"""
Load test: simulate many users using AutoML AI at the same time.

Run the server in dev mode (login check off, users told apart by a header):
    REQUIRE_AUTH=0 MODEL_SIGNING_KEY=test uvicorn main2:app --port 8000 --workers 1
Then:
    python loadtest.py --users 500 --concurrency 100 --jobs 150

Never use REQUIRE_AUTH=0 on a public server. For a deployed test, give each virtual
user a real Firebase token instead (edit headers() below).
"""
import argparse, asyncio, collections, random, statistics, time
import httpx


def make_csv(rows=300):
    rnd = random.Random(1)
    lines = ["age,income,city"]
    for _ in range(rows):
        g = rnd.randint(0, 2)
        lines.append(f"{rnd.gauss([25, 45, 65][g], 3):.1f},{rnd.gauss([30000, 90000, 50000][g], 4000):.0f},{['Patna', 'Delhi', 'Pune'][g]}")
    return "\n".join(lines).encode()


async def user(i, client, base, csv, run_job, stats):
    h = {"X-Dev-User": f"load-{i}"}

    async def call(name, method, path, **kw):
        t = time.time()
        try:
            r = await client.request(method, base + path, headers=h, timeout=300, **kw)
            stats["codes"][f"{name}:{r.status_code}"] += 1
            stats["lat"][name].append(time.time() - t)
            return r
        except Exception as e:
            stats["codes"][f"{name}:{type(e).__name__}"] += 1
            return None

    for _ in range(4):                                   # users retry a few times when told "busy"
        r = await call("upload", "POST", "/api/upload", files={"file": ("d.csv", csv, "text/csv")})
        if r is not None and r.status_code == 200:
            break
        await asyncio.sleep(random.uniform(1, 3))
    else:
        return
    if not run_job:
        return
    r = await call("start_job", "POST", "/api/jobs/cluster", json={"k_max": 5})
    if r is None or r.status_code != 200:
        return
    jid, t0 = r.json()["id"], time.time()
    while time.time() - t0 < 600:
        r = await call("poll", "GET", f"/api/jobs/{jid}")
        if r is not None and r.status_code == 200 and r.json()["status"] in ("done", "error"):
            stats["job_total"].append(time.time() - t0)
            stats["codes"][f"job:{r.json()['status']}"] += 1
            return
        await asyncio.sleep(1.5)


async def main(a):
    csv, stats = make_csv(), {"codes": collections.Counter(), "lat": collections.defaultdict(list), "job_total": []}
    sem = asyncio.Semaphore(a.concurrency)
    limits = httpx.Limits(max_connections=a.concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        async def one(i):
            async with sem:
                await user(i, client, a.base, csv, i < a.jobs, stats)
        t = time.time()
        await asyncio.gather(*(one(i) for i in range(a.users)))
        wall = time.time() - t
    print(f"\n{a.users} users, concurrency {a.concurrency}, {a.jobs} of them ran a clustering job -> {wall:.0f}s total")
    for k, v in sorted(stats["codes"].items()):
        print(f"  {k:24s} {v}")
    for name, xs in stats["lat"].items():
        if xs:
            xs.sort()
            print(f"  {name:10s} median {statistics.median(xs):.2f}s   p95 {xs[int(len(xs) * .95) - 1]:.2f}s   max {xs[-1]:.2f}s")
    if stats["job_total"]:
        xs = sorted(stats["job_total"])
        print(f"  job (queue + run) median {statistics.median(xs):.1f}s   p95 {xs[int(len(xs) * .95) - 1]:.1f}s   max {xs[-1]:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--users", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--jobs", type=int, default=150)
    asyncio.run(main(p.parse_args()))