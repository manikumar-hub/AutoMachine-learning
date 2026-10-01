from typing import Any, Dict, List, Literal, Optional, Union
import io
import re

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import base64
import os
import sys
import time
import json
import hmac
import hashlib
import secrets
import threading
import contextvars
import urllib.request
import traceback
from collections.abc import MutableMapping
from pathlib import Path
from fastapi import Depends
from starlette.concurrency import run_in_threadpool
import joblib
import matplotlib
matplotlib.use("Agg")  # headless — no GUI needed on a server
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from load_data.cleancol import try_convert_numeric

from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso
)
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


# ======================================================================
# REQUEST MODELS
# ======================================================================

class TargetRequest(BaseModel):
    target: str


class FeatureSelectRequest(BaseModel):
    # As many columns as you want: a list, or a comma separated string
    features: Optional[Union[List[str], str]] = None       # keep only these
    drop_columns: Optional[Union[List[str], str]] = None   # keep all except these


class DropNaConfig(BaseModel):
    columns_threshold: Optional[float] = None  # drop cols missing more than this ratio
    rows: bool = False                          # drop rows with missing values
    how: Literal["any", "all"] = "any"
    subset: Optional[List[str]] = None


class FillNaConfig(BaseModel):
    numeric: Optional[Literal["mean", "median", "mode", "zero", "constant"]] = None
    categorical: Optional[Literal["mode", "unknown", "constant"]] = None
    constant: Optional[Any] = None
    per_column: Dict[str, str] = {}


class FeatureEngineeringRequest(BaseModel):
    drop_duplicates: bool = False
    dropna: Optional[DropNaConfig] = None
    fillna: Optional[FillNaConfig] = None
    outliers: Literal["none", "clip", "remove"] = "none"


# ======================================================================
# CONFIG  (all overridable with environment variables)
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "autoaiml")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "1") != "0"          # REQUIRE_AUTH=0 only for local dev
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_MODEL_UPLOAD_MB = int(os.getenv("MAX_MODEL_UPLOAD_MB", "300"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_MINUTES", "30")) * 60
MAX_ACTIVE_USERS = int(os.getenv("MAX_ACTIVE_USERS", "10"))
MAX_PARALLEL_TRAININGS = int(os.getenv("MAX_PARALLEL_TRAININGS", "2"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

_key_env = os.getenv("MODEL_SIGNING_KEY", "")
if _key_env:
    MODEL_SIGNING_KEY = _key_env.encode()
else:
    MODEL_SIGNING_KEY = secrets.token_bytes(32)
    print("WARNING: MODEL_SIGNING_KEY is not set. Exported models will stop loading after a "
          "server restart. Set MODEL_SIGNING_KEY to a long random string.")


# ======================================================================
# PER-USER STATE
# Every signed-in user gets their own dataset / model memory, so two people
# using the app at the same time can never see or overwrite each other's data.
# `DATASET` and `MODEL_STORE` keep working exactly like before in the endpoints:
# they are small proxies that point at the current request's user.
# ======================================================================

_current_state = contextvars.ContextVar("current_state", default=None)
_STATES: Dict[str, Dict[str, Any]] = {}
_STATES_LOCK = threading.Lock()


def _new_state():
    return {
        "DATASET": {"df": None, "profile": None, "filename": None},
        "MODEL_STORE": {"bundle": None},
        "last_seen": time.time(),
    }


def _get_state(uid: str):
    now = time.time()
    with _STATES_LOCK:
        for key in [k for k, v in _STATES.items() if now - v["last_seen"] > SESSION_TTL_SECONDS]:
            del _STATES[key]                       # idle sessions free their memory
        state = _STATES.get(uid)
        if state is None:
            if len(_STATES) >= MAX_ACTIVE_USERS:
                oldest = min(_STATES, key=lambda k: _STATES[k]["last_seen"])
                if now - _STATES[oldest]["last_seen"] < 300:   # never kick out someone active
                    raise HTTPException(503, "Server is at capacity right now. Please try again in a few minutes.")
                del _STATES[oldest]
            state = _STATES[uid] = _new_state()
        state["last_seen"] = now
        return state


class _UserStore(MutableMapping):
    """Dict-like view of the current user's DATASET / MODEL_STORE."""

    def __init__(self, key):
        self._key = key

    def _d(self):
        state = _current_state.get()
        if state is None:
            raise HTTPException(401, "Please sign in first.")
        return state[self._key]

    def __getitem__(self, k): return self._d()[k]
    def __setitem__(self, k, v): self._d()[k] = v
    def __delitem__(self, k): del self._d()[k]
    def __iter__(self): return iter(list(self._d()))
    def __len__(self): return len(self._d())
    def clear(self): self._d().clear()


DATASET = _UserStore("DATASET")
MODEL_STORE = _UserStore("MODEL_STORE")


# ======================================================================
# AUTH — verifies the Firebase ID token the frontend already sends
# ======================================================================

_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
_certs_cache = {"certs": None, "expires": 0.0, "fetched": 0.0}
_certs_lock = threading.Lock()


def _firebase_certs(force=False):
    with _certs_lock:
        now = time.time()
        fresh = _certs_cache["certs"] and now < _certs_cache["expires"]
        if fresh and not (force and now - _certs_cache["fetched"] > 60):   # forced refresh at most 1/min
            return _certs_cache["certs"]
        try:
            with urllib.request.urlopen(_CERTS_URL, timeout=15) as resp:
                certs = json.loads(resp.read().decode())
                match = re.search(r"max-age=(\d+)", resp.headers.get("Cache-Control", ""))
        except Exception:
            print("AUTH ERROR: could not download Google's Firebase certificates:", file=sys.stderr)
            traceback.print_exc()
            if _certs_cache["certs"]:          # keep working with the last known certificates
                return _certs_cache["certs"]
            raise
        max_age = int(match.group(1)) if match else 3600
        _certs_cache.update(certs=certs, fetched=now, expires=now + max(300, min(max_age, 21600)))
        return certs


def _verify_firebase_token(token: str) -> str:
    """Returns the Firebase uid, or raises ValueError for an invalid / expired token."""
    from google.auth import jwt as google_jwt
    try:
        claims = google_jwt.decode(token, certs=_firebase_certs(), audience=FIREBASE_PROJECT_ID)
    except ValueError:
        # certificates may have rotated: refresh once (rate limited) and retry
        claims = google_jwt.decode(token, certs=_firebase_certs(force=True), audience=FIREBASE_PROJECT_ID)
    if claims.get("iss") != f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}":
        raise ValueError("wrong token issuer")
    uid = claims.get("sub")
    if not isinstance(uid, str) or not uid or len(uid) > 128:
        raise ValueError("token has no user id")
    return uid


async def _bind_user(request: Request):
    """Runs before every request: only /api/* needs a signed-in user."""
    if not request.url.path.startswith("/api"):
        return
    if not REQUIRE_AUTH:
        uid = "local-dev"
    else:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(401, "Please sign in first.")
        try:
            uid = await run_in_threadpool(_verify_firebase_token, token.strip())
        except ValueError:
            raise HTTPException(401, "Your session expired. Please sign in again.")
        except Exception as exc:
            print(f"AUTH ERROR (503): {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            raise HTTPException(503, "Could not verify your sign-in right now. Please try again.")
    _current_state.set(_get_state(uid))


# ======================================================================
# MODEL FILE SIGNING
# joblib/pickle files can run code when loaded, so the server only loads
# model files that it exported itself (HMAC-signed with MODEL_SIGNING_KEY).
# ======================================================================

_MODEL_MAGIC = b"AMLM1"


def _sign_payload(payload: bytes) -> bytes:
    return _MODEL_MAGIC + hmac.new(MODEL_SIGNING_KEY, payload, hashlib.sha256).digest() + payload


def _unsign_payload(blob: bytes) -> bytes:
    head = len(_MODEL_MAGIC)
    if len(blob) < head + 32 or not blob.startswith(_MODEL_MAGIC):
        raise ValueError("not a signed model file")
    signature, payload = blob[head:head + 32], blob[head + 32:]
    if not hmac.compare_digest(signature, hmac.new(MODEL_SIGNING_KEY, payload, hashlib.sha256).digest()):
        raise ValueError("signature mismatch")
    return payload


# ======================================================================
# APP
# ======================================================================

_TRAIN_SLOTS = threading.BoundedSemaphore(MAX_PARALLEL_TRAININGS)
_PLOT_LOCK = threading.Lock()          # matplotlib's pyplot is not thread-safe

app = FastAPI(title="AutoML AI", dependencies=[Depends(_bind_user)])
if CORS_ORIGINS:   # same-origin deployment needs no CORS; set CORS_ORIGINS only if the UI is hosted elsewhere
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/healthz/auth")
def healthz_auth():
    """Open this URL in the browser to see why sign-in verification fails (no secrets are shown)."""
    out = {"require_auth": REQUIRE_AUTH, "firebase_project_id": FIREBASE_PROJECT_ID}
    try:
        import google.auth  # noqa: F401
        from google.auth import jwt as _jwt  # noqa: F401
        out["google_auth_installed"] = True
    except Exception as exc:
        out["google_auth_installed"] = f"NO - {type(exc).__name__}: {exc}"
    try:
        import cryptography  # noqa: F401
        out["cryptography_installed"] = True
    except Exception as exc:
        out["cryptography_installed"] = f"NO - {type(exc).__name__}: {exc}"
    try:
        out["certificates"] = f"ok ({len(_firebase_certs())} keys)"
    except Exception as exc:
        out["certificates"] = f"FAILED - {type(exc).__name__}: {exc}"
    return out


# ======================================================================
# HELPERS
# ======================================================================

def _require_df():
    if DATASET["df"] is None:
        raise HTTPException(400, "Please upload a CSV file first.")
    return DATASET["df"]


def _require_target():
    df = _require_df()
    if "target" not in DATASET:
        raise HTTPException(400, "Please select target column first.")
    return df, DATASET["target"]


def _invalidate():
    for key in ("X_train", "X_test", "y_train", "y_test",
                "preprocessor", "model_results", "best_model",
                "trained_models", "features_used", "numerical_columns",
                "categorical_columns", "feature_examples"):
        DATASET.pop(key, None)
    MODEL_STORE["bundle"] = None


def _restore_pre_fe():
    if "df_before_fe" in DATASET:
        DATASET["df"] = DATASET.pop("df_before_fe")


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")
    return [v.strip() for v in value if v and v.strip()]


def _resolve_columns(df, names):
    lookup = {c.strip().lower(): c for c in df.columns}
    found, unknown = [], []
    for name in names:
        real = lookup.get(name.strip().lower())
        (found if real else unknown).append(real or name)
    return found, unknown


def _records(df):
    return df.astype(object).where(df.notnull(), None).to_dict(orient="records")


def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _fill(series, strategy, constant=None):
    if str(series.dtype) == "category":
        series = series.astype(object)
    numeric = _is_numeric(series)

    if strategy in ("mean", "median", "zero") and not numeric:
        raise HTTPException(400, f"'{strategy}' works only on numeric columns ('{series.name}' is not numeric).")
    if strategy == "unknown" and numeric:
        raise HTTPException(400, f"'unknown' works only on non-numeric columns ('{series.name}' is numeric).")

    if strategy == "mean":
        return series.fillna(series.mean())
    if strategy == "median":
        return series.fillna(series.median())
    if strategy == "mode":
        mode = series.mode(dropna=True)
        return series.fillna(mode.iloc[0]) if len(mode) else series
    if strategy == "zero":
        return series.fillna(0)
    if strategy == "unknown":
        return series.fillna("Unknown")
    if strategy == "constant":
        if constant is None:
            raise HTTPException(400, "Strategy 'constant' needs a 'constant' value.")
        return series.fillna(constant)
    if strategy == "ffill":
        return series.ffill()
    if strategy == "bfill":
        return series.bfill()
    raise HTTPException(400, f"Unknown fill strategy '{strategy}'.")


# ======================================================================
# BASIC ENDPOINTS
# ======================================================================
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Pass data into the template using a context dictionary
    context = {
       # 'request' is strictly required by Jinja2 in FastAPI
        "message": "Hello from the backend!"
    }
    return templates.TemplateResponse(request,"index.html", context)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is too large (max {MAX_UPLOAD_MB} MB).")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Could not read this file as CSV.")

    profile = {
        "filename": file.filename,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    DATASET.clear()
    MODEL_STORE["bundle"] = None
    DATASET["df"] = df
    DATASET["profile"] = profile
    DATASET["filename"] = file.filename
    return profile


@app.get('/api/information')
def get_profile():
    df = _require_df()
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": df.nunique().to_dict(),
        "head": _records(df.head()),
        "describe": df.describe(include="all").fillna("").to_dict()
    }


@app.post("/api/validate")
def validate_dataset():
    df = _require_df()
    validation = {
        "is_valid": True,
        "checks": {
            "has_rows": df.shape[0] > 0,
            "has_columns": df.shape[1] > 0,
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": int(df.isnull().sum().sum()),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
    }
    if df.shape[0] == 0 or df.shape[1] == 0:
        validation["is_valid"] = False
    return validation


@app.post("/api/target")
def select_target(request: TargetRequest):
    _restore_pre_fe()
    df = _require_df()
    target = request.target

    if target not in df.columns:
        raise HTTPException(400, f"Column '{target}' not found in dataset.")

    target_data = df[target]

    if (
        pd.api.types.is_bool_dtype(target_data)
        or pd.api.types.is_object_dtype(target_data)
        or pd.api.types.is_string_dtype(target_data)
        or isinstance(target_data.dtype, pd.CategoricalDtype)
    ):
        problem_type = "classification"
    elif pd.api.types.is_numeric_dtype(target_data):
        problem_type = "classification" if target_data.nunique() <= 15 else "regression"
    else:
        raise HTTPException(400, "Unable to determine problem type.")

    DATASET["target"] = target
    DATASET["problem_type"] = problem_type
    DATASET.pop("features", None)
    _invalidate()

    return {"target": target, "problem_type": problem_type, "unique_values": int(target_data.nunique())}


@app.post("/api/clean-columns")
def clean_columns():
    _restore_pre_fe()
    df = _require_df()

    if "problem_type" not in DATASET:
        raise HTTPException(400, "Please select target first")

    df = df.copy()
    target = DATASET["target"]
    problem_type = DATASET["problem_type"]
    changed_columns = {}

    if problem_type == "regression":
        cleaned_target, converted = try_convert_numeric(df[target])
        if converted:
            df[target] = cleaned_target
            changed_columns[target] = {
                "from": "object", "to": str(df[target].dtype),
                "reason": "Numeric values detected in regression target"
            }
        else:
            raise HTTPException(400, "Regression target could not be converted to numeric")

    for column in [c for c in df.columns if c != target]:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            cleaned_column, converted = try_convert_numeric(df[column])
            if converted:
                old_dtype = str(df[column].dtype)
                df[column] = cleaned_column
                changed_columns[column] = {
                    "from": old_dtype, "to": str(df[column].dtype),
                    "reason": "Numeric-looking values detected"
                }

    DATASET["df"] = df
    _invalidate()
    return {
        "message": "Column cleaning completed", "problem_type": problem_type,
        "target": target, "changed_columns": changed_columns
    }


# ======================================================================
# COLUMN OVERVIEW + FEATURE SELECTION
# ======================================================================

@app.get("/api/columns")
def get_columns():
    df = _require_df()
    target = DATASET.get("target")
    n = max(len(df), 1)
    columns, recommended_drop = [], []

    for col in df.columns:
        series = df[col]
        unique = int(series.nunique())
        missing = int(series.isnull().sum())
        suggestion = None

        if col != target:
            id_name = col.lower() == "id" or col.lower().endswith("id")
            if unique <= 1:
                suggestion = "constant"
            elif missing / n > 0.6:
                suggestion = "high_missing"
            elif unique / n > 0.95 and (not _is_numeric(series) or id_name):
                suggestion = "id_like"
            if suggestion:
                recommended_drop.append(col)

        columns.append({
            "name": col, "dtype": str(series.dtype), "missing": missing,
            "missing_pct": round(missing / n * 100, 2), "unique": unique,
            "is_target": col == target,
            "selected": col in DATASET["features"] if "features" in DATASET else col != target,
            "suggestion": suggestion
        })

    return {"target": target, "columns": columns, "recommended_drop": recommended_drop}


@app.post("/api/select-features")
def select_features(request: FeatureSelectRequest):
    df, target = _require_target()
    keep_in = _as_list(request.features)
    drop_in = _as_list(request.drop_columns)

    if keep_in and drop_in:
        raise HTTPException(400, "Use either 'features' or 'drop_columns', not both.")
    if not keep_in and drop_in is None:
        raise HTTPException(400, "Send 'features' (columns to keep) or 'drop_columns'.")

    names = keep_in if keep_in else drop_in
    chosen, unknown = _resolve_columns(df, names)
    if unknown:
        raise HTTPException(400, f"Columns not found: {unknown}. Available: {df.columns.tolist()}")

    if keep_in:
        features = [c for c in df.columns if c in chosen and c != target]
    else:
        features = [c for c in df.columns if c not in chosen and c != target]

    if not features:
        raise HTTPException(400, "At least one feature column is required.")

    DATASET["features"] = features
    _invalidate()
    return {
        "message": "Feature selection saved", "target": target,
        "selected_features": features,
        "dropped_columns": [c for c in df.columns if c not in features and c != target]
    }


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

@app.post("/api/feature-engineering")
def feature_engineering(request: FeatureEngineeringRequest):
    _, target = _require_target()

    if "df_before_fe" not in DATASET:
        DATASET["df_before_fe"] = DATASET["df"].copy()

    df = DATASET["df_before_fe"].copy()
    report = {
        "rows_before": int(len(df)), "columns_before": int(df.shape[1]),
        "missing_before": int(df.isnull().sum().sum()), "steps": []
    }

    n = len(df)
    df = df.dropna(subset=[target])
    if len(df) != n:
        report["steps"].append({"step": "drop_rows_missing_target", "rows_removed": n - len(df)})

    if request.drop_duplicates:
        n = len(df)
        df = df.drop_duplicates()
        report["steps"].append({"step": "drop_duplicates", "rows_removed": n - len(df)})

    if request.dropna:
        cfg = request.dropna
        if cfg.columns_threshold is not None:
            if not 0 <= cfg.columns_threshold <= 1:
                raise HTTPException(400, "columns_threshold must be between 0 and 1.")
            ratio = df.isnull().mean()
            to_drop = [c for c in df.columns if c != target and ratio[c] > cfg.columns_threshold]
            df = df.drop(columns=to_drop)
            report["steps"].append({
                "step": "drop_high_missing_columns", "threshold": cfg.columns_threshold,
                "columns_removed": to_drop
            })

        if cfg.rows:
            subset = cfg.subset or [c for c in df.columns if c != target]
            bad = [c for c in subset if c not in df.columns]
            if bad:
                raise HTTPException(400, f"Columns not found for dropna subset: {bad}")
            n = len(df)
            df = df.dropna(how=cfg.how, subset=subset)
            report["steps"].append({"step": "drop_rows_with_missing", "how": cfg.how, "rows_removed": n - len(df)})

    if request.fillna:
        cfg = request.fillna
        features = [c for c in df.columns if c != target]
        before = df.isnull().sum()

        for col, strategy in cfg.per_column.items():
            if col not in features:
                raise HTTPException(400, f"Column '{col}' not found in features.")
            df[col] = _fill(df[col], strategy, cfg.constant)

        for col in features:
            if col in cfg.per_column or not df[col].isnull().any():
                continue
            strategy = cfg.numeric if _is_numeric(df[col]) else cfg.categorical
            if strategy:
                df[col] = _fill(df[col], strategy, cfg.constant)

        after = df.isnull().sum()
        report["steps"].append({
            "step": "fillna",
            "filled": {c: int(before[c] - after[c]) for c in df.columns if before[c] - after[c] > 0}
        })

    if request.outliers != "none":
        num_cols = [c for c in df.columns if c != target and _is_numeric(df[c])]
        keep = pd.Series(True, index=df.index)
        clipped = {}

        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outside = (df[col] < low) | (df[col] > high)

            if request.outliers == "clip":
                if outside.any():
                    clipped[col] = int(outside.sum())
                    df[col] = df[col].clip(low, high)
            else:
                keep &= ~outside

        if request.outliers == "clip":
            report["steps"].append({"step": "clip_outliers", "values_clipped": clipped})
        else:
            n = len(df)
            df = df[keep]
            report["steps"].append({"step": "remove_outlier_rows", "rows_removed": n - len(df)})

    if len(df) == 0:
        raise HTTPException(400, "These options would remove every row. Try fillna instead of dropna.")

    df = df.reset_index(drop=True)
    DATASET["df"] = df
    if "features" in DATASET:
        DATASET["features"] = [c for c in DATASET["features"] if c in df.columns]
    _invalidate()

    report.update({
        "rows_after": int(len(df)), "columns_after": int(df.shape[1]),
        "missing_after": int(df.isnull().sum().sum()),
        "columns": df.columns.tolist(), "preview": _records(df.head())
    })
    return report


@app.post("/api/feature-engineering/reset")
def reset_feature_engineering():
    _require_df()
    _restore_pre_fe()
    _invalidate()
    return {"message": "Feature engineering undone", "rows": len(DATASET["df"])}


# ======================================================================
# PREPROCESS
# ======================================================================

@app.post("/api/preprocess")
def preprocess_dataset():
    df, target = _require_target()
    problem_type = DATASET["problem_type"]

    missing_target = int(df[target].isnull().sum())
    if missing_target:
        df = df[df[target].notnull()]
        if len(df) == 0:
            raise HTTPException(400, "Target column is empty.")

    y = df[target]
    features = [c for c in DATASET.get("features", df.columns) if c in df.columns and c != target]
    if not features:
        raise HTTPException(400, "No feature columns available.")

    X = df[features]
    stratify = y if problem_type == "classification" and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ],
        remainder="drop"
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    DATASET["X_train"] = X_train_processed
    DATASET["X_test"] = X_test_processed
    DATASET["y_train"] = y_train
    DATASET["y_test"] = y_test
    DATASET["preprocessor"] = preprocessor
    DATASET["features_used"] = features
    DATASET["numerical_columns"] = numerical_columns
    DATASET["categorical_columns"] = categorical_columns

    return {
        "message": "Preprocessing completed successfully", "problem_type": problem_type,
        "features_used": features, "rows_dropped_missing_target": missing_target,
        "train_rows": X_train_processed.shape[0], "test_rows": X_test_processed.shape[0],
        "train_features": X_train_processed.shape[1], "test_features": X_test_processed.shape[1],
        "numerical_columns": numerical_columns, "categorical_columns": categorical_columns
    }


# ======================================================================
# TRAIN — classification AND regression
# ======================================================================

@app.post("/api/train")
def train_models():
    if not _TRAIN_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "Server is busy training other models. Please try again in a minute.")
    try:
        return _train_models_impl()
    finally:
        _TRAIN_SLOTS.release()


def _train_models_impl():
    if DATASET["df"] is None:
        raise HTTPException(400, "Please upload a CSV file first.")
    if "target" not in DATASET:
        raise HTTPException(400, "Please select target column first.")
    if "X_train" not in DATASET:
        raise HTTPException(400, "Please run preprocessing first.")

    MODEL_STORE["bundle"] = None
    for key in ("trained_models", "model_results", "best_model", "feature_examples"):
        DATASET.pop(key, None)
    X_train, X_test = DATASET["X_train"], DATASET["X_test"]
    y_train, y_test = DATASET["y_train"], DATASET["y_test"]

    if DATASET["problem_type"] == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=1),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            #"SVC": SVC(probability=True, random_state=42),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=42)
        }

        results, trained = [], {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                results.append({
                    "model": name,
                    "accuracy": round(accuracy_score(y_test, pred), 4),
                    "precision": round(precision_score(y_test, pred, average="weighted", zero_division=0), 4),
                    "recall": round(recall_score(y_test, pred, average="weighted", zero_division=0), 4),
                    "f1_score": round(f1_score(y_test, pred, average="weighted", zero_division=0), 4)
                })
                trained[name] = model
                print(f"{name} trained successfully")
            except Exception as e:
                results.append({"model": name, "error": str(e)})

        successful = [r for r in results if "f1_score" in r]
        if not successful:
            raise HTTPException(500, "No model could be trained.")

        best_result = max(successful, key=lambda x: x["f1_score"])
        best_model = trained[best_result["model"]]
        cm = confusion_matrix(y_test, best_model.predict(X_test)).tolist()

        DATASET["model_results"] = results
        DATASET["best_model"] = best_result
        DATASET["trained_models"] = trained
        _save_feature_examples()

        return {
            "message": "Model training completed", "problem_type": "classification",
            "models": results, "best_model": best_result, "confusion_matrix": cm
        }

    # ---------------- regression ----------------
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        # PolynomialRegression removed: degree=2 on the already one-hot-encoded
        # feature matrix explodes into thousands of columns and can hang/OOM.
    }

    results, trained = [], {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            results.append({
                "model": name,
                "r2_score": round(r2_score(y_test, pred), 4),
                "mae": round(mean_absolute_error(y_test, pred), 4),
                "mse": round(mse, 4),
                "rmse": round(mse ** 0.5, 4)
            })
            trained[name] = model
            print(f"{name} trained successfully")
        except Exception as e:
            results.append({"model": name, "error": str(e)})

    successful = [r for r in results if "r2_score" in r]
    if not successful:
        raise HTTPException(500, "No model could be trained.")

    best_result = max(successful, key=lambda x: x["r2_score"])
    DATASET["model_results"] = results
    DATASET["best_model"] = best_result
    DATASET["trained_models"] = trained
    _save_feature_examples()

    return {
        "message": "Model training completed", "problem_type": "regression",
        "models": results, "best_model": best_result
    }


# ======================================================================
# DATA ANALYTICS — matplotlib + seaborn charts on the currently uploaded dataset
# ======================================================================

ACCENT = "#6366f1"

sns.set_theme(
    style="darkgrid",
    rc={
        "figure.facecolor": "#0f1421",
        "axes.facecolor": "#141a2b",
        "axes.edgecolor": "#222a3f",
        "grid.color": "#222a3f",
        "text.color": "#e8eaf2",
        "axes.labelcolor": "#8a91aa",
        "xtick.color": "#8a91aa",
        "ytick.color": "#8a91aa",
        "axes.titleweight": "bold",
    },
)


class AnalyzeRequest(BaseModel):
    kind: Literal["corr", "missing", "hist", "box", "count", "pair"]
    column: str = ""


def _to_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _numeric_df(df):
    return df.select_dtypes(include="number")


def _need_numeric(df, min_cols=1):
    num = _numeric_df(df)
    if num.shape[1] < min_cols:
        raise HTTPException(400, f"This chart needs at least {min_cols} numeric column(s).")
    return num


def _chart_corr(df, col):
    num = _need_numeric(df, 2)
    c = num.corr()
    size = max(5, min(12, len(c) * 0.7))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(c, annot=len(c) <= 12, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5, linecolor="#0f1421")
    ax.set_title("Correlation heatmap")
    return fig


def _chart_missing(df, col):
    m = df.isnull().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, len(m) * 0.35)))
    sns.barplot(x=m.values, y=m.index, color=ACCENT, ax=ax)
    ax.set_title("Missing values per column")
    ax.set_xlabel("Missing count")
    return fig


def _chart_hist(df, col):
    num = _need_numeric(df)
    cols = [col] if col in num.columns else list(num.columns[:12])
    n = len(cols)
    ncol = min(3, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        sns.histplot(num[c].dropna(), kde=True, color=ACCENT, ax=ax)
        ax.set_title(c)
        ax.set_xlabel("")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Distributions", fontweight="bold")
    fig.tight_layout()
    return fig


def _chart_box(df, col):
    num = _need_numeric(df)
    if col in num.columns:
        fig, ax = plt.subplots(figsize=(4, 5))
        sns.boxplot(y=num[col], color=ACCENT, ax=ax)
        ax.set_title(f"Box plot: {col}")
        return fig
    z = ((num - num.mean()) / num.std()).iloc[:, :10]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=z, palette="viridis", ax=ax)
    ax.set_title("Outliers (standardized)")
    ax.tick_params(axis="x", rotation=35)
    return fig


def _chart_count(df, col):
    if col not in df.columns:
        cats = [c for c in df.columns if df[c].nunique() <= 20]
        if not cats:
            raise HTTPException(400, "No low-cardinality column found. Pick a column.")
        col = cats[-1]
    order = df[col].value_counts().index[:15]
    fig, ax = plt.subplots(figsize=(8, max(3, len(order) * 0.4)))
    sns.countplot(data=df, y=col, order=order, color=ACCENT, ax=ax)
    ax.set_title(f"Value counts: {col}")
    return fig


def _chart_pair(df, col):
    num = _need_numeric(df, 2)
    sub = num.iloc[:, :5].dropna()
    sub = sub.sample(min(400, len(sub)), random_state=42)
    g = sns.pairplot(sub, corner=True, plot_kws={"color": ACCENT, "s": 12, "alpha": 0.6}, diag_kws={"color": ACCENT})
    g.figure.suptitle("Pair plot (first 5 numeric columns)", y=1.02, fontweight="bold")
    return g.figure


CHART_KINDS = {
    "corr": _chart_corr, "missing": _chart_missing, "hist": _chart_hist,
    "box": _chart_box, "count": _chart_count, "pair": _chart_pair,
}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    """Render a matplotlib/seaborn chart for the CSV already uploaded via /api/upload."""
    df = _require_df()
    if request.column and request.column not in df.columns:
        raise HTTPException(400, f"Column '{request.column}' not found in dataset.")
    with _PLOT_LOCK:
        fig = CHART_KINDS[request.kind](df, request.column)
        return {"kind": request.kind, "image": _to_png(fig)}


# ======================================================================
# MODEL EXPORT / IMPORT / PREDICTION
# ======================================================================

def _save_feature_examples():
    """One realistic example value per feature, used as input-form placeholders."""
    df = DATASET["df"]
    features = DATASET.get("features_used", [])
    examples = {}
    for col in features:
        s = df[col].dropna()
        if s.empty:
            examples[col] = ""
        elif _is_numeric(s):
            examples[col] = round(float(s.median()), 4)
        else:
            examples[col] = str(s.mode().iloc[0])
    DATASET["feature_examples"] = examples


def _active_bundle():
    """The model to predict with: an uploaded .joblib takes priority, otherwise
    fall back to whatever was just trained in this session."""
    if MODEL_STORE["bundle"] is not None:
        return MODEL_STORE["bundle"]

    if DATASET.get("trained_models") and DATASET.get("best_model"):
        name = DATASET["best_model"]["model"]
        return {
            "model_name": name,
            "model": DATASET["trained_models"][name],
            "preprocessor": DATASET["preprocessor"],
            "features": DATASET["features_used"],
            "numerical_columns": DATASET["numerical_columns"],
            "categorical_columns": DATASET["categorical_columns"],
            "feature_examples": DATASET.get("feature_examples", {}),
            "target": DATASET["target"],
            "problem_type": DATASET["problem_type"],
        }

    raise HTTPException(
        400,
        "No trained model available. Train a model first, or upload a "
        "previously downloaded .joblib model."
    )


def _bundle_schema(bundle):
    numeric = set(bundle["numerical_columns"])
    examples = bundle["feature_examples"]
    fields = [
        {
            "name": f,
            "type": "numeric" if f in numeric else "categorical",
            "example": examples.get(f, "")
        }
        for f in bundle["features"]
    ]
    return {
        "model": bundle["model_name"],
        "target": bundle["target"],
        "problem_type": bundle["problem_type"],
        "fields": fields
    }


@app.get("/api/model/download")
def download_model(model: Optional[str] = Query(None)):
    """Download the trained model + its preprocessing pipeline as one .joblib file."""
    if not DATASET.get("trained_models"):
        raise HTTPException(400, "No trained model to download yet. Run /api/train first.")

    name = model or DATASET["best_model"]["model"]
    if name not in DATASET["trained_models"]:
        raise HTTPException(400, f"Model '{name}' not found. Available: {list(DATASET['trained_models'])}")

    bundle = {
        "model_name": name,
        "model": DATASET["trained_models"][name],
        "preprocessor": DATASET["preprocessor"],
        "features": DATASET["features_used"],
        "numerical_columns": DATASET["numerical_columns"],
        "categorical_columns": DATASET["categorical_columns"],
        "feature_examples": DATASET.get("feature_examples", {}),
        "target": DATASET["target"],
        "problem_type": DATASET["problem_type"],
    }

    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    buf = io.BytesIO(_sign_payload(buf.getvalue()))
    target_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(DATASET["target"])).strip("._") or "target"
    return StreamingResponse(
        buf, media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{target_name}_prediction.joblib"'}
    )


@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """Load a previously downloaded .joblib model bundle for predictions."""
    data = await file.read()
    if len(data) > MAX_MODEL_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Model file is too large (max {MAX_MODEL_UPLOAD_MB} MB).")
    try:
        payload = _unsign_payload(data)
    except ValueError:
        raise HTTPException(
            400,
            "This model file was not exported by this server (or was modified), so it was not loaded. "
            "For safety only models downloaded from this app can be used. Train again and download a fresh copy."
        )
    try:
        bundle = await run_in_threadpool(joblib.load, io.BytesIO(payload))
    except Exception:
        raise HTTPException(400, "Could not read this file as a model bundle.")

    required = {"model", "preprocessor", "features", "target", "problem_type"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise HTTPException(400, "This file doesn't look like a model exported by this app.")

    MODEL_STORE["bundle"] = bundle
    return _bundle_schema(bundle)


@app.get("/api/model/schema")
def model_schema():
    """Feature list (+ example values for input placeholders) for the active model."""
    return _bundle_schema(_active_bundle())


class PredictRequest(BaseModel):
    values: Dict[str, Any]


@app.post("/api/predict")
def predict(request: PredictRequest):
    bundle = _active_bundle()
    features = bundle["features"]

    missing = [f for f in features if f not in request.values]
    if missing:
        raise HTTPException(400, f"Missing values for: {missing}")

    row = {}
    for f in features:
        v = request.values[f]
        if f in bundle["numerical_columns"]:
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"'{f}' must be a number, got {v!r}.")
        row[f] = v

    X = pd.DataFrame([row], columns=features)
    X_processed = bundle["preprocessor"].transform(X)
    model = bundle["model"]
    pred = model.predict(X_processed)[0]

    result = {
        "model": bundle["model_name"],
        "target": bundle["target"],
        "prediction": pred.item() if hasattr(pred, "item") else pred
    }

    if bundle["problem_type"] == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_processed)[0]
        classes = getattr(model, "classes_", range(len(proba)))
        result["probabilities"] = {
            str(c): round(float(p), 4) for c, p in zip(classes, proba)
        }

    return result
