import pandas as pd
def try_convert_numeric(series, threshold=0.90):
    """
    Convert an object column to numeric only when
    most non-null values look numeric.
    """

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(r"[$₹€£]", "", regex=True)
        .str.replace("%", "", regex=False)
    )

    numeric = pd.to_numeric(
        cleaned,
        errors="coerce"
    )

    non_null_count = series.notna().sum()

    if non_null_count == 0:
        return series, False

    numeric_ratio = numeric.notna().sum() / non_null_count

    if numeric_ratio >= threshold:
        return numeric, True

    return series, False