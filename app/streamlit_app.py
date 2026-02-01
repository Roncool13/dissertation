import os
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import shutil
import tempfile
from pathlib import Path
import dvc.api
import boto3

# -----------------------
# Utility
# -----------------------
def clip01(p, eps=1e-6):
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)

def get_expected_cols(model) -> Optional[List[str]]:
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                return list(cols)
    return None

def align_X(df: pd.DataFrame, expected_cols: Optional[List[str]], name: str) -> pd.DataFrame:
    """
    Return X with exact expected columns if available.
    If expected cols are unknown (pyfunc without schema), fall back to numeric cols (excluding date).
    """
    if expected_cols is None:
        keep = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if not keep:
            raise ValueError(f"{name}: cannot infer numeric columns for model input.")
        return df[keep].copy()

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing required columns: {missing[:15]} (showing 15). "
            f"Available cols sample: {list(df.columns)[:25]}"
        )
    return df[expected_cols].copy()


def sanitize_X(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure no NaNs reach estimators.

    - Numeric cols: coerce to float and fill NaN with 0.0
    - Non-numeric cols (e.g., symbol): fill NaN with 'UNKNOWN'
    """
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float).fillna(0.0)
        else:
            # keep categorical/text as object
            X[c] = X[c].astype("object")
            X.loc[X[c].isna(), c] = "UNKNOWN"
    return X


def proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(y=1) for each row in X."""
    X = sanitize_X(X)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    pred = model.predict(X)
    if isinstance(pred, (pd.DataFrame, pd.Series)):
        pred = pred.values
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] >= 2:
        return np.asarray(pred[:, 1], dtype=float)
    return np.asarray(pred.ravel(), dtype=float)

def get_row(df: pd.DataFrame, symbol: str, date: pd.Timestamp) -> Optional[pd.DataFrame]:
    out = df[(df["symbol"] == symbol) & (df["date"] == date)]
    if len(out) == 0:
        return None
    return out.iloc[[0]].copy()

def compute_gate_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """
    Vectorized gate (0/1) from commonly available columns.
    Priority: g_sent -> news_present_lag_* -> article_count_lag_* -> article_count
    """
    if "g_sent" in frame.columns:
        g = pd.to_numeric(frame["g_sent"], errors="coerce").fillna(0.0)
        return (g > 0.0).astype(float).values

    for gate_col in ["news_present_lag_3", "news_present_lag_1", "article_count_lag_3", "article_count_lag_1", "article_count"]:
        if gate_col in frame.columns:
            g = pd.to_numeric(frame[gate_col], errors="coerce").fillna(0.0)
            return (g > 0.0).astype(float).values

    return np.zeros(len(frame), dtype=float)

def init_history():
    return pd.DataFrame(columns=[
        "timestamp", "model", "symbol", "date", "p_up", "pred", "actual", "is_correct"
    ])

def history_stats(hist: pd.DataFrame) -> Dict[str, float]:
    if hist.empty:
        return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": np.nan}
    total = len(hist)
    correct = int((hist["is_correct"] == True).sum())
    incorrect = total - correct
    acc = correct / total if total > 0 else np.nan
    return {"total": total, "correct": correct, "incorrect": incorrect, "accuracy": acc}

def summarize_probs(name: str, p: np.ndarray) -> Dict[str, float]:
    p = np.asarray(p, dtype=float)
    return {
        f"{name}_mean": float(np.mean(p)),
        f"{name}_std": float(np.std(p)),
        f"{name}_min": float(np.min(p)),
        f"{name}_max": float(np.max(p)),
        f"{name}_pct_ge_0.5": float(np.mean(p >= 0.5)),
    }

# -----------------------
# DVC + AWS helpers
# -----------------------
@st.cache_resource
def aws_assume_role_env():
    """
    Assumes role and sets env vars. Expects Streamlit secrets:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, ASSUME_ROLE_ARN
    optional: AWS_REGION, ASSUME_ROLE_EXTERNAL_ID
    """
    os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_DEFAULT_REGION"] = st.secrets.get("AWS_REGION", "us-east-1")

    role_arn = st.secrets["ASSUME_ROLE_ARN"]
    external_id = st.secrets.get("ASSUME_ROLE_EXTERNAL_ID")

    sts = boto3.client("sts")
    kwargs = {
        "RoleArn": role_arn,
        "RoleSessionName": "streamlit-dvc-session",
        "DurationSeconds": 3600
    }
    if external_id:
        kwargs["ExternalId"] = external_id

    resp = sts.assume_role(**kwargs)
    creds = resp["Credentials"]

    os.environ["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = creds["SessionToken"]

@st.cache_data
def load_parquet(path: str, repo: str = ".", rev: str | None = None) -> pd.DataFrame:
    """
    Loads parquet either locally or from DVC remote using dvc.api.open.
    Caches fetched files in a temp dir.
    """
    path = str(path)

    if os.path.exists(path):
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    cache_dir = Path(tempfile.gettempdir()) / "dvc_streamlit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_target = cache_dir / Path(path).name

    if not local_target.exists():
        with dvc.api.open(path=path, repo=repo, rev=rev, mode="rb") as src:
            with open(local_target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    df = pd.read_parquet(local_target)
    df["date"] = pd.to_datetime(df["date"])
    return df

# -----------------------
# MLflow model loading
# -----------------------
@st.cache_resource
def load_model(uri: str):
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception:
        return mlflow.pyfunc.load_model(uri)

# -----------------------
# Batch prediction (Backtest-lite)
# -----------------------
def build_inputs_for_models(
    base: pd.DataFrame,
    sent_df: pd.DataFrame,
    pat_df: pd.DataFrame,
    ohlcv_expected: Optional[List[str]],
    sent_expected: Optional[List[str]],
    pat_expected: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Given base = OHLCV slice with symbol/date (+ label), attach sentiment/pattern columns as needed
    and return aligned X_ohlcv, X_sent, X_pat + gate vector.
    """
    key = ["symbol", "date"]
    base = base.copy()

    # Sentiment: left join (keep all OHLCV rows)
    sent_cols = key + [c for c in (sent_expected or []) if c not in key]
    sent_small = sent_df[sent_cols].copy() if sent_cols and all(c in sent_df.columns for c in sent_cols) else sent_df.copy()
    merged = base.merge(sent_small, on=key, how="left", suffixes=("", "_sentdup"))

    # If any duplicate-suffixed cols were created, prefer base (unsuffixed) and drop dup
    dup_cols = [c for c in merged.columns if c.endswith("_sentdup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    # Pattern: left join
    pat_cols = key + [c for c in (pat_expected or []) if c not in key]
    if "symbol" in pat_df.columns and "date" in pat_df.columns:
        pat_small = pat_df[pat_cols].copy() if pat_cols and all(c in pat_df.columns for c in pat_cols) else pat_df.copy()
        merged = merged.merge(pat_small, on=key, how="left", suffixes=("", "_patdup"))
        dup_cols = [c for c in merged.columns if c.endswith("_patdup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

    # Fill missing expert features with zeros (neutral)
    if sent_expected:
        for c in sent_expected:
            if c not in merged.columns:
                merged[c] = 0.0
        sent_num = [c for c in sent_expected if c not in key]
        if sent_num:
            merged[sent_num] = merged[sent_num].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if pat_expected:
        for c in pat_expected:
            if c not in merged.columns:
                merged[c] = 0.0
        pat_num = [c for c in pat_expected if c not in key]
        if pat_num:
            merged[pat_num] = merged[pat_num].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Ensure categorical keys are clean
    if "symbol" in merged.columns:
        merged["symbol"] = merged["symbol"].astype(str).fillna("UNK")

    # gate
    g = compute_gate_from_frame(merged)

    # Align X matrices
    X_ohlcv = align_X(merged, ohlcv_expected, "OHLCV(batch)")
    X_sent  = align_X(merged, sent_expected,  "SENT(batch)")
    X_pat   = align_X(merged, pat_expected,   "PAT(batch)")
    return X_ohlcv, X_sent, X_pat, g

def run_backtest_lite(
    ohlcv_slice: pd.DataFrame,
    sent_df: pd.DataFrame,
    pat_df: pd.DataFrame,
    label_col: str,
    threshold: float,
    no_trade: bool,
    delta: float,
    mode: str,
    m_ohlcv,
    m_sent,
    m_pat,
    m_fusion,
    ohlcv_expected,
    sent_expected,
    pat_expected,
) -> pd.DataFrame:
    """
    Returns dataframe with probabilities, predictions, correctness for a slice.
    mode: "OHLCV" or "FUSION"
    """
    # drop rows without label (can't score)
    df = ohlcv_slice.copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column `{label_col}` not found in OHLCV slice.")
    df = df.dropna(subset=[label_col]).copy()
    if df.empty:
        return df

    X_ohlcv, X_sent, X_pat, g = build_inputs_for_models(
        df, sent_df, pat_df, ohlcv_expected, sent_expected, pat_expected
    )

    p_ohlcv = clip01(proba(m_ohlcv, X_ohlcv))
    p_sent  = clip01(proba(m_sent,  X_sent))
    p_pat   = clip01(proba(m_pat,   X_pat))

    meta = pd.DataFrame({
        "p_ohlcv": p_ohlcv,
        "p_sent": p_sent,
        "p_pat": p_pat,
        "g_sent": g.astype(float),
        "p_ohlcv_x_sent": p_ohlcv * p_sent,
    })
    p_fusion = clip01(proba(m_fusion, meta[["p_ohlcv","p_sent","p_pat","g_sent","p_ohlcv_x_sent"]]))

    if mode == "OHLCV":
        p_up = p_ohlcv
        model_used = "OHLCV"
    else:
        p_up = p_fusion
        model_used = "Fusion(meta)"

        # handled below (single-row section)

    out = df[["symbol","date",label_col]].copy()
    out.rename(columns={label_col: "actual"}, inplace=True)
    out["model"] = model_used
    out["p_ohlcv"] = p_ohlcv
    out["p_sent"] = p_sent
    out["p_pat"] = p_pat
    out["g_sent"] = g
    out["p_fusion"] = p_fusion
    out["p_up"] = p_up

    # ---- Decision logic (supports NO-TRADE band) ----
    # traded = True if we actually take a directional view; False if NO-TRADE.
    p_up_arr = np.asarray(p_up, dtype=float)
    if no_trade:
        traded = (np.abs(p_up_arr - 0.5) >= float(delta))
    else:
        traded = np.ones_like(p_up_arr, dtype=bool)

    pred = (p_up_arr >= float(threshold)).astype(int)
    # Mark NO-TRADE predictions as -1 for readability
    pred = pred.astype(int)
    pred[~traded] = -1

    actual = np.asarray(out["actual"].values, dtype=int)
    is_correct = (pred == actual)
    # If no-trade, correctness is undefined
    is_correct = is_correct.astype(object)
    is_correct[~traded] = None

    out["pred"] = pred
    out["traded"] = traded
    out["is_correct"] = is_correct
    return out

# -----------------------
# Streamlit setup
# -----------------------
st.set_page_config(page_title="Dissertation Predictor", layout="wide")
st.title("📈 Dissertation Predictor: OHLCV vs Fusion (Meta LR)")

with st.sidebar:
    st.header("MLflow settings")
    tracking_uri = st.text_input(
        "MLflow Tracking URI",
        value=os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/Roncool13/dissertation-mlflow.mlflow")
    )
    mlflow.set_tracking_uri(tracking_uri)

    st.caption("Registry URIs used by default. Update stage/version if needed.")

    # Model URIs (your exact names)
    ohlcv_uri = st.text_input(
        "OHLCV model URI",
        value=os.environ.get("OHLCV_MODEL_URI", "models:/ohlcv_lr_multisym_2019_2023_persymnorm_true@production")
    )
    fusion_uri = st.text_input(
        "Fusion model URI",
        value=os.environ.get("FUSION_MODEL_URI", "models:/fusion_meta_lr_multisym@production")
    )
    pat_uri = st.text_input(
        "Pattern model URI",
        value=os.environ.get("PAT_MODEL_URI", "models:/pattern_lr_multisym_2019_2023@production")
    )
    sent_uri = st.text_input(
        "Sentiment model URI",
        value=os.environ.get("SENT_MODEL_URI", "models:/sentiment_lr_baseline_multisym@production")
    )

    st.divider()
    st.header("Feature data paths (DVC-aware)")
    ohlcv_path = st.text_input("OHLCV features parquet", value=os.environ.get("OHLCV_FEATS_PATH", "data/features/ohlcv_features.parquet"))
    sent_path  = st.text_input("Sentiment features parquet", value=os.environ.get("SENT_FEATS_PATH", "data/features/news_sentiment_features.parquet"))
    pat_path   = st.text_input("Pattern features parquet", value=os.environ.get("PAT_FEATS_PATH", "data/features/pattern_features.parquet"))
    label_col = st.text_input("Label column", value=os.environ.get("LABEL_COL", "y_up_5d"))

    st.divider()
    st.header("Prediction")
    threshold = st.slider("Decision threshold (UP if p >= threshold)", 0.40, 0.60, 0.50, 0.01)

    no_trade = st.checkbox("Enable NO-TRADE band around 0.5", value=False)
    delta = st.slider("NO-TRADE half-width (|p-0.5| < δ)", 0.00, 0.10, 0.03, 0.005)

    st.caption("If you see 'always UP', try threshold 0.52+ or add a no-trade band.")

    st.divider()
    st.header("History")
    if st.button("Reset history"):
        st.session_state["history"] = init_history()

# Init history
if "history" not in st.session_state:
    st.session_state["history"] = init_history()

# Ensure AWS assumed role (needed for DVC S3)
aws_assume_role_env()

# Load datasets
with st.spinner("Loading datasets (local or DVC remote)..."):
    ohlcv_df = load_parquet(ohlcv_path)
    sent_df = load_parquet(sent_path)
    pat_df  = load_parquet(pat_path)

# Sanity checks
for df_name, df in [("OHLCV", ohlcv_df), ("SENT", sent_df), ("PAT", pat_df)]:
    if "symbol" not in df.columns or "date" not in df.columns:
        st.error(f"{df_name} parquet must have columns: symbol, date")
        st.stop()

symbols = sorted(ohlcv_df["symbol"].dropna().unique().tolist())
min_date = pd.to_datetime(ohlcv_df["date"].min()).date()
max_date = pd.to_datetime(ohlcv_df["date"].max()).date()

# Load models
with st.spinner("Loading models from MLflow..."):
    m_ohlcv = load_model(ohlcv_uri)
    m_fusion = load_model(fusion_uri)
    m_pat = load_model(pat_uri)
    m_sent = load_model(sent_uri)

# Expected columns
ohlcv_expected = get_expected_cols(m_ohlcv)
pat_expected   = get_expected_cols(m_pat)
sent_expected  = get_expected_cols(m_sent)

# Tabs
tab_single, tab_range = st.tabs(["Single prediction", "Backtest-lite (date range)"])

# -----------------------
# Tab 1: Single prediction
# -----------------------
with tab_single:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Inputs")
        symbol = st.selectbox("Symbol", symbols, index=0, key="single_symbol")
        date = st.date_input("Date", value=max_date, min_value=min_date, max_value=max_date, key="single_date")
        model_choice = st.radio("Choose predictor", ["OHLCV only", "Fusion meta"], index=0, key="single_model")

    with c2:
        st.subheader("Dataset info")
        st.write(f"OHLCV rows: **{len(ohlcv_df):,}**")
        st.write(f"Date range: **{min_date} → {max_date}**")
        if label_col in ohlcv_df.columns:
            st.write(f"Label column: **{label_col}**")
        else:
            st.warning(f"Label `{label_col}` not found; actual comparison disabled.")

        # Quick debug: show whether probs look stuck
        if st.checkbox("Show model input schema (debug)", value=False):
            st.write("OHLCV expects:", ohlcv_expected)
            st.write("SENT expects:", sent_expected)
            st.write("PAT expects:", pat_expected)
            st.write("Fusion expects:", ["p_ohlcv","p_sent","p_pat","g_sent","p_ohlcv_x_sent"])
if st.button("Predict", key="single_predict"):
    d = pd.to_datetime(date)

    row_ohlcv = get_row(ohlcv_df, symbol, d)
    if row_ohlcv is None:
        st.error(f"No OHLCV row for {symbol} on {d.date()}")
        st.stop()

    row_sent = get_row(sent_df, symbol, d)  # may be None
    row_pat  = get_row(pat_df,  symbol, d)
    if row_pat is None:
        row_pat = row_ohlcv.copy()

    # Clean categorical key
    if "symbol" in row_ohlcv.columns:
        row_ohlcv["symbol"] = row_ohlcv["symbol"].astype(str).fillna("UNK")
    if row_sent is not None and "symbol" in row_sent.columns:
        row_sent["symbol"] = row_sent["symbol"].astype(str).fillna("UNK")
    if "symbol" in row_pat.columns:
        row_pat["symbol"] = row_pat["symbol"].astype(str).fillna("UNK")

    # Actual label (if available)
    actual = None
    if label_col in row_ohlcv.columns and not pd.isna(row_ohlcv[label_col].iloc[0]):
        actual = int(row_ohlcv[label_col].iloc[0])

    # ---- Expert probabilities ----
    X_ohlcv = align_X(row_ohlcv, ohlcv_expected, "OHLCV")
    p_ohlcv = float(clip01(proba(m_ohlcv, X_ohlcv))[0])

    # Sentiment: neutral row if missing
    if row_sent is None:
        row_sent = pd.DataFrame({"symbol": [symbol], "date": [d]})
    if sent_expected is not None:
        for c in sent_expected:
            if c not in row_sent.columns:
                row_sent[c] = 0.0
        num_cols = [c for c in sent_expected if c not in ["symbol", "date"]]
        if num_cols:
            row_sent[num_cols] = row_sent[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_sent = align_X(row_sent, sent_expected, "SENTIMENT")
    p_sent = float(clip01(proba(m_sent, X_sent))[0])

    # Pattern
    if pat_expected is not None:
        for c in pat_expected:
            if c not in row_pat.columns:
                row_pat[c] = 0.0
        num_cols = [c for c in pat_expected if c not in ["symbol", "date"]]
        if num_cols:
            row_pat[num_cols] = row_pat[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_pat = align_X(row_pat, pat_expected, "PATTERN")
    p_pat = float(clip01(proba(m_pat, X_pat))[0])

    g_sent = float(compute_gate_from_frame(row_sent)[0])

    meta = pd.DataFrame([{
        "p_ohlcv": p_ohlcv,
        "p_sent": p_sent,
        "p_pat": p_pat,
        "g_sent": g_sent,
        "p_ohlcv_x_sent": p_ohlcv * p_sent,
    }])
    p_fusion = float(clip01(proba(m_fusion, meta[["p_ohlcv","p_sent","p_pat","g_sent","p_ohlcv_x_sent"]]))[0])

    if model_choice == "OHLCV only":
        p_up = p_ohlcv
        model_used = "OHLCV"
    else:
        p_up = p_fusion
        model_used = "Fusion(meta)"

    # ---- Decision with optional NO-TRADE band ----
    conf = abs(float(p_up) - 0.5)
    if no_trade and (conf < float(delta)):
        pred = -1
    else:
        pred = int(p_up >= float(threshold))

    pred_label = "NO TRADE" if pred == -1 else ("UP" if pred == 1 else "DOWN")
    st.success(f"Prediction using **{model_used}**: **{pred_label}** (p_up={p_up:.4f})")

    with st.expander("Show expert probabilities (debug)"):
        st.write({
            "p_ohlcv": p_ohlcv,
            "p_sent": p_sent,
            "p_pat": p_pat,
            "g_sent": g_sent,
            "p_fusion": p_fusion,
            "threshold": float(threshold),
            "no_trade": bool(no_trade),
            "delta": float(delta),
            "confidence(|p-0.5|)": float(conf),
        })

    is_correct = None
    if actual is not None and pred != -1:
        is_correct = bool(pred == actual)
        st.write(f"Actual: **{'UP' if actual==1 else 'DOWN'}** → {'✅ Correct' if is_correct else '❌ Incorrect'}")
    elif actual is None:
        st.info("Actual label missing/NaN for this row; history scoring will ignore it.")
    else:
        st.info("NO TRADE: row will not be counted towards accuracy.")

    # Save to history
    new_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "model": model_used,
        "symbol": symbol,
        "date": d.date().isoformat(),
        "p_up": float(p_up),
        "pred": int(pred),
        "actual": (int(actual) if actual is not None else None),
        "is_correct": (bool(is_correct) if is_correct is not None else None),
    }])
    # Avoid FutureWarning on concat with empty/all-NA history
    if st.session_state["history"].empty:
        st.session_state["history"] = new_row.reset_index(drop=True)
    else:
        st.session_state["history"] = pd.concat([st.session_state["history"], new_row], ignore_index=True)
    st.subheader("Prediction history")
    hist = st.session_state["history"].copy()
    st.dataframe(hist.tail(100), width='stretch')
    scored = hist.dropna(subset=["is_correct"]).copy()
    stats = history_stats(scored)
    
    a, b, c, d = st.columns(4)
    a.metric("Total scored", stats["total"])
    b.metric("Correct", stats["correct"])
    c.metric("Incorrect", stats["incorrect"])
    d.metric("Accuracy", f"{stats['accuracy']*100:.2f}%" if stats["total"] > 0 else "N/A")
# -----------------------
# Tab 2: Backtest-lite (date range)
# -----------------------
with tab_range:
    st.subheader("Backtest-lite")
    st.caption("Runs predictions over a date range and reports accuracy + probability diagnostics. This is NOT a full backtest (no slippage/costs).")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        symbols_sel = st.multiselect("Symbols (choose 1+)", symbols, default=[symbols[0]], key="range_symbols")
    with c2:
        start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="range_start")
        end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="range_end")
    with c3:
        mode = st.radio("Predictor", ["OHLCV", "FUSION"], index=0, key="range_mode")
        max_rows = st.number_input("Max rows (safety)", min_value=200, max_value=200000, value=20000, step=200, key="range_maxrows")

    if st.button("Run backtest-lite", key="range_run"):
        if not symbols_sel:
            st.error("Select at least one symbol.")
            st.stop()
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
        if sd > ed:
            st.error("Start date must be <= end date.")
            st.stop()

        # Filter OHLCV slice
        sl = ohlcv_df[
            (ohlcv_df["symbol"].isin(symbols_sel)) &
            (ohlcv_df["date"] >= sd) &
            (ohlcv_df["date"] <= ed)
        ].copy().sort_values(["symbol","date"])

        if len(sl) == 0:
            st.warning("No OHLCV rows in selected range.")
            st.stop()

        if len(sl) > max_rows:
            st.warning(f"Slice too large ({len(sl):,} rows). Truncating to first {max_rows:,}.")
            sl = sl.iloc[:int(max_rows)].copy()

        with st.spinner("Computing predictions..."):
            out = run_backtest_lite(
                ohlcv_slice=sl,
                sent_df=sent_df,
                pat_df=pat_df,
                label_col=label_col,
                threshold=float(threshold),
                no_trade=bool(no_trade),
                delta=float(delta),
                mode=("OHLCV" if mode == "OHLCV" else "FUSION"),
                m_ohlcv=m_ohlcv,
                m_sent=m_sent,
                m_pat=m_pat,
                m_fusion=m_fusion,
                ohlcv_expected=ohlcv_expected,
                sent_expected=sent_expected,
                pat_expected=pat_expected,
            )

        if out.empty:
            st.warning("No scoreable rows (label missing) in this range.")
            st.stop()

        # Metrics (score ONLY traded rows if NO-TRADE is enabled)
        total = len(out)
        traded_mask = out["pred"] != -1
        traded_n = int(traded_mask.sum())
        coverage = traded_n / total if total else float("nan")
        
        if traded_n > 0:
            acc_traded = float(out.loc[traded_mask, "is_correct"].mean())
            up_rate_traded = float((out.loc[traded_mask, "pred"] == 1).mean())
        else:
            acc_traded = float("nan")
            up_rate_traded = float("nan")
        
        st.success(
            f"Traded accuracy: **{acc_traded*100:.2f}%** on **{traded_n:,}/{total:,}** rows "
            f"(coverage={coverage*100:.2f}%, threshold={float(threshold):.2f}, "
            f"{'NO-TRADE δ=' + str(float(delta)) if no_trade else 'no NO-TRADE'})"
        )
        st.write(f"UP prediction rate (traded only): **{up_rate_traded*100:.2f}%**")

        # Probability diagnostics
        st.write("Probability diagnostics:")
        st.write({
            **summarize_probs("p_ohlcv", out["p_ohlcv"].values),
            **summarize_probs("p_fusion", out["p_fusion"].values),
            **summarize_probs("p_up_used", out["p_up"].values),
        })

        # Confusion (simple)
        cm = pd.crosstab(
            out.loc[out["pred"] != -1, "actual"],
            out.loc[out["pred"] != -1, "pred"],
            rownames=["Actual"], colnames=["Pred"], dropna=False
        )
        st.write("Confusion matrix (0=DOWN, 1=UP):")
        st.dataframe(cm, width='stretch')
        if no_trade:
            st.write(f"NO-TRADE count: **{int((out['pred'] == -1).sum()):,}**")

        # Plot: accuracy over time (rolling)
        st.subheader("Rolling accuracy (by date)")
        tmp = out.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp_traded = tmp[tmp["pred"] != -1].copy()
        daily = tmp_traded.groupby("date", as_index=False)["is_correct"].mean()
        daily["roll_20d"] = daily["is_correct"].rolling(20, min_periods=5).mean()

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(daily["date"], daily["is_correct"], marker="o")
        plt.title("Daily accuracy")
        plt.xlabel("Date")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        st.pyplot(plt)

        plt.figure()
        plt.plot(daily["date"], daily["roll_20d"])
        plt.title("Rolling accuracy (20D)")
        plt.xlabel("Date")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        st.pyplot(plt)

        # Plot: probability histograms (used p_up)
        st.subheader("Probability distribution (p_up used)")
        plt.figure()
        plt.hist(out["p_up"].values, bins=40)
        plt.axvline(float(threshold), linestyle="--")
        plt.title("Histogram: p_up used")
        plt.xlabel("p_up")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(plt)

        st.subheader("Sample rows")
        st.dataframe(out.tail(200), width='stretch')
