import os
from typing import List, Optional, Dict
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
    if expected_cols is None:
        # Fallback: use all numeric cols except date
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

def proba(model, X: pd.DataFrame) -> float:
    # sklearn-like
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    # pyfunc-like
    pred = model.predict(X)
    if isinstance(pred, (pd.DataFrame, pd.Series)):
        pred = pred.values
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] >= 2:
        return float(pred[0, 1])
    return float(pred.ravel()[0])

def get_row(df: pd.DataFrame, symbol: str, date: pd.Timestamp) -> Optional[pd.DataFrame]:
    out = df[(df["symbol"] == symbol) & (df["date"] == date)]
    if len(out) == 0:
        return None
    return out.iloc[[0]].copy()

def compute_gate(row_sent: Optional[pd.DataFrame]) -> float:
    if row_sent is None:
        return 0.0
    # If explicit gate exists
    if "g_sent" in row_sent.columns:
        try:
            return float(row_sent["g_sent"].iloc[0])
        except Exception:
            pass
    # Derive from common cols
    for gate_col in ["news_present_lag_3", "news_present_lag_1", "article_count_lag_3", "article_count_lag_1", "article_count"]:
        if gate_col in row_sent.columns:
            try:
                return 1.0 if float(row_sent[gate_col].iloc[0]) > 0.0 else 0.0
            except Exception:
                return 0.0
    return 0.0

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

    # Your exact registry models (defaults)
    ohlcv_uri = st.text_input(
        "OHLCV model URI",
        value=os.environ.get("OHLCV_MODEL_URI", "models:/ohlcv_lr_multisym_2019_2023_persymnorm_true/Production")
    )
    fusion_uri = st.text_input(
        "Fusion model URI",
        value=os.environ.get("FUSION_MODEL_URI", "models:/fusion_meta_lr_multisym/Production")
    )
    pat_uri = st.text_input(
        "Pattern model URI",
        value=os.environ.get("PAT_MODEL_URI", "models:/pattern_lr_multisym_2019_2023/Production")
    )
    sent_uri = st.text_input(
        "Sentiment model URI",
        value=os.environ.get("SENT_MODEL_URI", "models:/sentiment_lr_baseline_multisym/Production")
    )

    st.divider()
    st.header("Feature data paths")
    # Adjust paths if your Streamlit deploy bundle differs
    ohlcv_path = st.text_input("OHLCV features parquet", value=os.environ.get("OHLCV_FEATS_PATH", "data/features/ohlcv_features.parquet"))
    sent_path  = st.text_input("Sentiment features parquet", value=os.environ.get("SENT_FEATS_PATH", "data/features/news_sentiment_features.parquet"))
    pat_path   = st.text_input("Pattern features parquet", value=os.environ.get("PAT_FEATS_PATH", "data/features/pattern_features.parquet"))

    label_col = st.text_input("Label column", value=os.environ.get("LABEL_COL", "y_up_5d"))

    st.divider()
    st.header("Prediction")
    model_choice = st.radio("Choose predictor", ["OHLCV only", "Fusion meta"], index=0)
    threshold = st.slider("Decision threshold (UP if p >= threshold)", 0.40, 0.60, 0.50, 0.01)

    st.divider()
    st.header("History")
    if st.button("Reset history"):
        st.session_state["history"] = init_history()

# Init history
if "history" not in st.session_state:
    st.session_state["history"] = init_history()

# -----------------------
# Cache: data + models
# -----------------------
# @st.cache_data
# def load_parquet(path: str) -> pd.DataFrame:
#     df = pd.read_parquet(path)
#     df["date"] = pd.to_datetime(df["date"])
#     return df

@st.cache_resource
def aws_assume_role_env():
    # Base creds come from Streamlit secrets
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

    # Set assumed-role creds into env so DVC/boto3 picks them up
    os.environ["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = creds["SessionToken"]

aws_assume_role_env()

@st.cache_data
def load_parquet(path: str, repo: str = ".", rev: str | None = None) -> pd.DataFrame:
    path = str(path)

    # If file already exists locally, use it
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # Otherwise, stream from DVC remote via dvc.api.open() and save locally
    cache_dir = Path(tempfile.gettempdir()) / "dvc_streamlit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_target = cache_dir / Path(path).name

    # If we already fetched it earlier, reuse
    if not local_target.exists():
        with dvc.api.open(path=path, repo=repo, rev=rev, mode="rb") as src:
            with open(local_target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    df = pd.read_parquet(local_target)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_model(uri: str):
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception:
        return mlflow.pyfunc.load_model(uri)

# Load datasets
with st.spinner("Loading datasets..."):
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

# Fusion expects meta schema (fixed)
FUSION_META_COLS = ["p_ohlcv", "p_sent", "p_pat", "g_sent", "p_ohlcv_x_sent"]

# -----------------------
# UI Inputs
# -----------------------
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Inputs")
    symbol = st.selectbox("Symbol", symbols, index=0)
    date = st.date_input("Date", value=max_date, min_value=min_date, max_value=max_date)

with c2:
    st.subheader("Dataset info")
    st.write(f"OHLCV rows: **{len(ohlcv_df):,}**")
    st.write(f"Date range: **{min_date} → {max_date}**")
    if label_col in ohlcv_df.columns:
        st.write(f"Label column: **{label_col}**")
    else:
        st.warning(f"Label `{label_col}` not found in OHLCV data; actual comparison disabled.")

# -----------------------
# Prediction
# -----------------------
if st.button("Predict"):
    d = pd.to_datetime(date)

    row_ohlcv = get_row(ohlcv_df, symbol, d)
    if row_ohlcv is None:
        st.error(f"No OHLCV row for {symbol} on {d.date()}")
        st.stop()

    row_sent = get_row(sent_df, symbol, d)  # can be None on no-news days
    row_pat  = get_row(pat_df,  symbol, d)
    if row_pat is None:
        # If pattern is missing for date, fall back to OHLCV row (only if pattern cols exist there)
        row_pat = row_ohlcv.copy()

    # Actual label if available
    actual = None
    if label_col in row_ohlcv.columns and not pd.isna(row_ohlcv[label_col].iloc[0]):
        actual = int(row_ohlcv[label_col].iloc[0])

    # Build model inputs
    X_ohlcv = align_X(row_ohlcv, ohlcv_expected, "OHLCV")
    p_ohlcv = proba(m_ohlcv, X_ohlcv)

    # Sentiment proba: if row missing, create neutral row
    if row_sent is None:
        # Create minimal row with required cols as zeros; will produce ~0.5 prob in LR
        row_sent = pd.DataFrame({"symbol":[symbol], "date":[d]})
    if sent_expected is not None:
        for c in sent_expected:
            if c not in row_sent.columns:
                row_sent[c] = 0.0
    X_sent = align_X(row_sent, sent_expected, "SENTIMENT")
    p_sent = proba(m_sent, X_sent)

    # Pattern proba
    if pat_expected is not None:
        for c in pat_expected:
            if c not in row_pat.columns:
                row_pat[c] = 0.0
    X_pat = align_X(row_pat, pat_expected, "PATTERN")
    p_pat = proba(m_pat, X_pat)

    g_sent = compute_gate(row_sent)

    # Fusion proba
    meta_row = pd.DataFrame([{
        "p_ohlcv": float(p_ohlcv),
        "p_sent": float(p_sent),
        "p_pat": float(p_pat),
        "g_sent": float(g_sent),
        "p_ohlcv_x_sent": float(p_ohlcv) * float(p_sent),
    }])
    X_fus = meta_row[FUSION_META_COLS].copy()
    p_fusion = proba(m_fusion, X_fus)

    # Choose final
    if model_choice == "OHLCV only":
        p_up = p_ohlcv
        model_used = "OHLCV"
    else:
        p_up = p_fusion
        model_used = "Fusion(meta)"

    pred = int(p_up >= threshold)
    pred_label = "UP" if pred == 1 else "DOWN"
    st.success(f"Prediction using **{model_used}**: **{pred_label}** (p_up={p_up:.4f})")

    with st.expander("Show expert probabilities"):
        st.write({
            "p_ohlcv": float(p_ohlcv),
            "p_sent": float(p_sent),
            "p_pat": float(p_pat),
            "g_sent": float(g_sent),
            "p_fusion": float(p_fusion),
        })

    # Compare actual
    is_correct = None
    if actual is not None:
        actual_label = "UP" if actual == 1 else "DOWN"
        is_correct = (pred == actual)
        st.write(f"Actual: **{actual_label}** → {'✅ Correct' if is_correct else '❌ Incorrect'}")
    else:
        st.info("Actual label not available (missing/NaN). Accuracy stats will ignore this row.")

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

    st.session_state["history"] = pd.concat([st.session_state["history"], new_row], ignore_index=True)

# -----------------------
# History + stats
# -----------------------
st.subheader("Prediction history")
hist = st.session_state["history"].copy()
st.dataframe(hist.tail(100), use_container_width=True)

hist_scored = hist.dropna(subset=["is_correct"]).copy()
stats = history_stats(hist_scored)

a, b, c, d = st.columns(4)
a.metric("Total scored", stats["total"])
b.metric("Correct", stats["correct"])
c.metric("Incorrect", stats["incorrect"])
d.metric("Accuracy", f"{stats['accuracy']*100:.2f}%" if stats["total"] > 0 else "N/A")