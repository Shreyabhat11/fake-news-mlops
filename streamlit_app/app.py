"""
streamlit_app/app.py
=====================
Streamlit dashboard for the Fake News Trend Drift Detector.

Pages:
    🔍 Predict         — interactive fake news classifier
    📊 Drift Monitor   — drift scores and charts
    🤖 Model Registry  — MLflow model versions
    📈 Live Metrics    — Prometheus metrics viewer
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

st.set_page_config(
    page_title="Fake News MLOps Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fake News Detector")
    st.caption("MLOps Monitoring Dashboard")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🔍 Predict", "📊 Drift Monitor", "🤖 Model Registry", "📈 Live Metrics"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"API: `{API_BASE_URL}`")

    # API health indicator
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3).json()
        api_status = health.get("status", "unknown")
        color = "green" if api_status == "healthy" else "orange"
        st.markdown(f"API status: :{color}[{api_status.upper()}]")
    except Exception:
        st.markdown("API status: :red[OFFLINE]")


# ── Helper functions ───────────────────────────────────────────────────────────

def api_predict(text: str, title: str = "") -> dict:
    payload = {"text": text, "title": title if title else None}
    try:
        r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_batch_predict(texts: list) -> dict:
    payload = {"texts": texts}
    try:
        r = requests.post(f"{API_BASE_URL}/batch_predict", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_drift_status() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/drift/status", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "unknown", "overall_drift_score": 0.0}


def get_model_info() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ── Page: Predict ──────────────────────────────────────────────────────────────

if page == "🔍 Predict":
    st.title("🔍 Fake News Classifier")
    st.markdown("Enter a news article to classify it as **FAKE** or **REAL**.")

    col1, col2 = st.columns([2, 1])

    with col1:
        title_input = st.text_input("Article Title (optional)")
        text_input = st.text_area(
            "Article Text",
            height=200,
            placeholder="Paste the news article text here ...",
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            predict_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

        # Demo examples
        with col_btn2:
            demo = st.selectbox(
                "Or try a demo",
                [
                    "— select example —",
                    "FAKE: Conspiracy example",
                    "REAL: Reuters example",
                ],
                label_visibility="collapsed",
            )

    if demo == "FAKE: Conspiracy example":
        text_input = (
            "SHOCKING: Government secretly admits to hiding UFO technology! "
            "Deep state operatives have confirmed that the mainstream media has been "
            "suppressing this information for decades. Anonymous sources reveal that "
            "miracle cures are being kept from the public to protect Big Pharma profits."
        )
        title_input = "Government Hides Shocking Truth About UFOs"

    elif demo == "REAL: Reuters example":
        text_input = (
            "The Federal Reserve held interest rates steady on Wednesday and signaled "
            "it remains in no hurry to resume cutting borrowing costs, as policymakers "
            "evaluate the evolving economic outlook and the potential effects of new "
            "government policies. The central bank kept its benchmark overnight interest "
            "rate in the 4.25%-4.50% range."
        )
        title_input = "Fed holds rates steady, signals cautious approach"

    with col2:
        st.subheader("Batch Predict")
        batch_text = st.text_area(
            "One article per line",
            height=200,
            placeholder="Article 1\nArticle 2\nArticle 3",
        )
        batch_btn = st.button("🔄 Batch Classify", use_container_width=True)

    # ── Single predict ─────────────────────────────────────────────────────────
    if predict_btn and text_input.strip():
        with st.spinner("Classifying..."):
            result = api_predict(text_input, title_input)

        if "error" in result:
            st.error(f"API error: {result['error']}")
        else:
            st.divider()
            label = result["prediction"]
            confidence = result["confidence"]
            fake_prob = result["fake_probability"]
            real_prob = result["real_probability"]

            # Big verdict
            if label == "FAKE":
                st.error(f"### 🚨 FAKE NEWS  —  {confidence:.1%} confidence")
            else:
                st.success(f"### ✅ REAL NEWS  —  {confidence:.1%} confidence")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Fake probability", f"{fake_prob:.2%}")
            col_m2.metric("Real probability", f"{real_prob:.2%}")
            col_m3.metric("Drift status", result.get("drift_status", "—").upper())
            col_m4.metric("Latency", f"{result.get('processing_time_ms', 0):.0f} ms")

            # Probability bar chart
            fig = go.Figure(go.Bar(
                x=["FAKE", "REAL"],
                y=[fake_prob, real_prob],
                marker_color=["#E24B4A", "#639922"],
                text=[f"{fake_prob:.1%}", f"{real_prob:.1%}"],
                textposition="auto",
            ))
            fig.update_layout(
                title="Prediction probability", yaxis_range=[0, 1],
                height=250, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Batch predict ──────────────────────────────────────────────────────────
    if batch_btn and batch_text.strip():
        texts = [t.strip() for t in batch_text.split("\n") if t.strip()]
        with st.spinner(f"Classifying {len(texts)} articles ..."):
            result = api_batch_predict(texts)

        if "error" in result:
            st.error(result["error"])
        else:
            preds = result.get("predictions", [])
            df = pd.DataFrame([{
                "#": i + 1,
                "Verdict": p["prediction"],
                "Confidence": f"{p['confidence']:.2%}",
                "Fake %": f"{p['fake_probability']:.1%}",
            } for i, p in enumerate(preds)])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Processed {result.get('total_processed', 0)} articles in "
                       f"{result.get('processing_time_ms', 0):.0f} ms")


# ── Page: Drift Monitor ────────────────────────────────────────────────────────

elif page == "📊 Drift Monitor":
    st.title("📊 Drift Monitoring")

    drift = get_drift_status()
    status = drift.get("status", "unknown")
    score = drift.get("overall_drift_score", 0.0)
    ts = drift.get("timestamp", "—")

    # Status banner
    if status == "critical":
        st.error(f"🔴 CRITICAL DRIFT — Score: {score:.4f}")
    elif status == "warning":
        st.warning(f"🟡 DRIFT WARNING — Score: {score:.4f}")
    elif status == "normal":
        st.success(f"🟢 No drift detected — Score: {score:.4f}")
    else:
        st.info("No drift report available yet. Run the monitoring pipeline.")

    if ts != "—":
        st.caption(f"Last checked: {ts}")

    if drift.get("should_retrain"):
        st.warning("⚠️ Auto-retraining should trigger on next orchestration cycle.")

    st.divider()

    # Detailed breakdown
    col1, col2, col3 = st.columns(3)

    emb = drift.get("embedding_drift", {})
    with col1:
        st.subheader("Embedding Drift")
        st.metric("Drift score", f"{emb.get('drift_score', 0):.4f}",
                  delta="⚠️ drifted" if emb.get("is_drifted") else "✅ normal",
                  delta_color="inverse")
        st.metric("Cosine distance", f"{emb.get('mean_cosine_distance', 0):.4f}")
        st.metric("PCA drift fraction", f"{emb.get('drift_fraction_pca', 0):.4f}")

    pred = drift.get("prediction_drift", {})
    with col2:
        st.subheader("Prediction Drift")
        st.metric("Prediction shift", f"{pred.get('prediction_shift', 0):.4f}",
                  delta="⚠️ drifted" if pred.get("is_drifted") else "✅ normal",
                  delta_color="inverse")
        st.metric("Reference FAKE%", f"{pred.get('reference_fake_pct', 0):.2%}")
        st.metric("Current FAKE%", f"{pred.get('current_fake_pct', 0):.2%}")

    conf = drift.get("confidence_drift", {})
    with col3:
        st.subheader("Confidence Drift")
        st.metric("Confidence drop", f"{conf.get('confidence_drop', 0):.4f}",
                  delta="⚠️ degraded" if conf.get("is_drifted") else "✅ stable",
                  delta_color="inverse")
        st.metric("Reference confidence", f"{conf.get('reference_mean_confidence', 0):.4f}")
        st.metric("Current confidence", f"{conf.get('current_mean_confidence', 0):.4f}")

    # Simulated drift trend chart
    st.divider()
    st.subheader("Drift Score Trend (simulated)")
    import numpy as np
    np.random.seed(42)
    n = 48
    dates = pd.date_range(end=datetime.utcnow(), periods=n, freq="30min")
    scores = np.clip(
        np.cumsum(np.random.randn(n) * 0.01) + np.random.uniform(0.05, 0.10), 0, 0.5
    )
    trend_df = pd.DataFrame({"time": dates, "drift_score": scores})
    fig = px.line(trend_df, x="time", y="drift_score",
                  title="Drift score over last 24h")
    fig.add_hline(y=0.15, line_dash="dash", line_color="orange",
                  annotation_text="Warning threshold")
    fig.add_hline(y=0.30, line_dash="dash", line_color="red",
                  annotation_text="Critical threshold")
    st.plotly_chart(fig, use_container_width=True)

    if drift.get("html_report"):
        st.info(f"📄 Full Evidently report: `{drift['html_report']}`")


# ── Page: Model Registry ───────────────────────────────────────────────────────

elif page == "🤖 Model Registry":
    st.title("🤖 Model Registry")
    model_info = get_model_info()

    if model_info:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Version", model_info.get("model_version", "—"))
        col2.metric("Classifier", model_info.get("classifier", "—"))
        col3.metric("Val F1", f"{model_info.get('val_f1', 0):.4f}" if model_info.get("val_f1") else "—")
        col4.metric("Test F1", f"{model_info.get('test_f1', 0):.4f}" if model_info.get("test_f1") else "—")

        st.caption(f"Loaded at: {model_info.get('loaded_at', '—')}")
        if model_info.get("canary_mode"):
            st.warning("🐤 Canary mode active — partial traffic to new model")
        if model_info.get("shadow_mode"):
            st.info("👥 Shadow mode active — challenger model logging predictions")
    else:
        st.warning("Could not reach the API. Is it running?")

    st.divider()
    st.subheader("Hot Reload Model")
    st.markdown("Force the API to reload the latest `models/model.pkl` without restarting.")
    if st.button("🔄 Reload Model", type="secondary"):
        try:
            r = requests.post(f"{API_BASE_URL}/model/reload", timeout=30)
            if r.status_code == 200:
                st.success(f"✅ Model reloaded: {r.json()}")
            else:
                st.error(f"Failed: {r.text}")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.markdown(f"📊 [Open MLflow UI]({MLFLOW_TRACKING_URI})", unsafe_allow_html=True)


# ── Page: Live Metrics ─────────────────────────────────────────────────────────

elif page == "📈 Live Metrics":
    st.title("📈 Live Metrics")
    st.markdown("Real-time view of Prometheus metrics from the inference API.")

    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)

    try:
        raw = requests.get(f"{API_BASE_URL}/metrics", timeout=10).text
        metrics_text = raw
    except Exception:
        metrics_text = "# Could not reach /metrics endpoint"

    def parse_metric(text, name):
        for line in text.split("\n"):
            if line.startswith(name) and not line.startswith("#"):
                parts = line.split(" ")
                if len(parts) >= 2:
                    try:
                        return float(parts[-1])
                    except Exception:
                        pass
        return None

    total_preds = parse_metric(metrics_text, "fakenews_predictions_total")
    drift_score_raw = parse_metric(metrics_text, "fakenews_drift_score")
    retrain_events = parse_metric(metrics_text, "fakenews_retrain_events_total")
    cache_hits = parse_metric(metrics_text, "fakenews_cache_hits_total")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total predictions", f"{total_preds:.0f}" if total_preds else "—")
    col2.metric("Current drift score", f"{drift_score_raw:.4f}" if drift_score_raw else "—")
    col3.metric("Retrain events", f"{retrain_events:.0f}" if retrain_events else "—")
    col4.metric("Cache hits", f"{cache_hits:.0f}" if cache_hits else "—")

    st.divider()
    with st.expander("Raw Prometheus metrics"):
        st.code(metrics_text[:3000], language="text")

    if auto_refresh:
        time.sleep(5)
        st.rerun()
