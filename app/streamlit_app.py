"""Streamlit UI for the v4 email spam-risk triage prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_model_comparison_tuning.predict import (
    CLASSIFICATION_THRESHOLD,
    DEFAULT_MODEL_PATH,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    load_default_model,
    predict_email_risk,
)
from v4_streamlit_app.batch import predict_batch
from v4_streamlit_app.explain import build_explanation
from v4_streamlit_app.feedback import log_feedback

EXAMPLES = {
    "Safe email": "Hi team, sharing the updated project notes from today's planning meeting. Please review before Friday.",
    "Spam email": "URGENT!!! You are a WINNER. Claim your FREE cash prize now at https://example.com!!!",
    "Medium-risk email": "Your account needs verification. Please login and review the latest security notice.",
}

RISK_STYLE = {
    "Low": {"color": "#3DDC97", "bg": "rgba(61, 220, 151, 0.12)", "icon": "OK", "verdict": "ALLOW"},
    "Medium": {"color": "#F2C94C", "bg": "rgba(242, 201, 76, 0.12)", "icon": "!", "verdict": "WARN"},
    "High": {"color": "#EF4444", "bg": "rgba(239, 68, 68, 0.14)", "icon": "X", "verdict": "QUARANTINE"},
}

CUSTOM_CSS = """
<style>
.risk-banner {
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    border-left: 6px solid var(--risk-color);
    background: var(--risk-bg);
}
.risk-banner .verdict {
    font-family: monospace;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--risk-color);
}
.risk-banner .subtext {
    color: #9CA3AF;
    font-size: 0.9rem;
    margin-top: 0.2rem;
}
.gauge-track {
    width: 100%;
    height: 10px;
    border-radius: 6px;
    background: #262B33;
    overflow: hidden;
    margin-top: 0.6rem;
}
.gauge-fill {
    height: 100%;
    border-radius: 6px;
    background: var(--risk-color);
}
.explain-item {
    font-family: monospace;
    font-size: 0.92rem;
    padding: 0.25rem 0;
    color: #D1D5DB;
}
.model-chip {
    display: inline-block;
    font-family: monospace;
    font-size: 0.78rem;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: #1F242C;
    color: #9CA3AF;
    border: 1px solid #2A2F38;
    margin-right: 0.4rem;
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Email Spam Risk Triage", page_icon=":shield:", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not DEFAULT_MODEL_PATH.exists():
        st.error(f"Missing v3 model at `{DEFAULT_MODEL_PATH}`. Run `poetry run python -m v3_model_comparison_tuning.main` first.")
        st.stop()

    load_default_model()
    render_header()

    single_tab, batch_tab = st.tabs(["Single Email", "Batch CSV"])
    with single_tab:
        render_single_email()
    with batch_tab:
        render_batch_upload()


def render_header() -> None:
    title_col, chip_col = st.columns([3, 2])
    with title_col:
        st.title(":shield: Email Spam Risk Triage")
        st.caption("Paste an email to estimate whether it should be allowed, warned, or quarantined.")
    with chip_col:
        st.markdown(
            f"""
            <div style="text-align:right; padding-top: 1.2rem;">
                <span class="model-chip">model: linear_svm (v3)</span>
                <span class="model-chip">threshold: {CLASSIFICATION_THRESHOLD:.2f}</span>
                <span class="model-chip">test F1: 0.9891</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()


def render_single_email() -> None:
    st.subheader("Analyze one email")
    selected_example = st.radio("Load an example", ["Custom"] + list(EXAMPLES), horizontal=True)
    default_text = "" if selected_example == "Custom" else EXAMPLES[selected_example]
    subject = st.text_input("Subject", value="")
    text = st.text_area("Email body", value=default_text, height=200)

    if st.button("Analyze Email", type="primary", use_container_width=False):
        if not text.strip():
            st.warning("Enter email text before analyzing.")
            return
        result = predict_email_risk(text, subject=subject)
        st.session_state["last_email_text"] = text
        st.session_state["last_prediction"] = result

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        explanation = build_explanation(st.session_state["last_email_text"], subject=subject)
        render_risk_banner(result)
        render_metric_row(result)
        render_explanation(explanation)
        render_feedback(st.session_state["last_email_text"], result)


def render_risk_banner(result: dict) -> None:
    style = RISK_STYLE[result["risk_level"]]
    probability_pct = result["spam_probability"] * 100
    st.markdown(
        f"""
        <div class="risk-banner" style="--risk-color:{style['color']}; --risk-bg:{style['bg']};">
            <div class="verdict">{style['icon']} {style['verdict']}</div>
            <div class="subtext">{result['prediction_label']} &middot; {result['risk_level']} risk &middot; spam probability {probability_pct:.1f}%</div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{probability_pct:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(result: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Prediction", result["prediction_label"])
    cols[1].metric("Spam probability", f"{result['spam_probability']:.1%}")
    cols[2].metric("Risk level", result["risk_level"])
    cols[3].metric("Recommended action", result["recommended_action"])
    st.caption(
        f"Risk tiers: low < {LOW_RISK_THRESHOLD:.0%} spam probability, "
        f"medium {LOW_RISK_THRESHOLD:.0%}-{HIGH_RISK_THRESHOLD:.0%}, high >= {HIGH_RISK_THRESHOLD:.0%}."
    )


def render_explanation(explanation: list[str]) -> None:
    st.markdown("#### Why this verdict")
    for bullet in explanation:
        st.markdown(f'<div class="explain-item">&gt; {bullet}</div>', unsafe_allow_html=True)


def render_feedback(email_text: str, prediction: dict) -> None:
    st.markdown("#### Was this prediction correct?")
    yes_col, no_col, _ = st.columns([1, 1, 4])
    if yes_col.button(":thumbsup: Yes"):
        log_feedback(email_text, prediction, "yes")
        st.success("Feedback logged.")
    if no_col.button(":thumbsdown: No"):
        log_feedback(email_text, prediction, "no")
        st.success("Feedback logged.")


def render_batch_upload() -> None:
    st.subheader("Batch CSV upload")
    st.write("Upload a CSV with required columns `email_id` and `email_text`. An optional `subject` column is supported.")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        input_df = pd.read_csv(uploaded_file)
        results = predict_batch(input_df)
    except Exception as exc:
        st.error(str(exc))
        return

    render_batch_summary(results)
    st.dataframe(
        results,
        use_container_width=True,
        column_config={
            "spam_probability": st.column_config.ProgressColumn(
                "Spam probability", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
    st.download_button(
        "Download results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="spam_risk_results.csv",
        mime="text/csv",
    )


def render_batch_summary(results: pd.DataFrame) -> None:
    counts = results["risk_level"].value_counts()
    total = len(results)
    cols = st.columns(4)
    cols[0].metric("Total emails", total)
    cols[1].metric("High risk", int(counts.get("High", 0)))
    cols[2].metric("Medium risk", int(counts.get("Medium", 0)))
    cols[3].metric("Low risk", int(counts.get("Low", 0)))


if __name__ == "__main__":
    main()
