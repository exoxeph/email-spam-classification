"""CLI smoke-test entry point for v4 helpers."""

from v3_model_comparison_tuning.predict import DEFAULT_MODEL_PATH, predict_email_risk
from v4_streamlit_app.explain import build_explanation


def main() -> None:
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train v3 first: missing model at {DEFAULT_MODEL_PATH}")

    sample = "URGENT: claim your free cash prize at https://example.com now!!!"
    result = predict_email_risk(sample)
    print(f"Prediction: {result['prediction_label']}")
    print(f"Probability: {result['spam_probability']:.2%}")
    print(f"Risk: {result['risk_level']}")
    print(f"Action: {result['recommended_action']}")
    print("Explanation:")
    for bullet in build_explanation(sample):
        print(f"- {bullet}")


if __name__ == "__main__":
    main()
