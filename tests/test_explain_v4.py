from v4_streamlit_app.explain import FALLBACK_EXPLANATION, build_explanation


def test_link_count_produces_link_bullet():
    bullets = build_explanation("", metadata={"link_count": 2})

    assert any("2 link(s)" in bullet for bullet in bullets)


def test_all_zero_metadata_produces_fallback():
    bullets = build_explanation(
        "",
        metadata={
            "link_count": 0,
            "exclamation_count": 0,
            "uppercase_ratio": 0,
            "currency_symbol_count": 0,
        },
    )

    assert bullets == [FALLBACK_EXPLANATION]


def test_suspicious_word_bullet_lists_matched_words():
    bullets = build_explanation("Claim your FREE prize", metadata={})

    suspicious_bullet = next(bullet for bullet in bullets if "suspicious word" in bullet)
    assert "claim" in suspicious_bullet
    assert "free" in suspicious_bullet
    assert "prize" in suspicious_bullet
