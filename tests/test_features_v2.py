import pandas as pd

from common.features import (
    METADATA_COLUMNS,
    build_column_transformer,
    build_feature_frame,
    build_metadata_features,
    count_currency_symbols,
    count_digits,
    count_exclamations,
    count_links,
    count_suspicious_words,
    uppercase_ratio,
)


def test_count_links_finds_common_variants():
    text = "Visit http://a.com, https://b.net, www.example.org and offer.com now"

    assert count_links(text) == 4


def test_count_exclamations_counts_mixed_punctuation():
    assert count_exclamations("Win now!!! Are you ready?!") == 4


def test_uppercase_ratio_handles_lowercase_uppercase_and_no_letters():
    assert uppercase_ratio("abc") == 0
    assert uppercase_ratio("ABC") == 1
    assert uppercase_ratio("123 !!!") == 0


def test_count_suspicious_words_is_case_insensitive_and_word_bounded():
    assert count_suspicious_words("FREE prize, click now. freeing") == 3


def test_count_currency_symbols_and_digits():
    assert count_currency_symbols("Win $100, \u00a320, \u20ac5, \u09f350") == 4
    assert count_digits("A1B22") == 3


def test_build_metadata_features_returns_expected_columns_without_nans():
    df = pd.DataFrame(
        {
            "subject": ["URGENT!!!", None],
            "body": ["Claim $100 at https://example.com", "plain note"],
        }
    )

    features = build_metadata_features(df)

    assert list(features.columns) == METADATA_COLUMNS
    assert not features.isna().any().any()
    assert features.loc[0, "subject_exclaim_count"] == 3
    assert features.loc[0, "link_count"] == 1


def test_build_feature_frame_adds_cleaned_combined_text_and_metadata():
    df = pd.DataFrame({"subject": ["Hello THERE"], "body": ["FREE   Money!!!"]})

    features = build_feature_frame(df)

    assert "combined_text" in features.columns
    assert features.loc[0, "combined_text"] == "hello there free money!!!"
    assert features.loc[0, "uppercase_ratio"] > 0


def test_build_column_transformer_has_expected_branches():
    transformer = build_column_transformer()

    assert [name for name, _, _ in transformer.transformers] == ["word_tfidf", "char_tfidf", "metadata"]
