import pandas as pd

from common.preprocess import build_dataset, clean_text, map_label


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_strips_extra_whitespace():
    assert clean_text("hello    world\n\nfoo") == "hello world foo"
    assert clean_text("  padded text  ") == "padded text"


def test_map_label_spam_to_1_and_ham_to_0():
    assert map_label("spam") == 1
    assert map_label("ham") == 0
    assert map_label("SPAM") == 1
    assert map_label("Ham") == 0


def test_map_label_rejects_unknown_value():
    import pytest

    with pytest.raises(ValueError):
        map_label("unknown")


def test_build_dataset_drops_empty_rows():
    df = pd.DataFrame(
        {
            "subject": ["Meeting tomorrow", "", None],
            "body": ["See you at 2pm", "", "   "],
            "label": ["ham", "spam", "spam"],
        }
    )
    X, y = build_dataset(df)
    assert len(X) == 1
    assert X.iloc[0] == "meeting tomorrow see you at 2pm"
    assert y.iloc[0] == 0


def test_build_dataset_drops_duplicates():
    df = pd.DataFrame(
        {
            "subject": ["Win now", "Win now"],
            "body": ["Click here", "Click here"],
            "label": ["spam", "spam"],
        }
    )
    X, y = build_dataset(df)
    assert len(X) == 1
