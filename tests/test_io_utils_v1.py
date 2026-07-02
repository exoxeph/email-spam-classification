import pandas as pd

from common.io_utils import load_artifact, load_dataset, save_artifact


def test_load_dataset_renames_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "Subject,Message,Spam/Ham,Date\n"
        "Win now,Click here,spam,2006-01-01\n"
        "Meeting,See you at 2pm,ham,2006-01-02\n"
    )

    df = load_dataset(csv_path)

    assert list(df.columns) == ["subject", "body", "label"]
    assert df.iloc[0]["subject"] == "Win now"
    assert df.iloc[0]["body"] == "Click here"
    assert df.iloc[0]["label"] == "spam"


def test_save_and_load_artifact_roundtrip(tmp_path):
    artifact_path = tmp_path / "nested" / "artifact.pkl"
    obj = {"hello": "world", "numbers": [1, 2, 3]}

    save_artifact(obj, artifact_path)
    loaded = load_artifact(artifact_path)

    assert artifact_path.exists()
    assert loaded == obj
