import pandas as pd

from common.io_utils import load_artifact, load_dataset, save_artifact


def test_load_dataset_maps_text_and_label_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("text,label\n" "Click here to win now,1\n" "See you at 2pm,0\n")

    df = load_dataset(csv_path)

    assert list(df.columns) == ["subject", "body", "label"]
    assert (df["subject"] == "").all()
    assert df.iloc[0]["body"] == "Click here to win now"
    assert df.iloc[0]["label"] == "spam"
    assert df.iloc[1]["label"] == "ham"


def test_save_and_load_artifact_roundtrip(tmp_path):
    artifact_path = tmp_path / "nested" / "artifact.pkl"
    obj = {"hello": "world", "numbers": [1, 2, 3]}

    save_artifact(obj, artifact_path)
    loaded = load_artifact(artifact_path)

    assert artifact_path.exists()
    assert loaded == obj
