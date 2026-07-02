import pandas as pd

from v1_basic_pipeline.train import split_dataset


def test_split_dataset_preserves_class_ratio():
    # 90 ham (0), 10 spam (1) - a 10% minority class, like real spam datasets.
    y = pd.Series([0] * 90 + [1] * 10)
    X = pd.Series([f"email {i}" for i in range(100)])

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    overall_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert abs(train_ratio - overall_ratio) < 0.05
    assert abs(test_ratio - overall_ratio) < 0.05
    assert len(X_test) == len(y_test) == 20  # TEST_SIZE=0.2 of 100
