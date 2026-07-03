from v3_model_comparison_tuning.threshold import THRESHOLD_COLUMNS, predict_with_threshold, sweep_thresholds


def test_sweep_thresholds_computes_expected_metrics():
    y_true = [0, 0, 1, 1]
    probabilities = [0.2, 0.6, 0.7, 0.9]

    table = sweep_thresholds(y_true, probabilities, thresholds=[0.5, 0.8])

    at_05 = table[table["threshold"] == 0.5].iloc[0]
    assert at_05["precision"] == 2 / 3
    assert at_05["recall"] == 1.0
    assert at_05["false_positive_count"] == 1
    assert at_05["false_negative_count"] == 0

    at_08 = table[table["threshold"] == 0.8].iloc[0]
    assert at_08["precision"] == 1.0
    assert at_08["recall"] == 0.5
    assert at_08["false_positive_count"] == 0
    assert at_08["false_negative_count"] == 1


def test_threshold_table_has_expected_columns_and_no_nans():
    table = sweep_thresholds([0, 1], [0.1, 0.9], thresholds=[0.5])

    assert list(table.columns) == THRESHOLD_COLUMNS
    assert not table.isna().any().any()


def test_predict_with_threshold_is_inclusive():
    assert predict_with_threshold(0.5, 0.5) == 1
    assert predict_with_threshold(0.49, 0.5) == 0
