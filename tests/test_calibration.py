"""Calibration mappings must fit validation rows and return both resolutions."""

from badminton_impact_ai.metrics import calibrate_fold_predictions


def test_calibration_returns_named_methods() -> None:
    rows = []
    for split in ("val", "test"):
        for index in range(20):
            probability = 0.1 + 0.8 * index / 19
            rows.append(
                {
                    "split": split,
                    "unique_impact_key_candidate": f"{split}_{index}",
                    "y_true_cls": int(index >= 10),
                    "y_prob": probability,
                    "cls_logit": __import__("math").log(probability / (1 - probability)),
                }
            )
    result = calibrate_fold_predictions(rows)
    assert {row["calibration_method"] for row in result} == {"none", "probability_logistic", "temperature", "isotonic"}
    assert {row["resolution"] for row in result} == {"view", "unique_impact"}
