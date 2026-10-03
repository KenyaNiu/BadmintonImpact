from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


def _figure_module():
    path = Path(__file__).parents[1] / "scripts" / "generate_result_figures.py"
    spec = spec_from_file_location("generate_result_figures", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reliability_points_skip_empty_bins_and_include_one() -> None:
    module = _figure_module()
    frame = pd.DataFrame({"y_prob": [0.02, 0.08, 0.55, 1.0], "y_true": [0, 1, 1, 1]})

    predicted, observed = module._reliability_points(frame, n_bins=10)

    assert predicted.tolist() == [0.05, 0.55, 1.0]
    assert observed.tolist() == [0.5, 1.0, 1.0]


def test_journal_figure_style_uses_at_least_eight_point_text() -> None:
    module = _figure_module()

    module._style()

    assert module.plt.rcParams["font.size"] >= 8
    assert module.plt.rcParams["xtick.labelsize"] >= 8
    assert module.plt.rcParams["ytick.labelsize"] >= 8
    assert module.plt.rcParams["legend.fontsize"] >= 8
