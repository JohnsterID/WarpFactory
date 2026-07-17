"""Tests for the shared explorer model and the Jupyter front end.

The model tests always run (no UI dependency). The JupyterExplorer
tests exercise the real widget tree headless via the Agg backend and
skip when ipywidgets is unavailable, mirroring the Qt test policy.
"""

import subprocess
import sys

import matplotlib
import numpy as np
import pytest

from warpfactory.interactive import METRIC_CATALOG, ExplorerModel

matplotlib.use("Agg")


def test_interactive_imports_without_ipywidgets():
    """The model must import and run when the jupyter extra is absent.

    Blocking ipywidgets in a subprocess simulates a core-only install
    and exercises the guarded-import path plus the placeholder error.
    """
    script = (
        "import sys\n"
        "sys.modules['ipywidgets'] = None\n"
        "from warpfactory.interactive import ExplorerModel, JupyterExplorer\n"
        "result = ExplorerModel().evaluate('Minkowski')\n"
        "assert abs(result.stress_energy['T_tt']).max() == 0.0\n"
        "try:\n"
        "    JupyterExplorer()\n"
        "except ImportError as err:\n"
        "    assert 'ipywidgets' in str(err)\n"
        "else:\n"
        "    raise AssertionError('placeholder must raise ImportError')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


class TestExplorerModel:
    def test_catalog_defaults_are_copies(self):
        model = ExplorerModel()
        defaults = model.defaults("Alcubierre")
        defaults["v_s"] = 99.0
        assert model.defaults("Alcubierre")["v_s"] == 2.0
        assert METRIC_CATALOG["Alcubierre"][1]["v_s"] == 2.0

    def test_minkowski_pipeline_is_vacuum(self):
        result = ExplorerModel().evaluate("Minkowski", diagnostics=True)
        assert np.allclose(result.stress_energy["T_tt"], 0.0, atol=1e-12)
        assert all(result.conditions.values())
        # No bubble parameters -> no quantum inequality verdict
        assert result.quantum_inequality is None
        assert len(result.horizons["ergosphere"]) == 0

    def test_alcubierre_diagnostics(self):
        result = ExplorerModel().evaluate("Alcubierre", diagnostics=True)
        # Superluminal bubble: energy conditions violated, ergosurface
        # present (g_tt = 0 crossing), frame dragging tilts the cones
        assert not result.conditions["weak"]
        assert not result.conditions["null"]
        assert len(result.horizons["ergosphere"]) > 0
        assert np.max(np.abs(result.light_cone_tilt)) > 0.1
        # Macroscopic wall is far above the Ford-Roman Planck limit
        assert not result.quantum_inequality["satisfied"]

    def test_diagnostics_off_by_default(self):
        result = ExplorerModel().evaluate("Alcubierre")
        assert result.conditions is None
        assert result.horizons is None
        assert result.quantum_inequality is None

    def test_custom_parameters_propagate(self):
        model = ExplorerModel()
        # sigma=4 keeps the wall inside the grid so the far field is
        # genuinely flat (see the tidal-force test for the same choice)
        slow = model.evaluate("Alcubierre", {"v_s": 0.5, "R": 1.0, "sigma": 4.0})
        fast = model.evaluate("Alcubierre", {"v_s": 4.0, "R": 1.0, "sigma": 4.0})
        assert slow.params["v_s"] == 0.5
        assert fast.params["v_s"] == 4.0
        # Faster bubble needs more (negative) wall energy
        assert (
            np.abs(fast.stress_energy["T_tt"]).max()
            > np.abs(slow.stress_energy["T_tt"]).max()
        )
        # Sharp-wall bubble is vacuum away from the wall
        assert np.allclose(slow.stress_energy["T_tt"][:20], 0.0, atol=1e-6)
        assert np.allclose(slow.stress_energy["T_tt"][-20:], 0.0, atol=1e-6)

    def test_custom_grid(self):
        x = np.linspace(-3.0, 3.0, 61)
        result = ExplorerModel(x=x).evaluate("Alcubierre")
        assert result.x.shape == (61,)
        assert result.stress_energy["T_tt"].shape == (61,)


ipywidgets = pytest.importorskip(
    "ipywidgets", reason="ipywidgets is not available; install the jupyter extra"
)

from warpfactory.interactive import JupyterExplorer  # noqa: E402


class TestJupyterExplorer:
    def test_initial_state(self):
        explorer = JupyterExplorer()
        assert explorer.metric_selector.value == "Alcubierre"
        assert explorer.get_parameters() == {"v_s": 2.0, "R": 1.0, "sigma": 0.5}
        assert explorer.result is not None
        assert "T_tt" in explorer.result.stress_energy
        # Diagnostics default on -> status shows the QI verdict
        assert "Ford-Roman QI" in explorer.status.value

    def test_metric_switch_rebuilds_sliders(self):
        explorer = JupyterExplorer()
        explorer.metric_selector.value = "Van Den Broeck"
        assert "B" in explorer.get_parameters()
        assert explorer.result.metric_name == "Van Den Broeck"
        explorer.metric_selector.value = "Minkowski"
        assert explorer.get_parameters() == {}
        assert len(explorer.parameter_box.children) == 0

    def test_slider_change_triggers_recompute(self):
        explorer = JupyterExplorer()
        rho_before = np.abs(explorer.result.stress_energy["T_tt"]).max()
        explorer.set_parameter("v_s", 4.0)
        rho_after = np.abs(explorer.result.stress_energy["T_tt"]).max()
        assert explorer.result.params["v_s"] == 4.0
        assert rho_after > rho_before

    def test_plot_layout_tracks_diagnostics(self):
        import matplotlib.pyplot as plt

        explorer = JupyterExplorer()
        fig = explorer.plot()
        assert len(fig.axes) == 3
        plt.close(fig)

        explorer.diagnostics_toggle.value = False
        fig = explorer.plot()
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_component_selector_changes_plot(self):
        import matplotlib.pyplot as plt

        explorer = JupyterExplorer()
        explorer.component_selector.value = "g_tx"
        fig = explorer.plot()
        line = fig.axes[0].lines[0]
        np.testing.assert_allclose(
            line.get_ydata(), np.asarray(explorer.result.metric["g_tx"], dtype=float)
        )
        plt.close(fig)

    def test_widget_tree_assembles(self):
        explorer = JupyterExplorer()
        tree = explorer.widget()
        assert isinstance(tree, ipywidgets.VBox)
        controls = tree.children[0]
        assert explorer.metric_selector in controls.children
        assert explorer.parameter_box in controls.children

    def test_sliders_recompute_on_release_by_default(self):
        # Per-drag-tick recomputes queue full pipeline+redraw runs in
        # the kernel and make the UI lag; sliders must default to
        # recompute-on-release.
        explorer = JupyterExplorer()
        assert all(
            not slider.continuous_update for slider in explorer._sliders.values()
        )
        eager = JupyterExplorer(continuous_update=True)
        assert all(slider.continuous_update for slider in eager._sliders.values())
        # The setting must survive a metric switch (sliders rebuild).
        explorer.metric_selector.value = "Van Den Broeck"
        assert all(
            not slider.continuous_update for slider in explorer._sliders.values()
        )

    def test_set_parameters_batches_into_one_recompute(self):
        explorer = JupyterExplorer()
        count_before = explorer.recompute_count
        result = explorer.set_parameters({"v_s": 4.0, "R": 2.0, "sigma": 1.0})
        assert explorer.recompute_count == count_before + 1
        assert result.params == {"v_s": 4.0, "R": 2.0, "sigma": 1.0}
        assert explorer.get_parameters() == {"v_s": 4.0, "R": 2.0, "sigma": 1.0}

    def test_set_parameter_still_recomputes_each_call(self):
        explorer = JupyterExplorer()
        count_before = explorer.recompute_count
        explorer.set_parameter("v_s", 3.0)
        explorer.set_parameter("R", 2.0)
        assert explorer.recompute_count == count_before + 2
