"""ipywidgets front end for interactive metric exploration in Jupyter.

The Jupyter explorer is the primary interactive UI for WarpFactory (the
Qt GUI is kept for compatibility but is maintenance-only). It goes
beyond the Qt feature set: alongside the metric-component and
stress-energy plots it shows light-cone tilt with horizon/ergosphere
markers, live energy-condition verdicts, and the Ford-Roman quantum
inequality assessment of the current bubble parameters.

Usage in a notebook:

    from warpfactory.interactive import JupyterExplorer
    JupyterExplorer().display()
"""

from __future__ import annotations

from typing import Dict, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from ..visualizer import ColorMaps
from .model import METRIC_CATALOG, ExplorationResult, ExplorerModel

# Slider ranges per known parameter: (min, max, step). Parameters not
# listed fall back to _DEFAULT_RANGE.
_PARAM_RANGES = {
    "v_s": (0.1, 10.0, 0.1),
    "R": (0.5, 5.0, 0.1),
    "sigma": (0.1, 8.0, 0.1),
    "B": (1.0, 5.0, 0.1),
    "thickness": (0.05, 1.0, 0.05),
}
_DEFAULT_RANGE = (0.1, 10.0, 0.1)

_COMPONENTS = ["g_tt", "g_tx", "g_xx", "g_yy", "g_zz"]


class JupyterExplorer:
    """Interactive warp-metric explorer built on ipywidgets.

    Parameters
    ----------
    x : np.ndarray, optional
        Spatial sampling line; defaults to the model's [-8, 8] line
    colormap : str
        Name of a warpfactory colormap for the line plots
    continuous_update : bool
        Recompute on every slider drag tick when True. The default
        (False) recomputes on slider release only: each recompute runs
        the full metric -> stress-energy -> diagnostics pipeline plus a
        matplotlib redraw (~0.1 s), and a drag emits dozens of ticks,
        so per-tick recomputes queue in the kernel and the UI lags
        behind the mouse.
    """

    def __init__(
        self,
        x: Optional[np.ndarray] = None,
        colormap: str = "redblue",
        continuous_update: bool = False,
    ):
        self.model = ExplorerModel(x=x)
        self.colormap = ColorMaps().get(colormap)
        self.continuous_update = continuous_update
        self.result: Optional[ExplorationResult] = None
        self.recompute_count = 0
        self._updates_held = False

        self.metric_selector = widgets.Dropdown(
            options=list(METRIC_CATALOG), description="Metric"
        )
        self.component_selector = widgets.Dropdown(
            options=_COMPONENTS, description="Component"
        )
        self.diagnostics_toggle = widgets.Checkbox(
            value=True, description="Diagnostics"
        )
        self.status = widgets.HTML()
        self.output = widgets.Output()

        self._sliders: Dict[str, widgets.FloatSlider] = {}
        self.parameter_box = widgets.VBox()
        self._build_sliders(self.model.defaults(self.metric_selector.value))

        self.metric_selector.observe(self._on_metric_changed, names="value")
        self.component_selector.observe(self._on_control_changed, names="value")
        self.diagnostics_toggle.observe(self._on_control_changed, names="value")

        self.recompute()

    # -- widget wiring -------------------------------------------------

    def _build_sliders(self, params: Dict[str, float]) -> None:
        self._sliders = {}
        rows = []
        for name, value in params.items():
            lo, hi, step = _PARAM_RANGES.get(name, _DEFAULT_RANGE)
            slider = widgets.FloatSlider(
                value=value,
                min=lo,
                max=hi,
                step=step,
                description=name,
                continuous_update=self.continuous_update,
            )
            slider.observe(self._on_control_changed, names="value")
            self._sliders[name] = slider
            rows.append(slider)
        self.parameter_box.children = tuple(rows)

    def _on_metric_changed(self, change: dict) -> None:
        self._build_sliders(self.model.defaults(change["new"]))
        self.recompute()

    def _on_control_changed(self, change: dict) -> None:
        if not self._updates_held:
            self.recompute()

    def get_parameters(self) -> Dict[str, float]:
        """Current slider values keyed by parameter name."""
        return {name: slider.value for name, slider in self._sliders.items()}

    def set_parameter(self, name: str, value: float) -> None:
        """Programmatically move a slider (triggers a recompute)."""
        self._sliders[name].value = value

    def set_parameters(self, params: Dict[str, float]) -> ExplorationResult:
        """Move several sliders with a single recompute at the end.

        Assigning slider values one by one fires one observer event
        (full pipeline + redraw) per slider; batching them keeps
        programmatic sweeps responsive.
        """
        self._updates_held = True
        try:
            for name, value in params.items():
                self._sliders[name].value = value
        finally:
            self._updates_held = False
        return self.recompute()

    # -- pipeline ------------------------------------------------------

    def recompute(self) -> ExplorationResult:
        """Re-run the pipeline for the current controls and redraw."""
        self.recompute_count += 1
        # Visible immediately in a live front end; the final status
        # overwrites it below.
        self.status.value = "<i>computing...</i>"
        self.result = self.model.evaluate(
            self.metric_selector.value,
            self.get_parameters(),
            diagnostics=self.diagnostics_toggle.value,
        )
        self._update_status()
        self._redraw()
        return self.result

    def _update_status(self) -> None:
        result = self.result
        if result is None or result.conditions is None:
            self.status.value = ""
            return
        parts = []
        for name, satisfied in result.conditions.items():
            color = "green" if satisfied else "red"
            state = "ok" if satisfied else "violated"
            parts.append(f"<span style='color:{color}'>{name}: {state}</span>")
        qi = result.quantum_inequality
        if qi is not None:
            color = "green" if qi["satisfied"] else "red"
            state = "ok" if qi["satisfied"] else "violated"
            parts.append(
                f"<span style='color:{color}'>Ford-Roman QI: {state} "
                f"(wall {qi['delta']:.3g} m vs limit {qi['delta_max']:.3g} m)</span>"
            )
        self.status.value = "Energy conditions -- " + " | ".join(parts)

    def _redraw(self) -> None:
        result = self.result
        if result is None:
            return
        with self.output:
            self.output.clear_output(wait=True)
            fig = self.plot(result)
            # display(fig) rather than plt.show(): renders inside the
            # Output widget and works headless (Agg) in tests.
            display(fig)
            plt.close(fig)

    # -- plotting ------------------------------------------------------

    def plot(self, result: Optional[ExplorationResult] = None) -> plt.Figure:
        """Draw the current exploration as a matplotlib figure.

        Public so notebooks (and tests) can render without a live
        widget front end.
        """
        if result is None:
            result = self.result
        if result is None:
            raise ValueError("nothing to plot; call recompute() first")

        show_causal = result.light_cone_tilt is not None
        n_rows = 3 if show_causal else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows), sharex=True)
        x = result.x

        component = self.component_selector.value
        values = np.asarray(result.metric.get(component, np.zeros_like(x)), dtype=float)
        axes[0].plot(x, values, color=self.colormap(0.8))
        axes[0].set_ylabel(component)
        axes[0].set_title(f"{result.metric_name}: {component}")

        rho = np.asarray(result.stress_energy["T_tt"], dtype=float)
        axes[1].plot(x, rho, color=self.colormap(0.2))
        axes[1].set_ylabel(r"$T_{tt}$")
        axes[1].set_title("Eulerian energy density")
        if result.causality_violations is not None and np.any(
            result.causality_violations
        ):
            axes[1].fill_between(
                x,
                rho.min(),
                rho.max(),
                where=result.causality_violations,
                alpha=0.15,
                color="red",
                label="causal structure altered",
            )
            axes[1].legend(loc="upper right")

        if show_causal:
            tilt = np.asarray(result.light_cone_tilt, dtype=float)
            axes[2].plot(x, tilt, color="black")
            axes[2].set_ylabel("light cone tilt [rad]")
            axes[2].set_title("Causal structure")
            for name, surface in (result.horizons or {}).items():
                if len(surface) > 0:
                    radius = float(np.max(np.abs(surface[:, 0])))
                    for sign in (-1.0, 1.0):
                        axes[2].axvline(
                            sign * radius, linestyle="--", alpha=0.6, label=name
                        )
            handles, labels = axes[2].get_legend_handles_labels()
            if labels:
                unique = dict(zip(labels, handles))
                axes[2].legend(unique.values(), unique.keys(), loc="upper right")

        axes[-1].set_xlabel("x")
        fig.tight_layout()
        return fig

    # -- notebook entry point -------------------------------------------

    def widget(self) -> widgets.VBox:
        """The assembled widget tree (without displaying it)."""
        controls = widgets.VBox(
            [
                self.metric_selector,
                self.parameter_box,
                self.component_selector,
                self.diagnostics_toggle,
            ]
        )
        return widgets.VBox([controls, self.status, self.output])

    def display(self) -> None:
        """Render the explorer in the current notebook cell."""
        display(self.widget())
        self._redraw()
