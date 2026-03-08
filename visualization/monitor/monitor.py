from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from sim.scheduler import MultiRateScheduler


@dataclass(slots=True)
class MonitorConfig:
    enable: bool = True
    realtime: bool = True
    max_points: int | None = None
    step_stride: int = 1
    title: str = "Monitor"
    xlabel: str = "t [s]"
    ylabel: str = "value"
    x_min: float | None = None
    x_max: float | None = None


@dataclass(slots=True)
class _Series:
    name: str
    group: str
    color: str
    x: list[float]
    y: list[float]
    line: object | None = None


class Monitor:
    """
    Real-time grouped 2D monitor.

    Usage:
      monitor.push(name="dist", color="red", data=dist, t=t_now, group="metrics")
      monitor.update(step=k)
    """

    def __init__(self, scheduler: MultiRateScheduler | None = None, cfg: MonitorConfig | None = None, **kwargs):
        if cfg is None:
            cfg = MonitorConfig(**kwargs)
        self.cfg = cfg
        self.sch = scheduler
        self.realtime = bool(cfg.realtime)
        self.max_points = None if cfg.max_points is None else max(10, int(cfg.max_points))
        self.step_stride = max(1, int(cfg.step_stride))

        backend_is_noninteractive = False
        if plt is not None:
            backend = str(plt.get_backend()).lower()
            backend_name = backend.split(".")[-1]
            noninteractive_backends = {
                "agg",
                "cairo",
                "pdf",
                "pgf",
                "ps",
                "svg",
                "template",
                "backend_inline",
            }
            backend_is_noninteractive = (
                backend_name in noninteractive_backends or "backend_inline" in backend
            )
            if cfg.enable and backend_is_noninteractive:
                print(f"[Monitor] disabled: non-interactive matplotlib backend '{backend}'")

        self.enable = bool(cfg.enable and (plt is not None) and (not backend_is_noninteractive))
        self.fig = None
        self._axes_by_group: dict[str, object] = {}
        self._series_by_key: dict[tuple[str, str], _Series] = {}
        self._group_order: list[str] = []
        self._dirty = False
        self._layout_dirty = False
        self._last_push_t = -1.0
        self._sample_index = 0

        if self.enable:
            plt.ion()
            self.fig = plt.figure(figsize=(8.0, 5.5))
            self.fig.suptitle(self.cfg.title)
            plt.show(block=False)
            plt.pause(0.001)

    def _next_x(self, t: float | None) -> float:
        if t is not None:
            return float(t)
        x = float(self._sample_index)
        self._sample_index += 1
        return x

    def _ensure_group(self, group: str) -> None:
        if group in self._group_order:
            return
        self._group_order.append(group)
        self._layout_dirty = True

    def _trim_series(self, s: _Series) -> None:
        if self.max_points is not None and len(s.x) > self.max_points:
            s.x = s.x[-self.max_points :]
            s.y = s.y[-self.max_points :]

    def push(
        self,
        name: str,
        color: str,
        data: float,
        t: float | None = None,
        group: str = "default",
        step: int | None = None,
    ) -> None:
        if not self.enable:
            return
        if step is not None and (step % self.step_stride) != 0:
            return
        key = (group, name)
        x = self._next_x(t)
        y = float(data)

        if key not in self._series_by_key:
            self._ensure_group(group)
            self._series_by_key[key] = _Series(
                name=name,
                group=group,
                color=color,
                x=[x],
                y=[y],
            )
        else:
            s = self._series_by_key[key]
            s.x.append(x)
            s.y.append(y)
            self._trim_series(s)

        self._last_push_t = x
        self._dirty = True

    def _grid_shape(self, n: int) -> tuple[int, int]:
        if n <= 1:
            return 1, 1
        cols = 1 if n <= 2 else 2
        rows = int(math.ceil(n / cols))
        return rows, cols

    def _rebuild_layout(self) -> None:
        if not self.enable or self.fig is None:
            return
        rows, cols = self._grid_shape(len(self._group_order))
        self.fig.clf()
        self.fig.suptitle(self.cfg.title)
        self._axes_by_group.clear()
        gs = self.fig.add_gridspec(rows, cols)

        for idx, group in enumerate(self._group_order):
            r = idx // cols
            c = idx % cols
            ax = self.fig.add_subplot(gs[r, c])
            ax.set_title(group)
            ax.set_xlabel(self.cfg.xlabel)
            ax.set_ylabel(self.cfg.ylabel)
            ax.grid(True, alpha=0.25)
            if self.cfg.x_min is not None and self.cfg.x_max is not None:
                ax.set_xlim(float(self.cfg.x_min), float(self.cfg.x_max))
            self._axes_by_group[group] = ax

        for s in self._series_by_key.values():
            ax = self._axes_by_group[s.group]
            (s.line,) = ax.plot(s.x, s.y, color=s.color, label=s.name, lw=1.6)

        for ax in self._axes_by_group.values():
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="best")

        self.fig.tight_layout()
        self._layout_dirty = False
        self._dirty = True

    def _update_lines(self) -> None:
        for s in self._series_by_key.values():
            ax = self._axes_by_group.get(s.group)
            if ax is None:
                continue
            if s.line is None:
                (s.line,) = ax.plot(s.x, s.y, color=s.color, label=s.name, lw=1.6)
                ax.legend(loc="best")
            else:
                s.line.set_data(s.x, s.y)
            ax.relim()
            ax.autoscale_view()
            if self.cfg.x_min is not None and self.cfg.x_max is not None:
                ax.set_xlim(float(self.cfg.x_min), float(self.cfg.x_max))

    def update(self, step: int | None = None) -> None:
        if not self.enable or not self._dirty:
            return
        if step is not None and self.sch is not None and not self.sch.should_visualization(step):
            return

        if self._layout_dirty:
            self._rebuild_layout()
        else:
            self._update_lines()

        if self.fig is None:
            return

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        self._dirty = False

    def clear(self) -> None:
        self._series_by_key.clear()
        self._group_order.clear()
        self._axes_by_group.clear()
        self._dirty = False
        self._layout_dirty = False
        self._sample_index = 0
        self._last_push_t = -1.0
        if self.enable and self.fig is not None:
            self.fig.clf()
            self.fig.suptitle(self.cfg.title)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def close(self, block: bool = False) -> None:
        if not self.enable or self.fig is None:
            return
        if block:
            plt.show(block=True)
        else:
            plt.close(self.fig)
