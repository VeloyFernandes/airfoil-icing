"""Droplet trajectories and local collection efficiency (Lagrangian method).

Droplets are released upstream with the local air velocity and tracked through
the panel-method flow field. Drag follows Schiller and Naumann; gravity is
neglected. Lengths are in chords, velocities in units of the free stream.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .panel import PanelFlow


@dataclass(frozen=True)
class Cloud:
    """Icing cloud and flight condition."""

    speed: float = 20.0          # free-stream speed [m/s]
    chord: float = 0.09          # [m]
    mvd: float = 20e-6           # droplet diameter [m]
    lwc: float = 1.0e-3          # liquid water content [kg/m^3]
    temperature: float = 248.15  # static air temperature [K]
    pressure: float = 101325.0   # [Pa]

    @property
    def air_density(self): return self.pressure / (287.05 * self.temperature)
    @property
    def air_viscosity(self):     # Sutherland
        T = self.temperature
        return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    @property
    def inertia(self):           # K = rho_w d^2 V / (18 mu c)
        return 1000.0 * self.mvd ** 2 * self.speed / (18 * self.air_viscosity * self.chord)
    @property
    def droplet_reynolds(self):  # based on the free-stream speed
        return self.air_density * self.speed * self.mvd / self.air_viscosity
    @property
    def reynolds(self):
        return self.air_density * self.speed * self.chord / self.air_viscosity


def _accel(flow, cloud, x, u):
    rel = flow.velocity(x) - u
    re = cloud.droplet_reynolds * np.hypot(rel[:, 0], rel[:, 1])
    return (1 + 0.15 * re ** 0.687)[:, None] * rel / cloud.inertia


def track(flow: PanelFlow, cloud: Cloud, y0, x0: float = -4.0, keep_paths: bool = False):
    """Track droplets released at (x0, y0).

    Returns (s_hit, paths): s_hit is the arc length of the impact point measured
    along the body polyline (NaN for a miss); paths is a list of trajectories.
    """
    y0 = np.atleast_1d(np.asarray(y0, float)); M = len(y0)
    x = np.column_stack([np.full(M, x0), y0]); u = flow.velocity(x)
    s_hit = np.full(M, np.nan); active = np.ones(M, bool)
    paths = [[p.copy()] for p in x] if keep_paths else None
    xmax = flow.nodes[:, 0].max()
    for _ in range(4000):
        if not active.any():
            break
        ia = np.where(active)[0]; xa, ua = x[ia], u[ia]
        d = np.min(np.hypot(xa[:, None, 0] - flow.mid[None, :, 0], xa[:, None, 1] - flow.mid[None, :, 1]), axis=1)
        dt = np.clip(0.12 * d, 0.0015, 0.08)[:, None]
        k1x, k1u = ua, _accel(flow, cloud, xa, ua)
        k2x, k2u = ua + dt / 2 * k1u, _accel(flow, cloud, xa + dt / 2 * k1x, ua + dt / 2 * k1u)
        k3x, k3u = ua + dt / 2 * k2u, _accel(flow, cloud, xa + dt / 2 * k2x, ua + dt / 2 * k2u)
        k4x, k4u = ua + dt * k3u, _accel(flow, cloud, xa + dt * k3x, ua + dt * k3u)
        xn = xa + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        un = ua + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        hit = flow.inside(xn)
        for j in np.where(hit)[0]:
            s_hit[ia[j]], xn[j] = _impact(flow, xa[j], xn[j])
        x[ia], u[ia] = xn, un
        if keep_paths:
            for j, i in enumerate(ia):
                paths[i].append(xn[j].copy())
        active[ia[hit]] = False
        active[ia[xn[:, 0] > xmax]] = False
    return s_hit, ([np.array(p) for p in paths] if keep_paths else None)


def _impact(flow, a, b):
    """First intersection of the step a -> b with the body; returns (arc length, point)."""
    p, r = flow.nodes[:-1], np.diff(flow.nodes, axis=0); d = b - a
    den = r[:, 0] * d[1] - r[:, 1] * d[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((a[0] - p[:, 0]) * d[1] - (a[1] - p[:, 1]) * d[0]) / den       # along the panel
        w = ((a[0] - p[:, 0]) * r[:, 1] - (a[1] - p[:, 1]) * r[:, 0]) / den  # along the step
    ok = (t >= 0) & (t <= 1) & (w >= 0) & (w <= 1)
    if not ok.any():
        k = np.argmin(np.hypot(*(flow.mid - b).T)); return flow.s_mid[k], flow.mid[k]
    k = np.where(ok)[0][np.argmin(w[ok])]
    return flow.s_nodes[k] + t[k] * flow.length[k], p[k] + t[k] * r[k]


@dataclass
class Collection:
    s: np.ndarray           # arc length of impact points, increasing
    beta: np.ndarray        # local collection efficiency at s
    y0: np.ndarray          # release heights of the droplets that hit
    limits: tuple           # arc length of the lower and upper impingement limits
    total: float            # overall collection efficiency E
    height: float           # projected frontal height of the body [chords]
    paths: list | None = None

    def at(self, s):
        """Collection efficiency interpolated at arc length s (zero outside the limits)."""
        return np.interp(s, self.s, self.beta, left=0.0, right=0.0)


def collection_efficiency(flow: PanelFlow, cloud: Cloud, n: int = 90, keep_paths: bool = False) -> Collection:
    """Local collection efficiency beta = dy0/ds from a fan of droplet trajectories."""
    ymin, ymax = flow.nodes[:, 1].min(), flow.nodes[:, 1].max()
    lift_shift = 0.45 * flow.cl                       # the stream tube that reaches the nose starts lower
    coarse = np.linspace(ymin - 0.25 - lift_shift, ymax + 0.15, 40)
    s, _ = track(flow, cloud, coarse)
    hit = np.where(~np.isnan(s))[0]
    if len(hit) == 0:
        return Collection(np.array([0, 1.0]), np.zeros(2), np.array([]), (0, 0), 0.0, ymax - ymin)
    lo, hi = coarse[max(hit[0] - 1, 0)], coarse[min(hit[-1] + 1, len(coarse) - 1)]
    y0 = np.linspace(lo, hi, n)
    s, paths = track(flow, cloud, y0, keep_paths=keep_paths)
    ok = ~np.isnan(s); yh, sh = y0[ok], s[ok]
    if ok.sum() < 3:
        return Collection(np.array([0, 1.0]), np.zeros(2), np.array([]), (0, 0), 0.0, ymax - ymin)
    order = np.argsort(sh); sh, yh = sh[order], yh[order]
    keep = np.concatenate([[True], np.diff(sh) > 1e-7]); sh, yh = sh[keep], yh[keep]
    beta = np.abs(np.gradient(yh, sh))
    beta = np.convolve(np.pad(beta, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    dy = y0[1] - y0[0]
    span = (yh.max() - yh.min()) + dy                 # half a spacing of uncertainty at each limit
    # close the curve at the impingement limits
    ds0 = dy / max(beta[0], 0.05); ds1 = dy / max(beta[-1], 0.05)
    sh = np.concatenate([[sh[0] - ds0], sh, [sh[-1] + ds1]]); beta = np.concatenate([[0], beta, [0]])
    return Collection(sh, beta, yh, (sh[0], sh[-1]), span / (ymax - ymin), ymax - ymin,
                      [p for p, k in zip(paths, ok) if k] if keep_paths else None)
