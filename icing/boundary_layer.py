"""Integral boundary layer on the panel-method pressure distribution.

Thwaites' method for the laminar part, Michel's criterion for transition (or
laminar separation, whichever comes first, or a forced trip), Head's
entrainment method for the turbulent part and the Squire-Young formula for the
profile drag. One-way coupled: the boundary layer does not feed back on the
outer flow, so the model is only meaningful while the flow stays attached.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .panel import PanelFlow


@dataclass
class Side:
    s: np.ndarray; x: np.ndarray; ue: np.ndarray; theta: np.ndarray; H: np.ndarray; cf: np.ndarray
    x_transition: float; forced: bool; x_separation: float | None


@dataclass
class Viscous:
    cd: float; upper: Side; lower: Side


def _H1(H):  return 3.3 + 0.8234 * (H - 1.1) ** -1.287 if H <= 1.6 else 3.3 + 1.5501 * (H - 0.6778) ** -3.064
def _H(H1):
    if H1 <= 3.32: return 3.0
    return 1.1 + (0.8234 / (H1 - 3.3)) ** (1 / 1.287) if H1 >= 5.3 else 0.6778 + (1.5501 / (H1 - 3.3)) ** (1 / 3.064)


def _march(s, x, ue, Re, s_trip=None):
    n = len(s); theta = np.zeros(n); H = np.zeros(n); cf = np.zeros(n)
    due = np.gradient(ue, s)
    integral = 0.0; turbulent = False; forced = False; x_tr = x[-1]; x_sep = None; H1 = 0.0
    for i in range(n):
        if not turbulent:
            if i == 0:
                theta[i] = np.sqrt(0.075 / (Re * max(due[i], 1e-6)))
            else:
                integral += 0.5 * (ue[i] ** 5 + ue[i - 1] ** 5) * (s[i] - s[i - 1])
                theta[i] = np.sqrt(0.45 * integral / (Re * max(ue[i], 1e-6) ** 6))
            lam = float(np.clip(theta[i] ** 2 * Re * due[i], -0.1, 0.1))
            if lam >= 0:
                H[i] = 2.61 - 3.75 * lam + 5.24 * lam ** 2; l = 0.22 + 1.57 * lam - 1.8 * lam ** 2
            else:
                H[i] = 2.088 + 0.0731 / (lam + 0.14); l = 0.22 + 1.402 * lam + 0.018 * lam / (lam + 0.107)
            cf[i] = 2 * l / (Re * ue[i] * theta[i]) if ue[i] > 1e-6 else 0.0
            re_x = Re * ue[i] * s[i]; re_t = Re * ue[i] * theta[i]
            michel = re_x > 1e3 and re_t > 1.174 * (1 + 22400 / re_x) * re_x ** 0.46
            trip = s_trip is not None and s[i] >= s_trip
            if lam <= -0.09 or michel or trip:
                turbulent = True; forced = trip and not (lam <= -0.09 or michel); x_tr = x[i]
                H[i] = 1.4; H1 = _H1(1.4)
            continue
        ds = s[i] - s[i - 1]

        def rhs(th, h1, u, du):
            h = _H(h1); c = 0.246 * 10 ** (-0.678 * h) * (Re * u * th) ** -0.268
            dth = c / 2 - (h + 2) * th / u * du
            dh1 = (0.0306 * (h1 - 3.0) ** -0.6169 - h1 * dth - h1 * th / u * du) / th if h1 > 3.0 else 0.0
            return dth, dh1
        a1, b1 = rhs(theta[i - 1], H1, ue[i - 1], due[i - 1])
        thp = max(theta[i - 1] + ds * a1, 1e-9); h1p = max(H1 + ds * b1, 3.05)
        a2, b2 = rhs(thp, h1p, ue[i], due[i])
        theta[i] = max(theta[i - 1] + ds * 0.5 * (a1 + a2), 1e-9); H1 = max(H1 + ds * 0.5 * (b1 + b2), 3.05)
        H[i] = _H(H1); cf[i] = 0.246 * 10 ** (-0.678 * H[i]) * (Re * ue[i] * theta[i]) ** -0.268
        if H[i] > 2.4 and x_sep is None:
            x_sep = x[i]
    return Side(s, x, ue, theta, H, cf, x_tr, forced, x_sep)


def analyse(flow: PanelFlow, reynolds: float, trip=None) -> Viscous:
    """Profile drag of the body.

    trip  optional (s_lower, s_upper): arc lengths, measured along the body
          polyline, where roughness trips the boundary layer on each side.
    """
    vt = flow.vt.copy()
    vt[1:-1] = 0.25 * vt[:-2] + 0.5 * vt[1:-1] + 0.25 * vt[2:]
    k = np.where((vt[:-1] < 0) & (vt[1:] >= 0))[0]
    k = k[np.argmin(np.abs(k - len(vt) / 2))]                       # stagnation point near the nose
    s0 = flow.s_mid[k] - vt[k] * (flow.s_mid[k + 1] - flow.s_mid[k]) / (vt[k + 1] - vt[k])
    cut = 3                                                         # ignore the panels at the trailing edge
    up = slice(k + 1, len(vt) - cut); lo = slice(k, cut - 1 if cut > 0 else None, -1)
    xle = flow.nodes[:, 0].min(); chord = flow.nodes[:, 0].max() - xle

    def side(sl, sign, s_trip):
        s = np.abs(flow.s_mid[sl] - s0); ue = sign * vt[sl]; x = (flow.mid[sl, 0] - xle) / chord
        return _march(s, x, np.maximum(ue, 1e-4), reynolds, None if s_trip is None else abs(s_trip - s0))
    upper = side(up, 1.0, None if trip is None else trip[1])
    lower = side(lo, -1.0, None if trip is None else trip[0])
    cd = sum(2 * sd.theta[-1] * sd.ue[-1] ** ((sd.H[-1] + 5) / 2) for sd in (upper, lower))
    return Viscous(cd, upper, lower)
