"""Checks and corrections applied to the 2024 project data."""
from __future__ import annotations

import numpy as np


def grid_convergence(coarse: float, medium: float, fine: float, r: float, safety: float = 1.25) -> dict:
    """Richardson extrapolation and grid convergence index (Roache; Celik et al. 2008)."""
    e32, e21 = coarse - medium, medium - fine
    ratio = e21 / e32 if e32 else np.nan
    out = {"e32": e32, "e21": e21, "ratio": ratio}
    if not (0 < ratio < 1):
        out.update(order=np.nan, extrapolated=np.nan, gci_fine=np.nan,
                   behaviour="oscillatory" if ratio < 0 else "not converging")
        return out
    p = np.log(1 / ratio) / np.log(r)
    ext = fine + (fine - medium) / (r ** p - 1)          # = (r^p fine - medium) / (r^p - 1)
    out.update(order=p, extrapolated=ext, gci_fine=safety * abs((fine - medium) / fine) / (r ** p - 1),
               behaviour="monotonic")
    return out


def blockage_factor(frontal_area: float, section_area: float) -> float:
    """Total (solid plus wake) blockage, epsilon = frontal area / (4 x section area).

    Barlow, Rae and Pope, Low-Speed Wind Tunnel Testing. The corrected dynamic
    pressure is q (1 + epsilon)^2, so coefficients are DIVIDED by (1 + epsilon)^2.
    """
    return frontal_area / (4 * section_area)


def correct_for_blockage(coefficient, epsilon):
    return np.asarray(coefficient) / (1 + epsilon) ** 2


def finite_wing_slope(a0: float, aspect_ratio: float, e: float = 1.0) -> float:
    """Lift-curve slope of a finite wing from the section slope a0 [1/rad] (lifting line)."""
    return a0 / (1 + a0 / (np.pi * e * aspect_ratio))


def effective_aspect_ratio(a0: float, a: float) -> float:
    """Aspect ratio that turns the section slope a0 into the measured slope a."""
    return a0 / (np.pi * (a0 / a - 1))


def induced_drag(cl, aspect_ratio: float, e: float = 1.0):
    return np.asarray(cl) ** 2 / (np.pi * e * aspect_ratio)


def coefficient(force: float, density: float, speed: float, area: float) -> float:
    return force / (0.5 * density * speed ** 2 * area)
