"""Two-dimensional potential flow around a closed airfoil: Hess-Smith panel method.

Constant-strength source panels plus one vortex strength common to all panels,
with the Kutta condition applied at the trailing edge. Lengths are in chords and
velocities in units of the free stream, which runs along +x. The body is given
as a closed polyline that starts at the trailing edge, runs along the lower
surface to the leading edge and back over the upper surface.
"""
from __future__ import annotations

import numpy as np


def naca4(code: str = "2412", n: int = 160, alpha: float = 0.0) -> np.ndarray:
    """Nodes of a NACA 4-digit airfoil with a closed trailing edge, pitched nose-up by alpha [rad]."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    i = np.arange(n + 1)
    x = 0.5 * (1 + np.cos(2 * np.pi * i / n))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x); dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    lower = i < n / 2
    px = np.where(lower, x + yt * np.sin(th), x - yt * np.sin(th))
    py = np.where(lower, yc - yt * np.cos(th), yc + yt * np.cos(th))
    px[-1], py[-1] = px[0], py[0]
    return rotate(np.column_stack([px, py]), alpha)


def rotate(nodes: np.ndarray, alpha: float) -> np.ndarray:
    """Pitch the body nose-up by alpha about the origin; the stream stays horizontal."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.column_stack([nodes[:, 0] * c + nodes[:, 1] * s, -nodes[:, 0] * s + nodes[:, 1] * c])


class PanelFlow:
    def __init__(self, nodes: np.ndarray):
        self.nodes = np.asarray(nodes, float)
        d = np.diff(self.nodes, axis=0)
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.cos = d[:, 0] / self.length
        self.sin = d[:, 1] / self.length
        self.mid = (self.nodes[:-1] + self.nodes[1:]) / 2
        self.s_nodes = np.concatenate([[0], np.cumsum(self.length)])
        self.s_mid = (self.s_nodes[:-1] + self.s_nodes[1:]) / 2
        self._solve()

    def _source(self, pts):
        """Velocity (u, v) at pts induced by unit-strength source panels; shape (M, N)."""
        dx = pts[:, None, 0] - self.nodes[None, :-1, 0]
        dy = pts[:, None, 1] - self.nodes[None, :-1, 1]
        xl = dx * self.cos + dy * self.sin
        yl = -dx * self.sin + dy * self.cos
        L = self.length
        ul = np.log((xl ** 2 + yl ** 2) / ((xl - L) ** 2 + yl ** 2)) / (4 * np.pi)
        vl = (np.arctan2(yl, xl - L) - np.arctan2(yl, xl)) / (2 * np.pi)
        return ul * self.cos - vl * self.sin, ul * self.sin + vl * self.cos

    def _solve(self):
        N = len(self.length)
        us, vs = self._source(self.mid)
        idx = np.arange(N)
        us[idx, idx] = -0.5 * self.sin; vs[idx, idx] = 0.5 * self.cos     # self-induced: half the strength, outward
        c, s = self.cos[:, None], self.sin[:, None]
        An = -us * s + vs * c                      # normal velocity from sources
        At = us * c + vs * s                       # tangential velocity from sources
        A = np.zeros((N + 1, N + 1)); b = np.zeros(N + 1)
        A[:N, :N] = An
        A[:N, N] = At.sum(axis=1)                  # vortex (-vs, us): normal component
        b[:N] = self.sin                           # -(U . n), U = (1, 0)
        for k in (0, N - 1):                       # Kutta: tangential velocities at the trailing edge cancel
            A[N, :N] += At[k]
            A[N, N] += -An[k].sum()
            b[N] -= self.cos[k]
        sol = np.linalg.solve(A, b)
        self.q, self.gamma = sol[:N], sol[N]
        self.vt = self.cos + At @ self.q - self.gamma * An.sum(axis=1)     # surface speed, positive along the node order
        self.cp = 1 - self.vt ** 2
        self.cl = -2 * self.gamma * self.length.sum()                      # Kutta-Joukowski, chord = 1

    def velocity(self, pts) -> np.ndarray:
        """Air velocity at field points, shape (M, 2)."""
        pts = np.atleast_2d(pts)
        us, vs = self._source(pts)
        u = 1 + us @ self.q - self.gamma * vs.sum(axis=1)
        v = vs @ self.q + self.gamma * us.sum(axis=1)
        return np.column_stack([u, v])

    def inside(self, pts) -> np.ndarray:
        """Crossing-number test against the body polygon."""
        pts = np.atleast_2d(pts)
        x, y = pts[:, None, 0], pts[:, None, 1]
        x1, y1 = self.nodes[None, :-1, 0], self.nodes[None, :-1, 1]
        x2, y2 = self.nodes[None, 1:, 0], self.nodes[None, 1:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            hit = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        return hit.sum(axis=1) % 2 == 1
