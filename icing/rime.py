"""Rime ice accretion: every droplet freezes where it lands.

The icing time is split into layers. For each layer the flow and the droplet
trajectories are recomputed around the current ice shape, and the surface is
moved along its outward normal by the thickness of ice caught in that layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from .panel import PanelFlow
from .droplets import Cloud, Collection, collection_efficiency

RIME_DENSITY = 917.0   # kg/m^3, the FENSAP-ICE default for rime


@dataclass
class IceResult:
    clean: np.ndarray                 # nodes of the clean body [chords]
    nodes: np.ndarray                 # nodes of the iced body [chords]
    layers: list = field(default_factory=list)       # body after each layer
    collections: list = field(default_factory=list)  # Collection of each layer
    mass_per_span: float = 0.0        # ice mass from the grown area [kg/m]
    caught_per_span: float = 0.0      # water caught by the droplet fan [kg/m]
    max_thickness: float = 0.0        # largest distance from the clean surface [m]


def polygon_area(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def _normals_and_curvature(nodes):
    d = np.diff(nodes, axis=0); L = np.hypot(d[:, 0], d[:, 1])
    t = d / L[:, None]
    n_panel = np.column_stack([-t[:, 1], t[:, 0]])            # outward for this node order
    n = np.zeros_like(nodes); n[1:-1] = n_panel[:-1] + n_panel[1:]
    n[0] = n[-1] = n_panel[0] + n_panel[-1]
    n /= np.hypot(n[:, 0], n[:, 1])[:, None]
    turn = t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0]         # sine of the turning angle
    kappa = np.zeros(len(nodes)); kappa[1:-1] = -turn / (0.5 * (L[:-1] + L[1:]))
    return n, np.maximum(kappa, 0.0)                           # convex curvature only


def _resample(nodes, fractions):
    d = np.diff(nodes, axis=0); s = np.concatenate([[0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    target = fractions * s[-1]
    out = np.column_stack([np.interp(target, s, nodes[:, 0]), np.interp(target, s, nodes[:, 1])])
    out[0], out[-1] = nodes[0], nodes[-1]
    return out


def accrete(nodes: np.ndarray, cloud: Cloud, duration: float, layers: int = 6,
            density: float = RIME_DENSITY, n_droplets: int = 90) -> IceResult:
    """Grow rime ice on the body for `duration` seconds in `layers` steps."""
    clean = np.asarray(nodes, float); body = clean.copy()
    d = np.diff(clean, axis=0); s = np.concatenate([[0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    fractions = s / s[-1]
    res = IceResult(clean, body)
    dt = duration / layers
    for _ in range(layers):
        flow = PanelFlow(body)
        col = collection_efficiency(flow, cloud, n_droplets)
        h0 = col.at(flow.s_nodes) * cloud.lwc * cloud.speed * dt / density / cloud.chord
        for _ in range(4):                                       # light smoothing along the surface
            h0[1:-1] = 0.25 * h0[:-2] + 0.5 * h0[1:-1] + 0.25 * h0[2:]
        n, kappa = _normals_and_curvature(body)
        # a layer of thickness h on a convex surface has area h (1 + h kappa / 2) per unit arc: conserve mass
        h = np.where(kappa > 1e-9, (np.sqrt(1 + 2 * kappa * h0) - 1) / np.maximum(kappa, 1e-9), h0)
        target = col.total * col.height * cloud.lwc * cloud.speed * dt / density / cloud.chord   # ice area caught
        base = polygon_area(body); scale = 1.0; iced = h > 0

        def grow(k):
            moved = body + k * h[:, None] * n
            sm = moved.copy()                                    # relax the fresh ice surface to keep it free of kinks
            sm[1:-1] = 0.25 * moved[:-2] + 0.5 * moved[1:-1] + 0.25 * moved[2:]
            moved[iced] = sm[iced]
            return _resample(moved, fractions)
        for _ in range(4):                                       # make the grown area equal the ice caught
            added = polygon_area(grow(scale)) - base
            if added <= 0 or target <= 0:
                break
            scale *= target / added
        body = grow(scale)
        res.layers.append(body.copy()); res.collections.append(col)
        res.caught_per_span += cloud.lwc * cloud.speed * dt * col.total * col.height * cloud.chord
    res.nodes = body
    res.mass_per_span = density * (polygon_area(body) - polygon_area(clean)) * cloud.chord ** 2
    dist = np.min(np.hypot(body[:, None, 0] - clean[None, :, 0], body[:, None, 1] - clean[None, :, 1]), axis=1)
    res.max_thickness = dist.max() * cloud.chord
    return res
