"""Regenerate the figures in docs/figures."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from icing import PanelFlow, naca4, Cloud, collection_efficiency, rime, boundary_layer, report_data as rd

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, RED, INK = "#2A6FB0", "#C4472D", "#10273A"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
cloud = Cloud()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, a in zip(axes, (0, 4, 8)):
    body = naca4("2412", 160, np.radians(a)); flow = PanelFlow(body)
    col = collection_efficiency(flow, cloud, 45, keep_paths=True)
    res = rime.accrete(body, cloud, 420, layers=6)
    for p in col.paths[::2]: ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=0.5, alpha=0.6)
    ax.fill(res.nodes[:, 0], res.nodes[:, 1], color="#C7E0F4", ec="#5FA3DD", lw=1)
    for layer in res.layers[:-1]: ax.plot(layer[:, 0], layer[:, 1], color=BLUE, lw=0.3)
    ax.fill(body[:, 0], body[:, 1], color=INK)
    ax.set_aspect("equal"); ax.set_xlim(-0.2, 0.4); ax.set_ylim(-0.17, 0.17); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{a} deg: {res.mass_per_span*250:.1f} g on the 25 cm model", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(OUT / "ice_shapes.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
flow = PanelFlow(naca4("2412", 160))
for d, c in ((10, "#8FB3CF"), (20, BLUE), (40, RED)):
    col = collection_efficiency(flow, Cloud(mvd=d * 1e-6))
    ax.plot((col.s - flow.s_nodes[80]) * 100, col.beta, color=c, label=f"{d} µm, E = {col.total:.2f}")
ax.set_xlabel("distance from the nose along the surface [% chord]"); ax.set_ylabel("collection efficiency"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "collection_efficiency.png"); plt.close(fig)

re_t = 1.2 * 20 * 0.09 / 1.81e-5
model = [boundary_layer.analyse(PanelFlow(naca4("2412", 200, np.radians(a))), re_t).cd for a in rd.ALPHA]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, key, lab in ((axes[0], "CL", "lift coefficient"), (axes[1], "CD", "drag coefficient")):
    ax.plot(rd.ALPHA, getattr(rd, f"CFD_CLEAN_{key}"), color=BLUE, label="Fluent, clean")
    ax.plot(rd.ALPHA, getattr(rd, f"CFD_ICED_{key}"), color=RED, label="Fluent, iced")
    ax.plot(rd.ALPHA, getattr(rd, f"EXP_CLEAN_{key}"), "o", color=BLUE, label="tunnel, clean")
    ax.plot(rd.EXP_ICED_ALPHA, getattr(rd, f"EXP_ICED_{key}"), "o", color=RED, label="tunnel, iced")
    ax.set_xlabel("angle of attack [deg]"); ax.set_ylabel(lab); ax.set_ylim(0)
axes[1].plot(rd.ALPHA, model, "--", color=INK, label="boundary-layer model, clean")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "report_data.png"); plt.close(fig)
print("figures written to", OUT)
