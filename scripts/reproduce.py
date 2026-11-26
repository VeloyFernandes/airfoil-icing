"""Recompute the checks on the 2024 project data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from icing import PanelFlow, naca4, Cloud, collection_efficiency, rime, boundary_layer, corrections as cor, report_data as rd

cloud = Cloud(speed=rd.SPEED, chord=rd.CHORD, mvd=rd.ICING["mvd"], lwc=rd.ICING["lwc"], temperature=rd.ICING["temperature"])
print(f"Icing case: K = {cloud.inertia:.3f}, droplet Re = {cloud.droplet_reynolds:.1f}, chord Re = {cloud.reynolds:,.0f}")
print("\nIce mass after 420 s on the 0.25 m model [g]")
for a in (0, 4, 8):
    body = naca4("2412", 160, np.radians(a))
    col = collection_efficiency(PanelFlow(body), cloud)
    single = cloud.lwc * cloud.speed * 420 * col.total * col.height * rd.CHORD * rd.SPAN * 1e3
    res = rime.accrete(body, cloud, 420, layers=6)
    extra = f"   FENSAP-ICE: {rd.FENSAP_ICE_MASS*1e3:.2f}" if a == 0 else ""
    print(f"  {a} deg: E = {col.total:.3f}, one step {single:.2f}, six layers {res.mass_per_span*rd.SPAN*1e3:.2f}, "
          f"max thickness {res.max_thickness*1e3:.1f} mm{extra}")

re_tunnel = 1.2 * rd.SPEED * rd.CHORD / 1.81e-5
print(f"\nClean profile drag, boundary-layer model at Re = {re_tunnel:,.0f}")
print("  alpha   model    CFD      tunnel")
for a in range(0, 9, 2):
    cd = boundary_layer.analyse(PanelFlow(naca4("2412", 200, np.radians(a))), re_tunnel).cd
    print(f"  {a:3d}    {cd:.4f}   {rd.CFD_CLEAN_CD[a]:.4f}   {rd.EXP_CLEAN_CD[a]:.4f}")

g = cor.grid_convergence(*rd.MESH_CL, rd.MESH_RATIO)
print(f"\nMesh study, lift: order {g['order']:.2f}, extrapolated {g['extrapolated']:.3f}, GCI {g['gci_fine']*100:.2f}%")
print(f"Table 4.5 lift coefficient with A = chord x span: {cor.coefficient(0.91017, 1.225, 20, rd.CHORD*rd.SPAN):.3f} (printed 0.00371)")
a0 = np.degrees((rd.CFD_CLEAN_CL[4] - rd.CFD_CLEAN_CL[0]) / 4); a_exp = np.degrees((rd.EXP_CLEAN_CL[4] - rd.EXP_CLEAN_CL[0]) / 4)
ar = rd.SPAN / rd.CHORD
print(f"Lift slope per degree: CFD {np.radians(a0):.4f}, tunnel {np.radians(a_exp):.4f}, free wing of AR {ar:.2f}: {np.radians(cor.finite_wing_slope(a0, ar)):.4f}")
h8 = np.ptp(naca4('2412', 80, np.radians(8))[:, 1]) * rd.CHORD
eps = cor.blockage_factor(h8 * rd.SPAN, rd.TUNNEL_SECTION)
print(f"Blockage at 8 deg: model fills {h8*rd.SPAN/rd.TUNNEL_SECTION*100:.1f}% of the section, coefficients divided by {(1+eps)**2:.3f}")
