import numpy as np
import pytest
from icing import PanelFlow, naca4, Cloud, collection_efficiency, rime, boundary_layer, corrections as cor, report_data as rd


def test_panel_method_lift_of_naca2412():
    assert PanelFlow(naca4("2412", 200)).cl == pytest.approx(0.257, abs=0.01)       # zero-lift angle near -2.1 deg
    slope = (PanelFlow(naca4("2412", 200, np.radians(4))).cl - PanelFlow(naca4("2412", 200)).cl) / np.radians(4)
    assert slope == pytest.approx(2 * np.pi * 1.09, rel=0.03)                       # thin-airfoil value plus thickness


def test_symmetric_section_has_no_lift_and_symmetric_catch():
    flow = PanelFlow(naca4("0012", 120))
    assert abs(flow.cl) < 1e-6
    col = collection_efficiency(flow, Cloud())
    mid = 0.5 * (col.limits[0] + col.limits[1])
    assert mid == pytest.approx(flow.s_nodes[60], abs=0.004)


def test_collection_efficiency_integrates_to_total():
    col = collection_efficiency(PanelFlow(naca4("2412", 120)), Cloud())
    assert np.trapezoid(col.beta, col.s) / col.height == pytest.approx(col.total, rel=0.03)
    assert 0 < col.beta.max() < 1


def test_bigger_droplets_are_caught_more():
    flow = PanelFlow(naca4("2412", 120))
    small = collection_efficiency(flow, Cloud(mvd=10e-6)).total
    large = collection_efficiency(flow, Cloud(mvd=40e-6)).total
    assert small < 0.3 < 0.6 < large


def test_single_shot_ice_mass_matches_fensap():
    cloud = Cloud(**{k: v for k, v in rd.ICING.items() if k != "duration"})
    col = collection_efficiency(PanelFlow(naca4("2412", 120)), cloud)
    mass = cloud.lwc * cloud.speed * rd.ICING["duration"] * col.total * col.height * rd.CHORD * rd.SPAN
    assert mass == pytest.approx(rd.FENSAP_ICE_MASS, rel=0.03)


def test_grown_ice_conserves_the_water_caught():
    res = rime.accrete(naca4("2412", 160), Cloud(), 420, layers=4)
    assert res.mass_per_span == pytest.approx(res.caught_per_span, rel=0.05)


def test_flat_plate_like_drag_level():
    flow = PanelFlow(naca4("0012", 200))
    cd = boundary_layer.analyse(flow, 3e6).cd
    assert 0.004 < cd < 0.008                                                      # measured: about 0.006


def test_grid_convergence_of_the_lift_coefficient():
    g = cor.grid_convergence(*rd.MESH_CL, rd.MESH_RATIO)
    assert g["order"] == pytest.approx(2.0, abs=0.01)
    assert g["extrapolated"] == pytest.approx(0.175, abs=1e-3)
    assert g["gci_fine"] == pytest.approx(0.0368, abs=1e-3)


def test_table_4_5_coefficients_used_a_unit_reference_area():
    assert cor.coefficient(0.91017, 1.225, 20, 1.0) == pytest.approx(0.00371, abs=2e-5)
    assert cor.coefficient(0.91017, 1.225, 20, rd.CHORD * rd.SPAN) == pytest.approx(0.165, abs=1e-3)
