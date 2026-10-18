"""Independent physical-ray reference for one ideal plane mirror + scalar UTD.

Only the existing SciPy coefficient reference is imported; no renderer code,
renderer image-source helpers, or renderer diffraction-point helpers are used.
The primary numerical solve minimizes a physical broken optical-path functional
in mirror tangent coordinates and the actual finite-edge coordinate. The source
is a locally collimated plane wave, so its phase is measured from a source plane,
not from a spherical point emitter. Units: mm. Convention: exp(+i omega t).

Run: python research/utd-reflected-fermat-reference.py [--summary]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("independent_utd", HERE / "utd-reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def unit(v):
    a = np.asarray(v, float)
    return a / np.linalg.norm(a)


def reflected_point(p, origin, normal):
    return p - 2 * np.dot(p - origin, normal) * normal


def reflected_direction(d, normal):
    return d - 2 * np.dot(d, normal) * normal


def tangent_basis(normal):
    tangent = unit(np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))]))
    return np.column_stack((tangent, np.cross(normal, tangent)))


def physical_fermat(order, src, di, dst, mp, mn, edge_center, edge_direction, half_length=300):
    """Minimize a physical broken eikonal without an image-source construction.

    x[:2] are mirror tangent coordinates, x[2] is the edge coordinate.
    Mirror equality is imposed exactly through the tangent parameterization;
    the finite-edge interval is a bound constraint. A stationary-system polish
    is used only after the constrained physical minimization has converged.
    """
    basis = tangent_basis(mn)
    edge = unit(edge_direction)
    eye = np.eye(3)

    def evaluate(x, want_hessian=False):
        m = mp + basis @ x[:2]
        q = edge_center + edge * x[2]
        am = np.column_stack((basis, np.zeros(3)))
        aq = np.column_stack((np.zeros((3, 2)), edge))
        value = 0.
        grad = np.zeros(3)
        hess = np.zeros((3, 3))

        def segment(delta, derivative):
            nonlocal value, grad, hess
            r = np.linalg.norm(delta)
            direction = delta / r
            value += r
            grad += derivative.T @ direction
            hess += derivative.T @ ((eye - np.outer(direction, direction)) / r) @ derivative

        if order == "RD":
            value += np.dot(di, m - src)
            grad += am.T @ di
            segment(q - m, aq - am)
            segment(dst - q, -aq)
        else:
            value += np.dot(di, q - src)
            grad += aq.T @ di
            segment(m - q, am - aq)
            segment(dst - m, -am)
        return (value, grad, hess) if want_hessian else (value, grad)

    # Starts use physical geometry only. They are deliberately not initialized
    # at a solution computed by reflection/unfolding.
    seeds = ([0., 0., 0.], [2.1, -1.3, .7], [-2.3, .9, -.8])
    solutions = []
    for seed in seeds:
        result = minimize(evaluate, seed, jac=True, method="SLSQP",
                          bounds=[(None, None), (None, None), (-half_length, half_length)],
                          options={"ftol": 1e-13, "maxiter": 300})
        if not np.isfinite(result.fun):
            raise AssertionError(f"Physical minimization failed: {result.message}")
        solutions.append(result)
    best = min(solutions, key=lambda r: r.fun)
    stationary = least_squares(lambda x: evaluate(x)[1], best.x,
                               jac=lambda x: evaluate(x, True)[2],
                               bounds=([-np.inf, -np.inf, -half_length],
                                       [np.inf, np.inf, half_length]),
                               xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=100)
    x = stationary.x
    value, gradient, hessian = evaluate(x, True)
    m = mp + basis @ x[:2]
    q = edge_center + edge * x[2]
    assert abs(x[2]) < half_length - 1, "Fixture must have an interior edge saddle"
    assert np.linalg.norm(gradient) < 1e-11
    assert np.linalg.eigvalsh(hessian).min() > 0, "Require a strict physical local minimum"
    assert abs(value - best.fun) < 1e-9
    return m, q, value, {
        "objective": float(value),
        "maxGradient": float(np.max(np.abs(gradient))),
        "hessianMinEigenvalue": float(np.linalg.eigvalsh(hessian).min()),
        "initialMinimizerGradient": float(np.linalg.norm(best.jac)),
        "initialMinimizerSuccess": bool(best.success),
        "initialMinimizerStatus": str(best.message),
        "threeStartObjectiveSpread": float(np.ptp([r.fun for r in solutions])),
        "iterations": int(best.nit),
    }


def fixture(order, angle, horizontal, mirror_variant, rotation, wavelength=.05):
    d_local = unit([.09 if angle else 0., np.sin(angle), np.cos(angle)])
    # Receiver offset relative to the incident beam axis controls the lit side.
    destination_local = 20 * d_local + np.array([horizontal, .4 if angle else .2, 0.])
    source_effective_local = -20 * d_local
    mp_local = np.array([0., 0., -10. if order == "RD" else 10.])
    mn_local = unit([-1., 0., 1.] if mirror_variant == "45deg" else [-.7, .25, 1.])
    if order == "RD":
        source_local = reflected_point(source_effective_local, mp_local, mn_local)
        direction_local = reflected_direction(d_local, mn_local)
        destination_physical_local = destination_local
    else:
        source_local = source_effective_local
        direction_local = d_local
        destination_physical_local = reflected_point(destination_local, mp_local, mn_local)
    if rotation:
        matrix = Rotation.from_euler("xyz", [23, 31, 17], degrees=True).as_matrix()
        translation = np.array([2., -3., 4.])
    else:
        matrix, translation = np.eye(3), np.zeros(3)
    point = lambda p: matrix @ np.asarray(p) + translation
    direction = lambda d: matrix @ np.asarray(d)
    src, di, dst = point(source_local), direction(direction_local), point(destination_physical_local)
    mp, mn = point(mp_local), direction(mn_local)
    center, edge = point([0, 0, 0]), direction([0, 1, 0])
    normal_front, tangent_front = direction([0, 0, -1]), direction([1, 0, 0])

    m, q, physical_phase, optimizer = physical_fermat(order, src, di, dst, mp, mn, center, edge)
    if order == "RD":
        incoming_at_mirror = di
        outgoing_at_mirror = unit(q - m)
        incident_at_edge = outgoing_at_mirror
        outgoing_at_edge = unit(dst - q)
        incoming_length = np.dot(m - src, di)
        source_launch = m - incoming_length * di
        segments = [incoming_length, np.linalg.norm(q-m), np.linalg.norm(dst-q)]
        outgoing_range = segments[2]
        effective_source = reflected_point(src, mp, mn)
        effective_direction = reflected_direction(di, mn)
        effective_destination = dst
        mirror_side_product = np.dot(src-mp,mn) * np.dot(q-mp,mn)
    else:
        incoming_at_mirror = unit(m-q)
        outgoing_at_mirror = unit(dst-m)
        incident_at_edge = di
        outgoing_at_edge = incoming_at_mirror
        incoming_length = np.dot(q-src, di)
        source_launch = q - incoming_length * di
        segments = [incoming_length, np.linalg.norm(m-q), np.linalg.norm(dst-m)]
        outgoing_range = segments[1]+segments[2]
        effective_source = src
        effective_direction = di
        effective_destination = reflected_point(dst,mp,mn)
        mirror_side_product = np.dot(q-mp,mn) * np.dot(dst-mp,mn)
    assert min(segments) > 0, "Every physical segment must run forward"
    assert mirror_side_product > 0, "Physical points must share a mirror half-space"
    mirror_law = np.linalg.norm(outgoing_at_mirror - reflected_direction(incoming_at_mirror,mn))
    stationarity = abs(np.dot(incident_at_edge-outgoing_at_edge,edge))
    sin_beta = np.linalg.norm(np.cross(incident_at_edge,edge))
    wi, wo = -incident_at_edge, outgoing_at_edge
    phi_i = np.arctan2(np.dot(wi,normal_front),np.dot(wi,tangent_front))
    phi_o = np.arctan2(np.dot(wo,normal_front),np.dot(wo,tangent_front))%(2*np.pi)
    gauge = np.dot(effective_direction,effective_destination-effective_source)
    excess = physical_phase - gauge
    k = 2*np.pi/wavelength
    transition_L = outgoing_range*sin_beta**2
    phase_spreading = np.exp(-1j*k*excess)/np.sqrt(outgoing_range)
    coefficients = {bc: reference.coefficient(k,2.,phi_i,phi_o,transition_L,sin_beta,bc)
                    for bc in ("soft","hard")}
    # For these forward half-plane fixtures no local planar GO reflection is
    # geometrically possible. The source-plane ray crosses z=0 on x<0 if lit.
    direct_weight = 1. if horizontal < 0 else 0. if horizontal > 0 else .5
    raw_edge = {bc: coefficients[bc]*phase_spreading for bc in coefficients}
    transported_edge = {bc: (-1 if bc=="soft" else 1)*raw_edge[bc] for bc in raw_edge}
    total = {bc: (-1 if bc=="soft" else 1)*(direct_weight+raw_edge[bc]) for bc in raw_edge}

    # Exact Sommerfeld half-plane field independently checks full GO + UTD
    # soft/hard amplitude, including obliquity via the transverse wave number.
    transverse_dst = destination_local.copy(); transverse_dst[1] = 0
    transverse_radius = np.linalg.norm(transverse_dst)
    phi_i_exact = np.arctan2(d_local[2],-d_local[0])
    phi_o_exact = np.arctan2(-destination_local[2],destination_local[0])%(2*np.pi)
    gauge_transverse = np.dot(d_local,transverse_dst)
    exact_total = {bc: (-1 if bc=="soft" else 1)*reference.sommerfeld(
        k*sin_beta,transverse_radius,phi_i_exact,phi_o_exact,bc)*np.exp(1j*k*gauge_transverse)
        for bc in coefficients}
    exact_error = max(abs(total[bc]-exact_total[bc]) for bc in total)

    # Closed-form unfolding is the comparator, not the numerical solver.
    delta = effective_destination-center
    axial = np.dot(delta,edge)
    rho = np.linalg.norm(delta-axial*edge)
    c = np.dot(effective_direction,edge)
    q_unfolded = center+edge*(axial-rho*c/np.sqrt(1-c*c))
    ro_unfolded = np.linalg.norm(effective_destination-q_unfolded)
    phase_unfolded = np.dot(effective_direction,q_unfolded-effective_source)+ro_unfolded
    unfolded_error = np.linalg.norm(q-q_unfolded)
    range_error = abs(outgoing_range-ro_unfolded)
    eikonal_error = abs(physical_phase-phase_unfolded)
    assert mirror_law < 1e-11
    assert stationarity < 1e-11
    assert unfolded_error < 1e-9
    assert range_error < 1e-10
    assert eikonal_error < 1e-10
    assert exact_error < 2e-9
    assert abs(sum(segments)-physical_phase) < 1e-10
    assert abs(np.dot(source_launch-src,di)) < 1e-11
    complex_array = lambda z: [float(z.real),float(z.imag)]
    label = f"{order}-{angle:.2f}-{horizontal:+.2f}-{mirror_variant}-{'rotated' if rotation else 'base'}"
    return {
        "id": label,"order":order,"angle":angle,"horizontalOffset":horizontal,
        "mirrorVariant":mirror_variant,"rotated":rotation,
        "mesh":{"vertices":[point(v).tolist() for v in [[0,-300,0],[300,-300,0],[300,300,0],[0,300,0]]],
                "faces":[[0,1,2],[0,2,3]]},
        "state":{"sourceOrigin":src.tolist(),"incidentDirection":di.tolist(),
                 "detectorCenter":dst.tolist(),
                 "detectorU":direction(reflected_direction(np.array([1.,0.,0.]),mn_local) if order=="DR" else [1,0,0]).tolist(),
                 "detectorV":direction(reflected_direction(np.array([0.,1.,0.]),mn_local) if order=="DR" else [0,1,0]).tolist(),"sigma0Mm":3.,"wavelengthMm":wavelength},
        "mirror":{"point":mp.tolist(),"normal":mn.tolist()},
        "expected":{
            "mirrorPoint":m.tolist(),"edgePoint":q.tolist(),"sourceLaunch":source_launch.tolist(),
            "physicalSegments":list(map(float,segments)),"physicalEikonal":float(physical_phase),
            "gauge":float(gauge),"phaseExcess":float(excess),"outgoingRange":float(outgoing_range),
            "sinBeta":float(sin_beta),"L":float(transition_L),"phi_i":float(phi_i),"phi_o":float(phi_o),
            "directWeight":direct_weight,"coefficient":{bc:complex_array(z) for bc,z in coefficients.items()},
            "edgeBeforeMirrorSign":{bc:complex_array(z) for bc,z in raw_edge.items()},
            "edge":{bc:complex_array(z) for bc,z in transported_edge.items()},
            "total":{bc:complex_array(z) for bc,z in total.items()},
            "exactTotal":{bc:complex_array(z) for bc,z in exact_total.items()},
            "unpolarizedPower":float(sum(abs(z)**2 for z in total.values())/2),
            "effectiveSource":effective_source.tolist(),"effectiveDirection":effective_direction.tolist(),
            "effectiveDestination":effective_destination.tolist(),
        },
        "verification":{"mirrorLawResidual":float(mirror_law),"edgeStationarityResidual":float(stationarity),
                        "stationaryPointVsUnfolding":float(unfolded_error),"rangeVsUnfolding":float(range_error),
                        "eikonalVsUnfolding":float(eikonal_error),"exactSommerfeldFieldError":float(exact_error),
                        "mirrorHalfspaceProduct":float(mirror_side_product),"optimizer":optimizer},
    }



def bounded_strip_fixture(order, angle, horizontal, mirror_variant, rotation):
    """Two physical edges of the UI-size plate, independently solved one by one."""
    f = fixture(order,angle,horizontal,mirror_variant,rotation)
    src=np.array(f["state"]["sourceOrigin"]);di=np.array(f["state"]["incidentDirection"])
    dst=np.array(f["state"]["detectorCenter"]);mp=np.array(f["mirror"]["point"]);mn=np.array(f["mirror"]["normal"])
    if rotation:
        matrix=Rotation.from_euler("xyz",[23,31,17],degrees=True).as_matrix()
        translation=np.array([2.,-3.,4.])
    else:matrix,translation=np.eye(3),np.zeros(3)
    point=lambda p:matrix@np.asarray(p)+translation
    direction=lambda d:matrix@np.asarray(d)
    front=direction([0,0,-1]);edge=direction([0,1,0])
    f["mesh"]={"vertices":[point(p).tolist() for p in [[-1,-10,0],[1,-10,0],[1,10,0],[-1,10,0]]],"faces":[[0,1,2],[0,2,3]]}
    f["state"]["sigma0Mm"]=1.
    f["id"]="bounded-"+f["id"]
    effective_dst=np.array(f["expected"]["effectiveDestination"])
    effective_di=np.array(f["expected"]["effectiveDirection"])
    gauge=f["expected"]["gauge"];k=2*np.pi/f["state"]["wavelengthMm"]
    complex_array=lambda z:[float(z.real),float(z.imag)]
    rows=[]
    for edge_x in (-1.,1.):
        center=point([edge_x,0,0]);tangent=direction([-edge_x,0,0])
        m,q,phase,opt=physical_fermat(order,src,di,dst,mp,mn,center,edge,10.)
        if order=="RD":
            incoming_mirror=di;outgoing_mirror=unit(q-m)
            incident=outgoing_mirror;outgoing=unit(dst-q)
            length=np.linalg.norm(dst-q)
            segments=[np.dot(m-src,di),np.linalg.norm(q-m),length]
        else:
            incoming_mirror=unit(m-q);outgoing_mirror=unit(dst-m)
            incident=di;outgoing=incoming_mirror
            segments=[np.dot(q-src,di),np.linalg.norm(m-q),np.linalg.norm(dst-m)]
            length=segments[1]+segments[2]
        sin_beta=np.linalg.norm(np.cross(incident,edge))
        phi_i=np.arctan2(np.dot(-incident,front),np.dot(-incident,tangent))
        phi_o=np.arctan2(np.dot(outgoing,front),np.dot(outgoing,tangent))%(2*np.pi)
        L=length*sin_beta**2;excess=phase-gauge
        coeff={bc:reference.coefficient(k,2.,phi_i,phi_o,L,sin_beta,bc) for bc in ("soft","hard")}
        raw={bc:coeff[bc]*np.exp(-1j*k*excess)/np.sqrt(length) for bc in coeff}
        transported={bc:(-1 if bc=="soft" else 1)*z for bc,z in raw.items()}
        # Each isolated-edge contribution is independently the exact half-plane
        # solution minus that half-plane's GO field. This does not claim the sum
        # is an exact finite-strip Maxwell solution.
        transverse=effective_dst-center-np.dot(effective_dst-center,edge)*edge
        radius=np.linalg.norm(transverse)
        pi_exact=np.arctan2(np.dot(-effective_di,front),np.dot(-effective_di,tangent))
        po_exact=np.arctan2(np.dot(transverse,front),np.dot(transverse,tangent))%(2*np.pi)
        crossing=-edge_x*(horizontal-edge_x)
        half_go=1. if crossing < -1e-10 else 0. if crossing > 1e-10 else .5
        exact={bc:reference.sommerfeld(k*sin_beta,radius,pi_exact,po_exact,bc)*np.exp(1j*k*np.dot(effective_di,transverse))-half_go for bc in coeff}
        exact_error=max(abs(raw[bc]-exact[bc]) for bc in raw)
        mirror_res=np.linalg.norm(outgoing_mirror-reflected_direction(incoming_mirror,mn))
        stationary=abs(np.dot(incident-outgoing,edge))
        assert exact_error<2e-9
        assert mirror_res<1e-11 and stationary<1e-11
        assert min(segments)>0
        rows.append({"edgeX":edge_x,"edgePoint":q.tolist(),"mirrorPoint":m.tolist(),"physicalEikonal":float(phase),
                     "physicalSegments":list(map(float,segments)),"phaseExcess":float(excess),"outgoingRange":float(length),
                     "sinBeta":float(sin_beta),"L":float(L),"phi_i":float(phi_i),"phi_o":float(phi_o),
                     "coefficient":{bc:complex_array(z) for bc,z in coeff.items()},
                     "edgeBeforeMirrorSign":{bc:complex_array(z) for bc,z in raw.items()},
                     "edge":{bc:complex_array(z) for bc,z in transported.items()},
                     "exactSommerfeldEdgeError":float(exact_error),"mirrorLawResidual":float(mirror_res),
                     "edgeStationarityResidual":float(stationary),"optimizer":opt})
    direct_weight=1. if abs(horizontal)>1+1e-10 else .5 if abs(abs(horizontal)-1)<1e-10 else 0.
    total={bc:(-1 if bc=="soft" else 1)*direct_weight+sum(complex(*r["edge"][bc]) for r in rows) for bc in ("soft","hard")}
    # Discard the unrelated half-plane expected field while retaining the
    # physical/effective frames and common gauge explicitly.
    f["expected"]={key:f["expected"][key] for key in ("effectiveSource","effectiveDirection","effectiveDestination","gauge")}
    f["expected"].update({"edges":rows,"directWeight":direct_weight,
                           "total":{bc:complex_array(z) for bc,z in total.items()},
                           "unpolarizedPower":float(sum(abs(z)**2 for z in total.values())/2)})
    f["verification"]={"exactSommerfeldEdgeError":max(r["exactSommerfeldEdgeError"] for r in rows),
                         "mirrorLawResidual":max(r["mirrorLawResidual"] for r in rows),
                         "edgeStationarityResidual":max(r["edgeStationarityResidual"] for r in rows)}
    return f


def generate():
    fixtures = [fixture(order, angle, horizontal, mirror, rotated)
                for order in ("RD","DR")
                for angle in (0.,.27)
                for horizontal in (-.6,0.,.6)
                for mirror in ("45deg","tilted")
                for rotated in (False,True)]
    fields = ("mirrorLawResidual","edgeStationarityResidual","stationaryPointVsUnfolding",
              "rangeVsUnfolding","eikonalVsUnfolding","exactSommerfeldFieldError")
    return {
        "method":"Independent SciPy constrained physical broken-path minimization; analytic gradient/Hessian stationarity polish; existing SciPy erfcx UTD coefficient and exact Sommerfeld field",
        "sourceModel":"Locally collimated plane wave referenced to its source plane, not a spherical emitter",
        "timeConvention":"exp(+i omega t); outgoing exp(-ik optical length)",
        "mirrorModel":"Infinite ideal scalar plane: soft reflection factor -1, hard +1; one selected RD or DR family",
        "cases":len(fixtures),
        "maxErrors":{field:max(f["verification"][field] for f in fixtures) for field in fields},
        "maxOptimizerGradient":max(f["verification"]["optimizer"]["maxGradient"] for f in fixtures),
        "maxThreeStartObjectiveSpread":max(f["verification"]["optimizer"]["threeStartObjectiveSpread"] for f in fixtures),
        "minPhysicalSegment":min(min(f["expected"]["physicalSegments"]) for f in fixtures),
        "limitations":["Independent physical-ray and canonical half-plane validation, not arbitrary mesh accuracy",
                       "No DD, general path-integral transfer, finite mirror clipping, endpoint/corner diffraction, or multiple-family coherent sum",
                       "No Gaussian beam amplitude factor or calibrated irradiance claim",
                       "No renderer or GPU verification unless executed by the companion JavaScript harness"],
        "fixtures":fixtures,
        "boundedStripFixtures":[bounded_strip_fixture(order,angle,horizontal,mirror,rotated)
                                for order in ("RD","DR") for angle in (0.,.27)
                                for horizontal in (-1.6,-1.,0.,1.,1.6)
                                for mirror in ("45deg","tilted") for rotated in (False,True)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary",action="store_true")
    args = parser.parse_args()
    report = generate()
    if args.summary:
        report.pop("fixtures")
        report["boundedStripCases"]=len(report.pop("boundedStripFixtures"))
    print(json.dumps(report,indent=2))
