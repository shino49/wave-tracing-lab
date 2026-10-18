"""Summarize measured reference errors separately from UTD model errors."""
import json
from pathlib import Path
import numpy as np
from scipy.special import hankel2
from asymptote import evaluate, SOURCE,E1,E2
ROOT=Path(__file__).resolve().parent

def z(v):return complex(*v)
def pair(v):return [float(v.real),float(v.imag)]
def extrapolate(rows,degree=2,count=3,index=0):
    rows=sorted(rows,key=lambda r:-r['h'])[-count:]
    return np.polynomial.polynomial.polyfit([r['h'] for r in rows],[z(r['field'][index]) for r in rows],degree)[0]
def metrics(got,ref):
    return {'complex_relative_error':float(abs(got-ref)/abs(ref)),
            'signed_relative_power_error':float(abs(got/ref)**2-1),
            'phase_error_degrees':float(np.angle(got/ref)*180/np.pi)}

def analyze():
    data=json.loads((ROOT/'convergence.json').read_text())
    control=json.loads((ROOT/'halfplane-validation.json').read_text())
    pmldata=json.loads((ROOT/'pml-validation.json').read_text())
    records=[];controls=[]
    for bc in ['soft','hard']:
        rows=[r for r in data if r['boundary']==bc]
        ref=extrapolate(rows);alternatives=[extrapolate(rows,2,4),extrapolate(rows,3,4),extrapolate(rows,4,5)]
        inc=-.25j*hankel2(0,8*np.linalg.norm(E1-SOURCE))
        rec={'boundary':bc,'k':8.,'reference_complex':pair(ref),
             'reference_conditional_power':float(abs(ref/inc)**2),
             'finest_grid_h':min(r['h'] for r in rows),
             'extrapolation_fit_max_spread_relative':float(max(abs(v-ref)/abs(ref) for v in alternatives)),
             'numerical_reference_uncertainty_budget_complex_relative':.04,
             'uncertainty_is_empirical_not_rigorous_bound':True,
             'comparisons':{}}
        for name,planar in [('native_local_plane_both_regions',True),('sequential_curvature_corrected',False)]:
            a=evaluate(8,boundary=bc,planar_both=planar);u=z(a['field'])
            rec['comparisons'][name]={**metrics(u,ref),'complex_field':pair(u),'conditional_power':float(abs(u/inc)**2),
                                       'reduced_distances':[a['L1'],a['L2']]}
        records.append(rec)
        crows=[r for r in control if r['boundary']==bc]
        cref=extrapolate(crows)
        controls.append({'boundary':bc,'exact_complex':crows[0]['exact'],'extrapolated_complex':pair(cref),
                         **metrics(cref,z(crows[0]['exact']))})
    pml=[]
    for row in pmldata:
        baseline=next((r for r in data if r['h']==row['h'] and r['boundary']==row['boundary']),None)
        if baseline:pml.append({'boundary':row['boundary'],'h':row['h'],'bounds':row['bounds'],'strength':row['pml_strength'],
                               'relative_field_change_at_same_grid':abs(z(row['field'][0])-z(baseline['field'][0]))/abs(z(baseline['field'][0]))})
    result={'scope':'Independent 2D Helmholtz DD field benchmark; not native PLT recursion or GPU parity validation.',
        'geometry':{'source':SOURCE.tolist(),'edge1':E1.tolist(),'edge2':E2.tolist(),'receiver':[4.,0.],
                    'screens':'x=0,y<=1 and x=2,y<=2, both z-invariant'},
        'convention':'exp(+i omega t); outgoing -i H_0^(2)(kr)/4; (Delta+k^2)u=-delta',
        'source_normalization':'conditional power = |u(P)/G(k |E1-S|)|^2',
        'reference':'5-point conservative cell-centred full Helmholtz, complex-coordinate PML; h+h^2 extrapolation',
        'cases':records,'exact_halfplane_controls':controls,'pml_controls':pml,
        'native_bundle_integration_status':'Not validated. At k8 this fixture cannot support two disjoint 3-sigma minimum-uncertainty native interaction regions from the stated source. Conditional field validation only.',
        'high_frequency_k16_status':'Exploratory only. Mesh extrapolants disagree materially; not certified and not used for any accuracy claim.',
        'not_claimed':['3D Gaussian/bundle normalization','finite-edge endpoints','grazing or overlapping transitions',
                       'unrestricted multiple diffraction','absolute irradiance','global error bounds','hardware GPU validation']}
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':print(json.dumps(analyze(),indent=2))
