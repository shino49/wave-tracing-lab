"""Fast mathematical tests for the isolated DD reference (no app imports)."""
import sys,unittest,json
from pathlib import Path
import numpy as np
from scipy.special import hankel2
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/dd-reference'))
from asymptote import SOURCE,E1,E2,evaluate,angle
from exact_halfplane import green
from helmholtz import solve
from analyze import extrapolate

class ReferenceTests(unittest.TestCase):
    def test_true_double_shadow(self):
        P=np.array([4.,0.])
        def y_at_x(a,b,x):return a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0])
        self.assertLess(y_at_x(SOURCE,E2,0),1)
        self.assertLess(y_at_x(E1,P,2),2)
        self.assertLess(y_at_x(SOURCE,P,0),1)
        self.assertGreater(y_at_x(E2,np.array([-4.,0.]),0),1)
        self.assertGreater(E2[1],E1[1])
    def test_no_plane_second_spreading(self):
        a=evaluate()
        s,d,r=a['leg_lengths']
        self.assertAlmostEqual(a['L1'],s*d/(s+d))
        self.assertAlmostEqual(a['L2'],d*r/(d+r))
        self.assertAlmostEqual(a['spreading_factors'][1],1/np.sqrt(r))
        self.assertNotAlmostEqual(a['spreading_factors'][1],np.sqrt(d/(r*(d+r))))
        b=evaluate(planar_both=True)
        self.assertAlmostEqual(b['L1'],d);self.assertAlmostEqual(b['L2'],r)
    def test_exact_green_flat_plane_images(self):
        k,r,phi,rs,phis=8,3,.6,2,2.4
        for bc in ['soft','hard']:
            direct=np.sqrt(r*r+rs*rs-2*r*rs*np.cos(phi-phis))
            image=np.sqrt(r*r+rs*rs-2*r*rs*np.cos(phi+phis))
            reference=-.25j*(hankel2(0,k*direct)+(-1 if bc=='soft' else 1)*hankel2(0,k*image))
            self.assertLess(abs(green(k,r,phi,rs,phis,bc,n=1)-reference),2e-14)
    def test_exact_green_reciprocity(self):
        for bc in ['soft','hard']:
            a=green(8,3,.5,2,4.5,bc)
            b=green(8,2,4.5,3,.5,bc)
            self.assertLess(abs(a-b),1e-14)
    def test_curvature_plane_converge_in_gtd_limit(self):
        for bc in ['soft','hard']:
            errors=[]
            for k in [8,1e5]:
                a=complex(*evaluate(k,boundary=bc)['field'])
                b=complex(*evaluate(k,boundary=bc,planar_both=True)['field'])
                errors.append(abs(a-b)/abs(a))
            self.assertLess(errors[1],errors[0]/100)
    def test_cylindrical_spreading_and_optical_phase(self):
        k=1e5;ray=np.array([1.,-1.])/np.sqrt(2)
        for bc in ['soft','hard']:
            a=complex(*evaluate(k,receiver=E2+3*ray,boundary=bc)['field'])
            b=complex(*evaluate(k,receiver=E2+6*ray,boundary=bc)['field'])
            ratio=b/a*np.exp(1j*k*3)
            self.assertLess(abs(ratio-1/np.sqrt(2)),2e-5)
    def test_approved_fixture_reference_controls(self):
        report=json.loads((ROOT/'research/dd-reference/validation.json').read_text())
        self.assertEqual(len(report['cases']),2)
        for c in report['exact_halfplane_controls']:
            self.assertLess(c['complex_relative_error'],.006)
        for c in report['cases']:
            self.assertLess(c['extrapolation_fit_max_spread_relative'],.01)
            self.assertTrue(c['uncertainty_is_empirical_not_rigorous_bound'])
        self.assertIn('Not validated',report['native_bundle_integration_status'])
    def test_pde_input_rejections(self):
        for kw in [{'k':0},{'h':-1},{'boundary':'absorbing'},{'screens':3}]:
            with self.assertRaises(ValueError):solve(**kw)
if __name__=='__main__':unittest.main()
