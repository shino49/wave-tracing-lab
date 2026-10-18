"""Fast checks of retained independent numerical evidence (does not rerun PDE)."""
import json,math,pathlib
r=json.loads((pathlib.Path(__file__).resolve().parent/'validation.json').read_text())
assert len(r['cases']) in (4,5)
for c in r['cases']:
 assert c['status'].startswith('empirically converged')
 assert c['empirical_complex_relative_uncertainty_budget']==1e-4
 for pair in c['reference_complex']:assert all(math.isfinite(x) for x in pair)
 if c['fixture']=='long_gap':
  assert max(c['controls']['independent_refinement_relative_change'])<(1e-6 if c['k']==512 else 1.2e-8)
  assert c['controls']['single_halfplane_relative_error']<(1e-6 if c['k']==512 else 8.2e-9)
  if 'centre_factor_prototype_power_comparison' in c:assert c['centre_factor_prototype_power_comparison']['full_native_traversal_status'].startswith('Not established here')
assert r['boundary']=='soft Dirichlet only'
print('PASS: retained converged soft references; retained controls finite and consistent. Native traversal remains separately scoped.')
