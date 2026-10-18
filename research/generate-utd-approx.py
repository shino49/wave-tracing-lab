#!/usr/bin/env python3
"""Generate scalar-UTD numerical data independently of the JavaScript kernel.

Requires NumPy/SciPy. Coefficients are computed from SciPy's complex erfcx;
fixtures use the separate reference equations in utd-reference.py and, for
half-planes, the exact Sommerfeld erfc solution (not the UTD implementation).
No expected values are obtained by executing JavaScript or WGSL.
"""
from pathlib import Path
import importlib.util
import json
import re
import sys
import numpy as np
from numpy.polynomial.chebyshev import chebinterpolate, chebval
from scipy.special import erfcx

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('utd_reference', ROOT / 'research/utd-reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
E = np.exp(1j * np.pi / 4)
DEGREE = 40


def scaled_erfcx(s):
    t = (s + 1) / 2
    u = t / (1 - t)
    return erfcx(E * u) / (1 - t)


def pair(z):
    return [float(np.real(z)), float(np.imag(z))]


def install_coefficients(coefficients):
    js = 'export const ERFCX_CHEBYSHEV = ' + json.dumps([pair(c) for c in coefficients], separators=(',', ':')) + ';'
    rows = ',\n  '.join('vec2f(' + ','.join(format(v, '.17g') for v in pair(c)) + ')' for c in coefficients)
    wgsl = f'const UTD_ERFCX_CHEBYSHEV = array<vec2f,{len(coefficients)}>(\n  {rows}\n);'
    # Keep the exact polynomial, but emit separate scalar streams with literal
    # coefficients rather than a dynamically indexed vector recurrence.
    scalar_steps = []
    for j in range(len(coefficients) - 1, 0, -1):
        re_c, im_c = (format(v, '.17g') for v in pair(coefficients[j]))
        scalar_steps.append(f'// coefficient {j}\n{{let nr=x*br-cr+({re_c});let ni=x*bi-ci+({im_c});cr=br;ci=bi;br=nr;bi=ni;}}')
    re_0, im_0 = (format(v, '.17g') for v in pair(coefficients[0]))
    wgsl += ('\n\n// Same degree-40 Chebyshev polynomial and coefficients as the f64 reference.\n'
        '// Separate scalar streams and literal indices avoid vector/dynamic-array\n'
        '// lowering in this numerically sensitive recurrence; no change to UTD physics.\n'
        'fn utd_erfcx_scaled(t:f32)->vec2f {\n let s=2.0*t-1.0;let x=2.0*s;\n'
        ' var br=0.0;var bi=0.0;var cr=0.0;var ci=0.0;\n '
        + '\n '.join(scalar_steps)
        + f'\n return vec2f(s*br-cr+({re_0}),s*bi-ci+({im_0}));\n}}')

    for filename, body in [('utd-math.js', js), ('utd-functions.wgsl', wgsl)]:
        target = ROOT / 'dist' / filename
        original = target.read_text()
        updated, count = re.subn(r'// BEGIN GENERATED COEFFICIENTS\n.*?// END GENERATED COEFFICIENTS',
            '// BEGIN GENERATED COEFFICIENTS\n' + body + '\n// END GENERATED COEFFICIENTS', original, flags=re.S)
        assert count == 1
        target.write_text(updated)


def create_fixtures():
    transitions = [{'x': float(x), 'expected': pair(reference.transition(x))}
        for x in np.unique(np.r_[0, np.logspace(-28, 20, 721), [0.1, 1, 4, 6, 10, 100]])]
    coefficients = []
    for n in (1, 1.00001, 1.05, 1.25, 1.5, 1.75, 1.99, 2):
        for k in (.1, 1, 100, 1e6):
            for pi in (.15, .6, 1.2, 2.5):
                angles = list(np.linspace(0, n * np.pi, 13))
                angles += [center + offset for center in (np.pi - pi, np.pi + pi, 2 * n * np.pi - np.pi - pi)
                    for offset in (-1e-8, -1e-12, 0, 1e-12, 1e-8)]
                for po in angles:
                    if not 0 <= po <= n * np.pi:
                        continue
                    for boundary in ('soft', 'hard'):
                        args = [k, n, pi, float(po), 1, 1, boundary]
                        coefficients.append({'args': args, 'expected': pair(reference.coefficient(*args))})
    # This is an exact, independently formulated physical comparison.
    halfplanes = []
    for boundary in ('soft', 'hard'):
        for kr in (.1, 1, 10, 100, 1000):
            for pi in (.15, .6, 1.2, 2.5):
                angles = list(np.linspace(0, 2 * np.pi, 81))
                angles += [center + offset for center in (np.pi + pi, np.pi - pi)
                    for offset in (-1e-8, -1e-12, 0, 1e-12, 1e-8)]
                for po in angles:
                    args = [kr, 1, 2, pi, float(po), boundary]
                    halfplanes.append({'args': args, 'expected': pair(reference.sommerfeld(kr, 1, pi, po, boundary))})
    # High-frequency seam tests: f32-rounded inputs, dense offsets around EPS.
    seam = []
    for kr in (1, 1000, 1e5, 1e6):
        for pi in (.15, .6, 1.2, 2.5):
            for boundary in ('soft', 'hard'):
                for center in (np.pi - pi, np.pi + pi):
                    for offset in (-1e-4, -4e-6, -2.1e-6, -2e-6, -1.9e-6, -1e-6, 0, 1e-6, 1.9e-6, 2e-6, 2.1e-6, 4e-6, 1e-4):
                        po = center + offset
                        kr32, pi32, po32 = map(lambda x: float(np.float32(x)), (kr, pi, po))
                        args = [kr32, 1, 2, pi32, po32, boundary]
                        seam.append({'args': args, 'offset': offset, 'expected': pair(reference.sommerfeld(kr32, 1, pi32, po32, boundary))})
    out = {'provenance': 'SciPy erfcx reference plus exact Sommerfeld erfc; never derived from JS/WGSL',
        'timeConvention': 'exp(+i omega t), outgoing exp(-ikr)',
        'transition': transitions, 'coefficient': coefficients, 'halfplane': halfplanes, 'float32Seam': seam}
    (ROOT / 'research/utd-fixtures.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
    return {key: len(value) for key, value in out.items() if isinstance(value, list)}


def main():
    coefficients = chebinterpolate(scaled_erfcx, DEGREE)
    s = np.unique(np.r_[np.linspace(-1, 1, 100001), 2 * (1 - 10. ** np.arange(-16, 0, .01)) - 1,
        2 * (10. ** np.arange(-16, 0, .01)) - 1])
    expected = np.zeros(len(s), dtype=complex)
    end = s == 1
    expected[~end] = scaled_erfcx(s[~end])
    expected[end] = 1 / (np.sqrt(np.pi) * E)
    errors = np.abs(chebval(s, coefficients) - expected)
    approximation = {'degree': DEGREE, 'verificationSamples': len(s), 'maximumAbsoluteHError': float(errors.max()),
        'mapping': 'H(t)=(1+u) erfcx(exp(i*pi/4)*u); t=u/(1+u); Chebyshev coordinate s=2*t-1',
        'coefficients': [pair(c) for c in coefficients]}
    assert errors.max() < 4e-13
    (ROOT / 'research/utd-approximation.json').write_text(json.dumps(approximation, indent=2) + '\n')
    install_coefficients(coefficients)
    counts = create_fixtures()
    print(json.dumps({'approximationMaxAbsoluteError': float(errors.max()), 'fixtureCounts': counts}, indent=2))


if __name__ == '__main__':
    main()
