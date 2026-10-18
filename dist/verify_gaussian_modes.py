"""Independent scalar/paraxial Wave Tracing mode-overlap checks.

No upstream repository code is executed or copied. The reference evaluates the
Fresnel field first, then convolves its irradiance with the detector Gaussian.
The tested estimator instead transports backward Gaussian modes, intersects
all aperture pieces with a local footprint, squares their aggregate overlap,
and sums nonnegative mode powers. Distances are millimetres.

Run: python research/verify_gaussian_modes.py
Requires numpy and scipy. Writes physics-validation.json next to this file.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy.special import roots_legendre, roots_hermitenorm

@lru_cache(None)
def legendre(n):
    return roots_legendre(n)

@lru_cache(None)
def normal_rule(n):
    nodes,weights=roots_hermitenorm(n)
    return nodes,weights/np.sqrt(2*np.pi)

@dataclass
class Scene:
    wavelength: float=.00055
    distance: float=2000.
    slit_width: float=.08
    separation: float=.30
    height: float=.18
    detector_sigma: float=.035
    source_sigma: float=.7

def intervals(scene,axis):
    if axis=='y': return [(-scene.height/2,scene.height/2)]
    return [(c-scene.slit_width/2,c+scene.slit_width/2)
            for c in (-scene.separation/2,scene.separation/2)]

def source(t,scene):
    return np.exp(-t*t/(4*scene.source_sigma**2))

def mode_power_1d(X,xi,scene,axis='x',order=16,cutoff=5.):
    """One q-mode sample, with proposal density cancellation included."""
    k=2*np.pi/scene.wavelength
    rho=scene.distance/k
    S=2*scene.detector_sigma**2
    D=S*S+rho*rho
    sigmaL=np.sqrt(D/(2*S))
    c=-sigmaL*xi
    a=S/(2*D)
    b=rho/(2*D)
    phase_linear=-k*X/scene.distance+S*S*c/(D*rho)
    nodes,weights=legendre(order)
    total=0j
    for lo,hi in intervals(scene,axis):
        if cutoff is not None:
            lo=max(lo,c-cutoff*sigmaL)
            hi=min(hi,c+cutoff*sigmaL)
        if hi<=lo: continue
        t=(lo+hi)/2+(hi-lo)*nodes/2
        amplitude=source(t,scene)*np.exp(-a*t*t+2*a*c*t)
        phase=b*t*t+phase_linear*t
        total+=(hi-lo)/2*np.sum(weights*amplitude*np.exp(1j*phase))
    return float(abs(total)**2/(scene.wavelength*scene.distance))

def expected_mode_1d(X,scene,axis='x',order=16,cutoff=5.,mode_order=96):
    xi,weights=normal_rule(mode_order)
    powers=np.array([mode_power_1d(X,z,scene,axis,order,cutoff) for z in xi])
    mean=float(weights@powers)
    variance=max(0.,float(weights@(powers*powers)-mean*mean))
    return mean,variance

def reference_1d(X,scene,axis='x',aperture_order=100,pixel_order=64):
    """Independent Fresnel propagation followed by intensity convolution."""
    xi,pixel_weights=normal_rule(pixel_order)
    Y=X+scene.detector_sigma*xi
    nodes,weights=legendre(aperture_order)
    field=np.zeros(Y.shape,dtype=complex)
    for lo,hi in intervals(scene,axis):
        t=(lo+hi)/2+(hi-lo)*nodes/2
        phase=np.pi*(Y[:,None]-t[None,:])**2/(scene.wavelength*scene.distance)
        field+=(hi-lo)/2*np.sum(weights[None,:]*source(t,scene)[None,:]*np.exp(1j*phase),axis=1)
    return float(pixel_weights@(abs(field)**2)/(scene.wavelength*scene.distance))

def run():
    cases=[Scene(),Scene(wavelength=.00042,distance=1500.,slit_width=.12,separation=.42),
           Scene(wavelength=.00068,distance=4000.,slit_width=.04,separation=.18),
           Scene(wavelength=.00042,distance=4000.,slit_width=.04,separation=.42),
           Scene(wavelength=.00068,distance=1500.,slit_width=.12,separation=.18)]
    positions={'x':[0.,.7,2.1,5.,8.,12.], 'y':[0.,.7,2.1,5.,7.5]}
    rows=[]
    for index,scene in enumerate(cases):
        for axis,ps in positions.items():
            for X in ps:
                ref=reference_1d(X,scene,axis)
                mode,_=expected_mode_1d(X,scene,axis,64,None)
                t5,var=expected_mode_1d(X,scene,axis,16,5.)
                t5high,_=expected_mode_1d(X,scene,axis,64,5.)
                rows.append(dict(scene=index,axis=axis,position=X,reference=ref,
                                 mode_untruncated=mode,mode_t5_n16=t5,
                                 relative_untruncated_error=abs(mode-ref)/max(ref,1e-30),
                                 relative_t5_error=abs(t5-ref)/max(ref,1e-30),
                                 relative_n16_error=abs(t5-t5high)/max(t5high,1e-30),
                                 sample_cv=np.sqrt(var)/max(t5,1e-30)))
    # Two-dimensional measurement factorizes because geometry/source/mode pdf do.
    products=[]
    for index,scene in enumerate(cases):
        for X,Y in [(0.,0.),(.7,.7),(5.,2.1),(12.,7.5)]:
            mx,_=expected_mode_1d(X,scene,'x',16,5.)
            my,_=expected_mode_1d(Y,scene,'y',16,5.)
            ref=reference_1d(X,scene,'x')*reference_1d(Y,scene,'y')
            products.append(dict(scene=index,position=[X,Y],reference=ref,mode=mx*my,
                                 relative_error=abs(mx*my-ref)/max(ref,1e-30)))
    # Envelope is a conservative ellipse around tensor cutoff at every distance.
    sig=np.array([.035,.06]); lam=.00042; k=2*np.pi/lam; T=5.
    z=np.linspace(0,4000,101)
    exact=np.sqrt(sig[None,:]**2+(z[:,None]/(2*k*sig[None,:]))**2)
    axes=np.sqrt(2)*T*(sig[None,:]+z[:,None]/(2*k*sig[None,:]))
    maximum_ellipse_coordinate=float(np.max(np.sum((T*exact/axes)**2,axis=1)))
    # Isotropic Gaussian mode sourcing satisfies sigma_x*sigma_k = 1/2.
    uncertainty=float(.035*(1/(2*.035)))
    report={
      'units':'millimetres; incident peak irradiance = 1',
      'scenes':[asdict(s) for s in cases],
      'method':'Backward Gaussian-mode overlap, q sampled from normal centred toward aperture; tensor footprint T=5; independent Fresnel plus Gaussian detector reference',
      'summary':{
        'max_relative_untruncated_error':max(r['relative_untruncated_error'] for r in rows),
        'max_relative_T5_error_1d':max(r['relative_t5_error'] for r in rows),
        'max_relative_T5_error_2d':max(r['relative_error'] for r in products),
        'max_relative_16_vs_64_quadrature_error':max(r['relative_n16_error'] for r in rows),
        'max_single_sample_coefficient_of_variation_1d':max(r['sample_cv'] for r in rows),
        'max_tensor_corner_ellipse_coordinate':maximum_ellipse_coordinate,
        'minimum_uncertainty_product':uncertainty,
      },
      'one_dimensional_cases':rows,'two_dimensional_cases':products,
      'energy_identity': 'In the untruncated continuum model, integral over the entire sensor plane equals integral of source irradiance over both aperture rectangles, by unitarity of Fresnel propagation and unit-integral detector Gaussian. A finite displayed screen captures only part of this power.',
      'limitations':['Scalar, monochromatic, paraxial, thin amplitude mask; no polarization, UTD, mirror/multibounce, evanescent fields, finite thickness, or Maxwell/BEM claim.',
                     'T=5 Gaussian-tail truncation and finite quadrature are numerical approximations.',
                     'These checks validate the mathematical CPU estimator; the actual WebGPU implementation still needs output comparison.']}
    target=Path(__file__).with_name('physics-validation.json')
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2))
    print('Wrote',target)
    assert report['summary']['max_relative_untruncated_error']<1e-7
    assert report['summary']['max_relative_16_vs_64_quadrature_error']<1e-6
    assert report['summary']['max_tensor_corner_ellipse_coordinate']<=1+1e-14
    assert report['summary']['max_relative_T5_error_2d']<1e-3

if __name__=='__main__': run()

# Optional numerical energy convergence check (also run by this script).
if __name__=='__main__':
    from scipy.integrate import simpson
    from scipy.special import erf
    def input_flux_1d(scene,axis):
        s=scene.source_sigma
        return sum(s*np.sqrt(np.pi/2)*(erf(hi/(np.sqrt(2)*s))-erf(lo/(np.sqrt(2)*s)))
                   for lo,hi in intervals(scene,axis))
    def screen_flux_1d(scene,axis,halfspan):
        Y=np.linspace(-halfspan,halfspan,8193)
        nodes,weights=legendre(160)
        field=np.zeros(Y.shape,dtype=complex)
        for lo,hi in intervals(scene,axis):
            t=(lo+hi)/2+(hi-lo)*nodes/2
            phase=np.pi*(t[None,:]**2-2*Y[:,None]*t[None,:])/(scene.wavelength*scene.distance)
            field+=(hi-lo)/2*np.sum(weights[None,:]*source(t,scene)[None,:]*np.exp(1j*phase),axis=1)
        return float(simpson(abs(field)**2,x=Y)/(scene.wavelength*scene.distance))
    scene=Scene()
    total=input_flux_1d(scene,'x')*input_flux_1d(scene,'y')
    energy=[]
    for halfspan in [12.,50.,100.,200.]:
        received=screen_flux_1d(scene,'x',halfspan)*screen_flux_1d(scene,'y',halfspan)
        energy.append(dict(square_halfspan_mm=halfspan,received_power=received,
                           transmitted_power=total,captured_fraction=received/total))
    actual_display=screen_flux_1d(scene,'x',12.)*screen_flux_1d(scene,'y',7.5)
    target=Path(__file__).with_name('physics-validation.json')
    report=json.loads(target.read_text())
    report['energy_convergence']=energy
    report['point_detector_power_in_default_display']={
        'received_power':actual_display,'transmitted_power':total,
        'captured_fraction':actual_display/total,
        'note':'Point-detector reference; Gaussian filtering preserves whole-plane power and slightly modifies finite-window boundaries.'}
    target.write_text(json.dumps(report,indent=2))
    print('Numerical finite-window energy convergence:',json.dumps(energy))
    assert all(0<row['captured_fraction']<1 for row in energy)
    assert energy[-1]['captured_fraction']>.98
