/**
 * Bounded scalar, paraxial, incoherent circular-pinhole optics.
 * All geometry is millimetres; wavelengths are explicitly nanometres.
 * This module does not import or use the local-UTD transport estimator.
 *
 * h(r) = |2π∫₀ᵃ ρ exp(iπρ²/(λ z_eff)) J₀(2πρr/(λz)) dρ|²
 *        / ((λz)² πa²),    z_eff⁻¹ = s⁻¹ + z⁻¹.
 * Analytic infinite-plane normalization only: ∫ h d²r = 1.
 * Finite detector/kernel mass is NEVER renormalized or wrapped back in.
 */
import {VISIBLE_BAND, SPECTRAL_MATERIALS, CIE_Y_INTEGRAL, cieXYZ,
  sourceSpectrum, reflectanceSpectrum, wavelengthNmToMm} from './plt-spectrum.js';

export const PINHOLE_OPTICS_SCOPE = 'finite-distance circular-pupil scalar Fresnel; incoherent scene points; paraxial; 360–830 nm CIE 1931 2°; physical mm; fixed relative radiometry';
export const PINHOLE_SPECTRAL_BASES = SPECTRAL_MATERIALS;
const PI = Math.PI;
const GAUSS = Object.freeze({
  1: [[0], [2]],
  2: [[-0.5773502691896257, 0.5773502691896257], [1, 1]],
  4: [[-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526],
      [0.34785484513745385,0.6521451548625461,0.6521451548625461,0.34785484513745385]],
  8: [[-0.9602898564975363,-0.7966664774136267,-0.525532409916329,-0.1834346424956498,0.1834346424956498,0.525532409916329,0.7966664774136267,0.9602898564975363],
      [0.1012285362903763,0.2223810344533745,0.3137066458778873,0.362683783378362,0.362683783378362,0.3137066458778873,0.2223810344533745,0.1012285362903763]],
});
function positive(x, name) { if (!Number.isFinite(x) || x <= 0) throw new RangeError(`${name} must be finite and positive`); return x; }
function abortError() { const e = new Error('Pinhole optics generation aborted'); e.name = 'AbortError'; return e; }
function checkAbort(hooks) { if (hooks.signal?.aborted || hooks.shouldAbort?.()) throw abortError(); }
async function checkpoint(hooks, progress) {
  checkAbort(hooks); (hooks.onProgress ?? hooks.progress)?.(progress);
  if (hooks.yield) await hooks.yield(); else await new Promise(resolve => setTimeout(resolve, 0));
  checkAbort(hooks);
}

// Integer cylindrical Bessel functions: normalized Miller recurrence near the
// turning point; stable forward recurrence from a high-accuracy asymptotic
// expansion when x exceeds every requested order. No large-q pupil sampling.
function asymptoticBessel(x, order) {
  const mu = 4 * order * order;
  let term = 1, p = 1, q = 0;
  for (let k = 1; k <= 24; k++) {
    term *= (mu - (2 * k - 1) ** 2) / (8 * k * x);
    if (k & 1) q += ((k >> 1) & 1 ? -1 : 1) * term;
    else p += ((k >> 1) & 1 ? -1 : 1) * term;
  }
  const phase = x - order * PI / 2 - PI / 4;
  return Math.sqrt(2 / (PI * x)) * (Math.cos(phase) * p - Math.sin(phase) * q);
}
function besselOrders(x, highest, out) {
  if (x === 0) { out.fill(0); out[0] = 1; return out; }
  if (x > Math.max(40, highest + 8)) {
    out[0] = asymptoticBessel(x, 0); out[1] = asymptoticBessel(x, 1);
    for (let n = 1; n < highest; n++) out[n+1] = 2*n/x*out[n] - out[n-1];
    return out;
  }
  const top = Math.ceil(Math.max(highest + 32, x + 40 + Math.sqrt(40*x)) / 2)*2;
  let jp = 0, j = 1, sum = 0;
  out.fill(0);
  for (let n = top; n >= 1; n--) {
    const jm = 2*n/x*j-jp; jp = j; j = jm;
    if (n-1 <= highest) out[n-1] = j;
    if ((n-1)%2 === 0) sum += (n===1 ? 1 : 2)*j;
    if (Math.abs(j) > 1e100) {
      j *= 1e-100; jp *= 1e-100; sum *= 1e-100;
      for(let k=n-1;k<=highest;k++) out[k] *= 1e-100;
    }
  }
  for (let n=0;n<=highest;n++) out[n] /= sum;
  return out;
}
export function besselJ0(x) { if (!Number.isFinite(x)) throw new RangeError('Nonfinite Bessel argument'); return besselOrders(Math.abs(x),1,new Float64Array(2))[0]; }
export function besselJ1(x) { if (!Number.isFinite(x)) throw new RangeError('Nonfinite Bessel argument'); return Math.sign(x)*besselOrders(Math.abs(x),1,new Float64Array(2))[1]; }
function sphericalBesselSeries(n,x) {
  let leading=1; for(let k=1;k<=n;k++) leading *= x/(2*k+1);
  let term=1,sum=1;
  for(let k=1;k<100;k++) {
    term *= -x*x/(2*k*(2*n+2*k+1)); sum += term;
    if(Math.abs(term)<2e-17*Math.max(1,Math.abs(sum))) break;
  }
  return leading*sum;
}
/** Dimensionless Fresnel amplitude, divided by pupil area, irrelevant phase omitted.
 * Exact Zernike expansion of the pupil chirp, not a fitted blur or Gaussian.
 * Intended bounded rig: β = πa²(1/s+1/z)/λ <= 14.
 */
export function makeFresnelEvaluator({diameterMm,depthMm,detectorDistanceMm=50,wavelengthNm,seriesTerms=28}) {
  const a=positive(diameterMm,'diameterMm')/2, s=positive(depthMm,'depthMm'),z=positive(detectorDistanceMm,'detectorDistanceMm');
  const lambda=wavelengthNmToMm(wavelengthNm),area=PI*a*a;
  const beta=PI*a*a*(1/s+1/z)/lambda;
  if(beta>16) throw new RangeError('Fresnel phase exceeds the validated bounded pinhole rig (β ≤ 16)');
  if(!Number.isInteger(seriesTerms)||seriesTerms<8||seriesTerms>64) throw new RangeError('seriesTerms must be an integer 8…64');
  const coefficients=Float64Array.from({length:seriesTerms},(_,n)=>(2*n+1)*sphericalBesselSeries(n,beta/2));
  const orders=new Float64Array(2*seriesTerms),qScale=2*PI*a/(lambda*z),densityScale=area/(lambda*z)**2;
  function amplitudeSquared(radiusMm) {
    if(!Number.isFinite(radiusMm)||radiusMm<0) throw new RangeError('radiusMm must be finite and nonnegative');
    const q=qScale*radiusMm;
    if(q<1e-7) return coefficients[0]**2;
    besselOrders(q,orders.length-1,orders);
    let re=0,im=0;
    for(let n=0;n<seriesTerms;n++) {
      const t=coefficients[n]*orders[2*n+1];
      switch(n%4) {case 0:re+=t;break;case 1:im-=t;break;case 2:re-=t;break;case 3:im+=t;break;}
    }
    return 4*(re*re+im*im)/(q*q);
  }
  return Object.freeze({diameterMm,depthMm:s,detectorDistanceMm:z,wavelengthNm,areaMm2:area,beta,seriesTerms,
    density:radiusMm=>densityScale*amplitudeSquared(radiusMm),amplitudeSquared});
}
export function fresnelPsfDensity(radiusMm,options) { return makeFresnelEvaluator(options).density(radiusMm); }
export function geometricPsfRadius({diameterMm,depthMm,detectorDistanceMm=50}) { return positive(diameterMm,'diameterMm')/2*(1+positive(detectorDistanceMm,'detectorDistanceMm')/positive(depthMm,'depthMm')); }
export function geometricPsfDensity(radiusMm,options) { if(!Number.isFinite(radiusMm)||radiusMm<0)throw new RangeError('Invalid radius');const radius=geometricPsfRadius(options);return radiusMm<radius?1/(PI*radius*radius):0; }

/** Shared fixed spectral measure: D65 * declared passive reflectance * CIE XYZ / ∫ybar dλ.
 * Narrowband is unit integrated spectral power at the chosen wavelength; it is
 * deliberately not normalized by ybar(λ), so wavelength luminance is preserved.
 */
export function spectralQuadrature({mode='broadband',wavelengthNm=550,spectralStepNm=5,basisNames=PINHOLE_SPECTRAL_BASES,source='D65'}={}) {
  if(!['broadband','narrowband'].includes(mode))throw new RangeError('mode must be broadband or narrowband');
  if(!Array.isArray(basisNames)||!basisNames.length||basisNames.some(n=>!SPECTRAL_MATERIALS.includes(n)))throw new RangeError('Unknown spectral basis');
  positive(spectralStepNm,'spectralStepNm');if(spectralStepNm>20)throw new RangeError('spectralStepNm must not exceed 20 nm');
  const wavelengths=[];
  if(mode==='narrowband') {wavelengthNmToMm(wavelengthNm);wavelengths.push([wavelengthNm,1]);}
  else {
    const intervals=Math.ceil(VISIBLE_BAND.widthNm/spectralStepNm),step=VISIBLE_BAND.widthNm/intervals;
    for(let i=0;i<=intervals;i++)wavelengths.push([VISIBLE_BAND.minNm+i*step,step*(i===0||i===intervals?.5:1)]);
  }
  return wavelengths.map(([nm,weight])=>({wavelengthNm:nm,weightNm:weight,
    basisXYZ:basisNames.map(name=>cieXYZ(nm).map(c=>c*weight*reflectanceSpectrum(nm,name)*(mode==='broadband'?sourceSpectrum(nm,source):1)/CIE_Y_INTEGRAL))}));
}
function validateGrid(options) {
  const resolution=options.resolution??512;
  if(!Number.isInteger(resolution)||resolution<8||resolution>2048)throw new RangeError('resolution must be an integer 8…2048');
  const sensorWidthMm=positive(options.sensorWidthMm??8,'sensorWidthMm'),detectorDistanceMm=positive(options.detectorDistanceMm??50,'detectorDistanceMm');
  const diameterMm=positive(options.diameterMm,'diameterMm');
  if(diameterMm<.04||diameterMm>.5)throw new RangeError('diameterMm must be within the validated 0.04…0.50 mm range');
  const depths=options.depths??[200,250,300];
  if(!Array.isArray(depths)||!depths.length||depths.some(s=>!Number.isFinite(s)||s<200||s>300))throw new RangeError('depths must lie within 200…300 mm');
  return {resolution,sensorWidthMm,detectorDistanceMm,diameterMm,depths};
}
/** Generate compact nonperiodic radial XYZ tables. Every finite source-to-sensor
 * offset is covered, including quadrature offsets. Work is bounded and yielded.
 * No 2D FFT, periodic PSF, image-dependent scaling, or sensor normalization here.
 */
export async function generateKernels(options,hooks={}) {
  const grid=validateGrid(options),{resolution,sensorWidthMm,detectorDistanceMm,diameterMm,depths}=grid;
  const basisNames=options.basisNames??[...PINHOLE_SPECTRAL_BASES];
  const samples=spectralQuadrature({...options,basisNames});
  const pixelWidthMm=sensorWidthMm/resolution,areaMm2=PI*(diameterMm/2)**2,throughput=areaMm2/detectorDistanceMm**2;
  const shortestNm=options.mode==='narrowband'?(options.wavelengthNm??550):VISIBLE_BAND.minNm;
  const radialStepMm=options.radialStepMm??Math.min(pixelWidthMm/4,shortestNm*1e-6*detectorDistanceMm/(diameterMm*12));
  positive(radialStepMm,'radialStepMm');
  const maxRadiusMm=options.maxRadiusMm??Math.SQRT2*sensorWidthMm;
  positive(maxRadiusMm,'maxRadiusMm');
  const radialCount=Math.ceil(maxRadiusMm/radialStepMm)+2;
  if(radialCount>200000)throw new RangeError('Radial table exceeds bounded allocation');
  const layers=[];let work=0;const total=depths.length*samples.length;
  for(const depthMm of depths) {
    checkAbort(hooks);
    const basis=basisNames.map(name=>({name,xyz:new Float64Array(radialCount*3),colorXYZ:new Float64Array(3)}));
    for(const sample of samples) {
      const evaluator=makeFresnelEvaluator({diameterMm,depthMm,detectorDistanceMm,wavelengthNm:sample.wavelengthNm,seriesTerms:options.seriesTerms??28});
      for(let b=0;b<basis.length;b++)for(let c=0;c<3;c++)basis[b].colorXYZ[c]+=sample.basisXYZ[b][c];
      for(let i=0;i<radialCount;i++) {
        const h=evaluator.density(i*radialStepMm),offset=3*i;
        for(let b=0;b<basis.length;b++) {
          const weights=sample.basisXYZ[b],out=basis[b].xyz;
          out[offset]+=h*weights[0];out[offset+1]+=h*weights[1];out[offset+2]+=h*weights[2];
        }
      }
      await checkpoint(hooks,{phase:'radial-kernels',completed:++work,total,depthMm,wavelengthNm:sample.wavelengthNm});
    }
    layers.push({depthMm,geometricRadiusMm:geometricPsfRadius({diameterMm,depthMm,detectorDistanceMm}),basis});
  }
  return { ...grid,basisNames:[...basisNames],mode:options.mode??'broadband',wavelengthNm:options.wavelengthNm??550,
    source:options.source??'D65',spectralStepNm:options.spectralStepNm??5,spectralSamples:samples.length,
    pixelWidthMm,pixelAreaMm2:pixelWidthMm**2,areaMm2,throughput,radialStepMm,maxRadiusMm:(radialCount-1)*radialStepMm,radialCount,
    layers,scope:PINHOLE_OPTICS_SCOPE,normalization:'analytic infinite-plane pupil normalization only; finite-detector loss retained',
    inputConvention:'source maps = radiance L*cos^4(theta), no area, distance, CIE, or exposure factor',
    outputConvention:'sensor irradiance XYZ; sum(image)*pixelAreaMm2 gives detector power XYZ'};
}
/** Interpolate physical PSF density XYZ; asking beyond the generated domain is
 * an error, never an implicit tail cut or periodic access. */
export function sampleRadialXYZ(kernels,depthIndex,basisIndex,radiusMm,out=new Float64Array(3)) {
  if(!Number.isFinite(radiusMm)||radiusMm<0||radiusMm>kernels.maxRadiusMm)throw new RangeError('PSF radius outside generated nonperiodic domain');
  const data=kernels.layers[depthIndex]?.basis[basisIndex]?.xyz;if(!data)throw new RangeError('Unknown PSF depth/basis');
  const p=radiusMm/kernels.radialStepMm,i=Math.min(Math.floor(p),kernels.radialCount-2),f=p-i,k=3*i;
  for(let c=0;c<3;c++)out[c]=data[k+c]*(1-f)+data[k+3+c]*f;
  return out;
}
function quadrantDiskArea(x,y,r) {
  x=Math.min(Math.abs(x),r);y=Math.min(Math.abs(y),r);
  const t=Math.sqrt(Math.max(0,r*r-y*y));
  if(x<=t)return x*y;
  const primitive=v=>.5*(v*Math.sqrt(Math.max(0,r*r-v*v))+r*r*Math.asin(Math.min(1,v/r)));
  return t*y+primitive(x)-primitive(t);
}
/** Exact uniform geometric disk / rectangle overlap, including subpixel rims. */
export function diskRectangleArea(radiusMm,x0,x1,y0,y1) {
  const r=positive(radiusMm,'radiusMm');
  if(![x0,x1,y0,y1].every(Number.isFinite)||x0>x1||y0>y1)throw new RangeError('Invalid rectangle');
  if(x0>=r||x1<=-r||y0>=r||y1<=-r)return 0;
  const primitive=(x,y)=>Math.sign(x)*Math.sign(y)*quadrantDiskArea(x,y,r);
  return Math.max(0,primitive(x1,y1)-primitive(x0,y1)-primitive(x1,y0)+primitive(x0,y0));
}
/** Integrate an XYZ PSF over one detector pixel. Returns probability mass XYZ,
 * before the single apertureArea / z² throughput multiplier. */
export function integratePixelXYZ(kernels,depthIndex,basisIndex,xMm,yMm,{kind='wave',pixelQuadrature=4,pixelWidthMm=kernels.pixelWidthMm}={},out=new Float64Array(3)) {
  positive(pixelWidthMm,'pixelWidthMm');if(!Number.isFinite(xMm)||!Number.isFinite(yMm))throw new RangeError('Invalid pixel position');
  const layer=kernels.layers[depthIndex],basis=layer?.basis[basisIndex];if(!basis)throw new RangeError('Unknown PSF depth/basis');
  const half=pixelWidthMm/2;out.fill(0);
  if(kind==='geometric') {
    const radius=layer.geometricRadiusMm,mass=diskRectangleArea(radius,xMm-half,xMm+half,yMm-half,yMm+half)/(PI*radius*radius);
    for(let c=0;c<3;c++)out[c]=mass*basis.colorXYZ[c];return out;
  }
  if(kind!=='wave')throw new RangeError('Unknown kernel kind');
  const quadrature=GAUSS[pixelQuadrature];if(!quadrature)throw new RangeError('pixelQuadrature must be 1, 2, 4 or 8');
  const [nodes,weights]=quadrature,data=basis.xyz,dr=kernels.radialStepMm;
  for(let j=0;j<nodes.length;j++)for(let i=0;i<nodes.length;i++) {
    const r=Math.hypot(xMm+half*nodes[i],yMm+half*nodes[j]);
    if(r>kernels.maxRadiusMm)throw new RangeError('Pixel exceeds generated nonperiodic PSF domain');
    const p=r/dr,k=Math.min(Math.floor(p),kernels.radialCount-2),f=p-k,offset=3*k,w=weights[i]*weights[j]*half*half;
    for(let c=0;c<3;c++)out[c]+=w*(data[offset+c]*(1-f)+data[offset+3+c]*f);
  }
  return out;
}
function nextPowerOfTwo(n){let p=1;while(p<n)p*=2;return p;}
/** Build one real, zero-padded linear-convolution kernel, with origin at [0,0].
 * Signed lags ±(N−1) are embedded modulo paddedSize as FFT indexing ONLY.
 * paddedSize >= 2N−1 ensures no source/output pair aliases; unused cells are zero.
 * Do not sum the padded kernel and normalize it: its domain is a difference grid,
 * not a detector image. Consumer crops the first NxN output pixels.
 */
export async function buildPaddedKernel(kernels,depthIndex,basisIndex,{kind='wave',paddedSize=nextPowerOfTwo(2*kernels.resolution-1),pixelQuadrature=4}={},hooks={}) {
  const n=kernels.resolution;if(!Number.isInteger(paddedSize)||paddedSize<2*n-1)throw new RangeError('paddedSize must be at least 2*resolution-1');
  const channels=Array.from({length:3},()=>new Float32Array(paddedSize*paddedSize)),value=new Float64Array(3),pitch=kernels.pixelWidthMm;
  function store(x,y) {const at=((y+paddedSize)%paddedSize)*paddedSize+(x+paddedSize)%paddedSize;for(let c=0;c<3;c++)channels[c][at]=value[c]*kernels.throughput;}
  for(let y=0;y<n;y++) {
    for(let x=y;x<n;x++) {
      integratePixelXYZ(kernels,depthIndex,basisIndex,x*pitch,y*pitch,{kind,pixelQuadrature},value);
      store(x,y);if(x)store(-x,y);if(y){store(x,-y);if(x)store(-x,-y);}
      if(x!==y) {store(y,x);if(y)store(-y,x);if(x){store(y,-x);if(y)store(-y,-x);}}
    }
    if(y%8===0)await checkpoint(hooks,{phase:'pixel-kernel',completed:y+1,total:n,depthIndex,basisIndex,kind});
  }
  checkAbort(hooks);
  return {size:paddedSize,paddedSize,resolution:n,channels,kind,depthIndex,basisIndex,pixelQuadrature,throughput:kernels.throughput};
}
/** Exact-position point calibration, bypassing scene/source pixel splatting.
 * Source has unit integrated radiance density in projected sensor coordinates.
 * Row zero is positive detector y. Returned values are pixel-average irradiance.
 */
export async function renderPointDetector(kernels,{depthIndex=0,basisIndex=0,centerMm=[0,0],kind='wave',pixelQuadrature=4,mass=1}={},hooks={}) {
  if(!Array.isArray(centerMm)||centerMm.length!==2||!centerMm.every(Number.isFinite)||!Number.isFinite(mass)||mass<0)throw new RangeError('Invalid point calibration');
  const n=kernels.resolution,pitch=kernels.pixelWidthMm,image=new Float32Array(n*n*3),value=new Float64Array(3),powerXYZ=new Float64Array(3);
  for(let y=0;y<n;y++) {
    const dy=(n/2-y-.5)*pitch-centerMm[1];
    for(let x=0;x<n;x++) {
      const dx=(x+.5-n/2)*pitch-centerMm[0];
      integratePixelXYZ(kernels,depthIndex,basisIndex,dx,dy,{kind,pixelQuadrature},value);
      for(let c=0;c<3;c++){const p=value[c]*kernels.throughput*mass;image[3*(y*n+x)+c]=p/kernels.pixelAreaMm2;powerXYZ[c]+=p;}
    }
    if(y%8===0)await checkpoint(hooks,{phase:'point-detector',completed:y+1,total:n,kind});
  }
  return {resolution:n,xyz:image,powerXYZ,centerMm:[...centerMm],kind};
}
