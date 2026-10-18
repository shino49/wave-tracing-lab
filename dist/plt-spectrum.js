/** Wavelength-sampled color for the relative scalar PLT estimator.
 * Colorimetry is CIE XYZ; this does NOT calibrate the native angular estimator.
 * nm is the integration measure; geometric lengths remain millimetres.
 */
import {CIE_START_NM,CIE_XYZ_1NM,D65_START_NM,CIE_D65_1NM} from './plt-cie.js';
export const VISIBLE_BAND=Object.freeze({minNm:360,maxNm:830,widthNm:470});
export const SPECTRAL_MATERIALS=Object.freeze(['neutral','red','green','blue']);
export const SPECTRAL_SOURCES=Object.freeze(['D65','E','blackbody3000']);
export const SPECTRAL_SCOPE='360–830 nm sampled scalar transport; CIE 1931 2° XYZ; fixed relative normalization, not calibrated radiometry';
export const CIE_Y_INTEGRAL=CIE_XYZ_1NM.reduce((sum,row,i)=>sum+row[1]*(i===0||i===CIE_XYZ_1NM.length-1?.5:1),0);
export function wavelengthNmToMm(nm){if(!Number.isFinite(nm)||nm<VISIBLE_BAND.minNm||nm>VISIBLE_BAND.maxNm)throw new RangeError('Visible wavelength must be within 360…830 nm');return nm*1e-6;}
export function cieXYZ(nm){
 if(!Number.isFinite(nm))throw new RangeError('Nonfinite wavelength');
 if(nm<CIE_START_NM||nm>CIE_START_NM+CIE_XYZ_1NM.length-1)return [0,0,0];
 const p=nm-CIE_START_NM,i=Math.min(Math.floor(p),CIE_XYZ_1NM.length-2),f=p-i;
 return CIE_XYZ_1NM[i].map((v,c)=>v*(1-f)+CIE_XYZ_1NM[i+1][c]*f);
}
export function sourceSpectrum(nm,source='D65'){
 wavelengthNmToMm(nm);if(source==='E')return 1;if(source==='blackbody3000'){const c2=14387768.775039338,T=3000;return (560/nm)**5*Math.expm1(c2/(560*T))/Math.expm1(c2/(nm*T));}if(source!=='D65')throw new RangeError('Unknown spectral source');
 const p=nm-D65_START_NM,i=Math.min(Math.floor(p),CIE_D65_1NM.length-2),f=p-i;
 // The published D65 table is 100 at 560 nm. No per-image or source-dependent
 // luminance normalization: source strength scales this spectral density.
 return (CIE_D65_1NM[i]*(1-f)+CIE_D65_1NM[i+1]*f)/100;
}
export function reflectanceSpectrum(nm,name='neutral'){
 wavelengthNmToMm(nm);if(name==='neutral')return 1;
 // Deliberately synthetic, analytic passive reflectors, NOT measured materials
 // or RGB upsampling: each is floor .04 plus a Gaussian lobe of amplitude .92.
 const parameters={red:[620,42],green:[535,32],blue:[450,25]}[name];
 if(!parameters)throw new RangeError('Unknown spectral reflectance');
 const [center,sigma]=parameters;return .04+.92*Math.exp(-.5*((nm-center)/sigma)**2);
}
export function hashSpectralIndex(seed,pixel,sample=0,tag=0){
 let x=(seed^Math.imul(pixel+1,0x9e3779b1)^Math.imul(sample+1,0x85ebca6b)^tag)>>>0;
 x=Math.imul(x^(x>>>16),0x7feb352d);x=Math.imul(x^(x>>>15),0x846ca68b);return (x^(x>>>16))>>>0;
}
function radicalInverse2(n){let x=n>>>0;x=((x>>>16)|(x<<16))>>>0;x=(((x&0x00ff00ff)<<8)|((x&0xff00ff00)>>>8))>>>0;x=(((x&0x0f0f0f0f)<<4)|((x&0xf0f0f0f0)>>>4))>>>0;x=(((x&0x33333333)<<2)|((x&0xcccccccc)>>>2))>>>0;x=(((x&0x55555555)<<1)|((x&0xaaaaaaaa)>>>1))>>>0;return x/4294967296;}
export function sampleVisibleWavelength(seed,pixel,sample){
 for(const [v,n] of [[seed,'seed'],[pixel,'pixel'],[sample,'sample']])if(!Number.isInteger(v)||v<0||v>0xffffffff)throw new RangeError(`Invalid spectral ${n} index`);
 // Cranley–Patterson randomized base-2 stratification. Every 2^m prefix has
 // one point per stratum (after rotation); each marginal PDF is uniform nm^-1.
 // Independent of tile order, requested spp and path-variable random draws.
 const shift=(hashSpectralIndex(seed,pixel,0,0x57415645)+.5)/4294967296;
 const u=(radicalInverse2(sample)+shift)%1,nm=VISIBLE_BAND.minNm+VISIBLE_BAND.widthNm*u,pdfPerNm=1/VISIBLE_BAND.widthNm;
 return {nm,wavelengthMm:wavelengthNmToMm(nm),pdfPerNm,xyzWeight:cieXYZ(nm).map(v=>v/(pdfPerNm*CIE_Y_INTEGRAL))};
}
export function validateSpectralOptions(o){
 if(!['monochrome','visible'].includes(o.spectralMode??'monochrome'))throw new RangeError('Invalid spectral mode');
 if(o.spectralMode==='visible'){
  if(!Number.isInteger(o.seed===undefined?7391:o.seed)||(o.seed===undefined?7391:o.seed)<0||(o.seed===undefined?7391:o.seed)>0xffffffff)throw new RangeError('Invalid spectral seed');
  if(!SPECTRAL_SOURCES.includes(o.spectralSource??'D65'))throw new RangeError('Invalid spectral source');
  if(!SPECTRAL_MATERIALS.includes(o.spectralMaterial??'neutral'))throw new RangeError('Invalid spectral material');
 }
 return o;
}
export function xyzToLinearSRGB(xyz){
 if(!xyz||xyz.length!==3||!Array.from(xyz).every(v=>Number.isFinite(v)&&v>=0))throw new RangeError('XYZ must be finite and nonnegative');
 const [x,y,z]=xyz;
 // D65-relative XYZ to linear sRGB, no chromatic adaptation (E is warmer).
 return [3.2409699419045226*x-1.537383177570094*y-.4986107602930034*z,-.9692436362808796*x+1.8759675015077202*y+.04155505740717559*z,.05563007969699366*x-.20397695888897652*y+1.0569715142428786*z];
}
export function linearToSRGB(v){if(!Number.isFinite(v)||v<0)throw new RangeError('Display value must be finite and nonnegative');return v<=.0031308?12.92*v:1.055*v**(1/2.4)-.055;}
export function displaySpectralXYZ(xyz,exposure=0){
 if(!Number.isFinite(exposure)||exposure<-20||exposure>20)throw new RangeError('Invalid display exposure');
 const rgb=xyzToLinearSRGB(xyz),gain=2**exposure;
 // Display-only out-of-gamut clipping and exponential compression; untouched
 // raw XYZ remains the estimator. No hidden path/sample clamp or denoiser.
 return {rgb:rgb.map(v=>Math.round(255*linearToSRGB(-Math.expm1(-gain*Math.max(0,v))))),negativeChannels:rgb.filter(v=>v<0).length,aboveUnitChannels:rgb.filter(v=>v>1).length};
}
export function spectralDisplaySummary(xyz){
 if(!xyz||xyz.length%3)throw new RangeError('Malformed XYZ image');let negativeChannels=0,aboveUnitChannels=0;
 for(let i=0;i<xyz.length;i+=3){const rgb=xyzToLinearSRGB(xyz.subarray?xyz.subarray(i,i+3):xyz.slice(i,i+3));negativeChannels+=rgb.filter(v=>v<0).length;aboveUnitChannels+=rgb.filter(v=>v>1).length;}
 return {negativeLinearSRGBChannels:negativeChannels,linearSRGBChannelsAboveOne:aboveUnitChannels,totalChannels:xyz.length,display:'negative linear sRGB clipped only for display; fixed exponential compression then standard sRGB transfer; raw XYZ unchanged'};
}
