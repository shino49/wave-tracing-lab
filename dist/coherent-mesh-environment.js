/** Incoherent infinite-environment radiance and cosine angular sampling.
 * Radiance is integrated over the visible spectral measure, never a field.
 * The spectrum is independent of the coherent point/disk source spectrum. */
import {validateCoherentSpectrum} from './coherent-mesh-spectrum.js';
import {hashSpectralIndex} from './plt-spectrum.js';
import {dot,cross,normalize} from './mesh-geometry.js';
const own=(o,k)=>Object.prototype.hasOwnProperty.call(o,k);
function record(v,keys,name){if(!v||typeof v!=='object'||Array.isArray(v)||![Object.prototype,null].includes(Object.getPrototypeOf(v)))throw new TypeError(`${name} must be a plain object`);for(const k of Object.keys(v))if(!keys.includes(k))throw new Error(`Unsupported ${name} field ${k}`);}
function finite(v,min,max,name){if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max)throw new RangeError(`Invalid ${name}`);return v;}
function unit(v,name){if(!Array.isArray(v)||v.length!==3||v.some(x=>typeof x!=='number'||!Number.isFinite(x))||!Number.isFinite(Math.hypot(...v))||Math.hypot(...v)<1e-12)throw new RangeError(`Invalid ${name}`);const length=Math.hypot(...v);return Math.abs(length-1)<=8*Number.EPSILON?[...v]:v.map(x=>x/length);}
export function validateCoherentLighting(value){
 if(value==null)return null;
 record(value,['environment','indirectSamples'],'lighting');
 const indirectSamples=own(value,'indirectSamples')?value.indirectSamples:0;
 if(![0,1,2,4,8].includes(indirectSamples))throw new RangeError('indirectSamples must be 0, 1, 2, 4 or 8');
 let environment=null;
 if(value.environment!=null){
  const e=value.environment;record(e,['radiance','spectrum','wavelengthNm','up','upper','lower','samples','seed'],'environment');
  const radiance=finite(e.radiance,0,1,'environment radiance'),upper=finite(own(e,'upper')?e.upper:1,0,10,'upper hemisphere'),lower=finite(own(e,'lower')?e.lower:.2,0,10,'lower hemisphere');
  const samples=own(e,'samples')?e.samples:8,seed=own(e,'seed')?e.seed:7391;
  if(![4,8,16,32].includes(samples))throw new RangeError('Environment samples must be 4, 8, 16 or 32');
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw new RangeError('Environment seed must be uint32');
  const spectrum=validateCoherentSpectrum(e.spectrum??{mode:'broadband',illuminant:'D65'});
  const wavelengthNm=finite(own(e,'wavelengthNm')?e.wavelengthNm:550,360,830,'environment wavelengthNm');
  environment=Object.freeze({radiance,spectrum,wavelengthNm,up:Object.freeze(unit(e.up??[0,1,0],'environment up')),upper,lower,samples,seed});
 }
 return Object.freeze({environment,indirectSamples});
}
/** Direction points FROM the shading point toward the infinite light. */
export function environmentAngularRadiance(environment,direction){return environment?environment.radiance*(dot(environment.up,direction)>=0?environment.upper:environment.lower):0;}
/** Equal-area rectangular strata on the cosine disk parameter square. Every
 * stratum has independently hashed 2-D jitter. Seeds select MC realizations;
 * this is not deterministic golden-angle quadrature or a cached image field.
 * Consumers divide the sum by count; cosine/pdf = pi cancels Lambert's pi. */
export function cosineDirection(normal,index,count,seed){
 const n=unit(normal,'cosine normal');
 if(!Number.isInteger(count)||count<1||count>1024||!Number.isInteger(index)||index<0||index>=count||!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw new RangeError('Invalid cosine sampling parameters');
 // Powers of two use a near-square grid; other counts use radial strata.
 const nx=(count&(count-1))===0?2**Math.floor(Math.log2(count)/2):count,ny=count/nx;
 const u=((index%nx)+(hashSpectralIndex(seed,index,0,0x434f5331)+.5)/4294967296)/nx;
 const v=(Math.floor(index/nx)+(hashSpectralIndex(seed,index,1,0x434f5332)+.5)/4294967296)/ny;
 const axis=Math.abs(n[2])<.9?[0,0,1]:[0,1,0],a=normalize(cross(axis,n)),b=cross(n,a),r=Math.sqrt(u),phi=2*Math.PI*v,z=Math.sqrt(1-u);
 return n.map((x,i)=>z*x+r*(Math.cos(phi)*a[i]+Math.sin(phi)*b[i]));
}
