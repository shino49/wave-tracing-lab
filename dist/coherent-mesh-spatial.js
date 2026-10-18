/** Uniform-area disk of mutually independent isotropic scalar monopoles.
 * This is a source intensity measure, not a Lambertian surface or arbitrary
 * cross-spectral-density model. Never sum fields between returned sources.
 */
export function validateCoherentSpatialSource(value){
 if(value==null)return null;
 if(typeof value!=='object'||Array.isArray(value)||Object.keys(value).some(k=>!['type','radiusMm','normal','samples','seed'].includes(k)))throw new TypeError('Invalid spatial source fields');
 const {type,radiusMm,normal,samples,seed=7391}=value;
 if(type!=='disk'||!Number.isFinite(radiusMm)||radiusMm<0||radiusMm>1000)throw new RangeError('Disk radius must be within 0…1000 mm');
 if(!Array.isArray(normal)||normal.length!==3||normal.some(x=>!Number.isFinite(x)||Math.abs(x)>10000)||!(Math.hypot(...normal)>=1e-12))throw new RangeError('Disk normal must be a nonzero finite 3-vector');
 if(!Number.isInteger(samples)||samples<1||samples>256||!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw new RangeError('Disk samples must be 1…256 and seed uint32');
 const length=Math.hypot(...normal),n=Math.abs(length-1)<=8*Number.EPSILON?[...normal]:normal.map(x=>x/length);
 return Object.freeze({type,radiusMm,normal:Object.freeze(n),samples,seed});
}
export function prepareCoherentSpatialSources(source){
 const spatial=validateCoherentSpatialSource(source.spatial);
 const {spatial:ignored,...point}=source;
 if(!spatial||spatial.radiusMm===0)return [{source:point,weight:1,index:0}];
 const n=spatial.normal,axis=Math.abs(n[2])<.9?[0,0,1]:[0,1,0];
 const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 let u=cross(axis,n),len=Math.hypot(...u);u=u.map(x=>x/len);const v=cross(n,u);
 // Equal-area radial midpoint spiral: deterministic quadrature, not a nested
 // random prefix. Seed rotates its azimuth; no statistical error bar claimed.
 const rotation=2*Math.PI*(spatial.seed/4294967296),golden=Math.PI*(3-Math.sqrt(5));
 return Array.from({length:spatial.samples},(_,index)=>{const radius=spatial.radiusMm*Math.sqrt((index+.5)/spatial.samples),angle=rotation+index*golden;
  const position=source.position.map((x,k)=>x+radius*(Math.cos(angle)*u[k]+Math.sin(angle)*v[k]));
  if(position.some(x=>Math.abs(x)>10000))throw new RangeError('Disk sample outside coordinate limits');
  return {source:{...point,position},weight:1/spatial.samples,index};
 });
}
