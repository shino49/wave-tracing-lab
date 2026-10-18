/** Pure paraxial Gaussian packets, independent of the scalar UTD field kernel.
 * Units: mm, radians, watts. Covariances use physical transverse ray slopes.
 * exp(+i omega t), outgoing carrier exp(-i k s). No scattering decomposition
 * is implied by these utilities. In particular, resetting a waist at a region
 * or treating arbitrary outgoing field samples as incoherent packets is invalid.
 */
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const scale=(a,s)=>a.map(x=>x*s);
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>Math.sqrt(dot(a,a));
const unit=a=>{const n=norm(a);if(!(Number.isFinite(n)&&n>0))throw new RangeError('Direction magnitude must be positive and finite');return scale(a,1/n);};
const transpose=a=>[a[0],a[2],a[1],a[3]];
const determinant=a=>a[0]*a[3]-a[1]*a[2];
const inverse=a=>{const d=determinant(a);if(!(d>0))throw new RangeError('Covariance must be positive definite');return [a[3]/d,-a[1]/d,-a[2]/d,a[0]/d];};
const mm=(a,b)=>[a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3]];
const sym=a=>[a[0],(a[1]+a[2])/2,(a[1]+a[2])/2,a[3]];
const finite=(a,n,name)=>{if(!Array.isArray(a)||a.length!==n||!a.every(Number.isFinite))throw new RangeError(`${name} requires ${n} finite values`);return [...a];};
const symmetric=(a,name)=>{finite(a,4,name);if(Math.abs(a[1]-a[2])>1e-11*Math.max(1,...a.map(Math.abs)))throw new RangeError(`${name} must be symmetric`);return sym(a);};
const positive=a=>{if(!(a.every(Number.isFinite)&&a[0]>0&&a[3]>0&&Number.isFinite(determinant(a))&&determinant(a)>0))throw new RangeError('Covariance must be positive definite');};
const modulo=a=>Math.atan2(Math.sin(a),Math.cos(a));
const clone=p=>({...p,position:[...p.position],direction:[...p.direction],u:[...p.u],v:[...p.v],positionCovariance:[...p.positionCovariance],crossCovariance:[...p.crossCovariance],angularCovariance:[...p.angularCovariance]});
const cmul=(a,b)=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];

/** X is intensity position covariance. B is inverse wavefront radius matrix.
 * C=X B, T=B X B+(4 k² X)^-1 enforce pure-state uncertainty exactly.
 * The frame is orthonormal and right handed; wavelengths and footprints cannot
 * be zero. Envelope clipping is a traversal approximation, not renormalization.
 */
export function createGaussianPacket({position=[0,0,0],direction=[0,0,1],u=[1,0,0],wavelengthMm,
  positionCovariance=[1,0,0,1],curvature=[0,0,0,0],powerW=1,phaseRad=0}={}){
  finite(position,3,'position');finite(direction,3,'direction');finite(u,3,'u');
  if(!(Number.isFinite(wavelengthMm)&&wavelengthMm>0&&Number.isFinite(powerW)&&powerW>=0&&Number.isFinite(phaseRad)))throw new RangeError('Invalid wavelength, power or phase');
  const d=unit(direction),basis=add(u,scale(d,-dot(u,d)));
  if(norm(basis)<1e-10*norm(u))throw new RangeError('Frame axis is parallel to propagation');
  const e1=unit(basis),e2=cross(d,e1),X=symmetric(positionCovariance,'positionCovariance'),B=symmetric(curvature,'curvature');positive(X);
  const k=2*Math.PI/wavelengthMm;
  if(!Number.isFinite(k)||k===0)throw new RangeError('Wavenumber is outside finite range');
  const C=mm(X,B),T=add(mm(mm(B,X),B),scale(inverse(X),1/(4*k*k)));positive(T);
  return {kind:'pure-paraxial-Gaussian',position:[...position],direction:d,u:e1,v:e2,wavelengthMm,k,
    positionCovariance:X,crossCovariance:C,angularCovariance:sym(T),powerW,phaseRad:modulo(phaseRad),
    opticalPathMm:0,gouyPhaseRad:0,normalization:'integral of intensity over perpendicular plane equals powerW'};
}

export function packetCurvature(packet){return sym(mm(transpose(packet.crossCovariance),inverse(packet.positionCovariance)));}
export function packetPurityResidual(packet){
  const {positionCovariance:X,crossCovariance:C,angularCovariance:T,k}=packet;
  const expected=add(mm(mm(transpose(C),inverse(X)),C),scale(inverse(X),1/(4*k*k)));
  return Math.max(...T.map((x,i)=>Math.abs(x-expected[i])))/Math.max(...T.map(Math.abs),1e-300);
}

/** Free-space phase-space shear. The waist age is never reset. */
export function propagatePacket(packet,distanceMm){
  if(!(Number.isFinite(distanceMm)&&distanceMm>=0))throw new RangeError('Propagation distance must be finite and nonnegative');
  if(distanceMm===0)return clone(packet);
  const s=distanceMm,{positionCovariance:X,crossCovariance:C,angularCovariance:T,k}=packet;
  if(!Number.isFinite(k*s)||!Number.isFinite(packet.opticalPathMm+s))throw new RangeError('Propagation phase or optical path overflows');
  const Xn=sym(add(add(X,scale(add(C,transpose(C)),s)),scale(T,s*s))),Cn=add(C,scale(T,s));positive(Xn);
  // D=I+s B-i s X^-1/(2k). Both eigenvalues lie in the lower half plane;
  // choose their continuous summed argument (-2pi,0), including a focus.
  const B=packetCurvature(packet),Xi=inverse(X),D=B.map((x,i)=>[(i===0||i===3?1:0)+s*x,-s*Xi[i]/(2*k)]);
  const ad=cmul(D[0],D[3]),bc=cmul(D[1],D[2]),det=[ad[0]-bc[0],ad[1]-bc[1]];
  let arg=Math.atan2(det[1],det[0]);if(arg>0)arg-=2*Math.PI;
  const gouy=-arg/2;
  return {...clone(packet),position:add(packet.position,scale(packet.direction,s)),positionCovariance:Xn,crossCovariance:Cn,
    opticalPathMm:packet.opticalPathMm+s,gouyPhaseRad:packet.gouyPhaseRad+gouy,
    phaseRad:modulo(packet.phaseRad-k*s+gouy)};
}

/** Rigid orthogonal transform, including ideal specular reflection. Covariances
 * remain components in the transformed transverse frame; no footprint scaling.
 * An improper transform reverses v to retain a right-handed frame and applies
 * the matching transverse coordinate reflection to all covariance blocks.
 */
export function transformPacket(packet,rotation,translation=[0,0,0],scalarPhase=0){
  finite(rotation,9,'rotation');finite(translation,3,'translation');
  if(!Number.isFinite(scalarPhase))throw new RangeError('Scalar phase must be finite');
  const rows=[rotation.slice(0,3),rotation.slice(3,6),rotation.slice(6,9)];
  if(rows.some((a,i)=>rows.some((b,j)=>Math.abs(dot(a,b)-(i===j?1:0))>1e-10)))throw new RangeError('Transform must be orthogonal');
  const apply=a=>rows.map(r=>dot(r,a)),d=apply(packet.direction),u=apply(packet.u),v=cross(d,u),oldV=apply(packet.v),sign=dot(v,oldV)>0?1:-1;
  const parity=a=>[a[0],a[1]*sign,a[2]*sign,a[3]];
  return {...clone(packet),position:add(apply(packet.position),translation),direction:d,u,v,
    positionCovariance:parity(packet.positionCovariance),crossCovariance:parity(packet.crossCovariance),angularCovariance:parity(packet.angularCovariance),phaseRad:modulo(packet.phaseRad+scalarPhase)};
}

/** Normalized irradiance in the packet's transverse plane, W/mm². */
export function packetIntensity(packet,transverse){
  finite(transverse,2,'transverse');const X=packet.positionCovariance,Xi=inverse(X),[x,y]=transverse;
  const q=Xi[0]*x*x+(Xi[1]+Xi[2])*x*y+Xi[3]*y*y;
  return packet.powerW/(2*Math.PI*Math.sqrt(determinant(X)))*Math.exp(-q/2);
}

/** Normalized complex field sqrt(W)/mm in the same plane. Carrier phase is
 * carried independently of the local quadratic wavefront curvature. */
export function packetField(packet,transverse){
  if(packet.coherentPhaseDefined===false||!Number.isFinite(packet.phaseRad))throw new RangeError('Coherent field phase is undefined for this packet proposal');
  const intensity=packetIntensity(packet,transverse),B=packetCurvature(packet),[x,y]=transverse;
  const phase=packet.phaseRad-packet.k*(B[0]*x*x+(B[1]+B[2])*x*y+B[3]*y*y)/2;
  return [Math.sqrt(intensity)*Math.cos(phase),Math.sqrt(intensity)*Math.sin(phase)];
}

/** Finite area detector normal to packet direction. Midpoint quadrature uses
 * actual mm² weights; changing pixel count does not rescale incident power.
 * This is an area sensor measurement, not an outgoing scattering-mode rule.
 */
export function measurePacketRectangle(packet,{center=[0,0],widthMm,heightMm,nx=64,ny=64}={}){
  finite(center,2,'detector center');if(!(widthMm>0&&heightMm>0&&Number.isFinite(widthMm)&&Number.isFinite(heightMm)&&Number.isInteger(nx)&&Number.isInteger(ny)&&nx>0&&ny>0&&nx<=2048&&ny<=2048))throw new RangeError('Invalid detector quadrature');
  const area=widthMm*heightMm/(nx*ny);let powerW=0;
  for(let j=0;j<ny;j++)for(let i=0;i<nx;i++)powerW+=area*packetIntensity(packet,[center[0]+((i+.5)/nx-.5)*widthMm,center[1]+((j+.5)/ny-.5)*heightMm]);
  return {powerW,quadrature:'deterministic-midpoint-area',samples:nx*ny,areaWeightMm2:area,unit:'W'};
}

/** Approximation diagnostics, not an automatic accuracy certificate. rmsSlope
 * and expected omitted quartic phase expose when a caller's declared paraxial
 * error budget is exceeded. The latter comes from the small-angle expansion
 * of sqrt(k²-|q|²), using the fourth moment of a 2D Gaussian direction law.
 */
export function packetParaxialDiagnostics(packet,distanceMm=0){
  if(!(Number.isFinite(distanceMm)&&distanceMm>=0))throw new RangeError('Diagnostic distance must be finite and nonnegative');
  const T=packet.angularCovariance,trace=T[0]+T[3],traceSquare=T[0]**2+2*T[1]*T[2]+T[3]**2;
  return {rmsSlope:Math.sqrt(trace),maxPrincipalRmsSlope:Math.sqrt((trace+Math.hypot(T[0]-T[3],2*T[1]))/2),
    expectedOmittedQuarticPhaseRad:packet.k*distanceMm*(trace**2+2*traceSquare)/8,
    certificate:false,approximation:'scalar, narrow angular spectrum, paraxial propagation'};
}
