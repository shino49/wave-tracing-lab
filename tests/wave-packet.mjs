import assert from 'node:assert/strict';
import {createGaussianPacket,propagatePacket,transformPacket,packetCurvature,packetPurityResidual,packetIntensity,packetField,measurePacketRectangle,packetParaxialDiagnostics} from '../dist/wave-packet.js';
const close=(a,b,t=2e-10)=>assert.ok(Math.abs(a-b)<=t*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const arrays=(a,b,t)=>a.forEach((x,i)=>close(x,b[i],t));
const phase=(a,b)=>close(Math.atan2(Math.sin(a-b),Math.cos(a-b)),0,1e-9);
const p=createGaussianPacket({wavelengthMm:.05,position:[1,2,3],positionCovariance:[.16,.03,.03,.49],curvature:[.02,.013,.013,-.08],powerW:3});
assert.ok(packetPurityResidual(p)<2e-13);arrays(packetCurvature(p),[.02,.013,.013,-.08]);
let count=0;
for(const X of [[.16,0,0,.49],[.16,.03,.03,.49],[.001,0,0,.0004]])for(const B of [[0,0,0,0],[.1,0,0,.02],[-.1,0,0,-.07],[-.1,.04,.04,-.02]]){
  const a=createGaussianPacket({wavelengthMm:.05,positionCovariance:X,curvature:B,powerW:3});
  for(const [s,t] of [[0,0],[1,2],[7,13],[50,80]]){
    const one=propagatePacket(a,s+t),two=propagatePacket(propagatePacket(a,s),t);
    for(const key of ['position','direction','u','v','positionCovariance','crossCovariance','angularCovariance'])arrays(one[key],two[key]);
    phase(one.phaseRad,two.phaseRad);close(one.gouyPhaseRad,two.gouyPhaseRad);close(one.opticalPathMm,s+t);close(one.powerW,3);
    assert.ok(packetPurityResidual(one)<2e-8,`purity residual ${packetPurityResidual(one)}`);
    const f=packetField(one,[.1,.2]);close(f[0]**2+f[1]**2,packetIntensity(one,[.1,.2]));count++;
  }
}
const R=[0,-1,0,0,0,-1,1,0,0],translation=[5,8,-2];
for(const s of [0,10,100]){
  const one=transformPacket(propagatePacket(p,s),R,translation),two=propagatePacket(transformPacket(p,R,translation),s);
  for(const key of ['position','direction','u','v','positionCovariance','crossCovariance','angularCovariance'])arrays(one[key],two[key]);
  phase(one.phaseRad,two.phaseRad);
}
const reflect=[1,0,0,0,1,0,0,0,-1];
const mirrored=transformPacket(p,reflect,[0,0,0],Math.PI),roundtrip=transformPacket(mirrored,reflect,[0,0,0],Math.PI);
for(const key of ['position','direction','u','v','positionCovariance','crossCovariance','angularCovariance'])arrays(roundtrip[key],p[key]);
phase(roundtrip.phaseRad,p.phaseRad);close(packetIntensity(p,[.2,.3]),packetIntensity(mirrored,[.2,-.3]));
const source=createGaussianPacket({wavelengthMm:.05,positionCovariance:[1,0,0,.36],powerW:2});
const area={widthMm:12,heightMm:7.2,nx:128,ny:128};close(measurePacketRectangle(source,area).powerW,2,5e-9);
const half=measurePacketRectangle(source,{center:[-3,0],widthMm:6,heightMm:7.2,nx:64,ny:128}).powerW+measurePacketRectangle(source,{center:[3,0],widthMm:6,heightMm:7.2,nx:64,ny:128}).powerW;
close(half,measurePacketRectangle(source,area).powerW);
for(const powerW of [0,1,3.5]){
  const scaled=createGaussianPacket({wavelengthMm:.05,positionCovariance:[1,0,0,.36],powerW});
  close(measurePacketRectangle(scaled,area).powerW,measurePacketRectangle(source,area).powerW*powerW/2);
}
const finiteArea={widthMm:2.6,heightMm:1.4};
const m64=measurePacketRectangle(source,{...finiteArea,nx:64,ny:64}).powerW,m128=measurePacketRectangle(source,{...finiteArea,nx:128,ny:128}).powerW,m256=measurePacketRectangle(source,{...finiteArea,nx:256,ny:256}).powerW;
assert.ok(Math.abs(m256-m128)<Math.abs(m128-m64)*.26,'Midpoint area refinement must converge quadratically');
assert.throws(()=>createGaussianPacket({wavelengthMm:0}));assert.throws(()=>createGaussianPacket({wavelengthMm:.05,positionCovariance:[1,2,2,1]}));
assert.throws(()=>createGaussianPacket({wavelengthMm:.05,curvature:[0,1,0,0]}));assert.throws(()=>propagatePacket(p,-1));
assert.throws(()=>transformPacket(p,[2,0,0,0,1,0,0,0,1]));
assert.throws(()=>createGaussianPacket({wavelengthMm:Number.MIN_VALUE}));
assert.throws(()=>createGaussianPacket({wavelengthMm:.05,curvature:[1e300,0,0,1e300]}));
assert.throws(()=>transformPacket(p,[1,0,0,0,1,0,0,0,1],[0,0,0],NaN));
const pd=packetParaxialDiagnostics(p,10);assert.ok(pd.rmsSlope>0&&pd.expectedOmittedQuarticPhaseRad>0);assert.equal(pd.certificate,false);
console.log(JSON.stringify({passed:true,propagationCases:count,rigidTransformCases:3,idealMirrorRoundTrip:true,finiteAreaPowerW:measurePacketRectangle(source,area).powerW,emittedPowerW:2,partitionInvariant:true,powerLinearity:true,quadratureConvergence:{m64,m128,m256},scope:'Pure paraxial Gaussian propagation and area measurement only; no scattering-mode decomposition or end-to-end PLT claim'}));
