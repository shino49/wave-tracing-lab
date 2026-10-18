import assert from 'node:assert/strict';
import {traceDoubleDiffraction,DD_LIMITS} from '../dist/utd-dd.js';
import {evaluateDDRecord} from '../dist/utd-dd-gpu.js';
import {packetField,packetIntensity,packetPurityResidual,propagatePacket,measurePacketRectangle} from '../dist/wave-packet.js';
import {inspectFixture} from '../research/dd-frustum-check.mjs';
import {transformPoint,add,sub,scale,dot,length,normalize,cross} from '../dist/mesh-geometry.js';
const close=(a,b,rel=2e-11,abs=1e-13)=>assert.ok(Math.abs(a-b)<=abs+rel*Math.max(Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const arrayClose=(a,b,rel,abs)=>a.forEach((x,i)=>close(x,b[i],rel,abs));
const valid=r=>assert.equal(r.valid,true,JSON.stringify(r.admissionReasons));
const invalid=(c,code)=>{const r=traceDoubleDiffraction(c);assert.equal(r.valid,false);assert.equal(r.coefficientRecord.valid,false);assert.equal(r.conditionalPower,null);assert.equal(r.conditionalWeight,null);if(code)assert.ok(r.admissionReasons.some(x=>x.code===code),JSON.stringify(r.admissionReasons));assert.equal(evaluateDDRecord(r.coefficientRecord).valid,false);return r;};
const canonical=traceDoubleDiffraction();valid(canonical);
assert.equal(canonical.admissionPolicy,'native-centre-conditional');
assert.equal(canonical.physicallyCalibrated,false);assert.equal(canonical.fullPacketScatteringValidated,false);
assert.equal(canonical.terminalConnection,'centre-connected-next-event');assert.equal(canonical.outgoingBundleClearanceProven,false);
assert.equal(canonical.originalCollimationGuard.applicable,false);assert.equal(canonical.originalCollimationGuard.passed,false);
assert.ok(canonical.originalCollimationGuard.stages[0].phaseCurvatureRad>.25);
assert.equal(canonical.originalCollimationGuard.stages[0].knownPhysicalPhase,true);
assert.equal(canonical.originalCollimationGuard.stages[1].knownPhysicalPhase,false);
assert.equal(canonical.originalCollimationGuard.stages[1].phaseCurvatureRad,null);
assert.equal(canonical.originalCollimationGuard.stages[1].passed,null);
assert.ok(canonical.originalCollimationGuard.stages[1].proposalQuadraticPhaseScaleRad>.25);
assert.equal(canonical.packetStates.length,6);assert.equal(canonical.regions.length,2);
assert.ok(Object.values(canonical.checks).every(Boolean));
assert.equal(canonical.mesh.triangleCount,4);
assert.deepEqual(Object.keys(canonical.coefficientRecord).sort(),['angles','ro1','ro2','ri2','k','boundary','valid'].sort());

let fixtureCases=0;
for(const k of [128,256,512])for(const boundary of ['soft','hard']){
  const r=traceDoubleDiffraction({k,boundary});valid(r);fixtureCases++;
  const q=Math.sqrt(256/k),reference=inspectFixture({k,sigma:.07*q,aim1:.1*q,aim2:.175*q});
  assert.equal(reference.proof.passed,true);
  arrayClose(r.coefficientRecord.angles,reference.nativeCentreEvaluation.angles);
  for(const key of ['ro1','ro2','ri2'])close(r.coefficientRecord[key],reference.nativeCentreEvaluation[key]);
  close(r.conditionalPower,reference.nativeCentreEvaluation.powers[boundary],1e-12,1e-24);
  close(r.conditionalPower,evaluateDDRecord(r.coefficientRecord).power,1e-12,1e-24);
  for(const [i,region] of r.regions.entries()){
    close(region.near,[reference.first,reference.second][i].near);
    const trace=r.geometryTrace[i];assert.equal(trace.steps[0].ballisticLengthWavelengths,16);
    close(trace.steps[0].to,16*2*Math.PI/k);
    assert.equal(trace.steps[0].overlapFactor,1.001);
    close(trace.steps[0].rayInterval[1],trace.steps[0].to*1.001);
    assert.equal(trace.steps[0].rayTriangleTests,r.mesh.triangleCount);
    assert.deepEqual(trace.steps[0].excludedTriangleIds,[]);
    assert.equal(trace.steps.at(-1).decision,'region');
    assert.ok(trace.steps.at(-1).diffusiveProgress>=trace.steps.at(-1).minimumProgress);
    assert.equal(r.localStages[i].finiteSourceReducedDistanceUsed,false);
    assert.equal(r.localStages[i].localCoherentContributionCount,1);
    assert.equal(r.localStages[i].directContributionCount,0);
    assert.ok(r.localStages[i].endpointGuardRatio>=1);
    close(r.localStages[i].stationaryWidth,Math.sqrt([r.coefficientRecord.ro1,r.coefficientRecord.ro2][i]/k));
    close(r.localStages[i].requiredHalfLength,2*r.localStages[i].stationaryWidth);
    assert.ok(r.localStages[i].minTransitionArgument>=DD_LIMITS.minTransitionArgument);
    assert.ok(r.localStages[i].transitionTerms.every(t=>Number.isFinite(t.argument)&&Number.isFinite(t.termMagnitude)));
    assert.ok(region.stationaryPoint.ellipsoidValue<1&&region.stationaryPoint.frustum.inside);
    assert.ok(Object.values(region.proxyBoundaryMargins).every(x=>x>0));
  }
}
// A fixed regression value is not an accuracy certificate or a full-wave fit.
close(canonical.conditionalPower,5.71382456757761e-8,1e-12,1e-24);
const hard=traceDoubleDiffraction({boundary:'hard'});close(hard.conditionalPower,3.730265878937685e-7,1e-12,1e-24);

let shapeIntegralError=0,maximumPurityResidual=0;
for(const p of canonical.packetStates){
  assert.equal(p.powerW,1);assert.ok(p.positionCovariance[0]>0);assert.ok(p.angularCovariance[0]>0);
  close(p.angularCovariance[0],canonical.packetStates[0].angularCovariance[0]);
  assert.ok(packetPurityResidual(p)<1e-12);maximumPurityResidual=Math.max(maximumPurityResidual,packetPurityResidual(p));
  close(p.physicalDistanceMm,p.opticalPathMm);
  const sigma=Math.sqrt(p.positionCovariance[0]);
  const integral=measurePacketRectangle(p,{widthMm:12*sigma,heightMm:12*sigma,nx:128,ny:128}).powerW;
  shapeIntegralError=Math.max(shapeIntegralError,Math.abs(integral-1));close(integral,1,0,5e-9);
  if(p.coherentPhaseDefined){const field=packetField(p,[.03,.05]);close(field[0]**2+field[1]**2,packetIntensity(p,[.03,.05]));}
  else{assert.equal(p.phaseRad,null);assert.equal(p.gouyPhaseRad,null);assert.throws(()=>packetField(p,[0,0]),/undefined/);}
}
const [source,in1,out1,in2,out2,receiver]=canonical.packetStates;
for(const [a,b,s] of [[source,in1,canonical.regions[0].near],[out1,in2,canonical.regions[1].near],[out2,receiver,length(sub(receiver.position,out2.position))]]){
  const propagated=propagatePacket({...a,phaseRad:0,gouyPhaseRad:0},s);
  for(const key of ['positionCovariance','crossCovariance','angularCovariance','position'])arrayClose(propagated[key],b[key]);
  close(a.opticalPathMm+s,b.opticalPathMm);
  close(b.proposalVirtualAgeMm-a.proposalVirtualAgeMm,s);
}
for(const [a,b,cone] of [[in1,out1,canonical.cones[1]],[in2,out2,canonical.cones[2]]]){
  close(a.opticalPathMm,b.opticalPathMm);
  assert.ok(b.proposalVirtualAgeMm>a.proposalVirtualAgeMm);
  const X=(cone.radius0/3)**2,T=(cone.tanAngle/3)**2,C=Math.sqrt(X*T-1/(4*b.k*b.k));
  close(b.positionCovariance[0],X);close(b.angularCovariance[0],T);close(b.crossCovariance[0],C);
  close(b.curvature[0],C/X);close(b.powerW,1);
  assert.equal(cone.radius0,3*Math.sqrt(b.positionCovariance[0]));
  assert.equal(cone.tanAngle,3*Math.sqrt(b.angularCovariance[0]));
}
assert.equal(canonical.geometryTrace[1].region.cone.radius0,canonical.cones[1].radius0);
assert.equal(canonical.geometryTrace[1].region.cone.tanAngle,canonical.cones[1].tanAngle);
close(receiver.opticalPathMm,canonical.physicalPath.centreAxisDistanceMm);
assert.ok(Math.abs(receiver.opticalPathMm-receiver.proposalVirtualAgeMm)>1);
assert.ok(Math.abs(canonical.physicalPath.edgeBrokenPathDistanceMm-receiver.opticalPathMm)>1e-3);

// The packet covariance drives the same conservative cones used for traversal.
let envelopeTests=0;
for(const [p,cone] of [[source,canonical.cones[0]],[out1,canonical.cones[1]],[out2,canonical.cones[2]]])for(const s of [0,.1,1,5,20,100]){
  const X=p.positionCovariance[0],C=p.crossCovariance[0],T=p.angularCovariance[0];
  assert.ok(3*Math.sqrt(X+2*C*s+T*s*s)<=cone.radius0+cone.tanAngle*s+1e-12);envelopeTests++;
}
// A turn fit must contain the entire previous frustum, not just its centre.
let capContainmentTests=0;
for(const [i,old] of canonical.regions.entries()){
  const cone=canonical.cones[i+1],u=normalize(cross(old.direction,[0,0,1])),v=normalize(cross(old.direction,u));
  for(const t of [old.near,old.far])for(let n=0;n<128;n++){
    const radius=old.cone.radius0+old.cone.tanAngle*t,angle=2*Math.PI*n/128;
    const p=add(add(old.cone.origin,scale(old.direction,t)),add(scale(u,radius*Math.cos(angle)),scale(v,radius*Math.sin(angle))));
    const offset=sub(p,cone.origin),axial=dot(offset,cone.direction),radial=length(sub(offset,scale(cone.direction,axial)));
    assert.ok(radial<=cone.radius0+cone.tanAngle*axial+1e-11);capContainmentTests++;
  }
}

// Proposal energy remains one; only the separately named conditional weight scales.
for(const inputWeight of [0,.25,2,1e12]){
  const r=traceDoubleDiffraction({inputWeight});valid(r);
  close(r.conditionalPower,canonical.conditionalPower,1e-12,1e-24);
  close(r.conditionalWeight,inputWeight*canonical.conditionalPower,1e-12,1e-24);
  r.packetStates.forEach((p,i)=>{assert.equal(p.powerW,1);arrayClose(p.positionCovariance,canonical.packetStates[i].positionCovariance);close(p.conditionalWeight,inputWeight*canonical.packetStates[i].conditionalWeight,1e-12,1e-24);});
}

// Full rigid motion, with both a rotated edge and rotated cross-section.
let rigidTransformCases=0;
for(const rotationDeg of [[31,27,19],[0,0,90],[0,90,0],[45,20,70],[-70,10,150]]){
  const c={rotationDeg,translation:[3,-5,8]},r=traceDoubleDiffraction(c);valid(r);rigidTransformCases++;
  close(r.conditionalPower,canonical.conditionalPower,2e-10,1e-23);
  arrayClose(r.coefficientRecord.angles,canonical.coefficientRecord.angles);
  r.packetStates.forEach((p,i)=>{arrayClose(p.position,transformPoint(canonical.packetStates[i].position,c));arrayClose(p.positionCovariance,canonical.packetStates[i].positionCovariance);close(p.opticalPathMm,canonical.packetStates[i].opticalPathMm);});
}
for(const proxySubdivisions of [2,8,32]){
  const r=traceDoubleDiffraction({proxySubdivisions});valid(r);assert.equal(r.mesh.triangleCount,4*proxySubdivisions);
  close(r.conditionalPower,canonical.conditionalPower,2e-10,1e-23);
  for(const stage of r.geometryTrace)assert.equal(stage.steps[0].rayTriangleTests,4*proxySubdivisions);
}
const broad=traceDoubleDiffraction({proxyHalfLength:40,proxyBottom:-40});valid(broad);close(broad.conditionalPower,canonical.conditionalPower,2e-10,1e-23);
let localGridCases=0,minimumGridEndpointRatio=Infinity;
for(const sigma of [.0675,.07,.0725])for(const aim1 of [.09,.1,.11])for(const aim2 of [.15,.175,.2])for(const y of [-2.25,-2,-1.75]){
  const r=traceDoubleDiffraction({sigma,aim1,aim2,receiver:[16,y,0]});valid(r);localGridCases++;
  minimumGridEndpointRatio=Math.min(minimumGridEndpointRatio,...r.localStages.map(s=>s.endpointGuardRatio));
}

// Negative controls exercise real admission, rather than standalone primitives.
let negativeControls=0;
const low=invalid({k:64},'initial-traversal');negativeControls++;
assert.equal(low.geometryTrace[0].steps[0].ballisticLengthWavelengths,16);
assert.equal(low.geometryTrace[0].steps[0].decision,'short-progress-continue-ballistic');
assert.equal(low.geometryTrace[0].steps[1].ballisticLengthWavelengths,64);
const pointed=invalid({aim1:0,aim2:0},'second-traversal');negativeControls++;
assert.equal(pointed.geometryTrace[1].reason,'opaque-ray-hit');
assert.ok(pointed.geometryTrace[1].steps.at(-1).hit.triangleId<2,'Previously visited screen may not be excluded from ballistic rays');
for(const [c,code] of [[{proxyHalfLength:.3},'region1-proxy-boundaries'],[{proxyBottom:.9999},'region1-proxy-boundaries'],
  [{receiver:[16,2.8,0]},'region2-away-from-transition'],[{receiver:[8.00001,-2,0]},'region2-away-from-grazing'],
  [{receiver:[16,4,0]},'local2-shadow'],[{proxySubdivisions:33},'triangle-safety-cap'],
  [{boundary:'unpolarized'},'unsupported-config-or-numerics'],[{source:[-2,0,.01]},'unsupported-config-or-numerics'],
  [{k:NaN},'unsupported-config-or-numerics'],[{sigma:0},'unsupported-config-or-numerics'],
  [{inputWeight:-1},'unsupported-config-or-numerics'],[{rotationDeg:[Infinity,0,0]},'unsupported-config-or-numerics'],
  [{ballisticStart:.1},'unsupported-config-or-numerics'],[{mesh:{}},'unsupported-config-or-numerics'],
  [{proxySubdivisions:1.5},'unsupported-config-or-numerics'],[{receiver:[16,-2]},'unsupported-config-or-numerics']]){
  invalid(c,code);negativeControls++;
}
assert.throws(()=>packetField({...source,phaseRad:NaN},[0,0]));
assert.throws(()=>packetField({...source,coherentPhaseDefined:false},[0,0]));
console.log(JSON.stringify({passed:true,fixtureCases,localGridCases,minimumGridEndpointRatio,rigidTransformCases,negativeControls,envelopeTests,capContainmentTests,
  maximumPurityResidual,shapeIntegralError,actualBallisticStartWavelengths:16,shortProgressControl:[16,64],
  default:{k:canonical.parameters.k,softConditionalPower:canonical.conditionalPower,hardConditionalPower:hard.conditionalPower,
    endpointGuardRatios:canonical.localStages.map(s=>s.endpointGuardRatio),
    minimumTransitionArguments:canonical.localStages.map(s=>s.minTransitionArgument),
    originalCollimationGuard:canonical.originalCollimationGuard,
    physicalDistanceMm:receiver.opticalPathMm,proposalVirtualAgeMm:receiver.proposalVirtualAgeMm},
  scope:canonical.scope,physicallyCalibrated:false,fullPacketScatteringValidated:false}));
