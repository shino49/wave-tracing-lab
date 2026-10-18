// Independent one-mirror + one-region CPU verification. No renderer helpers are
// used by the SciPy physical Fermat solve or independent erfcx coefficients.
// Run: node tests/utd-reflected-independent.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {prepareReflectedUTDScene,evaluateReflectedPoint,reflectPoint,reflectDirection,mirrorIntersection} from '../dist/utd-mirror.js';
import {stablePlaneExcess} from '../dist/utd-scene.js';
import {add,sub,scale,dot,length,distance,normalize} from '../dist/mesh-geometry.js';

const root=new URL('../',import.meta.url);
const reference=JSON.parse(execFileSync('python',['research/utd-reflected-fermat-reference.py'],{
  cwd:root,encoding:'utf8',maxBuffer:20*1024*1024,
}));
const max={field:0,power:0,edgeField:0,point:0,mirrorPoint:0,range:0,sinBeta:0,transitionL:0,
  angle:0,physicalEikonal:0,phaseExcess:0,mirrorLaw:0,stationarity:0,reflectionHelper:0,rigidField:0};
const negativeControl={omitSoftMirrorSign:0,useOnlyLastOutgoingLegForSpreading:0,useOnlyLastOutgoingLegForPhase:0};
const rows=[],baseFields=new Map();
const assertClose=(key,value,tolerance,id)=>{
  max[key]=Math.max(max[key],value);
  assert(value<=tolerance,`${id}: ${key} error ${value} > ${tolerance}`);
};
const complexDistance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const multiply=(a,b)=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const angleDistance=(a,b)=>Math.abs(Math.atan2(Math.sin(a-b),Math.cos(a-b)));
for(const fixture of reference.boundedStripFixtures){
  const id=fixture.id,expected=fixture.expected,order=fixture.order==='RD'?'before':'after';
  const scene=prepareReflectedUTDScene(fixture.mesh,{...fixture.state,mirror:{...fixture.mirror,order}});
  assert(scene.supported,`${id}: admission rejected: ${JSON.stringify(scene.unsupported)}`);
  const result=evaluateReflectedPoint(scene,fixture.state.detectorCenter,{details:true});
  assert(result.valid,`${id}: invalid sample: ${result.invalidReasons}`);
  for(const field of [result.soft,result.hard,...result.contributions.flatMap(c=>[c.soft,c.hard])]){
    assert.equal(field.length,2,`${id}: scalar complex field has exactly two components`);
    assert(field.every(Number.isFinite),`${id}: finite scalar complex field`);
  }
  assert.equal(result.edgeCount,2,`${id}: need both physical strip edges`);
  assert.equal(result.directWeight,expected.directWeight,`${id}: GO weight`);
  assert.equal(result.reflectionCount,0,`${id}: forward local plate reflection should be absent`);
  for(const bc of ['soft','hard'])assertClose('field',complexDistance(result[bc],expected.total[bc]),1e-8,id);
  assertClose('power',Math.abs(result.f-expected.unpolarizedPower),1e-8,id);
  assertClose('power',Math.abs(result.power-result.f),1e-14,id);
  assert.equal(result.quantity,'one selected reflected-family conditional directional power transfer');
  assert.match(result.phaseGauge,/unfolded/);
  assert.equal(scene.mirrorTransport.reflection.soft,-1);
  assert.equal(scene.mirrorTransport.reflection.hard,1);
  const effectiveDestination=order==='after'?reflectPoint(fixture.state.detectorCenter,fixture.mirror):fixture.state.detectorCenter;
  assertClose('reflectionHelper',distance(effectiveDestination,expected.effectiveDestination),1e-11,id);
  assertClose('reflectionHelper',distance(scene.params.sourceOrigin,expected.effectiveSource),1e-11,id);
  assertClose('reflectionHelper',distance(scene.params.incidentDirection,expected.effectiveDirection),1e-11,id);
  assertClose('reflectionHelper',distance(reflectPoint(reflectPoint(fixture.state.detectorCenter,fixture.mirror),fixture.mirror),fixture.state.detectorCenter),1e-11,id);
  assertClose('reflectionHelper',distance(reflectDirection(reflectDirection(fixture.state.incidentDirection,fixture.mirror),fixture.mirror),fixture.state.incidentDirection),1e-11,id);

  const used=new Set();
  for(const actual of result.contributions){
    const edge=expected.edges.reduce((a,b)=>distance(actual.point,a.edgePoint)<distance(actual.point,b.edgePoint)?a:b);
    assert(!used.has(edge.edgeX),`${id}: repeated expected edge`);used.add(edge.edgeX);
    assertClose('point',distance(actual.point,edge.edgePoint),1e-9,id);
    assertClose('range',Math.abs(actual.ro-edge.outgoingRange),1e-10,id);
    assertClose('sinBeta',Math.abs(actual.sinBeta-edge.sinBeta),1e-12,id);
    assertClose('transitionL',Math.abs(actual.L-edge.L),1e-10,id);
    assertClose('angle',angleDistance(actual.phi_i,edge.phi_i),1e-11,id);
    assertClose('angle',angleDistance(actual.phi_o,edge.phi_o),1e-11,id);
    for(const bc of ['soft','hard'])assertClose('edgeField',complexDistance(actual[bc],edge.edge[bc]),1e-8,id);

    const q=actual.point,di=fixture.state.incidentDirection,source=fixture.state.sourceOrigin,dst=fixture.state.detectorCenter;
    let intersection,incoming,outgoing,eikonal;
    if(order==='before'){
      const sourceWavefrontPoint=sub(q,scale(scene.params.incidentDirection,dot(sub(q,scene.params.sourceOrigin),scene.params.incidentDirection)));
      intersection=mirrorIntersection(sourceWavefrontPoint,q,fixture.mirror);
      assert(intersection,`${id}: missing RD mirror hit`);
      incoming=di;outgoing=normalize(sub(q,intersection.point));
      const projectedIncomingLength=dot(sub(intersection.point,source),di);
      assert(projectedIncomingLength>0);
      eikonal=projectedIncomingLength+distance(intersection.point,q)+distance(q,dst);
    }else{
      intersection=mirrorIntersection(q,effectiveDestination,fixture.mirror);
      assert(intersection,`${id}: missing DR mirror hit`);
      incoming=normalize(sub(intersection.point,q));outgoing=normalize(sub(dst,intersection.point));
      eikonal=dot(sub(q,source),di)+distance(q,intersection.point)+distance(intersection.point,dst);
      // These controls prove that the fixture can detect substituting the final
      // leg for the complete edge -> mirror -> receiver range.
      const last=distance(intersection.point,dst),wrongScale=Math.sqrt(actual.ro/last);
      negativeControl.useOnlyLastOutgoingLegForSpreading=Math.max(negativeControl.useOnlyLastOutgoingLegForSpreading,
        complexDistance(actual.hard,actual.hard.map(v=>v*wrongScale)));
      const omitted=distance(q,intersection.point),wrongPhase=[Math.cos(scene.params.k*omitted),Math.sin(scene.params.k*omitted)];
      negativeControl.useOnlyLastOutgoingLegForPhase=Math.max(negativeControl.useOnlyLastOutgoingLegForPhase,
        complexDistance(actual.hard,multiply(actual.hard,wrongPhase)));
    }
    assertClose('mirrorPoint',distance(intersection.point,edge.mirrorPoint),1e-9,id);
    assertClose('mirrorLaw',distance(outgoing,reflectDirection(incoming,fixture.mirror)),1e-11,id);
    assertClose('physicalEikonal',Math.abs(eikonal-edge.physicalEikonal),1e-10,id);
    const excess=stablePlaneExcess(scene.params.incidentDirection,q,effectiveDestination);
    assertClose('phaseExcess',Math.abs(excess-(eikonal-expected.gauge)),1e-10,id);
    assertClose('phaseExcess',Math.abs(excess-edge.phaseExcess),1e-10,id);
    const edgeDirection=normalize(sub(fixture.mesh.vertices[3],fixture.mesh.vertices[0]));
    const atEdgeIn=order==='before'?normalize(sub(q,intersection.point)):di;
    const atEdgeOut=order==='before'?normalize(sub(dst,q)):normalize(sub(intersection.point,q));
    assertClose('stationarity',Math.abs(dot(sub(atEdgeIn,atEdgeOut),edgeDirection)),1e-11,id);
    negativeControl.omitSoftMirrorSign=Math.max(negativeControl.omitSoftMirrorSign,
      complexDistance(actual.soft,edge.edgeBeforeMirrorSign.soft));
  }
  assert.equal(used.size,2);
  const rigidKey=id.replace(/-(rotated|base)$/,'');
  if(!fixture.rotated)baseFields.set(rigidKey,result);
  else{
    const base=baseFields.get(rigidKey);assert(base);
    for(const bc of ['soft','hard'])assertClose('rigidField',complexDistance(result[bc],base[bc]),1e-9,id);
  }
  rows.push({id,order,valid:result.valid,edgeCount:result.edgeCount,directWeight:result.directWeight,
    fieldError:Math.max(...['soft','hard'].map(bc=>complexDistance(result[bc],expected.total[bc]))),
    powerError:Math.abs(result.f-expected.unpolarizedPower)});
}
for(const [name,difference] of Object.entries(negativeControl))assert(difference>.01,`Negative control ${name} not sensitive enough: ${difference}`);
const {fixtures,boundedStripFixtures,...physicalReference}=reference;
const report={suite:'utd-reflected-independent',cpuExecuted:true,gpuExecuted:false,
  method:'Actual reflected-family CPU renderer versus independent 3-variable constrained physical Fermat minimization and SciPy erfcx coefficients',
  reference:physicalReference,rendererCases:rows.length,edgeContributions:2*rows.length,maxErrors:max,negativeControl,
  cases:rows,limitations:[
    'One selected infinite ideal mirror + one local scalar GO/UTD family only',
    'Each coefficient checked against exact isolated half-plane Sommerfeld solution; coherent two-edge strip sum is not asserted to be exact strip scattering',
    'No independent GPU execution, finite mirror aperture, endpoint/corner, DD, general PLT, or cross-family coherent-phase claim',
  ]};
fs.writeFileSync(new URL('../research/utd-reflected-independent-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({suite:report.suite,cpuExecuted:true,rendererCases:rows.length,edgeContributions:report.edgeContributions,maxErrors:max,negativeControl},null,2));
