import assert from 'node:assert/strict';
import {
  makeAperture,sampleFSD,pdfFSD,wrappedNormalPDF,diffractionPointDirection,
  stableBrokenPathExcess,evaluateFSD,evaluateOperator,FSD_LIMITS,
} from '../dist/plt-fsd.js';
import {preparePLTScene,collectRegionGeometry,coneContains} from '../dist/plt-geometry.js';
import {add,sub,scale,dot,cross,normalize,distance,diffractionPoint} from '../dist/mesh-geometry.js';
import {coefficient,add as ca,mul as cm,scale as cs,cis,abs2} from '../dist/utd-math.js';

let checks=0;
function check(name,fn){fn();checks++;console.log(`PASS ${name}`);}
const close=(a,b,tol=1e-11)=>assert(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const vectorClose=(a,b,tol=1e-11)=>a.forEach((x,i)=>close(x,b[i],tol));
function rng(seed=17){let a=seed>>>0;return()=>{a=(1664525*a+1013904223)>>>0;return a/4294967296;};}
const mesh={vertices:[[0,-100,0],[100,-100,0],[100,100,0],[0,100,0]],faces:[[0,1,2],[0,2,3]]};
const scene=preparePLTScene(mesh),source=[-3,0,-4];
const region=collectRegionGeometry(scene,{center:[0,0,0],direction:[0,0,1],radius:20,depth:40});
const incoming=normalize(sub(region.region.center,source)),aperture=makeAperture(region,incoming,100);
const coneTo=destination=>({origin:source,direction:normalize(sub(destination,source)),radius0:50,tanAngle:.1,near:0,far:1000});

check('one collected, clipped edge; finite physical provenance',()=>{
  assert.equal(aperture.edges.length,1);assert.equal(aperture.edges[0].l,40);
  assert.equal(aperture.edges[0].originalLength,200);
  assert(aperture.edges[0].clippedStart&&aperture.edges[0].clippedEnd);
  assert.equal(aperture.k,100);assert.equal(aperture.boundaryEpsilon,2e-6);
  assert.throws(()=>makeAperture(region,incoming,0),RangeError);
  assert.throws(()=>makeAperture(region,incoming,1,'dielectric'),RangeError);
});

check('wrapped-normal angular density integrates to one over all widths',()=>{
  for(const sigma of [.025,.1,.4,.99,1,2,5,30]) {
    let integral=0;const steps=24000;
    for(let j=0;j<steps;j++)integral+=wrappedNormalPDF((j+.5)*2*Math.PI/steps,sigma)*2*Math.PI/steps;
    close(integral,1,3e-12);
    close(wrappedNormalPDF(.123,sigma),wrappedNormalPDF(.123+8*Math.PI,sigma),1e-12);
    close(wrappedNormalPDF(.123,sigma),wrappedNormalPDF(-.123,sigma),1e-12);
  }
  assert.throws(()=>wrappedNormalPDF(0,0),RangeError);
});

check('finite-source directional saddle obeys Keller tangential conservation',()=>{
  const edge=aperture.edges[0],random=rng();
  for(let i=0;i<300;i++) {
    const p=add(edge.v,scale(edge.e,(random()-.5)*edge.l*.8));
    const wi=normalize(sub(source,p)),c=dot(wi,edge.e),s=Math.sqrt(1-c*c),phi=random()*2*Math.PI;
    const wo=add(scale(add(scale(edge.tff,Math.cos(phi)),scale(edge.nff,Math.sin(phi))),s),scale(edge.e,-c));
    const got=diffractionPointDirection(edge,source,wo);assert(got);vectorClose(got.point,p,2e-11);close(got.fermatResidual,0,2e-12);
    const dst=add(p,scale(wo,7)),finite=diffractionPoint(edge,source,dst);assert(finite);vectorClose(finite.point,p,2e-11);
  }
  assert.equal(diffractionPointDirection(edge,source,edge.e),null);
});

check('sampling uses all edge-plus-direct branches and the stated measure',()=>{
  const random=rng(871),count=20000;let direct=0,edge=0,rejected=0;
  for(let i=0;i<count;i++) {
    const sample=sampleFSD(aperture,source,random);
    assert.equal(sample.diagnostics.solidAngleUnbiased,false);
    if(!sample.valid){rejected++;continue;}
    if(sample.isDirect){direct++;assert.equal(sample.weight,2);assert.equal(sample.pdf.measure,'discrete');close(sample.pdf.value,.5);}
    else {
      edge++;assert.equal(sample.pdf.measure,'angle');close(sample.weight*sample.pdf.value,1);
      close(sample.pdf.value,pdfFSD(aperture,source,sample.wo));
      close(sample.auxiliary.positionDensity,1/40);close(sample.auxiliary.edgeProbability,.5);
      const hit=diffractionPointDirection(aperture.edges[0],source,sample.wo);assert(hit);vectorClose(hit.point,sample.point,2e-10);
    }
  }
  assert(Math.abs(direct/count-.5)<.015);assert(edge>9000);assert(rejected<50);
  assert.throws(()=>sampleFSD(aperture,source,()=>1),RangeError);
});

check('proposal angle integral uses N+1 direct mixture, not solid angle',()=>{
  const edge=aperture.edges[0],wi=normalize(sub(source,edge.v)),c=dot(wi,edge.e),s=Math.sqrt(1-c*c);
  let integral=0;const steps=12000;
  for(let i=0;i<steps;i++){
    const p=(i+.5)*2*Math.PI/steps,wo=add(scale(add(scale(edge.tff,Math.cos(p)),scale(edge.nff,Math.sin(p))),s),scale(edge.e,-c));
    integral+=pdfFSD(aperture,source,wo)*2*Math.PI/steps;
  }
  close(integral,.5,2e-11);
});

check('stable path excess resolves near-forward cancellation',()=>{
  close(stableBrokenPathExcess([0,0,0],[5,0,0],[10,0,0]),0);
  close(stableBrokenPathExcess([0,0,0],[3,4,0],[6,0,0]),4);
  const got=stableBrokenPathExcess([-1e9,0,0],[0,1,0],[1e9,0,0]);
  close(got,1e-9,1e-15);assert(got>0);
  assert.equal(2*Math.hypot(1e9,1)-2e9,0,'naive subtraction loses this excess');
  const src=[-3,1,-4],p=[0,0,0],dst=[5,2,4],offset=[1e5,-3e4,8e4];
  close(stableBrokenPathExcess(src,p,dst),stableBrokenPathExcess(add(src,offset),add(p,offset),add(dst,offset)),2e-11);
});

check('local evaluator matches separately assembled direct and edge fields',()=>{
  for(const dst of [[-3,0,4],[5,2,4],[4,0,-3]]) {
    const ret=evaluateFSD(scene,aperture,source,dst,coneTo(dst));assert(ret.valid);assert(ret.edgeCount>0);assert(ret.F>=0);
    const hit=diffractionPoint(aperture.edges[0],source,dst),w=aperture.edges[0];
    const pi=Math.atan2(dot(hit.wi,w.nff),dot(hit.wi,w.tff));let po=Math.atan2(dot(hit.wo,w.nff),dot(hit.wo,w.tff));if(po<0)po+=2*Math.PI;
    const L=hit.ro*hit.sinBeta**2,phase=cs(cis(-100*(hit.ri+hit.ro-distance(source,dst))),1/Math.sqrt(hit.ro));
    const soft=ca([ret.directWeight,0],cm(coefficient(100,2,pi,po,L,hit.sinBeta,'soft',2e-6),phase));
    const hard=ca([ret.directWeight,0],cm(coefficient(100,2,pi,po,L,hit.sinBeta,'hard',2e-6),phase));
    vectorClose(ret.soft,soft,2e-12);vectorClose(ret.hard,hard,2e-12);close(ret.F,(abs2(soft)+abs2(hard))/2);
    assert.deepEqual(evaluateOperator(ret.operator),{F:ret.F,f:ret.F,soft:ret.soft,hard:ret.hard});
  }
});

check('direct shadow seam uses the matched half-amplitude convention',()=>{
  const dst=[3,0,4],ret=evaluateFSD(scene,aperture,source,dst,coneTo(dst));
  assert.equal(ret.directWeight,.5);assert(ret.diagnostics.directBoundary);assert(ret.F>.24&&ret.F<.26);
  const left=[3-1e-7,0,4],right=[3+1e-7,0,4];
  assert.equal(evaluateFSD(scene,aperture,source,left,coneTo(left)).directWeight,.5);
  assert.equal(evaluateFSD(scene,aperture,source,right,coneTo(right)).directWeight,.5);
});

check('real occluder blocks the edge legs and the direct term',()=>{
  const blocker={vertices:[[-10,-30,-2],[10,-30,-2],[10,30,-2],[-10,30,-2]],faces:[[0,1,2],[0,2,3]]};
  const blocked=preparePLTScene([{mesh},{mesh:blocker}]),dst=[-3,0,4];
  const ret=evaluateFSD(blocked,aperture,source,dst,coneTo(dst));
  assert.equal(ret.directWeight,0);assert.equal(ret.edgeCount,0);assert.equal(ret.F,0);
});

check('finite endpoints are excluded explicitly; no endpoint clamping',()=>{
  const edge=aperture.edges[0],short={...edge,a:[0,-1,0],b:[0,1,0],v:[0,0,0],l:2,originalA:[0,-1,0],originalB:[0,1,0]};
  const ap={...aperture,edges:[short]},src=[-3,5,-4],dst=[5,5,4];
  const ret=evaluateFSD(scene,ap,src,dst,{...coneTo(dst),origin:src});
  assert.equal(ret.edgeCount,0);assert(ret.diagnostics.endpointExclusions>0);
  assert(ret.diagnostics.excluded.some(x=>x.reason==='no-finite-edge-stationary-point'));
  const at=evaluateFSD(scene,ap,[-3,1,-4],[5,1,4],{...coneTo(dst),origin:[-3,1,-4]});
  assert(at.diagnostics.excluded.some(x=>x.reason==='endpoint-stationary-point-not-modeled'));
});

check('near endpoints and low frequency remain visibly qualified approximations',()=>{
  const edge=aperture.edges[0],small={...edge,a:[0,-.1,0],b:[0,.1,0],v:[0,0,0],l:.2,originalA:[0,-.1,0],originalB:[0,.1,0]};
  const ret=evaluateFSD(scene,{...aperture,k:.1,edges:[small]},source,[5,0,4],coneTo([5,0,4]));
  assert(ret.valid);assert(ret.diagnostics.warnings.includes('low-frequency-asymptotic-approximation'));
  assert(ret.diagnostics.warnings.includes('finite-endpoint-neighborhood-not-modeled'));
  assert.equal(ret.diagnostics.fullWaveValidated,false);
});

check('rigid 3D transforms preserve center connection and power',()=>{
  const q=.713,c=Math.cos(q),s=Math.sin(q),shift=[123,-45,81],axis=normalize([.3,-.7,.6]);
  const rotate=p=>add(add(scale(p,c),scale(cross(axis,p),s)),scale(axis,dot(axis,p)*(1-c))),transform=p=>add(rotate(p),shift);
  const transformed=preparePLTScene({vertices:mesh.vertices.map(transform),faces:mesh.faces});
  const reg=collectRegionGeometry(transformed,{center:transform([0,0,0]),direction:rotate([0,0,1]),radius:20,depth:40});
  const ap=makeAperture(reg,rotate(incoming),100),dst=[5,2,4],before=evaluateFSD(scene,aperture,source,dst,coneTo(dst));
  const cone=coneTo(dst),after=evaluateFSD(transformed,ap,transform(source),transform(dst),{...cone,origin:transform(source),direction:rotate(cone.direction)});
  close(after.F,before.F,3e-11);vectorClose(after.soft,before.soft,3e-11);assert.equal(after.edgeCount,before.edgeCount);
});

check('a region coherently sums every contributing edge before squaring',()=>{
  const square={vertices:[[0,-10,0],[10,-10,0],[10,10,0],[0,10,0]],faces:mesh.faces};
  const s=preparePLTScene(square),r=collectRegionGeometry(s,{center:[0,0,0],direction:[0,0,1],radius:40,depth:80});
  const a=makeAperture(r,incoming,100),dst=[5,2,4],ret=evaluateFSD(s,a,source,dst,coneTo(dst));
  assert.equal(a.edges.length,4);assert(ret.records.length>=3);
  const incoherent=ret.records.reduce((sum,edge)=>sum+evaluateOperator({...ret.operator,directAmplitude:0,edges:[edge]}).F,ret.directWeight**2);
  assert(Math.abs(ret.F-incoherent)>1e-7,'inter-edge cross terms must actually contribute');
  const reversed=evaluateFSD(s,{...a,edges:[...a.edges].reverse()},source,dst,coneTo(dst));
  close(ret.F,reversed.F,1e-12);
  const selected=new Set(),random=rng(719);let direct=0;
  for(let i=0;i<10000;i++){
    const sample=sampleFSD(a,source,random);
    if(sample.isDirect){direct++;assert.equal(sample.weight,5);}
    else if(sample.valid)selected.add(sample.edgeId);
  }
  assert.equal(selected.size,4);assert(Math.abs(direct/10000-.2)<.015);
});

check('boundary response is scalar soft/hard or their unpolarized intensity mean',()=>{
  const dst=[5,2,4],cone=coneTo(dst),u=evaluateFSD(scene,aperture,source,dst,cone);
  const s=evaluateFSD(scene,{...aperture,boundary:'soft'},source,dst,cone),h=evaluateFSD(scene,{...aperture,boundary:'hard'},source,dst,cone);
  close(s.F,abs2(u.soft));close(h.F,abs2(u.hard));close(u.F,(s.F+h.F)/2);
});

check('scale-free tangent band is symmetric in aperture, PDF, sample and evaluation',()=>{
  for(const sign of [-1,1]) {
    assert.equal(makeAperture(region,normalize([1,0,sign*5e-13]),100).edges.length,0);
    assert.equal(pdfFSD(aperture,source,normalize([1,0,sign*5e-13])),0);
    const dst=[5,0,sign*2e-13],ret=evaluateFSD(scene,aperture,source,dst,coneTo(dst));
    assert.equal(ret.edgeCount,0);assert.equal(ret.F,1);
  }
  const edge=aperture.edges[0],wi=normalize(sub(source,edge.v));
  const phiI=Math.atan2(dot(wi,edge.nff),dot(wi,edge.tff)),sigma=Math.sqrt(45/(100*5));
  // Force a Box–Muller proposal exactly onto the outgoing plate tangent.
  const gaussianMagnitude=(Math.PI-phiI)/sigma,u=Math.exp(-.5*gaussianMagnitude**2);
  const numbers=[0,.5,u,.5,.75],sample=sampleFSD(aperture,source,()=>numbers.shift());
  assert.equal(sample.valid,false);assert.equal(sample.weight,0);
  assert.equal(sample.diagnostics.reason,'outgoing-inside-wedge');
});

check('empty aperture has the direct discrete event and no invented diffraction',()=>{
  const empty=makeAperture({...region,wedges:[]},incoming,100),dst=[-3,0,4];
  const sample=sampleFSD(empty,source,rng());assert(sample.valid&&sample.isDirect);assert.equal(sample.weight,1);assert.equal(pdfFSD(empty,source,[0,0,1]),0);
  const ret=evaluateFSD(scene,empty,source,dst,coneTo(dst));assert.equal(ret.F,1);assert.equal(ret.edgeCount,0);
  const narrow={origin:source,direction:[0,0,-1],radius0:.01,tanAngle:.01,near:0,far:100};
  assert.equal(coneContains(narrow,dst),false);assert.equal(evaluateFSD(scene,empty,source,dst,narrow).F,0);
});

console.log(JSON.stringify({suite:'plt-fsd',checks,gpuExecuted:false,claim:'Local finite-source center connections, angular proposal identities and CPU operator records; not a full-wave or absolute-energy validation.'}));
