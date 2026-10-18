import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {traceCameraPath,randomGenerator} from '../dist/plt-core.js';
import {makeAperture,sampleFSD,pdfFSD} from '../dist/plt-fsd.js';
import {preparePLTScene,collectRegionGeometry} from '../dist/plt-geometry.js';
import {add,sub,scale,normalize,dot,cross,length} from '../dist/mesh-geometry.js';
const close=(a,b,t=1e-11)=>assert(Math.abs(a-b)<=t*Math.max(1,Math.abs(a),Math.abs(b)),a+' != '+b);
const quad=(vertices,materialIndex)=>({mesh:{vertices,faces:[[0,1,2],[0,2,3]]},materialIndex,surface:'open-plate'});
const finish=(entries,materials)=>{const scene=preparePLTScene(entries,{materials});scene.emitters=scene.triangles.filter(t=>materials[t.materialIndex].emission>0);scene.emitterArea=scene.emitters.reduce((s,t)=>s+t.area,0);return scene;};
const job=()=>({events:[],terms:[],paths:[],width:1,height:1,samplesPerPixel:1,stats:new Proxy({approximationWarnings:{}},{get:(o,p)=>p in o?o[p]:0})});
const beam=()=>({origin:[0,0,.5],direction:[0,0,-1],radius0:.25,tanAngle:0,near:1e-7,far:Infinity,k:2*Math.PI/.025,wavelengthMm:.025});
const trace=(scene,nee,maxDepth)=>{const j=job();traceCameraPath(scene,beam(),{wave:true,nee,maxDepth,boundary:'unpolarized'},()=>0,j);return j;};
const report={createdUtc:new Date().toISOString(),sourceHashes:{},checks:{},claim:'Independent proposal-measure and deterministic component-level MIS checks; no solid-angle-unbiased or global-energy claim.'};

// Independent derivative of the actual (edge coordinate, azimuth) -> sphere map.
const scene=preparePLTScene({vertices:[[0,-100,0],[100,-100,0],[100,100,0],[0,100,0]],faces:[[0,1,2],[0,2,3]]});
const source=[-3,0,-4],region=collectRegionGeometry(scene,{center:[0,0,0],direction:[0,0,1],radius:20,depth:40}),ap=makeAperture(region,normalize(sub(region.region.center,source)),.05),random=randomGenerator(83471);
const e=ap.edges[0],a=dot(sub(source,e.v),e.e),rho=length(sub(sub(source,e.v),scale(e.e,a))),t=3,phi=1.32;
const wo=(t,p)=>{const c=(t-a)/Math.hypot(rho,t-a),s=Math.sqrt(1-c*c);return add(scale(add(scale(e.tff,Math.cos(p)),scale(e.nff,Math.sin(p))),s),scale(e.e,c));};
const h=1e-5,dt=scale(sub(wo(t+h,phi),wo(t-h,phi)),1/(2*h)),dp=scale(sub(wo(t,phi+h),wo(t,phi-h)),1/(2*h));
const jac=length(cross(dt,dp)),exact=rho*rho/Math.hypot(rho,t-a)**3,ri=Math.hypot(rho,t-a),g=pdfFSD(ap,source,wo(t,phi)),qOmega=g*ri/(e.l*(1-((t-a)/ri)**2));
assert(Math.abs(jac/exact-1)<1e-8);
report.checks.solidAngleJacobian={numeric:jac,analytic:exact,relativeError:Math.abs(jac/exact-1),returnedAngleDensity:g,actualSolidAngleDensity:qOmega};
const n=100000;let sum=0,sum2=0,direct=0,rejected=0;
for(let i=0;i<n;i++){const s=sampleFSD(ap,source,random);if(!s.valid){rejected++;continue;}sum+=s.weight;sum2+=s.weight*s.weight;if(s.isDirect)direct++;}
const mean=sum/n,se=Math.sqrt((sum2/n-mean*mean)/n),expect=1+2*Math.PI;
assert.equal(rejected,0);assert(Math.abs(mean-expect)<4*se);
report.checks.artificialConstantIntegrand={samples:n,direct,rejected,mean,standardError:se,analyticNativeMixedMeasure:expect,physicalConservationTest:false};

// One deterministic diffuse bounce by an actual finite edge, followed by a light.
// Zero RNG gives a normal outgoing BSDF ray in both NEE modes, while NEE selects
// a known emitter vertex. It is intentionally not a Monte Carlo mean test.
const plate=quad([[-10,-10,0],[.1,-10,0],[.1,10,0],[-10,10,0]],0),emitter=quad([[-.4,-.4,1],[-.4,.4,1],[.4,.4,1],[.4,-.4,1]],1);
const simple=finish([plate,emitter],[{type:'diffuse',albedo:.63,emission:0},{type:'diffuse',albedo:0,emission:4,twoSidedEmission:true}]);
const off=trace(simple,false,2),on=trace(simple,true,2);
assert.equal(off.events.length,1);assert.equal(on.events.length,1);assert(off.events[0].edges.length>0);assert(off.events[0].referenceF>0);
close(off.events[0].referenceF,on.events[0].referenceF);
const offOrganic=off.terms.slice(-2),onOrganic=on.terms.slice(-2);
assert.equal(offOrganic[0].eventIds.length,0);assert.equal(offOrganic[1].eventIds.length,1);
assert.equal(onOrganic[0].neeMatched,true);assert.equal(onOrganic[1].neeMatched,false);
const bsdfPDF=1/Math.PI,lightPDF=1/.64,mis=bsdfPDF**2/(bsdfPDF**2+lightPDF**2);
close(onOrganic[0].weight,offOrganic[0].weight*mis);
close(onOrganic[1].weight,offOrganic[1].weight);
close(onOrganic[1].weight,4);
report.checks.immediateOrganicFSD={localF:on.events[0].referenceF,bsdfMIS:mis,baseWeightWithNEE:onOrganic[0].weight,baseWeightWithoutNEE:offOrganic[0].weight,addedFSDWeightWithNEE:onOrganic[1].weight,addedFSDWeightWithoutNEE:offOrganic[1].weight,addedFSDPower:4*on.events[0].referenceF,pass:true};

// A subsequent ordinary surface makes all carried terms part of that surface's
// BSDF/NEE pair again. Use a tilted second diffuse plate and a third emitter.
const z=Math.SQRT1_2,second=quad([[-z,-1,1-z],[z,-1,1+z],[z,1,1+z],[-z,1,1-z]],1),third=quad([[1,-.4,-.4],[1,.4,-.4],[1,.4,.4],[1,-.4,.4]],2);
const again=finish([plate,second,third],[{type:'diffuse',albedo:.63,emission:0},{type:'diffuse',albedo:.8,emission:0},{type:'diffuse',albedo:0,emission:4,twoSidedEmission:true}]);
const secondOff=trace(again,false,3),secondOn=trace(again,true,3),lastOff=secondOff.terms.slice(-2),lastOn=secondOn.terms.slice(-2);
assert.equal(secondOff.events.length,1);assert.equal(secondOn.events.length,1);
for(const t of lastOn)assert.equal(t.neeMatched,true);
assert.equal(lastOn[0].eventIds.length,0);assert.equal(lastOn[1].eventIds.length,1);
const q=2/(.64*Math.SQRT1_2),w=bsdfPDF**2/(bsdfPDF**2+q**2);
close(lastOn[0].weight,lastOff[0].weight*w);close(lastOn[1].weight,lastOff[1].weight*w);
report.checks.subsequentSurfaceRestoresMIS={bsdfMIS:w,baseWeightRatio:lastOn[0].weight/lastOff[0].weight,addedFSDWeightRatio:lastOn[1].weight/lastOff[1].weight,allOrganicTermsMatched:true,pass:true};
for(const file of ['dist/plt-core.js','dist/plt-fsd.js','dist/plt-geometry.js'])report.sourceHashes[file]=crypto.createHash('sha256').update(fs.readFileSync(new URL('../'+file,import.meta.url))).digest('hex');
report.allPass=true;
fs.writeFileSync(new URL('plt-sampling-review-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));

