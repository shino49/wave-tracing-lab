import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {preparePLTScene,segmentVisible} from '../dist/plt-geometry.js';
import {prepareMesh,transformMesh,transformPoint,distance} from '../dist/mesh-geometry.js';
import {evaluateMeshDiffraction,prepareMeshDiffraction,COHERENT_DIFFRACTION_LIMITS} from '../dist/coherent-mesh-diffraction.js';
import {stableBrokenPathExcess} from '../dist/plt-fsd.js';

const PI=Math.PI,TAU=2*PI;
const gold=JSON.parse(fs.readFileSync(new URL('../research/coherent-mesh-diffraction-reference.json',import.meta.url)));
const report={scope:'First-order spherical-source scalar mesh UTD; no general full-wave, impedance, multiple-diffraction or flux certification',checks:{},thresholds:gold.thresholds};
let checks=0;
const check=(name,f)=>{f();checks++;console.log(`PASS ${name}`);};
const near=(a,b,t=2e-11)=>assert(Math.abs(a-b)<=t*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const nearComplex=(a,b,t=2e-11)=>{near(a[0],b[0],t);near(a[1],b[1],t);};
const field=r=>[r.amplitude*Math.cos(r.phaseRadReduced),r.amplitude*Math.sin(r.phaseRadReduced)];
const sum=routes=>routes.reduce((s,r)=>{const f=field(r);return[s[0]+f[0],s[1]+f[1]];},[0,0]);
const material=(boundary='soft')=>({type:'mirror',reflectance:1,phaseRad:boundary==='soft'?PI:0});
const plate=(lo=0,hi=100,h=100,z=0)=>({vertices:[[lo,-h,z],[hi,-h,z],[hi,h,z],[lo,h,z]],faces:[[0,1,2],[0,2,3]]});
const scene=(boundary='soft',mesh=plate())=>preparePLTScene(mesh,{materials:[material(boundary)]});
const central=r=>r.routes.find(p=>Math.abs(p.point[0])<1e-8&&Math.abs(p.point[2])<1e-8);
const src=[-3,0,-4];

check('spherical spreading, reduced distance, phase and skew edge geometry match independent SciPy',()=>{
  let max=0;
  for(const c of gold.finiteSource){
    const out=evaluateMeshDiffraction(scene(c.boundary),c.source,c.receiver,c),r=central(out);assert(r);
    const value=field(r),error=Math.hypot(value[0]-c.field[0],value[1]-c.field[1]);max=Math.max(max,error);
    nearComplex(value,c.field,gold.thresholds.edgeComplexAbsolute);
    for(const key of ['ri','ro','L','sinBeta'])near(r[key],c[key]);
    r.point.forEach((x,i)=>near(x,c.point[i]));
  }
  report.checks.independentSpherical={cases:gold.finiteSource.length,maxComplexAbsoluteError:max};
});

function totalHalfPlate(s,source,receiver,options){
  const out=evaluateMeshDiffraction(s,source,receiver,options),value=sum(out.routes),R=distance(source,receiver),A=Math.sqrt((options.power??1)/(4*PI)),k=TAU/options.wavelengthMm;
  const directWeight=out.directBoundary?.5:segmentVisible(s,source,receiver)?1:0;
  value[0]+=directWeight*A/R;
  if(source[2]*receiver[2]>0){
    const image=[source[0],source[1],-source[2]],t=-image[2]/(receiver[2]-image[2]);
    const bounce=image.map((x,i)=>x+t*(receiver[i]-x)),boundary=out.reflectionBoundaries.length>0;
    if((bounce[0]>=0&&bounce[0]<=100&&Math.abs(bounce[1])<=100)||boundary){
      const r=distance(image,receiver),weight=boundary?.5:1,phase=-k*stableBrokenPathExcess(source,bounce,receiver)+s.materials[0].phaseRad;
      value[0]+=weight*A/r*Math.cos(phase);value[1]+=weight*A/r*Math.sin(phase);
    }
  }
  return {value,out};
}
check('finite-source direct and reflection seams pair with half-weight GO fields',()=>{
  let maxCentreError=0,maxSideGap=0;
  for(const boundary of ['soft','hard'])for(const z of [10,-10]){
    const s=scene(boundary),o={wavelengthMm:.05};
    const middle=totalHalfPlate(s,src,[7.5,0,z],o),left=totalHalfPlate(s,src,[7.5-1e-8,0,z],o),right=totalHalfPlate(s,src,[7.5+1e-8,0,z],o);
    if(z>0)assert.equal(middle.out.directBoundary,true);else assert.equal(middle.out.reflectionBoundaries.length,1);
    const mean=left.value.map((x,i)=>(x+right.value[i])/2),err=Math.hypot(...mean.map((x,i)=>x-middle.value[i]));
    const gap=Math.hypot(...left.value.map((x,i)=>x-right.value[i]));maxCentreError=Math.max(maxCentreError,err);maxSideGap=Math.max(maxSideGap,gap);
    nearComplex(middle.value,mean,2e-9);assert(gap<2e-8);
  }
  report.checks.seams={maxCentreVersusSidesMean:maxCentreError,maxLeftRightGap:maxSideGap};
});

check('scalar reciprocity, power scaling and a common source phase gauge are preserved',()=>{
  for(const boundary of ['soft','hard']){
    const s=scene(boundary),dst=[2.4,7.3,9.1],o={wavelengthMm:.005};
    const forward=evaluateMeshDiffraction(s,src,dst,o),reverse=evaluateMeshDiffraction(s,dst,src,o);
    const reversed=new Map(reverse.routes.map(r=>[r.edgeId,r]));
    for(const r of forward.routes)nearComplex(field(r),field(reversed.get(r.edgeId)),2e-10);
    const high=evaluateMeshDiffraction(s,src,dst,{...o,power:9});
    nearComplex(sum(high.routes),sum(forward.routes).map(x=>3*x));
    const shifted=evaluateMeshDiffraction(s,src,dst,{...o,referenceDistance:distance(src,dst)+.001});
    const base=sum(forward.routes),phase=TAU/.005*.001,expected=[base[0]*Math.cos(phase)-base[1]*Math.sin(phase),base[0]*Math.sin(phase)+base[1]*Math.cos(phase)];
    nearComplex(sum(shifted.routes),expected,2e-10);
  }
  report.checks.reciprocityAndScaling=true;
});

function griddedPlate(nx=4,ny=8){
  const vertices=[],faces=[];
  for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++)vertices.push([100*x/nx,-100+200*y/ny,0]);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const a=y*(nx+1)+x,b=a+1,c=b+nx+1,d=c-1;faces.push([a,b,c],[a,c,d]);}
  return {vertices,faces};
}
check('physical edge merging removes tessellation multiplicity; 3-D rigid transforms preserve the field',()=>{
  const s=scene(),fine=scene('soft',griddedPlate()),dst=[2.4,7.3,9.1],o={wavelengthMm:.005};
  assert.equal(s.wedges.length,4);assert.equal(fine.wedges.length,4);
  nearComplex(sum(evaluateMeshDiffraction(s,src,dst,o).routes),sum(evaluateMeshDiffraction(fine,src,dst,o).routes),3e-10);
  const tr={rotationDeg:[23,31,-17],translation:[4,-3,7]},rotated=scene('soft',transformMesh(prepareMesh(plate()),tr));
  const original=evaluateMeshDiffraction(s,src,dst,o),changed=evaluateMeshDiffraction(rotated,transformPoint(src,tr),transformPoint(dst,tr),o);
  nearComplex(sum(original.routes),sum(changed.routes),3e-10);report.checks.subdivisionAndRigidTransform=true;
});

check('scene-global physical boundaries survive cross-entry T-junctions and partial coplanar overlays',()=>{
  const rect=(lo,hi,bot,top)=>({vertices:[[lo,bot,0],[hi,bot,0],[hi,top,0],[lo,top,0]],faces:[[0,1,2],[0,2,3]]});
  const fixtures=[
    {name:'partial-overlap',edges:4,a:[rect(-2,2,-2,2)],b:[rect(-2,1,-2,2),rect(-1,2,-2,2)]},
    {name:'T-junction',edges:4,a:[rect(-2,2,-2,2)],b:[rect(-2,0,-2,2),rect(0,2,-2,0),rect(0,2,0,2)]},
    {name:'partial-overlap-L-union',edges:6,a:[rect(-2,0,-2,2),rect(0,2,-2,0)],b:[rect(-2,2,-2,0),rect(-2,0,-1,2)]},
  ];
  const values=[];
  for(const fixture of fixtures){
    const scenes=[fixture.a,fixture.b].map(meshes=>preparePLTScene(meshes.map(mesh=>({mesh})),{materials:[material()]}));
    for(const s of scenes)assert.equal(prepareMeshDiffraction(s).eligible.length,fixture.edges);
    let max=0;
    for(const dst of [[1,.4,5],[0,0,5],[1,.4,-5],[0,0,-5]]){
      const outputs=scenes.map(s=>evaluateMeshDiffraction(s,[-1,.2,-3],dst,{wavelengthMm:.03}));
      assert.equal(outputs[0].directBoundary,outputs[1].directBoundary);
      assert.equal(outputs[0].reflectionBoundaries.length,outputs[1].reflectionBoundaries.length);
      assert.equal(outputs[1].diagnostics.materialExcludedEdges,0);
      const fields=outputs.map(o=>sum(o.routes));nearComplex(fields[0],fields[1],2e-11);
      max=Math.max(max,Math.hypot(...fields[0].map((x,i)=>x-fields[1][i])));
    }
    values.push({name:fixture.name,physicalEdges:fixture.edges,maxComplexAbsoluteError:max});
  }
  report.checks.coplanarUnion=values;
});

check('occlusion checks the full scene and never hides whole adjacent faces',()=>{
  const blocker={vertices:[[-5,-10,-2],[5,-10,-2],[5,10,-2],[-5,10,-2]],faces:[[0,1,2],[0,2,3]]};
  const s=preparePLTScene([{mesh:plate(),materialIndex:0},{mesh:blocker,materialIndex:1}],{materials:[material(),{type:'absorber'}]});
  const out=evaluateMeshDiffraction(s,src,[5,0,10],{wavelengthMm:.05});
  assert(!central(out));assert(out.diagnostics.excluded.some(x=>x.reason==='occluded-edge-leg'));
  assert(out.diagnostics.materialExcludedEdges>0);assert(out.diagnostics.warnings.includes('edge-material-diffraction-not-modeled'));
  report.checks.fullSceneOcclusion=true;
});

check('material admission rejects absorption, loss, arbitrary phase and mixed boundary seams',()=>{
  for(const m of [{type:'absorber'}, {type:'mirror',reflectance:.9,phaseRad:PI}, {type:'mirror',reflectance:1,phaseRad:.2}]){
    const s=preparePLTScene(plate(),{materials:[m]}),out=evaluateMeshDiffraction(s,src,[5,0,10],{wavelengthMm:.05});
    assert.equal(out.routes.length,0);assert.equal(out.diagnostics.materialExcludedEdges,4);
  }
  const mesh=griddedPlate(1,2),s=preparePLTScene({mesh,materialIndices:[0,0,1,1]},{materials:[material('soft'),material('hard')]});
  assert(prepareMeshDiffraction(s).excluded.some(x=>x.reason==='inconsistent-scalar-boundary-across-edge'));
  const mutable=scene();assert.equal(prepareMeshDiffraction(mutable).eligible.length,4);mutable.materials[0].phaseRad=.3;
  assert.equal(prepareMeshDiffraction(mutable).eligible.length,0);
  const hard=[0,TAU,-TAU].map(phaseRad=>evaluateMeshDiffraction(preparePLTScene(plate(),{materials:[{type:'mirror',reflectance:1,phaseRad}]}),src,[5,0,10],{wavelengthMm:.05}));
  for(const item of hard){assert(item.routes.length>0);assert.equal(item.diagnostics.materialExcludedEdges,0);nearComplex(sum(item.routes),sum(hard[0].routes));}
  report.checks.materialAdmission=true;
});

check('endpoint and low-frequency estimates are explicit; safety caps reject rather than truncate',()=>{
  const endpoint=evaluateMeshDiffraction(scene(),[-3,100,-4],[5,100,10],{wavelengthMm:.05});
  assert(endpoint.diagnostics.excluded.some(x=>x.reason==='endpoint-stationary-point-not-modeled'));
  assert(endpoint.diagnostics.warnings.includes('endpoint-dominated-diffraction-not-modeled'));
  const low=evaluateMeshDiffraction(scene(),src,[5,0,10],{wavelengthMm:50});assert(low.diagnostics.warnings.includes('low-kL-asymptotic-regime'));assert(central(low));
  const entries=Array.from({length:17},(_,i)=>({mesh:plate(3*i,3*i+1,1,i),materialIndex:0}));
  const tooMany=preparePLTScene(entries,{materials:[material()]}),out=evaluateMeshDiffraction(tooMany,[-2,0,-4],[2,0,30],{wavelengthMm:.05});
  assert.equal(out.routes.length,0);assert.equal(out.diagnostics.valid,false);assert(out.diagnostics.invalidReasons.includes('eligible-edge-safety-cap'));
  assert.equal(COHERENT_DIFFRACTION_LIMITS.maxEligibleEdges,64);
  assert.throws(()=>evaluateMeshDiffraction(scene(),src,[2,0,10],{wavelengthMm:0}),RangeError);
  assert.throws(()=>evaluateMeshDiffraction(scene(),src,[2,0,10],{wavelengthMm:1,power:-1}),RangeError);
  assert.equal(evaluateMeshDiffraction(scene(),src,[2,0,10],{wavelengthMm:1,power:0}).routes.length,0);
  report.checks.explicitLimits=true;
});

check('the same mesh edge operator reproduces the bounded two-slit Fresnel source limit',()=>{
  const ref=gold.doubleSlitFresnel,[[a,b],[c,d]]=ref.intervals,H=ref.plateHalfHeight,W=ref.plateHalfWidth;
  const entries=[[-W,a],[b,c],[d,W]].map(([lo,hi])=>({mesh:plate(lo,hi,H),materialIndex:0}));
  const results=[];
  for(const boundary of ['soft','hard']){
    const s=preparePLTScene(entries,{materials:[material(boundary)]});let max=0,numerator=0,denominator=0,warned=0;
    for(const fixture of ref.cases){
      const out=evaluateMeshDiffraction(s,ref.source,fixture.receiver,ref),u=sum(out.routes),R=distance(ref.source,fixture.receiver),A=1/(Math.sqrt(4*PI)*R);
      u[0]+=(out.directBoundary?.5:segmentVisible(s,ref.source,fixture.receiver)?1:0)*A;
      const normalized=u.map(x=>x/A),error=Math.hypot(...normalized.map((x,i)=>x-fixture.relativeField[i]));
      max=Math.max(max,error);numerator+=error*error;denominator+=fixture.relativeField[0]**2+fixture.relativeField[1]**2;if(out.diagnostics.warnings.length)warned++;
    }
    results.push({boundary,samples:ref.cases.length,maxNormalizedComplexAbsoluteError:max,relativeL2FieldError:Math.sqrt(numerator/denominator),samplesWithApproximationWarnings:warned});
    assert(max<gold.thresholds.apertureRelativeFieldMaximumAbsolute,`Fresnel approximation ${boundary} error ${max}`);
  }
  report.checks.doubleSlitFresnel=results;
});

check('single-knife shadow transition has independent wavelength and distance Fresnel scaling',()=>{
  const ref=gold.knifeFresnelScaling,results=[];
  for(const boundary of ['soft','hard']){
    const s=scene(boundary,plate(0,ref.plateHalfWidth,ref.plateHalfHeight));let max=0,shadowMax=0,seamMax=0;
    for(const fixture of ref.cases){
      const out=evaluateMeshDiffraction(s,fixture.source,fixture.receiver,fixture),u=sum(out.routes),R=distance(fixture.source,fixture.receiver),A=1/(Math.sqrt(4*PI)*R);
      u[0]+=(out.directBoundary?.5:segmentVisible(s,fixture.source,fixture.receiver)?1:0)*A;
      const normalized=u.map(x=>x/A),error=Math.hypot(...normalized.map((x,i)=>x-fixture.relativeField[i]));
      max=Math.max(max,error);if(fixture.fresnelParameter>0)shadowMax=Math.max(shadowMax,error);
      if(fixture.fresnelParameter===0)seamMax=Math.max(seamMax,error);
    }
    assert(max<gold.thresholds.knifeRelativeFieldMaximumAbsolute,`Knife-edge ${boundary} error ${max}`);
    results.push({boundary,samples:ref.cases.length,maxNormalizedComplexAbsoluteError:max,shadowMaximumError:shadowMax,seamMaximumError:seamMax});
  }
  report.checks.knifeFresnelScaling=results;
});

report.checksCount=checks;report.allPass=true;
report.moduleSha256=crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/coherent-mesh-diffraction.js',import.meta.url))).digest('hex');
fs.writeFileSync(new URL('../research/coherent-mesh-diffraction-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
