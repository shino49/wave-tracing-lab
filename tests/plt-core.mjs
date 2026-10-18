import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import * as p from '../dist/plt-core.js';
import {preparePLTScene} from '../dist/plt-geometry.js';
import {prepareMesh,add,scale,normalize,dot} from '../dist/mesh-geometry.js';
import {evaluateOperator,wrappedNormalPDF,sampleFSD} from '../dist/plt-fsd.js';
import {totalPlane} from '../dist/utd-math.js';
const root=new URL('../',import.meta.url),began=performance.now();
const report={scope:p.PLT_SCOPE,createdUtc:new Date().toISOString(),checks:{},warnings:[],failures:[],sourceHashes:{},independence:'SciPy erfcx local coefficients, exact erfc Sommerfeld half-plane control, independent finite-area emitter quadrature, and geometric transport controls; no full-wave 3D or solid-angle-unbiased FSD claim'};
const sourceFiles=['dist/plt-core.js','dist/plt-fsd.js','dist/plt-geometry.js','dist/mesh-geometry.js','dist/utd-math.js','research/plt-independent-reference.py','tests/plt-core.mjs'];
const sha=f=>crypto.createHash('sha256').update(fs.readFileSync(new URL(f,root))).digest('hex');
for(const f of sourceFiles)report.sourceHashes[f]=sha(f);
const rel=(a,b)=>Math.sqrt(a.reduce((s,v,i)=>s+(v-b[i])**2,0)/Math.max(1e-300,b.reduce((s,v)=>s+v*v,0)));
const close=(a,b,t=1e-11)=>assert.ok(Math.abs(a-b)<=t*Math.max(1,Math.abs(a),Math.abs(b)),`${a} differs from ${b}`);
const check=async(name,f)=>{try{report.checks[name]=await f();console.log('PASS',name);}catch(e){report.failures.push({name,message:e.message});console.error('FAIL',name,e.stack);}};
const quad=(vertices,materialIndex=0)=>({mesh:{vertices,faces:[[0,1,2],[0,2,3]]},materialIndex,surface:'open-plate'});
function finish(entries,materials){const s=preparePLTScene(entries,{materials});s.emitters=s.triangles.filter(t=>materials[t.materialIndex].emission>0);s.emitterArea=s.emitters.reduce((a,t)=>a+t.area,0);return s;}
function emptyJob(samples=1){return {width:1,height:1,samplesPerPixel:samples,events:[],terms:[],paths:[],stats:{approximationWarnings:{},...Object.fromEntries(['diffractingEvaluations','twoDiffractingPaths','twoDiffractingContributions','maxDiffractingPerPath','traversals','escaped','contributions','fsdEvaluations','edgeCoefficients','endpointExclusions','invalidEdges','regionInteractions','surfaceInteractions','samplingRejected','nullRestarts','twoFSDPaths','twoFSDContributions','twoSurfaceContributions','maxFSDPerPath','maxSurfacePerPath','maxDepthReached'].map(k=>[k,0]))}};}
const beam=(origin,direction)=>({origin,direction:normalize(direction),radius0:0,tanAngle:0,near:1e-7,far:Infinity,k:2*Math.PI/.025,wavelengthMm:.025});
const baseOpt={width:8,height:8,samplesPerPixel:8,maxDepth:6,wavelengthMm:.025,wave:true,nee:true,boundary:'unpolarized',seed:7391};
let gallery,ray,mirrorScene,surfaceAddition;
await check('cameraProjectedPixelMeasure',()=>{
 const camera={position:[1,2,3],target:[1,2,2],up:[0,1,0],fov:60},rng=p.randomGenerator(713),n=12000;let mx=0,my=0,mxx=0;
 for(let i=0;i<n;i++){const b=p.sampleCamera(camera,3,4,8,8,rng,.025,true),f=p.cameraFrame(camera),z=dot(b.direction,f.forward),x=dot(b.direction,f.right)/z,y=dot(b.direction,f.up)/z;mx+=x;my+=y;mxx+=x*x;assert.equal(b.initialWeight,1);}
 const half=Math.tan(Math.PI/6),lo=-half/4,hi=0,ex=(lo+hi)/2,ey=ex,variance=(hi-lo)**2/12;
 assert.ok(Math.abs(mx/n-ex)<5*Math.sqrt(variance/n));assert.ok(Math.abs(my/n-ey)<5*Math.sqrt(variance/n));
 return {samples:n,meanProjectedX:mx/n,meanProjectedY:my/n,expected:ex,varianceProjectedX:mxx/n-(mx/n)**2,expectedVariance:variance,interpretation:'Uniform projected-pixel area with native virtual perspective importance/PDF cancellation. No extra cos^3 or physical aperture factor is inserted.'};
});
await check('uniformEmitterCameraRadiance',async()=>{
 const s=finish([quad([[-100,-100,0],[100,-100,0],[100,100,0],[-100,100,0]])],[{type:'diffuse',albedo:0,emission:7,twoSidedEmission:true}]),camera={position:[0,0,3],target:[0,0,0],up:[0,1,0],fov:95};
 const records=[];for(const wave of [false,true])for(const samplesPerPixel of [1,7]){const j=await p.buildPLTJob(s,camera,{...baseOpt,wave,samplesPerPixel,maxDepth:1});assert.ok(j.reference.every(v=>Math.abs(v-7)<1e-13));records.push({wave,samplesPerPixel,mean:j.measurement.mean,max:j.measurement.max});}
 return {records,resolutionOrOffAxisNormalization:false};
});
await check('cosineAreaMISAndSampleConvergence',()=>{
 const rho=.63,E=4,h=2,a=.6,s=finish([quad([[-10,-10,0],[10,-10,0],[10,10,0],[-10,10,0]],0),quad([[-a,-a,h],[-a,a,h],[a,a,h],[a,-a,h]],1)],[{type:'diffuse',albedo:rho,emission:0},{type:'diffuse',albedo:0,emission:E,twoSidedEmission:false}]);
 // A direct two-dimensional midpoint quadrature of rho E / pi * cos_s cos_l / r^2 dA.
 let expected=0;const q=400,dx=2*a/q;for(let i=0;i<q;i++)for(let j=0;j<q;j++){const x=-a+(i+.5)*dx,y=-a+(j+.5)*dx;expected+=rho*E/Math.PI*h*h/(h*h+x*x+y*y)**2*dx*dx;}
 const results=[];
 for(const nee of [false,true]){const n=65536,j=emptyJob(n),rng=p.randomGenerator(48373),b=beam([0,0,1],[0,0,-1]),vals=[];let sum=0,sq=0;
 for(let i=0;i<n;i++){const first=j.terms.length;p.traceCameraPath(s,b,{...baseOpt,wave:false,nee,maxDepth:2},rng,j);const v=j.terms.slice(first).reduce((z,t)=>z+t.weight,0);sum+=v;sq+=v*v;if([1024,4096,16384,65536].includes(i+1))vals.push({samples:i+1,mean:sum/(i+1),absoluteError:Math.abs(sum/(i+1)-expected)});}
 const mean=sum/n,se=Math.sqrt(Math.max(0,sq/n-mean*mean)/n);assert.ok(Math.abs(mean-expected)<6*se+1e-6,`MIS=${nee}, ${mean}, reference ${expected}, SE ${se}`);assert.ok(mean>0&&mean<rho*E);results.push({nee,mean,standardError:se,checkpoints:vals,relativeError:Math.abs(mean-expected)/expected});
 }
 assert.ok(results[1].standardError<results[0].standardError/4);
 return {expected,sourceRadiance:E,albedo:rho,squareHalfWidth:a,distance:h,area:4*a*a,results,normalization:'Area selection probability, distance²/light cosine Jacobian, Lambertian cosine/pi, BSDF continuation and power-heuristic MIS jointly checked against independent quadrature'};
});
await check('twoMirrorInteractions',()=>{
 const t=.45,plate=(c,u,mi)=>quad([add(add(c,scale(u,-t)),[0,0,-1]),add(add(c,scale(u,t)),[0,0,-1]),add(add(c,scale(u,t)),[0,0,1]),add(add(c,scale(u,-t)),[0,0,1])],mi);
 mirrorScene=finish([plate([0,-1,0],[1,1,0],0),plate([0,1,0],[1,-1,0],1),quad([[-2,.5,-1],[-2,1.5,-1],[-2,1.5,1],[-2,.5,1]],2)],[{type:'mirror',albedo:.7,emission:0},{type:'mirror',albedo:.8,emission:0},{type:'diffuse',albedo:0,emission:5,twoSidedEmission:true}]);
 const records=[];for(const wave of [false,true]){const j=emptyJob(),b={...beam([-2,-1,0],[1,0,0]),radius0:wave?.001:0};p.traceCameraPath(mirrorScene,b,{...baseOpt,wave,maxDepth:4},p.randomGenerator(1),j);const out=p.replayPLT(j)[0];close(out,5*.7*.8);assert.equal(j.stats.maxSurfacePerPath,2);records.push({wave,radiance:out,expected:2.8,stats:j.stats});}return records;
});
await check('nativePendingSurfaceAddition',()=>{
 const j=emptyJob(),b={...beam([-2,-1,0],[1,0,0]),radius0:.75};p.traceCameraPath(mirrorScene,b,{...baseOpt,nee:false,maxDepth:4},p.randomGenerator(1),j);
 assert.equal(j.events.length,2);assert.ok(j.events.every(e=>e.edges.length>0&&e.referenceF>0));const [f1,f2]=j.events.map(e=>e.referenceF),expected=5*(.7+f1)*(.8+f2),actual=p.replayPLT(j)[0];close(actual,expected,2e-13);
 assert.deepEqual(j.terms.map(t=>({weight:t.weight,eventIds:t.eventIds})),[{weight:2.8,eventIds:[]},{weight:4,eventIds:[0]},{weight:3.5,eventIds:[1]},{weight:5,eventIds:[0,1]}]);surfaceAddition=j.events;
 return {localPowers:[f1,f2],directAmplitudes:j.events.map(e=>e.directAmplitude),expected,actual,positiveTerms:j.terms,derivation:'Native non-sampled pending FSD adds previous pre-BSDF intensity. Two surfaces therefore yield E*(rho1+F1)*(rho2+F2), four positive branches exactly once, including F1*F2.'};
});
await check('organicCameraAndRepeatedFSD',async()=>{
 const s=p.createPLTScene();ray=await p.buildPLTJob(s,s.camera,{...baseOpt,wave:false});gallery=await p.buildPLTJob(s,s.camera,baseOpt);
 assert.ok(gallery.reference.every(x=>Number.isFinite(x)&&x>=0));assert.ok(gallery.stats.twoFSDPaths>0);assert.ok(gallery.stats.twoFSDContributions>0);const repeated=gallery.terms.filter(t=>t.eventIds.filter(i=>gallery.events[i].edges.length>0&&gallery.events[i].referenceF>0).length>=2);assert.ok(repeated.length>0,'At least two actual nonempty UTD coefficients must contribute to a camera pixel');assert.ok(rel(gallery.reference,ray.reference)>.001);assert.equal(gallery.sceneSummary.triangles,ray.sceneSummary.triangles);assert.equal(gallery.sceneSummary.wedges,ray.sceneSummary.wedges);
 const explicit=Array(gallery.reference.length).fill(0);for(const t of gallery.terms){let value=t.weight;for(const i of t.eventIds)value*=gallery.events[i].referenceF;explicit[t.pixel]+=value/gallery.samplesPerPixel;}assert.ok(rel(explicit,gallery.reference)<1e-15);assert.deepEqual(Array.from(p.replayPLT(gallery,gallery.events.map(e=>e.referenceF))),Array.from(gallery.reference));
 const zero=p.replayPLT(gallery,gallery.events.map(()=>0));assert.ok(zero.every((v,i)=>v<=gallery.reference[i]+1e-12));assert.ok(rel(zero,gallery.reference)>.001);
 return {wave:gallery.measurement,ray:ray.measurement,stats:gallery.stats,positiveTapeTerms:gallery.terms.length,nonemptyRepeatedCoefficientTerms:repeated.length,exampleTerms:repeated.slice(0,3).map(t=>({pixel:t.pixel,weight:t.weight,eventIds:t.eventIds,edgeIds:t.eventIds.map(i=>gallery.events[i].edges.map(e=>e.edgeId)),power:t.eventIds.reduce((v,i)=>v*gallery.events[i].referenceF,t.weight)/gallery.samplesPerPixel})),scene:gallery.sceneSummary,replayExact:true,zeroOperatorNegativeControlChangesImage:true};
});
await check('nativeLocalOperatorAndSommerfeld',()=>{
 assert.ok(gallery);const operators=[...gallery.events.filter(e=>e.edges.length>0).slice(0,160),...surfaceAddition];assert.ok(operators.length>20);
 const cases=[];for(const boundary of ['soft','hard'])for(const phiI of [.37,1.13])for(const phiO of [.5,1.8,2.7,3.4,4.8,5.8])cases.push({k:41,r:3.7,phiI,phiO,boundary});
 const r=spawnSync('python',['research/plt-independent-reference.py'],{cwd:root,input:JSON.stringify({operators,sommerfeld:cases}),encoding:'utf8'});assert.equal(r.status,0,r.stderr);const ref=JSON.parse(r.stdout);
 let maxPowerError=0,maxComplexError=0;for(let i=0;i<operators.length;i++){const a=evaluateOperator(operators[i]),b=ref.operators[i];maxPowerError=Math.max(maxPowerError,Math.abs(a.F-b.F)/Math.max(1,a.F,b.F));for(const bc of ['soft','hard'])maxComplexError=Math.max(maxComplexError,Math.hypot(a[bc][0]-b[bc][0],a[bc][1]-b[bc][1]));}assert.ok(maxPowerError<2e-10);assert.ok(maxComplexError<2e-10);
 let sommerfeldError=0;for(let i=0;i<cases.length;i++){const c=cases[i],z=totalPlane(c.k,c.r,2,c.phiI,c.phiO,c.boundary);sommerfeldError=Math.max(sommerfeldError,Math.hypot(z[0]-ref.sommerfeld[i][0],z[1]-ref.sommerfeld[i][1]));}assert.ok(sommerfeldError<2e-9);
 return {operatorCount:operators.length,maxRelativePowerError:maxPowerError,maxComplexAmplitudeError:maxComplexError,sommerfeldCases:cases.length,maxSommerfeldComplexError:sommerfeldError,qualification:'Tests native local-plane center-connected factors only. Exact Sommerfeld agreement is limited to the canonical plane-wave half-plane, not finite-source or arbitrary 3-D multiple scattering.'};
});
await check('sourceScalingAndArbitraryOBJ',async()=>{
 const s2=p.createPLTScene({emission:10}),j2=await p.buildPLTJob(s2,s2.camera,baseOpt);assert.ok(rel(j2.reference,Array.from(gallery.reference,v=>v*2))<2e-14);
 const obj='v -1 -1 -1\nv 1 -1 -1\nv 0 1.8 -1\nv 0 0 1.5\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n';const records=[];
 for(const options of [{objText:obj},{objText:obj,modelRotation:[21,39,7],modelTranslation:[1.1,.3,-1.2],modelScale:1.25}]){const s=p.createPLTScene(options),j=await p.buildPLTJob(s,s.camera,{...baseOpt,wave:false,samplesPerPixel:32});const jw=await p.buildPLTJob(s,s.camera,{...baseOpt,samplesPerPixel:8});records.push({image:Array.from(j.reference),waveImage:Array.from(jw.reference),triangles:s.triangles.length,wedges:s.wedges.length});}
 assert.equal(records[0].triangles,10);assert.ok(rel(records[0].image,records[1].image)>.01);assert.ok(rel(records[0].waveImage,records[1].waveImage)>.01);const movedCamera=p.orbitCamera(s2.camera,20,4,1),jc=await p.buildPLTJob(s2,movedCamera,{...baseOpt,wave:false});assert.ok(rel(jc.reference,Array.from(ray.reference,v=>v*2))>.01);
 return {emissionScaleError:rel(j2.reference,Array.from(gallery.reference,v=>v*2)),importedOBJTriangleCount:records[0].triangles,importedOBJWedgeCount:records[0].wedges,modelTransformRelativeImageChange:rel(records[0].image,records[1].image),waveModelTransformRelativeImageChange:rel(records[0].waveImage,records[1].waveImage),cameraMoveRelativeImageChange:rel(jc.reference,Array.from(ray.reference,v=>v*2)),sameSceneGeometryUsedForRayAndWave:true};
});
await check('rigidSceneCameraCovariance',async()=>{
 // Diffuse continuation uses a world-axis basis: common-random-number equality is
 // not appropriate there. Mirrors isolate deterministic covariant continuation.
 const rotate=v=>[.36*v[0]-.48*v[1]+.8*v[2],.8*v[0]+.6*v[1],-.48*v[0]+.64*v[1]+.6*v[2]],move=v=>add(rotate(v),[2.1,-.7,3.2]);
 const transformed=finish(mirrorScene.meshes.map(({mesh,materialIndex,surface})=>({mesh:{vertices:mesh.vertices.map(move),faces:mesh.authoredFaces},materialIndex,surface})),mirrorScene.materials);
 const powers=[];for(const [s,b] of [[mirrorScene,beam([-2,-1,0],[1,0,0])],[transformed,beam(move([-2,-1,0]),rotate([1,0,0]))]]){const j=emptyJob();p.traceCameraPath(s,{...b,radius0:.001},{...baseOpt,maxDepth:4},p.randomGenerator(1),j);powers.push(p.replayPLT(j)[0]);}close(powers[0],powers[1]);
 const s=p.createPLTScene({preset:'gallery'}),camera={...s.camera};s.materials=s.materials.map(m=>({...m,albedo:0}));const tr=finish(s.meshes.map(({mesh,materialIndex,surface})=>({mesh:{vertices:mesh.vertices.map(move),faces:mesh.authoredFaces},materialIndex,surface})),s.materials),tc={...camera,position:move(camera.position),target:move(camera.target),up:rotate(camera.up)};
 const a=await p.buildPLTJob(s,camera,{...baseOpt,maxDepth:4,samplesPerPixel:8}),b=await p.buildPLTJob(tr,tc,{...baseOpt,maxDepth:4,samplesPerPixel:8});const error=rel(a.reference,b.reference);assert.ok(error<1e-7,`rigid transform error ${error}`);return {mirrorPowers:powers,cameraImageRelativeError:error,fsdEvents:[a.events.length,b.events.length],note:'All geometry, source and camera are rotated/translated together. Diffuse Monte Carlo basis covariance is statistical, rather than per-seed identical.'};
});
await check('wrappedAngularPDF',()=>{const output=[];for(const sigma of [.01,.2,1,4]){const n=32768,d=2*Math.PI/n;let integral=0;for(let i=0;i<n;i++)integral+=wrappedNormalPDF(-Math.PI+(i+.5)*d,sigma)*d;close(integral,1,2e-12);output.push({sigma,integral});}return {output,nativeAuxiliaryMeasure:'angle, not solid angle; no claimed auxiliary-edge length/Jacobian cancellation'};});
await check('nativeKellerAuxiliaryMeasure',()=>{
 // For a single open-plate edge, z=-t/sqrt(r²+t²), t uniform in [-a,a].
 // q_native(phi)=.5*q(phi|t), hence E[w*g]=g(direct)+(1/(2a))*integral(dt dphi g).
 // This has an exact analytic value for g=1+z². It is distinct from dOmega.
 const a=1,r=Math.SQRT2,edge={a:[0,0,-a],b:[0,0,a],v:[0,0,0],e:[0,0,1],l:2*a,nff:[0,1,0],nbf:[0,-1,0],tff:[-1,0,0],alpha:0,edgeId:0},aperture={edges:[edge],center:[0,0,0],k:1},src=[1,1,0],rng=p.randomGenerator(731997),n=131072;
 let sum=0,sq=0,direct=0;for(let i=0;i<n;i++){const s=sampleFSD(aperture,src,rng);assert.ok(s.valid);const v=s.weight*(1+s.wo[2]**2);sum+=v;sq+=v*v;direct+=Number(s.isDirect);}
 const expected=1+2*Math.PI*(2-r/a*Math.atan(a/r)),c=a/Math.sqrt(r*r+a*a),solidAngleValue=1+4*Math.PI*(c+c**3/3),mean=sum/n,se=Math.sqrt((sq/n-mean**2)/n);
 assert.ok(Math.abs(mean-expected)<6*se);assert.ok(Math.abs(mean-solidAngleValue)>20*se);
 return {samples:n,mean,standardError:se,independentAnalyticNativeMeasure:expected,distinctSolidAngleIntegral:solidAngleValue,directMixtureFraction:direct/n,solidAngleDensityJacobian:'q_Omega=q_angle*r/(edgeLength*sinBeta^3), for one nonoverlapping edge chart',conclusion:'Implemented inverse-PDF weight matches the native auxiliary-edge-position/angular estimator. It cannot also be described as an unbiased dOmega estimator without a change of measure.'};
});
await check('organicTwoScreenSourceAndConvergence',async()=>{
 const scene=p.createPLTScene({preset:'double-edge'}),camera={...scene.camera,position:[-2,.8,0],target:[8,1.8,0],fov:18},runs=[];let fine;
 for(const samplesPerPixel of [16,64,256]){const j=await p.buildPLTJob(scene,camera,{...baseOpt,samplesPerPixel,wavelengthMm:.025});const repeated=j.terms.filter(t=>{
   const groups=t.eventIds.filter(i=>j.events[i].referenceF>0&&j.events[i].edges.length).map(i=>new Set(j.events[i].edges.map(e=>scene.wedges.find(w=>w.edgeId===e.edgeId)?.meshId)));
   return groups.some((g,i)=>g.has(0)&&groups.slice(i+1).some(h=>h.has(1)));
 });assert.ok(repeated.length>0,'source-reaching contribution must traverse distinct first and second screen edges');
 const pure=repeated.filter(t=>t.eventIds.filter(i=>j.events[i].referenceF>0&&j.events[i].edges.length&&j.events[i].directAmplitude===0).length>=2);assert.ok(pure.length>0,'At least one positive double-shadow contribution must have two nonempty UTD operators with zero direct amplitude');
 const ddMean=repeated.reduce((sum,t)=>sum+t.eventIds.reduce((v,i)=>v*j.events[i].referenceF,t.weight),0)/(64*samplesPerPixel);
 runs.push({samplesPerPixel,rawMean:j.measurement.mean,rawMax:j.measurement.max,distinctScreenDDTerms:repeated.length,pureDoubleShadowTerms:pure.length,pureDoubleShadowMean:pure.reduce((sum,t)=>sum+t.eventIds.reduce((v,i)=>v*j.events[i].referenceF,t.weight),0)/(64*samplesPerPixel),ddRawMean:ddMean,example:pure.slice(0,1).map(t=>({pixel:t.pixel,eventIds:t.eventIds,operators:t.eventIds.map(i=>({F:j.events[i].referenceF,directAmplitude:j.events[i].directAmplitude,edges:j.events[i].edges.map(e=>({edgeId:e.edgeId,point:e.point}))}))}))});fine=j;
 }
 assert.ok(Math.abs(runs[2].rawMean-runs[1].rawMean)<.1*Math.abs(runs[2].rawMean));assert.ok(Math.abs(runs[2].ddRawMean-runs[1].ddRawMean)<.1*Math.abs(runs[2].ddRawMean));
 const geometric=await p.buildPLTJob(scene,camera,{...baseOpt,wave:false,samplesPerPixel:256});assert.ok(rel(fine.reference,geometric.reference)>.01);
 return {camera,wavelengthMm:.025,finite3DScene:scene.counts,runs,rawRayMean:geometric.measurement.mean,waveRayRelativeDifference:rel(fine.reference,geometric.reference),convergenceMeaning:'Fixed footprint/resolution and fixed estimator; prefix sample arithmetic means only. Mixed direct-plus-edge DD stabilizes here. Rare pure-double-shadow terms are coverage witnesses, not sample-convergence-certified measurements. This is not a 3D full-wave or global-energy validation.'};
});
await check('invalidInputsAndCancellation',async()=>{for(const x of [{width:0},{height:9.2},{samplesPerPixel:0},{maxDepth:99},{wavelengthMm:0},{boundary:'invented'}])assert.throws(()=>p.validatePLTOptions(x));const s=p.createPLTScene(),c=new AbortController();c.abort();await assert.rejects(()=>p.buildPLTJob(s,s.camera,baseOpt,{signal:c.signal}),{name:'AbortError'});return {validation:true,cancelledBeforeWork:true};});
for(const f of sourceFiles)if(sha(f)!==report.sourceHashes[f])report.failures.push({name:'sourceIntegrity',message:`${f} changed during validation; rerun on stable sources.`});
report.warnings.push('No claim of global coherent amplitudes across stochastic path events. Positive local intensity factors are multiplied, and paths are incoherently accumulated.','Native auxiliary Keller-angle FSD sampling has no established solid-angle-unbiased interpretation; source linearity does not certify global conservation.','Camera pixel footprint changes with image resolution. Sample-count convergence at fixed resolution is distinct from resolution/footprint convergence.');
report.elapsedMs=performance.now()-began;report.allPass=!report.failures.length;fs.writeFileSync(new URL('research/plt-core-validation.json',root),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({allPass:report.allPass,elapsedMs:report.elapsedMs,failures:report.failures},null,2));if(!report.allPass)process.exitCode=1;
