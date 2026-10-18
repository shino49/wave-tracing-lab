/** Independent scientific-scope/ablation review for the bounded 3-D demo.
 * Uses real native CPU paths. No physical GPU/browser or full-wave accuracy claim.
 * Optional --render-review (or --probe-dir=PATH) reviews actual 64x48 checkpoint renders and retraces
 * their hottest pixels, retaining exact per-sample/term contribution evidence.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {createPLTDemoScene,PLT_DEMO_DEFAULTS,PLT_DEMO_SCENE_ID} from '../dist/plt-demo-scene.js';
import {buildPLTJob,buildPLTBatch,createPLTRandomStates,replayPLT,sampleCamera,randomGenerator} from '../dist/plt-core.js';
import {evaluateOperator} from '../dist/plt-fsd.js';
const root=new URL('../',import.meta.url),checks=[],failures=[],began=performance.now();
const files=['dist/plt-demo-scene.js','dist/plt-core.js','dist/plt-fsd.js','dist/plt-geometry.js','dist/mesh-geometry.js','dist/utd-math.js','research/plt-independent-reference.py','tests/plt-demo-physics.mjs'];
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const sourceHashes=Object.fromEntries(files.map(f=>[f,sha(fs.readFileSync(new URL(f,root)))]));
const test=async(name,fn)=>{try{const evidence=await fn();checks.push({name,evidence});console.log('PASS',name);}catch(error){failures.push({name,error:error.stack});console.error('FAIL',name,error.stack);}};
const close=(a,b,tol=2e-12)=>assert(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} differs from ${b}`);
const closeArray=(a,b,tol)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
const difference=(a,b)=>Array.from(a,(v,i)=>v-b[i]);
const l1=(a,b)=>mean(difference(a,b).map(Math.abs));
const correlation=(a,b)=>{const ma=mean(a),mb=mean(b);let cross=0,aa=0,bb=0;for(let i=0;i<a.length;i++){cross+=(a[i]-ma)*(b[i]-mb);aa+=(a[i]-ma)**2;bb+=(b[i]-mb)**2;}return aa*bb>0?cross/Math.sqrt(aa*bb):null;};
const statsImage=(image,off)=>{const d=difference(image,off),rank=Array.from(image).sort((a,b)=>b-a);return {mean:mean(image),utdOffMean:mean(off),signedUTDDifference:mean(d),absoluteUTDDifference:mean(d.map(Math.abs)),max:rank[0],largestPixelShare:rank[0]/sum(image),largestFivePixelShare:sum(rank.slice(0,5))/sum(image),positiveDifferencePixels:d.filter(v=>v>1e-12).length,negativeDifferencePixels:d.filter(v=>v< -1e-12).length};};
const scene=createPLTDemoScene(),options={...PLT_DEMO_DEFAULTS.options,width:16,height:12,samplesPerPixel:16};
let job,off;
await test('fixture is finite 3-D geometry with physical material boundaries and explicit limited scope',()=>{
 assert.equal(scene.preset,PLT_DEMO_SCENE_ID);assert.equal(scene.triangles.length,204);assert.equal(scene.wedges.length,89);assert.equal(scene.unsupportedEdges.length,0);
 assert.equal(scene.wedges.filter(w=>w.meshId===0).length,4,'checkerboard seams must not become diffracting edges');
 assert(scene.triangles.some(t=>t.materialIndex===5));assert(scene.meshes.some(m=>m.mesh.isClosed));
 assert(scene.materials.every(m=>m.type==='diffuse'&&m.albedo>=0&&m.albedo<=1));
 assert.match(scene.demoMetadata.scope,/relative camera units/);assert.match(PLT_DEMO_DEFAULTS.scaleNote,/not visible light|红外/);
 assert.equal(scene.demoMetadata.baffleThickness,.4);assert.equal(scene.demoMetadata.apertureWidth,1.2);assert.equal(scene.demoMetadata.apertureHeight,2.9);
 return {triangles:scene.triangles.length,wedges:scene.wedges.length,floorOuterEdges:4,lengthUnit:'mm',wavelengthMm:PLT_DEMO_DEFAULTS.options.wavelengthMm,wavelengthMicrometres:1000*PLT_DEMO_DEFAULTS.options.wavelengthMm,prism:'opaque diffuse solid; no refraction or dispersion'};
});
await test('real native camera paths contain nonempty source-reaching UTD operators',async()=>{
 job=await buildPLTJob(scene,scene.camera,options,{yieldEvery:2000});assert(job.reference.every(v=>Number.isFinite(v)&&v>=0));
 const used=new Set(job.terms.flatMap(t=>t.eventIds));assert([...used].some(id=>job.events[id].edges.length>0&&job.events[id].referenceF>0));
 assert(job.stats.diffractingEvaluations>0);assert(job.stats.edgeCoefficients>0);
 return {options,measurement:job.measurement,stats:job.stats,usedEvents:used.size,usedNonemptyOperators:[...used].filter(id=>job.events[id].edges.length).length};
});
await test('same-path coefficient ablation retains geometry, RNG outcome, weights and direct seam convention',()=>{
 assert(job);const before=sha(JSON.stringify({terms:job.terms,events:job.events}));
 const powers=Float64Array.from(job.events,event=>evaluateOperator({...event,edges:[]}).F);
 job.events.forEach((event,i)=>close(powers[i],event.directAmplitude**2,1e-15));off=replayPLT(job,powers);
 const independent=new Float64Array(job.reference.length);for(const term of job.terms){let value=term.weight;for(const id of term.eventIds)value*=job.events[id].directAmplitude**2;independent[term.pixel]+=value/job.samplesPerPixel;}
 closeArray(off,independent,1e-15);assert.equal(sha(JSON.stringify({terms:job.terms,events:job.events})),before);assert(l1(job.reference,off)>1e-5);
 const allOperatorsZero=replayPLT(job,job.events.map(()=>0));assert(l1(off,allOperatorsZero)>1e-5,'F=0 removes direct local transfer too and is a distinct control');
 return {...statsImage(job.reference,off),meaning:'Same realized native path tape, with all local UTD complex coefficients zero. Not an unbiased no-diffraction estimator or the ray-rendered limit.',zeroAllOperatorsIsDifferent:true};
});
await test('touching-frame internal coefficient diagnostic is distinct from a welded-geometry control',()=>{
 assert(job);const approximately=(a,b)=>Math.abs(a-b)<1e-6,between=(x,a,b)=>x>=a-1e-6&&x<=b+1e-6;
 const wedges=new Map(scene.wedges.map(w=>[w.edgeId,w]));
 const internal=record=>{const wedge=wedges.get(record.edgeId);if(![4,5,6,7].includes(wedge?.meshId))return false;const [x,y]=record.point;
  const outer=(approximately(Math.abs(x),3.1)&&between(y,-1.5,3.3))||((approximately(y,-1.5)||approximately(y,3.3))&&between(x,-3.1,3.1));
  const aperture=(approximately(Math.abs(x),.6)&&between(y,-.4,2.5))||((approximately(y,-.4)||approximately(y,2.5))&&between(x,-.6,.6));return !outer&&!aperture;};
 const records=job.events.flatMap(event=>event.edges),internalRecords=records.filter(internal),filtered=replayPLT(job,job.events.map(event=>evaluateOperator({...event,edges:event.edges.filter(record=>!internal(record))}).F));
 assert(filtered.every(v=>Number.isFinite(v)&&v>=0));return {allRecords:records.length,internalRecords:internalRecords.length,absoluteImageChange:l1(filtered,job.reference),meaning:'Coefficient-only diagnostic on four touching authored boxes. Does not weld the geometry, change native sampling topology, or establish equivalence to a continuous aperture slab.'};
});
await test('k-only same-operator replay isolates coefficient and phase dependence without resampling',()=>{
 assert(job);const before=sha(JSON.stringify(job.events));const powers=job.events.map(event=>evaluateOperator({...event,k:event.k*2,edges:event.edges.map(edge=>({...edge,k:edge.k*2}))}).F);
 const changed=replayPLT(job,powers);assert(changed.every(v=>Number.isFinite(v)&&v>=0));assert(l1(changed,job.reference)>1e-7);assert.equal(sha(JSON.stringify(job.events)),before);
 for(const event of job.events)if(!event.edges.length)close(evaluateOperator({...event,k:event.k*2}).F,event.directAmplitude**2,1e-15);
 return {nominalWavelengthMm:options.wavelengthMm,replayedWavelengthMm:options.wavelengthMm/2,mean:mean(changed),absoluteMeanDifference:l1(changed,job.reference),meaning:'Operator sensitivity only. Geometry, sampled paths and proposal PDFs remain from the original wavelength; this is not a wavelength-rerendered estimator.'};
});
await test('independent SciPy local operators agree for source-reaching demo events',()=>{
 assert(job);const used=[...new Set(job.terms.flatMap(t=>t.eventIds))],operators=used.filter(i=>job.events[i].edges.length).slice(0,64).map(i=>job.events[i]);assert(operators.length>0);
 const r=spawnSync('python',['research/plt-independent-reference.py'],{cwd:root,input:JSON.stringify({operators}),encoding:'utf8'});assert.equal(r.status,0,r.stderr);const references=JSON.parse(r.stdout).operators;
 let maxRelativePowerError=0,maxComplexError=0;operators.forEach((o,i)=>{const got=evaluateOperator(o),want=references[i];maxRelativePowerError=Math.max(maxRelativePowerError,Math.abs(got.F-want.F)/Math.max(1,got.F,want.F));for(const channel of ['soft','hard'])maxComplexError=Math.max(maxComplexError,Math.hypot(got[channel][0]-want[channel][0],got[channel][1]-want[channel][1]));});
 assert(maxRelativePowerError<2e-10);assert(maxComplexError<2e-10);return {operators:operators.length,maxRelativePowerError,maxComplexError,scope:'Local native UTD arithmetic only; independent special-function implementation does not validate the global sampling measure.'};
});
await test('zero emission is black and source strength scales raw camera response linearly',async()=>{
 const control={...options,width:8,height:8,samplesPerPixel:8},baseline=await buildPLTJob(scene,scene.camera,control,{yieldEvery:2000});
 const dark=createPLTDemoScene({emission:0}),half=createPLTDemoScene({emission:4});
 assert.equal(dark.emitters.length,0);assert.equal(dark.emitterArea,0);
 const a=await buildPLTJob(dark,dark.camera,control,{yieldEvery:2000}),b=await buildPLTJob(half,half.camera,control,{yieldEvery:2000});
 assert(a.reference.every(v=>v===0));closeArray(b.reference,Array.from(baseline.reference,v=>v/2),2e-12);
 return {zeroMaximum:Math.max(...a.reference),sourceScale:.5,maximumScalingError:Math.max(...Array.from(b.reference,(v,i)=>Math.abs(v-baseline.reference[i]/2))),appliesTo:'both main and fill finite area emitters; raw response before display'};
});
await test('resolution changes native camera footprint and ray/wave is not a coefficient-only ablation',()=>{
 const a=sampleCamera(scene.camera,0,0,48,36,randomGenerator(17),.2,true),b=sampleCamera(scene.camera,0,0,96,72,randomGenerator(17),.2,true),ray=sampleCamera(scene.camera,0,0,48,36,randomGenerator(17),.2,false);
 close(a.radius0,2*b.radius0);close(a.tanAngle,2*b.tanAngle);assert.equal(ray.radius0,0);assert.equal(ray.tanAngle,0);
 return {at48:{radius0:a.radius0,tanAngle:a.tanAngle},at96:{radius0:b.radius0,tanAngle:b.tanAngle},consequence:'spp convergence must hold resolution fixed; image resolution is also a model/footprint change. Ray/wave additionally changes traversal and variable-length random streams.'};
});

const probeArg=process.argv.find(v=>v.startsWith('--probe-dir='));
const probeDirectory=probeArg?probeArg.slice('--probe-dir='.length):process.argv.includes('--render-review')?new URL('../research/plt-demo-v26/',import.meta.url).pathname:null;
let renderReview=null;
if(probeDirectory)await test('actual 64x48 checkpoint evidence and exact selected-pixel contribution distributions',async()=>{
 const directory=probeDirectory,runs=[];const raw=[];
 for(const seed of [7391,8319])for(const spp of [32,64,128]){
  const filename=`quality64-seed${seed}-wave-${spp}spp.json`,bytes=fs.readFileSync(path.join(directory,filename)),r=JSON.parse(bytes);
  assert.equal(r.options.width,64);assert.equal(r.options.height,48);assert.equal(r.options.samplesPerPixel,spp);assert.equal(r.options.seed,seed);assert.equal(r.options.maxDepth,2);assert.equal(r.options.wavelengthMm,.2);assert.deepEqual(r.camera,scene.camera);
  assert(r.image.length===3072&&r.utdOffImage.length===3072&&r.image.every(v=>Number.isFinite(v)&&v>=0)&&r.utdOffImage.every(v=>Number.isFinite(v)&&v>=0));
  runs.push({filename,sha256:sha(bytes),options:r.options,statistics:statsImage(r.image,r.utdOffImage),warnings:r.stats.approximationWarnings,fsdEvaluations:r.stats.fsdEvaluations,edgeCoefficients:r.stats.edgeCoefficients,elapsedMs:r.elapsedMs});raw.push(r);
 }
 const pair=raw.filter(r=>r.options.samplesPerPixel===128),d=pair.map(r=>difference(r.image,r.utdOffImage));
 const seedComparison={signedDifferences:d.map(mean),differenceCorrelation:correlation(d[0],d[1]),fullImageSeedAbsoluteDifference:l1(pair[0].image,pair[1].image),differenceImageSeedAbsoluteDifference:l1(d[0],d[1])};
 const distributions=[];
 for(const r of pair){
  const rank=(a)=>Array.from(a,(v,pixel)=>({v:Math.abs(v),pixel})).sort((a,b)=>b.v-a.v).slice(0,6).map(p=>p.pixel);
  const selected=[...new Set([...rank(r.image),...rank(difference(r.image,r.utdOffImage))])],rngStates=createPLTRandomStates(r.options);
  for(const pixel of selected){
   let total=0,offTotal=0,largestSample=0,largestTerm=0,largestDirectOnlyTerm=0,pureShadow=0;const topSamples=[];
   for(let sampleStart=0;sampleStart<128;sampleStart+=4){
    await buildPLTBatch(scene,scene.camera,r.options,{tile:{x:pixel%64,y:Math.floor(pixel/64),width:1,height:1},sampleStart,samplesPerPixel:4,rngStates,yieldEvery:1000,audit:{beginPath({sample,job:batch}){const first=batch.terms.length;return {onEnd(){let value=0,direct=0;for(const term of batch.terms.slice(first)){const v=term.eventIds.reduce((a,id)=>a*batch.events[id].referenceF,term.weight),off=term.eventIds.reduce((a,id)=>a*batch.events[id].directAmplitude**2,term.weight);value+=v;direct+=off;largestTerm=Math.max(largestTerm,v);if(term.eventIds.length&&term.eventIds.every(id=>batch.events[id].edges.length===0))largestDirectOnlyTerm=Math.max(largestDirectOnlyTerm,v);if(term.eventIds.some(id=>batch.events[id].directAmplitude===0&&batch.events[id].edges.length>0))pureShadow+=v;}total+=value;offTotal+=direct;largestSample=Math.max(largestSample,value);topSamples.push({sample,response:value,directOnly:direct});}};}}});
   }
   close(total/128,r.image[pixel],2e-10);close(offTotal/128,r.utdOffImage[pixel],2e-10);
   distributions.push({seed:r.options.seed,pixel,x:pixel%64,y:Math.floor(pixel/64),full:total/128,off:offTotal/128,pureShadowResponse:pureShadow/128,largestSample,largestSampleImageContribution:largestSample/128,largestSampleFractionOfPixel:total>0?largestSample/total:0,largestSampleFractionOfWholeFrame:largestSample/128/sum(r.image),largestTermImageContribution:largestTerm/128,largestNoEdgeOperatorTermImageContribution:largestDirectOnlyTerm/128,topSamples:topSamples.sort((a,b)=>b.response-a.response).slice(0,5)});
  }
 }
 renderReview={runs,seedComparison,selectedPixelDistributions:distributions,scope:'Two seeds and fixed-resolution sample prefixes are descriptive smoke evidence, not formal convergence or a full-wave accuracy certificate.',selectedPixelRule:'Union of the six brightest pixels and six largest absolute UTD-change pixels per 128-spp seed; exact full-frame camera/RNG retrace.'};
 fs.writeFileSync(new URL('../research/plt-demo-physics-render-review.json',import.meta.url),JSON.stringify({generatedAt:new Date().toISOString(),sourceHashes,...renderReview},null,2)+'\n');
 return {runs:runs.length,seeds:[7391,8319],prefixes:[32,64,128],selectedPixels:distributions.length,seedComparison,maximumSingleSampleFractionOfWholeFrame:Math.max(...distributions.map(d=>d.largestSampleFractionOfWholeFrame))};
});
await test('review source provenance stays unchanged during this run',()=>{for(const f of files)assert.equal(sha(fs.readFileSync(new URL(f,root))),sourceHashes[f],`${f} changed during review`);return sourceHashes;});
const report={suite:'plt-demo-physics',generatedAt:new Date().toISOString(),elapsedMs:performance.now()-began,passed:checks.length,failed:failures.length,sourceHashes,checks,failures,renderReviewFile:renderReview?'research/plt-demo-physics-render-review.json':null,browserValidated:false,GPUExecuted:false,physicalAccuracyClaim:false,interpretation:'Actual 3-D native PLT aperture/edge response. Local coherent UTD operator with incoherently accumulated positive paths; relative camera response; non-solid-angle-normalized native angular proposal; finite depth and declared asymptotic warnings.'};
fs.writeFileSync(new URL('../research/plt-demo-physics-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:checks.length,failed:failures.length,renderReview:!!renderReview,elapsedMs:report.elapsedMs}));if(failures.length)process.exitCode=1;
