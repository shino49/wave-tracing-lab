/** Independent visible complete-field validation. Production is read-only here.
 * Uses original official CIE CSVs, SI Planck law and independent dense numerical
 * integration. Template/runtime sections also compare the previous full-route
 * monochrome implementation; those are regression checks, not full-wave proof.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import * as spectrum from '../dist/coherent-mesh-spectrum.js';
import {prepareCoherentMeshScene,compileCoherentSource,sumCoherentRoutes,makeCoherentMeshPreset,detectorPoint} from '../dist/coherent-mesh-core.js';
import * as transport from '../dist/coherent-mesh-transport.js';
import * as runtime from '../dist/coherent-mesh-runtime.js';
import * as templates from '../dist/coherent-mesh-spectral-templates.js';
const root = new URL('../', import.meta.url);
const read = p => fs.readFileSync(new URL(p,root));
const sha = p => createHash('sha256').update(read(p)).digest('hex');
const report = {suite:'coherent-mesh-spectral-independent',generatedAt:new Date().toISOString(),GPUExecuted:false,browserValidated:false,fullWaveAccuracyClaim:false,checks:[],failures:[]};
async function check(name, fn) { const start=performance.now(); try { const evidence=await fn();report.checks.push({name,ms:performance.now()-start,evidence});console.log('PASS',name); } catch(error) {report.failures.push({name,error:error.stack});console.error('FAIL',name,error.stack);} }
const near=(a,b,rel=2e-10,abs=2e-13,label='comparison')=>assert(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=abs+rel*Math.max(Math.abs(a),Math.abs(b)),`${label}: ${a} != ${b}`);
const nearArray=(a,b,rel=2e-10,abs=2e-13)=>{assert.equal(a.length,b.length);a.forEach((x,i)=>near(x,b[i],rel,abs,`component ${i}`));};
const parse=p=>read(p).toString().trim().split(/\r?\n/).map(row=>row.split(',').map(Number));
const cie=parse('research/spectral-data/CIE_xyz_1931_2deg.csv'),d65=parse('research/spectral-data/CIE_std_illum_D65.csv');
function table(rows,nm){const a=Math.max(0,Math.min(rows.length-1,Math.floor(nm)-rows[0][0])),b=Math.min(rows.length-1,a+1),t=nm-rows[a][0];return rows[a].slice(1).map((x,i)=>x+t*(rows[b][i+1]-x));}
function planck(nm){const lambda=nm*1e-9,h=6.62607015e-34,c=299792458,k=1.380649e-23;return 2*h*c*c/(lambda**5*Math.expm1(h*c/(lambda*k*3000)));}
function density(d,nm){if(d.mode==='gaussian')return Math.exp(-4*Math.log(2)*((nm-d.centerNm)/d.fwhmNm)**2);return d.illuminant==='E'?1:d.illuminant==='D65'?table(d65,nm)[0]/100:planck(nm)/planck(560);}
function dense(d,transfer=()=>1,step=.05){const divisions=Math.ceil(470/step),h=470/divisions;let area=0,scalar=0;const xyz=[0,0,0];for(let i=0;i<=divisions;i++){const nm=360+i*h,p=density(d,nm)*(i===0||i===divisions?.5:1),v=transfer(nm),cmf=table(cie,nm);area+=p;scalar+=p*v;for(let c=0;c<3;c++)xyz[c]+=p*v*cmf[c];}return {scalar:scalar/area,xyz:xyz.map(x=>x/area)};}
const descriptors=[{mode:'broadband',illuminant:'E'},{mode:'broadband',illuminant:'D65'},{mode:'broadband',illuminant:'blackbody3000'},{mode:'gaussian',centerNm:548.3,fwhmNm:27.4}];
await check('official CIE files are checksum verified independently of runtime tables',()=>{assert.equal(sha('research/spectral-data/CIE_xyz_1931_2deg.csv'),'fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1');assert.equal(sha('research/spectral-data/CIE_std_illum_D65.csv'),'e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb');return {observerRows:cie.length,daylightRows:d65.length};});
await check('source CDF inversion is monotonic and independently jittered in every probability stratum',()=>{const results=[];for(const descriptor of descriptors){const p=spectrum.prepareCoherentSpectrum(descriptor,{spectralSamples:128,seed:451});let previous=-Infinity;for(let i=0;i<=2048;i++){const u=i/2048,nm=p.quantile(u);assert(nm>=previous&&nm>=360&&nm<=830);near(p.cdf(nm),u,0,4e-13);previous=nm;}const nodes=p.sampleNodes(31),jitter=nodes.map((n,i)=>{const u=p.cdf(n.wavelengthNm);assert(u>i/128&&u<(i+1)/128);return u*128-i;});assert(new Set(jitter.map(x=>x.toFixed(7))).size>120,'a single shifted grid is not independently jittered');near(nodes.reduce((s,n)=>s+n.weight,0),1);for(const n of nodes)nearArray(n.xyz,table(cie,n.wavelengthNm),2e-13,2e-15);assert.deepEqual(nodes,p.sampleNodes(31));assert.notDeepEqual(nodes,p.sampleNodes(32));assert.notDeepEqual(nodes,spectrum.prepareCoherentSpectrum(descriptor,{spectralSamples:128,seed:452}).sampleNodes(31));results.push({descriptor,maximumCDFRoundtripError:Math.max(...nodes.map(n=>Math.abs(p.quantile(p.cdf(n.wavelengthNm))-n.wavelengthNm)))});}return results;});
await check('neutral source has unit total spectral power and actual XYZ without CIE-Y renormalization',()=>{const result=[];for(const descriptor of descriptors){const expected=dense(descriptor),sum=[0,0,0],seeds=512;for(let seed=0;seed<seeds;seed++){const p=spectrum.prepareCoherentSpectrum(descriptor,{spectralSamples:128,seed});for(const n of p.sampleNodes(79))for(let c=0;c<3;c++)sum[c]+=n.weight*n.xyz[c]/seeds;}nearArray(sum,expected.xyz,0,.00028);if(descriptor.mode==='broadband')assert(expected.xyz[1]<.7,'equal total visible power must not have arbitrary Y=1');result.push({descriptor,expectedXYZ:expected.xyz,sampledEnsembleXYZ:sum});}return result;});
await check('spectral lines are exact, power-normalized, merged only at identical wavelength and mono-equivalent',()=>{const raw={mode:'lines',lines:[{wavelengthNm:650,powerFraction:2},{wavelengthNm:450,powerFraction:1},{wavelengthNm:650,powerFraction:3},{wavelengthNm:530,powerFraction:0}]};const valid=spectrum.validateCoherentSpectrum(raw);raw.lines[0].powerFraction=1e10;assert.deepEqual(valid.lines.map(n=>n.wavelengthNm),[450,650]);nearArray(valid.lines.map(n=>n.powerFraction),[1/6,5/6]);assert.deepEqual(spectrum.validateCoherentSpectrum(valid),valid);const p=spectrum.prepareCoherentSpectrum(valid,{spectralSamples:16,seed:1});assert(p.exact&&!p.stochastic);assert.equal(p.samples,2);assert.deepEqual(p.sampleNodes(1),p.sampleNodes(900));assert.deepEqual(p.sampleNodes(0),spectrum.prepareCoherentSpectrum(valid,{spectralSamples:128,seed:999}).sampleNodes(0));for(const nm of [360,435.8,550,830]){const mono=spectrum.prepareCoherentSpectrum({wavelengthMm:nm/1e6}),line=spectrum.prepareCoherentSpectrum({mode:'lines',lines:[{wavelengthNm:nm,powerFraction:Number.MAX_VALUE}]});assert.deepEqual(line.sampleNodes(4),mono.sampleNodes(9));}return {lines:valid.lines,qualityAndSeedInvariant:true};});
await check('Gaussian edge truncation and narrow-band limit retain total power and finite color',()=>{const evidence=[];for(const centerNm of [360,517.3,830])for(const fwhmNm of [1e-6,.001,.1]){const p=spectrum.prepareCoherentSpectrum({mode:'gaussian',centerNm,fwhmNm},{spectralSamples:128,seed:779}),nodes=p.sampleNodes(44),average=[0,0,0];for(const n of nodes){assert(n.wavelengthNm>=360&&n.wavelengthNm<=830);for(let c=0;c<3;c++)average[c]+=n.xyzWeight[c];}near(nodes.reduce((a,n)=>a+n.weight,0),1);if(fwhmNm<=1e-6)nearArray(average,table(cie,centerNm),0,3e-8);evidence.push({centerNm,fwhmNm,XYZ:average});}return evidence;});
await check('invalid source descriptors, seeds, sample indices and quality fail closed',()=>{for(const value of [{mode:'rgb'},{mode:'broadband',illuminant:'A'},{mode:'broadband',illuminant:'D65',unknown:true},{mode:'gaussian',centerNm:550,fwhmNm:0},{mode:'gaussian',centerNm:550,fwhmNm:1e-100},{mode:'lines',lines:[]},{mode:'lines',lines:[{wavelengthNm:550,powerFraction:0}]},{mode:'lines',lines:[{wavelengthNm:550,powerFraction:Infinity}]},{mode:'lines',lines:[{wavelengthNm:550,powerFraction:-1}]}])assert.throws(()=>spectrum.prepareCoherentSpectrum(value));for(const seed of [-1,.5,2**32,NaN,Infinity,'1'])assert.throws(()=>spectrum.prepareCoherentSpectrum(descriptors[0],{seed}));for(const spectralSamples of [0,1,15,17,256,NaN,'64'])assert.throws(()=>spectrum.prepareCoherentSpectrum(descriptors[0],{spectralSamples}));for(const index of [-1,.5,2**32,NaN,Infinity,'1',null])assert.throws(()=>spectrum.prepareCoherentSpectrum(descriptors[0]).sampleNodes(index));});
await check('ensemble interference converges against dense intensity-first quadrature, including long OPD',()=>{const evidence=[],seeds=192;for(const descriptor of descriptors.slice(0,3))for(const opd of [.0187,.2713]){const transfer=nm=>1+.96*Math.cos(2*Math.PI*opd*1e6/nm+.73),oracle=dense(descriptor,transfer,.0025),coarse=dense(descriptor,transfer,.005);nearArray(oracle.xyz,coarse.xyz,0,2e-7);const rows=[];for(const count of [16,32,64,128]){let sum=0,square=0;for(let seed=0;seed<seeds;seed++){const p=spectrum.prepareCoherentSpectrum(descriptor,{spectralSamples:count,seed}),estimate=p.sampleNodes(117).reduce((s,n)=>s+n.weight*n.xyz[1]*transfer(n.wavelengthNm),0),error=estimate-oracle.xyz[1];sum+=error;square+=error*error;}rows.push({count,meanError:sum/seeds,RMSE:Math.sqrt(square/seeds)});}assert(rows.at(-1).RMSE<rows[0].RMSE*.65,`128-sample ensemble failed to improve on 16: ${JSON.stringify(rows)}`);assert(Math.abs(rows.at(-1).meanError)<rows.at(-1).RMSE*.3+2e-5);evidence.push({descriptor,opticalPathDifferenceMm:opd,referenceXYZ:oracle.xyz,seedEnsemble:rows});}return evidence;});


await check('every wavelength recomputes complete direct/R/RR/D/RD/DR fields and GO seam phases',()=>{
  let compared=0,templateCount=0,maxAbsoluteComplexError=0;const families=new Set();
  const scenes=[];
  for(const preset of ['lloyd','two-mirror','double-slit','reflected-edge']){
    const p=runtime.prepareCoherentMeshTask({preset,power:2.7,phaseShiftRad:.37});
    scenes.push({...p,source:{...p.source,phaseRad:.83},receivers:Array.from({length:15},(_,i)=>detectorPoint(p.detector,(i*7+3)%17,(i*11+5)%17,17,17))});
  }
  const q=(vertices,materialIndex)=>({mesh:{vertices,faces:[[0,1,2],[0,2,3]]},materialIndex,surface:'open-plate'});
  const entries=[q([[0,-100,0],[100,-100,0],[100,100,0],[0,100,0]],0),q([[-10,-100,-100],[-10,100,-100],[-10,100,100],[-10,-100,100]],1)];
  for(const phaseRad of [0,Math.PI]){
    const scene=prepareCoherentMeshScene(entries,[{type:'mirror',reflectance:1,phaseRad},{type:'mirror',reflectance:.64,phaseRad:.37}],{maxReflections:2});
    for(const reverse of [false,true])for(const z of [-10,10])for(const delta of [-1e-9,0,1e-9]){
      const src=[-3,0,-4],dst=[42.5+delta,0,z];
      scenes.push({scene,source:{position:reverse?dst:src,wavelengthMm:.00055,power:2.7,phaseRad:.83},receivers:[reverse?src:dst],configuration:{diffraction:true,mixedDiffraction:true}});
    }
  }
  for(const fixture of scenes){const compiled=templates.compileCoherentSpectralSource(fixture.scene,fixture.source);
    for(const receiver of fixture.receivers){const prepared=templates.prepareCoherentSpectralSample(fixture.scene,compiled,receiver,fixture.configuration);templateCount+=prepared.templates.length;
      for(const nm of [360,401.37,450,550,650,830]){
        const expected=transport.evaluateMeshTransport(fixture.scene,compileCoherentSource(fixture.scene,{...fixture.source,wavelengthMm:nm/1e6}),receiver,fixture.configuration);
        assert.equal(prepared.templates.length,expected.routes.length);
        const evaluated=prepared.templates.map(t=>templates.evaluateCoherentSpectralTemplate(t,nm));
        for(let i=0;i<evaluated.length;i++){const a=evaluated[i],b=expected.routes[i],real=b.amplitude*Math.cos(b.phaseRadReduced),imag=b.amplitude*Math.sin(b.phaseRadReduced);
          near(a.amplitude,b.amplitude,5e-11,2e-14);nearArray([a.real,a.imag],[real,imag],5e-8,2e-12);maxAbsoluteComplexError=Math.max(maxAbsoluteComplexError,Math.hypot(a.real-real,a.imag-imag));families.add(b.kind==='reflection'?`R${b.sequence.length}`:b.interactionOrder??b.kind);compared++;
        }
        const actual=templates.evaluateCoherentSpectralSampleCPU(prepared.templates,[{wavelengthNm:nm,weight:1,xyz:table(cie,nm)}]),reference=sumCoherentRoutes(expected.routes);
        near(actual.coherent,reference.coherent,3e-8,5e-14);near(actual.incoherent,reference.incoherent,3e-12,2e-14);near(actual.geometric,expected.geometricPower,3e-12,2e-14);
      }
    }
  }
  for(const family of ['direct','R1','R2','diffraction','RD','DR'])assert(families.has(family),`fixture omitted ${family}`);
  return {geometricTemplates:templateCount,perWavelengthRoutesCompared:compared,routeFamilies:[...families],maxAbsoluteComplexError};
});
await check('Lloyd geometry matches independent image-source intensity and source power/phase normalization',()=>{
  const data=makeCoherentMeshPreset('lloyd',{phaseShiftRad:.37}),scene=prepareCoherentMeshScene(data.entries,data.materials),src={position:data.source.position,wavelengthMm:.00055,power:2.7,phaseRad:.83};let compared=0;
  for(const receiver of [[0,.11,50],[1,3.05,50],[-1,5.89,50]]){
    const rd=Math.hypot(...receiver.map((x,i)=>x-src.position[i])),image=[src.position[0],-src.position[1],src.position[2]],rr=Math.hypot(...receiver.map((x,i)=>x-image[i])),excess=4*src.position[1]*receiver[1]/(rr+rd),a=Math.sqrt(src.power/(4*Math.PI))/rd,b=Math.sqrt(src.power/(4*Math.PI))/rr;
    const prepared=templates.prepareCoherentSpectralSample(scene,templates.compileCoherentSpectralSource(scene,src),receiver);
    assert.equal(prepared.templates.length,2);
    for(const nm of [360,435.8,550,632.8,830]){
      const expected=a*a+b*b+2*a*b*Math.cos(Math.PI+.37-2*Math.PI*excess/(nm/1e6));
      const result=templates.evaluateCoherentSpectralSampleCPU(prepared.templates,[{wavelengthNm:nm,weight:1,xyz:table(cie,nm)}]);near(result.coherent,expected,2e-10,1e-16);
      const shifted=templates.prepareCoherentSpectralSample(scene,templates.compileCoherentSpectralSource(scene,{...src,power:src.power*4,phaseRad:-1.37}),receiver);
      const scaled=templates.evaluateCoherentSpectralSampleCPU(shifted.templates,[{wavelengthNm:nm,weight:1,xyz:table(cie,nm)}]);near(scaled.coherent,4*expected,2e-10,1e-16);compared++;
    }
  }
  return {comparisons:compared,normalization:'A=sqrt(P/(4pi)); point-source intensity P/(4pi r²); source phase global'};
});
await check('spectral integration sums intensities and XYZ after complete route interference',()=>{
  const routes=[{kind:1,amplitudeScale:1,interactionPhaseRad:0,opticalExcessMm:0},{kind:1,amplitudeScale:.7,interactionPhaseRad:.23,opticalExcessMm:.0057}],plan=spectrum.prepareCoherentSpectrum({mode:'lines',lines:[{wavelengthNm:430,powerFraction:.37},{wavelengthNm:690,powerFraction:.63}]}),nodes=plan.sampleNodes(0),expected={coherent:0,incoherent:1.49,geometric:1.49,coherentXYZ:[0,0,0],incoherentXYZ:[0,0,0],geometricXYZ:[0,0,0]};let wrongRe=0,wrongIm=0;
  for(const n of nodes){const phase=.23-2*Math.PI*.0057/(n.wavelengthNm/1e6),re=1+.7*Math.cos(phase),im=.7*Math.sin(phase),intensity=re*re+im*im;expected.coherent+=n.weight*intensity;for(let c=0;c<3;c++){expected.coherentXYZ[c]+=n.weight*intensity*n.xyz[c];expected.incoherentXYZ[c]+=n.weight*1.49*n.xyz[c];expected.geometricXYZ[c]+=n.weight*1.49*n.xyz[c];}wrongRe+=Math.sqrt(n.weight)*re;wrongIm+=Math.sqrt(n.weight)*im;}
  const actual=templates.evaluateCoherentSpectralSampleCPU(routes,nodes);for(const k of ['coherent','incoherent','geometric'])near(actual[k],expected[k]);for(const k of ['coherentXYZ','incoherentXYZ','geometricXYZ'])nearArray(actual[k],expected[k]);assert(Math.abs(actual.coherent-(wrongRe*wrongRe+wrongIm*wrongIm))>.1);return {correctIntensity:actual.coherent,incorrectCrossWavelengthFieldSquare:wrongRe*wrongRe+wrongIm*wrongIm,XYZ:actual.coherentXYZ};
});

const imageNames=['coherent','incoherent','geometric','crossTerm','coherentXYZ','incoherentXYZ','geometricXYZ'];
const routeCounts=['validRoutes','directRoutes','reflectedRoutes','diffractedRoutes','mixedRoutes','reflectionDiffractionRoutes','diffractionReflectionRoutes','boundaryRoutes','orthogonalCornerImages','maxRoutesAtSample','samplesWithModelWarnings'];
await check('runtime exact line mixture equals weighted monochrome renders in every scalar and XYZ channel',async()=>{
 const evidence=[];
 for(const preset of ['lloyd','double-slit','reflected-edge']){
  const options={preset,resolution:16,pixelQuadrature:1,backend:'cpu',power:2.7,coherence:.41};
  const lines=[{wavelengthNm:435.8,powerFraction:.3},{wavelengthNm:632.8,powerFraction:.7}],actual=await runtime.executeCoherentMeshTask({...options,spectrum:{mode:'lines',lines}});
  const monos=await Promise.all(lines.map(line=>runtime.executeCoherentMeshTask({...options,wavelengthNm:line.wavelengthNm})));
  assert(actual.spectralComplete&&actual.spectralSamples===2&&actual.complete);assert.equal(actual.diagnostics.productionCPUShadow,false);
  for(const key of imageNames){const expected=Float64Array.from(monos[0][key],(v,i)=>.3*v+.7*monos[1][key][i]);nearArray(actual[key],expected,1e-7,1e-13);}
  for(const p of [0,19,157,255])near(actual.crossTerm[p],actual.coherent[p]-actual.incoherent[p],2e-12,1e-16);
  const xChromaticity=Array.from({length:256},(_,i)=>{const xyz=actual.coherentXYZ.slice(3*i,3*i+3);return xyz[0]/Math.max(1e-30,xyz[0]+xyz[1]+xyz[2]);});const chromaticitySpan=Math.max(...xChromaticity)-Math.min(...xChromaticity);assert(chromaticitySpan>.01,`actual wavelength-dependent color collapsed to a fixed tint: ${chromaticitySpan}`);
  evidence.push({preset,lines,scalarPixelsCompared:256*4,XYZComponentsCompared:256*9,xChromaticitySpan:chromaticitySpan});
 }
 return evidence;
});
await check('runtime broadband pixels equal independent per-wavelength full-route sums and share no frequency cross terms',async()=>{
 const evidence=[];
 for(const preset of ['two-mirror','double-slit','reflected-edge']){
  const options={preset,resolution:16,pixelQuadrature:2,backend:'cpu',power:1.7,coherence:.64,spectrum:{mode:'broadband',illuminant:'D65'},spectralSamples:16,spectralSeed:2749},actual=await runtime.executeCoherentMeshTask(options),p=runtime.prepareCoherentMeshTask(options),plan=spectrum.prepareCoherentSpectrum(p.source,{spectralSamples:16,seed:2749});let maxDifference=0;
  for(const pixel of [0,19,75,111,157,255]){
   const expected={coherent:0,incoherent:0,geometric:0,coherentXYZ:[0,0,0],incoherentXYZ:[0,0,0],geometricXYZ:[0,0,0]};
   for(let sy=0;sy<2;sy++)for(let sx=0;sx<2;sx++){
    const receiver=detectorPoint(p.detector,pixel%16,Math.floor(pixel/16),16,16,(sx+.5)/2,(sy+.5)/2),nodes=plan.sampleNodes((pixel*2+sy)*2+sx);
    for(const node of nodes){const field=transport.evaluateMeshTransport(p.scene,compileCoherentSource(p.scene,{...p.source,wavelengthMm:node.wavelengthNm/1e6}),receiver,p.configuration),sum=sumCoherentRoutes(field.routes,{coherence:.64}),values={coherent:sum.coherent,incoherent:sum.incoherent,geometric:field.geometricPower};for(const [name,v]of Object.entries(values)){expected[name]+=v*node.weight/4;for(let c=0;c<3;c++)expected[name+'XYZ'][c]+=v*node.weight*table(cie,node.wavelengthNm)[c]/4;}}
   }
   for(const key of ['coherent','incoherent','geometric']){near(actual[key][pixel],expected[key],1e-7,1e-13);for(let c=0;c<3;c++){const a=actual[key+'XYZ'][pixel*3+c];near(a,expected[key+'XYZ'][c],1e-7,1e-13);maxDifference=Math.max(maxDifference,Math.abs(a-expected[key+'XYZ'][c]));}}
  }
  evidence.push({preset,independentFullRouteSpatialWavelengths:6*4*16,maxAbsoluteXYZDifference:maxDifference});
 }
 return evidence;
});
let canonical,baseOptions={preset:'reflected-edge',resolution:16,pixelQuadrature:1,backend:'cpu',spectrum:{mode:'broadband',illuminant:'D65'},spectralSamples:16,spectralSeed:451,power:2.7,coherence:.63};
await check('previews contain whole spectra, copy every XYZ channel and preserve canonical source snapshots',async()=>{
 const previews=[],raw=structuredClone(baseOptions);canonical=await runtime.executeCoherentMeshTask(raw,{progress(p){if(p.phase==='setup'){raw.spectrum.illuminant='E';raw.spectralSeed=12345;}},preview(p){assert(p.spectralComplete&&p.spectralSamples===16&&!p.complete);for(const key of imageNames){assert.equal(p[key].length,p.rowCount*p.width*(key.endsWith('XYZ')?3:1));assert(p[key].every(Number.isFinite));}previews.push({frame:p,copies:Object.fromEntries(imageNames.map(k=>[k,p[k].slice()]))});}});
 assert(previews.length>=2);assert.equal(canonical.configuration.spectrum.illuminant,'D65');assert.equal(canonical.configuration.spectralSeed,451);
 for(const {frame,copies}of previews)for(const key of imageNames){assert.deepEqual(frame[key],copies[key]);const channels=key.endsWith('XYZ')?3:1;assert.deepEqual(frame[key],canonical[key].slice(channels*frame.rowStart*16,channels*(frame.rowStart+frame.rowCount)*16));}
 const fresh=await runtime.executeCoherentMeshTask(baseOptions);for(const key of imageNames)assert.deepEqual(fresh[key],canonical[key]);return {previews:previews.length,scalarAndXYZChannelSnapshots:7,repeatDeterministic:true};
});
const synthetic=(sampleCount)=>Object.fromEntries(['coherent','incoherent','geometric','coherentXYZ','incoherentXYZ','geometricXYZ'].map((key,i)=>[key,new Float32Array(sampleCount*(key.endsWith('XYZ')?3:1)).fill(i+1)]));
await check('GPU loss after committed spectral rows discards every XYZ/scalar partial and all route counters',async()=>{
 let calls=0,resets=0,destroyed=0;const events=[],previews=[];
 const actual=await runtime.executeCoherentMeshTask({...baseOptions,backend:'auto'},{spectralGpuFactory:async()=>({backend:{kind:'webgpu',cpuShadow:false},async evaluate({sampleCount}){calls++;if(calls===2)throw new Error('injected spectral device loss after first tile');return synthetic(sampleCount);},destroy(){destroyed++;}}),reset(r){resets++;events.push('reset');assert.equal(r.attempt,1);},preview(p){previews.push(p);events.push(`preview${p.attempt}`);}});
 assert.equal(calls,2);assert.equal(resets,1);assert.equal(destroyed,1);assert.equal(actual.backend.kind,'cpu');assert.equal(actual.diagnostics.attempt,1);assert.deepEqual(events,['preview0','reset','preview1','preview1']);
 for(const key of imageNames)assert.deepEqual(actual[key],canonical[key]);for(const key of routeCounts)assert.equal(actual.diagnostics[key],canonical.diagnostics[key]);assert.deepEqual(actual.diagnostics.edgeWarningCounts,canonical.diagnostics.edgeWarningCounts);assert.deepEqual(actual.diagnostics.edgeExclusionCounts,canonical.diagnostics.edgeExclusionCounts);assert.equal(actual.diagnostics.spectralFieldSamples,16*16*16);
 for(const p of previews.filter(p=>p.attempt===1))for(const key of imageNames){const c=key.endsWith('XYZ')?3:1;assert.deepEqual(p[key],canonical[key].slice(c*p.rowStart*16,c*(p.rowStart+p.rowCount)*16));}
 return {calls,resets,destroyed,allSevenImagesCPUExact:true,routeAndWarningCountersCPUExact:true};
});
await check('malformed/nonfinite spectral GPU output is withheld and causes one complete CPU replay',async()=>{
 const mutations=[r=>delete r.coherentXYZ,r=>r.incoherentXYZ=new Float32Array(1),r=>r.geometricXYZ[0]=NaN,r=>r.coherentXYZ[0]=-1,r=>r.coherent[0]=Infinity];
 for(const mutate of mutations){let resets=0,calls=0;const r=await runtime.executeCoherentMeshTask({...baseOptions,backend:'auto'},{spectralGpuFactory:async()=>({backend:{kind:'webgpu'},async evaluate({sampleCount}){calls++;const value=synthetic(sampleCount);mutate(value);return value;},destroy(){}}),reset(){resets++;},preview(p){assert.equal(p.attempt,1,'bad candidate must not produce a preview');}});assert.equal(calls,1);assert.equal(resets,1);for(const key of imageNames)assert.deepEqual(r[key],canonical[key]);}
 return {malformedCases:mutations.length,publicationBeforeReset:false};
});
await check('abort and caller failures suppress stale spectral publication without fallback masquerade',async()=>{
 const controller=new AbortController();let previews=0,resets=0,destroyed=0;
 await assert.rejects(runtime.executeCoherentMeshTask({...baseOptions,backend:'auto'},{signal:controller.signal,spectralGpuFactory:async()=>({backend:{kind:'webgpu'},async evaluate({sampleCount}){controller.abort();return synthetic(sampleCount);},destroy(){destroyed++;}}),preview(){previews++;},reset(){resets++;}}),e=>e.name==='AbortError');assert.equal(previews,0);assert.equal(resets,0);assert.equal(destroyed,1);
 let calls=0;await assert.rejects(runtime.executeCoherentMeshTask({...baseOptions,backend:'auto'},{spectralGpuFactory:async()=>({backend:{kind:'webgpu'},async evaluate({sampleCount}){calls++;return synthetic(sampleCount);},destroy(){}}),preview(){throw new Error('independent observer failure');},reset(){resets++;}}),/independent observer failure/);assert.equal(calls,1);assert.equal(resets,0);
 const afterPreview=new AbortController();previews=0;await assert.rejects(runtime.executeCoherentMeshTask(baseOptions,{signal:afterPreview.signal,preview(){previews++;afterPreview.abort();}}),e=>e.name==='AbortError');assert.equal(previews,1);
 await assert.rejects(runtime.executeCoherentMeshTask({...baseOptions,resolution:512,pixelQuadrature:8,spectralSamples:128,maxRuntimeMs:10}),e=>e.name==='TimeoutError');
 return {cancelledBeforePreview:true,callbackFailuresNotGPUFallback:true,CPUPreviewCancellation:true,boundedTimeout:true};
});
await check('zero-power broadband image is exactly black without erasing selected spectral metadata',async()=>{const r=await runtime.executeCoherentMeshTask({...baseOptions,power:0,spectralSamples:128});for(const key of imageNames)assert(r[key].every(v=>v===0));assert.equal(r.spectralSamples,128);assert(r.spectralComplete);return {allChannelsZero:true,spectralSamples:r.spectralSamples};});



report.sourceHashes=Object.fromEntries(['dist/coherent-mesh-spectrum.js','dist/coherent-mesh-runtime.js','dist/coherent-mesh-transport.js','dist/coherent-mesh-spectral-gpu.js','dist/coherent-mesh-spectral-templates.js','dist/coherent-mesh-spectral-compute.wgsl','tests/coherent-mesh-spectral-independent.mjs'].map(p=>[p,sha(p)]));report.passed=report.checks.length;report.failed=report.failures.length;fs.writeFileSync(new URL('research/coherent-mesh-spectral-independent.json',root),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.passed,failed:report.failed}));if(report.failed)process.exitCode=1;
