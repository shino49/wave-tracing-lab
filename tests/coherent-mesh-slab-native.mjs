/** Native Dawn/SwiftShader only. Never launches a browser or claims hardware execution. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {create,globals} from 'webgpu';
import {executeCoherentMeshTask,prepareCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
import {createCoherentMeshSpectralGPURunner} from '../dist/coherent-mesh-spectral-gpu.js';
import {createCoherentMeshGPURunner} from '../dist/coherent-mesh-gpu.js';
import {displaySpectralXYZ} from '../dist/plt-spectrum.js';
import {makeCoherentCameraPreset,traceCoherentCamera} from '../dist/coherent-mesh-camera.js';
import {sceneFromCoherentData} from '../dist/coherent-mesh-scene.js';
Object.assign(globalThis,globals);
assert.equal(process.env.WEBGPU_EXECUTION_BACKEND,'vulkan');
const gpu=create(['backend=vulkan']),mode=process.argv[2]??'fast';
const spectralShader=fs.readFileSync('dist/utd-functions.wgsl','utf8')+'\n'+fs.readFileSync('dist/coherent-mesh-spectral-compute.wgsl','utf8'),monoShader=fs.readFileSync('dist/coherent-mesh-compute.wgsl','utf8');
const exposureEV=['interference','detector-clear'].includes(mode)?14:18;
const prefix='research/slab-',report={mode,scope:'Native Dawn Vulkan SwiftShader software execution. CPU geometry; native WebGPU integration. No browser, physical GPU, or CPU shadow.',exposureEV,parity:[],cases:[],hashes:{},startedAt:new Date().toISOString()};
for(const f of ['coherent-mesh-slab.js','coherent-mesh-spectral-templates.js','coherent-mesh-runtime.js','coherent-mesh-lighting-runtime.js','coherent-mesh-environment.js','coherent-mesh-camera.js','coherent-mesh-spectral-gpu.js','coherent-mesh-spectral-compute.wgsl'])report.hashes[f]=createHash('sha256').update(fs.readFileSync('dist/'+f)).digest('hex');
for(const f of fs.readdirSync('dist').filter(f=>/^coherent-mesh-.*\.(js|wgsl)$/.test(f)))report.hashes[f]=createHash('sha256').update(fs.readFileSync('dist/'+f)).digest('hex');
const save=()=>fs.writeFileSync(prefix+mode+'-report.json',JSON.stringify(report,null,2)+'\n');
const preset=makeCoherentCameraPreset(),camera={...preset.camera,mirrorBounces:2},spectrum={mode:'broadband',illuminant:'D65'},base={preset:'object-imaging',measurement:'camera',camera,diffraction:true,mixedDiffraction:false,resolution:16,pixelQuadrature:1,spectralSamples:16,spectralSeed:7391,maxRuntimeMs:600000};
const keys=['coherent','geometric','incoherent','crossTerm','coherentXYZ','geometricXYZ','incoherentXYZ'];
async function run(name,input={},backend='auto'){
 const start=performance.now();let firstCompleteTileMs=null,firstPreviewMs=null,firstLightingRowMs=null,last=-1;
 const r=await executeCoherentMeshTask({...base,...input,backend},{gpuFactory:o=>createCoherentMeshGPURunner({...o,gpu,shaderSource:monoShader}),spectralGpuFactory:o=>createCoherentMeshSpectralGPURunner({...o,gpu,shaderSource:spectralShader}),preview:p=>{if(firstPreviewMs===null)firstPreviewMs=performance.now()-start;if(p.rowCount===1&&p.lightingComplete===false&&firstLightingRowMs===null)firstLightingRowMs=performance.now()-start;if(p.spatialComplete&&firstCompleteTileMs===null){firstCompleteTileMs=performance.now()-start;console.log('FIRST_TILE '+name+' '+firstCompleteTileMs);}},progress:p=>{const step=Math.floor((p.fraction??0)*10);if(step!==last){last=step;console.log(name+' '+step*10+'% '+Math.round(performance.now()-start)+'ms');}}});
 assert(r.complete&&r.spatialComplete&&r.spectralComplete);assert.equal(r.diagnostics.productionCPUShadow,false);if(backend==='auto'){assert.equal(r.backend.kind,'hybrid');assert.equal(r.diagnostics.fallbackReason,null);}r.firstCompleteTileMs=firstCompleteTileMs;r.firstPreviewMs=firstPreviewMs;r.firstLightingRowMs=firstLightingRowMs;return r;
}
function errors(a,b){const out={};for(const k of keys){let s=0,t=0,max=0,peak=0;for(let i=0;i<a[k].length;i++){const d=a[k][i]-b[k][i];s+=d*d;t+=b[k][i]**2;max=Math.max(max,Math.abs(d));peak=Math.max(peak,Math.abs(b[k][i]));}out[k]={relativeL2:Math.sqrt(s/(t||1)),relativeToPeak:max/(peak||1),maxAbsolute:max};}return out;}
function output(name,r){
 for(const k of keys)fs.writeFileSync(prefix+name+'-'+k+'.f64',Buffer.from(r[k].buffer,r[k].byteOffset,r[k].byteLength));
 for(const k of ['coherent','geometric','incoherent','crossTerm']){const rgb=Buffer.alloc(r.width*r.height*3);for(let i=0;i<r.width*r.height;i++){if(k==='crossTerm'){const c=r.crossTerm[i],a=Math.min(255,Math.round(255*Math.sqrt(Math.abs(c)*2**18)));rgb.set(c>=0?[a,0,0]:[0,0,a],3*i);}else rgb.set(displaySpectralXYZ(r[k+'XYZ'].subarray(i*3,i*3+3),exposureEV).rgb,i*3);}fs.writeFileSync(prefix+name+'-'+k+'.ppm',Buffer.concat([Buffer.from(`P6\n${r.width} ${r.height}\n255\n`),rgb]));}
 const entry={name,width:r.width,height:r.height,firstPreviewMs:r.firstPreviewMs,firstLightingRowMs:r.firstLightingRowMs,firstCompleteTileMs:r.firstCompleteTileMs,timings:r.timings,backend:r.backend,diagnostics:r.diagnostics,configuration:r.configuration,rawFormat:'Unmodified float64 little-endian row-major; XYZ interleaved.',display:`Fixed +${exposureEV}EV displaySpectralXYZ, no normalization or filtering. Cross term red-positive/blue-negative sqrt(abs(c)*2^18).`};report.cases.push(entry);save();console.log('CASE '+JSON.stringify(entry));
}

const glass=JSON.parse(fs.readFileSync('dist/examples/coherent-glass-imaging.json','utf8')),arm=JSON.parse(fs.readFileSync('dist/examples/coherent-slab-interference.json','utf8'));
const clear=structuredClone(glass);delete clear.slab;
const glassInput={scene:glass,measurement:'camera',camera:glass.camera,maxReflections:0,diffraction:false,lighting:glass.lighting,spectrum};
const armInput={scene:arm,measurement:'detector',camera:null,maxReflections:1,diffraction:false,spectrum:{mode:'monochrome'},lighting:null};
function armOracle(sceneData,r){
 const s=sceneData.source.position,h=sceneData.slab.thickness,n=sceneData.slab.index,lambda=sceneData.source.wavelengthMm,k=2*Math.PI/lambda,C=Math.sqrt(sceneData.source.power/(4*Math.PI)),mirror=sceneData.materials.find(m=>m.type==='mirror');const checks=[];
 for(const [x,y] of [[0,0],[Math.floor(r.width/2),Math.floor(r.height/2)],[r.width-1,r.height-1]]){
  const d=sceneData.detector,b=d.center.map((v,i)=>v+((x+.5)/r.width-.5)*d.width*d.u[i]+(.5-(y+.5)/r.height)*d.height*d.v[i]),rho=Math.hypot(b[0]-s[0],b[1]-s[1]),A=Math.abs(b[2]-s[2])-h;
  let lo=0,hi=1;for(let j=0;j<100;j++){const p=(lo+hi)/2;if(A*p/Math.sqrt(1-p*p)+h*p/Math.sqrt(n*n-p*p)<rho)lo=p;else hi=p;}
  const p=(lo+hi)/2,c=Math.sqrt(1-p*p),q=Math.sqrt(n*n-p*p),L=h*n/q,opl=A/c+n*L,lr=Math.hypot(b[0]-s[0],b[1]+s[1],b[2]-s[2]),G=1/(c*Math.sqrt((A/c+h/q)*(A/c**3+h*n*n/q**3))),T=4*c*q/(c+q)**2;
  const nodes=sceneData.slab.absorption??[{wavelengthNm:360,value:0}];let alpha=nodes.at(-1).value,nm=lambda*1e6;if(nm<=nodes[0].wavelengthNm)alpha=nodes[0].value;else for(let j=1;j<nodes.length;j++)if(nm<=nodes[j].wavelengthNm){const a=nodes[j-1],b=nodes[j];alpha=a.value+(b.value-a.value)*(nm-a.wavelengthNm)/(b.wavelengthNm-a.wavelengthNm);break;}
  const at=C*G*T*Math.exp(-alpha*L/2),ar=C*Math.sqrt(mirror.reflectance)/lr,inc=at*at+ar*ar,cross=2*at*ar*Math.cos(k*(lr-opl)-(mirror.phaseRad??Math.PI)),coherent=inc+cross,index=y*r.width+x;
  const expected={coherent,incoherent:inc,geometric:inc,crossTerm:cross},actual=Object.fromEntries(Object.keys(expected).map(key=>[key,r[key][index]]));
  for(const key of Object.keys(expected))assert(Math.abs(actual[key]-expected[key])/inc<1e-4,JSON.stringify({key,actual:actual[key],expected:expected[key],x,y}));checks.push({x,y,glassLength:L,alpha,expected,actual});
 }
 return {checks,scope:'Independent scalar Fermat bisection, s-Fresnel product, ray-tube Jacobian, image-source mirror, per-route Beer field loss, then complex cross term. No production geometry or field helpers.'};
}
function comparison(a,b){let aa=0,bb=0,ab=0;for(let i=0;i<a.length;i++){aa+=a[i]*a[i];bb+=b[i]*b[i];ab+=a[i]*b[i];}return {cosine:ab/Math.sqrt(aa*bb),relativeL2:Math.sqrt((aa+bb-2*ab)/bb)};}
function coverage(sceneData,n=512){const p=prepareCoherentMeshTask({...glassInput,scene:sceneData}),counts={},branches={},unsupported={},lengths=[];let lateralSum=0,lateralMax=0,lateralCount=0;
 const old=prepareCoherentMeshTask({...glassInput,scene:clear});
 for(let y=0;y<n;y++)for(let x=0;x<n;x++){const t=traceCoherentCamera(p.scene,p.configuration.camera,x,y,n,n,.5,.5,7391);counts[t.status]=(counts[t.status]??0)+1;for(const e of t.slabEvents??[]){branches[e.branch]=(branches[e.branch]??0)+1;if(e.glassLength)lengths.push(e.glassLength);}if(t.slabUnsupported)unsupported[t.slabUnsupported]=(unsupported[t.slabUnsupported]??0)+1;if(t.status==='diffuse'&&t.slabEvents?.[0]?.branch==='TT'){const c=traceCoherentCamera(old.scene,old.configuration.camera,x,y,n,n,.5,.5,7391);if(c.status==='diffuse'&&c.materialIndex===t.materialIndex){const d=Math.hypot(...c.point.map((v,i)=>v-t.point[i]));lateralSum+=d;lateralMax=Math.max(d,lateralMax);lateralCount++;}}}
 return {counts,branches,unsupported,transmittedSameMaterialHitShiftMm:{samples:lateralCount,mean:lateralSum/lateralCount,max:lateralMax},glassLengths:{samples:lengths.length,min:lengths.reduce((a,b)=>Math.min(a,b),Infinity),max:lengths.reduce((a,b)=>Math.max(a,b),-Infinity)},scope:'Deterministic geometric diagnostic with seed7391; runtime sample seeds can differ. Fresnel branch roulette, finite internal order, no denoising.'};}
try{
 if(mode==='parity'){
  const monochromes={};const absorbing=structuredClone(arm);absorbing.slab.absorption=[{wavelengthNm:360,value:0},{wavelengthNm:830,value:6}];
  for(const [name,input] of [['arm-mono',armInput],['arm-absorbing450',{...armInput,scene:{...absorbing,source:{...absorbing.source,wavelengthMm:.00045}}}],['arm-absorbing650',{...armInput,scene:{...absorbing,source:{...absorbing.source,wavelengthMm:.00065}}}],['arm-D65-absorbing',{...armInput,scene:absorbing,spectrum}],['camera-mono',{...glassInput,spectrum:{mode:'monochrome'}}],['camera-D65',glassInput],['camera-D65-indirect',{...glassInput,lighting:{...glass.lighting,indirectSamples:2}}]]){
   const cpu=await run(name+'-cpu',input,'cpu'),native=await run(name+'-native',input);const e=errors(native,cpu);for(const k of keys)assert(e[k].maxAbsolute<1e-4*Math.max(...cpu.coherent,...cpu[k],1e-20),JSON.stringify({name,k,...e[k]}));if(name==='arm-absorbing450'||name==='arm-absorbing650')monochromes[name]=native;const oracle=name.startsWith('arm-')&&!name.includes('D65')?armOracle(input.scene,native):null;report.parity.push({name,oracle,errors:e,cpuMs:cpu.timings.totalMs,nativeMs:native.timings.totalMs,diagnostics:native.diagnostics});save();console.log('PARITY '+name);
  }
  const mixed=await run('arm-two-lines',{...armInput,scene:absorbing,spectrum:{mode:'lines',lines:[{wavelengthNm:450,powerFraction:.5},{wavelengthNm:650,powerFraction:.5}]}});const mixing={};for(const key of keys){let maximum=0,peak=0;for(let i=0;i<mixed[key].length;i++){const expected=.5*(monochromes['arm-absorbing450'][key][i]+monochromes['arm-absorbing650'][key][i]);maximum=Math.max(maximum,Math.abs(mixed[key][i]-expected));peak=Math.max(peak,Math.abs(expected));}mixing[key]=maximum/(peak||1);assert(mixing[key]<1e-4);}report.twoLineIntensityMixing=mixing;
  const air=structuredClone(arm);air.slab.index=1;const none=structuredClone(air);delete none.slab;const a=await run('air',{...armInput,scene:air}),b=await run('no-slab',{...armInput,scene:none});for(const k of keys)assert.deepEqual(a[k],b[k]);report.airIdentity='Bit-identical native outputs for n=1 alpha=0 and no slab';
 }else if(mode==='camera'){
  output('clear512',await run('clear512',{...glassInput,scene:clear,resolution:512}));
  output('glass512',await run('glass512',{...glassInput,resolution:512}));report.cameraCoverage=coverage(glass);assert(report.cameraCoverage.branches.TT>0);assert(report.cameraCoverage.branches.R>0);assert(report.cameraCoverage.transmittedSameMaterialHitShiftMm.max>1e-5);
 }else if(mode==='glass'){
  output('glass-final512',await run('glass-final512',{...glassInput,resolution:512}));report.cameraCoverage=coverage(glass);assert(report.cameraCoverage.branches.TT>0);assert(report.cameraCoverage.branches.R>0);assert(report.cameraCoverage.transmittedSameMaterialHitShiftMm.max>1e-5);
 }else if(mode==='color'){
  const neutral=structuredClone(glass);neutral.slab.absorption=[{wavelengthNm:360,value:0},{wavelengthNm:830,value:0}];const a=await run('unabsorbing64',{...glassInput,scene:neutral,resolution:64}),b=await run('absorbing64',{...glassInput,resolution:64});output('unabsorbing64',a);output('absorbing64',b);const xyz=image=>{const v=[0,0,0];for(let i=0;i<image.coherentXYZ.length;i++)v[i%3]+=image.coherentXYZ[i];return v;};const unabsorbing=xyz(a),absorbing=xyz(b);report.absorptionColor={unabsorbingXYZSum:unabsorbing,absorbingXYZSum:absorbing,XYZRatio:absorbing.map((v,i)=>v/unabsorbing[i]),scope:'Identical slab geometry/Fresnel branches and samples; only extinction spectrum changed. Full coherent source plus sky, fixed +18EV.'};assert(Math.max(...report.absorptionColor.XYZRatio)-Math.min(...report.absorptionColor.XYZRatio)>.01);
 }else if(mode==='interference'){
  const shifted=structuredClone(arm);shifted.slab.thickness+=arm.source.wavelengthMm/(2*(arm.slab.index-1));
  const a=await run('arm512',{...armInput,resolution:512}),b=await run('arm-pi512',{...armInput,scene:shifted,resolution:512});output('arm512',a);output('arm-pi512',b);report.interference={oracle:armOracle(arm,a),shiftedOracle:armOracle(shifted,b),deltaThicknessMm:shifted.slab.thickness-arm.slab.thickness,crossTerm:comparison(a.crossTerm,b.crossTerm),coherent:comparison(a.coherent,b.coherent),note:'Thickness increment predicts pi relative phase at normal incidence; finite-angle corrections and ray-tube amplitude change retained.'};assert(report.interference.crossTerm.cosine<-.99);
 }else if(mode==='detector-clear'){const noSlab=structuredClone(arm);delete noSlab.slab;output('arm-clear512',await run('arm-clear512',{...armInput,scene:noSlab,resolution:512}));}else if(mode==='coverage'){report.cameraCoverage=coverage(glass);console.log(JSON.stringify(report.cameraCoverage));}
 else throw Error('Unknown mode '+mode);
 report.passed=true;
}catch(e){report.passed=false;report.error=e.stack;process.exitCode=1;console.error(e);}finally{report.completedAt=new Date().toISOString();report.hashesAtEnd=Object.fromEntries(Object.keys(report.hashes).map(f=>[f,createHash('sha256').update(fs.readFileSync('dist/'+f)).digest('hex')]));report.changedDuringRun=Object.keys(report.hashes).filter(f=>report.hashes[f]!==report.hashesAtEnd[f]);save();}
