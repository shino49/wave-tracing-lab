/** DOM/control-flow QA with happy-dom + a Canvas 2D spy. No real browser, raster
 * renderer, graphics driver, WebGPU compilation or GPU execution is validated. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {builtInSceneOBJ} from '../dist/utd-scene.js';
import {evaluateReflectedPoint,reflectPoint} from '../dist/utd-mirror.js';
import {add,scale} from '../dist/mesh-geometry.js';
import {mockGPU,cpuFixture} from './helpers/utd-mock-gpu.mjs';
const {Window}=await import(process.env.WAVE_TEST_DOM||'happy-dom');
const win=new Window({url:'https://test.local',settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});
win.document.write(fs.readFileSync(new URL('../dist/mesh.html',import.meta.url),'utf8'));
const calls=[],images=[],contexts=new WeakMap(),noop=()=>{};
win.HTMLCanvasElement.prototype.getContext=function(){
  if(!contexts.has(this)){const canvas=this;contexts.set(this,new Proxy({
    createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4),width:w,height:h}),
    createRadialGradient:()=>({addColorStop:noop}),createLinearGradient:()=>({addColorStop:noop}),measureText:s=>({width:s.length*7}),
    putImageData:(im)=>images.push({canvas,image:im}),
  },{get:(t,k)=>k in t?t[k]:(...args)=>calls.push({canvas,method:k,args}),set:(t,k,v)=>(t[k]=v,true)}));}
  return contexts.get(this);
};
win.HTMLCanvasElement.prototype.getBoundingClientRect=()=>({width:640,height:420,top:0,left:0,right:640,bottom:420});
Object.assign(globalThis,{window:win,document:win.document,devicePixelRatio:1,ResizeObserver:class{observe(){}disconnect(){}},requestAnimationFrame:cb=>1});
let fetches=[];
const noGPU=()=>{Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});globalThis.fetch=async url=>{fetches.push(String(url));throw new Error('Unexpected fetch during local CPU/OBJ flow: '+url);};};
noGPU();
const $=id=>document.getElementById(id),tests=[],failures=[];
const test=async(name,fn)=>{try{await fn();tests.push(name);}catch(error){failures.push({name,error:error.stack});}};
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
const until=async fn=>{for(let i=0;i<1000;i++){if(fn())return;await delay(5);}throw new Error('Timed out waiting for test condition');};
const change=(id,value,event='input')=>{$(id).value=String(value);$(id).dispatchEvent(new win.Event(event,{bubbles:true}));};
const fileChange=file=>{Object.defineProperty($('meshFile'),'files',{value:[file],configurable:true});$('meshFile').dispatchEvent(new win.Event('change',{bubbles:true}));};
const reset=()=>{$('meshReset').click();noGPU();};
const assertIdle=()=>{assert.equal(utdLab.getState().running,false);assert.equal($('meshRun').disabled,false);assert.equal($('meshCancel').disabled,true);};
const assertNoResult=()=>{assert.equal(utdLab.getState().backend,undefined);assert.equal($('meshResultsPending').hidden,false);assert.equal($('meshPeak').textContent,'—');};
await import('../dist/mesh-app.js');

await test('Initial geometry is valid, controls are labelled, and no result is claimed',async()=>{
  assert.equal(utdLab.getState().sceneSupported,true);assert.equal(utdLab.getState().triangles,2);assert.equal($('meshError').hidden,true);assertIdle();assertNoResult();
  const ids=Array.from(document.querySelectorAll('[id]'),e=>e.id);assert.equal(new Set(ids).size,ids.length);
  for(const e of $('meshForm').querySelectorAll('input,select'))assert(document.querySelector(`label[for="${e.id}"]`),`${e.id} needs an associated label`);
  for(const e of document.querySelectorAll('canvas'))assert(e.getAttribute('aria-label'));
  assert.match($('meshStatus').getAttribute('role'),/status/);assert.equal($('meshError').getAttribute('role'),'alert');assert.equal(fetches.length,0);
  assert.equal($('meshTransport').value,'single');assert.equal($('meshMirrorDistance').value,'10');assert.equal($('meshMirrorDistance').disabled,true);assert.equal($('meshMirrorControls').hidden,true);
  assert.equal($('meshTransport').options.length,3);assert.match($('meshMirrorHelp').textContent,/45°.*无限理想标量镜面/);assert.match($('meshMirrorHelp').textContent,/无有限孔径.*无衍射边缘/);
});
await test('Automatic no-GPU run labels CPU fallback and renders all invalid pixels visibly',async()=>{
  await utdLab.run();const state=utdLab.getState();assert.equal(state.backend,'CPU 参考');assert(state.valid>0&&state.valid<33*33);assertIdle();
  assert.match($('meshBackend').textContent,/CPU 参考/);assert.match($('meshParity').textContent,/GPU 未获本次验证/);assert.match($('meshWarnings').textContent,/此浏览器未提供 WebGPU/);
  assert.equal(globalThis.utdLabValidation.backend,'CPU 参考');assert.equal(globalThis.utdLabValidation.parity,null);assert.equal($('meshResultsPending').hidden,true);
  const image=images.at(-1).image;assert.equal(image.data.length,33*33*4);let masked=0;for(let i=0;i<33*33;i++){const [r,g,b,a]=image.data.slice(i*4,i*4+4);assert.equal(a,255);if((r===43||r===55)&&g===r+3&&b===r+4)masked++;}assert.equal(masked,33*33-state.valid);assert.equal(fetches.length,0);
});
await test('Camera drag, keyboard orbit, wheel zoom and reset cannot mutate physics',async()=>{
  const before=JSON.stringify(utdLab.getScene()),revision=utdLab.getState().revision,canvas=$('meshSceneCanvas');
  canvas.dispatchEvent(new win.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));canvas.dispatchEvent(new win.KeyboardEvent('keydown',{key:'ArrowUp',bubbles:true}));
  canvas.dispatchEvent(new win.WheelEvent('wheel',{deltaY:40,bubbles:true,cancelable:true}));
  canvas.dispatchEvent(new win.PointerEvent('pointerdown',{pointerId:1,clientX:20,clientY:20}));canvas.dispatchEvent(new win.PointerEvent('pointermove',{pointerId:1,clientX:60,clientY:35}));canvas.dispatchEvent(new win.PointerEvent('pointerup',{pointerId:1,clientX:60,clientY:35}));$('meshResetCamera').click();
  assert.equal(JSON.stringify(utdLab.getScene()),before);assert.equal(utdLab.getState().revision,revision);assert.equal(utdLab.getState().backend,'CPU 参考');
});
await test('Each built-in model changes the actual mesh and physical interpretation',async()=>{
  const snapshots=[];for(const preset of ['box','wedge','opaque-double-slit','screen']){change('meshPreset',preset,'change');assertNoResult();assert.equal($('meshError').hidden,true);assert(utdLab.getState().sceneSupported);snapshots.push(JSON.stringify(utdLab.getScene().mesh.vertices));assert.equal($('meshFormat').value,['box','wedge'].includes(preset)?'closed-solid':'open-plate');}
  assert.equal(new Set(snapshots).size,4);assert.equal(fetches.length,0);
});
await test('Geometry unit, scale and rotations feed physical coordinates',async()=>{
  reset();const base=JSON.stringify(utdLab.getScene().mesh.vertices);
  for(const [id,value]of [['meshUnit',1.1],['meshScale',1.1],['meshRotateX',5],['meshRotateY',5],['meshRotateZ',5]]){reset();change(id,value);assertNoResult();await delay(180);assert.equal($('meshError').hidden,true,`${id}: ${$('meshError').textContent}`);assert.notEqual(JSON.stringify(utdLab.getScene().mesh.vertices),base,id);}
});
await test('Beam, incidence, detector and boundary controls update solver parameters',async()=>{
  for(const [id,value,verify]of [
    ['meshWavelength',.04,s=>assert.equal(s.params.wavelengthMm,.04)],['meshBeamRadius',1.2,s=>assert.equal(s.params.sigma0Mm,1.2)],
    ['meshYaw',3,s=>assert(Math.abs(s.params.incidentDirection[0])>.01)],['meshPitch',3,s=>assert(Math.abs(s.params.incidentDirection[1])>.01)],
    ['meshDistance',25,s=>assert.equal(s.params.detectorCenter[2],25)],['meshSpan',8,s=>assert.equal(s.params.detectorWidthMm,8)],['meshBoundary','hard',s=>assert.equal(s.params.boundary,'hard')],
  ]){reset();change(id,value);await delay(180);assert.equal($('meshError').hidden,true,`${id}: ${$('meshError').textContent}`);verify(utdLab.getScene());assertNoResult();assert.match($('meshParity').textContent,/待重新核对/);}
});
await test('Invalid numeric and unsupported surface fail before computing',async()=>{
  for(const [id,value]of [['meshScale',0],['meshWavelength','NaN'],['meshSpan','Infinity'],['meshPitch',90],['meshBeamRadius',.05],['meshFormat','closed-solid']]){reset();change(id,value);await utdLab.run();assertIdle();assertNoResult();assert.equal($('meshError').hidden,false,`${id} must reject`);assert.match($('meshStatus').textContent,/停止/);}
});
await test('CPU cancellation, immediate rerun and overlapping runs publish only latest result',async()=>{
  reset();change('meshBackendSelect','cpu');const a=utdLab.run();assert.equal(utdLab.getState().running,true);utdLab.cancel();await a;assertIdle();assertNoResult();assert.match($('meshStatus').textContent,/已取消/);
  const old=utdLab.run();change('meshBoundary','hard');const latest=utdLab.run();await Promise.all([old,latest]);assertIdle();assert.equal(utdLab.getState().backend,'CPU 参考');assert.equal(utdLab.getScene().params.boundary,'hard');assert.equal($('meshResultsPending').hidden,true);
});
await test('Reset during running job prevents obsolete output and restores all defaults',async()=>{
  change('meshPreset','box','change');change('meshYaw',5);const running=utdLab.run();$('meshReset').click();await running;assertIdle();assertNoResult();assert.equal(utdLab.getState().modelName,'有限薄屏');assert.equal($('meshPreset').value,'screen');assert.equal($('meshYaw').value,'0');assert.equal($('meshBoundary').value,'soft');assert.equal($('meshBackendSelect').value,'auto');assert.equal($('meshBackend').textContent,'待运行');assert.equal($('meshParity').textContent,'尚未执行');
});
await test('Cancel during asynchronous backend discovery never publishes stale CPU fallback',async()=>{
  reset();const ready=deferred();Object.defineProperty(globalThis,'navigator',{value:{gpu:{requestAdapter:()=>ready.promise}},configurable:true});const running=utdLab.run();utdLab.cancel();ready.resolve(null);await running;assertIdle();assertNoResult();assert.match($('meshStatus').textContent,/已取消/);noGPU();
});
await test('File limit rejects before reading and malformed file preserves current geometry',async()=>{
  reset();const before=JSON.stringify(utdLab.getScene().mesh.vertices);let reads=0;fileChange({name:'huge.obj',size:2*1024*1024+1,text:async()=>{reads++;return '';}});await until(()=>$('meshError').textContent.includes('2 MiB'));assert.equal(reads,0);assert.equal(JSON.stringify(utdLab.getScene().mesh.vertices),before);
  fileChange({name:'bad.obj',size:16,text:async()=>'<html>bad</html>'});await until(()=>$('meshError').textContent.includes('无法读取 OBJ')&&!$('meshError').textContent.includes('2 MiB'));assert.equal(JSON.stringify(utdLab.getScene().mesh.vertices),before);assertNoResult();
});
await test('OBJ material URLs never fetch or upload; successful file becomes actual model',async()=>{
  reset();const text='mtllib https://example.invalid/do-not-fetch.mtl\nusemtl remote\n'+builtInSceneOBJ('screen');fileChange({name:'local-only.obj',size:text.length,text:async()=>text});await until(()=>utdLab.getState().modelName==='local-only.obj');assert.equal($('meshError').hidden,true);assert.match($('meshWarnings').textContent,/不加载外部文件/);assert.equal(fetches.length,0);assertNoResult();
});
await test('Slower old file cannot overwrite newer file or a reset model',async()=>{
  reset();const slow=deferred();fileChange({name:'old.obj',size:200,text:()=>slow.promise});fileChange({name:'new.obj',size:200,text:async()=>builtInSceneOBJ('screen')});await until(()=>utdLab.getState().modelName==='new.obj');slow.resolve(builtInSceneOBJ('box'));await delay(10);assert.equal(utdLab.getState().modelName,'new.obj');assert.equal(utdLab.getState().triangles,2);
  const late=deferred();fileChange({name:'late.obj',size:200,text:()=>late.promise});$('meshReset').click();late.resolve(builtInSceneOBJ('box'));await delay(10);assert.equal(utdLab.getState().modelName,'有限薄屏');assert.equal(utdLab.getState().triangles,2);
});
await test('Mock GPU compile failure falls back visibly and frees allocated device',async()=>{
  reset();const mock=mockGPU({compileMessages:[{type:'error',lineNum:1,linePos:1,message:'mock compile failure'}]});await utdLab.run();assertIdle();assert.equal(utdLab.getState().backend,'CPU 参考');assert.match($('meshWarnings').textContent,/mock compile failure/);assert.match($('meshParity').textContent,/GPU 未获本次验证/);mock.assertDestroyed();noGPU();
});
await test('Mock GPU parity rejection replaces invalid GPU output with labelled CPU result',async()=>{
  reset();const fixture=cpuFixture(utdLab.getScene(),33,33);for(let i=0;i<33*33;i++)if(fixture.data[i*4+2])fixture.data[i*4]+=10;const mock=mockGPU({fixture});await utdLab.run();assertIdle();assert.equal(utdLab.getState().backend,'CPU 参考');assert.match($('meshWarnings').textContent,/数值核对未通过/);assert.match($('meshParity').textContent,/GPU 未获本次验证/);mock.assertDestroyed();noGPU();
});
await test('Parity report preserves failed GPU values and copies only on user action',async()=>{
  reset();const fixture=cpuFixture(utdLab.getScene(),33,33);for(let i=0;i<33*33;i++)if(fixture.data[i*4+2])fixture.data[i*4]+=10;
  const mock=mockGPU({fixture,adapterInfo:{vendor:'test-vendor',description:'test-adapter',isFallbackAdapter:true}});
  let copies=0,copied='';navigator.clipboard={writeText:async text=>{copies++;copied=text;}};
  await utdLab.run();assert.equal(copies,0);assert.equal($('meshDiagnostics').hidden,false);
  const d=utdLabDiagnostic;assert.equal(d.adapter.vendor,'test-vendor');assert.equal(d.parity.samples.length,25);assert.equal(d.params.boundary,'soft');
  assert.equal(d.parity.passed,false);assert.equal(d.dimensions.width,33);assert.match(d.buildId,/diagnostics/);
  const sample=d.parity.samples.find(p=>p.status==='numeric-mismatch');assert(sample.gpu.f>sample.cpu.f+9);
  assert.equal(utdLabValidation.backend,'CPU 参考');assert(!('objText' in d));
  $('meshCopyDiagnostics').click();await delay(0);assert.equal(copies,1);assert.deepEqual(JSON.parse(copied),d);assert.match($('meshCopyStatus').textContent,/已复制/);
  mock.assertDestroyed();noGPU();
});
await test('Report copy failure gives a selectable local report without network upload',async()=>{
  assert(utdLabDiagnostic);$('meshCopyDiagnostics').click();await delay(0);
  assert.equal($('meshDiagnosticDetails').open,true);assert.match($('meshCopyStatus').textContent,/请复制/);
  assert.equal(JSON.parse($('meshDiagnosticText').value).parity.samples.length,25);
});
await test('Changing parameters clears report and suppresses delayed copy status',async()=>{
  const done=deferred();navigator.clipboard={writeText:()=>done.promise};$('meshCopyDiagnostics').click();
  change('meshWavelength',.04);assert.equal($('meshDiagnostics').hidden,true);assert.equal(utdLabDiagnostic,null);
  assert.equal($('meshDiagnosticText').value,'');done.resolve();await delay(0);assert.equal($('meshCopyStatus').textContent,'');reset();
});
await test('Mock GPU device loss during work uses visible CPU fallback',async()=>{
  reset();const done=deferred(),entered=deferred();const mock=mockGPU({fixture:cpuFixture(utdLab.getScene(),33,33),workWait:done.promise,onWork:()=>entered.resolve()});const running=utdLab.run();await entered.promise;mock.lose('mock device loss');await delay(0);done.resolve();await running;assert.equal(utdLab.getState().backend,'CPU 参考');assert.match($('meshWarnings').textContent,/mock device loss/);assert.match($('meshParity').textContent,/GPU 未获本次验证/);mock.assertDestroyed();noGPU();
});
await test('Prior mock GPU validation claim is cleared on preset, file, cancel and reset',async()=>{
  for(const changeKind of ['preset','file','cancel','reset']){
    reset();const mock=mockGPU({fixture:cpuFixture(utdLab.getScene(),33,33)});await utdLab.run();assert.equal(utdLab.getState().backend,'WebGPU');assert.match($('meshParity').textContent,/25 点通过/);assert(globalThis.utdLabValidation.parity.passed);mock.assertDestroyed();
    if(changeKind==='preset')change('meshPreset','box','change');
    if(changeKind==='file'){fileChange({name:'replacement.obj',size:200,text:async()=>builtInSceneOBJ('screen')});await until(()=>utdLab.getState().modelName==='replacement.obj');}
    if(changeKind==='cancel')utdLab.cancel();if(changeKind==='reset')$('meshReset').click();
    assertNoResult();assert.doesNotMatch($('meshParity').textContent,/点通过/,changeKind+' retains stale verification claim');assert(!globalThis.utdLabValidation||!globalThis.utdLabValidation.parity?.passed,changeKind+' retains stale debug validation');noGPU();
  }
});
await test('Pagehide cancels work without publishing partial data',async()=>{
  reset();const running=utdLab.run();win.dispatchEvent(new win.Event('pagehide'));await running;assertIdle();assertNoResult();
});

const nearVector=(a,b)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert(Math.abs(v-b[i])<1e-9,`${a} differs from ${b}`));};
const expectedPixel=(r,x,y)=>{
  if(!r.valid){const z=(x+y)%4<2?43:55;return [z,z+3,z+4,255];}
  const db=10*Math.log10(Math.max(Math.fround(r.f),1e-4)),t=Math.max(0,Math.min(1,(db+40)/46)),q=Math.pow(t,1.6);
  return [...new Uint8ClampedArray([9+220*q,22+216*q,16+175*q,255])];
};
await test('Both mirror orders render actual CPU reflected-family pixels and physical frames',async()=>{
  for(const order of ['before','after']){
    reset();const original=utdLab.getScene().params;change('meshTransport',order,'change');change('meshBackendSelect','cpu');await utdLab.run();assertIdle();
    assert.equal($('meshError').hidden,true,$('meshError').textContent);assert.equal(utdLab.getState().backend,'CPU 参考');assert.equal(utdLab.getState().transport,order);
    const scene=utdLab.getScene(),t=scene.mirrorTransport,p=t.physical,display=utdLab.getViewScene();assert(scene.supported);assert.equal(t.order,order);
    nearVector(scene.params.sourceOrigin,original.sourceOrigin);nearVector(scene.params.detectorCenter,original.detectorCenter);
    assert.equal($('meshMirrorControls').hidden,false);assert.equal($('meshMirrorDistance').disabled,false);
    assert.equal(display.incomingCone,null);nearVector(display.sourceOrigin,p.sourceOrigin);nearVector(display.incidentDirection,p.incidentDirection);
    nearVector(display.detector.center,p.detectorCenter);nearVector(display.detector.u,p.detectorU);nearVector(display.detector.v,p.detectorV);
    assert.deepEqual(display.foldedPath,t.foldedPath);assert.equal(display.foldedPath.length,4);assert.equal(display.mirror.illustrative,true);assert.deepEqual(display.mirror.normal,t.mirror.normal);
    if(order==='before'){assert.notDeepEqual(display.sourceOrigin,scene.params.sourceOrigin);nearVector(reflectPoint(display.sourceOrigin,t.mirror),scene.params.sourceOrigin);}
    else {assert.notDeepEqual(display.detector.center,scene.params.detectorCenter);nearVector(reflectPoint(display.detector.center,t.mirror),scene.params.detectorCenter);nearVector(display.detector.u,[0,0,1]);}
    const image=images.at(-1).image;let valid=0,peak=0;
    for(let y=0;y<33;y++)for(let x=0;x<33;x++){
      const dst=add(p.detectorCenter,add(scale(p.detectorU,((x+.5)/33-.5)*p.detectorWidthMm),scale(p.detectorV,(.5-(y+.5)/33)*p.detectorHeightMm)));
      const r=evaluateReflectedPoint(scene,dst);assert.deepEqual([...image.data.slice((y*33+x)*4,(y*33+x+1)*4)],expectedPixel(r,x,y),`${order} pixel ${x},${y}`);
      if(r.valid){valid++;peak=Math.max(peak,Math.fround(r.f));}
    }
    assert.equal(valid,utdLab.getState().valid);assert(valid>0&&valid<1089);assert.equal($('meshPeak').textContent,peak.toPrecision(5));
    assert.match($('meshResultsTitle').textContent,/选定反射路径族/);assert.match($('meshRegionNotice').textContent,/完整局部 GO \+ D/);assert.match($('meshRegionNotice').textContent,/未合成独立直达路径族/);
    assert.match($('meshTransportSummary').textContent,/DD.*通用多区域/);assert.match($('meshHeatmapRange').textContent,/展开轴向距离/);assert.match($('meshSceneNotice').textContent,/无限理想标量镜面.*有限绘图片仅作示意/);
    const labels=calls.filter(c=>c.canvas===$('meshSceneCanvas')&&c.method==='fillText').map(c=>c.args[0]);assert(labels.includes('真实源'));assert(labels.includes('真实接收面'));assert(labels.includes('∞ 理想镜面 · 示意片无孔径'));
    assert.equal(fetches.length,0);
  }
});
await test('Reflected endpoint, box-centre and slit masks remain gray without path repair',async()=>{
  for(const order of ['before','after'])for(const preset of ['screen','box']){
    reset();change('meshPreset',preset,'change');change('meshTransport',order);change('meshBackendSelect','cpu');await utdLab.run();assert.equal($('meshError').hidden,true);
    const x=16,y=preset==='screen'?0:16,image=images.at(-1).image;assert.deepEqual([...image.data.slice((y*33+x)*4,(y*33+x+1)*4)],expectedPixel({valid:false},x,y));
  }
  reset();change('meshPreset','opaque-double-slit','change');change('meshTransport','after');change('meshBackendSelect','cpu');await utdLab.run();assert.equal($('meshError').hidden,true);
  assert.deepEqual([...images.at(-1).image.data.slice((16*33+16)*4,(16*33+17)*4)],expectedPixel({valid:false},16,16));
  change('meshTransport','before');await utdLab.run();assertIdle();assertNoResult();assert.equal(utdLab.getScene().supported,false);assert.match($('meshError').textContent,/真实源.*镜面同一侧/);assert.equal(utdLab.getScene().mirrorTransport.order,'before');
});
await test('Unsafe mirror separation is rejected visibly and never falls back to an old scene',async()=>{
  for(const separation of [.001,30,0,-1]){
    reset();change('meshTransport','before');await utdLab.run();assert.equal(utdLab.getState().backend,'CPU 参考');
    change('meshMirrorDistance',separation);assertNoResult();assert.equal(utdLab.getViewScene(),null);await utdLab.run();assertIdle();assertNoResult();assert.equal($('meshError').hidden,false);assert.match($('meshStatus').textContent,/停止/);
    if(separation>0){assert.equal(utdLab.getScene().supported,false);assert.equal(utdLab.getScene().mirrorTransport.order,'before');assert.doesNotMatch($('meshError').textContent,/mirror-/);}
    else {assert.equal(utdLab.getScene(),null);assert.equal(utdLab.getViewScene(),null);assert.match($('meshError').textContent,/镜面.*有限数值/);}
    assert.doesNotMatch($('meshRegionNotice').textContent,/点有效/);
  }
});
await test('Changing mirror route during CPU work cancels and only the newest family can publish',async()=>{
  reset();change('meshTransport','before');change('meshBackendSelect','cpu');const old=utdLab.run();await until(()=>$('meshStatus').textContent.includes('已计算'));
  change('meshTransport','after','change');assertIdle();assertNoResult();assert.equal(utdLabValidation,null);assert.equal(utdLab.getViewScene(),null);
  const latest=utdLab.run();await Promise.all([old,latest]);assertIdle();assert.equal(utdLab.getState().backend,'CPU 参考');assert.equal(utdLab.getScene().mirrorTransport.order,'after');
  assert.deepEqual(utdLab.getViewScene().detector.center,utdLab.getScene().mirrorTransport.physical.detectorCenter);
  change('meshTransport','single','change');assertNoResult();assert.doesNotMatch($('meshRegionNotice').textContent,/点有效/);await delay(180);assert.equal(utdLab.getScene().mirrorTransport,undefined);assert.equal(utdLab.getViewScene().mirror,null);
});
await test('Changing mirror route during backend discovery suppresses stale fallback and diagnostic claims',async()=>{
  reset();change('meshTransport','before');const ready=deferred();Object.defineProperty(globalThis,'navigator',{value:{gpu:{requestAdapter:()=>ready.promise}},configurable:true});
  const running=utdLab.run();change('meshTransport','after','change');ready.resolve(null);await running;assertIdle();assertNoResult();assert.equal(utdLabDiagnostic,null);assert.equal(utdLabValidation,null);assert.equal($('meshDiagnosticText').value,'');noGPU();
  await delay(180);assert.equal(utdLab.getScene().mirrorTransport.order,'after');assert.equal(utdLab.getState().backend,undefined);
});
await test('Mirror no-GPU fallback keeps reflected-family scope and cancel clears every result claim',async()=>{
  reset();change('meshTransport','after');await utdLab.run();assert.equal(utdLab.getState().backend,'CPU 参考');assert.match($('meshWarnings').textContent,/WebGPU.*CPU 参考/);
  assert.match($('meshWarnings').textContent,/完整局部 GO \+ D/);assert.match($('meshWarnings').textContent,/无.*有限孔径.*衍射边缘/);assert.match($('meshWarnings').textContent,/DD.*多区域/);assert.match($('meshWarnings').textContent,/几何路径畅通不等于波动计算准确/);
  utdLab.cancel();assertIdle();assertNoResult();assert.equal(utdLabValidation,null);assert.doesNotMatch($('meshRegionNotice').textContent,/点有效/);assert.match($('meshResultsTitle').textContent,/选定反射路径族/);
});
await test('Reset during reflected run restores original transport and mirror defaults',async()=>{
  reset();change('meshTransport','before');change('meshMirrorDistance',11);change('meshBackendSelect','cpu');const running=utdLab.run();$('meshReset').click();await running;assertIdle();assertNoResult();
  assert.equal($('meshTransport').value,'single');assert.equal(utdLab.getState().transport,'single');assert.equal($('meshMirrorDistance').value,'10');assert.equal($('meshMirrorDistance').disabled,true);assert.equal($('meshMirrorControls').hidden,true);
  assert.equal(utdLab.getScene().mirrorTransport,undefined);assert.equal(utdLab.getViewScene().mirror,null);assert.equal(utdLab.getViewScene().foldedPath,null);assert(utdLab.getViewScene().incomingCone);
  assert.equal($('meshResultsTitle').textContent,'条件功率传递因子 f');assert.doesNotMatch($('meshSceneNotice').textContent,/无限/);assert.equal($('meshParity').textContent,'尚未执行');
});
console.log(JSON.stringify({suite:'mesh-dom',passed:tests.length,failed:failures.length,browserValidated:false,visualQA:false,gpuExecutionValidated:false,coverage:'happy-dom controls, Canvas call/data spies and mocked GPU lifecycle only',tests,failures},null,2));
await win.happyDOM.close();if(failures.length)process.exitCode=1;
