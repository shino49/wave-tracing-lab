/** Structural accessibility and workflow regression checks. No browser layout claim. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {Window} from 'happy-dom';
const html = fs.readFileSync(new URL('../dist/transport.html', import.meta.url), 'utf8');
const css = fs.readFileSync(new URL('../dist/transport-style.css', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('../dist/transport-app.js', import.meta.url), 'utf8');
const w = new Window({settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});
w.document.write(html);
const d = w.document, $ = id => d.getElementById(id);
let count = 0;
function test(name, fn) {fn(); console.log(`PASS ${name}`); count++;}
function isDisclosureHidden(el) {for(let p=el.parentElement;p;p=p.parentElement) if(p.localName==='details'&&!p.open)return true;return false;}
function before(a,b) {return Boolean(a.compareDocumentPosition(b)&w.Node.DOCUMENT_POSITION_FOLLOWING);}
test('Every ID is unique and application literal references resolve',()=>{
 const ids=[...d.querySelectorAll('[id]')].map(e=>e.id);
 assert.equal(ids.length,new Set(ids).size);
 for(const [,id] of app.matchAll(/\$\(['"]([^'"]+)['"]\)/g)) assert($(id),`Missing #${id}`);
});
test('Scene, essentials, render action and output follow reading order',()=>{
 assert(before($('trPreset'),$('trMeasurementMode')));
 assert(before($('trMeasurementMode'),$('trRun')));
 assert(before($('trRun'),$('trCoherentCanvas')));
 assert(!isDisclosureHidden($('trPreset')));
 assert(!isDisclosureHidden($('trRun')));
});
test('All form controls have labels and all buttons have accessible names',()=>{
 for(const el of d.querySelectorAll('input,select,textarea')) assert(el.getAttribute('aria-label')||el.getAttribute('aria-labelledby')||d.querySelector(`label[for="${el.id}"]`),`Unlabelled ${el.id}`);
 for(const el of d.querySelectorAll('button')) {assert(el.textContent.trim()||el.getAttribute('aria-label'),`Unnamed ${el.id}`);assert.equal(el.type,'button',`Unexpected form submission ${el.id}`);}
});
test('Warnings, progress and status remain outside closed disclosures',()=>{
 for(const id of ['trCriticalWarnings','trConfigurationWarning','trSlabPropagationWarning','trStatus','trProgress']) assert(!isDisclosureHidden($(id)),`${id} hidden by disclosure`);
 assert($('trStatus').closest('[aria-live="polite"]'));
 assert($('trProgress').getAttribute('aria-label'));
});
test('Advanced controls and comparisons are initially collapsed',()=>{
 for(const id of ['trComparisons','trDiagnostics','trCameraSettings','trSourceSettings','trMaterialSettings','trLightSettings','trPathSettings','trQualitySettings','trImportSettings']) {
  const el=$(id);assert(el,`Missing ${id}`);assert.equal(el.localName,'details');assert.equal(el.open,false);assert(el.querySelector(':scope > summary')?.textContent.trim());
 }
 for(const id of ['trCoherence','trPhase','trQuadrature','trBackendSelect','trOBJ','trSceneJSON']) assert(isDisclosureHidden($(id)),`${id} unexpectedly overwhelms first view`);
});
test('Primary result stays visible and all scientific canvases stay native512',()=>{
 assert(!isDisclosureHidden($('trCoherentCanvas')));
 for(const id of ['trCoherentCanvas','trGeometricCanvas','trBaselineCanvas','trCrossCanvas']) {
 const el=$(id);assert.equal(el.width,512);assert.equal(el.height,512);assert(el.getAttribute('aria-label'));
 }
 assert.equal($('trSceneCanvas').tabIndex,0);
});
test('All labels, descriptions and local fragment links target existing nodes',()=>{
 for(const el of d.querySelectorAll('[for]'))for(const id of el.getAttribute('for').split(/\s+/))assert($(id),`Invalid for=${id}`);
 for(const el of d.querySelectorAll('[aria-labelledby],[aria-describedby]'))for(const a of ['aria-labelledby','aria-describedby'])for(const id of (el.getAttribute(a)||'').split(/\s+/).filter(Boolean))assert($(id),`Invalid ${a}=${id}`);
 for(const el of d.querySelectorAll('a[href^="#"]'))assert($(el.getAttribute('href').slice(1)),`Broken ${el.href}`);
});
test('Keyboard focus and hidden-state CSS safeguards remain available',()=>{
 assert(css.includes(':focus-visible'));assert(/\[hidden\][^{]*\{[^}]*display\s*:\s*none\s*!important/.test(css));assert(css.includes('@media'));
});
// Exercise presentation interactions against the actual application without a browser.
w.HTMLCanvasElement.prototype.getContext = function(){return new Proxy({
 createImageData:(width,height)=>({width,height,data:new Uint8ClampedArray(width*height*4)}),
 createRadialGradient:()=>({addColorStop(){}}),createLinearGradient:()=>({addColorStop(){}}),measureText:t=>({width:String(t).length*7})
},{get:(t,k)=>k in t?t[k]:()=>{}});};
w.HTMLCanvasElement.prototype.getBoundingClientRect=()=>({width:640,height:400,top:0,left:0,right:640,bottom:400});
Object.assign(globalThis,{window:w,document:d,devicePixelRatio:1,ResizeObserver:class{observe(){}disconnect(){}}});
Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});
let workerCount=0;globalThis.Worker=class{constructor(){workerCount++;}postMessage(){}terminate(){}};
await import('../dist/transport-app.js');
const change=(id,value)=>{$(id).value=String(value);$(id).dispatchEvent(new w.Event('input',{bubbles:true}));};
test('Scene summaries track presets and spectrum; quality changes do not run',()=>{
 for(const preset of ['lloyd','two-mirror','double-slit','reflected-edge','object-imaging']){
  change('trPreset',preset);assert($('trCurrentScene').textContent.trim());assert($('trPresetDescription').textContent.trim());
 }
 assert.match($('trCurrentSetup').textContent,/三维物体成像/);
 change('trSpectrumMode','monochrome');change('trWavelength',610);assert.match($('trCurrentSetup').textContent,/610 nm 单色/);
 $('trCameraFast').click();assert.equal($('trCameraFast').getAttribute('aria-pressed'),'true');assert.equal($('trCameraFine').getAttribute('aria-pressed'),'false');
 $('trCameraFine').click();assert.equal($('trCameraFine').getAttribute('aria-pressed'),'true');assert.equal($('trCameraFast').getAttribute('aria-pressed'),'false');assert.equal(workerCount,0);
 change('trMeasurementMode','detector');assert.match($('trModeHelp').textContent,/探测器模式/);
});
test('Glass scene summaries identify custom geometry and visible unsupported-path warnings',()=>{
 $('trGlassMirrorCameraPreset').click();assert.match($('trCurrentSetup').textContent,/玻璃已启用/);assert($('trCurrentScene').textContent.trim());
 change('trOrder',2);$('trDiffraction').checked=true;$('trDiffraction').dispatchEvent(new w.Event('input',{bubbles:true}));
 assert.equal($('trSlabPropagationWarning').hidden,false);assert(!isDisclosureHidden($('trSlabPropagationWarning')));
});
test('Critical result cautions remain concise and full evidence stays collapsed',()=>{
 const source=app.slice(app.indexOf('function showResultCautions('),app.indexOf('function resultEvidence('));
 const show=new Function('$','boundedText','finite','includesDiffraction',source+';return showResultCautions;')($,v=>typeof v==='string'?v:'',Number.isFinite,()=>true);
 show({diagnostics:{modelIncomplete:true,cameraSlabUnsupportedSamples:2,sourceSlabUnsupportedSamples:3,lightingSlabUnsupportedRays:4,cameraMirrorLimitSamples:5,cameraSlabTruncatedPowerSum:.3,edgeExclusionCounts:{outside:9},fallbackReason:'GPU unavailable'}},{},[]);
 const text=$('trCriticalWarnings').textContent;assert.match(text,/模型不完整/);assert.match(text,/物理黑色或吸收/);assert.match(text,/次数上限/);assert.match(text,/尾项/);assert.match(text,/被排除/);assert.match(text,/CPU/);
 assert(!isDisclosureHidden($('trCriticalWarnings')));assert(isDisclosureHidden($('trWarnings')));assert(isDisclosureHidden($('trDiffractionSummary')));
 show({diagnostics:{}},{},[]);assert.equal($('trCriticalWarnings').textContent,'');
 $('trGlassMirrorCameraPreset').click();change('trMeasurementMode','detector');assert.equal($('trConfigurationWarning').hidden,false);assert.match($('trConfigurationWarning').textContent,/仅支持相机模式/);
 assert.match($('trCurrentScene').textContent,/玻璃/);
});
test('Help opens scope and skip link target can receive focus',()=>{
 assert.equal($('trControlsTitle').tabIndex,-1);assert.equal($('trScope').open,false);d.querySelector('.tr-method-link').click();assert.equal($('trScope').open,true);
});
test('Opening comparisons/advanced sections does not alter configuration or start workers',()=>{
 const state=globalThis.coherentMeshLab.getState();for(const el of d.querySelectorAll('details'))el.open=true;
 assert.deepEqual(globalThis.coherentMeshLab.getState(),state);assert.equal(workerCount,0);
});
w.close();
console.log(JSON.stringify({passed:true,tests:count,structuralDOMValidated:true,browserLayoutValidated:false},null,2));
