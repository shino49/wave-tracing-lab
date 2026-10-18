import assert from 'node:assert/strict';
import {prepareCoherentMeshScene,compileCoherentSource,evaluateGeometricRoutes,sumCoherentRoutes,makeCoherentMeshPreset,detectorPoint,validateDetector} from '../dist/coherent-mesh-core.js';
import {prepareCoherentMeshTask,executeCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
const close=(a,b,t=1e-10)=>assert(Math.abs(a-b)<=t*Math.max(1e-15,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const test=(name,f)=>{f();console.log('PASS',name);};
function state(preset='lloyd',options={}){const p=makeCoherentMeshPreset(preset,options),scene=prepareCoherentMeshScene(p.entries,p.materials,{maxReflections:2}),source={...p.source,power:1,wavelengthMm:.00055};return {scene,source,compiled:compileCoherentSource(scene,source)};}
const monopole=(s,d,phase=0)=>{const r=Math.hypot(...d.map((x,i)=>x-s[i])),a=1/(Math.sqrt(4*Math.PI)*r),p=phase-2*Math.PI*r/.00055;return [a*Math.cos(p),a*Math.sin(p)];};
test('Lloyd mirror equals independent image-source field, including absolute spreading',()=>{
 const s=state();for(const d of [[0,.2,50],[1,2,50],[-2,5,50]]){const routes=evaluateGeometricRoutes(s.scene,s.compiled,d).routes;assert.equal(routes.length,2);const u=monopole(s.source.position,d),v=monopole([0,-.08,-50],d,Math.PI),expected=(u[0]+v[0])**2+(u[1]+v[1])**2;close(sumCoherentRoutes(routes).coherent,expected,2e-9);}
});
test('perpendicular mirror geometry chooses one double-bounce ordering, not both',()=>{
 const s=state('two-mirror');for(const d of [[1,2,50],[2,1,50],[4,2.3,50],[2,2,50],[2+1e-8,2,50],[2-1e-8,2,50]]){const routes=evaluateGeometricRoutes(s.scene,s.compiled,d).routes;assert.equal(routes.length,4);assert.equal(routes.filter(r=>r.sequence.length===2).length,1);let re=0,im=0;for(const x of [-.08,.08])for(const y of [-.08,.08]){const phase=(x<0?Math.PI:0)+(y<0?Math.PI:0),u=monopole([x,y,-50],d,phase);re+=u[0];im+=u[1];}close(sumCoherentRoutes(routes).coherent,re*re+im*im,2e-8);}
});
test('phase flip changes cross-term sign; randomized control and source scaling',()=>{
 const a=state(),b=state('lloyd',{phaseShiftRad:-Math.PI}),d=[1,2,50];const ra=evaluateGeometricRoutes(a.scene,a.compiled,d).routes,rb=evaluateGeometricRoutes(b.scene,b.compiled,d).routes,va=sumCoherentRoutes(ra),vb=sumCoherentRoutes(rb);close(va.crossTerm,-vb.crossTerm,1e-10);close(va.incoherent,vb.incoherent);assert(sumCoherentRoutes(ra,{coherence:0}).crossTerm===0);
 const twice=compileCoherentSource(a.scene,{...a.source,power:2});close(sumCoherentRoutes(evaluateGeometricRoutes(a.scene,twice,d).routes).coherent,2*va.coherent);
 const only=evaluateGeometricRoutes(a.scene,a.compiled,d,{blockReflected:true}).routes;assert.equal(only.length,1);close(sumCoherentRoutes(only).coherent,sumCoherentRoutes(only).incoherent);
});
test('triangle subdivisions do not duplicate a physical mirror route',()=>{
 const vertices=[[-10,0,-49],[10,0,-49],[10,0,49],[-10,0,49],[0,0,0]],materials=[{type:'mirror',reflectance:1,phaseRad:Math.PI}];
 const scene=prepareCoherentMeshScene([{mesh:{vertices,faces:[[0,1,4],[1,2,4],[2,3,4],[3,0,4]]}}],materials);assert.equal(scene.mirrorPlanes.length,1);assert.equal(scene.routeSequences.length,2);const c=compileCoherentSource(scene,{position:[0,.08,-50],power:1,wavelengthMm:.00055});assert.equal(evaluateGeometricRoutes(scene,c,[0,3,50]).routes.length,2);
});
test('source/geometry/receiver scaling preserves phase and inverse-square density',()=>{
 const a=state(),d=[1,2,50],data=makeCoherentMeshPreset();const entries=data.entries.map(e=>({...e,mesh:{vertices:e.mesh.vertices.map(v=>v.map(x=>x*4)),faces:e.mesh.faces}}));const scene=prepareCoherentMeshScene(entries,data.materials),c=compileCoherentSource(scene,{position:a.source.position.map(x=>x*4),power:1,wavelengthMm:.0022});const va=sumCoherentRoutes(evaluateGeometricRoutes(a.scene,a.compiled,d).routes),vb=sumCoherentRoutes(evaluateGeometricRoutes(scene,c,d.map(x=>x*4)).routes);close(vb.coherent,va.coherent/16,1e-10);
});
test('separate coplanar mesh entries and T-junctions are not physical rims',()=>{
 const vertices=[[-2,0,-2],[2,0,-2],[2,0,2],[-2,0,2]],materials=[{type:'mirror',reflectance:1,phaseRad:Math.PI}],src={position:[0,1,-1],wavelengthMm:.00055};
 const a=prepareCoherentMeshScene([{mesh:{vertices,faces:[[0,1,2],[0,2,3]]}}],materials),b=prepareCoherentMeshScene([{mesh:{vertices,faces:[[0,1,2]]}},{mesh:{vertices,faces:[[0,2,3]]}}],materials);
 const ar=evaluateGeometricRoutes(a,compileCoherentSource(a,src),[0,1,1]).routes,br=evaluateGeometricRoutes(b,compileCoherentSource(b,src),[0,1,1]).routes;close(ar[1].amplitude,br[1].amplitude);assert(!br[1].boundary);
 const middle=[0,0,0],c=prepareCoherentMeshScene([{mesh:{vertices:[...vertices,middle],faces:[[0,1,2]]}},{mesh:{vertices:[...vertices,middle],faces:[[0,4,3],[4,2,3]]}}],materials),cr=evaluateGeometricRoutes(c,compileCoherentSource(c,src),[0,1,1]).routes;close(ar[1].amplitude,cr[1].amplitude);assert(!cr[1].boundary);
});
test('opaque mesh actually blocks a geometrical segment',()=>{
 const p=makeCoherentMeshPreset();p.entries.push({mesh:{vertices:[[-1,-1,0],[1,-1,0],[1,2,0],[-1,2,0]],faces:[[0,1,2],[0,2,3]]},materialIndex:0});const s=prepareCoherentMeshScene(p.entries,p.materials),c=compileCoherentSource(s,{position:p.source.position,wavelengthMm:.00055});assert.equal(evaluateGeometricRoutes(s,c,[0,1,50]).routes.length,0);
});
test('unsupported material / geometry capacity / detector axes fail explicitly',()=>{
 assert.throws(()=>prepareCoherentMeshScene(makeCoherentMeshPreset().entries,[{type:'diffuse'}]),/rough|diffuse/);assert.throws(()=>validateDetector({center:[0,0,0],u:[1,0,0],v:[1,0,0],width:1,height:1}),/orthogonal/);
 const s=state();assert.throws(()=>compileCoherentSource(s.scene,{position:[0,0,NaN]}));assert.throws(()=>sumCoherentRoutes([{amplitude:Infinity,phaseRadReduced:0}]));
});
const r=await executeCoherentMeshTask({resolution:32,backend:'cpu'});assert(r.complete&&r.coherent.length===1024&&r.diagnostics.productionCPUShadow===false);for(let i=0;i<1024;i++)close(r.coherent[i]-r.incoherent[i],r.crossTerm[i],1e-9);console.log('PASS complete image / cross-term / native size');
const controller=new AbortController();let partial=0;await assert.rejects(executeCoherentMeshTask({resolution:32,backend:'cpu'},{signal:controller.signal,preview(){partial++;controller.abort();}}),e=>e.name==='AbortError');assert.equal(partial,1);console.log('PASS cancellation never returns complete image');
console.log('10 coherent mesh core groups passed');
