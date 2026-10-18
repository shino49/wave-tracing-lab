import assert from 'node:assert/strict';
import fs from 'node:fs';
import {prepareCoherentMeshScene,compileCoherentSource,sumCoherentRoutes} from '../dist/coherent-mesh-core.js';
import {evaluateMeshTransport,coherentRouteCapacity} from '../dist/coherent-mesh-transport.js';
import {evaluateMixedDiffraction} from '../dist/coherent-mesh-mixed.js';
import {executeCoherentMeshTask,prepareCoherentMeshTask} from '../dist/coherent-mesh-runtime.js';
const quad=(v,m)=>({mesh:{vertices:v,faces:[[0,1,2],[0,2,3]]},materialIndex:m,surface:'open-plate'});
const entries=[quad([[0,-100,0],[100,-100,0],[100,100,0],[0,100,0]],0),quad([[-6,-100,-6],[-.1,-100,-6],[-.1,100,-6],[-6,100,-6]],1)];
const materials=[{type:'mirror',reflectance:1,phaseRad:Math.PI},{type:'mirror',reflectance:.64,phaseRad:.37}];
const scene=prepareCoherentMeshScene(entries,materials,{maxReflections:2}),source={position:[-3,1.3,-4],wavelengthMm:.05,power:2.7},receiver=[-2,-.7,10];
const field=r=>[r.amplitude*Math.cos(r.phaseRadReduced),r.amplitude*Math.sin(r.phaseRadReduced)];
const near=(a,b,t=2e-12)=>assert(Math.abs(a-b)<t,`${a} != ${b}`);
let groups=0;const test=async(n,f)=>{await f();groups++;console.log('PASS',n)};
await test('RD and DR match independent SciPy complex field, finite spreading and physical geometry',()=>{
 const routes=evaluateMixedDiffraction(scene,compileCoherentSource(scene,source),receiver).routes;
 const gold={RD:{q:[0,.3882526681827189,0],b:[-2.25,1.0720631670456797,-6],ri:8.59251320610425,ro:10.255939443552053,L:4.622771177174412,u:[.0007533007156798428,.00006881679359350906]},DR:{q:[0,.9308699344627533,0],b:[-.5454545454545454,.4860872250638207,-6],ri:5.013607184980045,ro:22.15084054258742,L:4.066109176214920,u:[-.00015792341391640172,-.0004329512058896184]}};
 for(const order of ['RD','DR']){const r=routes.find(r=>r.interactionOrder===order&&Math.abs(r.point[0])<1e-9&&Math.abs(r.point[2])<1e-9),g=gold[order];assert(r);for(const key of ['ri','ro','L'])near(r[key],g[key]);r.point.forEach((x,i)=>near(x,g.q[i]));r.familyData.mirrorPoint.forEach((x,i)=>near(x,g.b[i]));field(r).forEach((x,i)=>near(x,g.u[i]));}
});
await test('source phase applied once and source strength keeps square-root amplitude law',()=>{
 const base=evaluateMixedDiffraction(scene,compileCoherentSource(scene,source),receiver).routes;
 const changed=evaluateMixedDiffraction(scene,compileCoherentSource(scene,{...source,power:4*source.power,phaseRad:.61}),receiver).routes;
 assert.equal(base.length,changed.length);base.forEach((r,i)=>{near(changed[i].amplitude,2*r.amplitude);const a=field(r),b=field(changed[i]);near(b[0],2*(a[0]*Math.cos(.61)-a[1]*Math.sin(.61)));near(b[1],2*(a[0]*Math.sin(.61)+a[1]*Math.cos(.61)));});
});
await test('capacity accounts for all eligible mixed families without silent truncation',()=>{
 const q=coherentRouteCapacity(scene,{diffraction:true,mixedDiffraction:true});assert.equal(q.count,scene.routeSequences.length+q.diffraction.eligible.length+q.mixed.count);assert(q.mixed.count>0);
 assert.throws(()=>coherentRouteCapacity(scene,{diffraction:false,mixedDiffraction:true}));
 const noGO=prepareCoherentMeshScene(entries,materials,{maxReflections:1});assert.throws(()=>coherentRouteCapacity(noGO,{diffraction:true,mixedDiffraction:true}));
});
await test('new actual mesh scene sends reflected illumination into geometric shadow',()=>{
 const p=prepareCoherentMeshTask({preset:'reflected-edge'}),c=compileCoherentSource(p.scene,p.source),receiver=[18,0,10];
 const enabled=evaluateMeshTransport(p.scene,c,receiver,{diffraction:true,mixedDiffraction:true}),disabled=evaluateMeshTransport(p.scene,c,receiver,{diffraction:true});
 assert(enabled.routes.some(r=>r.kind==='mixed-diffraction'&&r.interactionOrder==='RD'));assert.equal(enabled.geometricPower,0);assert(sumCoherentRoutes(enabled.routes).coherent>1e5*sumCoherentRoutes(disabled.routes).coherent);
 assert(!evaluateMeshTransport(p.scene,c,receiver,{diffraction:true,mixedDiffraction:true,blockReflected:true}).routes.some(r=>r.kind==='mixed-diffraction'));
});
await test('reflection ablation counts only active families rather than rejecting unused capacity',()=>{
 const q=(v)=>quad(v,0),s=prepareCoherentMeshScene([q([[1,2,0],[10,2,0],[10,11,0],[1,11,0]]),q([[1,-3,2],[10,-3,2],[10,-3,11],[1,-3,11]]),q([[-4,2,2],[-4,11,2],[-4,11,11],[-4,2,11]])],[{type:'mirror',reflectance:1,phaseRad:Math.PI}],{maxReflections:2});
 assert.throws(()=>coherentRouteCapacity(s,{diffraction:true,mixedDiffraction:true}),/65/);
 const c=coherentRouteCapacity(s,{diffraction:true,mixedDiffraction:true,blockReflected:true});assert.equal(c.mixed,null);assert.equal(c.count,1+c.diffraction.eligible.length);assert(c.count<65);
 assert.throws(()=>evaluateMeshTransport(s,compileCoherentSource(s,{position:[-1,-1,-1]}),[1,1,1],{mixedDiffraction:true}),/requires/);
});
await test('runtime exports correct mixed counts and fixed native grid tiles',async()=>{
 let tiles=0;const r=await executeCoherentMeshTask({preset:'reflected-edge',resolution:16,backend:'cpu'},{preview:p=>{assert(!('scene'in p.configuration));tiles++;}});
 assert.equal(r.width,16);assert.equal(tiles,2);assert(r.diagnostics.mixedRoutes>0);assert.equal(r.diagnostics.reflectionDiffractionRoutes,r.diagnostics.mixedRoutes);assert.equal(r.diagnostics.diffractionReflectionRoutes,0);assert.equal(r.diagnostics.productionCPUShadow,false);assert(r.coherent.some(x=>x>0));
});
await test('pre-aborted render never starts setup, even with malformed scene data',async()=>{
 const abort=new AbortController();abort.abort(new DOMException('Cancelled','AbortError'));let setup=0;
 await assert.rejects(executeCoherentMeshTask({scene:{invalid:true}},{signal:abort.signal,progress:()=>setup++}),{name:'AbortError'});assert.equal(setup,0);
});
fs.writeFileSync('research/coherent-mesh-v30/mixed-validation.json',JSON.stringify({groups,passed:true,independentOracle:'SciPy equations in coherent-mesh-mixed-physics-review.md'},null,2)+'\n');
console.log(groups,'mixed coherent mesh groups passed');
