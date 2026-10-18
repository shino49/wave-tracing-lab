/** Reproducible CPU-source + exact-Fraction anchored derivation checks.
 * Run: node tests/plt-resident-anchor-derivation.mjs
 * Requires Python 3; no third-party Python packages, GPU or network required.
 */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {preparePLTScene,rayNearest} from '../dist/plt-geometry.js';
import {sampleCamera,cameraFrame,randomGenerator} from '../dist/plt-core.js';
import {add,sub,scale,dot,normalize} from '../dist/mesh-geometry.js';
const materials=[{type:'mirror',albedo:.5,emission:0},{type:'mirror',albedo:.75,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true},{type:'mirror',albedo:.25,emission:0}];
function quad(axis,position,lo,hi,materialIndex){const a=(axis+1)%3,b=(axis+2)%3,point=(u,v)=>{const p=[0,0,0];p[axis]=position;p[a]=u;p[b]=v;return p;};return {mesh:{vertices:[point(lo,lo),point(hi,lo),point(hi,hi),point(lo,hi)],faces:[[0,1,2],[0,2,3]]},materialIndex};}
const out=[];
for(const cfg of [{scale:1,mirrors:1},{scale:1,mirrors:2},{scale:4,mirrors:2,shift:[-32,16,-8],perm:[2,0,1]},{scale:.5,mirrors:3,shift:[-16,8,-32],perm:[2,1,0]},{scale:8,mirrors:2,shift:[16,-16,0],perm:[1,0,2]},{scale:.25,mirrors:2,shift:[16,-8,32],perm:[1,2,0]},{scale:1/64,mirrors:2,shift:[1,-2,3],perm:[0,2,1]}]){
 const {scale:s,mirrors:m,shift=[0,0,0],perm=[0,1,2]}=cfg;
 const point=p=>perm.map((a,k)=>s*p[a]+shift[k]);
 const entries=m>=2?[quad(0,2,-8,8,0),quad(2,3,-8,8,1)]:[quad(2,2,-8,8,0)];if(m===3)entries.push(quad(1,2.5,-8,8,3));entries.push(quad(2,-2,-16,16,2));
 entries.forEach(e=>e.mesh.vertices=e.mesh.vertices.map(point));
 const camera={position:point([0,0,0]),target:point(m===3?[4,3,4]:m===2?[4,1,4]:[0,0,2]),up:perm.map(a=>[0,1,0][a]),fov:m===3?5:m===2?10:20};
 const scene=preparePLTScene(entries,{materials:structuredClone(materials)});
 for(const seed of [7391,1,0xdeadbeef])for(let pixel=0;pixel<64;pixel++){
  const rng=randomGenerator((seed^Math.imul(pixel+1,0x9e3779b1))>>>0);
  for(let sample=0;sample<3;sample++){
   const vals=[rng(),rng()];let at=0;const beam=sampleCamera(camera,pixel%8,Math.floor(pixel/8),8,8,()=>vals[at++],.025,false);
   const f=cameraFrame(camera),half=Math.tan((camera.fov*Math.PI/180)/2),u=(2*(pixel%8+vals[0])/8-1)*1*half,v=(1-2*(Math.floor(pixel/8)+vals[1])/8)*half;
   const raw=add(f.forward,add(scale(f.right,u),scale(f.up,v)));
   let O=beam.origin,D=beam.direction;const steps=[];
   for(let depth=0;depth<6;depth++){
    const h=rayNearest(scene,O,D,{minDistance:1e-7,maxDistance:Infinity});if(!h)break;
    const t=h.triangle,a=t.normal.findIndex(x=>Math.abs(x)===1),mirror=scene.materials[h.materialIndex??t.materialIndex].type==='mirror';
    steps.push({O,D,Q:normalize(D),point:h.point,t:h.distance,axis:a,plane:t.a[a],mirror});
    if(!mirror)break;
    let n=dot(t.normal,D)<0?t.normal:scale(t.normal,-1);D=sub(D,scale(n,2*dot(D,n)));O=h.point;
   }
   assert.equal(steps.filter(s=>s.mirror).length,m,'Every designed path executes the complete mirror chain');assert.equal(steps.length,m+1,'Each path terminates at the designed emitter');
   out.push({cfg,seed,pixel,sample,C:camera.position,R:raw,steps});
  }
 }
}

const pythonFile=new URL('./plt-resident-anchor-derivation.py',import.meta.url);
const result=spawnSync('python3',[fileURLToPath(pythonFile)],{input:JSON.stringify(out),encoding:'utf8',maxBuffer:1024*1024});
const files=['tests/plt-resident-anchor-derivation.mjs','tests/plt-resident-anchor-derivation.py','dist/plt-core.js','dist/mesh-geometry.js','dist/plt-geometry.js'];
const report={createdAt:new Date().toISOString(),command:'node tests/plt-resident-anchor-derivation.mjs',nodeVersion:process.version,scope:'Actual CPU camera and rayNearest source operations checked with exact Python Fraction recurrence; not GPU execution or an unconditional language-level floating-point proof',fullCPUShadowMandatory:true,actualGPUExecuted:false,physicalGPUValidated:false,browserValidated:false,hypotRelativeErrorAssumptionU64:16,geometryNormalOrZeroArithmeticRequired:true,fixtures:[...new Map(out.map(p=>[JSON.stringify(p.cfg),p.cfg])).values()],seeds:[7391,1,0xdeadbeef],sourceSHA256:Object.fromEntries(files.map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(new URL('../'+name,import.meta.url))).digest('hex')]))};
if(result.status!==0){report.allPass=false;report.failure={status:result.status,error:result.error?.message,stderr:result.stderr};}
else Object.assign(report,JSON.parse(result.stdout));
fs.writeFileSync(new URL('../research/plt-resident-anchor-derivation-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
assert(report.allPass,JSON.stringify(report.failure));
console.log(JSON.stringify(report,null,2));
