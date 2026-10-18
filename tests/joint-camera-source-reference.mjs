/** Independent actual-CPU camera/source-order oracle; no GPU arithmetic is simulated. */
import fs from 'node:fs';import assert from 'node:assert/strict';import crypto from 'node:crypto';import {spawnSync} from 'node:child_process';import {fileURLToPath} from 'node:url';
import {sampleCamera,cameraFrame} from '../dist/plt-core.js';
import {add,scale,normalize} from '../dist/mesh-geometry.js';
const fixtures=[];
for(const direction of [[4,1,4],[4,3,4],[0,0,1],[1,0,0],[1,1,1],[1e-6,1,1],[1,-1,-1],[-1,.125,4],[1,2**-1000,1]])for(const fov of [5,10,42,119.999,120])for(const [width,height] of [[1,1],[8,8],[13,7],[1,192],[256,1],[256,192]]){
 const position=direction.some(x=>x!==0&&Math.abs(x)<1e-290)?[0,0,0]:[16,-32,8],target=direction.map((x,i)=>x+position[i]);const camera={position,target,up:[0,1,0],fov};fixtures.push({camera,width,height});
}
const words=[0,1,0x0000ffff,0x00010000,0x007fffff,0x00800001,0x00ffffff,0x01000001,0x7fffffff,0x80000001,0xffffff00,0xfffffffe,0xffffffff];
const rows=[];
for(const fixture of fixtures){const {camera,width,height}=fixture,frame=cameraFrame(camera),half=Math.tan(((camera.fov*Math.PI)/180)/2),positions=[[0,0],[width-1,height-1],[Math.floor(width/2),Math.floor(height/2)]];
 for(const [x,y] of positions)for(let index=0;index<words.length;index++){
  const word0=words[index],word1=words[(index*7+3)%words.length],j0=word0/2**32,j1=word1/2**32;let at=0;const beam=sampleCamera(camera,x,y,width,height,()=>[j0,j1][at++],.025,false);
  const u=(2*(x+j0)/width-1)*(width/height)*half,v=(1-2*(y+j1)/height)*half,raw=add(frame.forward,add(scale(frame.right,u),scale(frame.up,v)));
  assert.deepEqual(normalize(raw),beam.direction,'CPU source reconstruction differs from actual sampleCamera');
  rows.push({fixture:fixtures.indexOf(fixture),width,height,x,y,word0,word1,frame,half,u,v,raw});
 }
}
const oracle=new URL('../research/joint-camera-source-reference.py',import.meta.url),result=spawnSync('python3',[fileURLToPath(oracle)],{input:JSON.stringify(rows),encoding:'utf8',maxBuffer:4*1024*1024});assert.equal(result.status,0,result.stderr);
const files=['tests/joint-camera-source-reference.mjs','research/joint-camera-source-reference.py','dist/plt-core.js','dist/mesh-geometry.js'];const report={createdAt:new Date().toISOString(),allPass:false,actualGPUExecuted:false,fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,fixtures:fixtures.length,cases:rows.length,nodeVersion:process.version,sourceSHA256:Object.fromEntries(files.map(path=>[path,crypto.createHash('sha256').update(fs.readFileSync(new URL('../'+path,import.meta.url))).digest('hex')])),...JSON.parse(result.stdout)};
fs.writeFileSync(new URL('../research/joint-camera-source-reference-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');assert(report.allPass,JSON.stringify(report.failures.slice(0,8)));console.log(JSON.stringify(report,null,2));
