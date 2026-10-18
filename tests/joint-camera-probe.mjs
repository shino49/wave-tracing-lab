/** Focused native Q44 camera/ratio probe. No transport admission. */
import assert from 'node:assert/strict';import fs from 'node:fs';import crypto from 'node:crypto';import {create,globals} from 'webgpu';
import {packResidentCamera} from '../dist/plt-resident-camera.js';import {cameraFrame} from '../dist/plt-core.js';Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert.equal(backend,'vulkan');
const baseFiles=['plt-resident-interval.wgsl','plt-resident-geometry.wgsl','plt-resident-axis-guard.wgsl'];
const source=baseFiles.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8')).join('\n');
const report={createdAt:new Date().toISOString(),actualGPUExecuted:false,allPass:false,fullCPUShadowMandatory:true,backend,filesSHA256:Object.fromEntries(baseFiles.map(f=>[f,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+f,import.meta.url))).digest('hex')])),mutations:[]};
const path=new URL('../research/joint-camera-probe-'+(execute?'validation':'compile-validation')+'.json',import.meta.url),save=()=>fs.writeFileSync(path,JSON.stringify(report,null,2)+'\n');
const dv=new DataView(new ArrayBuffer(8));const value=b=>{dv.setUint32(0,b,true);return dv.getFloat32(0,true);};
const F=(n,d=1n)=>({n,d}),add=(a,b)=>F(a.n*b.d+b.n*a.d,a.d*b.d),neg=a=>F(-a.n,a.d),mul=(a,b)=>F(a.n*b.n,a.d*b.d),div=(a,b)=>{const s=b.n<0n?-1n:1n;return F(a.n*b.d*s,a.d*b.n*s);},cmp=(a,b)=>a.n*b.d<b.n*a.d?-1:a.n*b.d>b.n*a.d?1:0;
function exact(x){if(x===0)return F(0n);dv.setFloat64(0,x);const u=dv.getBigUint64(0),e=Number(u>>52n&2047n)-1075,m=((u&((1n<<52n)-1n))|(1n<<52n))*(u>>63n?-1n:1n);return e>=0?F(m<<BigInt(e)):F(m,1n<<BigInt(-e));}
const signed=(lo,hi)=>BigInt.asIntN(64,BigInt(lo)|(BigInt(hi)<<32n));
let seed=7391;const word=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;};const rows=[];
for(let n=0;n<384;n++){
 const sizes=[[8,8],[64,16],[16,64],[256,192],[3,7],[7,3]], [width,height]=sizes[n%sizes.length],position=[0,0,0],target=[(word()/2**32-.5)*2,(word()/2**32-.5)*2,.2+word()/2**32],camera={position,target,fov:n%5===0?120:n%5===1?5:5+115*word()/2**32};
 const edges=[0,1,0x7fffffff,0x80000000,0xfffffffe,0xffffffff],w0=n<36?edges[n%6]:word(),w1=n<36?edges[Math.floor(n/6)]:word(),pixel=word()%(width*height),parameters=new Uint32Array(packResidentCamera(camera,{width,height}));assert.equal(parameters[26],1);
 const x=pixel%width,y=Math.floor(pixel/width),frame=cameraFrame(camera),half=Math.tan(((camera.fov*Math.PI/180)/2)),u=(2*(x+w0/2**32)/width-1)*(width/height)*half,v=(1-2*(y+w1/2**32)/height)*half,raw=frame.forward.map((f,i)=>f+(frame.right[i]*u+frame.up[i]*v));
 rows.push({name:'camera-'+n,camera,width,height,w0,w1,pixel,parameters,raw});
}
// Actual exact-axis frames and fail-closed domain/corruption cases.
for(const target of [[0,0,1],[1,0,0],[0,1,0]]){const camera={position:[0,0,0],target,up:target[1]?[0,0,1]:[0,1,0],fov:5},width=8,height=8,parameters=new Uint32Array(packResidentCamera(camera,{width,height})),pixel=19,w0=12345678,w1=87654321,frame=cameraFrame(camera),half=Math.tan(((camera.fov*Math.PI/180)/2)),u=(2*(3+w0/2**32)/8-1)*half,v=(1-2*(2+w1/2**32)/8)*half;rows.push({name:'axis-'+target,camera,width,height,w0,w1,pixel,parameters,raw:frame.forward.map((f,i)=>f+(frame.right[i]*u+frame.up[i]*v))});}
for(const kind of ['unsupported-aspect','bad-pixel','missing-valid','coefficient-overflow','bad-zero-mask']){const r=structuredClone(rows[0]);r.name=kind;r.parameters=new Uint32Array(r.parameters);r.invalid=true;if(kind==='unsupported-aspect')r.parameters=new Uint32Array(packResidentCamera(r.camera,{width:256,height:1}));if(kind==='bad-pixel')r.pixel=r.width*r.height;if(kind==='missing-valid')r.parameters[26]=0;if(kind==='coefficient-overflow')r.parameters[1]=0x7fffffff;if(kind==='bad-zero-mask')r.parameters[27]=7;rows.push(r);}
report.cases=rows.length;
const harness=`struct CameraInput {camera:RACameraParameters, data:vec4u}
@group(0) @binding(2) var<storage,read> inputs:array<CameraInput>;
@group(0) @binding(3) var<storage,read_write> outputs:array<vec4u>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 let p=inputs[id.x];ra_bad=false;let s=ra_camera_initialize(p.camera,p.data.x,p.data.y,p.data.z);let o=id.x*9u;
 outputs[o]=bitcast<vec4u>(vec4f(s.raw.x.lo,s.raw.x.hi,s.raw.y.lo,s.raw.y.hi));outputs[o+1u]=vec4u(bitcast<u32>(s.raw.z.lo),bitcast<u32>(s.raw.z.hi),s.valid,select(0u,1u,ra_bad));
 for(var k=0u;k<3u;k++){for(var i=0u;i<3u;i++){if(i==k){continue;}let index=k*2u+i-select(0u,1u,i>k);let r=s.ratios[index];outputs[o+2u+index]=bitcast<vec4u>(vec4f(r.center,r.residual.lo,r.residual.hi,0));}}
 outputs[o+8u]=vec4u(0xa71c2026u,0u,0u,0u);
}`;
let gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();
async function run(src){const start=performance.now(),module=device.createShaderModule({code:src+harness}),info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(x=>x.type==='error'),[]);const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}),compileMs=performance.now()-start;if(!execute)return {compileMs};
 const data=new Uint32Array(rows.length*32);rows.forEach((r,i)=>{data.set(r.parameters,i*32);data.set([r.pixel,r.w0,r.w1,0],i*32+28);});const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:rows.length*144,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:rows.length*144,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
 try{device.queue.writeBuffer(input,0,data);const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:input}},{binding:3,resource:{buffer:output}}]}),enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(rows.length);pass.end();enc.copyBufferToBuffer(output,0,read,0,rows.length*144);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange().slice(0));read.unmap();return {words,compileMs};}finally{input.destroy();output.destroy();read.destroy();}}
function inspect(words){const failures=[];let checks=0,accepted=0,unknown=0;const widths=[];
 for(let r=0;r<rows.length;r++){const row=rows[r],o=r*36;assert.equal(words[o+32],0xa71c2026);if(row.invalid){if(words[o+6]!==0)failures.push({name:row.name,reason:'invalid camera accepted'});continue;}if(words[o+6]!==1){unknown++;continue;}accepted++;
  for(let i=0;i<3;i++){const raw=exact(row.raw[i]),lo=exact(value(words[o+2*i])),hi=exact(value(words[o+2*i+1]));if(cmp(lo,raw)>0||cmp(raw,hi)>0)failures.push({name:row.name,reason:'CPU source raw outside',i,raw:row.raw[i],lo:value(words[o+2*i]),hi:value(words[o+2*i+1])});checks++;}
  for(let k=0;k<3;k++)for(let i=0;i<3;i++){if(i===k||row.raw[k]===0)continue;const index=k*2+i-Number(i>k),p=o+8+index*4,center=exact(value(words[p])),lo=add(center,exact(value(words[p+1]))),hi=add(center,exact(value(words[p+2]))),ratio=div(exact(row.raw[i]),exact(row.raw[k]));if(cmp(lo,ratio)>0||cmp(ratio,hi)>0)failures.push({name:row.name,reason:'CPU raw ratio outside',i,k,center:value(words[p]),residual:[value(words[p+1]),value(words[p+2])],truth:row.raw[i]/row.raw[k]});checks++;widths.push(value(words[p+2])-value(words[p+1]));}
 }
 widths.sort((a,b)=>a-b);return {checks,accepted,unknown,failures,medianRatioResidualWidth:widths[widths.length>>1],maximumRatioResidualWidth:widths.at(-1)};}
try{save();const output=await run(source);report.compileMs=output.compileMs;if(execute){report.actualGPUExecuted=true;Object.assign(report,inspect(output.words));assert.equal(report.failures.length,0,JSON.stringify(report.failures.slice(0,10)));assert(report.accepted>rows.length*.95);for(const [name,from,to]of [
 ['drop-jitter-low-bits','rac_scaled_jitter(a,word0)','rac_scaled_jitter(a,word0&0xffff0000u)'],
 ['drop-ratio-residual','return RACameraRatio(center,residual);','return RACameraRatio(center,RAI(0,0));'],
 ['drop-camera-error-and-quotient-remainder','var residual=ra_iadd(lowPart,RAI(0.0,0x1p-32));','var residual=lowPart;'],
 ]){let altered=source.replace(from,to);if(name==='drop-camera-error-and-quotient-remainder')altered=altered.replace('residual=ra_inflate(residual,correction);','residual=residual;');assert.notEqual(altered,source);const result=inspect((await run(altered)).words);assert(result.failures.length>0,'Mutation survived '+name);report.mutations.push({name,rejected:true,failures:result.failures.length,first:result.failures[0]});save();}}
 report.allPass=true;console.log(JSON.stringify(report,null,2));}catch(error){report.failure={message:error.message,stack:error.stack};throw error;}finally{save();device.destroy();gpu=null;}
