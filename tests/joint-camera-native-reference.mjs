/** Independent exact-Fraction oracle for actual native Q44 camera limb/ratio outputs. */
import fs from 'node:fs';import assert from 'node:assert/strict';import crypto from 'node:crypto';import {spawnSync} from 'node:child_process';import {fileURLToPath} from 'node:url';
import {create,globals} from 'webgpu';import {packResidentCamera} from '../dist/plt-resident-camera.js';import {cameraFrame,sampleCamera} from '../dist/plt-core.js';import {add,scale,normalize} from '../dist/mesh-geometry.js';Object.assign(globalThis,globals);
const execute=process.argv.includes('--execute'),mutate=process.argv.includes('--mutate'),backend=execute?process.env.WEBGPU_EXECUTION_BACKEND:'null';if(execute)assert.equal(backend,'vulkan');
const files=['plt-resident-interval.wgsl','plt-resident-geometry.wgsl','plt-resident-axis-guard.wgsl'],texts=files.map(f=>fs.readFileSync(new URL('../dist/'+f,import.meta.url),'utf8')),source=texts.join('\n');
const pair=x=>{const b=BigInt.asUintN(64,BigInt(x));return[Number(b&0xffffffffn),Number(b>>32n)];};
const rows=[],put=(name,op,a=0n,b=0n,extra={})=>rows.push({name,op,a:pair(a),b:pair(b),parameters:Array(28).fill(0),pixel:0,word0:0,word1:0,ne:512,de:512,signMask:0,scale:44,...extra});
const edges=[0n,1n,65535n,65536n,0x7fffffffn,0x80000000n,0xfffffffen,0xffffffffn];
for(const a of edges)for(const b of edges)put('mul32-edges',0,a,b,{mustValid:true});
let state=0xc477119b;const rand=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return state>>>0;};
for(let i=0;i<320;i++){const a=BigInt(rand()),b=BigInt(rand());put('mul32-random',0,a,b,{mustValid:true});const n=(BigInt(rand()&0x3ffff)<<32n)|BigInt(rand()),signed=i%2?-n:n;put('jitter-signed-random',2,signed,b,{mustValid:true});put('integer-signed-random',1,signed,BigInt(i%257),{mustValid:true});put('integer-to-directed-bounds',3,signed,0n,{mustValid:true});put('ratio-signed-random',5,signed,(i%3?-1n:1n)*((BigInt(rand()&0x3ffff)<<32n)|0x10000000n|BigInt(rand())),{mustValid:true});}
for(const a of [0n,1n,255n,256n,511n,512n,513n,(1n<<24n)-1n,1n<<24n,(1n<<32n)-1n,1n<<32n,(1n<<49n)-1n])for(const b of [1n<<28n,(1n<<28n)+1n,3n<<28n,(1n<<48n)-1n]){put('quotient-remainder-edges',4,a,b,{mustValid:true});put('ratio-remainder-edges',5,a,b,{mustValid:true});}
for(const a of [(1n<<31n)-1n,1n<<32n,(1n<<32n)-1n,-1n,-(1n<<32n),(1n<<62n)-1n])for(const b of [1n,-1n,(1n<<32n)+1n])put('signed-add-carry-borrow',7,a,b,{mustValid:true});
put('integer-overflow',1,1n<<62n,4n,{mustInvalid:true});put('signed-add-overflow',7,(1n<<63n)-1n,1n,{mustInvalid:true});put('small-ratio-denominator',5,1n,1n,{mustInvalid:true});put('zero-ratio-denominator',5,1n,0n,{mustInvalid:true});put('sticky-ratio',5,1n<<44n,3n<<44n,{sticky:true,mustInvalid:true});
for(const extra of [{ne:513},{de:513}])put('unsupported-error-radius',5,1n<<44n,3n<<44n,{...extra,mustInvalid:true});
put('unsupported-numerator-magnitude',5,1n<<50n,3n<<44n,{mustInvalid:true});put('unsupported-denominator-magnitude',5,1n<<44n,1n<<50n,{mustInvalid:true});
const words=[0,1,0xffff,0x1000001,0x7fffffff,0x80000001,0xfffffffe,0xffffffff];
for(const direction of [[4,1,4],[4,3,4],[0,0,1],[1,1,1],[-1,.125,4],[1e-4,1,1],[1,2**-1000,1]])for(const fov of [5,120])for(const [width,height] of [[8,8],[13,7],[48,192],[256,64]]){
 const position=direction.some(x=>x!==0&&Math.abs(x)<1e-290)?[0,0,0]:[16,-32,8],camera={position,target:direction.map((x,k)=>position[k]+x),up:[0,1,0],fov},frame=cameraFrame(camera),half=Math.tan(((fov*Math.PI/180)/2)),aspect=width/height,parameters=Array.from(new Uint32Array(packResidentCamera(camera,{width,height})));
 assert.equal(parameters[26],1);
 for(const [x,y] of [[0,0],[width-1,height-1],[Math.floor(width/2),Math.floor(height/2)]])for(let i=0;i<words.length;i++){
  const word0=words[i],word1=words[(i*3+1)%words.length],j0=word0/2**32,j1=word1/2**32,u=(2*(x+j0)/width-1)*aspect*half,v=(1-2*(y+j1)/height)*half,raw=add(frame.forward,add(scale(frame.right,u),scale(frame.up,v)));let at=0;assert.deepEqual(normalize(raw),sampleCamera(camera,x,y,width,height,()=>[j0,j1][at++],.025,false).direction);
  put('actual-camera-source',6,0n,0n,{parameters,pixel:y*width+x,word0,word1,signMask:i,reference:{width,height,x,y,frame,half,aspect,raw}});
 }
}
for(const sign of [1n,-1n])for(const n of [1n,5n,7n])put('exact-ratio-floor-remainder',5,sign*(n<<44n),3n<<44n,{ne:0,de:0,mustValid:true});
const sample=rows.find(r=>r.op===6);for(const [name,change] of [['invalid-marker',p=>p[26]=0],['invalid-zero-mask',p=>p[27]=8],['unsupported-coefficient',p=>p[1]=0x40000000],['false-exact-zero',p=>p[27]=7]]){const parameters=sample.parameters.slice();change(parameters);put(name,6,0n,0n,{parameters,pixel:sample.pixel,word0:sample.word0,word1:sample.word1,mustInvalid:true});}
put('invalid-pixel',6,0n,0n,{parameters:sample.parameters,pixel:sample.parameters[24]*sample.parameters[25],mustInvalid:true});
const harness=`
struct Input { header:vec4u, a:vec4u, control:vec4u, p:RACameraParameters }
@group(0) @binding(2) var<storage,read> inputs:array<Input>;
@group(0) @binding(3) var<storage,read_write> outputs:array<u32>;
fn out_pair(base:u32,at:u32,v:vec2u){outputs[base+at]=v.x;outputs[base+at+1u]=v.y;}
fn out_interval(base:u32,at:u32,v:RAI){out_pair(base,at,bitcast<vec2u>(vec2f(v.lo,v.hi)));}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let p=inputs[id.x];let base=id.x*48u;ra_bad=(p.control.z&256u)!=0u;var valid=1u;let a=p.a.xy;let b=p.a.zw;
 if(p.header.x==0u){out_pair(base,4u,rac_mul32(a.x,b.x));}
 else if(p.header.x==1u){out_pair(base,4u,rac_scaled_integer(a,b.x));}
 else if(p.header.x==2u){out_pair(base,4u,rac_scaled_jitter(a,b.x));}
 else if(p.header.x==3u){out_interval(base,4u,rac_signed_bounds(a,p.control.w));}
 else if(p.header.x==4u){out_pair(base,4u,rac_quotient(a,b));}
 else if(p.header.x==5u){let r=rac_make_ratio(a,b,p.control.x,p.control.y);outputs[base+4u]=bitcast<u32>(r.center);out_interval(base,5u,r.residual);}
 else if(p.header.x==7u){out_pair(base,4u,rac_signed_add(a,b));}
 else{let s=ra_camera_initialize(p.p,p.header.y,p.header.z,p.header.w);valid=s.valid;if(valid==1u){for(var i=0u;i<3u;i++){out_pair(base,4u+i*2u,rac_raw_integer(p.p,i,p.header.y%p.p.metadata.x,p.header.y/p.p.metadata.x,p.header.z,p.header.w));out_interval(base,10u+i*2u,ra_component(s.raw,i));}for(var k=0u;k<3u;k++){for(var i=0u;i<3u;i++){if(i==k){continue;}let index=k*2u+i-select(0u,1u,i>k);let d=ra_component(s.raw,k);if(d.lo==0.0&&d.hi==0.0){continue;}let r=ra_camera_ratio(s,p.control.z&7u,i,k);outputs[base+16u+index*5u]=bitcast<u32>(r.center);out_interval(base,17u+index*5u,r.residual);out_interval(base,19u+index*5u,r.full);}}}}
 outputs[base]=select(0u,1u,ra_bad);outputs[base+1u]=valid;outputs[base+2u]=p.header.x;outputs[base+3u]=0xca221122u;
}`;
const sha=s=>crypto.createHash('sha256').update(s).digest('hex'),report={createdAt:new Date().toISOString(),backend,allPass:false,actualGPUExecuted:false,executionRequested:execute,fullCPUShadowMandatory:true,physicalGPUValidated:false,browserValidated:false,cases:rows.length,sourceSHA256:Object.fromEntries(files.map((f,i)=>[f,sha(texts[i])])),hostSHA256:sha(fs.readFileSync(new URL('../dist/plt-resident-camera.js',import.meta.url))),testSHA256:sha(fs.readFileSync(new URL(import.meta.url))),oracleSHA256:sha(fs.readFileSync(new URL('../research/joint-camera-native-reference.py',import.meta.url))),mutations:[]};
const reportPath=new URL('../research/joint-camera-native-reference-'+(execute?'validation':'compile-validation')+'.json',import.meta.url),save=()=>fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');let gpu=create([`backend=${backend}`]);const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice();report.adapter={description:adapter.info.description,vendor:adapter.info.vendor,device:adapter.info.device};
async function compile(src){const start=performance.now(),module=device.createShaderModule({code:src+harness});assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==='error'),[]);const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});return{pipeline,compileMs:performance.now()-start};}
async function run(pipeline){const words=new Uint32Array(rows.length*40);rows.forEach((r,i)=>words.set([r.op,r.pixel,r.word0,r.word1,...r.a,...r.b,r.ne,r.de,r.signMask|(r.sticky?256:0),r.scale,...r.parameters],i*40));const input=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:rows.length*192,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:rows.length*192,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{device.queue.writeBuffer(input,0,words);const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:input}},{binding:3,resource:{buffer:output}}]}),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(rows.length);pass.end();encoder.copyBufferToBuffer(output,0,read,0,rows.length*192);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const payload=rows.map((r,i)=>({...r,output:Array.from(values.slice(i*48,i*48+48))})),result=spawnSync('python3',[fileURLToPath(new URL('../research/joint-camera-native-reference.py',import.meta.url))],{input:JSON.stringify(payload),encoding:'utf8',maxBuffer:4*1024*1024});assert.equal(result.status,0,result.stderr);return JSON.parse(result.stdout);}finally{input.destroy();output.destroy();read.destroy();}}
try{const baseline=await compile(source);report.compileMs=baseline.compileMs;save();if(execute){report.baseline=await run(baseline.pipeline);report.actualGPUExecuted=true;save();assert.equal(report.baseline.failureCount,0,JSON.stringify(report.baseline.failures.slice(0,10)));assert(report.baseline.cameraAccepted>800,'camera corpus unexpectedly mostly rejected');if(mutate){const mutations=[
 ['drop-low-eight-jitter-bits',s=>s.replace('fn rac_scaled_jitter(a:vec2u,b:u32)->vec2u {\n let p=rac_mul64_32(rac_abs(a),b);','fn rac_scaled_jitter(a:vec2u,b:u32)->vec2u {\n let p=rac_mul64_32(rac_abs(a),b&0xffffff00u);')],
 ['drop-camera-error-envelope',s=>s.replace('errors[i]=512u;','errors[i]=0u;')],
 ['drop-ratio-floor-remainder',s=>s.replace('var residual=ra_iadd(lowPart,RAI(0.0,0x1p-32));','var residual=lowPart;')],
 ['drop-ratio-perturbation',s=>s.replace('residual=ra_inflate(residual,correction);','residual=residual;')],
 ['drop-argument-range-check',s=>s.replace('if(numeratorError>512u||denominatorError>512u||n.y>=0x00040000u||d.y>=0x00040000u)', 'if(false)')],
 ['drop-integer-carry',s=>s.replace('a.y+b.y+select(0u,1u,low<a.x)','a.y+b.y')]
 ];for(const[name,change]of mutations){const altered=change(source);assert.notEqual(altered,source,name);const compiled=await compile(altered),result=await run(compiled.pipeline);assert(result.failureCount>0,name+' survived');report.mutations.push({name,rejected:true,compileMs:compiled.compileMs,failureCount:result.failureCount,firstFailure:result.failures[0]});save();}}}report.allPass=true;console.log(JSON.stringify(report,null,2));}finally{save();device.destroy();gpu=null;}
