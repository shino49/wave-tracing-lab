/* Bounded deterministic receiver sweep. CPU is a labelled reference backend.
 * GPU buffers and runtime parity are implemented in a separate adapter below.
 * A cancelled/obsolete run is never returned as a successful image. */
import {evaluatePoint,computeReceiverPoint} from './utd-scene.js';

export class CancelledError extends Error { constructor(){super('计算已取消');this.name='AbortError';} }
const check=signal=>{if(signal?.aborted)throw new CancelledError();};
const tick=()=>new Promise(r=>setTimeout(r,0));

export function makeCPUReference(scene,width=33,height=33){
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>97||height>97)throw new Error('接收网格超出安全范围（最多 97 × 97）');
  let disposed=false;
  return {
    type:'CPU 参考',width,height,
    async run({signal,onProgress=()=>{}}={}){
      if(disposed)throw new Error('计算引擎已关闭');
      check(signal);const data=new Float32Array(width*height*4),reasons=new Map();let valid=0;
      for(let y=0;y<height;y++){
        for(let x=0;x<width;x++){
          check(signal);if(disposed)throw new CancelledError();
          const p=computeReceiverPoint(scene,x,y,width,height),r=evaluatePoint(scene,p);
          const i=(y*width+x)*4;
          if(r.valid&&Number.isFinite(Math.fround(r.f))&&Number.isFinite(Math.fround(r.geometricPower))){data[i]=r.f;data[i+1]=r.geometricPower;data[i+2]=1;valid++;}
          else {data[i]=0;data[i+1]=0;data[i+2]=0;for(const reason of r.invalidReasons??[])reasons.set(reason,(reasons.get(reason)||0)+1);}
          data[i+3]=r.edgeCount||0;
        }
        onProgress((y+1)/height);await tick();
      }
      check(signal);if(disposed)throw new CancelledError();return {data,width,height,valid,invalid:width*height-valid,reasons:Object.fromEntries(reasons),backend:'CPU 参考',parity:null};
    },
    destroy(){disposed=true;},
  };
}

/** A spatially spread gate includes both axes, the middle, dark and bright values.
 * It checks backend agreement, never asserts physical correctness or real-device
 * validation before it actually executes. If the fixed grid misses every valid
 * pixel, add one bounded, mask-independent 9x9 grid before rejecting it.
 * This remains a sampled check, not an exhaustive numerical certificate. */
export function parityGate(scene,result,{boundaryEpsilon=2e-6}={}){
  const {width:w,height:h,data}=result;
  const coords=[0,Math.floor((w-1)/4),Math.floor((w-1)/2),Math.floor(3*(w-1)/4),w-1];
  const rows=[0,Math.floor((h-1)/4),Math.floor((h-1)/2),Math.floor(3*(h-1)/4),h-1];
  let worst=0,valid=0,maskDisagreements=0,numericDisagreements=0;
  const samples=[],visited=new Set();let coverage='base-5x5';
  const samplePoint=(x,y,stage)=>{
    const key=y*w+x;
    if(stage!=='base'&&visited.has(key))return;
    visited.add(key);
    const position=computeReceiverPoint(scene,x,y,w,h);
    const expected=evaluatePoint(scene,position,{boundaryEpsilon});
    const i=(y*w+x)*4,got=data[i],gpuValid=data[i+2]>.5;
    const sample={x,y,stage,position,cpu:{valid:expected.valid,f:expected.f,geometricPower:expected.geometricPower,edgeCount:expected.edgeCount},
      gpu:{valid:gpuValid,f:got,geometricPower:data[i+1],validityFlag:data[i+2],edgeCount:data[i+3]},status:'both-invalid'};
    samples.push(sample);
    if(gpuValid!==expected.valid){maskDisagreements++;sample.status='validity-mismatch';return;}
    if(!gpuValid)return;valid++;
    const error=Math.abs(got-expected.f)/(2e-5+.005*Math.abs(expected.f));
    const gerror=Math.abs(data[i+1]-expected.geometricPower)/(2e-5+.005*Math.abs(expected.geometricPower));
    const failed=!Number.isFinite(got)||got<0||!Number.isFinite(data[i+1])||error>1||gerror>1;
    sample.status=failed?'numeric-mismatch':'pass';
    sample.normalizedToleranceRatio={f:error,geometricPower:gerror};
    if(failed)numericDisagreements++;
    worst=Math.max(worst,error,gerror);
  };
  // Keep the original 25 checks and their order, including repeated coordinates
  // on grids smaller than 5. Supplemental locations never depend on GPU values.
  for(const y of rows)for(const x of coords)samplePoint(x,y,'base');
  const needsCoverage=()=>valid===0&&numericDisagreements===0&&maskDisagreements===0;
  if(needsCoverage()){
    coverage='refined-9x9';
    const fineX=Array.from({length:9},(_,i)=>Math.floor(i*(w-1)/8));
    const fineY=Array.from({length:9},(_,i)=>Math.floor(i*(h-1)/8));
    for(const y of fineY)for(const x of fineX)samplePoint(x,y,'refined');
  }
  const report={passed:!numericDisagreements&&!maskDisagreements&&valid>0,points:samples.length,validPoints:valid,normalizedToleranceRatio:worst,
    coverage:{strategy:coverage,basePoints:25,supplementalPoints:samples.length-25,uniquePoints:visited.size,exhaustive:visited.size===w*h},
    tolerance:{absolute:2e-5,relative:.005,boundaryEpsilon},numericDisagreements,maskDisagreements,samples};
  const reject=message=>{const error=new Error(message);error.parity=report;throw error;};
  if(numericDisagreements){const first=samples.find(s=>s.status==='numeric-mismatch');reject(`GPU / CPU 数值核对未通过，已拒绝这次 GPU 图像（首个差异像素 ${first.x}, ${first.y}）`);}
  if(maskDisagreements)reject(`GPU / CPU 有效区域不一致（${maskDisagreements} 点），已拒绝这次 GPU 图像`);
  if(!valid)reject('GPU 核对点中没有有效计算点，无法确认后端一致性');
  return report;
}

export async function createUTDEngine(scene,width,height,{backend='auto',onLost=()=>{}}={}){
  if(backend==='cpu')return {engine:makeCPUReference(scene,width,height),fallbackReason:null};
  try {
    const {makeGPUEngine}=await import('./utd-gpu-engine.js');
    return {engine:await makeGPUEngine(scene,width,height,{onLost}),fallbackReason:null};
  } catch(error){return {engine:makeCPUReference(scene,width,height),fallbackReason:error.message||String(error)};}
}
