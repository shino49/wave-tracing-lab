import {executeResidentHardwareTest,validateResidentHardwareTestRequest} from './plt-resident-test-runtime.js';
/** No GPU request on module load. A newer request cancels and supersedes the old
 * generation; cancelled/deadline-expired images are never posted.
 */
export function installResidentHardwareTestWorker(scope,execute=executeResidentHardwareTest){
 let active=null;
 scope.onmessage=async({data})=>{
  if(data?.type==='cancel'){if(active&&(data.id===undefined||data.id===active.id))active.controller.abort();return;}
  // Malformed envelopes must neither start work nor cancel a valid generation.
  if(!data||typeof data!=='object'||Array.isArray(data)||!Number.isSafeInteger(data.id)||data.id<1)return;
  try{
   if(Object.keys(data).some(key=>!['type','id','request'].includes(key))||(data.type!==undefined&&data.type!=='run')||!data.request)throw new TypeError('Invalid resident hardware-test worker request');
   validateResidentHardwareTestRequest(data.request);
  }catch(error){scope.postMessage({id:data.id,type:'error',error:String(error.message).slice(0,2000),name:error.name,diagnostic:null});return;}
  active?.controller.abort();
  const token={id:data?.id,controller:new AbortController()};active=token;
  const current=()=>active===token&&!token.controller.signal.aborted;
  const post=message=>{if(current())scope.postMessage({id:token.id,...message});};
  try{
   const result=await execute(data?.request,{signal:token.controller.signal,progress:progress=>post({type:'progress',progress})});
   post({type:'result',result});
  }catch(error){if(error.name!=='AbortError')post({type:'error',error:String(error.message??error).slice(0,2000),name:error.name??'Error',diagnostic:error.code?{code:error.code}:null});}
  finally{if(active===token)active=null;}
 };
 return ()=>{active?.controller.abort();active=null;scope.onmessage=null;};
}
if(typeof self!=='undefined'&&typeof WorkerGlobalScope!=='undefined'&&self instanceof WorkerGlobalScope)installResidentHardwareTestWorker(self);
