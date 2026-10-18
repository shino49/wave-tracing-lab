import {executePLTDemoTask} from './plt-demo-runtime.js';
/** Explicit user start only. Superseded and cancelled generations never publish. */
export function installPLTDemoWorker(scope,execute=executePLTDemoTask){
 let active=null;
 scope.onmessage=async({data:request})=>{
  if(request?.type==='cancel'){if(active&&(request.id===undefined||request.id===active.id))active.controller.abort();return;}
  if(!request||typeof request!=='object'||Array.isArray(request)||!Number.isSafeInteger(request.id)||request.id<1)return;
  active?.controller.abort();const token={id:request.id,controller:new AbortController()};active=token;
  const current=()=>active===token&&!token.controller.signal.aborted;
  const post=message=>{if(current())scope.postMessage({id:token.id,...message});};
  try{
   const result=await execute(request,{signal:token.controller.signal,progress:p=>post({type:'progress',...p}),preview:p=>post({type:'preview',...p})});post({type:'result',result});
  }catch(error){if(error.name!=='AbortError')post({type:'error',error:String(error.message??error).slice(0,2048),name:String(error.name??'Error').slice(0,128)});}
  finally{if(active===token)active=null;}
 };
 return ()=>{active?.controller.abort();active=null;scope.onmessage=null;};
}
if(typeof self!=='undefined'&&typeof WorkerGlobalScope!=='undefined'&&self instanceof WorkerGlobalScope)installPLTDemoWorker(self);
