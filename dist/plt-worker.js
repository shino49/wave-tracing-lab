import {executePLTTask} from './plt-runtime.js';
let active=null;
self.onmessage=async({data:request})=>{
 if(request?.type==='cancel'){if(active&&(request.id===undefined||request.id===active.id))active.controller.abort();return;}
 active?.controller.abort();
 const token={id:request?.id,controller:new AbortController()};active=token;
 const current=()=>active===token&&!token.controller.signal.aborted;
 const post=message=>{if(current())self.postMessage({id:token.id,...message});};
 try{
  const result=await executePLTTask(request,{signal:token.controller.signal,progress:p=>post({type:'progress',...p}),preview:p=>post({type:'preview',...p})});
  post({type:'result',result});
 }catch(error){if(error.name!=='AbortError')post({type:'error',error:error.message,name:error.name,diagnostic:error.diagnostic??error.diagnostics??null});}
 finally{if(active===token)active=null;}
};
