import {executeCoherentTask} from './coherent-runtime.js';
self.onmessage=async event=>{
 const request=event.data;
 try{const result=await executeCoherentTask(request,{progress:text=>self.postMessage({type:'progress',id:request.id,text})});self.postMessage({type:'result',id:request.id,result});}
 catch(error){self.postMessage({type:'error',id:request.id,error:error.message,diagnostic:error.diagnostic??error.diagnostics??null});}
};
