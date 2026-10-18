const $=id=>document.getElementById(id),PI=Math.PI;
const DEFAULTS={screenCount:'2',separation:'2.3',offset:'.4',gamma:'1',phase:'0',inputPower:'1',detectorCenter:'0',detectorHalfAngle:'.55',resolution:'64',backend:'auto'};
let job=0,active=null,result=null,view='intensity',angleView='forward',diagnostic=null,refinement=null,stale=false,copyRevision=0;
const set=(id,text)=>{$(id).textContent=text;};
const number=v=>Number.isFinite(v)?v.toPrecision(4):'—',scientific=v=>Number.isFinite(v)?v.toExponential(2):'—',ms=v=>Number.isFinite(v)?`${v.toFixed(1)} ms`:'—';
function readOptions(){return {screenCount:Number($('screenCount').value),separation:Number($('separation').value),offset:Number($('offset').value),gamma:Number($('gamma').value),phase:Number($('phase').value)*PI/180,inputPower:Number($('inputPower').value),detectorCenter:Number($('detectorCenter').value)*PI/180,detectorHalfAngle:Number($('detectorHalfAngle').value),n:Number($('resolution').value)};}
function labels(){
 for(const id of ['separation','offset'])set(`${id}Value`,`${Number($(id).value).toFixed(2)} L₀`);
 for(const id of ['gamma','inputPower'])set(`${id}Value`,Number($(id).value).toFixed(2));
 for(const id of ['phase','detectorCenter'])set(`${id}Value`,`${Number($(id).value).toFixed(0)}°`);
 set('detectorHalfAngleValue',`${(Number($('detectorHalfAngle').value)*180/PI).toFixed(1)}°`);
 const one=$('screenCount').value==='1';$('separation').disabled=one;$('offset').disabled=one;$('presetOne').classList.toggle('active',one);$('presetTwo').classList.toggle('active',!one);
 $('showPhase').disabled=Number($('gamma').value)<1||stale||!!(result&&result.options.gamma<1);if($('showPhase').disabled&&view==='phase')setView('intensity');
}
function busy(value){$('coherentRun').disabled=value;$('coherentCancel').disabled=!value;$('refineCheck').disabled=value||!result||stale||result.options.n>=144;$('coherentField').setAttribute('aria-busy',String(value));}
function clearDiagnostic(){diagnostic=null;copyRevision++;globalThis.coherentLabDiagnostic=null;$('diagnosticText').value='';set('copyStatus','');}
function clearMetrics(){set('validationNotice','尚无本次参数的有效数值证据。独立参考仅覆盖指定基准几何，改变几何后应进行网格加密检查。');for(const id of ['detectorSignal','incoherentSignal','outgoingRatio','solveResidual','referenceError','solveTime','cpuTime','gpuTime','totalTime'])set(id,'—');set('detectorFraction','相对入射通量 —');set('gpuParity','尚未运行');set('refinementError','尚未运行');}
function stop({message='已取消；未完成的结果不会显示。',clear=false}={}){
 job++;if(active){active.worker.terminate();active.resolve(null);active=null;}busy(false);set('coherentStatus',message);if(clear){clearDiagnostic();set('coherentWarnings','');result=null;refinement=null;stale=false;$('coherentPending').hidden=false;$('coherentStale').hidden=true;clearMetrics();set('coherentBackend','待运行');drawEmpty();}busy(false);
}
function markChanged(){
 if(active)stop({message:'参数已改变，旧计算已取消。'});clearDiagnostic();set('coherentWarnings','');stale=true;labels();refinement=null;$('coherentStale').hidden=!result;clearMetrics();set('coherentBackend','待重新计算');set('coherentStatus','参数已改变。点击“计算波场”应用；原图暂留作参考。');$('refineCheck').disabled=true;$('coherentError').hidden=true;
}
function reset(){stop({clear:true,message:'已恢复默认参数。'});for(const [id,value]of Object.entries(DEFAULTS))$(id).value=value;labels();diagnostic=null;$('diagnosticText').value='';set('coherentWarnings','');set('copyStatus','');$('coherentError').hidden=true;view='intensity';setView(view);angleView='forward';$('showForward').classList.add('active');$('showForward').setAttribute('aria-pressed','true');$('showFull').classList.remove('active');$('showFull').setAttribute('aria-pressed','false');}
function createWorker(){if(globalThis.__COHERENT_WORKER_FACTORY__)return globalThis.__COHERENT_WORKER_FACTORY__();if(typeof Worker==='undefined')throw new Error('此浏览器不支持模块 Worker，无法安全运行本模式。');return new Worker(new URL('./coherent-worker.js',import.meta.url),{type:'module'});}
async function run(action='run'){
 stop({message:'正在准备计算…'});const id=++job;const options=readOptions();$('coherentError').hidden=true;set('copyStatus','');busy(true);if(action==='run'){clearDiagnostic();set('coherentWarnings','');clearMetrics();$('coherentStale').hidden=!result;stale=!!result;labels();set('coherentBackend','计算中');}
 return new Promise(resolve=>{
  let worker;try{worker=createWorker();}catch(error){fail(error.message,null);resolve(null);return;}
  active={worker,resolve,id};
  worker.onmessage=event=>{const data=event.data;if(id!==job||data.id!==id||active?.worker!==worker)return;if(data.type==='progress'){set('coherentStatus',data.text);return;}worker.terminate();active=null;
   if(data.type==='error'){fail(data.error,data.diagnostic);resolve(null);return;}
   if(data.type!=='result'){fail('计算 Worker 返回了未知消息。');resolve(null);return;}
   if(data.result.action==='refine'){
    refinement=data.result.refinement;set('refinementError',`${refinement.coarseN}→${refinement.fineN} · ${scientific(refinement.fieldError)}`);set('coherentStatus',refinement.pass?'网格加密核对通过；当前显示仍是原节点数结果。':'网格加密核对未通过；请提高节点数后重算。');if(!refinement.pass){set('coherentError','场或探测算子的加密误差超过 2×10⁻⁴，请使用更精细网格。');$('coherentError').hidden=false;}saveDiagnostic();busy(false);resolve(data.result);return;
   }
   result=data.result;stale=false;refinement=null;$('coherentPending').hidden=true;$('coherentStale').hidden=true;renderResult();busy(false);resolve(result);
  };
  worker.onerror=event=>{if(id!==job||active?.worker!==worker)return;worker.terminate();active=null;fail(event.message||'计算 Worker 无法启动。');resolve(null);};
  worker.onmessageerror=()=>{if(id!==job||active?.worker!==worker)return;worker.terminate();active=null;fail('无法读取计算 Worker 的数据。');resolve(null);};
  try{worker.postMessage({id,action,options,backend:$('backend').value});}catch(error){worker.terminate();active=null;fail(error.message);resolve(null);}
 });
}
function fail(message,extra=null){busy(false);set('coherentError',message);$('coherentError').hidden=false;set('coherentStatus','计算停止；未通过的结果不会显示。');set('coherentBackend','停止');result=null;refinement=null;stale=false;$('coherentPending').hidden=false;$('coherentStale').hidden=true;clearMetrics();drawEmpty();clearDiagnostic();diagnostic={error:message,detail:extra,options:readOptions()};$('diagnosticText').value=JSON.stringify(diagnostic,null,2);busy(false);}
function saveDiagnostic(){if(!result)return;copyRevision++;diagnostic={schema:1,timestamp:new Date().toISOString(),options:result.options,backend:result.backend,measurement:{detectorFlux:result.measurement.detectorFlux,baselineDetectorFlux:result.baseline.detectorFlux,totalFlux:result.measurement.totalFlux},timings:result.timings,parity:result.parity,diagnostics:result.diagnostics,warnings:result.warnings,refinement};$('diagnosticText').value=JSON.stringify(diagnostic,null,2);globalThis.coherentLabDiagnostic=diagnostic;}
function renderResult(){
 labels();const r=result,m=r.measurement;set('coherentBackend',r.backend);set('detectorSignal',number(m.detectorFlux));set('incoherentSignal',number(r.baseline.detectorFlux));set('detectorFraction',`相对入射通量 ${(100*m.detectorFlux/r.options.inputPower).toFixed(3)}%`);set('outgoingRatio',(m.totalFlux/r.options.inputPower).toFixed(6));set('solveResidual',scientific(r.diagnostics.relativeResidual));const ref=r.diagnostics.reference;set('referenceError',ref?.applicable?scientific(ref.nearFieldError):'此几何无固定参考');set('refinementError',r.options.n>=144?'已用最高节点数 144':'尚未运行');
 set('gpuParity',r.parity?.passed?`通过 · ${r.grid.width*r.grid.height} 点`:r.parity?'未通过 · 已用 CPU':'未运行 · CPU 参考');set('solveTime',ms(r.timings.solveMs));set('cpuTime',ms(r.timings.cpuFieldMs));set('gpuTime',ms(r.timings.gpuMs));set('totalTime',ms(r.timings.totalMs));set('coherentWarnings',r.warnings.join(' '));set('coherentStatus',`已完成 ${r.options.screenCount} 屏耦合解，每屏 ${r.options.n} 节点。探测信号使用未经图像归一化的标量通量。`);
 set('screenTwoLabel',r.options.screenCount===2?`区域 B · x = ${r.options.separation.toFixed(2)}`:'单屏基准 · 无区域 B');set('validationNotice',ref?.applicable?`默认几何独立 SciPy 参考：复场 ${scientific(ref.nearFieldError)}；角场 ${scientific(ref.farFieldError)}；探测算子 ${scientific(ref.detectorOperatorError)}。这些检查不替代空间网格收敛或物理实验。`:'几何已改变：本次残差和能量检查有效，固定默认参考误差不适用于这里。可运行网格加密核对。');
 saveDiagnostic();drawField();drawAngular();
}
function setView(next){if(next==='phase'&&(stale||Number($('gamma').value)<1||(result&&result.options.gamma<1)))return;view=next;for(const [id,v]of [['showIntensity','intensity'],['showPhase','phase']]){$(id).classList.toggle('active',v===view);$(id).setAttribute('aria-pressed',String(v===view));}if(result)drawField();}
function context(id){const c=$(id),ctx=c.getContext('2d');if(!ctx)throw new Error('Canvas 2D 不可用。');return {c,ctx,w:c.width,h:c.height};}
function drawEmpty(){set('fieldPeak','—');set('fieldScale','平均强度 E|u|² · 等待计算');set('fieldNotice','灰色窄带排除未经收敛验证的近边界点。图像色阶不改变探测信号。');for(const id of ['coherentField','coherentAngular']){const {ctx,w,h}=context(id);ctx.fillStyle='#0a1310';ctx.fillRect(0,0,w,h);}}
function color(t){const stops=[[8,19,15],[20,51,38],[41,102,73],[118,182,123],[244,232,183]],x=Math.max(0,Math.min(1,t))*4,i=Math.min(3,Math.floor(x)),f=x-i;return stops[i].map((v,k)=>Math.round(v*(1-f)+stops[i+1][k]*f));}
function phaseColor(phi){const h=(phi+PI)/(2*PI)*6,c=.7,x=c*(1-Math.abs(h%2-1)),m=.18;const v=h<1?[c,x,0]:h<2?[x,c,0]:h<3?[0,c,x]:h<4?[0,x,c]:h<5?[x,0,c]:[c,0,x];return v.map(q=>Math.round(255*(q+m)));}
function drawField(){
 if(!result)return;const {ctx,w,h}=context('coherentField'),g=result.grid,pad={l:60,r:25,t:30,b:48},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;ctx.fillStyle='#09110f';ctx.fillRect(0,0,w,h);
 const buf=document.createElement('canvas');buf.width=g.width;buf.height=g.height;const b=buf.getContext('2d'),im=b.createImageData(g.width,g.height);let peak=0;for(const value of g.intensity)peak=Math.max(peak,value);const max=peak||1;
 for(let i=0;i<g.width*g.height;i++){let rgb;if(!g.mask[i])rgb=(Math.floor(i/g.width)+i)%2?[51,56,54]:[42,47,45];else if(view==='phase'){const re=g.coherentField?.[i*2]??0,imag=g.coherentField?.[i*2+1]??0;rgb=phaseColor(Math.atan2(imag,re));if(re*re+imag*imag<1e-14)rgb=[20,25,23];}else rgb=color((10*Math.log10(Math.max(g.intensity[i]/max,1e-12))+45)/45);im.data.set([...rgb,255],i*4);}b.putImageData(im,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(buf,pad.l,pad.t,pw,ph);
 const xMin=-2.8,xMax=5.5,yMin=-3,yMax=3,X=x=>pad.l+(x-xMin)/(xMax-xMin)*pw,Y=y=>pad.t+(yMax-y)/(yMax-yMin)*ph;
 ctx.font='13px system-ui';ctx.textAlign='center';ctx.fillStyle='#9cb0a2';ctx.strokeStyle='#77998435';ctx.lineWidth=1;
 for(const x of [-2,-1,0,1,2,3,4,5]){ctx.beginPath();ctx.moveTo(X(x),pad.t+ph);ctx.lineTo(X(x),pad.t+ph+5);ctx.stroke();ctx.fillText(String(x),X(x),h-24);}ctx.textAlign='right';for(const y of [-3,-2,-1,0,1,2,3])ctx.fillText(String(y),pad.l-12,Y(y)+4);
 ctx.textAlign='left';ctx.fillText('y / L₀',9,18);ctx.textAlign='right';ctx.fillText('x / L₀',w-18,h-7);
 for(let i=0;i<result.screens.length;i++){const [x,lo,hi]=result.screens[i];ctx.strokeStyle=i?'#8ed8c7':'#edc885';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(X(x),Y(lo));ctx.lineTo(X(x),Y(hi));ctx.stroke();ctx.font='bold 13px system-ui';ctx.fillStyle=i?'#b9ecdc':'#f3d69f';ctx.textAlign='center';ctx.fillText(i?'B':'A',X(x),Y(hi)-10);}
 ctx.strokeStyle='#4d6659';ctx.lineWidth=1;ctx.strokeRect(pad.l,pad.t,pw,ph);
 const count=g.mask.reduce((a,b)=>a+(b?0:1),0);set('fieldScale',view==='phase'?'相干分量 arg(u) · 总混合态无唯一相位':'平均强度 E|u|² · 峰值相对 −45…0 dB');set('fieldPeak',view==='phase'?'−π → +π':`峰值 ${number(peak)}`);$('fieldGradient').style.background=view==='phase'?'linear-gradient(90deg,#e32e2e,#e3e32e,#2ee32e,#2ee3e3,#2e2ee3,#e32ee3,#e32e2e)':'';set('fieldNotice',`灰色为近屏排除区：距屏 < ${(12/result.options.n).toFixed(3)} L₀，${count} / ${g.width*g.height} 像素不作预测。${view==='phase'?'显示正交源模态相干分量的相位。':'峰值相对 dB 色阶仅用于显示，不改变下方探测通量。'}`);
}
function drawAngular(){
 if(!result)return;const {ctx,w,h}=context('coherentAngular'),pad={l:61,r:25,t:28,b:44},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b,theta=result.measurement.angleTheta,values=result.measurement.angleFlux,baseline=result.baseline.angleFlux;ctx.fillStyle='#141e19';ctx.fillRect(0,0,w,h);let peak=0;for(const a of [values,baseline])for(const v of a)peak=Math.max(peak,v);peak=peak||1;const minAngle=angleView==='forward'?-PI/2:-PI,maxAngle=angleView==='forward'?PI/2:PI;peak=0;for(const a of [values,baseline])for(let i=0;i<a.length;i++)if(theta[i]>=minAngle&&theta[i]<=maxAngle)peak=Math.max(peak,a[i]);peak=peak||1;const X=t=>pad.l+(t-minAngle)/(maxAngle-minAngle)*pw,Y=v=>pad.t+ph-v/peak*ph;
 const lo=result.options.detectorCenter-result.options.detectorHalfAngle,hi=result.options.detectorCenter+result.options.detectorHalfAngle;ctx.fillStyle='#486f5138';ctx.fillRect(X(lo),pad.t,X(hi)-X(lo),ph);ctx.strokeStyle='#2b3a31';ctx.lineWidth=1;ctx.font='12px system-ui';ctx.fillStyle='#81998b';ctx.textAlign='right';for(let i=0;i<=4;i++){const val=peak*i/4,y=Y(val);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText(number(val),pad.l-10,y+4);}ctx.textAlign='center';for(const deg of (angleView==='forward'?[-90,-45,0,45,90]:[-180,-90,0,90,180])){const x=X(deg*PI/180);ctx.fillText(`${deg}°`,x,h-20);}
 for(const [a,c,dash]of [[baseline,'#d6b579',[7,6]],[values,'#b9e0b2',[]]]){ctx.strokeStyle=c;ctx.lineWidth=2.2;ctx.setLineDash(dash);ctx.beginPath();let started=false;for(let i=0;i<theta.length;i++){if(theta[i]<minAngle||theta[i]>maxAngle)continue;const x=X(theta[i]),y=Y(a[i]);if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);}ctx.stroke();}ctx.setLineDash([]);ctx.fillStyle='#9ab3a4';ctx.textAlign='right';ctx.fillText('θ',w-11,h-20);
}
$('coherentForm').addEventListener('submit',event=>{event.preventDefault();run();});$('coherentCancel').addEventListener('click',()=>stop({clear:true}));$('coherentReset').addEventListener('click',reset);$('refineCheck').addEventListener('click',()=>run('refine'));$('showIntensity').addEventListener('click',()=>setView('intensity'));$('showPhase').addEventListener('click',()=>setView('phase'));
for(const element of $('coherentForm').querySelectorAll('input,select'))element.addEventListener(element.tagName==='SELECT'?'change':'input',markChanged);
for(const [id,count]of [['presetOne',1],['presetTwo',2]])$(id).addEventListener('click',()=>{$('screenCount').value=String(count);markChanged();run();});
$('copyDiagnostics').addEventListener('click',async()=>{if(!diagnostic){set('copyStatus','尚无可复制的诊断报告。');return;}const revision=copyRevision;try{await navigator.clipboard.writeText(JSON.stringify(diagnostic,null,2));if(revision!==copyRevision)return;set('copyStatus','已复制。');}catch{if(revision!==copyRevision)return;$('coherentDiagnostics').open=true;$('diagnosticText').focus();$('diagnosticText').select();set('copyStatus','自动复制不可用，请复制上方可选文本。');}});
for(const [id,v]of [['showForward','forward'],['showFull','full']])$(id).addEventListener('click',()=>{angleView=v;for(const [button,value]of [['showForward','forward'],['showFull','full']]){$(button).classList.toggle('active',value===v);$(button).setAttribute('aria-pressed',String(value===v));}if(result)drawAngular();});
window.addEventListener('pagehide',()=>stop({message:'计算已停止。'}));
for(const [id,value]of Object.entries(DEFAULTS))$(id).value=value;labels();drawEmpty();
globalThis.coherentLab={run,cancel:()=>stop({clear:true}),reset,getState:()=>({running:!!active,job,stale,view,angleView,backend:result?.backend,options:readOptions(),result,refinement}),getDiagnostic:()=>diagnostic};
if(!globalThis.__COHERENT_TEST__)setTimeout(()=>run(),40);
