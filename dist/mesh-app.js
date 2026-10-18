import {prepareMesh,add,scale,cross,normalize,MESH_LIMITS} from './mesh-geometry.js';
import {prepareUTDScene,builtInSceneOBJ} from './utd-scene.js';
import {prepareMirrorExperiment} from './utd-mirror.js';
import {createUTDEngine,makeCPUReference} from './utd-engine.js';
import {setupMeshView} from './mesh-view.js';
const $=id=>document.getElementById(id),form=$('meshForm');
let objText=builtInSceneOBJ('screen'),modelName='有限薄屏',scene=null,result=null,engine=null,controller=null,revision=0,previewTimer=null;
const texture=document.createElement('canvas'),view=setupMeshView($('meshSceneCanvas'));
const issueNames={
 'geometry-validation-limit':'超出几何检查安全上限','imprecise-weld-geometry':'焊接位置不足以确定精确几何','nonconvex-solid':'实体未通过整体凸性检查','intersecting-solid-components':'实体组件重叠、嵌套或接触，无法作为独立楔形求解','overlapping-surface-triangles':'存在重叠的面片','intersecting-plate-components':'薄屏组件相交或接触','self-intersecting-surface':'表面自交或存在非拓扑接触',
 'nonmanifold':'存在非流形边','inconsistent-winding':'相邻面绕序不一致','open-solid':'实体有开放边，不能解释为封闭实体','zero-volume':'实体体积为零或数值分辨率不足','inward-winding':'实体面朝内，请在建模软件中修正绕序','closed-plate':'封闭体不能解释为开放薄屏','nonplanar-open-plate':'折叠／曲面开放薄壳不在本阶段范围内','concave-edge':'凹楔不在当前物理模型范围内','ambiguous-wedge':'相邻面无法定义一致的凸楔','multiple-depth-regions':'几何超出一次交互区域，需要多区域传播','plane-curvature-guard':'波前曲率超过局部准直近似范围','triangle-safety-cap':'超过当前求解器 128 三角形安全上限','edge-safety-cap':'超过当前求解器边缘安全上限',
 'mirror-crossing-geometry':'网格接触或跨过理想镜面；整个网格须严格位于镜面同一侧',
 'mirror-wrong-halfspace':'真实源与整个真实接收面须和网格位于镜面同一侧；请检查镜面间距与展开距离',
 'mirror-source-footprint':'保留的 3σ 源截面接触或跨过镜面',
 'mirror-grazing-order':'入射包络掠射镜面，或方向不符合所选反射次序',
 'mirror-source-order':'所有局部几何须位于展开源相位面的前方',
 'mirror-missing-region':'所选反射路径族没有实际局部 UTD 交互区域',
 'mirror-region-overlap':'完整交互椭球接触镜面；反射与局部 UTD 区域必须分离',
 'mirror-grazing-output':'接收面保留路径族接近镜面掠射',
 'mirror-projected-source-side':'局部平面波的源投影跨过镜面，无法保证所有保留路径具有所选反射次序',
 'mirror-extra-contact':'折线路径的保守包围界遇到镜像几何，可能存在额外接触或阴影转换；整个路径族被拒绝',
 'mirror-no-crossing':'展开中心路径没有恰好穿过镜面一次',
 'nonplanar-reflection-facet':'合并后的反射面并不共面，包含尚未解析的浅折边；不会自动压平或遗漏其衍射',
};
function setError(message=''){$('meshError').textContent=message;$('meshError').hidden=!message;}
function translateWarning(s){return ({
 'Conditional directional UTD power transfer f; not irradiance or a calibrated Gaussian-mode solution.':'显示条件性 UTD 功率传递因子 f，不是标定辐照度或精确高斯模式解。',
 'The 3σ envelope truncation, two-Fresnel-width endpoint screen and local-plane phase-curvature screen are disclosed approximations.':'采用 3σ 包络截断、两个 Fresnel 宽度的端点检查与局部准直曲率检查；这些是公开的适用性近似。',
 'Finite-edge endpoint/corner diffraction, multiple interaction regions and repeated reflections/diffractions are unsupported.':'不含有限边端点／角点衍射、多交互区域及重复反射／衍射。',
 'The finite 3σ beam envelope misses the mesh; only the direct conditional plane-wave term is available.':'3σ 波束包络没有遇到几何，只能计算直接条件项。',
 'OBJ materials are ignored. No external files are requested; triangles are geometry only, with materials chosen downstream.':'OBJ 材质已忽略，不加载外部文件。物理边界条件由当前实验选项指定。',
 'Texture/normal indices were validated; geometry normals are computed from triangle winding.':'纹理／法向索引已检查；计算法向由实际三角形绕序决定。',
 'One selected family: one infinite ideal scalar mirror and one coherent local GO + UTD interaction. Other mirror/direct families are not added.':'仅计算一个选定反射路径族：一个无限理想标量镜面与一个相干局部 GO + UTD 区域。保留完整局部 GO + D；不与其他镜面或独立直达路径族合成。',
 'The mirror has no finite aperture or diffracting rim. The full Gaussian waist age and unfolded range are preserved; no extra spreading or PDF weight is applied.':'镜面无限延伸，没有有限孔径或衍射边缘。保留原高斯束腰的传播历程与完整展开距离，不增加扩散或 PDF 权重。',
 'The folded-path bounding-box admission test is conservative and can reject clear layouts. Endpoint, corner, two-diffraction and general multi-region transport remain unsupported.':'折线路径包围界准入检查是保守的，可能拒绝本可畅通的布局。仍不支持端点、角点、两次衍射（DD）或通用多区域传播。',
 'Clear geometric paths do not certify wave accuracy: higher-order and remote diffraction, including nearby omitted penumbrae, are not modeled.':'几何路径畅通不等于波动计算准确：未建模高阶或远处的衍射效应，包括邻近但未计入的半影区。',
 })[s]||s;}
function setWarnings(messages){const list=$('meshWarnings');list.replaceChildren();for(const message of [...new Set(messages.filter(Boolean))]){const li=document.createElement('li');li.textContent=translateWarning(message);list.append(li);}list.hidden=!list.childElementCount;}
let gpuDiagnostic=null;
function setDiagnostic(report){
  gpuDiagnostic=report;globalThis.utdLabDiagnostic=report;
  $('meshDiagnostics').hidden=!report;$('meshDiagnosticText').value=report?JSON.stringify(report,(_,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v,2):'';
  $('meshCopyStatus').textContent='';$('meshDiagnosticDetails').open=false;
}
$('meshCopyDiagnostics').addEventListener('click',async()=>{
  const report=gpuDiagnostic;if(!report)return;
  const text=$('meshDiagnosticText').value;
  try{await navigator.clipboard.writeText(text);if(gpuDiagnostic===report)$('meshCopyStatus').textContent='已复制。可以粘贴发给我排查。';}
  catch{if(gpuDiagnostic!==report)return;$('meshDiagnosticDetails').open=true;$('meshDiagnosticText').focus();$('meshDiagnosticText').select();$('meshCopyStatus').textContent='请复制下方报告后发给我。';}
});
let selectedTransport='single';
function updateTransportUI(){
  selectedTransport=$('meshTransport').value;const reflected=selectedTransport!=='single';
  $('meshMirrorControls').hidden=!reflected;$('meshMirrorDistance').disabled=!reflected;
  $('meshResultsTitle').textContent=reflected?'选定反射路径族 · 条件 f':'条件功率传递因子 f';
  $('meshHeatmapTitle').textContent=reflected?'真实接收面 · 选定反射路径族':'接收面 · 二维分布';
  $('meshTransportSummary').textContent=reflected?'仅此反射路径族的条件 f：保留完整局部 GO + D，不与独立直达路径族合成。不支持两次衍射（DD）或通用多区域传播。':'一个局部 GO + UTD 区域的条件功率传递；不含多区域传播。';
  $('meshSceneNotice').textContent=reflected?'三维图使用真实源、折线路径与真实接收面；中心折线仅示意所选次序，不代表每条 GO / D 路径。45° 无限理想标量镜面的有限绘图片仅作示意，没有孔径或衍射边缘。':'网格与接收面按真实坐标显示。相机视角不参与物理计算；包络为示意。';
  $('meshBeamFrameHelp').textContent=reflected?'下方入射角、源腰面与接收距离沿用展开坐标；镜面将路径折回真实空间，三维图不显示虚源或虚接收面。镜面间距沿世界 z 轴从网格中心量取。':'入射角使用世界坐标；接收距离沿世界 z 轴从网格中心量取。';
  $('meshHeatmapCanvas').setAttribute('aria-label',reflected?'真实接收面上选定反射路径族的条件功率传递因子 f，保留完整局部 GO 加 D，不合成独立直达路径族':'接收面上计算所得的条件功率传递因子 f，以相对条件值 1 的分贝显示');
}
function pending(text='调整参数后，点击“运行计算”。'){
  updateTransportUI();$('meshResultsPending').hidden=false;$('meshResultsPending').textContent=text;result=null;globalThis.utdLabValidation=null;setDiagnostic(null);$('meshParity').textContent='当前参数待核对';$('meshBackend').textContent='待运行';$('meshPeak').textContent='—';$('meshElapsed').textContent='—';
  $('meshRegionNotice').textContent='尚无当前参数的计算结果。灰色表示超出适用范围；数值相对入射条件值 1，非标定辐照度。';
  for(const id of ['meshHeatmapCanvas','meshProfileCanvas']){const canvas=$(id);canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);}
  texture.width=1;texture.height=1;texture.getContext('2d').clearRect(0,0,1,1);view.setTexture(null);
}
function stop(mark=true){revision++;setDiagnostic(null);controller?.abort();engine?.destroy();engine=null;controller=null;globalThis.utdLabValidation=null;$('meshCancel').disabled=true;$('meshRun').disabled=false;if(mark){$('meshStatus').textContent='已取消。可以调整参数后重新运行。';pending('计算已取消，未展示未完成结果。');}}
const num=(id,min,max)=>{const n=Number($(id).value);if(!Number.isFinite(n)||n<min||n>max)throw new Error(`${$(id).closest('label')?.textContent.trim()||id}：请输入 ${min} 至 ${max} 的有限数值`);return n;};
function prepare(){
  updateTransportUI();scene=null;view.setScene(null);
  const unit=num('meshUnit',1e-6,1e6)*num('meshScale',1e-6,1e6);
  const mesh=prepareMesh(objText,{unitScale:unit,rotationDeg:['meshRotateX','meshRotateY','meshRotateZ'].map(id=>num(id,-36000,36000))});
  const yaw=num('meshYaw',-180,180)*Math.PI/180,pitch=num('meshPitch',-89,89)*Math.PI/180;
  const di=[Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)],c=mesh.bounds.center;
  const sourceDistance=Math.max(5,Math.hypot(...mesh.bounds.size));
  const distance=num('meshDistance',.001,1e5),span=num('meshSpan',.001,1e5);
  const state={wavelengthMm:num('meshWavelength',1e-6,1e3),sigma0Mm:num('meshBeamRadius',1e-5,1e4),
    sourceOrigin:add(c,scale(di,-sourceDistance)),incidentDirection:di,detectorCenter:add(c,[0,0,distance]),
    detectorU:[1,0,0],detectorV:[0,1,0],detectorWidthMm:span,detectorHeightMm:span,
    surface:$('meshFormat').value,boundary:$('meshBoundary').value};
  if(!['single','before','after'].includes(selectedTransport))throw new Error('请选择有效的传播路径。');
  scene=selectedTransport==='single'?prepareUTDScene(mesh,state):prepareMirrorExperiment(mesh,state,{order:selectedTransport,distanceMm:num('meshMirrorDistance',.001,1e5)});
  $('meshName').textContent=modelName;$('meshTriangles').textContent=mesh.triangles.length.toLocaleString();
  $('meshPhysicalEdges').textContent=(scene.physicalWedges?.length??scene.fullWedges?.length??mesh.diffractionEdges.length).toLocaleString();
  $('meshGatheredEdges').textContent=(scene.wedges?.length||0).toLocaleString();
  const transport=scene.mirrorTransport,physical=transport?.physical??scene.params;
  $('meshHeatmapRange').textContent=`局部 u / v · ${span} × ${span} mm · ${transport?'展开轴向距离':'距网格中心'} ${distance} mm`;
  const u=normalize(cross(Math.abs(di[1])<.9?[0,1,0]:[1,0,0],di)),v=cross(di,u);
  // The kernel keeps unfolded coordinates. Only physical fields reach the view;
  // omit its straight cone in reflected mode rather than draw a fictitious source.
  view.setScene({mesh,sourceOrigin:physical.sourceOrigin,incidentDirection:physical.incidentDirection,incomingCone:transport?null:scene.incomingCone,region:scene.region?{...scene.region,basis:{u,v,w:di}}:null,
    detector:{center:physical.detectorCenter,u:physical.detectorU,v:physical.detectorV,width:span,height:span},
    mirror:transport?{...transport.mirror,displayCenter:transport.centralMirrorPoint,illustrative:true}:null,
    foldedPath:transport?.foldedPath??null,mirrorOrder:transport?.order,supported:scene.supported});
  view.draw();
  const unsupported=scene.unsupported??[],messages=unsupported.map(v=>issueNames[v.code]||v.message||String(v));
  setWarnings([...(scene.warnings??[]),...messages,'OBJ 解析上限为 12,000 三角形；当前波动物理求解的安全上限为 128 三角形，超限会拒绝计算。']);
  if(!scene.supported)throw new Error(messages.join('；')||'这个场景超出当前物理模型的范围');
  return scene;
}
function updatePreview(){try{setError();prepare();$('meshStatus').textContent=scene.mirrorTransport?'几何已检查。点击运行，仅计算选定反射路径族的条件 f。':'几何已检查。点击运行，计算单区域条件功率传递。';}catch(error){setError(error.message);$('meshStatus').textContent='几何或参数未通过检查，尚未计算。';}}
function invalidate(){stop(false);scene=null;view.setScene(null);pending();$('meshParity').textContent='参数已改变，待重新核对';clearTimeout(previewTimer);previewTimer=setTimeout(updatePreview,150);}
for(const input of form.querySelectorAll('input[type=number],select'))if(input.id!=='meshPreset')input.addEventListener('input',invalidate);
$('meshTransport').addEventListener('change',()=>{if($('meshTransport').value!==selectedTransport)invalidate();});
$('meshPreset').addEventListener('change',()=>{stop(false);const name=$('meshPreset').value;objText=builtInSceneOBJ(name);modelName=$('meshPreset').selectedOptions[0].textContent;$('meshFile').value='';$('meshFormat').value=['box','wedge'].includes(name)?'closed-solid':'open-plate';pending();updatePreview();});
$('meshFile').addEventListener('change',async()=>{
  const file=$('meshFile').files?.[0];if(!file)return;stop(false);const rev=revision;pending('正在本地读取 OBJ…');
  try{if(file.size>MESH_LIMITS.maxBytes)throw new Error('OBJ 超过 2 MiB 上限');const text=await file.text();if(rev!==revision)return;prepareMesh(text);objText=text;modelName=file.name;pending();updatePreview();}
  catch(error){if(rev===revision){setError(`无法读取 OBJ：${error.message}`);pending('文件未通过检查，未替换已有几何。');}}
});
$('meshResetCamera').addEventListener('click',()=>view.resetCamera());
$('meshCancel').addEventListener('click',()=>stop(true));
$('meshReset').addEventListener('click',()=>{stop(false);form.reset();objText=builtInSceneOBJ('screen');modelName='有限薄屏';$('meshFile').value='';scene=null;pending();setError();$('meshBackend').textContent='待运行';$('meshParity').textContent='尚未执行';view.resetCamera();updatePreview();});
form.addEventListener('submit',e=>e.preventDefault());
function renderResults(r){
  const {data,width:w,height:h}=r;texture.width=w;texture.height=h;const tc=texture.getContext('2d'),im=tc.createImageData(w,h);let peak=0;
  for(let i=0;i<w*h;i++){
    const valid=data[4*i+2]>.5,v=data[4*i],o=i*4;
    if(valid){peak=Math.max(peak,v);const db=10*Math.log10(Math.max(v,1e-4)),t=Math.max(0,Math.min(1,(db+40)/46)),q=Math.pow(t,1.6);im.data[o]=9+220*q;im.data[o+1]=22+216*q;im.data[o+2]=16+175*q;}
    else{const x=i%w,y=Math.floor(i/w),z=(x+y)%4<2?43:55;im.data[o]=z;im.data[o+1]=z+3;im.data[o+2]=z+4;}im.data[o+3]=255;
  }
  tc.putImageData(im,0,0);const heat=$('meshHeatmapCanvas'),hc=heat.getContext('2d');hc.imageSmoothingEnabled=false;hc.clearRect(0,0,heat.width,heat.height);hc.drawImage(texture,0,0,heat.width,heat.height);
  view.setTexture(texture);view.draw();$('meshResultsPending').hidden=true;$('meshPeak').textContent=peak.toPrecision(5);
  const p=$('meshProfileCanvas'),pc=p.getContext('2d'),W=p.width,H=p.height,m={l:55,r:20,t:18,b:32},row=Math.floor(h/2),span=scene.params.detectorWidthMm;
  pc.clearRect(0,0,W,H);pc.fillStyle='#0e1914';pc.fillRect(0,0,W,H);pc.font='12px system-ui';pc.lineWidth=1;
  const px=x=>m.l+x/(w-1)*(W-m.l-m.r),py=db=>m.t+(6-db)/46*(H-m.t-m.b);
  for(const db of [-40,-30,-20,-10,0,6]){pc.strokeStyle='#294132';pc.beginPath();pc.moveTo(m.l,py(db));pc.lineTo(W-m.r,py(db));pc.stroke();pc.fillStyle='#99af9f';pc.fillText(db+' dB',3,py(db)+4);}
  for(const x of [0,(w-1)/2,w-1])pc.fillText(((x/(w-1)-.5)*span).toFixed(1)+' mm',px(x)-14,H-8);
  for(const [column,color,dash] of [[1,'#d5ae72',[5,4]],[0,'#bce5a8',[]]]){pc.strokeStyle=color;pc.lineWidth=2;pc.setLineDash(dash);pc.beginPath();let drawing=false;for(let x=0;x<w;x++){const i=(row*w+x)*4;if(data[i+2]<.5){drawing=false;continue;}const db=Math.max(-40,Math.min(6,10*Math.log10(Math.max(data[i+column],1e-4))));if(drawing)pc.lineTo(px(x),py(db));else pc.moveTo(px(x),py(db));drawing=true;}pc.stroke();}pc.setLineDash([]);
  $('meshRegionNotice').textContent=`${scene.mirrorTransport?'绿色：选定反射路径族的完整局部 GO + D；虚线：同一路径族的局部 GO。未合成独立直达路径族。':'绿色：UTD；虚线：几何直接／反射。'}${r.valid} / ${w*h} 点有效，${r.invalid} 点因端点、掠入射、低频或缺少一阶路径而留空。显示 −40…+6 dB（参考 f = 1），不是每张图重新归一化。`;
}
async function run(){
  stop(false);clearTimeout(previewTimer);const rev=revision;setError();pending('正在检查场景和计算后端…');let active;
  try{active=prepare();}catch(e){setError(e.message);$('meshStatus').textContent='未通过场景检查，已停止。';pending('场景不在当前适用范围内。');return;}
  const w=Number($('meshResolution').value);controller=new AbortController();const signal=controller.signal;
  $('meshRun').disabled=true;$('meshCancel').disabled=false;const start=performance.now();let fallbackReason=null;
  const onProgress=p=>{if(rev===revision)$('meshStatus').textContent=`${engine?.type||'初始化'} · 已计算 ${Math.round(p*100)}%，结果尚未通过最终检查。`;};
  try{
    const created=await createUTDEngine(active,w,w,{backend:$('meshBackendSelect').value,onLost:msg=>{fallbackReason=msg;}});
    if(rev!==revision){created.engine.destroy();return;}engine=created.engine;fallbackReason=created.fallbackReason;
    $('meshBackend').textContent=engine.type;$('meshParity').textContent=engine.type==='WebGPU'?'GPU 编译通过，数值核对待运行':'CPU 参考，未执行 GPU';
    let output;
    try{output=await engine.run({signal,onProgress});}
    catch(error){if(error.name==='AbortError'||rev!==revision)throw error;if(engine.type!=='WebGPU')throw error;fallbackReason=error.message;if(error.diagnostic)setDiagnostic(error.diagnostic);engine.destroy();engine=makeCPUReference(active,w,w);$('meshBackend').textContent=engine.type;output=await engine.run({signal,onProgress});}
    if(rev!==revision||signal.aborted)return;
    result=output;renderResults(result);$('meshElapsed').textContent=((performance.now()-start)/1000).toFixed(2)+' s';
    $('meshParity').textContent=result.parity?.passed?`本次 GPU / CPU ${result.parity.points} 点通过`:'CPU 参考；GPU 未获本次验证';
    $('meshStatus').textContent=result.valid?`${engine.type} 完成。有效点 ${result.valid}，其余点已明确留空。`:'没有有效接收点。请检查端点距离、源／接收布局与单区域适用范围。';
    if(fallbackReason)setWarnings([...(active.warnings??[]),`WebGPU 未用于这次最终结果：${fallbackReason}。已明确使用 CPU 参考。`]);
    globalThis.utdLabValidation={backend:result.backend,parity:result.parity,valid:result.valid,total:w*w,sceneTriangles:active.mesh.triangles.length,edges:active.wedges.length};
  }catch(error){if(error.name!=='AbortError'&&rev===revision){setError(error.message||String(error));$('meshStatus').textContent='计算未完成，未展示未检查的结果。';pending('计算失败或超出适用范围。');}}
  finally{if(rev===revision){engine?.destroy();engine=null;controller=null;$('meshRun').disabled=false;$('meshCancel').disabled=true;}}
}
$('meshRun').addEventListener('click',run);
window.addEventListener('resize',()=>{if(result)renderResults(result);else view.draw();});
window.addEventListener('pagehide',()=>stop(false));
globalThis.utdLab={run,cancel:()=>stop(true),getState:()=>({modelName,sceneSupported:scene?.supported,triangles:scene?.mesh.triangles.length,edges:scene?.wedges.length,backend:result?.backend,running:!!controller,valid:result?.valid,revision,transport:selectedTransport}),getScene:()=>scene,getViewScene:()=>view.getScene()};
pending();updatePreview();
