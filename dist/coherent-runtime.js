import {solveCoherent,evaluateGrid,evaluatePoints,measureCoherent,relativeError,DEFAULT_OPTIONS} from './coherent-core.js';
export const DISPLAY_GRID=Object.freeze({width:144,height:108,xMin:-2.8,xMax:5.5,yMin:-3,yMax:3});
const now=()=>globalThis.performance?.now?.()??Date.now();
export function validateRequest(request){
 const o={...DEFAULT_OPTIONS,...request.options};
 if(!['cpu','auto'].includes(request.backend||'auto'))throw new Error('Unsupported backend');
 for(const [name,lo,hi] of [['gamma',0,1],['phase',-Math.PI,Math.PI],['inputPower',.25,2],['detectorCenter',-Math.PI/4,Math.PI/4],['detectorHalfAngle',.1,.6]])if(!Number.isFinite(o[name])||o[name]<lo-1e-12||o[name]>hi+1e-12)throw new RangeError(`${name} exceeds the published interface limits`);
 if(!['run','refine'].includes(request.action||'run'))throw new Error('Unsupported action');return o;
}
export async function executeCoherentTask(request,{progress=()=>{},gpuRunner}={}){
 const options=validateRequest(request),start=now(),timings={},warnings=[];
 progress('正在组装并求解完整双向耦合系统…');
 let t=now();const solution=solveCoherent(options);timings.solveMs=now()-t;
 if(request.action==='refine'){
  const nextN=options.n===64?96:144;if(nextN===options.n)throw new Error('144 节点已是本模式的最高参考分辨率。');
  progress(`正在比较每屏 ${options.n} 与 ${nextN} 个边界节点…`);
  t=now();const fine=solveCoherent({...options,n:nextN});
  // Both grids use the coarse exclusion mask; compare on its accepted domain.
  const coarseGrid=evaluateGrid(solution,DISPLAY_GRID),fineGrid=evaluatePoints(fine,coarseGrid.xy,fine.state);
  const coarse=[],refined=[];for(let i=0;i<coarseGrid.mask.length;i++)if(coarseGrid.mask[i])for(let m=0;m<6;m++){coarse.push(coarseGrid.fieldsRaw[i*6+m]);refined.push(fineGrid.fieldsRaw[i*6+m]);}
  const fieldError=relativeError(coarse,refined),detectorOperatorError=relativeError(solution.measurement.detectorMatrixRaw,fine.measurement.detectorMatrixRaw);
  const detectorSignalChange=Math.abs(solution.measurement.detectorFlux-fine.measurement.detectorFlux)/Math.max(Math.abs(fine.measurement.detectorFlux),1e-30);
  const threshold=2e-4,pass=fieldError<=threshold&&detectorOperatorError<=threshold;
  return {action:'refine',options,refinement:{coarseN:options.n,fineN:nextN,fieldError,detectorOperatorError,detectorSignalChange,threshold,pass,validPixels:coarseGrid.diagnostics.validPixels,elapsedMs:now()-t},timings:{totalMs:now()-start}};
 }
 progress('正在计算 CPU 双精度参考场与标量通量…');
 t=now();const cpuGrid=evaluateGrid(solution,DISPLAY_GRID);let grid=cpuGrid;timings.cpuFieldMs=now()-t;
 const measurement=solution.measurement,baseline=measureCoherent(solution,{...options,gamma:0});
 let backend='CPU 双精度',parity=null,gpuDiagnostics=null;timings.gpuMs=null;
 if((request.backend||'auto')==='auto'){
  if(!gpuRunner&&!globalThis.navigator?.gpu)warnings.push('此浏览器的计算 Worker 未提供 WebGPU；已使用 CPU 双精度参考。');
  else{
   progress('正在用 WebGPU 计算复场，并逐点核对 CPU 参考…');
   try{
    const runner=gpuRunner??(await import('./coherent-gpu.js')).computeCoherentGPU;
    const gpu=await runner(solution,DISPLAY_GRID,{reference:grid,state:solution.state,onProgress:f=>progress(`WebGPU 场计算与回读：${Math.round(100*f)}%`)});
    grid=gpu;backend='CPU + WebGPU';parity=gpu.parity;gpuDiagnostics={...gpu.diagnostics,adapter:gpu.adapter,buildId:gpu.buildId,shaderFingerprintFNV1a:gpu.shaderFingerprintFNV1a};timings.gpuMs=gpu.timings?.gpuDispatchReadbackWallMs??gpu.timings?.dispatchReadbackMs??gpu.timings?.gpuMs??null;
    timings.gpuDetails=gpu.timings??null;
    if(!parity?.passed)throw new Error('GPU 未提供通过的全场准入结果');
   }catch(error){
    backend='CPU 双精度';grid=cpuGrid;warnings.push(`WebGPU 未获准显示：${error.message}。已切回 CPU 参考。`);gpuDiagnostics=error.diagnostic??error.diagnostics??{error:error.message};parity={...gpuDiagnostics?.parity,passed:false};
   }
  }
 }
 if(!(solution.diagnostics.orthonormalizationError<1e-10))throw new Error('源通量正交性检查未通过，拒绝显示结果。');
 if(!solution.measurement.diagnostics.energyPass)throw new Error('全角通量算子检查未通过，拒绝显示结果。');
 timings.totalMs=now()-start;
 return {action:'run',options,screens:solution.screens,grid,measurement,baseline,backend,parity,timings,warnings,diagnostics:{...solution.diagnostics,grid:grid.diagnostics,gpu:gpuDiagnostics,scope:'2D coupled scalar Helmholtz; calibrated scalar angular flux; no electromagnetic watts',executionContext:typeof WorkerGlobalScope!=='undefined'&&globalThis instanceof WorkerGlobalScope?'browser module Worker':'non-browser test execution',serverCompute:false}};
}
