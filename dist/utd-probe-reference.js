// Read-only CPU reference and decoder for optional failure diagnostics.
import {evaluatePoint,computeReceiverPoint,stablePlaneExcess} from './utd-scene.js';
import {coefficient,stableCotTransition,erfcxScaled,cis,mul,abs2} from './utd-math.js';
export const UTD_PROBE_STRIDE_VEC4=34;
export const UTD_PROBE_ROWS=[
 ['dst.x','dst.y','dst.z','pixel.x'],
 ['stationary.x','stationary.y','stationary.z','ro'],
 ['wo.x','wo.y','wo.z','sinBeta'],
 ['phi_i','phi_o','n','L'],
 ['excess','theta','phase.re','phase.im'],
 ['Dsoft.re','Dsoft.im','Dhard.re','Dhard.im'],
 ['edgeSoft.re','edgeSoft.im','edgeHard.re','edgeHard.im'],
 ['directWeight','reflectionCount','cisNormSquared','geometricPower'],
 ...['difference.plus','difference.minus','sum.plus','sum.minus'].flatMap(prefix=>[
  ['beta','side','nearest','delta'].map(k=>`${prefix}.${k}`),
  ['t','h','ratio','u'].map(k=>`${prefix}.${k}`),
  ['z','H.re','H.im','kL'].map(k=>`${prefix}.${k}`),
  ['cot.re','cot.im','sincT','cosH'].map(k=>`${prefix}.${k}`),
 ]),
 ['atanInput.wiN','atanInput.wiT','atanInput.woN','atanInput.woT'],
 ['literalH.446.re','literalH.446.im','literalH.639.re','literalH.639.im'],
 ['literalH.812.re','literalH.812.im','literalH.986.re','literalH.986.im'],
 ['k','L','sinBeta','coefficientDenominator'],
 ['reconstructedSoft.re','reconstructedSoft.im','reconstructedHard.re','reconstructedHard.im'],
 ['reconstructedPower','geometricPower','directPlusReflectionInvalid','boundaryR'],
 ['cis.re','cis.im','literalSinc.024979','literalCos.012489'],
 ['sqrtKL','inverseSqrtRo','rawSinTheta','rawCosTheta'],
 ['rawAtan2.wi','rawAtan2.wo','sqrtRo','clippedEdgeLength'],
 ['expMinusExcess','sqrtK','fmaResidual','mulSubResidual'],
];
const add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),scale=(a,s)=>a.map(x=>x*s),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),len=a=>Math.hypot(...a);
const sinc=x=>x===0?1:Math.sin(x)/x;
const power=(s,h,b)=>b==='soft'?abs2(s):b==='hard'?abs2(h):(abs2(s)+abs2(h))/2;
function makeUTDProbeReferencesInternal(scene,pixels,width,height){
 const edgeLimit=Math.min(2,scene.wedges.length),k=scene.params.k,di=scene.params.incidentDirection,wi=scale(di,-1),boundary=scene.params.boundary;
 return pixels.flatMap(([x,y],pixelIndex)=>{
  const dst=computeReceiverPoint(scene,x,y,width,height),cpu=evaluatePoint(scene,dst,{details:true,boundaryEpsilon:2e-6});
  let goSoft=[...cpu.soft],goHard=[...cpu.hard];
  for(const c of cpu.contributions||[]){goSoft=sub(goSoft,c.soft);goHard=sub(goHard,c.hard);}
  const traces=scene.wedges.slice(0,edgeLimit).map(w=>{
   const mid=scale(add(w.a,w.b),.5),e=scale(sub(w.b,w.a),1/len(sub(w.b,w.a))),c=dot(di,e),sb=Math.sqrt(Math.max(0,1-c*c));
   const d=sub(dst,mid),da=dot(d,e),rho=len(sub(d,scale(e,da))),axial=da-rho*c/sb,point=add(mid,scale(e,axial)),out=sub(dst,point),ro=len(out),wo=scale(out,1/ro);
   const wiN=dot(wi,w.nff),wiT=dot(wi,w.tff),woN=dot(wo,w.nff),woT=dot(wo,w.tff),phii=Math.atan2(wiN,wiT),rawPo=Math.atan2(woN,woT),phio=rawPo<0?rawPo+2*Math.PI:rawPo;
   const n=2-w.alpha/Math.PI,L=ro*sb*sb,excess=stablePlaneExcess(di,point,dst),theta=-k*excess,circle=cis(theta),phase=scale(circle,1/Math.sqrt(ro));
   const ds=coefficient(k,n,phii,phio,L,sb,'soft',2e-6),dh=coefficient(k,n,phii,phio,L,sb,'hard',2e-6),es=mul(ds,phase),eh=mul(dh,phase),rows=[];
   rows.push([...dst,x],[...point,ro],[...wo,sb],[phii,phio,n,L],[excess,theta,...phase],[...ds,...dh],[...es,...eh],[cpu.directWeight,cpu.reflectionCount,abs2(circle),cpu.geometricPower]);
   for(const beta of [phio-phii,phio+phii])for(const side of [1,-1]){
    const nearest=Math.floor((beta+side*Math.PI)/(2*Math.PI*n)+.5),delta=beta-(2*Math.PI*n*nearest-side*Math.PI),t=delta*.5,h=delta/(2*n),ratio=Math.sign(delta)*n*sinc(t)*Math.cos(h)/sinc(h),kL=k*L,u=Math.sqrt(2*kL)*Math.abs(Math.sin(t)),z=u/(1+u),H=erfcxScaled(z),cot=stableCotTransition(beta,n,kL,side,2e-6);
    rows.push([beta,side,nearest,delta],[t,h,ratio,u],[z,...H,kL],[...cot,sinc(t),Math.cos(h)]);
   }
   rows.push([wiN,wiT,woN,woT],[...erfcxScaled(.4461701888069467),...erfcxScaled(.6392366981032546)],[...erfcxScaled(.812819295910126),...erfcxScaled(.9860957183805704)],[k,L,sb,2*n*Math.sqrt(2*Math.PI*k)*sb]);
   rows.push(null,null,[...circle,sinc(.024979197860971247),Math.cos(.012489598930485624)],[Math.sqrt(k*L),1/Math.sqrt(ro),Math.sin(theta),Math.cos(theta)],[phii,rawPo,Math.sqrt(ro),len(sub(w.b,w.a))]);
   const fk=Math.fround(k),a=Math.fround(1+2**-23),fused=Math.fround(a*fk-fk),unfused=Math.fround(Math.fround(a*fk)-fk);
   rows.push([Math.exp(-excess),Math.sqrt(k),fused,unfused]);
   return {pixel:[x,y],pixelIndex,edgeId:w.edgeId,rows,es,eh,fmaPermitted:[fused,unfused]};
  });
  const sumSoft=traces.reduce((s,t)=>add(s,t.es),goSoft),sumHard=traces.reduce((s,t)=>add(s,t.eh),goHard);
  return traces.map((t,edgeIndex)=>{t.rows[28]=[...sumSoft,...sumHard];t.rows[29]=[power(sumSoft,sumHard,boundary),cpu.geometricPower,null,boundary==='soft'?-1:boundary==='hard'?1:0];const {es,eh,...record}=t;return {...record,edgeIndex,originalCPU:{f:cpu.f,valid:cpu.valid,edgeCount:cpu.edgeCount}};});
 });
}
export function decodeUTDProbe(data,pixels,edgeCount){
 const count=Math.min(2,edgeCount);return pixels.flatMap((pixel,pixelIndex)=>Array.from({length:count},(_,edgeIndex)=>{
  const offset=(pixelIndex*count+edgeIndex)*34*4;
  const rows=UTD_PROBE_ROWS.map((keys,row)=>Object.fromEntries(keys.map((key,col)=>[key,data[offset+row*4+col]])));
  return {pixel,pixelIndex,edgeIndex,rows};
 }));
}
// Expected function outputs evaluated at the GPU-reported inputs, to isolate stages.
export function assessUTDProbe(decodedRecords){
 return decodedRecords.map(record=>{
  const v=Object.assign({},...record.rows),r={pixel:record.pixel,edgeIndex:record.edgeIndex};
  try{
   const k=v.k,n=v.n,pi=v.phi_i,po=v.phi_o,L=v.L,sb=v.sinBeta;
   r.coefficientAtGPUInputs={soft:coefficient(k,n,pi,po,L,sb,'soft',2e-6),hard:coefficient(k,n,pi,po,L,sb,'hard',2e-6)};
   r.phaseAtGPUInputs=scale(cis(v.theta),1/Math.sqrt(v.ro));
   r.cisAtGPUInputs=cis(v.theta);
   r.anglesAtGPUInputs={phi_i:Math.atan2(v['atanInput.wiN'],v['atanInput.wiT']),rawPhi_o:Math.atan2(v['atanInput.woN'],v['atanInput.woT'])};
   r.branches=['difference.plus','difference.minus','sum.plus','sum.minus'].map(prefix=>{
    const z=v[`${prefix}.z`],beta=v[`${prefix}.beta`],side=v[`${prefix}.side`],kL=v[`${prefix}.kL`],t=v[`${prefix}.t`],h=v[`${prefix}.h`];
    return {prefix,HAtGPUInput:erfcxScaled(z),cotAtGPUInputs:stableCotTransition(beta,n,kL,side,2e-6),sincAtGPUInput:sinc(t),cosAtGPUInput:Math.cos(h),uAtGPUInputs:Math.sqrt(2*kL)*Math.abs(Math.sin(t))};
   });
   const cot=prefix=>[v[`${prefix}.cot.re`],v[`${prefix}.cot.im`]],a=add(cot('difference.plus'),cot('difference.minus')),b=add(cot('sum.plus'),cot('sum.minus'));
   const denom=2*n*Math.sqrt(2*Math.PI*k)*sb,prefactor=[Math.SQRT1_2,-Math.SQRT1_2];
   r.coefficientFromGPUCots={soft:scale(mul(prefactor,sub(a,b)),-1/denom),hard:scale(mul(prefactor,add(a,b)),-1/denom)};
   r.edgeFromGPUCoefficientAndPhase={soft:mul([v['Dsoft.re'],v['Dsoft.im']],[v['phase.re'],v['phase.im']]),hard:mul([v['Dhard.re'],v['Dhard.im']],[v['phase.re'],v['phase.im']])};
   const fk=Math.fround(k),factor=Math.fround(1+2**-23);
   r.intrinsicsAtGPUInputs={sqrtKL:Math.sqrt(k*L),sqrtK:Math.sqrt(k),sqrtRo:Math.sqrt(v.ro),inverseSqrtRo:1/Math.sqrt(v.ro),expMinusExcess:Math.exp(-v.excess),rawSinTheta:Math.sin(v.theta),rawCosTheta:Math.cos(v.theta),fmaFused:Math.fround(factor*fk-fk),fmaUnfused:Math.fround(Math.fround(factor*fk)-fk)};
  }catch(error){r.referenceError=String(error?.message||error);}
  return r;
 });
}

export function makeUTDProbeReferences(scene,pixels,width,height){
 try{return makeUTDProbeReferencesInternal(scene,pixels,width,height);}
 catch(error){return pixels.map(pixel=>({pixel,referenceError:String(error?.message||error),scope:'Diagnostic CPU tracing assumes finite, non-edge-parallel stationary points. GPU trace and original parity failure must be retained.'}));}
}
export function makeUTDProbeShader(source,template,pixels,width,height){
 if(!Array.isArray(pixels)||pixels.length!==3||!pixels.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isInteger)&&p[0]>=0&&p[0]<width&&p[1]>=0&&p[1]<height))throw new Error('Diagnostic pixels must be three in-bounds integer coordinate pairs.');
 const declaration='const UTD_DIAGNOSTIC_PIXELS = array<vec2u,3>('+pixels.map(([x,y])=>'vec2u('+x+'u,'+y+'u)').join(',')+');';
 const marker=/const UTD_DIAGNOSTIC_PIXELS = array<vec2u,3>\([^\n]*\);/;
 if(!marker.test(template))throw new Error('Diagnostic shader pixel declaration missing.');
 return source+'\n'+template.replace(marker,declaration);
}
