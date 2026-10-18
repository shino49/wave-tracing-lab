/** Additive incoherent environment and exactly one diffuse indirect vertex.
 * Exact camera/sample geometry; source routes remain coherent up to the first
 * diffuse scattering event. No image-space interpolation or irradiance cache. */
import {slabAbsorptionAt,slabIsInvisible} from './coherent-mesh-slab.js';
import {compileCoherentSource} from './coherent-mesh-core.js';
import {coherentRouteCapacity,evaluateMeshTransport} from './coherent-mesh-transport.js';
import {compileCoherentSpectralSource,prepareCoherentSpectralSample,evaluateCoherentSpectralSampleCPU,packCoherentSpectralTemplates} from './coherent-mesh-spectral-templates.js';
import {packCoherentSpectralCotDeltas} from './coherent-mesh-spectral-gpu.js';
import {prepareCoherentSpectrum} from './coherent-mesh-spectrum.js';
import {traceCoherentCamera,traceCoherentRay,projectCoherentSurfaceRoutes,diffuseReflectanceAt} from './coherent-mesh-camera.js';
import {cosineDirection,environmentAngularRadiance} from './coherent-mesh-environment.js';
import {cieXYZ,hashSpectralIndex} from './plt-spectrum.js';
const scalars=['coherent','incoherent','geometric'],xyzs=scalars.map(k=>k+'XYZ');
const now=()=>globalThis.performance?.now?.()??Date.now();
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
const front=h=>h.status==='diffuse'&&h.normal.reduce((s,v,i)=>s+v*h.viewDirection[i],0)>0;
const seedFor=(seed,index,domain)=>hashSpectralIndex(seed,index,0,domain)>>>0;
const rho=(scene,hit,nm)=>diffuseReflectanceAt(scene.materials[hit.materialIndex].reflectance,nm);
function validateFields(fields,count,spectral){
 for(const key of ['coherent','incoherent',...(spectral?['geometric',...xyzs]:[])]){
  if(!fields?.[key]||fields[key].length!==count*(key.endsWith('XYZ')?3:1))throw new Error('Malformed lighting field result dimensions');
  for(const v of fields[key])if(!Number.isFinite(v)||v<0)throw new Error('Invalid lighting field intensity');
 }
}
export async function integrateCoherentLighting({scene,source,camera,options:o,lighting,runner=null,spectralPlan,spatialSources,images,signal,check=()=>{},progress=()=>{},preview=null,timings={},diagnostics={}}){
 if(!lighting||(!lighting.environment&&!lighting.indirectSamples))return;
 const started=now(),n=o.resolution,q=o.pixelQuadrature,pixelWeight=1/(q*q),env=lighting.environment,ns=lighting.indirectSamples,budget=camera.mirrorBounces??2;
 const spectral=spectralPlan.mode!=='monochrome',capacity=coherentRouteCapacity(scene,o).count,monoXYZ=cieXYZ(source.wavelengthMm*1e6);
 const envPlan=env?prepareCoherentSpectrum({power:1,wavelengthMm:(env.wavelengthNm??550)/1e6,spectrum:env.spectrum},{spectralSamples:o.spectralSamples,seed:seedFor(env.seed,0,0x45535043)}):null;
 const rootSeed=env?.seed??o.spectralSeed??7391,activeSlab=!slabIsInvisible(scene.slab);
 const attenuation=(nm,L)=>L>0?Math.exp(-slabAbsorptionAt(scene.slab,nm)*L):1;
 function recordSlab(ray){if(!activeSlab)return;diagnostics.lightingSlabEvents=(diagnostics.lightingSlabEvents??0)+(ray.slabEvents?.length??0);diagnostics.lightingSlabUnsupportedRays=(diagnostics.lightingSlabUnsupportedRays??0)+(ray.status==='slab-unsupported'?1:0);diagnostics.lightingSlabTruncatedPower=(diagnostics.lightingSlabTruncatedPower??0)+(ray.slabTruncatedPower??0);}
 for(const key of ['lightingEnvironmentRays','lightingEnvironmentEscapes','lightingSecondaryHits','lightingSourceSamples','lightingSlabUnsupportedRays'])diagnostics[key]??=0;
 let operations=0;
 async function checkpoint(){check();if(++operations%128===0){await tick();check();}}
 function traceFrom(hit,direction,sampleSeed){const ray=traceCoherentRay(scene,hit.point,direction,budget,scene.triangles[hit.triangleId],sampleSeed);recordSlab(ray);return ray;}
 async function visibleSky(hit,index,domain){
  let total=0;const escapes=activeSlab?[]:null,seed=seedFor(rootSeed,index,domain);
  for(let k=0;k<env.samples;k++){
   const ray=traceFrom(hit,cosineDirection(hit.normal,k,env.samples,seed),seedFor(seed,k,0x534b534c));diagnostics.lightingEnvironmentRays++;
   if(ray.status==='miss'){diagnostics.lightingEnvironmentEscapes++;const weight=ray.throughput*environmentAngularRadiance(env,ray.escapeDirection);total+=weight;if(escapes)escapes.push({weight:weight/env.samples,glassLength:ray.glassLength??0});}
   await checkpoint();
  }
  return {amplitude:total/env.samples,escapes};
 }
 function addSky(pixel,index,amplitude,primary=null,secondary=null,extraGlassLength=0,escapes=null){
  if(amplitude===0)return;
  let value=0;const xyz=[0,0,0];
  for(const node of envPlan.sampleNodes(index)){
   const glassLength=(primary?.glassLength??0)+(secondary?.glassLength??0)+extraGlassLength,skyWeight=escapes?escapes.reduce((sum,r)=>sum+r.weight*attenuation(node.wavelengthNm,r.glassLength),0):1;
   const weight=node.weight*amplitude*skyWeight*attenuation(node.wavelengthNm,glassLength)*(primary?rho(scene,primary,node.wavelengthNm):1)*(secondary?rho(scene,secondary,node.wavelengthNm):1);
   value+=weight;for(let c=0;c<3;c++)xyz[c]+=weight*node.xyz[c];
  }
  for(const key of scalars)images[key][pixel]+=value*pixelWeight;
  for(const key of xyzs)for(let c=0;c<3;c++)images[key][3*pixel+c]+=xyz[c]*pixelWeight;
 }
 const compiled=ns?spatialSources.map(emitter=>({...emitter,compiled:spectral?compileCoherentSpectralSource(scene,emitter.source):compileCoherentSource(scene,emitter.source)})):[];
 async function sourceBatch(batch,emitter){
  const count=batch.length,records=new (runner?Float32Array:Float64Array)(count*capacity*(spectral?16:4));
  const spectralRecords=spectral&&runner?new Float32Array(count*spectralPlan.samples*8):null;
  const fields=Object.fromEntries([...scalars.map(k=>[k,new Float64Array(count)]),...xyzs.map(k=>[k,new Float64Array(3*count)])]);
  for(let s=0;s<count;s++){
   const b=batch[s],t=projectCoherentSurfaceRoutes(spectral?prepareCoherentSpectralSample(scene,emitter.compiled,b.secondary.point,o):evaluateMeshTransport(scene,emitter.compiled,b.secondary.point,o),b.secondary.point,b.secondary.normal);
   if((t.slabDiagnostics?.unsupportedSide||t.slabDiagnostics?.possibleUnsupportedReflectedSlab))diagnostics.sourceSlabUnsupportedSamples=(diagnostics.sourceSlabUnsupportedSamples??0)+1;
   if(t.routes.length>capacity)throw new Error('Lighting route capacity overflow');
   const factor=nm=>b.primary.throughput*b.secondary.throughput*rho(scene,b.primary,nm)*rho(scene,b.secondary,nm)/Math.PI*attenuation(nm,(b.primary.glassLength??0)+(b.secondary.glassLength??0));
   if(spectral){
    // Source node realization is domain-separated from both sampled directions.
    const nodeIndex=seedFor(rootSeed,b.index,0x534e4f44);
    if(runner){spectralPlan.writePackedNodes(nodeIndex,spectralRecords,s*spectralPlan.samples*8);for(let z=0;z<spectralPlan.samples;z++){const at=(s*spectralPlan.samples+z)*8,nm=Math.max(360,Math.min(830,1e6/(spectralRecords[at]+spectralRecords[at+1])));spectralRecords[at+3]*=factor(nm);spectralRecords[at+7]=slabAbsorptionAt(scene.slab,nm);}packCoherentSpectralTemplates(t.templates,records,s*capacity*16,packCoherentSpectralCotDeltas);}
    else{const value=evaluateCoherentSpectralSampleCPU(t.templates,spectralPlan.sampleNodes(nodeIndex).map(node=>({...node,weight:node.weight*factor(node.wavelengthNm)})));for(const k of scalars)fields[k][s]=value[k];for(const k of xyzs)fields[k].set(value[k],3*s);}
   }else{
    const f=factor(source.wavelengthMm*1e6);fields.geometric[s]=t.geometricPower*f;
    let re=0,im=0,inc=0;for(let j=0;j<t.routes.length;j++){const r=t.routes[j],a=r.amplitude*Math.sqrt(f),at=(s*capacity+j)*4;records[at]=a;records[at+1]=r.phaseRadReduced;if(!runner){re+=a*Math.cos(r.phaseRadReduced);im+=a*Math.sin(r.phaseRadReduced);inc+=a*a;}}
    if(!runner){fields.coherent[s]=re*re+im*im;fields.incoherent[s]=inc;}
   }
   diagnostics.lightingSourceSamples++;await checkpoint();
  }
  let result=fields;
  if(runner){try{result=await runner.evaluate({records,sampleCount:count,routeCount:capacity,...(spectral?{spectralRecords,wavelengthCount:spectralPlan.samples}:{}),signal});validateFields(result,count,spectral);}catch(error){throw Object.assign(error instanceof Error?error:new Error(String(error)),{coherentGPUFailure:true});}}
  validateFields(result,count,spectral);check();
  for(const [key,tk] of [['readbackMs','readbackMs'],['queueWaitMs','gpuQueueWaitMs'],['uploadEncodeMs','gpuUploadEncodeMs']])timings[tk]=(timings[tk]??0)+(result.timing?.[key]??0);
  const weight=pixelWeight*emitter.weight/ns;
  for(let s=0;s<count;s++){
   const p=batch[s].pixel,inc=result.incoherent[s],coh=o.coherence*result.coherent[s]+(1-o.coherence)*inc,geo=spectral?result.geometric[s]:fields.geometric[s];
   images.coherent[p]+=coh*weight;images.incoherent[p]+=inc*weight;images.geometric[p]+=geo*weight;images.crossTerm[p]+=(coh-inc)*weight;
   for(let c=0;c<3;c++){const at=3*p+c,fi=3*s+c,iv=spectral?result.incoherentXYZ[fi]:inc*monoXYZ[c],cv=spectral?o.coherence*result.coherentXYZ[fi]+(1-o.coherence)*iv:coh*monoXYZ[c],gv=spectral?result.geometricXYZ[fi]:geo*monoXYZ[c];images.coherentXYZ[at]+=cv*weight;images.incoherentXYZ[at]+=iv*weight;images.geometricXYZ[at]+=gv*weight;}
  }
 }
 for(let y=0;y<n;y++){
  check();let batch=[];
  const flush=async()=>{if(batch.length){for(const emitter of compiled)await sourceBatch(batch,emitter);batch=[];}};
  for(let x=0;x<n;x++)for(let sy=0;sy<q;sy++)for(let sx=0;sx<q;sx++){
   const pixel=y*n+x,index=(pixel*q+sy)*q+sx,primary=traceCoherentCamera(scene,camera,x,y,n,n,(sx+.5)/q,(sy+.5)/q,o.spectralSeed);
   if(env&&primary.status==='miss')addSky(pixel,index,primary.throughput*environmentAngularRadiance(env,primary.escapeDirection),null,null,primary.glassLength??0);
   if(front(primary)){
    if(env){const sky=await visibleSky(primary,index,0x50534b59);addSky(pixel,index,primary.throughput*(sky.escapes?1:sky.amplitude),primary,null,0,sky.escapes);}
    const directionSeed=seedFor(rootSeed,index,0x494e4452);
    for(let k=0;k<ns;k++){
     const secondary=traceFrom(primary,cosineDirection(primary.normal,k,ns,directionSeed),seedFor(directionSeed,k,0x494e534c));
     if(front(secondary)){
      diagnostics.lightingSecondaryHits++;
      const pairIndex=seedFor(rootSeed,index,k+0x50414952);
      if(env){const sky=await visibleSky(secondary,pairIndex,0x53534b59);addSky(pixel,seedFor(rootSeed,pairIndex,0x45534e44),primary.throughput*secondary.throughput*(sky.escapes?1:sky.amplitude)/ns,primary,secondary,0,sky.escapes);}
      batch.push({pixel,index:pairIndex,primary,secondary});if(batch.length===128)await flush();
     }
     await checkpoint();
    }
   }
   await checkpoint();
  }
  await flush();check();progress({phase:'lighting',fraction:(y+1)/n,rowStart:y,rowCount:1,message:`环境与一次漫反射 · ${y+1}/${n} 行`});if(preview)await preview({rowStart:y,rowCount:1});await tick();check();
 }
 timings.lightingMs=(timings.lightingMs??0)+now()-started;
}
