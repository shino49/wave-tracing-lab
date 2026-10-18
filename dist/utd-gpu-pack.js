/** GPU ABI for utd-functions.wgsl + utd-compute.wgsl. Coordinates are rebased.
 * Float32 positions are never formed from large unreduced world coordinates.
 * The CPU geometry object remains unchanged. No renderer/device side effects.
 */
export const UTD_GPU_BOUNDARY_EPS = 2e-6;
export const UTD_GPU_LIMITS = Object.freeze({ maxTriangles: 128, maxPhysicalEdges: 192, maxResolution: 97 });
export const UTD_GPU_UNIFORM_BYTES = 160;
export const UTD_GPU_BINDINGS = Object.freeze({ uniforms: 0, triangles: 1, edges: 2, adjacency: 3, output: 4, facets: 5 });
const sub = (a,b)=>a.map((x,i)=>x-b[i]);
const dot = (a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const norm = a=>Math.hypot(...a);
const paddedFloat = (length,stride)=>new Float32Array(Math.max(stride,length));

export function packUTDSceneGPU(scene,width,height,boundaryEps=UTD_GPU_BOUNDARY_EPS) {
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>97||height>97)
    throw new Error('UTD GPU resolution must be integer 1..97 in each dimension.');
  if(scene.mesh.triangles.length>128||scene.physicalWedges.length>192||scene.wedges.length>192)
    throw new Error('UTD GPU safety cap is 128 triangles and 192 physical/clipped edges.');
  if(scene.gpuSafety && !scene.gpuSafety.safe) throw new Error(`UTD GPU numerical guard: ${scene.gpuSafety.reasons.join('; ')}`);
  if(Math.abs(boundaryEps-UTD_GPU_BOUNDARY_EPS)>1e-12)
    throw new Error('GPU seam epsilon must match the compiled UTD_BOUNDARY_EPS (2e-6 radians).');
  const p=scene.params, origin=scene.gpuSafety?.rebaseOrigin??scene.mesh.bounds.center;
  const local=v=>sub(v,origin);
  const physical=scene.physicalWedges;
  const frames=new Map(scene.fullWedges.map(w=>[w.edgeId,w]));
  const physicalIndices=new Map(physical.map((w,i)=>[w.edgeId,i]));
  const allEdges=[...physical.map(w=>({...w,...(frames.get(w.edgeId)??{}),hasFrame:frames.has(w.edgeId)})),
    ...scene.wedges.map(w=>({...w,hasFrame:true}))];
  const triangles=paddedFloat(scene.mesh.triangles.length*12,12);
  scene.mesh.triangles.forEach((t,i)=>triangles.set([...local(t.a),0,...local(t.b),0,...local(t.c),0],i*12));
  const edges=paddedFloat(allEdges.length*24,24);
  const adjacency=new Uint32Array(Math.max(4,(allEdges.length+scene.facets.length)*4));
  const setMask=(row,ids)=>{for(const id of ids){if(id<0||id>=128)throw new Error('Invalid triangle adjacency ID.');adjacency[row*4+(id>>>5)]|=1<<(id&31);}};
  allEdges.forEach((w,i)=>{
    const delta=sub(w.b,w.a),length=norm(delta),axis=delta.map(x=>x/length);
    const a0=dot(sub(w.originalA??w.a,w.a),axis),a1=dot(sub(w.originalB??w.b,w.a),axis);
    const nff=w.nff??[0,0,0],tff=w.tff??[0,0,0],nbf=w.nbf??[0,0,0];
    edges.set([...local(w.a),w.hasFrame?1:0,...local(w.b),physicalIndices.get(w.edgeId)??-1,
      ...nff,Math.min(a0,a1),...tff,Math.max(a0,a1),...nbf,0,
      w.alpha,length,w.clippedStart?1:0,w.clippedEnd?1:0],i*24);
    setMask(i,w.triangleIds);
  });
  const facets=paddedFloat(scene.facets.length*8,8);
  scene.facets.forEach((facet,i)=>{
    facets.set([...facet.normal,0,...local(facet.point),0],i*8);
    setMask(allEdges.length+i,facet.triangleIds);
  });
  function createUniforms(baseRow=0,batchRows=8) {
    if(!Number.isInteger(baseRow)||baseRow<0||baseRow>=height||!Number.isInteger(batchRows)||batchRows<1)
      throw new Error('Invalid UTD GPU row batch.');
    const data=new ArrayBuffer(UTD_GPU_UNIFORM_BYTES),f=new Float32Array(data),u=new Uint32Array(data);
    u.set([width,height,baseRow,Math.min(batchRows,height-baseRow)],0);
    u.set([scene.mesh.triangles.length,physical.length,scene.wedges.length,scene.facets.length],4);
    f.set([p.k,p.boundary==='soft'?-1:p.boundary==='hard'?1:0,p.minKr,p.minSinBeta],8);
    f.set([...local(p.sourceOrigin),scene.incomingCone.radius0],12);
    f.set([...p.incidentDirection,scene.incomingCone.tanAngle],16);
    f.set([...local(p.detectorCenter),p.detectorWidthMm],20);
    f.set([...p.detectorU,p.detectorHeightMm],24);
    f.set([...p.detectorV,scene.region?1:0],28);
    f.set([...(scene.region?local(scene.region.center):[0,0,0]),scene.region?.radius??1],32);
    f.set([p.endpointFresnelGuard,boundaryEps,scene.supported?1:0,p.surface==='open-plate'?1:0],36);
    return new Uint8Array(data);
  }
  return {triangles,edges,adjacency,facets,uniforms:createUniforms(),createUniforms,
    width,height,outputByteLength:width*height*16,rebaseOrigin:[...origin],boundaryEpsilon:boundaryEps,
    bindings:UTD_GPU_BINDINGS,uniformByteLength:UTD_GPU_UNIFORM_BYTES,
    counts:{triangles:scene.mesh.triangles.length,physicalEdges:physical.length,clippedEdges:scene.wedges.length,facets:scene.facets.length}};
}
