/** Experimental resident geometry upload. No source geometry leaves the browser.
 * All offsets/strides are in 16-byte vec4 records, not bytes. WGSL consumes one
 * read-only array<vec4u>. Physical tolerances remain the double reference's;
 * broad-phase padding only enlarges candidate sets. Ambiguous f32 predicates
 * flag the batch for rejection, never authorize a different physical path.
 */
export const RESIDENT_GEOMETRY_LIMITS=Object.freeze({triangles:4096,wedges:24576,nodes:8191,stack:32,materials:4096});
export const RESIDENT_GEOMETRY_LAYOUT=Object.freeze({version:1,header:8,triangle:5,wedge:8,node:4,material:1,emitter:1});
export const RESIDENT_GEOMETRY_FLAGS=Object.freeze({invalid:1,ambiguous:2});
export const RESIDENT_GEOMETRY_REASONS=Object.freeze({"rayCoplanar": 256, "rayParallel": 512, "rayOrigin": 1024, "rayBarycentric": 2048, "rayInterval": 4096, "coneInterval": 8192, "coneFlat": 16384, "coneLeft": 32768, "coneThinInterval": 65536, "coneMinimum": 131072, "coneSection": 262144, "coneTie": 524288, "clipDegenerate": 1048576, "clipDiscriminant": 2097152, "clipSlabBoundary": 4194304, "clipSlabParallel": 8388608, "clipEmpty": 16777216, "clipEndpoint": 33554432, "clipCollapse": 67108864});
const F32_EPS=2**-23,MAX_F32=3.4028234663852886e38;
const vsub=(a,b)=>a.map((x,i)=>x-b[i]),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],len=a=>Math.hypot(...a);
const fail=message=>{throw new RangeError(`Resident geometry: ${message}`);};
const checkInt=(v,lo,hi,name)=>{if(!Number.isSafeInteger(v)||v<lo||v>hi)fail(`${name} outside ${lo}…${hi}`);return v;};
const checkVec=(v,name)=>{if(!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))fail(`${name} must be a finite 3-vector`);return v;};
const epsilon=(...points)=>Math.max(1e-10,128*Number.EPSILON*Math.max(1,...points.flat().map(Math.abs)));
const scratch=new ArrayBuffer(4),scratchF=new Float32Array(scratch),scratchU=new Uint32Array(scratch);
function adjacent(v,up){scratchF[0]=v;if(v===0)return up?2**-149:-(2**-149);scratchU[0]+=((v>0)===up)?1:-1;return scratchF[0];}
function outward(v,up){let f=Math.fround(v);if(!Number.isFinite(f))fail('unrepresentable f32 bound');if(up?f<v:f>v)f=adjacent(f,up);return f;}
const scalar=(v,name,{positive=false}={})=>{const f=Math.fround(v);if(!Number.isFinite(v)||!Number.isFinite(f)||(positive&&!(v>0))||(v!==0&&f===0))fail(`${name} is unresolved in f32`);return f;};

export function packResidentScene(scene){
 const L=RESIDENT_GEOMETRY_LIMITS,S=RESIDENT_GEOMETRY_LAYOUT;
 if(scene?.type!=='plt-triangle-scene'||!Array.isArray(scene.triangles)||!Array.isArray(scene.wedges)||!scene.bvh)fail('a prepared immutable PLT scene is required');
 const nt=checkInt(scene.triangles.length,1,L.triangles,'triangle count'),nw=checkInt(scene.wedges.length,0,L.wedges,'wedge count');
 const nn=checkInt(scene.bvh.nodes.length,1,L.nodes,'BVH node count');
 const origin=checkVec(scene.bounds?.center,'bounds center').slice();
 let inputError=0,vertexInputError=0,minFeature=Infinity,maxCoordinate=0;
 const unique=new Map(),sourceVertices=new Set(),axisPlanes=[new Map(),new Map(),new Map()];
 const local=(p,track=false)=>{checkVec(p,'point');const d=vsub(p,origin),q=d.map(x=>scalar(x,'position'));
  if(track){const key=q.join(','),source=p.join(',');if(unique.has(key)&&unique.get(key)!==source)fail('distinct vertices collapse after f32 rebase');unique.set(key,source);sourceVertices.add(source);}
  for(let i=0;i<3;i++){inputError=Math.max(inputError,Math.abs(d[i]-q[i]));if(track)vertexInputError=Math.max(vertexInputError,Math.abs(d[i]-q[i]));maxCoordinate=Math.max(maxCoordinate,Math.abs(d[i]));}return q;};
 const tris=scene.triangles.map((t,i)=>{
  if(t.id!==i)fail('triangle IDs must be stable dense array indices');
  const a=local(t.a,true),b=local(t.b,true),c=local(t.c,true),ab=vsub(b,a),ac=vsub(c,a),bc=vsub(c,b),area2=len(cross(ab,ac)),edge=Math.max(len(ab),len(ac),len(bc));
  for(let k=0;k<3;k++)if(a[k]===b[k]&&a[k]===c[k]){if(t.a[k]!==t.b[k]||t.a[k]!==t.c[k])fail(`triangle ${i} loses a nonconstant coordinate in f32`);axisPlanes[k].set(a[k],t.a[k]);}
  const originalArea2=len(cross(vsub(t.b,t.a),vsub(t.c,t.a)));
  if(!(area2>0)||!Number.isFinite(Math.fround(area2))||Math.abs(area2/originalArea2-1)>1e-5)fail(`triangle ${i} collapses or changes area in f32`);
  const altitude=area2/edge;minFeature=Math.min(minFeature,altitude,len(ab),len(ac),len(bc));
  const n=checkVec(t.normal,'triangle normal').map(x=>scalar(x,'normal'));if(Math.abs(dot(cross(ab,ac),n)/area2-1)>1e-5)fail(`triangle ${i} has unresolved orientation`);
  if(!(altitude>64*F32_EPS*edge))fail(`triangle ${i} has unresolved aspect ratio`);
  return {a,b,c,normal:n,centroid:local(t.centroid??t.a),epsilon:epsilon(t.a,t.b,t.c),area:scalar(t.area??originalArea2/2,'triangle area',{positive:true}),radius:scalar(t.radius??edge,'triangle radius'),material:checkInt(t.materialIndex??0,0,L.materials-1,'material index'),id:i};
 });
 if(inputError*64>=minFeature)fail('scene scale separation is unresolved in f32');
 // Narrow interval guard eligibility is independent of the ordinary packer.
 // Source/rebase values must be exact f32 in a modest normal range, and all
 // triangles must have exact axis normals and well-conditioned 2-D area.
 // The 2^-16 source/rebase grid bounds each scaled coordinate by 2^24,
 // so original-source minus rebase subtraction is exact in binary64.
 // Auxiliary centroid rounding does not alter the immutable triangle query.
 const onAxisGrid=x=>Math.fround(x)===x&&Math.abs(x)<=256&&Number.isInteger(x*65536);
 let axisEligible=nt<=64&&vertexInputError===0&&origin.every(onAxisGrid)&&minFeature>=2**-12&&maxCoordinate<=256,axisKappa=1;
 for(const t of scene.triangles){
  const vertices=[t.a,t.b,t.c];if(vertices.some(p=>p.some(x=>!onAxisGrid(x))))axisEligible=false;
  const axis=[0,1,2].find(k=>t.a[k]===t.b[k]&&t.a[k]===t.c[k]&&Math.abs(t.normal[k])===1&&t.normal.every((x,j)=>j===k||x===0));
  if(axis===undefined){axisEligible=false;continue;}
  const i=(axis+1)%3,j=(axis+2)%3,e1=vsub(t.b,t.a),e2=vsub(t.c,t.a),a=e1[i]*e2[j],b=e1[j]*e2[i],kappa=(Math.abs(a)+Math.abs(b))/Math.abs(a-b);
  if(!Number.isFinite(kappa)||kappa>16)axisEligible=false;else axisKappa=Math.max(axisKappa,kappa);
 }
 // One extra f32 step also encloses binary64 rounding of the quotient.
 const axisKappaUpper=adjacent(outward(axisKappa,true),true);axisEligible&&=axisKappaUpper<=16;
 const axisGuard=Object.freeze({eligible:axisEligible,globalKappa:axisKappaUpper,domain:'at most64 exact-f32 axis triangles on 2^-16 source/rebase grid; abs world/local <=256; min feature >=2^-12; area cancellation <=16',assumption:'normal binary64 basic arithmetic; Math.hypot relative error <=16*2^-53; full shadow still required'});
 const wedges=scene.wedges.map((w,i)=>{for(const p of [w.a,w.b])if(!sourceVertices.has(checkVec(p,'wedge endpoint').join(',')))fail(`wedge ${i} endpoint is not an original triangle vertex`);const a=local(w.a,true),b=local(w.b,true),l=len(vsub(b,a));if(!(l>0)||Math.abs(l/w.l-1)>1e-5)fail(`wedge ${i} collapses in f32`);
  const frame={};for(const key of ['e','n1','n2','t1','t2'])frame[key]=checkVec(w[key],`wedge ${key}`).map(x=>scalar(x,`wedge ${key}`));
  return {a,b,v:a.map((x,k)=>(x+b[k])/2),l:scalar(w.l,'wedge length',{positive:true}),alpha:scalar(w.alpha,'wedge alpha'),epsilon:epsilon(w.a),edgeId:checkInt(w.edgeId,0,0xffffffff,'wedge edge ID'),...frame};});
 // Validate the exact uploaded tree, including every leaf ID and the maximum
 // pending DFS stack. No assumption about the preparer's usual balanced tree.
 const seen=new Set(),covered=new Set(),leaves=[],nodes=Array(nn),pending=[{id:scene.bvh.root,depth:1}];let maxStack=1,maxDepth=0;
 while(pending.length){maxStack=Math.max(maxStack,pending.length);if(maxStack>L.stack)fail('BVH stack exceeds shader capacity');const {id,depth}=pending.pop();checkInt(id,0,nn-1,'BVH child');if(seen.has(id))fail('BVH contains cycle or shared child');seen.add(id);maxDepth=Math.max(maxDepth,depth);if(maxDepth>L.stack)fail('BVH depth exceeds shader stack capacity');
  const n=scene.bvh.nodes[id],lo=checkVec(n.min,'BVH min').map((x,k)=>outward(x-origin[k],false)),hi=checkVec(n.max,'BVH max').map((x,k)=>outward(x-origin[k],true));
  if(lo.some((x,k)=>x>hi[k]))fail('inverted BVH bounds');
  const center=lo.map((x,k)=>Math.fround(x+(hi[k]-x)/2)),radius=outward(Math.hypot(...lo.map((x,k)=>Math.max(Math.abs(x-center[k]),Math.abs(hi[k]-center[k])))),true);
  const record={lo,hi,center,radius,left:0xffffffff,right:0xffffffff,start:leaves.length,count:0};
  if(n.triangleIds!==null&&n.triangleIds!==undefined){if(!Array.isArray(n.triangleIds)||!n.triangleIds.length)fail('empty or malformed BVH leaf');record.count=n.triangleIds.length;
   for(const tid of n.triangleIds){checkInt(tid,0,nt-1,'BVH triangle');if(covered.has(tid))fail('duplicate BVH triangle');covered.add(tid);leaves.push(tid);for(const p of [tris[tid].a,tris[tid].b,tris[tid].c])if(p.some((x,k)=>x<lo[k]||x>hi[k]))fail('BVH leaf does not enclose uploaded triangle');}
  }else{record.left=checkInt(n.left,0,nn-1,'BVH left');record.right=checkInt(n.right,0,nn-1,'BVH right');pending.push({id:n.right,depth:depth+1},{id:n.left,depth:depth+1});}
  nodes[id]=record;
 }
 if(seen.size!==nn||covered.size!==nt)fail('BVH has unreachable nodes or missing triangles');
 for(const n of nodes)if(!n.count)for(const id of [n.left,n.right])if(nodes[id].lo.some((x,k)=>x<n.lo[k])||nodes[id].hi.some((x,k)=>x>n.hi[k]))fail('BVH parent does not enclose child');
 const defaultMaterials=Array.from({length:Math.max(...tris.map(t=>t.material))+1},()=>({type:'diffuse',albedo:0,emission:0}));
 const materials=(scene.materials?.length?scene.materials:defaultMaterials).map((m,i)=>{if(!['diffuse','mirror'].includes(m.type??'diffuse'))fail(`material ${i} type unsupported`);const albedo=scalar(m.albedo??0,'albedo'),emission=scalar(m.emission??0,'emission');if(albedo<0||albedo>1||emission<0)fail('invalid material power');return {albedo,emission,type:m.type==='mirror'?1:0,twoSidedEmission:m.twoSidedEmission?1:0};});
 checkInt(materials.length,1,L.materials,'material count');if(tris.some(t=>t.material>=materials.length))fail('missing material');
 const emitters=(scene.emitters??scene.triangles.filter(t=>materials[t.materialIndex??0].emission>0)).map(t=>checkInt(t.id,0,nt-1,'emitter triangle'));
 if(new Set(emitters).size!==emitters.length||emitters.some(id=>materials[tris[id].material].emission<=0))fail('invalid emitter table');
 const emitterArea=emitters.reduce((s,id)=>s+scene.triangles[id].area,0);if(scene.emitterArea!==undefined&&Math.abs(scene.emitterArea-emitterArea)>Number.EPSILON*Math.max(1,emitterArea)*nt)fail('emitter area mismatch');
 const offsets={triangle:S.header,wedge:S.header+nt*S.triangle};offsets.node=offsets.wedge+nw*S.wedge;offsets.leaf=offsets.node+nn*S.node;offsets.material=offsets.leaf+Math.ceil(leaves.length/4);offsets.emitter=offsets.material+materials.length*S.material;
 const slots=offsets.emitter+emitters.length*S.emitter,buffer=new ArrayBuffer(slots*16),data=new Uint32Array(buffer),floats=new Float32Array(buffer);
 const U=(at,values)=>data.set(values,at*4),F=(at,values)=>floats.set(values,at*4);
 U(0,[0x52475031,S.version,nt,nw]);U(1,[nn,materials.length,emitters.length,scene.bvh.root]);U(2,[offsets.triangle,offsets.wedge,offsets.node,offsets.leaf]);U(3,[offsets.material,offsets.emitter,slots,maxStack]);F(4,[...origin.map(Math.fround),scalar(emitterArea,'emitter area')]);F(5,[maxCoordinate,minFeature,inputError,epsilon(...scene.triangles.flatMap(t=>[t.a,t.b,t.c]))]);F(6,[...origin.map((x,k)=>x-Math.fround(origin[k])),0]);U(7,[scene.unsupportedEdges?.length??0,maxDepth,axisEligible?1:0,0]);floats[31]=axisGuard.globalKappa;
 for(const [i,t] of tris.entries()){const o=offsets.triangle+i*S.triangle;F(o,[...t.a,t.epsilon]);F(o+1,[...t.b,t.area]);F(o+2,[...t.c,t.radius]);F(o+3,t.normal);data[(o+3)*4+3]=t.material;F(o+4,t.centroid);data[(o+4)*4+3]=t.id;}
 for(const [i,w] of wedges.entries()){const o=offsets.wedge+i*S.wedge;F(o,[...w.a,w.epsilon]);F(o+1,[...w.b,w.l]);F(o+2,[...w.v,w.alpha]);F(o+3,w.e);data[(o+3)*4+3]=w.edgeId;for(const [j,k] of ['n1','n2','t1','t2'].entries())F(o+4+j,w[k]);}
 for(const [i,n] of nodes.entries()){const o=offsets.node+i*S.node;F(o,n.lo);F(o+1,n.hi);F(o+2,[...n.center,n.radius]);U(o+3,[n.left,n.right,n.start,n.count]);}
 data.set(leaves,offsets.leaf*4);
 for(const [i,m] of materials.entries()){const o=offsets.material+i;F(o,[m.albedo,m.emission]);data[o*4+2]=m.type;data[o*4+3]=m.twoSidedEmission;}
 let cumulative=0;for(const [i,id] of emitters.entries()){const o=offsets.emitter+i;data[o*4]=id;cumulative+=scene.triangles[id].area;floats[o*4+1]=tris[id].area;floats[o*4+2]=cumulative;}
 const toLocal=point=>checkVec(point,'query point').map((x,k)=>{const value=scalar(x-origin[k],'query point');if(axisPlanes[k].has(value)&&axisPlanes[k].get(value)!==x)fail('query point collapses onto an exact surface plane');return value;});
 return {data,buffer,origin,metadata:Object.freeze({version:S.version,bytes:buffer.byteLength,offsets,triangles:nt,wedges:nw,nodes:nn,materials:materials.length,emitters:emitters.length,emitterArea,maxStack,maxDepth,minFeature,inputError,vertexInputError,maxCoordinate,axisGuard,unsupportedEdges:scene.unsupportedEdges?.length??0,precision:'f32 local coordinates; ambiguous batches rejected'}),toLocal,toWorld:point=>checkVec(point,'local point').map((x,k)=>x+origin[k])};
}
