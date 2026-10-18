// Single-region deterministic scalar UTD mesh transport, independently written.
// Prepend utd-functions.wgsl. CPU reference: utd-scene.js evaluatePoint.
// Storage ABI: utd-gpu-pack.js. Positions have already been locally rebased.
// Conditional directional power transfer f, NOT irradiance or vector EM.
struct UTDParams {
  dims:vec4u,          // width, height, baseRow, batchRows
  counts:vec4u,        // triangles, physical edges, clipped edges, facets
  controls:vec4f,      // k, boundary (-1 soft, 0 unpolarized, +1 hard), minKr, minSinBeta
  source:vec4f,        // origin, incoming cone radius0
  incident:vec4f,      // propagation direction, cone tanAngle
  detector:vec4f,      // center, width
  detectorU:vec4f,     // u axis, height
  detectorV:vec4f,     // v axis, hasRegion
  region:vec4f,        // center, radius (axes radius,radius,2radius)
  limits:vec4f,        // endpoint guard, angular epsilon, scene supported, open plate
};
struct UTDTriangle { a:vec4f,b:vec4f,c:vec4f };
struct UTDEdge {
  a:vec4f, b:vec4f, // a.w oriented-frame flag; b.w physical-edge index
  nff:vec4f, tff:vec4f, // .w original start/end projection from a along a->b
  nbf:vec4f, geometry:vec4f, // alpha, clipped length, clipped start/end flags
};
struct UTDFacet { normal:vec4f, point:vec4f };
@group(0) @binding(0) var<uniform> up:UTDParams;
@group(0) @binding(1) var<storage,read> utri:array<UTDTriangle>;
@group(0) @binding(2) var<storage,read> uedge:array<UTDEdge>;
@group(0) @binding(3) var<storage,read> uadj:array<vec4u>;
@group(0) @binding(4) var<storage,read_write> uout:array<vec4f>;
@group(0) @binding(5) var<storage,read> ufacet:array<UTDFacet>;

struct UTDStationary {
  point:vec3f, ro:f32, wo:vec3f, sinBeta:f32,
  axial:f32, margin:f32, exists:u32, inside:u32,
};
struct UTDGO { weight:f32, invalid:u32 };
struct UTDReflection { soft:vec2f, hard:vec2f, count:u32, invalid:u32 };

fn utd_ray_epsilon(p:vec3f)->f32 {
  // Geometry tolerance is distinct from the angular cot/GO seam epsilon.
  return max(0.000001,0.00000025*max(1.0,max(abs(p.x),max(abs(p.y),abs(p.z)))));
}
fn utd_in_cone(p:vec3f)->bool {
  let q=p-up.source.xyz;let depth=dot(q,up.incident.xyz);
  return depth>=0.0 && length(q-depth*up.incident.xyz)<=up.source.w+up.incident.w*depth+utd_ray_epsilon(p);
}
fn utd_region_value(p:vec3f)->f32 {
  let q=p-up.region.xyz;let depth=dot(q,up.incident.xyz);
  return (dot(q,q)-0.75*depth*depth)/(up.region.w*up.region.w);
}
fn utd_mask_has(mask:vec4u,index:u32)->bool {
  return (mask[index>>5u] & (1u<<(index&31u)))!=0u;
}
fn utd_ray_hit(origin:vec3f,direction:vec3f,index:u32,tmin:f32,tmax:f32)->f32 {
  let tri=utri[index];let e1=tri.b.xyz-tri.a.xyz;let e2=tri.c.xyz-tri.a.xyz;
  let p=cross(direction,e2);let det=dot(e1,p);
  if(abs(det)<=0.0000001*length(e1)*length(e2)){return -1.0;}
  let inv=1.0/det;let s=origin-tri.a.xyz;let u=dot(s,p)*inv;
  if(u< -0.000001 || u>1.000001){return -1.0;}
  let q=cross(s,e1);let v=dot(direction,q)*inv;
  if(v< -0.000001 || u+v>1.000001){return -1.0;}
  let t=dot(e2,q)*inv;
  if(t<tmin||t>tmax){return -1.0;}
  return t;
}
fn utd_occluded(origin:vec3f,direction:vec3f,tmin:f32,tmax:f32,ignore:vec4u)->bool {
  for(var i=0u;i<up.counts.x;i=i+1u){
    if(!utd_mask_has(ignore,i)&&utd_ray_hit(origin,direction,i,tmin,tmax)>=0.0){return true;}
  }
  return false;
}
fn utd_edge_visible(index:u32,point:vec3f,dst:vec3f)->bool {
  let tol=utd_ray_epsilon(point);let outgoing=dst-point;let r=length(outgoing);
  if(r<=tol){return false;}
  return !utd_occluded(point,-up.incident.xyz,tol,1e30,uadj[index]) &&
    !utd_occluded(point,outgoing/r,tol,max(tol,r-utd_ray_epsilon(dst)),uadj[index]);
}
fn utd_stationary(edge:UTDEdge,dst:vec3f)->UTDStationary {
  var h:UTDStationary;
  let middle=(edge.a.xyz+edge.b.xyz)*0.5;let e=normalize(edge.b.xyz-edge.a.xyz);
  let c=dot(up.incident.xyz,e);let sb=sqrt(max(0.0,1.0-c*c));
  if(sb<0.00000001){return h;}
  let d=dst-middle;let da=dot(d,e);let rho=length(d-da*e);
  let axial=da-rho*c/sb;
  h.point=middle+axial*e;let out=dst-h.point;h.ro=length(out);h.sinBeta=sb;
  h.axial=axial;h.margin=min(axial+edge.geometry.y*0.5,edge.geometry.y*0.5-axial);
  if(h.ro<=utd_ray_epsilon(dst)){return h;}
  h.wo=out/h.ro;h.exists=1u;
  h.inside=select(0u,1u,h.margin>=-utd_ray_epsilon(h.point));
  return h;
}
fn utd_angles(edge:UTDEdge,h:UTDStationary)->vec2f {
  let wi=-up.incident.xyz;
  let pi=utd_atan2_axes(dot(wi,edge.nff.xyz),dot(wi,edge.tff.xyz));
  var po=utd_atan2_axes(dot(h.wo,edge.nff.xyz),dot(h.wo,edge.tff.xyz));
  if(po<0.0){po=po+2.0*UTD_PI;}
  return vec2f(pi,po);
}
fn utd_segment_distance(point:vec3f,a:vec3f,b:vec3f)->f32 {
  let d=b-a;let ll=dot(d,d);return length(point-(a+clamp(dot(point-a,d)/ll,0.0,1.0)*d));
}
fn utd_boundary_at(point:vec3f,tolerance:f32,mask:vec4u,restrictFacet:bool)->vec2u {
  var count=0u;var first=0u;
  for(var i=0u;i<up.counts.y;i=i+1u){
    if(restrictFacet&&!any((mask&uadj[i])!=vec4u(0u))){continue;}
    let edge=uedge[i];
    if(utd_segment_distance(point,edge.a.xyz,edge.b.xyz)<=tolerance){
      if(count==0u){first=i;}count=count+1u;
    }
  }
  return vec2u(count,first);
}
fn utd_region_has_saddle(physicalIndex:u32,dst:vec3f)->bool {
  for(var j=0u;j<up.counts.z;j=j+1u){
    let edge=uedge[up.counts.y+j];
    if(u32(edge.b.w)==physicalIndex){let h=utd_stationary(edge,dst);if(h.exists==1u&&h.inside==1u){return true;}}
  }
  return false;
}
fn utd_direct(dst:vec3f)->UTDGO {
  var go:UTDGO;if(!utd_in_cone(dst)){return go;}
  let wi=-up.incident.xyz;let tol=utd_ray_epsilon(dst);var nearest=1e30;var hit=false;
  for(var i=0u;i<up.counts.x;i=i+1u){let t=utd_ray_hit(dst,wi,i,tol,nearest);if(t>=0.0){hit=true;nearest=t;}}
  var boundaries=0u;var regionSaddle=false;
  for(var i=0u;i<up.counts.y;i=i+1u){
    let edge=uedge[i];if(edge.a.w==0.0){continue;}
    let h=utd_stationary(edge,dst);if(h.exists==0u||h.inside==0u){continue;}
    let a=utd_angles(edge,h);
    if(abs(UTD_PI-abs(a.y-a.x))<=up.limits.y){
      boundaries=boundaries+1u;regionSaddle=regionSaddle||utd_region_has_saddle(i,dst);
    }
  }
  if(hit){
    let border=utd_boundary_at(dst+nearest*wi,max(tol,up.limits.y*nearest),vec4u(0u),false);
    if(border.x>1u){go.invalid=1u;}
    if(border.x>0u&&boundaries==0u){boundaries=1u;regionSaddle=utd_region_has_saddle(border.y,dst);}
  }
  if(boundaries>1u){go.invalid=1u;}
  if(boundaries>0u){
    for(var i=0u;i<up.counts.x;i=i+1u){
      let t=utd_ray_hit(dst,wi,i,tol,1e30);
      if(t>=0.0){
        let blockerBorder=utd_boundary_at(dst+t*wi,max(tol,up.limits.y*t),vec4u(0u),false);
        if(blockerBorder.x==0u){return go;}
      }
    }
    if(!regionSaddle){go.invalid=1u;}go.weight=0.5;return go;
  }
  go.weight=select(1.0,0.0,hit);return go;
}
fn utd_stable_excess(point:vec3f,dst:vec3f)->f32 {
  let v=dst-point;let parallel=dot(v,up.incident.xyz);let r=length(v);
  if(parallel>0.0){let perp=v-parallel*up.incident.xyz;return dot(perp,perp)/(r+parallel);}
  return r-parallel;
}
fn utd_reflections(dst:vec3f)->UTDReflection {
  var result:UTDReflection;if(up.detectorV.w==0.0){return result;}
  let di=up.incident.xyz;let tol=utd_ray_epsilon(dst);
  for(var i=0u;i<up.counts.w;i=i+1u){
    let facet=ufacet[i];var normal=facet.normal.xyz;var inc=dot(di,normal);
    if(up.limits.w==1.0&&inc>0.0){normal=-normal;inc=-inc;}
    if(inc>=-up.controls.w){continue;}
    let dr=di-2.0*inc*normal;let den=dot(dr,normal);let r=dot(dst-facet.point.xyz,normal)/den;
    if(r<=tol){continue;}
    let point=dst-dr*r;let mask=uadj[up.counts.y+up.counts.z+i];var onFacet=false;
    for(var j=0u;j<up.counts.x;j=j+1u){
      if(utd_mask_has(mask,j)&&utd_ray_hit(dst,-dr,j,tol,r+tol)>=0.0){onFacet=true;break;}
    }
    // A symmetric physical-edge half-weight band includes both sides. Do not
    // let f32 barycentric rounding discard its slightly outside-facet half.
    let border=utd_boundary_at(point,max(tol,up.limits.y*r),mask,true);
    if((!onFacet&&border.x==0u)||!utd_in_cone(point)||utd_region_value(point)>1.0+tol){continue;}
    if(utd_occluded(point,-di,tol,1e30,mask)||utd_occluded(point,dr,tol,max(tol,r-tol),mask)){continue;}
    if(border.x>1u){result.invalid=1u;}
    let weight=select(1.0,0.5,border.x>0u);
    let phase=utd_cis(-up.controls.x*utd_stable_excess(point,dst));
    result.soft=result.soft-weight*phase;result.hard=result.hard+weight*phase;result.count=result.count+1u;
  }
  return result;
}
fn utd_power(soft:vec2f,hard:vec2f)->f32 {
  if(up.controls.y<0.0){return dot(soft,soft);}
  if(up.controls.y>0.0){return dot(hard,hard);}
  return (dot(soft,soft)+dot(hard,hard))*0.5;
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3u) {
  let x=id.x;let y=up.dims.z+id.y;
  if(x>=up.dims.x||y>=up.dims.y||id.y>=up.dims.w){return;}
  if(up.limits.z==0.0){uout[y*up.dims.x+x]=vec4f(0.0);return;}
  let dst=up.detector.xyz+((f32(x)+0.5)/f32(up.dims.x)-0.5)*up.detector.w*up.detectorU.xyz+
    (0.5-(f32(y)+0.5)/f32(up.dims.y))*up.detectorU.w*up.detectorV.xyz;
  let direct=utd_direct(dst);let reflections=utd_reflections(dst);
  var soft=vec2f(direct.weight,0.0)+reflections.soft;
  var hard=vec2f(direct.weight,0.0)+reflections.hard;
  let geometric=utd_power(soft,hard);
  var invalid=up.limits.z==0.0||direct.invalid!=0u||reflections.invalid!=0u;var edgeCount=0u;
  for(var j=0u;j<up.counts.z;j=j+1u){
    let index=up.counts.y+j;let edge=uedge[index];let h=utd_stationary(edge,dst);
    if(h.exists==0u){invalid=true;continue;}
    let width=sqrt(h.ro/(up.controls.x*h.sinBeta*h.sinBeta));
    if(h.inside==0u){
      let endpoint=select(edge.b.xyz,edge.a.xyz,distance(h.point,edge.a.xyz)<distance(h.point,edge.b.xyz));
      if(h.margin>=-up.limits.x*width&&utd_edge_visible(index,endpoint,dst)){invalid=true;}
      continue;
    }
    let angles=utd_angles(edge,h);let n=2.0-edge.geometry.x/UTD_PI;
    if(angles.y>n*UTD_PI+up.limits.y||!utd_edge_visible(index,h.point,dst)){continue;}
    if(h.sinBeta<up.controls.w||up.controls.x*h.ro*h.sinBeta*h.sinBeta<up.controls.z){invalid=true;continue;}
    if(min(angles.y,n*UTD_PI-angles.y)<up.controls.w){invalid=true;continue;}
    let along=h.axial+edge.geometry.y*0.5;
    let originalMargin=min(abs(along-edge.nff.w),abs(along-edge.tff.w));
    if(originalMargin<up.limits.x*width){invalid=true;}
    if((edge.geometry.z>0.0||edge.geometry.w>0.0)&&h.margin<up.limits.x*width){invalid=true;}
    let L=h.ro*h.sinBeta*h.sinBeta;
    let phase=utd_cis(-up.controls.x*utd_stable_excess(h.point,dst))/sqrt(h.ro);
    let ds=utd_coefficient(up.controls.x,n,angles.x,angles.y,L,h.sinBeta,-1.0);
    let dh=utd_coefficient(up.controls.x,n,angles.x,angles.y,L,h.sinBeta,1.0);
    soft=soft+utd_cmul(ds,phase);hard=hard+utd_cmul(dh,phase);edgeCount=edgeCount+1u;
  }
  var power=utd_power(soft,hard);
  // Comparison also rejects NaN without relying on optional shader builtins.
  if(!(power>=0.0&&power<1e30)){power=0.0;invalid=true;}
  if(direct.weight==0.0&&reflections.count==0u&&edgeCount==0u){invalid=true;}
  if(direct.weight==0.5&&edgeCount==0u){invalid=true;}
  if(up.detectorV.w==1.0&&distance(dst,up.region.xyz)<=2.0*up.region.w){invalid=true;}
  uout[y*up.dims.x+x]=vec4f(power,geometric,select(1.0,0.0,invalid),f32(edgeCount));
}
