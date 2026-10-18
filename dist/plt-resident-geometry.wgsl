// Resident geometry ABI v1; offsets/strides are vec4 (16-byte) records.
// rg_flags != 0 invalidates the whole candidate batch. Never clear it inside
// a query. Error envelopes classify ambiguity; they do not enlarge a physical
// triangle, cone, slab, or finite wedge. BVH padding is candidate-only.
@group(0) @binding(1) var<storage, read> rg_data: array<vec4u>;
var<private> rg_flags: u32 = 0u;
// Support identity is carried from actual section arithmetic, never inferred
// from a point being numerically close to a vertex or edge.
var<private> rg_section_mask: u32;
var<private> rg_contact_feature: vec4u;
var<private> rg_transport_feature: vec4u;
var<private> rg_region_feature: vec4u;
var<private> rg_visibility_active: bool = false;
var<private> rg_visibility_source: vec3f;
var<private> rg_visibility_destination: vec3f;
const RG_INVALID: u32 = 1u;
const RG_AMBIGUOUS: u32 = 2u;
const RG_FMAX: f32 = 3.4028234663852886e38;
const RG_EPS: f32 = 1.1920928955078125e-7;
struct RGBeam { origin: vec3f, direction: vec3f, radius0: f32, tanAngle: f32, near: f32, far: f32 }
struct RGRegion { center: vec3f, direction: vec3f, radius: f32, depth: f32 }
struct RGHit { valid: u32, triangle: u32, distance: f32, point: vec3f, normal: vec3f, material: u32 }
struct RGTriangle { a: vec3f, b: vec3f, c: vec3f, normal: vec3f, area: f32, material: u32, id: u32 }
struct RGMaterial { albedo: f32, emission: f32, kind: u32, twoSidedEmission: u32 }
struct RGEdge { valid: u32, index: u32, edgeId: u32, a: vec3f, b: vec3f, v: vec3f, e: vec3f, n1: vec3f, n2: vec3f, t1: vec3f, t2: vec3f, l: f32, alpha: f32, originalA: vec3f, originalB: vec3f, clippedStart: u32, clippedEnd: u32 }
struct RGNode { lo: vec3f, hi: vec3f, center: vec3f, radius: f32, left: u32, right: u32, start: u32, count: u32 }
struct RGSection { valid: u32, radial: f32, point: vec3f }
fn rg_f(i: u32) -> vec4f { return bitcast<vec4f>(rg_data[i]); }
fn rg_triangle_count() -> u32 { return rg_data[0].z; }
fn rg_wedge_count() -> u32 { return rg_data[0].w; }
fn rg_material_count() -> u32 { return rg_data[1].y; }
fn rg_emitter_count() -> u32 { return rg_data[1].z; }
fn rg_emitter_area() -> f32 { return rg_f(4u).w; }
fn rg_emitter_index(i: u32) -> u32 { if(i>=rg_emitter_count()){rg_flags|=RG_INVALID;return 0u;} return rg_data[rg_data[3].y+i].x; }
fn rg_unsupported_count() -> u32 { return rg_data[7].x; }
fn rg_triangle(i: u32) -> RGTriangle { var t: RGTriangle; if(i>=rg_triangle_count()){rg_flags|=RG_INVALID;return t;} let o=rg_data[2].x+i*5u; t.a=rg_f(o).xyz;t.b=rg_f(o+1u).xyz;t.c=rg_f(o+2u).xyz;t.normal=rg_f(o+3u).xyz;t.area=rg_f(o+1u).w;t.material=rg_data[o+3u].w;t.id=rg_data[o+4u].w;return t; }
fn rg_material(i: u32) -> RGMaterial { var m:RGMaterial;if(i>=rg_material_count()){rg_flags|=RG_INVALID;return m;}let o=rg_data[3].x+i;return RGMaterial(rg_f(o).x,rg_f(o).y,rg_data[o].z,rg_data[o].w); }
fn rg_node(i:u32)->RGNode {var n:RGNode;if(i>=rg_data[1].x){rg_flags|=RG_INVALID;return n;}let o=rg_data[2].z+i*4u;let ids=rg_data[o+3u];return RGNode(rg_f(o).xyz,rg_f(o+1u).xyz,rg_f(o+2u).xyz,rg_f(o+2u).w,ids.x,ids.y,ids.z,ids.w);}
fn rg_leaf(i:u32)->u32 {return rg_data[rg_data[2].w+i/4u][i%4u];}
fn rg_absmax(p:vec3f)->f32{return max(abs(p.x),max(abs(p.y),abs(p.z)));}
fn rg_world(p:vec3f)->vec3f{return p+rg_f(4u).xyz+rg_f(6u).xyz;}
fn rg_epsilon(p:vec3f)->f32 {let world=rg_world(p);return max(1e-10,2.842170943040401e-14*max(1.0,rg_absmax(world)));}
fn rg_endpoint_epsilon(a:vec3f,b:vec3f)->f32 {return max(rg_epsilon(a),rg_epsilon(b));}
fn rg_round(scaleValue:f32)->f32{return 32.0*RG_EPS*max(scaleValue,1e-30)+4.0*rg_f(5u).z;}
fn rg_finite(v:f32)->bool{return abs(v)<=RG_FMAX;}
fn rg_finite3(v:vec3f)->bool{return all(abs(v)<=vec3f(RG_FMAX));}
fn rg_valid_beam(b:RGBeam)->bool {let norm=dot(b.direction,b.direction);if(!rg_finite3(b.origin)||!rg_finite3(b.direction)||!rg_finite(b.near)||!rg_finite(b.far)||!rg_finite(b.radius0)||!rg_finite(b.tanAngle)||!(norm>0.0)||abs(norm-1.0)>2e-6||b.near<0.0||b.far<b.near||b.radius0<0.0||b.tanAngle<0.0){rg_flags|=RG_INVALID;return false;}return true;}
fn rg_box(n:RGNode,b:RGBeam,farLimit:f32)->f32 {
 var lo=b.near;var hi=min(b.far,farLimit);let pad=max(rg_round(max(rg_absmax(b.origin),max(rg_absmax(n.lo),rg_absmax(n.hi)))),max(rg_epsilon(b.origin),max(rg_epsilon(n.lo),rg_epsilon(n.hi))));
 for(var k=0u;k<3u;k++){let a=n.lo[k]-pad;let c=n.hi[k]+pad;let d=b.direction[k];let o=b.origin[k];if(d==0.0){if(o<a||o>c){return -1.0;}}else{let t0=(a-o)/d;let t1=(c-o)/d;lo=max(lo,min(t0,t1));hi=min(hi,max(t0,t1));if(lo>hi){return -1.0;}}}return lo;
}
fn rg_on_axis_plane(t:RGTriangle,p:vec3f)->bool {for(var k=0u;k<3u;k++){if(t.a[k]==t.b[k]&&t.a[k]==t.c[k]&&p[k]==t.a[k]){return true;}}return false;}
fn rg_axis_endpoint(t:RGTriangle)->bool {
 // An exact endpoint on an axis-aligned plane has only that one plane
 // crossing along a non-coplanar segment. This preserves tiny physical
 // endpoint exclusions even when distance - epsilon rounds to distance.
 // Generic near-plane endpoints are deliberately not treated as exact.
 if(!rg_visibility_active){return false;}
 for(var k=0u;k<3u;k++){if(t.a[k]==t.b[k]&&t.a[k]==t.c[k]){if(rg_visibility_source[k]==t.a[k]||rg_visibility_destination[k]==t.a[k]){return true;}}}return false;
}
fn rg_triangle_ray(t:RGTriangle,b:RGBeam,farLimit:f32)->RGHit {
 var h:RGHit;if(rg_axis_endpoint(t)){return h;}let e1=t.b-t.a;let e2=t.c-t.a;let p=cross(b.direction,e2);let det=dot(e1,p);let scaleDet=length(e1)*length(e2);let threshold=max(1e-38,scaleDet*1e-13);let errDet=32.0*RG_EPS*scaleDet;
 if(abs(det)<=threshold){let plane=dot(b.origin-t.a,t.normal);if(abs(plane)<=rg_round(max(rg_absmax(b.origin),rg_absmax(t.a)))){rg_flags|=RG_AMBIGUOUS|256u;}return h;}
 if(abs(det)<=errDet){rg_flags|=RG_AMBIGUOUS|512u;return h;}
 let s=b.origin-t.a;let u=dot(s,p)/det;let q=cross(s,e1);let v=dot(b.direction,q)/det;
 let baryErr=64.0*RG_EPS*max(1.0,length(s)*max(length(e1),length(e2))/abs(det));
 if(u< -1e-12-baryErr||u>1.0+1e-12+baryErr||v< -1e-12-baryErr||u+v>1.0+1e-12+baryErr){return h;}
 let distanceValue=dot(e2,q)/det;let farValue=min(farLimit,b.far);let distanceErr=rg_round(max(abs(distanceValue),max(rg_absmax(b.origin),rg_absmax(t.a))))*max(1.0,scaleDet/abs(det));
 if(distanceValue<b.near-distanceErr||distanceValue>farValue+distanceErr){return h;}
 if(distanceValue==0.0&&b.near>0.0){if(!rg_on_axis_plane(t,b.origin)){rg_flags|=RG_AMBIGUOUS|1024u;}return h;}
 if(abs(u+1e-12)<=baryErr||abs(v+1e-12)<=baryErr||abs(1.0+1e-12-u-v)<=baryErr){rg_flags|=RG_AMBIGUOUS|2048u;return h;}
 // Exact zero is an endpoint exclusion, not a whole-face exclusion. An
 // uncertain nonzero distance near either endpoint rejects the candidate.
 if(distanceValue!=0.0&&(abs(distanceValue-b.near)<=distanceErr||abs(distanceValue-farValue)<=distanceErr)){rg_flags|=RG_AMBIGUOUS|4096u;return h;}
 if(u< -1e-12||u>1.0+1e-12||v< -1e-12||u+v>1.0+1e-12||distanceValue<b.near||distanceValue>farValue){return h;}
 h.valid=1u;h.triangle=t.id;h.distance=distanceValue;h.point=b.origin+b.direction*distanceValue;
 // Preserve exact incidence on a known axis plane after rounded ray arithmetic.
 for(var k=0u;k<3u;k++){if(t.a[k]==t.b[k]&&t.a[k]==t.c[k]){h.point[k]=t.a[k];}}
 h.normal=t.normal;h.material=t.material;return h;
}
fn rg_ray(b:RGBeam)->RGHit {
 var best:RGHit;if(!rg_valid_beam(b)){return best;}var stack:array<u32,32>;stack[0]=rg_data[1].w;var count=1u;var visited=0u;
 loop {if(count==0u){break;}visited++;if(visited>rg_data[1].x){rg_flags|=RG_INVALID;return RGHit();}count--;let n=rg_node(stack[count]);var farLimit=b.far;if(best.valid!=0u){farLimit=best.distance;}
  if(rg_box(n,b,farLimit)<0.0){continue;}
  if(n.count>0u){for(var i=0u;i<n.count;i++){let hit=rg_triangle_ray(rg_triangle(rg_leaf(n.start+i)),b,farLimit);if(hit.valid!=0u){if(best.valid==0u||hit.distance<best.distance||(hit.distance==best.distance&&hit.triangle<best.triangle)){best=hit;farLimit=best.distance;}}}}
  else {let left=rg_box(rg_node(n.left),b,farLimit);let right=rg_box(rg_node(n.right),b,farLimit);let need=select(0u,1u,left>=0.0)+select(0u,1u,right>=0.0);if(count+need>32u){rg_flags|=RG_INVALID;return RGHit();}
   if(left>=0.0&&right>=0.0){if(left<right){stack[count]=n.right;stack[count+1u]=n.left;}else{stack[count]=n.left;stack[count+1u]=n.right;}count+=2u;}else if(left>=0.0){stack[count]=n.left;count++;}else if(right>=0.0){stack[count]=n.right;count++;}}
 }return best;
}
fn rg_visible(src:vec3f,dst:vec3f,endpointEpsilon:f32)->bool {
 if(rg_flags!=0u){return false;}
 if(!rg_finite(endpointEpsilon)||!rg_finite3(src)||!rg_finite3(dst)||endpointEpsilon<0.0){rg_flags|=RG_INVALID;return false;}let d=length(dst-src);if(d==0.0){return true;}let e=min(endpointEpsilon,d/4.0);let previous=rg_flags;rg_visibility_active=e>0.0;rg_visibility_source=src;rg_visibility_destination=dst;let h=rg_ray(RGBeam(src,(dst-src)/d,0.0,0.0,e,d-e));rg_visibility_active=false;return h.valid==0u&&rg_flags==previous;
}
fn rg_cone_contains(b:RGBeam,p:vec3f,tolerance:f32)->bool {let q=p-b.origin;let t=dot(q,b.direction);return t>=b.near-tolerance&&t<=b.far+tolerance&&length(q-b.direction*t)<=b.radius0+b.tanAngle*t+tolerance;}
fn rg_sphere_cone(n:RGNode,b:RGBeam)->bool {
 let delta=n.center-b.origin;let axial=dot(delta,b.direction);let pad=max(rg_round(max(rg_absmax(n.center),rg_absmax(b.origin))+n.radius+abs(axial)),max(rg_epsilon(n.center),rg_epsilon(b.origin)));
 if(axial+n.radius<b.near-pad||axial-n.radius>b.far+pad){return false;}let upper=max(0.0,min(b.far,axial+n.radius+pad));let radial=length(delta-b.direction*axial);return radial<=n.radius+b.radius0+b.tanAngle*upper+pad*(1.0+b.tanAngle);
}
// Same endpoint order and up-to-six-point section as utd-scene.js sectionAt.
fn rg_section(t:RGTriangle,b:RGBeam,depths:vec3f,depth:f32,tol:f32,flat:bool)->RGSection {
 rg_section_mask=0u;var out:array<vec3f,6>;var masks:array<u32,6>;var points:array<vec3f,3>;points[0]=t.a;points[1]=t.b;points[2]=t.c;var count=0u;
 for(var i=0u;i<3u;i++){let j=(i+1u)%3u;if(abs(depths[i]-depth)<=tol){out[count]=points[i];masks[count]=1u<<i;count++;}if((depths[i]<depth&&depths[j]>depth)||(depths[i]>depth&&depths[j]<depth)){let u=(depth-depths[i])/(depths[j]-depths[i]);out[count]=points[i]+(points[j]-points[i])*u;masks[count]=(1u<<i)|(1u<<j);count++;}}
 var result:RGSection;if(count==0u){return result;}let axis=b.origin+b.direction*depth;
 if(count>=3u&&flat){var inside=true;for(var i=0u;i<3u;i++){if(!(dot(cross(points[(i+1u)%3u]-points[i],axis-points[i]),t.normal)>=-tol)){inside=false;break;}}if(inside){var strictInside=true;for(var i=0u;i<3u;i++){let edge=points[(i+1u)%3u]-points[i];let offset=axis-points[i];if(dot(cross(edge,offset),t.normal)<=rg_round(length(edge)*length(offset))){strictInside=false;}}rg_section_mask=select(0u,7u,strictInside);return RGSection(1u,0.0,axis);}}
 var closest=out[0];var support=masks[0];var radial=length(closest-axis);
 for(var i=0u;i<count;i++){for(var j=i+1u;j<count;j++){let delta=out[j]-out[i];let ll=dot(delta,delta);var u=0.0;if(ll>0.0){u=clamp(dot(axis-out[i],delta)/ll,0.0,1.0);}let q=out[i]+delta*u;let r=length(q-axis);if(r<radial){radial=r;closest=q;if(u==0.0){support=masks[i];}else if(u==1.0){support=masks[j];}else{let sharedMask=masks[i]|masks[j];support=0u;if(sharedMask==3u||sharedMask==5u||sharedMask==6u){support=sharedMask;}else if(u>64.0*RG_EPS&&u<1.0-64.0*RG_EPS){support=7u;}}}}}
 rg_section_mask=support;return RGSection(1u,radial,closest);
}
fn rg_cone_value(t:RGTriangle,b:RGBeam,z:vec3f,depth:f32,tol:f32,flat:bool)->f32 {let s=rg_section(t,b,z,depth,tol,flat);if(s.valid==0u){return RG_FMAX;}return s.radial-b.radius0-b.tanAngle*depth;}
fn rg_contact(t:RGTriangle,b:RGBeam)->RGHit {
 rg_contact_feature=vec4u(0u);var result:RGHit;let z=vec3f(dot(t.a-b.origin,b.direction),dot(t.b-b.origin,b.direction),dot(t.c-b.origin,b.direction));let minZ=min(z.x,min(z.y,z.z));let maxZ=max(z.x,max(z.y,z.z));let lo=max(b.near,minZ);let hi=min(b.far,maxZ);
 let scaleValue=max(rg_absmax(b.origin),max(rg_absmax(t.a),max(rg_absmax(t.b),rg_absmax(t.c))));let errorValue=rg_round(scaleValue+max(abs(minZ),abs(maxZ)));let tol=max(rg_f(rg_data[2].x+t.id*5u).w,rg_epsilon(b.origin));
 if(lo>hi){if(lo-hi<=errorValue){rg_flags|=RG_AMBIGUOUS|8192u;}return result;}
 let span=maxZ-minZ;let flat=span<=tol;if(span>0.0&&abs(span-tol)<=errorValue){rg_flags|=RG_AMBIGUOUS|16384u;return result;}
 let left=rg_cone_value(t,b,z,lo,tol,flat);var contact=lo;
 if(left<=tol){if(left>-errorValue&&b.radius0+b.tanAngle*lo>0.0){rg_flags|=RG_AMBIGUOUS|32768u;return result;}}
 else {if(hi-lo<=tol){if(left<=tol+errorValue){rg_flags|=RG_AMBIGUOUS|65536u;}return result;}
  var a=lo;var cHi=hi;let gr=0.6180339887498949;var c=cHi-gr*(cHi-a);var d=a+gr*(cHi-a);var fc=rg_cone_value(t,b,z,c,tol,flat);var fd=rg_cone_value(t,b,z,d,tol,flat);
  for(var i=0u;i<65u;i++){if(fc<fd){cHi=d;d=c;fd=fc;c=cHi-gr*(cHi-a);fc=rg_cone_value(t,b,z,c,tol,flat);}else{a=c;c=d;fc=fd;d=a+gr*(cHi-a);fd=rg_cone_value(t,b,z,d,tol,flat);}}
  let mid=(a+cHi)/2.0;var bestDepth=mid;var bestValue=rg_cone_value(t,b,z,mid,tol,flat);let highValue=rg_cone_value(t,b,z,hi,tol,flat);if(highValue<bestValue){bestValue=highValue;bestDepth=hi;}if(fc<bestValue){bestValue=fc;bestDepth=c;}if(fd<bestValue){bestValue=fd;bestDepth=d;}
  if(abs(bestValue-tol)<=errorValue){rg_flags|=RG_AMBIGUOUS|131072u;return result;}if(bestValue>tol){return result;}
  var lower=lo;var upper=bestDepth;for(var i=0u;i<65u;i++){let middle=(lower+upper)/2.0;if(rg_cone_value(t,b,z,middle,tol,flat)<=tol){upper=middle;}else{lower=middle;}}contact=upper;
 }
 let section=rg_section(t,b,z,contact,tol,flat);if(section.valid==0u){rg_flags|=RG_AMBIGUOUS|262144u;return result;}
 result.valid=1u;result.triangle=t.id;result.distance=contact;result.point=section.point;result.normal=t.normal;result.material=t.material;let mask=rg_section_mask;if(mask==1u||mask==2u||mask==4u){if(rg_vertex_contact(result,b)){rg_contact_feature=vec4u(1u,t.id,mask,0u);}}else if(mask==3u||mask==5u||mask==6u){rg_contact_feature=vec4u(2u,t.id,mask,0u);}else if(mask==7u){rg_contact_feature=vec4u(3u,t.id,7u,0u);}return result;
}
fn rg_vertex_contact(h:RGHit,b:RGBeam)->bool {
 let t=rg_triangle(h.triangle);var points:array<vec3f,3>;points[0]=t.a;points[1]=t.b;points[2]=t.c;
 for(var i=0u;i<3u;i++){if(all(h.point==points[i])&&h.distance==dot(points[i]-b.origin,b.direction)){
  let errorValue=rg_round(rg_absmax(b.origin)+rg_absmax(points[i])+abs(h.distance));
  for(var j=0u;j<3u;j++){if(j!=i&&dot(points[j]-b.origin,b.direction)-h.distance<=errorValue){return false;}}return true;
 }}return false;
}
fn rg_feature_vertex(f:vec4u)->vec3f {let t=rg_triangle(f.y);if(f.z==1u){return t.a;}if(f.z==2u){return t.b;}return t.c;}
fn rg_feature_a(f:vec4u)->vec3f {let t=rg_triangle(f.y);if(f.z==6u){return t.b;}return t.a;}
fn rg_feature_b(f:vec4u)->vec3f {let t=rg_triangle(f.y);if(f.z==3u){return t.b;}return t.c;}
fn rg_same_edge_feature(a:vec4u,b:vec4u)->bool {if(a.x!=2u||b.x!=2u){return false;}let aa=rg_feature_a(a);let ab=rg_feature_b(a);let ba=rg_feature_a(b);let bb=rg_feature_b(b);return (all(aa==ba)&&all(ab==bb))||(all(aa==bb)&&all(ab==ba));}
fn rg_edge_in_triangle(edge:vec4u,triangle:u32)->bool {if(edge.x!=2u){return false;}let a=rg_feature_a(edge);let b=rg_feature_b(edge);let t=rg_triangle(triangle);return (all(a==t.a)||all(a==t.b)||all(a==t.c))&&(all(b==t.a)||all(b==t.b)||all(b==t.c));}
fn rg_get_transport_feature()->vec4u{return rg_transport_feature;}
fn rg_set_region_feature(f:vec4u) {rg_region_feature=vec4u(0u);if(f.x==0u){return;}if(f.x>2u||f.y>=rg_triangle_count()||(f.x==1u&&f.z!=1u&&f.z!=2u&&f.z!=4u)||(f.x==2u&&f.z!=3u&&f.z!=5u&&f.z!=6u)){rg_flags|=RG_INVALID;return;}rg_region_feature=f;}
fn rg_clear_region_feature(){rg_region_feature=vec4u(0u);}
fn rg_cone_impl(b:RGBeam,transport:bool)->RGHit {
 rg_transport_feature=vec4u(0u);var best:RGHit;var bestFeature=vec4u(0u);var tieFlags=0u;if(!rg_valid_beam(b)){return best;}var stack:array<u32,32>;stack[0]=rg_data[1].w;var count=1u;var visited=0u;
 // No candidate truncation. Each actual triangle gets the full 65+65 search.
 // Fixed full far interval avoids order-dependent clipped minimizers.
 loop {if(count==0u){break;}visited++;if(visited>rg_data[1].x){rg_flags|=RG_INVALID;return RGHit();}count--;let n=rg_node(stack[count]);if(!rg_sphere_cone(n,b)){continue;}
  if(n.count>0u){for(var i=0u;i<n.count;i++){let t=rg_triangle(rg_leaf(n.start+i));let h=rg_contact(t,b);let feature=rg_contact_feature;if(h.valid!=0u){var preferCandidate=false;var retainBest=false;if(best.valid!=0u&&h.triangle!=best.triangle){let tieError=rg_round(max(abs(h.distance),abs(best.distance))+rg_absmax(b.origin));
 if(abs(h.distance-best.distance)<=tieError){
  // The same strict axial-minimum vertex yields identical double-reference
  // depth arithmetic in both triangles. This is a proven stable-ID tie.
  let sameVertex=h.distance==best.distance&&all(h.point==best.point)&&rg_vertex_contact(h,b)&&rg_vertex_contact(best,b);
  // Transport uses only axial depth, not contact triangle identity. For
  // exactly the same original finite edge, convex cone intersection has one
  // earliest axial depth. Its f32 numeric error remains shadow-audited.
  let sameTransportEdge=transport&&rg_same_edge_feature(feature,bestFeature);
  // A proven interior triangle contact dominates a contact on an
  // original edge contained in that triangle. This set-inclusion relation
  // resolves ordering without comparing uncertain f32 contact distances.
  preferCandidate=transport&&feature.x==3u&&rg_edge_in_triangle(bestFeature,h.triangle);
  retainBest=transport&&bestFeature.x==3u&&rg_edge_in_triangle(feature,best.triangle);
  if(!sameVertex&&!sameTransportEdge&&!preferCandidate&&!retainBest){tieFlags|=RG_AMBIGUOUS|524288u;}
 }else if(h.distance<best.distance){tieFlags=0u;}
 }if(best.valid==0u||preferCandidate||(!retainBest&&(h.distance<best.distance||(h.distance==best.distance&&h.triangle<best.triangle)))){best=h;bestFeature=feature;}}}}
  else {if(count+2u>32u){rg_flags|=RG_INVALID;return RGHit();}stack[count]=n.right;stack[count+1u]=n.left;count+=2u;}
 }rg_flags|=tieFlags;if(transport&&bestFeature.x<3u){rg_transport_feature=bestFeature;}return best;
}
fn rg_cone(b:RGBeam)->RGHit{return rg_cone_impl(b,false);}
fn rg_cone_transport(b:RGBeam)->RGHit{return rg_cone_impl(b,true);}
fn rg_metric(p:vec3f,r:RGRegion)->vec3f {let q=p-r.center;let z=dot(q,r.direction);return (q-r.direction*z)/r.radius+r.direction*(z/(2.0*r.radius));}
fn rg_clip_impl(index:u32,r:RGRegion,feature:vec4u)->RGEdge {
 var edge:RGEdge;if(index>=rg_wedge_count()||!rg_finite3(r.center)||!rg_finite3(r.direction)||!rg_finite(r.radius)||!rg_finite(r.depth)||!(r.radius>0.0)||r.depth<0.0||abs(dot(r.direction,r.direction)-1.0)>2e-6){rg_flags|=RG_INVALID;return edge;}
 let o=rg_data[2].y+index*8u;let originalA=rg_f(o).xyz;let originalB=rg_f(o+1u).xyz;let a=rg_metric(originalA,r);let b=rg_metric(originalB,r);if(max(rg_absmax(a),rg_absmax(b))>1e8){rg_flags|=RG_INVALID;return edge;}let d=b-a;let A=dot(d,d);let B=2.0*dot(a,d);let C=dot(a,a)-1.0;var disc=B*B-4.0*A*C;let discScale=B*B+abs(4.0*A*C);
 if(A==0.0){rg_flags|=RG_AMBIGUOUS|1048576u;return edge;}if(abs(disc)<=32.0*RG_EPS*max(discScale,1e-30)){rg_flags|=RG_AMBIGUOUS|2097152u;return edge;}
 if(disc<0.0&&disc>-1e-12*discScale){disc=0.0;}if(disc<0.0){return edge;}
 // Algebraically identical quadratic roots, evaluated around the closest
 // point to avoid subtracting two large squared quantities for long edges.
 let middle=-B/(2.0*A);let closest=a+d*middle;let half=sqrt(max(0.0,(1.0-dot(closest,closest))/A));var u0=max(0.0,middle-half);var u1=min(1.0,middle+half);
 if(r.depth<RG_FMAX){var za=dot(originalA-r.center,r.direction);let dz=dot(originalB-originalA,r.direction);
  // A strict-minimum source vertex is exactly on its contact region's front
  // slab by construction. Preserve that incidence by identity, not epsilon.
  if(feature.x==1u){let p=rg_feature_vertex(feature);if(all(originalA==p)){za=0.0;}else if(all(originalB==p)){za=-dz;}}
  let spatialError=rg_round(max(rg_absmax(originalA),rg_absmax(r.center)));if(abs(dz)<1e-14){let tol=rg_f(o).w;if(za< -tol||za>r.depth+tol){if(abs(za+tol)<=spatialError||abs(za-r.depth-tol)<=spatialError){rg_flags|=RG_AMBIGUOUS|4194304u;}return edge;}}
  else {if(abs(dz)<=rg_round(length(originalB-originalA))){rg_flags|=RG_AMBIGUOUS|8388608u;return edge;}let v0=-za/dz;let v1=(r.depth-za)/dz;u0=max(u0,min(v0,v1));u1=min(u1,max(v0,v1));}}
 if(u1-u0<=1e-12){if(abs(u1-u0)<=64.0*RG_EPS){rg_flags|=RG_AMBIGUOUS|16777216u;}return edge;}
 if((u0>0.0&&u0<64.0*RG_EPS)||(u1<1.0&&1.0-u1<64.0*RG_EPS)||u1-u0<=64.0*RG_EPS){rg_flags|=RG_AMBIGUOUS|33554432u;return edge;}
 let delta=originalB-originalA;let pa=originalA+delta*u0;let pb=originalA+delta*u1;if(!(length(pb-pa)>0.0)){rg_flags|=RG_AMBIGUOUS|67108864u;return edge;}edge.valid=1u;edge.index=index;edge.edgeId=rg_data[o+3u].w;edge.a=pa;edge.b=pb;edge.v=(pa+pb)/2.0;edge.e=rg_f(o+3u).xyz;edge.n1=rg_f(o+4u).xyz;edge.n2=rg_f(o+5u).xyz;edge.t1=rg_f(o+6u).xyz;edge.t2=rg_f(o+7u).xyz;edge.l=length(pb-pa);edge.alpha=rg_f(o+2u).w;edge.originalA=originalA;edge.originalB=originalB;edge.clippedStart=select(0u,1u,u0>1e-10);edge.clippedEnd=select(0u,1u,u1<1.0-1e-10);return edge;
}

// Generic clipping remains provenance-free. The resident walker explicitly
// selects active clipping and restores the matching region's saved identity.
fn rg_clip(index:u32,r:RGRegion)->RGEdge{return rg_clip_impl(index,r,vec4u(0u));}
fn rg_clip_active(index:u32,r:RGRegion)->RGEdge{return rg_clip_impl(index,r,rg_region_feature);}
