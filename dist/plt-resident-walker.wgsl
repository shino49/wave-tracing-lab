// Compile-time opt-in keeps experimental interval graphs out of stable pipelines.
override RR_AXIS_INTERVAL_ENABLED:bool=false;
override RR_ANCHOR_ENABLED:bool=false;
override RR_TIGHT_ENABLED:bool=false;
override RR_CAMERA_ENABLED:bool=false;
// Experimental genuine resident transport. CPU provides immutable geometry and
// initial RNG prefixes only. No CPU path/event tape is consumed by this shader.
// Admission remains exhaustive CPU shadow audit; this is NOT an unchecked lane.
// config.y opts into paused failed samples (header.w=3) for whole-path CPU replay.
struct RRParameters {
  frame: vec4u, tile: vec4u, run: vec4u, batch: vec4u,
  cameraOrigin: vec4f, cameraForward: vec4f, cameraRight: vec4f, cameraUp: vec4f,
  optics: vec4f, config: vec4u,
};
struct RRLane {
  header: vec4u, counters: vec4u, pending: vec4u, trace: vec4u, progress: vec4u,
  origin: vec4f, direction: vec4f, previousOrigin: vec4f, previousDirection: vec4f,
  previousPosition: vec4f, pendingCenter: vec4f, pendingDirection: vec4f, scalars: vec4f,
  vertexHeader:vec4u,vertexCenter:vec4f,vertexPoint:vec4f,vertexNormal:vec4f,vertexRegion:vec4f,vertexDirection:vec4f,vertexMeta:vec4u,traversal:vec4f,fsdHeader:vec4u,fsdMeta:vec4u,vertexFeature:vec4u,pendingFeature:vec4u,intervalOriginError:vec4f,intervalDirectionError:vec4f,intervalMeta:vec4u,
};
struct RRAState { prepared:RAQueryState, progress:RAScanProgress, plane:RAPlaneState };
struct RRAnchorState { anchor:RAAnchor, result:RAResult, proofMeta:vec4u, checked:u32 };
struct RRCameraState { camera:RACameraState, proofMeta:vec4u };
@group(0) @binding(10) var<storage,read_write> rr_camera_states:array<RRCameraState>;
@group(0) @binding(11) var<storage,read> rr_camera_parameters:RACameraParameters;
@group(0) @binding(9) var<storage,read_write> rr_anchor_states:array<RRAnchorState>;
@group(0) @binding(8) var<storage,read_write> rr_axis_states:array<RRAState>;
struct RRVertex { valid:u32, hit:RGHit, center:vec3f, distance:f32, region:RGRegion, feature:vec4u };
struct RRLight { valid:u32, direction:vec3f, point:vec3f, distance:f32, cosine:f32, pdf:f32, emission:f32 };
@group(0) @binding(0) var<uniform> rr: RRParameters;
@group(0) @binding(2) var<storage,read> rr_initial: array<vec4u>;
@group(0) @binding(3) var<storage,read_write> rr_lanes: array<RRLane>;
@group(0) @binding(4) var<storage,read_write> rr_audit: array<vec4u>;
@group(0) @binding(5) var<storage,read_write> rr_status: array<atomic<u32>,4>;
const RR_PI = 3.14159265358979323846;
const RR_TAU = 6.28318530717958647692;
const RR_FAR = 1e30;
const RR_AUDIT_STRIDE = 387u;
const RR_VERTEX_OFFSET = 20u;
const RR_EVENT_OFFSET = 308u;
const RR_CONTRIBUTION_OFFSET = 338u;
const RR_SCATTER_OFFSET = 362u;
@group(0) @binding(6) var<storage,read_write> rr_records: array<vec4u>;
@group(0) @binding(7) var<storage,read_write> rr_weights: array<vec4u>;
var<private> rr_lane_index:u32;
fn rr_weight(s:ptr<function,RRLane>,index:u32)->f32 { return bitcast<f32>(rr_weights[rr_lane_index*48u+index/4u][index%4u]); }
fn rr_set_weight(s:ptr<function,RRLane>,index:u32,value:f32) { rr_weights[rr_lane_index*48u+index/4u][index%4u]=bitcast<u32>(value); }
fn rr_saved(s:ptr<function,RRLane>,index:u32)->f32 { return bitcast<f32>(rr_weights[rr_lane_index*48u+32u+index/4u][index%4u]); }
fn rr_beam(s:ptr<function,RRLane>)->RGBeam { return RGBeam((*s).origin.xyz,(*s).direction.xyz,(*s).origin.w,(*s).direction.w,1e-7,RR_FAR); }
fn rr_previous_beam(s:ptr<function,RRLane>)->RGBeam { return RGBeam((*s).previousOrigin.xyz,(*s).previousDirection.xyz,(*s).previousOrigin.w,(*s).previousDirection.w,1e-7,RR_FAR); }
fn rr_mis(a:f32,b:f32)->f32 { if(a<=0.0){return 0.0;} return a*a/(a*a+b*b); }
fn rr_ok(v:f32)->bool { return v==v && abs(v)<1e30; }
fn rr_radius(b:RGBeam,t:f32)->f32 { return b.radius0+b.tanAngle*t; }
// One geometric segment per dispatch: avoids a nested whole-walk kernel and
// keeps continuation on GPU even on long near-contact/null searches.
fn rr_traverse(b:RGBeam,wave:bool,previousTravel:f32,segment:u32)->RRVertex {
 var v:RRVertex;
 if(!wave){let h=rg_ray(b);if(h.valid!=0u){v.valid=1u;v.hit=h;v.center=h.point;v.distance=h.distance;}return v;}
 if(segment>16u){rg_flags|=32u;return v;}
 var span=RR_FAR;if(segment<16u){span=min(65536.0,16.0*exp2(2.0*f32(segment)))*rr.optics.z;}
 var query=b;query.near=max(b.near,previousTravel);query.far=min(b.far,previousTravel+span*1.001);
 let h=rg_ray(query);if(h.valid!=0u){let radius=rr_radius(b,h.distance);v.valid=1u;v.hit=h;v.center=h.point;v.distance=h.distance;v.region=RGRegion(h.point,b.direction,radius,2.0*radius);return v;}
 let travelled=previousTravel+span;if(travelled>=b.far){return v;}
 query=b;query.near=travelled;let contact=rg_cone_transport(query);let feature=rg_get_transport_feature();if(contact.valid==0u){return v;}
 if(contact.distance-travelled<rr_radius(b,travelled)/2.0){v.valid=2u;v.distance=travelled;return v;}
 let t=contact.distance;let center=b.origin+b.direction*t;let radius=rr_radius(b,t);query=b;query.near=max(t,b.near);query.far=t+2.0*radius;
 v.valid=1u;v.hit=rg_ray(query);v.center=center;v.distance=t;v.region=RGRegion(center,b.direction,radius,2.0*radius);v.feature=feature;return v;
}

fn rr_cosine(normal:vec3f,rng:ptr<function,u32>)->vec4f {
  let a=RR_TAU*rr_float(rng);let radius=sqrt(rr_float(rng));let z=sqrt(max(0.0,1.0-radius*radius));
  let axis=select(vec3f(0,1,0),vec3f(0,0,1),abs(normal.z)<0.9);
  let u=normalize(cross(axis,normal));let v=cross(normal,u);
  let trig=rf_sincos(a);return vec4f(normal*z+u*(radius*trig.x)+v*(radius*trig.y),z/RR_PI);
}
fn rr_light(point:vec3f,rng:ptr<function,u32>)->RRLight {
  var result:RRLight;let count=rg_emitter_count();if(count==0u){return result;}
  var choice=rr_float(rng)*rg_emitter_area();var index=rg_emitter_index(count-1u);
  for(var i=0u;i<count;i++){let candidate=rg_emitter_index(i);choice-=rg_triangle(candidate).area;if(choice<=0.0){index=candidate;break;}}
  let t=rg_triangle(index);let a=sqrt(rr_float(rng));let b=rr_float(rng);
  var p=t.a*(1.0-a)+t.b*(a*(1.0-b))+t.c*(a*b);for(var axis=0u;axis<3u;axis++){if(t.a[axis]==t.b[axis]&&t.a[axis]==t.c[axis]){p[axis]=t.a[axis];}}let delta=p-point;let d=length(delta);
  if(d<1e-8){return result;}let direction=delta/d;let m=rg_material(t.material);
  let cosine=select(max(0.0,-dot(t.normal,direction)),abs(dot(t.normal,direction)),m.twoSidedEmission!=0u);
  result.valid=1u;result.direction=direction;result.point=p;result.distance=d;result.cosine=cosine;result.emission=m.emission;
  result.pdf=RR_FAR;if(cosine>0.0){result.pdf=d*d/(rg_emitter_area()*cosine);}return result;
}
fn rr_add(s:ptr<function,RRLane>,base:u32,kind:u32,weight:f32,misWeight:f32) {
  if(!(weight>0.0)){return;}
  var value=0.0;
  for(var i=0u;i<(*s).counters.z;i++) {
    var term=rr_weight(s,i);if(kind==1u&&rr_weights[rr_lane_index*48u+16u+i/4u][i%4u]!=0u){term*=misWeight;}
    value+=term*weight;
  }
  if(!rr_ok(value)||value<0.0){rg_flags|=64u;return;}
  let index=(*s).trace.z;
  if(index>=12u){rg_flags|=16u;return;}
  rr_audit[base+RR_CONTRIBUTION_OFFSET+index*2u]=vec4u(kind,(*s).counters.x,0u,0u);
  rr_audit[base+RR_CONTRIBUTION_OFFSET+index*2u+1u]=bitcast<vec4u>(vec4f(value,weight,misWeight,0));
  (*s).trace.z+=1u;
  let adjusted=value-(*s).scalars.w;let next=(*s).scalars.x+adjusted;(*s).scalars.w=(next-(*s).scalars.x)-adjusted;(*s).scalars.x=next;
}
fn rr_scatter(s:ptr<function,RRLane>,base:u32,kind:u32,direction:vec3f,pdf:f32,weight:f32,valid:u32,edgeId:u32,isDirect:u32,point:vec3f) {
 let index=(*s).progress.w;if(index>=6u){rg_flags|=16u;return;}let at=base+RR_SCATTER_OFFSET+index*4u;
 rr_audit[at]=vec4u(kind,(*s).counters.x,valid,edgeId);rr_audit[at+1u]=bitcast<vec4u>(vec4f(direction,pdf));rr_audit[at+2u]=bitcast<vec4u>(vec4f(weight,f32(isDirect),0,0));rr_audit[at+3u]=bitcast<vec4u>(vec4f(point,0));(*s).progress.w+=1u;
}
fn rr_axis_requested()->bool{return rr.config.z==2u||rr.config.z==4u||rr.config.z==6u||rr.config.z==8u||rr.config.z==10u;}
fn rr_anchor_requested()->bool{return rr.config.z==6u||(RR_TIGHT_ENABLED&&rr.config.z==8u)||(RR_CAMERA_ENABLED&&rr.config.z==10u);}
fn rr_anchor_marker()->u32{return select(select(3u,4u,rr.config.z==8u),5u,rr.config.z==10u);}
// CPU-state enclosure initialization. Immutable host constants and raw RNG
// words only; no CPU camera/path tape. The scalar RI operations are outward
// rounded and fail unknown outside their explicitly supported normal range.
fn rri_point(v:f32)->RAI{return ra_ipoint(bitcast<u32>(v));}
fn rri_enclose(v:f32,e:f32)->RAI{return ra_iadd(rri_point(v),ra_ibounds(bitcast<u32>(-e),bitcast<u32>(e)));}
fn rri_vector(v:vec3f,e:vec3f)->RAVec3{return RAVec3(rri_enclose(v.x,e.x),rri_enclose(v.y,e.y),rri_enclose(v.z,e.z));}
fn rri_valid3(v:RAVec3)->bool{return !ra_bad;}
fn rri_error(v:RAI,center:f32)->f32{let d=ra_isub(v,rri_point(center));if(ra_bad){return 1e30;}return max(abs(d.lo),abs(d.hi));}
fn rri_random(word:u32)->RAI{if(word==0u){return rri_point(0.0);}return ra_imul(ra_pad(bitcast<u32>(f32(word)),1u),rri_point(0x1p-32));}
fn rri_camera_raw(pixel:u32,word0:u32,word1:u32)->RAVec3 {
 let x=rri_point(f32(pixel%rr.frame.x));let y=rri_point(f32(pixel/rr.frame.x));
 let width=rri_point(f32(rr.frame.x));let height=rri_point(f32(rr.frame.y));let one=rri_point(1.0);let two=rri_point(2.0);
 let half=rri_enclose(rr.optics.x,bitcast<f32>(rr.config.w));let aspect=ra_idiv(width,height);
 let u=ra_imul(ra_imul(ra_isub(ra_idiv(ra_imul(two,ra_iadd(x,rri_random(word0))),width),one),aspect),half);
 let v=ra_imul(ra_isub(one,ra_idiv(ra_imul(two,ra_iadd(y,rri_random(word1))),height)),half);
 let f=rri_vector(rr.cameraForward.xyz,vec3f(rr.cameraForward.w));let r=rri_vector(rr.cameraRight.xyz,vec3f(rr.cameraRight.w));let up=rri_vector(rr.cameraUp.xyz,vec3f(rr.cameraUp.w));
 let raw=RAVec3(ra_iadd(f.x,ra_iadd(ra_imul(r.x,u),ra_imul(up.x,v))),ra_iadd(f.y,ra_iadd(ra_imul(r.y,u),ra_imul(up.y,v))),ra_iadd(f.z,ra_iadd(ra_imul(r.z,u),ra_imul(up.z,v))));
 return raw;
}
fn rri_initialize(s:ptr<function,RRLane>){
 (*s).intervalMeta=vec4u(0u,3u,0u,0u);(*s).intervalOriginError=vec4f(0);(*s).intervalDirectionError=vec4f(0);
 if(!RR_AXIS_INTERVAL_ENABLED||!rr_axis_requested()||rr.run.w!=0u||rg_data[7].z!=1u){return;}
 // Compact RA operators retain integer outward steps at every arithmetic
 // boundary. Their sticky failure is reset only here, never within the graph.
 // Host upload validation guarantees no subnormal/NaN camera constants.
 ra_bad=false;
 let raw=rri_camera_raw((*s).header.x,rr_rng_words[0],rr_rng_words[1]);
 // CPU sampleCamera uses this operation order and reciprocal-then-multiply
 // normalization. ra_cpu_normalize includes the declared hypot <=16*u64
 // relative-error allowance; the extra 1e-13 preserves the earlier slack.
 let unit=ra_cpu_normalize(raw);if(ra_bad){return;}
 let extra=ra_ibounds(bitcast<u32>(-1e-13),bitcast<u32>(1e-13));let normalized=RAVec3(ra_iadd(unit.x,extra),ra_iadd(unit.y,extra),ra_iadd(unit.z,extra));
 let error=vec3f(rri_error(normalized.x,(*s).direction.x),rri_error(normalized.y,(*s).direction.y),rri_error(normalized.z,(*s).direction.z));
 if(ra_bad||any(error>=vec3f(1e29))){return;}
 (*s).intervalDirectionError=vec4f(error,0);(*s).intervalOriginError=vec4f(vec3f(rr.cameraOrigin.w),0);// Provenance2 is granted only after the actual camera envelope succeeds.
 // It survives only the explicitly checked exact-axis sign flip below.
 (*s).intervalMeta.x=select(select(1u,2u,rr.config.z==4u),rr_anchor_marker(),RR_ANCHOR_ENABLED&&rr_anchor_requested());
}
fn rr_start(s:ptr<function,RRLane>) {
  (*s).counters=vec4u(0u,0u,1u,0u);(*s).pending=vec4u(0u,0u,1u,0u);
  (*s).trace=vec4u(0u);(*s).progress=vec4u(2166136261u,(*s).header.y,1u,0u);
  (*s).scalars.x=0.0;(*s).scalars.w=0.0;
  rr_rng_count=0u;rr_rng_hash=2166136261u;
  var rng=(*s).header.y;let pixel=(*s).header.x;let x=pixel%rr.frame.x;let y=pixel/rr.frame.x;
  let u=(2.0*(f32(x)+rr_float(&rng))/f32(rr.frame.x)-1.0)*rr.optics.y*rr.optics.x;
  let v=(1.0-2.0*(f32(y)+rr_float(&rng))/f32(rr.frame.y))*rr.optics.x;
  let direction=normalize(rr.cameraForward.xyz+rr.cameraRight.xyz*u+rr.cameraUp.xyz*v);
  let pixelWidth=20.0*rr.optics.x/f32(rr.frame.y);let wave=rr.run.w!=0u;
  (*s).origin=vec4f(rr.cameraOrigin.xyz,select(0.0,0.75*pixelWidth,wave));
  (*s).direction=vec4f(direction,select(0.0,pixelWidth/10.0,wave));
  (*s).previousPosition=vec4f(rr.cameraOrigin.xyz,0.0);rr_weights[rr_lane_index*48u].x=bitcast<u32>(1.0);rr_weights[rr_lane_index*48u+16u].x=1u;
  (*s).header.y=rng;(*s).header.w=1u;(*s).traversal=vec4f(0);(*s).vertexMeta=vec4u(0u);(*s).fsdHeader=vec4u(0u);(*s).fsdMeta=vec4u(0u);(*s).vertexFeature=vec4u(0u);(*s).pendingFeature=vec4u(0u);rri_initialize(s);
}
fn rr_finish(s:ptr<function,RRLane>,base:u32,termination:u32) {
  // A rejected path carries no reusable geometry proof across CPU replay.
  if(rg_flags!=0u){(*s).intervalMeta.x=0u;}
  rr_audit[base]=vec4u((*s).header.x,(*s).header.z,rr.run.w,rg_flags);
  rr_audit[base+1u]=vec4u((*s).progress.y,(*s).header.y,rr_rng_count,rr_rng_hash);
  rr_audit[base+2u]=vec4u((*s).trace.xyz,termination);
  rr_audit[base+3u]=bitcast<vec4u>(vec4f((*s).scalars.x,f32((*s).counters.x),f32((*s).counters.y),f32((*s).progress.w)));
  rr_audit[base+386u]=vec4u((*s).intervalMeta.z,(*s).intervalMeta.x,(*s).intervalMeta.y,(*s).intervalMeta.w);
  // A failed sample contributes nothing. The host must replay its entire path
  // from progress.y and supply its canonical final RNG before this lane resumes.
  if(rg_flags==0u){let adjusted=(*s).scalars.x-(*s).scalars.z;let next=(*s).scalars.y+adjusted;(*s).scalars.z=(next-(*s).scalars.y)-adjusted;(*s).scalars.y=next;}
  (*s).header.z+=1u;(*s).header.w=select(0u,2u,(*s).header.z>=rr.run.x+rr.run.y);
  if(rg_flags!=0u&&rr.config.y!=0u){(*s).header.w=3u;}
}
@compute @workgroup_size(32)
fn rr_init_anchored(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=rr.tile.z*rr.tile.w){return;}
  var s:RRLane;s.header=vec4u(rr_initial[id.x].xy,rr.run.x,0u);rr_lanes[id.x]=s;
  if(RR_ANCHOR_ENABLED){rr_anchor_states[id.x].proofMeta=vec4u(0u);rr_anchor_states[id.x].checked=0u;}
}

fn rr_store_vertex(s:ptr<function,RRLane>,v:RRVertex,count:u32) {
 (*s).vertexFeature=v.feature;
 (*s).vertexHeader=vec4u(v.valid,v.hit.valid,v.hit.triangle,v.hit.material);(*s).vertexCenter=vec4f(v.center,v.distance);(*s).vertexPoint=vec4f(v.hit.point,v.hit.distance);(*s).vertexNormal=vec4f(v.hit.normal,0);(*s).vertexRegion=vec4f(v.region.center,v.region.radius);(*s).vertexDirection=vec4f(v.region.direction,v.region.depth);(*s).vertexMeta=vec4u(count,0u,1u,0u);(*s).fsdHeader=vec4u(0u);(*s).fsdMeta=vec4u(0u);
}
fn rr_load_vertex(s:ptr<function,RRLane>)->RRVertex {
 return RRVertex((*s).vertexHeader.x,RGHit((*s).vertexHeader.y,(*s).vertexHeader.z,(*s).vertexPoint.w,(*s).vertexPoint.xyz,(*s).vertexNormal.xyz,(*s).vertexHeader.w),(*s).vertexCenter.xyz,(*s).vertexCenter.w,RGRegion((*s).vertexRegion.xyz,(*s).vertexDirection.xyz,(*s).vertexRegion.w,(*s).vertexDirection.w),(*s).vertexFeature);
}
fn rr_base(lane:u32,s:ptr<function,RRLane>)->u32{return (lane*rr.run.y+((*s).header.z-rr.run.x))*RR_AUDIT_STRIDE;}
fn rr_flush_rng(s:ptr<function,RRLane>,base:u32,oldDraws:u32){for(var i=oldDraws;i<min(rr_rng_count,64u);i++){rr_audit[base+4u+i/4u][i%4u]=rr_rng_words[i];}(*s).trace.w=rr_rng_count;(*s).progress.x=rr_rng_hash;}
@compute @workgroup_size(32)
fn rr_traverse_step(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
  rr_lane_index=lane;var s=rr_lanes[lane];if(s.header.w>=2u||(s.header.w==1u&&s.vertexMeta.z!=0u)){return;}rg_flags=s.pending.w;
  rr_rng_count=s.trace.w;rr_rng_hash=s.progress.x;let oldDraws=select(s.trace.w,0u,s.header.w==0u);if(s.header.w==0u){rr_start(&s);}let base=rr_base(lane,&s);var done=false;var termination=0u;
  // There is no certified propagated origin/direction envelope after scattering
  // yet. The conservative coverage guard quarantines that unsupported history
  // before any next query; it is not a heuristic distance threshold or a claim
  // that primary-segment predicates have been generally certified.
  if(rr.config.z==1u&&(s.counters.x>0u||s.counters.y>0u)){rg_flags|=134217730u;rr_flush_rng(&s,base,oldDraws);s.pending.w=rg_flags;rr_finish(&s,base,7u);rr_lanes[lane]=s;return;}
  if(rr.config.z>=2u&&(!rr_axis_requested()||rr.run.w!=0u||s.intervalMeta.x==0u||(rr.config.z==4u&&s.intervalMeta.x!=2u)||((rr.config.z==6u||rr.config.z==8u||rr.config.z==10u)&&(!RR_ANCHOR_ENABLED||!rr_anchor_requested()||s.intervalMeta.x!=rr_anchor_marker()))||s.counters.y>0u)){rg_flags|=134217730u;rr_flush_rng(&s,base,oldDraws);s.pending.w=rg_flags;rr_finish(&s,base,7u);rr_lanes[lane]=s;return;}
  let beam=rr_beam(&s);let vertex=rr_traverse(beam,rr.run.w!=0u,s.traversal.x,u32(s.traversal.y));

  if(vertex.valid==2u&&rg_flags==0u){s.traversal.x=vertex.distance;s.traversal.y+=1.0;s.vertexMeta.z=0u;rr_flush_rng(&s,base,oldDraws);rr_lanes[lane]=s;return;}
  var apertureCount=0u;
  if(vertex.valid!=0u&&rr.run.w!=0u&&vertex.region.radius>0.0){rg_set_region_feature(vertex.feature);apertureCount=rf_count(vertex.region,beam.direction);}
  let vi=s.trace.x;
  if(vi>=72u){rg_flags|=16u;done=true;termination=7u;}
  else {
    let at=base+RR_VERTEX_OFFSET+vi*4u;let kind=select(select(2u,1u,vertex.hit.valid!=0u),0u,vertex.valid==0u);
    rr_audit[at]=vec4u(kind,select(0xffffffffu,vertex.hit.triangle,vertex.hit.valid!=0u),apertureCount,s.counters.x);
    rr_audit[at+1u]=bitcast<vec4u>(vec4f(vertex.center,vertex.distance));
    rr_audit[at+2u]=bitcast<vec4u>(vec4f(beam.direction,vertex.region.radius));
    rr_audit[at+3u]=bitcast<vec4u>(vec4f(beam.origin,beam.tanAngle));s.trace.x+=1u;
  }
  if(vertex.valid==0u){done=true;termination=1u;}
  rr_store_vertex(&s,vertex,apertureCount);rr_flush_rng(&s,base,oldDraws);
  // Delay all transport/escape completion until every staged geometry bound
  // passes. The three guard passes run immediately before records/scatter.
  if(RR_AXIS_INTERVAL_ENABLED&&rr_axis_requested()&&rg_flags==0u){s.vertexMeta.w=1u;done=false;}
  if(rg_flags!=0u){done=true;termination=7u;}s.pending.w=rg_flags;if(done){rr_finish(&s,base,termination);}rr_lanes[lane]=s;
}
// These stages persist only GPU-computed intervals, never CPU path state.
// Splitting the bounded expression graphs avoids the observed software-driver
// pipeline compiler failure without reducing their numerical guarantees.
fn rr_axis_pending(s:RRLane)->bool{return RR_AXIS_INTERVAL_ENABLED&&rr_axis_requested()&&s.header.w==1u&&s.vertexMeta.w==1u;}
// Separate camera/raw pass avoids adding a larger graph to traversal or prepare.
// Reconstruct the same two GPU words from the immutable original sample prefix;
// this private invocation never advances the persistent transport RNG or audit.
@compute @workgroup_size(32)
fn rr_anchor_init_step(@builtin(global_invocation_id) id:vec3u){
 if(!RR_ANCHOR_ENABLED||!rr_anchor_requested()||id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 let s=rr_lanes[lane];if(!rr_axis_pending(s)||s.counters.x!=0u){return;}
 rr_anchor_states[lane].proofMeta=vec4u(0u);rr_anchor_states[lane].checked=0u;
 if(s.intervalMeta.x!=rr_anchor_marker()||s.counters.y!=0u){return;}
 var rng=s.progress.y;rr_rng_count=0u;rr_rng_hash=2166136261u;let word0=rr_word(&rng);let word1=rr_word(&rng);
 ra_bad=false;let raw=rri_camera_raw(s.header.x,word0,word1);if(ra_bad){return;}
 var anchor=ra_anchor_initialize(raw);if(RR_TIGHT_ENABLED&&rr.config.z==8u){anchor=ra_anchor_tight_refine_norm(anchor);}rr_anchor_states[lane].anchor=anchor;
 rr_anchor_states[lane].proofMeta=vec4u(anchor.valid,s.header.z,s.progress.y,0u);
}
fn rr_camera_owned(s:RRLane,lane:u32)->bool {
 if(rr.config.z!=10u){return true;}
 if(!RR_CAMERA_ENABLED){return false;}
 let camera=rr_camera_states[lane];
 return camera.camera.valid==1u&&all(camera.proofMeta==vec4u(1u,s.header.z,s.progress.y,5u));
}
fn rr_anchor_owned(s:RRLane,state:RRAnchorState)->bool {
 return RR_ANCHOR_ENABLED&&rr_anchor_requested()&&s.intervalMeta.x==rr_anchor_marker()&&state.proofMeta.x==1u&&state.proofMeta.y==s.header.z&&state.proofMeta.z==s.progress.y&&state.proofMeta.w==s.counters.x&&state.anchor.valid==1u;
}
@compute @workgroup_size(32)
fn rr_anchor_prepare_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 var prepared:RAQueryState;
 if(RR_ANCHOR_ENABLED){rr_anchor_states[lane].checked=0u;}
 if(RR_ANCHOR_ENABLED&&rr_anchor_requested()){prepared=ra_prepare_unit(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz,select(0u,1u,rr_anchor_owned(s,rr_anchor_states[lane])));}
 else if(rr.config.z==4u){prepared=ra_prepare_unit(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz,select(0u,1u,s.intervalMeta.x==2u));}
 else{prepared=ra_prepare(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz);}
 rr_axis_states[lane].prepared=prepared;rr_axis_states[lane].progress=ra_scan_begin(rr_load_vertex(&s).hit,prepared);
}
@compute @workgroup_size(32)
fn rr_anchor_plane_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 let s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 let state=rr_axis_states[lane];
 if(RR_ANCHOR_ENABLED&&rr_anchor_requested()){
  if(state.progress.valid!=1u||state.progress.cursor>=rg_triangle_count()||!rr_anchor_owned(s,rr_anchor_states[lane])){rr_axis_states[lane].plane=ra_plane_unknown(state.progress.cursor,RA_UNSUPPORTED);return;}
  if(RR_TIGHT_ENABLED&&rr.config.z==8u){
   rr_axis_states[lane].plane=ra_anchor_tight_plane_query(rg_triangle(state.progress.cursor),state.progress.cursor,s.origin.xyz,s.intervalOriginError.xyz,state.prepared,rr_anchor_states[lane].anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w));
  }else{rr_axis_states[lane].plane=ra_anchor_plane_query(rg_triangle(state.progress.cursor),state.progress.cursor,s.origin.xyz,s.intervalOriginError.xyz,state.prepared,rr_anchor_states[lane].anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w));}
 }else{rr_axis_states[lane].plane=ra_scan_plane(s.origin.xyz,s.intervalOriginError.xyz,state.prepared,state.progress);}
}
@compute @workgroup_size(32)
fn rr_axis_scan_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 rr_axis_states[lane].progress=ra_scan_step(rr_load_vertex(&s).hit,rr_axis_states[lane].plane,rr_axis_states[lane].progress);
}
// Dedicated point/residual pass; pending transport remains blocked until finish.
@compute @workgroup_size(32)
fn rr_anchor_point_step(@builtin(global_invocation_id) id:vec3u){
 if(!RR_ANCHOR_ENABLED||!rr_anchor_requested()||id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}let state=rr_axis_states[lane];
 let anchorState=rr_anchor_states[lane];
 var enclosed=RAAnchorResult(ra_failure(RA_UNSUPPORTED),anchorState.anchor);
 if(rr_anchor_owned(s,anchorState)){
  let candidate=rr_load_vertex(&s).hit;
  if(RR_TIGHT_ENABLED&&rr.config.z==8u){
   enclosed=ra_anchor_tight_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,candidate,state.prepared,ra_scan_end(candidate,state.progress),anchorState.anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w));
  }else{enclosed=ra_anchor_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,candidate,state.prepared,ra_scan_end(candidate,state.progress),anchorState.anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w));}
  if(enclosed.result.accepted==1u&&candidate.valid==1u&&rg_material(candidate.material).kind==1u){
   enclosed.anchor=ra_anchor_reflect(enclosed.anchor);
   if(enclosed.anchor.valid!=1u){enclosed.result=ra_failure(RA_UNSUPPORTED);}
  }
 }
 rr_anchor_states[lane].anchor=enclosed.anchor;rr_anchor_states[lane].result=enclosed.result;rr_anchor_states[lane].checked=1u;
}
@compute @workgroup_size(32)
fn rr_anchor_finish_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 let vertex=rr_load_vertex(&s);let state=rr_axis_states[lane];
 var enclosed:RAResult;
 if(RR_ANCHOR_ENABLED&&rr_anchor_requested()){
  let anchorState=rr_anchor_states[lane];enclosed=anchorState.result;
  if(anchorState.checked!=1u||!rr_anchor_owned(s,anchorState)){enclosed=ra_failure(RA_UNSUPPORTED);}
  rr_anchor_states[lane].checked=0u;
  if(enclosed.accepted==0u){rr_anchor_states[lane].proofMeta.x=0u;}
  else if(vertex.hit.valid!=0u){rr_anchor_states[lane].proofMeta.w+=1u;}
 }else{enclosed=ra_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,vertex.hit,state.prepared,ra_scan_end(vertex.hit,state.progress));}
 s.intervalMeta.z+=1u;s.intervalMeta.w=enclosed.reason;s.vertexMeta.w=0u;
 rg_flags=s.pending.w;rr_rng_count=s.trace.w;rr_rng_hash=s.progress.x;let base=rr_base(lane,&s);
 if(enclosed.accepted==0u){rg_flags|=268435458u;s.pending.w=rg_flags;rr_finish(&s,base,7u);}
 else if(vertex.valid==0u){rr_finish(&s,base,1u);}
 else if(vertex.hit.valid!=0u){s.intervalOriginError=vec4f(enclosed.originError,enclosed.planeValue);s.intervalDirectionError.w=enclosed.planeResidual;s.intervalMeta.y=enclosed.planeAxis;}
 rr_lanes[lane]=s;
}
// Record geometry is GPU-generated, not a CPU event replay. Per-event links
// preserve stable wedge summation order despite concurrent pool allocation.
@compute @workgroup_size(32)
fn rr_records_step(@builtin(global_invocation_id) id:vec3u) {
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 rr_lane_index=lane;var s=rr_lanes[lane];if(s.header.w!=1u||s.vertexMeta.z==0u||s.vertexMeta.w!=0u||s.pending.x==0u){return;}rg_flags=s.pending.w;
 let base=rr_base(lane,&s);let vertex=rr_load_vertex(&s);let region=RGRegion(s.pendingCenter.xyz,s.pendingDirection.xyz,s.pendingCenter.w,s.pendingDirection.w);let ei=s.trace.y;rg_set_region_feature(s.pendingFeature);
 if(ei>=6u){s.pending.w|=16u;rr_lanes[lane]=s;return;}
 let at=base+RR_EVENT_OFFSET+ei*5u;
 if(s.fsdMeta.w==0u){
  let contains=rg_cone_contains(rr_previous_beam(&s),vertex.center,1e-10);var direct=0.0;if(contains&&rg_visible(s.previousOrigin.xyz,vertex.center,rg_endpoint_epsilon(s.previousOrigin.xyz,vertex.center))){direct=1.0;}
  rr_audit[at+1u]=bitcast<vec4u>(vec4f(0,direct,0,0));s.fsdHeader=vec4u(0u,0xffffffffu,0xffffffffu,0u);s.fsdMeta=vec4u(2166136261u,0u,select(0u,1u,contains),1u);
 }
 // One finite wedge per dispatch; all remaining scan state stays resident.
 // This is intentionally conservative compiler/watchdog staging, not pruning.
 if(s.fsdHeader.x<rg_wedge_count()){
  let record=rf_record(region,s.previousDirection.xyz,rr.optics.w,s.previousOrigin.xyz,vertex.center,s.fsdHeader.x);s.fsdMeta.y|=record.diagnostics;s.fsdHeader.x+=1u;
  if(record.valid!=0u){
   let index=atomicAdd(&rr_status[2],1u);if(index>=131072u){rg_flags|=16u;}
   else{let out=index*4u;rr_records[out]=vec4u(lane*rr.run.y+(s.header.z-rr.run.x),ei,record.edgeId,0xffffffffu);rr_records[out+1u]=bitcast<vec4u>(vec4f(record.point,record.ri));rr_records[out+2u]=bitcast<vec4u>(vec4f(record.ro,record.sinBeta,record.phiI,record.phiO));rr_records[out+3u]=bitcast<vec4u>(vec4f(record.n,record.phaseExcess,record.L,record.fermatResidual));
    if(s.fsdHeader.y==0xffffffffu){s.fsdHeader.y=index;}else{rr_records[s.fsdHeader.z*4u].w=index;}s.fsdHeader.z=index;s.fsdHeader.w+=1u;s.fsdMeta.x=(s.fsdMeta.x^record.edgeId)*16777619u;if(s.fsdMeta.z!=0u&&record.angularSeam!=0u){rr_audit[at+1u].y=bitcast<u32>(0.5);}
   }
  }
 }
 if(s.fsdHeader.x<rg_wedge_count()&&rg_flags==0u){s.vertexMeta.z=2u;s.pending.w=rg_flags;rr_lanes[lane]=s;return;}
 rr_audit[at]=vec4u(s.fsdHeader.w,s.fsdMeta.x,s.counters.x,s.pending.y);rr_audit[at+2u]=vec4u(0u,0u,bitcast<u32>(f32(s.fsdMeta.y)),s.fsdHeader.y);rr_audit[at+3u]=bitcast<vec4u>(vec4f(vertex.center,0));rr_audit[at+4u]=bitcast<vec4u>(vec4f(s.previousOrigin.xyz,0));s.vertexMeta.z=1u;
 s.vertexMeta.y=ei+1u;s.trace.y+=1u;s.pending.w=rg_flags;rr_lanes[lane]=s;
}
@compute @workgroup_size(32)
fn rr_operator_step(@builtin(global_invocation_id) id:vec3u) {
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 rr_lane_index=lane;var s=rr_lanes[lane];if(s.header.w!=1u||s.vertexMeta.z!=1u||s.vertexMeta.w!=0u||s.vertexMeta.y==0u){return;}rg_flags=s.pending.w;let base=rr_base(lane,&s);let at=base+RR_EVENT_OFFSET+(s.vertexMeta.y-1u)*5u;let count=rr_audit[at].x;let direct=bitcast<vec4f>(rr_audit[at+1u]).y;
 var soft=vec2f(direct,0);var hard=soft;var index=rr_audit[at+2u].w;
 for(var i=0u;i<count;i++) {
  if(index>=131072u){rg_flags|=16u;break;}let ro=index*4u;let values=bitcast<vec4f>(rr_records[ro+2u]);let scales=bitcast<vec4f>(rr_records[ro+3u]);
  let phase=utd_cis(-rr.optics.w*scales.y)/sqrt(values.x);
  soft+=utd_cmul(utd_coefficient(rr.optics.w,scales.x,values.z,values.w,scales.z,values.y,-1.0),phase);
  hard+=utd_cmul(utd_coefficient(rr.optics.w,scales.x,values.z,values.w,scales.z,values.y,1.0),phase);index=rr_records[ro].w;
 }
 if(index!=0xffffffffu){rg_flags|=16u;}let sp=dot(soft,soft);let hp=dot(hard,hard);let F=select(select((sp+hp)*0.5,hp,rr.frame.w==1u),sp,rr.frame.w==0u);
 rr_audit[at+1u]=bitcast<vec4u>(vec4f(F,direct,soft));rr_audit[at+2u].x=bitcast<u32>(hard.x);rr_audit[at+2u].y=bitcast<u32>(hard.y);
 if(!rr_ok(F)||F<0.0){rg_flags|=64u;}
 if(s.pending.y!=0u){for(var i=0u;i<s.counters.z;i++){rr_set_weight(&s,i,rr_weight(&s,i)*F);}}
 else {let oldCount=s.counters.z;if(oldCount+s.counters.w>64u){rg_flags|=16u;}else{for(var i=0u;i<s.counters.w;i++){rr_set_weight(&s,oldCount+i,rr_saved(&s,i)*F);rr_weights[rr_lane_index*48u+16u+(oldCount+i)/4u][(oldCount+i)%4u]=0u;}s.counters.z+=s.counters.w;}}
 s.pending.x=0u;s.progress.z=max(s.progress.z,s.counters.z);s.pending.w=rg_flags;rr_lanes[lane]=s;
}
@compute @workgroup_size(32)
fn rr_scatter_step(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
  rr_lane_index=lane;var s=rr_lanes[lane];if(s.header.w!=1u||s.vertexMeta.z!=1u||s.vertexMeta.w!=0u){if(s.header.w!=2u){atomicAdd(&rr_status[0],1u);}atomicOr(&rr_status[1],s.pending.w);return;}rg_flags=s.pending.w;
  rr_rng_count=s.trace.w;rr_rng_hash=s.progress.x;let oldDraws=s.trace.w;let base=rr_base(lane,&s);let vertex=rr_load_vertex(&s);let beam=rr_beam(&s);let apertureCount=s.vertexMeta.x;var done=rg_flags!=0u;var termination=7u;
  if(!done) {
    s.previousOrigin=s.origin;s.previousDirection=s.direction;s.counters.w=s.counters.z;
    for(var i=0u;i<s.counters.z;i++){rr_weights[rr_lane_index*48u+32u+i/4u][i%4u]=bitcast<u32>(rr_weight(&s,i));}
    if(vertex.hit.valid!=0u) {
      let hit=vertex.hit;let material=rg_material(hit.material);let n=select(-hit.normal,hit.normal,dot(hit.normal,beam.direction)<0.0);
      if(material.emission>0.0&&(material.twoSidedEmission!=0u||dot(hit.normal,beam.direction)<0.0)) {
        var misWeight=1.0;
        if(rr.config.x!=0u&&s.pending.z==0u&&s.previousPosition.w>0.0){let d=distance(hit.point,s.previousPosition.xyz);let cosine=abs(dot(hit.normal,beam.direction));let lightPDF=d*d/(rg_emitter_area()*max(cosine,1e-30));misWeight=rr_mis(s.previousPosition.w,lightPDF);}
        rr_add(&s,base,1u,material.emission,misWeight);
      }
      if(!(material.albedo>0.0)){done=true;termination=2u;}
      else {
        var rng=s.header.y;
        if(rr.config.x!=0u&&material.kind==0u&&s.counters.x+1u<rr.frame.z){let light=rr_light(hit.point,&rng);if(light.valid!=0u&&light.cosine>0.0){let cosine=max(0.0,dot(n,light.direction));if(cosine>0.0&&rg_visible(hit.point,light.point,1e-6)){let bsdfPDF=cosine/RR_PI;rr_add(&s,base,2u,material.albedo*bsdfPDF/light.pdf*light.emission*rr_mis(light.pdf,bsdfPDF),1.0);}}}
        var next:vec3f;var pdf=0.0;var delta=1u;
        if(material.kind==1u){next=beam.direction-n*(2.0*dot(beam.direction,n));}else{let sampled=rr_cosine(n,&rng);next=sampled.xyz;pdf=sampled.w;delta=0u;}
        if(rr_axis_requested()){
         // Exact axis reflection changes only a sign, so the stored CPU beam's
         // componentwise direction enclosure is preserved without re-normalizing.
         if(material.kind!=1u||s.intervalMeta.y>=3u){s.intervalMeta.x=0u;}
         else{var flipped=beam.direction;flipped[s.intervalMeta.y]=-flipped[s.intervalMeta.y];if(any(next!=flipped)){s.intervalMeta.x=0u;}}
        }
        rr_scatter(&s,base,select(1u,2u,material.kind==1u),next,pdf,material.albedo,1u,0xffffffffu,0u,hit.point);
        for(var i=0u;i<s.counters.z;i++){rr_set_weight(&s,i,rr_weight(&s,i)*material.albedo);rr_weights[rr_lane_index*48u+16u+i/4u][i%4u]=1u;}
        s.origin=vec4f(hit.point,rr_radius(beam,hit.distance));s.direction=vec4f(next,beam.tanAngle);
        if(apertureCount>0u){s.pending.x=1u;s.pending.y=0u;s.pendingCenter=vec4f(vertex.region.center,vertex.region.radius);s.pendingDirection=vec4f(vertex.region.direction,vertex.region.depth);s.pendingFeature=vertex.feature;}
        s.previousPosition=vec4f(hit.point,pdf);s.pending.z=delta;s.header.y=rng;s.counters.x+=1u;
      }
    } else if(apertureCount>0u) {
      s.intervalMeta.x=0u; // FSD directions have no camera-unit provenance.
      rg_set_region_feature(vertex.feature);var rng=s.header.y;let sampled=rf_sample(vertex.region,beam.direction,rr.optics.w,s.previousPosition.xyz,&rng);s.header.y=rng;
      rr_scatter(&s,base,3u,sampled.direction,sampled.pdf,sampled.weight,sampled.valid,sampled.edgeId,sampled.isDirect,sampled.point);
      if(sampled.valid==0u||!(sampled.weight>0.0)||!rr_ok(sampled.weight)){done=true;termination=3u;}
      else {
        for(var i=0u;i<s.counters.z;i++){rr_set_weight(&s,i,rr_weight(&s,i)*sampled.weight);}
        let radius=rr_radius(beam,vertex.distance);let c=clamp(dot(beam.direction,sampled.direction),-1.0,1.0);let ss=sqrt(max(0.0,1.0-c*c));
        let projected=radius*sqrt(1.0+3.0*ss*ss);let axial=radius*sqrt(1.0+3.0*c*c);
        s.origin=vec4f(vertex.center,projected+beam.tanAngle*axial);s.direction=vec4f(sampled.direction,beam.tanAngle);
        s.pending.x=1u;s.pending.y=1u;s.pendingCenter=vec4f(vertex.region.center,vertex.region.radius);s.pendingDirection=vec4f(vertex.region.direction,vertex.region.depth);s.pendingFeature=vertex.feature;
        s.previousPosition=vec4f(vertex.center,0);s.pending.z=1u;s.counters.x+=1u;
      }
    } else {
      s.intervalMeta.x=0u; // Unmodelled null-origin shift revokes the enclosure.
      s.counters.y+=1u;if(s.counters.y>64u){rg_flags|=32u;done=true;termination=7u;}
      else {let shift=vertex.distance+max(1e-6,vertex.region.radius*0.001);s.origin=vec4f(beam.origin+beam.direction*shift,rr_radius(beam,shift));}
    }
  }
  if(s.counters.x>=rr.frame.z){done=true;termination=4u;}if(rg_flags!=0u){done=true;termination=7u;}
  rr_flush_rng(&s,base,oldDraws);s.pending.w=rg_flags;s.vertexMeta.z=0u;s.traversal=vec4f(0);if(done){rr_finish(&s,base,termination);}rr_lanes[lane]=s;
  if(s.header.w!=2u){atomicAdd(&rr_status[0],1u);}atomicOr(&rr_status[1],rg_flags);
}

// Legacy entrypoints preserve their original statically accessed bindings.
// WebGPU auto-layout includes storage bindings even behind a false override;
// only the explicitly compiled anchored entrypoint aliases access binding 9.
@compute @workgroup_size(32)
fn rr_init(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=rr.tile.z*rr.tile.w){return;}
  var s:RRLane;s.header=vec4u(rr_initial[id.x].xy,rr.run.x,0u);rr_lanes[id.x]=s;
}
@compute @workgroup_size(32)
fn rr_axis_prepare_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 var prepared:RAQueryState;
 if(rr.config.z==4u){prepared=ra_prepare_unit(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz,select(0u,1u,s.intervalMeta.x==2u));}
 else{prepared=ra_prepare(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz);}
 rr_axis_states[lane].prepared=prepared;rr_axis_states[lane].progress=ra_scan_begin(rr_load_vertex(&s).hit,prepared);
}
@compute @workgroup_size(32)
fn rr_axis_plane_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 let s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 rr_axis_states[lane].plane=ra_scan_plane(s.origin.xyz,s.intervalOriginError.xyz,rr_axis_states[lane].prepared,rr_axis_states[lane].progress);
}
@compute @workgroup_size(32)
fn rr_axis_finish_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 let vertex=rr_load_vertex(&s);let state=rr_axis_states[lane];
 let enclosed=ra_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,vertex.hit,state.prepared,ra_scan_end(vertex.hit,state.progress));
 s.intervalMeta.z+=1u;s.intervalMeta.w=enclosed.reason;s.vertexMeta.w=0u;
 rg_flags=s.pending.w;rr_rng_count=s.trace.w;rr_rng_hash=s.progress.x;let base=rr_base(lane,&s);
 if(enclosed.accepted==0u){rg_flags|=268435458u;s.pending.w=rg_flags;rr_finish(&s,base,7u);}
 else if(vertex.valid==0u){rr_finish(&s,base,1u);}
 else if(vertex.hit.valid!=0u){s.intervalOriginError=vec4f(enclosed.originError,enclosed.planeValue);s.intervalDirectionError.w=enclosed.planeResidual;s.intervalMeta.y=enclosed.planeAxis;}
 rr_lanes[lane]=s;
}

// Camera-only entrypoints keep extra private resource bindings out of all legacy layouts.
fn rr_camera_anchor_owned(s:RRLane,state:RRAnchorState,lane:u32)->bool {
 return rr_camera_owned(s,lane)&&rr_anchor_owned(s,state);
}
@compute @workgroup_size(32)
fn rr_init_camera(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=rr.tile.z*rr.tile.w){return;}
  var s:RRLane;s.header=vec4u(rr_initial[id.x].xy,rr.run.x,0u);rr_lanes[id.x]=s;
  if(RR_ANCHOR_ENABLED){rr_anchor_states[id.x].proofMeta=vec4u(0u);rr_anchor_states[id.x].checked=0u;}
 if(RR_CAMERA_ENABLED){rr_camera_states[id.x].proofMeta=vec4u(0u);rr_camera_states[id.x].camera.valid=0u;}
}
@compute @workgroup_size(32)
fn rr_camera_init_step(@builtin(global_invocation_id) id:vec3u){
 if(!RR_ANCHOR_ENABLED||!rr_anchor_requested()||id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 let s=rr_lanes[lane];if(!rr_axis_pending(s)||s.counters.x!=0u){return;}
 rr_anchor_states[lane].proofMeta=vec4u(0u);rr_anchor_states[lane].checked=0u;
 if(s.intervalMeta.x!=rr_anchor_marker()||s.counters.y!=0u){return;}
 var rng=s.progress.y;rr_rng_count=0u;rr_rng_hash=2166136261u;let word0=rr_word(&rng);let word1=rr_word(&rng);
 ra_bad=false;var raw:RAVec3;
 if(RR_CAMERA_ENABLED&&rr.config.z==10u){
  rr_camera_states[lane].proofMeta=vec4u(0u);rr_camera_states[lane].camera.valid=0u;
  let camera=ra_camera_initialize(rr_camera_parameters,s.header.x,word0,word1);
  if(ra_bad||camera.valid!=1u){return;}
  rr_camera_states[lane].camera=camera;rr_camera_states[lane].proofMeta=vec4u(1u,s.header.z,s.progress.y,5u);raw=camera.raw;
 }else{raw=rri_camera_raw(s.header.x,word0,word1);}
 if(ra_bad){return;}
 var anchor=ra_anchor_initialize(raw);if((RR_TIGHT_ENABLED&&rr.config.z==8u)||(RR_CAMERA_ENABLED&&rr.config.z==10u)){anchor=ra_anchor_tight_refine_norm(anchor);}rr_anchor_states[lane].anchor=anchor;
 rr_anchor_states[lane].proofMeta=vec4u(anchor.valid,s.header.z,s.progress.y,0u);
}
@compute @workgroup_size(32)
fn rr_camera_prepare_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 var prepared:RAQueryState;
 if(RR_ANCHOR_ENABLED){rr_anchor_states[lane].checked=0u;}
 if(RR_ANCHOR_ENABLED&&rr_anchor_requested()){prepared=ra_prepare_unit(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz,select(0u,1u,rr_camera_anchor_owned(s,rr_anchor_states[lane],lane)));}
 else if(rr.config.z==4u){prepared=ra_prepare_unit(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz,select(0u,1u,s.intervalMeta.x==2u));}
 else{prepared=ra_prepare(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,s.intervalDirectionError.xyz);}
 rr_axis_states[lane].prepared=prepared;rr_axis_states[lane].progress=ra_scan_begin(rr_load_vertex(&s).hit,prepared);
}
@compute @workgroup_size(32)
fn rr_camera_plane_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 let s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 let state=rr_axis_states[lane];
 if(RR_ANCHOR_ENABLED&&rr_anchor_requested()){
  if(state.progress.valid!=1u||state.progress.cursor>=rg_triangle_count()||!rr_camera_anchor_owned(s,rr_anchor_states[lane],lane)){rr_axis_states[lane].plane=ra_plane_unknown(state.progress.cursor,RA_UNSUPPORTED);return;}
  if(RR_CAMERA_ENABLED&&rr.config.z==10u){
   rr_axis_states[lane].plane=ra_camera_anchor_plane_query(rg_triangle(state.progress.cursor),state.progress.cursor,s.origin.xyz,s.intervalOriginError.xyz,state.prepared,rr_anchor_states[lane].anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w),rr_camera_states[lane].camera);
  }else if(RR_TIGHT_ENABLED&&rr.config.z==8u){
   rr_axis_states[lane].plane=ra_anchor_tight_plane_query(rg_triangle(state.progress.cursor),state.progress.cursor,s.origin.xyz,s.intervalOriginError.xyz,state.prepared,rr_anchor_states[lane].anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w));
  }else{rr_axis_states[lane].plane=ra_anchor_plane_query(rg_triangle(state.progress.cursor),state.progress.cursor,s.origin.xyz,s.intervalOriginError.xyz,state.prepared,rr_anchor_states[lane].anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w));}
 }else{rr_axis_states[lane].plane=ra_scan_plane(s.origin.xyz,s.intervalOriginError.xyz,state.prepared,state.progress);}
}
@compute @workgroup_size(32)
fn rr_camera_point_step(@builtin(global_invocation_id) id:vec3u){
 if(!RR_ANCHOR_ENABLED||!rr_anchor_requested()||id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}let state=rr_axis_states[lane];
 let anchorState=rr_anchor_states[lane];
 var enclosed=RAAnchorResult(ra_failure(RA_UNSUPPORTED),anchorState.anchor);
 if(rr_camera_anchor_owned(s,anchorState,lane)){
  let candidate=rr_load_vertex(&s).hit;
  if(RR_CAMERA_ENABLED&&rr.config.z==10u){
   enclosed=ra_camera_anchor_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,candidate,state.prepared,ra_scan_end(candidate,state.progress),anchorState.anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w),rr_camera_states[lane].camera);
  }else if(RR_TIGHT_ENABLED&&rr.config.z==8u){
   enclosed=ra_anchor_tight_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,candidate,state.prepared,ra_scan_end(candidate,state.progress),anchorState.anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w));
  }else{enclosed=ra_anchor_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,candidate,state.prepared,ra_scan_end(candidate,state.progress),anchorState.anchor,rr.cameraOrigin.xyz,vec3f(rr.cameraOrigin.w));}
  if(enclosed.result.accepted==1u&&candidate.valid==1u&&rg_material(candidate.material).kind==1u){
   enclosed.anchor=ra_anchor_reflect(enclosed.anchor);
   if(enclosed.anchor.valid!=1u){enclosed.result=ra_failure(RA_UNSUPPORTED);}
  }
 }
 rr_anchor_states[lane].anchor=enclosed.anchor;rr_anchor_states[lane].result=enclosed.result;rr_anchor_states[lane].checked=1u;
}
@compute @workgroup_size(32)
fn rr_camera_finish_step(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=rr.batch.y){return;}let lane=rr.batch.x+id.x;if(lane>=rr.tile.z*rr.tile.w){return;}
 var s=rr_lanes[lane];if(!rr_axis_pending(s)){return;}
 let vertex=rr_load_vertex(&s);let state=rr_axis_states[lane];
 var enclosed:RAResult;
 if(RR_ANCHOR_ENABLED&&rr_anchor_requested()){
  let anchorState=rr_anchor_states[lane];enclosed=anchorState.result;
  if(anchorState.checked!=1u||!rr_camera_anchor_owned(s,anchorState,lane)){enclosed=ra_failure(RA_UNSUPPORTED);}
  rr_anchor_states[lane].checked=0u;
  if(enclosed.accepted==0u){rr_anchor_states[lane].proofMeta.x=0u;}
  else if(vertex.hit.valid!=0u){rr_anchor_states[lane].proofMeta.w+=1u;}
 }else{enclosed=ra_finish(s.origin.xyz,s.intervalOriginError.xyz,s.direction.xyz,vertex.hit,state.prepared,ra_scan_end(vertex.hit,state.progress));}
 s.intervalMeta.z+=1u;s.intervalMeta.w=enclosed.reason;s.vertexMeta.w=0u;
 rg_flags=s.pending.w;rr_rng_count=s.trace.w;rr_rng_hash=s.progress.x;let base=rr_base(lane,&s);
 if(enclosed.accepted==0u){rg_flags|=268435458u;s.pending.w=rg_flags;rr_finish(&s,base,7u);}
 else if(vertex.valid==0u){rr_finish(&s,base,1u);}
 else if(vertex.hit.valid!=0u){s.intervalOriginError=vec4f(enclosed.originError,enclosed.planeValue);s.intervalDirectionError.w=enclosed.planeResidual;s.intervalMeta.y=enclosed.planeAxis;}
 rr_lanes[lane]=s;
}
