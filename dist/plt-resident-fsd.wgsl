// Resident local FSD include. Requires unchanged utd-functions.wgsl, RG geometry,
// rr_word(ptr<function,u32>) and rr_float(ptr<function,u32>) from the walker.
// All geometric points use the packer's shared rebased coordinate system.
// No edge array/cap: every operation rescans all stable scene wedges.
// This is native local-plane UTD and a Keller-angle proposal with auxiliary
// edge position. It is NOT an absolute-energy/full-wave validation.
const RF_TAU:f32 = 6.28318530717958647692;
const RF_MIN_SIN:f32 = 0.001;
const RF_TANGENT:f32 = 1e-12;
const RF_EPS:f32 = 1.1920928955078125e-7;
const RF_MAX_PHASE:f32 = 100000.0;
// rg_flags bits 1/2 belong to geometry. FSD owns 4 (invalid phase/math), 8
// (Box-Muller exact zero), while predicate uncertainty uses shared bit 2.
// Diagnostics: 1 endpoint, 2 grazing, 4 coefficient, 8 low kL,
// 16 surface grazing, 32 physical endpoint neighborhood, 64 region truncation,
// 128 reflected seam approximation, 256 phase, 512 missing previous cone.
struct RFOriented { edge:RGEdge, nff:vec3f, tff:vec3f, nbf:vec3f, valid:u32 }
struct RFSaddle { valid:u32, point:vec3f, wi:vec3f, wo:vec3f, ri:f32, ro:f32,
  axial:f32, sinBeta:f32, fermatResidual:f32, atEndpoint:u32 }
struct RFSample { valid:u32, direction:vec3f, point:vec3f, weight:f32, pdf:f32,
  edgeId:u32, count:u32, isDirect:u32, selected:u32, diagnostics:u32 }
struct RFRecord { valid:u32, point:vec3f, ri:f32, ro:f32, sinBeta:f32,
  phiI:f32, phiO:f32, n:f32, phaseExcess:f32, L:f32, fermatResidual:f32,
  edgeId:u32, diagnostics:u32, excluded:u32, angularSeam:u32,
  endpointFresnelWidth:f32 }
struct RFEvent { F:f32, edgeCount:u32, soft:vec2f, hard:vec2f,
  directAmplitude:f32, directBoundary:u32, diagnostics:u32,
  endpointExclusions:u32, invalidEdges:u32, edgeHash:u32 }
fn rf_finite(x:f32)->bool { return abs(x)<=3.402823466e38; }
fn rf_finite3(x:vec3f)->bool { return all(abs(x)<=vec3f(3.402823466e38)); }
// Deterministic f32 trigonometry for stochastic geometry. Device intrinsic
// sin/cos accuracy is insufficient for the fixed 2e-5 + 3e-6 geometry gate.
// Explicit fma retains the small residual when subtracting many pi/2 periods.
// Taylor degree 13/12 on [-pi/4,pi/4] has truncation below f32 rounding.
fn rf_sincos(theta:f32)->vec2f {
  if(!rf_finite(theta)||abs(theta)>RF_MAX_PHASE){rg_flags|=4u;return vec2f(1.0,0.0);}
  let quadrant=i32(round(theta*0.63661977236758134308));let q=f32(quadrant);
  let coarse=fma(-q,1.5703125,theta);
  // Both q*201 and q*63 stay exactly representable for |theta|<=100000.
  // A third split also bounds error if a backend lowers fma unfused.
  let medium=fma(-q,0.00048065185546875,coarse);
  let x=fma(-q,0.0000031749394278692313217,medium);let x2=x*x;
  var sp:f32=1.60590438368216145994e-10;
  sp=fma(sp,x2,-2.50521083854417187751e-8);
  sp=fma(sp,x2,2.75573192239858925110e-6);
  sp=fma(sp,x2,-1.98412698412698412527e-4);
  sp=fma(sp,x2,8.33333333333333321769e-3);
  sp=fma(sp,x2,-1.66666666666666657415e-1);
  let sine=x*fma(sp,x2,1.0);
  var cp:f32=2.08767569878680989792e-9;
  cp=fma(cp,x2,-2.75573192239858882710e-7);
  cp=fma(cp,x2,2.48015873015873015844e-5);
  cp=fma(cp,x2,-1.38888888888888894189e-3);
  cp=fma(cp,x2,4.16666666666666643537e-2);
  cp=fma(cp,x2,-0.5);
  let cosine=fma(cp,x2,1.0);
  switch(quadrant&3){
    case 0:{return vec2f(cosine,sine);}
    case 1:{return vec2f(-sine,cosine);}
    case 2:{return vec2f(-cosine,-sine);}
    default:{return vec2f(sine,-cosine);}
  }
}
fn rf_wrap(x:f32)->f32 { let a=x-RF_TAU*floor(x/RF_TAU); return select(a,0.0,a>=RF_TAU); }
fn rf_signed(x:f32)->f32 { let a=rf_wrap(x);return select(a,a-RF_TAU,a>UTD_PI); }
fn rf_unit(x:vec3f)->vec3f {
  let l=length(x); if(!(l>0.0)||!rf_finite(l)){rg_flags|=4u;return vec3f(0.0,0.0,1.0);}return x/l;
}
fn rf_positive(n:vec3f,d:vec3f)->bool {
  let value=dot(n,d);let bound=8.0*RF_EPS*dot(abs(n),abs(d));
  if(bound>0.0&&abs(value-RF_TANGENT)<=bound){rg_flags|=2u;}
  return value>RF_TANGENT;
}
fn rf_outside(edge:RFOriented,d:vec3f)->bool {
  if(rf_positive(edge.nff,d)){return true;}return rf_positive(edge.nbf,d);
}
fn rf_orient(index:u32,region:RGRegion,incomingDir:vec3f)->RFOriented {
  var out:RFOriented;out.edge=rg_clip_active(index,region);
  if(out.edge.valid==0u){return out;}
  let wi=-incomingDir;let first=rf_positive(out.edge.n1,wi);
  if(!first&&!rf_positive(out.edge.n2,wi)){return out;}
  out.nff=select(out.edge.n2,out.edge.n1,first);
  out.nbf=select(out.edge.n1,out.edge.n2,first);
  out.tff=select(out.edge.t2,out.edge.t1,first);
  out.edge.e=rf_unit(cross(out.nff,out.tff));
  if(!(out.edge.l>0.0)||out.edge.alpha<0.0||out.edge.alpha>UTD_PI){rg_flags|=4u;return out;}
  out.valid=1u;return out;
}
fn rf_count(region:RGRegion,incomingDir:vec3f)->u32 {
  var count=0u;let incoming=rf_unit(incomingDir);
  for(var i=0u;i<rg_wedge_count();i++){count+=rf_orient(i,region,incoming).valid;}
  return count;
}
fn rf_wrapped_normal(angle:f32,sigma:f32)->f32 {
  if(!(sigma>0.0)||!rf_finite(sigma)||!rf_finite(angle)){rg_flags|=4u;return 0.0;}
  let x=rf_signed(angle);
  if(sigma<1.0){
    let limit=i32(ceil(9.0*sigma/RF_TAU))+1;var sum=0.0;
    for(var j=-limit;j<=limit;j++){let z=(x+f32(j)*RF_TAU)/sigma;sum+=exp(-0.5*z*z);}
    return sum/(sqrt(RF_TAU)*sigma);
  }
  var sum=1.0;
  for(var j=1u;j<100u;j++){
    let js=f32(j)*sigma;let amplitude=exp(-0.5*js*js);
    if(amplitude<1e-17){break;}sum+=2.0*amplitude*rf_sincos(f32(j)*x).x;
  }
  return max(0.0,sum/RF_TAU);
}
fn rf_direction_saddle(edge:RGEdge,source:vec3f,direction:vec3f)->RFSaddle {
  var out:RFSaddle;out.wo=rf_unit(direction);
  let s=source-edge.v;let sa=dot(s,edge.e);let radial=length(s-edge.e*sa);
  let cosine=dot(out.wo,edge.e);let sine=sqrt(max(0.0,1.0-cosine*cosine));
  if(abs(sine-RF_MIN_SIN)<=8.0*RF_EPS){rg_flags|=2u;}
  let tolerance=2.842170943040401e-14*max(1.0,max(max(abs(rg_world(source).x),max(abs(rg_world(source).y),abs(rg_world(source).z))),max(abs(rg_world(edge.v).x),max(abs(rg_world(edge.v).y),abs(rg_world(edge.v).z)))));
  if(sine<RF_MIN_SIN||radial<=tolerance){return out;}
  let axial=sa+radial*cosine/sine;
  let margin=abs(axial)-edge.l*0.5;
  if(abs(margin)<=16.0*RF_EPS*max(1.0,max(abs(axial),edge.l))){rg_flags|=2u;}
  if(margin>0.0){return out;}
  out.point=edge.v+edge.e*axial;out.ri=distance(source,out.point);
  if(!(out.ri>0.0)){return out;}
  out.wi=(source-out.point)/out.ri;out.axial=axial;out.sinBeta=sine;
  out.fermatResidual=dot(out.wi,edge.e)+cosine;out.valid=1u;return out;
}
fn rf_stationary(edge:RGEdge,source:vec3f,destination:vec3f)->RFSaddle {
  var out:RFSaddle;
  let s=source-edge.v;let d=destination-edge.v;let e=rf_unit(edge.e);
  let sa=dot(s,e);let da=dot(d,e);let sr=length(s-e*sa);let dr=length(d-e*da);
  let tolerance=1.4210854715202004e-14*max(1.0,max(edge.l,max(length(s),max(length(d),length(rg_world(edge.v))))));
  if(sr+dr<=tolerance){return out;}
  var axial=sa+(da-sa)*(sr/(sr+dr));let endpoint=abs(axial)-edge.l*0.5;
  if(abs(endpoint)<=16.0*RF_EPS*max(1.0,max(abs(axial),edge.l))){rg_flags|=2u;}
  if(axial< -edge.l*0.5-tolerance||axial>edge.l*0.5+tolerance){return out;}
  axial=clamp(axial,-edge.l*0.5,edge.l*0.5);out.point=edge.v+e*axial;
  let incident=source-out.point;let outgoing=destination-out.point;
  out.ri=length(incident);out.ro=length(outgoing);
  if(out.ri<=tolerance||out.ro<=tolerance){return out;}
  out.wi=incident/out.ri;out.wo=outgoing/out.ro;out.axial=axial;
  let cosine=dot(out.wi,e);out.sinBeta=sqrt(max(0.0,1.0-cosine*cosine));
  out.atEndpoint=select(0u,1u,abs(abs(axial)-edge.l*0.5)<=tolerance);
  out.fermatResidual=dot(out.wi,e)+dot(out.wo,e);out.valid=1u;return out;
}
fn rf_angles(edge:RFOriented,wi:vec3f,wo:vec3f)->vec2f {
  return vec2f(rf_wrap(utd_atan2_axes(dot(wi,edge.nff),dot(wi,edge.tff))),
    rf_wrap(utd_atan2_axes(dot(wo,edge.nff),dot(wo,edge.tff))));
}
fn rf_pdf(region:RGRegion,incomingDir:vec3f,k:f32,source:vec3f,direction:vec3f)->f32 {
  if(!(k>0.0)||!rf_finite(k)){rg_flags|=4u;return 0.0;}var count=0u;var density=0.0;let incoming=rf_unit(incomingDir);
  for(var i=0u;i<rg_wedge_count();i++){
    let edge=rf_orient(i,region,incoming);if(edge.valid==0u){continue;}count++;
    let p=rf_direction_saddle(edge.edge,source,direction);
    if(p.valid==0u){continue;}if(!rf_outside(edge,p.wi)||!rf_outside(edge,p.wo)){continue;}
    let angles=rf_angles(edge,p.wi,p.wo);let sigma=sqrt(45.0/(k*p.ri));
    density+=0.5*(rf_wrapped_normal(angles.y-(UTD_PI+angles.x),sigma)+rf_wrapped_normal(angles.y-(UTD_PI-angles.x),sigma));
  }
  return density/f32(count+1u);
}
// Exact upper 32 bits of a 32x32 product, without f32 branch rounding.
fn rf_discrete(word:u32,count:u32)->u32 {
  let a=word&65535u;let b=word>>16u;let c=count&65535u;let d=count>>16u;
  let ac=a*c;let bc=b*c+(ac>>16u);let ad=a*d+(bc&65535u);
  return b*d+(bc>>16u)+(ad>>16u);
}
fn rf_sample(region:RGRegion,incomingDir:vec3f,k:f32,source:vec3f,rng:ptr<function,u32>)->RFSample {
  var out:RFSample; if(!(k>0.0)||!rf_finite(k)){rg_flags|=4u;return out;}out.direction=vec3f(0.0,0.0,1.0);out.point=region.center;out.edgeId=0xffffffffu;
  let incoming=rf_unit(incomingDir);out.count=rf_count(region,incoming);
  out.selected=rr_choice(rng,out.count+1u);
  if(out.selected==out.count){
    let delta=region.center-source;let dist=length(delta);
    let tolerance=2.842170943040401e-14*max(1.0,max(max(abs(rg_world(source).x),max(abs(rg_world(source).y),abs(rg_world(source).z))),max(abs(rg_world(region.center).x),max(abs(rg_world(region.center).y),abs(rg_world(region.center).z)))));
    if(dist<=tolerance){out.diagnostics=1u;return out;}
    out.direction=delta/dist;out.valid=1u;out.isDirect=1u;out.weight=f32(out.count+1u);out.pdf=1.0/out.weight;return out;
  }
  var ordinal=0u;var selected:RFOriented;
  for(var i=0u;i<rg_wedge_count();i++){
    let edge=rf_orient(i,region,incoming);if(edge.valid==0u){continue;}
    if(ordinal==out.selected){selected=edge;break;}ordinal++;
  }
  if(selected.valid==0u){rg_flags|=4u;return out;}
  let coordinate=rr_float(rng)-0.5;let edge=selected.edge;out.edgeId=edge.edgeId;
  out.point=edge.v+edge.e*(coordinate*edge.l);let ri=distance(source,out.point);
  if(!(ri>0.0)){out.diagnostics=2u;return out;}let wi=(source-out.point)/ri;
  if(!rf_outside(selected,wi)){out.diagnostics=2u;return out;}
  let cosine=dot(wi,edge.e);let sine=sqrt(max(0.0,1.0-cosine*cosine));
  if(abs(sine-RF_MIN_SIN)<=8.0*RF_EPS){rg_flags|=2u;}
  if(sine<RF_MIN_SIN){out.diagnostics=3u;return out;}
  let phiI=utd_atan2_axes(dot(wi,selected.nff),dot(wi,selected.tff));let sigma=sqrt(45.0/(k*ri));
  let u=rr_float(rng);let v=rr_float(rng);let branch=rr_choice(rng,2u);
  if(u==0.0){out.diagnostics=6u;rg_flags|=8u;return out;}
  let gaussian=sqrt(-2.0*log(u))*rf_sincos(RF_TAU*v).x;
  let phiO=select(UTD_PI-phiI,UTD_PI+phiI,branch==0u)+sigma*gaussian;
  let outgoingTrig=rf_sincos(phiO);
  out.direction=rf_unit((selected.tff*outgoingTrig.x+selected.nff*outgoingTrig.y)*sine-edge.e*cosine);
  if(!rf_outside(selected,out.direction)){out.diagnostics=4u;return out;}
  out.pdf=rf_pdf(region,incoming,k,source,out.direction);
  if(!(out.pdf>0.0)||!rf_finite(1.0/out.pdf)){out.diagnostics=5u;return out;}
  out.weight=1.0/out.pdf;out.valid=1u;return out;
}
fn rf_excess(source:vec3f,point:vec3f,destination:vec3f)->f32 {
  let a=point-source;let b=destination-point;let ra=length(a);let rb=length(b);let r=distance(source,destination);
  if(!(ra>0.0&&rb>0.0)){return 0.0;}let du=a/ra-b/rb;return (ra/(ra+rb+r))*rb*dot(du,du);
}
// Predicate guard does not change the 2e-6 seam. Unresolved classifications
// invalidate the entire walker job through rg_flags, never widen the seam.
fn rf_seam(delta:f32)->bool {
  if(abs(abs(delta)-UTD_BOUNDARY_EPS)<=8.0*RF_EPS){rg_flags|=2u;}
  return abs(delta)<=UTD_BOUNDARY_EPS;
}
fn rf_coefficient_seams(phiI:f32,phiO:f32,n:f32) {
  for(var j=0u;j<4u;j++){
    let beta=select(phiO-phiI,phiO+phiI,j>=2u);let side=select(1.0,-1.0,(j&1u)!=0u);
    let nearest=floor((beta+side*UTD_PI)/(2.0*UTD_PI*n)+0.5);
    let delta=beta-(2.0*UTD_PI*n*nearest-side*UTD_PI);let seam=rf_seam(delta);
  }
}
fn rf_record(region:RGRegion,incomingDir:vec3f,k:f32,source:vec3f,destination:vec3f,index:u32)->RFRecord {
  var out:RFRecord;if(!(k>0.0)||!rf_finite(k)){rg_flags|=4u;return out;}let oriented=rf_orient(index,region,rf_unit(incomingDir));
  if(oriented.valid==0u){return out;}let edge=oriented.edge;out.edgeId=edge.edgeId;
  let hit=rf_stationary(edge,source,destination);
  if(hit.valid==0u){out.excluded=1u;return out;}
  if(hit.atEndpoint!=0u){out.excluded=2u;out.diagnostics|=1u;return out;}
  if(!rf_outside(oriented,hit.wi)||!rf_outside(oriented,hit.wo)){return out;}
  if(!rg_visible(source,hit.point,rg_endpoint_epsilon(source,hit.point))||!rg_visible(hit.point,destination,rg_endpoint_epsilon(hit.point,destination))){return out;}
  let angles=rf_angles(oriented,hit.wi,hit.wo);out.phiI=angles.x;out.phiO=angles.y;out.n=2.0-edge.alpha/UTD_PI;
  if(out.phiI>out.n*UTD_PI||out.phiO>out.n*UTD_PI){return out;}
  if(abs(hit.sinBeta-RF_MIN_SIN)<=8.0*RF_EPS){rg_flags|=2u;}
  if(hit.sinBeta<RF_MIN_SIN){out.excluded=3u;out.diagnostics|=2u;return out;}
  out.L=hit.ro*hit.sinBeta*hit.sinBeta;
  if(!(out.L>0.0)||!rf_finite(k*out.L)){out.excluded=4u;out.diagnostics|=4u;rg_flags|=4u;return out;}
  if(k*out.L<10.0){out.diagnostics|=8u;}
  if(min(min(out.phiI,out.phiO),min(out.n*UTD_PI-out.phiI,out.n*UTD_PI-out.phiO))<RF_MIN_SIN){out.diagnostics|=16u;}
  let width=sqrt(hit.ro/k)/hit.sinBeta;out.endpointFresnelWidth=width;
  let physicalMargin=min(distance(hit.point,edge.originalA),distance(hit.point,edge.originalB));
  let clippedMargin=min(distance(hit.point,edge.a),distance(hit.point,edge.b));
  if(physicalMargin<2.0*width){out.diagnostics|=32u;}
  if((edge.clippedStart!=0u||edge.clippedEnd!=0u)&&clippedMargin<2.0*width){out.diagnostics|=64u;}
  out.angularSeam=select(0u,1u,rf_seam(abs(out.phiO-out.phiI)-UTD_PI));
  if(rf_seam(out.phiO+out.phiI-UTD_PI)||rf_seam(out.phiO+out.phiI-(2.0*out.n-1.0)*UTD_PI)){out.diagnostics|=128u;}
  rf_coefficient_seams(out.phiI,out.phiO,out.n);
  out.phaseExcess=rf_excess(source,hit.point,destination);
  if(!rf_finite(k*out.phaseExcess)||abs(k*out.phaseExcess)>RF_MAX_PHASE){out.excluded=5u;out.diagnostics|=256u;rg_flags|=4u;return out;}
  out.point=hit.point;out.ri=hit.ri;out.ro=hit.ro;out.sinBeta=hit.sinBeta;out.fermatResidual=hit.fermatResidual;out.valid=1u;return out;
}
fn rf_evaluate(region:RGRegion,incomingDir:vec3f,k:f32,boundary:u32,source:vec3f,destination:vec3f,previousBeam:RGBeam)->RFEvent {
  var out:RFEvent;out.edgeHash=2166136261u;if(!(k>0.0)||!rf_finite(k)||boundary>2u){rg_flags|=4u;return out;}
  let contains=rg_cone_contains(previousBeam,destination,1e-10);
  if(contains&&rg_visible(source,destination,rg_endpoint_epsilon(source,destination))){out.directAmplitude=1.0;}
  // First streaming scan establishes diagnostics and the direct half weight.
  // Second scan preserves CPU direct-first, stable-wedge accumulation order.
  for(var i=0u;i<rg_wedge_count();i++){
    let record=rf_record(region,incomingDir,k,source,destination,i);out.diagnostics|=record.diagnostics;
    if(record.excluded==1u||record.excluded==2u){out.endpointExclusions++;}
    else if(record.excluded!=0u){out.invalidEdges++;}
    if(record.valid==0u){continue;}out.edgeCount++;
    out.edgeHash=(out.edgeHash^record.edgeId)*16777619u;
    if(contains&&record.angularSeam!=0u){out.directBoundary=1u;}
  }
  if(out.directBoundary!=0u){out.directAmplitude=0.5;}
  out.soft=vec2f(out.directAmplitude,0.0);out.hard=vec2f(out.directAmplitude,0.0);
  for(var i=0u;i<rg_wedge_count();i++){
    let record=rf_record(region,incomingDir,k,source,destination,i);
    if(record.valid==0u){continue;}
    let phase=utd_cis(-k*record.phaseExcess)/sqrt(record.ro);
    out.soft+=utd_cmul(utd_coefficient(k,record.n,record.phiI,record.phiO,record.L,record.sinBeta,-1.0),phase);
    out.hard+=utd_cmul(utd_coefficient(k,record.n,record.phiI,record.phiO,record.L,record.sinBeta,1.0),phase);
  }
  let softPower=dot(out.soft,out.soft);let hardPower=dot(out.hard,out.hard);
  out.F=select(select((softPower+hardPower)*0.5,hardPower,boundary==1u),softPower,boundary==0u);
  if(!rf_finite(out.F)||!all(abs(out.soft)<=vec2f(3.402823466e38))||!all(abs(out.hard)<=vec2f(3.402823466e38))){rg_flags|=4u;}
  return out;
}
