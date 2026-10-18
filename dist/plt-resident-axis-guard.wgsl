// Bounded axis-ray stability guard. Experimental; full CPU shadow is mandatory.
// Requires plt-resident-interval.wgsl and plt-resident-geometry.wgsl.
// This is a complete scan of at most 64 triangles, independent of GPU BVH.
// It is NOT an unchecked admission lane or a universal CPU/WGSL certification.
//
// Host contract (rg_data[7].z != 0): every original triangle is axis-aligned;
// source/rebased vertices and rebase are multiples of 2^-16 in +/-256;
// Nonzero queried axis components must be bounded away from zero by 2^-12.
// source normals are exactly signed coordinate axes; source/world/local and
// camera coordinates have magnitude <= 256; minimum feature >= 2^-12;
// all source edge differences/triangle areas are finite normal or exact zero.
// rg_data[7].w is an UPWARD bound 1 <= kappa <= 16, where for each triangle
// A=e1_i*e2_j-e1_j*e2_i, B=abs(e1_i*e2_j)+abs(e1_j*e2_i), kappa=B/abs(A),
// using the CPU's immutable source-edge values. rg_f(6).w is zero input error.
//
// Reference arithmetic contract: binary64, normal finite intermediates or
// exact zero, round-to-nearest basic arithmetic; hypot3 relative error <=16u64,
// u64=2^-53. Only the unit-query proof below additionally covers underflow
// of final CPU normalization components; triangle/point proofs still require
// normal-or-zero intermediates. ECMAScript does NOT guarantee hypot accuracy:
// https://tc39.es/ecma262/#sec-math.hypot . The full independent shadow remains
// the decisive check of this assumption and of the CPU's own BVH behavior.
//
// Origins use actual GPU centers plus CPU deviation radii. Never materialize
// center +/- radius before subtraction from a triangle: that would destroy a
// tiny CPU residual about the previous exact surface plane. No face is skipped.
// A previous coplanar face is excluded only when its full t interval is below
// the CPU's physical near distance. Stored CPU beam direction and its freshly
// normalized CPU query direction are distinct. Exact axis reflection preserves
// the component error radii; callers must reject all other continuations.
//
// For an axis-normal triangle, det=-Q_k*A*(1+delta_d), numerator=S_k*A*
// (1+delta_n), with abs(delta_d),abs(delta_n)<=gamma3(u64)*kappa. Subtraction
// and division rounding give relative t error <=32*u64*kappa (kappa<=16).
// The unsnapped CPU hit-plane residual is conservatively bounded by
// 64*u64*kappa*abs(plane-O_k)+4*u64*512. Only the GPU point is snapped.
// U,V use a projected formula with explicit bounded-domain CPU64 error.
// CPU64 correctly rounded basic operations stay inside representable f32
// outward endpoints. The real-axis t interval is enlarged for CPU64 rounding.

const RA_UNSUPPORTED: u32 = 1u;
const RA_INTERVAL: u32 = 2u;
const RA_DETERMINANT: u32 = 4u;
const RA_BARYCENTRIC: u32 = 8u;
const RA_NEAR_FAR: u32 = 16u;
const RA_ORDER: u32 = 32u;
const RA_CANDIDATE: u32 = 64u;
const RA_NUMERIC: u32 = 128u;
// Diagnostic sub-bits preserve the common numeric rejection bit.
const RA_NUMERIC_DIRECTION: u32 = 256u;
const RA_NUMERIC_DISTANCE: u32 = 512u;
const RA_NUMERIC_POINT: u32 = 1024u;
struct RAResult {
 accepted: u32, reason: u32, distance: RI, originError: vec3f,
 planeAxis: u32, planeValue: f32, planeResidual: f32,
}
struct RATriangleResult {
 // 0 definite miss, 1 definite hit, 2 unknown. Unknown never means empty.
 classification: u32, reason: u32, axis: u32, distance: RAI, planeDelta: RAI,
}
// Compact RA-only backend: endpoints zero or normal in [2^-100,2^48].
// Inputs are checked once; each arithmetic result is integer-stepped before
// reuse. Sticky failure means UNKNOWN. Add cannot overflow; mul/div exponent
// preflights run before evaluation. Unproved zero/underflow is never accepted.
struct RAI { lo:f32, hi:f32 }
struct RAVec3 { x:RAI, y:RAI, z:RAI }
var<private> ra_bad:bool;
fn ra_supported(bits:u32)->bool {let m=bits&0x7fffffffu;return m==0u||(m>=0x0d800000u&&m<=0x57800000u);}
fn ra_iunknown(reason:u32)->RAI {ra_bad=true;return RAI(0.0,0.0);}
fn ra_ibounds(lo:u32,hi:u32)->RAI {
 let bitsOK=ra_supported(lo)&&ra_supported(hi);
 let l=bitcast<f32>(select(0u,lo,bitsOK));let h=bitcast<f32>(select(0u,hi,bitsOK));let good=bitsOK&&(l<=h);
 ra_bad=select(true,ra_bad,good);return RAI(select(0.0,l,good),select(0.0,h,good));
}
fn ra_ipoint(bits:u32)->RAI {return ra_ibounds(bits,bits);}
fn ra_point(x:f32)->RAI {return ra_ipoint(bitcast<u32>(x));}
fn ra_ivalid(a:RAI)->bool {return !ra_bad;}
fn ra_ilo(a:RAI)->f32 {return a.lo;}
fn ra_ihi(a:RAI)->f32 {return a.hi;}
fn ra_round_bounds(bits:u32,n:u32,exact:bool)->RAI {
 let sign=(bits&0x80000000u)!=0u;let sourceOK=ra_supported(bits)&&(exact||(bits&0x7fffffffu)!=0u);
 let lo=select(select(bits-n,bits+n,sign),bits,exact);let hi=select(select(bits+n,bits-n,sign),bits,exact);
 ra_bad=select(true,ra_bad,sourceOK);return ra_ibounds(select(0u,lo,sourceOK),select(0u,hi,sourceOK));
}
fn ra_pad(bits:u32,n:u32)->RAI {return ra_round_bounds(bits,n,false);}
fn ra_add_scalar(a:f32,b:f32)->RAI {return ra_round_bounds(bitcast<u32>(a+b),1u,a==0.0||b==0.0||a == -b);}
fn ra_ineg(a:RAI)->RAI {return RAI(-a.hi,-a.lo);}
fn ra_iadd(a:RAI,b:RAI)->RAI {let l=ra_add_scalar(a.lo,b.lo);let h=ra_add_scalar(a.hi,b.hi);return RAI(l.lo,h.hi);}
fn ra_isub(a:RAI,b:RAI)->RAI {return ra_iadd(a,ra_ineg(b));}
fn ra_mul_scalar(a:f32,b:f32)->RAI {
 let zero=a==0.0||b==0.0;let e=i32((bitcast<u32>(a)>>23u)&255u)+i32((bitcast<u32>(b)>>23u)&255u)-254;
 let good=zero||(e>=-120&&e<=124);ra_bad=select(true,ra_bad,good);
 return ra_round_bounds(bitcast<u32>(select(0.0,a,good)*select(0.0,b,good)),1u,zero);
}
fn ra_imul(a:RAI,b:RAI)->RAI {
 let p=ra_mul_scalar(a.lo,b.lo);let q=ra_mul_scalar(a.lo,b.hi);let r=ra_mul_scalar(a.hi,b.lo);let s=ra_mul_scalar(a.hi,b.hi);
 return RAI(min(min(p.lo,q.lo),min(r.lo,s.lo)),max(max(p.hi,q.hi),max(r.hi,s.hi)));
}
fn ra_div_scalar(a:f32,b:f32)->RAI {
 let e=i32((bitcast<u32>(a)>>23u)&255u)-i32((bitcast<u32>(b)>>23u)&255u);
 let good=b!=0.0&&(a==0.0||(e>=-119&&e<=124));ra_bad=select(true,ra_bad,good);
 return ra_round_bounds(bitcast<u32>(select(0.0,a,good)/select(1.0,b,good)),8u,a==0.0);
}
fn ra_idiv(a:RAI,b:RAI)->RAI {
 // A denominator has one strict sign. Transform a negative denominator by
 // negating BOTH intervals, then monotonicity selects just two endpoint
 // quotients. This is the same interval hull as four-corner division.
 let good=b.lo>0.0||b.hi<0.0;ra_bad=select(true,ra_bad,good);let neg=b.hi<0.0;
 let al=select(a.lo,-a.hi,neg);let ah=select(a.hi,-a.lo,neg);
 let bl=select(1.0,select(b.lo,-b.hi,neg),good);let bh=select(1.0,select(b.hi,-b.lo,neg),good);
 let l=ra_div_scalar(al,select(bl,bh,al>=0.0));let h=ra_div_scalar(ah,select(bh,bl,ah>=0.0));
 return RAI(l.lo,h.hi);
}
fn ra_isquare(a:RAI)->RAI {
 let l=ra_mul_scalar(a.lo,a.lo);let h=ra_mul_scalar(a.hi,a.hi);
 return RAI(select(min(l.lo,h.lo),0.0,a.lo<=0.0&&a.hi>=0.0),max(l.hi,h.hi));
}
fn ra_sqrt_scalar(x:f32)->RAI {
 if(x==0.0){return RAI(0.0,0.0);}if(x<0.0){return ra_iunknown(1u);}
 let inv=ra_pad(bitcast<u32>(inverseSqrt(x)),8u);return ra_idiv(RAI(1.0,1.0),inv);
}
fn ra_sqrt(a:RAI)->RAI {let l=ra_sqrt_scalar(a.lo);let h=ra_sqrt_scalar(a.hi);return RAI(l.lo,h.hi);}
fn ra_idot3(a:RAVec3,b:RAVec3)->RAI {return ra_iadd(ra_iadd(ra_imul(a.x,b.x),ra_imul(a.y,b.y)),ra_imul(a.z,b.z));}
fn ra_ilength3(a:RAVec3)->RAI {return ra_sqrt(ra_iadd(ra_iadd(ra_isquare(a.x),ra_isquare(a.y)),ra_isquare(a.z)));}
fn ra_symmetric(radius:f32)->RAI {
 if(radius<0.0){return ra_iunknown(1u);}return ra_ibounds(bitcast<u32>(-radius),bitcast<u32>(radius));
}
fn ra_centered(center:f32,radius:f32)->RAI {return ra_iadd(ra_point(center),ra_symmetric(radius));}
fn ra_absmax(a:RAI)->f32 {return max(abs(a.lo),abs(a.hi));}
fn ra_absmin(a:RAI)->f32 {if(a.lo<=0.0&&a.hi>=0.0){return 0.0;}return min(abs(a.lo),abs(a.hi));}
fn ra_component(v:RAVec3,k:u32)->RAI {if(k==0u){return v.x;}if(k==1u){return v.y;}return v.z;}
fn ra_vec_valid(v:RAVec3)->bool {return !ra_bad;}
fn ra_cross(a:RAVec3,b:RAVec3)->RAVec3 {
 return RAVec3(ra_isub(ra_imul(a.y,b.z),ra_imul(a.z,b.y)),ra_isub(ra_imul(a.z,b.x),ra_imul(a.x,b.z)),ra_isub(ra_imul(a.x,b.y),ra_imul(a.y,b.x)));
}
fn ra_difference(a:vec3f,b:vec3f)->RAVec3 {return RAVec3(ra_isub(ra_point(a.x),ra_point(b.x)),ra_isub(ra_point(a.y),ra_point(b.y)),ra_isub(ra_point(a.z),ra_point(b.z)));}
fn ra_origin_difference(origin:vec3f,error:vec3f,a:vec3f)->RAVec3 {
 let d=ra_difference(origin,a);return RAVec3(ra_iadd(d.x,ra_symmetric(error.x)),ra_iadd(d.y,ra_symmetric(error.y)),ra_iadd(d.z,ra_symmetric(error.z)));
}
fn ra_inflate(a:RAI,error:RAI)->RAI {if(error.lo<0.0){return ra_iunknown(1u);}return ra_iadd(a,ra_symmetric(error.hi));}
fn ra_cpu_normalize(d:RAVec3)->RAVec3 {
 var n=ra_ilength3(d);n=ra_inflate(n,ra_imul(ra_point(ra_absmax(n)),ra_ipoint(0x27000000u)));
 if(ra_bad||n.lo<=0.0){let bad=ra_iunknown(1u);return RAVec3(bad,bad,bad);}
 let inverse=ra_idiv(RAI(1.0,1.0),n);return RAVec3(ra_imul(d.x,inverse),ra_imul(d.y,inverse),ra_imul(d.z,inverse));
}
// Restricted camera/axis-mirror invariant, never inferred from a GPU norm.
// CPU normalize is fl64(raw * fl64(1/hypot(raw))). With u=2^-53 and
// |hypot/||raw||-1|<=16u, finite normal arithmetic gives
// L=(1-u)^2/(1+16u) <= ||D|| <= U=(1+u)^2/(1-16u).
// A signed coordinate-axis reflection changes exactly one component sign;
// therefore every such stored beam retains this SAME norm bound. Re-normalizing
// for the CPU query gives Q_i/D_i in [L/U,U/L], both within 37u of 1.
// For zero D_i the ratio is undefined but Q_i is exactly zero.
//
// Tiny CPU components do not need to be assumed normal. With gradual underflow
// |fl64(x)-x|<=u*|x|+tau, tau=2^-1075, camera normalization instead gives
// ||D|| in [L-2tau,U+2tau] subset [1-19u,1+19u]. Its raw norm and reciprocal
// must be normal, as bounded by the SUCCESSFUL compact camera initialization.
// The subsequent query then satisfies |Q_i-D_i|<38u*|D_i|+tau. A nonzero
// compact interval has M=absmax>=2^-100; 64u*M dominates 38u*M+tau. A singleton
// zero remains exact. Compact underflow/range failures still become UNKNOWN.
// This does not turn the implementation-dependent CPU hypot assumption into a
// language guarantee, nor remove the mandatory complete independent shadow.
fn ra_unit_query_error(a:RAI)->RAI {
 return ra_imul(ra_point(ra_absmax(a)),ra_ipoint(0x28000000u)); // 64*u64 = 2^-47
}
fn ra_cpu_unit_query(d:RAVec3)->RAVec3 {
 return RAVec3(ra_inflate(d.x,ra_unit_query_error(d.x)),ra_inflate(d.y,ra_unit_query_error(d.y)),ra_inflate(d.z,ra_unit_query_error(d.z)));
}
fn ra_close(candidate:f32,reference:RAI)->bool {
 let difference=ra_isub(reference,ra_point(candidate));
 let limit=ra_iadd(ra_ipoint(0x37a7c5acu),ra_imul(ra_ipoint(0x3649539bu),ra_point(ra_absmin(reference))));
 return !ra_bad&&ra_absmax(difference)<=limit.lo;
}
fn ra_export(a:RAI)->RI {return RI(bitcast<u32>(a.lo),bitcast<u32>(a.hi),select(0u,1u,ra_bad));}
fn ra_triangle_axis(t:RGTriangle)->u32 {
 var axis=3u;
 for(var k=0u;k<3u;k++){
  if(t.a[k]==t.b[k]&&t.a[k]==t.c[k]&&abs(t.normal[k])==1.0){
   if(axis!=3u){return 3u;}axis=k;
  }
 }
 if(axis==3u){return axis;}
 for(var k=0u;k<3u;k++){if(k!=axis&&t.normal[k]!=0.0){return 3u;}}
 return axis;
}
fn ra_unknown_triangle(reason:u32)->RATriangleResult {
 return RATriangleResult(2u,reason,3u,ra_iunknown(RI_BAD_DOMAIN),ra_iunknown(RI_BAD_DOMAIN));
}
// Preserve the CPU dot's x+y then z order while exploiting exact zero axis
// edge components. The three term values have already been outward rounded.
fn ra_axis_dot(ti:RAI,tj:RAI,tk:RAI,axis:u32)->RAI {
 var lo=vec3f(0.0);var hi=vec3f(0.0);let i=(axis+1u)%3u;let j=(axis+2u)%3u;
 lo[i]=ti.lo;lo[j]=tj.lo;lo[axis]=tk.lo;hi[i]=ti.hi;hi[j]=tj.hi;hi[axis]=tk.hi;
 return ra_iadd(ra_iadd(RAI(lo.x,hi.x),RAI(lo.y,hi.y)),RAI(lo.z,hi.z));
}
// One triangle's bounded plane/projected setup (88 bytes). kind=0 is a
// proven miss, 2 unknown, 3 ready for barycentric test. No omitted triangles.
struct RAPlaneState {kind:u32,reason:u32,axis:u32,index:u32,distance:RAI,planeDelta:RAI,wi:RAI,wj:RAI,ai:RAI,aj:RAI,bi:RAI,bj:RAI,area:RAI}
fn ra_plane_value(kind:u32,reason:u32,index:u32,axis:u32,distance:RAI,delta:RAI)->RAPlaneState {
 let z=RAI(0.0,0.0);return RAPlaneState(kind,reason,axis,index,distance,delta,z,z,z,z,z,z,z);
}
fn ra_plane_unknown(index:u32,reason:u32)->RAPlaneState {return ra_plane_value(2u,reason,index,3u,RAI(0.0,0.0),RAI(0.0,0.0));}
fn ra_plane_query(t:RGTriangle,index:u32,origin:vec3f,error:vec3f,q:RAVec3,kappa:RAI)->RAPlaneState {
 let axis=ra_triangle_axis(t);if(axis==3u){return ra_plane_unknown(index,RA_UNSUPPORTED);}
 let s=ra_origin_difference(origin,error,t.a);let i=(axis+1u)%3u;let j=(axis+2u)%3u;
 let ai=ra_isub(ra_point(t.b[i]),ra_point(t.a[i]));let aj=ra_isub(ra_point(t.b[j]),ra_point(t.a[j]));
 let bi=ra_isub(ra_point(t.c[i]),ra_point(t.a[i]));let bj=ra_isub(ra_point(t.c[j]),ra_point(t.a[j]));
 if(ra_bad){return ra_plane_unknown(index,RA_INTERVAL);}
 let qk=ra_component(q,axis);let delta=ra_ineg(ra_component(s,axis));
 // Exact zero direction yields exact zero determinant in the CPU reference.
 if(qk.lo==0.0&&qk.hi==0.0){return ra_plane_value(0u,0u,index,axis,RAI(0.0,0.0),delta);}
 if(ra_absmin(qk)<0x1p-12){return ra_plane_unknown(index,RA_DETERMINANT);}
 let planeDistance=ra_idiv(delta,qk);
 var distance=planeDistance;
 if(!ra_ivalid(distance)){return ra_plane_unknown(index,RA_INTERVAL);}
 distance=ra_inflate(distance,ra_imul(ra_imul(ra_point(ra_absmax(distance)),kappa),ra_ipoint(0x27800000u))); //32u64
 if(!ra_ivalid(distance)){return ra_plane_unknown(index,RA_INTERVAL);}
 // CPU near=1e-7 is between these adjacent f32 values. Its far is Infinity;
 // beyond the finite GPU far is UNKNOWN, never a certified CPU miss.
 let near=ra_ibounds(0x33d6bf94u,0x33d6bf95u);
 if(ra_ihi(distance)<ra_ilo(near)){return ra_plane_value(0u,0u,index,axis,distance,delta);}
 // Projected barycentrics avoid a very large repeated interval cross/dot
 // compiler graph. Explicit CPU64 forward-error padding bridges the original
 // MT arithmetic, rather than silently substituting a different predicate.
 // Existing host bounds imply: |S_i|<=513, |Q_i|<=2, |Q_k|>=2^-12,
 // true altitude>=2^-13 (factor-two host slack), max edge<=1024, kappa<=16. Thus the exact determinant
 // exceeds the CPU parallel threshold by >250x, even after CPU64 rounding.
 // For each barycentric numerator, sumAbsTerms/|det| <=
 // 513*(1+2*4096)*sqrt(2)*8192 < 2^36. CPU MT numerator/determinant/division
 // error is <128*u64*2^36=2^-10; use 2^-9 for extra headroom. This is an
 // AMBIGUITY band: physical triangle bounds remain unchanged.
 let qi=ra_component(q,i);let qj=ra_component(q,j);let si=ra_component(s,i);let sj=ra_component(s,j);
 let area=ra_isub(ra_imul(ai,bj),ra_imul(aj,bi));
 if(ra_bad){return ra_plane_unknown(index,RA_INTERVAL);}
 if(ra_absmin(area)==0.0){return ra_plane_unknown(index,RA_DETERMINANT);}
 let wi=ra_iadd(si,ra_imul(qi,planeDistance));let wj=ra_iadd(sj,ra_imul(qj,planeDistance));
 return RAPlaneState(3u,0u,axis,index,distance,delta,wi,wj,ai,aj,bi,bj,area);
}
fn ra_plane_test(p:RAPlaneState)->RATriangleResult {
 if(p.kind==0u){return RATriangleResult(0u,0u,p.axis,p.distance,p.planeDelta);}
 if(p.kind!=3u){return ra_unknown_triangle(p.reason);}
 let wi=p.wi;let wj=p.wj;let ai=p.ai;let aj=p.aj;let bi=p.bi;let bj=p.bj;let area=p.area;
 let distance=p.distance;let delta=p.planeDelta;let axis=p.axis;let near=ra_ibounds(0x33d6bf94u,0x33d6bf95u);
 let geometricU=ra_idiv(ra_isub(ra_imul(wi,bj),ra_imul(wj,bi)),area);
 let geometricV=ra_idiv(ra_isub(ra_imul(ai,wj),ra_imul(aj,wi)),area);
 let u=ra_inflate(geometricU,ra_point(0x1p-9));let v=ra_inflate(geometricV,ra_point(0x1p-9));let uv=ra_iadd(u,v);
 if(ra_bad){return ra_unknown_triangle(RA_INTERVAL);}
 // For rejection use OUTER original barycentric thresholds; for acceptance
 // require strict interior. The sliver between either is deliberately unknown.
 let negativeTolerance=bitcast<f32>(0xab8cbccdu); // <= -1e-12
 let upperTolerance=bitcast<f32>(0x3f800001u); // >= 1+1e-12
 if(ra_ihi(u)<negativeTolerance||ra_ilo(u)>upperTolerance||ra_ihi(v)<negativeTolerance||ra_ilo(uv)>upperTolerance){
  return RATriangleResult(0u,0u,axis,distance,delta);
 }
 if(!(ra_ilo(u)>0.0&&ra_ilo(v)>0.0&&ra_ihi(uv)<1.0)){return ra_unknown_triangle(RA_BARYCENTRIC);}
 if(ra_ilo(distance)<ra_ihi(near)||ra_ihi(distance)>bitcast<f32>(0x7149f2c9u)){return ra_unknown_triangle(RA_NEAR_FAR);}
 return RATriangleResult(1u,0u,axis,distance,delta);
}
fn ra_triangle_query(t:RGTriangle,origin:vec3f,error:vec3f,q:RAVec3,kappa:RAI)->RATriangleResult {return ra_plane_test(ra_plane_query(t,t.id,origin,error,q,kappa));}
fn ra_failure(reason:u32)->RAResult {
 return RAResult(0u,reason,RI(0u,0u,1u),vec3f(0.0),3u,0.0,0.0);
}
// Directions/errors describe the CPU STORED beam, not its normalized query.
// originError is relative to the exact real rebased CPU origin. The caller may
// retain these radii through exact axis sign flips only. Returned point radii
// satisfy a stricter absolute 2e-5 check; this never relaxes camera-anchored
// 2e-5+3e-6*abs(reference) admission. Unknown must trigger the existing fallback.
// Staged API. Persist each output only with its original immutable ray and
// candidate. No scatter or origin update may occur until ra_finish accepts.
// Query state is 64 bytes; scan state is 40 bytes (natural storage layout).
struct RAQueryState {valid:u32,reason:u32,stored:RAVec3,query:RAVec3,kappa:RAI}
struct RAScanState {valid:u32,reason:u32,found:u32,winner:RATriangleResult}
// Cursor scan state is 52 bytes. Each plane/test pair processes one original
// triangle and advances exactly once. Finalization requires the full count.
struct RAScanProgress {valid:u32,reason:u32,cursor:u32,haveWinner:u32,haveOther:u32,otherDistance:f32,winner:RATriangleResult}
fn ra_progress_failure(reason:u32)->RAScanProgress {var result:RAScanProgress;result.reason=reason;return result;}
fn ra_scan_begin(candidate:RGHit,prepared:RAQueryState)->RAScanProgress {
 if(prepared.valid!=1u){return ra_progress_failure(prepared.reason|RA_INTERVAL);}
 if(candidate.valid>1u){return ra_progress_failure(RA_CANDIDATE);}
 if(candidate.valid!=0u){
  if(candidate.triangle>=rg_triangle_count()){return ra_progress_failure(RA_CANDIDATE);}
  let t=rg_triangle(candidate.triangle);if(candidate.material!=t.material||!all(candidate.normal==t.normal)){return ra_progress_failure(RA_CANDIDATE);}
 }
 var result:RAScanProgress;result.valid=1u;return result;
}
fn ra_scan_plane(origin:vec3f,error:vec3f,prepared:RAQueryState,progress:RAScanProgress)->RAPlaneState {
 ra_bad=false;
 if(prepared.valid!=1u||progress.valid!=1u){return ra_plane_unknown(progress.cursor,prepared.reason|progress.reason|RA_INTERVAL);}
 if(progress.cursor>=rg_triangle_count()){return ra_plane_unknown(progress.cursor,RA_CANDIDATE);}
 return ra_plane_query(rg_triangle(progress.cursor),progress.cursor,origin,error,prepared.query,prepared.kappa);
}
fn ra_scan_step(candidate:RGHit,plane:RAPlaneState,previous:RAScanProgress)->RAScanProgress {
 ra_bad=false;var progress=previous;
 if(progress.valid!=1u){return progress;}
 if(plane.index!=progress.cursor||progress.cursor>=rg_triangle_count()){return ra_progress_failure(RA_CANDIDATE);}
 let hit=ra_plane_test(plane);
 if(hit.classification==2u||ra_bad){return ra_progress_failure(hit.reason|select(0u,RA_INTERVAL,ra_bad));}
 if(candidate.valid!=0u&&progress.cursor==candidate.triangle){
  if(hit.classification!=1u){return ra_progress_failure(RA_CANDIDATE);}
  progress.winner=hit;progress.haveWinner=1u;
 }else if(hit.classification==1u){
  if(candidate.valid==0u){return ra_progress_failure(RA_CANDIDATE);}
  if(progress.haveOther==0u||hit.distance.lo<progress.otherDistance){progress.otherDistance=hit.distance.lo;}progress.haveOther=1u;
 }
 progress.cursor+=1u;return progress;
}
fn ra_scan_end(candidate:RGHit,progress:RAScanProgress)->RAScanState {
 if(progress.valid!=1u){return ra_scan_failure(progress.reason);}
 if(progress.cursor!=rg_triangle_count()){return ra_scan_failure(RA_CANDIDATE);}
 if(candidate.valid==0u){return RAScanState(1u,0u,0u,progress.winner);}
 if(progress.haveWinner!=1u){return ra_scan_failure(RA_CANDIDATE);}
 if(progress.haveOther!=0u&&!(progress.winner.distance.hi<progress.otherDistance)){return ra_scan_failure(RA_ORDER);}
 return RAScanState(1u,0u,1u,progress.winner);
}
fn ra_query_failure(reason:u32)->RAQueryState {let z=RAI(0.0,0.0);let v=RAVec3(z,z,z);return RAQueryState(0u,reason,v,v,z);}
fn ra_scan_failure(reason:u32)->RAScanState {var w:RATriangleResult;return RAScanState(0u,reason,0u,w);}
fn ra_prepare_impl(origin:vec3f,originError:vec3f,direction:vec3f,directionError:vec3f,unitProvenance:bool)->RAQueryState {
 ra_bad=false;
 if(rg_data[7].z==0u||rg_f(6u).w!=0.0||rg_triangle_count()==0u||rg_triangle_count()>64u){return ra_query_failure(RA_UNSUPPORTED);}
 let kappa=ra_ipoint(rg_data[7].w);
 if(!ra_ivalid(kappa)||ra_ilo(kappa)<1.0||ra_ihi(kappa)>16.0){return ra_query_failure(RA_UNSUPPORTED);}
 if(!all(abs(origin)<=vec3f(256.0))||!all(abs(direction)<=vec3f(2.0))||!all(originError>=vec3f(0.0))||!all(originError<=vec3f(1.0))||!all(directionError>=vec3f(0.0))||!all(directionError<=vec3f(1.0))){return ra_query_failure(RA_UNSUPPORTED);}
 let stored=RAVec3(ra_centered(direction.x,directionError.x),ra_centered(direction.y,directionError.y),ra_centered(direction.z,directionError.z));
 if(!ra_vec_valid(stored)||!ra_close(direction.x,stored.x)||!ra_close(direction.y,stored.y)||!ra_close(direction.z,stored.z)){return ra_query_failure(RA_NUMERIC|RA_NUMERIC_DIRECTION);}
 var query:RAVec3;
 if(unitProvenance){query=ra_cpu_unit_query(stored);}else{query=ra_cpu_normalize(stored);}
 if(!ra_vec_valid(query)){return ra_query_failure(RA_INTERVAL);}
 if(ra_absmax(query.x)>2.0||ra_absmax(query.y)>2.0||ra_absmax(query.z)>2.0){return ra_query_failure(RA_UNSUPPORTED);}
 return RAQueryState(1u,0u,stored,query,kappa);
}
// Generic standalone rays may be arbitrarily nonunit. Keep their full CPU
// normalization; an approximate norm check is NOT sufficient provenance.
fn ra_prepare(origin:vec3f,originError:vec3f,direction:vec3f,directionError:vec3f)->RAQueryState {
 return ra_prepare_impl(origin,originError,direction,directionError,false);
}
// provenance=1 may only be issued by successful resident camera initialization
// and retained through verified exact-axis CPU/GPU mirror sign flips. It is
// private resident state, never a host-supplied per-ray claim. Every unsupported
// scatter, reset/replay, failed initialization or invalid history must clear it.
// Missing or unknown provenance rejects instead of silently choosing a shortcut.
fn ra_prepare_unit(origin:vec3f,originError:vec3f,direction:vec3f,directionError:vec3f,provenance:u32)->RAQueryState {
 if(provenance!=1u){return ra_query_failure(RA_UNSUPPORTED);}
 return ra_prepare_impl(origin,originError,direction,directionError,true);
}
fn ra_scan(origin:vec3f,originError:vec3f,candidate:RGHit,prepared:RAQueryState)->RAScanState {
 ra_bad=false;if(prepared.valid!=1u){return ra_scan_failure(prepared.reason|RA_INTERVAL);}
 let query=prepared.query;let kappa=prepared.kappa;
 var winner:RATriangleResult;var haveWinner=false;var haveOther=false;var otherDistance=0.0;
 if(candidate.valid>1u){return ra_scan_failure(RA_CANDIDATE);}
 if(candidate.valid!=0u){
  if(candidate.triangle>=rg_triangle_count()){return ra_scan_failure(RA_CANDIDATE);}
  let triangle=rg_triangle(candidate.triangle);
  if(candidate.material!=triangle.material||!all(candidate.normal==triangle.normal)){return ra_scan_failure(RA_CANDIDATE);}
 }
 // Audit ALL authored triangles, even ones the GPU BVH did not visit. A single
 // call site avoids duplicating the large RI graph in driver compilation.
 for(var index=0u;index<rg_triangle_count();index++){
  let hit=ra_triangle_query(rg_triangle(index),origin,originError,query,kappa);
  if(hit.classification==2u){return ra_scan_failure(hit.reason);}
  if(candidate.valid!=0u&&index==candidate.triangle){
   if(hit.classification!=1u){return ra_scan_failure(RA_CANDIDATE);}
   winner=hit;haveWinner=true;
  }else if(hit.classification==1u){
   if(candidate.valid==0u){return ra_scan_failure(RA_CANDIDATE);}
   if(!haveOther||ra_ilo(hit.distance)<otherDistance){otherDistance=ra_ilo(hit.distance);}haveOther=true;
  }
 }
 if(candidate.valid==0u){return RAScanState(1u,0u,0u,winner);}
 if(!haveWinner){return ra_scan_failure(RA_CANDIDATE);}
 if(haveOther&&!(ra_ihi(winner.distance)<otherDistance)){return ra_scan_failure(RA_ORDER);}
 return RAScanState(1u,0u,1u,winner);
}
// One tangential component without an interval-valued loop. The common CPU
// query reciprocal r=fl64(1/hypot(D)) cancels: Q_i/Q_k equals D_i/D_k times
// (1+e_i)/(1+e_k), |e_i|,|e_k|<=u64. For G=(D_i/D_k)*planeDelta, combining
// the established 32*u64*kappa t error, query component multiplications,
// hit-product rounding and the product-dependent final addition gives
// [((1+u64)^3/(1-u64))*(1+32*kappa*u64)-1] < 37*kappa*u64.
// Use 64*kappa*u64*|G| plus 4*u64*512 for the world-origin addition. All
// stored-direction and plane-delta intervals are retained; no GPU point
// direction is substituted. The normalized-query t audit remains separate.
fn ra_point_error(origin:f32,error:f32,tangent:RAI,axis:RAI,delta:RAI,kappa:RAI,point:f32)->f32 {
 let product=ra_imul(ra_idiv(tangent,axis),delta);
 let offset=ra_iadd(ra_isub(ra_point(origin),ra_point(point)),ra_iadd(ra_symmetric(error),product));
 let rounding=ra_iadd(ra_ipoint(0x2a800000u),ra_imul(ra_imul(ra_point(ra_absmax(product)),kappa),ra_ipoint(0x28000000u)));
 let total=ra_iadd(ra_point(ra_absmax(offset)),rounding);return total.hi;
}
fn ra_finish(origin:vec3f,originError:vec3f,direction:vec3f,candidate:RGHit,prepared:RAQueryState,scanned:RAScanState)->RAResult {
 ra_bad=false;
 if(prepared.valid!=1u||scanned.valid!=1u){return ra_failure(prepared.reason|scanned.reason|RA_INTERVAL);}
 if(candidate.valid==0u){if(scanned.found!=0u){return ra_failure(RA_CANDIDATE);}return RAResult(1u,0u,RI(0u,0u,0u),originError,3u,0.0,0.0);}
 if(candidate.valid!=1u||scanned.found!=1u){return ra_failure(RA_CANDIDATE);}
 let stored=prepared.stored;let query=prepared.query;let kappa=prepared.kappa;let winner=scanned.winner;
 if(candidate.point[winner.axis]!=rg_triangle(candidate.triangle).a[winner.axis]){return ra_failure(RA_CANDIDATE);}
 // Preserve normal-facing/emission branches as well as the geometric hit.
 let storedAxis=ra_component(stored,winner.axis);
 if(direction[winner.axis]==0.0||(direction[winner.axis]<0.0&&storedAxis.hi>=0.0)||(direction[winner.axis]>0.0&&storedAxis.lo<=0.0)){return ra_failure(RA_DETERMINANT);}
 if(!ra_close(candidate.distance,winner.distance)){return ra_failure(RA_NUMERIC|RA_NUMERIC_DISTANCE);}
 let residual=ra_iadd(ra_imul(ra_imul(ra_point(ra_absmax(winner.planeDelta)),kappa),ra_ipoint(0x28000000u)),ra_ipoint(0x2a800000u));
 if(ra_bad){return ra_failure(RA_INTERVAL);}
 var errors=vec3f(residual.hi);
 let first=(winner.axis+1u)%3u;let second=(winner.axis+2u)%3u;
 errors[first]=ra_point_error(origin[first],originError[first],ra_component(stored,first),storedAxis,winner.planeDelta,kappa,candidate.point[first]);
 errors[second]=ra_point_error(origin[second],originError[second],ra_component(stored,second),storedAxis,winner.planeDelta,kappa,candidate.point[second]);
 if(ra_bad){return ra_failure(RA_INTERVAL);}
 let audited=ra_iadd(ra_point(max(errors.x,max(errors.y,errors.z))),ra_ipoint(0x2b800000u));
 if(ra_bad||audited.hi>bitcast<f32>(0x37a7c5acu)){return ra_failure(RA_NUMERIC|RA_NUMERIC_POINT);}
 return RAResult(1u,0u,ra_export(winner.distance),errors,winner.axis,candidate.point[winner.axis],ra_ihi(residual));
}

// Convenience form only. Native walkers should use the staged functions to
// avoid creating one enormous driver compiler graph from the whole audit.
fn ra_check(origin:vec3f,originError:vec3f,direction:vec3f,directionError:vec3f,candidate:RGHit)->RAResult {
 let prepared=ra_prepare(origin,originError,direction,directionError);
 let scanned=ra_scan(origin,originError,candidate,prepared);
 return ra_finish(origin,originError,direction,candidate,prepared,scanned);
}

// v20 research candidate: camera-anchored raw-direction enclosure.
// No existing v19 API above is changed. Complete CPU shadow remains mandatory.
// Invariant in exact real arithmetic, with PHYSICAL (not unfolded) residual e:
// O_i = A_i + V_i*s_a + e_i, A_i=sign_i*C_i+B_i, V_i=sign_i*raw_i,
// s_a=(lastPlane-A_a)/V_a (or 0 initially), |e_i|<=residual_i.
// residual is CPU64-only; never replace it with candidate-centred originError.
// B is an exact 2^-16-grid integer. Conversion to f32 must roundtrip exactly.
struct RAAnchor {
 raw:RAVec3, rawNorm:RAI, offsetGrid:vec3i, signMask:u32,
 residual:vec3f, axis:u32, plane:f32, valid:u32,
}
struct RAAnchorResult { result:RAResult, anchor:RAAnchor }
fn ra_anchor_initialize(raw:RAVec3)->RAAnchor {
 ra_bad=false;var a:RAAnchor;a.raw=raw;a.axis=3u;
 // The compact camera computation must have succeeded before this API.
 let x=ra_ibounds(bitcast<u32>(raw.x.lo),bitcast<u32>(raw.x.hi));
 let y=ra_ibounds(bitcast<u32>(raw.y.lo),bitcast<u32>(raw.y.hi));
 let z=ra_ibounds(bitcast<u32>(raw.z.lo),bitcast<u32>(raw.z.hi));
 // A strict nonzero sign proves CPU64 camera/query component products normal.
 // Generic unit mode retains its separate subnormal extension.
 for(var k=0u;k<3u;k++){let r=ra_component(raw,k);if(r.lo<=0.0&&r.hi>=0.0&&!(r.lo==0.0&&r.hi==0.0)){return a;}}
 a.rawNorm=ra_ilength3(RAVec3(x,y,z));
 if(!ra_bad&&a.rawNorm.lo>=0x1p-100&&a.rawNorm.hi<=0x1p48){a.valid=1u;}
 return a;
}
fn ra_anchor_sign(a:RAAnchor,k:u32)->f32 {return select(1.0,-1.0,(a.signMask&(1u<<k))!=0u);}
fn ra_anchor_offset(a:RAAnchor,k:u32)->f32 {return f32(a.offsetGrid[k])*0x1p-16;}
fn ra_anchor_direction(a:RAAnchor,k:u32)->RAI {
 let r=ra_component(a.raw,k);if((a.signMask&(1u<<k))!=0u){return ra_ineg(r);}return r;
}
fn ra_anchor_valid(a:RAAnchor)->bool {
 if(a.valid!=1u||a.axis>3u||a.signMask>7u||any(a.residual<vec3f(0.0))||any(a.residual>vec3f(1.0))){return false;}
 if(any(a.offsetGrid>vec3i(201326592))||any(a.offsetGrid<vec3i(-201326592))){return false;}
 for(var k=0u;k<3u;k++){if(i32(f32(a.offsetGrid[k]))!=a.offsetGrid[k]){return false;}}
 return a.rawNorm.lo>0.0&&a.rawNorm.hi<=0x1p48;
}
// p-A_k, centered before any uncertain absolute world endpoint is created.
fn ra_anchor_delta(a:RAAnchor,k:u32,p:f32,camera:vec3f,cameraError:vec3f)->RAI {
 let fixed=ra_isub(ra_point(p),ra_point(ra_anchor_offset(a,k)));
 let c=ra_centered(camera[k],cameraError[k]);
 return ra_isub(fixed,ra_imul(ra_point(ra_anchor_sign(a,k)),c));
}
// A_k-p, the companion centered expression used for point/barycentric tests.
fn ra_anchor_base(a:RAAnchor,k:u32,p:f32,camera:vec3f,cameraError:vec3f)->RAI {
 let fixed=ra_isub(ra_point(ra_anchor_offset(a,k)),ra_point(p));
 let c=ra_centered(camera[k],cameraError[k]);
 return ra_iadd(fixed,ra_imul(ra_point(ra_anchor_sign(a,k)),c));
}
fn ra_anchor_local_error(product:RAI,kappa:RAI)->RAI {
 return ra_iadd(ra_imul(ra_imul(ra_point(ra_absmax(product)),kappa),ra_ipoint(0x28000000u)),ra_ipoint(0x2a800000u));
}
fn ra_anchor_tangent_residual(a:RAAnchor,i:u32,k:u32,ratio:RAI,actualDelta:RAI,kappa:RAI)->RAI {
 let inherited=ra_iadd(ra_point(a.residual[i]),ra_imul(ra_point(ra_absmax(ratio)),ra_point(a.residual[k])));
 return ra_iadd(inherited,ra_anchor_local_error(ra_imul(ratio,actualDelta),kappa));
}
fn ra_anchor_plane_query(t:RGTriangle,index:u32,origin:vec3f,totalOriginError:vec3f,prepared:RAQueryState,a:RAAnchor,camera:vec3f,cameraError:vec3f)->RAPlaneState {
 ra_bad=false;
 if(prepared.valid!=1u||!ra_anchor_valid(a)||any(cameraError<vec3f(0.0))||any(cameraError>vec3f(1.0))){return ra_plane_unknown(index,RA_UNSUPPORTED);}
 let k=ra_triangle_axis(t);if(k==3u){return ra_plane_unknown(index,RA_UNSUPPORTED);}
 let qk=ra_component(prepared.query,k);let vk=ra_anchor_direction(a,k);
 let actualDelta=ra_ineg(ra_component(ra_origin_difference(origin,totalOriginError,t.a),k));
 if(qk.lo==0.0&&qk.hi==0.0){return ra_plane_value(0u,0u,index,k,RAI(0.0,0.0),actualDelta);}
 if(ra_absmin(qk)<0x1p-12||ra_absmin(vk)==0.0){return ra_plane_unknown(index,RA_DETERMINANT);}
 let delta=ra_anchor_delta(a,k,t.a[k],camera,cameraError);
 var parameter:RAI;
 if(a.axis==3u){parameter=ra_idiv(ra_isub(delta,ra_symmetric(a.residual[k])),vk);}
 else if(a.axis==k){
  // Essential cancellation. The previous face is tested, not skipped.
  let separation=ra_isub(ra_point(t.a[k]),ra_point(a.plane));
  parameter=ra_idiv(ra_isub(separation,ra_symmetric(a.residual[k])),vk);
 }else{
  let previous=ra_idiv(ra_anchor_delta(a,a.axis,a.plane,camera,cameraError),ra_anchor_direction(a,a.axis));
  parameter=ra_isub(ra_idiv(ra_isub(delta,ra_symmetric(a.residual[k])),vk),previous);
 }
 var distance=ra_imul(a.rawNorm,parameter);
 // Q_k/(V_k/||raw||) differs from one by <32u. Including MT's32*kappa*u
 // and inversion gives <128*kappa*u; this does not alter admission tolerance.
 distance=ra_inflate(distance,ra_imul(ra_imul(ra_point(ra_absmax(distance)),prepared.kappa),ra_ipoint(0x28800000u)));
 if(ra_bad){return ra_plane_unknown(index,RA_INTERVAL);}
 let near=ra_ibounds(0x33d6bf94u,0x33d6bf95u);
 if(distance.hi<near.lo){return ra_plane_value(0u,0u,index,k,distance,actualDelta);}
 let i=(k+1u)%3u;let j=(k+2u)%3u;
 let ai=ra_isub(ra_point(t.b[i]),ra_point(t.a[i]));let aj=ra_isub(ra_point(t.b[j]),ra_point(t.a[j]));
 let bi=ra_isub(ra_point(t.c[i]),ra_point(t.a[i]));let bj=ra_isub(ra_point(t.c[j]),ra_point(t.a[j]));
 let area=ra_isub(ra_imul(ai,bj),ra_imul(aj,bi));
 if(ra_bad||ra_absmin(area)==0.0){return ra_plane_unknown(index,RA_DETERMINANT);}
 let ri=ra_idiv(ra_anchor_direction(a,i),vk);let rj=ra_idiv(ra_anchor_direction(a,j),vk);
 let ei=ra_anchor_tangent_residual(a,i,k,ri,actualDelta,prepared.kappa);
 let ej=ra_anchor_tangent_residual(a,j,k,rj,actualDelta,prepared.kappa);
 let wi=ra_inflate(ra_iadd(ra_anchor_base(a,i,t.a[i],camera,cameraError),ra_imul(ri,delta)),ei);
 let wj=ra_inflate(ra_iadd(ra_anchor_base(a,j,t.a[j],camera,cameraError),ra_imul(rj,delta)),ej);
 if(ra_bad){return ra_plane_unknown(index,RA_INTERVAL);}
 return RAPlaneState(3u,0u,k,index,distance,actualDelta,wi,wj,ai,aj,bi,bj,area);
}
fn ra_anchor_point_error(a:RAAnchor,i:u32,k:u32,delta:RAI,actualDelta:RAI,kappa:RAI,point:f32,camera:vec3f,cameraError:vec3f)->vec2f {
 let ratio=ra_idiv(ra_anchor_direction(a,i),ra_anchor_direction(a,k));
 let residual=ra_anchor_tangent_residual(a,i,k,ratio,actualDelta,kappa);
 let offset=ra_iadd(ra_anchor_base(a,i,point,camera,cameraError),ra_imul(ratio,delta));
 let total=ra_iadd(ra_point(ra_absmax(offset)),residual);
 return vec2f(total.hi,residual.hi);
}
fn ra_anchor_finish(origin:vec3f,totalOriginError:vec3f,direction:vec3f,candidate:RGHit,prepared:RAQueryState,scanned:RAScanState,a:RAAnchor,camera:vec3f,cameraError:vec3f)->RAAnchorResult {
 ra_bad=false;
 if(prepared.valid!=1u||scanned.valid!=1u||!ra_anchor_valid(a)){return RAAnchorResult(ra_failure(prepared.reason|scanned.reason|RA_INTERVAL),a);}
 if(candidate.valid==0u){
  if(scanned.found!=0u){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
  return RAAnchorResult(RAResult(1u,0u,RI(0u,0u,0u),totalOriginError,3u,0.0,0.0),a);
 }
 if(candidate.valid!=1u||scanned.found!=1u){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
 let winner=scanned.winner;let k=winner.axis;let triangle=rg_triangle(candidate.triangle);
 if(k>=3u||candidate.point[k]!=triangle.a[k]){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
 let storedAxis=ra_component(prepared.stored,k);
 if(direction[k]==0.0||(direction[k]<0.0&&storedAxis.hi>=0.0)||(direction[k]>0.0&&storedAxis.lo<=0.0)){return RAAnchorResult(ra_failure(RA_DETERMINANT),a);}
 if(!ra_close(candidate.distance,winner.distance)){return RAAnchorResult(ra_failure(RA_NUMERIC|RA_NUMERIC_DISTANCE),a);}
 let normalResidual=ra_anchor_local_error(winner.planeDelta,prepared.kappa);
 let delta=ra_anchor_delta(a,k,triangle.a[k],camera,cameraError);
 var errors=vec3f(normalResidual.hi);var residual=errors;
 let i=(k+1u)%3u;let j=(k+2u)%3u;
 let first=ra_anchor_point_error(a,i,k,delta,winner.planeDelta,prepared.kappa,candidate.point[i],camera,cameraError);
 let second=ra_anchor_point_error(a,j,k,delta,winner.planeDelta,prepared.kappa,candidate.point[j],camera,cameraError);
 errors[i]=first.x;residual[i]=first.y;errors[j]=second.x;residual[j]=second.y;
 let audited=ra_iadd(ra_point(max(errors.x,max(errors.y,errors.z))),ra_ipoint(0x2b800000u));
 if(ra_bad){return RAAnchorResult(ra_failure(RA_INTERVAL),a);}
 if(audited.hi>bitcast<f32>(0x37a7c5acu)){return RAAnchorResult(ra_failure(RA_NUMERIC|RA_NUMERIC_POINT),a);}
 var next=a;next.axis=k;next.plane=triangle.a[k];next.residual=residual;
 return RAAnchorResult(RAResult(1u,0u,ra_export(winner.distance),errors,k,triangle.a[k],normalResidual.hi),next);
}
fn ra_anchor_reflect(a:RAAnchor)->RAAnchor {
 var next=a;
 if(!ra_anchor_valid(a)||a.axis>=3u||abs(a.plane)>256.0){next.valid=0u;return next;}
 let scaled=a.plane*0x1p16;let grid=i32(scaled);
 if(f32(grid)!=scaled){next.valid=0u;return next;}
 let value=2*grid-a.offsetGrid[a.axis];
 if(value>201326592||value< -201326592||i32(f32(value))!=value){next.valid=0u;return next;}
 next.offsetGrid[a.axis]=value;next.signMask=a.signMask^(1u<<a.axis);
 // Physical CPU residual is unchanged by the sign/offset representation update.
 return next;
}

// Opt-in anchor-tight research helpers. Existing v20 APIs above are unchanged.
// q is merely a finite representable center; no native-division accuracy is
// used to justify the identity A/B = q + (A-q*B)/B.
struct RAAnchorTightRatio { center:f32, residual:RAI, full:RAI }
fn ra_anchor_tight_intersect(a:RAI,b:RAI)->RAI {
 let lo=max(a.lo,b.lo);let hi=min(a.hi,b.hi);
 if(lo>hi){return ra_iunknown(RA_INTERVAL);}return RAI(lo,hi);
}
fn ra_anchor_tight_ratio(a:RAI,b:RAI)->RAAnchorTightRatio {
 if(!(b.lo>0.0||b.hi<0.0)){let bad=ra_iunknown(RA_INTERVAL);return RAAnchorTightRatio(0.0,bad,bad);}
 let seed=ra_div_scalar(a.lo,b.lo);
 if(ra_bad){return RAAnchorTightRatio(0.0,seed,seed);}
 let q=seed.lo;
 let residual=ra_idiv(ra_isub(a,ra_imul(ra_point(q),b)),b);
 let full=ra_iadd(ra_point(q),residual);
 return RAAnchorTightRatio(q,residual,full);
}
fn ra_anchor_tight_product_add(base:RAI,ratio:RAAnchorTightRatio,delta:RAI)->RAI {
 return ra_iadd(ra_iadd(base,ra_imul(ra_point(ratio.center),delta)),ra_imul(ratio.residual,delta));
}

fn ra_anchor_tight_plane_query(t:RGTriangle,index:u32,origin:vec3f,totalOriginError:vec3f,prepared:RAQueryState,a:RAAnchor,camera:vec3f,cameraError:vec3f)->RAPlaneState {
 ra_bad=false;
 if(prepared.valid!=1u||!ra_anchor_valid(a)||any(cameraError<vec3f(0.0))||any(cameraError>vec3f(1.0))){return ra_plane_unknown(index,RA_UNSUPPORTED);}
 let k=ra_triangle_axis(t);if(k==3u){return ra_plane_unknown(index,RA_UNSUPPORTED);}
 let qk=ra_component(prepared.query,k);let vk=ra_anchor_direction(a,k);
 let actualDelta=ra_ineg(ra_component(ra_origin_difference(origin,totalOriginError,t.a),k));
 if(qk.lo==0.0&&qk.hi==0.0){return ra_plane_value(0u,0u,index,k,RAI(0.0,0.0),actualDelta);}
 if(ra_absmin(qk)<0x1p-12||ra_absmin(vk)==0.0){return ra_plane_unknown(index,RA_DETERMINANT);}
 let delta=ra_anchor_tight_delta(a,k,t.a[k],camera,cameraError);
 var parameter:RAI;
 if(a.axis==3u){parameter=ra_idiv(ra_isub(delta,ra_symmetric(a.residual[k])),vk);}
 else if(a.axis==k){
  // Essential cancellation. The previous face is tested, not skipped.
  let separation=ra_isub(ra_point(t.a[k]),ra_point(a.plane));
  parameter=ra_idiv(ra_isub(separation,ra_symmetric(a.residual[k])),vk);
 }else{
  let previous=ra_idiv(ra_anchor_tight_delta(a,a.axis,a.plane,camera,cameraError),ra_anchor_direction(a,a.axis));
  parameter=ra_isub(ra_idiv(ra_isub(delta,ra_symmetric(a.residual[k])),vk),previous);
 }
 var distance=ra_imul(a.rawNorm,parameter);
 // Independent algebraic form retains the candidate/previous direction ratio
 // inside the subtraction before applying N/Vk. Both contain the same exact
 // N*((p-O_k)/Vk), with the physical e_k retained; intersect, never union.
 var physicalDelta=ra_isub(delta,ra_symmetric(a.residual[k]));
 if(a.axis==k){physicalDelta=ra_isub(ra_isub(ra_point(t.a[k]),ra_point(a.plane)),ra_symmetric(a.residual[k]));}
 else if(a.axis<3u){
  let previousDelta=ra_anchor_tight_delta(a,a.axis,a.plane,camera,cameraError);
  let ratio=ra_idiv(vk,ra_anchor_direction(a,a.axis));
  physicalDelta=ra_isub(ra_isub(delta,ra_imul(ratio,previousDelta)),ra_symmetric(a.residual[k]));
 }
 let normAxis=ra_idiv(a.rawNorm,vk);
 let alternative=ra_imul(normAxis,physicalDelta);
 distance=ra_anchor_tight_intersect(distance,alternative);
 // Q_k/(V_k/||raw||) differs from one by <32u. Including MT's32*kappa*u
 // and inversion gives <128*kappa*u; this does not alter admission tolerance.
 distance=ra_inflate(distance,ra_imul(ra_imul(ra_point(ra_absmax(distance)),prepared.kappa),ra_ipoint(0x28800000u)));
 if(ra_bad){return ra_plane_unknown(index,RA_INTERVAL);}
 let near=ra_ibounds(0x33d6bf94u,0x33d6bf95u);
 if(distance.hi<near.lo){return ra_plane_value(0u,0u,index,k,distance,actualDelta);}
 let i=(k+1u)%3u;let j=(k+2u)%3u;
 let ai=ra_isub(ra_point(t.b[i]),ra_point(t.a[i]));let aj=ra_isub(ra_point(t.b[j]),ra_point(t.a[j]));
 let bi=ra_isub(ra_point(t.c[i]),ra_point(t.a[i]));let bj=ra_isub(ra_point(t.c[j]),ra_point(t.a[j]));
 let area=ra_isub(ra_imul(ai,bj),ra_imul(aj,bi));
 if(ra_bad||ra_absmin(area)==0.0){return ra_plane_unknown(index,RA_DETERMINANT);}
 let ri=ra_anchor_tight_ratio(ra_anchor_direction(a,i),vk);let rj=ra_anchor_tight_ratio(ra_anchor_direction(a,j),vk);
 let ei=ra_anchor_tangent_residual(a,i,k,ri.full,actualDelta,prepared.kappa);
 let ej=ra_anchor_tangent_residual(a,j,k,rj.full,actualDelta,prepared.kappa);
 let wi=ra_inflate(ra_anchor_tight_product_add(ra_anchor_tight_base(a,i,t.a[i],camera,cameraError),ri,delta),ei);
 let wj=ra_inflate(ra_anchor_tight_product_add(ra_anchor_tight_base(a,j,t.a[j],camera,cameraError),rj,delta),ej);
 if(ra_bad){return ra_plane_unknown(index,RA_INTERVAL);}
 return RAPlaneState(3u,0u,k,index,distance,actualDelta,wi,wj,ai,aj,bi,bj,area);
}
fn ra_anchor_tight_point_error(a:RAAnchor,i:u32,k:u32,delta:RAI,actualDelta:RAI,kappa:RAI,point:f32,camera:vec3f,cameraError:vec3f)->vec2f {
 let ratio=ra_anchor_tight_ratio(ra_anchor_direction(a,i),ra_anchor_direction(a,k));
 let residual=ra_anchor_tangent_residual(a,i,k,ratio.full,actualDelta,kappa);
 let offset=ra_anchor_tight_product_add(ra_anchor_tight_base(a,i,point,camera,cameraError),ratio,delta);
 let total=ra_iadd(ra_point(ra_absmax(offset)),residual);
 return vec2f(total.hi,residual.hi);
}
fn ra_anchor_tight_finish(origin:vec3f,totalOriginError:vec3f,direction:vec3f,candidate:RGHit,prepared:RAQueryState,scanned:RAScanState,a:RAAnchor,camera:vec3f,cameraError:vec3f)->RAAnchorResult {
 ra_bad=false;
 if(prepared.valid!=1u||scanned.valid!=1u||!ra_anchor_valid(a)){return RAAnchorResult(ra_failure(prepared.reason|scanned.reason|RA_INTERVAL),a);}
 if(candidate.valid==0u){
  if(scanned.found!=0u){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
  return RAAnchorResult(RAResult(1u,0u,RI(0u,0u,0u),totalOriginError,3u,0.0,0.0),a);
 }
 if(candidate.valid!=1u||scanned.found!=1u){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
 let winner=scanned.winner;let k=winner.axis;let triangle=rg_triangle(candidate.triangle);
 if(k>=3u||candidate.point[k]!=triangle.a[k]){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
 let storedAxis=ra_component(prepared.stored,k);
 if(direction[k]==0.0||(direction[k]<0.0&&storedAxis.hi>=0.0)||(direction[k]>0.0&&storedAxis.lo<=0.0)){return RAAnchorResult(ra_failure(RA_DETERMINANT),a);}
 if(!ra_close(candidate.distance,winner.distance)){return RAAnchorResult(ra_failure(RA_NUMERIC|RA_NUMERIC_DISTANCE),a);}
 let normalResidual=ra_anchor_local_error(winner.planeDelta,prepared.kappa);
 let delta=ra_anchor_tight_delta(a,k,triangle.a[k],camera,cameraError);
 var errors=vec3f(normalResidual.hi);var residual=errors;
 let i=(k+1u)%3u;let j=(k+2u)%3u;
 let first=ra_anchor_tight_point_error(a,i,k,delta,winner.planeDelta,prepared.kappa,candidate.point[i],camera,cameraError);
 let second=ra_anchor_tight_point_error(a,j,k,delta,winner.planeDelta,prepared.kappa,candidate.point[j],camera,cameraError);
 errors[i]=first.x;residual[i]=first.y;errors[j]=second.x;residual[j]=second.y;
 let audited=ra_iadd(ra_point(max(errors.x,max(errors.y,errors.z))),ra_ipoint(0x2b800000u));
 if(ra_bad){return RAAnchorResult(ra_failure(RA_INTERVAL),a);}
 if(audited.hi>bitcast<f32>(0x37a7c5acu)){return RAAnchorResult(ra_failure(RA_NUMERIC|RA_NUMERIC_POINT),a);}
 var next=a;next.axis=k;next.plane=triangle.a[k];next.residual=residual;
 return RAAnchorResult(RAResult(1u,0u,ra_export(winner.distance),errors,k,triangle.a[k],normalResidual.hi),next);
}

// Exact Sterbenz subtraction only; other additions keep the existing padding.
fn ra_anchor_tight_add_scalar(a:f32,b:f32)->RAI {
 let x=abs(a);let y=abs(b);
 if(a==0.0||b==0.0||a == -b||((a<0.0)!=(b<0.0)&&max(x,y)<=2.0*min(x,y))){return ra_point(a+b);}
 return ra_add_scalar(a,b);
}
fn ra_anchor_tight_iadd(a:RAI,b:RAI)->RAI {
 let l=ra_anchor_tight_add_scalar(a.lo,b.lo);let h=ra_anchor_tight_add_scalar(a.hi,b.hi);return RAI(l.lo,h.hi);
}
// Exact source-grid subtraction when both grid membership and the resulting
// f32 integer conversion are checked. Otherwise preserve the old enclosure.
fn ra_anchor_tight_grid_delta(a:RAAnchor,k:u32,p:f32)->RAI {
 if(abs(p)<=4096.0){
  let scaled=p*0x1p16;let grid=i32(scaled);
  if(f32(grid)==scaled){
   let difference=grid-a.offsetGrid[k];
   if(i32(f32(difference))==difference){return ra_point(f32(difference)*0x1p-16);}
  }
 }
 return ra_isub(ra_point(p),ra_point(ra_anchor_offset(a,k)));
}
fn ra_anchor_tight_delta(a:RAAnchor,k:u32,p:f32,camera:vec3f,cameraError:vec3f)->RAI {
 let fixed=ra_anchor_tight_grid_delta(a,k,p);
 let center=select(-camera[k],camera[k],(a.signMask&(1u<<k))!=0u);
 let centered=ra_anchor_tight_iadd(fixed,ra_point(center));
 // Keep the physical camera packing radius until after center cancellation.
 return ra_iadd(centered,ra_symmetric(cameraError[k]));
}
fn ra_anchor_tight_base(a:RAAnchor,k:u32,p:f32,camera:vec3f,cameraError:vec3f)->RAI {
 return ra_ineg(ra_anchor_tight_delta(a,k,p,camera,cameraError));
}
// sqrt(S)-q = (S-q*q)/(sqrt(S)+q), q>0. The old norm is independently
// valid; its intersection with this residual form still encloses true ||raw||.
// Call after the legacy initializer without clearing its sticky failure.
fn ra_anchor_tight_refine_norm(a:RAAnchor)->RAAnchor {
 var next=a;
 if(ra_bad||!ra_anchor_valid(a)){next.valid=0u;return next;}
 let squared=ra_iadd(ra_iadd(ra_isquare(a.raw.x),ra_isquare(a.raw.y)),ra_isquare(a.raw.z));
 let q=ra_point(a.rawNorm.lo);
 let residual=ra_idiv(ra_isub(squared,ra_imul(q,q)),ra_iadd(a.rawNorm,q));
 next.rawNorm=ra_anchor_tight_intersect(a.rawNorm,ra_iadd(q,residual));
 if(ra_bad||next.rawNorm.lo<=0.0){next.valid=0u;}
 return next;
}

// Q44 immutable affine camera reconstruction. Append-only research candidate.
// No floating-point two-sum/FMA assumption. All jitter bits remain integers.
// The CPU source-order/quantization bridge is derived in camera-guard-derivation.md.
struct RACameraParameters { coefficients:array<vec4u,6>, metadata:vec4u }
struct RACameraRatio { center:f32, residual:RAI }
struct RACameraState { raw:RAVec3, ratios:array<RACameraRatio,6>, valid:u32, pad0:u32, pad1:u32, pad2:u32 }
struct RACameraProduct { low:u32, middle:u32, high:u32 }
fn rac_zero(a:vec2u)->bool{return (a.x|a.y)==0u;}
fn rac_neg(a:vec2u)->vec2u {let low=0u-a.x;return vec2u(low,~a.y+select(0u,1u,a.x==0u));}
fn rac_abs(a:vec2u)->vec2u {if((a.y&0x80000000u)!=0u){return rac_neg(a);}return a;}
fn rac_less(a:vec2u,b:vec2u)->bool{return a.y<b.y||(a.y==b.y&&a.x<b.x);}
fn rac_add(a:vec2u,b:vec2u)->vec2u {let low=a.x+b.x;return vec2u(low,a.y+b.y+select(0u,1u,low<a.x));}
fn rac_sub(a:vec2u,b:vec2u)->vec2u{return rac_add(a,rac_neg(b));}
fn rac_signed_add(a:vec2u,b:vec2u)->vec2u {
 let c=rac_add(a,b);if(((a.y^b.y)&0x80000000u)==0u&&((a.y^c.y)&0x80000000u)!=0u){ra_bad=true;}return c;
}
fn rac_shift(a:vec2u)->vec2u{return vec2u(a.x<<1u,(a.y<<1u)|(a.x>>31u));}
fn rac_mul32(a:u32,b:u32)->vec2u {
 let a0=a&65535u;let a1=a>>16u;let b0=b&65535u;let b1=b>>16u;
 let p=a0*b0;let t=a1*b0+(p>>16u);let w=t&65535u;let v=t>>16u;
 let s=a0*b1+w;return vec2u((s<<16u)|(p&65535u),a1*b1+v+(s>>16u));
}
fn rac_mul64_32(a:vec2u,b:u32)->RACameraProduct {
 let low=rac_mul32(a.x,b);let high=rac_mul32(a.y,b);let middle=low.y+high.x;
 return RACameraProduct(low.x,middle,high.y+select(0u,1u,middle<low.y));
}
fn rac_scaled_integer(a:vec2u,b:u32)->vec2u {
 let p=rac_mul64_32(rac_abs(a),b);if(p.high!=0u||(p.middle&0x80000000u)!=0u){ra_bad=true;}
 let c=vec2u(p.low,p.middle);if((a.y&0x80000000u)!=0u){return rac_neg(c);}return c;
}
fn rac_scaled_jitter(a:vec2u,b:u32)->vec2u {
 let p=rac_mul64_32(rac_abs(a),b);let c=vec2u(p.middle,p.high);
 if((c.y&0x80000000u)!=0u){ra_bad=true;}if((a.y&0x80000000u)!=0u){return rac_neg(c);}return c;
}
fn rac_coefficient(p:RACameraParameters,index:u32)->vec2u {
 let i=index*2u;return vec2u(p.coefficients[i/4u][i%4u],p.coefficients[(i+1u)/4u][(i+1u)%4u]);
}
fn rac_raw_integer(p:RACameraParameters,i:u32,x:u32,y:u32,word0:u32,word1:u32)->vec2u {
 let c=rac_coefficient(p,i*3u);let a=rac_coefficient(p,i*3u+1u);let b=rac_coefficient(p,i*3u+2u);
 var z=rac_signed_add(c,rac_scaled_integer(a,x));z=rac_signed_add(z,rac_scaled_integer(b,y));
 z=rac_signed_add(z,rac_scaled_jitter(a,word0));return rac_signed_add(z,rac_scaled_jitter(b,word1));
}
fn rac_bits(a:vec2u)->u32 {if(a.y!=0u){return 64u-countLeadingZeros(a.y);}return 32u-countLeadingZeros(a.x);}
fn rac_right(a:vec2u,s:u32)->vec2u {
 if(s==0u){return a;}if(s<32u){return vec2u((a.x>>s)|(a.y<<(32u-s)),a.y>>s);}return vec2u(a.y>>(s-32u),0u);
}
// Positive integer / 2^scale: exact directed endpoint conversion via bit fields.
fn rac_positive_bounds(a:vec2u,scale:u32)->RAI {
 if(rac_zero(a)){return RAI(0.0,0.0);}let n=rac_bits(a);let shift=select(0u,n-24u,n>24u);
 let mantissa=rac_right(a,shift).x;let normalized=select(mantissa<<(24u-n),mantissa,n>=24u);
 let exponent=i32(n)-1-i32(scale)+127;if(exponent<=0||exponent>=255){return ra_iunknown(RA_INTERVAL);}
 let low=(u32(exponent)<<23u)|(normalized&0x7fffffu);
 var exact=true;
 if(shift>0u&&shift<32u){exact=(a.x&((1u<<shift)-1u))==0u;}
 else if(shift>=32u){exact=a.x==0u&&(shift==32u||(a.y&((1u<<(shift-32u))-1u))==0u);}
 return ra_ibounds(low,low+select(1u,0u,exact));
}
fn rac_signed_bounds(a:vec2u,scale:u32)->RAI {let r=rac_positive_bounds(rac_abs(a),scale);if((a.y&0x80000000u)!=0u){return ra_ineg(r);}return r;}
fn rac_raw_bounds(a:vec2u,error:u32)->RAI {
 let lo=rac_signed_bounds(rac_signed_add(a,rac_neg(vec2u(error,0u))),44u);
 let hi=rac_signed_bounds(rac_signed_add(a,vec2u(error,0u)),44u);return RAI(lo.lo,hi.hi);
}
// Exact unsigned floor((a/b)*2^32). Preconditions: a,b<2^50, b>0.
// Each remainder is <b before shift, so its next value fits below2^51.
fn rac_quotient(a:vec2u,b:vec2u)->vec2u {
 var remainder=vec2u(0u);var quotient=vec2u(0u);
 for(var pos=96u;pos>0u;pos--){
  let bit=pos-1u;var incoming=0u;if(bit>=32u){let source=bit-32u;if(source<32u){incoming=(a.x>>source)&1u;}else{incoming=(a.y>>(source-32u))&1u;}}
  remainder=rac_shift(remainder);remainder.x|=incoming;
  if((quotient.y&0x80000000u)!=0u){ra_bad=true;}quotient=rac_shift(quotient);
  if(!rac_less(remainder,b)){remainder=rac_sub(remainder,b);quotient.x|=1u;}
 }
 return quotient;
}
fn rac_make_ratio(numerator:vec2u,denominator:vec2u,numeratorError:u32,denominatorError:u32)->RACameraRatio {
 let n=rac_abs(numerator);let d=rac_abs(denominator);
 if(numeratorError>512u||denominatorError>512u||n.y>=0x00040000u||d.y>=0x00040000u){ra_bad=true;return RACameraRatio(0.0,RAI(0,0));}
 if(rac_zero(d)||rac_less(d,vec2u(0x10000000u,0u))){ra_bad=true;return RACameraRatio(0.0,RAI(0,0));}
 let q=rac_quotient(n,d);let nbits=rac_bits(q);let shift=select(0u,nbits-24u,nbits>24u);
 let top=rac_right(q,shift);var centerInteger=top;
 for(var j=0u;j<shift;j++){centerInteger=rac_shift(centerInteger);}
 let center=rac_positive_bounds(centerInteger,32u).lo;
 let lowPart=rac_positive_bounds(rac_sub(q,centerInteger),32u);
 // Floor division contributes [0,2^-32]; retaining it is mandatory.
 var residual=ra_iadd(lowPart,RAI(0.0,0x1p-32));
 // d>=2^-16 and each source radius<=2^-35 imply d-E>d/2.
 // Thus 2*E*(1+n/d)/d is a valid conservative correction. Evaluate only
 // upward scalar endpoints, with the same old integer-step allowances.
 let da=rac_positive_bounds(d,44u);
 let qa=rac_positive_bounds(q,32u);
 let ratioUpper=ra_add_scalar(qa.hi,0x1p-32).hi;
 let onePlus=ra_add_scalar(1.0,ratioUpper).hi;
 var correction=RAI(0.0,0.0);
 if(numeratorError!=0u||denominatorError!=0u){
  let divided=ra_div_scalar(onePlus,da.lo).hi;
  correction=RAI(0.0,ra_mul_scalar(divided,0x1p-34).hi);
 }
 residual=ra_inflate(residual,correction);
 if(((numerator.y^denominator.y)&0x80000000u)!=0u){return RACameraRatio(-center,ra_ineg(residual));}
 return RACameraRatio(center,residual);
}
fn ra_camera_initialize(p:RACameraParameters,pixel:u32,word0:u32,word1:u32)->RACameraState {
 var state:RACameraState;
 // Preserve caller failure; a new initialization must explicitly reset it.
 if(ra_bad||p.metadata.z!=1u||p.metadata.x<1u||p.metadata.x>256u||p.metadata.y<1u||p.metadata.y>192u||p.metadata.w>7u||p.metadata.x>4u*p.metadata.y||p.metadata.y>4u*p.metadata.x||pixel>=p.metadata.x*p.metadata.y){return state;}
 var raw:array<vec2u,3>;var errors:array<u32,3>;
 for(var i=0u;i<3u;i++){
  // An independent triangle inequality proves every signed partial sum fits.
  let c=rac_abs(rac_coefficient(p,i*3u));let x=rac_abs(rac_coefficient(p,i*3u+1u));let y=rac_abs(rac_coefficient(p,i*3u+2u));
  if(c.y>=0x10000000u||x.y>=0x10000000u||y.y>=0x10000000u){return state;}
  let bound=rac_signed_add(rac_signed_add(c,rac_scaled_integer(x,p.metadata.x)),rac_scaled_integer(y,p.metadata.y));if(ra_bad||bound.y>=0x10000000u){return state;}
  raw[i]=rac_raw_integer(p,i,pixel%p.metadata.x,pixel/p.metadata.x,word0,word1);
  if(rac_abs(raw[i]).y>=0x00040000u){return state;}// |raw Q44|<2^50.
  errors[i]=512u;
  if((p.metadata.w&(1u<<i))!=0u){if(!rac_zero(c)||!rac_zero(x)||!rac_zero(y)){return state;}errors[i]=0u;}
 }
 state.raw=RAVec3(rac_raw_bounds(raw[0],errors[0]),rac_raw_bounds(raw[1],errors[1]),rac_raw_bounds(raw[2],errors[2]));
 for(var k=0u;k<3u;k++){
  let dk=ra_component(state.raw,k);
  if(dk.lo<=0.0&&dk.hi>=0.0){
   // Exact-zero components cannot be denominators. Cache unused zero slots.
   if(!(dk.lo==0.0&&dk.hi==0.0)){return state;}continue;
  }
  for(var i=0u;i<3u;i++){if(i==k){continue;}let index=k*2u+i-select(0u,1u,i>k);state.ratios[index]=rac_make_ratio(raw[i],raw[k],errors[i],errors[k]);}
 }
 if(!ra_bad){state.valid=1u;}return state;
}
fn ra_camera_ratio(state:RACameraState,signMask:u32,i:u32,k:u32)->RAAnchorTightRatio {
 if(state.valid!=1u||i>=3u||k>=3u){let bad=ra_iunknown(RA_INTERVAL);return RAAnchorTightRatio(0.0,bad,bad);}
 let denominator=ra_component(state.raw,k);if(!(denominator.lo>0.0||denominator.hi<0.0)){let bad=ra_iunknown(RA_INTERVAL);return RAAnchorTightRatio(0.0,bad,bad);}
 if(i==k){return RAAnchorTightRatio(1.0,RAI(0,0),RAI(1,1));}
 let index=k*2u+i-select(0u,1u,i>k);var ratio=state.ratios[index];
 if(((signMask>>i)^(signMask>>k))%2u!=0u){ratio.center=-ratio.center;ratio.residual=ra_ineg(ratio.residual);}
 return RAAnchorTightRatio(ratio.center,ratio.residual,ra_iadd(ra_point(ratio.center),ratio.residual));
}

// Camera fixed-point ratios share the original jitter and retain physical CPU residuals.
fn ra_camera_anchor_plane_query(t:RGTriangle,index:u32,origin:vec3f,totalOriginError:vec3f,prepared:RAQueryState,a:RAAnchor,camera:vec3f,cameraError:vec3f,c:RACameraState)->RAPlaneState {
 ra_bad=false;
 if(prepared.valid!=1u||!ra_anchor_valid(a)||c.valid!=1u||any(cameraError<vec3f(0.0))||any(cameraError>vec3f(1.0))){return ra_plane_unknown(index,RA_UNSUPPORTED);}
 let k=ra_triangle_axis(t);if(k==3u){return ra_plane_unknown(index,RA_UNSUPPORTED);}
 let qk=ra_component(prepared.query,k);let vk=ra_anchor_direction(a,k);
 let actualDelta=ra_ineg(ra_component(ra_origin_difference(origin,totalOriginError,t.a),k));
 if(qk.lo==0.0&&qk.hi==0.0){return ra_plane_value(0u,0u,index,k,RAI(0.0,0.0),actualDelta);}
 if(ra_absmin(qk)<0x1p-12||ra_absmin(vk)==0.0){return ra_plane_unknown(index,RA_DETERMINANT);}
 let delta=ra_anchor_tight_delta(a,k,t.a[k],camera,cameraError);
 var parameter:RAI;
 if(a.axis==3u){parameter=ra_idiv(ra_isub(delta,ra_symmetric(a.residual[k])),vk);}
 else if(a.axis==k){
  // Essential cancellation. The previous face is tested, not skipped.
  let separation=ra_isub(ra_point(t.a[k]),ra_point(a.plane));
  parameter=ra_idiv(ra_isub(separation,ra_symmetric(a.residual[k])),vk);
 }else{
  let previous=ra_idiv(ra_anchor_tight_delta(a,a.axis,a.plane,camera,cameraError),ra_anchor_direction(a,a.axis));
  parameter=ra_isub(ra_idiv(ra_isub(delta,ra_symmetric(a.residual[k])),vk),previous);
 }
 var distance=ra_imul(a.rawNorm,parameter);
 // Independent algebraic form retains the candidate/previous direction ratio
 // inside the subtraction before applying N/Vk. Both contain the same exact
 // N*((p-O_k)/Vk), with the physical e_k retained; intersect, never union.
 var physicalDelta=ra_isub(delta,ra_symmetric(a.residual[k]));
 if(a.axis==k){physicalDelta=ra_isub(ra_isub(ra_point(t.a[k]),ra_point(a.plane)),ra_symmetric(a.residual[k]));}
 else if(a.axis<3u){
  let previousDelta=ra_anchor_tight_delta(a,a.axis,a.plane,camera,cameraError);
  let ratio=ra_camera_ratio(c,a.signMask,k,a.axis);
  physicalDelta=ra_isub(ra_anchor_tight_product_add(delta,ratio,ra_ineg(previousDelta)),ra_symmetric(a.residual[k]));
 }
 let normAxis=ra_idiv(a.rawNorm,vk);
 let alternative=ra_imul(normAxis,physicalDelta);
 distance=ra_anchor_tight_intersect(distance,alternative);
 // Q_k/(V_k/||raw||) differs from one by <32u. Including MT's32*kappa*u
 // and inversion gives <128*kappa*u; this does not alter admission tolerance.
 distance=ra_inflate(distance,ra_imul(ra_imul(ra_point(ra_absmax(distance)),prepared.kappa),ra_ipoint(0x28800000u)));
 if(ra_bad){return ra_plane_unknown(index,RA_INTERVAL);}
 let near=ra_ibounds(0x33d6bf94u,0x33d6bf95u);
 if(distance.hi<near.lo){return ra_plane_value(0u,0u,index,k,distance,actualDelta);}
 let i=(k+1u)%3u;let j=(k+2u)%3u;
 let ai=ra_isub(ra_point(t.b[i]),ra_point(t.a[i]));let aj=ra_isub(ra_point(t.b[j]),ra_point(t.a[j]));
 let bi=ra_isub(ra_point(t.c[i]),ra_point(t.a[i]));let bj=ra_isub(ra_point(t.c[j]),ra_point(t.a[j]));
 let area=ra_isub(ra_imul(ai,bj),ra_imul(aj,bi));
 if(ra_bad||ra_absmin(area)==0.0){return ra_plane_unknown(index,RA_DETERMINANT);}
 let ri=ra_camera_ratio(c,a.signMask,i,k);let rj=ra_camera_ratio(c,a.signMask,j,k);
 let ei=ra_anchor_tangent_residual(a,i,k,ri.full,actualDelta,prepared.kappa);
 let ej=ra_anchor_tangent_residual(a,j,k,rj.full,actualDelta,prepared.kappa);
 let wi=ra_inflate(ra_anchor_tight_product_add(ra_anchor_tight_base(a,i,t.a[i],camera,cameraError),ri,delta),ei);
 let wj=ra_inflate(ra_anchor_tight_product_add(ra_anchor_tight_base(a,j,t.a[j],camera,cameraError),rj,delta),ej);
 if(ra_bad){return ra_plane_unknown(index,RA_INTERVAL);}
 return RAPlaneState(3u,0u,k,index,distance,actualDelta,wi,wj,ai,aj,bi,bj,area);
}
fn ra_camera_anchor_point_error(a:RAAnchor,i:u32,k:u32,delta:RAI,actualDelta:RAI,kappa:RAI,point:f32,camera:vec3f,cameraError:vec3f,c:RACameraState)->vec2f {
 let ratio=ra_camera_ratio(c,a.signMask,i,k);
 let residual=ra_anchor_tangent_residual(a,i,k,ratio.full,actualDelta,kappa);
 let offset=ra_anchor_tight_product_add(ra_anchor_tight_base(a,i,point,camera,cameraError),ratio,delta);
 let total=ra_iadd(ra_point(ra_absmax(offset)),residual);
 return vec2f(total.hi,residual.hi);
}
fn ra_camera_anchor_finish(origin:vec3f,totalOriginError:vec3f,direction:vec3f,candidate:RGHit,prepared:RAQueryState,scanned:RAScanState,a:RAAnchor,camera:vec3f,cameraError:vec3f,c:RACameraState)->RAAnchorResult {
 ra_bad=false;
 if(prepared.valid!=1u||scanned.valid!=1u||!ra_anchor_valid(a)||c.valid!=1u){return RAAnchorResult(ra_failure(prepared.reason|scanned.reason|RA_INTERVAL),a);}
 if(candidate.valid==0u){
  if(scanned.found!=0u){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
  return RAAnchorResult(RAResult(1u,0u,RI(0u,0u,0u),totalOriginError,3u,0.0,0.0),a);
 }
 if(candidate.valid!=1u||scanned.found!=1u){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
 let winner=scanned.winner;let k=winner.axis;let triangle=rg_triangle(candidate.triangle);
 if(k>=3u||candidate.point[k]!=triangle.a[k]){return RAAnchorResult(ra_failure(RA_CANDIDATE),a);}
 let storedAxis=ra_component(prepared.stored,k);
 if(direction[k]==0.0||(direction[k]<0.0&&storedAxis.hi>=0.0)||(direction[k]>0.0&&storedAxis.lo<=0.0)){return RAAnchorResult(ra_failure(RA_DETERMINANT),a);}
 if(!ra_close(candidate.distance,winner.distance)){return RAAnchorResult(ra_failure(RA_NUMERIC|RA_NUMERIC_DISTANCE),a);}
 let normalResidual=ra_anchor_local_error(winner.planeDelta,prepared.kappa);
 let delta=ra_anchor_tight_delta(a,k,triangle.a[k],camera,cameraError);
 var errors=vec3f(normalResidual.hi);var residual=errors;
 let i=(k+1u)%3u;let j=(k+2u)%3u;
 let first=ra_camera_anchor_point_error(a,i,k,delta,winner.planeDelta,prepared.kappa,candidate.point[i],camera,cameraError,c);
 let second=ra_camera_anchor_point_error(a,j,k,delta,winner.planeDelta,prepared.kappa,candidate.point[j],camera,cameraError,c);
 errors[i]=first.x;residual[i]=first.y;errors[j]=second.x;residual[j]=second.y;
 let audited=ra_iadd(ra_point(max(errors.x,max(errors.y,errors.z))),ra_ipoint(0x2b800000u));
 if(ra_bad){return RAAnchorResult(ra_failure(RA_INTERVAL),a);}
 if(audited.hi>bitcast<f32>(0x37a7c5acu)){return RAAnchorResult(ra_failure(RA_NUMERIC|RA_NUMERIC_POINT),a);}
 var next=a;next.axis=k;next.plane=triangle.a[k];next.residual=residual;
 return RAAnchorResult(RAResult(1u,0u,ra_export(winner.distance),errors,k,triangle.a[k],normalResidual.hi),next);
}
