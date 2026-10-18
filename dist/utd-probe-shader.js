// Separate diagnostic entrypoint; the production shader files remain unchanged.
export const UTD_PROBE_SHADER=String.raw`// Diagnostic entrypoint only. Append to the original complete shader.
// Same six bindings; dispatch exactly 6 workgroups; requires at least 204 vec4 outputs.
// Host must replace UTD_DIAGNOSTIC_PIXELS with three selected pixel coordinates.
const UTD_DIAGNOSTIC_PIXELS = array<vec2u,3>(vec2u(0u,16u),vec2u(8u,16u),vec2u(16u,16u));
@compute @workgroup_size(1)
fn utd_diagnostic_probe(@builtin(global_invocation_id) id:vec3u) {
  let sample=id.x;let edgeLimit=min(2u,up.counts.z);if(edgeLimit==0u||sample>=3u*edgeLimit){return;}
  let which=sample/edgeLimit;let pixel=UTD_DIAGNOSTIC_PIXELS[which];let x=pixel.x;let y=pixel.y;if(x>=up.dims.x||y>=up.dims.y){return;}
  let dst=up.detector.xyz+((f32(x)+0.5)/f32(up.dims.x)-0.5)*up.detector.w*up.detectorU.xyz+
    (0.5-(f32(y)+0.5)/f32(up.dims.y))*up.detectorU.w*up.detectorV.xyz;
  let edge=uedge[up.counts.y+sample%edgeLimit];let h=utd_stationary(edge,dst);let angles=utd_angles(edge,h);
  let n=2.0-edge.geometry.x/UTD_PI;let L=h.ro*h.sinBeta*h.sinBeta;let k=up.controls.x;
  let excess=utd_stable_excess(h.point,dst);let theta=-k*excess;let circle=utd_cis(theta);let phase=circle/sqrt(h.ro);
  let ds=utd_coefficient(k,n,angles.x,angles.y,L,h.sinBeta,-1.0);
  let dh=utd_coefficient(k,n,angles.x,angles.y,L,h.sinBeta,1.0);
  let direct=utd_direct(dst);let reflections=utd_reflections(dst);
  let goSoft=vec2f(direct.weight,0.0)+reflections.soft;let goHard=vec2f(direct.weight,0.0)+reflections.hard;
  let base=sample*34u;if(base+33u>=arrayLength(&uout)){return;}
  uout[base+0u]=vec4f(dst,f32(x));
  uout[base+1u]=vec4f(h.point,h.ro);
  uout[base+2u]=vec4f(h.wo,h.sinBeta);
  uout[base+3u]=vec4f(angles,n,L);
  uout[base+4u]=vec4f(excess,theta,phase);
  uout[base+5u]=vec4f(ds,dh);
  uout[base+6u]=vec4f(utd_cmul(ds,phase),utd_cmul(dh,phase));
  uout[base+7u]=vec4f(direct.weight,f32(reflections.count),dot(circle,circle),utd_power(goSoft,goHard));
  let diff=angles.y-angles.x;let sum=angles.y+angles.x;let kL=k*L;
  for(var branch=0u;branch<4u;branch=branch+1u){
    let beta=select(diff,sum,branch>=2u);let side=select(1.0,-1.0,(branch&1u)!=0u);
    let nearest=floor((beta+side*UTD_PI)/(2.0*UTD_PI*n)+0.5);
    let delta=beta-(2.0*UTD_PI*n*nearest-side*UTD_PI);
    let t=delta*0.5;let hAngle=delta/(2.0*n);
    let ratio=sign(delta)*n*utd_sinc(t)*utd_cos_small(hAngle)/utd_sinc(hAngle);
    let rootKL=sqrt(kL);let u=sqrt(2.0)*rootKL*abs(t*utd_sinc(t));let inv=1.0/(1.0+u);let z=u*inv;
    let H=utd_erfcx_scaled(z);let cot=utd_cot_transition(beta,n,kL,side);
    let row=base+8u+4u*branch;
    uout[row]=vec4f(beta,side,nearest,delta);
    uout[row+1u]=vec4f(t,hAngle,ratio,u);
    uout[row+2u]=vec4f(z,H,kL);
    uout[row+3u]=vec4f(cot,utd_sinc(t),utd_cos_small(hAngle));
  }
  let wi=-up.incident.xyz;
  uout[base+24u]=vec4f(dot(wi,edge.nff.xyz),dot(wi,edge.tff.xyz),dot(h.wo,edge.nff.xyz),dot(h.wo,edge.tff.xyz));
  uout[base+25u]=vec4f(utd_erfcx_scaled(0.4461701888069467),utd_erfcx_scaled(0.6392366981032546));
  uout[base+26u]=vec4f(utd_erfcx_scaled(0.812819295910126),utd_erfcx_scaled(0.9860957183805704));
  uout[base+27u]=vec4f(k,L,h.sinBeta,2.0*n*sqrt(2.0*UTD_PI*k)*h.sinBeta);
  var soft=goSoft;var hard=goHard;
  for(var j=0u;j<edgeLimit;j=j+1u){
    let ej=uedge[up.counts.y+j];let hj=utd_stationary(ej,dst);let aj=utd_angles(ej,hj);
    let nj=2.0-ej.geometry.x/UTD_PI;let Lj=hj.ro*hj.sinBeta*hj.sinBeta;
    let pj=utd_cis(-k*utd_stable_excess(hj.point,dst))/sqrt(hj.ro);
    soft=soft+utd_cmul(utd_coefficient(k,nj,aj.x,aj.y,Lj,hj.sinBeta,-1.0),pj);
    hard=hard+utd_cmul(utd_coefficient(k,nj,aj.x,aj.y,Lj,hj.sinBeta,1.0),pj);
  }
  uout[base+28u]=vec4f(soft,hard);
  uout[base+29u]=vec4f(utd_power(soft,hard),utd_power(goSoft,goHard),f32(direct.invalid+reflections.invalid),up.controls.y);
  uout[base+30u]=vec4f(circle,utd_sinc(0.024979197860971247),utd_cos_small(0.012489598930485624));
  uout[base+31u]=vec4f(sqrt(kL),1.0/sqrt(h.ro),sin(theta),cos(theta));
  uout[base+32u]=vec4f(atan2(dot(wi,edge.nff.xyz),dot(wi,edge.tff.xyz)),atan2(dot(h.wo,edge.nff.xyz),dot(h.wo,edge.tff.xyz)),sqrt(h.ro),length(edge.b.xyz-edge.a.xyz));
  uout[base+33u]=vec4f(exp(-excess),sqrt(k),fma(1.0000001192092896,k,-k),1.0000001192092896*k-k);
}
`;
