// Single local region: Gaussian detector mode -> two finite slits -> coherent source.
// Across modes only nonnegative POWERS accumulate. mm units. See method.html.
struct Params { dims:vec4f, physical:vec4f, beam:vec4f, sampling:vec4f };
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> accumulation:array<vec4f>;
const PI:f32=3.141592653589793;
const N=array<f32,16>(-0.989400935,-0.944575023,-0.865631202,-0.755404408,-0.617876244,-0.458016778,-0.281603551,-0.095012510,0.095012510,0.281603551,0.458016778,0.617876244,0.755404408,0.865631202,0.944575023,0.989400935);
const W=array<f32,16>(0.027152459,0.062253524,0.095158512,0.124628971,0.149595989,0.169156519,0.182603415,0.189450610,0.189450610,0.182603415,0.169156519,0.149595989,0.124628971,0.095158512,0.062253524,0.027152459);
fn hash(x:u32)->u32 { var v=x;v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u); }
fn normal2(pixel:u32,sample:u32)->vec2f {let s=hash(pixel+(sample+1u)*747796405u);let u=max((f32(hash(s))+0.5)/4294967296.0,1e-8);let v=(f32(hash(s^0x68bc21ebu))+0.5)/4294967296.0;let r=sqrt(-2.0*log(u));return r*vec2f(cos(2.0*PI*v),sin(2.0*PI*v));}
fn overlap(lo:f32,hi:f32,X:f32,xi:f32)->vec2f {
 let lambda=p.physical.x;let L=p.physical.y;let k=2.0*PI/lambda;let rho=L/k;let S=2.0*p.beam.x*p.beam.x;let D=S*S+rho*rho;let sigmaL=sqrt(D/(2.0*S));
 // Rectangle cutoff lies inside the conservative elliptical cone.
 let c=-sigmaL*xi;let a=max(lo,c-p.beam.w*sigmaL);let b=min(hi,c+p.beam.w*sigmaL);if(b<=a){return vec2f(0.0);}
 let halfWidth=(b-a)*0.5;let mid=(a+b)*0.5;let beta=rho/(2.0*D);let gamma=-k*X/L-xi*sigmaL*S*S/(rho*D);var value=vec2f(0.0);
 for(var j=0u;j<16u;j++){let t=mid+halfWidth*N[j];let amp=exp(-t*t/(4.0*p.beam.y*p.beam.y)-t*t/(4.0*sigmaL*sigmaL)-xi*t/(2.0*sigmaL));let phase=beta*t*t+gamma*t;value+=W[j]*amp*vec2f(cos(phase),sin(phase));}return halfWidth*value;
}
// Specialised paraxial cone-plane query, followed by ellipse-rectangle collection.
// Inflated axes conservatively contain the exact tensor cutoff throughout propagation.
fn coneHits(lo:f32,hi:f32,halfHeight:f32,xi:vec2f)->bool {
 let k=2.0*PI/p.physical.x;let L=p.physical.y;let sigma=p.beam.x;let rho=L/k;let sigmaL=sqrt(sigma*sigma+rho*rho/(4.0*sigma*sigma));
 let centre=-sigmaL*xi;let axis=sqrt(2.0)*p.beam.w*(sigma+L/(2.0*k*sigma));
 let closest=clamp(centre,vec2f(lo,-halfHeight),vec2f(hi,halfHeight));let r=(closest-centre)/axis;return dot(r,r)<=1.0;
}
fn filterInterval(lo:f32,hi:f32,X:f32)->f32{let mid=(lo+hi)*0.5;let h=(hi-lo)*0.5;var v=0.0;for(var j=0u;j<16u;j++){let t=mid+h*N[j];v+=W[j]*exp(-t*t/(2.0*p.beam.y*p.beam.y)-(t-X)*(t-X)/(2.0*p.beam.x*p.beam.x));}return v*h/(sqrt(2.0*PI)*p.beam.x);}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3u){let width=u32(p.dims.x);let height=u32(p.dims.y);if(id.x>=width||id.y>=height){return;}let index=id.y*width+id.x;let X=((f32(id.x)+0.5)/f32(width)-0.5)*p.dims.z;let Y=((f32(id.y)+0.5)/f32(height)-0.5)*p.dims.w;
 let halfSlit=p.physical.z*0.5;let halfSeparation=p.physical.w*0.5;let halfHeight=p.beam.z*0.5;var sum=0.0;var sum2=0.0;
 for(var s=0u;s<u32(p.sampling.y);s++){let xi=normal2(index,u32(p.sampling.x)+s);var ax=vec2f(0.0);if(coneHits(-halfSeparation-halfSlit,-halfSeparation+halfSlit,halfHeight,xi)){ax+=overlap(-halfSeparation-halfSlit,-halfSeparation+halfSlit,X,xi.x);}if(coneHits(halfSeparation-halfSlit,halfSeparation+halfSlit,halfHeight,xi)){ax+=overlap(halfSeparation-halfSlit,halfSeparation+halfSlit,X,xi.x);}let ay=overlap(-halfHeight,halfHeight,Y,xi.y);let v=dot(ax,ax)*dot(ay,ay)/pow(p.physical.x*p.physical.y,2.0);sum+=v;sum2+=v*v;}
 let geo=(filterInterval(-halfSeparation-halfSlit,-halfSeparation+halfSlit,X)+filterInterval(halfSeparation-halfSlit,halfSeparation+halfSlit,X))*filterInterval(-halfHeight,halfHeight,Y);
 var old=accumulation[index];if(p.sampling.x==0.0){old=vec4f(0.0);}accumulation[index]=vec4f(old.x+sum,old.y+sum2,geo,old.w+p.sampling.y);
}
