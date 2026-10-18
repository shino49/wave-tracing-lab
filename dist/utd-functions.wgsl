// Independently implemented scalar UTD, exp(+i omega t), outgoing exp(-ikr).
// Equations: Kouyoumjian & Pathak (1974), NASA CR-2401; arxiv:1905.10562.
// This is a shader include: prepend to the renderer shader; no bindings/entrypoint.
// Exterior wedge index 1 <= n <= 2; positive k,L; 0 < sinBeta <= 1.
// complex = vec2f(real, imaginary); boundaryR = -1 soft, +1 hard.
const UTD_PI:f32 = 3.14159265358979323846;
const UTD_SQRT_HALF:f32 = 0.70710678118654752440;
// A narrow angular seam matches GO half weights with the cot pole convention.
// Call utd_step for accompanying GO weights; do NOT independently use 0/1 there.
const UTD_BOUNDARY_EPS:f32 = 0.000002;

// BEGIN GENERATED COEFFICIENTS
const UTD_ERFCX_CHEBYSHEV = array<vec2f,41>(
  vec2f(0.75393091236043497,-0.31228840899201349),
  vec2f(-0.34180228289066317,-0.21798615414677372),
  vec2f(-0.062861325985977309,0.12853057714762112),
  vec2f(0.047109696155292445,0.02288046950922288),
  vec2f(0.010703996179539949,-0.017747532128594992),
  vec2f(-0.0064420676756608035,-0.0055503724079850793),
  vec2f(-0.0028813059466919356,0.0021409415472675246),
  vec2f(0.00056862318075534546,0.0014472723328808241),
  vec2f(0.00068563701037428727,-4.9450766296487991e-05),
  vec2f(8.00690605642227e-05,-0.0002986107478057238),
  vec2f(-0.0001146212286691504,-8.2426630994954068e-05),
  vec2f(-5.5315199850017518e-05,3.4793481443824278e-05),
  vec2f(4.6902690135752497e-06,3.0251467084912864e-05),
  vec2f(1.3959134565580604e-05,3.9150513837218029e-06),
  vec2f(4.5647548774284433e-06,-5.2200874408738724e-06),
  vec2f(-1.2752390761123345e-06,-3.0947278460898653e-06),
  vec2f(-1.6233345797968494e-06,-1.3924122021516045e-07),
  vec2f(-4.352971400272024e-07,6.7038048773959425e-07),
  vec2f(1.8646172777348196e-07,3.4821628848131346e-07),
  vec2f(1.9647829765595673e-07,4.6609894507684354e-09),
  vec2f(5.147218906924492e-08,-8.3882569903653972e-08),
  vec2f(-2.3033023669921256e-08,-4.3906671585754552e-08),
  vec2f(-2.5070336140409211e-08,-1.6117347429578107e-09),
  vec2f(-7.4466950221350794e-09,1.0408598148692869e-08),
  vec2f(2.4550536874448571e-09,6.0762871279256173e-09),
  vec2f(3.3114303223621766e-09,6.3809059391308982e-10),
  vec2f(1.2172580910720608e-09,-1.2482353240459216e-09),
  vec2f(-1.867371176388653e-10,-8.8597003175606683e-10),
  vec2f(-4.3678666374836994e-10,-1.7949659915598621e-10),
  vec2f(-2.0738174727610249e-10,1.3473300388645825e-10),
  vec2f(-4.6004622875411713e-12,1.2994487058197887e-10),
  vec2f(5.4487417288112008e-11,4.1717867642783561e-11),
  vec2f(3.4800487694804736e-11,-1.049995185648915e-11),
  vec2f(6.3970482028084032e-12,-1.8233612451879475e-11),
  vec2f(-5.7958583731823269e-12,-8.5205149169271914e-12),
  vec2f(-5.5106481708434545e-12,-2.4183839213052823e-13),
  vec2f(-1.8959549377627367e-12,2.2833530459352479e-12),
  vec2f(3.4742500204656359e-13,1.5486077867209485e-12),
  vec2f(7.7579135527563137e-13,3.381813799186708e-13),
  vec2f(4.3933572904325674e-13,-2.4694220245882412e-13),
  vec2f(1.2292504412743751e-13,-2.5116248850357489e-13)
);

// Same degree-40 Chebyshev polynomial and coefficients as the f64 reference.
// Separate scalar streams and literal indices avoid vector/dynamic-array
// lowering in this numerically sensitive recurrence; no change to UTD physics.
fn utd_erfcx_scaled(t:f32)->vec2f {
 let s=2.0*t-1.0;let x=2.0*s;
 var br=0.0;var bi=0.0;var cr=0.0;var ci=0.0;
 // coefficient 40
{let nr=x*br-cr+(1.2292504412743751e-13);let ni=x*bi-ci+(-2.5116248850357489e-13);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 39
{let nr=x*br-cr+(4.3933572904325674e-13);let ni=x*bi-ci+(-2.4694220245882412e-13);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 38
{let nr=x*br-cr+(7.7579135527563137e-13);let ni=x*bi-ci+(3.381813799186708e-13);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 37
{let nr=x*br-cr+(3.4742500204656359e-13);let ni=x*bi-ci+(1.5486077867209485e-12);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 36
{let nr=x*br-cr+(-1.8959549377627367e-12);let ni=x*bi-ci+(2.2833530459352479e-12);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 35
{let nr=x*br-cr+(-5.5106481708434545e-12);let ni=x*bi-ci+(-2.4183839213052823e-13);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 34
{let nr=x*br-cr+(-5.7958583731823269e-12);let ni=x*bi-ci+(-8.5205149169271914e-12);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 33
{let nr=x*br-cr+(6.3970482028084032e-12);let ni=x*bi-ci+(-1.8233612451879475e-11);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 32
{let nr=x*br-cr+(3.4800487694804736e-11);let ni=x*bi-ci+(-1.049995185648915e-11);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 31
{let nr=x*br-cr+(5.4487417288112008e-11);let ni=x*bi-ci+(4.1717867642783561e-11);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 30
{let nr=x*br-cr+(-4.6004622875411713e-12);let ni=x*bi-ci+(1.2994487058197887e-10);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 29
{let nr=x*br-cr+(-2.0738174727610249e-10);let ni=x*bi-ci+(1.3473300388645825e-10);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 28
{let nr=x*br-cr+(-4.3678666374836994e-10);let ni=x*bi-ci+(-1.7949659915598621e-10);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 27
{let nr=x*br-cr+(-1.867371176388653e-10);let ni=x*bi-ci+(-8.8597003175606683e-10);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 26
{let nr=x*br-cr+(1.2172580910720608e-09);let ni=x*bi-ci+(-1.2482353240459216e-09);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 25
{let nr=x*br-cr+(3.3114303223621766e-09);let ni=x*bi-ci+(6.3809059391308982e-10);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 24
{let nr=x*br-cr+(2.4550536874448571e-09);let ni=x*bi-ci+(6.0762871279256173e-09);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 23
{let nr=x*br-cr+(-7.4466950221350794e-09);let ni=x*bi-ci+(1.0408598148692869e-08);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 22
{let nr=x*br-cr+(-2.5070336140409211e-08);let ni=x*bi-ci+(-1.6117347429578107e-09);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 21
{let nr=x*br-cr+(-2.3033023669921256e-08);let ni=x*bi-ci+(-4.3906671585754552e-08);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 20
{let nr=x*br-cr+(5.147218906924492e-08);let ni=x*bi-ci+(-8.3882569903653972e-08);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 19
{let nr=x*br-cr+(1.9647829765595673e-07);let ni=x*bi-ci+(4.6609894507684354e-09);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 18
{let nr=x*br-cr+(1.8646172777348196e-07);let ni=x*bi-ci+(3.4821628848131346e-07);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 17
{let nr=x*br-cr+(-4.352971400272024e-07);let ni=x*bi-ci+(6.7038048773959425e-07);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 16
{let nr=x*br-cr+(-1.6233345797968494e-06);let ni=x*bi-ci+(-1.3924122021516045e-07);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 15
{let nr=x*br-cr+(-1.2752390761123345e-06);let ni=x*bi-ci+(-3.0947278460898653e-06);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 14
{let nr=x*br-cr+(4.5647548774284433e-06);let ni=x*bi-ci+(-5.2200874408738724e-06);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 13
{let nr=x*br-cr+(1.3959134565580604e-05);let ni=x*bi-ci+(3.9150513837218029e-06);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 12
{let nr=x*br-cr+(4.6902690135752497e-06);let ni=x*bi-ci+(3.0251467084912864e-05);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 11
{let nr=x*br-cr+(-5.5315199850017518e-05);let ni=x*bi-ci+(3.4793481443824278e-05);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 10
{let nr=x*br-cr+(-0.0001146212286691504);let ni=x*bi-ci+(-8.2426630994954068e-05);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 9
{let nr=x*br-cr+(8.00690605642227e-05);let ni=x*bi-ci+(-0.0002986107478057238);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 8
{let nr=x*br-cr+(0.00068563701037428727);let ni=x*bi-ci+(-4.9450766296487991e-05);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 7
{let nr=x*br-cr+(0.00056862318075534546);let ni=x*bi-ci+(0.0014472723328808241);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 6
{let nr=x*br-cr+(-0.0028813059466919356);let ni=x*bi-ci+(0.0021409415472675246);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 5
{let nr=x*br-cr+(-0.0064420676756608035);let ni=x*bi-ci+(-0.0055503724079850793);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 4
{let nr=x*br-cr+(0.010703996179539949);let ni=x*bi-ci+(-0.017747532128594992);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 3
{let nr=x*br-cr+(0.047109696155292445);let ni=x*bi-ci+(0.02288046950922288);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 2
{let nr=x*br-cr+(-0.062861325985977309);let ni=x*bi-ci+(0.12853057714762112);cr=br;ci=bi;br=nr;bi=ni;}
 // coefficient 1
{let nr=x*br-cr+(-0.34180228289066317);let ni=x*bi-ci+(-0.21798615414677372);cr=br;ci=bi;br=nr;bi=ni;}
 return vec2f(s*br-cr+(0.75393091236043497),s*bi-ci+(-0.31228840899201349));
}
// END GENERATED COEFFICIENTS

fn utd_cmul(ca:vec2f,cb:vec2f)->vec2f { return vec2f(ca.x*cb.x-ca.y*cb.y,ca.x*cb.y+ca.y*cb.x); }
fn utd_atan2_axes(y:f32,x:f32)->f32 {
  // WGSL does not bound atan2 accuracy when x or y is zero. Handle exact
  // axes before calling the intrinsic, including either sign of zero.
  // The origin has no direction; edge-parallel geometry is masked elsewhere.
  if(x==0.0) {
    if(y>0.0){return 0.5*UTD_PI;}
    if(y<0.0){return -0.5*UTD_PI;}
    return 0.0;
  }
  if(y==0.0) {
    if(x<0.0) {
      let negativeY=(bitcast<u32>(y)&0x80000000u)!=0u;
      return select(UTD_PI,-UTD_PI,negativeY);
    }
    return y;
  }
  return atan2(y,x);
}
fn utd_cis(theta:f32)->vec2f {
  // Local phase bound <= 100000 rad is checked before GPU packing.
  // Split pi/2: 201/128 is exact, and q*201 remains exactly representable.
  let quadrant=i32(round(theta*0.63661977236758134308));
  let q=f32(quadrant);
  let coarse=fma(-q,1.5703125,theta);
  let reduced=fma(-q,0.00048382679489661923,coarse);
  let c=cos(reduced);let s=sin(reduced);
  switch(quadrant&3) {
    case 0: { return vec2f(c,s); }
    case 1: { return vec2f(-s,c); }
    case 2: { return vec2f(-c,-s); }
    default: { return vec2f(s,-c); }
  }
}
fn utd_sinc(x:f32)->f32 {
  if(abs(x)<=0.5) {
    let q=x*x;
    return 1.0+q*(-0.16666666666666666667+q*(0.00833333333333333333+q*(-0.00019841269841269841+q*0.00000275573192239858907)));
  }
  return sin(x)/x;
}
fn utd_cos_small(x:f32)->f32 {
  if(abs(x)<=0.5) {
    let q=x*x;
    return 1.0+q*(-0.5+q*(0.04166666666666666667+q*(-0.00138888888888888889+q*(0.00002480158730158730-q*0.00000027557319223986))));
  }
  return cos(x);
}
// The scalar erfcx evaluator is generated beside its coefficients above.
fn utd_transition(x:f32)->vec2f {
  if(x<=0.0) { return vec2f(0.0); }
  let u=sqrt(x); let t=u/(1.0+u);
  return sqrt(UTD_PI)*t*utd_cmul(vec2f(UTD_SQRT_HALF),utd_erfcx_scaled(t));
}
fn utd_cot_transition(beta:f32,n:f32,kL:f32,side:f32)->vec2f {
  let nearest=floor((beta+side*UTD_PI)/(2.0*UTD_PI*n)+0.5);
  let delta=beta-(2.0*UTD_PI*n*nearest-side*UTD_PI);
  if(abs(delta)<=UTD_BOUNDARY_EPS) { return vec2f(0.0); }
  let t=delta*0.5; let h=delta/(2.0*n);
  let ratio=sign(delta)*n*utd_sinc(t)*utd_cos_small(h)/utd_sinc(h);
  let rootKL=sqrt(kL); let u=sqrt(2.0)*rootKL*abs(t*utd_sinc(t));
  let inv=1.0/(1.0+u);
  return side*sqrt(2.0*UTD_PI)*(rootKL*inv)*ratio*
    utd_cmul(vec2f(UTD_SQRT_HALF),utd_erfcx_scaled(u*inv));
}
fn utd_coefficient(k:f32,n:f32,phi_i:f32,phi_o:f32,L:f32,sinBeta:f32,boundaryR:f32)->vec2f {
  if(n==1.0) { return vec2f(0.0); }
  let d=phi_o-phi_i; let s=phi_o+phi_i; let kL=k*L;
  let a=utd_cot_transition(d,n,kL,1.0)+utd_cot_transition(d,n,kL,-1.0);
  let b=utd_cot_transition(s,n,kL,1.0)+utd_cot_transition(s,n,kL,-1.0);
  return -utd_cmul(vec2f(UTD_SQRT_HALF,-UTD_SQRT_HALF),a+boundaryR*b)/
    (2.0*n*sqrt(2.0*UTD_PI*k)*sinBeta);
}
fn utd_step(x:f32)->f32 {
  if(abs(x)<=UTD_BOUNDARY_EPS) { return 0.5; }
  return select(0.0,1.0,x>0.0);
}
fn utd_go_plane(k:f32,r:f32,n:f32,phi_i:f32,phi_o:f32,boundaryR:f32)->vec2f {
  let d=phi_o-phi_i; let s=phi_o+phi_i;
  return utd_cis(k*r*cos(d))*utd_step(UTD_PI-abs(d))+
    boundaryR*utd_cis(k*r*cos(s))*utd_step(UTD_PI-s)+
    boundaryR*utd_cis(k*r*cos(s-2.0*n*UTD_PI))*utd_step(s-(2.0*n*UTD_PI-UTD_PI));
}
fn utd_total_plane(k:f32,r:f32,n:f32,phi_i:f32,phi_o:f32,boundaryR:f32)->vec2f {
  return utd_go_plane(k,r,n,phi_i,phi_o,boundaryR)+
    utd_cmul(utd_coefficient(k,n,phi_i,phi_o,r,1.0,boundaryR),utd_cis(-k*r))/sqrt(r);
}
