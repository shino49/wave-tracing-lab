/** Bounded 2-D scalar Helmholtz reference, not native Wave Tracing/UTD.
 * exp(+i omega t), G=-i H_0^(2)(kr)/4; soft finite screens, fully coupled solve.
 * All numerical state is Float64. Source covariance is retained as L L†;
 * images show sum |u L|², whereas angular measurements use k dtheta flux.
 * Bessel rational coefficients: Cephes, Copyright 1984, 1987, 1989, 2000
 * Stephen L. Moshier; SciPy XSF https://github.com/scipy/xsf (BSD-3-Clause).
 * Algorithm/coefficients verified against official xsf/cephes/j0.h.
 */
/*
 * Bessel coefficient adaptation: BSD 3-Clause License
 * Copyright (c) 2024, SciPy
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

const PI=Math.PI, EULER=.5772156649015328606, MODES=[[-2,-.55,0,1.25],[-1.2,.6,-.65,1.4],[-2.6,.2,.8,.9]];
export const COHERENT_SCOPE='Coupled 2D scalar Helmholtz finite-screen reference; scalar flux, not electromagnetic watts; not native Wave Tracing or UTD';
export const defaultState=Object.freeze({gamma:1,phase:0,inputPower:1,detectorCenter:0,detectorHalfAngle:.55});
export const DEFAULT_OPTIONS=Object.freeze({screenCount:2,separation:2.3,offset:.4,n:64,nq:320,nang:160,...defaultState});
export const COHERENT_LIMITS=Object.freeze({k:12,qmax:9,separation:[1.8,3],offset:[-.2,1],nodes:[64,96,144],maxGridPoints:100000,maskDistanceNumerator:12});
const PP=[7.96936729297347051624e-4,8.28352392107440799803e-2,1.23953371646414299388,5.44725003058768775090,8.74716500199817011941,5.30324038235394892183,.999999999999999997821];
const PQ=[9.24408810558863637013e-4,8.56288474354474431428e-2,1.25352743901058953537,5.47097740330417105182,8.76190883237069594232,5.30605288235394617618,1.00000000000000000218];
const QP=[-1.13663838898469149931e-2,-1.28252718670509318512,-1.95539544257735972385e1,-9.32060152123768231369e1,-1.77681167980488050595e2,-1.47077505154951170175e2,-5.14105326766599330220e1,-6.05014350600728481186];
const QQ=[6.43178256118178023184e1,8.56430025976980587198e2,3.88240183605401609683e3,7.24046774195652478189e3,5.93072701187316984827e3,2.06209331660327847417e3,2.42005740240291393179e2];
const YP=[1.55924367855235737965e4,-1.46639295903971606143e7,5.43526477051876500413e9,-9.82136065717911466409e11,8.75906394395366999549e13,-3.46628303384729719441e15,4.42733268572569800351e16,-1.84950800436986690637e16];
const YQ=[1.04128353664259848412e3,6.26107330137134956842e5,2.68919633393814121987e8,8.64002487103935000337e10,2.02979612750105546709e13,3.17157752842975028269e15,2.50596256172653059228e17];
const RP=[-4.79443220978201773821e9,1.95617491946556577543e12,-2.49248344360967716204e14,9.70862251047306323952e15];
const RQ=[4.99563147152651017219e2,1.73785401676374683123e5,4.84409658339962045305e7,1.11855537045356834862e10,2.11277520115489217587e12,3.10518229857422583814e14,3.18121955943204943306e16,1.71086294081043136091e18];
const pol=(x,c)=>c.reduce((s,v)=>s*x+v,0), pol1=(x,c)=>c.reduce((s,v)=>s*x+v,1);
export function besselJ0(x){
 x=Math.abs(x); if(x<=5){if(x<1e-5)return 1-x*x/4; const z=x*x;return (z-5.78318596294678452118)*(z-30.4712623436620863991)*pol(z,RP)/pol1(z,RQ);}
 const w=5/x,z=w*w,p=pol(z,PP)/pol(z,PQ),q=pol(z,QP)/pol1(z,QQ);
 return ((p+w*q)*Math.cos(x)+(p-w*q)*Math.sin(x))/Math.sqrt(PI*x);
}
export function besselY0(x){
 if(x===0)return -Infinity;if(x<0)return NaN;
 if(x<=5)return pol(x*x,YP)/pol1(x*x,YQ)+2/PI*Math.log(x)*besselJ0(x);
 const w=5/x,z=w*w,p=pol(z,PP)/pol(z,PQ),q=pol(z,QP)/pol1(z,QQ);
 return ((w*q-p)*Math.cos(x)+(p+w*q)*Math.sin(x))/Math.sqrt(PI*x);
}
export function green(r,k=12){return [-.25*besselY0(k*r),-.25*besselJ0(k*r)];}
const bounded=(v,name,lo,hi)=>{if(!Number.isFinite(v)||v<lo||v>hi)throw new RangeError(`${name} must be in [${lo},${hi}]`);return v;};
function checkedOptions(o={}){
 const c={...DEFAULT_OPTIONS,...o};
 if(![1,2].includes(c.screenCount))throw new RangeError('screenCount must be 1 or 2');
 bounded(c.separation,'separation',1.8,3);bounded(c.offset,'offset',-.2,1);
 if(![64,96,144].includes(c.n))throw new RangeError('n must be 64, 96, or 144');
 if(!Number.isInteger(c.nq)||c.nq<160||c.nq>640)throw new RangeError('nq must be an integer in [160,640]');
 if(!Number.isInteger(c.nang)||c.nang<80||c.nang>320)throw new RangeError('nang must be an integer in [80,320]');
 checkState(c);return c;
}
function checkState(o={}){
 const c={...defaultState,...o};bounded(c.gamma,'gamma',0,1);bounded(c.phase,'phase',-1000,1000);bounded(c.inputPower,'inputPower',0,1e6);
 bounded(c.detectorCenter,'detectorCenter',-PI,PI);bounded(c.detectorHalfAngle,'detectorHalfAngle',.01,PI);
 return c;
}
export function gaussLegendre(a,b,n){
 const x=new Float64Array(n),w=new Float64Array(n),mid=(a+b)/2,half=(b-a)/2;
 for(let i=0;i<Math.ceil(n/2);i++){
  let z=Math.cos(PI*(i+.75)/(n+.5)),pp=0;
  for(let it=0;it<50;it++){let p1=1,p2=0;for(let j=1;j<=n;j++){const p3=p2;p2=p1;p1=((2*j-1)*z*p2-(j-1)*p3)/j;}pp=n*(z*p1-p2)/(z*z-1);const dz=p1/pp;z-=dz;if(Math.abs(dz)<2e-16)break;}
  // Recompute derivative at final root, avoiding stale weights.
  let p1=1,p2=0;for(let j=1;j<=n;j++){const p3=p2;p2=p1;p1=((2*j-1)*z*p2-(j-1)*p3)/j;}pp=n*(z*p1-p2)/(z*z-1);
  x[i]=mid-half*z;x[n-1-i]=mid+half*z;w[i]=w[n-1-i]=2*half/((1-z*z)*pp*pp);
 }return {x,w};
}
export function multiplyComplex(a,b,rows,inner,cols){
 const out=new Float64Array(2*rows*cols);
 for(let i=0;i<rows;i++)for(let k=0;k<inner;k++){const p=2*(i*inner+k),ar=a[p],ai=a[p+1];for(let j=0;j<cols;j++){const q=2*(k*cols+j),r=2*(i*cols+j);out[r]+=ar*b[q]-ai*b[q+1];out[r+1]+=ar*b[q+1]+ai*b[q];}}
 return out;
}
export function gramMatrix(fields,weights,rows,cols=3,scale=1){
 const out=new Float64Array(2*cols*cols);
 for(let p=0;p<rows;p++)for(let a=0;a<cols;a++)for(let b=0;b<cols;b++){
  const ia=2*(p*cols+a),ib=2*(p*cols+b),t=2*(a*cols+b),w=scale*weights[p];
  out[t]+=w*(fields[ia]*fields[ib]+fields[ia+1]*fields[ib+1]);out[t+1]+=w*(fields[ia]*fields[ib+1]-fields[ia+1]*fields[ib]);
 }return out;
}
function choleskyWhitening(g,n){
 const l=new Float64Array(2*n*n);
 for(let i=0;i<n;i++)for(let j=0;j<=i;j++){
  let re=g[2*(i*n+j)],im=g[2*(i*n+j)+1];
  for(let k=0;k<j;k++){const a=2*(i*n+k),b=2*(j*n+k);re-=l[a]*l[b]+l[a+1]*l[b+1];im-=l[a+1]*l[b]-l[a]*l[b+1];}
  if(i===j){if(!(re>1e-12))throw new Error('Incident flux Gram matrix is not positive definite');l[2*(i*n+j)]=Math.sqrt(re);}
  else{const d=l[2*(j*n+j)];l[2*(i*n+j)]=re/d;l[2*(i*n+j)+1]=im/d;}
 }
 // T = inv(L†), so T† G T = I. No eigensolver or intensity normalization.
 const t=new Float64Array(2*n*n);
 for(let j=0;j<n;j++)for(let i=n-1;i>=0;i--){let re=i===j?1:0,im=0;for(let k=i+1;k<n;k++){const a=2*(k*n+i),b=2*(k*n+j);re-=l[a]*t[b]+l[a+1]*t[b+1];im-=l[a]*t[b+1]-l[a+1]*t[b];}const d=l[2*(i*n+i)];t[2*(i*n+j)]=re/d;t[2*(i*n+j)+1]=im/d;}
 return t;
}
export function createSources(nq=320){
 const {x:q,w}=gaussLegendre(-9,9,nq),kx=new Float64Array(nq),normalizers=new Float64Array(3);
 for(let j=0;j<nq;j++){kx[j]=Math.sqrt(144-q[j]*q[j]);for(let m=0;m<3;m++)normalizers[m]+=w[j]*Math.exp(-((q[j]-MODES[m][2])**2)/(2*MODES[m][3]**2));}
 for(let m=0;m<3;m++)normalizers[m]=Math.sqrt(normalizers[m]);
 const src={q,w,kx,normalizers,nq,modes:MODES.map(x=>x.slice())};
 src.spectrumRaw=spectrumAt(src,q);src.weightedSpectrumRaw=new Float64Array(src.spectrumRaw.length);
 for(let j=0;j<nq;j++)for(let m=0;m<6;m++)src.weightedSpectrumRaw[j*6+m]=src.spectrumRaw[j*6+m]*w[j]/Math.sqrt(2*PI*kx[j]);
 src.gram=gramMatrix(src.spectrumRaw,w,nq);src.transform=choleskyWhitening(src.gram,3);
 const white=multiplyComplex(src.spectrumRaw,src.transform,nq,3,3);src.orthonormalizationError=identityError(gramMatrix(white,w,nq));return src;
}
function spectrumAt(src,q){
 const out=new Float64Array(q.length*6);
 for(let j=0;j<q.length;j++)for(let m=0;m<3;m++){const kx=Math.sqrt(144-q[j]*q[j]),mode=MODES[m],v=Math.exp(-((q[j]-mode[2])**2)/(4*mode[3]**2))/src.normalizers[m],phase=kx*mode[0]+q[j]*mode[1],p=6*j+2*m;out[p]=v*Math.cos(phase);out[p+1]=v*Math.sin(phase);}return out;
}
export function incidentFields(src,xy){
 const out=new Float64Array(xy.length*3),nn=xy.length/2;
 for(let i=0;i<nn;i++)for(let j=0;j<src.nq;j++){
  const phase=-(xy[2*i]*src.kx[j]+xy[2*i+1]*src.q[j]),c=Math.cos(phase),s=Math.sin(phase);
  for(let m=0;m<3;m++){const a=6*j+2*m,b=6*i+2*m,ar=src.weightedSpectrumRaw[a],ai=src.weightedSpectrumRaw[a+1];out[b]+=c*ar-s*ai;out[b+1]+=s*ar+c*ai;}
 }return out;
}
export function assembleSystem(screens,n){
 const h=PI/n,nodes=n*screens.length,xy=new Float64Array(nodes*2),t=new Float64Array(n),cs=new Float64Array(n),basis=new Float64Array(n*n),matrix=new Float64Array(2*nodes*nodes),logw=new Float64Array(n*n);
 for(let i=0;i<n;i++){t[i]=(i+.5)*h;cs[i]=Math.cos(t[i]);for(let m=0;m<n;m++)basis[i*n+m]=Math.cos(m*t[i]);}
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){let sum=0;for(let m=1;m<n;m++)sum+=basis[i*n+m]*basis[j*n+m]/m;logw[i*n+j]=-h*Math.log(2)-2*h*sum;}
 screens.forEach(([x,lo,hi],si)=>{for(let j=0;j<n;j++){const p=2*(si*n+j);xy[p]=x;xy[p+1]=(lo+hi)/2+(hi-lo)/2*cs[j];}});
 for(let i=0;i<nodes;i++)for(let j=0;j<nodes;j++){
  const p=2*(i*nodes+j),si=Math.floor(i/n),sj=Math.floor(j/n),r=Math.hypot(xy[2*i]-xy[2*j],xy[2*i+1]-xy[2*j+1]);
  if(si!==sj){matrix[p]=-.25*h*besselY0(12*r);matrix[p+1]=-.25*h*besselJ0(12*r);continue;}
  const jj=besselJ0(12*r);let regular;
  if(i===j){const [,lo,hi]=screens[si];regular=-(Math.log(12*(hi-lo)/4)+EULER)/(2*PI);}
  else regular=-.25*besselY0(12*r)+jj*Math.log(Math.abs(cs[i%n]-cs[j%n]))/(2*PI);
  matrix[p]=-jj*logw[(i%n)*n+j%n]/(2*PI)+h*regular;matrix[p+1]=-.25*h*jj;
 }return {xy,t,matrix,nodes,h};
}
/** In-place complex LU elimination with partial pivoting; RHS has nrhs columns. */
export function complexSolve(matrix,rhs,n,nrhs=3){
 const a=matrix.slice(),b=rhs.slice();let minPivot=Infinity,maxPivot=0;
 for(let k=0;k<n;k++){
  let pivot=k,best=0;for(let i=k;i<n;i++){const p=2*(i*n+k),v=a[p]*a[p]+a[p+1]*a[p+1];if(v>best){best=v;pivot=i;}}
  if(!(best>1e-28))throw new Error('Singular/ill-conditioned BEM matrix');minPivot=Math.min(minPivot,Math.sqrt(best));maxPivot=Math.max(maxPivot,Math.sqrt(best));
  if(pivot!==k){for(let j=k;j<n;j++)for(let z=0;z<2;z++){const p=2*(pivot*n+j)+z,q=2*(k*n+j)+z,tmp=a[p];a[p]=a[q];a[q]=tmp;}for(let j=0;j<2*nrhs;j++){const p=2*pivot*nrhs+j,q=2*k*nrhs+j,tmp=b[p];b[p]=b[q];b[q]=tmp;}}
  const d=2*(k*n+k),dr=a[d],di=a[d+1],den=dr*dr+di*di;
  for(let i=k+1;i<n;i++){
   const p=2*(i*n+k),lr=(a[p]*dr+a[p+1]*di)/den,li=(a[p+1]*dr-a[p]*di)/den;a[p]=a[p+1]=0;
   for(let j=k+1;j<n;j++){const q=2*(k*n+j),r=2*(i*n+j);a[r]-=lr*a[q]-li*a[q+1];a[r+1]-=lr*a[q+1]+li*a[q];}
   for(let j=0;j<nrhs;j++){const q=2*(k*nrhs+j),r=2*(i*nrhs+j);b[r]-=lr*b[q]-li*b[q+1];b[r+1]-=lr*b[q+1]+li*b[q];}
  }
 }
 for(let i=n-1;i>=0;i--)for(let j=0;j<nrhs;j++){
  const p=2*(i*nrhs+j);let re=b[p],im=b[p+1];for(let k=i+1;k<n;k++){const q=2*(i*n+k),r=2*(k*nrhs+j);re-=a[q]*b[r]-a[q+1]*b[r+1];im-=a[q]*b[r+1]+a[q+1]*b[r];}
  const d=2*(i*n+i),dr=a[d],di=a[d+1],den=dr*dr+di*di;b[p]=(re*dr+im*di)/den;b[p+1]=(im*dr-re*di)/den;
 }return {solution:b,pivotRatio:minPivot/maxPivot};
}
export function relativeError(a,b){let d=0,s=0;for(let i=0;i<a.length;i++){d+=(a[i]-b[i])**2;s+=b[i]**2;}return Math.sqrt(d/Math.max(s,1e-300));}
function identityError(a){let s=0;for(let i=0;i<3;i++)for(let j=0;j<3;j++){const p=2*(i*3+j);s+=(a[p]-(i===j?1:0))**2+a[p+1]**2;}return Math.sqrt(s);}
export function solveBoundary(options={}){
 const c=checkedOptions(options),screens=[[0,-1.5,1.5]];if(c.screenCount===2)screens.push([c.separation,c.offset-1.3,c.offset+1.3]);
 const src=createSources(c.nq),sys=assembleSystem(screens,c.n),incident=incidentFields(src,sys.xy),rhs=Float64Array.from(incident,x=>-x),solved=complexSolve(sys.matrix,rhs,sys.nodes),phiRaw=solved.solution;
 const res=relativeError(multiplyComplex(sys.matrix,phiRaw,sys.nodes,sys.nodes,3),rhs);
 if(!Number.isFinite(res)||res>1e-10)throw new Error(`BEM residual rejected: ${res}`);
 return {options:c,k:12,n:c.n,screens,sources:src,boundary:{xy:sys.xy,t:sys.t,phiRaw,weightedPhiRaw:Float64Array.from(phiRaw,x=>x*sys.h),nodes:sys.nodes,h:sys.h},diagnostics:{relativeResidual:res,algebraicResidual:res,orthonormalizationError:src.orthonormalizationError,pivotRatio:solved.pivotRatio,residualPass:res<1e-10,maskDistance:12/c.n,scalarFluxUnits:true,scope:COHERENT_SCOPE}};
}
export function createState(solution,options={}){
 // In this orthonormal basis, gamma changes off-diagonal coherence only.
 // Fixed p=[.2,.5,.3]; rho=P[(1-gamma)diag(p)+gamma vv†].
 const c=checkState({...solution.options,...options}),factorOrthonormal=new Float64Array(24),v=new Float64Array([Math.sqrt(.2),0,Math.sqrt(.5)*Math.cos(c.phase),Math.sqrt(.5)*Math.sin(c.phase),Math.sqrt(.3)*Math.cos(-.3),Math.sqrt(.3)*Math.sin(-.3)]);
 const scale=Math.sqrt(c.inputPower*c.gamma),pop=[.2,.5,.3];
 for(let m=0;m<3;m++){factorOrthonormal[8*m]=scale*v[2*m];factorOrthonormal[8*m+1]=scale*v[2*m+1];factorOrthonormal[2*(m*4+m+1)]=Math.sqrt(c.inputPower*(1-c.gamma)*pop[m]);}
 const factorRaw=multiplyComplex(solution.sources.transform,factorOrthonormal,3,3,4),coherentRaw=multiplyComplex(solution.sources.transform,v,3,3,1);
 return {...c,factorOrthonormal,factorRaw,coherentRaw,factorCount:4,sourceCount:3,rank:c.inputPower===0?0:c.gamma===1?1:3};
}
export function intensityFromFields(fields,factorRaw,factorCount=4){
 const n=fields.length/6,out=new Float64Array(n);
 for(let i=0;i<n;i++)for(let r=0;r<factorCount;r++){let re=0,im=0;for(let m=0;m<3;m++){const a=6*i+2*m,b=2*(m*factorCount+r);re+=fields[a]*factorRaw[b]-fields[a+1]*factorRaw[b+1];im+=fields[a]*factorRaw[b+1]+fields[a+1]*factorRaw[b];}out[i]+=re*re+im*im;}return out;
}
export function pointIsValid(solution,x,y){return solution.screens.every(([sx,lo,hi])=>Math.hypot(x-sx,Math.max(lo-y,0,y-hi))>=12/solution.n);}
export function evaluatePoints(solution,xy,state=solution.state,options={}){
 if(!state)state=createState(solution);if(xy.length%2||xy.length/2>COHERENT_LIMITS.maxGridPoints||!xy.every(Number.isFinite))throw new RangeError('Invalid field coordinates');
 const nn=xy.length/2,fieldsRaw=new Float64Array(nn*6),mask=new Uint8Array(nn),{sources:src,boundary:b,k}=solution;
 for(let i=0;i<nn;i++){
  const x=xy[2*i],y=xy[2*i+1];if(options.mask!==false&&!pointIsValid(solution,x,y))continue;mask[i]=1;
  for(let j=0;j<src.nq;j++){const phase=-(x*src.kx[j]+y*src.q[j]),c=Math.cos(phase),s=Math.sin(phase);for(let m=0;m<3;m++){const a=6*j+2*m,p=6*i+2*m;fieldsRaw[p]+=c*src.weightedSpectrumRaw[a]-s*src.weightedSpectrumRaw[a+1];fieldsRaw[p+1]+=c*src.weightedSpectrumRaw[a+1]+s*src.weightedSpectrumRaw[a];}}
  for(let j=0;j<b.nodes;j++){const kr=k*Math.hypot(x-b.xy[2*j],y-b.xy[2*j+1]),gr=-.25*besselY0(kr),gi=-.25*besselJ0(kr);for(let m=0;m<3;m++){const a=6*j+2*m,p=6*i+2*m;fieldsRaw[p]+=gr*b.weightedPhiRaw[a]-gi*b.weightedPhiRaw[a+1];fieldsRaw[p+1]+=gr*b.weightedPhiRaw[a+1]+gi*b.weightedPhiRaw[a];}}
 }
 const intensity=intensityFromFields(fieldsRaw,state.factorRaw),coherentField=multiplyComplex(fieldsRaw,state.coherentRaw,nn,3,1);
 let validPixels=0,maxIntensity=0,finite=true;for(let i=0;i<nn;i++){validPixels+=mask[i];finite&&=Number.isFinite(intensity[i])&&intensity[i]>=0;maxIntensity=Math.max(maxIntensity,intensity[i]);}finite&&=fieldsRaw.every(Number.isFinite);
 if(!finite)throw new Error('Nonfinite/negative field output rejected');
 return {xy,fieldsRaw,intensity,coherentField,mask,diagnostics:{finite,nonnegative:true,validPixels,maskedPixels:nn-validPixels,maxIntensity,maskDistance:12/solution.n,intensityIsFlux:false}};
}
export function evaluateGrid(solution,{width=144,height=108,xMin=-2.8,xMax=5.5,yMin=-3,yMax=3}={},state=solution.state){
 if(!Number.isInteger(width)||!Number.isInteger(height)||width<2||height<2||width*height>COHERENT_LIMITS.maxGridPoints||![xMin,xMax,yMin,yMax].every(Number.isFinite)||xMin>=xMax||yMin>=yMax)throw new RangeError('Invalid grid');
 const xy=new Float64Array(width*height*2);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const p=2*(y*width+x);xy[p]=xMin+(xMax-xMin)*x/(width-1);xy[p+1]=yMax-(yMax-yMin)*y/(height-1);}
 return {...evaluatePoints(solution,xy,state),width,height,xMin,xMax,yMin,yMax};
}
export function farFields(solution,theta){
 const out=new Float64Array(theta.length*6),b=solution.boundary,src=solution.sources,fac=1/Math.sqrt(16*PI*solution.k);
 for(let i=0;i<theta.length;i++){
  const c=Math.cos(theta[i]),s=Math.sin(theta[i]),q=solution.k*s;
  if(c>0&&Math.abs(q)<=9){const raw=spectrumAt(src,[q]),d=Math.sqrt(c/2);for(let m=0;m<3;m++){const p=6*i+2*m;out[p]=d*(raw[2*m]-raw[2*m+1]);out[p+1]=d*(raw[2*m]+raw[2*m+1]);}}
  for(let j=0;j<b.nodes;j++){const phase=solution.k*(c*b.xy[2*j]+s*b.xy[2*j+1]),co=Math.cos(phase),si=Math.sin(phase),fr=fac*(co+si),fi=fac*(si-co);for(let m=0;m<3;m++){const a=6*j+2*m,p=6*i+2*m;out[p]+=fr*b.weightedPhiRaw[a]-fi*b.weightedPhiRaw[a+1];out[p+1]+=fr*b.weightedPhiRaw[a+1]+fi*b.weightedPhiRaw[a];}}
 }return out;
}
function piecewiseAngles(breaks,n){
 const x=[],w=[];for(let j=0;j<breaks.length-1;j++){if(breaks[j+1]-breaks[j]<1e-14)continue;const g=gaussLegendre(breaks[j],breaks[j+1],n);x.push(...g.x);w.push(...g.w);}return {theta:Float64Array.from(x),weights:Float64Array.from(w)};
}
function fullAngles(n){const a=Math.asin(.75);return piecewiseAngles([-PI,-PI/2,-a,a,PI/2,PI],n);}
function detectorAngles(center,half,n){
 const lo=center-half,hi=center+half,a=Math.asin(.75),breaks=[lo,hi];
 // Split at all incident-spectrum support discontinuities and forward/backward boundaries.
 for(let period=-1;period<=1;period++)for(const t of [-PI,-PI/2,-a,a,PI/2,PI]){const v=t+2*PI*period;if(v>lo&&v<hi)breaks.push(v);}
 return piecewiseAngles(breaks.sort((a,b)=>a-b),n);
}
export function measureCoherent(solution,options={}){
 const state=createState(solution,options),n=solution.options.nang;
 if(!solution.angular){const a=fullAngles(n);solution.angular={...a,fieldsRaw:farFields(solution,a.theta)};}
 const all=solution.angular,flux=intensityFromFields(all.fieldsRaw,state.factorRaw);let totalFlux=0,forwardFlux=0,backwardFlux=0;
 for(let i=0;i<flux.length;i++){flux[i]*=solution.k;const v=all.weights[i]*flux[i];totalFlux+=v;if(Math.cos(all.theta[i])>0)forwardFlux+=v;else backwardFlux+=v;}
 const key=`${state.detectorCenter}:${state.detectorHalfAngle}:${n}`;
 if(!solution.detectorCache||solution.detectorCache.key!==key){const d=detectorAngles(state.detectorCenter,state.detectorHalfAngle,n);solution.detectorCache={...d,key,fieldsRaw:farFields(solution,d.theta)};}
 const det=solution.detectorCache,df=intensityFromFields(det.fieldsRaw,state.factorRaw);let detectorFlux=0;for(let i=0;i<df.length;i++){df[i]*=solution.k;detectorFlux+=det.weights[i]*df[i];}
 const energyMatrixRaw=gramMatrix(all.fieldsRaw,all.weights,all.theta.length,3,solution.k),detectorMatrixRaw=gramMatrix(det.fieldsRaw,det.weights,det.theta.length,3,solution.k),white=multiplyComplex(all.fieldsRaw,solution.sources.transform,all.theta.length,3,3),energyError=identityError(gramMatrix(white,all.weights,all.theta.length,3,solution.k));
 const finite=[detectorFlux,totalFlux,forwardFlux,backwardFlux].every(Number.isFinite),nonnegative=flux.every(x=>x>=0)&&df.every(x=>x>=0);
 if(!finite||!nonnegative)throw new Error('Invalid scalar-flux measurement');
 return {state,detectorFlux,totalFlux,forwardFlux,backwardFlux,detectorMatrixRaw,energyMatrixRaw,angleTheta:all.theta,angleFlux:flux,detectorTheta:det.theta,detectorAngleFlux:df,diagnostics:{finite,nonnegative,energyOperatorError:energyError,relativeFluxError:state.inputPower?Math.abs(totalFlux-state.inputPower)/state.inputPower:0,energyPass:energyError<5e-4,fluxUnits:'scalar current per invariant transverse length',noOutputRenormalization:true}};
}
export function solveCoherent(options={}){
 const start=globalThis.performance?.now?.()??Date.now(),solution=solveBoundary(options);
 solution.measurement=measureCoherent(solution,options);solution.state=solution.measurement.state;
 solution.diagnostics.reference=validateReference(solution);
 if(solution.diagnostics.reference.pass===false)throw new Error('Independent complex-field fixture rejected this solve');
 solution.diagnostics={...solution.diagnostics,...solution.measurement.diagnostics,elapsedMs:(globalThis.performance?.now?.()??Date.now())-start};
 return solution;
}

/** Independent SciPy gold, only for validation; never used to form solved output. */
export const COHERENT_REFERENCE={"one_screen":{"source":"research/coherent-closure/bem-results.json; SciPy independent 144-node solution","xy":[5.0,-4.0,5.0,-3.0,5.0,-2.0,5.0,-1.0,5.0,0.0,5.0,1.0,5.0,2.0,5.0,3.0,5.0,4.0],"nearRaw":[-0.0014154065292498888,-0.015207488197519553,-0.0015571880942757909,1.4714828363077874e-06,-0.0014265500003921357,-0.003446798520408892,-0.011737168347153867,0.014030362380324216,0.0012037095078729874,-0.0015637029110929551,-0.000573947727690764,4.3934300416220685e-05,-0.010512271119816632,-0.01268206226430907,0.002471769805746267,0.000900293840312507,0.00188821075732604,-0.0008320768959280142,-0.010827130436906585,-0.004611219078452272,0.0034351077854929413,0.0012095337240793994,0.002128825789926285,0.0017701766278882293,0.004854786597917826,0.006080660410957728,0.0013545679701616464,-0.005025139869254486,0.005542132173719239,-0.005447964717097728,-0.004473997646978506,0.002879502318043986,0.0015828610083852244,0.007131413122211577,-0.002223118277378977,0.014654958578362333,-0.003290351962600309,0.0019806833090714416,-0.0020286890195084922,0.008911360315933196,-0.01361576990975909,0.01884430964494707,0.0007011910235143132,0.0027169339185996945,0.009285043173998616,-0.0006616205895158388,0.021996011465982204,0.015685160400592233,0.0013692422580271716,-0.001483919014548962,-0.004223570773835032,0.0066738760707548165,-0.018834981131584876,-0.00015007580104596649],"theta":[-3.1242659127907686,-1.9146738076078007,-1.40668758921604,-0.8546246329011526,0.17307243005416745,0.9115684668574959,1.5112375531610374,2.210989847555599,3.1273295470293925],"farRaw":[0.24842298592262965,-0.4653006430117827,-0.1966501713143324,0.4610871785201426,-0.25419583780386273,-0.43168675661458356,0.007186510149804151,0.014639788042730825,0.001601992359638972,-8.617136375833836e-05,0.0030508102322100443,0.004048511924596747,-0.005623663425493035,0.014578943762820077,0.00021218906237931094,-0.0006449878914769497,-0.0007255544245528709,0.0019966682272003753,-0.01362435245470139,-0.01462346973456914,-0.004054042216574667,-0.0007430632220873939,-0.00678488914800866,-0.006826510655921741,-0.015609018558751367,-0.020003449210319768,-0.018619154386528697,0.010242164019947264,-0.062243722430947813,-0.013437419042210186,0.001125122783258832,-0.0046155554280177145,-0.006205638508745209,0.01020267412702053,-0.01976786422032817,0.0022516061547775378,0.0009913048009018734,0.0003344504590663367,0.00939681146290464,-0.003346344959693936,0.014870887095606612,0.007835355773886266,5.946286331396282e-06,-0.004612585527154135,-0.008135255382773104,0.008560294417051712,-0.01999023968886511,-0.0020903537873336424,0.16109434828177685,-0.5148009814098742,-0.2707959728057012,0.385337898251297,-0.2485591886561277,-0.5133043920561293],"detectorMatrixRaw":[0.012635263167599203,-8.786995850660776e-20,-2.4136598601276038e-05,-8.72451212471033e-05,0.0024542472227321987,0.0009038999783543951,-2.4136598601276065e-05,8.724512124710334e-05,0.003843517616666882,-3.005168355712809e-20,0.00733379461715236,0.004855881128832163,0.0024542472227321982,-0.0009038999783543953,0.00733379461715236,-0.004855881128832162,0.02278093729401823,-6.626434964050603e-20]},"two_screens":{"source":"research/coherent-closure/bem-results.json; SciPy independent 144-node solution","xy":[5.0,-4.0,5.0,-3.0,5.0,-2.0,5.0,-1.0,5.0,0.0,5.0,1.0,5.0,2.0,5.0,3.0,5.0,4.0],"nearRaw":[-0.0008591294823738268,-0.014030073353162566,-0.0030364556435529206,-0.0003064121933580005,-0.003565185512208931,-0.0036284780343269016,-0.012560428353229132,0.014091882813637498,-0.004339692592312507,0.001161242567120363,-0.010736427742199503,0.0014719219881717841,-0.013134843619176134,-0.011957487186265962,0.0020052899678499984,0.0011265073259175869,0.0012360549181707255,-0.0007701902590927,-0.009716776934296269,-0.007015517742425509,0.00042164261624055843,0.000796236383350267,-0.001238691798753619,-0.0006916353429763898,0.0026595045890499963,0.0032908182286929266,-5.2462321140769874e-05,0.0009599589788283902,-0.00030740934386405683,0.003054680428895673,0.0007996686988022278,0.0019152585719989998,0.0019482428043331012,0.001085219026859624,0.003905292575593522,0.004110343486713554,0.00028989385104009543,0.0005335465772546115,-0.002747283602410583,0.005151649443927415,-0.011113869735442546,0.011088673495237622,0.0016089334145122786,-0.0009910603068431253,0.011077840252554821,0.0018578184297379672,0.023873849438936615,0.021072633097904154,0.003207757192969907,0.0008034435007434188,-0.00347225357005736,0.008181137025002373,-0.01808937811413969,0.004023390751572816],"theta":[-3.1242659127907686,-1.9146738076078007,-1.40668758921604,-0.8546246329011526,0.17307243005416745,0.9115684668574959,1.5112375531610374,2.210989847555599,3.1273295470293925],"farRaw":[0.2495372330634848,-0.46590274354214656,-0.1994992245816621,0.4582077706277848,-0.25721305990584764,-0.43892148847925744,0.008255333484303694,0.01596221413049931,0.003155669334588055,-0.001934452275838097,0.006961628427169966,0.0025769597909395155,-0.007139591311814566,0.015250233448674544,0.001380526947277807,-0.000786264036129075,0.0005584467790905526,0.0024479173213742553,-0.012377407462357439,-0.012905387449301794,-0.0029007496763334283,0.001436271273176715,-0.0068343994632009564,-0.002451976370616129,-0.012066382128286882,-0.016393490995238402,-0.018012902921088203,0.0034319088651250246,-0.05511669835787533,-0.02456898260606727,0.0012215957923062883,-0.00409701708241262,-0.004563547949419656,0.012992737889671402,-0.01852697031307051,0.009029556877942688,-0.0014243520937930413,0.0023049007412735086,0.008652221802444439,-0.002434055425787031,0.011861503792244371,0.009516892491428924,0.004021598861243883,-0.007620166701505004,-0.00734396096311644,0.007591167205400136,-0.016349041079175394,-0.004072316277630592,0.16255366119352996,-0.5140135831097516,-0.272462232982905,0.3813337260712949,-0.24836640735426246,-0.5215095136756792],"detectorMatrixRaw":[0.012333453859047586,-7.72324432205088e-20,0.0007297556113448204,0.0006412403757514692,0.0027719878623970015,0.0026517020709599997,0.0007297556113448204,-0.0006412403757514691,0.0032635402860862195,-2.2356124637486242e-20,0.006567106559649103,0.004454723730406078,0.002771987862397001,-0.0026517020709599997,0.006567106559649103,-0.004454723730406078,0.021396157882458394,-3.841549628308464e-20]}};

export function validateReference(solution){
 const applicable=solution.screens.length===1||(Math.abs(solution.options.separation-2.3)<1e-12&&Math.abs(solution.options.offset-.4)<1e-12);
 if(!applicable)return {applicable:false,pass:null,reason:'Geometry differs from the independent default fixture'};
 const ref=COHERENT_REFERENCE[solution.screens.length===1?'one_screen':'two_screens'];
 const near=evaluatePoints(solution,Float64Array.from(ref.xy),solution.state,{mask:false}).fieldsRaw;
 const far=farFields(solution,Float64Array.from(ref.theta)),det=gaussLegendre(-.55,.55,160),dfield=farFields(solution,det.x),matrix=gramMatrix(dfield,det.w,160,3,solution.k);
 const nearFieldError=relativeError(near,ref.nearRaw),farFieldError=relativeError(far,ref.farRaw),detectorOperatorError=relativeError(matrix,ref.detectorMatrixRaw),threshold=2e-7;
 return {applicable:true,pass:Math.max(nearFieldError,farFieldError,detectorOperatorError)<threshold,nearFieldError,farFieldError,detectorOperatorError,threshold,source:ref.source};
}
