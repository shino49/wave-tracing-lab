/** Independent anchored-axis CPU reference review. No native GPU execution.
 * Run: node tests/plt-resident-anchor-reference.mjs
 * Uses actual sampleCamera, normalize, rayNearestBrute and mirror operations;
 * exact BigInt rational checks evaluate recurrence errors without f32 reuse.
 * Conditional on the documented binary64/hypot and geometry arithmetic contract.
 * Optional raw-camera underflow extension is checked analytically below. The
 * initial production candidate instead rejects raw intervals crossing/touching
 * zero unless exactly zero; the extension is not its admission basis. This does
 * not independently certify every barycentric/BVH/estimator operation.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {cameraFrame,sampleCamera} from '../dist/plt-core.js';
import {normalize,sub,scale,dot,add} from '../dist/mesh-geometry.js';
import {rayNearestBrute} from '../dist/plt-geometry.js';
const abs=x=>x<0n?-x:x;function gcd(a,b){a=abs(a);b=abs(b);while(b){[a,b]=[b,a%b];}return a;}
function R(n,d=1n){if(d<0n){n=-n;d=-d;}const g=gcd(n,d);return{n:n/g,d:d/g};}
const qadd=(a,b)=>R(a.n*b.d+b.n*a.d,a.d*b.d),qneg=a=>R(-a.n,a.d),qsub=(a,b)=>qadd(a,qneg(b)),qmul=(a,b)=>R(a.n*b.n,a.d*b.d),qdiv=(a,b)=>R(a.n*b.d,a.d*b.n),qabs=a=>R(abs(a.n),a.d),cmp=(a,b)=>Math.sign(Number(a.n*b.d-b.n*a.d)),value=a=>Number(a.n)/Number(a.d);
function rat(x){assert(Number.isFinite(x));const b=new ArrayBuffer(8),v=new DataView(b);v.setFloat64(0,x);const w=v.getBigUint64(0),e=Number(w>>52n&2047n),m=w&0xfffffffffffffn,s=w>>63n?-1n:1n,ex=e?e-1075:-1074,n=s*(e?m+(1n<<52n):m);return ex>=0?R(n<<BigInt(ex)):R(n,1n<<BigInt(-ex));}
const zero=rat(0),u=rat(2**-53);let seed=417936;function random(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return(seed>>>0)/2**32;}
function sceneBox(side){const ts=[];for(let k=0;k<3;k++)for(const sign of[-1,1]){const i=(k+1)%3,j=(k+2)%3,p=(x,y)=>{const a=[0,0,0];a[k]=sign*side;a[i]=x;a[j]=y;return a;};const ps=[p(-side,-side),p(side,-side),p(side,side),p(-side,side)];for(const ids of[[0,1,2],[0,2,3]]){let n=[0,0,0];n[k]=sign;const[a,b,c]=ids.map(i=>ps[i]);ts.push({id:ts.length,a,b,c,normal:n,materialIndex:0});}}return {triangles:ts};}
let steps=0,pointChecks=0,distanceChecks=0,nonzeroNormal=0,rawRatioDifferent=0,maxRatio=0,maxNormal=0,worst=null,idealMiss=null,zeroFloorViolations=0;
for(const side of[1/64,.125,1,4,64,256])for(let sample=0;sample<96;sample++){
 const C=[0,1,2].map(()=>side*(random()-.5)),target=[0,1,2].map(()=>side*(random()*1.5-.75));
 const camera={position:C,target,up:[0,1,0],fov:5+110*random()},x=Math.floor(8*random()),y=Math.floor(8*random()),rs=[random(),random()];let ri=0;const beam=sampleCamera(camera,x,y,8,8,()=>rs[ri++],.025,false);const frame=cameraFrame(camera),half=Math.tan((camera.fov*Math.PI/180)/2),cx=(2*(x+rs[0])/8-1)*half,cy=(1-2*(y+rs[1])/8)*half;const raw=add(frame.forward,add(scale(frame.right,cx),scale(frame.up,cy)));
 const qr=raw.map(rat),qc=C.map(rat);let O=C,D=beam.direction,sign=[1,1,1],B=[zero,zero,zero],E=[zero,zero,zero];const scene=sceneBox(side);
 for(let depth=0;depth<6;depth++){
  const Q=normalize(D),hit=rayNearestBrute(scene,O,D,{minDistance:1e-7,maxDistance:Infinity});assert(hit);const axis=hit.triangle.normal.findIndex(x=>x!==0),P=rat(hit.triangle.a[axis]);assert(Math.abs(Q[axis])>=2**-12,'fixture must satisfy queried-axis contract');const V=qr.map((a,k)=>qmul(rat(sign[k]),a));const H=qmul(rat(sign[axis]),qsub(P,B[axis])),S=qdiv(qsub(H,qc[axis]),qr[axis]);const ideal=qr.map((a,k)=>qadd(qmul(rat(sign[k]),qadd(qc[k],qmul(a,S))),B[k]));const Ehit=hit.point.map((x,k)=>qsub(rat(x),ideal[k]));
  const delta=qsub(P,rat(O[axis])),tIdeal=qdiv(delta,rat(Q[axis])),tErr=qabs(qsub(rat(hit.distance),tIdeal)),tBound=qmul(qmul(rat(32),u),qabs(tIdeal));assert(cmp(tErr,tBound)<=0,'MT distance error exceeds32u');distanceChecks++;
  for(let i=0;i<3;i++){
   const ratio=qdiv(V[i],V[axis]),G=qmul(ratio,delta),local=i===axis?Ehit[i]:qadd(qsub(Ehit[i],E[i]),qmul(ratio,E[axis]));const bound=qadd(qmul(qmul(rat(64),u),qabs(G)),qmul(qmul(rat(4),u),rat(512)));const ratioErr=value(qdiv(qabs(local),bound));
   if(cmp(qabs(local),qmul(qmul(rat(64),u),qabs(G)))>0)zeroFloorViolations++;
   assert(cmp(qabs(local),bound)<=0,`raw residual recurrence failure side${side}sample${sample}depth${depth}axis${i}`);pointChecks++;if(ratioErr>maxRatio){maxRatio=ratioErr;worst={side,sample,depth,axis,i,local:value(local),bound:value(bound)};}
   if(cmp(qdiv(rat(D[i]),rat(D[axis])),ratio)!==0)rawRatioDifferent++;
  }
  const residual=value(qsub(rat(hit.point[axis]),P));if(residual!==0){nonzeroNormal++;if(Math.abs(residual)>maxNormal){maxNormal=Math.abs(residual);idealMiss={side,sample,depth,axis,O,D,Q,P:value(P),actualPoint:hit.point,residual};}}
  E=Ehit;B[axis]=qsub(qmul(rat(2),P),B[axis]);sign[axis]*=-1;const n=dot(hit.geometricNormal,D)<0?hit.geometricNormal:scale(hit.geometricNormal,-1);D=sub(D,scale(n,2*dot(D,n)));O=hit.point;steps++;
 }
}

// These strict rational inequalities are the normal-arithmetic derivation,
// not fits to the preceding finite corpus.
const one=rat(1),up=qadd(one,u),down=qsub(one,u),u2=qmul(u,u),pow=(a,n)=>Array(n).fill(a).reduce(qmul,one),tau=R(1n,1n<<1075n);
let constantChecks=0;
for(let k=1;k<=16;k++){
 const factor=qsub(qmul(qdiv(pow(up,4),pow(down,2)),qadd(one,qmul(rat(32*k),u))),one);
 assert(cmp(factor,qmul(rat(39*k),u))<0);constantChecks++;
 // Camera product underflow changes ||D||/(c||R||) by at most4tau<u².
 // Queried axis products are normal because |Q_k|>=2^-12 and unit norm.
 const rawLo=qdiv(pow(down,3),qmul(qadd(up,u2),qadd(one,qmul(rat(16),u))));
 const rawHi=qdiv(pow(up,3),qmul(qsub(down,u2),qsub(one,qmul(rat(16),u))));
 assert(cmp(qsub(rawHi,one),qmul(rat(21),u))<0);assert(cmp(qsub(one,rawLo),qmul(rat(21),u))<0);constantChecks+=2;
 const distanceUpper=qdiv(qadd(one,qmul(rat(32*k),u)),qsub(one,qmul(rat(64),u)));
 const distanceLower=qdiv(qsub(one,qmul(rat(32*k),u)),qadd(one,qmul(rat(64),u)));
 assert(cmp(qsub(distanceUpper,one),qmul(rat(128*k),u))<0);assert(cmp(qsub(one,distanceLower),qmul(rat(128*k),u))<0);constantChecks+=2;
}
assert(cmp(qmul(rat(4),tau),u2)<0);constantChecks++;
// Q_i = cbR_i*relativeFactors + additiveError, |additiveError|<4tau,
// because the query reciprocal b<2. |Q_k|>=2^-12, |plane-O_k|<=513
// imply t<2^23. Additional tangent/product/add underflow is <2^26tau.
// This fits in the existing2048u absolute floor with world addition <=513u.
assert(cmp(qadd(qmul(rat(2**26),tau),qmul(rat(513),u)),qmul(rat(2048),u))<0);constantChecks++;
const p1=256,p2=-256+2**-16,b1=2*p1,b2=2*p2-b1;
assert.notEqual(Math.fround(b2),b2,'test must expose non-exact f32 unfolded offset');
assert(nonzeroNormal>0,'CPU plane-snap mutation must be detected');
assert(rawRatioDifferent>0,'exact raw/normalized ratio mutation must be detected');
assert(zeroFloorViolations>0,'removing CPU point absolute floor must be detected');
const hashes=Object.fromEntries(['plt-core.js','plt-geometry.js','mesh-geometry.js'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/'+name,import.meta.url))).digest('hex')]));
const report={allPass:true,actualGPUExecuted:false,scope:'independent exact-rational CPU recurrence/constant review',nodeVersion:process.version,sourceSHA256:hashes,testSHA256:crypto.createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex'),samples:576,steps,pointChecks,distanceChecks,constantChecks,rawRatioDifferent,nonzeroNormal,zeroFloorViolations,maxRatio,maxNormal,worst,idealMiss,offsetCounterexample:{p1,p2,b1,b2,f32:Math.fround(b2),error:Math.fround(b2)-b2},limitations:['Conditional nearest-rounding binary64 and Math.hypot relative error<=16u','Assumes existing bounded axis geometry MT predicate contract; no general BVH equivalence certification','Actual CPU box fixtures use kappa1; exact constant inequalities cover1<=kappa<=16','Optional underflow inequalities are theory only; current anchor mode rejects raw intervals crossing/touching zero unless exact zero','No production default, tolerance change, native GPU dispatch or performance claim']};
fs.writeFileSync(new URL('../research/plt-resident-anchor-reference-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
