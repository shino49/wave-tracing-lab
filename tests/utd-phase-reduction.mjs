import assert from 'node:assert/strict';
import fs from 'node:fs';
const f=Math.fround,PI=f(Math.PI),TAU=f(2*Math.PI),HI=f(1.5703125),LO=f(0.00048382679489661923),INV=f(0.63661977236758134308);
const fma=(a,b,c)=>f(a*b+c);
function reduced(theta){let quadrant=Math.round(f(theta*INV)),r=fma(-quadrant,HI,theta);r=fma(-quadrant,LO,r);let c=f(Math.cos(r)),s=f(Math.sin(r));switch(quadrant&3){case 0:return {r,p:[c,s]};case 1:return {r,p:[-s,c]};case 2:return {r,p:[-c,-s]};default:return {r,p:[s,-c]};}}
function naive(theta){let q=Math.round(f(theta/TAU)),r=f(theta-f(q*TAU));return [f(Math.cos(r)),f(Math.sin(r))];}
let count=0,max={error:0},naiveMax={error:0},maxR=0,seed=0x1eafd00d;
function check(t){t=f(t);let ideal=[Math.cos(t),Math.sin(t)],a=reduced(t),b=naive(t),error=Math.max(...a.p.map((v,i)=>Math.abs(v-ideal[i]))),ne=Math.max(...b.map((v,i)=>Math.abs(v-ideal[i])));count++;maxR=Math.max(maxR,Math.abs(a.r));if(error>max.error)max={error,input:t,got:a.p,ideal,residual:a.r};if(ne>naiveMax.error)naiveMax={error:ne,input:t,got:b,ideal};}
for(let i=0;i<=1000000;i++)check(-1e5+.2*i);
for(let i=0;i<1000000;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;check((seed/4294967296*2-1)*1e5);}
for(let q=-63662;q<=63662;q++)for(let d of [-Math.PI/4,-1e-4,0,1e-4,Math.PI/4])check(q*Math.PI/2+d);
const report={count,range:[-100000,100000],algorithm:'Cody-Waite split pi/2 via fma; f32 input and each intermediate',constants:{HI,LO,INV},max,maxR,naiveMax};console.log(JSON.stringify(report,null,2));

assert(max.error<2.5e-7);assert(maxR<.8);assert(naiveMax.error>.005);
