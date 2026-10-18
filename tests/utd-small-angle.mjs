import assert from 'node:assert/strict';
import fs from 'node:fs';
const f=Math.fround,m=(a,b)=>f(a*b),a=(a,b)=>f(a+b),fm=(a,b,c)=>f(a*b+c);
const C=[-1/6,1/120,-1/5040,1/362880].map(f),D=[-.5,1/24,-1/720,1/40320,-1/3628800].map(f);
function poly(x,coeff,fused){let q=m(x,x),p=coeff.at(-1);for(let i=coeff.length-2;i>=0;i--)p=fused?fm(p,q,coeff[i]):a(m(p,q),coeff[i]);return fused?fm(p,q,1):a(m(p,q),1);}
const reports=[];for(let fused of [false,true]){let r={fused,count:1000001,maxSincError:0,maxCosError:0,maxSinProductError:0};for(let i=0;i<=1000000;i++){let x=f(-.5+i/1e6),sc=poly(x,C,fused),cs=poly(x,D,fused);r.maxSincError=Math.max(r.maxSincError,Math.abs(sc-(x===0?1:Math.sin(x)/x)));r.maxCosError=Math.max(r.maxCosError,Math.abs(cs-Math.cos(x)));r.maxSinProductError=Math.max(r.maxSinProductError,Math.abs(m(x,sc)-Math.sin(x)));}reports.push(r);}console.log(JSON.stringify(reports,null,2));

for(const r of reports){ assert(r.maxSincError<5e-8); assert(r.maxCosError<5e-8); assert(r.maxSinProductError<5e-8); }
