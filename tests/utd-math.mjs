// Expected numbers come only from research/generate-utd-approx.py / SciPy.
// Float32 emulation quantifies rounding; it is NOT a claim of GPU execution.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as U from '../dist/utd-math.js';
const fixture = JSON.parse(fs.readFileSync(new URL('../research/utd-fixtures.json', import.meta.url)));
const report = { provenance: fixture.provenance, gpuExecuted: false, maxima: {} };
for (const kind of ['transition', 'coefficient', 'halfplane']) {
  let max = 0;
  for (const c of fixture[kind]) {
    const got = kind === 'transition' ? U.transition(c.x) : kind === 'coefficient' ? U.coefficient(...c.args) : U.totalPlane(...c.args);
    assert(got.every(Number.isFinite));
    const error = U.abs(U.sub(got, c.expected));
    max = Math.max(max, error);
    assert(error < 2e-11, `${kind} ${c.args || c.x}: ${error}`);
  }
  report.maxima[kind] = { cases: fixture[kind].length, maxAbsoluteError: max };
}
assert.deepEqual(U.transition(0), [0, 0]);
assert.deepEqual(U.transition(Infinity), [1, 0]);
assert.throws(() => U.transition(-1), RangeError);
assert.throws(() => U.coefficient(1, 2, 0.5, 1, 1, 0), RangeError);
assert.throws(() => U.coefficient(1, 2, 0.5, 1, 1, 1, 'metal'), RangeError);
for (const boundary of ['soft', 'hard']) for (const pi of [0.1, 0.9, 2.8]) for (const po of [0.2, 1, 2.9]) {
  assert.deepEqual(U.coefficient(1e8, 1, pi, po, 1, 1, boundary), [0, 0]);
}
for (const n of [1.01, 1.25, 1.5, 1.75, 2]) {
  for (const k of [0.1, 1, 100, 1e6]) {
    for (const boundary of ['soft', 'hard']) {
      // Reciprocity and oblique-cone normalization are coefficient identities.
      const a = U.coefficient(k, n, .7, 1.8, 1, 1, boundary);
      const b = U.coefficient(k, n, 1.8, .7, 1, 1, boundary);
      assert(U.abs(U.sub(a, b)) < 1e-12);
      assert(U.abs(U.sub(U.coefficient(k, n, .7, 1.8, 1, .4, boundary), U.scale(a, 2.5))) < 1e-12);
      for (const center of [Math.PI - .6, Math.PI + .6, 2 * n * Math.PI - Math.PI - .6]) {
        if (center <= 0 || center >= n * Math.PI) continue;
        const v = U.totalPlane(k, 1, n, .6, center, boundary);
        const l = U.totalPlane(k, 1, n, .6, center - 1e-12, boundary);
        const r = U.totalPlane(k, 1, n, .6, center + 1e-12, boundary);
        assert(U.abs(U.sub(l, r)) < 5e-6, 'UTD total field is continuous across its GO seam');
        assert(U.abs(U.sub(v, U.scale(U.add(l, r), .5))) < 1e-9);
      }
    }
  }
}
const pole = U.stableCotTransition(Math.PI, 2, 100, -1);
assert.deepEqual(pole, [0, 0]);
assert(U.abs(U.stableCotTransition(Math.PI + 1e-12, 2, 100, -1)) > 1, 'near-pole value must not be zeroed');

// Every arithmetic stage below is rounded to f32, matching the shader's
// non-fused algebra; hardware trig/FMA behavior can add a small difference.
const f = Math.fround, P = f(Math.PI), H = f(Math.SQRT1_2), EPS = f(2e-6);
const add = (a,b) => a.map((v,i)=>f(v+b[i]));
const mul = (a,b) => [f(f(a[0]*b[0])-f(a[1]*b[1])),f(f(a[0]*b[1])+f(a[1]*b[0]))];
const scale = (a,s) => a.map(v=>f(v*s));
const cis = x => [f(Math.cos(x)),f(Math.sin(x))];
const cc = U.ERFCX_CHEBYSHEV.map(c=>c.map(f));
function cheb(t) {
  const s=f(f(2*t)-1); let b=[0,0],c=[0,0];
  for(let j=40;j>=1;--j) {const next=add(add(scale(b,f(2*s)),scale(c,-1)),cc[j]);c=b;b=next;}
  return add(add(scale(b,s),scale(c,-1)),cc[0]);
}
function trans(x) {x=f(x);if(x===0)return [0,0]; const u=f(Math.sqrt(x)),t=f(u/f(1+u));return scale(mul([H,H],cheb(t)),f(f(Math.sqrt(P))*t));}
function sinc(x) {if(Math.abs(x)<.001){const q=f(x*x);return f(f(1-f(q/6))+f(f(q*q)/120));}return f(f(Math.sin(x))/x);}
function cot(beta,n,kL,side) {
  const nearest=Math.floor(f(f(f(beta+f(side*P))/f(f(2*P)*n))+.5));
  const delta=f(beta-f(f(f(f(2*P)*n)*nearest)-f(side*P)));
  if(Math.abs(delta)<=EPS)return [0,0];
  const t=f(delta*.5), h=f(delta/f(2*n));
  const ratio=f(f(f(f(Math.sign(delta)*n)*sinc(t))*f(Math.cos(h)))/sinc(h));
  const root=f(Math.sqrt(kL)),u=f(f(f(Math.sqrt(2))*root)*Math.abs(f(Math.sin(t))));
  const inv=f(1/f(1+u));
  return scale(mul([H,H],cheb(f(u*inv))),f(f(f(side*f(Math.sqrt(f(2*P))))*f(root*inv))*ratio));
}
function coef(k,n,pi,po,L,sinBeta,boundary) {
  if(n===1)return [0,0];const d=f(po-pi),s=f(po+pi),kL=f(k*L),R=boundary==='soft'?-1:1;
  const a=add(cot(d,n,kL,1),cot(d,n,kL,-1)),b=add(cot(s,n,kL,1),cot(s,n,kL,-1));
  const den=f(f(f(2*n)*f(Math.sqrt(f(f(2*P)*k))))*sinBeta);
  return mul([H,-H],add(a,scale(b,R))).map(x=>f(-x/den));
}
function step(x) {return Math.abs(x)<=EPS?.5:x>0?1:0;}
function go(k,r,n,pi,po,boundary) {
  const d=f(po-pi),s=f(po+pi),R=boundary==='soft'?-1:1,kr=f(k*r),twoNP=f(f(2*n)*P);
  return add(scale(cis(f(kr*f(Math.cos(d)))),step(f(P-Math.abs(d)))),
    add(scale(cis(f(kr*f(Math.cos(s)))),f(R*step(f(P-s)))),
      scale(cis(f(kr*f(Math.cos(f(s-twoNP))))),f(R*step(f(s-f(twoNP-P)))))));
}
function total(args) {const [k,r,n,pi,po,bc]=args;return add(go(...args),scale(mul(coef(k,n,pi,po,r,1,bc),cis(f(-f(k*r)))),f(1/f(Math.sqrt(r)))));}
let maxFloatTransition=0;
for(const c of fixture.transition){const e=U.abs(U.sub(trans(c.x),U.transition(f(c.x))));maxFloatTransition=Math.max(maxFloatTransition,e);assert(e<4e-7);}
report.maxima.float32Transition={cases:fixture.transition.length,maxAbsoluteError:maxFloatTransition};
const seamByKR={};
for(const c of fixture.float32Seam) {
  const value=total(c.args),error=U.abs(U.sub(value,c.expected));
  const kr=c.args[0];const row=seamByKR[kr]??={cases:0,maxAbsoluteError:0,worstOffset:0};++row.cases;
  if(error>row.maxAbsoluteError){row.maxAbsoluteError=error;row.worstOffset=c.offset;}
  assert(value.every(Number.isFinite));
  assert(error < .25, `f32 ${c.args}: ${error}`);
}
report.float32SommerfeldSeams=seamByKR;
report.caveat='f32 emulation is not GPU execution; full canonical field phase error grows with kr. Use stable optical path differences in a renderer.';
console.log(JSON.stringify(report,null,2));
fs.writeFileSync(new URL('../research/utd-kernel-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
