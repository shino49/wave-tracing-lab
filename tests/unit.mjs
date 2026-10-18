import assert from 'node:assert/strict';
import fs from 'node:fs';
import {parameters,waveSample,geometric,normals,MODEL} from '../dist/physics.js';
const cases=JSON.parse(fs.readFileSync(new URL('../research/sample-fixtures.json',import.meta.url)));
let worst=0;for(const c of cases){const p=parameters(c.state),v=waveSample(c.X,c.Y,c.xi,c.eta,p);assert(Number.isFinite(v)&&v>=0);const e=Math.abs(v-c.expected)/Math.max(c.expected,1e-20);worst=Math.max(worst,e);assert(e<1e-10);}
const p=parameters({lambda:550,slitWidth:.08,separation:.3,screenDistance:2.5});
for(const X of [0,.5,2,5,12]){assert(Math.abs(waveSample(X,2,.3,-.7,p)-waveSample(-X,-2,-.3,.7,p))<1e-12);assert(geometric(X,0,p)>=0);}
for(let i=0;i<1000;i++)assert(normals(i,12).every(Number.isFinite));
const ids=[...fs.readFileSync(new URL('../dist/index.html',import.meta.url),'utf8').matchAll(/id="([^"]+)"/g)].map(x=>x[1]);assert.equal(ids.length,new Set(ids).size);
for(const file of ['index.html','app.js','physics.js','engine.js','scene.js','style.css','method.html','kernel.wgsl'])assert(fs.statSync(new URL('../dist/'+file,import.meta.url)).size>0);
console.log(JSON.stringify({fixtures:cases.length,worstRelativeError:worst,symmetry:true,nonnegative:true,randomNumbersFinite:true,uniqueIds:ids.length}));
