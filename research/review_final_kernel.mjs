// Independent fixture and conservative region-query verification.
// Does not execute the upstream wave_tracer repository or a GPU.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {parameters,waveSample,coneHits,hash} from '../dist/physics.js';
const dir=path.dirname(fileURLToPath(import.meta.url));
let maxRelativeError=0;
const fixtures=JSON.parse(fs.readFileSync(path.join(dir,'sample-fixtures.json'),'utf8'));
for(const row of fixtures){
 const p=parameters(row.state),got=waveSample(row.X,row.Y,row.xi,row.eta,p);
 const err=Math.abs(got-row.expected)/Math.max(1e-30,Math.abs(row.expected));
 maxRelativeError=Math.max(maxRelativeError,err);
}
let falseNegatives=0,rejected=0,count=0,seed=0x582913;
const uniform=()=>{seed=hash(seed+0x19955);return(seed+.5)/4294967296;};
for(const lambda of [420,550,680])for(const screenDistance of [1.5,2,4])
 for(const slitWidth of [.04,.08,.12])for(const separation of [.18,.3,.42])for(let n=0;n<500;n++){
  const p=parameters({lambda,screenDistance,slitWidth,separation});
  const xi=-8+16*uniform(),eta=-8+16*uniform();
  for(const sign of [-1,1]){
   const lo=sign*separation/2-slitWidth/2,hi=sign*separation/2+slitWidth/2;
   const hit=coneHits(lo,hi,-p.height/2,p.height/2,xi,eta,p);
   const cx=-p.sigmaL*xi,cy=-p.sigmaL*eta;
   const retained=Math.min(hi,cx+p.tail*p.sigmaL)>Math.max(lo,cx-p.tail*p.sigmaL)&&
      Math.min(p.height/2,cy+p.tail*p.sigmaL)>Math.max(-p.height/2,cy-p.tail*p.sigmaL);
   count++;if(!hit)rejected++;if(retained&&!hit)falseNegatives++;
  }
 }
const report={
 status:'Theory and CPU implementation approved for the explicitly restricted scalar/paraxial, single-interaction Gaussian-mode Wave Tracing model.',
 gpuExecutionVerified:false,visualQAVerified:false,
 independentFixtureCount:fixtures.length,maxRelativeError,
 conservativeConeQueries:count,retainedRegionFalseNegatives:falseNegatives,rejectedCandidates:rejected,
 hashes:Object.fromEntries(['physics.js','kernel.wgsl'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(path.join(dir,'../dist',name))).digest('hex')])),
 scope:'Pure Gaussian sensor modes propagated backward through conservative elliptical cones; exact local thin-mask/source overlap; incoherent accumulation across modes; no UTD, polarization, finite-thickness, reflector or multibounce claim.',
 browserSafeguard:'A runtime GPU/CPU test is useful but is not evidence that a GPU was actually exercised during release review. The 16 test points must cover both image axes and the central region.'
};
fs.writeFileSync(path.join(dir,'final-theory-review.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
if(maxRelativeError>1e-8||falseNegatives!==0)process.exit(1);
