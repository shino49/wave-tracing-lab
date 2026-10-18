import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import * as c from '../dist/coherent-core.js';
const root=new URL('../',import.meta.url),gold=JSON.parse(fs.readFileSync(new URL('research/coherent-closure/bem-results.json',root))),controls=JSON.parse(fs.readFileSync(new URL('research/coherent-closure/bem-controls.json',root)));
const report={scope:c.COHERENT_SCOPE,createdUtc:new Date().toISOString(),source:'Independent NumPy/SciPy cosine-product quadrature artifacts, plus runtime JavaScript solve and separate algebra controls',thresholds:{rawReference:2e-7,residual:1e-10,energy:5e-4,gridMesh:2e-5,algebra:2e-11},cases:{},checks:{}};
const flat=z=>Float64Array.from(z.flat(Infinity)),err=c.relativeError,mat=c.multiplyComplex;
const less=(x,t,message)=>assert.ok(Number.isFinite(x)&&x<t,`${message}: ${x} >= ${t}`);
const sum=(x,w,scale=1)=>x.reduce((s,v,i)=>s+scale*w[i]*v,0);
function adjoint(a,n,m){const b=new Float64Array(a.length);for(let i=0;i<n;i++)for(let j=0;j<m;j++){b[2*(j*n+i)]=a[2*(i*m+j)];b[2*(j*n+i)+1]=-a[2*(i*m+j)+1];}return b;}
function intensityGeneral(f,l,rows,modes,rank){const u=mat(f,l,rows,modes,rank),o=new Float64Array(rows);for(let i=0;i<rows;i++)for(let r=0;r<rank;r++)o[i]+=u[2*(i*rank+r)]**2+u[2*(i*rank+r)+1]**2;return o;}
function traceProduct(a,b,n=3){let re=0,im=0;for(let i=0;i<n;i++)for(let j=0;j<n;j++){const p=2*(i*n+j),q=2*(j*n+i);re+=a[p]*b[q]-a[p+1]*b[q+1];im+=a[p]*b[q+1]+a[p+1]*b[q];}return [re,im];}
function selectedGridError(a,b){let d=0,s=0,max=0;for(let i=0;i<a.mask.length;i++)if(a.mask[i]&&b.mask[i])for(let j=0;j<6;j++){const p=6*i+j,v=a.fieldsRaw[p]-b.fieldsRaw[p];d+=v*v;s+=b.fieldsRaw[p]**2;max=Math.max(max,Math.abs(v));}return {relative:Math.sqrt(d/s),maxAbsolute:max};}
const solveStart=performance.now();
for(const screenCount of [1,2]){
 const name=screenCount===1?'one_screen':'two_screens',artifact=gold.cases[name].full_complex_artifact,runs=[64,96,144].map(n=>c.solveCoherent({screenCount,n})),fine=runs[2],nearXY=Float64Array.from(artifact.near_plane_y.flatMap(y=>[5,y]));
 const fields=c.evaluatePoints(fine,nearXY).fieldsRaw,far=c.farFields(fine,artifact.far_field_theta),nearError=err(fields,flat(artifact.near_plane_total_raw_gaussian_fields_complex)),farError=err(far,flat(artifact.far_field_total_outgoing_raw_gaussian_fields_complex)),densityError=err(fine.boundary.phiRaw,flat(artifact.weighted_boundary_density_raw_complex));
 less(nearError,2e-7,`${name} full401 near raw field`);less(farError,2e-7,`${name} full800 far raw field`);less(densityError,2e-7,`${name} boundary144 density`);
 const g64=c.evaluateGrid(runs[0]),g96=c.evaluatePoints(runs[1],g64.xy),gridMesh=selectedGridError(g64,g96);less(gridMesh.relative,2e-5,`${name} 64to96 visible grid`);
 const g96own=c.evaluateGrid(runs[1]),g144=c.evaluatePoints(runs[2],g96own.xy),gridMesh96to144=selectedGridError(g96own,g144);less(gridMesh96to144.relative,2e-5,`${name} 96to144 visible grid`);
 const near64=c.evaluatePoints(runs[0],nearXY).fieldsRaw,near96=c.evaluatePoints(runs[1],nearXY).fieldsRaw,meshError=err(near64,near96);less(meshError,2e-7,`${name} 64to96 near plane`);
 const srcRef=flat(artifact.source_gram_complex);less(err(fine.sources.gram,srcRef),2e-11,'source flux Gram');
 for(const run of runs){less(run.diagnostics.relativeResidual,1e-10,'residual');less(run.diagnostics.energyOperatorError,5e-4,'energy');assert.ok(run.diagnostics.reference.pass);}
 // Exact reconstruction of reference's different saved source density using its own whitening map.
 const oldL=new Float64Array(24),oldV=[1/1.5,0,Math.cos(.7)/1.5,Math.sin(.7)/1.5,Math.cos(-.3)/3,Math.sin(-.3)/3];
 for(let m=0;m<3;m++){oldL[8*m]=Math.sqrt(.55)*oldV[2*m];oldL[8*m+1]=Math.sqrt(.55)*oldV[2*m+1];oldL[2*(m*4+m+1)]=Math.sqrt(.45*[.2,.5,.3][m]);}
 const oldRaw=mat(flat(artifact.source_whitening_transform_complex),oldL,3,3,4),detRaw=c.farFields(fine,artifact.detector_theta),oldSignal=sum(c.intensityFromFields(detRaw,oldRaw),artifact.detector_angle_weights,12),oldExpected=gold.cases[name].mesh_summaries[2].density_measurement.mixed_signal;
 less(Math.abs(oldSignal-oldExpected),2e-11,'saved reference source-basis/covariance equivalence');
 report.cases[name]={rawNear401Error:nearError,rawFar800Error:farError,rawBoundary144Error:densityError,mesh64to96Near:meshError,mesh64to96VisibleGrid:gridMesh,mesh96to144VisibleGrid:gridMesh96to144,maskedPixels:g64.diagnostics.maskedPixels,visiblePixels:g64.diagnostics.validPixels,referenceMixedSignal:oldSignal,referenceMixedSignalError:Math.abs(oldSignal-oldExpected),residuals:runs.map(s=>s.diagnostics.relativeResidual),energyOperatorErrors:runs.map(s=>s.diagnostics.energyOperatorError),runtimeReferences:runs.map(s=>s.diagnostics.reference)};
}
const s=c.solveCoherent({gamma:.45,phase:.7}),base=c.measureCoherent(s,{gamma:0}),end=c.measureCoherent(s,{gamma:1,phase:.7}),mix=c.measureCoherent(s,{gamma:.45,phase:.7});
less(Math.abs(mix.detectorFlux-(.55*base.detectorFlux+.45*end.detectorFlux)),2e-12,'partial coherence interpolation');
const phase0=c.measureCoherent(s,{gamma:0,phase:0}),phase1=c.measureCoherent(s,{gamma:0,phase:1.7});assert.equal(phase0.detectorFlux,phase1.detectorFlux);
const phases=Array.from({length:17},(_,i)=>2*Math.PI*i/16),phaseFlux=phases.map(phase=>c.measureCoherent(s,{gamma:1,phase}).detectorFlux),phaseRange=Math.max(...phaseFlux)-Math.min(...phaseFlux);assert.ok(phaseRange>.001,'coherent interference must be observable');
for(const gamma of [0,.1,.5,1])for(const phase of [0,.7,2.3])for(const inputPower of [0,.03,1,5]){
 const m=c.measureCoherent(s,{gamma,phase,inputPower}),one=c.measureCoherent(s,{gamma,phase,inputPower:1}),state=m.state,rho=mat(state.factorRaw,adjoint(state.factorRaw,3,4),3,4,3),power=traceProduct(s.sources.gram,rho),signal=traceProduct(m.detectorMatrixRaw,rho);
 less(Math.abs(power[0]-inputPower),2e-11,'Gram-metric source power');less(Math.abs(power[1]),2e-11,'real source power');less(Math.abs(m.detectorFlux-one.detectorFlux*inputPower),2e-11,'input power scaling');less(Math.abs(signal[0]-m.detectorFlux),2e-11,'density matrix/factor detector equivalence');assert.ok(m.detectorFlux>=0&&m.detectorFlux<=inputPower+1e-10);
 const orthRho=mat(state.factorOrthonormal,adjoint(state.factorOrthonormal,3,4),3,4,3);for(let j=0;j<3;j++)less(Math.abs(orthRho[2*(3*j+j)]-inputPower*[.2,.5,.3][j]),2e-12,'coherence keeps populations fixed');
}
report.checks.coherence={partialInterpolationError:Math.abs(mix.detectorFlux-(.55*base.detectorFlux+.45*end.detectorFlux)),gammaZeroPhaseInvariant:true,populationsFixed:true,sourceGramPowerCorrect:true,powerScaling:true,nonnegativePSDMeasurements:true,phasePeakToPeak:phaseRange,phaseFlux};
// Arbitrary invertible complex source-coordinate change: F'=F B, L'=B^-1 L.
const B=new Float64Array([1,0,.2,.1,-.1,.1, .1,-.2,1.3,0,.3,.1, -.2,0,.1,.2,.8,0]),state=c.createState(s,{gamma:.61,phase:1.1}),fields=s.angular.fieldsRaw,changed=mat(fields,B,fields.length/6,3,3),newL=c.complexSolve(B,state.factorRaw,3,4).solution;
const originalIntensity=c.intensityFromFields(fields,state.factorRaw),changedIntensity=c.intensityFromFields(changed,newL),basisError=err(changedIntensity,originalIntensity);less(basisError,2e-11,'invertible source basis invariance');
// Redundant frame refinement [F/sqrt2,F/sqrt2], with dual factors [L/sqrt2;L/sqrt2].
const rows=fields.length/6,overF=new Float64Array(rows*12),overL=new Float64Array(6*4*2);for(let i=0;i<rows;i++)for(let m=0;m<6;m++)for(let z=0;z<2;z++)overF[2*(i*6+m)+z]=fields[2*(i*3+m%3)+z]/Math.sqrt(2);
for(let m=0;m<6;m++)for(let r=0;r<4;r++)for(let z=0;z<2;z++)overL[2*(m*4+r)+z]=state.factorRaw[2*((m%3)*4+r)+z]/Math.sqrt(2);
const refinementError=err(intensityGeneral(overF,overL,rows,6,4),originalIntensity);less(refinementError,2e-11,'overcomplete source-frame refinement');
report.checks.sourceBasis={invertibleIntensityError:basisError,redundantRefinementIntensityError:refinementError};
// Coarse solve admits conservation but must fail independent complex accuracy: retained negative control.
const coarseSys=c.assembleSystem(s.screens,12),coarseIncident=c.incidentFields(s.sources,coarseSys.xy),coarsePhi=c.complexSolve(coarseSys.matrix,Float64Array.from(coarseIncident,v=>-v),coarseSys.nodes).solution,coarse={...s,n:12,boundary:{xy:coarseSys.xy,nodes:coarseSys.nodes,phiRaw:coarsePhi,weightedPhiRaw:Float64Array.from(coarsePhi,x=>coarseSys.h*x)},angular:null,detectorCache:null};
const badFar=c.farFields(coarse,s.angular.theta),coarseError=err(badFar,s.angular.fieldsRaw),coarseEnergy=c.measureCoherent(coarse).diagnostics.energyOperatorError;assert.ok(coarseError>.1);less(coarseEnergy,5e-4,'coarse control remains conservative');
report.checks.coarseNegativeControl={nodes:12,farComplexRelativeError:coarseError,energyOperatorError:coarseEnergy,conservationDoesNotCertifyAccuracy:true};
// One-pass cascade deliberately omits feedback into screen 1.
const sys=c.assembleSystem(s.screens,64),inc=c.incidentFields(s.sources,sys.xy),n=64,a11=new Float64Array(2*n*n),a22=new Float64Array(2*n*n),a21=new Float64Array(2*n*n);
for(let i=0;i<n;i++)for(let j=0;j<n;j++)for(let z=0;z<2;z++){a11[2*(i*n+j)+z]=sys.matrix[2*(i*128+j)+z];a22[2*(i*n+j)+z]=sys.matrix[2*((i+n)*128+j+n)+z];a21[2*(i*n+j)+z]=sys.matrix[2*((i+n)*128+j)+z];}
const p1=c.complexSolve(a11,Float64Array.from(inc.slice(0,n*6),v=>-v),n).solution,feedback=mat(a21,p1,n,n,3),rhs2=Float64Array.from(inc.slice(n*6),(v,i)=>-v-feedback[i]),p2=c.complexSolve(a22,rhs2,n).solution,onePassPhi=Float64Array.from([...p1,...p2]),onePass={...s,boundary:{...s.boundary,phiRaw:onePassPhi,weightedPhiRaw:Float64Array.from(onePassPhi,v=>v*Math.PI/n)}};
const onePassFarError=err(c.farFields(onePass,s.angular.theta),s.angular.fieldsRaw);assert.ok(onePassFarError>.01,'two-way coupling must affect solution');report.checks.twoWayCoupling={onePassFarComplexError:onePassFarError};
// Angular-aperture partition, including wrapped back-facing detector windows.
const front=c.measureCoherent(s,{detectorCenter:0,detectorHalfAngle:Math.PI/2}),back=c.measureCoherent(s,{detectorCenter:Math.PI,detectorHalfAngle:Math.PI/2}),full=c.measureCoherent(s,{detectorCenter:0,detectorHalfAngle:Math.PI});
less(Math.abs(front.detectorFlux+back.detectorFlux-full.detectorFlux),2e-10,'angular detector partition');less(Math.abs(full.detectorFlux-full.totalFlux),2e-10,'full aperture equals total outgoing flux');
report.checks.detectorPartition={error:Math.abs(front.detectorFlux+back.detectorFlux-full.detectorFlux),fullApertureError:Math.abs(full.detectorFlux-full.totalFlux)};
// Source angular quadrature refinement, compared at the same physical points.
const src640=c.solveCoherent({nq:480}),sampleXY=Float64Array.from(c.COHERENT_REFERENCE.two_screens.xy),sourceQuadratureError=err(c.evaluatePoints(s,sampleXY).fieldsRaw,c.evaluatePoints(src640,sampleXY).fieldsRaw);less(sourceQuadratureError,2e-7,'source angular quadrature refinement');report.checks.sourceQuadratureError=sourceQuadratureError;
// Actual finite curvature by exact angular-spectrum y derivatives, independently saved gold values.
const curvatures=[];for(const d of controls.incident_curvature_diagnostics){const m=d.raw_gaussian_mode-1,x=d.screen_x,y=d.central_ray_y;let ur=0,ui=0,dr=0,di=0,ddr=0,ddi=0;for(let j=0;j<s.sources.nq;j++){const p=6*j+2*m,phase=-(x*s.sources.kx[j]+y*s.sources.q[j]),co=Math.cos(phase),si=Math.sin(phase),ar=s.sources.weightedSpectrumRaw[p],ai=s.sources.weightedSpectrumRaw[p+1],re=co*ar-si*ai,im=co*ai+si*ar,q=s.sources.q[j];ur+=re;ui+=im;dr+=q*im;di-=q*re;ddr-=q*q*re;ddi-=q*q*im;}const den=ur*ur+ui*ui,rr=(dr*ur+di*ui)/den,ri=(di*ur-dr*ui)/den,curvature=(ddi*ur-ddr*ui)/den-2*rr*ri;less(Math.abs(curvature-d.phase_curvature),2e-10,'exact source phase curvature');assert.ok(Math.abs(curvature)>.1);curvatures.push(curvature);}report.checks.phaseCurvature=curvatures;
// Bounded-geometry corners plus centre; actual JS64↔96 image and angular comparison.
report.geometrySweep=[];for(const separation of [1.8,2.4,3])for(const offset of [-.2,.4,1]){
 const lo=c.solveCoherent({separation,offset,n:64}),hi=c.solveCoherent({separation,offset,n:96}),g=c.evaluateGrid(lo,{width:65,height:49}),h=c.evaluatePoints(hi,g.xy),ge=selectedGridError(g,h),fe=err(lo.angular.fieldsRaw,hi.angular.fieldsRaw);less(ge.relative,2e-5,'bounded geometry visible grid convergence');less(fe,2e-7,'bounded geometry far convergence');report.geometrySweep.push({separation,offset,fieldGrid64to96:ge,far64to96:fe,residual:lo.diagnostics.relativeResidual,energyOperatorError:lo.diagnostics.energyOperatorError});
}
// Type/finite/mask and explicit rejection contracts.
assert.throws(()=>c.solveCoherent({separation:0}));assert.throws(()=>c.solveCoherent({offset:4}));assert.throws(()=>c.solveCoherent({n:16}));assert.throws(()=>c.measureCoherent(s,{gamma:-.1}));assert.throws(()=>c.measureCoherent(s,{inputPower:-1}));
const onScreen=c.evaluatePoints(s,new Float64Array([0,0,2.3,.4]));assert.deepEqual(Array.from(onScreen.mask),[0,0]);assert.ok(onScreen.fieldsRaw.every(v=>v===0));assert.ok(onScreen.intensity.every(v=>v===0));
report.checks.maskedSingularPointsZero=true;report.elapsedMs=performance.now()-solveStart;report.sourceSha256=crypto.createHash('sha256').update(fs.readFileSync(new URL('dist/coherent-core.js',root))).digest('hex');report.allPass=true;
fs.writeFileSync(new URL('research/coherent-browser-core-validation.json',root),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({allPass:true,cases:report.cases,checks:report.checks,geometrySweep:report.geometrySweep,elapsedMs:report.elapsedMs},null,2));
