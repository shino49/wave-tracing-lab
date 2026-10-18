/** Independent admission tests. No production edits, no finite-crop renormalization.
 * Oracle fixtures were produced by direct GL Hankel quadrature, not the Zernike
 * expansion used in production. Rebuild with research/pinhole-independent-oracle.py.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {makeFresnelEvaluator,generateKernels,renderPointDetector,buildPaddedKernel,
  integratePixelXYZ,geometricPsfRadius,diskRectangleArea} from '../dist/pinhole-optics.js';
import {pinholeDepthLayers,pinholeDepthWeights,projectPinholePoint} from '../dist/pinhole-scene.js';
const hooks={yield:()=>{}};
const report={scope:'Independent scalar-Fresnel math and sampling admission; no GPU or full-Maxwell validation',passed:false};
const close=(a,b,t,label)=>assert.ok(Math.abs(a-b)<=t,`${label}: ${a} != ${b} (tolerance ${t})`);
const comparison=(a,b)=>{let difference=0,sum=0;for(let i=1;i<a.length;i+=3){difference+=Math.abs(a[i]-b[i]);sum+=b[i];}return difference/sum;};

// A genuinely separate formulation and library, including oscillatory far tails.
const oracle=JSON.parse(fs.readFileSync(new URL('../research/pinhole-independent-oracle.json',import.meta.url)));
let worstScaled=0,worstRelative=0;
for(const c of oracle.cases){
  const got=makeFresnelEvaluator(c).density(c.radiusMm),scale=Math.PI*(c.diameterMm/2)**2/(c.wavelengthNm*1e-6*50)**2;
  const scaled=Math.abs(got-c.density)/scale,relative=Math.abs(got-c.density)/Math.max(c.density,1e-12*scale);
  worstScaled=Math.max(worstScaled,scaled);worstRelative=Math.max(worstRelative,relative);
  assert.ok(scaled<2e-11&&relative<1e-6,'Independent Hankel quadrature mismatch');
}
report.hankel={cases:oracle.cases.length,worstPeakScaledError:worstScaled,worstLocalRelativeError:worstRelative,oracle512To1024Error:oracle.max512To1024ErrorScaled};

// Closed-form on-axis chirped pupil, pupil area exactly once, Airy width limit.
for(const d of [.04,.24,.5])for(const nm of [360,550,830]){
 const e=makeFresnelEvaluator({diameterMm:d,depthMm:200,wavelengthNm:nm});
 const h0=e.areaMm2/(nm*1e-6*50)**2*(Math.sin(e.beta/2)/(e.beta/2))**2;
 close(e.density(0),h0,1e-10*Math.max(1,h0),'On-axis analytic integral');
}
const halfWidth=nm=>{const e=makeFresnelEvaluator({diameterMm:.04,depthMm:200,wavelengthNm:nm});let low=0,high=2;for(let i=0;i<60;i++){const mid=(low+high)/2;if(e.density(mid)>e.density(0)/2)low=mid;else high=mid;}return (low+high)/2;};
const blueHalfWidth=halfWidth(360),redHalfWidth=halfWidth(830),widthRatio=redHalfWidth/blueHalfWidth;
close(widthRatio,830/360,.001,'Airy-limit wavelength scaling');
const airyRoot=3.8317059702075125;
for(const nm of [360,550,830]){const e=makeFresnelEvaluator({diameterMm:.04,depthMm:200,wavelengthNm:nm}),r=airyRoot*nm*1e-6*50/(Math.PI*.04);assert.ok(e.density(r)/e.density(0)<1e-4,'First Airy minimum filled only weakly by finite-distance defocus');}
report.wavelength={blueHalfWidthMm:blueHalfWidth,redHalfWidthMm:redHalfWidth,widthRatio,expectedRatio:830/360};

// Enlarge a physically integrated disk: finite mass approaches one from below.
report.infinitePlane=[];
for(const diameterMm of [.04,.5])for(const wavelengthNm of [360,830]){
 const e=makeFresnelEvaluator({diameterMm,depthMm:200,wavelengthNm}),masses=[];let sum=0;
 for(let i=0;i<64000;i++){const r=(i+.5)*.001;sum+=2*Math.PI*r*.001*e.density(r);if([3999,15999,63999].includes(i))masses.push(sum);}
 assert.ok(masses[0]<masses[1]&&masses[1]<masses[2]&&masses[2]<1.00002&&masses[2]>.996,'Infinite-plane mass convergence');
 report.infinitePlane.push({diameterMm,wavelengthNm,radiiMm:[4,16,64],masses});
}

// Default eight inverse-depth layers: quantify rather than hide approximation.
const depths=Array.from(pinholeDepthLayers(8));let worstDepth={relativeL1:0};
for(const wavelengthNm of [360,550,830])for(let j=0;j<depths.length-1;j++){
 const depthMm=2/(1/depths[j]+1/depths[j+1]),weights=pinholeDepthWeights(depthMm,depths),o={diameterMm:.5,wavelengthNm};
 close(weights.weights[0],.5,1e-12,'Reciprocal depth midpoint');
 const a=makeFresnelEvaluator({...o,depthMm:depths[j]}),b=makeFresnelEvaluator({...o,depthMm:depths[j+1]}),e=makeFresnelEvaluator({...o,depthMm});
 let l1=0,mass=0,maxError=0,peak=0;
 for(let i=0;i<6000;i++){const r=(i+.5)*.001,w=2*Math.PI*r*.001,v=e.density(r),error=Math.abs(v-(a.density(r)+b.density(r))/2);l1+=error*w;mass+=v*w;peak=Math.max(peak,v);maxError=Math.max(maxError,error);}
 const row={wavelengthNm,depthMm,intervalIndex:j,relativeL1:l1/mass,maxErrorOverPeak:maxError/peak};
 if(row.relativeL1>worstDepth.relativeL1)worstDepth=row;
 assert.ok(row.relativeL1<.0003&&row.maxErrorOverPeak<.0025,'Eight-layer depth approximation regression');
}
report.depthInterpolation={depthsMm:depths,worst:worstDepth};

// Actual native-512 finite pixels, including demanding spectral/aperture endpoints.
report.pixelRefinement=[];
for(const [diameterMm,wavelengthNm] of [[.04,830],[.24,550],[.5,360],[.5,830]]){
 const options={resolution:512,diameterMm,depths:[200],basisNames:['neutral'],mode:'narrowband',wavelengthNm};
 const k=await generateKernels(options,hooks),ref=await generateKernels({...options,radialStepMm:k.radialStepMm/4},hooks);
 const p4=await renderPointDetector(k,{pixelQuadrature:4},hooks),p8=await renderPointDetector(k,{pixelQuadrature:8},hooks),r8=await renderPointDetector(ref,{pixelQuadrature:8},hooks);
 const row={diameterMm,wavelengthNm,radialStepMm:k.radialStepMm,pixel4To8RelativeL1:comparison(p4.xyz,p8.xyz),radialRefinementRelativeL1:comparison(p8.xyz,r8.xyz),combinedRelativeL1:comparison(p4.xyz,r8.xyz),powerRelativeError:Math.abs(p4.powerXYZ[1]-r8.powerXYZ[1])/r8.powerXYZ[1],detectorCollectedFraction:p4.powerXYZ[1]/(k.throughput*k.layers[0].basis[0].colorXYZ[1])};
 assert.ok(row.combinedRelativeL1<.0015&&row.powerRelativeError<.0004,'Native512 pixel/radial refinement regression');
 assert.ok(row.detectorCollectedFraction>0&&row.detectorCollectedFraction<1,'Finite-detector power must retain tail losses');
 report.pixelRefinement.push(row);
 if(diameterMm===.24){
  // Same physical point at an exact half-pixel center, independent of resolution.
  const fine=await generateKernels({...options,resolution:1024,radialStepMm:ref.radialStepMm},hooks),f8=await renderPointDetector(fine,{pixelQuadrature:8},hooks);
  let difference=0,magnitude=0;for(let y=0;y<512;y++)for(let x=0;x<512;x++){let avg=0;for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)avg+=f8.xyz[3*((2*y+dy)*1024+2*x+dx)+1]/4;const v=r8.xyz[3*(y*512+x)+1];difference+=Math.abs(v-avg);magnitude+=avg;}
  report.resolution={coarse:512,fine:1024,relativeL1:difference/magnitude,powerRelativeError:Math.abs(r8.powerXYZ[1]-f8.powerXYZ[1])/f8.powerXYZ[1]};
  assert.ok(report.resolution.relativeL1<.0001&&report.resolution.powerRelativeError<.00002,'Exact-point irradiance resolution invariance');
 }
}

// Exact finite-hole geometry; no unfair perfect geometric point comparison.
for(const diameterMm of [.04,.5])for(const depthMm of [200,300]){
 const r=geometricPsfRadius({diameterMm,depthMm});close(r,diameterMm/2*(1+50/depthMm),1e-15,'Geometric finite-aperture radius');
 close(diskRectangleArea(r,-1,1,-1,1),Math.PI*r*r,1e-14,'Full geometric aperture mass');
 close(diskRectangleArea(r,0,1,0,1),Math.PI*r*r/4,1e-14,'Geometric quadrant mass');
}
const projection=projectPinholePoint([10,20,-250]);assert.equal(projection.xMm,-2);assert.equal(projection.yMm,-4);assert.equal(projection.magnification,-.2);
report.projection={inputPointMm:[10,20,-250],detectorPointMm:[projection.xMm,projection.yMm],magnification:projection.magnification};

// Difference-grid coverage is not periodic extension of the physical PSF.
const indexOptions={resolution:16,diameterMm:.04,depths:[200],basisNames:['neutral'],mode:'narrowband',wavelengthNm:830,radialStepMm:.002};
const k=await generateKernels(indexOptions,hooks),padded=await buildPaddedKernel(k,0,0,{},hooks),m=padded.paddedSize;
assert.equal(m,32);for(const [dx,dy] of [[0,0],[1,0],[-1,0],[15,15],[-15,-15],[-15,15]]){
 const expect=integratePixelXYZ(k,0,0,dx*k.pixelWidthMm,dy*k.pixelWidthMm),idx=((dy+m)%m)*m+(dx+m)%m;
 for(let c=0;c<3;c++)close(padded.channels[c][idx],Math.fround(expect[c]*k.throughput),1e-30,'Nonperiodic signed lag embedding');
}
for(let i=0;i<m;i++)assert.equal(padded.channels[1][16*m+i]+padded.channels[1][i*m+16],0,'Unused gap cannot contain wrapped tails');
report.linearConvolution={sourceSide:16,paddedSide:m,verifiedExtremeLag:15,unusedGapZero:true};
report.visibilityScope={model:'Center-pupil first-surface geometry plus circular full-pupil PSFs',fullDepthWorstParallaxMm:50*.25*(1/200-1/300),fullDepthWorstParallaxPixels512:50*.25*(1/200-1/300)/(8/512),limitation:'Finite-pupil partial occlusion and newly visible source patches require a source-dependent pupil; rim diagnostics are not a mathematical error bound.'};
report.passed=true;
fs.writeFileSync(new URL('../research/pinhole-independent-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
console.log('Independent pinhole physical derivation and sampling checks passed.');
