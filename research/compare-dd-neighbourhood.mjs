// Fixed parameter grid; selected before reference powers were inspected.
// This compares conditional native centre factors, NOT a finite Gaussian mode integral.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {traceDoubleDiffraction} from '../dist/utd-dd.js';
const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url)));
const references=read('./dd-bem/validation.json');
const independent=read('./dd-neighbourhood-independent.json');
const rows=[];
for(const k of [64,128,256,512]){
  const q=Math.sqrt(256/k),ref=references.cases.find(c=>c.fixture==='long_gap'&&c.k===k);
  for(const sigma of [.0675,.07,.0725])for(const aim1 of [.09,.10,.11])for(const aim2 of [.15,.175,.20])for(const y of [-2.25,-2,-1.75]){
    const parameters={k,sigma:sigma*q,aim1:aim1*q,aim2:aim2*q,receiver:[16,y,0]};
    const r=traceDoubleDiffraction(parameters),ri=ref?.receivers.findIndex(p=>p[1]===y),referencePower=ri>=0?ref.reference_conditional_power[ri]:null;
    rows.push({parameters,admittedByNativeConditionalPolicy:r.valid,admissionReasons:r.admissionReasons,
      independentGeometryReviewed:k===256,clipMargins:r.localStages?.map(s=>s.endpointGuardRatio)??[],
      originalCollimationDiagnostic:r.originalCollimationGuard??r.phaseCurvatureDiagnostics??null,
      referenceStatus:ref?.status??null,referenceEmpiricallyConverged:ref?.status.startsWith('empirically converged')??false,
      referencePower,nativePower:r.conditionalPower,
      signedRelativePowerError:r.valid&&referencePower!==null?r.conditionalPower/referencePower-1:null});
  }
}
const groups=[64,128,256,512].map(k=>{
  const group=rows.filter(r=>r.parameters.k===k),errors=group.flatMap(r=>r.signedRelativePowerError===null?[]:[r.signedRelativePowerError]);
  return{k,total:group.length,nativeAdmissionPass:group.filter(r=>r.admittedByNativeConditionalPolicy).length,
    referenceCompared:errors.length,referenceEmpiricallyConverged:group.every(r=>r.referenceEmpiricallyConverged),minSignedPowerError:errors.length?Math.min(...errors):null,maxSignedPowerError:errors.length?Math.max(...errors):null};
});
const report={scope:'Experimental native-style centre conditional powers only; not a calibrated multimode renderer or finite Gaussian scattering validation',
  policy:'Full fixed Cartesian grids; all rejected and admitted cases retained; no post-result error acceptance threshold',
  scaling:'sigma, aim1 and aim2 scale by sqrt(256/k); fixed source, screens and receiver heights',
  coreSourceSHA256:crypto.createHash('sha256').update(fs.readFileSync(new URL('../dist/utd-dd.js',import.meta.url))).digest('hex'),
  independentK256Geometry:independent.summary,groups,rows};
fs.writeFileSync(new URL('./dd-neighbourhood-comparison.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(groups,null,2));
