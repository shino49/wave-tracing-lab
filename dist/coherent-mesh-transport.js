import {slabIsInvisible,assertSlabRouteSupported,segmentIntersectsSlab} from './coherent-mesh-slab.js';
/** Shared coherent transport composition. Geometry families and diffraction
 * enter one complex route set before any detector squares the field. */
import {evaluateGeometricRoutes,reducedPhase} from './coherent-mesh-core.js';
import {distance} from './mesh-geometry.js';
import {stableBrokenPathExcess} from './plt-fsd.js';
import {evaluateMeshDiffraction,prepareMeshDiffraction} from './coherent-mesh-diffraction.js';
import {prepareMixedDiffraction,evaluateMixedDiffraction} from './coherent-mesh-mixed.js';
export function coherentRouteCapacity(scene,{diffraction=false,mixedDiffraction=false,blockReflected=false}={}){
 if(mixedDiffraction&&(!diffraction||scene.maxReflections!==2))throw new Error('Mixed diffraction requires diffraction and GO order 2');
 const prepared=diffraction?prepareMeshDiffraction(scene):null,mixed=mixedDiffraction&&!blockReflected?prepareMixedDiffraction(scene):null,count=(blockReflected?1:scene.routeSequences.length)+(slabIsInvisible(scene.slab)?0:1+(blockReflected||scene.maxReflections===0?0:2*scene.mirrorPlanes.length))+(prepared?.eligible.length??0)+(mixed?.count??0);
 if(count>65||prepared?.supported===false)throw new Error('Complete coherent route set exceeds 65 families; no silent path truncation');
 return {count,diffraction:prepared,mixed};
}
export function evaluateMeshTransport(scene,compiled,receiver,{diffraction=false,blockReflected=false,mixedDiffraction=false}={}){
 if(mixedDiffraction&&(!diffraction||scene.maxReflections!==2))throw new Error('Mixed diffraction requires diffraction and GO order 2');
 const geometric=evaluateGeometricRoutes(scene,compiled,receiver,{blockReflected}),routes=geometric.routes;let geometricPower=routes.reduce((s,r)=>s+r.amplitude*r.amplitude,0);
 if(!diffraction)return {...geometric,geometricPower,diffractionDiagnostics:null};
 const d=evaluateMeshDiffraction(scene,compiled.source.position,receiver,{wavelengthMm:compiled.source.wavelengthMm,power:compiled.source.power,referenceDistance:geometric.directDistance});
 if(d.diagnostics.invalidReasons.some(r=>['eligible-edge-safety-cap','coplanar-union-safety-cap','nonfinite-wavenumber','nonfinite-reduced-distance','nonfinite-diffracted-field','phase-resolution-limit'].includes(r)))throw new Error(`Invalid coherent diffraction state: ${d.diagnostics.invalidReasons.join(', ')}`);
 if(d.directBoundary&&segmentIntersectsSlab(scene.slab,compiled.source.position,receiver))throw new Error('Unsupported diffraction boundary intersects dielectric slab');
 if(d.directBoundary){let direct=routes.find(r=>r.kind==='direct');const amplitude=compiled.amplitudeScale/geometric.directDistance;geometricPower+=.25*amplitude*amplitude-(direct?.amplitude??0)**2;if(!direct){direct={kind:'direct',sequence:[],vertices:[compiled.source.position,receiver],pathLength:geometric.directDistance,excess:0,phaseRadReduced:reducedPhase(compiled.source.phaseRad)};routes.unshift(direct);}direct.amplitude=.5*amplitude;direct.interactionPhaseRad=compiled.source.phaseRad;direct.boundary=true;}
 // Reconcile exact first-reflection seams in the SAME angular convention as
 // the UTD pole. A mesh barycentric band must not create a different seam.
 if(!blockReflected&&scene.maxReflections>0){const handled=new Set();for(const seam of d.reflectionBoundaries){const plane=scene.mirrorPlanes.find(p=>p.triangles.some(t=>seam.triangleIds.includes(t.id)));if(!plane||handled.has(plane.id))continue;handled.add(plane.id);const p=seam.point,L=distance(compiled.source.position,p)+distance(p,receiver),excess=stableBrokenPathExcess(compiled.source.position,p,receiver),amplitude=compiled.amplitudeScale/L;let r=routes.find(r=>r.kind==='reflection'&&r.sequence.length===1&&r.sequence[0]===plane.id);geometricPower+=.25*amplitude*amplitude-(r?.amplitude??0)**2;if(!r){r={kind:'reflection',sequence:[plane.id],vertices:[compiled.source.position,p,receiver],pathLength:L,excess};routes.push(r);}r.amplitude=.5*amplitude;r.interactionPhaseRad=compiled.source.phaseRad+(seam.boundary==='soft'?Math.PI:0);r.phaseRadReduced=reducedPhase(r.interactionPhaseRad-compiled.k*excess);r.boundary=true;}}
 for(const r of d.routes)assertSlabRouteSupported(scene.slab,r.vertices??[compiled.source.position,r.point,receiver],'diffraction');
 for(const r of d.routes)routes.push({...r,phaseRadReduced:reducedPhase(r.phaseRadReduced+compiled.source.phaseRad)});
 if(mixedDiffraction&&!blockReflected){
  const mixed=evaluateMixedDiffraction(scene,compiled,receiver);if(!mixed.diagnostics.valid)throw new Error(`Invalid mixed diffraction state: ${mixed.diagnostics.invalidReasons.join(', ')}`);
  const handled=new Map();for(const seam of mixed.goSeams){const key=seam.sequence.join(','),previous=handled.get(key);if(previous){if(JSON.stringify(previous)!==JSON.stringify(seam.mixedSeamEvent))d.diagnostics.warnings.push('joint-mixed-GO-boundaries-need-higher-order-diffraction');continue;}handled.set(key,seam.mixedSeamEvent);const old=routes.find(r=>r.kind==='reflection'&&r.sequence.join(',')===key);geometricPower+=seam.amplitude**2-(old?.amplitude??0)**2;if(old)Object.assign(old,seam);else routes.push(seam);}
  for(const r of mixed.routes)assertSlabRouteSupported(scene.slab,r.vertices,'mixed diffraction');routes.push(...mixed.routes);d.diagnostics.mixed=mixed.diagnostics;d.diagnostics.warnings=[...new Set([...d.diagnostics.warnings,...mixed.diagnostics.warnings])];d.diagnostics.excluded.push(...mixed.diagnostics.excluded);d.diagnostics.edgeEvaluations+=mixed.diagnostics.edgeEvaluations;d.diagnostics.omittedFamilies=d.diagnostics.omittedFamilies.filter(x=>x!=='reflection-before-or-after-diffraction').concat(mixed.diagnostics.omittedFamilies);
 }
 for(const r of routes)if(r.kind!=='transmission')assertSlabRouteSupported(scene.slab,r.vertices??[compiled.source.position,r.point,receiver],r.kind);
 return {routes,geometricPower,directDistance:geometric.directDistance,diffractionDiagnostics:d.diagnostics,slabDiagnostics:geometric.slabDiagnostics};
}
