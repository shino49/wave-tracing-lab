/** One ideal infinite scalar mirror composed with ONE coherent UTD region.
 * This is a selected path family, not general multi-region wave transport.
 * Reflection is an isometry: the original waist, propagation age, local GO+D
 * field and total unfolded range are retained. No new amplitude, pdf, inverse
 * square or Gaussian weight is introduced. Only |E|² is passed to the kernel.
 */
import {add,sub,scale,dot,cross,length,distance,normalize,prepareMesh} from './mesh-geometry.js';
import {prepareUTDScene,evaluatePoint} from './utd-scene.js';

const vec=(v,name)=>{
  if(!Array.isArray(v)||v.length!==3||v.some(x=>!Number.isFinite(x)))throw new Error(`${name} requires three finite coordinates.`);
  return [...v];
};
function plane(mirror) {
  const point=vec(mirror?.point,'Mirror point'),raw=vec(mirror?.normal,'Mirror normal');
  if(length(raw)<1e-12)throw new Error('Mirror normal must be nonzero.');
  return {point,normal:normalize(raw)};
}
export function reflectPoint(point,mirror) {
  const m=plane(mirror);return sub(point,scale(m.normal,2*dot(sub(point,m.point),m.normal)));
}
export function reflectDirection(direction,mirror) {
  const m=plane(mirror);return sub(direction,scale(m.normal,2*dot(direction,m.normal)));
}
export function mirrorIntersection(a,b,mirror) {
  const m=plane(mirror),d=sub(b,a),den=dot(d,m.normal);
  if(Math.abs(den)<1e-14*length(d))return null;
  const t=dot(sub(m.point,a),m.normal)/den;
  return t>0&&t<1?{point:add(a,scale(d,t)),fraction:t}:null;
}
function corners(p) {
  return [-1,1].flatMap(u=>[-1,1].map(v=>add(p.detectorCenter,add(scale(p.detectorU,u*p.detectorWidthMm/2),scale(p.detectorV,v*p.detectorHeightMm/2)))));
}
function extent(points,frame,origin) {
  const coordinates=points.map(p=>frame.map(axis=>dot(sub(p,origin),axis)));
  return {min:frame.map((_,i)=>Math.min(...coordinates.map(p=>p[i]))),max:frame.map((_,i)=>Math.max(...coordinates.map(p=>p[i])))};
}
function intersects(a,b,eps) {return a.min.every((x,i)=>x<=b.max[i]+eps&&a.max[i]>=b.min[i]-eps);}

export function prepareReflectedUTDScene(input,state={}) {
  const mirror=plane(state.mirror),order=state.mirror?.order;
  if(!['before','after'].includes(order))throw new Error('Mirror order must be before or after the local UTD region.');
  const mesh=input?.triangles?input:prepareMesh(input);
  const sourceOrigin=vec(state.sourceOrigin??[0,0,-5],'Physical source'),incidentDirection=normalize(vec(state.incidentDirection??[0,0,1],'Physical incident direction'));
  const detectorCenter=vec(state.detectorCenter??[0,0,20],'Physical detector'),detectorU=normalize(vec(state.detectorU??[1,0,0],'Physical detector u')),detectorV=normalize(vec(state.detectorV??[0,1,0],'Physical detector v'));
  const effective={...state,sourceOrigin,incidentDirection,detectorCenter,detectorU,detectorV};
  delete effective.mirror;
  if(order==='before') {
    effective.sourceOrigin=reflectPoint(sourceOrigin,mirror);
    effective.incidentDirection=reflectDirection(incidentDirection,mirror);
  } else {
    effective.detectorCenter=reflectPoint(detectorCenter,mirror);
    effective.detectorU=reflectDirection(detectorU,mirror);
    effective.detectorV=reflectDirection(detectorV,mirror);
  }
  const scene=prepareUTDScene(mesh,effective),p=scene.params;
  const physical={...p,sourceOrigin,incidentDirection,detectorCenter,detectorU,detectorV};
  const issues=[];
  const reject=(code,message)=>issues.push({code,message});
  const relativeScale=Math.max(1,mesh.bounds.radius,distance(sourceOrigin,mirror.point),distance(detectorCenter,mirror.point));
  // Larger than the kernel's f32 ray-position tolerance (2.5e-7*scale),
  // including accumulated coordinate/Householder roundoff near a contact.
  const eps=Math.max(1e-5,1e-5*relativeScale);
  const signed=x=>dot(sub(x,mirror.point),mirror.normal);
  const values=mesh.vertices.map(signed),lo=Math.min(...values),hi=Math.max(...values);
  const side=lo>eps?1:hi< -eps?-1:0;
  if(!side)reject('mirror-crossing-geometry','The entire mesh must lie strictly on one side of the ideal mirror.');
  const physicalCorners=corners(physical),effectiveCorners=corners(p);
  if(side&&([sourceOrigin,...physicalCorners].some(x=>side*signed(x)<=eps)))
    reject('mirror-wrong-halfspace','Source and the entire physical receiver must share the mesh halfspace.');
  const sourceDiskSupport=scene.incomingCone.radius0*Math.sqrt(Math.max(0,1-dot(mirror.normal,incidentDirection)**2));
  if(side&&side*signed(sourceOrigin)<=sourceDiskSupport+eps)
    reject('mirror-source-footprint','The retained 3-sigma source disk touches or crosses the mirror.');
  const nd=dot(mirror.normal,p.incidentDirection),toward=order==='before'?side*nd:-side*nd;
  const alpha=Math.atan(scene.incomingCone.tanAngle),coneMargin=toward*Math.cos(alpha)-Math.sqrt(Math.max(0,1-toward*toward))*Math.sin(alpha);
  if(!side||coneMargin<=p.minSinBeta)
    reject('mirror-grazing-order','The incident cone is grazing or points in the wrong direction for the selected mirror order.');
  if(mesh.vertices.some(x=>dot(sub(x,p.sourceOrigin),p.incidentDirection)<=eps))
    reject('mirror-source-order','All local geometry must be forward of the unfolded source phase plane.');
  if(!scene.region)reject('mirror-missing-region','The selected family must contain an actual local UTD interaction region.');
  if(scene.region) {
    const support=scene.region.radius*Math.sqrt(1+3*nd*nd);
    if(side*signed(scene.region.center)<=support+eps)
      reject('mirror-region-overlap','The complete retained interaction ellipsoid touches the mirror; distinct reflection and UTD regions are required.');
  }
  if(order==='after'&&side) {
    const separation=Math.min(...values.map(x=>side*x))+Math.min(...effectiveCorners.map(x=>-side*signed(x)));
    const maximumRange=Math.max(...mesh.vertices.flatMap(a=>effectiveCorners.map(b=>distance(a,b))));
    const minCos=separation/Math.max(maximumRange,eps);
    if(minCos*Math.cos(alpha)-Math.sqrt(Math.max(0,1-minCos*minCos))*Math.sin(alpha)<=p.minSinBeta)
      reject('mirror-grazing-output','The retained receiver family approaches a grazing mirror reflection.');
  }
  // Every selected segment is a straight line between a source-phase-plane
  // projection, a point on the original mesh, and/or a receiver point. Their
  // convex hull lies within this oriented bounding box. If ANY triangle of the
  // reflected mesh overlaps it, reject the WHOLE family. Thus no folded leg can
  // hide an extra mesh contact or an unpaired image-edge shadow transition.
  // This deliberately rejects some clear layouts; it never admits on a sample.
  let u=sub(mirror.normal,scale(p.incidentDirection,nd));
  if(length(u)<1e-8)u=sub(p.detectorU,scale(p.incidentDirection,dot(p.detectorU,p.incidentDirection)));
  if(length(u)<1e-8)u=sub(p.detectorV,scale(p.incidentDirection,dot(p.detectorV,p.incidentDirection)));
  u=normalize(u);const frame=[u,normalize(cross(p.incidentDirection,u)),p.incidentDirection];
  const points=[...mesh.vertices,...effectiveCorners];
  const sourceProjections=points.map(x=>sub(x,scale(p.incidentDirection,dot(sub(x,p.sourceOrigin),p.incidentDirection))));
  if(side&&sourceProjections.some(x=>(order==='before'?-side:side)*signed(x)<=eps))
    reject('mirror-projected-source-side','The local plane-wave source projections cross the mirror; not every retained path has the selected reflection order.');
  const swept=extent([...points,...sourceProjections],frame,mirror.point);
  const possibleBlockers=mesh.triangles.filter(t=>intersects(swept,extent([t.a,t.b,t.c].map(x=>reflectPoint(x,mirror)),frame,mirror.point),eps)).map(t=>t.id);
  if(possibleBlockers.length)reject('mirror-extra-contact','The conservative folded-path bound meets image geometry. Extra contacts or shadow transitions are unsupported; this entire family is rejected.');
  const localCenter=scene.region?.center??mesh.bounds.center;
  const bounce=order==='before'?mirrorIntersection(p.sourceOrigin,localCenter,mirror):mirrorIntersection(localCenter,p.detectorCenter,mirror);
  if(!bounce)reject('mirror-no-crossing','The central unfolded path does not cross the mirror exactly once.');
  const mirrorTransport={order,mirror,physical,side,reflection:{soft:-1,hard:1},
    quantity:'one selected reflected-family conditional directional power transfer',
    phaseGauge:'carrier-removed in unfolded coordinates; not phase-compatible with another path family',
    admission:{conservative:true,possibleBlockers,frame,sweptBound:swept,epsilon:eps},
    foldedPath:bounce?(order==='before'?[sourceOrigin,bounce.point,localCenter,detectorCenter]:[sourceOrigin,localCenter,bounce.point,detectorCenter]):[],
    centralMirrorPoint:bounce?.point??null};
  scene.mirrorTransport=mirrorTransport;
  scene.unsupported.push(...issues);scene.supported=scene.unsupported.length===0;
  scene.model='one-ideal-mirror-one-local-utd-region';
  scene.warnings=scene.warnings.filter(s=>!s.startsWith('Finite-edge endpoint/corner'));
  scene.warnings.push('One selected family: one infinite ideal scalar mirror and one coherent local GO + UTD interaction. Other mirror/direct families are not added.',
    'The mirror has no finite aperture or diffracting rim. The full Gaussian waist age and unfolded range are preserved; no extra spreading or PDF weight is applied.',
    'The folded-path bounding-box admission test is conservative and can reject clear layouts. Endpoint, corner, two-diffraction and general multi-region transport remain unsupported.',
    'Clear geometric paths do not certify wave accuracy: higher-order and remote diffraction, including nearby omitted penumbrae, are not modeled.');
  return scene;
}

/** Convenience layout: physical folding of the existing unfolded laboratory.
 * Fixed 45-degree infinite mirror, with a user-selected axial separation.
 */
export function prepareMirrorExperiment(input,state={},options={}) {
  const mesh=input?.triangles?input:prepareMesh(input),order=options.order;
  const distanceMm=options.distanceMm??10;
  if(!Number.isFinite(distanceMm)||distanceMm<=0)throw new Error('Mirror separation must be positive and finite.');
  const mirror={point:add(mesh.bounds.center,[0,0,order==='before'?-distanceMm:distanceMm]),normal:normalize([-1,0,1]),order};
  const physical={...state,mirror};
  if(order==='before') {
    physical.sourceOrigin=reflectPoint(state.sourceOrigin??[0,0,-20],mirror);
    physical.incidentDirection=reflectDirection(state.incidentDirection??[0,0,1],mirror);
  } else if(order==='after') {
    physical.detectorCenter=reflectPoint(state.detectorCenter??[0,0,20],mirror);
    physical.detectorU=reflectDirection(state.detectorU??[1,0,0],mirror);
    physical.detectorV=reflectDirection(state.detectorV??[0,1,0],mirror);
  }
  return prepareReflectedUTDScene(mesh,physical);
}

export function evaluateReflectedPoint(scene,physicalDestination,options={}) {
  const t=scene.mirrorTransport;if(!t)throw new Error('Expected a reflected-family scene.');
  const position=vec(physicalDestination,'Physical receiver point'),p=t.physical,q=sub(position,p.detectorCenter);
  const eps=Math.min(t.admission.epsilon/16,64*Number.EPSILON*Math.max(1,...position.map(Math.abs),...p.detectorCenter.map(Math.abs)));
  if(Math.abs(dot(q,cross(p.detectorU,p.detectorV)))>eps||
    Math.abs(dot(q,p.detectorU))>p.detectorWidthMm/2+eps||Math.abs(dot(q,p.detectorV))>p.detectorHeightMm/2+eps)
    throw new Error('Receiver lies outside the prepared reflected-family detector support.');
  const dst=t.order==='after'?reflectPoint(position,t.mirror):position;
  const result=evaluatePoint(scene,dst,options);
  // A unit ideal mirror reverses only the soft scalar component. The common
  // sign leaves power unchanged; retain it for explicitly gauged field tests.
  return {...result,soft:result.soft.map(x=>-x),hard:[...result.hard],
    ...(result.contributions?{contributions:result.contributions.map(c=>({...c,soft:c.soft.map(x=>-x),hard:[...c.hard]}))}:{}),
    phaseGauge:t.phaseGauge,quantity:t.quantity};
}
