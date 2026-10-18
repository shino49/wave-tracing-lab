// Experimental geometric validation of native-style successive region queries.
import {prepareMesh,gatherConeTriangles,add,sub,scale,dot,length,normalize,rayTriangleIntersect} from '../dist/mesh-geometry.js';
import {firstConeTriangleContact} from '../dist/utd-scene.js';
import {coefficient,mul,abs2} from '../dist/utd-math.js';
export function inspectFixture(config={}) {
const X={source:[-2,0,0],E1:[0,1,0],E2:[8,2,0],receiver:[16,-2,0],k:64,...config};
const {source,E1,E2,receiver,k}=X;
let obj='',id=1;for(const[x,y]of[E1,E2]){for(const p of [[x,-20,-20],[x,y,-20],[x,y,20],[x,-20,20]])obj+=`v ${p.join(' ')}\n`;obj+=`f ${id} ${id+1} ${id+2}\nf ${id} ${id+2} ${id+3}\n`;id+=4;}
const mesh=prepareMesh(obj),sigma=X.sigma??Math.sqrt(length(sub(E1,source))/(2*k)),di=normalize(sub(add(E1,[0,X.aim1??0,0]),source));
const cone0={origin:source,direction:di,radius0:3*sigma,tanAngle:3/(2*k*sigma),near:0,far:Infinity};
function bounds(cone,near,far){const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];for(const t of[near,far]){const c=add(cone.origin,scale(cone.direction,t)),r=cone.radius0+cone.tanAngle*t;for(let j=0;j<3;j++){const a=r*Math.sqrt(Math.max(0,1-cone.direction[j]**2));min[j]=Math.min(min[j],c[j]-a);max[j]=Math.max(max[j],c[j]+a);}}return{min,max};}
function discover(cone){const contacts=gatherConeTriangles(mesh,cone).map(t=>firstConeTriangleContact(t,cone)).filter(Boolean).sort((a,b)=>a.distance-b.distance);if(!contacts.length)return null;const t=contacts[0].distance,r=cone.radius0+cone.tanAngle*t,near=t,far=t+2*r,frustum={...cone,near,far};const triangles=gatherConeTriangles(mesh,frustum).filter(t=>firstConeTriangleContact(t,frustum));return{cone,center:add(cone.origin,scale(cone.direction,t)),near,far,depth:far-near,radius:r,triangles:triangles.map(t=>t.id),bounds:bounds(cone,near,far)};}
function inFrustum(region,p){const q=sub(p,region.cone.origin),t=dot(q,region.cone.direction),r=region.cone.radius0+region.cone.tanAngle*t;return t>=region.near-1e-9&&t<=region.far+1e-9&&dot(q,q)-t*t<=r*r+1e-8;}
function outgoing(region,d){const c=region.center,di=region.cone.direction,tau=region.cone.tanAngle,co=dot(di,d),si=Math.sqrt(Math.max(0,1-co*co));let r0=0;for(const t of[region.near,region.far]){const s=t-region.near,r=region.cone.radius0+tau*t;r0=Math.max(r0,r+s*si-tau*s*co+tau*r*si);}return{origin:c,direction:d,radius0:r0,tanAngle:tau,near:0,far:Infinity};}
function ballisticTraverse(cone){let dist=0;const steps=[];for(let i=0;i<8;i++){const B=(X.ballisticStart??16)*(2*Math.PI/k)*4**i;const hits=mesh.triangles.map(t=>rayTriangleIntersect(cone.origin,cone.direction,t,{minDistance:dist+1e-8,maxDistance:dist+B*1.001})).filter(h=>h&&h.distance<=dist+B*1.001);if(hits.length)return{blocked:true,steps,hits};dist+=B;const region=discover({...cone,near:dist});steps.push({from:dist-B,to:dist,coneContact:region?.near});if(!region)return{missing:true,steps};if(region.near-dist>=(cone.radius0+cone.tanAngle*dist)/2)return{region,steps};}return{missing:true,steps};}
const t0=ballisticTraverse(cone0);if(!t0.region){return {parameters:X,proof:{passed:false,reason:'initial-traversal',trace:t0}};}
const first=t0.region;const cone1=outgoing(first,normalize(sub(add(E2,[0,X.aim2??0,0]),first.center)));
const t1=ballisticTraverse(cone1);if(!t1.region){return {parameters:X,proof:{passed:false,reason:'second-traversal',trace:t1}};}
const second=t1.region;
const disjoint=first.bounds.max.some((v,i)=>v<second.bounds.min[i]||second.bounds.max[i]<first.bounds.min[i]);
const outside=!inFrustum(second,receiver),only1=first.triangles.every(i=>i<2),only2=second.triangles.every(i=>i>=2);
const angle=(v)=>{const a=Math.atan2(v[0],-v[1]);return a<0?a+2*Math.PI:a;};
const ro1=length(sub(second.center,E1)),ri2=length(sub(E2,first.center)),ro2=length(sub(receiver,E2));
const angles=[angle(sub(source,E1)),angle(sub(second.center,E1)),angle(sub(first.center,E2)),angle(sub(receiver,E2))];
function blocked(a,b){const q=sub(b,a),r=length(q),d=scale(q,1/r);return mesh.triangles.some(t=>{const h=rayTriangleIntersect(a,d,t,{minDistance:1e-8});return h&&h.distance<r-1e-8;});}
const outgoingClear=!blocked(second.center,receiver),firstLocalShadow=blocked(source,second.center),secondLocalShadow=blocked(first.center,receiver);
const powers={};for(const boundary of['soft','hard']){const D1=coefficient(k,2,angles[0],angles[1],ro1,1,boundary),D2=coefficient(k,2,angles[2],angles[3],ro2,1,boundary);powers[boundary]=abs2(mul(D1,D2))/(ro1*ro2);}
return {parameters:X,sigma,cone0,first,cone1,second,ballisticTrace:[t0.steps,t1.steps],proof:{firstSourceSeparated:first.near>0,edge1InFrustum:inFrustum(first,E1),edge2InFrustum:inFrustum(second,E2),only1,only2,aabbDisjoint:disjoint,receiverOutside:outside,outgoingClear,firstLocalShadow,secondLocalShadow,passed:first.near>0&&inFrustum(first,E1)&&inFrustum(second,E2)&&only1&&only2&&disjoint&&outside&&outgoingClear&&firstLocalShadow&&secondLocalShadow},nativeCentreEvaluation:{angles,ro1,ri2,ro2,powers},scope:'Conservative containing cone, retained angular slope, geometric native centre anchoring; deterministic selected-family local-plane UTD. Not a calibrated Gaussian-mode decomposition.'};
}
if (import.meta.url===new URL(process.argv[1], 'file://').href) console.log(JSON.stringify(inspectFixture(process.argv[2]?JSON.parse(process.argv[2]):{}),null,2));
