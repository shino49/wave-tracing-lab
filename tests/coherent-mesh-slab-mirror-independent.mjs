import assert from 'node:assert/strict';
import {prepareCoherentMeshScene,compileCoherentSource,evaluateGeometricRoutes,reflectedSlabRoutes} from '../dist/coherent-mesh-core.js';
const add=(a,b)=>a.map((v,i)=>v+b[i]), sub=(a,b)=>a.map((v,i)=>v-b[i]), mul=(a,s)=>a.map(v=>v*s), dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0), cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]], norm=a=>Math.hypot(...a), unit=a=>mul(a,1/norm(a));
const near=(a,b,t=2e-9)=>assert.ok(Math.abs(a-b)<t*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const quad=(vertices,materialIndex=0)=>({mesh:{vertices,faces:[[0,1,2],[0,2,3]]},materialIndex,surface:'open-plate'});
const materials=[{type:'mirror',reflectance:.81,phaseRad:.37},{type:'opaque'}];
const mirror=quad([[-10,0,-15],[10,0,-15],[10,0,15],[-10,0,15]]);
const slab={center:[0,2.1,5],u:[1,0,0],v:[0,1,0],width:10,height:4,thickness:2,index:1.5,absorption:[{wavelengthNm:360,value:.2},{wavelengthNm:830,value:.2}]};
const source=[-1,1,-10],receiver=[2,1,10];
const make=(entries=[mirror],s=slab)=>prepareCoherentMeshScene(entries,materials,{maxReflections:1,slab:s});
const routes=(scene,a=source,b=receiver)=>evaluateGeometricRoutes(scene,compileCoherentSource(scene,{position:a,wavelengthMm:.00055,phaseRad:.2}),b).routes.filter(r=>r.sequence.length&&r.kind==='transmission');
const scene=make(),r=routes(scene)[0];assert.ok(r);assert.equal(r.interactionOrder,'R-TT');
// Independently differentiate the physical optical path on each constrained surface.
const opl=vertices=>vertices.slice(1).reduce((s,p,i)=>s+norm(sub(p,vertices[i]))*(i===2?slab.index:1),0);
near(opl(r.vertices),r.opticalPathLength);near(r.excess,r.opticalPathLength-norm(sub(receiver,source)));near(r.pathLength,r.vertices.slice(1).reduce((s,p,i)=>s+norm(sub(p,r.vertices[i])),0));
for(const [j,tangents] of [[1,[[1,0,0],[0,0,1]]],[2,[slab.u,slab.v]],[3,[slab.u,slab.v]]])for(const t of tangents){const h=1e-4,p=r.vertices.map(v=>[...v]),m=r.vertices.map(v=>[...v]);p[j]=add(p[j],mul(t,h));m[j]=add(m[j],mul(t,-h));near((opl(p)-opl(m))/(2*h),0,2e-8);assert.ok(opl(p)>=opl(r.vertices)-1e-10);}
const vacuumBounce=[.5,0,0];assert.ok(norm(sub(r.vertices[1],vacuumBounce))>1e-4,'refraction must move the mirror saddle');
const reversed=routes(scene,receiver,source)[0];assert.equal(reversed.interactionOrder,'TT-R');near(reversed.amplitude,r.amplitude);near(reversed.excess,r.excess);r.vertices.forEach((p,i)=>p.forEach((x,k)=>near(x,reversed.vertices.at(-1-i)[k])));
// Direct numerical ray-tube area Jacobian: perturb physical launch angles,
// reflect and refract by vector Snell, intersect a screen normal to arrival.
const hit=(o,d,p,n)=>add(o,mul(d,dot(sub(p,o),n)/dot(d,n)));
const snell=(d,n,n1,n2)=>{if(dot(d,n)<0)n=mul(n,-1);const c=dot(d,n),eta=n1/n2;return add(mul(sub(d,mul(n,c)),eta),mul(n,Math.sqrt(1-eta*eta*(1-c*c))));};
const d0=unit(sub(r.vertices[1],source)),basis1=unit(cross(d0,[0,1,0])),basis2=cross(d0,basis1),arrival=unit(sub(receiver,r.vertices[3]));
const trace=(x,y)=>{let d=unit(add(d0,add(mul(basis1,x),mul(basis2,y)))),p=hit(source,d,[0,0,0],[0,1,0]);d=sub(d,mul([0,1,0],2*d[1]));p=hit(p,d,r.entry,[0,0,1]);d=snell(d,[0,0,1],1,slab.index);p=hit(p,d,r.exit,[0,0,1]);d=snell(d,[0,0,1],slab.index,1);return hit(p,d,receiver,arrival);};
const h=1e-6,dx=mul(sub(trace(h,0),trace(-h,0)),1/(2*h)),dy=mul(sub(trace(0,h),trace(0,-h)),1/(2*h)),numericG=1/Math.sqrt(norm(cross(dx,dy)));near(numericG,r.spreading,2e-8);
near(r.amplitude,r.unabsorbedAmplitude*Math.exp(-.1*r.glassLength));
// Rigidly rotate the entire problem out of coordinate axes.
const rot=p=>{const a=.63,b=-.41;const q=[Math.cos(a)*p[0]-Math.sin(a)*p[1],Math.sin(a)*p[0]+Math.cos(a)*p[1],p[2]];return [q[0],Math.cos(b)*q[1]-Math.sin(b)*q[2],Math.sin(b)*q[1]+Math.cos(b)*q[2]];};
const rotated=make([quad(mirror.mesh.vertices.map(rot))],{...slab,center:rot(slab.center),u:rot(slab.u),v:rot(slab.v)}),rr=routes(rotated,rot(source),rot(receiver))[0];near(rr.amplitude,r.amplitude);near(rr.excess,r.excess);r.vertices.map(rot).forEach((p,i)=>p.forEach((x,k)=>near(x,rr.vertices[i][k])));
// A finite mirror missing the saddle cannot create its infinite-plane image.
assert.equal(routes(make([quad([[-10,0,-15],[10,0,-15],[10,0,-5],[-10,0,-5]])])).length,0);
// Block the folded physical source leg; the unfolded virtual leg is clear.
const mid=mul(add(source,r.vertices[1]),.5),blocker=quad([add(mid,[-.1,-.1,0]),add(mid,[.1,-.1,0]),add(mid,[.1,.1,0]),add(mid,[-.1,.1,0])],1);
assert.equal(routes(make([mirror,blocker])).length,0);
assert.equal(evaluateGeometricRoutes(scene,compileCoherentSource(scene,{position:source}),receiver,{blockReflected:true}).routes.some(x=>x.sequence.length),false);
// Tilted slab can carry the ray across the INFINITE mirror plane away from
// the finite physical patch. Only adjacent bounce vertices need same side.
const N=[-.8,0,.6],d=unit([.1,0,1]),entry=mul(d,1),inside=snell(d,N,1,1.5),exit=add(entry,mul(inside,2/dot(inside,N))),a=[d[0],0,-d[2]],b=add(exit,mul(d,.1));assert.ok(a[0]*b[0]<0);
const tilted={center:add(entry,N),u:[.6,0,.8],v:[0,1,0],width:4,height:4,thickness:2,index:1.5};
const tiltedMirror=quad([[0,-.2,-.2],[0,.2,-.2],[0,.2,.2],[0,-.2,.2]]),ts=make([tiltedMirror],tilted),tr=reflectedSlabRoutes(ts,compileCoherentSource(ts,{position:a}),b,ts.mirrorPlanes[0]);
assert.equal(tr.length,1,'valid tilted slab route with opposite-side endpoints');near(norm(tr[0].vertices[1]),0);near(norm(sub(tr[0].entry,entry)),0);near(norm(sub(tr[0].exit,exit)),0);
console.log('PASS independent slab+mirror physical Fermat derivatives, shifted saddle, reciprocal vertices, numerical ray-tube Jacobian, rigid rotation, finite mirror, physical blocker, reflection ablation, and tilted-slab opposite-side endpoints');
