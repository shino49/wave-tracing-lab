/** Bounded homogeneous parallel dielectric. mm; alpha is POWER extinction/mm.
 * Scalar s-channel only. Source transport includes first transmission, never
 * a Fabry–Perot sum; camera reflection order is a separate ray truncation. */
import {add,sub,scale,dot,cross,length,normalize,distance} from './mesh-geometry.js';
const finite=(x,a,b,name)=>{if(typeof x!=='number'||!Number.isFinite(x)||x<a||x>b)throw new RangeError(`${name} must be finite within ${a}…${b}`);return x;};
const vec=(x,name)=>{if(!Array.isArray(x)||x.length!==3)throw new TypeError(`${name} requires 3 coordinates`);return Array.from(x,v=>finite(v,-10000,10000,name));};
export function validateCoherentSlab(value){
 if(value===undefined||value===null)return null;
 if(typeof value!=='object'||Array.isArray(value)||![Object.prototype,null].includes(Object.getPrototypeOf(value)))throw new TypeError('slab must be one object');
 for(const k of Object.keys(value))if(!['center','u','v','width','height','thickness','index','absorption','internalReflections'].includes(k))throw new Error(`slab unsupported field ${k}`);
 const center=vec(value.center??[0,0,0],'slab.center'),u0=vec(value.u??[1,0,0],'slab.u'),v0=vec(value.v??[0,1,0],'slab.v');if(length(u0)<1e-12||length(v0)<1e-12)throw new Error('slab axes must be nonzero');
 const u=normalize(u0),v=normalize(v0);if(Math.abs(dot(u,v))>1e-10)throw new Error('slab axes must be orthogonal');
 const width=finite(value.width??20,1e-4,1000,'slab.width'),height=finite(value.height??20,1e-4,1000,'slab.height'),thickness=finite(value.thickness??2,1e-6,1000,'slab.thickness'),index=finite(value.index??1.5,1,3,'slab.index'),internalReflections=value.internalReflections??0;
 if(![0,1,2].includes(internalReflections))throw new Error('slab.internalReflections must be 0, 1 or 2');
 const absorption=value.absorption??[{wavelengthNm:360,value:0},{wavelengthNm:830,value:0}];if(!Array.isArray(absorption)||!absorption.length||absorption.length>32)throw new Error('slab absorption requires 1–32 spectral nodes');
 const curve=Array.from(absorption,(p,i)=>{if(!p||typeof p!=='object'||Object.keys(p).some(k=>!['wavelengthNm','value'].includes(k)))throw new Error('Invalid slab absorption node');const wavelengthNm=finite(p.wavelengthNm,360,830,'absorption wavelength'),a=finite(p.value,0,1e6,'absorption power coefficient');if(i&&wavelengthNm<=absorption[i-1].wavelengthNm)throw new Error('Slab absorption wavelengths must increase');return Object.freeze({wavelengthNm,value:a});});
 const n=cross(u,v);for(const a of [-1,1])for(const b of [-1,1])for(const c of [-1,1])vec(center.map((x,i)=>x+a*u[i]*width/2+b*v[i]*height/2+c*n[i]*thickness/2),'slab corner');
 return Object.freeze({center:Object.freeze(center),u:Object.freeze(u),v:Object.freeze(v),width,height,thickness,index,absorption:Object.freeze(curve),internalReflections});
}
export const slabNormal=s=>normalize(cross(s.u,s.v));
export const slabIsInvisible=s=>!s||(s.index===1&&s.absorption.every(p=>p.value===0));
export function absorptionAt(curve,nm){if(!curve?.length)return 0;if(nm<=curve[0].wavelengthNm)return curve[0].value;for(let i=1;i<curve.length;i++)if(nm<=curve[i].wavelengthNm){const a=curve[i-1],b=curve[i],t=(nm-a.wavelengthNm)/(b.wavelengthNm-a.wavelengthNm);return a.value*(1-t)+b.value*t;}return curve.at(-1).value;}
export const slabAbsorptionAt=(s,nm)=>absorptionAt(s?.absorption,nm);
export function slabContainsPoint(s,p){if(!s)return false;const d=sub(p,s.center);return Math.abs(dot(d,s.u))<=s.width/2&&Math.abs(dot(d,s.v))<=s.height/2&&Math.abs(dot(d,slabNormal(s)))<=s.thickness/2;}
/** OBB intersection includes side faces so unsupported lateral routes fail closed. */
export function intersectSlabRay(s,origin,direction,minT=0,maxT=Infinity){
 if(!s)return null;const axes=[s.u,s.v,slabNormal(s)],halves=[s.width/2,s.height/2,s.thickness/2],names=['u','v','normal'],d=sub(origin,s.center);let lo=-Infinity,hi=Infinity,enter,leave;
 for(let i=0;i<3;i++){const o=dot(d,axes[i]),v=dot(direction,axes[i]);if(Math.abs(v)<1e-15){if(Math.abs(o)>halves[i])return null;continue;}let a=(-halves[i]-o)/v,b=(halves[i]-o)/v,na=scale(axes[i],-1),nb=axes[i],fa=names[i]+'-',fb=names[i]+'+';if(a>b){[a,b]=[b,a];[na,nb]=[nb,na];[fa,fb]=[fb,fa];}if(a>lo){lo=a;enter={normal:na,face:fa};}if(b<hi){hi=b;leave={normal:nb,face:fb};}if(lo>hi)return null;}
 const entering=lo>=minT,t=entering?lo:hi,hit=entering?enter:leave;if(!hit||t<minT||t>maxT)return null;return {t,point:add(origin,scale(direction,t)),...hit,entering};
}
export function segmentIntersectsSlab(s,a,b){if(!s||slabIsInvisible(s))return false;const delta=sub(b,a),d=length(delta);if(d<1e-12)return slabContainsPoint(s,a);return !!intersectSlabRay(s,a,scale(delta,1/d),1e-9,d-1e-9);}
export function fresnelS(n1,n2,cosI){cosI=Math.min(1,Math.max(0,cosI));if(n1===n2)return {r:0,t:1,R:0,T:1,cosT:cosI,tir:false};const sin2=(n1/n2)**2*Math.max(0,1-cosI*cosI);if(sin2>=1)return {r:1,t:0,R:1,T:0,cosT:0,tir:true};const cosT=Math.sqrt(1-sin2),den=n1*cosI+n2*cosT,r=(n1*cosI-n2*cosT)/den,t=2*n1*cosI/den;return {r,t,R:r*r,T:Math.max(0,1-r*r),cosT,tir:false};}
export function refractSlabDirection(direction,normal,n1,n2){let N=normal;let c=-dot(direction,N);if(c<0){N=scale(N,-1);c=-c;}const f=fresnelS(n1,n2,c);if(f.tir)return null;return normalize(add(scale(direction,n1/n2),scale(N,(n1/n2)*c-f.cosT)));}
export function assertSlabRouteSupported(s,vertices,kind){if(slabIsInvisible(s))return;for(let i=0;i<vertices.length-1;i++)if(segmentIntersectsSlab(s,vertices[i],vertices[i+1]))throw new Error(`Unsupported ${kind} route intersects dielectric slab; this crossing family is not modeled (only one mirror plus one broad-face slab transmission is supported; disable diffraction and use reflection order 0 or 1)`);}
/** Fermat transmission and ray-tube spreading, not 1/OPL. */
export function solveSlabTransmission(s,a,b){
 if(!s||slabIsInvisible(s))return null;
 if(slabContainsPoint(s,a)||slabContainsPoint(s,b))throw new Error('Source/receiver inside or on dielectric slab is unsupported');
 const N=slabNormal(s),za=dot(sub(a,s.center),N),zb=dot(sub(b,s.center),N),h=s.thickness;if(za*zb>=0||Math.abs(za)<=h/2||Math.abs(zb)<=h/2)return null;
 const sign=za<0?1:-1,normal=scale(N,sign),A1=Math.abs(za)-h/2,A2=Math.abs(zb)-h/2,A=A1+A2,delta=sub(b,a),tangent=sub(delta,scale(normal,A+h)),rho=length(tangent),e=rho?scale(tangent,1/rho):s.u,n=s.index;
 let lo=0,hi=1;for(let i=0;i<80;i++){const p=(lo+hi)/2,c=Math.sqrt(1-p*p),q=Math.sqrt(n*n-p*p);if(A*p/c+h*p/q<rho)lo=p;else hi=p;}const p=(lo+hi)/2,c=Math.sqrt(1-p*p),q=Math.sqrt(n*n-p*p);
 const entry=add(a,add(scale(normal,A1),scale(e,A1*p/c))),exit=add(entry,add(scale(normal,h),scale(e,h*p/q)));
 const within=x=>Math.abs(dot(sub(x,s.center),s.u))<=s.width/2&&Math.abs(dot(sub(x,s.center),s.v))<=s.height/2;if(!within(entry)||!within(exit))return null;
 const glassLength=h*n/q,pathLength=A/c+glassLength,opticalPathLength=A/c+n*glassLength,directDistance=distance(a,b);
 // Stable geometric excess: sum |leg|*(1-cos(angle(leg,direct))).
 // Rationalized 1-cos uses transverse cross product; avoid subtraction of O(mm).
 const D=scale(delta,1/directDistance),vertices=[a,entry,exit,b];let geometricExcess=0;for(let i=0;i<3;i++){const leg=sub(vertices[i+1],vertices[i]),L=length(leg),proj=dot(leg,D),perp=cross(leg,D);geometricExcess+=dot(perp,perp)/(L+proj);}
 const excess=geometricExcess+(n-1)*glassLength,G=1/(c*Math.sqrt((A/c+h/q)*(A/(c*c*c)+h*n*n/(q*q*q)))),f01=fresnelS(1,n,c),f10=fresnelS(n,1,q/n);
 return {vertices,entry,exit,p,cosAir:c,cosGlass:q/n,glassLength,internalLength:glassLength,pathLength,opticalPathLength,directDistance,excess,spreading:G,fieldTransmission:f01.t*f10.t,slab:s};
}
/** Exact convex triangle/box SAT. Touching is excluded with overlap because
 * coincident optical interfaces have no ordering in this bounded model. */
export function slabIntersectsTriangle(s,t){
 if(slabIsInvisible(s))return false;
 const basis=[s.u,s.v,slabNormal(s)],half=[s.width/2,s.height/2,s.thickness/2],p=[t.a,t.b,t.c].map(x=>sub(x,s.center)),edges=[sub(p[1],p[0]),sub(p[2],p[1]),sub(p[0],p[2])],axes=[...basis,cross(edges[0],edges[1]),...edges.flatMap(e=>basis.map(b=>cross(e,b)))];
 for(const axis of axes){if(length(axis)<1e-14)continue;const projections=p.map(x=>dot(x,axis)),radius=half.reduce((r,h,i)=>r+h*Math.abs(dot(basis[i],axis)),0),tol=1e-10*length(axis);if(Math.min(...projections)>radius+tol||Math.max(...projections)<-radius-tol)return false;}return true;
}
export function validateSlabMeshSeparation(s,triangles){if(!slabIsInvisible(s)&&triangles.some(t=>slabIntersectsTriangle(s,t)))throw new Error('Dielectric slab intersects or touches mesh geometry; nested/intersecting media and objects are unsupported');return true;}
