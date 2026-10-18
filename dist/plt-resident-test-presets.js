/** Exact public subset of tests/plt-resident-tight-transport.mjs fixtures.
 * Structured mirror panels add a separately frozen dyadic 14-triangle fixture.
 * No user geometry, material, camera, wave or spectral input is accepted.
 */
import {preparePLTScene} from './plt-geometry.js';

export const RESIDENT_HARDWARE_TEST_PRESETS=Object.freeze([
 Object.freeze({id:'two-mirrors-emission',name:'Two mirrors',expectedPower:3,mirrors:2}),
 Object.freeze({id:'four-scale-shifted-zxy',name:'Scaled and shifted two mirrors',expectedPower:3,mirrors:2}),
 Object.freeze({id:'half-scale-three-mirrors-negative-permutation',name:'Three mirrors, negative translation',expectedPower:.75,mirrors:3}),
 Object.freeze({id:'structured-mirror-panels',name:'Structured mirror panels',expectedPower:null,expectedPowers:Object.freeze([0,.75,1.5,3,6]),mirrors:null,maxMirrors:2}),
]);
const specifications={
 'two-mirrors-emission':{mirrors:2,scale:1,shift:[0,0,0],perm:[0,1,2]},
 'four-scale-shifted-zxy':{mirrors:2,scale:4,shift:[-32,16,-8],perm:[2,0,1]},
 'half-scale-three-mirrors-negative-permutation':{mirrors:3,scale:.5,shift:[-16,8,-32],perm:[2,1,0]},
};
function quad(axis,position,lo,hi,materialIndex){
 const a=(axis+1)%3,b=(axis+2)%3,point=(u,v)=>{const p=[0,0,0];p[axis]=position;p[a]=u;p[b]=v;return p;};
 return {mesh:{vertices:[point(lo,lo),point(hi,lo),point(hi,hi),point(lo,hi)],faces:[[0,1,2],[0,2,3]]},materialIndex};
}
function structuredMirrorPanels(){
 const rectangle=(axis,position,u0,u1,v0,v1,materialIndex)=>{
  const a=(axis+1)%3,b=(axis+2)%3,point=(u,v)=>{const p=[0,0,0];p[axis]=position;p[a]=u;p[b]=v;return p;};
  return {mesh:{vertices:[point(u0,v0),point(u1,v0),point(u1,v1),point(u0,v1)],faces:[[0,1,2],[0,2,3]]},materialIndex};
 };
 const materials=[{type:'mirror',albedo:.5,emission:0},{type:'mirror',albedo:.75,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true},{type:'diffuse',albedo:0,emission:2,twoSidedEmission:true},{type:'diffuse',albedo:0,emission:0}];
 const entries=[rectangle(0,2,-8,8,1,2,0),rectangle(2,3,-8,8,-8,8,1)];
 for(const [x0,x1] of [[-6,-4],[6,8]]){entries.push(rectangle(2,-2,x0,x1,0,2,2));entries.push(rectangle(2,-2,x0,x1,2,4,3));}
 entries.push(rectangle(2,1,.9375,1.0625,.21875,.28125,4));
 const camera={position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10};
 const scene=preparePLTScene(entries,{materials});scene.emitters=scene.triangles.filter(t=>scene.materials[t.materialIndex].emission>0);scene.emitterArea=scene.emitters.reduce((sum,t)=>sum+t.area,0);scene.camera=camera;
 return {id:'structured-mirror-panels',scene,camera,expectedPower:null,expectedPowers:[0,.75,1.5,3,6],mirrors:null,maxMirrors:2,oracle:'dyadic-mirror-panels-v1'};
}
export function createResidentHardwareTestFixture(id){
 const metadata=RESIDENT_HARDWARE_TEST_PRESETS.find(p=>p.id===id);
 if(!metadata)throw new RangeError('Unknown resident hardware-test preset');
 if(id==='structured-mirror-panels')return structuredMirrorPanels();
 const {mirrors,scale,shift,perm}=specifications[id];
 const materials=[{type:'mirror',albedo:.5,emission:0},{type:'mirror',albedo:.75,emission:0},{type:'diffuse',albedo:0,emission:8,twoSidedEmission:true},{type:'mirror',albedo:.25,emission:0}];
 const entries=[quad(0,2,-8,8,0),quad(2,3,-8,8,1)];
 if(mirrors===3)entries.push(quad(1,2.5,-8,8,3));
 entries.push(quad(2,-2,-16,16,2));
 let camera=mirrors===3?{position:[0,0,0],target:[4,3,4],up:[0,1,0],fov:5}:{position:[0,0,0],target:[4,1,4],up:[0,1,0],fov:10};
 const point=p=>perm.map((axis,k)=>scale*p[axis]+shift[k]),direction=p=>perm.map(axis=>p[axis]);
 for(const entry of entries)entry.mesh.vertices=entry.mesh.vertices.map(point);
 camera={...camera,position:point(camera.position),target:point(camera.target),up:direction(camera.up)};
 const scene=preparePLTScene(entries,{materials});
 scene.emitters=scene.triangles.filter(t=>scene.materials[t.materialIndex].emission>0);
 scene.emitterArea=scene.emitters.reduce((sum,t)=>sum+t.area,0);scene.camera=camera;
 return {id,scene,camera,expectedPower:metadata.expectedPower,mirrors,scale,shift:[...shift],perm:[...perm]};
}
