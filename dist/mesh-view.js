/**
 * Display-only, perspective Canvas 2D viewer of the actual world-space mesh.
 * No irradiance / UTD calculation occurs here. The receiver texture must be
 * supplied by the numeric solver. Mesh coordinates and winding are unchanged.
 *
 * setupMeshView(canvas, { getState, onCameraChange } = {}) returns:
 *   setScene(scene), setTexture(canvas|null), draw(), resetCamera(),
 *   setCamera({yaw,pitch,zoom}), getCamera(), destroy(). Angles are radians.
 *
 * Scene contract (world coordinates in mm):
 * { mesh: preparedMesh,
 *   sourceOrigin: [x,y,z], incidentDirection: [x,y,z],
 *   incomingCone: {origin,direction,near,far,radius0,tanAngle},
 *     // Gaussian envelope sigma0/slope/tail also accepted for display.
 *   region: {center, axes: [halfExtentX,halfExtentY,halfExtentZ],
 *            basis: {u:[x,y,z],v:[x,y,z],w:[x,y,z]}},
 *     // Also half-axis vectors, {min,max}, {bounds:{min,max}},
 *     // or unit axis vectors + halfSize. frame is accepted instead of basis.
 *   detector: {center, u:[x,y,z], v:[x,y,z], width,height},
 *   mirror: {point,normal,displayCenter,illustrative:true} | null,
 *   foldedPath: [[x,y,z], ...] | null, mirrorOrder: 'before'|'after',
 *     // Physical vertices only. Infinite mirror's finite drawing has no rim.
 *   texture: canvas, label: string }
 * u/v are orthonormal receiver basis vectors; texture top is +v.
 * Source aliases: source: vector or {origin|position}; direction: vector.
 * getState() may supply the same scene; explicit setScene takes precedence.
 * Full two-axis orbit is supported without altering the physical scene.
 */
export function setupMeshView(canvas, options = {}) {
  if(typeof options === 'function') options = { getState: options };
  const ctx = canvas.getContext('2d', {alpha:true});
  if(!ctx) throw new Error('Canvas 2D is unavailable for the geometry viewer.');
  const TAU=Math.PI*2;
  const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const mul=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const unit=(a,fallback=[0,0,1])=>{const n=Math.hypot(...a);return n>1e-12?mul(a,1/n):fallback;};
  const vec=(a)=>Array.isArray(a)&&a.length===3&&a.every(Number.isFinite);
  const finite=(v,fallback)=>Number.isFinite(v)?v:fallback;
  const clamp=(n,a,b)=>Math.min(b,Math.max(a,n));
  const midpoint=(points)=>mul(points.reduce(add,[0,0,0]),1/points.length);
  const initial={yaw:-.92,pitch:.43,zoom:1};
  let camera={...initial}, explicitScene=null, suppliedTexture=null, disposed=false;
  let width=1,height=1,dpr=1,drag=null,pinch=null;
  const pointers=new Map(), listeners=[];
  const basis=(direction)=>{
    const d=unit(direction),u=unit(cross(Math.abs(d[1])<.92?[0,1,0]:[1,0,0],d));
    return {u,v:unit(cross(d,u)),d};
  };
  const boundsFor=(points)=>{
    const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
    for(const p of points)if(vec(p))for(let i=0;i<3;i++){min[i]=Math.min(min[i],p[i]);max[i]=Math.max(max[i],p[i]);}
    if(!Number.isFinite(min[0]))return {min:[-1,-1,-1],max:[1,1,1],center:[0,0,0],radius:1};
    const center=mul(add(min,max),.5),radius=Math.max(Math.hypot(...sub(max,min))*.5,1e-5);
    return {min,max,center,radius};
  };
  function regionCorners(region,mesh) {
    const raw=region?.bounds||region;
    if(vec(raw?.min)&&vec(raw?.max)) {
      const out=[];
      for(let x=0;x<2;x++)for(let y=0;y<2;y++)for(let z=0;z<2;z++)out.push([raw[x?'max':'min'][0],raw[y?'max':'min'][1],raw[z?'max':'min'][2]]);
      return out;
    }
    if(vec(region?.center)&&vec(region.axes)) {
      const frame=region.basis||region.frame||{};
      const halfAxes=[mul(frame.u||[1,0,0],region.axes[0]),mul(frame.v||[0,1,0],region.axes[1]),mul(frame.w||[0,0,1],region.axes[2])];
      return regionCorners({...region,axes:halfAxes,halfSize:undefined,halfExtents:undefined},mesh);
    }
    if(vec(region?.center)&&Array.isArray(region.axes)&&region.axes.length===3&&region.axes.every(vec)) {
      let axes=region.axes;
      const half=region.halfSize||region.halfExtents;
      if(vec(half))axes=axes.map((a,i)=>mul(unit(a),half[i]));
      const out=[];
      for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])out.push(add(region.center,add(add(mul(axes[0],x),mul(axes[1],y)),mul(axes[2],z))));
      return out;
    }
    // No invented interaction region if no physical region is supplied.
    return [];
  }
  function sceneParts(scene) {
    const mesh=scene.mesh;
    const meshBounds=mesh.bounds||boundsFor(mesh.vertices||mesh.triangles.flatMap(t=>[t.a,t.b,t.c]));
    const center=meshBounds.center||mul(add(meshBounds.min,meshBounds.max),.5);
    const radius=meshBounds.radius||1;
    const cone=scene.incomingCone||scene.cone;
    const direction=unit(scene.incidentDirection||scene.direction||cone?.direction||[0,0,1]);
    const source=scene.sourceOrigin||(vec(scene.source)?scene.source:scene.source?.origin||scene.source?.position)||cone?.origin||add(center,mul(direction,-Math.max(10,radius*2)));
    const disk=scene.sourceDisk,sourceDiskPoints=disk?.radiusMm>0&&vec(disk.normal)?Array.from({length:64},(_,i)=>{const b=basis(unit(disk.normal)),t=TAU*i/64;return add(source,add(mul(b.u,disk.radiusMm*Math.cos(t)),mul(b.v,disk.radiusMm*Math.sin(t))));}):[];
    const detector=scene.detector||{};
    const detectorCenter=detector.center||detector.origin||add(center,mul(direction,finite(scene.detectorDistance,20)));
    const frame=basis(direction);
    const u=unit(detector.u||frame.u),v=unit(detector.v||frame.v);
    const dw=Math.max(1e-6,finite(detector.width,finite(detector.span,10)));
    const dh=Math.max(1e-6,finite(detector.height,dw));
    const point=(x,y)=>add(detectorCenter,add(mul(u,x*dw),mul(v,y*dh)));
    const detectorPoints=[point(-.5,.5),point(.5,.5),point(.5,-.5),point(-.5,-.5)];
    const corners=regionCorners(scene.region,mesh);
    const foldedPath=Array.isArray(scene.foldedPath)?scene.foldedPath.filter(vec):null;
    let mirrorData=null;
    if(vec(scene.mirror?.point)&&vec(scene.mirror?.normal)) {
      const normal=unit(scene.mirror.normal),mb=basis(normal);
      const patchCenter=vec(scene.mirror.displayCenter)?scene.mirror.displayCenter:scene.mirror.point;
      // Drawing extent only. Never use these corners in any physical calculation.
      const half=Math.max(2,Math.min(5,radius*.35));
      const patch=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>add(patchCenter,add(mul(mb.u,a*half),mul(mb.v,b*half))));
      mirrorData={center:patchCenter,patch};
    }
    let coneData=null;
    if(cone) {
      const origin=vec(cone.origin)?cone.origin:source,d=unit(cone.direction||direction),cb=basis(d);
      const near=Math.max(0,finite(cone.near,0));
      const far=Math.max(near+1e-8,finite(cone.far,Math.max(near,dot(sub(center,origin),d)+radius)));
      const radiusAt=t=>cone.radius0!==undefined||cone.tanAngle!==undefined?
        Math.max(0,finite(cone.radius0,0)+finite(cone.tanAngle,0)*t):
        Math.max(0,finite(cone.tail,3)*Math.hypot(finite(cone.sigma0,1),finite(cone.slope,0)*t));
      const ring=t=>Array.from({length:32},(_,i)=>add(add(origin,mul(d,t)),add(mul(cb.u,radiusAt(t)*Math.cos(i*TAU/32)),mul(cb.v,radiusAt(t)*Math.sin(i*TAU/32)))));
      coneData={origin,d,near,far,first:ring(near),last:ring(far)};
    }
    let physicalCamera=null;if(scene.measurement==='camera'&&vec(scene.camera?.position)&&vec(scene.camera?.target)){const c=scene.camera,f=unit(sub(c.target,c.position)),r=unit(cross(f,c.up)),u=cross(r,f),h=Math.hypot(...sub(c.target,c.position))*Math.tan(c.fovDeg*Math.PI/360);physicalCamera={position:c.position,target:c.target,corners:[[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>add(c.target,add(mul(r,x*h),mul(u,y*h))))};}
    const all=[...mesh.triangles.flatMap(t=>[t.a,t.b,t.c]),source,...sourceDiskPoints,...detectorPoints,...corners];
    if(physicalCamera)all.push(physicalCamera.position,...physicalCamera.corners);
    if(coneData)all.push(...coneData.first,...coneData.last);
    if(mirrorData)all.push(...mirrorData.patch);
    if(foldedPath)all.push(...foldedPath);
    return {mesh,meshBounds,center,source,sourceDiskPoints,direction,detectorCenter,detectorPoints,point,corners,coneData,foldedPath,mirrorData,physicalCamera,bounds:boundsFor(all)};
  }
  function draw() {
    if(disposed)return;
    const rect=canvas.getBoundingClientRect();
    width=Math.max(1,rect.width||canvas.clientWidth||640);height=Math.max(1,rect.height||canvas.clientHeight||400);
    dpr=Math.min(globalThis.devicePixelRatio||globalThis.window?.devicePixelRatio||1,2);
    const pw=Math.round(width*dpr),ph=Math.round(height*dpr);
    if(canvas.width!==pw||canvas.height!==ph){canvas.width=pw;canvas.height=ph;}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
    const scene=explicitScene||options.getState?.()||{};
    if(!scene.mesh?.triangles?.length) {
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#6f8278';ctx.font='11px system-ui,sans-serif';
      ctx.fillText('等待载入真实三维几何',width/2,height/2);return;
    }
    const p=sceneParts(scene),{mesh,center,source,detectorCenter,detectorPoints,corners,coneData,foldedPath,mirrorData,bounds}=p;
    const viewRadius=Math.max(bounds.radius,1e-4),target=bounds.center;
    // Use a fixed-distance perspective camera and change focal length to zoom.
    // It remains outside the scene even at large zoom, avoiding a near-plane jump.
    const dist=viewRadius*3.6;
    const outward=[Math.sin(camera.yaw)*Math.cos(camera.pitch),Math.sin(camera.pitch),Math.cos(camera.yaw)*Math.cos(camera.pitch)];
    const position=add(target,mul(outward,dist)),forward=mul(outward,-1);
    const right=[Math.cos(camera.yaw),0,-Math.sin(camera.yaw)],up=unit(cross(right,forward));
    const focal=Math.min(width,height)*1.18*camera.zoom;
    const project=point=>{const q=sub(point,position),depth=dot(q,forward);return {x:width*.5+dot(q,right)/depth*focal,y:height*.50-dot(q,up)/depth*focal,depth};};
    const path=(points,close=false)=>{ctx.beginPath();points.forEach((v,i)=>{const q=project(v);i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y);});if(close)ctx.closePath();};
    const line=(a,b,color='#739583',thickness=1,dashes=[])=>{path([a,b]);ctx.strokeStyle=color;ctx.lineWidth=thickness;ctx.setLineDash(dashes);ctx.stroke();ctx.setLineDash([]);};
    const polygon=(points,fill,stroke=null,thickness=1)=>{path(points,true);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=thickness;ctx.stroke();}};
    const queue=[];
    const item=(points,fn,offset=0)=>queue.push({depth:project(midpoint(points)).depth+offset,fn});
    // Metric grid uses scene coordinates. It is a display aid, not an aperture.
    const size=Math.max(...sub(bounds.max,bounds.min)),rawStep=size/9;
    const power=10**Math.floor(Math.log10(Math.max(rawStep,1e-9)));
    const step=[1,2,5,10].map(v=>v*power).find(v=>v>=rawStep)||power*10;
    const floor=bounds.min[1]-viewRadius*.12;
    const xmin=Math.floor(bounds.min[0]/step)*step,xmax=Math.ceil(bounds.max[0]/step)*step;
    const zmin=Math.floor(bounds.min[2]/step)*step,zmax=Math.ceil(bounds.max[2]/step)*step;
    for(let x=xmin;x<=xmax+step*.01;x+=step)line([x,floor,zmin],[x,floor,zmax],'#8aac960e',.65);
    for(let z=zmin;z<=zmax+step*.01;z+=step)line([xmin,floor,z],[xmax,floor,z],'#8aac960e',.65);
    // Incoming central axis and envelope are explicitly schematic.
    // A reflected family never gets a fictitious straight source/receiver axis.
    if(!mirrorData&&!p.physicalCamera)line(source,detectorCenter,'#c7cf9840',.8,[4,5]);
    if(foldedPath)for(let i=1;i<foldedPath.length;i++){
      const a=foldedPath[i-1],b=foldedPath[i];
      item([a,b],()=>line(a,b,scene.supported===false?'#d28c70':'#e0cb8e',1.45,[6,4]));
    }
    if(mirrorData){
      const {patch}=mirrorData;
      item(patch,()=>{
        polygon(patch,'#8daac525');
        // Dashed limits belong to the illustration, never to a physical rim.
        path([...patch,patch[0]]);ctx.strokeStyle='#9bb6cf88';ctx.lineWidth=.8;ctx.setLineDash([2,6]);ctx.stroke();ctx.setLineDash([]);
      });
    }
    if(coneData) {
      const {first,last}=coneData;
      for(let j=0;j<32;j++) {
        const k=(j+1)%32,quad=[first[j],first[k],last[k],last[j]];
        item(quad,()=>polygon(quad,'#bddc9807'));
      }
      item(first,()=>{path([...first,first[0]]);ctx.strokeStyle='#c2d69c45';ctx.lineWidth=.8;ctx.setLineDash([3,5]);ctx.stroke();ctx.setLineDash([]);});
      item(last,()=>{path([...last,last[0]]);ctx.strokeStyle='#c2d69c38';ctx.lineWidth=.8;ctx.setLineDash([3,5]);ctx.stroke();ctx.setLineDash([]);});
      for(const j of [0,8,16,24])item([first[j],last[j]],()=>line(first[j],last[j],'#b9dca533',.8,[3,5]));
    }
    // Real triangle faces, lit using their actual world-space normal.
    // No triangle-edge stroke: coplanar triangulation must not look diffractive.
    const light=unit([-.4,.8,-.55]);
    for(const triangle of mesh.triangles) {
      const points=[triangle.a,triangle.b,triangle.c];
      if(points.some(v=>!vec(v)))continue;
      const n=vec(triangle.normal)?triangle.normal:unit(cross(sub(points[1],points[0]),sub(points[2],points[0])));
      const value=.26+.57*Math.abs(dot(n,light));
      const fill=`rgb(${Math.round(37+46*value)},${Math.round(57+65*value)},${Math.round(44+51*value)})`;
      item(points,()=>polygon(points,fill));
    }
    const physical=mesh.diffractionEdges||(mesh.edges||[]).filter(e=>e.kind==='boundary'||e.kind==='crease');
    for(const edge of physical)if(vec(edge.a)&&vec(edge.b)&&(!edge.kind||edge.kind==='boundary'||edge.kind==='crease')) {
      item([edge.a,edge.b],()=>line(edge.a,edge.b,edge.kind==='crease'?'#c7e7b9':'#aaca9b',edge.kind==='crease'?1.35:1.05),-viewRadius*1e-5);
    }
    // An explicit region is drawn as a wireframe box, never substituted for mesh.
    if(corners.length===8)for(let i=0;i<8;i++)for(const bit of [1,2,4])if((i&bit)===0) {
      const a=corners[i],b=corners[i|bit];item([a,b],()=>line(a,b,'#93b7b056',.7,[3,4]));
    }
    item(detectorPoints,()=>{
      polygon(detectorPoints,'#0d211caf','#76b7a78f',1);
      const image=suppliedTexture||scene.texture;
      if(image?.width>0&&image?.height>0) {
        ctx.save();path(detectorPoints,true);ctx.clip();ctx.globalAlpha=.78;
        const cells=10;
        for(let y=0;y<cells;y++)for(let x=0;x<cells;x++) {
          const x0=x/cells,x1=(x+1)/cells,y0=y/cells,y1=(y+1)/cells;
          const q=[project(p.point(x0-.5,.5-y0)),project(p.point(x1-.5,.5-y0)),project(p.point(x1-.5,.5-y1)),project(p.point(x0-.5,.5-y1))];
          const src=[[x0*image.width,y0*image.height],[x1*image.width,y0*image.height],[x1*image.width,y1*image.height],[x0*image.width,y1*image.height]];
          textureTriangle(image,[src[0],src[1],src[2]],[q[0],q[1],q[2]]);
          textureTriangle(image,[src[0],src[2],src[3]],[q[0],q[2],q[3]]);
        }
        ctx.restore();
      } else {
        for(let j=1;j<4;j++) {
          const a=j/4-.5;line(p.point(a,-.5),p.point(a,.5),'#699b8e26',.65);line(p.point(-.5,a),p.point(.5,a),'#699b8e26',.65);
        }
      }
      polygon(detectorPoints,null,'#93cfbb',1);
    });
    if(p.sourceDiskPoints.length)item(p.sourceDiskPoints,()=>polygon(p.sourceDiskPoints,'#e3ba7530','#f0dca5',1));
    if(p.physicalCamera){const c=p.physicalCamera;for(const corner of c.corners)line(c.position,corner,'#9bcde66a',.8,[3,4]);polygon(c.corners,null,'#9bcde66a',.8);item([c.position],()=>{const q=project(c.position);ctx.fillStyle='#aaddf5';ctx.fillRect(q.x-4,q.y-3,8,6);});}
    item([source],()=>{
      const s=project(source),glow=ctx.createRadialGradient(s.x,s.y,0,s.x,s.y,17);
      glow.addColorStop(0,'#e3ba7560');glow.addColorStop(1,'#e3ba7500');ctx.fillStyle=glow;ctx.beginPath();ctx.arc(s.x,s.y,17,0,TAU);ctx.fill();
      ctx.fillStyle='#f0dca5';ctx.beginPath();ctx.arc(s.x,s.y,3.4,0,TAU);ctx.fill();
    });
    queue.sort((a,b)=>b.depth-a.depth).forEach(({fn})=>fn());
    // Labels refer to geometry, not inferred propagation or a decorative fringe.
    const label=(world,text,dx=0,dy=0,color='#a1b3a4')=>{
      const q=project(world),font=width<400?8:9;
      const x=clamp(q.x+dx,12,width-12),y=clamp(q.y+dy,46,height-57);
      ctx.font=`${font}px "PingFang SC",system-ui,sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';
      const tw=ctx.measureText(text).width;
      ctx.fillStyle='#101a16c9';ctx.fillRect(clamp(x-tw/2-5,2,width-tw-10),y-9,tw+10,18);ctx.fillStyle=color;ctx.fillText(text,clamp(x,tw/2+7,width-tw/2-7),y);
    };
    label(source,p.sourceDiskPoints.length?'独立发射体圆盘':mirrorData?'真实源':'入射波束',0,-21,'#ddc795');
    const top=mesh.bounds?.max||mesh.vertices?.[0]||center;
    label(top,'真实网格',0,-17,'#b9d2b0');
    if(p.physicalCamera)label(p.physicalCamera.position,'几何针孔相机',0,-17,'#aaddf5');
    label(detectorPoints[1],p.physicalCamera?'参考平面 · 非相机接收面':mirrorData?'真实接收面':'接收面',10,-12,'#96c6b7');
    if(mirrorData)label(mirrorData.center,'∞ 理想镜面 · 示意片无孔径',0,23,'#adc3db');
    if(corners.length===8)label(corners[3],'单交互区域',-7,15,'#829e94');
    drawAxes(right,up);
    // Scale bar describes the grid, not the mesh's authored unit setting.
    ctx.textAlign='left';ctx.textBaseline='top';ctx.fillStyle='#577363';ctx.font='8px "SFMono-Regular",Consolas,monospace';
    ctx.fillText(`GRID ${Number(step.toPrecision(3))} mm`,17,43);
  }
  function textureTriangle(image,s,t) {
    const den=s[0][0]*(s[1][1]-s[2][1])+s[1][0]*(s[2][1]-s[0][1])+s[2][0]*(s[0][1]-s[1][1]);
    if(Math.abs(den)<1e-12)return;
    const solve=key=>{
      const v=t.map(p=>p[key]);
      return [
        (v[0]*(s[1][1]-s[2][1])+v[1]*(s[2][1]-s[0][1])+v[2]*(s[0][1]-s[1][1]))/den,
        (v[0]*(s[2][0]-s[1][0])+v[1]*(s[0][0]-s[2][0])+v[2]*(s[1][0]-s[0][0]))/den,
        (v[0]*(s[1][0]*s[2][1]-s[2][0]*s[1][1])+v[1]*(s[2][0]*s[0][1]-s[0][0]*s[2][1])+v[2]*(s[0][0]*s[1][1]-s[1][0]*s[0][1]))/den
      ];
    };
    const x=solve('x'),y=solve('y');
    ctx.save();ctx.beginPath();ctx.moveTo(t[0].x,t[0].y);ctx.lineTo(t[1].x,t[1].y);ctx.lineTo(t[2].x,t[2].y);ctx.closePath();ctx.clip();
    ctx.transform(x[0],y[0],x[1],y[1],x[2],y[2]);ctx.drawImage(image,0,0);ctx.restore();
  }
  function drawAxes(right,up) {
    const ox=width-42,oy=height-62;
    const axes=[{v:[1,0,0],text:'x',color:'#c99483'},{v:[0,1,0],text:'y',color:'#abc59b'},{v:[0,0,1],text:'z',color:'#78afa9'}];
    ctx.font='9px "SFMono-Regular",Consolas,monospace';ctx.textAlign='center';ctx.textBaseline='middle';
    for(const {v,text,color} of axes) {
      const dx=dot(v,right)*20,dy=-dot(v,up)*20;
      ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(ox+dx,oy+dy);ctx.stroke();
      ctx.fillStyle=color;ctx.fillText(text,ox+dx*1.38,oy+dy*1.38);
    }
  }
  function announceCamera() { options.onCameraChange?.({...camera}); }
  function setCamera(next={}) {
    camera={yaw:finite(next.yaw,camera.yaw),pitch:finite(next.pitch,camera.pitch),zoom:clamp(finite(next.zoom,camera.zoom),.2,8)};
    // Angles wrap without clamping either orbit axis.
    camera.yaw=((camera.yaw%TAU)+TAU)%TAU;camera.pitch=((camera.pitch%TAU)+TAU)%TAU;
    draw();announceCamera();return {...camera};
  }
  function resetCamera() {camera={...initial};draw();announceCamera();return {...camera};}
  const listen=(type,fn,opts)=>{canvas.addEventListener(type,fn,opts);listeners.push([type,fn,opts]);};
  const startDrag=()=>{if(pointers.size===1){const point=[...pointers.values()][0];drag={...point,yaw:camera.yaw,pitch:camera.pitch};pinch=null;}else if(pointers.size>=2){const [a,b]=[...pointers.values()];pinch={distance:Math.hypot(a.x-b.x,a.y-b.y),zoom:camera.zoom};drag=null;}};
  listen('pointerdown',event=>{
    if(event.button!==0&&event.pointerType!=='touch')return;
    pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});canvas.setPointerCapture?.(event.pointerId);startDrag();
  });
  listen('pointermove',event=>{
    if(!pointers.has(event.pointerId))return;
    pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
    if(pointers.size>=2&&pinch){const [a,b]=[...pointers.values()];setCamera({zoom:pinch.zoom*Math.hypot(a.x-b.x,a.y-b.y)/Math.max(1,pinch.distance)});}
    else if(drag)setCamera({yaw:drag.yaw-(event.clientX-drag.x)*.008,pitch:drag.pitch+(event.clientY-drag.y)*.008});
  });
  const endPointer=event=>{pointers.delete(event.pointerId);if(canvas.hasPointerCapture?.(event.pointerId))canvas.releasePointerCapture?.(event.pointerId);drag=null;pinch=null;if(pointers.size)startDrag();};
  listen('pointerup',endPointer);listen('pointercancel',endPointer);listen('lostpointercapture',endPointer);
  listen('wheel',event=>{event.preventDefault();setCamera({zoom:camera.zoom*Math.exp(-clamp(event.deltaY,-160,160)*.002)});},{passive:false});
  listen('keydown',event=>{
    const actions={ArrowLeft:()=>setCamera({yaw:camera.yaw+.12}),ArrowRight:()=>setCamera({yaw:camera.yaw-.12}),ArrowUp:()=>setCamera({pitch:camera.pitch-.12}),ArrowDown:()=>setCamera({pitch:camera.pitch+.12}),'+':()=>setCamera({zoom:camera.zoom*1.15}),'=':()=>setCamera({zoom:camera.zoom*1.15}),'-':()=>setCamera({zoom:camera.zoom/1.15}),Home:resetCamera};
    if(actions[event.key]){event.preventDefault();actions[event.key]();}
  });
  const Observer=globalThis.ResizeObserver||globalThis.window?.ResizeObserver;
  const observer=Observer?new Observer(()=>draw()):null;observer?.observe(canvas);
  draw();
  return {
    draw,
    setScene(scene){explicitScene=scene||null;draw();},
    getScene(){return explicitScene;},
    setTexture(texture){suppliedTexture=texture||null;draw();},
    resetCamera,reset:resetCamera,setCamera,getCamera:()=>({...camera}),
    destroy(){disposed=true;observer?.disconnect();listeners.forEach(([type,fn,opts])=>canvas.removeEventListener(type,fn,opts));pointers.clear();},
  };
}
