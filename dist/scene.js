/**
 * Display-only projected 3D apparatus. This module never computes irradiance.
 * Dimensions are deliberately exaggerated for legibility (the UI says schematic).
 * The screen texture is supplied by the numeric simulation through getHeatmap().
 */
export function setupScene(canvas, getState, getHeatmap) {
  const ctx = canvas.getContext('2d', { alpha: true });
  let orbit = 0;
  let drag = null;
  let width = 1, height = 1, dpr = 1;
  let disposed = false;
  const accent = '#b9e0b2';
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const sub = (a,b) => a.map((v,i)=>v-b[i]);
  const norm = a => { const n=Math.hypot(...a)||1; return a.map(v=>v/n); };
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const num = (v, fallback) => Number.isFinite(Number(v)) ? Number(v) : fallback;

  function draw() {
    if (disposed || !ctx) return;
    const rect = canvas.getBoundingClientRect();
    width = Math.max(1, rect.width); height = Math.max(1, rect.height);
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pw = Math.round(width*dpr), ph = Math.round(height*dpr);
    if (canvas.width !== pw || canvas.height !== ph) { canvas.width=pw; canvas.height=ph; }
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,width,height);
    const st = getState?.() || {};
    const params = st.params || st;
    const slitWidth = clamp(num(params.slitWidth, .08), .04, .12);
    const separation = clamp(num(params.separation, .3), .18, .42);
    const distance = clamp(num(params.screenDistance, 2.5), 1.5, 4);
    const wavelength = clamp(num(params.lambda, 550), 420, 680);
    const hue = 275-(wavelength-420)/260*268;
    const beam = `hsla(${hue.toFixed(1)},55%,70%,`;
    const screenX = 2.25+(distance-1.5)*.34;
    const apertureX = -.7;
    const sourceX = -3.2;
    const sep = separation/.3*.70;
    const slit = slitWidth/.08*.15;
    const halfY = 1.31, halfZ = .87;
    const angle = (28+orbit*.68)*Math.PI/180;
    const camera = [Math.sin(angle)*10, -Math.cos(angle)*10, 5.0];
    const target = [-.16,0,-.06];
    const forward = norm(sub(target,camera));
    const right = norm(cross(forward,[0,0,1]));
    const up = norm(cross(right,forward));
    const focal = Math.min(width*1.34,height*2.63);
    const center = [width*.5,height*.48];
    function project(p) {
      const v=sub(p,camera), depth=dot(v,forward);
      return {x:center[0]+dot(v,right)/depth*focal,y:center[1]-dot(v,up)/depth*focal,depth};
    }
    function path(points, close=false) {
      ctx.beginPath(); points.forEach((p,i)=>{const q=project(p);if(i)ctx.lineTo(q.x,q.y);else ctx.moveTo(q.x,q.y);});if(close)ctx.closePath();
    }
    function polygon(points, fill, stroke, lineWidth=1) {
      path(points,true); if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lineWidth;ctx.stroke();}
    }
    function line(a,b,color,w=1,dashes=[]) {
      path([a,b]);ctx.strokeStyle=color;ctx.lineWidth=w;ctx.setLineDash(dashes);ctx.stroke();ctx.setLineDash([]);
    }
    const plane = (x,y0,y1,z0=-halfZ,z1=halfZ) => [[x,y0,z1],[x,y1,z1],[x,y1,z0],[x,y0,z0]];
    const queue=[];
    const item=(pts,drawItem)=>queue.push({depth:pts.reduce((a,p)=>a+project(p).depth,0)/pts.length,draw:drawItem});

    // Sparse optical-bench grid. Its world coordinates are schematic only.
    for(let x=-4;x<=4;x+=.5)line([x,-2,-1.22],[x,2,-1.22],'#92b39b0b',.65);
    for(let y=-2;y<=2;y+=.5)line([-4,y,-1.22],[4,y,-1.22],'#92b39b0b',.65);
    line([sourceX-.3,0,-1.2],[screenX+.4,0,-1.2],'#779d821d',.8,[3,4]);
    line([sourceX,0,0],[screenX,0,0],'#a0cda331',.8,[3,5]);

    // Two schematic propagation envelopes. No ray path encodes intensity.
    for(const side of [-1,1]) {
      const y=side*sep/2;
      const start=[[apertureX,y-slit/2,.72],[apertureX,y+slit/2,.72],[apertureX,y+slit/2,-.72],[apertureX,y-slit/2,-.72]];
      const end=[[screenX,-halfY,halfZ],[screenX,halfY,halfZ],[screenX,halfY,-halfZ],[screenX,-halfY,-halfZ]];
      item([...start,...end],()=>{
        polygon([start[0],end[0],end[1],start[1]],beam+'.022)',null);
        polygon([start[3],end[3],end[2],start[2]],beam+'.025)',null);
        for(let i=0;i<4;i++)line(start[i],end[i],beam+'.19)',.8,[3,5]);
      });
      item([[sourceX,0,0],...start],()=>{
        polygon([[sourceX,0,0],start[0],start[1]],beam+'.045)',null);
        polygon([[sourceX,0,0],start[3],start[2]],beam+'.035)',null);
        line([sourceX,0,0],[apertureX,y,.72],beam+'.24)',.7);
        line([sourceX,0,0],[apertureX,y,-.72],beam+'.20)',.7);
      });
    }

    // A slab made from three opaque strips leaves two real, visible apertures.
    const ranges=[[-halfY,-sep/2-slit/2],[-sep/2+slit/2,sep/2-slit/2],[sep/2+slit/2,halfY]];
    for (const [y0,y1] of ranges) {
      const front=plane(apertureX,y0,y1);
      item(front,()=>{
        polygon(plane(apertureX-.085,y0,y1),'#16231e','#587260',.7);
        polygon([[apertureX-.085,y0,halfZ],[apertureX,y0,halfZ],[apertureX,y1,halfZ],[apertureX-.085,y1,halfZ]],'#415247','#729080',.6);
        polygon(front,'#24392b','#789382',.85);
        polygon([[apertureX,y0,halfZ],[apertureX-.085,y0,halfZ],[apertureX-.085,y0,-halfZ],[apertureX,y0,-halfZ]],'#19271e','#63816c',.6);
      });
    }
    for (const side of [-1,1]) {
      const y=side*sep/2;
      line([apertureX,y-slit/2,halfZ],[apertureX,y-slit/2,-halfZ],beam+'.62)',1.2);
    }

    const screen=plane(screenX,-halfY,halfY);
    item(screen,()=>{
      polygon(screen,'#0b1713','#7ba38c',1);
      const supplied=getHeatmap?.();
      const texture=supplied?.canvas || supplied;
      if(texture && texture.width>0 && texture.height>0) {
        ctx.save();
        path(screen,true);ctx.clip();ctx.globalAlpha=.96;
        // Subdivision retains a useful perspective impression with Canvas 2D.
        const cols=14,rows=3;
        for(let r=0;r<rows;r++)for(let c=0;c<cols;c++) {
          const u0=c/cols,u1=(c+1)/cols,v0=r/rows,v1=(r+1)/rows;
          const point=(u,v)=>project([screenX,-halfY+u*halfY*2,halfZ-v*halfZ*2]);
          const a=point(u0,v0),b=point(u1,v0),cc=point(u1,v1),d=point(u0,v1);
          const sx0=u0*texture.width,sx1=u1*texture.width,sy0=v0*texture.height,sy1=v1*texture.height;
          textureTriangle(texture,[[sx0,sy0],[sx1,sy0],[sx1,sy1]],[a,b,cc]);
          textureTriangle(texture,[[sx0,sy0],[sx1,sy1],[sx0,sy1]],[a,cc,d]);
        }
        ctx.restore();
      }
      polygon(screen,null,'#91b89c',1.1);
      const offset=-.07;
      polygon([[screenX,-halfY,halfZ],[screenX+offset,-halfY,halfZ+.06],[screenX+offset,halfY,halfZ+.06],[screenX,halfY,halfZ]],'#344f3e','#6d957d',.6);
      line([screenX,0,-halfZ],[screenX,0,-1.19],'#6c8d75',2);
      line([screenX,-.42,-1.2],[screenX,.42,-1.2],'#6c8d75',2);
    });

    item([[sourceX,0,0]],()=>{
      const p=project([sourceX,0,0]);
      const glow=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,22);
      glow.addColorStop(0,beam+'.25)');glow.addColorStop(1,beam+'0)');
      ctx.fillStyle=glow;ctx.beginPath();ctx.arc(p.x,p.y,22,0,Math.PI*2);ctx.fill();
      line([sourceX,0,-.12],[sourceX,0,-1.18],'#6d8472',1.5);
      line([sourceX,-.26,-1.2],[sourceX,.26,-1.2],'#6d8472',2);
      ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.fillStyle='#dcc492';ctx.fill();
      ctx.beginPath();ctx.arc(p.x,p.y,2,0,Math.PI*2);ctx.fillStyle='#fff9d9';ctx.fill();
    });
    queue.sort((a,b)=>b.depth-a.depth).forEach(q=>q.draw());

    // Apparatus labels are attached to geometry, never to an inferred fringe.
    const label=(p,text,align='center',dx=0,dy=0)=>{
      const q=project(p);ctx.font=`${width<440?9:10}px "PingFang SC",system-ui,sans-serif`;
      ctx.textAlign=align;ctx.textBaseline='top';ctx.fillStyle='#9bad9e';
      ctx.fillText(text,q.x+dx,q.y+dy);
    };
    label([sourceX,0,-1.34],'光源');
    label([apertureX,0,-1.34],'双缝');
    label([screenX,0,-1.34],'接收屏');
    const a=project([apertureX,.08,-1.51]),b=project([screenX,.08,-1.51]);
    if(width>=450) {
      ctx.strokeStyle='#587563';ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
      for(const p of [a,b]){ctx.beginPath();ctx.moveTo(p.x,p.y-3);ctx.lineTo(p.x,p.y+3);ctx.stroke();}
      ctx.textAlign='center';ctx.textBaseline='top';ctx.font='9px "SFMono-Regular",Consolas,monospace';ctx.fillStyle=accent;
      ctx.fillText(`L = ${distance.toFixed(2)} m`,(a.x+b.x)/2,(a.y+b.y)/2+5);
    }

    function textureTriangle(img,s,t) {
      const den=s[0][0]*(s[1][1]-s[2][1])+s[1][0]*(s[2][1]-s[0][1])+s[2][0]*(s[0][1]-s[1][1]);
      if(Math.abs(den)<1e-8)return;
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
      ctx.transform(x[0],y[0],x[1],y[1],x[2],y[2]);ctx.drawImage(img,0,0);ctx.restore();
    }
  }
  function setOrbit(value) { orbit=clamp(num(value,0),-45,45);draw();return orbit; }
  function resetCamera() { setOrbit(0); }
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,orbit};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{
    if(!drag)return;
    setOrbit(drag.orbit+(e.clientX-drag.x)*.23);
    const slider=document.getElementById('orbit');
    const output=document.getElementById('orbitValue');
    if(slider){slider.value=String(Math.round(orbit));slider.dispatchEvent(new Event('input',{bubbles:true}));}
    if(output)output.textContent=`${Math.round(orbit)}°`;
  });
  const end=e=>{drag=null;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);};
  canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);
  const observer=new ResizeObserver(()=>draw());observer.observe(canvas);
  draw();
  return {draw,resetCamera,setOrbit,destroy(){disposed=true;observer.disconnect();}};
}
