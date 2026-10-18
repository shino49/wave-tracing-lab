/** Bounded CPU fallback for the physical pinhole convolution. No shadow run. */
const now=()=>globalThis.performance?.now?.()??Date.now();
const cancelled=signal=>signal?.reason instanceof Error?signal.reason:Object.assign(new Error('Pinhole render cancelled'),{name:'AbortError'});
function check(signal){if(signal?.aborted)throw cancelled(signal);}
function validSignal(signal){if(signal!=null&&(typeof signal!=='object'||typeof signal.aborted!=='boolean'||typeof signal.addEventListener!=='function'||typeof signal.removeEventListener!=='function'))throw new TypeError('Invalid CPU cancellation signal');}
function powerOfTwo(n){return Number.isInteger(n)&&n>=2&&(n&(n-1))===0;}
function fftLine(data,n,base,stride,inverse){
 for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){const a=2*(base+i*stride),b=2*(base+j*stride),r=data[a],im=data[a+1];data[a]=data[b];data[a+1]=data[b+1];data[b]=r;data[b+1]=im;}}
 for(let len=2;len<=n;len*=2){const angle=(inverse?2:-2)*Math.PI/len,wr=Math.cos(angle),wi=Math.sin(angle);for(let start=0;start<n;start+=len){let ur=1,ui=0;for(let j=0;j<len/2;j++){const a=2*(base+(start+j)*stride),b=2*(base+(start+j+len/2)*stride),r=data[b]*ur-data[b+1]*ui,im=data[b]*ui+data[b+1]*ur,ar=data[a],ai=data[a+1];data[a]=ar+r;data[a+1]=ai+im;data[b]=ar-r;data[b+1]=ai-im;const t=ur*wr-ui*wi;ui=ur*wi+ui*wr;ur=t;}}}
 if(inverse)for(let i=0;i<n;i++){const at=2*(base+i*stride);data[at]/=n;data[at+1]/=n;}
}
export async function pinholeCPUFFT(data,size,{inverse=false,signal}={}){
 if(!powerOfTwo(size)||size>1024||!(data instanceof Float64Array)||data.length!==2*size*size)throw new RangeError('Invalid bounded CPU FFT');
 let yielded=now();for(let axis=0;axis<2;axis++)for(let line=0;line<size;line++){check(signal);fftLine(data,size,axis?line:line*size,axis?size:1,inverse);if(now()-yielded>12){await new Promise(r=>setTimeout(r,0));check(signal);yielded=now();}}
 return data;
}
export function createPinholeCPURunner({width,fftSize=2*width,outputChannels=6,signal}={}){
 if(!powerOfTwo(width)||width<8||width>512||fftSize!==2*width||outputChannels!==6)throw new RangeError('Invalid CPU convolution size');
 const elements=fftSize*fftSize,complexBytes=2*elements*8,accumulators=Array.from({length:outputChannels},()=>new Float64Array(2*elements));
 let destroyed=false,finalized=false,busy=false,convolutions=0,transforms=0,fftMs=0,snapshots=0,snapshotTransforms=0,snapshotMs=0,lastYield=now(),failure=null;
 const ensure=s=>{if(destroyed||finalized)throw new Error('CPU convolution runner is closed');validSignal(signal);validSignal(s);check(signal);check(s);};
 const destroy=error=>{destroyed=true;failure??=error instanceof Error?error:new Error('CPU convolution runner is closed');for(let i=0;i<accumulators.length;i++)accumulators[i]=null;};
 const exclusive=async(fn,s)=>{
  // A rejected concurrent call cannot unlock or mutate the active operation.
  if(busy)throw new Error('CPU convolution methods must be sequential');
  try{ensure(s);}catch(error){destroy(error);throw error;}
  busy=true;try{return await fn();}catch(error){destroy(error);throw error;}finally{busy=false;}
 };
 // Observe both signals (and explicit destruction) even while an FFT yields.
 const operationSignal=s=>({get aborted(){return destroyed||signal?.aborted||s?.aborted;},get reason(){return destroyed?failure:signal?.aborted?signal.reason:s?.reason;}});
 const transform=async(data,inverse,s,snapshot=false)=>{const t=now();await pinholeCPUFFT(data,fftSize,{inverse,signal:operationSignal(s)});ensure(s);if(snapshot)snapshotTransforms++;else{fftMs+=now()-t;transforms++;}};
 const read=async(localSignal,snapshot)=>{
  ensure(localSignal);const started=now(),channels=[],groups=Array.from({length:2},()=>({peak:0,minimum:0,maximumImaginary:0}));let maximumImaginary=0;
  // Exactly one complex scratch allocation per snapshot, reused for each channel.
  // At width 512 this is 16 MiB; the six resident spectra are never cloned together.
  const scratch=snapshot?new Float64Array(2*elements):null;
  for(let c=0;c<outputChannels;c++){
   ensure(localSignal);const data=snapshot?scratch:accumulators[c];if(snapshot)data.set(accumulators[c]);
   await transform(data,true,localSignal,snapshot);ensure(localSignal);
   const channel=new Float64Array(width*width),group=groups[Math.floor(c/3)];
   for(let y=0;y<width;y++)for(let x=0;x<width;x++){
    const at=2*(y*fftSize+x),value=data[at],imaginary=Math.abs(data[at+1]);
    if(!Number.isFinite(value)||!Number.isFinite(imaginary))throw new Error('Nonfinite CPU FFT result');
    channel[y*width+x]=value;group.peak=Math.max(group.peak,value);group.minimum=Math.min(group.minimum,value);group.maximumImaginary=Math.max(group.maximumImaginary,imaginary);maximumImaginary=Math.max(maximumImaginary,imaginary);
   }
   channels.push(channel);
  }
  // Match the existing runtime's wave-XYZ / geometric-XYZ roundoff policy.
  // Leave bounded signed roundoff for that existing clamp/diagnostic step.
  const roundoff=groups.map(group=>({...group,tolerance:Math.max(1e-20,group.peak*2e-10)}));
  for(const group of roundoff){if(group.minimum < -group.tolerance)throw new Error('CPU FFT result contains significant negative light');if(group.maximumImaginary>group.tolerance)throw new Error('CPU FFT result contains significant imaginary residual');}
  ensure(localSignal);if(snapshot){snapshots++;snapshotMs+=now()-started;}else finalized=true;
  return {channels,backend:'CPU f64 FFT fallback',validation:{finite:true,cpuShadow:false,roundoff},stats:{convolutions,transforms,fftMs,maximumImaginary,snapshots,snapshotTransforms,snapshotMs,snapshotScratchComplexBytes:snapshots?complexBytes:0,estimatedSnapshotPeakComplexBytes:snapshots?(outputChannels+1)*complexBytes:0,estimatedPeakComplexBytes:(outputChannels+2)*complexBytes}};
 };
 return {
  async convolveAccumulate({source,kernels,signal:localSignal}={}){return exclusive(async()=>{
   ensure(localSignal);if(!(source instanceof Float32Array||source instanceof Float64Array)||source.length!==width*width||!source.every(v=>Number.isFinite(v)&&v>=0)||!Array.isArray(kernels)||kernels.length>12)throw new Error('Invalid CPU convolution source');
   const spectrum=new Float64Array(2*elements);for(let y=0;y<width;y++)for(let x=0;x<width;x++)spectrum[2*(y*fftSize+x)]=source[y*width+x];await transform(spectrum,false,localSignal);
   for(const kernel of kernels){ensure(localSignal);const {values,channel,weight=1}=kernel;if(!Number.isInteger(channel)||channel<0||channel>=outputChannels||!Number.isFinite(weight)||weight<0||!(values instanceof Float32Array||values instanceof Float64Array)||values.length!==elements||!values.every(v=>Number.isFinite(v)&&v>=0))throw new Error('Invalid CPU convolution kernel');
    const transformed=new Float64Array(2*elements);for(let i=0;i<elements;i++)transformed[2*i]=values[i];await transform(transformed,false,localSignal);
    const sum=accumulators[channel];for(let i=0;i<elements;i++){const at=2*i,a=spectrum[at],b=spectrum[at+1],c=transformed[at],d=transformed[at+1];sum[at]+=(a*c-b*d)*weight;sum[at+1]+=(a*d+b*c)*weight;}
    convolutions++;if(now()-lastYield>12){await new Promise(r=>setTimeout(r,0));ensure(localSignal);lastYield=now();}
   }
  },localSignal);},
  /** Non-final read: snapshots never modify or expose the frequency accumulators. */
  async snapshotResults({signal:localSignal}={}){return exclusive(()=>read(localSignal,true),localSignal);},
  /** Final read retains its existing in-place inverse and closed-runner contract. */
  async readResults({signal:localSignal}={}){return exclusive(()=>read(localSignal,false),localSignal);},
  destroy,
 };
}
