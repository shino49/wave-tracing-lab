/** Immutable Q44 affine camera coefficients. Research-only, no per-sample tape. */
import {cameraFrame} from './plt-core.js';
export const RESIDENT_CAMERA_PARAMETER_BYTES=112;
const Q=44n, LIMIT=1n<<60n;
const gcd=(a,b)=>{a=a<0n?-a:a;while(b){[a,b]=[b,a%b];}return a;};
const rat=(n,d=1n)=>{if(d<0n){n=-n;d=-d;}const g=gcd(n,d);return {n:n/g,d:d/g};};
const add=(a,b)=>rat(a.n*b.d+b.n*a.d,a.d*b.d),neg=a=>rat(-a.n,a.d),mul=(a,b)=>rat(a.n*b.n,a.d*b.d),div=(a,b)=>rat(a.n*b.d,a.d*b.n);
function exact(value){if(!Number.isFinite(value))throw new RangeError('Camera coefficient must be finite');if(value===0)return rat(0n);const data=new DataView(new ArrayBuffer(8));data.setFloat64(0,value);const bits=data.getBigUint64(0),e=Number(bits>>52n&2047n)-1075,m=((bits&((1n<<52n)-1n))|(1n<<52n))*(bits>>63n?-1n:1n);if((bits>>52n&2047n)===0n)throw new RangeError('Subnormal camera constant unsupported');return e>=0?rat(m<<BigInt(e)):rat(m,1n<<BigInt(-e));}
function quantize(a){const sign=a.n<0n?-1n:1n,n=(a.n<0n?-a.n:a.n)<<Q;return sign*((n+a.d/2n)/a.d);}
export function packResidentCamera(camera,options){
 const data=new Uint32Array(RESIDENT_CAMERA_PARAMETER_BYTES/4),width=options.width,height=options.height;
 // Unsupported cameras are explicit invalid immutable state, never clamped.
 if(!Number.isInteger(width)||width<1||width>256||!Number.isInteger(height)||height<1||height>192)return data.buffer;
 const aspect=width/height,half=Math.tan((((camera.fov??42)*Math.PI/180)/2));
 const frame=cameraFrame(camera),vectors=[frame.forward,frame.right,frame.up];
 if(aspect<.25||aspect>4||!(half>0&&half<=2)||vectors.flat().some(x=>!Number.isFinite(x)||Math.abs(x)>2))return data.buffer;
 let zeroMask=0;const coefficients=[];
 try{
  const A=exact(aspect),H=exact(half),AH=mul(A,H),W=rat(BigInt(width)),T=rat(BigInt(height)),two=rat(2n);
  for(let i=0;i<3;i++){
   const F=exact(frame.forward[i]),R=exact(frame.right[i]),U=exact(frame.up[i]);
   const c=add(add(F,neg(mul(R,AH))),mul(U,H));
   const x=div(mul(mul(two,R),AH),W),y=neg(div(mul(mul(two,U),H),T));
   const ints=[c,x,y].map(quantize),abs=x=>x<0n?-x:x;
   if(abs(ints[0])+BigInt(width)*abs(ints[1])+BigInt(height)*abs(ints[2])>=LIMIT)return data.buffer;
   coefficients.push(...ints);
   if(F.n===0n&&R.n===0n&&U.n===0n)zeroMask|=1<<i;
  }
 }catch{return data.buffer;}
 coefficients.forEach((value,i)=>{const n=BigInt.asUintN(64,value);data[i*2]=Number(n&0xffffffffn);data[i*2+1]=Number(n>>32n);});
 data.set([width,height,1,zeroMask],24);return data.buffer;
}
