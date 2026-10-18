/** Scheduling only: every nonzero source/depth/material contribution appears
 * exactly once. Source support is not detector-energy coverage or an error bound. */
export async function planPinholeContributions(tasks,{signal}={}) {
 const check=()=>{if(signal?.aborted)throw signal.reason??Object.assign(new Error('Contribution scheduling cancelled'),{name:'AbortError'});};check();
 if(!Array.isArray(tasks))throw new TypeError('Expected bounded contribution tasks');
 if(!tasks.length)return {tasks:[],checkpoints:[],activeSourcePairs:0,totalSourceY:0};
 const size=tasks[0].values.length,bases=1+Math.max(...tasks.map(t=>t.basis));
 if(tasks.length>64||size>512*512||bases>4)throw new RangeError('Contribution schedule capacity exceeded');
 const active=new Uint8Array(size*bases),seen=new Uint8Array(size*bases);let activeCount=0,totalSourceY=0;
 for(const t of tasks){if(t.values.length!==size||!Number.isInteger(t.basis)||t.basis<0||!Number.isFinite(t.sourceY)||t.sourceY<0)throw new Error('Invalid contribution task');totalSourceY+=t.sourceY;for(let i=0;i<size;i++)if(t.values[i]>0&&!active[t.basis*size+i]){active[t.basis*size+i]=1;activeCount++;}}
 const remaining=tasks.map((task,originalIndex)=>({...task,originalIndex})),ordered=[];let represented=0,cumulativeY=0,coverCount=0;
 while(remaining.length){check();let best=0,bestNew=-1;for(let j=0;j<remaining.length;j++){const t=remaining[j];let fresh=0;for(let i=0;i<size;i++)if(t.values[i]>0&&!seen[t.basis*size+i])fresh++;if(fresh>bestNew||(fresh===bestNew&&t.sourceY>remaining[best].sourceY)){best=j;bestNew=fresh;}}
  const t=remaining.splice(best,1)[0];for(let i=0;i<size;i++)if(t.values[i]>0&&!seen[t.basis*size+i]){seen[t.basis*size+i]=1;represented++;}cumulativeY+=t.sourceY;
  ordered.push({...t,representedSourceFraction:activeCount?represented/activeCount:1,sourcePowerFraction:totalSourceY?Math.min(1,cumulativeY/totalSourceY):1});
  if(!coverCount&&represented===activeCount)coverCount=ordered.length;
  await new Promise(r=>setTimeout(r,0));check();
 }
 const checkpoints=[coverCount,coverCount+Math.ceil((ordered.length-coverCount)/2)].filter((x,i,a)=>x>0&&x<ordered.length&&a.indexOf(x)===i);
 return {tasks:ordered,checkpoints,activeSourcePairs:activeCount,totalSourceY,firstFullSourceSupport:coverCount};
}
