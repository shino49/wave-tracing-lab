// Node-only test bridge for the unchanged browser worker module. This checks
// real inter-thread message cloning and termination, not a browser Worker.
import {parentPort} from 'node:worker_threads';
globalThis.self={postMessage:message=>parentPort.postMessage(message)};
await import('../../dist/coherent-worker.js');
parentPort.on('message',data=>self.onmessage({data}));
parentPort.postMessage({type:'ready'});
