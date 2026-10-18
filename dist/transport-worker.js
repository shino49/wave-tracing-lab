import {executeCoherentMeshTask} from './coherent-mesh-runtime.js';
function boundedError(value, depth = 0) {
  if (depth > 4 || ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return undefined;
  if (typeof value === 'string') return value.slice(0, 1000);
  if (typeof value === 'number') return Number.isFinite(value) ? value : String(value);
  if (value === null || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.slice(0, 16).map(item => boundedError(item, depth + 1));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).slice(0, 24).filter(([key]) => !/^(customOBJ|objText|sourceText)$/i.test(key)).map(([key, item]) => [key.slice(0, 80), boundedError(item, depth + 1)]).filter(([, item]) => item !== undefined));
}
/** Only the active generation may publish; explicit owner cancellation is
 * silent, while an internal/unsolicited AbortError remains a visible error. */
export function installTransportWorker(scope, execute = executeCoherentMeshTask) {
  let active = null;
  scope.onmessage = async ({data: request}) => {
    if (request?.type === 'cancel') { if (active && (request.id === undefined || request.id === active.id)) active.controller.abort(); return; }
    if (!request || typeof request !== 'object' || Array.isArray(request) || !Number.isSafeInteger(request.id) || request.id < 1) return;
    active?.controller.abort();
    const token = {id: request.id, controller: new AbortController()}; active = token;
    const current = () => active === token && !token.controller.signal.aborted;
    const post = message => { if (current()) scope.postMessage({...message, id: token.id}); };
    try {
      const result = await execute(request, {signal: token.controller.signal,
        progress: value => post({...(typeof value === 'number' ? {fraction: value} : value), type: 'progress'}),
        preview: value => post({...value, type: 'preview'}), reset: value => post({...value, type: 'reset'})});
      post({type: 'result', result});
    } catch (error) { post({type: 'error', error: String(error?.message ?? error).slice(0, 2048), name: String(error?.name ?? 'Error').slice(0, 128), diagnostics: boundedError(error?.diagnostics), timings: boundedError(error?.timings), backend: boundedError(error?.backend)}); }
    finally { if (active === token) active = null; }
  };
  return () => { active?.controller.abort(); active = null; scope.onmessage = null; };
}
if (typeof self !== 'undefined' && typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) installTransportWorker(self);
