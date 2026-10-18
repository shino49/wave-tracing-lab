import {executePinholeTask} from './pinhole-runtime.js';
function boundedError(value, depth = 0) {
  if (depth > 4 || ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return undefined;
  if (typeof value === 'string') return value.slice(0, 1000);
  if (typeof value === 'number') return Number.isFinite(value) ? value : String(value);
  if (value === null || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.slice(0, 16).map(v => boundedError(v, depth + 1));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).slice(0, 24).map(([k, v]) => [k.slice(0, 80), boundedError(v, depth + 1)]).filter(([, v]) => v !== undefined));
}
/** Only the active requested generation can publish. Actual owner cancellation
 * is silent; an unsolicited/internal AbortError is still a visible failure. */
export function installPinholeWorker(scope, execute = executePinholeTask) {
  let active = null;
  scope.onmessage = async ({data: request}) => {
    if (request?.type === 'cancel') {
      if (active && (request.id === undefined || request.id === active.id)) active.controller.abort();
      return;
    }
    if (!request || typeof request !== 'object' || Array.isArray(request) || !Number.isSafeInteger(request.id) || request.id < 1) return;
    active?.controller.abort();
    const token = {id: request.id, controller: new AbortController()};
    active = token;
    const current = () => active === token && !token.controller.signal.aborted;
    const post = message => { if (current()) scope.postMessage({...message, id: token.id}); };
    try {
      const result = await execute(request, {
        signal: token.controller.signal,
        progress: value => post({...value, type: 'progress'}),
        preview: value => post({...value, type: 'preview'}),
        reset: value => post({...value, type: 'reset'})
      });
      post({type: 'result', result});
    } catch (error) {
      post({type: 'error', error: String(error?.message ?? error).slice(0, 2048), name: String(error?.name ?? 'Error').slice(0, 128), diagnostics: boundedError(error?.diagnostics), timings: boundedError(error?.timings), backend: boundedError(error?.backend)});
    } finally {
      if (active === token) active = null;
    }
  };
  return () => { active?.controller.abort(); active = null; scope.onmessage = null; };
}
if (typeof self !== 'undefined' && typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) installPinholeWorker(self);
