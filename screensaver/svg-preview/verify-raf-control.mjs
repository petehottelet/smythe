import assert from 'node:assert/strict';
import {summarizeCallbacks,runControl} from './raf-control.mjs';

const exact=Array.from({length:3601},(_,index)=>index*1000/60);
assert.equal(summarizeCallbacks(exact).durationMs,60000);
assert.equal(summarizeCallbacks(exact).averageCallbacksPerSecond,60);
const slower=Array.from({length:3361},(_,index)=>index*1000/56);
assert.equal(summarizeCallbacks(slower).averageCallbacksPerSecond,56);
assert.throws(()=>summarizeCallbacks(exact.slice(0,-1)),/sixty seconds/);
assert.throws(()=>summarizeCallbacks([0,Infinity]),/Invalid/);
assert.throws(()=>summarizeCallbacks([0,60000,60000]),/increase/);
assert.throws(()=>summarizeCallbacks([1,0,60001]),/increase/);
await assert.rejects(runControl('unused.json',true),/Repetition/);
await assert.rejects(runControl(import.meta.filename,1),/overwrite/);
console.log('RAF control raw-summary and admission checks passed');
