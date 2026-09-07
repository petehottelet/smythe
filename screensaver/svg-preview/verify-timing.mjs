import assert from 'node:assert/strict';
import {createFrameGate} from './timing.mjs';

function sample(refresh,fps,jitter=false){
  const gate=createFrameGate(),draws=[];
  // Sample callback timestamps, not CPU/render work or a benchmark result.
  for(let frame=0;frame<refresh*20;frame++){
    const now=frame*1000/refresh+(jitter?.65*Math.sin(frame*1.731):0);
    if(gate.shouldRender(now,fps))draws.push(now);
  }
  return draws;
}

for(const refresh of [60,120,144]){
  for(const fps of [15,30,45,60]){
    for(const jitter of [false,true]){
      const draws=sample(refresh,fps,jitter);
      assert.ok(Math.abs(draws.length-fps*20)<=1,`${refresh}Hz / ${fps}FPS / jitter ${jitter}: ${draws.length}`);
      assert.ok(draws.every((now,i)=>i===0||now>draws[i-1]));
      // Hardware callback quantization permits varying intervals, but the
      // average cadence must converge on the selected rate, including 45 FPS.
      const achieved=(draws.length-1)*1000/(draws.at(-1)-draws[0]);
      assert.ok(Math.abs(achieved-fps)<.1,`${refresh}Hz / ${fps}FPS: ${achieved}`);
    }
  }
}

const gate=createFrameGate();
assert.equal(gate.shouldRender(0,60),true);
assert.equal(gate.shouldRender(1,60),false);
assert.equal(gate.shouldRender(10000,60),true,'one draw after a ten-second stall');
for(let i=0;i<20;i++)assert.equal(gate.shouldRender(10000,60),false,'no catch-up drawing burst');
assert.equal(gate.shouldRender(10016.7,60),true);
assert.equal(gate.shouldRender(10017,30),true,'rate change starts a new cadence');
assert.equal(gate.shouldRender(10018,30),false);
assert.equal(gate.shouldRender(0,30),true,'clock restart resets its phase');
gate.reset();assert.equal(gate.shouldRender(0,30),true,'explicit lifecycle reset');
for(const [now,fps] of [[NaN,60],[Infinity,60],[0,0],[0,-1],[0,NaN],[0,Infinity]]){
  assert.throws(()=>gate.shouldRender(now,fps),RangeError);
}
console.log('Drawing cadence: 15/30/45/60 FPS at 60/120/144 Hz, jitter, missed slots, rate changes and reset passed.');
