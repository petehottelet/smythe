import assert from 'node:assert/strict';
import {WORLD,createWorld,advance,project,glyphAt,lightAt,resetCamera} from './model.mjs';

const snapshots=[30,60,144].map(fps=>{
  const world=createWorld();for(let frame=0;frame<fps*10;frame++)advance(world,1/fps,{sideways:1,forward:1});
  return {camera:world.camera,time:world.rainTime,light:world.columns.map(column=>lightAt(world,column)),
    glyphs:world.columns.map(column=>glyphAt(world,column,27,192))};
});
for(const sample of snapshots.slice(1)){
  assert.ok(Math.abs(sample.camera.x-snapshots[0].camera.x)<1e-8);
  assert.ok(Math.abs(sample.camera.z-snapshots[0].camera.z)<1e-8);
  assert.ok(Math.abs(sample.time-snapshots[0].time)<1e-8);
  sample.light.forEach((light,index)=>assert.ok(Math.abs(light-snapshots[0].light[index])<1e-8));
  assert.deepEqual(sample.glyphs,snapshots[0].glyphs);
}
const world=createWorld(),column={x:2,z:8};
const near=project(world,column,20,1920,1080),far=project(world,{...column,z:32},20,1920,1080);
advance(world,1/WORLD.sidewaysSpeed,{sideways:1},true);
const movedNear=project(world,column,20,1920,1080),movedFar=project(world,{...column,z:32},20,1920,1080);
assert.equal((near.x-movedNear.x)/(far.x-movedFar.x),4);assert.equal(near.size/far.size,4);
assert.equal(movedNear.depth,near.depth);assert.equal(world.rainTime,0);
const identity=world.columns.map(item=>item.seed);for(let i=0;i<10000;i++)advance(world,.01,{sideways:-1,forward:1},true);
assert.ok(world.camera.x>=-70&&world.camera.x<70&&world.camera.z>=0&&world.camera.z<60);
assert.deepEqual(world.columns.map(item=>item.seed),identity);resetCamera(world);assert.deepEqual(world.camera,{x:0,z:0});
const before=project(world,column,20,1920,1080);advance(world,.1,{forward:1},true);const after=project(world,column,20,1920,1080);
assert.ok(after.depth<before.depth&&after.size>before.size);
for(const value of [-1,NaN,Infinity])assert.throws(()=>advance(world,value),RangeError);
console.log('Passed: 30/60/144 FPS equivalence, 4:1 parallax, scale, pause, identity, wrapping, reset, and invalid elapsed time.');
