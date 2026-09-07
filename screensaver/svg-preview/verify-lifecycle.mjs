// Execute the real host module with deterministic browser/REGL boundaries.
// This tests scheduling and measurement lifecycle, not graphics performance.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {createFrameGate} from './timing.mjs';

const source=readFileSync(new URL('./rain.js',import.meta.url),'utf8');
// Imports are replaced by the explicit boundaries below; the host body runs unchanged.
const host=source.replace(/^import .+;\r?$/gm,'');
assert.ok(!/^import /m.test(host),'Every host dependency must have an explicit test boundary');

async function preview(initialPaused=false){
  let now=0,nextId=0;
  const callbacks=new Map(),timers=new Map(),elements=new Map();
  const element=()=>({textContent:'',disabled:false,dataset:{},
    addEventListener(){},setAttribute(){},closest(){return null;},
    classList:{add(){},remove(){}}});
  const get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
  const document={hidden:false,body:element(),documentElement:element(),
    getElementById:get,querySelectorAll:()=>[],
    querySelector:selector=>selector===':focus-visible'?null:get(selector),addEventListener(){}};
  const config={preset:'classic',fps:60,originalMix:10,resolution:.75};
  const regl=()=>argument=>{if(typeof argument==='function')argument();};
  Object.assign(regl,{_gl:{FRAGMENT_SHADER:1,HIGH_FLOAT:2,getShaderPrecisionFormat:()=>({precision:23})},
    limits:{maxTextureSize:8192,maxRenderbufferSize:8192},poll(){},destroy(){},read:()=>new Uint8Array(4)});
  const sandbox={document,location:{href:'http://localhost/preview/'},navigator:{userAgent:'deterministic-test'},console,URL,Uint8Array,
    innerWidth:1920,innerHeight:1080,devicePixelRatio:1,
    performance:{now:()=>now},matchMedia:()=>({matches:initialPaused,addEventListener(){}}),
    addEventListener(){},requestAnimationFrame:callback=>{const id=++nextId;callbacks.set(id,callback);return id;},
    cancelAnimationFrame:id=>callbacks.delete(id),
    setTimeout:(callback,delay)=>{const id=++nextId;timers.set(id,{callback,delay});return id;},
    clearTimeout:id=>timers.delete(id),
    mountSettings:()=>({isOpen:()=>false,open(){}}),serializeSettingsUrl:()=>'',
    SCHEMA:[],PRESETS:[],presetValues:()=>({...config}),readConfig:()=>({...config}),engineConfig:value=>value,
    makeRain(){},makeBloom(){},makePalette(){},
    makeSimulationScope:()=>callback=>callback(),
    makePipeline:()=>[{outputs:{primary:{}},ready:Promise.resolve(),setSize(){},execute(){}}],
    createFrameGate,createREGL:()=>regl,SVG_GLYPHS:{catalog_sha256:'test-catalog'}};
  await runInNewContext(`(async()=>{${host}\n})()`,sandbox,{filename:'rain.js',timeout:1000});
  assert.equal(sandbox.GlyphRainPreview?.version,'2','The production host must initialize');
  return {api:sandbox.GlyphRainPreview,pending:()=>callbacks.size,
    tick(milliseconds=17){
      now+=milliseconds;
      const scheduled=[...callbacks.values()];callbacks.clear();
      for(const callback of scheduled)callback(now);
      return scheduled.length;
    },
    timeout(){
      const entry=[...timers].find(([,timer])=>timer.delay>=10000);
      assert.ok(entry,'A benchmark deadline must exist');
      timers.delete(entry[0]);entry[1].callback();
    }};
}

for(const initialPaused of [false,true]){
  const page=await preview(initialPaused);
  assert.equal(page.pending(),initialPaused?0:1);
  // Reusing a page must not accumulate invisible callback chains after samples.
  for(let repetition=0;repetition<3;repetition++){
    const result=page.api.runBenchmark({warmupSeconds:0,durationSeconds:.03});
    for(let ticks=0;page.api.benchmarkRunning()&&ticks<10;ticks++){
      assert.equal(page.tick(),1,'One callback must own each animation step');
    }
    assert.equal(page.api.benchmarkRunning(),false,'The bounded sample must finish');
    const receipt=await result;
    assert.equal(receipt.status,'completed');
    assert.ok(receipt.sampleBoundary.last.tick>receipt.sampleBoundary.first.tick);
    assert.ok(receipt.sampleBoundary.last.time>receipt.sampleBoundary.first.time);
    assert.ok(receipt.measuredDurationSeconds>=.03,'Stop is measured from the first sampled callback');
    assert.equal(receipt.samples.frameIntervalMs.reduce((a,b)=>a+b,0)/1000,receipt.measuredDurationSeconds);
    assert.equal(page.api.stats().paused,initialPaused);
    assert.equal(page.pending(),initialPaused?0:1,'Completion must restore exactly one loop, or remain paused');
    if(!initialPaused){
      for(let ticks=0;ticks<10;ticks++){
        assert.equal(page.tick(),1,'Completed measurements must not leave duplicate callbacks');
        assert.equal(page.pending(),1);
      }
    }
  }
  const timeout=page.api.runBenchmark({warmupSeconds:0,durationSeconds:.03});
  page.timeout();
  assert.equal((await timeout).invalidReason,'Measurement timed out');
  assert.equal(page.pending(),initialPaused?0:1,'An external timeout must preserve the same loop invariant');
  page.api.pause(true);assert.equal(page.pending(),0);
  page.api.pause(false);page.api.pause(false);assert.equal(page.pending(),1,'Repeated resume must stay idempotent');
}
console.log('Passed: benchmark completion/timeout, prior pause restoration, repeated in-page sampling, and one RAF chain.');
