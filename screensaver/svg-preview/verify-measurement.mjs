// Deterministic protocol/CLI tests. No browser or performance sample is started.
import assert from 'node:assert/strict';
import {mkdtempSync,readFileSync,writeFileSync,existsSync,unlinkSync,rmdirSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {dirname,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {PROTOCOL,SCHEDULE,HARNESS_FILES,expectedConfig,expectedEngineConfig,summarize,validateStartup,validateReceipt,aggregateReceipts,classifyBackend} from './measurement.mjs';
import {measureRun} from './measure.mjs';
import {hashFiles} from './provenance.mjs';
import {rendererSources} from './source-files.mjs';

const directory=dirname(fileURLToPath(import.meta.url)),hash='a'.repeat(64);
const sourceNames=[...new Set([...rendererSources(directory),'reference/catalog.json','reference/provenance.json','reference/LICENSE'])].sort();
const sources=hashFiles(directory,sourceNames);
const browser={version:{product:'Chrome/test',revision:'fixed-revision'},commandLine:{arguments:['chrome','--headless']},headless:true,
  gl:{renderer:'WebKit WebGL',version:'WebGL 1',unmaskedRenderer:'ANGLE NVIDIA GeForce',extensions:[]},
  systemInfo:{gpu:{devices:[{deviceString:'NVIDIA GeForce'}]}}};
function fixture(preset='classic',repetition=1){
  const startup={version:'2',paused:true,camera:{x:0,z:0},simulationTick:0,rainTime:0,
    config:expectedConfig(preset),catalogCount:192,referenceGlyphCount:56,referenceSequenceLength:57,totalVisibleShapes:248,glyphMix:.1,
    viewport:{width:1920,height:1080,dpr:1,renderScale:.75,physicalWidth:1440,physicalHeight:810}};
  const intervals=Array(3600).fill(16.6667),cpu=Array(3601).fill(.2),elapsed=intervals.reduce((a,b)=>a+b,0)/1000;
  const position=SCHEDULE.findIndex(([p,r])=>p===preset&&r===repetition)+1,start=Date.parse('2026-09-07T00:00:00Z')+position*70000;
  return {protocol:PROTOCOL,status:'completed',warmupSeconds:5,requestedDurationSeconds:60,measuredDurationSeconds:elapsed,
    attemptStartedAt:new Date(start).toISOString(),attemptFinishedAt:new Date(start+65001).toISOString(),
    scenario:{preset,repetition,position},startup,sceneUrl:'http://localhost/preview/',engineConfig:expectedEngineConfig(preset,'http://localhost/preview/'),
    initialStats:{...structuredClone(startup),paused:false},finalStats:{...structuredClone(startup),paused:false,rainTime:65.1,simulationTick:3906},
    sampleBoundary:{first:{tick:300,time:5},last:{tick:3906,time:65.1}},
    samples:{frameIntervalMs:intervals,cpuSubmissionMs:cpu},frameIntervalMs:summarize(intervals),cpuSubmissionMs:summarize(cpu),
    averageFramesPerSecond:intervals.length/elapsed,catalogCount:192,referenceGlyphCount:56,catalogSha256:hash,referenceCatalogSha256:sources['reference/catalog.json'],
    sourceSha256:{...sources},servedSourceSha256:{...sources},harnessSha256:Object.fromEntries(HARNESS_FILES.map(name=>[name,hash])),
    sourceStable:true,harnessStable:true,protocolStable:true,protocolSha256:hash,browser:structuredClone(browser),browserErrors:[],host:{cpu:'fixture',powerProfile:'fixture',competingWorkloads:'fixture'}};
}
assert.equal(validateReceipt(fixture()).valid,true);
assert.equal(validateReceipt(fixture()).targetPassed,true);
const damage=[
  r=>{r.measuredDurationSeconds=59.98;},r=>{r.samples.frameIntervalMs[0]=-1;},
  r=>{r.samples.cpuSubmissionMs[0]=NaN;},r=>{r.samples.cpuSubmissionMs.pop();},
  r=>{r.frameIntervalMs.p95=1;},r=>{r.averageFramesPerSecond=1000;},
  r=>{r.measuredDurationSeconds+=1;},r=>{r.sourceStable=false;},r=>{r.protocolStable=false;},
  r=>{r.servedSourceSha256['rain.js']=hash;},r=>{delete r.sourceSha256['base-glyphs.js'];delete r.servedSourceSha256['base-glyphs.js'];},
  r=>{r.startup.simulationTick=1;},r=>{r.startup.rainTime=.01;},r=>{r.startup.paused=false;},
  r=>{r.finalStats.config.originalMix=0;},r=>{r.finalStats.viewport.physicalWidth=1920;},
  r=>{r.browser.headless=false;},r=>{r.browserErrors.push('context failure');},
  r=>{r.sampleBoundary.first=null;},r=>{r.scenario.position=0;},r=>{delete r.catalogSha256;}
];
for(const key of ['warmupSeconds','requestedDurationSeconds','measuredDurationSeconds'])damage.push(r=>{r[key]=Infinity;});
damage.push(r=>{r.sampleBoundary.last.time=Infinity;},r=>{r.sampleBoundary.first.time=NaN;});
damage.push(r=>{r.requestedDurationSeconds=61;},r=>{r.warmupSeconds=6;},r=>{r.referenceCatalogSha256=hash;},
  r=>{delete r.harnessSha256['measure.mjs'];},r=>{r.host.powerProfile='not recorded';},r=>{r.host.competingWorkloads='';},r=>{r.engineConfig.fallSpeed=99;});
for(const mutate of damage){const receipt=fixture();mutate(receipt);assert.equal(validateReceipt(receipt).valid,false,mutate.toString());}
assert.throws(()=>validateStartup({...fixture().startup,config:{...expectedConfig('classic'),fps:30}},'classic'),/configuration/);
assert.equal(classifyBackend(browser),'hardware-reported');
assert.equal(classifyBackend({...browser,gl:{...browser.gl,unmaskedRenderer:'ANGLE SwiftShader'}}),'software');
assert.equal(classifyBackend({gl:{renderer:'WebKit'}}),'unknown');
assert.equal(classifyBackend({...browser,gl:{...browser.gl,unmaskedRenderer:'WebKit WebGL'}}),'unknown');
assert.equal(classifyBackend({...browser,gl:{...browser.gl,unmaskedRenderer:'ANGLE Intel'}}),'unknown','Unmasked vendor must match physical-device evidence');
const observed=structuredClone(browser);
observed.gl.unmaskedRenderer='ANGLE (NVIDIA, NVIDIA GeForce RTX 4090 (0x00002684) Direct3D11 vs_5_0 ps_5_0, D3D11)';
observed.systemInfo.gpu.devices=[{vendorId:4318,deviceString:'NVIDIA GeForce RTX 4090'},
  {vendorId:5140,deviceString:'Microsoft Basic Render Driver'}];
observed.systemInfo.gpu.auxAttributes={glRenderer:observed.gl.unmaskedRenderer};
assert.equal(classifyBackend(observed),'hardware-reported','An unused fallback adapter must not override the actual RTX4090 context');
observed.gl.unmaskedRenderer='WebKit WebGL';assert.equal(classifyBackend(observed),'unknown');
observed.gl.unmaskedRenderer='ANGLE SwiftShader';assert.equal(classifyBackend(observed),'software');
observed.gl.unmaskedRenderer='ANGLE NVIDIA';observed.systemInfo.gpu.auxAttributes.glRenderer='ANGLE Intel';
assert.equal(classifyBackend(observed),'unknown','Conflicting active-context metadata must not qualify hardware');
const campaign=SCHEDULE.map(([preset,repetition])=>fixture(preset,repetition));
assert.equal(aggregateReceipts(campaign).targetPassed,true);
const ephemeral=structuredClone(campaign);
for(const [index,receipt] of ephemeral.entries())receipt.browser.commandLine.arguments.push(`--user-data-dir=profile${index}`,`--remote-debugging-port=${10000+index}`,`--crashpad-handler-pid=${100+index}`);
assert.equal(aggregateReceipts(ephemeral).targetPassed,true);
ephemeral[1].browser.commandLine.arguments.push('--disable-gpu');assert.throws(()=>aggregateReceipts(ephemeral),/launch configuration/);
assert.throws(()=>aggregateReceipts(campaign.slice(1)),/six/);
assert.throws(()=>aggregateReceipts([...campaign.slice(0,5),campaign[0]]),/position/);
for(const change of [r=>{r.sourceSha256['rain.js']=hash;},r=>{r.browser.version.product='Other';},r=>{r.host.cpu='Other';}]){
  const copy=structuredClone(campaign);change(copy[1]);assert.throws(()=>aggregateReceipts(copy),/mismatch/);
}
const reversed=structuredClone(campaign);reversed[1].attemptStartedAt=reversed[0].attemptStartedAt;
assert.throws(()=>aggregateReceipts(reversed),/order|overlap/);
const slow=structuredClone(campaign);slow[2].samples.frameIntervalMs=Array(3000).fill(20);slow[2].samples.cpuSubmissionMs=Array(3001).fill(.2);
slow[2].measuredDurationSeconds=60;slow[2].averageFramesPerSecond=50;slow[2].frameIntervalMs=summarize(slow[2].samples.frameIntervalMs);slow[2].cpuSubmissionMs=summarize(slow[2].samples.cpuSubmissionMs);
assert.equal(aggregateReceipts(slow).status,'completed');assert.equal(aggregateReceipts(slow).targetPassed,false,'One slow valid repetition fails the combined target');

const scratch=mkdtempSync(join(tmpdir(),'smythe-measurement-test-'));
const generated=[];
const priorPower=process.env.SMYTHE_MEASURE_POWER_PROFILE,priorWork=process.env.SMYTHE_MEASURE_WORKLOADS;
process.env.SMYTHE_MEASURE_POWER_PROFILE='fixture';process.env.SMYTHE_MEASURE_WORKLOADS='fixture';
try{
  for(const [index,receipt] of campaign.entries()){
    const path=join(scratch,`run-${index}.json`);writeFileSync(path,JSON.stringify(receipt));generated.push(path);
  }
  const aggregatePath=join(scratch,'aggregate.json'),aggregateScript=fileURLToPath(new URL('./aggregate-measurements.mjs',import.meta.url));
  execFileSync(process.execPath,[aggregateScript,aggregatePath,...generated],{stdio:'pipe',windowsHide:true});
  generated.push(aggregatePath);
  const aggregate=JSON.parse(readFileSync(aggregatePath,'utf8'));assert.equal(aggregate.targetPassed,true);assert.equal(aggregate.receipts.length,6);
  assert.throws(()=>execFileSync(process.execPath,[aggregateScript,aggregatePath,...generated.slice(0,6)],{stdio:'pipe',windowsHide:true}),/existing aggregate/);
  const failurePath=join(scratch,'launch-failure.json');let calls=0;
  const failed=await measureRun({url:'http://localhost/preview/',output:failurePath},{command(){calls++;throw new Error('fixture launch failed');}});
  assert.equal(failed.status,'failed');assert.equal(failed.failure.stage,'launch');
  assert.equal(JSON.parse(readFileSync(failurePath,'utf8')).failure.message,'fixture launch failed');
  const before=calls;
  await assert.rejects(()=>measureRun({url:'http://localhost/preview/',output:failurePath},{command(){calls++;}}),/Refusing to replace/);
  assert.equal(calls,before,'Overwrite rejection must precede browser work');
  const order=[],good=fixture();
  const command=(...args)=>{
    order.push(args);
    if(args[0]!=='eval')return {};
    const code=args[1];
    if(code==='GlyphRainPreview.inspect()')return {result:good.startup};
    if(code.includes('UNMASKED_RENDERER_WEBGL'))return {result:browser.gl};
    if(code.includes('const {engineConfig}'))return {result:good.engineConfig};
    if(code.includes('Object.fromEntries(await Promise.all'))return {result:sources};
    if(code==='globalThis.__svgMeasurement !== null')return {result:true};
    if(code==='globalThis.__svgMeasurement'){
      const sample=structuredClone(good);
      for(const key of ['harnessSha256','sourceSha256','servedSourceSha256','protocolSha256','browser','host'])delete sample[key];
      return {result:sample};
    }
    if(code==='globalThis.__svgErrors')return {result:[]};
    return {result:'started'};
  };
  const complete=await measureRun({url:'http://localhost/preview/',output:join(scratch,'complete.json')},{command,metadata:async()=>structuredClone(browser),wait:async()=>{}});
  assert.equal(complete.status,'completed',JSON.stringify(complete.validation.errors));
  assert.deepEqual(order.slice(0,5),[['open','about:blank'],['set','media','reduced-motion'],['set','viewport','1920','1080'],['open','http://localhost/preview/'],['wait','--fn','globalThis.GlyphRainPreview?.version === "2"']]);
  assert.ok(order.findIndex(args=>args[1]?.includes('Object.fromEntries(await Promise.all'))<order.findIndex(args=>args[1]?.includes('GlyphRainPreview.runBenchmark')));
  good.startup.simulationTick=1;
  const invalid=await measureRun({url:'http://localhost/preview/',output:join(scratch,'invalid-start.json')},{command,metadata:async()=>browser});
  assert.equal(invalid.failure.stage,'startup');
}finally{
  for(const name of ['launch-failure.json','complete.json','invalid-start.json'])if(existsSync(join(scratch,name)))unlinkSync(join(scratch,name));
  for(const path of generated)if(existsSync(path))unlinkSync(path);
  rmdirSync(scratch);
  if(priorPower===undefined)delete process.env.SMYTHE_MEASURE_POWER_PROFILE;else process.env.SMYTHE_MEASURE_POWER_PROFILE=priorPower;
  if(priorWork===undefined)delete process.env.SMYTHE_MEASURE_WORKLOADS;else process.env.SMYTHE_MEASURE_WORKLOADS=priorWork;
}
console.log('Passed: startup ordering, retained launch failure, overwrite guard, raw reconciliation, backend scope, complete six-run gates, and corrupt receipt rejection.');
