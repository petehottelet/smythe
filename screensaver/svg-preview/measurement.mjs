// Pure receipt validation. No browser startup or graphics work happens here.
import {presetValues,engineConfig} from './config.mjs';
import {hashesMatch} from './provenance.mjs';
import {normalizeLaunchArguments} from './browser-metadata.mjs';

export const PROTOCOL='regl-20260907-v1';
export const SCHEDULE=[['classic',1],['3d',1],['3d',2],['classic',2],['classic',3],['3d',3]];
export const HARNESS_FILES=['measure.mjs','measurement.mjs','aggregate-measurements.mjs','browser-metadata.mjs','browser-launch.mjs','provenance.mjs','browser-command.mjs','source-files.mjs'];
const normalize=value=>value&&typeof value==='object'?(Array.isArray(value)?value.map(normalize)
  :Object.fromEntries(Object.keys(value).sort().map(key=>[key,normalize(value[key])]))):value;
const canonical=value=>JSON.stringify(normalize(value));
export const same=(a,b)=>canonical(a)===canonical(b);
const close=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-6*Math.max(1,Math.abs(a),Math.abs(b));

export function summarize(items){
  const sorted=[...items].sort((a,b)=>a-b),quantile=f=>{
    if(!sorted.length)return null;
    const index=(sorted.length-1)*f,left=Math.floor(index);
    return sorted[left]+(sorted[Math.ceil(index)]-sorted[left])*(index-left);
  };
  return {samples:items.length,mean:items.length?items.reduce((a,b)=>a+b,0)/items.length:null,
    median:quantile(.5),p95:quantile(.95),max:sorted.at(-1)??null};
}

export function expectedConfig(preset){
  if(!['classic','3d'].includes(preset))throw new Error('The primary campaign supports Classic and 3D only');
  return {...presetValues(preset),resolution:.75,numColumns:80,density:1,originalMix:10,fps:60};
}

export function expectedEngineConfig(preset,url){
  const address=new URL(url);
  if(!['http:','https:'].includes(address.protocol)||!['localhost','127.0.0.1','[::1]'].includes(address.hostname))throw new Error('Scene must use a local HTTP URL');
  return {...engineConfig(expectedConfig(preset)),glyphMSDFURL:new URL('./reference/matrixcode_msdf.png',address).href,
    generatedAtlasURL:new URL('./generated-sdf.png',address).href};
}

export function validateStartup(initial,preset){
  if(initial?.version!=='2'||!initial.paused||initial.simulationTick!==0||initial.rainTime!==0
    ||!same(initial.camera,{x:0,z:0}))throw new Error('Expected paused zero-tick, zero-time startup at the camera origin');
  if(!same(initial.config,expectedConfig(preset)))throw new Error('The resolved configuration differs from the frozen scenario');
  if(!same(initial.viewport,{width:1920,height:1080,dpr:1,renderScale:.75,physicalWidth:1440,physicalHeight:810}))throw new Error('Unexpected viewport, DPR, or drawing-buffer workload');
  if(initial.catalogCount!==192||initial.referenceGlyphCount!==56||initial.referenceSequenceLength!==57
    ||initial.totalVisibleShapes!==248||initial.glyphMix!==.1)throw new Error('Unexpected catalog counts or mix');
}

export function classifyBackend(browser){
  const renderer=browser?.gl?.unmaskedRenderer??'';
  const devices=browser?.systemInfo?.gpu?.devices??[];
  const deviceNames=devices.map(device=>[device.vendorString,device.deviceString].filter(Boolean).join(' '));
  const auxiliary=browser?.systemInfo?.gpu?.auxAttributes?.glRenderer??'';
  const software=/swiftshader|llvmpipe|softpipe|software|lavapipe|microsoft basic render/i;
  // An enumerated fallback adapter is not evidence that this context uses it.
  if(software.test(renderer))return 'software';
  if(software.test(auxiliary))return 'unknown';
  const vendors=[[/nvidia|geforce/i,[0x10de]],[/amd|radeon/i,[0x1002,0x1022]],[/intel/i,[0x8086]],
    [/apple/i,[0x106b]],[/qualcomm|adreno/i,[0x5143]],[/arm|mali/i,[0x13b5]]];
  if(vendors.some(([pattern,ids])=>pattern.test(renderer)&&(!auxiliary||pattern.test(auxiliary))&&devices.some((device,index)=>
    ids.includes(device.vendorId)||pattern.test(deviceNames[index]))))return 'hardware-reported';
  return 'unknown';
}

export function validateReceipt(receipt){
  const errors=[],check=(condition,message)=>{if(!condition)errors.push(message);};
  if(!receipt||typeof receipt!=='object')return {valid:false,errors:['Receipt must be an object'],targetPassed:false};
  check(receipt.status==='completed','Receipt did not complete');
  check(receipt.protocol===PROTOCOL,'Wrong or missing protocol');
  check(Number.isFinite(Date.parse(receipt.attemptStartedAt))&&Number.isFinite(Date.parse(receipt.attemptFinishedAt))
    &&Date.parse(receipt.attemptFinishedAt)>=Date.parse(receipt.attemptStartedAt),'Missing or invalid attempt timestamps');
  check([receipt.warmupSeconds,receipt.requestedDurationSeconds,receipt.measuredDurationSeconds].every(Number.isFinite)
    &&receipt.warmupSeconds===5&&receipt.requestedDurationSeconds===60&&receipt.measuredDurationSeconds>=60,'Wrong or invalid primary warmup/sample duration');
  try{validateStartup(receipt.startup,receipt.scenario?.preset);}catch(error){errors.push(error.message);}
  try{check(same(receipt.engineConfig,expectedEngineConfig(receipt.scenario?.preset,receipt.sceneUrl)),'Resolved engine configuration differs from the frozen scenario');}catch(error){errors.push(error.message);}
  check(receipt.sourceStable===true&&receipt.harnessStable===true,'Source or harness changed');
  check(receipt.protocolStable===true&&/^[0-9a-f]{64}$/.test(receipt.protocolSha256??''),'Protocol identity missing or changed');
  for(const key of ['sourceSha256','harnessSha256']){
    const map=receipt[key];check(map&&Object.keys(map).length>0&&Object.values(map).every(hash=>typeof hash==='string'&&/^[0-9a-f]{64}$/.test(hash)),`Invalid ${key}`);
  }
  check(same(Object.keys(receipt.harnessSha256??{}).sort(),[...HARNESS_FILES].sort()),'Incomplete or unexpected harness inventory');
  check(Boolean(receipt.sourceSha256&&receipt.servedSourceSha256&&hashesMatch(receipt.sourceSha256,receipt.servedSourceSha256)),'Served sources differ');
  check(['rain.js','config.mjs','timing.mjs','glyphs.js','base-glyphs.js','generated-sdf.png',
    'reference/matrixcode_msdf.png','reference/catalog.json','engine/manifest.json'].every(name=>receipt.sourceSha256?.[name]),'Missing required renderer/catalog bindings');
  const first=receipt.initialStats,last=receipt.finalStats;
  check(Boolean(first&&last&&same(first.config,last.config)&&same(first.config,receipt.startup?.config)
    &&same(first.viewport,last.viewport)&&same(first.viewport,receipt.startup?.viewport)),'Workload changed during sampling');
  check(Boolean(first&&last&&same(first.camera,{x:0,z:0})&&same(last.camera,{x:0,z:0})),'Manual camera movement during sampling');
  check(receipt.catalogCount===192&&receipt.referenceGlyphCount===56,'Incomplete catalogs');
  check(typeof receipt.catalogSha256==='string'&&/^[0-9a-f]{64}$/.test(receipt.catalogSha256),'Missing original catalog identity');
  check(typeof receipt.referenceCatalogSha256==='string'&&/^[0-9a-f]{64}$/.test(receipt.referenceCatalogSha256),'Missing reference catalog identity');
  check(receipt.referenceCatalogSha256===receipt.sourceSha256?.['reference/catalog.json'],'Reference catalog identity differs from its source hash');
  const boundaries=receipt.sampleBoundary;
  check(Boolean(boundaries&&Number.isInteger(boundaries.first?.tick)&&Number.isInteger(boundaries.last?.tick)
    &&boundaries.first.tick>=0&&boundaries.last.tick>boundaries.first.tick
    &&Number.isFinite(boundaries.first.time)&&Number.isFinite(boundaries.last.time)
    &&boundaries.first.time>=0&&boundaries.last.time>boundaries.first.time),'Missing actual sample boundary simulation state');
  const intervals=receipt.samples?.frameIntervalMs,cpu=receipt.samples?.cpuSubmissionMs;
  const rawValid=Array.isArray(intervals)&&intervals.length>1&&intervals.every(n=>Number.isFinite(n)&&n>0)
    &&Array.isArray(cpu)&&cpu.length===intervals.length+1&&cpu.every(n=>Number.isFinite(n)&&n>=0);
  check(rawValid,'Malformed raw timing samples');
  let recomputed=null;
  if(rawValid){
    const elapsed=intervals.reduce((a,b)=>a+b,0)/1000;
    recomputed={frameIntervalMs:summarize(intervals),cpuSubmissionMs:summarize(cpu),averageFramesPerSecond:intervals.length/elapsed};
    check(close(elapsed,receipt.measuredDurationSeconds),'Raw intervals do not reconcile to duration');
    for(const key of ['frameIntervalMs','cpuSubmissionMs']){
      check(Object.entries(recomputed[key]).every(([field,value])=>close(value,receipt[key]?.[field])),`${key} summary differs from raw samples`);
    }
    check(close(recomputed.averageFramesPerSecond,receipt.averageFramesPerSecond),'Draw rate differs from raw samples');
  }
  const browser=receipt.browser;
  check(Boolean(browser?.version?.product&&browser?.version?.revision&&Array.isArray(browser?.commandLine?.arguments)),'Missing actual browser build or launch arguments');
  check(browser?.headless===true,'Primary campaign requires confirmed headless mode');
  check(Boolean(browser?.gl?.renderer&&browser?.gl?.version&&Array.isArray(browser?.gl?.extensions)),'Missing GL identity');
  check(Array.isArray(receipt.browserErrors)&&receipt.browserErrors.length===0,'Browser reported errors');
  for(const field of ['powerProfile','competingWorkloads'])check(typeof receipt.host?.[field]==='string'
    &&receipt.host[field].trim()!==''&&receipt.host[field]!=='not recorded',`Missing host declaration: ${field}`);
  check(receipt.scenario?.position===SCHEDULE.findIndex(([preset,rep])=>preset===receipt.scenario?.preset&&rep===receipt.scenario?.repetition)+1,'Scenario position mismatch');
  return {valid:errors.length===0,errors,recomputed,backend:classifyBackend(browser),
    targetPassed:errors.length===0&&recomputed.frameIntervalMs.p95<=16.7&&recomputed.averageFramesPerSecond>=59.5};
}

export function aggregateReceipts(receipts){
  if(!Array.isArray(receipts)||receipts.length!==6)throw new Error('The primary campaign requires exactly six retained repetitions');
  const sorted=[...receipts].sort((a,b)=>a.scenario?.position-b.scenario?.position);
  if(!sorted.every((r,i)=>r.scenario?.position===i+1))throw new Error('Missing or duplicated scenario position');
  for(const [index,receipt] of sorted.entries()){
    for(const key of ['sourceSha256','harnessSha256','protocolSha256'])if(!same(receipt[key],sorted[0][key]))throw new Error(`Campaign ${key} mismatch`);
    if(!same(receipt.host,sorted[0].host))throw new Error('Campaign host mismatch');
    for(const key of ['version','gl','headless'])if(!same(receipt.browser?.[key],sorted[0].browser?.[key]))throw new Error(`Campaign browser ${key} mismatch`);
    if(!same(normalizeLaunchArguments(receipt.browser?.commandLine?.arguments??[]),normalizeLaunchArguments(sorted[0].browser?.commandLine?.arguments??[])))throw new Error('Campaign browser launch configuration mismatch');
    if(!same(receipt.browser?.systemInfo?.gpu?.devices,sorted[0].browser?.systemInfo?.gpu?.devices))throw new Error('Campaign GPU device mismatch');
    const samePreset=sorted.find(other=>other.scenario.preset===receipt.scenario.preset);
    if(!same(receipt.engineConfig,samePreset.engineConfig))throw new Error('Campaign resolved engine configuration mismatch');
    if(index&&Date.parse(receipt.attemptStartedAt)<Date.parse(sorted[index-1].attemptFinishedAt))throw new Error('Campaign session order differs or attempts overlap');
  }
  const checks=sorted.map(validateReceipt),valid=checks.every(check=>check.valid);
  return {protocol:PROTOCOL,status:valid?'completed':'invalid',evidenceStatus:'diagnostic-awaiting-independent-review',
    claimable:false,repeatCount:3,attemptCount:receipts.length,schedule:sorted.map(r=>r.scenario),
    targetPassed:valid&&checks.every(check=>check.targetPassed),
    hardwareTargetPassed:valid&&checks.every(check=>check.targetPassed&&check.backend==='hardware-reported'),
    checks,perPreset:Object.fromEntries(['classic','3d'].map(preset=>[preset,sorted.filter(r=>r.scenario.preset===preset).map(r=>({
      repetition:r.scenario.repetition,frameIntervalP95Ms:r.frameIntervalMs?.p95,averageDrawsPerSecond:r.averageFramesPerSecond,cpuSubmissionP95Ms:r.cpuSubmissionMs?.p95}))]))};
}
