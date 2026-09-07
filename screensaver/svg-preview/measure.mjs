// Local, single-attempt collection. Never replaces a receipt or starts a provider.
import {browserCommand} from './browser-command.mjs';
import {browserMetadata} from './browser-metadata.mjs';
import {existsSync,mkdirSync,writeFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {hashFiles,hashesMatch,assertServedSources} from './provenance.mjs';
import os from 'node:os';
import {rendererSources} from './source-files.mjs';
import {PROTOCOL,SCHEDULE,HARNESS_FILES,validateStartup,validateReceipt} from './measurement.mjs';

const directory=dirname(fileURLToPath(import.meta.url));
const PROTOCOL_PATH='../../benchmarks/renderer_performance_20260907.md';
const HARNESS=HARNESS_FILES;

export async function measureRun({url,output,durationSeconds=60,warmupSeconds=5,repetition=1},dependencies={}){
  if(!url||!output)throw new Error('Usage: node measure.mjs LOCAL_HTTP_URL OUTPUT_JSON [DURATION=60] [WARMUP=5] [REPETITION=1]');
  const address=new URL(url),target=resolve(output);
  if(!['http:','https:'].includes(address.protocol)||!['localhost','127.0.0.1','[::1]'].includes(address.hostname))throw new Error('Use a local HTTP preview URL');
  if(!Number.isFinite(durationSeconds)||durationSeconds<=0||durationSeconds>600||!Number.isFinite(warmupSeconds)||warmupSeconds<0||warmupSeconds>600)throw new Error('Invalid duration');
  if(!Number.isInteger(repetition)||repetition<1||repetition>3)throw new Error('Repetition must be 1, 2, or 3');
  if(existsSync(target))throw new Error(`Refusing to replace an existing measurement receipt: ${target}`);
  const preset=address.searchParams.get('preset')??'classic';
  const position=SCHEDULE.findIndex(([name,rep])=>name===preset&&rep===repetition)+1;
  if(!position)throw new Error('Use preset=classic or preset=3d');
  const session=`smythe-regl-measure-${Date.now()}-${process.pid}`,sourceNames=[...new Set([
    ...rendererSources(directory),'reference/catalog.json','reference/provenance.json','reference/LICENSE'])].sort();
  const initialHashes=hashFiles(directory,sourceNames),initialHarness=hashFiles(directory,HARNESS);
  const receipt={schemaVersion:3,protocol:PROTOCOL,status:'failed',evidenceStatus:'diagnostic',repeatCount:1,
    scenario:{preset,repetition,position},sceneUrl:address.href,scheduleSeed:20260907,
    initialization:'Frozen GLSL coordinate/time hash functions; paused tick/time zero; no external field-seed parameter',
    protocolSha256:hashFiles(directory,[PROTOCOL_PATH])[PROTOCOL_PATH],
    sourceSha256:initialHashes,harnessSha256:initialHarness,sourceStable:false,harnessStable:false,
    requestedDurationSeconds:durationSeconds,warmupSeconds,createdAt:new Date().toISOString(),
    attemptStartedAt:new Date().toISOString(),stage:'launch',browserErrors:[],
    host:{os:os.platform(),release:os.release(),architecture:os.arch(),cpu:os.cpus()[0]?.model??null,
      logicalCpuCount:os.cpus().length,totalSystemMemoryBytes:os.totalmem(),
      powerProfile:process.env.SMYTHE_MEASURE_POWER_PROFILE??'not recorded',
      competingWorkloads:process.env.SMYTHE_MEASURE_WORKLOADS??'not recorded'},
    harness:{browserTool:'agent-browser',pollSeconds:5,viewport:[1920,1080],dpr:1,
      scope:'Headless local-browser callback/submission measurement. Polling and host workloads can affect intervals.'}};
  let command=dependencies.command;
  const evaluate=code=>command('eval',code).result;
  const wait=dependencies.wait??(ms=>new Promise(resolve=>setTimeout(resolve,ms)));
  const clock=dependencies.clock??Date.now;
  try{
    if(!command){
      const browser=process.env.AGENT_BROWSER_BINARY||(process.platform==='win32'
        ?join(process.env.APPDATA||'','npm/node_modules/agent-browser/bin/agent-browser-win32-x64.exe'):'agent-browser');
      command=(...args)=>browserCommand(browser,['--session',session,'--json',...args]);
      receipt.harness.browserToolVersion=execFileSync(browser,['--version'],{encoding:'utf8',timeout:10000,windowsHide:true}).trim();
    }else receipt.harness.browserToolVersion='injected test boundary';
    command('open','about:blank');command('set','media','reduced-motion');command('set','viewport','1920','1080');
    command('open',address.href);command('wait','--fn','globalThis.GlyphRainPreview?.version === "2"');
    receipt.stage='startup';receipt.startup=evaluate('GlyphRainPreview.inspect()');
    validateStartup(receipt.startup,preset);
    receipt.browser=await (dependencies.metadata??browserMetadata)(command('get','cdp-url'));
    receipt.browser.gl=evaluate(`(()=>{
      const gl=document.getElementById('rain').getContext('webgl'),debug=gl.getExtension('WEBGL_debug_renderer_info');
      return {renderer:gl.getParameter(gl.RENDERER),vendor:gl.getParameter(gl.VENDOR),version:gl.getParameter(gl.VERSION),
        shadingLanguageVersion:gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
        unmaskedRenderer:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):null,
        unmaskedVendor:debug?gl.getParameter(debug.UNMASKED_VENDOR_WEBGL):null,
        extensions:gl.getSupportedExtensions().sort(),contextAttributes:gl.getContextAttributes()};
    })()`);
    if(!receipt.browser.headless)throw new Error('The primary campaign requires a confirmed headless browser');
    receipt.engineConfig=evaluate("(async()=>{const {engineConfig}=await import('./config.mjs');return engineConfig(GlyphRainPreview.stats().config);})()");
    receipt.stage='source-verification';
    receipt.servedSourceSha256=evaluate(`(async()=>Object.fromEntries(await Promise.all(${JSON.stringify(sourceNames)}.map(async name=>{
      const response=await fetch(new URL(name,location.href),{cache:'no-store'});if(!response.ok)throw new Error(name+' failed to load');
      const digest=await crypto.subtle.digest('SHA-256',await response.arrayBuffer());
      return [name,[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')];
    }))))()`);
    assertServedSources(initialHashes,receipt.servedSourceSha256);
    receipt.referenceCatalogSha256=receipt.servedSourceSha256['reference/catalog.json'];
    evaluate(`globalThis.__svgErrors=[];addEventListener('error',e=>__svgErrors.push(String(e.message)));
      addEventListener('unhandledrejection',e=>__svgErrors.push(String(e.reason)));
      const priorError=console.error;console.error=(...args)=>{__svgErrors.push(args.map(String).join(' '));priorError.apply(console,args);};'monitoring'`);
    receipt.stage='sampling';
    evaluate(`globalThis.__svgMeasurement=null; GlyphRainPreview.runBenchmark(${JSON.stringify({warmupSeconds,durationSeconds})}).then(value=>globalThis.__svgMeasurement=value); 'started'`);
    const deadline=clock()+(warmupSeconds+durationSeconds+30)*1000;
    let sample=null;
    while(clock()<deadline){
      await wait(5000);
      if(evaluate('globalThis.__svgMeasurement !== null')){sample=evaluate('globalThis.__svgMeasurement');break;}
    }
    if(!sample)throw new Error('Measurement timed out');
    for(const key of ['status','invalidReason','measurement','createdAt','warmupSeconds','requestedDurationSeconds',
      'measuredDurationSeconds','catalogCount','catalogSha256','referenceGlyphCount','frameIntervalMs','cpuSubmissionMs',
      'averageFramesPerSecond','samples','sampleBoundary','initialStats','finalStats','userAgent','exclusions']){
      if(Object.hasOwn(sample,key))receipt[key]=sample[key];
    }
    receipt.browserErrors=evaluate('globalThis.__svgErrors');receipt.stage='completed';
  }catch(error){
    receipt.status='failed';receipt.failure={stage:receipt.stage,name:error.name,message:String(error.message).slice(0,4000)};
    receipt.invalidReason=receipt.failure.message;
  }finally{
    try{
      try{
        receipt.sourceStable=hashesMatch(initialHashes,hashFiles(directory,sourceNames));
        receipt.harnessStable=hashesMatch(initialHarness,hashFiles(directory,HARNESS));
        receipt.protocolStable=receipt.protocolSha256===hashFiles(directory,[PROTOCOL_PATH])[PROTOCOL_PATH];
      }catch(error){receipt.status='failed';receipt.provenanceError=String(error.message);}
      receipt.attemptFinishedAt=new Date().toISOString();
      receipt.validation=validateReceipt(receipt);
      if(receipt.status==='completed'&&!receipt.validation.valid)receipt.status='invalid';
      mkdirSync(dirname(target),{recursive:true});writeFileSync(target,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
    }finally{try{command?.('close');}catch{}}
  }
  return receipt;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const [url,output,duration='60',warmup='5',repetition='1']=process.argv.slice(2);
  const result=await measureRun({url,output,durationSeconds:Number(duration),warmupSeconds:Number(warmup),repetition:Number(repetition)});
  console.log(JSON.stringify({status:result.status,output,scenario:result.scenario,validation:result.validation,
    averageDrawsPerSecond:result.averageFramesPerSecond,frameIntervalP95Ms:result.frameIntervalMs?.p95},null,2));
  if(result.status!=='completed')process.exitCode=1;
}
