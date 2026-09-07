// Run against a local HTTP server. This measures rendering, never provider calls.
// node screensaver/svg-preview/measure.mjs http://127.0.0.1:8000/screensaver/svg-preview/ receipt.json
import {browserCommand} from './browser-command.mjs';
import {existsSync,mkdirSync,writeFileSync} from 'node:fs';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {hashFiles,hashesMatch,assertServedSources} from './provenance.mjs';
import os from 'node:os';

const [url,output,durationArg='60',warmupArg='5']=process.argv.slice(2);
if(!url||!output)throw new Error('Usage: node measure.mjs LOCAL_HTTP_URL OUTPUT_JSON [DURATION_SECONDS=60] [WARMUP_SECONDS=5]');
const parsed=new URL(url);
if(!['http:','https:'].includes(parsed.protocol)||!['localhost','127.0.0.1','[::1]'].includes(parsed.hostname))throw new Error('Use a local HTTP preview URL');
const durationSeconds=Number(durationArg),warmupSeconds=Number(warmupArg);
if(!Number.isFinite(durationSeconds)||durationSeconds<=0||durationSeconds>600||!Number.isFinite(warmupSeconds)||warmupSeconds<0)throw new Error('Invalid duration');
const target=resolve(output);
if(existsSync(target))throw new Error(`Refusing to replace an existing measurement receipt: ${target}`);
let browser=process.env.AGENT_BROWSER_BINARY||'agent-browser';
if(process.platform==='win32'&&!process.env.AGENT_BROWSER_BINARY){
  const native=join(process.env.APPDATA||'', 'npm/node_modules/agent-browser/bin/agent-browser-win32-x64.exe');
  if(!existsSync(native))throw new Error('Set AGENT_BROWSER_BINARY to the native agent-browser executable');
  browser=native;
}
const session=`smythe-svg-measure-${Date.now()}`;
const sourceDirectory=dirname(fileURLToPath(import.meta.url));
const sourceNames=['model.mjs','rain.js','glyphs.js','index.html','style.css'];
const localHashes=()=>hashFiles(sourceDirectory,sourceNames);
const initialHashes=localHashes();
const harnessHashes=()=>hashFiles(sourceDirectory,['measure.mjs','provenance.mjs','browser-command.mjs']);
const initialHarnessHashes=harnessHashes();
function command(...args){
  return browserCommand(browser,['--session',session,'--json',...args]);
}
try{
  command('open',url);command('set','viewport','1920','1080');
  const initial=command('eval','({ready:!!globalThis.GlyphRainPreview,dpr:devicePixelRatio,viewport:[innerWidth,innerHeight]})').result;
  if(!initial.ready||initial.dpr!==1||initial.viewport[0]!==1920||initial.viewport[1]!==1080)throw new Error(`Unexpected browser state: ${JSON.stringify(initial)}`);
  const servedHashes=command('eval',`(async()=>Object.fromEntries(await Promise.all(${JSON.stringify(sourceNames)}.map(async name=>{
    const response=await fetch(new URL(name,location.href),{cache:'no-store'});if(!response.ok)throw new Error(name+' failed to load');
    const digest=await crypto.subtle.digest('SHA-256',await response.arrayBuffer());
    return [name,[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')];
  }))))()`).result;
  assertServedSources(initialHashes,servedHashes);
  command('eval',`globalThis.__svgMeasurement=null; GlyphRainPreview.runBenchmark(${JSON.stringify({warmupSeconds,durationSeconds})}).then(value=>globalThis.__svgMeasurement=value); 'started'`);
  console.log(`Measuring 1920×1080 at DPR 1: ${warmupSeconds}s warmup, ${durationSeconds}s sample.`);
  const deadline=Date.now()+(warmupSeconds+durationSeconds+30)*1000;
  let receipt=null;
  while(Date.now()<deadline){
    await new Promise(resolve=>setTimeout(resolve,5000));
    const complete=command('eval','globalThis.__svgMeasurement !== null').result;
    if(complete){receipt=command('eval','globalThis.__svgMeasurement').result;break;}
  }
  if(!receipt)throw new Error('Measurement timed out');
  receipt.host={os:os.platform(),release:os.release(),architecture:os.arch(),cpu:os.cpus()[0]?.model??null,
    logicalCpuCount:os.cpus().length,totalSystemMemoryBytes:os.totalmem()};
  receipt.harness={browserTool:'agent-browser',pollSeconds:5,viewport:[1920,1080],dpr:1,
    scope:'Headless local-browser measurement. Polling and other host workloads can affect callback intervals.'};
  receipt.sourceSha256=initialHashes;
  receipt.servedSourceSha256=servedHashes;
  receipt.sourceStable=hashesMatch(initialHashes,localHashes());
  receipt.harnessSha256=initialHarnessHashes;
  receipt.harnessStable=hashesMatch(initialHarnessHashes,harnessHashes());
  if(!receipt.sourceStable){receipt.status='invalid';receipt.invalidReason='Source files changed during measurement';}
  if(!receipt.harnessStable){receipt.status='invalid';receipt.invalidReason='Measurement harness changed during measurement';}
  receipt.evidenceStatus='diagnostic';
  receipt.repeatCount=1;
  receipt.diagnosticReasons=['A single renderer run does not establish repeatability or a comparative performance claim.'];
  if(durationSeconds<60||receipt.measuredDurationSeconds<60)receipt.diagnosticReasons.push('The sample is shorter than the 60-second renderer protocol.');
  if(warmupSeconds<5)receipt.diagnosticReasons.push('Warmup is shorter than the 5-second renderer protocol.');
  if(receipt.catalogCount!==192)receipt.diagnosticReasons.push('The catalog does not contain the full 192 glyphs.');
  receipt.timestampSource='Browser host clock at measurement completion; elapsed durations use performance.now().';
  mkdirSync(dirname(target),{recursive:true});writeFileSync(target,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({status:receipt.status,output:target,framesPerSecond:receipt.averageFramesPerSecond,
    frameIntervalP95Ms:receipt.frameIntervalMs.p95,cpuSubmissionP95Ms:receipt.cpuSubmissionMs.p95,
    cachePeakMiB:receipt.finalStats.cache.peakBytes/1024/1024,gate:receipt.gate},null,2));
  if(receipt.status!=='completed')process.exitCode=1;
}finally{try{command('close');}catch{}}
