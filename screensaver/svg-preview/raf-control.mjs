// Diagnostic blank-page RAF cadence. No rain rendering or performance gate.
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {resolve,dirname,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import os from 'node:os';
import {browserCommand} from './browser-command.mjs';
import {browserArguments} from './browser-launch.mjs';
import {browserMetadata} from './browser-metadata.mjs';
import {classifyBackend} from './measurement.mjs';
import {hashFiles,hashesMatch} from './provenance.mjs';

export const CONTROL_FILES=['raf-control.mjs','timing.mjs','browser-command.mjs','browser-launch.mjs',
  'browser-metadata.mjs','measurement.mjs','provenance.mjs'];

export function summarizeCallbacks(timestamps){
  if(!Array.isArray(timestamps)||timestamps.length<2||timestamps.some(value=>!Number.isFinite(value)))throw new Error('Invalid callback timestamps');
  const intervals=timestamps.slice(1).map((value,index)=>value-timestamps[index]);
  if(intervals.some(value=>value<=0))throw new Error('Callbacks must increase strictly');
  const durationMs=timestamps.at(-1)-timestamps[0];
  if(durationMs<60000)throw new Error('At least sixty seconds of actual callbacks required');
  const sorted=[...intervals].sort((a,b)=>a-b),at=(sorted.length-1)*.95,index=Math.floor(at);
  return {callbackCount:timestamps.length,intervalCount:intervals.length,durationMs,
    averageCallbacksPerSecond:intervals.length*1000/durationMs,
    intervalMs:{mean:durationMs/intervals.length,p95:sorted[index]+(sorted[Math.min(index+1,sorted.length-1)]-sorted[index])*(at-index),max:sorted.at(-1)}};
}

export async function runControl(output,repetition){
  if(!Number.isInteger(repetition)||repetition<1||repetition>3)throw new Error('Repetition must be 1..3');
  const target=resolve(output);if(existsSync(target))throw new Error('Refusing to overwrite a control receipt');
  const directory=dirname(fileURLToPath(import.meta.url)),root=resolve(directory,'../..');
  const protocol=join(root,'benchmarks/renderer_cadence_control_20260907.md');
  const session=`smythe-raf-control-${Date.now()}-${process.pid}`;
  const browser=process.env.AGENT_BROWSER_BINARY||(process.platform==='win32'
    ?join(process.env.APPDATA||'','npm/node_modules/agent-browser/bin/agent-browser-win32-x64.exe'):'agent-browser');
  const command=(...args)=>browserCommand(browser,browserArguments(session,args));
  const receipt={schemaVersion:1,campaignId:'glyph-rain-regl-20260907-cadence-control1',status:'failed',
    evidenceStatus:'exploratory-diagnostic',claimable:false,repetition,warmupSeconds:5,requestedSeconds:60,
    scope:'Blank-page RAF callbacks and unchanged 60fps timing gate; no rain, physical presentation or GPU-time measurement',
    attemptStartedAt:new Date().toISOString(),sourceSha256:hashFiles(directory,CONTROL_FILES),
    protocolSha256:hashFiles(dirname(protocol),['renderer_cadence_control_20260907.md'])['renderer_cadence_control_20260907.md'],
    host:{platform:os.platform(),release:os.release(),cpus:os.cpus()[0]?.model,logicalCpuCount:os.cpus().length,
      powerProfile:process.env.SMYTHE_MEASURE_POWER_PROFILE||'not recorded',
      workloads:process.env.SMYTHE_MEASURE_WORKLOADS||'not recorded'}};
  try{
    receipt.browserToolVersion=execFileSync(browser,['--version'],{encoding:'utf8',timeout:10000,windowsHide:true}).trim();
    command('open','about:blank');command('set','media','reduced-motion');command('set','viewport','1920','1080');
    receipt.browser=await browserMetadata(command('get','cdp-url'));
    const moduleUrl=`data:text/javascript;base64,${readFileSync(join(directory,'timing.mjs')).toString('base64')}`;
    receipt.startup=command('eval',`(async()=>{
      const canvas=document.createElement('canvas');canvas.width=2;canvas.height=2;
      const gl=canvas.getContext('webgl',{alpha:false,antialias:false,preserveDrawingBuffer:true});
      if(!gl)throw new Error('WebGL unavailable');const debug=gl.getExtension('WEBGL_debug_renderer_info');
      const info={renderer:gl.getParameter(gl.RENDERER),vendor:gl.getParameter(gl.VENDOR),version:gl.getParameter(gl.VERSION),
        unmaskedRenderer:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):null,
        unmaskedVendor:debug?gl.getParameter(debug.UNMASKED_VENDOR_WEBGL):null,contextAttributes:gl.getContextAttributes()};
      const {createFrameGate}=await import(${JSON.stringify(moduleUrl)}),gate=createFrameGate();
      const data=globalThis.__rafControl={raw:[],selected:[],errors:[],done:false,warmupStartedAt:performance.now()};
      addEventListener('error',event=>data.errors.push(String(event.message)));
      addEventListener('unhandledrejection',event=>data.errors.push(String(event.reason)));
      addEventListener('visibilitychange',()=>data.errors.push('Visibility changed to '+document.visibilityState));
      function frame(now){
        const selected=gate.shouldRender(now,60);
        if(now-data.warmupStartedAt>=5000){
          data.raw.push(now);if(selected)data.selected.push(now);
          if(data.raw.at(-1)-data.raw[0]>=60000&&data.selected.at(-1)-data.selected[0]>=60000){data.done=true;return;}
        }
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
      return {width:innerWidth,height:innerHeight,dpr:devicePixelRatio,visibility:document.visibilityState,
        reducedMotion:matchMedia('(prefers-reduced-motion: reduce)').matches,gl:info};
    })()`).result;
    receipt.browser.gl=receipt.startup.gl;receipt.backend=classifyBackend(receipt.browser);
    if(receipt.startup.width!==1920||receipt.startup.height!==1080||receipt.startup.dpr!==1||receipt.startup.visibility!=='visible'
      ||!receipt.startup.reducedMotion||!receipt.browser.headless)throw new Error('Unexpected blank-page environment');
    const deadline=Date.now()+150000;
    while(!command('eval','globalThis.__rafControl.done').result){
      if(Date.now()>deadline)throw new Error('Control sample timed out');
      await new Promise(resolve=>setTimeout(resolve,5000));
    }
    receipt.samples=command('eval','globalThis.__rafControl').result;
    if(receipt.samples.errors.length)throw new Error('Browser errors during control');
    receipt.raw=summarizeCallbacks(receipt.samples.raw);receipt.selected=summarizeCallbacks(receipt.samples.selected);
    if(receipt.samples.raw[0]-receipt.samples.warmupStartedAt<5000)throw new Error('Warmup too short');
    const rawSet=new Set(receipt.samples.raw);
    if(receipt.samples.selected.some(time=>!rawSet.has(time)))throw new Error('Selected callback was not in raw stream');
    receipt.status='completed';
  }catch(error){receipt.failure={name:error.name,message:String(error.message).slice(0,4000)};}
  finally{
    receipt.attemptFinishedAt=new Date().toISOString();
    receipt.sourcesUnchanged=hashesMatch(receipt.sourceSha256,hashFiles(directory,CONTROL_FILES));
    if(!receipt.sourcesUnchanged){receipt.status='failed';receipt.failure={name:'SourceMismatch',message:'Control sources changed during sampling'};}
    try{mkdirSync(dirname(target),{recursive:true});writeFileSync(target,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});}
    finally{try{command('close');}catch{}}
  }
  return receipt;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const [output,repetition]=process.argv.slice(2);if(!output)throw new Error('Pass NEW_OUTPUT_JSON REPETITION');
  const receipt=await runControl(output,Number(repetition));
  console.log(JSON.stringify({status:receipt.status,output,backend:receipt.backend,raw:receipt.raw,selected:receipt.selected,failure:receipt.failure},null,2));
  if(receipt.status!=='completed')process.exitCode=1;
}
