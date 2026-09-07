// Ten-minute travel/resize check. This is not an FPS measurement.
// node screensaver/svg-preview/soak.mjs LOCAL_HTTP_URL OUTPUT_JSON [SECONDS=600]
import {browserCommand} from './browser-command.mjs';
import {existsSync,mkdirSync,writeFileSync} from 'node:fs';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {hashFiles,hashesMatch,assertServedSources} from './provenance.mjs';
const [url,output,secondsArg='600']=process.argv.slice(2),seconds=Number(secondsArg);
if(!url||!output||!Number.isFinite(seconds)||seconds<=0||seconds>3600)throw new Error('Usage: node soak.mjs LOCAL_HTTP_URL OUTPUT_JSON [SECONDS=600]');
const address=new URL(url);
if(!['http:','https:'].includes(address.protocol)||!['localhost','127.0.0.1','[::1]'].includes(address.hostname))throw new Error('Use a local HTTP URL');
const target=resolve(output);if(existsSync(target))throw new Error(`Refusing to replace an existing receipt: ${target}`);
const browser=process.env.AGENT_BROWSER_BINARY||(process.platform==='win32'
  ?join(process.env.APPDATA||'','npm/node_modules/agent-browser/bin/agent-browser-win32-x64.exe'):'agent-browser');
const sourceDirectory=dirname(fileURLToPath(import.meta.url));
const sourceNames=['model.mjs','rain.js','glyphs.js','index.html','style.css'];
const hashes=()=>hashFiles(sourceDirectory,sourceNames);
const initialHashes=hashes(),session=`smythe-svg-soak-${Date.now()}`,samples=[],errors=[];
const harnessHashes=()=>hashFiles(sourceDirectory,['soak.mjs','provenance.mjs','browser-command.mjs']);
const initialHarnessHashes=harnessHashes();
function command(...args){
  return browserCommand(browser,['--session',session,'--json',...args]);
}
const evaluate=code=>command('eval',code).result;
try{
  command('open',url);command('set','viewport','1920','1080');
  const initial=evaluate('({state:GlyphRainPreview.inspect(),stats:GlyphRainPreview.stats(),catalogCount:SVG_GLYPHS.glyphs.length,catalogSha256:SVG_GLYPHS.catalog_sha256??null})');
  const servedHashes=evaluate(`(async()=>Object.fromEntries(await Promise.all(${JSON.stringify(sourceNames)}.map(async name=>{
    const response=await fetch(new URL(name,location.href),{cache:'no-store'});if(!response.ok)throw new Error(name+' failed to load');
    const digest=await crypto.subtle.digest('SHA-256',await response.arrayBuffer());
    return [name,[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')];
  }))))()`);
  assertServedSources(initialHashes,servedHashes);
  const started=performance.now(),directions=[['ArrowUp','ArrowRight'],['ArrowUp','ArrowLeft'],['ArrowDown','ArrowLeft'],['ArrowDown','ArrowRight']];
  const viewports=[[1920,1080],[1280,720],[390,844],[2560,1440]];
  let iteration=0;
  while((performance.now()-started)/1000<seconds){
    const keys=directions[iteration%directions.length],viewport=viewports[iteration%viewports.length];
    command('set','viewport',...viewport.map(String));
    evaluate(`for(const code of ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'])dispatchEvent(new KeyboardEvent('keyup',{code,bubbles:true}));for(const code of ${JSON.stringify(keys)})dispatchEvent(new KeyboardEvent('keydown',{code,bubbles:true}));'moving'`);
    const remaining=seconds-(performance.now()-started)/1000;
    await new Promise(resolve=>setTimeout(resolve,Math.max(0,Math.min(10000,remaining*1000))));
    const sample=evaluate('({state:GlyphRainPreview.inspect(),stats:GlyphRainPreview.stats(),heapBytes:performance.memory?.usedJSHeapSize??null})');
    sample.elapsedSeconds=(performance.now()-started)/1000;samples.push(sample);
    const {x,z}=sample.state.camera;
    if(!Number.isFinite(x)||!Number.isFinite(z)||x < -70||x>=70||z<0||z>=60)errors.push('Camera escaped the finite repeating world');
    if(JSON.stringify(sample.state.columnIdentities)!==JSON.stringify(initial.state.columnIdentities))errors.push('Column identities changed');
    if(sample.stats.cache.bytes>sample.stats.cache.capBytes||sample.stats.cache.peakBytes>sample.stats.cache.capBytes)errors.push('Image cache exceeded its byte cap');
    if(sample.stats.viewport.width!==viewport[0]||sample.stats.viewport.height!==viewport[1])errors.push('Resize retained the previous render dimensions');
    if(sample.state.rainTime < initial.state.rainTime||(samples.length>1&&sample.state.rainTime<samples.at(-2).state.rainTime))errors.push('Resize or travel reset rain time');
    iteration++;
  }
  evaluate("dispatchEvent(new Event('blur'));GlyphRainPreview.pause(true);'paused'");
  const stopped=evaluate('GlyphRainPreview.stats()');await new Promise(resolve=>setTimeout(resolve,200));
  const final=evaluate('GlyphRainPreview.stats()');
  if(stopped.frames!==final.frames||JSON.stringify(stopped.camera)!==JSON.stringify(final.camera))errors.push('Paused renderer kept drawing or moving after focus release');
  const sourceStable=hashesMatch(initialHashes,hashes()),harnessStable=hashesMatch(initialHarnessHashes,harnessHashes());
  if(!sourceStable)errors.push('Source files changed during soak');
  if(!harnessStable)errors.push('Soak harness changed during soak');
  const receipt={schemaVersion:1,status:errors.length?'failed':'passed',createdAt:new Date().toISOString(),requestedSeconds:seconds,
    actualSeconds:(performance.now()-started)/1000,scope:'Travel, resize, state persistence, focus release and bounded raw image-cache storage; not an FPS or total GPU/process-memory claim',
    evidenceStatus:seconds>=600&&initial.catalogCount===192?'completed-soak':'diagnostic-short-or-partial-soak',sourceSha256:initialHashes,
    servedSourceSha256:servedHashes,sourceStable,harnessSha256:initialHarnessHashes,harnessStable,iterations:iteration,
    initial,final,samples,errors:[...new Set(errors)],userAgent:evaluate('navigator.userAgent')};
  mkdirSync(dirname(target),{recursive:true});writeFileSync(target,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({status:receipt.status,output:target,iterations:iteration,actualSeconds:receipt.actualSeconds,errors:receipt.errors},null,2));
  if(errors.length)process.exitCode=1;
}finally{try{command('close');}catch{}}
