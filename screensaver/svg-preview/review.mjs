// Fixed-step visual and functional evidence only; these checks do not measure FPS.
import assert from 'node:assert/strict';
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {browserCommand} from './browser-command.mjs';
import {hashFiles,hashesMatch,assertServedSources} from './provenance.mjs';
import {rendererSources} from './source-files.mjs';

const [url,output]=process.argv.slice(2);
if(!url||!output)throw new Error('Usage: node review.mjs LOCAL_HTTP_URL OUTPUT_JSON');
const address=new URL(url);
if(!['http:','https:'].includes(address.protocol)||!['localhost','127.0.0.1','[::1]'].includes(address.hostname))throw new Error('Use a local HTTP URL');
const target=resolve(output);if(existsSync(target))throw new Error('Refusing to replace an existing visual review');
const directory=dirname(fileURLToPath(import.meta.url));
const sourceNames=[...new Set([...rendererSources(directory),'catalog.html','reference/catalog.json','reference/provenance.json','reference/LICENSE'])].sort();
const sourceSha256=hashFiles(directory,sourceNames),harnessNames=['review.mjs','browser-command.mjs','provenance.mjs','source-files.mjs'];
const harnessSha256=hashFiles(directory,harnessNames);
const browser=process.env.AGENT_BROWSER_BINARY||(process.platform==='win32'?join(process.env.APPDATA||'','npm/node_modules/agent-browser/bin/agent-browser-win32-x64.exe'):'agent-browser');
const session=`smythe-reference-review-${Date.now()}`;
const command=(...args)=>browserCommand(browser,['--session',session,'--json',...args]);
const evaluate=code=>command('eval',code).result;
const checks=[],samples=[];
function check(name,actual,passed,expected){checks.push({name,passed:Boolean(passed),expected,actual});assert.ok(passed,`${name}: ${JSON.stringify(actual)}`);}
function openScene(preset,originalMix){
  const scene=new URL(url);scene.search='';scene.searchParams.set('preset',preset);scene.searchParams.set('originalMix',originalMix);
  command('open',scene.href);command('wait','--fn','globalThis.GlyphRainPreview?.version === "2"');
  const initial=evaluate('GlyphRainPreview.stats()');
  check(`startup/${preset}/${originalMix}`,initial,initial.paused&&initial.rainTime===0&&initial.simulationTick===0&&initial.config.preset===preset&&initial.config.originalMix===originalMix,'Reduced-motion startup, paused at tick 0 with requested preset and mix');
  evaluate('GlyphRainPreview.stepForReview(228); "stepped"');return evaluate('GlyphRainPreview.inspectFrame()');
}
function startBenchmark(){return evaluate('globalThis.__reviewResult=null;globalThis.__reviewPriorPaused=GlyphRainPreview.stats().paused;GlyphRainPreview.runBenchmark({warmupSeconds:0,durationSeconds:60}).then(result=>globalThis.__reviewResult=result);({priorPaused:__reviewPriorPaused,running:GlyphRainPreview.benchmarkRunning()})');}
function interrupted(name,reason){
  command('wait','--fn','globalThis.__reviewResult !== null');
  const actual=evaluate('({result:__reviewResult,paused:GlyphRainPreview.stats().paused,priorPaused:__reviewPriorPaused,running:GlyphRainPreview.benchmarkRunning()})');
  check(name,actual,actual.result.status==='invalid'&&reason.test(actual.result.invalidReason)&&actual.paused===actual.priorPaused&&!actual.running,'Measurement invalidated for the tested action and prior pause restored');return actual;
}
try{
  command('open','about:blank');command('set','media','reduced-motion');command('set','viewport','1920','1080');command('open',url);
  command('wait','--fn','globalThis.GlyphRainPreview?.version === "2"');
  const servedSourceSha256=evaluate(`(async()=>Object.fromEntries(await Promise.all(${JSON.stringify(sourceNames)}.map(async name=>{
    const response=await fetch(new URL(name,location.href),{cache:'no-store'});if(!response.ok)throw new Error(name+' failed to load');
    const digest=await crypto.subtle.digest('SHA-256',await response.arrayBuffer());return [name,[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')];
  }))))()`);
  assertServedSources(sourceSha256,servedSourceSha256);check('servedSources',Object.keys(servedSourceSha256).length,hashesMatch(sourceSha256,servedSourceSha256),'Every declared source and asset matches local bytes');
  const viewport=evaluate('({width:innerWidth,height:innerHeight,dpr:devicePixelRatio,reduced:matchMedia("(prefers-reduced-motion: reduce)").matches})');
  check('viewport',viewport,viewport.width===1920&&viewport.height===1080&&viewport.dpr===1&&viewport.reduced,'1920x1080 DPR 1 with reduced motion');
  for(const [preset,mix] of [['classic',0],['classic',10],['classic',100],['operator',10],['3d',10]]){
    const sample=openScene(preset,mix);samples.push({preset,originalMix:mix,...sample});
    check(`frame/${preset}/${mix}`,sample,sample.pixels>0&&sample.darkFraction>0&&sample.darkFraction<.99&&Math.abs(sample.rainTime-3.8)<1e-10&&sample.simulationTick===228,'Nonblank frame after 228 fixed 60Hz steps; 0 < dark fraction < 0.99');
  }
  const before=evaluate('globalThis.__reviewOrigin=document.querySelector("#rain").toDataURL();GlyphRainPreview.stats()');
  const moved=evaluate('GlyphRainPreview.moveForReview(12,18);({changed:__reviewOrigin!==document.querySelector("#rain").toDataURL(),...GlyphRainPreview.stats()})');
  check('paused3DMovement',moved,moved.changed&&moved.paused&&moved.rainTime===before.rainTime&&moved.camera.x===12&&moved.camera.z===18,'Move to (12,18), change pixels, preserve paused time');
  command('press','r');const reset=evaluate('({exactPixels:__reviewOrigin===document.querySelector("#rain").toDataURL(),...GlyphRainPreview.inspect()})');
  check('reset',reset,reset.exactPixels&&reset.camera.x===0&&reset.camera.z===0&&reset.rainTime===before.rainTime&&reset.heldKeys.length===0,'R restores exact drawing-buffer PNG bytes and origin without advancing time');
  const baseProvenance=evaluate('({count:BASE_GLYPHS.count,source:BASE_GLYPHS.source,canvas:BASE_GLYPHS.canvas,fillRule:BASE_GLYPHS.fill_rule})');
  const originalProvenance=evaluate('({count:SVG_GLYPHS.glyphs.length,version:SVG_GLYPHS.version,catalogSha256:SVG_GLYPHS.catalog_sha256})');
  command('open',new URL('catalog.html',url).href);command('wait','--fn','document.querySelectorAll("#catalog svg").length > 0');
  const inspectCatalog=()=>evaluate('({svgs:document.querySelectorAll("#catalog svg").length,paths:document.querySelectorAll("#catalog path").length,emptyPaths:[...document.querySelectorAll("#catalog path")].filter(path=>!path.getAttribute("d")?.trim()).length,emptyGlyphs:[...document.querySelectorAll("#catalog svg")].filter(svg=>!svg.querySelector("path")).length,active:document.querySelector(".catalog-heading button[aria-pressed=true]")?.id})');
  const originals=inspectCatalog();check('originalCatalog',originals,originals.svgs===192&&originals.paths>=192&&!originals.emptyPaths&&!originals.emptyGlyphs&&originals.active==='originals','Default catalog displays 192 nonempty original SVGs');
  command('click','#reference');const reference=inspectCatalog();check('referenceCatalog',reference,reference.svgs===56&&reference.paths>=56&&!reference.emptyPaths&&!reference.emptyGlyphs&&reference.active==='reference','Reference button displays 56 nonempty imported SVGs');
  const screenshotFrame=openScene('classic',10);command('wait','--fn','document.body.classList.contains("idle")');
  const screenshotState=evaluate('({idle:document.body.classList.contains("idle"),paused:GlyphRainPreview.stats().paused,rainTime:GlyphRainPreview.stats().rainTime,focused:document.activeElement?.tagName})');
  check('screenshotState',screenshotState,screenshotState.idle&&screenshotState.paused&&Math.abs(screenshotState.rainTime-3.8)<1e-10,'Normal idle chrome at paused 3.8 seconds');
  command('screenshot',join(directory,'preview.png'));const png=readFileSync(join(directory,'preview.png'));
  const dimensions={width:png.readUInt32BE(16),height:png.readUInt32BE(20)};check('screenshotDimensions',dimensions,dimensions.width===1920&&dimensions.height===1080,'Actual 1920x1080 browser screenshot');
  openScene('3d',10);const settingsStart=startBenchmark();check('settingsBenchmarkStarted',settingsStart,settingsStart.running&&settingsStart.priorPaused,'Start a 60-second benchmark from paused state');
  command('press','s');interrupted('settingsInterrupt',/Settings opened/);const opened=evaluate('document.querySelector("dialog").open');check('settingsOpened',opened,opened,'S opens settings');
  command('press','Escape');const closed=evaluate('!document.querySelector("dialog").open');check('settingsClosed',closed,closed,'Escape closes settings');
  command('focus','[data-direction="ArrowUp"]');evaluate('globalThis.__reviewClickDetail=null;document.querySelector("[data-direction=ArrowUp]").addEventListener("click",event=>globalThis.__reviewClickDetail=event.detail,{once:true});"listening"');
  const navigationStart=startBenchmark();check('navigationBenchmarkStarted',navigationStart,navigationStart.running&&navigationStart.priorPaused,'Start another benchmark from paused state');
  command('press','Enter');interrupted('keyboardButtonInterrupt',/Keyboard navigation/);const clickDetail=evaluate('__reviewClickDetail');check('keyboardClickDetail',clickDetail,clickDetail===0,'Enter triggers the detail 0 movement-button path');
  // Context loss intentionally leaves this scene unable to render: exercise last.
  const contextAvailable=evaluate('Boolean(document.querySelector("#rain").getContext("webgl").getExtension("WEBGL_lose_context"))');let contextLoss={available:contextAvailable,status:'unavailable'};
  if(contextAvailable){
    startBenchmark();evaluate('document.querySelector("#rain").getContext("webgl").getExtension("WEBGL_lose_context").loseContext();"requested"');const loss=interrupted('contextLossInterrupt',/WebGL context lost/);
    const stoppedAt=evaluate('GlyphRainPreview.stats().frames');command('wait','250');const after=evaluate('GlyphRainPreview.stats().frames');
    check('contextLossStopsFrames',{stoppedAt,after},after===stoppedAt,'No additional draw frames in 250ms after context loss');contextLoss={available:contextAvailable,result:loss.result,framesAtStop:stoppedAt,framesAfter250Ms:after};
  }
  const sourceStable=hashesMatch(sourceSha256,hashFiles(directory,sourceNames)),harnessStable=hashesMatch(harnessSha256,hashFiles(directory,harnessNames));
  check('sourceStability',sourceStable,sourceStable,'Renderer and catalog sources unchanged');check('harnessStability',harnessStable,harnessStable,'Review harness unchanged');
  const receipt={schemaVersion:2,status:checks.every(item=>item.passed)?'passed':'failed',createdAt:new Date().toISOString(),evidenceStatus:'automated-visual-and-interaction-review',
    scope:'Actual browser checks at fixed simulation steps. Not a timing campaign or human visual-preference judgment.',rendererVersion:evaluate('GlyphRainPreview.version'),userAgent:evaluate('navigator.userAgent'),viewport,
    sourceSha256,servedSourceSha256,harnessSha256,sourceStable,harnessStable,originalProvenance,baseProvenance,checks,samples,contextLoss,
    screenshot:{path:'screensaver/svg-preview/preview.png',sha256:hashFiles(directory,['preview.png'])['preview.png'],...dimensions,state:screenshotState,frame:screenshotFrame,method:'Actual browser screenshot after 228 fixed 60Hz steps; normal idle chrome, no compositing.'}};
  mkdirSync(dirname(target),{recursive:true});writeFileSync(target,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({output:target,status:receipt.status,checks:checks.length,samples:samples.length,contextLossAvailable:contextAvailable},null,2));
}finally{try{command('close');}catch{}}
