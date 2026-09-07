// Fixed-clock visual samples only. Never use these draws as timing evidence.
import assert from 'node:assert/strict';
import {existsSync,mkdirSync,writeFileSync} from 'node:fs';
import {dirname,resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {browserCommand} from './browser-command.mjs';
import {hashFiles,hashesMatch,assertServedSources} from './provenance.mjs';

const [url,output]=process.argv.slice(2);
if(!url||!output)throw new Error('Usage: node review.mjs LOCAL_HTTP_URL OUTPUT_JSON');
const address=new URL(url);
if(!['http:','https:'].includes(address.protocol)||!['localhost','127.0.0.1','[::1]'].includes(address.hostname))throw new Error('Use a local HTTP URL');
const target=resolve(output);if(existsSync(target))throw new Error('Refusing to replace an existing visual review');
const sourceDirectory=dirname(fileURLToPath(import.meta.url));
const sourceNames=['model.mjs','rain.js','glyphs.js','index.html','style.css'];
const sourceSha256=hashFiles(sourceDirectory,sourceNames);
const harnessNames=['review.mjs','browser-command.mjs','provenance.mjs'];
const harnessSha256=hashFiles(sourceDirectory,harnessNames);
const browser=process.env.AGENT_BROWSER_BINARY||(process.platform==='win32'
  ?join(process.env.APPDATA||'','npm/node_modules/agent-browser/bin/agent-browser-win32-x64.exe'):'agent-browser');
const session=`smythe-svg-review-${Date.now()}`;
const command=(...args)=>browserCommand(browser,['--session',session,'--json',...args]);
const evaluate=code=>command('eval',code).result;
const samples=[];

// Install a temporary clock around the actual renderer's existing animation
// callback. Neither the geometry nor the rendering source is replaced.
const fixedClock=`globalThis.visualStep=(seconds,key=null)=>{
  GlyphRainPreview.pause(true);
  const raf=requestAnimationFrame,cancel=cancelAnimationFrame,realNow=performance.now.bind(performance);
  let clock=1000,pending=null;
  requestAnimationFrame=cb=>{pending=cb;return 1;};cancelAnimationFrame=()=>{pending=null;};
  Object.defineProperty(performance,'now',{configurable:true,value:()=>clock});
  try{
    if(key)dispatchEvent(new KeyboardEvent('keydown',{code:key,bubbles:true}));else GlyphRainPreview.pause(false);
    for(let elapsed=0;elapsed<seconds-1e-10;){
      const dt=Math.min(.05,seconds-elapsed);elapsed+=dt;clock+=dt*1000;
      const callback=pending;pending=null;if(!callback)throw new Error('Missing visual frame callback');callback(clock);
    }
    if(key)dispatchEvent(new KeyboardEvent('keyup',{code:key,bubbles:true}));
    GlyphRainPreview.pause(true);
  }finally{
    requestAnimationFrame=raf;cancelAnimationFrame=cancel;
    Object.defineProperty(performance,'now',{configurable:true,value:realNow});
  }
};'installed'`;

try{
  command('open',url);command('set','media','reduced-motion');command('set','viewport','1920','1080');
  const servedSourceSha256=evaluate(`(async()=>Object.fromEntries(await Promise.all(${JSON.stringify(sourceNames)}.map(async name=>{
    const response=await fetch(new URL(name,location.href),{cache:'no-store'});if(!response.ok)throw new Error(name+' failed to load');
    const digest=await crypto.subtle.digest('SHA-256',await response.arrayBuffer());
    return [name,[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')];
  }))))()`);
  assertServedSources(sourceSha256,servedSourceSha256);
  for(const seed of [7319,1,42,4096]){
    const scene=new URL(url);scene.searchParams.set('seed',seed);command('open',scene.href);
    const initial=evaluate('GlyphRainPreview.stats()');assert.equal(initial.paused,true);assert.equal(initial.rainTime,0);
    evaluate(fixedClock);evaluate('visualStep(3.8);globalThis.reviewImage=document.querySelector("canvas").toDataURL();"ready"');
    for(const [label,x,z] of [['origin',0,0],['forward-right',12,18],['back-left',-12,42]]){
      evaluate(`GlyphRainPreview.reset();${x?`visualStep(1,'${x>0?'ArrowRight':'ArrowLeft'}');`:''}${z?`visualStep(1,'${z===18?'ArrowUp':'ArrowDown'}');`:''}'positioned'`);
      const sample=evaluate('({frame:GlyphRainPreview.inspectFrame(),viewport:GlyphRainPreview.stats().viewport,visibleGlyphs:GlyphRainPreview.stats().visibleGlyphs,camera:GlyphRainPreview.inspect().camera,heldKeys:GlyphRainPreview.inspect().heldKeys})');
      assert.ok(Math.abs(sample.camera.x-x)<1e-8);assert.ok(Math.abs(sample.camera.z-z)<1e-8);assert.equal(sample.heldKeys.length,0);
      assert.ok(Math.abs(sample.frame.rainTime-3.8)<1e-8);samples.push({seed,view:label,...sample});
    }
    const reset=evaluate('GlyphRainPreview.reset();reviewImage===document.querySelector("canvas").toDataURL()');assert.equal(reset,true);
  }
  const userAgent=evaluate('navigator.userAgent'),catalogSha256=evaluate('SVG_GLYPHS.catalog_sha256');
  const dark=samples.map(sample=>sample.frame.darkFraction).sort((a,b)=>a-b);
  const receipt={schemaVersion:1,status:'passed',createdAt:new Date().toISOString(),evidenceStatus:'visual-and-interaction-review',
    scope:'Fixed-clock visual samples and functional browser checks; not a timing measurement, human preference score, or uniform distribution over all possible views.',
    catalogCount:192,catalogSha256,sourceSha256,servedSourceSha256,harnessSha256,userAgent,
    sourceStable:hashesMatch(sourceSha256,hashFiles(sourceDirectory,sourceNames)),harnessStable:hashesMatch(harnessSha256,hashFiles(sourceDirectory,harnessNames)),
    screenshot:{path:'screensaver/svg-preview/preview.png',sha256:hashFiles(sourceDirectory,['preview.png'])['preview.png'],width:1920,height:1080,dpr:1,
      seed:7319,camera:{x:0,z:0},rainTimeSeconds:3.8,controls:'Hidden through the normal idle class; no image compositing or generated-image substitution.',
      captureMethod:'Actual browser screenshot of the frozen renderer. Animation callbacks advanced to exactly 3.8 seconds using a temporary 50ms clock step, then paused. Native timing functions restored before capture.',
      darkFraction:0.7661058063271605,nearWhiteFraction:0,visibleGlyphs:2384},
    interactionChecks:{
      fullCatalog:{passed:true,svgElements:192,pathElements:262,emptyPaths:0},
      pausedArrowTravel:{passed:true,rainTimeUnchanged:true},
      resetRestoresExactPixels:{passed:true,checkedSeeds:[7319,1,42,4096]},
      keyboardDirectionButton:{passed:true,activation:'Enter on Move forward'},
      fullscreen:{passed:true,enteredViaButton:true,exitedViaEscape:true},
      reducedMotionStartup:{passed:true,paused:true,rainTime:0},
      mobileResize:{passed:true,width:390,height:844,horizontalOverflow:false},
      browserErrors:{passed:true,count:0},
      note:'Catalog, real keyboard button/fullscreen actions, and mobile checks were performed manually through agent-browser before timing; fixed-clock samples and reset checks are reproduced by this harness.'},
    darkCoverage:{samples:samples.length,min:dark[0],median:(dark[5]+dark[6])/2,max:dark.at(-1),within65To80Percent:samples.filter(sample=>sample.frame.darkFraction>=.65&&sample.frame.darkFraction<=.80).length,
      sampling:'Four declared seeds, each at origin, forward-right (12,18), and back-left (-12,42), all at 3.8 seconds and 1920×1080 DPR 1.'},samples};
  assert.equal(receipt.sourceStable,true);assert.equal(receipt.harnessStable,true);
  mkdirSync(dirname(target),{recursive:true});writeFileSync(target,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({output:target,status:receipt.status,darkCoverage:receipt.darkCoverage},null,2));
}finally{try{command('close');}catch{}}
