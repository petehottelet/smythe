import {mountSettings,serializeSettingsUrl} from './settings.mjs';
import {SCHEMA,PRESETS,presetValues,readConfig,engineConfig} from './config.mjs';
import makeRain from './engine/rainPass.js';
import makeBloom from './engine/bloomPass.js';
import makePalette from './engine/palettePass.js';
import {makeSimulationScope,makePipeline} from './engine/utils.js';
import {createFrameGate} from './timing.mjs';

const canvas=document.getElementById('rain'),status=document.getElementById('load-status');
const held=new Set(),pointers=new Map(),reduced=matchMedia('(prefers-reduced-motion: reduce)');
const motion={x:0,z:0},simulation={time:0,tick:0};
const renderGate=createFrameGate();
let values=readConfig(location.href),scene=null,paused=reduced.matches,handle=null,last=0,accumulator=0;
let idleTimer=null,applying=false,frames=0,lastCpu=0,benchmark=null,contextLost=false,submissionSinceDraw=0;
const wrap=(n,span)=>((n%span)+span)%span;

async function createScene(config){
  const regl=globalThis.createREGL({canvas,pixelRatio:1,
    attributes:{alpha:false,antialias:false,preserveDrawingBuffer:true},
    extensions:['OES_texture_half_float','OES_texture_half_float_linear'],
    optionalExtensions:['EXT_color_buffer_half_float','WEBGL_color_buffer_float','OES_standard_derivatives']});
  try{
    const gl=regl._gl;
    if(!gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER,gl.HIGH_FLOAT)?.precision)throw new Error('This renderer requires high-precision fragment shaders');
    const scope=makeSimulationScope(regl,simulation);
    const context={regl,config,motion,simulation,lkg:{enabled:false,tileX:1,tileY:1}};
    const pipeline=makePipeline(context,[makeRain,makeBloom,makePalette]);
    const copy=regl({uniforms:{tex:pipeline.at(-1).outputs.primary}});
    await Promise.all(pipeline.map(pass=>pass.ready));
    const size=()=>{
      const scale=(devicePixelRatio||1)*config.resolution;
      const limit=Math.min(regl.limits.maxTextureSize,regl.limits.maxRenderbufferSize);
      const ratio=Math.min(1,limit/(innerWidth*scale),limit/(innerHeight*scale));
      canvas.width=Math.max(1,Math.ceil(innerWidth*scale*ratio));
      canvas.height=Math.max(1,Math.ceil(innerHeight*scale*ratio));
      regl.poll();for(const pass of pipeline)pass.setSize(canvas.width,canvas.height);
    };
    const draw=(advance=false,render=true)=>{
      regl.poll();scope(()=>{for(const pass of pipeline)pass.execute(render,advance);if(render)copy();});
      if(render)frames++;
    };
    size();draw(true);
    return {regl,config,size,draw,destroy:()=>regl.destroy(),pixels:()=>regl.read(),
      gpu:{renderer:regl.limits.renderer,vendor:regl.limits.vendor,version:regl.limits.version}};
  }catch(error){regl.destroy();throw error;}
}
function direction(){
  const keys=new Set([...held,...pointers.values()]);
  return {x:Number(keys.has('ArrowRight'))-Number(keys.has('ArrowLeft')),
    z:Number(keys.has('ArrowUp'))-Number(keys.has('ArrowDown'))};
}
function move(dt){
  if(values.preset!=='3d')return;
  const vector=direction(),diagonal=vector.x&&vector.z?Math.SQRT1_2:1;
  motion.x=wrap(motion.x+vector.x*12*dt*diagonal+70,140)-70;
  motion.z=wrap(motion.z+vector.z*18*dt*diagonal,60);
}
function moving(){return values.preset==='3d'&&(held.size||pointers.size);}
function active(){return scene&&!applying&&!contextLost&&!document.hidden&&(!paused||moving());}
function schedule(){if(handle===null&&active()){last=performance.now();handle=requestAnimationFrame(frame);}}
function stop(){if(handle!==null)cancelAnimationFrame(handle);handle=null;}
function release(){held.clear();pointers.clear();if(paused)stop();}
function frame(now){
  handle=null;const start=performance.now(),dt=Math.max(0,Math.min(.1,(now-last)/1000));last=now;
  move(dt);
  if(!paused){
    accumulator+=dt;
    while(accumulator>=1/60){simulation.time+=1/60;simulation.tick++;scene.draw(true,false);accumulator-=1/60;}
  }
  const shouldRender=renderGate.shouldRender(now,values.fps);
  if(shouldRender)scene.draw(false);
  submissionSinceDraw+=performance.now()-start;
  if(shouldRender){lastCpu=submissionSinceDraw;submissionSinceDraw=0;collect(now,lastCpu);}
  // Measurement completion may already have restored playback via schedule().
  if(handle===null&&active())handle=requestAnimationFrame(frame);
}
function pause(value=!paused){
  if(benchmark)finishMeasurement('Playback changed during measurement');
  paused=Boolean(value);accumulator=0;
  const button=document.getElementById('pause');button.textContent=paused?'Play':'Pause';button.setAttribute('aria-pressed',String(paused));
  if(active())schedule();else stop();
}
function reset(){
  if(benchmark)finishMeasurement('Viewpoint reset during measurement');
  release();motion.x=0;motion.z=0;scene?.draw(false);
}
async function fullscreen(){
  try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}
  catch{status.textContent='Fullscreen is unavailable in this browser.';}
}
function updateChrome(){
  const is3d=values.preset==='3d';
  document.getElementById('mode').textContent=is3d?'Classic view':'Enter 3D';
  document.getElementById('mode').setAttribute('aria-pressed',String(is3d));
  document.getElementById('mode-label').textContent=is3d?'Glyph Rain / 3D':values.preset==='operator'?'Glyph Rain / Operator':'Glyph Rain';
  document.querySelector('.key-hint').textContent=is3d?'Arrow keys to move':'Enter 3D to explore';
  for(const button of document.querySelectorAll('[data-direction]'))button.disabled=!is3d;
}
function wake(){
  document.body.classList.remove('idle');clearTimeout(idleTimer);
  idleTimer=setTimeout(()=>{if(!moving()&&!sheet?.isOpen()&&!document.querySelector(':focus-visible'))document.body.classList.add('idle');},3200);
}
async function apply(next){
  if(benchmark)finishMeasurement('Settings changed during measurement');
  const prior={...values};applying=true;stop();release();status.textContent='Loading the rain…';
  scene?.destroy();scene=null;simulation.time=0;simulation.tick=0;accumulator=0;motion.x=0;motion.z=0;
  try{scene=await createScene(engineConfig(next));values={...next};updateChrome();status.textContent='';}
  catch(error){
    status.textContent='The new settings could not be loaded.';
    try{scene=await createScene(engineConfig(prior));values=prior;}catch{status.textContent='WebGL rendering is unavailable. Open the glyph catalog to view the SVGs.';}
    throw error;
  }finally{applying=false;schedule();}
}
const sheet=mountSettings({container:document.getElementById('settings-root'),schema:SCHEMA,presets:PRESETS,
  initialConfig:values,initialPresetId:values.preset,
  onApply:async(next,{presetId})=>apply({...next,preset:presetId||values.preset}),onClose:()=>wake()});
function openSettings(){if(benchmark)finishMeasurement('Settings opened during measurement');release();wake();sheet.open();}
function editable(target){return target.closest?.('dialog,input,select,textarea,[contenteditable]');}
addEventListener('keydown',event=>{
  if(editable(event.target)||event.altKey||event.ctrlKey||event.metaKey)return;wake();
  if(event.code.startsWith('Arrow')){
    if(values.preset!=='3d')return;
    if(benchmark)finishMeasurement('Navigation during measurement');
    held.add(event.code);event.preventDefault();schedule();
  }else if(event.code==='Space'&&!event.repeat&&!event.target.closest?.('button,a')){event.preventDefault();pause();}
  else if(event.code==='KeyR'&&!event.repeat)reset();
  else if(event.code==='KeyF'&&!event.repeat)fullscreen();
  else if(event.code==='KeyS'&&!event.repeat){event.preventDefault();openSettings();}
});
addEventListener('keyup',event=>{held.delete(event.code);if(paused&&!moving())stop();});
addEventListener('blur',()=>{release();if(benchmark)finishMeasurement('Window lost focus');});
addEventListener('resize',()=>{if(benchmark)finishMeasurement('Viewport changed');if(scene){scene.size();scene.draw(false);}schedule();});
addEventListener('pointermove',wake,{passive:true});addEventListener('pointerdown',wake,{passive:true});addEventListener('focusin',wake);
document.addEventListener('visibilitychange',()=>{release();if(document.hidden){stop();if(benchmark)finishMeasurement('Page hidden');}else schedule();});
reduced.addEventListener('change',event=>{if(event.matches)pause(true);});
canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();contextLost=true;stop();status.textContent='Graphics context lost. Reload to resume the rain.';if(benchmark)finishMeasurement('WebGL context lost');});
document.getElementById('pause').addEventListener('click',()=>pause());
document.getElementById('reset').addEventListener('click',reset);
document.getElementById('fullscreen').addEventListener('click',fullscreen);
document.getElementById('settings').addEventListener('click',openSettings);
document.getElementById('mode').addEventListener('click',()=>{
  const next={...presetValues(values.preset==='3d'?'classic':'3d'),originalMix:values.originalMix};
  location.assign(serializeSettingsUrl(location.href,next,SCHEMA,{presetId:next.preset}));
});
for(const button of document.querySelectorAll('[data-direction]')){
  button.addEventListener('pointerdown',event=>{event.preventDefault();if(benchmark)finishMeasurement('Navigation during measurement');button.setPointerCapture(event.pointerId);pointers.set(event.pointerId,button.dataset.direction);wake();schedule();});
  const up=event=>{pointers.delete(event.pointerId);if(paused&&!moving())stop();};
  button.addEventListener('pointerup',up);button.addEventListener('pointercancel',up);button.addEventListener('lostpointercapture',up);
  button.addEventListener('click',event=>{if(event.detail!==0)return;if(benchmark)finishMeasurement('Keyboard navigation during measurement');held.add(button.dataset.direction);move(.08);held.clear();scene?.draw(false);});
}
function stats(){return {version:'2',renderer:'REGL adapted from m8e/matrix-rain',frames,cpuSubmissionMs:lastCpu,
  camera:{...motion},rainTime:simulation.time,simulationTick:simulation.tick,paused,config:{...values},
  catalogCount:192,referenceGlyphCount:56,referenceSequenceLength:57,totalVisibleShapes:248,
  glyphMix:values.originalMix/100,viewport:{width:innerWidth,height:innerHeight,dpr:devicePixelRatio||1,renderScale:values.resolution,
  physicalWidth:canvas.width,physicalHeight:canvas.height},gpu:scene?.gpu??null};}
function inspectFrame(){
  const pixels=scene.pixels();let dark=0,white=0;
  for(let i=0;i<pixels.length;i+=4){if(pixels[i]<=16&&pixels[i+1]<=16&&pixels[i+2]<=16)dark++;if(pixels[i]>=210&&pixels[i+1]>=210&&pixels[i+2]>=210)white++;}
  return {pixels:pixels.length/4,darkFraction:dark/(pixels.length/4),nearWhiteFraction:white/(pixels.length/4),source:'WebGL drawing buffer, controls excluded',...stats()};
}
const summary=items=>{const a=[...items].sort((a,b)=>a-b);const q=f=>{if(!a.length)return null;const i=(a.length-1)*f,l=Math.floor(i);return a[l]+(a[Math.ceil(i)]-a[l])*(i-l);};return {samples:a.length,mean:a.length?a.reduce((a,b)=>a+b,0)/a.length:null,median:q(.5),p95:q(.95),max:a.at(-1)??null};};
function collect(now,cpu){
  if(!benchmark||now<benchmark.sampleStart)return;const b=benchmark;
  if(b.first===null){b.first=now;b.firstState={time:simulation.time,tick:simulation.tick};}
  if(b.last!==null)b.intervals.push(now-b.last);b.last=now;b.lastState={time:simulation.time,tick:simulation.tick};b.cpu.push(cpu);
  if(now-b.first>=b.durationSeconds*1000)finishMeasurement();
}
function finishMeasurement(reason=null){
  if(!benchmark)return;const b=benchmark;benchmark=null;clearTimeout(b.timeout);
  const seconds=b.first===null?0:(b.last-b.first)/1000;
  const result={schemaVersion:2,status:reason?'invalid':'completed',invalidReason:reason,evidenceStatus:'diagnostic',
    measurement:'Browser draw callback intervals and CPU WebGL command submission accumulated across all simulation callbacks between draws; not GPU completion or display presentation',
    createdAt:new Date().toISOString(),warmupSeconds:b.warmupSeconds,requestedDurationSeconds:b.durationSeconds,measuredDurationSeconds:seconds,
    catalogCount:192,catalogSha256:globalThis.SVG_GLYPHS.catalog_sha256,referenceGlyphCount:56,
    frameIntervalMs:summary(b.intervals),cpuSubmissionMs:summary(b.cpu),averageFramesPerSecond:seconds?b.intervals.length/seconds:null,
    samples:{frameIntervalMs:b.intervals,cpuSubmissionMs:b.cpu},sampleBoundary:{first:b.firstState??null,last:b.lastState??null},
    initialStats:b.initial,finalStats:stats(),userAgent:navigator.userAgent,
    exclusions:['GPU execution time','physical display timing','power','total process/GPU memory','generation time']};
  pause(b.priorPaused);b.resolve(result);
}
function runBenchmark({warmupSeconds=5,durationSeconds=60}={}){
  if(benchmark||applying||contextLost||sheet.isOpen()||document.hidden)throw new Error('Keep the visible rain available with settings closed');
  if(!Number.isFinite(warmupSeconds)||warmupSeconds<0||!Number.isFinite(durationSeconds)||durationSeconds<=0||durationSeconds>600)throw new RangeError('Invalid measurement duration');
  const priorPaused=paused;pause(false);release();submissionSinceDraw=0;
  return new Promise(resolve=>{benchmark={resolve,priorPaused,warmupSeconds,durationSeconds,sampleStart:performance.now()+warmupSeconds*1000,
    first:null,last:null,intervals:[],cpu:[],initial:stats(),timeout:setTimeout(()=>finishMeasurement('Measurement timed out'),(warmupSeconds+durationSeconds+10)*1000)};schedule();});
}
try{
  await apply(values);pause(paused);wake();
  globalThis.GlyphRainPreview=Object.freeze({version:'2',stats,inspectFrame,pause,reset,runBenchmark,
    benchmarkRunning:()=>Boolean(benchmark),inspect:()=>({...stats(),heldKeys:[...held],pointerCount:pointers.size}),
    // Deliberate fixed steps for visual/interaction checks; never timing evidence.
    stepForReview:(steps=1)=>{if(!paused||benchmark||!Number.isInteger(steps)||steps<0||steps>3600)throw new Error('Pause and pass 0–3600 integer steps');for(let i=0;i<steps;i++){simulation.tick++;simulation.time+=1/60;scene.draw(true,false);}scene.draw(false);},
    moveForReview:(x,z)=>{if(!paused||!Number.isFinite(x)||!Number.isFinite(z))throw new Error('Pause and pass finite coordinates');motion.x=wrap(x+70,140)-70;motion.z=wrap(z,60);scene.draw(false);}
  });
}catch(error){console.error(error);status.textContent=`The rain could not start: ${error.message}. You can still view the glyph catalog.`;}
