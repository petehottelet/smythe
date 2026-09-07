import {WORLD,createWorld,advance,project,glyphAt,lightAt,resetCamera,wrap} from './model.mjs';

const catalog=globalThis.SVG_GLYPHS;
const status=document.getElementById('load-status');
if (!catalog?.glyphs?.length) {
  status.textContent='The generated SVG catalog is not available yet.';
  throw new Error('Expected the original catalog in glyphs.js');
}
const paths=catalog.glyphs.map(glyph=>{
  const path=new Path2D();
  for(const d of glyph.paths??[glyph.path]) path.addPath(new Path2D(d));
  return path;
});
const canvas=document.getElementById('rain'), ctx=canvas.getContext('2d',{alpha:false});
const world=createWorld(Number(new URLSearchParams(location.search).get('seed'))||7319);
const held=new Set(), pointerKeys=new Map();
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
let paused=reduced.matches, frameHandle=null, lastFrame=0, idleTimer=null;
let width=0,height=0,dpr=1,lastVisible=0,lastCpuMs=0,lastDirect=0,frames=0;
let benchmark=null, lastBenchmark=null;
const levels=[16,32,64,128], cache=new Map(), cacheCap=64*1024*1024;
let cacheBytes=0,peakCacheBytes=0,cacheHits=0,cacheMisses=0,evictions=0;

function paintGlyph(context,glyph,cell,head){
  // The broad halo carries exposure; a second tight rim keeps the core sharp.
  context.shadowColor=head?'rgba(155,255,71,1)':'rgba(100,255,48,.85)';
  context.shadowBlur=cell*(head?.27:.20);
  context.fillStyle=head?'#c1ff75':'#89f76e';context.fill(paths[glyph],'nonzero');
  context.shadowColor=head?'rgba(188,255,105,1)':'rgba(122,255,62,.75)';
  context.shadowBlur=cell*(head?.085:.05);context.fill(paths[glyph],'nonzero');
}

function sprite(glyph,size,head) {
  const cell=levels.find(level=>level>=size);
  // Large heads are uncommon; draw their paths directly instead of retaining
  // a second large atlas that would crowd body glyphs out of the memory cap.
  if(!cell||(head&&cell===128)) return null;
  const key=`${glyph}:${cell}:${+head}`;
  if(cache.has(key)){const value=cache.get(key);cache.delete(key);cache.set(key,value);cacheHits++;return value;}
  cacheMisses++;
  const pad=Math.ceil(cell*.30),side=cell+pad*2,bytes=side*side*4;
  while(cacheBytes+bytes>cacheCap&&cache.size){
    const oldest=cache.keys().next().value,entry=cache.get(oldest);
    cache.delete(oldest);cacheBytes-=entry.bytes;entry.image.width=0;entry.image.height=0;evictions++;
  }
  const image=typeof OffscreenCanvas==='function'?new OffscreenCanvas(side,side):Object.assign(document.createElement('canvas'),{width:side,height:side});
  const context=image.getContext('2d');
  context.translate(pad,pad);context.scale(cell/100,cell/100);
  paintGlyph(context,glyph,cell,head);
  const result={image,pad,cell,side,bytes};cache.set(key,result);cacheBytes+=bytes;peakCacheBytes=Math.max(peakCacheBytes,cacheBytes);
  return result;
}

function drawGlyph(glyph,x,y,size,alpha,head) {
  ctx.globalAlpha=alpha;
  const cached=sprite(glyph,size,head);
  if(cached){const ratio=size/cached.cell;ctx.drawImage(cached.image,x-cached.pad*ratio,y-cached.pad*ratio,cached.side*ratio,cached.side*ratio);}
  else{
    lastDirect++;
    ctx.save();ctx.translate(x,y);ctx.scale(size/100,size/100);
    paintGlyph(ctx,glyph,size,head);ctx.restore();
  }
}

const ordered=world.columns.map(column=>({column,depth:0}));
function draw(){
  ctx.globalAlpha=1;ctx.fillStyle='#000';ctx.fillRect(0,0,canvas.width,canvas.height);
  lastVisible=0;lastDirect=0;
  for(const item of ordered)item.depth=WORLD.near+wrap(item.column.z-world.camera.z-WORLD.near,WORLD.depth);
  ordered.sort((a,b)=>b.depth-a.depth);
  for(const {column,depth} of ordered){
    const anchor=project(world,column,0,width,height),size=anchor.size;
    if(anchor.x < -size || anchor.x > width+size)continue;
    const fade=Math.min(1,(depth-WORLD.near)/3,(WORLD.far-depth)/18);
    const light=fade*(.58+.42*(1-(depth-WORLD.near)/WORLD.depth));
    const front=lightAt(world,column),first=Math.floor(front);
    for(let tail=column.tail;tail>=0;tail--){
      const row=wrap(first-tail,WORLD.rows),position=project(world,column,row,width,height);
      if(position.y < -size || position.y > height+size)continue;
      const glyph=glyphAt(world,column,row,paths.length),head=tail===0&&column.head;
      const brightness=Math.max(0,(1-tail/(column.tail+1))*1.08+.08);
      const alpha=Math.min(1,light*Math.pow(brightness,.65)*(head?1.23:1));
      if(alpha<.018)continue;
      drawGlyph(glyph,(position.x-size/2)*dpr,position.y*dpr,size*dpr,alpha,head);lastVisible++;
    }
  }
  ctx.globalAlpha=1;frames++;
}

function direction(){
  const keys=new Set([...held,...pointerKeys.values()]);
  return {sideways:Number(keys.has('ArrowRight'))-Number(keys.has('ArrowLeft')),
    forward:Number(keys.has('ArrowUp'))-Number(keys.has('ArrowDown'))};
}
function moving(){return held.size>0||pointerKeys.size>0;}
function shouldRun(){return !document.hidden&&(!paused||moving()||benchmark);}
function frame(now){
  frameHandle=null;
  const dt=Math.max(0,Math.min(.1,(now-lastFrame)/1000));lastFrame=now;
  const start=performance.now();advance(world,dt,direction(),paused);draw();const end=performance.now();lastCpuMs=end-start;
  collect(now,end,lastCpuMs);
  if(shouldRun())frameHandle=requestAnimationFrame(frame);
}
function schedule(){if(frameHandle===null&&shouldRun()){lastFrame=performance.now();frameHandle=requestAnimationFrame(frame);}}
function stop(){if(frameHandle!==null)cancelAnimationFrame(frameHandle);frameHandle=null;}
function release(){held.clear();pointerKeys.clear();if(paused&&!benchmark)stop();}
function resize(){
  if(benchmark)finishBenchmark('Viewport changed during measurement');
  width=innerWidth;height=innerHeight;dpr=Math.min(2,devicePixelRatio||1);
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);draw();schedule();
}
function pause(value=!paused){
  if(benchmark)finishBenchmark('Playback changed during measurement');
  paused=value;document.getElementById('pause').textContent=paused?'Play':'Pause';
  document.getElementById('pause').setAttribute('aria-pressed',String(paused));
  if(shouldRun())schedule();else stop();
}
function reset(){if(benchmark)finishBenchmark('Camera reset during measurement');release();resetCamera(world);draw();}
async function fullscreen(){
  try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}
  catch{status.textContent='Fullscreen is unavailable in this browser.';}
}
function wake(){
  document.body.classList.remove('idle');clearTimeout(idleTimer);
  idleTimer=setTimeout(()=>{if(!moving()&&!document.querySelector(':focus-visible'))document.body.classList.add('idle');},3200);
}
addEventListener('keydown',event=>{
  wake();
  if(event.altKey||event.ctrlKey||event.metaKey)return;
  if(event.code.startsWith('Arrow')){
    if(benchmark)finishBenchmark('User navigation during measurement');
    held.add(event.code);event.preventDefault();schedule();
  }else if(event.code==='Space'&&!event.repeat&&event.target.tagName!=='BUTTON'&&event.target.tagName!=='A'){event.preventDefault();pause();}
  else if(event.code==='KeyR'&&!event.repeat)reset();
  else if(event.code==='KeyF'&&!event.repeat)fullscreen();
});
addEventListener('keyup',event=>{held.delete(event.code);if(paused&&!moving()&&!benchmark)stop();});
addEventListener('blur',()=>{release();if(benchmark)finishBenchmark('Window lost focus during measurement');});
addEventListener('resize',resize);
document.addEventListener('visibilitychange',()=>{
  release();if(document.hidden){stop();if(benchmark)finishBenchmark('Page hidden during measurement');}else schedule();
});
reduced.addEventListener('change',event=>{if(event.matches)pause(true);});
document.getElementById('pause').addEventListener('click',()=>pause());
document.getElementById('reset').addEventListener('click',reset);
document.getElementById('fullscreen').addEventListener('click',fullscreen);
for(const button of document.querySelectorAll('[data-direction]')){
  button.addEventListener('pointerdown',event=>{
    if(benchmark)finishBenchmark('Pointer navigation during measurement');
    event.preventDefault();button.setPointerCapture(event.pointerId);pointerKeys.set(event.pointerId,button.dataset.direction);wake();schedule();
  });
  const releasePointer=event=>{pointerKeys.delete(event.pointerId);if(paused&&!moving()&&!benchmark)stop();};
  button.addEventListener('pointerup',releasePointer);button.addEventListener('pointercancel',releasePointer);button.addEventListener('lostpointercapture',releasePointer);
  // Enter/Space activation takes one deliberate step; pointer input remains held.
  button.addEventListener('click',event=>{
    if(event.detail!==0)return;
    if(benchmark)finishBenchmark('Keyboard navigation during measurement');
    const key=button.dataset.direction;
    advance(world,.08,{sideways:key==='ArrowRight'?1:key==='ArrowLeft'?-1:0,
      forward:key==='ArrowUp'?1:key==='ArrowDown'?-1:0},true);draw();
  });
}
addEventListener('pointermove',wake,{passive:true});addEventListener('focusin',wake);addEventListener('pointerdown',wake,{passive:true});

function quantile(values,fraction){if(!values.length)return null;const sorted=[...values].sort((a,b)=>a-b);const index=(sorted.length-1)*fraction,lower=Math.floor(index);return sorted[lower]+(sorted[Math.ceil(index)]-sorted[lower])*(index-lower);}
function summary(values){return {samples:values.length,mean:values.length?values.reduce((a,b)=>a+b,0)/values.length:null,median:quantile(values,.5),p95:quantile(values,.95),max:values.length?Math.max(...values):null};}
function stats(){return {frames,visibleGlyphs:lastVisible,directVectorDraws:lastDirect,cpuSubmissionMs:lastCpuMs,
  cache:{entries:cache.size,bytes:cacheBytes,peakBytes:peakCacheBytes,capBytes:cacheCap,hits:cacheHits,misses:cacheMisses,evictions},
  renderBufferBytes:canvas.width*canvas.height*4,camera:{...world.camera},rainTime:world.rainTime,paused,
  viewport:{width,height,dpr,physicalWidth:canvas.width,physicalHeight:canvas.height}};}
function inspectFrame(){
  const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  let dark=0,nearWhite=0;
  for(let index=0;index<pixels.length;index+=4){
    if(pixels[index]<=16&&pixels[index+1]<=16&&pixels[index+2]<=16)dark++;
    if(pixels[index]>=210&&pixels[index+1]>=210&&pixels[index+2]>=210)nearWhite++;
  }
  const count=canvas.width*canvas.height;
  return {pixels:count,darkFraction:dark/count,nearWhiteFraction:nearWhite/count,
    darkDefinition:'All RGB channels <=16',nearWhiteDefinition:'All RGB channels >=210',
    source:'Rain canvas only; controls excluded',camera:{...world.camera},rainTime:world.rainTime};
}
function collect(raf,end,cpu){
  if(!benchmark)return;
  const b=benchmark;if(end<b.sampleStart)return;
  if(b.previousEnd!==null){b.intervals.push(end-b.previousEnd);b.rafIntervals.push(raf-b.previousRaf);}
  b.previousEnd=end;b.previousRaf=raf;b.cpu.push(cpu);b.visible.push(lastVisible);b.direct.push(lastDirect);
  if(b.firstEnd===null)b.firstEnd=end;b.lastEnd=end;
  if(end>=b.firstEnd+b.durationSeconds*1000)finishBenchmark();
}
function finishBenchmark(reason=null){
  if(!benchmark)return null;
  const b=benchmark;benchmark=null;clearTimeout(b.timeout);
  const elapsed=b.firstEnd===null?0:(b.lastEnd-b.firstEnd)/1000;
  const intervals=summary(b.intervals),cpu=summary(b.cpu);
  const result={schemaVersion:1,status:reason?'invalid':'completed',invalidReason:reason,
    createdAt:new Date().toISOString(),catalogVersion:catalog.version,catalogSha256:catalog.catalog_sha256??null,catalogCount:paths.length,seed:world.seed,
    measurement:'Browser animation callbacks and CPU canvas command submission; no GPU-completion or physical-display timing',
    warmupSeconds:b.warmupSeconds,requestedDurationSeconds:b.durationSeconds,measuredDurationSeconds:elapsed,
    frameIntervalMs:intervals,rafIntervalMs:summary(b.rafIntervals),cpuSubmissionMs:cpu,
    averageFramesPerSecond:elapsed>0?b.intervals.length/elapsed:null,
    estimatedMissed60HzIntervals:b.intervals.reduce((total,time)=>total+Math.max(0,Math.round(time/(1000/60))-1),0),
    visibleGlyphs:summary(b.visible),directVectorDraws:summary(b.direct),initialStats:b.initial,finalStats:stats(),
    samples:{frameIntervalMs:b.intervals,rafIntervalMs:b.rafIntervals,cpuSubmissionMs:b.cpu,
      visibleGlyphs:b.visible,directVectorDraws:b.direct},
    userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency??null,deviceMemoryGiB:navigator.deviceMemory??null,
    jsHeapBytes:performance.memory?{start:b.heapStart,end:performance.memory.usedJSHeapSize}:null,
    gate:{referenceResolution:width===1920&&height===1080&&dpr===1,p95FrameIntervalWithin16_7ms:intervals.p95!==null&&intervals.p95<=16.7,
      cacheWithin64MiB:peakCacheBytes<=cacheCap},
    exclusions:['GPU execution time','physical presentation timing','power draw','total process or GPU memory']};
  lastBenchmark=result;paused=b.priorPaused;world.camera.x=b.priorCamera.x;world.camera.z=b.priorCamera.z;world.rainTime=b.priorRainTime;
  document.getElementById('pause').textContent=paused?'Play':'Pause';document.getElementById('pause').setAttribute('aria-pressed',String(paused));
  draw();b.resolve(result);return result;
}
function runBenchmark({warmupSeconds=5,durationSeconds=60}={}){
  if(benchmark)throw new Error('A renderer measurement is already running');
  if(document.hidden)throw new Error('Keep the preview visible while measuring');
  if(!Number.isFinite(warmupSeconds)||warmupSeconds<0||!Number.isFinite(durationSeconds)||durationSeconds<=0||durationSeconds>600)throw new RangeError('Invalid measurement duration');
  return new Promise(resolve=>{
    const priorPaused=paused,priorCamera={...world.camera},priorRainTime=world.rainTime;
    release();paused=false;resetCamera(world);world.rainTime=0;
    const now=performance.now();benchmark={resolve,warmupSeconds,durationSeconds,priorPaused,priorCamera,priorRainTime,
      sampleStart:now+warmupSeconds*1000,previousEnd:null,previousRaf:null,firstEnd:null,lastEnd:null,
      intervals:[],rafIntervals:[],cpu:[],visible:[],direct:[],initial:stats(),heapStart:performance.memory?.usedJSHeapSize??null,
      timeout:setTimeout(()=>finishBenchmark('Measurement did not complete within its time window'),(warmupSeconds+durationSeconds+10)*1000)};
    document.getElementById('pause').textContent='Pause';document.getElementById('pause').setAttribute('aria-pressed','false');schedule();
  });
}

globalThis.GlyphRainPreview=Object.freeze({version:'1',runBenchmark,stats,inspectFrame,pause,reset,
  lastBenchmark:()=>lastBenchmark,benchmarkRunning:()=>Boolean(benchmark),
  inspect:()=>({seed:world.seed,columnIdentities:world.columns.map(column=>column.id),camera:{...world.camera},rainTime:world.rainTime,paused,heldKeys:[...held],pointerCount:pointerKeys.size})});
status.textContent='';resize();pause(paused);wake();
