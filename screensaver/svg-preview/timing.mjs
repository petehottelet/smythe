// Quantize animation callbacks to a requested drawing cadence. Each time slot
// produces at most one draw; missed slots are skipped after a long callback.
// Nearest-slot selection tolerates callback jitter without halving the rate.
export function createFrameGate(){
  let origin=0,lastTime=null,lastSlot=-1,rate=null;
  return {
    shouldRender(now,fps){
      if(!Number.isFinite(now)||!Number.isFinite(fps)||fps<=0)throw new RangeError('Pass finite time and a positive frame rate');
      if(lastTime===null||now<lastTime||fps!==rate){origin=now;lastSlot=-1;rate=fps;}
      lastTime=now;
      const slot=Math.floor((now-origin)*fps/1000+.5);
      if(slot<=lastSlot)return false;
      lastSlot=slot;
      return true;
    },
    reset(){lastTime=null;lastSlot=-1;rate=null;}
  };
}
