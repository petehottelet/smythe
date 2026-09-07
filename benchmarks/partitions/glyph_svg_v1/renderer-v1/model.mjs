// Independent, elapsed-time simulation. Rendering and input hosts share no state.
export const WORLD = Object.freeze({width:140, height:80, depth:60, near:4,
  far:64, columns:420, rows:70, pitch:80/70, glyphHeight:.85, focal:.9,
  sidewaysSpeed:12, forwardSpeed:18});

export const wrap = (value, span) => ((value % span) + span) % span;
export function hash(value) {
  value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
  value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
  return (value ^ (value >>> 16)) >>> 0;
}
export function createWorld(seed = 7319) {
  let state = seed >>> 0;
  const random = () => { state = (Math.imul(state,1664525)+1013904223)>>>0; return state/4294967296; };
  return {seed, camera:{x:0,z:0}, rainTime:0, columns:Array.from({length:WORLD.columns}, (_,id) => ({
    id, x:random()*WORLD.width-WORLD.width/2, z:random()*WORLD.depth+WORLD.near,
    phase:random()*WORLD.rows, speed:16+random()*14, tail:12+Math.floor(random()*13),
    head:random()>.18, seed:Math.floor(random()*0x7fffffff)
  }))};
}
export function advance(world, dt, {sideways=0,forward=0} = {}, paused=false) {
  if (!Number.isFinite(dt) || dt < 0) throw new RangeError('Elapsed time must be finite and nonnegative');
  sideways = Math.sign(sideways); forward = Math.sign(forward);
  const diagonal = sideways && forward ? Math.SQRT1_2 : 1;
  world.camera.x = wrap(world.camera.x + sideways*WORLD.sidewaysSpeed*dt*diagonal + WORLD.width/2, WORLD.width)-WORLD.width/2;
  world.camera.z = wrap(world.camera.z + forward*WORLD.forwardSpeed*dt*diagonal, WORLD.depth);
  if (!paused) world.rainTime += dt;
}
export function project(world, column, row, width, height) {
  const depth = WORLD.near + wrap(column.z-world.camera.z-WORLD.near, WORLD.depth);
  const scale = height*WORLD.focal/depth;
  const x = wrap(column.x-world.camera.x+WORLD.width/2,WORLD.width)-WORLD.width/2;
  return {depth,scale,x:width/2+x*scale,y:height/2+(row*WORLD.pitch-WORLD.height/2)*scale,
    size:WORLD.glyphHeight*scale};
}
export function glyphAt(world, column, row, count) {
  const identity = hash(column.seed + row*65537);
  const interval = .35 + (identity%550)/1000;
  // Equal elapsed times can land on opposite sides of an integer because of
  // accumulated floating-point error at different display refresh rates.
  const mutation = Math.floor((world.rainTime+(identity%1000)/1000)/interval+1e-9);
  return hash(identity+mutation*1013)%count;
}
export function lightAt(world, column) {
  return wrap(column.phase+world.rainTime*column.speed,WORLD.rows);
}
export function resetCamera(world) { world.camera.x=0; world.camera.z=0; }
