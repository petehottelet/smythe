import assert from 'node:assert/strict';
import upstream from './engine/upstream-config.mjs';
import {MATRIX_GREEN,PRESET_IDS,presetValues,readConfig,engineConfig} from './config.mjs';
for(const id of PRESET_IDS){
  const base=upstream({version:id}),values=readConfig(`https://example.test/?preset=${id}&palette=reference`),result=engineConfig(values);
  for(const key of ['numColumns','fallSpeed','cycleSpeed','raindropLength','animationSpeed','bloomSize','bloomStrength','resolution','forwardSpeed','volumetric','palette','cursorColor'])assert.deepEqual(result[key],key==='forwardSpeed'&&id!=='3d'?0:base[key],`${id}: ${key}`);
  assert.equal(result.glyphMix,.1);assert.equal(result.generatedCount,192);
  const green=engineConfig(presetValues(id));
  assert.deepEqual(green.palette.map(entry=>entry.at),base.palette.map(entry=>entry.at));
  assert.deepEqual(green.palette.map(entry=>entry.color.values[2]),base.palette.map(entry=>entry.color.values[2]));
  assert.ok(green.palette.every(entry=>entry.color.values[0]===MATRIX_GREEN.hue&&entry.color.values[1]===.8));
  assert.deepEqual(green.cursorColor,{space:'rgb',values:[162/255,1,216/255]});
}
for(const value of [0,10,100])assert.equal(engineConfig(readConfig(`https://example.test/?originalMix=${value}`)).glyphMix,value/100);
const malformed=readConfig('https://example.test/?originalMix=NaN&resolution=0&fps=0&numColumns=Infinity');
assert.equal(malformed.originalMix,10);assert.equal(malformed.resolution,.25);assert.equal(malformed.fps,15);assert.equal(malformed.numColumns,80);
const alias=readConfig('https://example.test/?version=1999&width=120&glyphFlip=true&angle=30');
assert.equal(alias.preset,'operator');assert.equal(alias.numColumns,120);assert.equal(alias.flip,true);assert.equal(alias.slant,30);
const applied=readConfig('https://example.test/?preset=3d&originalMix=37&autoTravel=false');
assert.equal(engineConfig(applied).volumetric,true);assert.equal(engineConfig(applied).forwardSpeed,0);
assert.equal(readConfig('https://example.test/?preset=classic&volumetric=true').preset,'classic');
console.log('Configuration: reference presets, mix endpoints, aliases, URL validation and paused automatic travel passed.');
