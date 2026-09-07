"""Offline contracts for the licensed REGL adaptation; GPU review is separate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "screensaver" / "svg-preview" / "engine"


def test_engine_source_receipt_binds_every_pass_and_shader():
    manifest = json.loads((ENGINE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "5ba90490453ceceb6812d6b1bc658a99a92411d0"
    assert manifest["license"] == "MIT"
    assert "Permission is hereby granted" in (ENGINE / manifest["license_path"]).read_text()
    assert hashlib.sha256((ENGINE / manifest["license_path"]).read_bytes()).hexdigest() == manifest["license_sha256"]
    files = manifest["files"]
    expected = {path.relative_to(ENGINE).as_posix() for path in ENGINE.glob("*.js")}
    expected |= {path.relative_to(ENGINE).as_posix() for path in ENGINE.glob("shaders/*.glsl")}
    assert {record["path"] for record in files} == expected
    assert len(files) == len(expected) == 15
    for record in files:
        data = (ENGINE / record["path"]).read_bytes()
        assert b"\r" not in data, record["path"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
        assert len(record["source_sha256"]) == 64
        assert len(record["source_git_blob"]) == 40
        assert bool(record["changes"]) == (record["sha256"] != record["source_sha256"])
    untouched = {
        "colorToRGB.js", "shaders/rainPass.intro.frag.glsl",
        "shaders/rainPass.raindrop.frag.glsl", "shaders/rainPass.effect.frag.glsl",
        "shaders/bloomPass.highPass.frag.glsl", "shaders/bloomPass.blur.frag.glsl",
        "shaders/bloomPass.combine.frag.glsl", "shaders/palettePass.frag.glsl",
    }
    assert {record["path"] for record in files if not record["changes"]} == untouched


def test_msdf_coordinates_use_high_precision_in_both_shader_stages():
    for path in ["rainPass.vert.glsl", "rainPass.frag.glsl"]:
        shader = (ENGINE / "shaders" / path).read_text(encoding="utf-8")
        assert "precision highp float;" in shader
        assert "precision lowp float;" not in shader


NODE_CHECKS = r"""
import assert from 'node:assert/strict';
const engine = __ENGINE__;
const {makeSimulationScope, makeDoubleBuffer, loadText} = await import(engine + '/utils.js');
const {default: makeRain} = await import(engine + '/rainPass.js');
const {default: makeBloom} = await import(engine + '/bloomPass.js');

const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
};
let pending = new Map(), fetched = [];
globalThis.fetch = async url => {
  const name = String(url);
  fetched.push(name);
  if (pending.has(name.split('/').at(-1))) await pending.get(name.split('/').at(-1)).promise;
  return {ok: !name.includes('missing'), status: name.includes('missing') ? 404 : 200,
    text: async () => name};
};
globalThis.Image = class { width=128; height=128; async decode() {} };
const matrix = new Proxy({}, {get: (_, key) => (...args) => key === 'create' ? new Float32Array(16) : (args[0] ?? [])});
globalThis.glMatrix = {mat2: matrix, mat4: matrix, vec2: matrix, vec3: matrix};

function makeRegl() {
  let context = {tick: 900, time: 400}, id = 0;
  const calls = [], fbos = [];
  const evaluate = (value, props) => typeof value === 'function' ? value(context, props) : value;
  const regl = spec => (props, callback) => {
    if (typeof props === 'function') { callback = props; props = {}; }
    props ??= {};
    if (callback) {
      const previous = context;
      context = {...context, ...Object.fromEntries(Object.entries(spec.context ?? {}).map(([key, value]) => [key, evaluate(value, props)]))};
      const uniforms = Object.fromEntries(Object.entries(spec.uniforms ?? {}).map(([key, value]) => [key, evaluate(value, props)]));
      try { callback(context, uniforms); } finally { context = previous; }
    } else {
      calls.push({spec, props, context: {...context},
        uniforms: Object.fromEntries(Object.entries(spec.uniforms ?? {}).map(([key, value]) => [key, evaluate(value, props)])),
        framebuffer: evaluate(spec.framebuffer, props)});
    }
  };
  regl.context = key => context => context[key];
  regl.prop = key => (_, props) => props[key];
  regl.texture = props => ({width: props.width ?? props.data?.width ?? 1, height: props.height ?? props.data?.height ?? 1});
  regl.framebuffer = props => {
    const fbo = {id: ++id, props, sizes: [], resize(w, h) { this.sizes.push([w, h]); }};
    fbos.push(fbo);
    return fbo;
  };
  regl.clear = () => {};
  return {regl, calls, fbos};
}

const result = {};
{
  const {regl} = makeRegl(), simulation = {tick: 0, time: 0};
  const scope = makeSimulationScope(regl, simulation), buffer = makeDoubleBuffer(regl, {});
  let first;
  scope((context, uniforms) => {
    assert.equal(uniforms.tick, 0); assert.equal(uniforms.time, 0);
    first = buffer.front(context); assert.notEqual(first, buffer.back(context));
  });
  simulation.tick = 1; simulation.time = 1/60;
  let second;
  scope((context, uniforms) => {
    assert.equal(uniforms.tick, 1); assert.equal(uniforms.time, 1/60);
    second = buffer.front(context); assert.notEqual(second, first);
    assert.equal(buffer.back(context), first);
  });
  for (let i = 0; i < 144; i++) scope(context => assert.equal(buffer.front(context), second));
  result.fixed_clock_and_pause_buffer = true;
}
{
  const {regl, calls} = makeRegl();
  const simulation = {tick: 0, time: 0}, motion = {x: 0, z: 0};
  const scope = makeSimulationScope(regl, simulation);
  const effectGate = deferred(); pending.set('rainPass.effect.frag.glsl', effectGate);
  const pass = makeRain({regl, motion, config: {
    numColumns: 80, density: 1, slant: 0, glyphMSDFURL: 'base.png',
    generatedAtlasURL: 'generated.png', generatedCount: 192, generatedGrid: [16,12],
    generatedPxRange: 16, glyphMix: .1
  }});
  let ready = false; pass.ready.then(() => {ready = true;});
  await new Promise(done => setTimeout(done, 0));
  assert.equal(ready, false, 'rain ready must await effect shader');
  effectGate.resolve(); await pass.ready; pending.clear();
  pass.setSize(1440, 810);
  scope(() => pass.execute(true, true));
  assert.equal(calls.length, 5, 'four compute passes and one render');
  const first = calls.at(-1).uniforms.symbolState;
  assert.equal(calls.at(-1).uniforms.generatedPxRange, 16);
  assert.deepEqual(calls.at(-1).uniforms.generatedGrid, [16,12]);
  assert.equal(calls[2].uniforms.glyphMix, .1);
  calls.length = 0;
  motion.x = 14; motion.z = 6;
  scope(() => pass.execute(true, false));
  assert.equal(calls.length, 1, 'paused camera draw must not execute compute');
  assert.equal(calls[0].uniforms.symbolState, first);
  assert.deepEqual(calls[0].uniforms.cameraOffset, [.2,.1]);
  calls.length = 0; simulation.tick++; simulation.time += 1/60;
  scope(() => pass.execute(false, true));
  assert.equal(calls.length, 4, 'catch-up step computes without rendering');
  calls.length = 0;
  scope(() => pass.execute(true, false));
  assert.notEqual(calls[0].uniforms.symbolState, first, 'next tick must change front buffer');
  result.rain_readiness_and_camera_only_draw = true;
}
{
  const {regl, fbos} = makeRegl(), combineGate = deferred();
  pending.set('bloomPass.combine.frag.glsl', combineGate);
  const pass = makeBloom({regl, config: {bloomStrength:.7,bloomSize:.4}}, {primary:{}});
  let ready = false; pass.ready.then(() => {ready = true;});
  await new Promise(done => setTimeout(done, 0));
  assert.equal(ready, false, 'bloom ready must await combine shader');
  combineGate.resolve(); await pass.ready; pending.clear();
  pass.setSize(1, 1);
  assert.equal(fbos.length, 16);
  assert.ok(fbos.every(fbo => fbo.sizes.at(-1).every(size => size >= 1)));
  pass.setSize(1440, 810);
  assert.deepEqual(fbos.slice(0, 5).map(fbo => fbo.sizes.at(-1)), [[576,324],[288,162],[144,81],[72,40],[36,20]]);
  result.bloom_readiness_and_minimum_dimensions = true;
}
{
  const missing = loadText('shaders/missing.glsl');
  await assert.rejects(missing.loaded, /Shader load failed \(404\)/);
  assert.ok(fetched.every(url => url.startsWith(engine + '/shaders/')));
  result.shader_urls_and_http_failure = true;
}
console.log(JSON.stringify(result));
"""


@pytest.fixture(scope="module")
def engine_receipt():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the offline engine contract checks")
    source = NODE_CHECKS.replace("__ENGINE__", json.dumps(ENGINE.as_uri()))
    completed = subprocess.run(
        [node, "--experimental-default-type=module", "--input-type=module", "-"],
        input=source, capture_output=True, text=True, check=False, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.parametrize("check", [
    "fixed_clock_and_pause_buffer", "rain_readiness_and_camera_only_draw",
    "bloom_readiness_and_minimum_dimensions", "shader_urls_and_http_failure",
])
def test_engine_runtime_contracts(engine_receipt, check):
    assert engine_receipt[check]
