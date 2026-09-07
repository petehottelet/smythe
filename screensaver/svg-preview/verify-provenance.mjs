import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,unlinkSync,rmdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {hashFiles,hashesMatch,assertServedSources} from './provenance.mjs';

const directory=mkdtempSync(join(tmpdir(),'smythe-renderer-provenance-'));
const names=['harness.mjs','renderer.js'];
try{
  for(const name of names)writeFileSync(join(directory,name),`original ${name}`);
  const initial=hashFiles(directory,names);
  assert.equal(initial['harness.mjs'].length,64);
  assertServedSources(initial,Object.fromEntries(Object.entries(initial).reverse()));
  assert.throws(()=>assertServedSources(initial,{'harness.mjs':initial['harness.mjs']}),/Served preview files/);
  assert.throws(()=>assertServedSources(initial,{...initial,'renderer.js':'wrong localhost source'}),/Served preview files/);
  writeFileSync(join(directory,'harness.mjs'),'modified after measurement began');
  assert.equal(hashesMatch(initial,hashFiles(directory,names)),false);
  writeFileSync(join(directory,'harness.mjs'),'original harness.mjs');
  assert.equal(hashesMatch(initial,hashFiles(directory,names)),true);
  writeFileSync(join(directory,'renderer.js'),'modified renderer');
  assert.equal(hashesMatch(initial,hashFiles(directory,names)),false);
  console.log('Passed: served-source mismatch, absent sources, key-order independence, and harness/renderer mutation detection.');
}finally{for(const name of names)unlinkSync(join(directory,name));rmdirSync(directory);}
