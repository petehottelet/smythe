import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtempSync,readFileSync,writeFileSync,unlinkSync,rmdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

const directory=mkdtempSync(join(tmpdir(),'smythe-measure-guard-'));
const output=join(directory,'receipt.json'), original='{"historical":true}\n';
try{
  writeFileSync(output,original);
  for(const helper of ['measure.mjs','soak.mjs']){
  const result=spawnSync(process.execPath,[fileURLToPath(new URL(`./${helper}`,import.meta.url)),
    'http://127.0.0.1:8768/screensaver/svg-preview/',output],{
    encoding:'utf8',timeout:10000,windowsHide:true,
    env:{...process.env,AGENT_BROWSER_BINARY:join(directory,'browser-must-not-run')}
  });
  assert.notEqual(result.status,0);
  assert.match(result.stderr,/Refusing to replace an existing (measurement )?receipt/);
  assert.equal(readFileSync(output,'utf8'),original);
  }
  console.log('Passed: both helpers reject existing receipts before browser launch and preserve them byte-for-byte.');
}finally{unlinkSync(output);rmdirSync(directory);}
