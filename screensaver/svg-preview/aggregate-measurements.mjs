// node aggregate-measurements.mjs NEW_OUTPUT_JSON RECEIPT_JSON... (all six runs)
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,dirname,relative} from 'node:path';
import {aggregateReceipts} from './measurement.mjs';

const [output,...inputs]=process.argv.slice(2);
if(!output||inputs.length!==6)throw new Error('Pass a new aggregate output followed by all six receipt paths');
const target=resolve(output);
if(existsSync(target))throw new Error('Refusing to replace an existing aggregate receipt');
if(new Set(inputs.map(path=>resolve(path))).size!==6)throw new Error('Receipt paths must be distinct');
const records=inputs.map(path=>{const bytes=readFileSync(path);return {path:relative(dirname(target),resolve(path)).replaceAll('\\','/'),
  sha256:createHash('sha256').update(bytes).digest('hex'),receipt:JSON.parse(bytes.toString('utf8'))};});
const result=aggregateReceipts(records.map(record=>record.receipt));
result.receipts=records.map(({path,sha256,receipt})=>({path,sha256,scenario:receipt.scenario,status:receipt.status}));
result.createdAt=new Date().toISOString();
mkdirSync(dirname(target),{recursive:true});writeFileSync(target,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({status:result.status,targetPassed:result.targetPassed,hardwareTargetPassed:result.hardwareTargetPassed,output:target}));
if(result.status!=='completed')process.exitCode=1;
