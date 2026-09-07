// node metadata-preflight.mjs NEW_RECEIPT_JSON
// Launch metadata and a 2x2 context only; no rain scene, warmup, or timing sample.
import {existsSync,mkdirSync,writeFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {resolve,dirname,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {browserCommand} from './browser-command.mjs';
import {browserArguments} from './browser-launch.mjs';
import {browserMetadata} from './browser-metadata.mjs';
import {classifyBackend,HARNESS_FILES} from './measurement.mjs';
import {hashFiles} from './provenance.mjs';

const [output]=process.argv.slice(2);
if(!output)throw new Error('Pass a new metadata-preflight receipt path');
const target=resolve(output);if(existsSync(target))throw new Error('Refusing to replace an existing metadata receipt');
const directory=dirname(fileURLToPath(import.meta.url)),session=`smythe-metadata-${Date.now()}-${process.pid}`;
const browser=process.env.AGENT_BROWSER_BINARY||(process.platform==='win32'
  ?join(process.env.APPDATA||'','npm/node_modules/agent-browser/bin/agent-browser-win32-x64.exe'):'agent-browser');
const command=(...args)=>browserCommand(browser,browserArguments(session,args));
const receipt={schemaVersion:1,status:'failed',scope:'Metadata-only preflight; blank page and 2x2 WebGL context; no rain or timing sample',
  evidenceStatus:'metadata-only',createdAt:new Date().toISOString(),
  harnessSha256:hashFiles(directory,HARNESS_FILES),preflightSha256:hashFiles(directory,['metadata-preflight.mjs'])['metadata-preflight.mjs']};
try{
  receipt.browserToolVersion=execFileSync(browser,['--version'],{encoding:'utf8',timeout:10000,windowsHide:true}).trim();
  command('open','about:blank');
  receipt.browser=await browserMetadata(command('get','cdp-url'));
  receipt.browser.gl=command('eval',`(()=>{
    const canvas=document.createElement('canvas');canvas.width=2;canvas.height=2;
    const gl=canvas.getContext('webgl',{alpha:false,antialias:false,preserveDrawingBuffer:true});
    if(!gl)throw new Error('WebGL unavailable');const debug=gl.getExtension('WEBGL_debug_renderer_info');
    return {renderer:gl.getParameter(gl.RENDERER),vendor:gl.getParameter(gl.VENDOR),version:gl.getParameter(gl.VERSION),
      unmaskedRenderer:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):null,
      unmaskedVendor:debug?gl.getParameter(debug.UNMASKED_VENDOR_WEBGL):null,
      extensions:gl.getSupportedExtensions().sort(),contextAttributes:gl.getContextAttributes()};
  })()`).result;
  if(!receipt.browser.headless||!receipt.browser.commandLine.arguments.includes('--enable-automation'))throw new Error('Expected headless browser with explicit automation flag');
  receipt.backend=classifyBackend(receipt.browser);receipt.status='passed';
}catch(error){receipt.failure={name:error.name,message:String(error.message).slice(0,4000)};}
finally{
  try{
    mkdirSync(dirname(target),{recursive:true});writeFileSync(target,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  }finally{try{command('close');}catch{}}
}
console.log(JSON.stringify({status:receipt.status,backend:receipt.backend,output:target,failure:receipt.failure},null,2));
if(receipt.status!=='passed')process.exitCode=1;
