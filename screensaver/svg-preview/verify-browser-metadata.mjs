import assert from 'node:assert/strict';
import {browserMetadata,normalizeLaunchArguments} from './browser-metadata.mjs';
import {browserArguments} from './browser-launch.mjs';

assert.deepEqual(browserArguments('fresh-session',['open','about:blank']),
  ['--session','fresh-session','--args','--enable-automation','--json','open','about:blank']);
for(const args of [['open','http://localhost/'],['get','cdp-url'],['eval','1'],['close']]){
  assert.deepEqual(browserArguments('fresh-session',args),['--session','fresh-session','--json',...args]);
}

let lastSocket,rejectMethod=null;
class Socket extends EventTarget{
  constructor(){super();lastSocket=this;this.closed=false;queueMicrotask(()=>this.dispatchEvent(new Event('open')));}
  send(text){
    const {id,method}=JSON.parse(text);
    const results={'Browser.getVersion':{product:'Chrome/fixture',revision:'revision'},
      'Browser.getBrowserCommandLine':{arguments:['chrome','--headless=new','--use-angle=d3d11','--user-data-dir=C:/private/profile','--remote-debugging-port=42133','--proxy-server=https://user:secret@host']},
      'SystemInfo.getInfo':{gpu:{devices:[{deviceString:'GPU fixture'}]}}};
    const reply=method===rejectMethod?{id,error:{message:'fixture metadata denied'}}:{id,result:results[method]};
    queueMicrotask(()=>this.dispatchEvent(new MessageEvent('message',{data:JSON.stringify(reply)})));
  }
  close(){this.closed=true;}
}
const metadata=await browserMetadata({cdpUrl:'ws://127.0.0.1:1234/devtools/browser/example'},{WebSocketImpl:Socket});
assert.equal(metadata.headless,true);assert.equal(metadata.version.product,'Chrome/fixture');
assert.ok(metadata.commandLine.arguments.includes('--use-angle=d3d11'));
assert.ok(metadata.commandLine.arguments.includes('--user-data-dir=C:/private/profile'));
assert.ok(metadata.commandLine.comparisonArguments.includes('--user-data-dir=<ephemeral>'));
assert.ok(metadata.commandLine.comparisonArguments.includes('--remote-debugging-port=<ephemeral>'));
assert.ok(!JSON.stringify(metadata).includes('secret'));
assert.deepEqual(normalizeLaunchArguments(['chrome','--crashpad-handler-pid=42','--user-data-dir','new-profile','--remote-debugging-pipe']),
  ['chrome','--crashpad-handler-pid=<ephemeral>','--user-data-dir=<ephemeral>','--remote-debugging-pipe']);
assert.equal(lastSocket.closed,true);
await browserMetadata('http://localhost:1234',{WebSocketImpl:Socket,fetchImpl:async url=>{
  assert.equal(url.href,'http://localhost:1234/json/version');
  return {ok:true,json:async()=>({webSocketDebuggerUrl:'ws://localhost:1234/devtools/browser/example'})};
}});
rejectMethod='Browser.getBrowserCommandLine';
await assert.rejects(()=>browserMetadata('ws://localhost:1234/devtools/browser/example',{WebSocketImpl:Socket}),/metadata denied/);
assert.equal(lastSocket.closed,true);
await assert.rejects(()=>browserMetadata('ws://example.com/devtools/browser/example',{WebSocketImpl:Socket}),/local CDP/);
await assert.rejects(()=>browserMetadata('http://localhost:1234',{WebSocketImpl:Socket,
  fetchImpl:async()=>({ok:true,json:async()=>({webSocketDebuggerUrl:'ws://example.com/devtools/browser/example'})})}),/Unexpected CDP/);
console.log('Passed: actual CDP version/arguments/devices, ephemeral/redacted values, discovery, error cleanup, and nonlocal rejection.');
