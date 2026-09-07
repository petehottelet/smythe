// Read-only Chrome DevTools metadata, collected outside timing samples.
export function normalizeLaunchArguments(args){
  const ephemeral=new Set(['--user-data-dir','--remote-debugging-port','--crashpad-handler-pid']);
  const result=[];
  for(let index=0;index<args.length;index++){
    const arg=args[index],key=arg.split('=')[0];
    if(ephemeral.has(key)){
      result.push(key+'=<ephemeral>');
      if(!arg.includes('=')&&args[index+1]&&!args[index+1].startsWith('--'))index++;
    }else result.push(arg);
  }
  return result;
}

export async function browserMetadata(endpoint,{fetchImpl=fetch,WebSocketImpl=WebSocket}={}){
  const candidate=typeof endpoint==='string'?endpoint:endpoint?.cdpUrl??endpoint?.url??endpoint?.value;
  const address=new URL(candidate);
  if(!['localhost','127.0.0.1','[::1]'].includes(address.hostname))throw new Error('Metadata requires a local CDP endpoint');
  let wsURL=address.href;
  if(['http:','https:'].includes(address.protocol)){
    const response=await fetchImpl(new URL('/json/version',address),{signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error('CDP version discovery failed');
    wsURL=(await response.json()).webSocketDebuggerUrl;
  }
  const socketURL=new URL(wsURL);
  if(!['ws:','wss:'].includes(socketURL.protocol)||!['localhost','127.0.0.1','[::1]'].includes(socketURL.hostname))throw new Error('Unexpected CDP WebSocket endpoint');
  const socket=new WebSocketImpl(socketURL.href),pending=new Map();let id=0;
  const opened=new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('CDP metadata connection timed out')),5000);
    socket.addEventListener('open',()=>{clearTimeout(timer);resolve();},{once:true});
    socket.addEventListener('error',()=>{clearTimeout(timer);reject(new Error('CDP metadata connection failed'));},{once:true});
  });
  socket.addEventListener('message',event=>{
    const message=JSON.parse(event.data),call=pending.get(message.id);if(!call)return;
    pending.delete(message.id);clearTimeout(call.timer);
    if(message.error)call.reject(new Error(message.error.message));else call.resolve(message.result);
  });
  const request=method=>new Promise((resolve,reject)=>{
    const key=++id,timer=setTimeout(()=>{pending.delete(key);reject(new Error(`CDP ${method} timed out`));},5000);
    pending.set(key,{resolve,reject,timer});socket.send(JSON.stringify({id:key,method}));
  });
  try{
    await opened;
    const [version,commandLine,systemInfo]=await Promise.all([
      request('Browser.getVersion'),request('Browser.getBrowserCommandLine'),request('SystemInfo.getInfo')]);
    const args=commandLine.arguments;
    if(!Array.isArray(args))throw new Error('Browser launch arguments unavailable');
    const headless=args.some(arg=>/^--headless(?:=|$)/.test(arg));
    // Retain actual arguments, redacting authenticated proxy values. Only the
    // three explicitly listed ephemeral fields are ignored during comparison.
    const argumentsSafe=args.map(arg=>/^--proxy-server=/.test(arg)?'--proxy-server=<redacted>':arg);
    return {version,commandLine:{arguments:argumentsSafe,comparisonArguments:normalizeLaunchArguments(argumentsSafe)},systemInfo,headless};
  }finally{
    for(const call of pending.values()){clearTimeout(call.timer);call.reject(new Error('CDP metadata collection closed'));}
    pending.clear();socket.close();
  }
}
