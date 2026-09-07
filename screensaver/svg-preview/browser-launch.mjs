// Chrome exposes Browser.getBrowserCommandLine only with this explicit flag.
// Apply it to the first about:blank launch; existing sessions retain the flag.
export function browserArguments(session,args){
  const launch=args[0]==='open'&&args[1]==='about:blank';
  return ['--session',session,...(launch?['--args','--enable-automation']:[]),'--json',...args];
}
