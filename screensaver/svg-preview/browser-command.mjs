import {execFileSync} from 'node:child_process';
import {mkdtempSync,openSync,closeSync,readFileSync,unlinkSync,rmdirSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';

export function browserCommand(browser,args){
  const directory=mkdtempSync(join(tmpdir(),'smythe-browser-command-'));
  const stdoutPath=join(directory,'stdout'),stderrPath=join(directory,'stderr');
  const stdout=openSync(stdoutPath,'w'),stderr=openSync(stderrPath,'w');
  try{
    // A newly started Windows browser daemon can inherit capture pipes. Files
    // let the CLI return without waiting for its long-lived child to close them.
    execFileSync(browser,args,{stdio:['ignore',stdout,stderr],timeout:45000,windowsHide:true});
    const raw=readFileSync(stdoutPath,'utf8'),response=JSON.parse(raw);
    if(!response.success)throw new Error(response.error||raw);
    return response.data;
  }catch(error){
    const details=readFileSync(stderrPath,'utf8');
    if(details)error.message+=`\n${details}`;
    throw error;
  }finally{
    closeSync(stdout);closeSync(stderr);
    unlinkSync(stdoutPath);unlinkSync(stderrPath);rmdirSync(directory);
  }
}
