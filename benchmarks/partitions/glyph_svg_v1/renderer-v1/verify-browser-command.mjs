import assert from 'node:assert/strict';
import {browserCommand} from './browser-command.mjs';

const inherited='const {spawn}=require("node:child_process");'+
  'spawn(process.execPath,["-e","setTimeout(()=>{},300)"],{stdio:["ignore",1,2],detached:true,windowsHide:true}).unref();'+
  'console.log(JSON.stringify({success:true,data:{inheritedHandles:true}}));';
assert.deepEqual(browserCommand(process.execPath,['-e',inherited]),{inheritedHandles:true});
assert.throws(()=>browserCommand(process.execPath,['-e','console.log(JSON.stringify({success:false,error:"expected rejection"}))']),/expected rejection/);
assert.throws(()=>browserCommand(process.execPath,['-e','console.error("expected stderr");process.exit(2)']),/expected stderr/);
console.log('Passed: inherited output handles, browser rejection, and command failure diagnostics.');
