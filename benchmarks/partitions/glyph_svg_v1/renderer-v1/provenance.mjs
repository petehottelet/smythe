import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';

export function hashFiles(directory,names){
  return Object.fromEntries(names.map(name=>[
    name,createHash('sha256').update(readFileSync(join(directory,name))).digest('hex')]));
}

export function hashesMatch(expected,actual){
  return Object.keys(expected).length===Object.keys(actual).length&&
    Object.keys(expected).every(name=>expected[name]===actual[name]);
}

export function assertServedSources(expected,actual){
  if(!hashesMatch(expected,actual))throw new Error('Served preview files do not match the local measurement sources');
}
