import upstreamConfig from './engine/upstream-config.mjs';
import {normalizeSettings,parseSettingsUrl,resolveSettingsSchema} from './settings.mjs';

export const PRESET_IDS=['classic','operator','3d'];
// Hue follows the user's reference screenshot; the original preset is retained.
export const MATRIX_GREEN={hue:137/360,saturation:.8,cursor:'#a2ffd8'};
const hslToRgb=({space,values})=>{
  if(space==='rgb')return values;
  const [h,s,l]=values,a=s*Math.min(l,1-l);
  return [0,8,4].map(n=>{const k=(n+h*12)%12;return l-a*Math.max(-1,Math.min(k-3,9-k,1));});
};
const hex=color=>'#'+hslToRgb(color).map(value=>Math.round(value*255).toString(16).padStart(2,'0')).join('');
const rgb=value=>({space:'rgb',values:[1,3,5].map(at=>parseInt(value.slice(at,at+2),16)/255)});

export const SCHEMA=resolveSettingsSchema([
  'originalMix','numColumns',
  {key:'palette',label:'Body palette',options:[{value:'matrix',label:'Matrix green'},{value:'reference',label:'Reference colors'},{value:'monochrome',label:'White'},{value:'amber',label:'Amber'}]},
  'backgroundColor','cursorColor','fallSpeed','cycleSpeed','raindropLength','animationSpeed',
  'bloomStrength',{key:'bloomSize',min:0},
  {key:'cursorIntensity',type:'range',label:'Leading glyph brightness',group:'Glow',min:0,max:4,step:.1,default:2},
  'resolution','flip',{key:'rotation',step:90},'slant',
  {key:'skipIntro',type:'checkbox',label:'Start with a full rain field',group:'Motion',default:true},
  {key:'autoTravel',group:'3D travel',help:'Applies in the 3D preset.',enabledWhen:v=>v.preset==='3d'},
  {key:'forwardSpeed',group:'3D travel',min:0,default:.25,enabledWhen:v=>v.preset==='3d'},
  {key:'density',type:'range',label:'3D density',group:'3D travel',min:.25,max:2,step:.25,default:1,enabledWhen:v=>v.preset==='3d'},
  {key:'fps',type:'range',label:'Frame rate limit',group:'View',min:15,max:60,step:15,default:60}
]);

export function presetValues(id='classic'){
  const preset=PRESET_IDS.includes(id)?id:'classic',config=upstreamConfig({version:preset});
  return normalizeSettings(SCHEMA,{
    preset,originalMix:10,numColumns:config.numColumns,palette:'matrix',
    backgroundColor:hex(config.backgroundColor),cursorColor:MATRIX_GREEN.cursor,
    fallSpeed:config.fallSpeed,cycleSpeed:config.cycleSpeed,raindropLength:config.raindropLength,
    animationSpeed:config.animationSpeed,bloomStrength:config.bloomStrength,bloomSize:config.bloomSize,
    cursorIntensity:config.cursorIntensity,resolution:config.resolution,flip:config.glyphFlip,
    rotation:config.glyphRotation,slant:config.slant*180/Math.PI,autoTravel:config.volumetric,
    forwardSpeed:config.forwardSpeed,density:config.density,skipIntro:config.skipIntro,fps:config.fps
  });
}
export const PRESETS=PRESET_IDS.map(id=>({id,label:id==='3d'?'3D':id==='operator'?'Operator':'Classic',values:presetValues(id)}));

export function readConfig(url){
  const address=new URL(url),q=address.searchParams;
  const alias={'1999':'operator','2003':'classic',throwback:'operator'};
  let id=q.get('preset')??q.get('version')??'classic';id=alias[id]??id;
  if(!q.has('preset')&&q.get('volumetric')==='true')id='3d';
  if(!PRESET_IDS.includes(id))id='classic';
  for(const [oldKey,newKey] of Object.entries({width:'numColumns',dropLength:'raindropLength',angle:'slant',glyphFlip:'flip',glyphRotation:'rotation'})){
    if(q.has(oldKey)&&!q.has(newKey))q.set(newKey,q.get(oldKey));
  }
  const values=parseSettingsUrl(address.href,SCHEMA,presetValues(id),{presets:PRESETS});
  if(values.palette==='reference'&&!q.has('cursorColor'))values.cursorColor=hex(upstreamConfig({version:id}).cursorColor);
  values.preset=id;
  return values;
}

export function engineConfig(values){
  const id=PRESET_IDS.includes(values.preset)?values.preset:'classic';
  const config=upstreamConfig({version:id}),base=presetValues(id),v=normalizeSettings(SCHEMA,values,base);
  for(const key of ['numColumns','fallSpeed','cycleSpeed','raindropLength','animationSpeed','bloomStrength','bloomSize','cursorIntensity','resolution','forwardSpeed','density','skipIntro','fps'])config[key]=v[key];
  Object.assign(config,{
    glyphMSDFURL:new URL('./reference/matrixcode_msdf.png',import.meta.url).href,
    generatedAtlasURL:new URL('./generated-sdf.png',import.meta.url).href,
    generatedGrid:[16,12],generatedCount:192,generatedPxRange:16,glyphMix:v.originalMix/100,
    glyphFlip:v.flip,glyphRotation:v.rotation,slant:v.slant*Math.PI/180,
    volumetric:id==='3d',forwardSpeed:v.autoTravel?v.forwardSpeed:0,
    backgroundColor:v.backgroundColor===base.backgroundColor?config.backgroundColor:rgb(v.backgroundColor),
    cursorColor:v.cursorColor===hex(config.cursorColor)?config.cursorColor:rgb(v.cursorColor)
  });
  if(v.palette!=='reference'){
    const hue=v.palette==='matrix'?MATRIX_GREEN.hue:v.palette==='amber'?.12:0;
    const saturation=v.palette==='matrix'?MATRIX_GREEN.saturation:v.palette==='amber'?.9:0;
    // Preserve each preset's exposure ramp; only its chromatic grade changes.
    config.palette=config.palette.map(entry=>({color:{space:'hsl',values:[hue,saturation,entry.color.values[2]]},at:entry.at}));
  }
  return config;
}
