// Optional controls for capabilities explicitly supplied by the renderer host.
export const SETTING_DEFINITIONS=Object.freeze({
  originalMix:{type:'range',label:'Original glyphs',group:'Look',min:0,max:100,step:1,default:10,unit:'%',help:'Share of cells drawn from Smythe’s original catalog.'},
  numColumns:{type:'range',label:'Columns',group:'Look',min:40,max:160,step:1,default:80},
  palette:{type:'select',label:'Palette',group:'Look',options:[]},
  backgroundColor:{type:'color',label:'Background',group:'Look',default:'#000000'},
  cursorColor:{type:'color',label:'Leading glyph',group:'Look',default:'#d8ffa8'},
  fallSpeed:{type:'range',label:'Fall speed',group:'Motion',min:.05,max:2,step:.05,default:.3},
  cycleSpeed:{type:'range',label:'Character changes',group:'Motion',min:0,max:.2,step:.005,default:.03},
  raindropLength:{type:'range',label:'Trail length',group:'Motion',min:.1,max:2,step:.05,default:.75},
  animationSpeed:{type:'range',label:'Animation speed',group:'Motion',min:.1,max:3,step:.1,default:1,unit:'×'},
  bloomStrength:{type:'range',label:'Glow amount',group:'Glow',min:0,max:2,step:.05,default:.7},
  bloomSize:{type:'range',label:'Glow spread',group:'Glow',min:.05,max:1,step:.05,default:.4},
  resolution:{type:'range',label:'Render scale',group:'View',min:.25,max:1,step:.05,default:.75,unit:'×',help:'Lower values reduce the render workload.'},
  flip:{type:'checkbox',label:'Mirror glyphs',group:'View',default:false},
  rotation:{type:'range',label:'Glyph rotation',group:'View',min:-180,max:180,step:1,default:0,unit:'°'},
  slant:{type:'range',label:'Column angle',group:'View',min:-60,max:60,step:1,default:0,unit:'°'},
  autoTravel:{type:'checkbox',label:'Travel automatically',group:'Travel',default:false},
  forwardSpeed:{type:'range',label:'Travel speed',group:'Travel',min:.05,max:3,step:.05,default:.2}
});

export function resolveSettingsSchema(schema){
  if(!Array.isArray(schema))throw new TypeError('Pass the settings supported by this engine as a schema array');
  const keys=new Set();
  return schema.map(entry=>{
    const key=typeof entry==='string'?entry:entry?.key;
    if(typeof key!=='string'||!/^\w+$/.test(key)||['__proto__','constructor','prototype','preset'].includes(key)||keys.has(key))throw new TypeError(`Invalid or duplicate setting key: ${key}`);
    keys.add(key);
    const field={...SETTING_DEFINITIONS[key],...(typeof entry==='object'?entry:{}),key};
    if(field.enabledWhen!==undefined&&typeof field.enabledWhen!=='function')throw new TypeError(`enabledWhen must be a function: ${key}`);
    if(!['range','checkbox','color','select'].includes(field.type)||!field.label)throw new TypeError(`Incomplete setting definition: ${key}`);
    if(field.type==='range'&&(!Number.isFinite(field.min)||!Number.isFinite(field.max)||field.min>=field.max||!Number.isFinite(field.step)||field.step<=0))throw new TypeError(`Invalid range: ${key}`);
    if(field.type==='select'){
      field.options=(field.options??[]).map(option=>typeof option==='string'?{value:option,label:option}:{...option});
      if(!field.options.length||field.options.some(option=>typeof option.value!=='string'||typeof option.label!=='string'))throw new TypeError(`Supply implemented choices for ${key}`);
      field.default??=field.options[0].value;
    }
    field.group??='Settings';
    return field;
  });
}

export const settingEnabled=(field,config)=>field.enabledWhen?Boolean(field.enabledWhen(config)):true;

function valueFor(field,value){
  if(field.type==='range'){
    if(value===null||value===undefined||typeof value==='boolean'||(typeof value==='string'&&!value.trim()))return undefined;
    const number=Number(value);if(!Number.isFinite(number))return undefined;
    const clamped=Math.max(field.min,Math.min(field.max,number));
    return Number(Math.max(field.min,Math.min(field.max,field.min+Math.round((clamped-field.min)/field.step)*field.step)).toFixed(8));
  }
  if(field.type==='checkbox'){
    if(value===true||value===1||value==='true'||value==='1')return true;
    if(value===false||value===0||value==='false'||value==='0')return false;
    return undefined;
  }
  if(field.type==='color')return typeof value==='string'&&/^#[0-9a-f]{6}$/i.test(value)?value.toLowerCase():undefined;
  return field.options.some(option=>option.value===value)?value:undefined;
}

export function normalizeSettings(schema,values={},base={}){
  const fields=resolveSettingsSchema(schema),result={...base,...values};
  for(const field of fields){
    result[field.key]=valueFor(field,values[field.key])??valueFor(field,base[field.key])??valueFor(field,field.default);
    if(result[field.key]===undefined)throw new TypeError(`No valid value or default for ${field.key}`);
  }
  return result;
}

export function parseSettingsUrl(url,schema,base={}, {presets=[]}={}){
  const address=new URL(url),preset=presets.find(item=>item.id===address.searchParams.get('preset'));
  const initial={...base,...preset?.values},values={};
  for(const field of resolveSettingsSchema(schema)){
    if(address.searchParams.has(field.key)){
      const value=valueFor(field,address.searchParams.get(field.key));
      if(value!==undefined)values[field.key]=value;
    }
  }
  return normalizeSettings(schema,values,initial);
}

export function serializeSettingsUrl(url,values,schema,{presetId=null}={}){
  const address=new URL(url),config=normalizeSettings(schema,values);
  for(const field of resolveSettingsSchema(schema))address.searchParams.set(field.key,String(config[field.key]));
  if(presetId)address.searchParams.set('preset',presetId);else address.searchParams.delete('preset');
  return address;
}

let nextSheet=0;
export function mountSettings({container,schema,presets=[],initialConfig={},initialPresetId=null,onApply,onClose=()=>{},syncUrl=true}){
  if(!container?.ownerDocument)throw new TypeError('Pass a DOM container for the settings sheet');
  if(typeof onApply!=='function')throw new TypeError('Pass an onApply callback for the supported renderer settings');
  const fields=resolveSettingsSchema(schema),document=container.ownerDocument,window=document.defaultView;
  if(new Set(presets.map(preset=>preset.id)).size!==presets.length||presets.some(preset=>!preset.id||!preset.label||!preset.values))throw new TypeError('Presets require unique ids, labels, and implemented configuration values');
  const id=`rain-settings-${++nextSheet}`,controls=new Map();
  let applied=normalizeSettings(fields,initialConfig),draft={...applied},previousFocus=null,open=false,busy=false,destroyed=false;
  let appliedPreset=initialPresetId??presets.find(preset=>preset.id===new URL(window.location.href).searchParams.get('preset'))?.id??'';
  if(appliedPreset&&!presets.some(preset=>preset.id===appliedPreset))throw new TypeError('Unknown initial preset');
  const element=(tag,className,text)=>{const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node;};
  const dialog=element('dialog','rain-settings');dialog.setAttribute('aria-labelledby',`${id}-title`);
  const form=element('form','rain-settings__form'),header=element('header','rain-settings__header');
  const title=element('h2','', 'Settings');title.id=`${id}-title`;
  const closeButton=element('button','rain-settings__close','×');closeButton.type='button';closeButton.setAttribute('aria-label','Close settings');
  header.append(title,closeButton);
  const body=element('div','rain-settings__body');
  let presetSelect=null;
  if(presets.length){
    const label=element('label','rain-settings__preset','Preset');presetSelect=element('select');presetSelect.id=`${id}-preset`;label.htmlFor=presetSelect.id;
    const custom=element('option','', 'Current settings');custom.value='';presetSelect.append(custom);
    for(const preset of presets){const option=element('option','',preset.label);option.value=preset.id;presetSelect.append(option);}
    presetSelect.value=appliedPreset;label.append(presetSelect);body.append(label);
    presetSelect.addEventListener('change',()=>{
      const preset=presets.find(item=>item.id===presetSelect.value);if(preset)setValues(preset.values);
    });
  }
  const groups=new Map();
  for(const field of fields){
    if(!groups.has(field.group)){
      const section=element('fieldset','rain-settings__group');section.append(element('legend','',field.group));groups.set(field.group,section);body.append(section);
    }
    const row=element('div',`rain-settings__field rain-settings__field--${field.type}`),label=element('label','rain-settings__label',field.label);
    const input=element(field.type==='select'?'select':'input');input.id=`${id}-${field.key}`;input.name=field.key;label.htmlFor=input.id;
    if(field.type==='select')for(const item of field.options){const option=element('option','',item.label);option.value=item.value;input.append(option);}
    else input.type=field.type;
    if(field.type==='range'){input.min=String(field.min);input.max=String(field.max);input.step=String(field.step);}
    const output=field.type==='range'?element('output','rain-settings__value'):null;
    if(output){output.htmlFor=input.id;output.setAttribute('aria-hidden','true');}
    const line=element('div','rain-settings__line');line.append(label);if(output)line.append(output);
    row.append(line,input);
    if(field.help){const help=element('p','rain-settings__help',field.help);help.id=`${input.id}-help`;input.setAttribute('aria-describedby',help.id);row.append(help);}
    controls.set(field.key,{field,input,output,row});groups.get(field.group).append(row);
    input.addEventListener('input',()=>{
      draft[field.key]=field.type==='checkbox'?input.checked:input.value;
      if(output)output.textContent=`${valueFor(field,input.value)}${field.unit??''}`;
      error.textContent='';refreshEnabled();
    });
  }
  const footer=element('footer','rain-settings__footer'),error=element('p','rain-settings__error');error.setAttribute('role','status');error.setAttribute('aria-live','polite');
  const actions=element('div','rain-settings__actions'),cancelButton=element('button','rain-settings__cancel','Cancel'),applyButton=element('button','rain-settings__apply','Apply');
  cancelButton.type='button';applyButton.type='submit';actions.append(cancelButton,applyButton);footer.append(error,actions);
  form.append(header,body,footer);dialog.append(form);container.append(dialog);

  function read(){return normalizeSettings(fields,draft,applied);}
  function refreshEnabled(){
    const config=read();
    for(const button of form.querySelectorAll('button'))button.disabled=busy;
    if(presetSelect)presetSelect.disabled=busy;
    for(const {field,input,row} of controls.values()){
      const enabled=settingEnabled(field,config);
      input.disabled=busy||!enabled;row.classList.toggle('is-unavailable',!enabled);
    }
  }
  function setValues(values){
    draft=normalizeSettings(fields,values,read());
    for(const {field,input,output} of controls.values()){
      if(field.type==='checkbox')input.checked=draft[field.key];else input.value=String(draft[field.key]);
      if(output)output.textContent=`${draft[field.key]}${field.unit??''}`;
    }
    error.textContent='';refreshEnabled();return read();
  }
  function close(reason='cancel'){
    if(!open||busy)return false;
    open=false;dialog.close();draft={...applied};setValues(applied);if(presetSelect)presetSelect.value=appliedPreset;
    if(previousFocus?.isConnected)previousFocus.focus({preventScroll:true});
    onClose(reason);return true;
  }
  function show(){
    if(destroyed)throw new Error('This settings sheet has been destroyed');if(open)return;
    previousFocus=document.activeElement;setValues(applied);if(presetSelect)presetSelect.value=appliedPreset;
    dialog.showModal();open=true;(presetSelect??controls.values().next().value?.input??closeButton).focus({preventScroll:true});
  }
  closeButton.addEventListener('click',()=>close());cancelButton.addEventListener('click',()=>close());
  dialog.addEventListener('cancel',event=>{event.preventDefault();close();});
  // Let native input keys work, while keeping R, Space and arrows away from the rain.
  dialog.addEventListener('keydown',event=>event.stopPropagation());
  dialog.addEventListener('keyup',event=>event.stopPropagation());
  form.addEventListener('submit',async event=>{
    event.preventDefault();if(busy)return;
    const config=read(),presetId=presetSelect?.value||null,changed=fields.filter(field=>config[field.key]!==applied[field.key]).map(field=>field.key);
    busy=true;form.setAttribute('aria-busy','true');error.textContent='';
    refreshEnabled();
    try{
      await onApply({...config},{changed,presetId});
      applied={...config};appliedPreset=presetId??'';
      if(syncUrl){const address=serializeSettingsUrl(window.location.href,applied,fields,{presetId});window.history.replaceState(window.history.state,'',address);}
      busy=false;close('apply');
    }catch(failure){error.textContent=failure?.message||'These settings could not be applied. Try again.';}
    finally{busy=false;form.removeAttribute('aria-busy');refreshEnabled();}
  });
  setValues(applied);
  return Object.freeze({element:dialog,open:show,close,read,setValues,
    isOpen:()=>open,
    destroy:()=>{if(busy)throw new Error('Wait for settings to finish applying');close('destroy');dialog.remove();destroyed=true;}
  });
}
