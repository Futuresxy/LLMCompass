// Local report rendering and calculator checks; requires Node 22+ and Chrome.
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, writeFile, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.join(root,'validation');
await mkdir(output,{recursive:true});
const profile=await mkdtemp(path.join(tmpdir(),'llmcompass-flow-cdp-'));
const chrome=spawn(process.env.REPORT_CHROME || 'google-chrome',[
  '--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
  '--no-first-run','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0',
  '--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;
try {
  const url=await new Promise((resolve,reject)=>{
    let log='';const timeout=setTimeout(()=>reject(new Error('Chrome debugging endpoint timeout')),15000);
    chrome.stderr.on('data',chunk=>{log+=chunk;const m=log.match(/DevTools listening on (ws:\/\/\S+)/);if(m){clearTimeout(timeout);resolve(m[1]);}});
    chrome.on('error',reject);chrome.on('exit',code=>{clearTimeout(timeout);reject(new Error('Chrome exited '+code));});
  });
  socket=new WebSocket(url);
  await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
  let next=0;const pending=new Map();
  socket.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}});
  const call=(method,params={},sessionId)=>new Promise((resolve,reject)=>{const id=++next;const timer=setTimeout(()=>{pending.delete(id);reject(new Error(method+' timeout'));},15000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});
  const target=await call('Target.createTarget',{url:'about:blank'});
  const attached=await call('Target.attachToTarget',{targetId:target.targetId,flatten:true});
  const c=(method,params={})=>call(method,params,attached.sessionId);
  const evaluate=async expression=>{const r=await c('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await c('Page.enable');await c('Runtime.enable');
  await c('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await c('Page.navigate',{url:pathToFileURL(path.join(root,'report.html')).href});
  for(let i=0;i<100;i++){if(await evaluate('document.readyState === "complete" && !!window.lastCalculation'))break;await new Promise(r=>setTimeout(r,50));}
  assert.equal(await evaluate('document.readyState'),'complete');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'),true);
  const png=async(name,clip)=>{const r=await c('Page.captureScreenshot',{format:'png',captureBeyondViewport:!!clip,...(clip?{clip}:{})});await writeFile(path.join(output,name),Buffer.from(r.data,'base64'));};
  await png('cover.png');
  for(let i=0;i<2;i++){
    const clip=await evaluate(`(()=>{const r=document.querySelectorAll('figure')[${i}].getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()`);
    await png(i?'mapping.png':'boundaries.png',clip);
  }
  if(await evaluate("!!document.getElementById('hardwarebw')")){
    for(let i=0;i<2;i++){
      const clip=await evaluate(`(()=>{const r=document.querySelectorAll('#hardwarebw .table-wrap')[${i}].getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()`);
      await png(i?'gpu-feed-budget.png':'hardware-bandwidth.png',clip);
    }
  }
  if(await evaluate("!!document.getElementById('microarch')")){
    const clip=await evaluate("(()=>{const r=document.querySelector('#microarch .table-wrap').getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()");
    await png('microarchitecture-metrics.png',clip);
  }
  const evidence=JSON.parse(await readFile(path.join(root,'evidence','derived.json'),'utf8'));
  const checks=[];
  const expectations={base:[16384,16384,4,786432,1048576],k2:[8192,8192,2,262144,524288],n2:[8192,8192,4,786432,1048576],short:[1024,4096,1,0,262144]};
  for(const [name,expected] of Object.entries(expectations)){
    const got=await evaluate(`(()=>{const e=document.getElementById('preset');e.value=${JSON.stringify(name)};e.dispatchEvent(new Event('change'));return window.lastCalculation})()`);
    assert.deepEqual([got.osIssues,got.wsIssues,got.folds,got.wsAccReadBytes,got.wsAccWriteBytes],expected);
    assert.equal(got.osFinalBytes,262144);checks.push({preset:name,...got});
  }
  await evaluate(`document.getElementById('preset').value='base';document.getElementById('preset').dispatchEvent(new Event('change'));`);
  for(const [section,filename] of [['example','comparison.png'],['calculator','calculator.png']]){
    await evaluate(`document.getElementById('${section}').scrollIntoView();new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
    await png(filename);
  }
  await evaluate('window.scrollTo(0,0)');
  const pdf=await c('Page.printToPDF',{printBackground:true,preferCSSPageSize:true});
  await writeFile(path.join(root,'report.pdf'),Buffer.from(pdf.data,'base64'));
  await c('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate('window.scrollTo(0,0)');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'),true);
  await png('mobile.png');
  const result={status:'passed',browser:'local Chrome via CDP',calculator_checks:checks,
    analytical_evidence_status:evidence.status,
    assertions:['Desktop/mobile document does not overflow','Four calculator presets checked','Two vector figures and comparison/table rendered','PDF generated'],
    note:'Browser validation is not hardware validation; visual inspection recorded separately.'};
  await writeFile(path.join(output,'browser.json'),JSON.stringify(result,null,2)+'\n');
  console.log('passed: '+path.join(output,'browser.json'));
} finally {
  if(socket)socket.close();chrome.kill('SIGTERM');
  await new Promise(resolve=>chrome.exitCode!==null?resolve():chrome.once('exit',resolve));
  await rm(profile,{recursive:true,force:true,maxRetries:6,retryDelay:100});
}
