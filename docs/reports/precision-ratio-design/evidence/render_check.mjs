// Offline report validation. Node 22+ and local Chrome; no network page loads.
import {spawn} from 'node:child_process';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output=path.join(root,'validation');await mkdir(output,{recursive:true});
const profile=await mkdtemp(path.join(tmpdir(),'precision-report-cdp-'));
const chrome=spawn('google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;
try{
 const url=await new Promise((resolve,reject)=>{let log='';const timeout=setTimeout(()=>reject(new Error('Chrome timeout')),15000);chrome.stderr.on('data',x=>{log+=x;const m=log.match(/DevTools listening on (ws:\/\/\S+)/);if(m){clearTimeout(timeout);resolve(m[1]);}});chrome.on('error',reject);chrome.on('exit',code=>{clearTimeout(timeout);reject(new Error('Chrome exited '+code));});});
 socket=new WebSocket(url);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
 let next=0;const pending=new Map();socket.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}});
 const call=(method,params={},sessionId)=>new Promise((resolve,reject)=>{const id=++next;const timer=setTimeout(()=>{pending.delete(id);reject(new Error(method+' timeout'));},15000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});
 const target=await call('Target.createTarget',{url:'about:blank'});const attached=await call('Target.attachToTarget',{targetId:target.targetId,flatten:true});const c=(method,params={})=>call(method,params,attached.sessionId);
 const evaluate=async expression=>{const r=await c('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await c('Page.enable');await c('Runtime.enable');await c('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});await c('Page.navigate',{url:pathToFileURL(path.join(root,'report.html')).href});
 for(let i=0;i<100;i++){if(await evaluate('document.readyState==="complete" && document.querySelectorAll("#results tbody tr").length===4'))break;await new Promise(r=>setTimeout(r,50));}
 assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
 const png=async name=>{const r=await c('Page.captureScreenshot',{format:'png'});await writeFile(path.join(output,name),Buffer.from(r.data,'base64'));};await png('cover.png');
 const checks=await evaluate(`(()=>{const result=[];for(const format of [...f.options].map(o=>o.value)){f.value=format;f.dispatchEvent(new Event('change'));for(const name of [...w.options].map(o=>o.value)){w.value=name;w.dispatchEvent(new Event('change'));if(document.querySelectorAll('#results tbody tr').length!==4)throw new Error('Missing rows');result.push({format,name,rows:4});}}return result;})()`);
 assert.equal(checks.length,28);
 await evaluate("f.value='mxfp4';w.value='dense_prefill_proxy';draw();document.getElementById('chart').scrollIntoView();");await png('chart.png');
 const pdf=await c('Page.printToPDF',{printBackground:true,preferCSSPageSize:true});await writeFile(path.join(root,'report.pdf'),Buffer.from(pdf.data,'base64'));
 await c('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await evaluate('window.scrollTo(0,0)');assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);await png('mobile.png');
 await writeFile(path.join(output,'browser.json'),JSON.stringify({status:'passed',offline:true,desktop_mobile_overflow:false,interactive_cases:checks.length,figure_count:await evaluate('document.querySelectorAll("svg").length'),pdf_generated:true},null,2)+'\n');console.log('passed: '+output);
}finally{if(socket)socket.close();chrome.kill('SIGTERM');await new Promise(resolve=>chrome.exitCode!==null?resolve():chrome.once('exit',resolve));await rm(profile,{recursive:true,force:true,maxRetries:6,retryDelay:100});}
