import{A as e,C as t,D as n,E as r,F as i,I as a,L as o,M as s,N as c,P as l,S as u,T as d,_ as f,b as p,c as m,d as ee,g as h,h as g,j as _,k as v,l as te,m as y,n as ne,p as re,r as b,t as ie,u as x,v as S,w as C,y as w,z as T}from"./app-DzHBvgKi.js";var E=`qse-conductor-token`;function ae(){try{return sessionStorage.getItem(E)??localStorage.getItem(E)}catch{return null}}function oe(e,t=!1){try{sessionStorage.removeItem(E),localStorage.removeItem(E),e&&(t?localStorage:sessionStorage).setItem(E,e)}catch{}}async function D(e,t,n){let r={"Content-Type":`application/json`},i=ae();i&&(r.Authorization=`Bearer ${i}`);let a;try{a=await fetch(u()+e,{method:t,headers:r,body:JSON.stringify(n),credentials:`same-origin`,cache:`no-store`})}catch(t){throw new p(e,0,`engine unreachable (${t.message})`,`offline`)}let o={};try{o=await a.json()}catch{}if(a.status===401)throw new p(e,401,`editing needs the admin token`,`token_needed`);if(!a.ok){let t=new p(e,a.status,o.error?.message??`HTTP ${a.status}`,o.error?.type??`error`);throw t.errors=o.error?.errors??[],t}return o}var O=encodeURIComponent,k=(e,t)=>e===`profiles`?`/v1/conductor/profiles/${O(t??``)}`:`/v1/conductor/${e}`,A={config:()=>C(`/v1/conductor/config`),meta:()=>C(`/v1/conductor/meta`),profiles:()=>C(`/v1/conductor/profiles`),doc:(e,t)=>C(k(e,t)),versions:(e,t)=>C(`${k(e,t)}/versions`),version:(e,t,n)=>C(`${k(e,t)}/versions/${n}`),diff:(e,t,n,r)=>C(`${k(e,t)}/diff?from=${n}&to=${r}`),check:(e,t,n)=>C(`${k(e,t)}/check`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({doc:n})}),put:(e,t,n,r,i)=>D(k(e,t),`PUT`,{doc:n,note:r,base_version:i}),restore:(e,t,n,r)=>D(`${k(e,t)}/restore`,`POST`,{version:n,note:r}),setMode:(e,t,n,r)=>D(`/v1/conductor/profiles/${O(e)}/mode`,`POST`,{mode:t,note:n,base_version:r}),roles:()=>C(`/v1/conductor/roles?fresh=1`),roleTest:e=>C(`/v1/conductor/roles/${O(e)}/test`,{method:`POST`,headers:{"Content-Type":`application/json`},body:`{}`}),scan:()=>C(`/v1/conductor/packages/scan`),classify:(e,t,n=!0)=>C(`/v1/conductor/classify`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({text:e,profile:t,tools:n})}),decisions:(e={})=>{let t=new URLSearchParams;return e.since&&t.set(`since`,String(e.since)),t.set(`limit`,String(e.limit??100)),e.session&&t.set(`session`,e.session),e.before&&t.set(`before`,String(e.before)),C(`/v1/conductor/decisions?${t}`)},decision:e=>C(`/v1/conductor/decisions/${O(e)}`),label:(e,t,n=``)=>D(`/v1/conductor/decisions/${O(e)}/label`,`POST`,{label:t,note:n}),events:(e={})=>{let t=new URLSearchParams;return e.since&&t.set(`since`,String(e.since)),t.set(`limit`,String(e.limit??100)),e.session&&t.set(`session`,e.session),e.task&&t.set(`task`,e.task),e.mode&&t.set(`mode`,e.mode),e.acted&&t.set(`acted`,`1`),C(`/v1/conductor/events?${t}`)},event:e=>C(`/v1/conductor/events/${O(e)}`),stats:(e=24)=>C(`/v1/conductor/stats?hours=${e}`),sessions:()=>C(`/v1/conductor/sessions`),session:e=>C(`/v1/conductor/sessions/${O(e)}`),exportJson:e=>C(`/v1/conductor/sessions/${O(e)}/export?format=json`),exportUrl:(e,t)=>`${u()}/v1/conductor/sessions/${O(e)}/export?format=${t}`,evals:()=>C(`/v1/conductor/evals`),watch:()=>C(`/v1/conductor/watch`),deciders:()=>C(`/v1/conductor/deciders/status`),deciderLoad:e=>D(`/v1/conductor/deciders/models/${O(e)}/load`,`POST`,{}),deciderUnload:e=>D(`/v1/conductor/deciders/models/${O(e)}/unload`,`POST`,{}),deciderTest:(e,t,n)=>C(`/v1/conductor/deciders/test`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({state:e,questions:t,model:n})})},se=.9,ce=`on is allowed at precision ≥ ${se} over ≥ 30 labelled decisions`;function le(e){return e.labelled<30?{allowed:!1,why:`${e.labelled} of 30 labelled decisions`}:e.precision==null||e.precision<.9?{allowed:!1,why:`precision ${ue(e.precision)} is below ${se}`}:{allowed:!0,why:`precision ${ue(e.precision)} over ${e.labelled}`}}var ue=e=>e==null?`—`:e.toFixed(2);function de(e){return e.some(e=>e!=null&&(e.indicative??(e.n??0)<10))}function fe(e){return e===`raw`?`raw`:`TandemLLM`}function pe(e,t){return!e.build||!t.build||e.build!==t.build?{ok:!1,why:`builds ${e.build??`?`} and ${t.build??`?`}`}:!e.night||!t.night||e.night!==t.night?{ok:!1,why:`nights ${e.night??`?`} and ${t.night??`?`}`}:{ok:!0,why:null}}function me(e){let t=new Map;for(let n of e){let e=t.get(n.suite)??{raw:null,tandem:null};n.arm===`raw`?e.raw=n:e.tandem=n,t.set(n.suite,e)}return[...t.entries()].map(([e,t])=>({suite:e,...t,cmp:t.raw&&t.tandem?pe(t.raw,t.tandem):{ok:!1,why:t.raw?`no TandemLLM arm`:`no raw arm`}}))}var he={mean:[`mean`,`tok/s`,2],ttft_p50_ms:[`TTFT`,`ms`,1],ms_blk:[`ms a block`,`ms`,2]};function ge(e){return e.filter(e=>e.stat in he).map(e=>{let[t,n,r]=he[e.stat],i=e.delta_pct>=0?`+`:`−`;return`${t} ${e.base.toFixed(r)} → ${e.cand.toFixed(r)} ${n} (${i}${Math.abs(e.delta_pct).toFixed(1)} %${e.noise_pct==null?``:`, noise ${e.noise_pct.toFixed(1)} %`})`}).join(`, `)}function _e(e,t){let n=e.filter(e=>e.kind===`speed_guard`),r=e=>e.build??{},i=n.find(e=>t.code_sha&&r(e).code_sha===t.code_sha||!t.code_sha&&t.version&&r(e).version===t.version),a=i??n.find(e=>t.version&&r(e).version===t.version),o=t.version??`unknown`;if(!a)return{tone:`warn`,status:`missing`,line:`speed guard not run for build ${o}`,detail:`the raw engine's never-worse gate (both views, same hour) has no result for the code that serves now.`};let s=a.stats??[],c=a.runs_each,l=ge(s),u=String(a.verdict??``),d=i?``:` It was recorded for code ${String(r(a).code_sha??`?`)}; the server runs ${t.code_sha??`?`}.`;return u===`no_noise_estimate`?{tone:i?`note`:`warn`,status:i?`no_noise_estimate`:`other_code`,line:`speed guard for build ${o}: one row pair, ${c??1} run a side${l?` — ${l}`:``}`,detail:`One run gives no noise estimate, so this row cannot say whether a difference is real; it is no pass and no failure.${d}`}:u===`worse`?{tone:`warn`,status:`worse`,line:`speed guard for build ${o}: ${String(a.why??`resolved worse`)}`,detail:`${l}.${d}`}:u===`pass`&&i?{tone:`ok`,status:`pass`,line:`speed guard passed for build ${o}${c?` (${c} runs a side)`:``}`,detail:l||null}:u===`pass`?{tone:`warn`,status:`other_code`,line:`speed guard passed for build ${o} on other code`,detail:`${l}${l?`.`:``}${d}`.trim()}:{tone:`warn`,status:`missing`,line:`speed guard for build ${o}: verdict ${u||`unknown`}`,detail:a.why??null}}var ve=[`cobalt`,`orange`,`aqua`,`violet`,`yellow`,`red`];function ye(e,t){let n=e==null?-1:t.indexOf(e);return n<0?`muted`:ve[n%ve.length]}function be(e,t,n=4){let r=Object.entries(e.by_class).sort((e,t)=>t[1]-e[1]),i=r.slice(0,n).map(([e])=>e),a=i.map(n=>({name:n,color:ye(n,t),values:e.hourly.map(e=>e.by_class[n]??0)}));return r.length>n&&a.push({name:`other`,color:`muted`,values:e.hourly.map(e=>Object.entries(e.by_class).filter(([e])=>!i.includes(e)).reduce((e,[,t])=>e+t,0))}),a}function xe(e){return JSON.parse(JSON.stringify(e))}function j(e,t,n){if(!t.length)return n;let[r,...i]=t,a=e??(typeof r==`number`?[]:{}),o=Array.isArray(a)?[...a]:{...a};return o[r]=j(a[r],i,n),o}function Se(e){return e.replace(/^\$\.?/,``).replace(/\[/g,` › `).replace(/\]/g,``).replace(/\./g,` › `)||`(document)`}function M(e,t=80){if(e==null)return`—`;let n=typeof e==`string`?e:JSON.stringify(e);return n.length>t?n.slice(0,t-1)+`…`:n}function Ce(e){let t=[];e.effort&&t.push(e.thinking&&e.thinking.startsWith(`off`)?`thinking off`:e.effort);for(let n of e.injected??[]){let e=n.parts.filter(e=>e.kind!==`effort`).map(e=>e.name);t.push(`${e.length?e.join(` + `):`note`} (${n.tokens.toLocaleString(`en-US`)} tokens)`)}return e.replayed?.count&&t.push(`replayed ${e.replayed.count}`),e.mode===`shadow`&&t.push(`shadow`),t.join(` · `)||`—`}var we={decider:`decider`,qwen:`Qwen`,sticky:`kept (same task)`,structural:`request shape`,fallback:`fail-open`,profile:`profile default`};function Te(e){let t=new Map;e.forEach((n,r)=>{for(let i of n.next_checker??[])for(let n of i.readings){let a=`${i.checker} ${n.gate}`;t.has(a)||t.set(a,Array(e.length).fill(null)),t.get(a)[r]=n.value}});let n=[`cobalt`,`orange`,`aqua`,`violet`,`yellow`,`red`];return{series:[...t.entries()].slice(0,6).map(([e,t],r)=>({name:e,color:n[r%n.length],values:t})),labels:e.map((e,t)=>`turn ${t+1}`)}}var Ee={thinking:`cobalt`,"end of turn":`orange`,"tool calls":`aqua`,session:`violet`,other:`muted`};function De(e){let t=[...new Set(e.hourly.flatMap(e=>Object.keys(e.by_family)))],n=Object.keys(Ee);return t.sort((e,t)=>n.indexOf(e)-n.indexOf(t)),t.map(t=>({name:t,color:Ee[t]??`muted`,values:e.hourly.map(e=>e.by_family[t]??0)}))}function Oe(e,t){let n=e?.properties?.tasks?.additionalProperties,r=(typeof n==`object`&&n?n.oneOf??[]:[]).find(e=>e.properties?.rule?.const===t);if(!r?.properties)return null;let{rule:i,mode:a,...o}=r.properties;return{...r,properties:o,required:(r.required??[]).filter(e=>e!==`rule`&&e!==`mode`)}}function ke(e){let t=Object.keys(e.tasks),n=t.filter(e=>/^C\d/.test(e)).sort();return[...n,...t.filter(e=>!n.includes(e))]}function Ae(e){return/^S\d/.test(e)}function je(e){let t=[];for(let n of e.gates)t.push(`${n.checker} (${n.gate}) ${n.values.map(e=>N(e)).join(`, `)}${n.noted?` · noted`:``}`);return e.compaction?.watching&&t.push(`after compaction ${e.compaction.count}: ${e.compaction.steps} step${e.compaction.steps===1?``:`s`}, no re-read yet`),e.progress&&t.push(e.progress.last?`last progress ${N(e.progress.minutes)} min ago (${e.progress.last})`:`no progress for ${N(e.progress.minutes)} min`),e.client_gone?t.push(`client gone after ${N(e.idle_minutes??0)} min`):e.idle_minutes!=null&&t.push(`idle ${N(e.idle_minutes)} min`),t.join(` · `)||`nothing to watch yet`}var N=e=>String(Number.isInteger(e)?e:Math.round(e*100)/100);function Me(e){return e.split(`
`).map(e=>e.trim()).filter(Boolean)}function Ne(e){return e.source===`structural`||e.shadow?.source===`structural`}function Pe(e,t,n,r,i=5){let a=e=>e.ts_ms,o=e=>e.ts_ms+Math.max(0,e.outcome?.total_ms??0),s=t.filter(e=>e.ts_ms!=null&&(x(e)||ee(e))),c=[...e.map(a),...e.map(o),...s.map(e=>e.ts_ms)],l=c.length?Math.min(...c):0,u=c.length?Math.max(...c):1,d=Math.max(1,u-l),f=e=>Math.min(1,Math.max(0,(e-l)/d)),p=[...e].sort((e,t)=>e.ts_ms-t.ts_ms),m={start:l,end:u,turns:[],acts:[],notes:[],compactions:[],checkers:[],ticks:[]};p.forEach((e,t)=>{let i=Ne(e);m.turns.push({id:e.id,x0:f(a(e)),x1:f(o(e)),color:ye(e.class_id,n),compaction:i,title:`${r(e.ts_ms)} · ${e.class_id??`—`} · ${e.outcome?.finish_reason??`running`}`}),i&&m.compactions.push({id:e.id,x:f(a(e)),title:`${r(e.ts_ms)} compaction (the client summarised the session)`});for(let t of e.injected??[])m.notes.push({id:`${e.id}:${t.pos}`,x:f(a(e)),shadow:!1,title:`${r(e.ts_ms)} router note, ${t.tokens} tokens (${t.reason})`});if(!e.injected?.length&&e.shadow&&e.shadow.notes?.length&&m.notes.push({id:`${e.id}:shadow`,x:f(a(e)),shadow:!0,title:`${r(e.ts_ms)} router note in shadow`}),e.next_checker?.length){let n=p[t+1]?a(p[t+1]):o(e),i=e.next_checker.every(e=>e.verdict===`PASS`),s=e.next_checker.map(e=>`${e.checker} ${e.verdict}${e.readings.length?`: ${e.readings.map(e=>`(${e.gate}) ${e.value} ${e.op} ${e.limit}`).join(`, `)}`:``}`).join(`; `);m.checkers.push({id:e.id,x:f(n),pass:i,title:`${r(n)} ${s}`})}});for(let e of s)m.acts.push({id:e.id??`${e.ts_ms}`,x:f(e.ts_ms),shadow:!x(e),task:e.task??`—`,title:`${r(e.ts_ms)} ${te({task:e.task,decision:e.decision,mode:e.mode,trigger:e.trigger??null,cap:e.cap??null})}`});for(let e=0;e<=i;e++)m.ticks.push({x:e/i,label:r(l+d*e/i)});return m}function Fe(e){let t=e.filter(e=>e.tier===1&&ee(e));if(!t.length)return null;let n=t.filter(e=>e.label===`right`||e.label===`wrong`);return{decisions:t.length,labelled:n.length,agree:n.filter(e=>e.label===`right`).length}}function P(e,t,n){let r=ye(e,t);return T`<span class="cls" style="--cls: var(--series-${r})" data-class=${e??`none`}>${n??e??`—`}</span>`}function Ie(e,t=.6){return e==null?T`<span class="conf muted">—</span>`:T`<span class="conf" title=${`confidence ${e.toFixed(4)}`}><span class="conf-bar ${e<t?`is-low`:``}"><span style="width:${Math.round(e*100)}%"></span></span>${e.toFixed(2)}</span>`}function F(e,t,n,r={}){return T`<div class="seg ${r.small?`seg-sm`:``} ${r.cls??``}" role="radiogroup" aria-label=${r.label??`mode`}>
    ${e.map(e=>T`<button type="button" class="seg-btn mode-${e} ${e===t?`is-on`:``}" role="radio" aria-checked=${e===t?`true`:`false`} ?disabled=${r.disabled} @click=${()=>e!==t&&n(e)}>${e}</button>`)}
  </div>`}function Le(e){return e?we[e]??e:`—`}var I=class extends c{constructor(...e){super(...e),this.kind=`confirm`,this.heading=``,this.lines=[],this.confirmLabel=`Confirm`,this.done=()=>void 0,this.error=null}submit(e){if(e.preventDefault(),this.kind===`token`){let e=(this.querySelector(`#cond-token`)?.value??``).trim();if(!e){this.error=`Enter the admin token.`;return}this.done(!0,e,this.querySelector(`#cond-remember`)?.checked??!1)}else this.done(!0)}render(){return T`<div class="modal-back" @click=${()=>this.done(!1)}></div>
      <form class="modal" role="dialog" aria-modal="true" aria-label=${this.heading} @submit=${e=>this.submit(e)}>
        <h2>${this.heading}</h2>
        ${this.lines.map(e=>T`<p class="muted small">${e}</p>`)}
        ${this.kind===`token`?T`<label class="field field-block"><span>Admin token</span><input id="cond-token" type="password" autocomplete="current-password" required autofocus /></label>
              <label class="switch"><input id="cond-remember" type="checkbox" /> remember on this device</label>`:o}
        ${this.error?T`<p class="err-ink small" role="alert">${this.error}</p>`:o}
        <div class="modal-actions">
          <button type="button" class="btn" @click=${()=>this.done(!1)}>Cancel</button>
          <button type="submit" class="btn btn-primary" data-confirm>${this.confirmLabel}</button>
        </div>
      </form>`}};s([i({attribute:!1})],I.prototype,`kind`,void 0),s([i({attribute:!1})],I.prototype,`heading`,void 0),s([i({attribute:!1})],I.prototype,`lines`,void 0),s([i({attribute:!1})],I.prototype,`confirmLabel`,void 0),s([i({attribute:!1})],I.prototype,`done`,void 0),s([l()],I.prototype,`error`,void 0),I=s([a(`qse-cond-dialog`)],I);function Re(e){return new Promise(t=>{let n=document.createElement(`qse-cond-dialog`);e(n),n.done=(e,r,i)=>{n.remove(),t({ok:e,token:r,remember:i})},document.body.appendChild(n)})}function ze(e,t,n){return Re(r=>{r.kind=`confirm`,r.heading=e,r.lines=t,r.confirmLabel=n}).then(e=>e.ok)}async function Be(e=`Changing the conductor needs the admin token (ops/make-secrets.sh made it; it is in secrets.env on the box).`){let t=await Re(t=>{t.kind=`token`,t.heading=`Unlock editing`,t.lines=[e,`It stays in this browser tab, or on this device if you tick the box, and is sent only with a change.`],t.confirmLabel=`Unlock`});return t.ok&&t.token&&oe(t.token,!!t.remember),t.ok&&!!t.token}async function L(e){try{return await e()}catch(t){if(t instanceof p&&t.type===`token_needed`&&(ae()&&oe(null),await Be()))return await e();throw t}}var R=class{constructor(e,t,n=null){this.host=e,this.kind=t,this.name=n,this.base=null,this.draft=null,this.loading=!0,this.loadError=null,this.errors=[],this.changes=[],this.note=``,this.saving=!1,this.saved=null,this.saveError=null,this.versions=null,this.showVersions=!1,this.showDiff=!1,this.onSaved=null,this.checkTimer=null}get dirty(){return this.changes.length>0}get locked(){return this.base?.locked??null}async load(){this.loading=!0,this.host.requestUpdate();try{this.base=await A.doc(this.kind,this.name),this.draft=xe(this.base.doc),this.errors=[],this.changes=[],this.loadError=null}catch(e){let n=t(e);this.loadError=`${n.endpoint}: ${n.message}`}finally{this.loading=!1,this.host.requestUpdate()}}set(e){this.draft=e,this.saved=null,this.host.requestUpdate(),this.checkTimer&&clearTimeout(this.checkTimer),this.checkTimer=setTimeout(()=>void this.check(),250)}async check(){if(this.draft){try{let e=await A.check(this.kind,this.name,this.draft);this.errors=e.errors,this.changes=e.changes}catch(e){this.errors=[t(e).message]}this.host.requestUpdate()}}reset(){this.base&&(this.draft=xe(this.base.doc),this.errors=[],this.changes=[],this.host.requestUpdate())}async save(){if(!this.draft||!this.base)return!1;if(!this.note.trim())return this.saveError=`Write a one-line change note first.`,this.host.requestUpdate(),!1;this.saving=!0,this.saveError=null,this.host.requestUpdate();try{let e=await L(()=>A.put(this.kind,this.name,this.draft,this.note.trim(),this.base.version));return this.saved=`version ${e.version}, reloaded ${e.loaded_at?_(e.loaded_at):`now`}`,this.note=``,await this.load(),this.showVersions&&await this.loadVersions(),this.onSaved?.(),!0}catch(e){let t=e;return this.saveError=t.status===409?`${t.message}. Reload to see it.`:t.message,t.errors?.length&&(this.errors=t.errors),!1}finally{this.saving=!1,this.host.requestUpdate()}}async loadVersions(){try{this.versions=(await A.versions(this.kind,this.name)).versions}catch(e){this.saveError=t(e).message}this.host.requestUpdate()}async restore(e){if(await ze(`Restore version ${e}?`,[`A new version equal to version ${e} becomes current, with the note "restore of v${e}". The server loads it for the next request.`],`Restore`))try{let t=await L(()=>A.restore(this.kind,this.name,e));this.saved=`version ${t.version} (restore of v${e}), reloaded ${t.loaded_at?_(t.loaded_at):`now`}`,await this.load(),await this.loadVersions(),this.onSaved?.()}catch(e){this.saveError=t(e).message,this.host.requestUpdate()}}};function z(e){return e.errors.length?T`<ul class="errors" role="alert" data-errors>${e.errors.slice(0,12).map(e=>T`<li>${e}</li>`)}</ul>`:o}function Ve(e){return e.length?T`<div class="diff" data-diff>
    ${e.map(e=>T`<div class="diff-row"><span>${Se(e.path)}</span><span class="diff-before">${M(e.before,160)}</span><span class="diff-after">${M(e.after,160)}</span></div>`)}
  </div>`:T`<p class="muted small">No changes.</p>`}function B(e,t){let n=e.base?.meta;return e.locked?T`<div class="savebar" data-savebar><span class="badge badge-lock">${g(`warn`)} locked</span><span class="status">${e.locked}. Every control is read-only.</span></div>`:T`
    ${e.showDiff?T`<section class="panel" data-diff-panel><header class="panel-head"><h2 class="panel-title">Changes against live (v${n?.version??0})</h2></header>${Ve(e.changes)}</section>`:o}
    ${e.showVersions?T`<section class="panel" data-versions><header class="panel-head"><h2 class="panel-title">Versions of ${t}</h2></header>
          ${e.versions==null?T`<p class="muted small">Loading…</p>`:T`<div class="table-scroll"><table class="table">
                <thead><tr><th>version</th><th>note</th><th>author</th><th>when</th><th></th></tr></thead>
                <tbody>${e.versions.map(t=>T`<tr data-version=${t.version}>
                    <td class="num">v${t.version}${t.current?T` <span class="badge badge-on">live</span>`:o}</td>
                    <td class="cell-wrap">${t.note??`—`}</td>
                    <td class="muted">${t.author??`—`}</td>
                    <td class="num muted">${t.created_at?r(t.created_at):`built in`}</td>
                    <td>${t.current?o:T`<button class="btn btn-sm" @click=${()=>e.restore(t.version)}>Restore</button>`}</td>
                  </tr>`)}</tbody></table></div>`}
        </section>`:o}
    <div class="savebar" data-savebar>
      <span class="status num" title=${n?.note??``}>live v${n?.version??0}${n?.loaded_at?T` · reloaded ${_(n.loaded_at)}`:o}</span>
      <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>e.note=t.target.value} aria-label="change note" maxlength="200" />
      <button class="btn btn-primary" ?disabled=${!e.dirty||e.saving||e.errors.length>0} @click=${()=>e.save()} data-save>${e.saving?`Saving…`:`Save as new version`}</button>
      <button class="btn btn-ghost" ?disabled=${!e.dirty} @click=${()=>e.reset()}>Reset</button>
      <button class="btn btn-ghost ${e.showDiff?`is-on`:``}" @click=${()=>{e.showDiff=!e.showDiff,e.check()}}>Diff (${e.changes.length})</button>
      <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():e.host.requestUpdate()}} data-versions-btn>Versions</button>
      ${e.saved?T`<span class="status" role="status" data-saved>${g(`check`)} ${e.saved}</span>`:o}
      ${e.saveError?T`<span class="err-ink small" role="alert">${e.saveError}</span>`:o}
    </div>`}var V=class extends c{constructor(...e){super(...e),this.row=null,this.ids=[],this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.row){this.labelling=!0;try{await L(()=>A.label(this.row.id,e)),this.row={...this.row,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=t(e).message}finally{this.labelling=!1}}}render(){let e=this.row;if(!e)return o;let t=e.stages.find(e=>!e.error&&!e.below);return T`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="decision" data-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title">${P(e.class_id,this.ids)} ${e.mode===`shadow`?T`<span class="badge badge-shadow">shadow</span>`:T`<span class="badge badge-on">on</span>`}</div>
            <div class="muted small num">${r(new Date(e.ts_ms).toISOString())} · ${e.profile} v${e.profile_version} · router v${e.router_version}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${g(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">
            <h3>Decision</h3>
            <dl class="kv">
              <div><dt>class</dt><dd>${e.class_id??`—`} ${Ie(e.confidence)}</dd></div>
              <div><dt>decided by</dt><dd>${Le(e.source)}${e.kept?T` <span class="badge badge-warn">${e.kept.replace(/_/g,` `)}</span>`:o}</dd></div>
              <div><dt>classifier time</dt><dd class="num">${v(e.classifier_ms)}</dd></div>
              <div><dt>effort</dt><dd>${e.effort??`unchanged`}${e.budget==null?o:T` <span class="muted num">budget ${e.budget}</span>`} <span class="muted">(${e.budget_source??`server`})</span></dd></div>
              <div><dt>thinking</dt><dd>${e.thinking??`on`}</dd></div>
              <div><dt>modes</dt><dd class="num">${Object.entries(e.decisions??{}).map(([e,t])=>`${e} ${t}`).join(` · `)}</dd></div>
            </dl>
          </section>
          ${e.stages.length?T`<section class="drawer-section"><h3>Classifier stages</h3><div class="stage-list">
                ${e.stages.map(e=>T`<div class="stage ${e===t?`is-accepted`:``} ${e.error?`is-error`:``}">
                    <strong>${e.stage===`decider`?`decider`:`Qwen`}</strong>
                    <div>${e.error?T`<span class="err-ink">${e.error}</span>`:T`${e.label} ${Ie(e.confidence,e.accept??.6)}${e.below?T` <span class="muted small">below ${e.accept}</span>`:o}`}
                      ${e.probs&&Object.keys(e.probs).length>1?T`<div class="probs">${Object.entries(e.probs).sort((e,t)=>t[1]-e[1]).slice(0,6).map(([e,t])=>T`<div class="prob"><span>${e}</span><span class="prob-bar"><span style="width:${Math.round(t*100)}%"></span></span><span>${t.toFixed(2)}</span></div>`)}</div>`:o}
                    </div>
                    <span class="num muted">${v(e.ms)}${e.cached?` (cached)`:``}</span>
                  </div>`)}</div></section>`:o}
          ${(e.injected??[]).map(e=>T`<section class="drawer-section" data-injected>
              <h3>Injected at token ${e.pos.toLocaleString(`en-US`)} · ${e.tokens.toLocaleString(`en-US`)} tokens ${re(e.text,`Copy the injected text`)}</h3>
              <p class="muted small">Because ${e.reason}. Parts: ${e.parts.map(e=>`${e.kind} ${e.name}`).join(`, `)}. The client never saw this text.</p>
              <pre class="note-text">${e.text}</pre>
            </section>`)}
          ${e.replayed?.count?T`<p class="muted small">Replayed ${e.replayed.count} earlier note(s), ${e.replayed.tokens.toLocaleString(`en-US`)} tokens, at the same positions, so the resident prefix kept matching.</p>`:o}
          ${e.events?.length?T`<section class="drawer-section" data-monitor-events><h3>The session monitor in this request</h3>
                ${e.events.map(e=>We(e))}</section>`:o}
          ${e.shadow?T`<section class="drawer-section"><h3>Shadow (not applied)</h3><pre class="note-text">${JSON.stringify(e.shadow,null,2)}</pre></section>`:o}
          ${e.digest?T`<section class="drawer-section"><h3>What the classifier read</h3><pre class="note-text">${e.digest}</pre></section>`:o}
          <section class="drawer-section">
            <h3>Outcome</h3>
            ${e.outcome?T`<dl class="kv">
                  <div><dt>finish</dt><dd>${e.outcome.finish_reason??`running`}</dd></div>
                  <div><dt>tool calls</dt><dd class="num">${e.outcome.tool_calls??0} ${(e.outcome.tool_names??[]).join(`, `)}</dd></div>
                  <div><dt>reasoning tokens</dt><dd class="num">${e.outcome.reasoning_tokens??`—`}</dd></div>
                  <div><dt>time</dt><dd class="num">${v(e.outcome.total_ms)}</dd></div>
                  ${e.next_checker?.length?T`<div><dt>next checker</dt><dd>${e.next_checker.map(e=>T`<span class="badge ${e.verdict===`PASS`?`badge-ok`:`badge-warn`}">${e.verdict}</span> <span class="muted small">${e.checker} ${e.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:o}
                </dl>`:T`<p class="muted small">The turn is still running.</p>`}
          </section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${e.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${g(`check`)} this was right</button>
              <button class="btn ${e.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${g(`x`)} this was wrong</button>
              ${e.label?T`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:o}
              ${this.msg?T`<span class="muted small" role="status">${this.msg}</span>`:o}
            </div>
            <p class="muted small">Labels feed the precision on the Compare screen and the promotion rule.</p>
          </section>
          <p class="muted small num">decision ${e.id} · request ${e.request_id??`—`} · session ${M(e.session_key,40)}</p>
        </div>
      </aside>`}};s([i({attribute:!1})],V.prototype,`row`,void 0),s([i({attribute:!1})],V.prototype,`ids`,void 0),s([i({attribute:!1})],V.prototype,`close`,void 0),s([l()],V.prototype,`labelling`,void 0),s([l()],V.prototype,`msg`,void 0),V=s([a(`qse-cond-decision`)],V);function He(e,t,n,r){return T`<button class="feed-row" data-decision=${e.id} @click=${()=>n(e)}>
    <span class="feed-time">${_(new Date(e.ts_ms).toISOString())}</span>
    <span class="feed-cls">${P(e.class_id,t)}</span>
    <span class="feed-conf">${Ie(e.confidence)} <span class="muted small">${Le(e.source)}</span></span>
    <span class="feed-what">${r}</span>
    <span class="feed-act">${e.acted?T`<span class="badge badge-on">acted</span>`:o}${e.mode===`shadow`||e.shadow?T` <span class="badge badge-shadow">shadow</span>`:o}${!e.acted&&!(e.mode===`shadow`||e.shadow)?T`<span class="badge">pass</span>`:o}</span>
  </button>`}function Ue(e){return x(e)?T`<span class="badge badge-on">acted</span>`:e.mode===`shadow`&&e.decision!==`PASS`?T`<span class="badge badge-shadow">shadow</span>`:T`<span class="badge">pass</span>`}function We(e){let t=(e.hidden_tokens??0)>0;return T`<div class="event-block" data-event=${e.id}>
    <div class="event-head">${Ue(e)} <strong>${te(e)}</strong></div>
    <dl class="kv">
      <div><dt>safe point</dt><dd>${e.sp??`—`} · rule ${e.rule??`—`} · policy v${e.policy_version??`—`}</dd></div>
      ${e.tokens_at_cut==null?o:T`<div><dt>reasoning tokens</dt><dd class="num">${e.tokens_at_cut.toLocaleString(`en-US`)}</dd></div>`}
      ${e.outcome?T`<div><dt>outcome</dt><dd>${e.outcome.kind??`—`}${e.outcome.finish?T` <span class="muted">(${e.outcome.finish})</span>`:o}</dd></div>`:o}
      ${e.next_checker?.length?T`<div><dt>next checker</dt><dd>${e.next_checker.map(e=>T`<span class="badge ${e.verdict===`PASS`?`badge-ok`:`badge-warn`}">${e.verdict}</span> <span class="muted small">${e.checker} ${e.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:o}
    </dl>
    ${e.text?T`<h4 class="small">${t?T`Hidden text at token ${(e.position??0).toLocaleString(`en-US`)} · ${(e.hidden_tokens??0).toLocaleString(`en-US`)} tokens`:`Visible text`} ${re(e.text,`Copy the text`)}</h4><pre class="note-text">${e.text}</pre>`:o}
    ${e.detail&&Object.keys(e.detail).length?T`<details><summary class="muted small">detail</summary><pre class="note-text">${JSON.stringify(e.detail,null,2)}</pre></details>`:o}
  </div>`}function Ge(e,t){return T`<button class="feed-row" data-event-row=${e.id} @click=${()=>t(e)}>
    <span class="feed-time">${_(new Date(e.ts_ms).toISOString())}</span>
    <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
    <span class="feed-conf muted small">${e.sp??``}</span>
    <span class="feed-what">${te(e)}${(e.hidden_tokens??0)>0?T` <span class="muted small num">${e.hidden_tokens} hidden tokens</span>`:o}</span>
    <span class="feed-act">${Ue(e)}</span>
  </button>`}var H=class extends c{constructor(...e){super(...e),this.ev=null,this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.ev){this.labelling=!0;try{await L(()=>A.label(this.ev.id,e)),this.ev={...this.ev,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=t(e).message}finally{this.labelling=!1}}}render(){let e=this.ev;return e?T`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="monitor decision" data-event-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title"><span class="badge mono">${e.task}</span> ${Ue(e)}</div>
            <div class="muted small num">${r(new Date(e.ts_ms).toISOString())} · ${e.profile??`no profile`} · ${e.client_kind??`—`}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${g(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">${We(e)}</section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${e.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${g(`check`)} this was right</button>
              <button class="btn ${e.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${g(`x`)} this was wrong</button>
              ${e.label?T`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:o}
              ${this.msg?T`<span class="muted small" role="status">${this.msg}</span>`:o}
            </div>
            <p class="muted small">Labels on shadow decisions feed the task's precision and its promotion to act (Compare).</p>
          </section>
          <p class="muted small num">event ${e.id} · request ${e.request_id??`—`} · session ${e.session_key?T`<a href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${M(e.session_key,40)}</a>`:`—`}</p>
        </div>
      </aside>`:o}};s([i({attribute:!1})],H.prototype,`ev`,void 0),s([i({attribute:!1})],H.prototype,`close`,void 0),s([l()],H.prototype,`labelling`,void 0),s([l()],H.prototype,`msg`,void 0),H=s([a(`qse-cond-event`)],H);var U=class extends c{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=m(()=>this.loadStats(),2e4),this.feedPoll=m(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await A.stats(24),this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await A.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await A.event(e)).event}catch(e){this.error=t(e).message}}renderMonitor(e){let t=De(e);return T`<div class="grid-2 grid-2-wide">
      ${h(`Monitor acts an hour, last 24 h`,e.acts?T`<qse-ribbon .series=${t} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:y(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${h(`Last actions of the monitor`,e.last_actions.length?T`<div class="feed" data-last-actions>${e.last_actions.map(e=>T`<button class="feed-row" data-action=${e.id} @click=${()=>this.openAction(e.id)}>
                <span class="feed-time">${_(new Date(e.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
                <span class="feed-conf muted small">${e.outcome?.kind??`running`}</span>
                <span class="feed-what">${te({task:e.task,decision:e.decision,mode:`act`,trigger:e.trigger,cap:e.cap})}</span>
                <span class="feed-act">${e.session_key?T`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:o}</span>
              </button>`)}</div>`:T`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:T`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${g(`external`)}</a>`})}
    </div>`}render(){let t=this.stats;if(this.error&&!t)return T`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${this.error}</div></div>`;if(!t)return f(6);let r=t.classifier,i=r.decider,a=r.qwen,s=i?.calls?i:a?.calls?a:i,c=Object.entries(t.roles),l=c.filter(([,e])=>e.up).length,u=t.monitor;return T`
      <div class="stats" data-tiles>
        ${S({label:`decisions, 24 h`,value:n(t.decisions),sub:T`<span class="num">${n(t.acted)}</span> acted · <span class="num">${n(t.shadow)}</span> in shadow`,title:`${t.injections} notes, ${t.injected_tokens} tokens injected`})}
        ${S({label:`monitor, 24 h`,value:u?n(u.acts):`—`,sub:u?T`acts · <span class="num">${n(u.shadow)}</span> in shadow · <span class="num" data-gave-up>${n(u.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!u,warn:(u?.gave_up??0)>0,title:u?`${n(u.hidden_tokens)} hidden tokens`:void 0})}
        ${S({label:`classifier p50 / p95${!i?.calls&&a?.calls?` (Qwen)`:``}`,value:s?.p50_ms==null?`—`:`${Math.round(s.p50_ms)} / ${Math.round(s.p95_ms??0)}`,unit:s?.p50_ms==null?void 0:`ms`,sub:T`decider ${n(i?.calls??0)} calls${a?T` · Qwen ${n(a.calls)} (${v(a.p50_ms)})`:o} · fail-open ${e(t.fail_open_rate,1)}`,warn:(t.fail_open_rate??0)>.1})}
        ${S({label:`roles online`,value:`${l} / ${c.length}`,sub:T`${c.map(([e,t])=>T`<span title=${t.error??``} class=${t.up?``:t.mode===`off`?`muted`:`err-ink`} data-role=${e}>${t.up?`●`:`○`} ${e} </span>`)}`,warn:c.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${h(`Decisions an hour, last 24 h`,t.decisions?T`<qse-ribbon .series=${be(t,this.ids)} .labels=${t.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:y(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${h(`Task classes, 24 h`,T`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${t.classes.map(e=>T`<tr><td>${P(e.class_id,this.ids)}</td><td class="num num-col">${n(e.decisions)}</td><td class="num num-col">${e.labelled?`${e.agree} / ${e.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(t.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(t.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${h(T`Live decisions ${this.paused?T`<span class="badge">paused</span>`:T`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?T`<div class="feed" data-feed>${this.feed.map(e=>He(e,this.ids,e=>this.open=e,Ce(e)))}</div>`:T`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:T`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${g(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${g(`external`)}</a>`})}
        ${h(`Profiles by client`,T`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(t=>T`<button class="pick" @click=${()=>w({screen:`profiles`,edit:e.name})}><span class="muted">${t}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(e=>T`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${u?this.renderMonitor(u):o}
      ${this.open?T`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:o}
      ${this.openEvent?T`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:o}
    `}};s([i({attribute:!1})],U.prototype,`ids`,void 0),s([i({attribute:!1})],U.prototype,`tiles`,void 0),s([i({attribute:!1})],U.prototype,`active`,void 0),s([l()],U.prototype,`stats`,void 0),s([l()],U.prototype,`feed`,void 0),s([l()],U.prototype,`error`,void 0),s([l()],U.prototype,`open`,void 0),s([l()],U.prototype,`openEvent`,void 0),s([l()],U.prototype,`paused`,void 0),U=s([a(`qse-cond-overview`)],U);var Ke=[`off`,`low`,`medium`,`high`,`xhigh`],W=class extends c{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new R(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),A.doc(`packages`).then(e=>this.packages=Object.keys(e.doc.packages)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(j(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await A.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=t(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return f(8);if(e.loadError)return T`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=t.classes.map(e=>e.id);return T`
      ${this.renderTest(n)}
      <details class="sec" open data-sec="classes">
        <summary>Task classes <span class="sec-sub">${t.classes.length} classes; a classifier sees only the descriptions</span></summary>
        <div class="sec-body">
          ${t.classes.map((e,r)=>this.renderClass(e,r,t.classes.length,n))}
          <button class="btn" data-add-class @click=${()=>this.addClass()}>${g(`plus`)} Add a class</button>
        </div>
      </details>
      <details class="sec" data-sec="classifier">
        <summary>Classifier stages <span class="sec-sub">asked in order until one answers at its threshold</span></summary>
        <div class="sec-body">
          ${t.classifier.stages.map((e,n)=>T`<div class="row-card" data-stage=${e.kind}>
              <div class="row-card-head">
                <strong>${n+1}. ${e.kind===`decider`?`decider (role decider: any model behind the Jev question shape)`:`Qwen, thinking off (role classifier)`}</strong>
                <span class="toolbar" style="margin:0">
                  <label class="switch"><input type="checkbox" .checked=${e.enabled} @change=${e=>this.upd([`classifier`,`stages`,n,`enabled`],e.target.checked)} /> enabled</label>
                  <button class="btn btn-ghost btn-sm" ?disabled=${n===0} @click=${()=>this.moveStage(n,-1)} aria-label="earlier">↑</button>
                  <button class="btn btn-ghost btn-sm" ?disabled=${n===t.classifier.stages.length-1} @click=${()=>this.moveStage(n,1)} aria-label="later">↓</button>
                </span>
              </div>
              <div class="form-grid">
                ${this.num(`accept at (probability)`,e.accept,e=>this.upd([`classifier`,`stages`,n,`accept`],e),{step:.01,min:0,max:1,hint:`below it, the next stage is asked`})}
                ${this.num(`timeout (ms)`,e.timeout_ms,e=>this.upd([`classifier`,`stages`,n,`timeout_ms`],Math.round(e)),{step:50,min:1,hint:`a slower answer counts as no answer`})}
              </div>
            </div>`)}
          <div class="form-grid">
            ${this.text(`question to the decider`,t.classifier.question,e=>this.upd([`classifier`,`question`],e))}
            ${this.num(`digest (characters)`,t.classifier.digest_chars,e=>this.upd([`classifier`,`digest_chars`],Math.round(e)),{step:100,min:200,hint:`the user message is cut to this, head and tail; the decider cuts again to its own limits`})}
            ${this.text(`package question`,t.classifier.package_question,e=>this.upd([`classifier`,`package_question`],e))}
            ${this.num(`package accept at`,t.classifier.package_accept,e=>this.upd([`classifier`,`package_accept`],e),{step:.01,min:0,max:1})}
          </div>
          <label class="field field-block"><span>Qwen's instruction (<code>{classes}</code> becomes the id and description list)</span>
            <textarea rows="5" .value=${t.classifier.qwen_prompt} @input=${e=>this.upd([`classifier`,`qwen_prompt`],e.target.value)}></textarea></label>
        </div>
      </details>
      <details class="sec" data-sec="stickiness">
        <summary>Stickiness and structure <span class="sec-sub">when a session changes class, and what the request shape decides</span></summary>
        <div class="sec-body">
          <div class="form-grid">
            ${this.num(`switch class at (probability)`,t.stickiness.switch,e=>this.upd([`stickiness`,`switch`],e),{step:.01,min:0,max:1,hint:`a session with a class moves to another only at this confidence`})}
            <label class="field"><span>opencode's compaction request</span>
              <select @change=${e=>this.upd([`structural`,`compaction`],e.target.value)}>
                ${[`keep`,`classify`,...n].map(e=>T`<option value=${e} ?selected=${e===t.structural.compaction}>${e===`keep`?`keep the session class`:e===`classify`?`ask a classifier`:`class ${e}`}</option>`)}
              </select><span class="hint">the turn itself; the session keeps its class after it</span></label>
            <label class="field"><span>a turn that only carries tool results</span>
              <select @change=${e=>this.upd([`structural`,`tool_turn`],e.target.value)}>
                <option value="keep" ?selected=${t.structural.tool_turn===`keep`}>keep the session class (no model call)</option>
                <option value="classify" ?selected=${t.structural.tool_turn===`classify`}>ask a classifier every turn</option>
              </select></label>
          </div>
        </div>
      </details>
      <details class="sec" data-sec="effort">
        <summary>Effort levels <span class="sec-sub">a budget and a sentence at the prompt's end; never the template's reasoning_effort</span></summary>
        <div class="sec-body">
          <p class="muted small">The template renders <code>reasoning_effort</code> as the first system tokens, so changing it per turn would cost a cold prefill. A level here sets the reasoning budget (0 = no cap below the backstop), thinking on or off (off uses <code>enable_thinking: false</code> only when the server's effort is medium, else a budget of 1), and a sentence in the tail note.</p>
          ${Ke.map(e=>T`<div class="row-card" data-effort=${e}>
              <div class="row-card-head"><strong class="mono">${e}</strong>
                <label class="switch"><input type="checkbox" .checked=${t.effort.levels[e].thinking} @change=${t=>this.upd([`effort`,`levels`,e,`thinking`],t.target.checked)} /> thinking</label></div>
              <div class="form-grid">${this.num(`budget (tokens, 0 = no cap)`,t.effort.levels[e].budget,t=>this.upd([`effort`,`levels`,e,`budget`],Math.round(t)),{step:256,min:0})}</div>
              <label class="field field-block"><span>sentence in the note</span><textarea rows="2" .value=${t.effort.levels[e].note} @input=${t=>this.upd([`effort`,`levels`,e,`note`],t.target.value)}></textarea></label>
            </div>`)}
          <label class="switch"><input type="checkbox" .checked=${t.effort.raise_on_fail} @change=${e=>this.upd([`effort`,`raise_on_fail`],e.target.checked)} /> a failing checker (RESULT: FAIL) raises the effort one level for that turn</label>
        </div>
      </details>
      <details class="sec" data-sec="injection">
        <summary>The note <span class="sec-sub">where hidden text goes and how much</span></summary>
        <div class="sec-body">
          <p class="muted small">The note goes into the last turn, right before its <code>&lt;|im_end|&gt;</code>. Later turns replay it at the same token position, so the resident prefix keeps matching. <code>{text}</code> is replaced by the effort sentence, the package and the rules.</p>
          <label class="field field-block"><span>wrapper</span><textarea rows="6" data-wrapper .value=${t.injection.wrapper} @input=${e=>this.upd([`injection`,`wrapper`],e.target.value)}></textarea></label>
          <div class="form-grid">
            ${this.num(`max tokens per note`,t.injection.max_note_tokens,e=>this.upd([`injection`,`max_note_tokens`],Math.round(e)),{step:500,min:16})}
            ${this.num(`max tokens replayed per session`,t.injection.max_replay_tokens,e=>this.upd([`injection`,`max_replay_tokens`],Math.round(e)),{step:1e3,min:0})}
            <label class="switch"><input type="checkbox" .checked=${t.injection.replay} @change=${e=>this.upd([`injection`,`replay`],e.target.checked)} /> replay notes on later turns</label>
          </div>
        </div>
      </details>
      ${z(e)}
      ${B(e,`the router`)}
    `}renderTest(e){let t=this.result;return T`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?T`<span class="err-ink small" role="alert">${this.testError}</span>`:o}
      </div>
      ${t?t.active?T`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${P(t.class_id??null,e)} ${Ie(t.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${Le(t.source??null)}${t.kept?T` <span class="badge badge-warn">${t.kept}</span>`:o} <span class="muted num">${v(t.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${t.effort??`unchanged`} · thinking ${t.thinking??`on`}${t.budget==null?``:` · budget ${t.budget}`}</dd></div>
                  <div><dt>profile</dt><dd>${t.profile} (${t.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(t.stages??[]).map(e=>T`<div class="stage ${e.error?`is-error`:e.below?``:`is-accepted`}"><strong>${e.stage===`decider`?`decider`:`Qwen`}</strong><span>${e.error??T`${e.label} ${Ie(e.confidence,e.accept??.6)}`}</span><span class="num muted">${v(e.ms)}</span></div>`)}</div>
              </div>
              <div>${t.note?T`<p class="muted small">The model would see this at the end of the last turn (${(t.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${t.note}</pre>`:T`<p class="muted small">No note: ${t.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:T`<p class="muted small" style="margin-top:12px">Profile ${t.profile} is raw (${t.mode}): the conductor does nothing for it.</p>`:o}
    </section>`}renderClass(e,t,n,r){let i=e.policy,a=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return T`<details class="row-card" data-class-row=${e.id} ?open=${a}>
      <summary>
        <span>${P(e.id,r,e.label)} <span class="muted small mono">${i.effort??`medium`}${(i.packages??[]).length?` · ${(i.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===0} @click=${()=>this.moveClass(t,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===n-1} @click=${()=>this.moveClass(t,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n<=1} @click=${()=>this.removeClass(t)} aria-label=${`remove ${e.id}`}>${g(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${e.id} ?readonly=${!a} @input=${e=>this.upd([`classes`,t,`id`],e.target.value)} data-class-id /><span class="hint">${a?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${e.label} @input=${e=>this.upd([`classes`,t,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${e.description} @input=${e=>this.upd([`classes`,t,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${F(Ke,i.effort??`medium`,e=>this.upd([`classes`,t,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let n=(i.packages??[]).includes(e);return T`<button class="pick ${n?`is-on`:``}" aria-pressed=${n?`true`:`false`} @click=${()=>this.upd([`classes`,t,`policy`,`packages`],n?(i.packages??[]).filter(t=>t!==e):[...i.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(i.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,t,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}num(e,t,n,r={}){return T`<label class="field"><span>${e}</span><input type="number" .value=${String(t)} step=${r.step??1} min=${r.min??``} max=${r.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&n(t)}} />${r.hint?T`<span class="hint">${r.hint}</span>`:o}</label>`}text(e,t,n){return T`<label class="field"><span>${e}</span><input .value=${t} @input=${e=>n(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(j(n,[`classifier`,`stages`],r))}};s([i({attribute:!1})],W.prototype,`ids`,void 0),s([i({attribute:!1})],W.prototype,`meta`,void 0),s([i({attribute:!1})],W.prototype,`active`,void 0),s([i({attribute:!1})],W.prototype,`onSaved`,void 0),s([l()],W.prototype,`packages`,void 0),s([l()],W.prototype,`testText`,void 0),s([l()],W.prototype,`testProfile`,void 0),s([l()],W.prototype,`testTools`,void 0),s([l()],W.prototype,`testing`,void 0),s([l()],W.prototype,`result`,void 0),s([l()],W.prototype,`testError`,void 0),W=s([a(`qse-cond-routing`)],W);var qe=[`off`,`shadow`,`act`],Je=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,Ye=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,G=class extends c{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new R(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),A.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(j(this.ed.draft,e,t))}render(){let e=this.ed,t=this.config?.server.monitor_mode;return T`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${e.loading&&!e.draft?f(8):e.loadError?T`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${e.loadError}</div></div>`:T`
            ${t===`raw`?T`<div class="guard-line" data-monitor-off>${g(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:o}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(e.draft)}
            ${this.renderRest(e.draft)}
            ${z(e)}
            ${B(e,`the monitor`)}`}`}renderTasks(e){let t=ke(e),n=t.filter(t=>e.tasks[t]&&/^C\d/.test(t)),r=t.filter(e=>Ae(e)),i=t.filter(e=>!n.includes(e)&&!r.includes(e)),a=r.map(t=>e.tasks[t]).find(e=>e.rule===`stall_watchdog`)?.notify;return T`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${n.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${n.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${r.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${r.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${a?.enabled?T`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:T`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, Khaled's ruling of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${r.map(t=>this.renderTask(t,e.tasks[t]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${i.join(`, `)}</span></summary>
        <div class="sec-body">${i.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>`}renderTask(e,t){let n=Oe(this.meta?.schemas?.monitor,t.rule),r=this.meta?.tasks?.[e];return T`<div class="row-card" data-monitor-task=${e}>
      <div class="row-card-head">
        <strong>${e}${r?T` · ${r.label}`:o} <span class="muted small mono">${t.rule}</span></strong>
        ${F(qe,t.mode,t=>this.upd([`tasks`,e,`mode`],t),{small:!0,label:`${e} mode without a profile`})}
      </div>
      ${r?.why?T`<p class="muted small">${r.why}</p>`:o}
      ${n?.properties?T`<div class="form-stack">${Object.entries(n.properties).map(([n,r])=>this.field(n,r,t[n],[`tasks`,e,n]))}</div>`:T`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,t,[`tasks`,e])}`}
    </div>`}field(e,t,n,r){let i=Array.isArray(t.type)?t.type[0]:t.type,a=e.replace(/_/g,` `),s=t.description?T`<span class="hint">${t.description}</span>`:o;if(t.enum)return T`<label class="field"><span>${a}</span><select data-param=${e} @change=${e=>this.upd(r,e.target.value)}>${t.enum.map(e=>T`<option value=${String(e)} ?selected=${e===n}>${String(e)}</option>`)}</select>${s}</label>`;if(i===`boolean`)return T`<label class="switch" data-param=${e}><input type="checkbox" .checked=${!!n} @change=${e=>this.upd(r,e.target.checked)} /> ${a}</label>${t.description?T`<p class="hint">${t.description}</p>`:o}`;if(i===`integer`||i===`number`)return T`<label class="field"><span>${a}</span><input type="number" data-param=${e} step=${i===`integer`?`1`:`0.01`} min=${t.minimum??o} max=${t.maximum??o} .value=${n==null?``:String(n)} @input=${e=>{let t=e.target.value;this.upd(r,t===``?null:i===`integer`?Math.round(Number(t)):Number(t))}} />${s}</label>`;if(i===`string`){let t=String(n??``);return Je.test(e)||t.length>60?T`<label class="field field-block"><span>${a}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(t.length/90)))} class=${Ye.test(e)?`mono`:``} data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)}></textarea>${s}</label>`:T`<label class="field"><span>${a}</span><input data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)} />${s}</label>`}if(i===`array`&&t.items&&(Array.isArray(t.items.type)?t.items.type[0]:t.items.type)===`string`){let i=!Array.isArray(n)&&Array.isArray(t.default),o=Array.isArray(n)?n:i?t.default:[];return T`<label class="field field-block"><span>${a} <span class="muted small">(one a line${i?`; the default, not saved yet`:``})</span></span><textarea rows=${Math.min(8,Math.max(2,o.length+1))} data-param=${e} .value=${o.join(`
`)} @input=${e=>this.upd(r,Me(e.target.value))}></textarea>${s}</label>`}if(i===`array`&&t.items?.properties){let i=Array.isArray(n)?n:[],o=t.items;return T`<fieldset class="fieldset" data-param=${e}><legend>${a} <span class="muted small">${i.length}</span></legend>${s}
        ${i.map((e,t)=>T`<div class="row-card row-card-sub" data-item=${String(e.id??t)}>
          <div class="row-card-head"><strong class="mono small">${String(e.id??`#${t+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(r,i.filter((e,n)=>n!==t))} aria-label=${`remove ${String(e.id??t+1)}`}>${g(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(o.properties).map(([n,i])=>this.field(n,i,e[n],[...r,t,n]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${e} @click=${()=>this.upd(r,[...i,Xe(o)])}>${g(`plus`)} Add</button>
      </fieldset>`}if(i===`object`&&t.properties){let i=n??{};return T`<fieldset class="fieldset" data-param=${e}><legend>${a}</legend>${s}
        <div class="form-stack">${Object.entries(t.properties).map(([e,n])=>i[e]===void 0&&!(t.required??[]).includes(e)?T`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...r,e],Xe(n))}>${g(`plus`)} ${e.replace(/_/g,` `)}</button>`:this.field(e,n,i[e],[...r,e]))}</div></fieldset>`}return this.jsonField(a,n,r)}jsonField(e,t,n){return T`<label class="field field-block"><span>${e} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(t,null,2)} @change=${t=>{try{let e=JSON.parse(t.target.value);if(n.length===2&&n[0]===`tasks`){let t=this.ed.draft.tasks[n[1]];this.upd(n,{...e,rule:t.rule,mode:t.mode})}else this.upd(n,e)}catch{this.ed.errors=[`${e}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(e){return T`
      <details class="sec" data-sec="caps">
        <summary>Caps, texts and the journal <span class="sec-sub">beyond a cap an act becomes GIVE_UP; a profile sets its own caps</span></summary>
        <div class="sec-body">
          <div class="form-grid">
            ${this.field(`acts_per_request`,{type:`integer`,minimum:0},e.caps.acts_per_request,[`caps`,`acts_per_request`])}
            ${this.field(`acts_per_session_hour`,{type:`integer`,minimum:0},e.caps.acts_per_session_hour,[`caps`,`acts_per_session_hour`])}
            ${this.field(`same_action_in_a_row`,{type:`boolean`,description:`off: the same hidden action twice in a row on one turn ends in GIVE_UP`},e.caps.same_action_in_a_row,[`caps`,`same_action_in_a_row`])}
            ${this.field(`tool_passes`,{type:`boolean`,description:`one journal row for every answer whose calls passed every check (findings are always journaled, passes always counted)`},e.journal.tool_passes,[`journal`,`tool_passes`])}
          </div>
          ${Object.entries(e.texts).map(([e,t])=>this.field(e,{type:`string`,description:e===`give_up`?`the one visible sentence of a GIVE_UP`:void 0},t,[`texts`,e]))}
          <div class="form-grid">
            ${this.field(`lineage_tokens`,{type:`integer`,minimum:1,description:`prompt tokens hashed into the session key when the client sends no id`},e.session.lineage_tokens,[`session`,`lineage_tokens`])}
            ${this.field(`join_window_s`,{type:`number`,minimum:0},e.session.join_window_s,[`session`,`join_window_s`])}
            ${this.field(`max_sessions`,{type:`integer`,minimum:1},e.session.max_sessions,[`session`,`max_sessions`])}
          </div>
        </div>
      </details>
      <details class="sec" data-sec="templates">
        <summary>Chat template shapes <span class="sec-sub">how a hidden turn, a reminder and a note are written in the model's own format</span></summary>
        <div class="sec-body">${Object.entries(e.templates).map(([e,t])=>this.field(e,{type:`string`},t,[`templates`,e]))}</div>
      </details>`}};s([i({attribute:!1})],G.prototype,`meta`,void 0),s([i({attribute:!1})],G.prototype,`onSaved`,void 0),s([l()],G.prototype,`config`,void 0),G=s([a(`qse-cond-monitor`)],G);function Xe(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,Xe(t)])):``}var K=class extends c{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=m(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await A.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}render(){let e=this.sessions;return h(T`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!e?T`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?f(2):e.length?T`<div class="form-stack" data-watch>${e.map(e=>this.row(e))}</div>`:T`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(e){let t=e.gates.some(e=>e.noted);return T`<div class="row-card" data-watch-session=${e.session}>
      <div class="row-card-head">
        <strong class="mono small">${e.session}</strong>
        <span class="toolbar" style="margin:0">
          ${e.watch?T`<span class="badge ${e.watch.mode===`act`?`badge-on`:`badge-shadow`}">${e.watch.task} ${e.watch.mode}</span>`:o}
          ${e.client_gone?T`<span class="badge badge-warn" data-client-gone>client gone</span>`:e.running?T`<span class="badge badge-on">running</span>`:o}
          ${e.watch?.notify?T`<span class="badge badge-warn" data-notify>push on</span>`:o}
          ${t?T`<span class="badge badge-warn">gate noted</span>`:o}</span>
      </div>
      <p class="small" style="margin:4px 0">${je(e)}</p>
      <p class="muted small num" style="margin:0">${e.watch?`${e.watch.profile||`—`} · ${e.watch.client||`—`} · `:``}${e.requests} requests · last ${r(new Date(e.last_seen*1e3).toISOString())}${e.note_at?` · last note ${r(new Date(e.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};s([l()],K.prototype,`sessions`,void 0),s([l()],K.prototype,`monitor`,void 0),s([l()],K.prototype,`error`,void 0),K=s([a(`qse-cond-watch`)],K);var q=class extends c{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=m(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await A.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}render(){let e=this.events,t=[...new Set((e??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!t.includes(this.task)&&t.push(this.task),T`${h(T`Monitor journal ${this.paused?T`<span class="badge">paused</span>`:T`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!e?T`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?f(4):e.length?T`<div class="feed" data-journal>${e.map(e=>Ge(e,e=>this.open=e))}</div>`:T`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:T`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${t.map(e=>T`<option value=${e} ?selected=${e===this.task}>${e}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${g(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?T`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:o}`}};s([i({attribute:!1})],q.prototype,`session`,void 0),s([l()],q.prototype,`events`,void 0),s([l()],q.prototype,`monitor`,void 0),s([l()],q.prototype,`error`,void 0),s([l()],q.prototype,`task`,void 0),s([l()],q.prototype,`kind`,void 0),s([l()],q.prototype,`paused`,void 0),s([l()],q.prototype,`open`,void 0),q=s([a(`qse-cond-journal`)],q);var Ze=[`off`,`low`,`medium`,`high`,`xhigh`],Qe={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},$e=new Set([`S1`,`S4`]),et=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],J=class extends c{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleTest={},this.ed=null,this.roles=new R(this,`roles`),this.residency=new R(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),A.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),A.doc(`packages`).then(e=>this.packages=Object.keys(e.doc.packages)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new R(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{this.roleHealth=(await A.roles()).health}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let t=await A.roleTest(e);this.roleTest={...this.roleTest,[e]:t.ok?`ok in ${v(t.ms)}: "${t.answer?.label}" at ${t.answer?.confidence.toFixed(2)}`:`failed: ${t.error}`}}catch(n){this.roleTest={...this.roleTest,[e]:t(n).message}}this.loadHealth()}render(){return T`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>T`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>w({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?T`<span class="badge badge-lock">${g(`warn`)} locked</span>`:T`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${r(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):T`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(j(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=j(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=j(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderEditor(e){if(e.loading&&!e.draft)return f(8);if(e.loadError)return T`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=!!e.locked,r=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],i=this.router?.classes??[];return T`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${g(`playground`)} Try in Playground</a></div>
      ${n?T`<div class="note" data-lock-reason>${g(`warn`)} ${e.locked}. Every control below is read-only.</div>`:o}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${t.description} ?disabled=${n} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${F([`raw`,`shadow`,`on`],t.mode,e=>this.upd([`mode`],e),{disabled:n,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${r.map(e=>{let r=t.clients.includes(e);return T`<button class="pick ${r?`is-on`:``}" ?disabled=${n} aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`clients`],r?t.clients.filter(t=>t!==e):[...t.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${n} @change=${e=>this.upd([`default_class`],e.target.value)}>${i.map(e=>T`<option value=${e.id} ?selected=${e.id===t.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(Qe).map(e=>T`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${F([`off`,`shadow`,`on`],t.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:n,small:!0,label:e})}<span class="small">${Qe[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${i.map(e=>{let r=t.routing[e.id]??{},i=Object.keys(r).length>0;return T`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${P(e.id,this.ids,e.label)}
                ${i?T`<button class="btn btn-ghost btn-sm" ?disabled=${n} @click=${()=>{let n={...t.routing};delete n[e.id],this.upd([`routing`],n)}}>use the router's</button>`:T`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${F(Ze,r.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...r,effort:t}),{disabled:n,small:!0,label:`effort override of ${e.id}`,cls:r.effort?``:`is-inherited`})}</div>
              <div class="pkg-picks">${this.packages.map(t=>{let i=r.packages??e.policy.packages??[],a=i.includes(t);return T`<button class="pick ${a?`is-on`:``}" ?disabled=${n} @click=${()=>this.upd([`routing`,e.id],{...r,packages:a?i.filter(e=>e!==t):[...i,t]})}>${t}</button>`})}</div>
            </div>`})}
        </div>
      </details>
      <details class="sec" data-sec="tasks">
        <summary>Session monitor tasks <span class="sec-sub">the frame (SRV-109) runs these; act needs the role online</span></summary>
        <div class="sec-body">${Object.entries(t.tasks).map(([e,t])=>this.renderTask(e,t.mode,t.cap,t.thresholds,n))}</div>
      </details>
      <details class="sec" data-sec="thinking">
        <summary>Thinking <span class="sec-sub">the effort when no class sets one</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${F(Ze,t.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:n,small:!0,label:`default effort`})}</div>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.thinking.hints} @change=${e=>this.upd([`thinking`,`hints`],e.target.checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113); a pack that is off turns its task off</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${et.map(([e,r,i])=>T`<label class="switch" data-pack=${e}><input type="checkbox" ?disabled=${n} .checked=${t.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${i} <span class="muted small">(${r}, ${t.toolgate[e]?t.tasks[r]?.mode??`off`:`off`})</span></label>`)}
          <label class="field"><span>Tier 1 timeout (ms)</span><input type="number" ?disabled=${n} .value=${String(t.toolgate.tier1_timeout_ms)} @input=${e=>this.upd([`toolgate`,`tier1_timeout_ms`],Math.round(Number(e.target.value)))} /></label>
        </div>
        <p class="muted small">With a pack on, its task runs in the mode set above; with one of them able to act, an answer's tool calls are held to its end and released whole. The floors, rules and reminders are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      <details class="sec" data-sec="context">
        <summary>Context <span class="sec-sub">the session tasks S1, S2, S4 (SRV-121): their numbers and the push</span></summary>
        <div class="sec-body"><div class="form-grid">
          <label class="field"><span>S1: failed runs of one gate before the blocked-stage note <span class="hint">count, default 3</span></span><input type="number" min="1" max="20" ?disabled=${n} data-context="s1_runs" .value=${String(t.context.s1_runs)} @input=${e=>this.ctx(`s1_runs`,`S1`,`fails`,Math.round(Number(e.target.value)))} /></label>
          <label class="field"><span>S4: minutes without progress before the note, and of silence before the client counts as gone <span class="hint">minutes, default 20</span></span><input type="number" min="1" max="600" ?disabled=${n} data-context="s4_minutes" .value=${String(t.context.s4_minutes)} @input=${e=>this.ctx(`s4_minutes`,`S4`,`minutes`,Math.round(Number(e.target.value)))} /></label>
          <label class="switch" data-context="notify"><input type="checkbox" ?disabled=${n} .checked=${t.context.notify} @change=${e=>this.upd([`context`,`notify`],e.target.checked)} /> NOTIFY: a push through ntfy to the phone when the client has gone <span class="badge ${t.context.notify?`badge-warn`:``}">${t.context.notify?`on`:`off`}</span></label>
        </div>
        <p class="muted small" data-notify-rule>Allowed by Khaled on 2026-09-30: off by default, on in the overnight profile. Only S4 in act pushes, never in shadow, once a turn end; the token is a file on the box. S2 (re-read after a compaction) has no number here: its texts and state files are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      <details class="sec" data-sec="hints">
        <summary>Speed hints <span class="badge badge-warn">not gated</span></summary>
        <div class="sec-body">
          <p class="warn-ink small">No SPD measurement is in the journal yet (SPD-73); the router's ≤ 1 ulp text rule and the never-worse gate come first.</p>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.stair_width} @change=${e=>this.upd([`hints`,`stair_width`],e.target.checked)} /> StairCut width hint</label>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.prefix_retention} @change=${e=>this.upd([`hints`,`prefix_retention`],e.target.checked)} /> prefix retention hint</label>
        </div>
      </details>
      ${z(e)}
      ${B(e,`profile ${e.name}`)}
    </div>`}renderTask(e,t,n,r,i){let a=this.meta?.tasks[e],s=a?.role&&a.tier===`1`,c=!s||!!this.roleHealth[a.role]?.up;return T`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${a?.family??``}</div></div>
      <div>${a?.label??e}<div class="task-why">${a?.why??``}</div>${s&&!c?T`<div class="warn-ink small">${g(`warn`)} the ${a.role} role is offline: act is refused</div>`:o}</div>
      <div class="task-mode">${F([`off`,`shadow`,`act`],t,t=>this.upd([`tasks`,e,`mode`],t),{disabled:i,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${$e.has(e)&&Object.keys(r).length?T`<p class="muted small" data-in-context=${e}>${Object.entries(r).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:o}
        ${Object.entries($e.has(e)?{}:r).map(([t,n])=>{let r=a?.thresholds[t];return T`<label class="field ${r&&r.default!==n?`is-changed`:``}"><span>${t.replace(/_/g,` `)} <span class="hint">${r?.unit??``}${r?`, default ${r.default}`:``}</span></span>
            <input type="number" ?disabled=${i} .value=${String(n)} step="any" @input=${n=>{let r=Number(n.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,t],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${i} min="0" max="10" .value=${String(n)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let e=this.roles;if(!e.draft)return e.loadError?T`<p class="err-ink small">${e.loadError}</p>`:o;let t=e.draft,n=(n,r)=>e.set(j(t,n,r));return T`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${Object.entries(t).map(([e,t])=>{let r=this.roleHealth[e];return T`<div class="row-card" data-role-row=${e}>
            <div class="row-card-head"><strong>${e}</strong>
              <span class="toolbar" style="margin:0">${r?T`<span class="badge ${r.up?`badge-ok`:t.mode===`off`?``:`badge-warn`}">${r.up?`up ${v(r.ms)}`:t.mode===`off`?`off`:`down: ${r.error??`?`}`}</span>`:o}
                <button class="btn btn-sm" @click=${()=>this.testRole(e)} data-role-test=${e}>Test</button></span></div>
            ${this.roleTest[e]?T`<p class="small" role="status">${this.roleTest[e]}</p>`:o}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${t.endpoint} @input=${t=>n([e,`endpoint`],t.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${t.model} @input=${t=>n([e,`model`],t.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(t.timeout_ms)} @input=${t=>n([e,`timeout_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>memory claim (GB)</span><input type="number" step="0.1" .value=${String(t.memory_gb)} @input=${t=>n([e,`memory_gb`],Number(t.target.value))} /></label>
            </div>
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${F([`off`,`shadow`,`on`],t.mode,t=>n([e,`mode`],t),{small:!0,label:`${e} mode`})}<span class="muted small mono">${t.contract}</span></div>
          </div>`})}
        ${z(e)}
        ${B(e,`the roles`)}
      </div>
    </details>`}renderResidency(){let e=this.residency;if(!e.draft)return e.loadError?T`<p class="err-ink small">${e.loadError}</p>`:o;let t=e.draft,n=(n,r)=>e.set(j(t,n,r)),r=(e,r=`0.1`,i=!1)=>T`<input type="number" step=${r} min="0" .value=${String(t[e[0]])} @input=${t=>{let r=Number(t.target.value);n(e,i?Math.round(r):r)}} data-res=${e[0]} />`,i=[`measured`,`peak`,`declared`];return T`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${r([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${r([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${r([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${t.order.join(`, `)} @change=${e=>n([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(t.claims).map(([e,r])=>T`<div class="row-card" data-claim=${e}>
            <div class="row-card-head"><strong>${e}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[e]:r,...i}=t.claims;n([`claims`],i)}} aria-label=${`remove the claim of ${e}`}>${g(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(r.gib)} @input=${t=>n([`claims`,e,`gib`],Number(t.target.value))} data-claim-gib=${e} /></label>
              <label class="field"><span>basis</span><select @change=${t=>n([`claims`,e,`basis`],t.target.value)}>${i.map(e=>T`<option value=${e} ?selected=${e===r.basis}>${e}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${r.note} @input=${t=>n([`claims`,e,`note`],t.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in t.claims} @click=${()=>{n([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${g(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${z(e)}
        ${B(e,`the residency limits`)}
      </div>
    </details>`}};s([i({attribute:!1})],J.prototype,`ids`,void 0),s([i({attribute:!1})],J.prototype,`meta`,void 0),s([i({attribute:!1})],J.prototype,`tiles`,void 0),s([i({attribute:!1})],J.prototype,`selected`,void 0),s([i({attribute:!1})],J.prototype,`onSaved`,void 0),s([l()],J.prototype,`router`,void 0),s([l()],J.prototype,`packages`,void 0),s([l()],J.prototype,`roleHealth`,void 0),s([l()],J.prototype,`roleTest`,void 0),s([l()],J.prototype,`newClaim`,void 0),J=s([a(`qse-cond-profiles`)],J);var Y=class extends c{constructor(...e){super(...e),this.ids=[],this.router=null,this.sel=null,this.found=null,this.scanError=null,this.ed=new R(this,`packages`)}connectedCallback(){super.connectedCallback(),this.ed.load(),A.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0)}usedBy(e){return(this.router?.classes??[]).filter(t=>(t.policy.packages??[]).includes(e)).map(e=>e.id)}upd(e,t){this.ed.draft&&this.ed.set(j(this.ed.draft,e,t))}async scan(){this.scanError=null;try{this.found=(await A.scan()).skills}catch(e){this.scanError=t(e).message}}register(e){let t=this.ed.draft,n={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.ed.set(j(t,[`packages`,e.name],n)),this.sel=e.name}addInline(){let e=this.ed.draft,t=1;for(;e.packages[`package-${t}`];)t++;let n=`package-${t}`;this.ed.set(j(e,[`packages`,n],{title:`Package ${t}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0})),this.sel=n}removePackage(e){let t=this.ed.draft,n={...t.packages};delete n[e],this.ed.set({...t,packages:n}),this.sel=null}render(){let e=this.ed;if(e.loading&&!e.draft)return f(6);if(e.loadError)return T`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,r=e.base?.status??{};return T`
      <section class="panel" data-packages>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills</h2><p class="panel-sub">The router injects one into the prompt when the turn's class names it and the client did not load it itself.</p></div>
          <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-add-package>${g(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${g(`search`)} Skills on the box</button></div></header>
        <div class="table-scroll"><table class="table">
          <thead><tr><th>name</th><th>source</th><th>on the box</th><th class="num-col">tokens</th><th>classes</th><th>enabled</th></tr></thead>
          <tbody>${Object.entries(t.packages).map(([e,t])=>{let i=r[e];return T`<tr class="req-row ${this.sel===e?`is-open`:``}" data-package-row=${e} @click=${()=>this.sel=e}>
              <td><strong>${e}</strong><div class="muted small">${t.title}</div></td>
              <td class="mono small">${t.source}</td>
              <td>${i?i.available?T`<span class="badge badge-ok">found</span>`:T`<span class="badge badge-warn" title=${i.path??``}>${i.error??`missing`}</span>`:T`<span class="badge">not saved</span>`}</td>
              <td class="num num-col">${i?n(i.tokens_est):`—`}</td>
              <td>${this.usedBy(e).map(e=>P(e,this.ids))}</td>
              <td>${t.enabled?`yes`:`no`}</td>
            </tr>`})}</tbody></table></div>
      </section>
      ${this.found?T`<section class="panel" style="margin-top:16px" data-found>
            <header class="panel-head"><h2 class="panel-title">Skills in ${t.skill_dirs.join(`, `)}</h2></header>
            ${this.found.length?T`<div class="table-scroll"><table class="table"><tbody>${this.found.map(e=>T`<tr><td><strong>${e.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${e.description}</div></td><td class="num">${n(e.chars)} chars</td>
                    <td>${e.registered||t.packages[e.name]?T`<span class="badge badge-ok">registered</span>`:T`<button class="btn btn-sm" @click=${()=>this.register(e)} data-register=${e.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:T`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ spark:~/.config/opencode/skills/</code>).</p>`}
          </section>`:this.scanError?T`<p class="err-ink small">${this.scanError}</p>`:o}
      ${this.sel&&t.packages[this.sel]?this.renderEditor(this.sel,t.packages[this.sel]):o}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${t.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${z(e)}
      ${B(e,`the packages`)}
    `}renderEditor(e,t){let n=t.source===`inline`,r=this.usedBy(e);return T`<section class="panel" style="margin-top:16px" data-package-editor=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e}</h2><p class="panel-sub">${r.length?T`used by ${r.join(`, `)}`:`no class uses it yet (add it on the Routing screen)`}</p></div>
        <div class="panel-tools"><button class="btn btn-sm btn-ghost" ?disabled=${r.length>0} title=${r.length?`a class uses it`:``} @click=${()=>this.removePackage(e)}>${g(`trash`)} Remove</button></div></header>
      <div class="form-grid">
        <label class="field"><span>title</span><input .value=${t.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>source</span><input .value=${t.source} @input=${t=>this.upd([`packages`,e,`source`],t.target.value)} /><span class="hint">inline, skill:&lt;name&gt; or file:&lt;path on the box&gt;</span></label>
        <label class="field"><span>max tokens</span><input type="number" .value=${String(t.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">longer text is cut at a line end</span></label>
        <label class="switch"><input type="checkbox" .checked=${t.enabled} @change=${t=>this.upd([`packages`,e,`enabled`],t.target.checked)} /> enabled</label>
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" .value=${t.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${n?T`<label class="field field-block"><span>text the model gets</span><textarea rows="14" data-package-body .value=${t.body} @input=${t=>this.upd([`packages`,e,`body`],t.target.value)}></textarea></label>`:T`<p class="muted small">The text is the file on the box, read when first needed and again when it changes; its YAML front matter is left out.</p>`}
    </section>`}};s([i({attribute:!1})],Y.prototype,`ids`,void 0),s([l()],Y.prototype,`router`,void 0),s([l()],Y.prototype,`sel`,void 0),s([l()],Y.prototype,`found`,void 0),s([l()],Y.prototype,`scanError`,void 0),Y=s([a(`qse-cond-packages`)],Y);var tt={not_loaded:`not loaded`,loading:`loading the weights`,warming:`warming up`,loaded:`loaded`,unloading:`unloading`,failed:`failed`};function nt(e){return e===`loaded`?`badge-ok`:e===`failed`?`badge-warn`:e===`loading`||e===`warming`||e===`unloading`?`badge-shadow`:``}function rt(e){if(!e||e.allowed)return null;let t=e.reason?ie[e.reason]??e.reason:`refused`;if(e.reason!==`refused_memory`)return t;let n=[`MemAvailable ${b(e.mem_available_gib)}`,`claim ${b(e.claim_gib)}`];return e.after_gib!=null&&n.push(`${b(e.after_gib)} left`,`${b(e.need_gib)} needed`),`${t} (${n.join(`, `)})`}function it(e){return e?e.allowed?`fits, ${b(e.after_gib)} left after the claim`:rt(e)??`refused`:`—`}function at(e){let t=Object.entries(e).sort((e,t)=>t[1]-e[1]);return t.length?t.map(([e,t])=>`${e} ${t}`).join(` · `):`no call yet`}function ot(e){let t=e.limits;return[t.context_tokens,t.question_tokens,t.max_options].map(e=>e==null?`—`:String(e)).join(` / `)}function st(e,t){let n=e.trim();if(!n)return`Write the question.`;let r=t.split(/[,\n]/).map(e=>e.trim()).filter(Boolean);return r.length<2?`Give at least two options.`:r.length>20?`At most 20 options (the model card’s limit).`:new Set(r).size===r.length?{type:`choice`,instructions:n,criteria:Object.fromEntries(r.map(e=>[e,null]))}:`Two options have the same name.`}function ct(e){return e.type===`noul`?[[`true`,e.noul??0],[`false`,1-(e.noul??0)]]:Object.entries(e.probabilities??{}).sort((e,t)=>t[1]-e[1])}var lt=[`on_start`,`on_demand`,`off`],X=class extends c{constructor(...e){super(...e),this.st=null,this.error=null,this.busy=null,this.said={},this.tState=`The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.`,this.tQuestion=`What kind of work does this request ask for?`,this.tOptions=`debug, code_edit, plan, chat`,this.tModel=`auto`,this.tOut=null,this.tErr=null,this.asking=!1,this.ed=new R(this,`deciders`),this.poller=null}connectedCallback(){super.connectedCallback(),this.ed.load(),this.poller=m(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{this.st=await A.deciders(),this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async act(e,n){if(n!==`unload`||await ze(`Unload ${e.name}?`,[`Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).`,`Loading it again goes through the residency planner.`],`Unload`)){this.busy=e.name,this.said={...this.said,[e.name]:``};try{let t=await L(()=>n===`load`?A.deciderLoad(e.name):A.deciderUnload(e.name));this.said={...this.said,[e.name]:n===`load`?`load started (${t.state??`loading`})`:`unloaded`}}catch(r){let i=r instanceof p&&r.status===409&&n===`load`?`refused by the residency planner: ${r.message}`:t(r).message;this.said={...this.said,[e.name]:i}}finally{this.busy=null,await this.load()}}}setLoad(e,t){let n=this.ed.draft;if(!n)return;let r=n.models.findIndex(t=>t.name===e),i=r>=0?n.models.map((e,n)=>n===r?{...e,load:t}:e):[...n.models,{name:e,load:t}];this.ed.set({...n,models:i})}async ask(){let e=st(this.tQuestion,this.tOptions);if(typeof e==`string`){this.tErr=e;return}this.asking=!0,this.tErr=null;try{this.tOut=await A.deciderTest(this.tState,{q:e},this.tModel)}catch(e){this.tOut=null,this.tErr=t(e).message}finally{this.asking=!1}}render(){let e=this.st;if(this.error&&!e)return T`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${this.error}</div></div>`;if(!e)return f(6);if(!e.enabled)return h(`Decision models`,T`<div class="state" data-deciders-off>
          <div class="state-title">${g(`warn`)} This engine serves no decision model</div>
          <p class="muted small">${e.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port behind the admin token, and this screen loads, unloads and tests them.</p>
        </div>`,{id:`deciders-off`});let t=e.models.filter(e=>e.state===`loaded`);return T`
      <div class="stats" data-decider-tiles>
        ${S({label:`loaded`,value:`${t.length} / ${e.models.length}`,sub:t.length?t.map(e=>e.name).join(`, `):`none: questions fail open`,warn:!t.length})}
        ${S({label:`on the device`,value:ne(t.reduce((e,t)=>e+t.device_bytes,0)),sub:`${e.device??`no device`} · own stream and memory pool`})}
        ${S({label:`a load now`,value:e.planner?.allowed?`fits`:`refused`,sub:it(e.planner),warn:!e.planner?.allowed})}
        ${S({label:`when they run`,value:`idle`,sub:`only while no request holds the engine (stage A)`})}
      </div>
      ${this.renderModels(e)}
      ${this.renderMemory(e)}
      ${this.renderQuestions(e)}
      ${this.renderTest(e)}
    `}renderModels(e){let t=this.ed.draft,r=e=>t?.models.find(t=>t.name===e)?.load??`on_demand`;return h(`Models`,T`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${e.models.map(e=>{let t=rt(e.refusal),i=this.said[e.name],a=e.state===`loaded`||e.state===`failed`||e.state===`warming`;return T`<tr data-decider-row=${e.name}>
              <td><strong>${e.name}</strong><div class="muted small">${e.family} · ${e.contract}${e.aliases.length?T` · ${e.aliases.join(`, `)}`:o}</div></td>
              <td><span class="badge ${nt(e.state)}" data-decider-state=${e.state}>${tt[e.state]}</span>${e.warm===!1?T` <span class="muted small">not warmed</span>`:o}</td>
              <td><select aria-label=${`load policy of ${e.name}`} data-decider-load=${e.name} @change=${t=>this.setLoad(e.name,t.target.value)}>
                ${lt.map(t=>T`<option value=${t} ?selected=${r(e.name)===t}>${t.replace(`_`,` `)}</option>`)}
              </select></td>
              <td class="num">${ot(e)}</td>
              <td class="small">${e.precision??T`<span class="muted">—</span>`}<div class="muted small mono" title=${e.weight_hash??``}>${e.weight_hash?e.weight_hash.slice(0,12):``}</div></td>
              <td class="num num-col">${n(e.questions)}</td>
              <td class="num num-col">${e.stats?.forward_ms_p50==null?`—`:`${v(e.stats.forward_ms_p50)} / ${v(e.stats.forward_ms_p95)}`}</td>
              <td class="num num-col ${e.stats?.errors?`err-ink`:``}">${e.stats?n(e.stats.errors+e.stats.deadline_missed):`—`}</td>
              <td>${a?T`<button class="btn btn-sm" ?disabled=${this.busy===e.name} @click=${()=>this.act(e,`unload`)} data-unload=${e.name}>Unload</button>`:T`<button class="btn btn-sm btn-primary" ?disabled=${this.busy===e.name||e.state===`loading`||e.state===`unloading`} @click=${()=>this.act(e,`load`)} data-load=${e.name}>Load</button>`}</td>
            </tr>
            ${t||e.error||i?T`<tr class="is-warn" data-decider-note=${e.name}><td colspan="9" class="small">
                  ${t?T`<span class="warn-ink" data-refusal=${e.name}>${g(`warn`)} ${t}</span> `:o}
                  ${e.error?T`<span class="err-ink">${e.error}</span> `:o}
                  ${i?T`<span role="status">${i}</span>`:o}
                </td></tr>`:o}`})}</tbody>
      </table></div>
      ${e.unserved?.length?T`<p class="muted small">In the document but not served by this engine: ${e.unserved.join(`, `)} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>`:o}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,{id:`deciders-models`,sub:`the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen`})}renderMemory(e){let t=e.planner,n=e.last_refusal;return h(`Memory`,T`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${b(t?.mem_available_gib)}</div></div>
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${b(t?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${b(t?.floor_gib)} + ${b(t?.margin_gib)}</div></div>
          <div class="stat ${t?.allowed?``:`is-warn`}"><div class="stat-label">a load now</div><div class="stat-value num small">${it(t)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${n?T`Last refused engine load: ${rt(n)} <span class="muted">(${n.source??``})</span>`:T`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable minus the claim stays at or above floor + margin, never during a hold, never beside another load. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,{id:`deciders-memory`})}renderQuestions(e){let t=this.ed,n=t.draft,r=e.models.map(e=>e.name),i=n?Object.entries(n.questions):[];return T`${h(`Questions`,n?T`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${i.map(([i,a])=>{let s=e.questions.find(e=>e.id===i);return T`<tr data-question=${i}>
                  <td class="mono">${i}</td>
                  <td><select aria-label=${`model for ${i}`} data-question-model=${i} @change=${e=>t.set(j(n,[`questions`,i],e.target.value))}>
                    ${[`auto`,...r].map(e=>T`<option value=${e} ?selected=${e===a}>${e===`auto`?`auto (the role’s model; the router picks)`:e}</option>`)}
                    ${a!==`auto`&&!r.includes(a)?T`<option value=${a} selected>${a} (not served)</option>`:o}
                  </select></td>
                  <td class="small">${at(s?.picks??{})}</td>
                </tr>`})}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); the session monitor's Tier 1 questions join this list when they land.</p>`:f(3),{id:`deciders-questions`,sub:`which model answers each question the conductor asks`})}
      ${z(t)}
      ${B(t,`the deciders document`)}`}renderTest(e){let t=this.tOut,r=t?.answers.q;return h(`Test box`,T`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${e=>this.tState=e.target.value} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${e=>this.tQuestion=e.target.value} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${e=>this.tOptions=e.target.value} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${e=>this.tModel=e.target.value} data-test-model>
            ${[`auto`,...e.models.map(e=>e.name)].map(e=>T`<option value=${e} ?selected=${e===this.tModel}>${e}</option>`)}
          </select></label>
        </div>
        <div class="savebar"><button class="btn btn-primary" ?disabled=${this.asking} @click=${()=>this.ask()} data-test-ask>${g(`send`)} ${this.asking?`Asking…`:`Ask`}</button>
          ${this.tErr?T`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>`:o}</div>
        ${t&&r?T`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${r.type===`choice`?r.choice:r.type===`noul`?(r.noul??0).toFixed(4):(r.score??0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${r.confidence.toFixed(2)} · ${t.model} · ${v(t.latency_ms)} · ${t.path} · ${n(t.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${ct(r).map(([e,t])=>T`<div class="prob-row" data-prob=${e}><span class="mono small">${e}</span><span class="conf-bar"><span style="width:${Math.round(t*100)}%"></span></span><span class="num small">${t.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${t.routing.reason}${t.routing.requested?` (wanted ${t.routing.requested})`:``}</p>
            </div>`:o}`,{id:`deciders-test`,sub:`one choice question, as /v1/systemone answers it (it runs in the idle window like any other)`})}};s([l()],X.prototype,`st`,void 0),s([l()],X.prototype,`error`,void 0),s([l()],X.prototype,`busy`,void 0),s([l()],X.prototype,`said`,void 0),s([l()],X.prototype,`tState`,void 0),s([l()],X.prototype,`tQuestion`,void 0),s([l()],X.prototype,`tOptions`,void 0),s([l()],X.prototype,`tModel`,void 0),s([l()],X.prototype,`tOut`,void 0),s([l()],X.prototype,`tErr`,void 0),s([l()],X.prototype,`asking`,void 0),X=s([a(`qse-cond-deciders`)],X);var Z=class extends c{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=m(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await A.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async loadSession(e){try{let t=await A.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(n){e===this.selected&&(this.turns=[]),this.error=t(n).message}}render(){if(this.error&&!this.list)return T`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${this.error}</div></div>`;if(!this.list)return f(6);let e=[...new Set(this.list.map(e=>e.client_kind??`other`))],t=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return T`
      ${this.selected?this.renderSession(this.selected):o}
      ${h(`Sessions, last 7 days`,t.length?T`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${t.map(e=>T`<tr class="req-row ${e.key===this.selected?`is-open`:``}" data-session=${e.key} @click=${()=>w({session:e.key})}>
                  <td class="mono small cell-wrap">${M(e.key,28)}</td><td>${e.client_kind??`—`}</td><td>${e.profile??`—`}</td>
                  <td>${P(e.class_id,this.ids)}</td><td class="num num-col">${n(e.turns)}</td>
                  <td class="num num-col">${e.notes?`${e.notes} (${n(e.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${e.acts?T`${n(e.acts)}${e.gave_ups?T` <span class="badge badge-warn">${e.gave_ups} gave up</span>`:o}`:`—`}</td>
                  <td class="num num-col" data-interventions>${n(e.interventions??0)}</td>
                  <td>${e.last_checker?T`<span class="badge ${e.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${e.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${d(new Date(e.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:y(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:T`<label class="field field-search">${g(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${e.map(e=>T`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?T`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:o}
    `}renderSession(e){let t=this.turns,i=this.live.find(t=>t.key===e);if(t==null)return f(4);let a=t.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,s=t.reduce((e,t)=>e+(t.injected?.length??0),0),c=this.events.filter(e=>x(e)).length,l=Te(t);return T`<section class="panel" style="margin-bottom:16px" data-timeline=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${M(e,44)} ${i?T`<span class="badge badge-on">live</span>`:o}</h2>
        <p class="panel-sub">${t.length} turns · ${s} hidden note${s===1?``:`s`} · ${c} monitor act${c===1?``:`s`}${i?` · class now ${i.class_id} · ${n(i.note_tokens)} note tokens replayed a turn`:``}${t.length?` · since ${r(new Date(t[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${A.exportUrl(e,`md`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${g(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${A.exportUrl(e,`json`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${g(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>w({session:null})}>${g(`x`)}</button></div></header>
      ${this.axis(Pe(t,this.events,this.ids,e=>_(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${t.map(e=>{let t=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/a*900)),n=(e.injected?.length??0)>0,r=!!e.shadow,i=(e.events??[]).filter(e=>e.decision!==`PASS`),s=i.some(e=>x(e))?`act`:i.length?`shadow`:null;return T`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${t}px; --cls: var(--series-${ye(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${Ce(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${n?T`<span class="mark"></span>`:r?T`<span class="mark is-shadow"></span>`:o}
            ${s?T`<span class="mark is-monitor ${s===`shadow`?`is-shadow`:``}" data-monitor-mark=${s}></span>`:o}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${l.series.length?T`<qse-ribbon .series=${l.series} .labels=${l.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:o}
      <div class="feed" style="margin-top:10px">${t.slice().reverse().map(e=>He(e,this.ids,e=>this.open=e,Ce(e)+((e.events??[]).some(e=>x(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?T`<div style="margin-top:12px"><qse-cond-journal .session=${e}></qse-cond-journal></div>`:o}
    </section>`}axis(e){let t=e=>`${(e*100).toFixed(3)}%`;return T`<figure class="axis" data-axis aria-label="the session on one time axis">
      <div class="axis-lane axis-acts" aria-label=${`${e.acts.length} monitor decisions`}>
        ${e.acts.map(e=>T`<span class="axis-mark ${e.shadow?`is-shadow`:`is-act`}" style="left:${t(e.x)}" title=${e.title} data-axis-act=${e.shadow?`shadow`:`act`}></span>`)}
        ${e.notes.map(e=>T`<span class="axis-mark is-note ${e.shadow?`is-shadow`:``}" style="left:${t(e.x)}" title=${e.title} data-axis-note></span>`)}
      </div>
      <div class="axis-lane axis-turns" aria-label=${`${e.turns.length} turns`}>
        ${e.turns.map(e=>T`<span class="axis-turn ${e.compaction?`is-compaction`:``}" style="left:${t(e.x0)};width:${t(Math.max(0,e.x1-e.x0))};--cls:var(--series-${e.color})" title=${e.title} data-axis-turn></span>`)}
      </div>
      <div class="axis-lane axis-events" aria-label=${`${e.compactions.length} compactions, ${e.checkers.length} checker results`}>
        ${e.compactions.map(e=>T`<span class="axis-mark is-compaction" style="left:${t(e.x)}" title=${e.title} data-axis-compaction></span>`)}
        ${e.checkers.map(e=>T`<span class="axis-mark ${e.pass?`is-pass`:`is-fail`}" style="left:${t(e.x)}" title=${e.title} data-axis-checker=${e.pass?`pass`:`fail`}></span>`)}
      </div>
      <div class="axis-ticks" aria-hidden="true">${e.ticks.map((n,r)=>T`<span class="num ${r===0?`is-first`:r===e.ticks.length-1?`is-last`:``}" style="left:${t(n.x)}">${n.label.slice(0,5)}</span>`)}</div>
      <figcaption class="muted small">One time axis, ${e.ticks[0]?.label??``} to ${e.ticks[e.ticks.length-1]?.label??``}: ${e.turns.length} turns (the gaps are the client's time), ${e.acts.filter(e=>!e.shadow).length} monitor actions ◆ above (◇ shadow, ${e.acts.filter(e=>e.shadow).length}), ${e.notes.length} router note${e.notes.length===1?``:`s`}, ${e.compactions.length} compaction${e.compactions.length===1?``:`s`} ▲ and ${e.checkers.length} checker result${e.checkers.length===1?``:`s`} below (● pass, ○ fail; the gate numbers are in each one's title).</figcaption>
    </figure>`}tier1(){let e=Fe(this.events);return e?T`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${e.agree} of ${e.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${e.decisions} shadow decisions)</span></p>`:o}};s([i({attribute:!1})],Z.prototype,`ids`,void 0),s([i({attribute:!1})],Z.prototype,`selected`,void 0),s([l()],Z.prototype,`list`,void 0),s([l()],Z.prototype,`live`,void 0),s([l()],Z.prototype,`turns`,void 0),s([l()],Z.prototype,`events`,void 0),s([l()],Z.prototype,`error`,void 0),s([l()],Z.prototype,`query`,void 0),s([l()],Z.prototype,`client`,void 0),s([l()],Z.prototype,`open`,void 0),Z=s([a(`qse-cond-sessions`)],Z);var Q=class extends c{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([A.stats(168),A.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async promote(e,n,r,i,a){if(await ze(`Promote ${r} to ${n===`task`?`act`:`on`} in ${e}?`,[n===`task`?`The session monitor will act on ${r} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${r} decisions instead of only journaling them: ${r===`effort`?`budgets and thinking off go to the engine`:r===`packages`?`skills and packages go into the prompt`:r===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${i.toFixed(2)} over ${a} labelled decisions. A new profile version records it.`],`Promote`))try{let t=await A.doc(`profiles`,e),o=n===`task`?j(t.doc,[`tasks`,r,`mode`],`act`):j(t.doc,[`decisions`,r],`on`),s=await L(()=>A.put(`profiles`,e,o,`${r} to ${n===`task`?`act`:`on`}, precision ${i.toFixed(2)} / ${a}`,t.version));this.msg=`${e} v${s.version}: ${r} promoted`,this.onSaved()}catch(e){this.msg=t(e).message}}async loadAb(){try{this.abVersions=(await A.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await A.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await A.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=t(e).message}}render(){return this.error&&!this.stats?T`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${this.error}</div></div>`:this.stats?T`
      ${this.msg?T`<div class="note note-info" role="status">${this.msg}</div>`:o}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:f(8)}renderPromotion(e){let t=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,r=(e.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return h(`Shadow precision and promotion`,T`<p class="muted small">${ce}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${e.precision.length||r.length?T`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${e.precision.map(e=>{let r=le(e),i=t(e.profile,e.decision);return T`<tr data-precision-row=${`${e.profile}/${e.decision}`}>
                    <td>${e.profile}</td><td class="mono">${e.decision}</td><td><span class="badge ${i===`on`?`badge-on`:i===`shadow`?`badge-shadow`:``}">${i}</span></td>
                    <td class="num num-col">${n(e.decisions)}</td><td class="num num-col">${n(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${r.allowed?``:`muted`}">${r.why}</td>
                    <td><button class="btn btn-sm ${r.allowed?`btn-primary`:``}" ?disabled=${!r.allowed||i!==`shadow`} title=${r.allowed?``:ce} data-promote=${`${e.profile}/${e.decision}`} @click=${()=>this.promote(e.profile??``,`decision`,e.decision,e.precision??0,e.labelled)}>Promote to on</button></td></tr>`})}
                ${r.map(e=>{let t=le(e);return T`<tr data-precision-row=${`task/${e.task}`}><td>—</td><td class="mono">${e.task}</td><td><span class="badge ${e.mode===`act`?`badge-on`:`badge-shadow`}">${e.mode}</span></td>
                    <td class="num num-col">${n(e.decisions)}</td><td class="num num-col">${n(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${t.allowed?``:`muted`}">${t.why}</td>
                    <td><button class="btn btn-sm ${t.allowed?`btn-primary`:``}" ?disabled=${!t.allowed||e.mode!==`shadow`} data-promote=${`task/${e.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,e.task,e.precision??0,e.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:y(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${e.monitor?o:T`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let t=this.reports.find(e=>e.kind===`classifier`);if(!t)return h(`Classifier accuracy`,y(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let n=t.stages??{},i=t.set??{},a=(n[this.stage]??Object.values(n)[0])?.confusion??{},o=i.classes??Object.keys(a);return h(T`Classifier accuracy <span class="muted small num">${i.name??``} · ${i.n??`?`} prompts · ${t.created_at?r(String(t.created_at)):``}</span>`,T`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(n).map(([t,n])=>T`<tr class="req-row ${t===this.stage?`is-open`:``}" @click=${()=>this.stage=t}><td class="mono">${t}</td><td class="num num-col">${e(n.accuracy,1)}</td><td class="num num-col">${n.answered} / ${n.n}</td><td class="num num-col">${n.accuracy_answered==null?`—`:e(n.accuracy_answered,1)}</td><td class="num num-col">${v(n.p50_ms)}</td><td class="num num-col">${v(n.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${o.length&&Object.keys(a).length?T`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...o,`—`].map(e=>T`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${o.map(e=>T`<tr><th>${P(e,this.ids)}</th>${[...o,`—`].map(t=>{let n=a[e]?.[t]??0;return T`<td class="cm-cell ${e===t?`cm-diag`:n?`cm-miss`:``}">${n||``}</td>`})}</tr>`)}</tbody></table></div>`:T`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}`,{id:`cond-classifier`})}renderSuites(){let t=this.reports.filter(e=>e.kind===`suite`),r=t.find(e=>e.file===this.suiteFile)??t[0],i=r?.rows??[],a=_e(this.reports,this.engine),s=me(i),c=r?.verdict,l=r?.comparison??[],u=r?.raw_row,d=r?.build,f=c?.indicative??de(i),p=(e,t=1)=>e==null?`—`:Number.isInteger(e)?n(e):e.toFixed(t);return h(T`TandemLLM against the raw engine ${r?T`<span class="muted small num">night ${r.night??`—`}${d?` · ${d.version??`?`} ${M(d.code_sha,12)}`:``}${r.profile?` · profile ${r.profile}`:``}</span>`:o}`,T`${a.tone===`ok`?T`<p class="small" data-guard-detail=${a.status}>${g(`check`)} ${a.line}${a.detail?T`: <span class="num">${a.detail}</span>`:o}.</p>`:T`<div class="guard-line ${a.tone===`note`?`guard-note`:``}" data-guard-detail=${a.status}>${g(a.tone===`note`?`info`:`warn`)}<span>${a.line}${a.detail?T`. <span class="small">${a.detail}</span>`:o}</span></div>`}
        ${t.length>1?T`<label class="field"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${t.map(e=>T`<option value=${String(e.file)} ?selected=${e===r}>${String(e.night??e.file)} · ${String(e.file)}</option>`)}</select></label>`:o}
        ${c?T`<div class="note" data-eval-verdict>
              <span class="badge ${c.better?`badge-ok`:c.better===!1?`badge-warn`:``}"><strong>${c.better?`Better`:c.better===!1?`Not better`:`Not shown yet`}</strong></span>${f?T` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:o} — ${c.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${u?.status??`—`}${u?.source?` (${u.source})`:``}${r?.driver?` · driver ${r.driver.mode}`:``}${r?.isolation?` · ${r.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:o}
        ${s.length?T`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>build · night</th></tr></thead>
              <tbody>${s.flatMap(t=>{let r=l.find(e=>e.suite===t.suite||e.suite_id===(t.raw??t.tandem)?.suite_id);return[t.raw,t.tandem].filter(Boolean).map((i,a)=>T`<tr data-suite=${t.suite} data-arm=${i.arm}>
                    <td>${a===0?t.suite:``}${a===0&&!t.cmp.ok?T` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${t.cmp.why}</div>`:o}${a===0&&r?T` <span class="badge ${r.verdict===`better`?`badge-ok`:r.verdict===`worse`||r.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${r.verdict}>${r.verdict}</span>`:o}${a===0&&(i.indicative??(i.n??0)<10)?T` <span class="badge" data-row-indicative>indicative</span>`:o}</td>
                    <td class="mono">${fe(i.arm)}${i.pristine===!1?T` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:o}</td>
                    <td class="num num-col">${i.n??`—`}</td>
                    <td class="num num-col">${i.pass_rate==null?`—`:e(i.pass_rate,0)}</td>
                    <td class="num num-col">${i.gates_passed??`—`}${i.gates_total?` / ${i.gates_total}`:``}</td>
                    <td class="num num-col">${p(i.minutes_to_pass)}</td><td class="num num-col">${p(i.wall_minutes)}</td>
                    <td class="num num-col">${i.tokens==null?`—`:n(Math.round(i.tokens))}</td><td class="num num-col">${p(i.decode_tps)}</td>
                    <td class="num num-col">${i.empty_stops??`—`}</td><td class="num num-col">${i.continuations??`—`}</td>
                    <td class="num num-col">${i.interventions??`—`}</td><td class="num num-col">${i.gave_ups??`—`}</td>
                    <td class="num num-col">${i.cache_hit==null?`—`:e(i.cache_hit,0)}</td>
                    <td class="small">${M(i.build,16)} · ${i.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts.</p>`:y(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return h(`A/B of two profile versions`,T`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>T`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>T`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(t=>T`<option value=${t.version} ?selected=${t.version===this[e]}>v${t.version} ${M(t.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?T`<div class="diff" data-ab>${this.abChanges.map(e=>T`<div class="diff-row"><span>${Se(e.path)}</span><span class="diff-before">${M(e.before,80)}</span><span class="diff-after">${M(e.after,80)}</span></div>`)}</div>`:T`<p class="muted small">The two versions are the same.</p>`:o}`,{id:`cond-ab`})}};s([i({attribute:!1})],Q.prototype,`ids`,void 0),s([i({attribute:!1})],Q.prototype,`tiles`,void 0),s([i({attribute:!1})],Q.prototype,`onSaved`,void 0),s([l()],Q.prototype,`stats`,void 0),s([l()],Q.prototype,`reports`,void 0),s([l()],Q.prototype,`engine`,void 0),s([l()],Q.prototype,`error`,void 0),s([l()],Q.prototype,`msg`,void 0),s([l()],Q.prototype,`stage`,void 0),s([l()],Q.prototype,`abProfile`,void 0),s([l()],Q.prototype,`abVersions`,void 0),s([l()],Q.prototype,`abA`,void 0),s([l()],Q.prototype,`abB`,void 0),s([l()],Q.prototype,`abChanges`,void 0),s([l()],Q.prototype,`abCounts`,void 0),s([l()],Q.prototype,`suiteFile`,void 0),Q=s([a(`qse-cond-compare`)],Q);var ut=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`deciders`,label:`Deciders`},{id:`sessions`,label:`Sessions`},{id:`compare`,label:`Compare`}],$=class extends c{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.guard=null,this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=m(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&ut.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r,i]=await Promise.all([A.profiles(),this.meta?Promise.resolve(this.meta):A.meta(),A.config(),A.doc(`router`),A.evals()]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.guard=_e(i.reports,i.engine),this.error=null}catch(e){let n=t(e);this.error=n.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${n.endpoint}: ${n.message}`}}async setMode(e){let n=this.tiles.find(e=>e.name===this.active);if(n&&!n.locked&&(e!==`on`||await ze(`Turn the ${n.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let t=await L(()=>A.setMode(n.name,e,`mode to ${e}`,n.version));this.reloaded=`v${t.version} · reloaded ${t.loaded_at?_(t.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=t(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),t=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return T`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${g(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>w({profile:e.target.value})}>
                ${this.tiles.map(e=>T`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?F([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):o}
            ${this.reloaded?T`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:o}
          </div>
        </div>
        ${this.switchError?T`<div class="guard-line" role="alert">${g(`warn`)} ${this.switchError}</div>`:o}
        ${this.error?T`<div class="guard-line" role="alert">${g(`warn`)} ${this.error}</div>`:o}
        ${this.guard&&this.guard.tone!==`ok`?T`<div class="guard-line ${this.guard.tone===`note`?`guard-note`:``}" data-guard=${this.guard.status}>${g(this.guard.tone===`note`?`info`:`warn`)}<span>${this.guard.line}${this.guard.detail?T`. <span class="small">${this.guard.detail}</span>`:o}</span></div>`:o}
        ${t.map(([e,t])=>T`<div class="guard-line" role="alert">${g(`warn`)} ${e}: ${t.error?.message} (kept version ${t.version})</div>`)}
        ${e?.locked?T`<div class="note" data-locked>${g(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:o}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${ut.map(e=>T`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:o}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return T`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return T`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return T`<qse-cond-monitor .meta=${e.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return T`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return T`<qse-cond-packages .ids=${e.ids}></qse-cond-packages>`;case`deciders`:return T`<qse-cond-deciders></qse-cond-deciders>`;case`sessions`:return T`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`compare`:return T`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};s([i({attribute:!1})],$.prototype,`params`,void 0),s([l()],$.prototype,`tiles`,void 0),s([l()],$.prototype,`defaultProfile`,void 0),s([l()],$.prototype,`meta`,void 0),s([l()],$.prototype,`config`,void 0),s([l()],$.prototype,`ids`,void 0),s([l()],$.prototype,`guard`,void 0),s([l()],$.prototype,`error`,void 0),s([l()],$.prototype,`switching`,void 0),s([l()],$.prototype,`reloaded`,void 0),s([l()],$.prototype,`switchError`,void 0),$=s([a(`qse-conductor`)],$);export{$ as QseConductor};