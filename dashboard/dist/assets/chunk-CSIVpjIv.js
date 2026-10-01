import{A as e,C as t,D as n,E as r,I as i,M as a,N as o,O as s,P as c,S as l,_ as u,a as d,b as f,c as p,d as m,f as h,g,h as _,j as v,k as y,m as b,o as x,p as S,s as C,u as ee,w,x as T,y as te}from"./app-DOH6NdT0.js";var E=`qse-conductor-token`;function ne(){try{return sessionStorage.getItem(E)??localStorage.getItem(E)}catch{return null}}function re(e,t=!1){try{sessionStorage.removeItem(E),localStorage.removeItem(E),e&&(t?localStorage:sessionStorage).setItem(E,e)}catch{}}async function ie(e,t,n){let r={"Content-Type":`application/json`},i=ne();i&&(r.Authorization=`Bearer ${i}`);let a;try{a=await fetch(te()+e,{method:t,headers:r,body:JSON.stringify(n),credentials:`same-origin`,cache:`no-store`})}catch(t){throw new u(e,0,`engine unreachable (${t.message})`,`offline`)}let o={};try{o=await a.json()}catch{}if(a.status===401)throw new u(e,401,`editing needs the admin token`,`token_needed`);if(!a.ok){let t=new u(e,a.status,o.error?.message??`HTTP ${a.status}`,o.error?.type??`error`);throw t.errors=o.error?.errors??[],t}return o}var D=encodeURIComponent,O=(e,t)=>e===`profiles`?`/v1/conductor/profiles/${D(t??``)}`:`/v1/conductor/${e}`,k={config:()=>T(`/v1/conductor/config`),meta:()=>T(`/v1/conductor/meta`),profiles:()=>T(`/v1/conductor/profiles`),doc:(e,t)=>T(O(e,t)),versions:(e,t)=>T(`${O(e,t)}/versions`),version:(e,t,n)=>T(`${O(e,t)}/versions/${n}`),diff:(e,t,n,r)=>T(`${O(e,t)}/diff?from=${n}&to=${r}`),check:(e,t,n)=>T(`${O(e,t)}/check`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({doc:n})}),put:(e,t,n,r,i)=>ie(O(e,t),`PUT`,{doc:n,note:r,base_version:i}),restore:(e,t,n,r)=>ie(`${O(e,t)}/restore`,`POST`,{version:n,note:r}),setMode:(e,t,n,r)=>ie(`/v1/conductor/profiles/${D(e)}/mode`,`POST`,{mode:t,note:n,base_version:r}),roles:()=>T(`/v1/conductor/roles?fresh=1`),roleTest:e=>T(`/v1/conductor/roles/${D(e)}/test`,{method:`POST`,headers:{"Content-Type":`application/json`},body:`{}`}),scan:()=>T(`/v1/conductor/packages/scan`),classify:(e,t,n=!0)=>T(`/v1/conductor/classify`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({text:e,profile:t,tools:n})}),decisions:(e={})=>{let t=new URLSearchParams;return e.since&&t.set(`since`,String(e.since)),t.set(`limit`,String(e.limit??100)),e.session&&t.set(`session`,e.session),e.before&&t.set(`before`,String(e.before)),T(`/v1/conductor/decisions?${t}`)},decision:e=>T(`/v1/conductor/decisions/${D(e)}`),label:(e,t,n=``)=>ie(`/v1/conductor/decisions/${D(e)}/label`,`POST`,{label:t,note:n}),events:(e={})=>{let t=new URLSearchParams;return e.since&&t.set(`since`,String(e.since)),t.set(`limit`,String(e.limit??100)),e.session&&t.set(`session`,e.session),e.task&&t.set(`task`,e.task),e.mode&&t.set(`mode`,e.mode),e.acted&&t.set(`acted`,`1`),T(`/v1/conductor/events?${t}`)},event:e=>T(`/v1/conductor/events/${D(e)}`),stats:(e=24)=>T(`/v1/conductor/stats?hours=${e}`),sessions:()=>T(`/v1/conductor/sessions`),session:e=>T(`/v1/conductor/sessions/${D(e)}`),exportJson:e=>T(`/v1/conductor/sessions/${D(e)}/export?format=json`),exportUrl:(e,t)=>`${te()}/v1/conductor/sessions/${D(e)}/export?format=${t}`,evals:()=>T(`/v1/conductor/evals`),watch:()=>T(`/v1/conductor/watch`)},ae=.9,oe=`on is allowed at precision ≥ ${ae} over ≥ 30 labelled decisions`;function se(e){return e.labelled<30?{allowed:!1,why:`${e.labelled} of 30 labelled decisions`}:e.precision==null||e.precision<.9?{allowed:!1,why:`precision ${ce(e.precision)} is below ${ae}`}:{allowed:!0,why:`precision ${ce(e.precision)} over ${e.labelled}`}}var ce=e=>e==null?`—`:e.toFixed(2);function le(e){return e.some(e=>e!=null&&(e.indicative??(e.n??0)<10))}function ue(e){return e===`raw`?`raw`:`TandemLLM`}function de(e,t){return!e.build||!t.build||e.build!==t.build?{ok:!1,why:`builds ${e.build??`?`} and ${t.build??`?`}`}:!e.night||!t.night||e.night!==t.night?{ok:!1,why:`nights ${e.night??`?`} and ${t.night??`?`}`}:{ok:!0,why:null}}function fe(e){let t=new Map;for(let n of e){let e=t.get(n.suite)??{raw:null,tandem:null};n.arm===`raw`?e.raw=n:e.tandem=n,t.set(n.suite,e)}return[...t.entries()].map(([e,t])=>({suite:e,...t,cmp:t.raw&&t.tandem?de(t.raw,t.tandem):{ok:!1,why:t.raw?`no TandemLLM arm`:`no raw arm`}}))}function pe(e,t){return e.some(e=>{if(e.kind!==`speed_guard`)return!1;let n=e.build??{};return t.code_sha&&n.code_sha===t.code_sha||!t.code_sha&&t.version&&n.version===t.version})?null:`speed guard not run for build ${t.version??`unknown`}`}var me=[`cobalt`,`orange`,`aqua`,`violet`,`yellow`,`red`];function A(e,t){let n=e==null?-1:t.indexOf(e);return n<0?`muted`:me[n%me.length]}function he(e,t,n=4){let r=Object.entries(e.by_class).sort((e,t)=>t[1]-e[1]),i=r.slice(0,n).map(([e])=>e),a=i.map(n=>({name:n,color:A(n,t),values:e.hourly.map(e=>e.by_class[n]??0)}));return r.length>n&&a.push({name:`other`,color:`muted`,values:e.hourly.map(e=>Object.entries(e.by_class).filter(([e])=>!i.includes(e)).reduce((e,[,t])=>e+t,0))}),a}function ge(e){return JSON.parse(JSON.stringify(e))}function j(e,t,n){if(!t.length)return n;let[r,...i]=t,a=e??(typeof r==`number`?[]:{}),o=Array.isArray(a)?[...a]:{...a};return o[r]=j(a[r],i,n),o}function _e(e){return e.replace(/^\$\.?/,``).replace(/\[/g,` › `).replace(/\]/g,``).replace(/\./g,` › `)||`(document)`}function M(e,t=80){if(e==null)return`—`;let n=typeof e==`string`?e:JSON.stringify(e);return n.length>t?n.slice(0,t-1)+`…`:n}function ve(e){let t=[];e.effort&&t.push(e.thinking&&e.thinking.startsWith(`off`)?`thinking off`:e.effort);for(let n of e.injected??[]){let e=n.parts.filter(e=>e.kind!==`effort`).map(e=>e.name);t.push(`${e.length?e.join(` + `):`note`} (${n.tokens.toLocaleString(`en-US`)} tokens)`)}return e.replayed?.count&&t.push(`replayed ${e.replayed.count}`),e.mode===`shadow`&&t.push(`shadow`),t.join(` · `)||`—`}var ye={decider:`decider`,qwen:`Qwen`,sticky:`kept (same task)`,structural:`request shape`,fallback:`fail-open`,profile:`profile default`};function be(e){let t=new Map;e.forEach((n,r)=>{for(let i of n.next_checker??[])for(let n of i.readings){let a=`${i.checker} ${n.gate}`;t.has(a)||t.set(a,Array(e.length).fill(null)),t.get(a)[r]=n.value}});let n=[`cobalt`,`orange`,`aqua`,`violet`,`yellow`,`red`];return{series:[...t.entries()].slice(0,6).map(([e,t],r)=>({name:e,color:n[r%n.length],values:t})),labels:e.map((e,t)=>`turn ${t+1}`)}}var xe={thinking:`cobalt`,"end of turn":`orange`,"tool calls":`aqua`,session:`violet`,other:`muted`};function Se(e){let t=[...new Set(e.hourly.flatMap(e=>Object.keys(e.by_family)))],n=Object.keys(xe);return t.sort((e,t)=>n.indexOf(e)-n.indexOf(t)),t.map(t=>({name:t,color:xe[t]??`muted`,values:e.hourly.map(e=>e.by_family[t]??0)}))}function Ce(e,t){let n=e?.properties?.tasks?.additionalProperties,r=(typeof n==`object`&&n?n.oneOf??[]:[]).find(e=>e.properties?.rule?.const===t);if(!r?.properties)return null;let{rule:i,mode:a,...o}=r.properties;return{...r,properties:o,required:(r.required??[]).filter(e=>e!==`rule`&&e!==`mode`)}}function we(e){let t=Object.keys(e.tasks),n=t.filter(e=>/^C\d/.test(e)).sort();return[...n,...t.filter(e=>!n.includes(e))]}function Te(e){return/^S\d/.test(e)}function Ee(e){let t=[];for(let n of e.gates)t.push(`${n.checker} (${n.gate}) ${n.values.map(e=>N(e)).join(`, `)}${n.noted?` · noted`:``}`);return e.compaction?.watching&&t.push(`after compaction ${e.compaction.count}: ${e.compaction.steps} step${e.compaction.steps===1?``:`s`}, no re-read yet`),e.progress&&t.push(e.progress.last?`last progress ${N(e.progress.minutes)} min ago (${e.progress.last})`:`no progress for ${N(e.progress.minutes)} min`),e.client_gone?t.push(`client gone after ${N(e.idle_minutes??0)} min`):e.idle_minutes!=null&&t.push(`idle ${N(e.idle_minutes)} min`),t.join(` · `)||`nothing to watch yet`}var N=e=>String(Number.isInteger(e)?e:Math.round(e*100)/100);function De(e){return e.split(`
`).map(e=>e.trim()).filter(Boolean)}function Oe(e){return e.source===`structural`||e.shadow?.source===`structural`}function ke(e,t,n,r,i=5){let a=e=>e.ts_ms,o=e=>e.ts_ms+Math.max(0,e.outcome?.total_ms??0),s=t.filter(e=>e.ts_ms!=null&&(C(e)||p(e))),c=[...e.map(a),...e.map(o),...s.map(e=>e.ts_ms)],l=c.length?Math.min(...c):0,u=c.length?Math.max(...c):1,d=Math.max(1,u-l),f=e=>Math.min(1,Math.max(0,(e-l)/d)),m=[...e].sort((e,t)=>e.ts_ms-t.ts_ms),h={start:l,end:u,turns:[],acts:[],notes:[],compactions:[],checkers:[],ticks:[]};m.forEach((e,t)=>{let i=Oe(e);h.turns.push({id:e.id,x0:f(a(e)),x1:f(o(e)),color:A(e.class_id,n),compaction:i,title:`${r(e.ts_ms)} · ${e.class_id??`—`} · ${e.outcome?.finish_reason??`running`}`}),i&&h.compactions.push({id:e.id,x:f(a(e)),title:`${r(e.ts_ms)} compaction (the client summarised the session)`});for(let t of e.injected??[])h.notes.push({id:`${e.id}:${t.pos}`,x:f(a(e)),shadow:!1,title:`${r(e.ts_ms)} router note, ${t.tokens} tokens (${t.reason})`});if(!e.injected?.length&&e.shadow&&e.shadow.notes?.length&&h.notes.push({id:`${e.id}:shadow`,x:f(a(e)),shadow:!0,title:`${r(e.ts_ms)} router note in shadow`}),e.next_checker?.length){let n=m[t+1]?a(m[t+1]):o(e),i=e.next_checker.every(e=>e.verdict===`PASS`),s=e.next_checker.map(e=>`${e.checker} ${e.verdict}${e.readings.length?`: ${e.readings.map(e=>`(${e.gate}) ${e.value} ${e.op} ${e.limit}`).join(`, `)}`:``}`).join(`; `);h.checkers.push({id:e.id,x:f(n),pass:i,title:`${r(n)} ${s}`})}});for(let e of s)h.acts.push({id:e.id??`${e.ts_ms}`,x:f(e.ts_ms),shadow:!C(e),task:e.task??`—`,title:`${r(e.ts_ms)} ${x({task:e.task,decision:e.decision,mode:e.mode,trigger:e.trigger??null,cap:e.cap??null})}`});for(let e=0;e<=i;e++)h.ticks.push({x:e/i,label:r(l+d*e/i)});return h}function Ae(e){let t=e.filter(e=>e.tier===1&&p(e));if(!t.length)return null;let n=t.filter(e=>e.label===`right`||e.label===`wrong`);return{decisions:t.length,labelled:n.length,agree:n.filter(e=>e.label===`right`).length}}function P(e,t,n){let r=A(e,t);return i`<span class="cls" style="--cls: var(--series-${r})" data-class=${e??`none`}>${n??e??`—`}</span>`}function F(e,t=.6){return e==null?i`<span class="conf muted">—</span>`:i`<span class="conf" title=${`confidence ${e.toFixed(4)}`}><span class="conf-bar ${e<t?`is-low`:``}"><span style="width:${Math.round(e*100)}%"></span></span>${e.toFixed(2)}</span>`}function I(e,t,n,r={}){return i`<div class="seg ${r.small?`seg-sm`:``} ${r.cls??``}" role="radiogroup" aria-label=${r.label??`mode`}>
    ${e.map(e=>i`<button type="button" class="seg-btn mode-${e} ${e===t?`is-on`:``}" role="radio" aria-checked=${e===t?`true`:`false`} ?disabled=${r.disabled} @click=${()=>e!==t&&n(e)}>${e}</button>`)}
  </div>`}function je(e){return e?ye[e]??e:`—`}var L=class extends e{constructor(...e){super(...e),this.kind=`confirm`,this.heading=``,this.lines=[],this.confirmLabel=`Confirm`,this.done=()=>void 0,this.error=null}submit(e){if(e.preventDefault(),this.kind===`token`){let e=(this.querySelector(`#cond-token`)?.value??``).trim();if(!e){this.error=`Enter the admin token.`;return}this.done(!0,e,this.querySelector(`#cond-remember`)?.checked??!1)}else this.done(!0)}render(){return i`<div class="modal-back" @click=${()=>this.done(!1)}></div>
      <form class="modal" role="dialog" aria-modal="true" aria-label=${this.heading} @submit=${e=>this.submit(e)}>
        <h2>${this.heading}</h2>
        ${this.lines.map(e=>i`<p class="muted small">${e}</p>`)}
        ${this.kind===`token`?i`<label class="field field-block"><span>Admin token</span><input id="cond-token" type="password" autocomplete="current-password" required autofocus /></label>
              <label class="switch"><input id="cond-remember" type="checkbox" /> remember on this device</label>`:c}
        ${this.error?i`<p class="err-ink small" role="alert">${this.error}</p>`:c}
        <div class="modal-actions">
          <button type="button" class="btn" @click=${()=>this.done(!1)}>Cancel</button>
          <button type="submit" class="btn btn-primary" data-confirm>${this.confirmLabel}</button>
        </div>
      </form>`}};y([a({attribute:!1})],L.prototype,`kind`,void 0),y([a({attribute:!1})],L.prototype,`heading`,void 0),y([a({attribute:!1})],L.prototype,`lines`,void 0),y([a({attribute:!1})],L.prototype,`confirmLabel`,void 0),y([a({attribute:!1})],L.prototype,`done`,void 0),y([v()],L.prototype,`error`,void 0),L=y([o(`qse-cond-dialog`)],L);function Me(e){return new Promise(t=>{let n=document.createElement(`qse-cond-dialog`);e(n),n.done=(e,r,i)=>{n.remove(),t({ok:e,token:r,remember:i})},document.body.appendChild(n)})}function Ne(e,t,n){return Me(r=>{r.kind=`confirm`,r.heading=e,r.lines=t,r.confirmLabel=n}).then(e=>e.ok)}async function Pe(e=`Changing the conductor needs the admin token (ops/make-secrets.sh made it; it is in secrets.env on the box).`){let t=await Me(t=>{t.kind=`token`,t.heading=`Unlock editing`,t.lines=[e,`It stays in this browser tab, or on this device if you tick the box, and is sent only with a change.`],t.confirmLabel=`Unlock`});return t.ok&&t.token&&re(t.token,!!t.remember),t.ok&&!!t.token}async function R(e){try{return await e()}catch(t){if(t instanceof u&&t.type===`token_needed`&&(ne()&&re(null),await Pe()))return await e();throw t}}var z=class{constructor(e,t,n=null){this.host=e,this.kind=t,this.name=n,this.base=null,this.draft=null,this.loading=!0,this.loadError=null,this.errors=[],this.changes=[],this.note=``,this.saving=!1,this.saved=null,this.saveError=null,this.versions=null,this.showVersions=!1,this.showDiff=!1,this.onSaved=null,this.checkTimer=null}get dirty(){return this.changes.length>0}get locked(){return this.base?.locked??null}async load(){this.loading=!0,this.host.requestUpdate();try{this.base=await k.doc(this.kind,this.name),this.draft=ge(this.base.doc),this.errors=[],this.changes=[],this.loadError=null}catch(e){let t=f(e);this.loadError=`${t.endpoint}: ${t.message}`}finally{this.loading=!1,this.host.requestUpdate()}}set(e){this.draft=e,this.saved=null,this.host.requestUpdate(),this.checkTimer&&clearTimeout(this.checkTimer),this.checkTimer=setTimeout(()=>void this.check(),250)}async check(){if(this.draft){try{let e=await k.check(this.kind,this.name,this.draft);this.errors=e.errors,this.changes=e.changes}catch(e){this.errors=[f(e).message]}this.host.requestUpdate()}}reset(){this.base&&(this.draft=ge(this.base.doc),this.errors=[],this.changes=[],this.host.requestUpdate())}async save(){if(!this.draft||!this.base)return!1;if(!this.note.trim())return this.saveError=`Write a one-line change note first.`,this.host.requestUpdate(),!1;this.saving=!0,this.saveError=null,this.host.requestUpdate();try{let e=await R(()=>k.put(this.kind,this.name,this.draft,this.note.trim(),this.base.version));return this.saved=`version ${e.version}, reloaded ${e.loaded_at?s(e.loaded_at):`now`}`,this.note=``,await this.load(),this.showVersions&&await this.loadVersions(),this.onSaved?.(),!0}catch(e){let t=e;return this.saveError=t.status===409?`${t.message}. Reload to see it.`:t.message,t.errors?.length&&(this.errors=t.errors),!1}finally{this.saving=!1,this.host.requestUpdate()}}async loadVersions(){try{this.versions=(await k.versions(this.kind,this.name)).versions}catch(e){this.saveError=f(e).message}this.host.requestUpdate()}async restore(e){if(await Ne(`Restore version ${e}?`,[`A new version equal to version ${e} becomes current, with the note "restore of v${e}". The server loads it for the next request.`],`Restore`))try{let t=await R(()=>k.restore(this.kind,this.name,e));this.saved=`version ${t.version} (restore of v${e}), reloaded ${t.loaded_at?s(t.loaded_at):`now`}`,await this.load(),await this.loadVersions(),this.onSaved?.()}catch(e){this.saveError=f(e).message,this.host.requestUpdate()}}};function B(e){return e.errors.length?i`<ul class="errors" role="alert" data-errors>${e.errors.slice(0,12).map(e=>i`<li>${e}</li>`)}</ul>`:c}function Fe(e){return e.length?i`<div class="diff" data-diff>
    ${e.map(e=>i`<div class="diff-row"><span>${_e(e.path)}</span><span class="diff-before">${M(e.before,160)}</span><span class="diff-after">${M(e.after,160)}</span></div>`)}
  </div>`:i`<p class="muted small">No changes.</p>`}function V(e,n){let r=e.base?.meta;return e.locked?i`<div class="savebar" data-savebar><span class="badge badge-lock">${h(`warn`)} locked</span><span class="status">${e.locked}. Every control is read-only.</span></div>`:i`
    ${e.showDiff?i`<section class="panel" data-diff-panel><header class="panel-head"><h2 class="panel-title">Changes against live (v${r?.version??0})</h2></header>${Fe(e.changes)}</section>`:c}
    ${e.showVersions?i`<section class="panel" data-versions><header class="panel-head"><h2 class="panel-title">Versions of ${n}</h2></header>
          ${e.versions==null?i`<p class="muted small">Loading…</p>`:i`<div class="table-scroll"><table class="table">
                <thead><tr><th>version</th><th>note</th><th>author</th><th>when</th><th></th></tr></thead>
                <tbody>${e.versions.map(n=>i`<tr data-version=${n.version}>
                    <td class="num">v${n.version}${n.current?i` <span class="badge badge-on">live</span>`:c}</td>
                    <td class="cell-wrap">${n.note??`—`}</td>
                    <td class="muted">${n.author??`—`}</td>
                    <td class="num muted">${n.created_at?t(n.created_at):`built in`}</td>
                    <td>${n.current?c:i`<button class="btn btn-sm" @click=${()=>e.restore(n.version)}>Restore</button>`}</td>
                  </tr>`)}</tbody></table></div>`}
        </section>`:c}
    <div class="savebar" data-savebar>
      <span class="status num" title=${r?.note??``}>live v${r?.version??0}${r?.loaded_at?i` · reloaded ${s(r.loaded_at)}`:c}</span>
      <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>e.note=t.target.value} aria-label="change note" maxlength="200" />
      <button class="btn btn-primary" ?disabled=${!e.dirty||e.saving||e.errors.length>0} @click=${()=>e.save()} data-save>${e.saving?`Saving…`:`Save as new version`}</button>
      <button class="btn btn-ghost" ?disabled=${!e.dirty} @click=${()=>e.reset()}>Reset</button>
      <button class="btn btn-ghost ${e.showDiff?`is-on`:``}" @click=${()=>{e.showDiff=!e.showDiff,e.check()}}>Diff (${e.changes.length})</button>
      <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():e.host.requestUpdate()}} data-versions-btn>Versions</button>
      ${e.saved?i`<span class="status" role="status" data-saved>${h(`check`)} ${e.saved}</span>`:c}
      ${e.saveError?i`<span class="err-ink small" role="alert">${e.saveError}</span>`:c}
    </div>`}var H=class extends e{constructor(...e){super(...e),this.row=null,this.ids=[],this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.row){this.labelling=!0;try{await R(()=>k.label(this.row.id,e)),this.row={...this.row,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=f(e).message}finally{this.labelling=!1}}}render(){let e=this.row;if(!e)return c;let n=e.stages.find(e=>!e.error&&!e.below);return i`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="decision" data-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title">${P(e.class_id,this.ids)} ${e.mode===`shadow`?i`<span class="badge badge-shadow">shadow</span>`:i`<span class="badge badge-on">on</span>`}</div>
            <div class="muted small num">${t(new Date(e.ts_ms).toISOString())} · ${e.profile} v${e.profile_version} · router v${e.router_version}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${h(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">
            <h3>Decision</h3>
            <dl class="kv">
              <div><dt>class</dt><dd>${e.class_id??`—`} ${F(e.confidence)}</dd></div>
              <div><dt>decided by</dt><dd>${je(e.source)}${e.kept?i` <span class="badge badge-warn">${e.kept.replace(/_/g,` `)}</span>`:c}</dd></div>
              <div><dt>classifier time</dt><dd class="num">${r(e.classifier_ms)}</dd></div>
              <div><dt>effort</dt><dd>${e.effort??`unchanged`}${e.budget==null?c:i` <span class="muted num">budget ${e.budget}</span>`} <span class="muted">(${e.budget_source??`server`})</span></dd></div>
              <div><dt>thinking</dt><dd>${e.thinking??`on`}</dd></div>
              <div><dt>modes</dt><dd class="num">${Object.entries(e.decisions??{}).map(([e,t])=>`${e} ${t}`).join(` · `)}</dd></div>
            </dl>
          </section>
          ${e.stages.length?i`<section class="drawer-section"><h3>Classifier stages</h3><div class="stage-list">
                ${e.stages.map(e=>i`<div class="stage ${e===n?`is-accepted`:``} ${e.error?`is-error`:``}">
                    <strong>${e.stage===`decider`?`decider`:`Qwen`}</strong>
                    <div>${e.error?i`<span class="err-ink">${e.error}</span>`:i`${e.label} ${F(e.confidence,e.accept??.6)}${e.below?i` <span class="muted small">below ${e.accept}</span>`:c}`}
                      ${e.probs&&Object.keys(e.probs).length>1?i`<div class="probs">${Object.entries(e.probs).sort((e,t)=>t[1]-e[1]).slice(0,6).map(([e,t])=>i`<div class="prob"><span>${e}</span><span class="prob-bar"><span style="width:${Math.round(t*100)}%"></span></span><span>${t.toFixed(2)}</span></div>`)}</div>`:c}
                    </div>
                    <span class="num muted">${r(e.ms)}${e.cached?` (cached)`:``}</span>
                  </div>`)}</div></section>`:c}
          ${(e.injected??[]).map(e=>i`<section class="drawer-section" data-injected>
              <h3>Injected at token ${e.pos.toLocaleString(`en-US`)} · ${e.tokens.toLocaleString(`en-US`)} tokens ${ee(e.text,`Copy the injected text`)}</h3>
              <p class="muted small">Because ${e.reason}. Parts: ${e.parts.map(e=>`${e.kind} ${e.name}`).join(`, `)}. The client never saw this text.</p>
              <pre class="note-text">${e.text}</pre>
            </section>`)}
          ${e.replayed?.count?i`<p class="muted small">Replayed ${e.replayed.count} earlier note(s), ${e.replayed.tokens.toLocaleString(`en-US`)} tokens, at the same positions, so the resident prefix kept matching.</p>`:c}
          ${e.events?.length?i`<section class="drawer-section" data-monitor-events><h3>The session monitor in this request</h3>
                ${e.events.map(e=>Re(e))}</section>`:c}
          ${e.shadow?i`<section class="drawer-section"><h3>Shadow (not applied)</h3><pre class="note-text">${JSON.stringify(e.shadow,null,2)}</pre></section>`:c}
          ${e.digest?i`<section class="drawer-section"><h3>What the classifier read</h3><pre class="note-text">${e.digest}</pre></section>`:c}
          <section class="drawer-section">
            <h3>Outcome</h3>
            ${e.outcome?i`<dl class="kv">
                  <div><dt>finish</dt><dd>${e.outcome.finish_reason??`running`}</dd></div>
                  <div><dt>tool calls</dt><dd class="num">${e.outcome.tool_calls??0} ${(e.outcome.tool_names??[]).join(`, `)}</dd></div>
                  <div><dt>reasoning tokens</dt><dd class="num">${e.outcome.reasoning_tokens??`—`}</dd></div>
                  <div><dt>time</dt><dd class="num">${r(e.outcome.total_ms)}</dd></div>
                  ${e.next_checker?.length?i`<div><dt>next checker</dt><dd>${e.next_checker.map(e=>i`<span class="badge ${e.verdict===`PASS`?`badge-ok`:`badge-warn`}">${e.verdict}</span> <span class="muted small">${e.checker} ${e.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:c}
                </dl>`:i`<p class="muted small">The turn is still running.</p>`}
          </section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${e.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${h(`check`)} this was right</button>
              <button class="btn ${e.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${h(`x`)} this was wrong</button>
              ${e.label?i`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:c}
              ${this.msg?i`<span class="muted small" role="status">${this.msg}</span>`:c}
            </div>
            <p class="muted small">Labels feed the precision on the Compare screen and the promotion rule.</p>
          </section>
          <p class="muted small num">decision ${e.id} · request ${e.request_id??`—`} · session ${M(e.session_key,40)}</p>
        </div>
      </aside>`}};y([a({attribute:!1})],H.prototype,`row`,void 0),y([a({attribute:!1})],H.prototype,`ids`,void 0),y([a({attribute:!1})],H.prototype,`close`,void 0),y([v()],H.prototype,`labelling`,void 0),y([v()],H.prototype,`msg`,void 0),H=y([o(`qse-cond-decision`)],H);function Ie(e,t,n,r){return i`<button class="feed-row" data-decision=${e.id} @click=${()=>n(e)}>
    <span class="feed-time">${s(new Date(e.ts_ms).toISOString())}</span>
    <span class="feed-cls">${P(e.class_id,t)}</span>
    <span class="feed-conf">${F(e.confidence)} <span class="muted small">${je(e.source)}</span></span>
    <span class="feed-what">${r}</span>
    <span class="feed-act">${e.acted?i`<span class="badge badge-on">acted</span>`:c}${e.mode===`shadow`||e.shadow?i` <span class="badge badge-shadow">shadow</span>`:c}${!e.acted&&!(e.mode===`shadow`||e.shadow)?i`<span class="badge">pass</span>`:c}</span>
  </button>`}function Le(e){return C(e)?i`<span class="badge badge-on">acted</span>`:e.mode===`shadow`&&e.decision!==`PASS`?i`<span class="badge badge-shadow">shadow</span>`:i`<span class="badge">pass</span>`}function Re(e){let t=(e.hidden_tokens??0)>0;return i`<div class="event-block" data-event=${e.id}>
    <div class="event-head">${Le(e)} <strong>${x(e)}</strong></div>
    <dl class="kv">
      <div><dt>safe point</dt><dd>${e.sp??`—`} · rule ${e.rule??`—`} · policy v${e.policy_version??`—`}</dd></div>
      ${e.tokens_at_cut==null?c:i`<div><dt>reasoning tokens</dt><dd class="num">${e.tokens_at_cut.toLocaleString(`en-US`)}</dd></div>`}
      ${e.outcome?i`<div><dt>outcome</dt><dd>${e.outcome.kind??`—`}${e.outcome.finish?i` <span class="muted">(${e.outcome.finish})</span>`:c}</dd></div>`:c}
      ${e.next_checker?.length?i`<div><dt>next checker</dt><dd>${e.next_checker.map(e=>i`<span class="badge ${e.verdict===`PASS`?`badge-ok`:`badge-warn`}">${e.verdict}</span> <span class="muted small">${e.checker} ${e.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:c}
    </dl>
    ${e.text?i`<h4 class="small">${t?i`Hidden text at token ${(e.position??0).toLocaleString(`en-US`)} · ${(e.hidden_tokens??0).toLocaleString(`en-US`)} tokens`:`Visible text`} ${ee(e.text,`Copy the text`)}</h4><pre class="note-text">${e.text}</pre>`:c}
    ${e.detail&&Object.keys(e.detail).length?i`<details><summary class="muted small">detail</summary><pre class="note-text">${JSON.stringify(e.detail,null,2)}</pre></details>`:c}
  </div>`}function ze(e,t){return i`<button class="feed-row" data-event-row=${e.id} @click=${()=>t(e)}>
    <span class="feed-time">${s(new Date(e.ts_ms).toISOString())}</span>
    <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
    <span class="feed-conf muted small">${e.sp??``}</span>
    <span class="feed-what">${x(e)}${(e.hidden_tokens??0)>0?i` <span class="muted small num">${e.hidden_tokens} hidden tokens</span>`:c}</span>
    <span class="feed-act">${Le(e)}</span>
  </button>`}var U=class extends e{constructor(...e){super(...e),this.ev=null,this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.ev){this.labelling=!0;try{await R(()=>k.label(this.ev.id,e)),this.ev={...this.ev,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=f(e).message}finally{this.labelling=!1}}}render(){let e=this.ev;return e?i`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="monitor decision" data-event-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title"><span class="badge mono">${e.task}</span> ${Le(e)}</div>
            <div class="muted small num">${t(new Date(e.ts_ms).toISOString())} · ${e.profile??`no profile`} · ${e.client_kind??`—`}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${h(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">${Re(e)}</section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${e.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${h(`check`)} this was right</button>
              <button class="btn ${e.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${h(`x`)} this was wrong</button>
              ${e.label?i`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:c}
              ${this.msg?i`<span class="muted small" role="status">${this.msg}</span>`:c}
            </div>
            <p class="muted small">Labels on shadow decisions feed the task's precision and its promotion to act (Compare).</p>
          </section>
          <p class="muted small num">event ${e.id} · request ${e.request_id??`—`} · session ${e.session_key?i`<a href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${M(e.session_key,40)}</a>`:`—`}</p>
        </div>
      </aside>`:c}};y([a({attribute:!1})],U.prototype,`ev`,void 0),y([a({attribute:!1})],U.prototype,`close`,void 0),y([v()],U.prototype,`labelling`,void 0),y([v()],U.prototype,`msg`,void 0),U=y([o(`qse-cond-event`)],U);var W=class extends e{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=d(()=>this.loadStats(),2e4),this.feedPoll=d(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await k.stats(24),this.error=null}catch(e){let t=f(e);this.error=`${t.endpoint}: ${t.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await k.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await k.event(e)).event}catch(e){this.error=f(e).message}}renderMonitor(e){let t=Se(e);return i`<div class="grid-2 grid-2-wide">
      ${S(`Monitor acts an hour, last 24 h`,e.acts?i`<qse-ribbon .series=${t} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:m(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${S(`Last actions of the monitor`,e.last_actions.length?i`<div class="feed" data-last-actions>${e.last_actions.map(e=>i`<button class="feed-row" data-action=${e.id} @click=${()=>this.openAction(e.id)}>
                <span class="feed-time">${s(new Date(e.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
                <span class="feed-conf muted small">${e.outcome?.kind??`running`}</span>
                <span class="feed-what">${x({task:e.task,decision:e.decision,mode:`act`,trigger:e.trigger,cap:e.cap})}</span>
                <span class="feed-act">${e.session_key?i`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:c}</span>
              </button>`)}</div>`:i`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:i`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${h(`external`)}</a>`})}
    </div>`}render(){let e=this.stats;if(this.error&&!e)return i`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!e)return b(6);let t=e.classifier,a=t.decider,o=t.qwen,s=a?.calls?a:o?.calls?o:a,l=Object.entries(e.roles),u=l.filter(([,e])=>e.up).length,d=e.monitor;return i`
      <div class="stats" data-tiles>
        ${_({label:`decisions, 24 h`,value:w(e.decisions),sub:i`<span class="num">${w(e.acted)}</span> acted · <span class="num">${w(e.shadow)}</span> in shadow`,title:`${e.injections} notes, ${e.injected_tokens} tokens injected`})}
        ${_({label:`monitor, 24 h`,value:d?w(d.acts):`—`,sub:d?i`acts · <span class="num">${w(d.shadow)}</span> in shadow · <span class="num" data-gave-up>${w(d.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!d,warn:(d?.gave_up??0)>0,title:d?`${w(d.hidden_tokens)} hidden tokens`:void 0})}
        ${_({label:`classifier p50 / p95${!a?.calls&&o?.calls?` (Qwen)`:``}`,value:s?.p50_ms==null?`—`:`${Math.round(s.p50_ms)} / ${Math.round(s.p95_ms??0)}`,unit:s?.p50_ms==null?void 0:`ms`,sub:i`decider ${w(a?.calls??0)} calls${o?i` · Qwen ${w(o.calls)} (${r(o.p50_ms)})`:c} · fail-open ${n(e.fail_open_rate,1)}`,warn:(e.fail_open_rate??0)>.1})}
        ${_({label:`roles online`,value:`${u} / ${l.length}`,sub:i`${l.map(([e,t])=>i`<span title=${t.error??``} class=${t.up?``:t.mode===`off`?`muted`:`err-ink`} data-role=${e}>${t.up?`●`:`○`} ${e} </span>`)}`,warn:l.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${S(`Decisions an hour, last 24 h`,e.decisions?i`<qse-ribbon .series=${he(e,this.ids)} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:m(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${S(`Task classes, 24 h`,i`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${e.classes.map(e=>i`<tr><td>${P(e.class_id,this.ids)}</td><td class="num num-col">${w(e.decisions)}</td><td class="num num-col">${e.labelled?`${e.agree} / ${e.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(e.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(e.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${S(i`Live decisions ${this.paused?i`<span class="badge">paused</span>`:i`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?i`<div class="feed" data-feed>${this.feed.map(e=>Ie(e,this.ids,e=>this.open=e,ve(e)))}</div>`:i`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:i`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${h(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${h(`external`)}</a>`})}
        ${S(`Profiles by client`,i`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(t=>i`<button class="pick" @click=${()=>g({screen:`profiles`,edit:e.name})}><span class="muted">${t}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(e=>i`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${d?this.renderMonitor(d):c}
      ${this.open?i`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:c}
      ${this.openEvent?i`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:c}
    `}};y([a({attribute:!1})],W.prototype,`ids`,void 0),y([a({attribute:!1})],W.prototype,`tiles`,void 0),y([a({attribute:!1})],W.prototype,`active`,void 0),y([v()],W.prototype,`stats`,void 0),y([v()],W.prototype,`feed`,void 0),y([v()],W.prototype,`error`,void 0),y([v()],W.prototype,`open`,void 0),y([v()],W.prototype,`openEvent`,void 0),y([v()],W.prototype,`paused`,void 0),W=y([o(`qse-cond-overview`)],W);var Be=[`off`,`low`,`medium`,`high`,`xhigh`],G=class extends e{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new z(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),k.doc(`packages`).then(e=>this.packages=Object.keys(e.doc.packages)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(j(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await k.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=f(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return b(8);if(e.loadError)return i`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=t.classes.map(e=>e.id);return i`
      ${this.renderTest(n)}
      <details class="sec" open data-sec="classes">
        <summary>Task classes <span class="sec-sub">${t.classes.length} classes; a classifier sees only the descriptions</span></summary>
        <div class="sec-body">
          ${t.classes.map((e,r)=>this.renderClass(e,r,t.classes.length,n))}
          <button class="btn" data-add-class @click=${()=>this.addClass()}>${h(`plus`)} Add a class</button>
        </div>
      </details>
      <details class="sec" data-sec="classifier">
        <summary>Classifier stages <span class="sec-sub">asked in order until one answers at its threshold</span></summary>
        <div class="sec-body">
          ${t.classifier.stages.map((e,n)=>i`<div class="row-card" data-stage=${e.kind}>
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
                ${[`keep`,`classify`,...n].map(e=>i`<option value=${e} ?selected=${e===t.structural.compaction}>${e===`keep`?`keep the session class`:e===`classify`?`ask a classifier`:`class ${e}`}</option>`)}
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
          ${Be.map(e=>i`<div class="row-card" data-effort=${e}>
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
      ${B(e)}
      ${V(e,`the router`)}
    `}renderTest(e){let t=this.result;return i`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?i`<span class="err-ink small" role="alert">${this.testError}</span>`:c}
      </div>
      ${t?t.active?i`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${P(t.class_id??null,e)} ${F(t.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${je(t.source??null)}${t.kept?i` <span class="badge badge-warn">${t.kept}</span>`:c} <span class="muted num">${r(t.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${t.effort??`unchanged`} · thinking ${t.thinking??`on`}${t.budget==null?``:` · budget ${t.budget}`}</dd></div>
                  <div><dt>profile</dt><dd>${t.profile} (${t.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(t.stages??[]).map(e=>i`<div class="stage ${e.error?`is-error`:e.below?``:`is-accepted`}"><strong>${e.stage===`decider`?`decider`:`Qwen`}</strong><span>${e.error??i`${e.label} ${F(e.confidence,e.accept??.6)}`}</span><span class="num muted">${r(e.ms)}</span></div>`)}</div>
              </div>
              <div>${t.note?i`<p class="muted small">The model would see this at the end of the last turn (${(t.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${t.note}</pre>`:i`<p class="muted small">No note: ${t.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:i`<p class="muted small" style="margin-top:12px">Profile ${t.profile} is raw (${t.mode}): the conductor does nothing for it.</p>`:c}
    </section>`}renderClass(e,t,n,r){let a=e.policy,o=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return i`<details class="row-card" data-class-row=${e.id} ?open=${o}>
      <summary>
        <span>${P(e.id,r,e.label)} <span class="muted small mono">${a.effort??`medium`}${(a.packages??[]).length?` · ${(a.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===0} @click=${()=>this.moveClass(t,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===n-1} @click=${()=>this.moveClass(t,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n<=1} @click=${()=>this.removeClass(t)} aria-label=${`remove ${e.id}`}>${h(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${e.id} ?readonly=${!o} @input=${e=>this.upd([`classes`,t,`id`],e.target.value)} data-class-id /><span class="hint">${o?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${e.label} @input=${e=>this.upd([`classes`,t,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${e.description} @input=${e=>this.upd([`classes`,t,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${I(Be,a.effort??`medium`,e=>this.upd([`classes`,t,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let n=(a.packages??[]).includes(e);return i`<button class="pick ${n?`is-on`:``}" aria-pressed=${n?`true`:`false`} @click=${()=>this.upd([`classes`,t,`policy`,`packages`],n?(a.packages??[]).filter(t=>t!==e):[...a.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(a.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,t,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}num(e,t,n,r={}){return i`<label class="field"><span>${e}</span><input type="number" .value=${String(t)} step=${r.step??1} min=${r.min??``} max=${r.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&n(t)}} />${r.hint?i`<span class="hint">${r.hint}</span>`:c}</label>`}text(e,t,n){return i`<label class="field"><span>${e}</span><input .value=${t} @input=${e=>n(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(j(n,[`classifier`,`stages`],r))}};y([a({attribute:!1})],G.prototype,`ids`,void 0),y([a({attribute:!1})],G.prototype,`meta`,void 0),y([a({attribute:!1})],G.prototype,`active`,void 0),y([a({attribute:!1})],G.prototype,`onSaved`,void 0),y([v()],G.prototype,`packages`,void 0),y([v()],G.prototype,`testText`,void 0),y([v()],G.prototype,`testProfile`,void 0),y([v()],G.prototype,`testTools`,void 0),y([v()],G.prototype,`testing`,void 0),y([v()],G.prototype,`result`,void 0),y([v()],G.prototype,`testError`,void 0),G=y([o(`qse-cond-routing`)],G);var Ve=[`off`,`shadow`,`act`],He=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,Ue=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,K=class extends e{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new z(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),k.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(j(this.ed.draft,e,t))}render(){let e=this.ed,t=this.config?.server.monitor_mode;return i`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${e.loading&&!e.draft?b(8):e.loadError?i`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`:i`
            ${t===`raw`?i`<div class="guard-line" data-monitor-off>${h(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:c}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(e.draft)}
            ${this.renderRest(e.draft)}
            ${B(e)}
            ${V(e,`the monitor`)}`}`}renderTasks(e){let t=we(e),n=t.filter(t=>e.tasks[t]&&/^C\d/.test(t)),r=t.filter(e=>Te(e)),a=t.filter(e=>!n.includes(e)&&!r.includes(e)),o=r.map(t=>e.tasks[t]).find(e=>e.rule===`stall_watchdog`)?.notify;return i`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${n.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${n.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${r.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${r.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${o?.enabled?i`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:i`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, Khaled's ruling of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${r.map(t=>this.renderTask(t,e.tasks[t]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${a.join(`, `)}</span></summary>
        <div class="sec-body">${a.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>`}renderTask(e,t){let n=Ce(this.meta?.schemas?.monitor,t.rule),r=this.meta?.tasks?.[e];return i`<div class="row-card" data-monitor-task=${e}>
      <div class="row-card-head">
        <strong>${e}${r?i` · ${r.label}`:c} <span class="muted small mono">${t.rule}</span></strong>
        ${I(Ve,t.mode,t=>this.upd([`tasks`,e,`mode`],t),{small:!0,label:`${e} mode without a profile`})}
      </div>
      ${r?.why?i`<p class="muted small">${r.why}</p>`:c}
      ${n?.properties?i`<div class="form-stack">${Object.entries(n.properties).map(([n,r])=>this.field(n,r,t[n],[`tasks`,e,n]))}</div>`:i`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,t,[`tasks`,e])}`}
    </div>`}field(e,t,n,r){let a=Array.isArray(t.type)?t.type[0]:t.type,o=e.replace(/_/g,` `),s=t.description?i`<span class="hint">${t.description}</span>`:c;if(t.enum)return i`<label class="field"><span>${o}</span><select data-param=${e} @change=${e=>this.upd(r,e.target.value)}>${t.enum.map(e=>i`<option value=${String(e)} ?selected=${e===n}>${String(e)}</option>`)}</select>${s}</label>`;if(a===`boolean`)return i`<label class="switch" data-param=${e}><input type="checkbox" .checked=${!!n} @change=${e=>this.upd(r,e.target.checked)} /> ${o}</label>${t.description?i`<p class="hint">${t.description}</p>`:c}`;if(a===`integer`||a===`number`)return i`<label class="field"><span>${o}</span><input type="number" data-param=${e} step=${a===`integer`?`1`:`0.01`} min=${t.minimum??c} max=${t.maximum??c} .value=${n==null?``:String(n)} @input=${e=>{let t=e.target.value;this.upd(r,t===``?null:a===`integer`?Math.round(Number(t)):Number(t))}} />${s}</label>`;if(a===`string`){let t=String(n??``);return He.test(e)||t.length>60?i`<label class="field field-block"><span>${o}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(t.length/90)))} class=${Ue.test(e)?`mono`:``} data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)}></textarea>${s}</label>`:i`<label class="field"><span>${o}</span><input data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)} />${s}</label>`}if(a===`array`&&t.items&&(Array.isArray(t.items.type)?t.items.type[0]:t.items.type)===`string`){let t=Array.isArray(n)?n:[];return i`<label class="field field-block"><span>${o} <span class="muted small">(one a line)</span></span><textarea rows=${Math.min(8,Math.max(2,t.length+1))} data-param=${e} .value=${t.join(`
`)} @input=${e=>this.upd(r,De(e.target.value))}></textarea>${s}</label>`}if(a===`array`&&t.items?.properties){let a=Array.isArray(n)?n:[],c=t.items;return i`<fieldset class="fieldset" data-param=${e}><legend>${o} <span class="muted small">${a.length}</span></legend>${s}
        ${a.map((e,t)=>i`<div class="row-card row-card-sub" data-item=${String(e.id??t)}>
          <div class="row-card-head"><strong class="mono small">${String(e.id??`#${t+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(r,a.filter((e,n)=>n!==t))} aria-label=${`remove ${String(e.id??t+1)}`}>${h(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(c.properties).map(([n,i])=>this.field(n,i,e[n],[...r,t,n]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${e} @click=${()=>this.upd(r,[...a,We(c)])}>${h(`plus`)} Add</button>
      </fieldset>`}if(a===`object`&&t.properties){let a=n??{};return i`<fieldset class="fieldset" data-param=${e}><legend>${o}</legend>${s}
        <div class="form-stack">${Object.entries(t.properties).map(([e,n])=>a[e]===void 0&&!(t.required??[]).includes(e)?i`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...r,e],We(n))}>${h(`plus`)} ${e.replace(/_/g,` `)}</button>`:this.field(e,n,a[e],[...r,e]))}</div></fieldset>`}return this.jsonField(o,n,r)}jsonField(e,t,n){return i`<label class="field field-block"><span>${e} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(t,null,2)} @change=${t=>{try{let e=JSON.parse(t.target.value);if(n.length===2&&n[0]===`tasks`){let t=this.ed.draft.tasks[n[1]];this.upd(n,{...e,rule:t.rule,mode:t.mode})}else this.upd(n,e)}catch{this.ed.errors=[`${e}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(e){return i`
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
      </details>`}};y([a({attribute:!1})],K.prototype,`meta`,void 0),y([a({attribute:!1})],K.prototype,`onSaved`,void 0),y([v()],K.prototype,`config`,void 0),K=y([o(`qse-cond-monitor`)],K);function We(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,We(t)])):``}var q=class extends e{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=d(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await k.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let t=f(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let e=this.sessions;return S(i`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!e?i`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?b(2):e.length?i`<div class="form-stack" data-watch>${e.map(e=>this.row(e))}</div>`:i`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(e){let n=e.gates.some(e=>e.noted);return i`<div class="row-card" data-watch-session=${e.session}>
      <div class="row-card-head">
        <strong class="mono small">${e.session}</strong>
        <span class="toolbar" style="margin:0">
          ${e.watch?i`<span class="badge ${e.watch.mode===`act`?`badge-on`:`badge-shadow`}">${e.watch.task} ${e.watch.mode}</span>`:c}
          ${e.client_gone?i`<span class="badge badge-warn" data-client-gone>client gone</span>`:e.running?i`<span class="badge badge-on">running</span>`:c}
          ${e.watch?.notify?i`<span class="badge badge-warn" data-notify>push on</span>`:c}
          ${n?i`<span class="badge badge-warn">gate noted</span>`:c}</span>
      </div>
      <p class="small" style="margin:4px 0">${Ee(e)}</p>
      <p class="muted small num" style="margin:0">${e.watch?`${e.watch.profile||`—`} · ${e.watch.client||`—`} · `:``}${e.requests} requests · last ${t(new Date(e.last_seen*1e3).toISOString())}${e.note_at?` · last note ${t(new Date(e.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};y([v()],q.prototype,`sessions`,void 0),y([v()],q.prototype,`monitor`,void 0),y([v()],q.prototype,`error`,void 0),q=y([o(`qse-cond-watch`)],q);var J=class extends e{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=d(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await k.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let t=f(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let e=this.events,t=[...new Set((e??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!t.includes(this.task)&&t.push(this.task),i`${S(i`Monitor journal ${this.paused?i`<span class="badge">paused</span>`:i`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!e?i`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?b(4):e.length?i`<div class="feed" data-journal>${e.map(e=>ze(e,e=>this.open=e))}</div>`:i`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:i`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${t.map(e=>i`<option value=${e} ?selected=${e===this.task}>${e}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${h(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?i`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:c}`}};y([a({attribute:!1})],J.prototype,`session`,void 0),y([v()],J.prototype,`events`,void 0),y([v()],J.prototype,`monitor`,void 0),y([v()],J.prototype,`error`,void 0),y([v()],J.prototype,`task`,void 0),y([v()],J.prototype,`kind`,void 0),y([v()],J.prototype,`paused`,void 0),y([v()],J.prototype,`open`,void 0),J=y([o(`qse-cond-journal`)],J);var Ge=[`off`,`low`,`medium`,`high`,`xhigh`],Ke={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},qe=new Set([`S1`,`S4`]),Je=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],Y=class extends e{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleTest={},this.ed=null,this.roles=new z(this,`roles`),this.residency=new z(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),k.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),k.doc(`packages`).then(e=>this.packages=Object.keys(e.doc.packages)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new z(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{this.roleHealth=(await k.roles()).health}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let t=await k.roleTest(e);this.roleTest={...this.roleTest,[e]:t.ok?`ok in ${r(t.ms)}: "${t.answer?.label}" at ${t.answer?.confidence.toFixed(2)}`:`failed: ${t.error}`}}catch(t){this.roleTest={...this.roleTest,[e]:f(t).message}}this.loadHealth()}render(){return i`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>i`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>g({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?i`<span class="badge badge-lock">${h(`warn`)} locked</span>`:i`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${t(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):i`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(j(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=j(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=j(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderEditor(e){if(e.loading&&!e.draft)return b(8);if(e.loadError)return i`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=!!e.locked,r=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],a=this.router?.classes??[];return i`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${h(`playground`)} Try in Playground</a></div>
      ${n?i`<div class="note" data-lock-reason>${h(`warn`)} ${e.locked}. Every control below is read-only.</div>`:c}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${t.description} ?disabled=${n} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${I([`raw`,`shadow`,`on`],t.mode,e=>this.upd([`mode`],e),{disabled:n,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${r.map(e=>{let r=t.clients.includes(e);return i`<button class="pick ${r?`is-on`:``}" ?disabled=${n} aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`clients`],r?t.clients.filter(t=>t!==e):[...t.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${n} @change=${e=>this.upd([`default_class`],e.target.value)}>${a.map(e=>i`<option value=${e.id} ?selected=${e.id===t.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(Ke).map(e=>i`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${I([`off`,`shadow`,`on`],t.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:n,small:!0,label:e})}<span class="small">${Ke[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${a.map(e=>{let r=t.routing[e.id]??{},a=Object.keys(r).length>0;return i`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${P(e.id,this.ids,e.label)}
                ${a?i`<button class="btn btn-ghost btn-sm" ?disabled=${n} @click=${()=>{let n={...t.routing};delete n[e.id],this.upd([`routing`],n)}}>use the router's</button>`:i`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${I(Ge,r.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...r,effort:t}),{disabled:n,small:!0,label:`effort override of ${e.id}`,cls:r.effort?``:`is-inherited`})}</div>
              <div class="pkg-picks">${this.packages.map(t=>{let a=r.packages??e.policy.packages??[],o=a.includes(t);return i`<button class="pick ${o?`is-on`:``}" ?disabled=${n} @click=${()=>this.upd([`routing`,e.id],{...r,packages:o?a.filter(e=>e!==t):[...a,t]})}>${t}</button>`})}</div>
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
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${I(Ge,t.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:n,small:!0,label:`default effort`})}</div>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.thinking.hints} @change=${e=>this.upd([`thinking`,`hints`],e.target.checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113); a pack that is off turns its task off</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${Je.map(([e,r,a])=>i`<label class="switch" data-pack=${e}><input type="checkbox" ?disabled=${n} .checked=${t.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${a} <span class="muted small">(${r}, ${t.toolgate[e]?t.tasks[r]?.mode??`off`:`off`})</span></label>`)}
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
      ${B(e)}
      ${V(e,`profile ${e.name}`)}
    </div>`}renderTask(e,t,n,r,a){let o=this.meta?.tasks[e],s=o?.role&&o.tier===`1`,l=!s||!!this.roleHealth[o.role]?.up;return i`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${o?.family??``}</div></div>
      <div>${o?.label??e}<div class="task-why">${o?.why??``}</div>${s&&!l?i`<div class="warn-ink small">${h(`warn`)} the ${o.role} role is offline: act is refused</div>`:c}</div>
      <div class="task-mode">${I([`off`,`shadow`,`act`],t,t=>this.upd([`tasks`,e,`mode`],t),{disabled:a,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${qe.has(e)&&Object.keys(r).length?i`<p class="muted small" data-in-context=${e}>${Object.entries(r).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:c}
        ${Object.entries(qe.has(e)?{}:r).map(([t,n])=>{let r=o?.thresholds[t];return i`<label class="field ${r&&r.default!==n?`is-changed`:``}"><span>${t.replace(/_/g,` `)} <span class="hint">${r?.unit??``}${r?`, default ${r.default}`:``}</span></span>
            <input type="number" ?disabled=${a} .value=${String(n)} step="any" @input=${n=>{let r=Number(n.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,t],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${a} min="0" max="10" .value=${String(n)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let e=this.roles;if(!e.draft)return e.loadError?i`<p class="err-ink small">${e.loadError}</p>`:c;let t=e.draft,n=(n,r)=>e.set(j(t,n,r));return i`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${Object.entries(t).map(([e,t])=>{let a=this.roleHealth[e];return i`<div class="row-card" data-role-row=${e}>
            <div class="row-card-head"><strong>${e}</strong>
              <span class="toolbar" style="margin:0">${a?i`<span class="badge ${a.up?`badge-ok`:t.mode===`off`?``:`badge-warn`}">${a.up?`up ${r(a.ms)}`:t.mode===`off`?`off`:`down: ${a.error??`?`}`}</span>`:c}
                <button class="btn btn-sm" @click=${()=>this.testRole(e)} data-role-test=${e}>Test</button></span></div>
            ${this.roleTest[e]?i`<p class="small" role="status">${this.roleTest[e]}</p>`:c}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${t.endpoint} @input=${t=>n([e,`endpoint`],t.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${t.model} @input=${t=>n([e,`model`],t.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(t.timeout_ms)} @input=${t=>n([e,`timeout_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>memory claim (GB)</span><input type="number" step="0.1" .value=${String(t.memory_gb)} @input=${t=>n([e,`memory_gb`],Number(t.target.value))} /></label>
            </div>
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${I([`off`,`shadow`,`on`],t.mode,t=>n([e,`mode`],t),{small:!0,label:`${e} mode`})}<span class="muted small mono">${t.contract}</span></div>
          </div>`})}
        ${B(e)}
        ${V(e,`the roles`)}
      </div>
    </details>`}renderResidency(){let e=this.residency;if(!e.draft)return e.loadError?i`<p class="err-ink small">${e.loadError}</p>`:c;let t=e.draft,n=(n,r)=>e.set(j(t,n,r)),r=(e,r=`0.1`,a=!1)=>i`<input type="number" step=${r} min="0" .value=${String(t[e[0]])} @input=${t=>{let r=Number(t.target.value);n(e,a?Math.round(r):r)}} data-res=${e[0]} />`,a=[`measured`,`peak`,`declared`];return i`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${r([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${r([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${r([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${t.order.join(`, `)} @change=${e=>n([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(t.claims).map(([e,r])=>i`<div class="row-card" data-claim=${e}>
            <div class="row-card-head"><strong>${e}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[e]:r,...i}=t.claims;n([`claims`],i)}} aria-label=${`remove the claim of ${e}`}>${h(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(r.gib)} @input=${t=>n([`claims`,e,`gib`],Number(t.target.value))} data-claim-gib=${e} /></label>
              <label class="field"><span>basis</span><select @change=${t=>n([`claims`,e,`basis`],t.target.value)}>${a.map(e=>i`<option value=${e} ?selected=${e===r.basis}>${e}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${r.note} @input=${t=>n([`claims`,e,`note`],t.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in t.claims} @click=${()=>{n([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${h(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${B(e)}
        ${V(e,`the residency limits`)}
      </div>
    </details>`}};y([a({attribute:!1})],Y.prototype,`ids`,void 0),y([a({attribute:!1})],Y.prototype,`meta`,void 0),y([a({attribute:!1})],Y.prototype,`tiles`,void 0),y([a({attribute:!1})],Y.prototype,`selected`,void 0),y([a({attribute:!1})],Y.prototype,`onSaved`,void 0),y([v()],Y.prototype,`router`,void 0),y([v()],Y.prototype,`packages`,void 0),y([v()],Y.prototype,`roleHealth`,void 0),y([v()],Y.prototype,`roleTest`,void 0),y([v()],Y.prototype,`newClaim`,void 0),Y=y([o(`qse-cond-profiles`)],Y);var X=class extends e{constructor(...e){super(...e),this.ids=[],this.router=null,this.sel=null,this.found=null,this.scanError=null,this.ed=new z(this,`packages`)}connectedCallback(){super.connectedCallback(),this.ed.load(),k.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0)}usedBy(e){return(this.router?.classes??[]).filter(t=>(t.policy.packages??[]).includes(e)).map(e=>e.id)}upd(e,t){this.ed.draft&&this.ed.set(j(this.ed.draft,e,t))}async scan(){this.scanError=null;try{this.found=(await k.scan()).skills}catch(e){this.scanError=f(e).message}}register(e){let t=this.ed.draft,n={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.ed.set(j(t,[`packages`,e.name],n)),this.sel=e.name}addInline(){let e=this.ed.draft,t=1;for(;e.packages[`package-${t}`];)t++;let n=`package-${t}`;this.ed.set(j(e,[`packages`,n],{title:`Package ${t}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0})),this.sel=n}removePackage(e){let t=this.ed.draft,n={...t.packages};delete n[e],this.ed.set({...t,packages:n}),this.sel=null}render(){let e=this.ed;if(e.loading&&!e.draft)return b(6);if(e.loadError)return i`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=e.base?.status??{};return i`
      <section class="panel" data-packages>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills</h2><p class="panel-sub">The router injects one into the prompt when the turn's class names it and the client did not load it itself.</p></div>
          <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-add-package>${h(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${h(`search`)} Skills on the box</button></div></header>
        <div class="table-scroll"><table class="table">
          <thead><tr><th>name</th><th>source</th><th>on the box</th><th class="num-col">tokens</th><th>classes</th><th>enabled</th></tr></thead>
          <tbody>${Object.entries(t.packages).map(([e,t])=>{let r=n[e];return i`<tr class="req-row ${this.sel===e?`is-open`:``}" data-package-row=${e} @click=${()=>this.sel=e}>
              <td><strong>${e}</strong><div class="muted small">${t.title}</div></td>
              <td class="mono small">${t.source}</td>
              <td>${r?r.available?i`<span class="badge badge-ok">found</span>`:i`<span class="badge badge-warn" title=${r.path??``}>${r.error??`missing`}</span>`:i`<span class="badge">not saved</span>`}</td>
              <td class="num num-col">${r?w(r.tokens_est):`—`}</td>
              <td>${this.usedBy(e).map(e=>P(e,this.ids))}</td>
              <td>${t.enabled?`yes`:`no`}</td>
            </tr>`})}</tbody></table></div>
      </section>
      ${this.found?i`<section class="panel" style="margin-top:16px" data-found>
            <header class="panel-head"><h2 class="panel-title">Skills in ${t.skill_dirs.join(`, `)}</h2></header>
            ${this.found.length?i`<div class="table-scroll"><table class="table"><tbody>${this.found.map(e=>i`<tr><td><strong>${e.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${e.description}</div></td><td class="num">${w(e.chars)} chars</td>
                    <td>${e.registered||t.packages[e.name]?i`<span class="badge badge-ok">registered</span>`:i`<button class="btn btn-sm" @click=${()=>this.register(e)} data-register=${e.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:i`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ spark:~/.config/opencode/skills/</code>).</p>`}
          </section>`:this.scanError?i`<p class="err-ink small">${this.scanError}</p>`:c}
      ${this.sel&&t.packages[this.sel]?this.renderEditor(this.sel,t.packages[this.sel]):c}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${t.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${B(e)}
      ${V(e,`the packages`)}
    `}renderEditor(e,t){let n=t.source===`inline`,r=this.usedBy(e);return i`<section class="panel" style="margin-top:16px" data-package-editor=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e}</h2><p class="panel-sub">${r.length?i`used by ${r.join(`, `)}`:`no class uses it yet (add it on the Routing screen)`}</p></div>
        <div class="panel-tools"><button class="btn btn-sm btn-ghost" ?disabled=${r.length>0} title=${r.length?`a class uses it`:``} @click=${()=>this.removePackage(e)}>${h(`trash`)} Remove</button></div></header>
      <div class="form-grid">
        <label class="field"><span>title</span><input .value=${t.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>source</span><input .value=${t.source} @input=${t=>this.upd([`packages`,e,`source`],t.target.value)} /><span class="hint">inline, skill:&lt;name&gt; or file:&lt;path on the box&gt;</span></label>
        <label class="field"><span>max tokens</span><input type="number" .value=${String(t.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">longer text is cut at a line end</span></label>
        <label class="switch"><input type="checkbox" .checked=${t.enabled} @change=${t=>this.upd([`packages`,e,`enabled`],t.target.checked)} /> enabled</label>
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" .value=${t.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${n?i`<label class="field field-block"><span>text the model gets</span><textarea rows="14" data-package-body .value=${t.body} @input=${t=>this.upd([`packages`,e,`body`],t.target.value)}></textarea></label>`:i`<p class="muted small">The text is the file on the box, read when first needed and again when it changes; its YAML front matter is left out.</p>`}
    </section>`}};y([a({attribute:!1})],X.prototype,`ids`,void 0),y([v()],X.prototype,`router`,void 0),y([v()],X.prototype,`sel`,void 0),y([v()],X.prototype,`found`,void 0),y([v()],X.prototype,`scanError`,void 0),X=y([o(`qse-cond-packages`)],X);var Z=class extends e{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=d(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await k.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let t=f(e);this.error=`${t.endpoint}: ${t.message}`}}async loadSession(e){try{let t=await k.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(t){e===this.selected&&(this.turns=[]),this.error=f(t).message}}render(){if(this.error&&!this.list)return i`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!this.list)return b(6);let e=[...new Set(this.list.map(e=>e.client_kind??`other`))],t=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return i`
      ${this.selected?this.renderSession(this.selected):c}
      ${S(`Sessions, last 7 days`,t.length?i`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${t.map(e=>i`<tr class="req-row ${e.key===this.selected?`is-open`:``}" data-session=${e.key} @click=${()=>g({session:e.key})}>
                  <td class="mono small cell-wrap">${M(e.key,28)}</td><td>${e.client_kind??`—`}</td><td>${e.profile??`—`}</td>
                  <td>${P(e.class_id,this.ids)}</td><td class="num num-col">${w(e.turns)}</td>
                  <td class="num num-col">${e.notes?`${e.notes} (${w(e.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${e.acts?i`${w(e.acts)}${e.gave_ups?i` <span class="badge badge-warn">${e.gave_ups} gave up</span>`:c}`:`—`}</td>
                  <td class="num num-col" data-interventions>${w(e.interventions??0)}</td>
                  <td>${e.last_checker?i`<span class="badge ${e.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${e.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${l(new Date(e.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:m(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:i`<label class="field field-search">${h(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${e.map(e=>i`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?i`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:c}
    `}renderSession(e){let n=this.turns,r=this.live.find(t=>t.key===e);if(n==null)return b(4);let a=n.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,o=n.reduce((e,t)=>e+(t.injected?.length??0),0),l=this.events.filter(e=>C(e)).length,u=be(n);return i`<section class="panel" style="margin-bottom:16px" data-timeline=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${M(e,44)} ${r?i`<span class="badge badge-on">live</span>`:c}</h2>
        <p class="panel-sub">${n.length} turns · ${o} hidden note${o===1?``:`s`} · ${l} monitor act${l===1?``:`s`}${r?` · class now ${r.class_id} · ${w(r.note_tokens)} note tokens replayed a turn`:``}${n.length?` · since ${t(new Date(n[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${k.exportUrl(e,`md`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${h(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${k.exportUrl(e,`json`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${h(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>g({session:null})}>${h(`x`)}</button></div></header>
      ${this.axis(ke(n,this.events,this.ids,e=>s(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${n.map(e=>{let t=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/a*900)),n=(e.injected?.length??0)>0,r=!!e.shadow,o=(e.events??[]).filter(e=>e.decision!==`PASS`),s=o.some(e=>C(e))?`act`:o.length?`shadow`:null;return i`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${t}px; --cls: var(--series-${A(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${ve(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${n?i`<span class="mark"></span>`:r?i`<span class="mark is-shadow"></span>`:c}
            ${s?i`<span class="mark is-monitor ${s===`shadow`?`is-shadow`:``}" data-monitor-mark=${s}></span>`:c}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${u.series.length?i`<qse-ribbon .series=${u.series} .labels=${u.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:c}
      <div class="feed" style="margin-top:10px">${n.slice().reverse().map(e=>Ie(e,this.ids,e=>this.open=e,ve(e)+((e.events??[]).some(e=>C(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?i`<div style="margin-top:12px"><qse-cond-journal .session=${e}></qse-cond-journal></div>`:c}
    </section>`}axis(e){let t=e=>`${(e*100).toFixed(3)}%`;return i`<figure class="axis" data-axis aria-label="the session on one time axis">
      <div class="axis-lane axis-acts" aria-label=${`${e.acts.length} monitor decisions`}>
        ${e.acts.map(e=>i`<span class="axis-mark ${e.shadow?`is-shadow`:`is-act`}" style="left:${t(e.x)}" title=${e.title} data-axis-act=${e.shadow?`shadow`:`act`}></span>`)}
        ${e.notes.map(e=>i`<span class="axis-mark is-note ${e.shadow?`is-shadow`:``}" style="left:${t(e.x)}" title=${e.title} data-axis-note></span>`)}
      </div>
      <div class="axis-lane axis-turns" aria-label=${`${e.turns.length} turns`}>
        ${e.turns.map(e=>i`<span class="axis-turn ${e.compaction?`is-compaction`:``}" style="left:${t(e.x0)};width:${t(Math.max(0,e.x1-e.x0))};--cls:var(--series-${e.color})" title=${e.title} data-axis-turn></span>`)}
      </div>
      <div class="axis-lane axis-events" aria-label=${`${e.compactions.length} compactions, ${e.checkers.length} checker results`}>
        ${e.compactions.map(e=>i`<span class="axis-mark is-compaction" style="left:${t(e.x)}" title=${e.title} data-axis-compaction></span>`)}
        ${e.checkers.map(e=>i`<span class="axis-mark ${e.pass?`is-pass`:`is-fail`}" style="left:${t(e.x)}" title=${e.title} data-axis-checker=${e.pass?`pass`:`fail`}></span>`)}
      </div>
      <div class="axis-ticks" aria-hidden="true">${e.ticks.map((n,r)=>i`<span class="num ${r===0?`is-first`:r===e.ticks.length-1?`is-last`:``}" style="left:${t(n.x)}">${n.label.slice(0,5)}</span>`)}</div>
      <figcaption class="muted small">One time axis, ${e.ticks[0]?.label??``} to ${e.ticks[e.ticks.length-1]?.label??``}: ${e.turns.length} turns (the gaps are the client's time), ${e.acts.filter(e=>!e.shadow).length} monitor actions ◆ above (◇ shadow, ${e.acts.filter(e=>e.shadow).length}), ${e.notes.length} router note${e.notes.length===1?``:`s`}, ${e.compactions.length} compaction${e.compactions.length===1?``:`s`} ▲ and ${e.checkers.length} checker result${e.checkers.length===1?``:`s`} below (● pass, ○ fail; the gate numbers are in each one's title).</figcaption>
    </figure>`}tier1(){let e=Ae(this.events);return e?i`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${e.agree} of ${e.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${e.decisions} shadow decisions)</span></p>`:c}};y([a({attribute:!1})],Z.prototype,`ids`,void 0),y([a({attribute:!1})],Z.prototype,`selected`,void 0),y([v()],Z.prototype,`list`,void 0),y([v()],Z.prototype,`live`,void 0),y([v()],Z.prototype,`turns`,void 0),y([v()],Z.prototype,`events`,void 0),y([v()],Z.prototype,`error`,void 0),y([v()],Z.prototype,`query`,void 0),y([v()],Z.prototype,`client`,void 0),y([v()],Z.prototype,`open`,void 0),Z=y([o(`qse-cond-sessions`)],Z);var Q=class extends e{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([k.stats(168),k.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let t=f(e);this.error=`${t.endpoint}: ${t.message}`}}async promote(e,t,n,r,i){if(await Ne(`Promote ${n} to ${t===`task`?`act`:`on`} in ${e}?`,[t===`task`?`The session monitor will act on ${n} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${n} decisions instead of only journaling them: ${n===`effort`?`budgets and thinking off go to the engine`:n===`packages`?`skills and packages go into the prompt`:n===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${r.toFixed(2)} over ${i} labelled decisions. A new profile version records it.`],`Promote`))try{let a=await k.doc(`profiles`,e),o=t===`task`?j(a.doc,[`tasks`,n,`mode`],`act`):j(a.doc,[`decisions`,n],`on`),s=await R(()=>k.put(`profiles`,e,o,`${n} to ${t===`task`?`act`:`on`}, precision ${r.toFixed(2)} / ${i}`,a.version));this.msg=`${e} v${s.version}: ${n} promoted`,this.onSaved()}catch(e){this.msg=f(e).message}}async loadAb(){try{this.abVersions=(await k.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await k.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await k.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=f(e).message}}render(){return this.error&&!this.stats?i`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`:this.stats?i`
      ${this.msg?i`<div class="note note-info" role="status">${this.msg}</div>`:c}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:b(8)}renderPromotion(e){let t=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,n=(e.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return S(`Shadow precision and promotion`,i`<p class="muted small">${oe}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${e.precision.length||n.length?i`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${e.precision.map(e=>{let n=se(e),r=t(e.profile,e.decision);return i`<tr data-precision-row=${`${e.profile}/${e.decision}`}>
                    <td>${e.profile}</td><td class="mono">${e.decision}</td><td><span class="badge ${r===`on`?`badge-on`:r===`shadow`?`badge-shadow`:``}">${r}</span></td>
                    <td class="num num-col">${w(e.decisions)}</td><td class="num num-col">${w(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${n.allowed?``:`muted`}">${n.why}</td>
                    <td><button class="btn btn-sm ${n.allowed?`btn-primary`:``}" ?disabled=${!n.allowed||r!==`shadow`} title=${n.allowed?``:oe} data-promote=${`${e.profile}/${e.decision}`} @click=${()=>this.promote(e.profile??``,`decision`,e.decision,e.precision??0,e.labelled)}>Promote to on</button></td></tr>`})}
                ${n.map(e=>{let t=se(e);return i`<tr data-precision-row=${`task/${e.task}`}><td>—</td><td class="mono">${e.task}</td><td><span class="badge ${e.mode===`act`?`badge-on`:`badge-shadow`}">${e.mode}</span></td>
                    <td class="num num-col">${w(e.decisions)}</td><td class="num num-col">${w(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${t.allowed?``:`muted`}">${t.why}</td>
                    <td><button class="btn btn-sm ${t.allowed?`btn-primary`:``}" ?disabled=${!t.allowed||e.mode!==`shadow`} data-promote=${`task/${e.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,e.task,e.precision??0,e.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:m(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${e.monitor?c:i`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let e=this.reports.find(e=>e.kind===`classifier`);if(!e)return S(`Classifier accuracy`,m(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let a=e.stages??{},o=e.set??{},s=(a[this.stage]??Object.values(a)[0])?.confusion??{},c=o.classes??Object.keys(s);return S(i`Classifier accuracy <span class="muted small num">${o.name??``} · ${o.n??`?`} prompts · ${e.created_at?t(String(e.created_at)):``}</span>`,i`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(a).map(([e,t])=>i`<tr class="req-row ${e===this.stage?`is-open`:``}" @click=${()=>this.stage=e}><td class="mono">${e}</td><td class="num num-col">${n(t.accuracy,1)}</td><td class="num num-col">${t.answered} / ${t.n}</td><td class="num num-col">${t.accuracy_answered==null?`—`:n(t.accuracy_answered,1)}</td><td class="num num-col">${r(t.p50_ms)}</td><td class="num num-col">${r(t.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${c.length&&Object.keys(s).length?i`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...c,`—`].map(e=>i`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${c.map(e=>i`<tr><th>${P(e,this.ids)}</th>${[...c,`—`].map(t=>{let n=s[e]?.[t]??0;return i`<td class="cm-cell ${e===t?`cm-diag`:n?`cm-miss`:``}">${n||``}</td>`})}</tr>`)}</tbody></table></div>`:i`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}`,{id:`cond-classifier`})}renderSuites(){let e=this.reports.filter(e=>e.kind===`suite`),t=e.find(e=>e.file===this.suiteFile)??e[0],r=t?.rows??[],a=pe(this.reports,this.engine),o=fe(r),s=t?.verdict,l=t?.comparison??[],u=t?.raw_row,d=t?.build,f=s?.indicative??le(r),p=(e,t=1)=>e==null?`—`:Number.isInteger(e)?w(e):e.toFixed(t);return S(i`TandemLLM against the raw engine ${t?i`<span class="muted small num">night ${t.night??`—`}${d?` · ${d.version??`?`} ${M(d.code_sha,12)}`:``}${t.profile?` · profile ${t.profile}`:``}</span>`:c}`,i`${a?i`<div class="guard-line" data-guard-detail>${h(`warn`)} ${a}</div>`:i`<p class="small">${h(`check`)} the never-worse gate ran for build ${this.engine.version}.</p>`}
        ${e.length>1?i`<label class="field"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${e.map(e=>i`<option value=${String(e.file)} ?selected=${e===t}>${String(e.night??e.file)} · ${String(e.file)}</option>`)}</select></label>`:c}
        ${s?i`<div class="note" data-eval-verdict>
              <span class="badge ${s.better?`badge-ok`:s.better===!1?`badge-warn`:``}"><strong>${s.better?`Better`:s.better===!1?`Not better`:`Not shown yet`}</strong></span>${f?i` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:c} — ${s.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${u?.status??`—`}${u?.source?` (${u.source})`:``}${t?.driver?` · driver ${t.driver.mode}`:``}${t?.isolation?` · ${t.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:c}
        ${o.length?i`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>build · night</th></tr></thead>
              <tbody>${o.flatMap(e=>{let t=l.find(t=>t.suite===e.suite||t.suite_id===(e.raw??e.tandem)?.suite_id);return[e.raw,e.tandem].filter(Boolean).map((r,a)=>i`<tr data-suite=${e.suite} data-arm=${r.arm}>
                    <td>${a===0?e.suite:``}${a===0&&!e.cmp.ok?i` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${e.cmp.why}</div>`:c}${a===0&&t?i` <span class="badge ${t.verdict===`better`?`badge-ok`:t.verdict===`worse`||t.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${t.verdict}>${t.verdict}</span>`:c}${a===0&&(r.indicative??(r.n??0)<10)?i` <span class="badge" data-row-indicative>indicative</span>`:c}</td>
                    <td class="mono">${ue(r.arm)}${r.pristine===!1?i` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:c}</td>
                    <td class="num num-col">${r.n??`—`}</td>
                    <td class="num num-col">${r.pass_rate==null?`—`:n(r.pass_rate,0)}</td>
                    <td class="num num-col">${r.gates_passed??`—`}${r.gates_total?` / ${r.gates_total}`:``}</td>
                    <td class="num num-col">${p(r.minutes_to_pass)}</td><td class="num num-col">${p(r.wall_minutes)}</td>
                    <td class="num num-col">${r.tokens==null?`—`:w(Math.round(r.tokens))}</td><td class="num num-col">${p(r.decode_tps)}</td>
                    <td class="num num-col">${r.empty_stops??`—`}</td><td class="num num-col">${r.continuations??`—`}</td>
                    <td class="num num-col">${r.interventions??`—`}</td><td class="num num-col">${r.gave_ups??`—`}</td>
                    <td class="num num-col">${r.cache_hit==null?`—`:n(r.cache_hit,0)}</td>
                    <td class="small">${M(r.build,16)} · ${r.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts.</p>`:m(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return S(`A/B of two profile versions`,i`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>i`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>i`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(t=>i`<option value=${t.version} ?selected=${t.version===this[e]}>v${t.version} ${M(t.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?i`<div class="diff" data-ab>${this.abChanges.map(e=>i`<div class="diff-row"><span>${_e(e.path)}</span><span class="diff-before">${M(e.before,80)}</span><span class="diff-after">${M(e.after,80)}</span></div>`)}</div>`:i`<p class="muted small">The two versions are the same.</p>`:c}`,{id:`cond-ab`})}};y([a({attribute:!1})],Q.prototype,`ids`,void 0),y([a({attribute:!1})],Q.prototype,`tiles`,void 0),y([a({attribute:!1})],Q.prototype,`onSaved`,void 0),y([v()],Q.prototype,`stats`,void 0),y([v()],Q.prototype,`reports`,void 0),y([v()],Q.prototype,`engine`,void 0),y([v()],Q.prototype,`error`,void 0),y([v()],Q.prototype,`msg`,void 0),y([v()],Q.prototype,`stage`,void 0),y([v()],Q.prototype,`abProfile`,void 0),y([v()],Q.prototype,`abVersions`,void 0),y([v()],Q.prototype,`abA`,void 0),y([v()],Q.prototype,`abB`,void 0),y([v()],Q.prototype,`abChanges`,void 0),y([v()],Q.prototype,`abCounts`,void 0),y([v()],Q.prototype,`suiteFile`,void 0),Q=y([o(`qse-cond-compare`)],Q);var Ye=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`sessions`,label:`Sessions`},{id:`compare`,label:`Compare`}],$=class extends e{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.guard=null,this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=d(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&Ye.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r,i]=await Promise.all([k.profiles(),this.meta?Promise.resolve(this.meta):k.meta(),k.config(),k.doc(`router`),k.evals()]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.guard=pe(i.reports,i.engine),this.error=null}catch(e){let t=f(e);this.error=t.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${t.endpoint}: ${t.message}`}}async setMode(e){let t=this.tiles.find(e=>e.name===this.active);if(t&&!t.locked&&(e!==`on`||await Ne(`Turn the ${t.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let n=await R(()=>k.setMode(t.name,e,`mode to ${e}`,t.version));this.reloaded=`v${n.version} · reloaded ${n.loaded_at?s(n.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=f(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),t=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return i`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${h(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>g({profile:e.target.value})}>
                ${this.tiles.map(e=>i`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?I([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):c}
            ${this.reloaded?i`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:c}
          </div>
        </div>
        ${this.switchError?i`<div class="guard-line" role="alert">${h(`warn`)} ${this.switchError}</div>`:c}
        ${this.error?i`<div class="guard-line" role="alert">${h(`warn`)} ${this.error}</div>`:c}
        ${this.guard?i`<div class="guard-line" data-guard>${h(`warn`)} ${this.guard}: the raw engine's never-worse gate (both views, same hour) has no result for the code that serves now.</div>`:c}
        ${t.map(([e,t])=>i`<div class="guard-line" role="alert">${h(`warn`)} ${e}: ${t.error?.message} (kept version ${t.version})</div>`)}
        ${e?.locked?i`<div class="note" data-locked>${h(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:c}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${Ye.map(e=>i`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:c}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return i`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return i`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return i`<qse-cond-monitor .meta=${e.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return i`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return i`<qse-cond-packages .ids=${e.ids}></qse-cond-packages>`;case`sessions`:return i`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`compare`:return i`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};y([a({attribute:!1})],$.prototype,`params`,void 0),y([v()],$.prototype,`tiles`,void 0),y([v()],$.prototype,`defaultProfile`,void 0),y([v()],$.prototype,`meta`,void 0),y([v()],$.prototype,`config`,void 0),y([v()],$.prototype,`ids`,void 0),y([v()],$.prototype,`guard`,void 0),y([v()],$.prototype,`error`,void 0),y([v()],$.prototype,`switching`,void 0),y([v()],$.prototype,`reloaded`,void 0),y([v()],$.prototype,`switchError`,void 0),$=y([o(`qse-conductor`)],$);export{$ as QseConductor};