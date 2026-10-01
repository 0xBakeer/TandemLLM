import{A as e,C as t,D as n,E as r,N as i,O as a,T as o,_ as s,a as c,b as l,c as u,d,f,g as p,j as m,k as h,l as g,m as _,p as v,s as ee,u as y,v as b,w as x,x as S,y as te}from"./app-xWj6ROMX.js";var C=`qse-conductor-token`;function ne(){try{return sessionStorage.getItem(C)??localStorage.getItem(C)}catch{return null}}function re(e,t=!1){try{sessionStorage.removeItem(C),localStorage.removeItem(C),e&&(t?localStorage:sessionStorage).setItem(C,e)}catch{}}async function w(e,t,n){let r={"Content-Type":`application/json`},i=ne();i&&(r.Authorization=`Bearer ${i}`);let a;try{a=await fetch(p()+e,{method:t,headers:r,body:JSON.stringify(n),credentials:`same-origin`,cache:`no-store`})}catch(t){throw new _(e,0,`engine unreachable (${t.message})`,`offline`)}let o={};try{o=await a.json()}catch{}if(a.status===401)throw new _(e,401,`editing needs the admin token`,`token_needed`);if(!a.ok){let t=new _(e,a.status,o.error?.message??`HTTP ${a.status}`,o.error?.type??`error`);throw t.errors=o.error?.errors??[],t}return o}var T=encodeURIComponent,E=(e,t)=>e===`profiles`?`/v1/conductor/profiles/${T(t??``)}`:`/v1/conductor/${e}`,D={config:()=>b(`/v1/conductor/config`),meta:()=>b(`/v1/conductor/meta`),profiles:()=>b(`/v1/conductor/profiles`),doc:(e,t)=>b(E(e,t)),versions:(e,t)=>b(`${E(e,t)}/versions`),version:(e,t,n)=>b(`${E(e,t)}/versions/${n}`),diff:(e,t,n,r)=>b(`${E(e,t)}/diff?from=${n}&to=${r}`),check:(e,t,n)=>b(`${E(e,t)}/check`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({doc:n})}),put:(e,t,n,r,i)=>w(E(e,t),`PUT`,{doc:n,note:r,base_version:i}),restore:(e,t,n,r)=>w(`${E(e,t)}/restore`,`POST`,{version:n,note:r}),setMode:(e,t,n,r)=>w(`/v1/conductor/profiles/${T(e)}/mode`,`POST`,{mode:t,note:n,base_version:r}),roles:()=>b(`/v1/conductor/roles?fresh=1`),roleTest:e=>b(`/v1/conductor/roles/${T(e)}/test`,{method:`POST`,headers:{"Content-Type":`application/json`},body:`{}`}),scan:()=>b(`/v1/conductor/packages/scan`),classify:(e,t,n=!0)=>b(`/v1/conductor/classify`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({text:e,profile:t,tools:n})}),decisions:(e={})=>{let t=new URLSearchParams;return e.since&&t.set(`since`,String(e.since)),t.set(`limit`,String(e.limit??100)),e.session&&t.set(`session`,e.session),e.before&&t.set(`before`,String(e.before)),b(`/v1/conductor/decisions?${t}`)},decision:e=>b(`/v1/conductor/decisions/${T(e)}`),label:(e,t,n=``)=>w(`/v1/conductor/decisions/${T(e)}/label`,`POST`,{label:t,note:n}),stats:(e=24)=>b(`/v1/conductor/stats?hours=${e}`),sessions:()=>b(`/v1/conductor/sessions`),session:e=>b(`/v1/conductor/sessions/${T(e)}`),exportUrl:(e,t)=>`${p()}/v1/conductor/sessions/${T(e)}/export?format=${t}`,evals:()=>b(`/v1/conductor/evals`)},O=.9,k=`on is allowed at precision ≥ ${O} over ≥ 30 labelled decisions`;function A(e){return e.labelled<30?{allowed:!1,why:`${e.labelled} of 30 labelled decisions`}:e.precision==null||e.precision<.9?{allowed:!1,why:`precision ${ie(e.precision)} is below ${O}`}:{allowed:!0,why:`precision ${ie(e.precision)} over ${e.labelled}`}}var ie=e=>e==null?`—`:e.toFixed(2);function ae(e,t){return!e.build||!t.build||e.build!==t.build?{ok:!1,why:`builds ${e.build??`?`} and ${t.build??`?`}`}:!e.night||!t.night||e.night!==t.night?{ok:!1,why:`nights ${e.night??`?`} and ${t.night??`?`}`}:{ok:!0,why:null}}function oe(e){let t=new Map;for(let n of e){let e=t.get(n.suite)??{raw:null,tandem:null};n.arm===`raw`?e.raw=n:e.tandem=n,t.set(n.suite,e)}return[...t.entries()].map(([e,t])=>({suite:e,...t,cmp:t.raw&&t.tandem?ae(t.raw,t.tandem):{ok:!1,why:t.raw?`no TandemLLM arm`:`no raw arm`}}))}function se(e,t){return e.some(e=>{if(e.kind!==`speed_guard`)return!1;let n=e.build??{};return t.code_sha&&n.code_sha===t.code_sha||!t.code_sha&&t.version&&n.version===t.version})?null:`speed guard not run for build ${t.version??`unknown`}`}var ce=[`cobalt`,`orange`,`aqua`,`violet`,`yellow`,`red`];function j(e,t){let n=e==null?-1:t.indexOf(e);return n<0?`muted`:ce[n%ce.length]}function le(e,t,n=4){let r=Object.entries(e.by_class).sort((e,t)=>t[1]-e[1]),i=r.slice(0,n).map(([e])=>e),a=i.map(n=>({name:n,color:j(n,t),values:e.hourly.map(e=>e.by_class[n]??0)}));return r.length>n&&a.push({name:`other`,color:`muted`,values:e.hourly.map(e=>Object.entries(e.by_class).filter(([e])=>!i.includes(e)).reduce((e,[,t])=>e+t,0))}),a}function ue(e){return JSON.parse(JSON.stringify(e))}function M(e,t,n){if(!t.length)return n;let[r,...i]=t,a=e??(typeof r==`number`?[]:{}),o=Array.isArray(a)?[...a]:{...a};return o[r]=M(a[r],i,n),o}function de(e){return e.replace(/^\$\.?/,``).replace(/\[/g,` › `).replace(/\]/g,``).replace(/\./g,` › `)||`(document)`}function N(e,t=80){if(e==null)return`—`;let n=typeof e==`string`?e:JSON.stringify(e);return n.length>t?n.slice(0,t-1)+`…`:n}function P(e){let t=[];e.effort&&t.push(e.thinking&&e.thinking.startsWith(`off`)?`thinking off`:e.effort);for(let n of e.injected??[]){let e=n.parts.filter(e=>e.kind!==`effort`).map(e=>e.name);t.push(`${e.length?e.join(` + `):`note`} (${n.tokens.toLocaleString(`en-US`)} tokens)`)}return e.replayed?.count&&t.push(`replayed ${e.replayed.count}`),e.mode===`shadow`&&t.push(`shadow`),t.join(` · `)||`—`}var fe={decider:`decider`,qwen:`Qwen`,sticky:`kept (same task)`,structural:`request shape`,fallback:`fail-open`,profile:`profile default`};function pe(e){let t=new Map;e.forEach((n,r)=>{for(let i of n.next_checker??[])for(let n of i.readings){let a=`${i.checker} ${n.gate}`;t.has(a)||t.set(a,Array(e.length).fill(null)),t.get(a)[r]=n.value}});let n=[`cobalt`,`orange`,`aqua`,`violet`,`yellow`,`red`];return{series:[...t.entries()].slice(0,6).map(([e,t],r)=>({name:e,color:n[r%n.length],values:t})),labels:e.map((e,t)=>`turn ${t+1}`)}}function F(e,t,n){let r=j(e,t);return i`<span class="cls" style="--cls: var(--series-${r})" data-class=${e??`none`}>${n??e??`—`}</span>`}function I(e,t=.6){return e==null?i`<span class="conf muted">—</span>`:i`<span class="conf" title=${`confidence ${e.toFixed(4)}`}><span class="conf-bar ${e<t?`is-low`:``}"><span style="width:${Math.round(e*100)}%"></span></span>${e.toFixed(2)}</span>`}function L(e,t,n,r={}){return i`<div class="seg ${r.small?`seg-sm`:``} ${r.cls??``}" role="radiogroup" aria-label=${r.label??`mode`}>
    ${e.map(e=>i`<button type="button" class="seg-btn mode-${e} ${e===t?`is-on`:``}" role="radio" aria-checked=${e===t?`true`:`false`} ?disabled=${r.disabled} @click=${()=>e!==t&&n(e)}>${e}</button>`)}
  </div>`}function R(e){return e?fe[e]??e:`—`}var z=class extends n{constructor(...e){super(...e),this.kind=`confirm`,this.heading=``,this.lines=[],this.confirmLabel=`Confirm`,this.done=()=>void 0,this.error=null}submit(e){if(e.preventDefault(),this.kind===`token`){let e=(this.querySelector(`#cond-token`)?.value??``).trim();if(!e){this.error=`Enter the admin token.`;return}this.done(!0,e,this.querySelector(`#cond-remember`)?.checked??!1)}else this.done(!0)}render(){return i`<div class="modal-back" @click=${()=>this.done(!1)}></div>
      <form class="modal" role="dialog" aria-modal="true" aria-label=${this.heading} @submit=${e=>this.submit(e)}>
        <h2>${this.heading}</h2>
        ${this.lines.map(e=>i`<p class="muted small">${e}</p>`)}
        ${this.kind===`token`?i`<label class="field field-block"><span>Admin token</span><input id="cond-token" type="password" autocomplete="current-password" required autofocus /></label>
              <label class="switch"><input id="cond-remember" type="checkbox" /> remember on this device</label>`:m}
        ${this.error?i`<p class="err-ink small" role="alert">${this.error}</p>`:m}
        <div class="modal-actions">
          <button type="button" class="btn" @click=${()=>this.done(!1)}>Cancel</button>
          <button type="submit" class="btn btn-primary" data-confirm>${this.confirmLabel}</button>
        </div>
      </form>`}};r([h({attribute:!1})],z.prototype,`kind`,void 0),r([h({attribute:!1})],z.prototype,`heading`,void 0),r([h({attribute:!1})],z.prototype,`lines`,void 0),r([h({attribute:!1})],z.prototype,`confirmLabel`,void 0),r([h({attribute:!1})],z.prototype,`done`,void 0),r([a()],z.prototype,`error`,void 0),z=r([e(`qse-cond-dialog`)],z);function B(e){return new Promise(t=>{let n=document.createElement(`qse-cond-dialog`);e(n),n.done=(e,r,i)=>{n.remove(),t({ok:e,token:r,remember:i})},document.body.appendChild(n)})}function V(e,t,n){return B(r=>{r.kind=`confirm`,r.heading=e,r.lines=t,r.confirmLabel=n}).then(e=>e.ok)}async function me(e=`Changing the conductor needs the admin token (ops/make-secrets.sh made it; it is in secrets.env on the box).`){let t=await B(t=>{t.kind=`token`,t.heading=`Unlock editing`,t.lines=[e,`It stays in this browser tab, or on this device if you tick the box, and is sent only with a change.`],t.confirmLabel=`Unlock`});return t.ok&&t.token&&re(t.token,!!t.remember),t.ok&&!!t.token}async function H(e){try{return await e()}catch(t){if(t instanceof _&&t.type===`token_needed`&&(ne()&&re(null),await me()))return await e();throw t}}var U=class{constructor(e,t,n=null){this.host=e,this.kind=t,this.name=n,this.base=null,this.draft=null,this.loading=!0,this.loadError=null,this.errors=[],this.changes=[],this.note=``,this.saving=!1,this.saved=null,this.saveError=null,this.versions=null,this.showVersions=!1,this.showDiff=!1,this.onSaved=null,this.checkTimer=null}get dirty(){return this.changes.length>0}get locked(){return this.base?.locked??null}async load(){this.loading=!0,this.host.requestUpdate();try{this.base=await D.doc(this.kind,this.name),this.draft=ue(this.base.doc),this.errors=[],this.changes=[],this.loadError=null}catch(e){let t=s(e);this.loadError=`${t.endpoint}: ${t.message}`}finally{this.loading=!1,this.host.requestUpdate()}}set(e){this.draft=e,this.saved=null,this.host.requestUpdate(),this.checkTimer&&clearTimeout(this.checkTimer),this.checkTimer=setTimeout(()=>void this.check(),250)}async check(){if(this.draft){try{let e=await D.check(this.kind,this.name,this.draft);this.errors=e.errors,this.changes=e.changes}catch(e){this.errors=[s(e).message]}this.host.requestUpdate()}}reset(){this.base&&(this.draft=ue(this.base.doc),this.errors=[],this.changes=[],this.host.requestUpdate())}async save(){if(!this.draft||!this.base)return!1;if(!this.note.trim())return this.saveError=`Write a one-line change note first.`,this.host.requestUpdate(),!1;this.saving=!0,this.saveError=null,this.host.requestUpdate();try{let e=await H(()=>D.put(this.kind,this.name,this.draft,this.note.trim(),this.base.version));return this.saved=`version ${e.version}, reloaded ${e.loaded_at?o(e.loaded_at):`now`}`,this.note=``,await this.load(),this.showVersions&&await this.loadVersions(),this.onSaved?.(),!0}catch(e){let t=e;return this.saveError=t.status===409?`${t.message}. Reload to see it.`:t.message,t.errors?.length&&(this.errors=t.errors),!1}finally{this.saving=!1,this.host.requestUpdate()}}async loadVersions(){try{this.versions=(await D.versions(this.kind,this.name)).versions}catch(e){this.saveError=s(e).message}this.host.requestUpdate()}async restore(e){if(await V(`Restore version ${e}?`,[`A new version equal to version ${e} becomes current, with the note "restore of v${e}". The server loads it for the next request.`],`Restore`))try{let t=await H(()=>D.restore(this.kind,this.name,e));this.saved=`version ${t.version} (restore of v${e}), reloaded ${t.loaded_at?o(t.loaded_at):`now`}`,await this.load(),await this.loadVersions(),this.onSaved?.()}catch(e){this.saveError=s(e).message,this.host.requestUpdate()}}};function W(e){return e.errors.length?i`<ul class="errors" role="alert" data-errors>${e.errors.slice(0,12).map(e=>i`<li>${e}</li>`)}</ul>`:m}function he(e){return e.length?i`<div class="diff" data-diff>
    ${e.map(e=>i`<div class="diff-row"><span>${de(e.path)}</span><span class="diff-before">${N(e.before,160)}</span><span class="diff-after">${N(e.after,160)}</span></div>`)}
  </div>`:i`<p class="muted small">No changes.</p>`}function G(e,t){let n=e.base?.meta;return e.locked?i`<div class="savebar" data-savebar><span class="badge badge-lock">${g(`warn`)} locked</span><span class="status">${e.locked}. Every control is read-only.</span></div>`:i`
    ${e.showDiff?i`<section class="panel" data-diff-panel><header class="panel-head"><h2 class="panel-title">Changes against live (v${n?.version??0})</h2></header>${he(e.changes)}</section>`:m}
    ${e.showVersions?i`<section class="panel" data-versions><header class="panel-head"><h2 class="panel-title">Versions of ${t}</h2></header>
          ${e.versions==null?i`<p class="muted small">Loading…</p>`:i`<div class="table-scroll"><table class="table">
                <thead><tr><th>version</th><th>note</th><th>author</th><th>when</th><th></th></tr></thead>
                <tbody>${e.versions.map(t=>i`<tr data-version=${t.version}>
                    <td class="num">v${t.version}${t.current?i` <span class="badge badge-on">live</span>`:m}</td>
                    <td class="cell-wrap">${t.note??`—`}</td>
                    <td class="muted">${t.author??`—`}</td>
                    <td class="num muted">${t.created_at?l(t.created_at):`built in`}</td>
                    <td>${t.current?m:i`<button class="btn btn-sm" @click=${()=>e.restore(t.version)}>Restore</button>`}</td>
                  </tr>`)}</tbody></table></div>`}
        </section>`:m}
    <div class="savebar" data-savebar>
      <span class="status num" title=${n?.note??``}>live v${n?.version??0}${n?.loaded_at?i` · reloaded ${o(n.loaded_at)}`:m}</span>
      <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>e.note=t.target.value} aria-label="change note" maxlength="200" />
      <button class="btn btn-primary" ?disabled=${!e.dirty||e.saving||e.errors.length>0} @click=${()=>e.save()} data-save>${e.saving?`Saving…`:`Save as new version`}</button>
      <button class="btn btn-ghost" ?disabled=${!e.dirty} @click=${()=>e.reset()}>Reset</button>
      <button class="btn btn-ghost ${e.showDiff?`is-on`:``}" @click=${()=>{e.showDiff=!e.showDiff,e.check()}}>Diff (${e.changes.length})</button>
      <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():e.host.requestUpdate()}} data-versions-btn>Versions</button>
      ${e.saved?i`<span class="status" role="status" data-saved>${g(`check`)} ${e.saved}</span>`:m}
      ${e.saveError?i`<span class="err-ink small" role="alert">${e.saveError}</span>`:m}
    </div>`}var K=class extends n{constructor(...e){super(...e),this.row=null,this.ids=[],this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.row){this.labelling=!0;try{await H(()=>D.label(this.row.id,e)),this.row={...this.row,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=s(e).message}finally{this.labelling=!1}}}render(){let e=this.row;if(!e)return m;let n=e.stages.find(e=>!e.error&&!e.below);return i`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="decision" data-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title">${F(e.class_id,this.ids)} ${e.mode===`shadow`?i`<span class="badge badge-shadow">shadow</span>`:i`<span class="badge badge-on">on</span>`}</div>
            <div class="muted small num">${l(new Date(e.ts_ms).toISOString())} · ${e.profile} v${e.profile_version} · router v${e.router_version}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${g(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">
            <h3>Decision</h3>
            <dl class="kv">
              <div><dt>class</dt><dd>${e.class_id??`—`} ${I(e.confidence)}</dd></div>
              <div><dt>decided by</dt><dd>${R(e.source)}${e.kept?i` <span class="badge badge-warn">${e.kept.replace(/_/g,` `)}</span>`:m}</dd></div>
              <div><dt>classifier time</dt><dd class="num">${t(e.classifier_ms)}</dd></div>
              <div><dt>effort</dt><dd>${e.effort??`unchanged`}${e.budget==null?m:i` <span class="muted num">budget ${e.budget}</span>`} <span class="muted">(${e.budget_source??`server`})</span></dd></div>
              <div><dt>thinking</dt><dd>${e.thinking??`on`}</dd></div>
              <div><dt>modes</dt><dd class="num">${Object.entries(e.decisions??{}).map(([e,t])=>`${e} ${t}`).join(` · `)}</dd></div>
            </dl>
          </section>
          ${e.stages.length?i`<section class="drawer-section"><h3>Classifier stages</h3><div class="stage-list">
                ${e.stages.map(e=>i`<div class="stage ${e===n?`is-accepted`:``} ${e.error?`is-error`:``}">
                    <strong>${e.stage===`decider`?`decider`:`Qwen`}</strong>
                    <div>${e.error?i`<span class="err-ink">${e.error}</span>`:i`${e.label} ${I(e.confidence,e.accept??.6)}${e.below?i` <span class="muted small">below ${e.accept}</span>`:m}`}
                      ${e.probs&&Object.keys(e.probs).length>1?i`<div class="probs">${Object.entries(e.probs).sort((e,t)=>t[1]-e[1]).slice(0,6).map(([e,t])=>i`<div class="prob"><span>${e}</span><span class="prob-bar"><span style="width:${Math.round(t*100)}%"></span></span><span>${t.toFixed(2)}</span></div>`)}</div>`:m}
                    </div>
                    <span class="num muted">${t(e.ms)}${e.cached?` (cached)`:``}</span>
                  </div>`)}</div></section>`:m}
          ${(e.injected??[]).map(e=>i`<section class="drawer-section" data-injected>
              <h3>Injected at token ${e.pos.toLocaleString(`en-US`)} · ${e.tokens.toLocaleString(`en-US`)} tokens ${ee(e.text,`Copy the injected text`)}</h3>
              <p class="muted small">Because ${e.reason}. Parts: ${e.parts.map(e=>`${e.kind} ${e.name}`).join(`, `)}. The client never saw this text.</p>
              <pre class="note-text">${e.text}</pre>
            </section>`)}
          ${e.replayed?.count?i`<p class="muted small">Replayed ${e.replayed.count} earlier note(s), ${e.replayed.tokens.toLocaleString(`en-US`)} tokens, at the same positions, so the resident prefix kept matching.</p>`:m}
          ${e.shadow?i`<section class="drawer-section"><h3>Shadow (not applied)</h3><pre class="note-text">${JSON.stringify(e.shadow,null,2)}</pre></section>`:m}
          ${e.digest?i`<section class="drawer-section"><h3>What the classifier read</h3><pre class="note-text">${e.digest}</pre></section>`:m}
          <section class="drawer-section">
            <h3>Outcome</h3>
            ${e.outcome?i`<dl class="kv">
                  <div><dt>finish</dt><dd>${e.outcome.finish_reason??`running`}</dd></div>
                  <div><dt>tool calls</dt><dd class="num">${e.outcome.tool_calls??0} ${(e.outcome.tool_names??[]).join(`, `)}</dd></div>
                  <div><dt>reasoning tokens</dt><dd class="num">${e.outcome.reasoning_tokens??`—`}</dd></div>
                  <div><dt>time</dt><dd class="num">${t(e.outcome.total_ms)}</dd></div>
                  ${e.next_checker?.length?i`<div><dt>next checker</dt><dd>${e.next_checker.map(e=>i`<span class="badge ${e.verdict===`PASS`?`badge-ok`:`badge-warn`}">${e.verdict}</span> <span class="muted small">${e.checker} ${e.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:m}
                </dl>`:i`<p class="muted small">The turn is still running.</p>`}
          </section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${e.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${g(`check`)} this was right</button>
              <button class="btn ${e.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${g(`x`)} this was wrong</button>
              ${e.label?i`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:m}
              ${this.msg?i`<span class="muted small" role="status">${this.msg}</span>`:m}
            </div>
            <p class="muted small">Labels feed the precision on the Compare screen and the promotion rule.</p>
          </section>
          <p class="muted small num">decision ${e.id} · request ${e.request_id??`—`} · session ${N(e.session_key,40)}</p>
        </div>
      </aside>`}};r([h({attribute:!1})],K.prototype,`row`,void 0),r([h({attribute:!1})],K.prototype,`ids`,void 0),r([h({attribute:!1})],K.prototype,`close`,void 0),r([a()],K.prototype,`labelling`,void 0),r([a()],K.prototype,`msg`,void 0),K=r([e(`qse-cond-decision`)],K);function ge(e,t,n,r){return i`<button class="feed-row" data-decision=${e.id} @click=${()=>n(e)}>
    <span class="feed-time">${o(new Date(e.ts_ms).toISOString())}</span>
    <span class="feed-cls">${F(e.class_id,t)}</span>
    <span class="feed-conf">${I(e.confidence)} <span class="muted small">${R(e.source)}</span></span>
    <span class="feed-what">${r}</span>
    <span class="feed-act">${e.acted?i`<span class="badge badge-on">acted</span>`:m}${e.mode===`shadow`||e.shadow?i` <span class="badge badge-shadow">shadow</span>`:m}${!e.acted&&!(e.mode===`shadow`||e.shadow)?i`<span class="badge">pass</span>`:m}</span>
  </button>`}var q=class extends n{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=c(()=>this.loadStats(),2e4),this.feedPoll=c(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await D.stats(24),this.error=null}catch(e){let t=s(e);this.error=`${t.endpoint}: ${t.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await D.decisions({limit:20})).decisions}catch{}}render(){let e=this.stats;if(this.error&&!e)return i`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${this.error}</div></div>`;if(!e)return d(6);let n=e.classifier,r=n.decider,a=n.qwen,o=Object.entries(e.roles),s=o.filter(([,e])=>e.up).length,c=e.monitor;return i`
      <div class="stats" data-tiles>
        ${f({label:`decisions, 24 h`,value:S(e.decisions),sub:i`<span class="num">${S(e.acted)}</span> acted · <span class="num">${S(e.shadow)}</span> in shadow`,title:`${e.injections} notes, ${e.injected_tokens} tokens injected`})}
        ${f({label:`gave up, 24 h`,value:c?.gave_up==null?`—`:S(c.gave_up),sub:c?`the session monitor`:`the session monitor is not on this server yet`,na:!c})}
        ${f({label:`classifier p50 / p95`,value:r?.p50_ms==null?`—`:`${Math.round(r.p50_ms)} / ${Math.round(r.p95_ms??0)}`,unit:r?.p50_ms==null?void 0:`ms`,sub:i`decider ${S(r?.calls??0)} calls${a?i` · Qwen ${S(a.calls)} (${t(a.p50_ms)})`:m} · fail-open ${x(e.fail_open_rate,1)}`,warn:(e.fail_open_rate??0)>.1})}
        ${f({label:`roles online`,value:`${s} / ${o.length}`,sub:i`${o.map(([e,t])=>i`<span title=${t.error??``} class=${t.up?``:t.mode===`off`?`muted`:`err-ink`} data-role=${e}>${t.up?`●`:`○`} ${e} </span>`)}`,warn:o.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${y(`Decisions an hour, last 24 h`,e.decisions?i`<qse-ribbon .series=${le(e,this.ids)} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:u(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${y(`Task classes, 24 h`,i`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${e.classes.map(e=>i`<tr><td>${F(e.class_id,this.ids)}</td><td class="num num-col">${S(e.decisions)}</td><td class="num num-col">${e.labelled?`${e.agree} / ${e.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(e.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(e.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${y(i`Live decisions ${this.paused?i`<span class="badge">paused</span>`:i`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?i`<div class="feed" data-feed>${this.feed.map(e=>ge(e,this.ids,e=>this.open=e,P(e)))}</div>`:i`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:i`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${g(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${g(`external`)}</a>`})}
        ${y(`Profiles by client`,i`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(t=>i`<button class="pick" @click=${()=>v({screen:`profiles`,edit:e.name})}><span class="muted">${t}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> or the raw model name always serves the engine untouched.</p>
            <dl class="kv">${this.tiles.map(e=>i`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${this.open?i`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:m}
    `}};r([h({attribute:!1})],q.prototype,`ids`,void 0),r([h({attribute:!1})],q.prototype,`tiles`,void 0),r([h({attribute:!1})],q.prototype,`active`,void 0),r([a()],q.prototype,`stats`,void 0),r([a()],q.prototype,`feed`,void 0),r([a()],q.prototype,`error`,void 0),r([a()],q.prototype,`open`,void 0),r([a()],q.prototype,`paused`,void 0),q=r([e(`qse-cond-overview`)],q);var _e=[`off`,`low`,`medium`,`high`,`xhigh`],J=class extends n{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new U(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),D.doc(`packages`).then(e=>this.packages=Object.keys(e.doc.packages)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(M(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await D.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=s(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return d(8);if(e.loadError)return i`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=t.classes.map(e=>e.id);return i`
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
          ${_e.map(e=>i`<div class="row-card" data-effort=${e}>
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
      ${W(e)}
      ${G(e,`the router`)}
    `}renderTest(e){let n=this.result;return i`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?i`<span class="err-ink small" role="alert">${this.testError}</span>`:m}
      </div>
      ${n?n.active?i`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${F(n.class_id??null,e)} ${I(n.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${R(n.source??null)}${n.kept?i` <span class="badge badge-warn">${n.kept}</span>`:m} <span class="muted num">${t(n.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${n.effort??`unchanged`} · thinking ${n.thinking??`on`}${n.budget==null?``:` · budget ${n.budget}`}</dd></div>
                  <div><dt>profile</dt><dd>${n.profile} (${n.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(n.stages??[]).map(e=>i`<div class="stage ${e.error?`is-error`:e.below?``:`is-accepted`}"><strong>${e.stage===`decider`?`decider`:`Qwen`}</strong><span>${e.error??i`${e.label} ${I(e.confidence,e.accept??.6)}`}</span><span class="num muted">${t(e.ms)}</span></div>`)}</div>
              </div>
              <div>${n.note?i`<p class="muted small">The model would see this at the end of the last turn (${(n.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${n.note}</pre>`:i`<p class="muted small">No note: ${n.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:i`<p class="muted small" style="margin-top:12px">Profile ${n.profile} is raw (${n.mode}): the conductor does nothing for it.</p>`:m}
    </section>`}renderClass(e,t,n,r){let a=e.policy,o=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return i`<details class="row-card" data-class-row=${e.id} ?open=${o}>
      <summary>
        <span>${F(e.id,r,e.label)} <span class="muted small mono">${a.effort??`medium`}${(a.packages??[]).length?` · ${(a.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===0} @click=${()=>this.moveClass(t,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===n-1} @click=${()=>this.moveClass(t,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n<=1} @click=${()=>this.removeClass(t)} aria-label=${`remove ${e.id}`}>${g(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${e.id} ?readonly=${!o} @input=${e=>this.upd([`classes`,t,`id`],e.target.value)} data-class-id /><span class="hint">${o?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${e.label} @input=${e=>this.upd([`classes`,t,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${e.description} @input=${e=>this.upd([`classes`,t,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${L(_e,a.effort??`medium`,e=>this.upd([`classes`,t,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let n=(a.packages??[]).includes(e);return i`<button class="pick ${n?`is-on`:``}" aria-pressed=${n?`true`:`false`} @click=${()=>this.upd([`classes`,t,`policy`,`packages`],n?(a.packages??[]).filter(t=>t!==e):[...a.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(a.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,t,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}num(e,t,n,r={}){return i`<label class="field"><span>${e}</span><input type="number" .value=${String(t)} step=${r.step??1} min=${r.min??``} max=${r.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&n(t)}} />${r.hint?i`<span class="hint">${r.hint}</span>`:m}</label>`}text(e,t,n){return i`<label class="field"><span>${e}</span><input .value=${t} @input=${e=>n(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(M(n,[`classifier`,`stages`],r))}};r([h({attribute:!1})],J.prototype,`ids`,void 0),r([h({attribute:!1})],J.prototype,`meta`,void 0),r([h({attribute:!1})],J.prototype,`active`,void 0),r([h({attribute:!1})],J.prototype,`onSaved`,void 0),r([a()],J.prototype,`packages`,void 0),r([a()],J.prototype,`testText`,void 0),r([a()],J.prototype,`testProfile`,void 0),r([a()],J.prototype,`testTools`,void 0),r([a()],J.prototype,`testing`,void 0),r([a()],J.prototype,`result`,void 0),r([a()],J.prototype,`testError`,void 0),J=r([e(`qse-cond-routing`)],J);var ve=[`off`,`low`,`medium`,`high`,`xhigh`],ye={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},Y=class extends n{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleTest={},this.ed=null,this.roles=new U(this,`roles`),this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.loadHealth(),D.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),D.doc(`packages`).then(e=>this.packages=Object.keys(e.doc.packages)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new U(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{this.roleHealth=(await D.roles()).health}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let n=await D.roleTest(e);this.roleTest={...this.roleTest,[e]:n.ok?`ok in ${t(n.ms)}: "${n.answer?.label}" at ${n.answer?.confidence.toFixed(2)}`:`failed: ${n.error}`}}catch(t){this.roleTest={...this.roleTest,[e]:s(t).message}}this.loadHealth()}render(){return i`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>i`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>v({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?i`<span class="badge badge-lock">${g(`warn`)} locked</span>`:i`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${l(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):i`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(M(this.ed.draft,e,t))}renderEditor(e){if(e.loading&&!e.draft)return d(8);if(e.loadError)return i`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=!!e.locked,r=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],a=this.router?.classes??[];return i`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${g(`playground`)} Try in Playground</a></div>
      ${n?i`<div class="note" data-lock-reason>${g(`warn`)} ${e.locked}. Every control below is read-only.</div>`:m}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${t.description} ?disabled=${n} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${L([`raw`,`shadow`,`on`],t.mode,e=>this.upd([`mode`],e),{disabled:n,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${r.map(e=>{let r=t.clients.includes(e);return i`<button class="pick ${r?`is-on`:``}" ?disabled=${n} aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`clients`],r?t.clients.filter(t=>t!==e):[...t.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${n} @change=${e=>this.upd([`default_class`],e.target.value)}>${a.map(e=>i`<option value=${e.id} ?selected=${e.id===t.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(ye).map(e=>i`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${L([`off`,`shadow`,`on`],t.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:n,small:!0,label:e})}<span class="small">${ye[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${a.map(e=>{let r=t.routing[e.id]??{},a=Object.keys(r).length>0;return i`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${F(e.id,this.ids,e.label)}
                ${a?i`<button class="btn btn-ghost btn-sm" ?disabled=${n} @click=${()=>{let n={...t.routing};delete n[e.id],this.upd([`routing`],n)}}>use the router's</button>`:i`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${L(ve,r.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...r,effort:t}),{disabled:n,small:!0,label:`effort override of ${e.id}`,cls:r.effort?``:`is-inherited`})}</div>
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
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${L(ve,t.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:n,small:!0,label:`default effort`})}</div>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.thinking.hints} @change=${e=>this.upd([`thinking`,`hints`],e.target.checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113)</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${[`risk`,`project`,`same_call`,`malformed`].map(e=>i`<label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${e.replace(`_`,` `)}</label>`)}
          <label class="field"><span>Tier 1 timeout (ms)</span><input type="number" ?disabled=${n} .value=${String(t.toolgate.tier1_timeout_ms)} @input=${e=>this.upd([`toolgate`,`tier1_timeout_ms`],Math.round(Number(e.target.value)))} /></label>
        </div></div>
      </details>
      <details class="sec" data-sec="context">
        <summary>Context <span class="sec-sub">session-level tasks S1 to S4</span></summary>
        <div class="sec-body"><div class="form-grid">
          <label class="field"><span>S1: fails of one gate before a note</span><input type="number" ?disabled=${n} .value=${String(t.context.s1_runs)} @input=${e=>this.upd([`context`,`s1_runs`],Math.round(Number(e.target.value)))} /></label>
          <label class="field"><span>S4: minutes without progress</span><input type="number" ?disabled=${n} .value=${String(t.context.s4_minutes)} @input=${e=>this.upd([`context`,`s4_minutes`],Math.round(Number(e.target.value)))} /></label>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.context.notify} @change=${e=>this.upd([`context`,`notify`],e.target.checked)} /> NOTIFY (ntfy to the phone) <span class="badge badge-warn">needs Khaled's approval</span></label>
        </div></div>
      </details>
      <details class="sec" data-sec="hints">
        <summary>Speed hints <span class="badge badge-warn">not gated</span></summary>
        <div class="sec-body">
          <p class="warn-ink small">No SPD measurement is in the journal yet (SPD-73); the router's ≤ 1 ulp text rule and the never-worse gate come first.</p>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.stair_width} @change=${e=>this.upd([`hints`,`stair_width`],e.target.checked)} /> StairCut width hint</label>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.prefix_retention} @change=${e=>this.upd([`hints`,`prefix_retention`],e.target.checked)} /> prefix retention hint</label>
        </div>
      </details>
      ${W(e)}
      ${G(e,`profile ${e.name}`)}
    </div>`}renderTask(e,t,n,r,a){let o=this.meta?.tasks[e],s=o?.role&&o.tier===`1`,c=!s||!!this.roleHealth[o.role]?.up;return i`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${o?.family??``}</div></div>
      <div>${o?.label??e}<div class="task-why">${o?.why??``}</div>${s&&!c?i`<div class="warn-ink small">${g(`warn`)} the ${o.role} role is offline: act is refused</div>`:m}</div>
      <div class="task-mode">${L([`off`,`shadow`,`act`],t,t=>this.upd([`tasks`,e,`mode`],t),{disabled:a,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${Object.entries(r).map(([t,n])=>{let r=o?.thresholds[t];return i`<label class="field ${r&&r.default!==n?`is-changed`:``}"><span>${t.replace(/_/g,` `)} <span class="hint">${r?.unit??``}${r?`, default ${r.default}`:``}</span></span>
            <input type="number" ?disabled=${a} .value=${String(n)} step="any" @input=${n=>{let r=Number(n.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,t],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${a} min="0" max="10" .value=${String(n)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let e=this.roles;if(!e.draft)return e.loadError?i`<p class="err-ink small">${e.loadError}</p>`:m;let n=e.draft,r=(t,r)=>e.set(M(n,t,r));return i`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${Object.entries(n).map(([e,n])=>{let a=this.roleHealth[e];return i`<div class="row-card" data-role-row=${e}>
            <div class="row-card-head"><strong>${e}</strong>
              <span class="toolbar" style="margin:0">${a?i`<span class="badge ${a.up?`badge-ok`:n.mode===`off`?``:`badge-warn`}">${a.up?`up ${t(a.ms)}`:n.mode===`off`?`off`:`down: ${a.error??`?`}`}</span>`:m}
                <button class="btn btn-sm" @click=${()=>this.testRole(e)} data-role-test=${e}>Test</button></span></div>
            ${this.roleTest[e]?i`<p class="small" role="status">${this.roleTest[e]}</p>`:m}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${n.endpoint} @input=${t=>r([e,`endpoint`],t.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${n.model} @input=${t=>r([e,`model`],t.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(n.timeout_ms)} @input=${t=>r([e,`timeout_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>memory claim (GB)</span><input type="number" step="0.1" .value=${String(n.memory_gb)} @input=${t=>r([e,`memory_gb`],Number(t.target.value))} /></label>
            </div>
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${L([`off`,`shadow`,`on`],n.mode,t=>r([e,`mode`],t),{small:!0,label:`${e} mode`})}<span class="muted small mono">${n.contract}</span></div>
          </div>`})}
        ${W(e)}
        ${G(e,`the roles`)}
      </div>
    </details>`}};r([h({attribute:!1})],Y.prototype,`ids`,void 0),r([h({attribute:!1})],Y.prototype,`meta`,void 0),r([h({attribute:!1})],Y.prototype,`tiles`,void 0),r([h({attribute:!1})],Y.prototype,`selected`,void 0),r([h({attribute:!1})],Y.prototype,`onSaved`,void 0),r([a()],Y.prototype,`router`,void 0),r([a()],Y.prototype,`packages`,void 0),r([a()],Y.prototype,`roleHealth`,void 0),r([a()],Y.prototype,`roleTest`,void 0),Y=r([e(`qse-cond-profiles`)],Y);var X=class extends n{constructor(...e){super(...e),this.ids=[],this.router=null,this.sel=null,this.found=null,this.scanError=null,this.ed=new U(this,`packages`)}connectedCallback(){super.connectedCallback(),this.ed.load(),D.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0)}usedBy(e){return(this.router?.classes??[]).filter(t=>(t.policy.packages??[]).includes(e)).map(e=>e.id)}upd(e,t){this.ed.draft&&this.ed.set(M(this.ed.draft,e,t))}async scan(){this.scanError=null;try{this.found=(await D.scan()).skills}catch(e){this.scanError=s(e).message}}register(e){let t=this.ed.draft,n={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.ed.set(M(t,[`packages`,e.name],n)),this.sel=e.name}addInline(){let e=this.ed.draft,t=1;for(;e.packages[`package-${t}`];)t++;let n=`package-${t}`;this.ed.set(M(e,[`packages`,n],{title:`Package ${t}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0})),this.sel=n}removePackage(e){let t=this.ed.draft,n={...t.packages};delete n[e],this.ed.set({...t,packages:n}),this.sel=null}render(){let e=this.ed;if(e.loading&&!e.draft)return d(6);if(e.loadError)return i`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=e.base?.status??{};return i`
      <section class="panel" data-packages>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills</h2><p class="panel-sub">The router injects one into the prompt when the turn's class names it and the client did not load it itself.</p></div>
          <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-add-package>${g(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${g(`search`)} Skills on the box</button></div></header>
        <div class="table-scroll"><table class="table">
          <thead><tr><th>name</th><th>source</th><th>on the box</th><th class="num-col">tokens</th><th>classes</th><th>enabled</th></tr></thead>
          <tbody>${Object.entries(t.packages).map(([e,t])=>{let r=n[e];return i`<tr class="req-row ${this.sel===e?`is-open`:``}" data-package-row=${e} @click=${()=>this.sel=e}>
              <td><strong>${e}</strong><div class="muted small">${t.title}</div></td>
              <td class="mono small">${t.source}</td>
              <td>${r?r.available?i`<span class="badge badge-ok">found</span>`:i`<span class="badge badge-warn" title=${r.path??``}>${r.error??`missing`}</span>`:i`<span class="badge">not saved</span>`}</td>
              <td class="num num-col">${r?S(r.tokens_est):`—`}</td>
              <td>${this.usedBy(e).map(e=>F(e,this.ids))}</td>
              <td>${t.enabled?`yes`:`no`}</td>
            </tr>`})}</tbody></table></div>
      </section>
      ${this.found?i`<section class="panel" style="margin-top:16px" data-found>
            <header class="panel-head"><h2 class="panel-title">Skills in ${t.skill_dirs.join(`, `)}</h2></header>
            ${this.found.length?i`<div class="table-scroll"><table class="table"><tbody>${this.found.map(e=>i`<tr><td><strong>${e.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${e.description}</div></td><td class="num">${S(e.chars)} chars</td>
                    <td>${e.registered||t.packages[e.name]?i`<span class="badge badge-ok">registered</span>`:i`<button class="btn btn-sm" @click=${()=>this.register(e)} data-register=${e.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:i`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ spark:~/.config/opencode/skills/</code>).</p>`}
          </section>`:this.scanError?i`<p class="err-ink small">${this.scanError}</p>`:m}
      ${this.sel&&t.packages[this.sel]?this.renderEditor(this.sel,t.packages[this.sel]):m}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${t.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${W(e)}
      ${G(e,`the packages`)}
    `}renderEditor(e,t){let n=t.source===`inline`,r=this.usedBy(e);return i`<section class="panel" style="margin-top:16px" data-package-editor=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e}</h2><p class="panel-sub">${r.length?i`used by ${r.join(`, `)}`:`no class uses it yet (add it on the Routing screen)`}</p></div>
        <div class="panel-tools"><button class="btn btn-sm btn-ghost" ?disabled=${r.length>0} title=${r.length?`a class uses it`:``} @click=${()=>this.removePackage(e)}>${g(`trash`)} Remove</button></div></header>
      <div class="form-grid">
        <label class="field"><span>title</span><input .value=${t.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>source</span><input .value=${t.source} @input=${t=>this.upd([`packages`,e,`source`],t.target.value)} /><span class="hint">inline, skill:&lt;name&gt; or file:&lt;path on the box&gt;</span></label>
        <label class="field"><span>max tokens</span><input type="number" .value=${String(t.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">longer text is cut at a line end</span></label>
        <label class="switch"><input type="checkbox" .checked=${t.enabled} @change=${t=>this.upd([`packages`,e,`enabled`],t.target.checked)} /> enabled</label>
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" .value=${t.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${n?i`<label class="field field-block"><span>text the model gets</span><textarea rows="14" data-package-body .value=${t.body} @input=${t=>this.upd([`packages`,e,`body`],t.target.value)}></textarea></label>`:i`<p class="muted small">The text is the file on the box, read when first needed and again when it changes; its YAML front matter is left out.</p>`}
    </section>`}};r([h({attribute:!1})],X.prototype,`ids`,void 0),r([a()],X.prototype,`router`,void 0),r([a()],X.prototype,`sel`,void 0),r([a()],X.prototype,`found`,void 0),r([a()],X.prototype,`scanError`,void 0),X=r([e(`qse-cond-packages`)],X);var Z=class extends n{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=c(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await D.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let t=s(e);this.error=`${t.endpoint}: ${t.message}`}}async loadSession(e){try{let t=await D.session(e);e===this.selected&&(this.turns=t.turns)}catch(t){e===this.selected&&(this.turns=[]),this.error=s(t).message}}render(){if(this.error&&!this.list)return i`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${this.error}</div></div>`;if(!this.list)return d(6);let e=[...new Set(this.list.map(e=>e.client_kind??`other`))],t=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return i`
      ${this.selected?this.renderSession(this.selected):m}
      ${y(`Sessions, last 7 days`,t.length?i`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${t.map(e=>i`<tr class="req-row ${e.key===this.selected?`is-open`:``}" data-session=${e.key} @click=${()=>v({session:e.key})}>
                  <td class="mono small cell-wrap">${N(e.key,28)}</td><td>${e.client_kind??`—`}</td><td>${e.profile??`—`}</td>
                  <td>${F(e.class_id,this.ids)}</td><td class="num num-col">${S(e.turns)}</td>
                  <td class="num num-col">${e.notes?`${e.notes} (${S(e.note_tokens)} tok)`:`—`}</td>
                  <td>${e.last_checker?i`<span class="badge ${e.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${e.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${te(new Date(e.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:u(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:i`<label class="field field-search">${g(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${e.map(e=>i`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?i`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:m}
    `}renderSession(e){let t=this.turns,n=this.live.find(t=>t.key===e);if(t==null)return d(4);let r=t.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,a=t.reduce((e,t)=>e+(t.injected?.length??0),0),o=pe(t);return i`<section class="panel" style="margin-bottom:16px" data-timeline=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${N(e,44)} ${n?i`<span class="badge badge-on">live</span>`:m}</h2>
        <p class="panel-sub">${t.length} turns · ${a} hidden note${a===1?``:`s`}${n?` · class now ${n.class_id} · ${S(n.note_tokens)} note tokens replayed a turn`:``}${t.length?` · since ${l(new Date(t[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${D.exportUrl(e,`md`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${g(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${D.exportUrl(e,`json`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${g(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>v({session:null})}>${g(`x`)}</button></div></header>
      <div class="strip" role="list" aria-label="turns">
        ${t.map(e=>{let t=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/r*900)),n=(e.injected?.length??0)>0,a=!!e.shadow;return i`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${t}px; --cls: var(--series-${j(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${P(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${n?i`<span class="mark"></span>`:a?i`<span class="mark is-shadow"></span>`:m}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. ◆ a hidden note went in, ◇ shadow would have.</p>
      ${o.series.length?i`<qse-ribbon .series=${o.series} .labels=${o.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:m}
      <div class="feed" style="margin-top:10px">${t.slice().reverse().map(e=>ge(e,this.ids,e=>this.open=e,P(e)))}</div>
    </section>`}};r([h({attribute:!1})],Z.prototype,`ids`,void 0),r([h({attribute:!1})],Z.prototype,`selected`,void 0),r([a()],Z.prototype,`list`,void 0),r([a()],Z.prototype,`live`,void 0),r([a()],Z.prototype,`turns`,void 0),r([a()],Z.prototype,`error`,void 0),r([a()],Z.prototype,`query`,void 0),r([a()],Z.prototype,`client`,void 0),r([a()],Z.prototype,`open`,void 0),Z=r([e(`qse-cond-sessions`)],Z);var Q=class extends n{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={}}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([D.stats(168),D.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let t=s(e);this.error=`${t.endpoint}: ${t.message}`}}async promote(e,t,n,r,i){if(await V(`Promote ${n} to ${t===`task`?`act`:`on`} in ${e}?`,[t===`task`?`The session monitor will act on ${n} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${n} decisions instead of only journaling them: ${n===`effort`?`budgets and thinking off go to the engine`:n===`packages`?`skills and packages go into the prompt`:n===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${r.toFixed(2)} over ${i} labelled decisions. A new profile version records it.`],`Promote`))try{let a=await D.doc(`profiles`,e),o=t===`task`?M(a.doc,[`tasks`,n,`mode`],`act`):M(a.doc,[`decisions`,n],`on`),s=await H(()=>D.put(`profiles`,e,o,`${n} to ${t===`task`?`act`:`on`}, precision ${r.toFixed(2)} / ${i}`,a.version));this.msg=`${e} v${s.version}: ${n} promoted`,this.onSaved()}catch(e){this.msg=s(e).message}}async loadAb(){try{this.abVersions=(await D.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await D.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await D.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=s(e).message}}render(){return this.error&&!this.stats?i`<div class="state state-error" role="alert"><div class="state-title">${g(`warn`)} ${this.error}</div></div>`:this.stats?i`
      ${this.msg?i`<div class="note note-info" role="status">${this.msg}</div>`:m}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:d(8)}renderPromotion(e){let t=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,n=e.monitor?.precision??[];return y(`Shadow precision and promotion`,i`<p class="muted small">${k}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${e.precision.length||n.length?i`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${e.precision.map(e=>{let n=A(e),r=t(e.profile,e.decision);return i`<tr data-precision-row=${`${e.profile}/${e.decision}`}>
                    <td>${e.profile}</td><td class="mono">${e.decision}</td><td><span class="badge ${r===`on`?`badge-on`:r===`shadow`?`badge-shadow`:``}">${r}</span></td>
                    <td class="num num-col">${S(e.decisions)}</td><td class="num num-col">${S(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${n.allowed?``:`muted`}">${n.why}</td>
                    <td><button class="btn btn-sm ${n.allowed?`btn-primary`:``}" ?disabled=${!n.allowed||r!==`shadow`} title=${n.allowed?``:k} data-promote=${`${e.profile}/${e.decision}`} @click=${()=>this.promote(e.profile??``,`decision`,e.decision,e.precision??0,e.labelled)}>Promote to on</button></td></tr>`})}
                ${n.map(e=>{let t=A(e);return i`<tr data-precision-row=${`task/${e.task}`}><td>—</td><td class="mono">${e.task}</td><td><span class="badge ${e.mode===`act`?`badge-on`:`badge-shadow`}">${e.mode}</span></td>
                    <td class="num num-col">${S(e.decisions)}</td><td class="num num-col">${S(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${t.allowed?``:`muted`}">${t.why}</td>
                    <td><button class="btn btn-sm ${t.allowed?`btn-primary`:``}" ?disabled=${!t.allowed||e.mode!==`shadow`} data-promote=${`task/${e.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,e.task,e.precision??0,e.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:u(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${e.monitor?m:i`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let e=this.reports.find(e=>e.kind===`classifier`);if(!e)return y(`Classifier accuracy`,u(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let n=e.stages??{},r=e.set??{},a=(n[this.stage]??Object.values(n)[0])?.confusion??{},o=r.classes??Object.keys(a);return y(i`Classifier accuracy <span class="muted small num">${r.name??``} · ${r.n??`?`} prompts · ${e.created_at?l(String(e.created_at)):``}</span>`,i`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(n).map(([e,n])=>i`<tr class="req-row ${e===this.stage?`is-open`:``}" @click=${()=>this.stage=e}><td class="mono">${e}</td><td class="num num-col">${x(n.accuracy,1)}</td><td class="num num-col">${n.answered} / ${n.n}</td><td class="num num-col">${n.accuracy_answered==null?`—`:x(n.accuracy_answered,1)}</td><td class="num num-col">${t(n.p50_ms)}</td><td class="num num-col">${t(n.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${o.length&&Object.keys(a).length?i`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...o,`—`].map(e=>i`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${o.map(e=>i`<tr><th>${F(e,this.ids)}</th>${[...o,`—`].map(t=>{let n=a[e]?.[t]??0;return i`<td class="cm-cell ${e===t?`cm-diag`:n?`cm-miss`:``}">${n||``}</td>`})}</tr>`)}</tbody></table></div>`:i`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}`,{id:`cond-classifier`})}renderSuites(){let e=this.reports.filter(e=>e.kind===`suite`).flatMap(e=>e.rows??[]),t=se(this.reports,this.engine),n=oe(e);return y(`TandemLLM against the raw engine`,i`${t?i`<div class="guard-line" data-guard-detail>${g(`warn`)} ${t}</div>`:i`<p class="small">${g(`check`)} the never-worse gate ran for build ${this.engine.version}.</p>`}
        ${n.length?i`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">tokens</th><th class="num-col">interventions</th><th>build · night</th><th class="num-col">n</th></tr></thead>
              <tbody>${n.flatMap(e=>[e.raw,e.tandem].filter(Boolean).map((t,n)=>i`<tr data-suite=${e.suite}>
                    <td>${n===0?e.suite:``}${n===0&&!e.cmp.ok?i` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${e.cmp.why}</div>`:m}</td>
                    <td class="mono">${t.arm}</td><td class="num num-col">${t.gates_passed??`—`}${t.gates_total?` / ${t.gates_total}`:``}</td>
                    <td class="num num-col">${t.minutes_to_pass??`—`}</td><td class="num num-col">${t.tokens==null?`—`:S(t.tokens)}</td>
                    <td class="num num-col">${t.interventions??`—`}</td><td class="small">${N(t.build,16)} · ${t.night??`—`}</td><td class="num num-col">${t.n??`—`}</td></tr>`))}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only.</p>`:u(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite into the box's evals/ directory.`)}`,{id:`cond-suites`})}renderAb(){return y(`A/B of two profile versions`,i`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>i`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>i`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(t=>i`<option value=${t.version} ?selected=${t.version===this[e]}>v${t.version} ${N(t.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?i`<div class="diff" data-ab>${this.abChanges.map(e=>i`<div class="diff-row"><span>${de(e.path)}</span><span class="diff-before">${N(e.before,80)}</span><span class="diff-after">${N(e.after,80)}</span></div>`)}</div>`:i`<p class="muted small">The two versions are the same.</p>`:m}`,{id:`cond-ab`})}};r([h({attribute:!1})],Q.prototype,`ids`,void 0),r([h({attribute:!1})],Q.prototype,`tiles`,void 0),r([h({attribute:!1})],Q.prototype,`onSaved`,void 0),r([a()],Q.prototype,`stats`,void 0),r([a()],Q.prototype,`reports`,void 0),r([a()],Q.prototype,`engine`,void 0),r([a()],Q.prototype,`error`,void 0),r([a()],Q.prototype,`msg`,void 0),r([a()],Q.prototype,`stage`,void 0),r([a()],Q.prototype,`abProfile`,void 0),r([a()],Q.prototype,`abVersions`,void 0),r([a()],Q.prototype,`abA`,void 0),r([a()],Q.prototype,`abB`,void 0),r([a()],Q.prototype,`abChanges`,void 0),r([a()],Q.prototype,`abCounts`,void 0),Q=r([e(`qse-cond-compare`)],Q);var be=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`sessions`,label:`Sessions`},{id:`compare`,label:`Compare`}],$=class extends n{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.guard=null,this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=c(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&be.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r,i]=await Promise.all([D.profiles(),this.meta?Promise.resolve(this.meta):D.meta(),D.config(),D.doc(`router`),D.evals()]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.guard=se(i.reports,i.engine),this.error=null}catch(e){let t=s(e);this.error=t.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${t.endpoint}: ${t.message}`}}async setMode(e){let t=this.tiles.find(e=>e.name===this.active);if(t&&!t.locked&&(e!==`on`||await V(`Turn the ${t.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let n=await H(()=>D.setMode(t.name,e,`mode to ${e}`,t.version));this.reloaded=`v${n.version} · reloaded ${n.loaded_at?o(n.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=s(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),t=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return i`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${g(`conductor`)} Conductor <span class="muted small num">TandemLLM beside ${this.config?.server.raw_model??`the raw engine`}</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>v({profile:e.target.value})}>
                ${this.tiles.map(e=>i`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?L([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):m}
            ${this.reloaded?i`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:m}
          </div>
        </div>
        ${this.switchError?i`<div class="guard-line" role="alert">${g(`warn`)} ${this.switchError}</div>`:m}
        ${this.error?i`<div class="guard-line" role="alert">${g(`warn`)} ${this.error}</div>`:m}
        ${this.guard?i`<div class="guard-line" data-guard>${g(`warn`)} ${this.guard}: the raw engine's never-worse gate (both views, same hour) has no result for the code that serves now.</div>`:m}
        ${t.map(([e,t])=>i`<div class="guard-line" role="alert">${g(`warn`)} ${e}: ${t.error?.message} (kept version ${t.version})</div>`)}
        ${e?.locked?i`<div class="note" data-locked>${g(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:m}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${be.map(e=>i`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:m}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return i`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return i`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`profiles`:return i`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return i`<qse-cond-packages .ids=${e.ids}></qse-cond-packages>`;case`sessions`:return i`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`compare`:return i`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};r([h({attribute:!1})],$.prototype,`params`,void 0),r([a()],$.prototype,`tiles`,void 0),r([a()],$.prototype,`defaultProfile`,void 0),r([a()],$.prototype,`meta`,void 0),r([a()],$.prototype,`config`,void 0),r([a()],$.prototype,`ids`,void 0),r([a()],$.prototype,`guard`,void 0),r([a()],$.prototype,`error`,void 0),r([a()],$.prototype,`switching`,void 0),r([a()],$.prototype,`reloaded`,void 0),r([a()],$.prototype,`switchError`,void 0),$=r([e(`qse-conductor`)],$);export{$ as QseConductor};