import{C as e,_ as t,b as n,d as r,g as i,h as a,i as o,t as s,v as c,y as l}from"./chunk-CkiyBafN.js";import{d as u,m as d,o as f}from"./chunk-BXmDkkzu.js";import{A as p,K as m,O as h,S as g,Z as _,bt as v,mt as y,nt as b,yt as x}from"./app-DTFfdS-y.js";function S(t,n,r){let i=h(t,n);return e`<span class="cls" style="--cls: var(--series-${i})" data-class=${t??`none`}>${r??t??`—`}</span>`}function C(t,n=.6){return t==null?e`<span class="conf muted">—</span>`:e`<span class="conf" title=${`confidence ${t.toFixed(4)}`}><span class="conf-bar ${t<n?`is-low`:``}"><span style="width:${Math.round(t*100)}%"></span></span>${t.toFixed(2)}</span>`}function w(t,n,r,i={}){return e`<div class="seg ${i.small?`seg-sm`:``} ${i.cls??``}" role="radiogroup" aria-label=${i.label??`mode`}>
    ${t.map(t=>e`<button type="button" class="seg-btn mode-${t} ${t===n?`is-on`:``}" role="radio" aria-checked=${t===n?`true`:`false`} ?disabled=${i.disabled} @click=${()=>t!==n&&r(t)}>${t}</button>`)}
  </div>`}function T(e){return e?g[e]??e:`—`}var E=class extends i{constructor(...e){super(...e),this.heading=``,this.lines=[],this.confirmLabel=`Confirm`,this.done=()=>void 0,this.onKey=e=>{e.key===`Escape`&&(e.preventDefault(),this.done(!1))},this.onRoute=()=>this.done(!1)}connectedCallback(){super.connectedCallback(),document.addEventListener(`keydown`,this.onKey),window.addEventListener(`hashchange`,this.onRoute),window.addEventListener(`popstate`,this.onRoute)}disconnectedCallback(){document.removeEventListener(`keydown`,this.onKey),window.removeEventListener(`hashchange`,this.onRoute),window.removeEventListener(`popstate`,this.onRoute),super.disconnectedCallback()}submit(e){e.preventDefault(),this.done(!0)}render(){return e`<div class="modal-back" @click=${()=>this.done(!1)}></div>
      <form class="modal" role="dialog" aria-modal="true" aria-label=${this.heading} @submit=${e=>this.submit(e)}>
        <h2>${this.heading}</h2>
        ${this.lines.map(t=>e`<p class="muted small">${t}</p>`)}
        <div class="modal-actions">
          <button type="button" class="btn" @click=${()=>this.done(!1)}>Cancel</button>
          <button type="submit" class="btn btn-primary" data-confirm>${this.confirmLabel}</button>
        </div>
      </form>`}};a([c({attribute:!1})],E.prototype,`heading`,void 0),a([c({attribute:!1})],E.prototype,`lines`,void 0),a([c({attribute:!1})],E.prototype,`confirmLabel`,void 0),a([c({attribute:!1})],E.prototype,`done`,void 0),E=a([l(`qse-cond-dialog`)],E);function D(e){return new Promise(t=>{let n=document.createElement(`qse-cond-dialog`),r=document.activeElement instanceof HTMLElement?document.activeElement:null;e(n);let i=!1;n.done=e=>{i||(i=!0,n.remove(),r&&r.isConnected&&r.focus(),t({ok:e}))},document.body.appendChild(n)})}function O(e,t,n){return D(r=>{r.heading=e,r.lines=t,r.confirmLabel=n}).then(e=>e.ok)}var k=class{constructor(e,t,n=null){this.host=e,this.kind=t,this.name=n,this.base=null,this.draft=null,this.loading=!0,this.loadError=null,this.errors=[],this.changes=[],this.note=``,this.saving=!1,this.saved=null,this.saveError=null,this.versions=null,this.showVersions=!1,this.showDiff=!1,this.onSaved=null,this.checkTimer=null}get dirty(){return this.changes.length>0}get locked(){return this.base?.locked??null}async load(){this.loading=!0,this.host.requestUpdate();try{this.base=await y.doc(this.kind,this.name),this.draft=p(this.base.doc),this.errors=[],this.changes=[],this.loadError=null}catch(e){let t=r(e);this.loadError=`${t.endpoint}: ${t.message}`}finally{this.loading=!1,this.host.requestUpdate()}}set(e){this.draft=e,this.saved=null,this.host.requestUpdate(),this.checkTimer&&clearTimeout(this.checkTimer),this.checkTimer=setTimeout(()=>void this.check(),250)}async check(){if(this.draft){try{let e=await y.check(this.kind,this.name,this.draft);this.errors=e.errors,this.changes=e.changes}catch(e){this.errors=[r(e).message]}this.host.requestUpdate()}}reset(){this.base&&(this.draft=p(this.base.doc),this.errors=[],this.changes=[],this.host.requestUpdate())}async save(){if(!this.draft||!this.base)return!1;if(!this.note.trim())return this.saveError=`Write a one-line change note first.`,this.host.requestUpdate(),!1;this.saving=!0,this.saveError=null,this.host.requestUpdate();try{let e=await y.put(this.kind,this.name,this.draft,this.note.trim(),this.base.version);return this.saved=`version ${e.version}, reloaded ${e.loaded_at?d(e.loaded_at):`now`}`,this.note=``,await this.load(),this.showVersions&&await this.loadVersions(),this.onSaved?.(),!0}catch(e){let t=e;return this.saveError=t.status===409?`${t.message}. Reload to see it.`:t.message,t.errors?.length&&(this.errors=t.errors),!1}finally{this.saving=!1,this.host.requestUpdate()}}async loadVersions(){try{this.versions=(await y.versions(this.kind,this.name)).versions}catch(e){this.saveError=r(e).message}this.host.requestUpdate()}async restore(e){if(await O(`Restore version ${e}?`,[`A new version equal to version ${e} becomes current, with the note "restore of v${e}". The server loads it for the next request.`],`Restore`))try{let t=await y.restore(this.kind,this.name,e);this.saved=`version ${t.version} (restore of v${e}), reloaded ${t.loaded_at?d(t.loaded_at):`now`}`,await this.load(),await this.loadVersions(),this.onSaved?.()}catch(e){this.saveError=r(e).message,this.host.requestUpdate()}}};function A(t){return t.errors.length?e`<ul class="errors" role="alert" data-errors>${t.errors.slice(0,12).map(t=>e`<li>${t}</li>`)}</ul>`:n}function j(t){return t.length?e`<div class="diff" data-diff>
    ${t.map(t=>e`<div class="diff-row"><span>${m(t.path)}</span><span class="diff-before">${_(t.before,160)}</span><span class="diff-after">${_(t.after,160)}</span></div>`)}
  </div>`:e`<p class="muted small">No changes.</p>`}function M(t,r){return e`<section class="panel" data-versions><header class="panel-head"><h2 class="panel-title">Versions of ${r}</h2></header>
    ${t.versions==null?e`<p class="muted small">Loading…</p>`:e`<div class="table-scroll"><table class="table">
          <thead><tr><th>version</th><th>note</th><th>author</th><th>when</th><th></th></tr></thead>
          <tbody>${t.versions.map(r=>e`<tr data-version=${r.version}>
              <td class="num">v${r.version}${r.current?e` <span class="badge badge-on">live</span>`:n}</td>
              <td class="cell-wrap">${r.note??`—`}</td>
              <td class="muted">${r.author??`—`}</td>
              <td class="num muted">${r.created_at?f(r.created_at):`built in`}</td>
              <td>${r.current?n:e`<button class="btn btn-sm" @click=${()=>t.restore(r.version)}>Restore</button>`}</td>
            </tr>`)}</tbody></table></div>`}
  </section>`}function N(t,r){let i=t.base?.meta;return t.locked?e`<div class="savebar is-idle" data-savebar><span class="badge badge-lock">${o(`warn`)} locked</span><span class="status">${t.locked}. Every control is read-only.</span></div>`:e`
    ${t.showDiff?e`<section class="panel" data-diff-panel><header class="panel-head"><h2 class="panel-title">Changes against live (v${i?.version??0})</h2></header>${j(t.changes)}</section>`:n}
    ${t.showVersions?M(t,r):n}
    <div class="savebar ${t.dirty||t.saving?``:`is-idle`}" data-savebar>
      <span class="status num" title=${i?.note??``}>live v${i?.version??0}${i?.loaded_at?e` · reloaded ${d(i.loaded_at)}`:n}</span>
      <input class="note-input" placeholder="one-line change note (required)" .value=${t.note} @input=${e=>t.note=e.target.value} aria-label="change note" maxlength="200" />
      <button class="btn btn-primary" ?disabled=${!t.dirty||t.saving||t.errors.length>0} @click=${()=>t.save()} data-save>${t.saving?`Saving…`:`Save as new version`}</button>
      <button class="btn btn-ghost" ?disabled=${!t.dirty} @click=${()=>t.reset()}>Reset</button>
      <button class="btn btn-ghost ${t.showDiff?`is-on`:``}" @click=${()=>{t.showDiff=!t.showDiff,t.check()}}>Diff (${t.changes.length})</button>
      <button class="btn btn-ghost ${t.showVersions?`is-on`:``}" @click=${()=>{t.showVersions=!t.showVersions,t.showVersions?t.loadVersions():t.host.requestUpdate()}} data-versions-btn>Versions</button>
      ${t.saved?e`<span class="status" role="status" data-saved>${o(`check`)} ${t.saved}</span>`:n}
      ${t.saveError?e`<span class="err-ink small" role="alert">${t.saveError}</span>`:n}
    </div>`}var P=class extends i{constructor(...e){super(...e),this.row=null,this.ids=[],this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.row){this.labelling=!0;try{await y.label(this.row.id,e),this.row={...this.row,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=r(e).message}finally{this.labelling=!1}}}render(){let t=this.row;if(!t)return n;let r=t.stages.find(e=>!e.error&&!e.below);return e`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="decision" data-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title">${t.kind===`side`?S(null,this.ids,`side request`):S(t.class_id,this.ids)} ${t.mode===`shadow`?e`<span class="badge badge-shadow">shadow</span>`:e`<span class="badge badge-on">on</span>`}</div>
            <div class="muted small num">${f(new Date(t.ts_ms).toISOString())} · ${t.profile} v${t.profile_version} · router v${t.router_version}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${o(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">
            <h3>Decision</h3>
            <dl class="kv">
              <div><dt>class</dt><dd>${t.class_id??`—`} ${C(t.confidence)}</dd></div>
              <div><dt>decided by</dt><dd>${T(t.source)}${t.kept?e` <span class="badge badge-warn">${t.kept.replace(/_/g,` `)}</span>`:n}</dd></div>
              <div><dt>classifier time</dt><dd class="num">${u(t.classifier_ms)}</dd></div>
              <div><dt>effort</dt><dd>${t.effort??`unchanged`}${t.budget==null?n:e` <span class="muted num">budget ${t.budget}</span>`} <span class="muted">(${t.budget_source??`server`})</span></dd></div>
              <div><dt>thinking</dt><dd>${t.thinking??`on`}</dd></div>
              <div><dt>modes</dt><dd class="num">${Object.entries(t.decisions??{}).map(([e,t])=>`${e} ${t}`).join(` · `)}</dd></div>
            </dl>
          </section>
          ${t.stages.length?e`<section class="drawer-section"><h3>Classifier stages</h3><div class="stage-list">
                ${t.stages.map(t=>e`<div class="stage ${t===r?`is-accepted`:``} ${t.error?`is-error`:``}">
                    <strong>${b(t.stage)}</strong>
                    <div>${t.error?e`<span class="err-ink">${t.error}</span>`:e`${t.label} ${C(t.confidence,t.accept??.6)}${t.below?e` <span class="muted small">below ${t.accept}</span>`:n}`}
                      ${t.stage===`qwen_context`&&t.prompt_tokens!=null?e`<div class="muted small num" data-context-prefill>${t.reason===`long`?`long prompt`:`follow-up`} · prompt ${t.prompt_tokens.toLocaleString(`en-US`)} tokens · prefilled ${(t.prefilled??0).toLocaleString(`en-US`)} · reused ${(t.reused??0).toLocaleString(`en-US`)}</div>`:n}
                      ${t.probs&&Object.keys(t.probs).length>1?e`<div class="probs">${Object.entries(t.probs).sort((e,t)=>t[1]-e[1]).slice(0,6).map(([t,n])=>e`<div class="prob"><span>${t}</span><span class="prob-bar"><span style="width:${Math.round(n*100)}%"></span></span><span>${n.toFixed(2)}</span></div>`)}</div>`:n}
                    </div>
                    <span class="num muted">${u(t.ms)}${t.cached?` (cached)`:``}</span>
                  </div>`)}</div></section>`:n}
          ${(t.injected??[]).map(t=>e`<section class="drawer-section" data-injected>
              <h3>Injected at token ${t.pos.toLocaleString(`en-US`)} · ${t.tokens.toLocaleString(`en-US`)} tokens ${s(t.text,`Copy the injected text`)}</h3>
              <p class="muted small">Because ${t.reason}. Parts: ${t.parts.map(e=>`${e.kind} ${e.name}`).join(`, `)}. The client never saw this text.</p>
              <pre class="note-text">${t.text}</pre>
            </section>`)}
          ${t.replayed?.count?e`<p class="muted small">Replayed ${t.replayed.count} earlier note(s), ${t.replayed.tokens.toLocaleString(`en-US`)} tokens, at the same positions, so the resident prefix kept matching.</p>`:n}
          ${t.events?.length?e`<section class="drawer-section" data-monitor-events><h3>The session monitor in this request</h3>
                ${t.events.map(e=>L(e))}</section>`:n}
          ${t.shadow?e`<section class="drawer-section"><h3>Shadow (not applied)</h3><pre class="note-text">${JSON.stringify(t.shadow,null,2)}</pre></section>`:n}
          ${t.digest?e`<section class="drawer-section"><h3>What the classifier read</h3><pre class="note-text">${t.digest}</pre></section>`:n}
          <section class="drawer-section">
            <h3>Outcome</h3>
            ${t.outcome?e`<dl class="kv">
                  <div><dt>finish</dt><dd>${t.outcome.finish_reason??`running`}</dd></div>
                  <div><dt>tool calls</dt><dd class="num">${t.outcome.tool_calls??0} ${(t.outcome.tool_names??[]).join(`, `)}</dd></div>
                  <div><dt>reasoning tokens</dt><dd class="num">${t.outcome.reasoning_tokens??`—`}</dd></div>
                  <div><dt>time</dt><dd class="num">${u(t.outcome.total_ms)}</dd></div>
                  ${t.next_checker?.length?e`<div><dt>next checker</dt><dd>${t.next_checker.map(t=>e`<span class="badge ${t.verdict===`PASS`?`badge-ok`:`badge-warn`}">${t.verdict}</span> <span class="muted small">${t.checker} ${t.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:n}
                </dl>`:e`<p class="muted small">The turn is still running.</p>`}
          </section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${t.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${o(`check`)} this was right</button>
              <button class="btn ${t.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${o(`x`)} this was wrong</button>
              ${t.label?e`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:n}
              ${this.msg?e`<span class="muted small" role="status">${this.msg}</span>`:n}
            </div>
            <p class="muted small">Labels feed the precision on the Compare screen and the promotion rule.</p>
          </section>
          <p class="muted small num">decision ${t.id} · request ${t.request_id??`—`} · session ${_(t.session_key,40)}</p>
        </div>
      </aside>`}};a([c({attribute:!1})],P.prototype,`row`,void 0),a([c({attribute:!1})],P.prototype,`ids`,void 0),a([c({attribute:!1})],P.prototype,`close`,void 0),a([t()],P.prototype,`labelling`,void 0),a([t()],P.prototype,`msg`,void 0),P=a([l(`qse-cond-decision`)],P);function F(t,r,i,a){return e`<button class="feed-row" data-decision=${t.id} @click=${()=>i(t)}>
    <span class="feed-time">${d(new Date(t.ts_ms).toISOString())}</span>
    <span class="feed-cls">${t.kind===`side`?S(null,r,`side request`):S(t.class_id,r)}</span>
    <span class="feed-conf">${C(t.confidence)} <span class="muted small">${T(t.source)}</span></span>
    <span class="feed-what">${a}</span>
    <span class="feed-act">${t.acted?e`<span class="badge badge-on">acted</span>`:n}${t.mode===`shadow`||t.shadow?e` <span class="badge badge-shadow">shadow</span>`:n}${!t.acted&&!(t.mode===`shadow`||t.shadow)?e`<span class="badge">pass</span>`:n}</span>
  </button>`}function I(t){return v(t)?e`<span class="badge badge-on">acted</span>`:t.mode===`shadow`&&t.decision!==`PASS`?e`<span class="badge badge-shadow">shadow</span>`:e`<span class="badge">pass</span>`}function L(t){let r=(t.hidden_tokens??0)>0;return e`<div class="event-block" data-event=${t.id}>
    <div class="event-head">${I(t)} <strong>${x(t)}</strong></div>
    <dl class="kv">
      <div><dt>safe point</dt><dd>${t.sp??`—`} · rule ${t.rule??`—`} · policy v${t.policy_version??`—`}</dd></div>
      ${t.tokens_at_cut==null?n:e`<div><dt>reasoning tokens</dt><dd class="num">${t.tokens_at_cut.toLocaleString(`en-US`)}</dd></div>`}
      ${t.outcome?e`<div><dt>outcome</dt><dd>${t.outcome.kind??`—`}${t.outcome.finish?e` <span class="muted">(${t.outcome.finish})</span>`:n}</dd></div>`:n}
      ${t.next_checker?.length?e`<div><dt>next checker</dt><dd>${t.next_checker.map(t=>e`<span class="badge ${t.verdict===`PASS`?`badge-ok`:`badge-warn`}">${t.verdict}</span> <span class="muted small">${t.checker} ${t.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:n}
    </dl>
    ${t.text?e`<h4 class="small">${r?e`Hidden text at token ${(t.position??0).toLocaleString(`en-US`)} · ${(t.hidden_tokens??0).toLocaleString(`en-US`)} tokens`:`Visible text`} ${s(t.text,`Copy the text`)}</h4><pre class="note-text">${t.text}</pre>`:n}
    ${t.detail&&Object.keys(t.detail).length?e`<details><summary class="muted small">detail</summary><pre class="note-text">${JSON.stringify(t.detail,null,2)}</pre></details>`:n}
  </div>`}function R(t,r){return e`<button class="feed-row" data-event-row=${t.id} @click=${()=>r(t)}>
    <span class="feed-time">${d(new Date(t.ts_ms).toISOString())}</span>
    <span class="feed-cls"><span class="badge mono">${t.task??`—`}</span></span>
    <span class="feed-conf muted small">${t.sp??``}</span>
    <span class="feed-what">${x(t)}${(t.hidden_tokens??0)>0?e` <span class="muted small num">${t.hidden_tokens} hidden tokens</span>`:n}</span>
    <span class="feed-act">${I(t)}</span>
  </button>`}var z=class extends i{constructor(...e){super(...e),this.ev=null,this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.ev){this.labelling=!0;try{await y.label(this.ev.id,e),this.ev={...this.ev,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=r(e).message}finally{this.labelling=!1}}}render(){let t=this.ev;return t?e`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="monitor decision" data-event-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title"><span class="badge mono">${t.task}</span> ${I(t)}</div>
            <div class="muted small num">${f(new Date(t.ts_ms).toISOString())} · ${t.profile??`no profile`} · ${t.client_kind??`—`}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${o(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">${L(t)}</section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${t.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${o(`check`)} this was right</button>
              <button class="btn ${t.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${o(`x`)} this was wrong</button>
              ${t.label?e`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:n}
              ${this.msg?e`<span class="muted small" role="status">${this.msg}</span>`:n}
            </div>
            <p class="muted small">Labels on shadow decisions feed the task's precision and its promotion to act (Compare).</p>
          </section>
          <p class="muted small num">event ${t.id} · request ${t.request_id??`—`} · session ${t.session_key?e`<a href=${`#/conductor?screen=sessions&session=${encodeURIComponent(t.session_key)}`}>${_(t.session_key,40)}</a>`:`—`}</p>
        </div>
      </aside>`:n}};a([c({attribute:!1})],z.prototype,`ev`,void 0),a([c({attribute:!1})],z.prototype,`close`,void 0),a([t()],z.prototype,`labelling`,void 0),a([t()],z.prototype,`msg`,void 0),z=a([l(`qse-cond-event`)],z);export{k as DocEditor,P as QseCondDecision,E as QseCondDialog,z as QseCondEvent,S as clsChip,C as confView,O as confirmDialog,j as diffList,A as errorsList,L as eventBlock,R as eventRow,F as feedRow,w as modeSeg,N as saveBar,T as sourceWords,M as versionsPanel};