import{$t as e,B as t,G as n,Jt as r,Kt as i,Lt as a,Nt as o,S as s,St as c,X as l,Xt as u,Yt as d,Zt as f,_t as p,ct as m,g as h,gt as g,kt as _,qt as v,w as y,yt as b,zt as x}from"./app-DgLGW9kB.js";function S(t,n,r){let i=s(t,n);return e`<span class="cls" style="--cls: var(--series-${i})" data-class=${t??`none`}>${r??t??`—`}</span>`}function C(t,n=.6){return t==null?e`<span class="conf muted">—</span>`:e`<span class="conf" title=${`confidence ${t.toFixed(4)}`}><span class="conf-bar ${t<n?`is-low`:``}"><span style="width:${Math.round(t*100)}%"></span></span>${t.toFixed(2)}</span>`}function w(t,n,r,i={}){return e`<div class="seg ${i.small?`seg-sm`:``} ${i.cls??``}" role="radiogroup" aria-label=${i.label??`mode`}>
    ${t.map(t=>e`<button type="button" class="seg-btn mode-${t} ${t===n?`is-on`:``}" role="radio" aria-checked=${t===n?`true`:`false`} ?disabled=${i.disabled} @click=${()=>t!==n&&r(t)}>${t}</button>`)}
  </div>`}function T(e){return e?h[e]??e:`—`}var E=class extends v{constructor(...e){super(...e),this.heading=``,this.lines=[],this.confirmLabel=`Confirm`,this.done=()=>void 0,this.onKey=e=>{e.key===`Escape`&&(e.preventDefault(),this.done(!1))},this.onRoute=()=>this.done(!1)}connectedCallback(){super.connectedCallback(),document.addEventListener(`keydown`,this.onKey),window.addEventListener(`hashchange`,this.onRoute),window.addEventListener(`popstate`,this.onRoute)}disconnectedCallback(){document.removeEventListener(`keydown`,this.onKey),window.removeEventListener(`hashchange`,this.onRoute),window.removeEventListener(`popstate`,this.onRoute),super.disconnectedCallback()}submit(e){e.preventDefault(),this.done(!0)}render(){return e`<div class="modal-back" @click=${()=>this.done(!1)}></div>
      <form class="modal" role="dialog" aria-modal="true" aria-label=${this.heading} @submit=${e=>this.submit(e)}>
        <h2>${this.heading}</h2>
        ${this.lines.map(t=>e`<p class="muted small">${t}</p>`)}
        <div class="modal-actions">
          <button type="button" class="btn" @click=${()=>this.done(!1)}>Cancel</button>
          <button type="submit" class="btn btn-primary" data-confirm>${this.confirmLabel}</button>
        </div>
      </form>`}};i([d({attribute:!1})],E.prototype,`heading`,void 0),i([d({attribute:!1})],E.prototype,`lines`,void 0),i([d({attribute:!1})],E.prototype,`confirmLabel`,void 0),i([d({attribute:!1})],E.prototype,`done`,void 0),E=i([u(`qse-cond-dialog`)],E);function D(e){return new Promise(t=>{let n=document.createElement(`qse-cond-dialog`),r=document.activeElement instanceof HTMLElement?document.activeElement:null;e(n);let i=!1;n.done=e=>{i||(i=!0,n.remove(),r&&r.isConnected&&r.focus(),t({ok:e}))},document.body.appendChild(n)})}function O(e,t,n){return D(r=>{r.heading=e,r.lines=t,r.confirmLabel=n}).then(e=>e.ok)}var k=class{constructor(e,t,n=null){this.host=e,this.kind=t,this.name=n,this.base=null,this.draft=null,this.loading=!0,this.loadError=null,this.errors=[],this.changes=[],this.note=``,this.saving=!1,this.saved=null,this.saveError=null,this.versions=null,this.showVersions=!1,this.showDiff=!1,this.onSaved=null,this.checkTimer=null}get dirty(){return this.changes.length>0}get locked(){return this.base?.locked??null}async load(){this.loading=!0,this.host.requestUpdate();try{this.base=await m.doc(this.kind,this.name),this.draft=y(this.base.doc),this.errors=[],this.changes=[],this.loadError=null}catch(e){let t=_(e);this.loadError=`${t.endpoint}: ${t.message}`}finally{this.loading=!1,this.host.requestUpdate()}}set(e){this.draft=e,this.saved=null,this.host.requestUpdate(),this.checkTimer&&clearTimeout(this.checkTimer),this.checkTimer=setTimeout(()=>void this.check(),250)}async check(){if(this.draft){try{let e=await m.check(this.kind,this.name,this.draft);this.errors=e.errors,this.changes=e.changes}catch(e){this.errors=[_(e).message]}this.host.requestUpdate()}}reset(){this.base&&(this.draft=y(this.base.doc),this.errors=[],this.changes=[],this.host.requestUpdate())}async save(){if(!this.draft||!this.base)return!1;if(!this.note.trim())return this.saveError=`Write a one-line change note first.`,this.host.requestUpdate(),!1;this.saving=!0,this.saveError=null,this.host.requestUpdate();try{let e=await m.put(this.kind,this.name,this.draft,this.note.trim(),this.base.version);return this.saved=`version ${e.version}, reloaded ${e.loaded_at?x(e.loaded_at):`now`}`,this.note=``,await this.load(),this.showVersions&&await this.loadVersions(),this.onSaved?.(),!0}catch(e){let t=e;return this.saveError=t.status===409?`${t.message}. Reload to see it.`:t.message,t.errors?.length&&(this.errors=t.errors),!1}finally{this.saving=!1,this.host.requestUpdate()}}async loadVersions(){try{this.versions=(await m.versions(this.kind,this.name)).versions}catch(e){this.saveError=_(e).message}this.host.requestUpdate()}async restore(e){if(await O(`Restore version ${e}?`,[`A new version equal to version ${e} becomes current, with the note "restore of v${e}". The server loads it for the next request.`],`Restore`))try{let t=await m.restore(this.kind,this.name,e);this.saved=`version ${t.version} (restore of v${e}), reloaded ${t.loaded_at?x(t.loaded_at):`now`}`,await this.load(),await this.loadVersions(),this.onSaved?.()}catch(e){this.saveError=_(e).message,this.host.requestUpdate()}}};function A(t){return t.errors.length?e`<ul class="errors" role="alert" data-errors>${t.errors.slice(0,12).map(t=>e`<li>${t}</li>`)}</ul>`:f}function j(r){return r.length?e`<div class="diff" data-diff>
    ${r.map(r=>e`<div class="diff-row"><span>${t(r.path)}</span><span class="diff-before">${n(r.before,160)}</span><span class="diff-after">${n(r.after,160)}</span></div>`)}
  </div>`:e`<p class="muted small">No changes.</p>`}function M(t,n){return e`<section class="panel" data-versions><header class="panel-head"><h2 class="panel-title">Versions of ${n}</h2></header>
    ${t.versions==null?e`<p class="muted small">Loading…</p>`:e`<div class="table-scroll"><table class="table">
          <thead><tr><th>version</th><th>note</th><th>author</th><th>when</th><th></th></tr></thead>
          <tbody>${t.versions.map(n=>e`<tr data-version=${n.version}>
              <td class="num">v${n.version}${n.current?e` <span class="badge badge-on">live</span>`:f}</td>
              <td class="cell-wrap">${n.note??`—`}</td>
              <td class="muted">${n.author??`—`}</td>
              <td class="num muted">${n.created_at?o(n.created_at):`built in`}</td>
              <td>${n.current?f:e`<button class="btn btn-sm" @click=${()=>t.restore(n.version)}>Restore</button>`}</td>
            </tr>`)}</tbody></table></div>`}
  </section>`}function N(t,n){let r=t.base?.meta;return t.locked?e`<div class="savebar is-idle" data-savebar><span class="badge badge-lock">${c(`warn`)} locked</span><span class="status">${t.locked}. Every control is read-only.</span></div>`:e`
    ${t.showDiff?e`<section class="panel" data-diff-panel><header class="panel-head"><h2 class="panel-title">Changes against live (v${r?.version??0})</h2></header>${j(t.changes)}</section>`:f}
    ${t.showVersions?M(t,n):f}
    <div class="savebar ${t.dirty||t.saving?``:`is-idle`}" data-savebar>
      <span class="status num" title=${r?.note??``}>live v${r?.version??0}${r?.loaded_at?e` · reloaded ${x(r.loaded_at)}`:f}</span>
      <input class="note-input" placeholder="one-line change note (required)" .value=${t.note} @input=${e=>t.note=e.target.value} aria-label="change note" maxlength="200" />
      <button class="btn btn-primary" ?disabled=${!t.dirty||t.saving||t.errors.length>0} @click=${()=>t.save()} data-save>${t.saving?`Saving…`:`Save as new version`}</button>
      <button class="btn btn-ghost" ?disabled=${!t.dirty} @click=${()=>t.reset()}>Reset</button>
      <button class="btn btn-ghost ${t.showDiff?`is-on`:``}" @click=${()=>{t.showDiff=!t.showDiff,t.check()}}>Diff (${t.changes.length})</button>
      <button class="btn btn-ghost ${t.showVersions?`is-on`:``}" @click=${()=>{t.showVersions=!t.showVersions,t.showVersions?t.loadVersions():t.host.requestUpdate()}} data-versions-btn>Versions</button>
      ${t.saved?e`<span class="status" role="status" data-saved>${c(`check`)} ${t.saved}</span>`:f}
      ${t.saveError?e`<span class="err-ink small" role="alert">${t.saveError}</span>`:f}
    </div>`}var P=class extends v{constructor(...e){super(...e),this.row=null,this.ids=[],this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.row){this.labelling=!0;try{await m.label(this.row.id,e),this.row={...this.row,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=_(e).message}finally{this.labelling=!1}}}render(){let t=this.row;if(!t)return f;let r=t.stages.find(e=>!e.error&&!e.below);return e`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="decision" data-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title">${t.kind===`side`?S(null,this.ids,`side request`):S(t.class_id,this.ids)} ${t.mode===`shadow`?e`<span class="badge badge-shadow">shadow</span>`:e`<span class="badge badge-on">on</span>`}</div>
            <div class="muted small num">${o(new Date(t.ts_ms).toISOString())} · ${t.profile} v${t.profile_version} · router v${t.router_version}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${c(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">
            <h3>Decision</h3>
            <dl class="kv">
              <div><dt>class</dt><dd>${t.class_id??`—`} ${C(t.confidence)}</dd></div>
              <div><dt>decided by</dt><dd>${T(t.source)}${t.kept?e` <span class="badge badge-warn">${t.kept.replace(/_/g,` `)}</span>`:f}</dd></div>
              <div><dt>classifier time</dt><dd class="num">${a(t.classifier_ms)}</dd></div>
              <div><dt>effort</dt><dd>${t.effort??`unchanged`}${t.budget==null?f:e` <span class="muted num">budget ${t.budget}</span>`} <span class="muted">(${t.budget_source??`server`})</span></dd></div>
              <div><dt>thinking</dt><dd>${t.thinking??`on`}</dd></div>
              <div><dt>modes</dt><dd class="num">${Object.entries(t.decisions??{}).map(([e,t])=>`${e} ${t}`).join(` · `)}</dd></div>
            </dl>
          </section>
          ${t.stages.length?e`<section class="drawer-section"><h3>Classifier stages</h3><div class="stage-list">
                ${t.stages.map(t=>e`<div class="stage ${t===r?`is-accepted`:``} ${t.error?`is-error`:``}">
                    <strong>${l(t.stage)}</strong>
                    <div>${t.error?e`<span class="err-ink">${t.error}</span>`:e`${t.label} ${C(t.confidence,t.accept??.6)}${t.below?e` <span class="muted small">below ${t.accept}</span>`:f}`}
                      ${t.stage===`qwen_context`&&t.prompt_tokens!=null?e`<div class="muted small num" data-context-prefill>${t.reason===`long`?`long prompt`:`follow-up`} · prompt ${t.prompt_tokens.toLocaleString(`en-US`)} tokens · prefilled ${(t.prefilled??0).toLocaleString(`en-US`)} · reused ${(t.reused??0).toLocaleString(`en-US`)}</div>`:f}
                      ${t.probs&&Object.keys(t.probs).length>1?e`<div class="probs">${Object.entries(t.probs).sort((e,t)=>t[1]-e[1]).slice(0,6).map(([t,n])=>e`<div class="prob"><span>${t}</span><span class="prob-bar"><span style="width:${Math.round(n*100)}%"></span></span><span>${n.toFixed(2)}</span></div>`)}</div>`:f}
                    </div>
                    <span class="num muted">${a(t.ms)}${t.cached?` (cached)`:``}</span>
                  </div>`)}</div></section>`:f}
          ${(t.injected??[]).map(t=>e`<section class="drawer-section" data-injected>
              <h3>Injected at token ${t.pos.toLocaleString(`en-US`)} · ${t.tokens.toLocaleString(`en-US`)} tokens ${b(t.text,`Copy the injected text`)}</h3>
              <p class="muted small">Because ${t.reason}. Parts: ${t.parts.map(e=>`${e.kind} ${e.name}`).join(`, `)}. The client never saw this text.</p>
              <pre class="note-text">${t.text}</pre>
            </section>`)}
          ${t.replayed?.count?e`<p class="muted small">Replayed ${t.replayed.count} earlier note(s), ${t.replayed.tokens.toLocaleString(`en-US`)} tokens, at the same positions, so the resident prefix kept matching.</p>`:f}
          ${t.events?.length?e`<section class="drawer-section" data-monitor-events><h3>The session monitor in this request</h3>
                ${t.events.map(e=>L(e))}</section>`:f}
          ${t.shadow?e`<section class="drawer-section"><h3>Shadow (not applied)</h3><pre class="note-text">${JSON.stringify(t.shadow,null,2)}</pre></section>`:f}
          ${t.digest?e`<section class="drawer-section"><h3>What the classifier read</h3><pre class="note-text">${t.digest}</pre></section>`:f}
          <section class="drawer-section">
            <h3>Outcome</h3>
            ${t.outcome?e`<dl class="kv">
                  <div><dt>finish</dt><dd>${t.outcome.finish_reason??`running`}</dd></div>
                  <div><dt>tool calls</dt><dd class="num">${t.outcome.tool_calls??0} ${(t.outcome.tool_names??[]).join(`, `)}</dd></div>
                  <div><dt>reasoning tokens</dt><dd class="num">${t.outcome.reasoning_tokens??`—`}</dd></div>
                  <div><dt>time</dt><dd class="num">${a(t.outcome.total_ms)}</dd></div>
                  ${t.next_checker?.length?e`<div><dt>next checker</dt><dd>${t.next_checker.map(t=>e`<span class="badge ${t.verdict===`PASS`?`badge-ok`:`badge-warn`}">${t.verdict}</span> <span class="muted small">${t.checker} ${t.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:f}
                </dl>`:e`<p class="muted small">The turn is still running.</p>`}
          </section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${t.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${c(`check`)} this was right</button>
              <button class="btn ${t.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${c(`x`)} this was wrong</button>
              ${t.label?e`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:f}
              ${this.msg?e`<span class="muted small" role="status">${this.msg}</span>`:f}
            </div>
            <p class="muted small">Labels feed the precision on the Compare screen and the promotion rule.</p>
          </section>
          <p class="muted small num">decision ${t.id} · request ${t.request_id??`—`} · session ${n(t.session_key,40)}</p>
        </div>
      </aside>`}};i([d({attribute:!1})],P.prototype,`row`,void 0),i([d({attribute:!1})],P.prototype,`ids`,void 0),i([d({attribute:!1})],P.prototype,`close`,void 0),i([r()],P.prototype,`labelling`,void 0),i([r()],P.prototype,`msg`,void 0),P=i([u(`qse-cond-decision`)],P);function F(t,n,r,i){return e`<button class="feed-row" data-decision=${t.id} @click=${()=>r(t)}>
    <span class="feed-time">${x(new Date(t.ts_ms).toISOString())}</span>
    <span class="feed-cls">${t.kind===`side`?S(null,n,`side request`):S(t.class_id,n)}</span>
    <span class="feed-conf">${C(t.confidence)} <span class="muted small">${T(t.source)}</span></span>
    <span class="feed-what">${i}</span>
    <span class="feed-act">${t.acted?e`<span class="badge badge-on">acted</span>`:f}${t.mode===`shadow`||t.shadow?e` <span class="badge badge-shadow">shadow</span>`:f}${!t.acted&&!(t.mode===`shadow`||t.shadow)?e`<span class="badge">pass</span>`:f}</span>
  </button>`}function I(t){return p(t)?e`<span class="badge badge-on">acted</span>`:t.mode===`shadow`&&t.decision!==`PASS`?e`<span class="badge badge-shadow">shadow</span>`:e`<span class="badge">pass</span>`}function L(t){let n=(t.hidden_tokens??0)>0;return e`<div class="event-block" data-event=${t.id}>
    <div class="event-head">${I(t)} <strong>${g(t)}</strong></div>
    <dl class="kv">
      <div><dt>safe point</dt><dd>${t.sp??`—`} · rule ${t.rule??`—`} · policy v${t.policy_version??`—`}</dd></div>
      ${t.tokens_at_cut==null?f:e`<div><dt>reasoning tokens</dt><dd class="num">${t.tokens_at_cut.toLocaleString(`en-US`)}</dd></div>`}
      ${t.outcome?e`<div><dt>outcome</dt><dd>${t.outcome.kind??`—`}${t.outcome.finish?e` <span class="muted">(${t.outcome.finish})</span>`:f}</dd></div>`:f}
      ${t.next_checker?.length?e`<div><dt>next checker</dt><dd>${t.next_checker.map(t=>e`<span class="badge ${t.verdict===`PASS`?`badge-ok`:`badge-warn`}">${t.verdict}</span> <span class="muted small">${t.checker} ${t.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:f}
    </dl>
    ${t.text?e`<h4 class="small">${n?e`Hidden text at token ${(t.position??0).toLocaleString(`en-US`)} · ${(t.hidden_tokens??0).toLocaleString(`en-US`)} tokens`:`Visible text`} ${b(t.text,`Copy the text`)}</h4><pre class="note-text">${t.text}</pre>`:f}
    ${t.detail&&Object.keys(t.detail).length?e`<details><summary class="muted small">detail</summary><pre class="note-text">${JSON.stringify(t.detail,null,2)}</pre></details>`:f}
  </div>`}function R(t,n){return e`<button class="feed-row" data-event-row=${t.id} @click=${()=>n(t)}>
    <span class="feed-time">${x(new Date(t.ts_ms).toISOString())}</span>
    <span class="feed-cls"><span class="badge mono">${t.task??`—`}</span></span>
    <span class="feed-conf muted small">${t.sp??``}</span>
    <span class="feed-what">${g(t)}${(t.hidden_tokens??0)>0?e` <span class="muted small num">${t.hidden_tokens} hidden tokens</span>`:f}</span>
    <span class="feed-act">${I(t)}</span>
  </button>`}var z=class extends v{constructor(...e){super(...e),this.ev=null,this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.ev){this.labelling=!0;try{await m.label(this.ev.id,e),this.ev={...this.ev,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=_(e).message}finally{this.labelling=!1}}}render(){let t=this.ev;return t?e`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="monitor decision" data-event-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title"><span class="badge mono">${t.task}</span> ${I(t)}</div>
            <div class="muted small num">${o(new Date(t.ts_ms).toISOString())} · ${t.profile??`no profile`} · ${t.client_kind??`—`}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${c(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">${L(t)}</section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${t.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${c(`check`)} this was right</button>
              <button class="btn ${t.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${c(`x`)} this was wrong</button>
              ${t.label?e`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:f}
              ${this.msg?e`<span class="muted small" role="status">${this.msg}</span>`:f}
            </div>
            <p class="muted small">Labels on shadow decisions feed the task's precision and its promotion to act (Compare).</p>
          </section>
          <p class="muted small num">event ${t.id} · request ${t.request_id??`—`} · session ${t.session_key?e`<a href=${`#/conductor?screen=sessions&session=${encodeURIComponent(t.session_key)}`}>${n(t.session_key,40)}</a>`:`—`}</p>
        </div>
      </aside>`:f}};i([d({attribute:!1})],z.prototype,`ev`,void 0),i([d({attribute:!1})],z.prototype,`close`,void 0),i([r()],z.prototype,`labelling`,void 0),i([r()],z.prototype,`msg`,void 0),z=i([u(`qse-cond-event`)],z);export{k as DocEditor,P as QseCondDecision,E as QseCondDialog,z as QseCondEvent,S as clsChip,C as confView,O as confirmDialog,j as diffList,A as errorsList,L as eventBlock,R as eventRow,F as feedRow,w as modeSeg,N as saveBar,T as sourceWords,M as versionsPanel};