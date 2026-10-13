import{c as e,d as t,f as n,g as r,i,l as a,p as o,t as s,u as c}from"./chunk-5VjiyoiJ.js";import{d as l,m as u,o as d}from"./chunk-BXmDkkzu.js";import{i as f}from"./chunk-CI4nmvus.js";import{$ as p,A as m,J as h,M as g,St as _,gt as v,it as y,w as b,xt as x}from"./app-DsqMc03P.js";function S(e,t,n){let i=m(e,t);return r`<span class="cls" style="--cls: var(--series-${i})" data-class=${e??`none`}>${n??e??`—`}</span>`}function C(e,t=.6){return e==null?r`<span class="conf muted">—</span>`:r`<span class="conf" title=${`confidence ${e.toFixed(4)}`}><span class="conf-bar ${e<t?`is-low`:``}"><span style="width:${Math.round(e*100)}%"></span></span>${e.toFixed(2)}</span>`}function w(e,t,n,i={}){return r`<div class="seg ${i.small?`seg-sm`:``} ${i.cls??``}" role="radiogroup" aria-label=${i.label??`mode`}>
    ${e.map(e=>r`<button type="button" class="seg-btn mode-${e} ${e===t?`is-on`:``}" role="radio" aria-checked=${e===t?`true`:`false`} ?disabled=${i.disabled} @click=${()=>e!==t&&n(e)}>${e}</button>`)}
  </div>`}function T(e){return e?b[e]??e:`—`}var E=class extends a{constructor(...e){super(...e),this.heading=``,this.lines=[],this.confirmLabel=`Confirm`,this.done=()=>void 0,this.onKey=e=>{e.key===`Escape`&&(e.preventDefault(),this.done(!1))},this.onRoute=()=>this.done(!1)}connectedCallback(){super.connectedCallback(),document.addEventListener(`keydown`,this.onKey),window.addEventListener(`hashchange`,this.onRoute),window.addEventListener(`popstate`,this.onRoute)}disconnectedCallback(){document.removeEventListener(`keydown`,this.onKey),window.removeEventListener(`hashchange`,this.onRoute),window.removeEventListener(`popstate`,this.onRoute),super.disconnectedCallback()}submit(e){e.preventDefault(),this.done(!0)}render(){return r`<div class="modal-back" @click=${()=>this.done(!1)}></div>
      <form class="modal" role="dialog" aria-modal="true" aria-label=${this.heading} @submit=${e=>this.submit(e)}>
        <h2>${this.heading}</h2>
        ${this.lines.map(e=>r`<p class="muted small">${e}</p>`)}
        <div class="modal-actions">
          <button type="button" class="btn" @click=${()=>this.done(!1)}>Cancel</button>
          <button type="submit" class="btn btn-primary" data-confirm>${this.confirmLabel}</button>
        </div>
      </form>`}};e([t({attribute:!1})],E.prototype,`heading`,void 0),e([t({attribute:!1})],E.prototype,`lines`,void 0),e([t({attribute:!1})],E.prototype,`confirmLabel`,void 0),e([t({attribute:!1})],E.prototype,`done`,void 0),E=e([n(`qse-cond-dialog`)],E);function D(e){return new Promise(t=>{let n=document.createElement(`qse-cond-dialog`),r=document.activeElement instanceof HTMLElement?document.activeElement:null;e(n);let i=!1;n.done=e=>{i||(i=!0,n.remove(),r&&r.isConnected&&r.focus(),t({ok:e}))},document.body.appendChild(n)})}function O(e,t,n){return D(r=>{r.heading=e,r.lines=t,r.confirmLabel=n}).then(e=>e.ok)}var k=class{constructor(e,t,n=null){this.host=e,this.kind=t,this.name=n,this.base=null,this.draft=null,this.loading=!0,this.loadError=null,this.errors=[],this.changes=[],this.note=``,this.saving=!1,this.saved=null,this.saveError=null,this.versions=null,this.showVersions=!1,this.showDiff=!1,this.onSaved=null,this.checkTimer=null}get dirty(){return this.changes.length>0}get locked(){return this.base?.locked??null}async load(){this.loading=!0,this.host.requestUpdate();try{this.base=await v.doc(this.kind,this.name),this.draft=g(this.base.doc),this.errors=[],this.changes=[],this.loadError=null}catch(e){let t=f(e);this.loadError=`${t.endpoint}: ${t.message}`}finally{this.loading=!1,this.host.requestUpdate()}}set(e){this.draft=e,this.saved=null,this.host.requestUpdate(),this.checkTimer&&clearTimeout(this.checkTimer),this.checkTimer=setTimeout(()=>void this.check(),250)}async check(){if(this.draft){try{let e=await v.check(this.kind,this.name,this.draft);this.errors=e.errors,this.changes=e.changes}catch(e){this.errors=[f(e).message]}this.host.requestUpdate()}}reset(){this.base&&(this.draft=g(this.base.doc),this.errors=[],this.changes=[],this.host.requestUpdate())}async save(){if(!this.draft||!this.base)return!1;if(!this.note.trim())return this.saveError=`Write a one-line change note first.`,this.host.requestUpdate(),!1;this.saving=!0,this.saveError=null,this.host.requestUpdate();try{let e=await v.put(this.kind,this.name,this.draft,this.note.trim(),this.base.version);return this.saved=`version ${e.version}, reloaded ${e.loaded_at?u(e.loaded_at):`now`}`,this.note=``,await this.load(),this.showVersions&&await this.loadVersions(),this.onSaved?.(),!0}catch(e){let t=e;return this.saveError=t.status===409?`${t.message}. Reload to see it.`:t.message,t.errors?.length&&(this.errors=t.errors),!1}finally{this.saving=!1,this.host.requestUpdate()}}async loadVersions(){try{this.versions=(await v.versions(this.kind,this.name)).versions}catch(e){this.saveError=f(e).message}this.host.requestUpdate()}async restore(e){if(await O(`Restore version ${e}?`,[`A new version equal to version ${e} becomes current, with the note "restore of v${e}". The server loads it for the next request.`],`Restore`))try{let t=await v.restore(this.kind,this.name,e);this.saved=`version ${t.version} (restore of v${e}), reloaded ${t.loaded_at?u(t.loaded_at):`now`}`,await this.load(),await this.loadVersions(),this.onSaved?.()}catch(e){this.saveError=f(e).message,this.host.requestUpdate()}}};function A(e){return e.errors.length?r`<ul class="errors" role="alert" data-errors>${e.errors.slice(0,12).map(e=>r`<li>${e}</li>`)}</ul>`:o}function j(e){return e.length?r`<div class="diff" data-diff>
    ${e.map(e=>r`<div class="diff-row"><span>${h(e.path)}</span><span class="diff-before">${p(e.before,160)}</span><span class="diff-after">${p(e.after,160)}</span></div>`)}
  </div>`:r`<p class="muted small">No changes.</p>`}function M(e,t){return r`<section class="panel" data-versions><header class="panel-head"><h2 class="panel-title">Versions of ${t}</h2></header>
    ${e.versions==null?r`<p class="muted small">Loading…</p>`:r`<div class="table-scroll"><table class="table">
          <thead><tr><th>version</th><th>note</th><th>author</th><th>when</th><th></th></tr></thead>
          <tbody>${e.versions.map(t=>r`<tr data-version=${t.version}>
              <td class="num">v${t.version}${t.current?r` <span class="badge badge-on">live</span>`:o}</td>
              <td class="cell-wrap">${t.note??`—`}</td>
              <td class="muted">${t.author??`—`}</td>
              <td class="num muted">${t.created_at?d(t.created_at):`built in`}</td>
              <td>${t.current?o:r`<button class="btn btn-sm" @click=${()=>e.restore(t.version)}>Restore</button>`}</td>
            </tr>`)}</tbody></table></div>`}
  </section>`}function N(e,t){let n=e.base?.meta;return e.locked?r`<div class="savebar is-idle" data-savebar><span class="badge badge-lock">${i(`warn`)} locked</span><span class="status">${e.locked}. Every control is read-only.</span></div>`:r`
    ${e.showDiff?r`<section class="panel" data-diff-panel><header class="panel-head"><h2 class="panel-title">Changes against live (v${n?.version??0})</h2></header>${j(e.changes)}</section>`:o}
    ${e.showVersions?M(e,t):o}
    <div class="savebar ${e.dirty||e.saving?``:`is-idle`}" data-savebar>
      <span class="status num" title=${n?.note??``}>live v${n?.version??0}${n?.loaded_at?r` · reloaded ${u(n.loaded_at)}`:o}</span>
      <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>e.note=t.target.value} aria-label="change note" maxlength="200" />
      <button class="btn btn-primary" ?disabled=${!e.dirty||e.saving||e.errors.length>0} @click=${()=>e.save()} data-save>${e.saving?`Saving…`:`Save as new version`}</button>
      <button class="btn btn-ghost" ?disabled=${!e.dirty} @click=${()=>e.reset()}>Reset</button>
      <button class="btn btn-ghost ${e.showDiff?`is-on`:``}" @click=${()=>{e.showDiff=!e.showDiff,e.check()}}>Diff (${e.changes.length})</button>
      <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():e.host.requestUpdate()}} data-versions-btn>Versions</button>
      ${e.saved?r`<span class="status" role="status" data-saved>${i(`check`)} ${e.saved}</span>`:o}
      ${e.saveError?r`<span class="err-ink small" role="alert">${e.saveError}</span>`:o}
    </div>`}var P=class extends a{constructor(...e){super(...e),this.row=null,this.ids=[],this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.row){this.labelling=!0;try{await v.label(this.row.id,e),this.row={...this.row,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=f(e).message}finally{this.labelling=!1}}}render(){let e=this.row;if(!e)return o;let t=e.stages.find(e=>!e.error&&!e.below);return r`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="decision" data-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title">${e.kind===`side`?S(null,this.ids,`side request`):S(e.class_id,this.ids)} ${e.mode===`shadow`?r`<span class="badge badge-shadow">shadow</span>`:r`<span class="badge badge-on">on</span>`}</div>
            <div class="muted small num">${d(new Date(e.ts_ms).toISOString())} · ${e.profile} v${e.profile_version} · router v${e.router_version}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${i(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">
            <h3>Decision</h3>
            <dl class="kv">
              <div><dt>class</dt><dd>${e.class_id??`—`} ${C(e.confidence)}</dd></div>
              <div><dt>decided by</dt><dd>${T(e.source)}${e.kept?r` <span class="badge badge-warn">${e.kept.replace(/_/g,` `)}</span>`:o}</dd></div>
              <div><dt>classifier time</dt><dd class="num">${l(e.classifier_ms)}</dd></div>
              <div><dt>effort</dt><dd>${e.effort??`unchanged`}${e.budget==null?o:r` <span class="muted num">budget ${e.budget}</span>`} <span class="muted">(${e.budget_source??`server`})</span></dd></div>
              <div><dt>thinking</dt><dd>${e.thinking??`on`}</dd></div>
              <div><dt>modes</dt><dd class="num">${Object.entries(e.decisions??{}).map(([e,t])=>`${e} ${t}`).join(` · `)}</dd></div>
            </dl>
          </section>
          ${e.stages.length?r`<section class="drawer-section"><h3>Classifier stages</h3><div class="stage-list">
                ${e.stages.map(e=>r`<div class="stage ${e===t?`is-accepted`:``} ${e.error?`is-error`:``}">
                    <strong>${y(e.stage)}</strong>
                    <div>${e.error?r`<span class="err-ink">${e.error}</span>`:r`${e.label} ${C(e.confidence,e.accept??.6)}${e.below?r` <span class="muted small">below ${e.accept}</span>`:o}`}
                      ${e.stage===`qwen_context`&&e.prompt_tokens!=null?r`<div class="muted small num" data-context-prefill>${e.reason===`long`?`long prompt`:`follow-up`} · prompt ${e.prompt_tokens.toLocaleString(`en-US`)} tokens · prefilled ${(e.prefilled??0).toLocaleString(`en-US`)} · reused ${(e.reused??0).toLocaleString(`en-US`)}</div>`:o}
                      ${e.probs&&Object.keys(e.probs).length>1?r`<div class="probs">${Object.entries(e.probs).sort((e,t)=>t[1]-e[1]).slice(0,6).map(([e,t])=>r`<div class="prob"><span>${e}</span><span class="prob-bar"><span style="width:${Math.round(t*100)}%"></span></span><span>${t.toFixed(2)}</span></div>`)}</div>`:o}
                    </div>
                    <span class="num muted">${l(e.ms)}${e.cached?` (cached)`:``}</span>
                  </div>`)}</div></section>`:o}
          ${(e.injected??[]).map(e=>r`<section class="drawer-section" data-injected>
              <h3>Injected at token ${e.pos.toLocaleString(`en-US`)} · ${e.tokens.toLocaleString(`en-US`)} tokens ${s(e.text,`Copy the injected text`)}</h3>
              <p class="muted small">Because ${e.reason}. Parts: ${e.parts.map(e=>`${e.kind} ${e.name}`).join(`, `)}. The client never saw this text.</p>
              <pre class="note-text">${e.text}</pre>
            </section>`)}
          ${e.replayed?.count?r`<p class="muted small">Replayed ${e.replayed.count} earlier note(s), ${e.replayed.tokens.toLocaleString(`en-US`)} tokens, at the same positions, so the resident prefix kept matching.</p>`:o}
          ${e.events?.length?r`<section class="drawer-section" data-monitor-events><h3>The session monitor in this request</h3>
                ${e.events.map(e=>L(e))}</section>`:o}
          ${e.shadow?r`<section class="drawer-section"><h3>Shadow (not applied)</h3><pre class="note-text">${JSON.stringify(e.shadow,null,2)}</pre></section>`:o}
          ${e.digest?r`<section class="drawer-section"><h3>What the classifier read</h3><pre class="note-text">${e.digest}</pre></section>`:o}
          <section class="drawer-section">
            <h3>Outcome</h3>
            ${e.outcome?r`<dl class="kv">
                  <div><dt>finish</dt><dd>${e.outcome.finish_reason??`running`}</dd></div>
                  <div><dt>tool calls</dt><dd class="num">${e.outcome.tool_calls??0} ${(e.outcome.tool_names??[]).join(`, `)}</dd></div>
                  <div><dt>reasoning tokens</dt><dd class="num">${e.outcome.reasoning_tokens??`—`}</dd></div>
                  <div><dt>time</dt><dd class="num">${l(e.outcome.total_ms)}</dd></div>
                  ${e.next_checker?.length?r`<div><dt>next checker</dt><dd>${e.next_checker.map(e=>r`<span class="badge ${e.verdict===`PASS`?`badge-ok`:`badge-warn`}">${e.verdict}</span> <span class="muted small">${e.checker} ${e.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:o}
                </dl>`:r`<p class="muted small">The turn is still running.</p>`}
          </section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${e.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${i(`check`)} this was right</button>
              <button class="btn ${e.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${i(`x`)} this was wrong</button>
              ${e.label?r`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:o}
              ${this.msg?r`<span class="muted small" role="status">${this.msg}</span>`:o}
            </div>
            <p class="muted small">Labels feed the precision on the Compare screen and the promotion rule.</p>
          </section>
          <p class="muted small num">decision ${e.id} · request ${e.request_id??`—`} · session ${p(e.session_key,40)}</p>
        </div>
      </aside>`}};e([t({attribute:!1})],P.prototype,`row`,void 0),e([t({attribute:!1})],P.prototype,`ids`,void 0),e([t({attribute:!1})],P.prototype,`close`,void 0),e([c()],P.prototype,`labelling`,void 0),e([c()],P.prototype,`msg`,void 0),P=e([n(`qse-cond-decision`)],P);function F(e,t,n,i){return r`<button class="feed-row" data-decision=${e.id} @click=${()=>n(e)}>
    <span class="feed-time">${u(new Date(e.ts_ms).toISOString())}</span>
    <span class="feed-cls">${e.kind===`side`?S(null,t,`side request`):S(e.class_id,t)}</span>
    <span class="feed-conf">${C(e.confidence)} <span class="muted small">${T(e.source)}</span></span>
    <span class="feed-what">${i}</span>
    <span class="feed-act">${e.acted?r`<span class="badge badge-on">acted</span>`:o}${e.mode===`shadow`||e.shadow?r` <span class="badge badge-shadow">shadow</span>`:o}${!e.acted&&!(e.mode===`shadow`||e.shadow)?r`<span class="badge">pass</span>`:o}</span>
  </button>`}function I(e){return _(e)?r`<span class="badge badge-on">acted</span>`:e.mode===`shadow`&&e.decision!==`PASS`?r`<span class="badge badge-shadow">shadow</span>`:r`<span class="badge">pass</span>`}function L(e){let t=(e.hidden_tokens??0)>0;return r`<div class="event-block" data-event=${e.id}>
    <div class="event-head">${I(e)} <strong>${x(e)}</strong></div>
    <dl class="kv">
      <div><dt>safe point</dt><dd>${e.sp??`—`} · rule ${e.rule??`—`} · policy v${e.policy_version??`—`}</dd></div>
      ${e.tokens_at_cut==null?o:r`<div><dt>reasoning tokens</dt><dd class="num">${e.tokens_at_cut.toLocaleString(`en-US`)}</dd></div>`}
      ${e.outcome?r`<div><dt>outcome</dt><dd>${e.outcome.kind??`—`}${e.outcome.finish?r` <span class="muted">(${e.outcome.finish})</span>`:o}</dd></div>`:o}
      ${e.next_checker?.length?r`<div><dt>next checker</dt><dd>${e.next_checker.map(e=>r`<span class="badge ${e.verdict===`PASS`?`badge-ok`:`badge-warn`}">${e.verdict}</span> <span class="muted small">${e.checker} ${e.reasons.slice(0,2).join(`; `)}</span>`)}</dd></div>`:o}
    </dl>
    ${e.text?r`<h4 class="small">${t?r`Hidden text at token ${(e.position??0).toLocaleString(`en-US`)} · ${(e.hidden_tokens??0).toLocaleString(`en-US`)} tokens`:`Visible text`} ${s(e.text,`Copy the text`)}</h4><pre class="note-text">${e.text}</pre>`:o}
    ${e.detail&&Object.keys(e.detail).length?r`<details><summary class="muted small">detail</summary><pre class="note-text">${JSON.stringify(e.detail,null,2)}</pre></details>`:o}
  </div>`}function R(e,t){return r`<button class="feed-row" data-event-row=${e.id} @click=${()=>t(e)}>
    <span class="feed-time">${u(new Date(e.ts_ms).toISOString())}</span>
    <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
    <span class="feed-conf muted small">${e.sp??``}</span>
    <span class="feed-what">${x(e)}${(e.hidden_tokens??0)>0?r` <span class="muted small num">${e.hidden_tokens} hidden tokens</span>`:o}</span>
    <span class="feed-act">${I(e)}</span>
  </button>`}var z=class extends a{constructor(...e){super(...e),this.ev=null,this.close=()=>void 0,this.labelling=!1,this.msg=null}async label(e){if(this.ev){this.labelling=!0;try{await v.label(this.ev.id,e),this.ev={...this.ev,label:e===`clear`?null:e},this.msg=e===`clear`?`label cleared`:`labelled ${e}`}catch(e){this.msg=f(e).message}finally{this.labelling=!1}}}render(){let e=this.ev;return e?r`<div class="drawer-back" @click=${()=>this.close()}></div>
      <aside class="drawer" role="dialog" aria-label="monitor decision" data-event-drawer>
        <header class="drawer-head">
          <div>
            <div class="panel-title"><span class="badge mono">${e.task}</span> ${I(e)}</div>
            <div class="muted small num">${d(new Date(e.ts_ms).toISOString())} · ${e.profile??`no profile`} · ${e.client_kind??`—`}</div>
          </div>
          <button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.close()}>${i(`x`)}</button>
        </header>
        <div class="drawer-body">
          <section class="drawer-section">${L(e)}</section>
          <section class="drawer-section">
            <h3>Was this right?</h3>
            <div class="toolbar" style="margin:0">
              <button class="btn ${e.label===`right`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`right`)} data-label="right">${i(`check`)} this was right</button>
              <button class="btn ${e.label===`wrong`?`btn-primary`:``}" ?disabled=${this.labelling} @click=${()=>this.label(`wrong`)} data-label="wrong">${i(`x`)} this was wrong</button>
              ${e.label?r`<button class="btn btn-ghost" @click=${()=>this.label(`clear`)}>clear</button>`:o}
              ${this.msg?r`<span class="muted small" role="status">${this.msg}</span>`:o}
            </div>
            <p class="muted small">Labels on shadow decisions feed the task's precision and its promotion to act (Compare).</p>
          </section>
          <p class="muted small num">event ${e.id} · request ${e.request_id??`—`} · session ${e.session_key?r`<a href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${p(e.session_key,40)}</a>`:`—`}</p>
        </div>
      </aside>`:o}};e([t({attribute:!1})],z.prototype,`ev`,void 0),e([t({attribute:!1})],z.prototype,`close`,void 0),e([c()],z.prototype,`labelling`,void 0),e([c()],z.prototype,`msg`,void 0),z=e([n(`qse-cond-event`)],z);export{k as DocEditor,P as QseCondDecision,E as QseCondDialog,z as QseCondEvent,S as clsChip,C as confView,O as confirmDialog,j as diffList,A as errorsList,L as eventBlock,R as eventRow,F as feedRow,w as modeSeg,N as saveBar,T as sourceWords,M as versionsPanel};