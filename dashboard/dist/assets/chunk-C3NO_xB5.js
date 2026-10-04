import{A as e,B as t,C as n,D as r,E as i,F as a,I as o,L as s,M as c,N as l,O as u,P as d,R as f,T as p,_ as m,b as ee,c as h,g,h as _,j as v,l as te,m as y,n as ne,p as re,r as b,t as ie,u as ae,v as x,y as S}from"./app-DAukfkhS.js";import{$ as oe,A as se,B as ce,C as le,D as ue,DocEditor as C,E as de,F as fe,G as pe,H as me,I as w,J as he,K as ge,L as _e,M as ve,N as ye,O as be,P as xe,Q as Se,R as T,S as Ce,T as we,U as Te,V as Ee,W as De,X as Oe,Y as ke,Z as Ae,_ as je,a as Me,b as Ne,c as Pe,clsChip as E,confView as Fe,confirmDialog as D,d as Ie,diffList as Le,errorsList as O,et as k,eventRow as Re,f as ze,feedRow as Be,g as Ve,h as He,i as Ue,j as We,k as Ge,l as Ke,m as qe,modeSeg as A,n as Je,o as Ye,p as Xe,q as Ze,s as Qe,saveBar as j,sourceWords as $e,t as et,u as tt,v as nt,versionsPanel as rt,w as it,withToken as M,x as at,y as ot,z as N}from"./chunk-0L57czuW.js";import{n as st,t as ct}from"./chunk-BiChSaT2.js";var P=class extends d{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=h(()=>this.loadStats(),2e4),this.feedPoll=h(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await k.stats(24),this.error=null}catch(e){let t=n(e);this.error=`${t.endpoint}: ${t.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await k.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await k.event(e)).event}catch(e){this.error=n(e).message}}renderMonitor(e){let n=be(e);return t`<div class="grid-2 grid-2-wide">
      ${g(`Monitor acts an hour, last 24 h`,e.acts?t`<qse-ribbon .series=${n} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:y(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${g(`Last actions of the monitor`,e.last_actions.length?t`<div class="feed" data-last-actions>${e.last_actions.map(e=>t`<button class="feed-row" data-action=${e.id} @click=${()=>this.openAction(e.id)}>
                <span class="feed-time">${c(new Date(e.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
                <span class="feed-conf muted small">${e.outcome?.kind??`running`}</span>
                <span class="feed-what">${te({task:e.task,decision:e.decision,mode:`act`,trigger:e.trigger,cap:e.cap})}</span>
                <span class="feed-act">${e.session_key?t`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:f}</span>
              </button>`)}</div>`:t`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:t`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${_(`external`)}</a>`})}
    </div>`}render(){let n=this.stats;if(this.error&&!n)return t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${this.error}</div></div>`;if(!n)return m(6);let r=n.classifier,i=r.decider,a=r.qwen,o=i?.calls?i:a?.calls?a:i,s=Object.entries(n.roles),c=s.filter(([,e])=>e.up).length,l=n.monitor;return t`
      <div class="stats" data-tiles>
        ${x({label:`decisions, 24 h`,value:u(n.decisions),sub:t`<span class="num">${u(n.acted)}</span> acted · <span class="num">${u(n.shadow)}</span> in shadow`,title:`${n.injections} notes, ${n.injected_tokens} tokens injected`})}
        ${x({label:`monitor, 24 h`,value:l?u(l.acts):`—`,sub:l?t`acts · <span class="num">${u(l.shadow)}</span> in shadow · <span class="num" data-gave-up>${u(l.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!l,warn:(l?.gave_up??0)>0,title:l?`${u(l.hidden_tokens)} hidden tokens`:void 0})}
        ${x({label:`classifier p50 / p95${!i?.calls&&a?.calls?` (Qwen)`:``}`,value:o?.p50_ms==null?`—`:`${Math.round(o.p50_ms)} / ${Math.round(o.p95_ms??0)}`,unit:o?.p50_ms==null?void 0:`ms`,sub:t`decider ${u(i?.calls??0)} calls${a?t` · Qwen ${u(a.calls)} (${e(a.p50_ms)})`:f} · fail-open ${v(n.fail_open_rate,1)}`,warn:(n.fail_open_rate??0)>.1})}
        ${x({label:`roles online`,value:`${c} / ${s.length}`,sub:t`${s.map(([e,n])=>t`<span title=${n.error??``} class=${n.up?``:n.mode===`off`?`muted`:`err-ink`} data-role=${e}>${n.up?`●`:`○`} ${e} </span>`)}`,warn:s.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${g(`Decisions an hour, last 24 h`,n.decisions?t`<qse-ribbon .series=${le(n,this.ids)} .labels=${n.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:y(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${g(`Task classes, 24 h`,t`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${n.classes.map(e=>t`<tr><td>${E(e.class_id,this.ids)}</td><td class="num num-col">${u(e.decisions)}</td><td class="num num-col">${e.labelled?`${e.agree} / ${e.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(n.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(n.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${g(t`Live decisions ${this.paused?t`<span class="badge">paused</span>`:t`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?t`<div class="feed" data-feed>${this.feed.map(e=>Be(e,this.ids,e=>this.open=e,Ne(e)))}</div>`:t`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:t`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${_(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${_(`external`)}</a>`})}
        ${g(`Profiles by client`,t`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(n=>t`<button class="pick" @click=${()=>S({screen:`profiles`,edit:e.name})}><span class="muted">${n}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(e=>t`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${l?this.renderMonitor(l):f}
      ${this.open?t`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:f}
      ${this.openEvent?t`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:f}
    `}};l([o({attribute:!1})],P.prototype,`ids`,void 0),l([o({attribute:!1})],P.prototype,`tiles`,void 0),l([o({attribute:!1})],P.prototype,`active`,void 0),l([a()],P.prototype,`stats`,void 0),l([a()],P.prototype,`feed`,void 0),l([a()],P.prototype,`error`,void 0),l([a()],P.prototype,`open`,void 0),l([a()],P.prototype,`openEvent`,void 0),l([a()],P.prototype,`paused`,void 0),P=l([s(`qse-cond-overview`)],P);var F=[`off`,`low`,`medium`,`high`,`xhigh`],lt=[`small`,`medium`,`large`],I=class extends d{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new C(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),k.doc(`packages`).then(e=>this.packages=je(e.doc)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(w(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await k.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=n(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return m(8);if(e.loadError)return t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${e.loadError}</div></div>`;let n=e.draft,r=n.classes.map(e=>e.id);return t`
      ${this.renderTest(r)}
      <details class="sec" open data-sec="classes">
        <summary>Task classes <span class="sec-sub">${n.classes.length} classes; a classifier sees only the descriptions</span></summary>
        <div class="sec-body">
          ${n.classes.map((e,t)=>this.renderClass(e,t,n.classes.length,r))}
          <button class="btn" data-add-class @click=${()=>this.addClass()}>${_(`plus`)} Add a class</button>
        </div>
      </details>
      <details class="sec" data-sec="classifier">
        <summary>Classifier stages <span class="sec-sub">asked in order until one answers at its threshold</span></summary>
        <div class="sec-body">
          ${n.classifier.stages.map((e,r)=>t`<div class="row-card" data-stage=${e.kind}>
              <div class="row-card-head">
                <strong>${r+1}. ${e.kind===`decider`?`decider (role decider: any model behind the Jev question shape)`:`Qwen, thinking off (role classifier)`}</strong>
                <span class="toolbar" style="margin:0">
                  <label class="switch"><input type="checkbox" .checked=${e.enabled} @change=${e=>this.upd([`classifier`,`stages`,r,`enabled`],e.target.checked)} /> enabled</label>
                  <button class="btn btn-ghost btn-sm" ?disabled=${r===0} @click=${()=>this.moveStage(r,-1)} aria-label="earlier">↑</button>
                  <button class="btn btn-ghost btn-sm" ?disabled=${r===n.classifier.stages.length-1} @click=${()=>this.moveStage(r,1)} aria-label="later">↓</button>
                </span>
              </div>
              <div class="form-grid">
                ${this.num(`accept at (probability)`,e.accept,e=>this.upd([`classifier`,`stages`,r,`accept`],e),{step:.01,min:0,max:1,hint:`below it, the next stage is asked`})}
                ${this.num(`timeout (ms)`,e.timeout_ms,e=>this.upd([`classifier`,`stages`,r,`timeout_ms`],Math.round(e)),{step:50,min:1,hint:`a slower answer counts as no answer`})}
              </div>
            </div>`)}
          ${this.renderContextStage(n)}
          <div class="form-grid">
            ${this.text(`question to the decider`,n.classifier.question,e=>this.upd([`classifier`,`question`],e))}
            ${this.num(`digest (characters)`,n.classifier.digest_chars,e=>this.upd([`classifier`,`digest_chars`],Math.round(e)),{step:100,min:200,hint:`Qwen's digest of the user message, head and tail; the decider's is sized the same way to its own limits`})}
            <label class="switch" data-context-line><input type="checkbox" .checked=${n.classifier.context_line??!1} @change=${e=>this.upd([`classifier`,`context_line`],e.target.checked)} /> context line: the previous class, the last tools, the file types, the checker's status and a compaction, at most 120 characters out of the tail's share (no path, no file text)</label>
            ${this.text(`package question`,n.classifier.package_question,e=>this.upd([`classifier`,`package_question`],e))}
            ${this.num(`package accept at`,n.classifier.package_accept,e=>this.upd([`classifier`,`package_accept`],e),{step:.01,min:0,max:1})}
          </div>
          <label class="field field-block"><span>Qwen's instruction (<code>{classes}</code> becomes the id and description list)</span>
            <textarea rows="5" .value=${n.classifier.qwen_prompt} @input=${e=>this.upd([`classifier`,`qwen_prompt`],e.target.value)}></textarea></label>
        </div>
      </details>
      <details class="sec" data-sec="stickiness">
        <summary>Stickiness and structure <span class="sec-sub">when a session changes class, and what the request shape decides</span></summary>
        <div class="sec-body">
          <div class="form-grid">
            ${this.num(`switch class at (probability)`,n.stickiness.switch,e=>this.upd([`stickiness`,`switch`],e),{step:.01,min:0,max:1,hint:`a session with a class moves to another only at this confidence`})}
            <label class="field"><span>opencode's compaction request</span>
              <select @change=${e=>this.upd([`structural`,`compaction`],e.target.value)}>
                ${[`keep`,`classify`,...r].map(e=>t`<option value=${e} ?selected=${e===n.structural.compaction}>${e===`keep`?`keep the session class`:e===`classify`?`ask a classifier`:`class ${e}`}</option>`)}
              </select><span class="hint">the turn itself; the session keeps its class after it</span></label>
            <label class="field"><span>a turn that only carries tool results</span>
              <select @change=${e=>this.upd([`structural`,`tool_turn`],e.target.value)}>
                <option value="keep" ?selected=${n.structural.tool_turn===`keep`}>keep the session class (no model call)</option>
                <option value="classify" ?selected=${n.structural.tool_turn===`classify`}>ask a classifier every turn</option>
              </select></label>
            <label class="field" data-tool-turn-effort><span>effort of a tool-result turn</span>
              <select aria-label="effort of a tool-result turn" @change=${e=>{let t=e.target.value;this.upd([`structural`,`tool_turn_effort`],t===`inherit`?void 0:t)}}>
                ${ze.map(e=>t`<option value=${e} ?selected=${e===(n.structural.tool_turn_effort??`inherit`)}>${e===`inherit`?`the class's (inherit)`:he(e,n,null)}</option>`)}
              </select><span class="hint">a turn whose newest message is a tool result (a screenshot a tool returned too) thinks at this level, its switch and cap; the first turn and every new user message keep the class's, the note stays the class's. A profile can set its own (Profiles › Thinking); a client's own budget wins</span></label>
            <label class="field" data-side-requests><span>a client's title or summary request</span>
              <select aria-label="a client's title or summary request" @change=${e=>this.upd([`structural`,`side_requests`],e.target.value)}>
                <option value="on" ?selected=${(n.structural.side_requests??`on`)===`on`}>a side request: no class, thinking off, short, first in the queue</option>
                <option value="off" ?selected=${n.structural.side_requests===`off`}>a turn, as any request</option>
              </select><span class="hint">known by its shape: no tools and the client's own fixed instruction (${(n.structural.side_shapes??Pe).map(e=>`${e.client??`a client`}'s ${e.kind}, at most ${e.max_tokens} tokens`).join(`; `)}). It asks no classifier, gets no note and leaves the session's class alone; thinking only when the client asks for a number</span></label>
          </div>
        </div>
      </details>
      <details class="sec" data-sec="effort">
        <summary>Effort levels <span class="sec-sub">a budget and a sentence at the prompt's end; never the template's reasoning_effort</span></summary>
        <div class="sec-body">
          <p class="muted small">The template renders <code>reasoning_effort</code> as the first system tokens, so changing it per turn would cost a cold prefill. A level here sets the reasoning budget (0 = no cap below the backstop), thinking on or off (off uses <code>enable_thinking: false</code> only when the server's effort is medium, else a budget of 1), and a sentence in the tail note.</p>
          <p class="muted small">The budget is a real cap (SRV-159): when the reasoning reaches it, the server closes it with the session monitor's hidden close, so the client sees only <code>&lt;/think&gt;</code> and the answer. Defaults: low 1,024, medium 8,192, high 16,384, xhigh no cap. A profile can override a level (Profiles › Thinking); a client's own budget always wins.</p>
          ${F.map(e=>t`<div class="row-card" data-effort=${e}>
              <div class="row-card-head"><strong class="mono">${e}</strong>
                <label class="switch"><input type="checkbox" .checked=${n.effort.levels[e].thinking} @change=${t=>this.upd([`effort`,`levels`,e,`thinking`],t.target.checked)} /> thinking</label></div>
              <div class="form-grid">${this.num(`budget (tokens, 0 = no cap)`,n.effort.levels[e].budget,t=>this.upd([`effort`,`levels`,e,`budget`],Math.round(t)),{step:256,min:0})}</div>
              <label class="field field-block"><span>sentence in the note</span><textarea rows="2" .value=${n.effort.levels[e].note} @input=${t=>this.upd([`effort`,`levels`,e,`note`],t.target.value)}></textarea></label>
            </div>`)}
          <label class="switch"><input type="checkbox" .checked=${n.effort.raise_on_fail} @change=${e=>this.upd([`effort`,`raise_on_fail`],e.target.checked)} /> a failing checker (RESULT: FAIL) raises the effort one level for that turn</label>
          ${this.renderSize(n.effort.size??Ke)}
        </div>
      </details>
      <details class="sec" data-sec="injection">
        <summary>The note <span class="sec-sub">where hidden text goes and how much</span></summary>
        <div class="sec-body">
          <p class="muted small">The note goes into the last turn, right before its <code>&lt;|im_end|&gt;</code>. Later turns replay it at the same token position, so the resident prefix keeps matching. <code>{text}</code> is replaced by the effort sentence, the package and the rules.</p>
          <label class="field field-block"><span>wrapper</span><textarea rows="6" data-wrapper .value=${n.injection.wrapper} @input=${e=>this.upd([`injection`,`wrapper`],e.target.value)}></textarea></label>
          <div class="form-grid">
            ${this.num(`max tokens per note`,n.injection.max_note_tokens,e=>this.upd([`injection`,`max_note_tokens`],Math.round(e)),{step:500,min:16})}
            ${this.num(`max tokens replayed per session`,n.injection.max_replay_tokens,e=>this.upd([`injection`,`max_replay_tokens`],Math.round(e)),{step:1e3,min:0})}
            <label class="switch"><input type="checkbox" .checked=${n.injection.replay} @change=${e=>this.upd([`injection`,`replay`],e.target.checked)} /> replay notes on later turns</label>
          </div>
        </div>
      </details>
      ${O(e)}
      ${j(e,`the router`)}
    `}renderTest(n){let r=this.result;return t`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?t`<span class="err-ink small" role="alert">${this.testError}</span>`:f}
      </div>
      ${r?r.active?t`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${E(r.class_id??null,n)} ${Fe(r.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${$e(r.source??null)}${r.kept?t` <span class="badge badge-warn">${r.kept}</span>`:f} <span class="muted num">${e(r.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${r.effort??`unchanged`} · thinking ${r.thinking??`on`}${r.budget==null?``:` · budget ${r.budget}`}</dd></div>
                  ${r.extra?.size?t`<div data-test-size><dt>task size</dt><dd>${ce(r.extra.size)}</dd></div>`:f}
                  <div><dt>profile</dt><dd>${r.profile} (${r.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(r.stages??[]).map(n=>t`<div class="stage ${n.error?`is-error`:n.below?``:`is-accepted`}"><strong>${me(n.stage)}</strong><span>${n.error??t`${n.label} ${Fe(n.confidence,n.accept??.6)}`}</span><span class="num muted">${e(n.ms)}</span></div>`)}</div>
              </div>
              <div>${r.note?t`<p class="muted small">The model would see this at the end of the last turn (${(r.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${r.note}</pre>`:t`<p class="muted small">No note: ${r.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:t`<p class="muted small" style="margin-top:12px">Profile ${r.profile} is raw (${r.mode}): the conductor does nothing for it.</p>`:f}
    </section>`}renderClass(e,n,r,i){let a=e.policy,o=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return t`<details class="row-card" data-class-row=${e.id} ?open=${o}>
      <summary>
        <span>${E(e.id,i,e.label)} <span class="muted small mono">${a.effort_by_size?lt.map(e=>N(a,e)).join(`/`):a.effort??`medium`}${(a.packages??[]).length?` · ${(a.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${n===0} @click=${()=>this.moveClass(n,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n===r-1} @click=${()=>this.moveClass(n,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${r<=1} @click=${()=>this.removeClass(n)} aria-label=${`remove ${e.id}`}>${_(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${e.id} ?readonly=${!o} @input=${e=>this.upd([`classes`,n,`id`],e.target.value)} data-class-id /><span class="hint">${o?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${e.label} @input=${e=>this.upd([`classes`,n,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${e.description} @input=${e=>this.upd([`classes`,n,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${A(F,a.effort??`medium`,e=>this.upd([`classes`,n,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div class="toolbar size-map" style="margin:0" data-size-map=${e.id}><span class="muted small">by task size (the effort above is the ceiling)</span>
        ${lt.map(r=>t`<label class="field"><span>${r}</span><select aria-label=${`${e.id} effort when ${r}`} @change=${e=>this.setSizeLevel(n,a.effort_by_size,r,e.target.value)}>
            <option value="" ?selected=${!a.effort_by_size?.[r]}>the class's</option>
            ${F.map(e=>t`<option value=${e} ?selected=${a.effort_by_size?.[r]===e}>${e}</option>`)}
          </select>${a.effort_by_size?.[r]&&N(a,r)!==a.effort_by_size[r]?t`<span class="hint">cut to ${N(a,r)}</span>`:f}</label>`)}
      </div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let r=(a.packages??[]).includes(e);return t`<button class="pick ${r?`is-on`:``}" aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`classes`,n,`policy`,`packages`],r?(a.packages??[]).filter(t=>t!==e):[...a.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(a.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,n,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}renderSize(e){let n=(t,n)=>this.upd([`effort`,`size`],w(e,t,n));return t`<div class="row-card" data-size-config>
      <div class="row-card-head"><strong>Task size</strong>
        <label class="switch"><input type="checkbox" .checked=${e.reset_on_compaction} @change=${e=>n([`reset_on_compaction`],e.target.checked)} /> a compaction starts the size over</label></div>
      <p class="muted small">Each signal is small below its first number, medium up to its second, large above. The task's size is the largest any counting signal reached, and it never falls within a task. A class's levels per size (in its row above) pick the effort under the class's effort; whether that acts is each profile's <em>effort by task size</em> (off, shadow, on).</p>
      ${tt.map(r=>{let i=e.signals[r.id]??Ke.signals[r.id];return t`<div class="size-signal" data-size-signal=${r.id}>
          <label class="switch"><input type="checkbox" .checked=${i.enabled} @change=${e=>n([`signals`,r.id,`enabled`],e.target.checked)} /> ${r.label}</label>
          <div class="form-grid">
            ${this.num(`small below (${r.unit})`,i.small,e=>n([`signals`,r.id],{...i,small:e}),{min:0,step:r.id===`session_tokens`?1e3:1})}
            ${this.num(`medium up to (${r.unit})`,i.medium,e=>n([`signals`,r.id],{...i,medium:e}),{min:0,step:r.id===`session_tokens`?1e3:1,hint:r.hint})}
          </div>
        </div>`})}
    </div>`}setSizeLevel(e,t,n,r){let i={...t??{}};r?i[n]=r:delete i[n],this.upd([`classes`,e,`policy`,`effort_by_size`],Object.keys(i).length?i:void 0)}renderContextStage(e){let n=nt(e),r=(t,r)=>this.upd([`classifier`,`context_stage`],{...n,...e.classifier.context_stage??{},[t]:r});return t`<div class="row-card" data-context-stage>
      <div class="row-card-head">
        <strong>Qwen in the conversation (in-context stage)</strong>
        ${A([`off`,`on`],n.mode,e=>r(`mode`,e),{small:!0,label:`in-context stage`})}
      </div>
      <p class="muted small">A follow-up or a long prompt is classified by the engine inside the conversation it already holds, right before the turn: its prompt up to the last message plus a hidden question, one letter per class, thinking off. The turn resumes from the anchors that prefill leaves. A short first message keeps the stages above; a request with images too.</p>
      <div class="form-grid">
        ${this.num(`long prompt from (tokens)`,n.min_tokens,e=>r(`min_tokens`,Math.round(e)),{step:1024,min:0,max:262144,hint:`a prompt at least this long goes to the stage`})}
        <label class="switch" data-context-followups><input type="checkbox" .checked=${n.followups} @change=${e=>r(`followups`,e.target.checked)} /> every follow-up (a user message after an earlier answer), whatever its length</label>
        ${this.num(`accept at (probability)`,n.accept,e=>r(`accept`,e),{step:.01,min:0,max:1,hint:`below it, the session's class`})}
      </div>
    </div>`}num(e,n,r,i={}){return t`<label class="field"><span>${e}</span><input type="number" .value=${String(n)} step=${i.step??1} min=${i.min??``} max=${i.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&r(t)}} />${i.hint?t`<span class="hint">${i.hint}</span>`:f}</label>`}text(e,n,r){return t`<label class="field"><span>${e}</span><input .value=${n} @input=${e=>r(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(w(n,[`classifier`,`stages`],r))}};l([o({attribute:!1})],I.prototype,`ids`,void 0),l([o({attribute:!1})],I.prototype,`meta`,void 0),l([o({attribute:!1})],I.prototype,`active`,void 0),l([o({attribute:!1})],I.prototype,`onSaved`,void 0),l([a()],I.prototype,`packages`,void 0),l([a()],I.prototype,`testText`,void 0),l([a()],I.prototype,`testProfile`,void 0),l([a()],I.prototype,`testTools`,void 0),l([a()],I.prototype,`testing`,void 0),l([a()],I.prototype,`result`,void 0),l([a()],I.prototype,`testError`,void 0),I=l([s(`qse-cond-routing`)],I);var ut=[`off`,`shadow`,`act`],dt=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,ft=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,L=class extends d{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new C(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),k.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(w(this.ed.draft,e,t))}render(){let e=this.ed,n=this.config?.server.monitor_mode;return t`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${e.loading&&!e.draft?m(8):e.loadError?t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${e.loadError}</div></div>`:t`
            ${n===`raw`?t`<div class="guard-line" data-monitor-off>${_(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:f}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(e.draft)}
            ${this.renderRest(e.draft)}
            ${O(e)}
            ${j(e,`the monitor`)}`}`}renderTasks(e){let n=Ge(e),r=n.filter(t=>e.tasks[t]&&/^C\d/.test(t)),i=n.filter(e=>we(e)),a=n.filter(e=>!r.includes(e)&&!i.includes(e)),o=i.map(t=>e.tasks[t]).find(e=>e.rule===`stall_watchdog`)?.notify;return t`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${r.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${r.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${i.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${i.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${o?.enabled?t`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:t`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, Khaled's ruling of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${i.map(t=>this.renderTask(t,e.tasks[t]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${a.join(`, `)}</span></summary>
        <div class="sec-body">${a.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>`}renderTask(e,n){let r=xe(this.meta?.schemas?.monitor,n.rule),i=this.meta?.tasks?.[e];return t`<div class="row-card" data-monitor-task=${e}>
      <div class="row-card-head">
        <strong>${e}${i?t` · ${i.label}`:f} <span class="muted small mono">${n.rule}</span></strong>
        ${A(ut,n.mode,t=>this.upd([`tasks`,e,`mode`],t),{small:!0,label:`${e} mode without a profile`})}
      </div>
      ${i?.why?t`<p class="muted small">${i.why}</p>`:f}
      ${r?.properties?t`<div class="form-stack">${Object.entries(r.properties).map(([t,r])=>this.field(t,r,n[t],[`tasks`,e,t]))}</div>`:t`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,n,[`tasks`,e])}`}
    </div>`}field(e,n,r,i){let a=Array.isArray(n.type)?n.type[0]:n.type,o=e.replace(/_/g,` `),s=n.description?t`<span class="hint">${n.description}</span>`:f;if(n.enum)return t`<label class="field"><span>${o}</span><select data-param=${e} @change=${e=>this.upd(i,e.target.value)}>${n.enum.map(e=>t`<option value=${String(e)} ?selected=${e===r}>${String(e)}</option>`)}</select>${s}</label>`;if(a===`boolean`)return t`<label class="switch" data-param=${e}><input type="checkbox" .checked=${!!r} @change=${e=>this.upd(i,e.target.checked)} /> ${o}</label>${n.description?t`<p class="hint">${n.description}</p>`:f}`;if(a===`integer`||a===`number`)return t`<label class="field"><span>${o}</span><input type="number" data-param=${e} step=${a===`integer`?`1`:`0.01`} min=${n.minimum??f} max=${n.maximum??f} .value=${r==null?``:String(r)} @input=${e=>{let t=e.target.value;this.upd(i,t===``?null:a===`integer`?Math.round(Number(t)):Number(t))}} />${s}</label>`;if(a===`string`){let n=String(r??``);return dt.test(e)||n.length>60?t`<label class="field field-block"><span>${o}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(n.length/90)))} class=${ft.test(e)?`mono`:``} data-param=${e} .value=${n} @input=${e=>this.upd(i,e.target.value)}></textarea>${s}</label>`:t`<label class="field"><span>${o}</span><input data-param=${e} .value=${n} @input=${e=>this.upd(i,e.target.value)} />${s}</label>`}if(a===`array`&&n.items&&(Array.isArray(n.items.type)?n.items.type[0]:n.items.type)===`string`){let a=!Array.isArray(r)&&Array.isArray(n.default),c=Array.isArray(r)?r:a?n.default:[];return t`<label class="field field-block"><span>${o} <span class="muted small">(one a line${a?`; the default, not saved yet`:``})</span></span><textarea rows=${Math.min(8,Math.max(2,c.length+1))} data-param=${e} .value=${c.join(`
`)} @input=${e=>this.upd(i,ue(e.target.value))}></textarea>${s}</label>`}if(a===`array`&&n.items?.properties){let a=Array.isArray(r)?r:[],c=n.items;return t`<fieldset class="fieldset" data-param=${e}><legend>${o} <span class="muted small">${a.length}</span></legend>${s}
        ${a.map((e,n)=>t`<div class="row-card row-card-sub" data-item=${String(e.id??n)}>
          <div class="row-card-head"><strong class="mono small">${String(e.id??`#${n+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(i,a.filter((e,t)=>t!==n))} aria-label=${`remove ${String(e.id??n+1)}`}>${_(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(c.properties).map(([t,r])=>this.field(t,r,e[t],[...i,n,t]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${e} @click=${()=>this.upd(i,[...a,pt(c)])}>${_(`plus`)} Add</button>
      </fieldset>`}if(a===`object`&&n.properties){let a=r??{};return t`<fieldset class="fieldset" data-param=${e}><legend>${o}</legend>${s}
        <div class="form-stack">${Object.entries(n.properties).map(([e,r])=>a[e]===void 0&&!(n.required??[]).includes(e)?t`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...i,e],pt(r))}>${_(`plus`)} ${e.replace(/_/g,` `)}</button>`:this.field(e,r,a[e],[...i,e]))}</div></fieldset>`}return this.jsonField(o,r,i)}jsonField(e,n,r){return t`<label class="field field-block"><span>${e} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(n,null,2)} @change=${t=>{try{let e=JSON.parse(t.target.value);if(r.length===2&&r[0]===`tasks`){let t=this.ed.draft.tasks[r[1]];this.upd(r,{...e,rule:t.rule,mode:t.mode})}else this.upd(r,e)}catch{this.ed.errors=[`${e}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(e){return t`
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
      </details>`}};l([o({attribute:!1})],L.prototype,`meta`,void 0),l([o({attribute:!1})],L.prototype,`onSaved`,void 0),l([a()],L.prototype,`config`,void 0),L=l([s(`qse-cond-monitor`)],L);function pt(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,pt(t)])):``}var R=class extends d{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=h(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await k.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let t=n(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let e=this.sessions;return g(t`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!e?t`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?m(2):e.length?t`<div class="form-stack" data-watch>${e.map(e=>this.row(e))}</div>`:t`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(e){let n=e.gates.some(e=>e.noted);return t`<div class="row-card" data-watch-session=${e.session}>
      <div class="row-card-head">
        <strong class="mono small">${e.session}</strong>
        <span class="toolbar" style="margin:0">
          ${e.watch?t`<span class="badge ${e.watch.mode===`act`?`badge-on`:`badge-shadow`}">${e.watch.task} ${e.watch.mode}</span>`:f}
          ${e.client_gone?t`<span class="badge badge-warn" data-client-gone>client gone</span>`:e.running?t`<span class="badge badge-on">running</span>`:f}
          ${e.watch?.notify?t`<span class="badge badge-warn" data-notify>push on</span>`:f}
          ${n?t`<span class="badge badge-warn">gate noted</span>`:f}</span>
      </div>
      <p class="small" style="margin:4px 0">${Oe(e)}</p>
      <p class="muted small num" style="margin:0">${e.watch?`${e.watch.profile||`—`} · ${e.watch.client||`—`} · `:``}${e.requests} requests · last ${r(new Date(e.last_seen*1e3).toISOString())}${e.note_at?` · last note ${r(new Date(e.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};l([a()],R.prototype,`sessions`,void 0),l([a()],R.prototype,`monitor`,void 0),l([a()],R.prototype,`error`,void 0),R=l([s(`qse-cond-watch`)],R);var z=class extends d{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=h(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await k.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let t=n(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let e=this.events,n=[...new Set((e??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!n.includes(this.task)&&n.push(this.task),t`${g(t`Monitor journal ${this.paused?t`<span class="badge">paused</span>`:t`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!e?t`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?m(4):e.length?t`<div class="feed" data-journal>${e.map(e=>Re(e,e=>this.open=e))}</div>`:t`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:t`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${n.map(e=>t`<option value=${e} ?selected=${e===this.task}>${e}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${_(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?t`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:f}`}};l([o({attribute:!1})],z.prototype,`session`,void 0),l([a()],z.prototype,`events`,void 0),l([a()],z.prototype,`monitor`,void 0),l([a()],z.prototype,`error`,void 0),l([a()],z.prototype,`task`,void 0),l([a()],z.prototype,`kind`,void 0),l([a()],z.prototype,`paused`,void 0),l([a()],z.prototype,`open`,void 0),z=l([s(`qse-cond-journal`)],z);var mt=[`off`,`low`,`medium`,`high`,`xhigh`],ht={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},gt=new Set([`S1`,`S4`]),_t=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],B=class extends d{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleContract=null,this.roleTest={},this.ed=null,this.roles=new C(this,`roles`),this.residency=new C(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),k.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),k.doc(`packages`).then(e=>this.packages=je(e.doc)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new C(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{let e=await k.roles();this.roleHealth=e.health,this.roleContract=e.contract??null}catch{}}async testRole(t){this.roleTest={...this.roleTest,[t]:`testing…`};try{let n=await k.roleTest(t);this.roleTest={...this.roleTest,[t]:n.ok?`ok in ${e(n.ms)}: "${n.answer?.label}" at ${n.answer?.confidence.toFixed(2)}`:`failed: ${n.error}`}}catch(e){this.roleTest={...this.roleTest,[t]:n(e).message}}this.loadHealth()}render(){return t`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>t`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>S({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?t`<span class="badge badge-lock">${_(`warn`)} locked</span>`:t`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${r(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):t`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(w(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=w(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=w(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderUi(e,n){let r=e.ui??{states:`off`},i=r.kinds??{},a=n||r.states!==`on`,o=Te(r),s=We(r),c=(e,n,i,o,c)=>t`<label class="field"><span>${n}</span><input type="number" inputmode="numeric" min=${i} max=${o} step=${c} ?disabled=${a||s.mode!==`on`} data-ui-progress=${e} .value=${String(s[e])}
        @input=${t=>{let n=Math.round(Number(t.target.value));Number.isFinite(n)&&n>=i&&n<=o&&this.upd([`ui`],{...r,progress:{...r.progress??{},mode:s.mode,[e]:n}})}} /></label>`;return t`<details class="sec" data-sec="ui">
      <summary>What the user sees <span class="sec-sub">TandemLLM's own steps as lines the client shows</span> <span class="badge ${r.states===`on`?`badge-on`:``}" data-ui-badge>${r.states}</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0"><span class="muted small">state lines</span>${A([`off`,`on`],r.states,e=>this.upd([`ui`],{...r,states:e}),{disabled:n,small:!0,label:`state lines`})}</div>
        <p class="muted small">Each step goes out at the moment it happens as one line, <code>› deciding: code_edit, high effort</code>. The server takes them out of the reasoning and the content a client sends back, so Qwen never sees them; they are not tokens and not in the usage. Only in mode on.</p>
        <div class="toolbar" style="margin:0" data-ui-channel><span class="muted small">where</span>${A([`reasoning`,`content`,`inline`],o,e=>this.upd([`ui`],{...r,states_channel:e}),{disabled:a,small:!0,label:`where the lines go`})}</div>
        <p class="muted small">${o===`inline`?t`Each line opens a thinking block of its own and is its title, in time order: opencode shows <code>Thought: › thinking 2 min, 4,096 tokens</code> between Qwen's thinking blocks, and the answer after them. Needs <code>reasoning_content</code>; with <code>&lt;think&gt;</code> in the content (Hermes) the lines go as in the content.`:o===`content`?t`In the content, outside the thinking: before the reasoning, between its parts, after it. opencode shows them in its text part; with <code>&lt;think&gt;</code> in the content (Hermes) a line said inside the block goes out right after <code>&lt;/think&gt;</code>.`:t`In the reasoning opencode and Hermes show (a <code>reasoning_content</code> delta, or inside <code>&lt;think&gt;</code>). A profile saved before this choice existed keeps them here.`}</p>
        <div class="pkg-picks" data-ui-kinds>${Ie.map(({id:e,example:n})=>{let o=i[e]!==!1;return t`<button class="pick ${o?`is-on`:``}" data-ui-kind=${e} title=${n} aria-pressed=${o?`true`:`false`} ?disabled=${a} @click=${()=>this.upd([`ui`],{...r,kinds:{...i,[e]:!o}})}>${e}</button>`})}</div>
        <div class="toolbar" style="margin:0" data-ui-progress-mode><span class="muted small">progress</span>${A([`off`,`on`],s.mode,e=>this.upd([`ui`],{...r,progress:{...r.progress??{},mode:e}}),{disabled:a,small:!0,label:`progress lines`})}</div>
        <p class="muted small">The numbers of a wait, templated: <code>waiting: opencode's title request runs first, position 1</code> while the request waits, <code>reading 35,189 tokens at ~750 tok/s, about 47 s left</code> during a long prefill, <code>thinking 4 min, 9,800 tokens</code> during a long thinking. Streams only; a request that waits opens its stream at its first waiting line.</p>
        <div class="form-grid">
          ${c(`prefill_every_s`,`a prefill line every (s)`,2,600,1)}
          ${c(`thinking_every_s`,`a thinking line every (s)`,10,3600,10)}
          ${c(`thinking_every_tokens`,`or every (tokens)`,256,262144,256)}
        </div>
        <div class="toolbar" style="margin:0" data-ui-narrator><span class="muted small">narrator</span>${A([`off`,`on`,`thoughts`],r.narrator??`off`,e=>this.upd([`ui`],{...r,narrator:e}),{disabled:a,small:!0,label:`narrator`})}</div>
        <p class="muted small">A small model inside the engine (the <code>narrator</code> role, Qwen3.5-0.8B) writes in plain words. <b>thoughts</b>: after each stretch of thinking (and at the long-thinking marks) one line with what Qwen thought about and decided, e.g. “Thought 1m 43s: planned the level layout …”, read from the thought alone; the other lines stay templates. <b>on</b>: it writes every line instead of the template. Qwen waits while it writes, at most 3 % of its time (the rest at the turn’s end); a line it cannot write in time is the template. It loads on its first line; the Roles section below holds its budget and mode.</p>
      </div>
    </details>`}renderToolTurn(e,n){let r=e.thinking.tool_turn_effort,i=Ze(this.router,null).level,a=Ze(this.router,e);return t`<label class="field ${r?``:`is-inherited`}" data-tool-turn-effort><span>effort of a tool-result turn</span>
      <select aria-label="effort of a tool-result turn" ?disabled=${n} @change=${e=>{let t=e.target.value;if(!this.ed?.draft)return;let n={...this.ed.draft.thinking,tool_turn_effort:t};t||delete n.tool_turn_effort,this.ed.set(w(this.ed.draft,[`thinking`],n))}}>
        <option value="" ?selected=${!r}>Routing's: ${i===`inherit`?`the class's`:i}</option>
        ${ze.map(e=>t`<option value=${e} ?selected=${r===e}>${e===`inherit`?`the class's (inherit)`:e}</option>`)}
      </select>
      <span class="hint">${he(a.level,this.router,e)}${a.source===`profile`?` (this profile)`:a.source===`router`?` (Routing)`:``}: a turn whose newest message is a tool result thinks at this level, its switch and cap; the first turn and every new user message keep the class's</span></label>`}renderBudgets(e,n){return t`<div data-budgets>
      <div class="muted small">Thinking budget per level <span class="muted">(reasoning tokens; empty keeps Routing's level, 0 is no cap below the backstop)</span></div>
      <div class="form-grid">
        ${et.map(r=>{let i=e.thinking.budgets?.[r],a=de(r,this.router,null).cap,o=de(r,this.router,e);return t`<label class="field ${i==null?`is-inherited`:``}" data-budget=${r}><span class="mono">${r}</span>
            <input type="number" inputmode="numeric" min="0" max=${Je} step="1024" ?disabled=${n} placeholder=${a==null?``:String(a)} .value=${i==null?``:String(i)}
              @change=${e=>{let t=e.target.value.trim();this.ed?.draft&&this.ed.set(w(this.ed.draft,[`thinking`],Se(this.ed.draft.thinking,r,t===``?null:Number(t))))}} />
            <span class="hint">${He(o.cap)}${o.source===`router`?` (Routing)`:o.source===`profile`?` (this profile)`:``}</span></label>`})}
      </div>
    </div>`}renderEditor(e){if(e.loading&&!e.draft)return m(8);if(e.loadError)return t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${e.loadError}</div></div>`;let n=e.draft,r=!!e.locked,i=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],a=this.router?.classes??[];return t`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${_(`playground`)} Try in Playground</a></div>
      ${r?t`<div class="note" data-lock-reason>${_(`warn`)} ${e.locked}. Every control below is read-only.</div>`:f}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${n.description} ?disabled=${r} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${A([`raw`,`shadow`,`on`],n.mode,e=>this.upd([`mode`],e),{disabled:r,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${i.map(e=>{let i=n.clients.includes(e);return t`<button class="pick ${i?`is-on`:``}" ?disabled=${r} aria-pressed=${i?`true`:`false`} @click=${()=>this.upd([`clients`],i?n.clients.filter(t=>t!==e):[...n.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${r} @change=${e=>this.upd([`default_class`],e.target.value)}>${a.map(e=>t`<option value=${e.id} ?selected=${e.id===n.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(ht).map(e=>t`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${A([`off`,`shadow`,`on`],n.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:r,small:!0,label:e})}<span class="small">${ht[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${a.map(e=>{let i=n.routing[e.id]??{},a=Object.keys(i).length>0;return t`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${E(e.id,this.ids,e.label)}
                ${a?t`<button class="btn btn-ghost btn-sm" ?disabled=${r} @click=${()=>{let t={...n.routing};delete t[e.id],this.upd([`routing`],t)}}>use the router's</button>`:t`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${A(mt,i.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...i,effort:t}),{disabled:r,small:!0,label:`effort override of ${e.id}`,cls:i.effort?``:`is-inherited`})}</div>
              <div class="pkg-picks">${this.packages.map(n=>{let a=i.packages??e.policy.packages??[],o=a.includes(n);return t`<button class="pick ${o?`is-on`:``}" ?disabled=${r} @click=${()=>this.upd([`routing`,e.id],{...i,packages:o?a.filter(e=>e!==n):[...a,n]})}>${n}</button>`})}</div>
            </div>`})}
        </div>
      </details>
      <details class="sec" data-sec="tasks">
        <summary>Session monitor tasks <span class="sec-sub">the frame (SRV-109) runs these; act needs the role online</span></summary>
        <div class="sec-body">${Object.entries(n.tasks).map(([e,t])=>this.renderTask(e,t.mode,t.cap,t.thresholds,r))}</div>
      </details>
      <details class="sec" data-sec="thinking">
        <summary>Thinking <span class="sec-sub">the effort when no class sets one</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${A(mt,n.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:r,small:!0,label:`default effort`})}</div>
          <label class="switch"><input type="checkbox" ?disabled=${r} .checked=${n.thinking.hints} @change=${e=>this.upd([`thinking`,`hints`],e.target.checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <div class="toolbar" style="margin:0" data-by-size><span class="muted small">effort by task size</span>${A([`off`,`shadow`,`on`],n.thinking.by_size??`shadow`,e=>this.upd([`thinking`,`by_size`],e),{disabled:r,small:!0,label:`effort by task size`,cls:n.thinking.by_size?``:`is-inherited`})}</div>
          <p class="muted small">Effort by task size (SRV-157): a class with levels per size (Routing › the class) thinks at its size's level, never above the class's effort; the size only rises within a task. <strong>off</strong>: the class's effort; <strong>shadow</strong>: the class's effort, and the journal shows the size and the effort it would set; <strong>on</strong>: the size picks.</p>
          ${this.renderToolTurn(n,r)}
          ${this.renderBudgets(n,r)}
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="library">
        <summary>Package library <span class="sec-sub">a searched skill for a new message or a stuck run (R1)</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0" data-library-mode><span class="muted small">library</span>${A([`off`,`shadow`,`on`],n.library??`off`,e=>this.upd([`library`],e),{disabled:r,small:!0,label:`package library`,cls:n.library?``:`is-inherited`})}</div>
          <p class="muted small">On a new user message, and on a low-quality or stuck outcome, the router searches the library (the <a href="#/conductor?screen=packages">Packages</a> screen) with the message, asks the package question among the best matches and puts the skill that fits at the prompt's tail. <strong>off</strong>: nothing; <strong>shadow</strong>: the journal shows the pick, nothing goes in; <strong>on</strong>: the skill and the card that tells the model how to search the library itself.</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113); a pack that is off turns its task off</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${_t.map(([e,i,a])=>t`<label class="switch" data-pack=${e}><input type="checkbox" ?disabled=${r} .checked=${n.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${a} <span class="muted small">(${i}, ${n.toolgate[e]?n.tasks[i]?.mode??`off`:`off`})</span></label>`)}
          <label class="field"><span>Tier 1 timeout (ms)</span><input type="number" ?disabled=${r} .value=${String(n.toolgate.tier1_timeout_ms)} @input=${e=>this.upd([`toolgate`,`tier1_timeout_ms`],Math.round(Number(e.target.value)))} /></label>
        </div>
        <p class="muted small">With a pack on, its task runs in the mode set above; with one of them able to act, an answer's tool calls are held to its end and released whole. The floors, rules and reminders are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      <details class="sec" data-sec="context">
        <summary>Context <span class="sec-sub">the session tasks S1, S2, S4 (SRV-121): their numbers and the push</span></summary>
        <div class="sec-body"><div class="form-grid">
          <label class="field"><span>S1: failed runs of one gate before the blocked-stage note <span class="hint">count, default 3</span></span><input type="number" min="1" max="20" ?disabled=${r} data-context="s1_runs" .value=${String(n.context.s1_runs)} @input=${e=>this.ctx(`s1_runs`,`S1`,`fails`,Math.round(Number(e.target.value)))} /></label>
          <label class="field"><span>S4: minutes without progress before the note, and of silence before the client counts as gone <span class="hint">minutes, default 20</span></span><input type="number" min="1" max="600" ?disabled=${r} data-context="s4_minutes" .value=${String(n.context.s4_minutes)} @input=${e=>this.ctx(`s4_minutes`,`S4`,`minutes`,Math.round(Number(e.target.value)))} /></label>
          <label class="switch" data-context="notify"><input type="checkbox" ?disabled=${r} .checked=${n.context.notify} @change=${e=>this.upd([`context`,`notify`],e.target.checked)} /> NOTIFY: a push through ntfy to the phone when the client has gone <span class="badge ${n.context.notify?`badge-warn`:``}">${n.context.notify?`on`:`off`}</span></label>
        </div>
        <p class="muted small" data-notify-rule>Allowed by Khaled on 2026-09-30: off by default, on in the overnight profile. Only S4 in act pushes, never in shadow, once a turn end; the token is a file on the box. S2 (re-read after a compaction) has no number here: its texts and state files are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      ${this.renderUi(n,r)}
      <details class="sec" data-sec="hints">
        <summary>Speed hints <span class="badge badge-warn">not gated</span></summary>
        <div class="sec-body">
          <p class="warn-ink small">No SPD measurement is in the journal yet (SPD-73); the router's ≤ 1 ulp text rule and the never-worse gate come first.</p>
          <label class="switch"><input type="checkbox" ?disabled=${r} .checked=${n.hints.stair_width} @change=${e=>this.upd([`hints`,`stair_width`],e.target.checked)} /> StairCut width hint</label>
          <label class="switch"><input type="checkbox" ?disabled=${r} .checked=${n.hints.prefix_retention} @change=${e=>this.upd([`hints`,`prefix_retention`],e.target.checked)} /> prefix retention hint</label>
        </div>
      </details>
      ${O(e)}
      ${j(e,`profile ${e.name}`)}
    </div>`}renderTask(e,n,r,i,a){let o=this.meta?.tasks[e],s=o?.role&&o.tier===`1`,c=!s||!!this.roleHealth[o.role]?.up;return t`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${o?.family??``}</div></div>
      <div>${o?.label??e}<div class="task-why">${o?.why??``}</div>${s&&!c?t`<div class="warn-ink small">${_(`warn`)} the ${o.role} role is offline: act is refused</div>`:f}</div>
      <div class="task-mode">${A([`off`,`shadow`,`act`],n,t=>this.upd([`tasks`,e,`mode`],t),{disabled:a,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${gt.has(e)&&Object.keys(i).length?t`<p class="muted small" data-in-context=${e}>${Object.entries(i).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:f}
        ${Object.entries(gt.has(e)?{}:i).map(([n,r])=>{let i=o?.thresholds[n];return t`<label class="field ${i&&i.default!==r?`is-changed`:``}"><span>${n.replace(/_/g,` `)} <span class="hint">${i?.unit??``}${i?`, default ${i.default}`:``}</span></span>
            <input type="number" ?disabled=${a} .value=${String(r)} step="any" @input=${t=>{let r=Number(t.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,n],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${a} min="0" max="10" .value=${String(r)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let n=this.roles;if(!n.draft)return n.loadError?t`<p class="err-ink small">${n.loadError}</p>`:f;let r=n.draft,i=(e,t)=>n.set(w(r,e,t)),a=ot(n.base?.meta.defaults),o=this.roleContract?.engine_tokenizer_sha256??null,s=this.meta?.roles,c=s?.kinds??Me,l=s?.windows??Ye,u=s?.fallbacks??[`PASS`,`stale`,`skip`,`next-role`],d=Qe.filter(e=>!s||s.writes.includes(e));return t`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${n.base?.meta.defaults?.length?t`<div class="note" data-role-defaults>${_(`warn`)} Version ${n.base.version} was stored before the role contract and loaded with ${n.base.meta.defaults.length} default${n.base.meta.defaults.length===1?``:`s`} (marked <span class="badge">default</span> below; the journal has the list). Save the roles to make them explicit.</div>`:f}
        <p class="muted small">The contract (SRV-133): when a role may use the GPU, whether a Qwen request cuts it, how long its caller waits before the fallback, what it may write, and the tokenizer it reads, which decides what it may share with Qwen. Engine tokenizer: <code class="mono" data-engine-sha>${o?`${o.slice(0,12)}…`:`not recorded`}</code></p>
        ${Object.entries(r).map(([n,r])=>{let s=this.roleHealth[n],p=new Set((a[n]??[]).map(e=>e.split(`=`)[0])),m=e=>p.has(e)?t` <span class="badge" data-default=${`${n}.${e}`}>default</span>`:f,ee=Ae(r.kind,r.window),h=oe(r.kind,r.writes),g=_e(r,o);return t`<div class="row-card" data-role-row=${n}>
            <div class="row-card-head"><strong>${n}</strong>
              <span class="toolbar" style="margin:0">${s?t`<span class="badge ${s.up?`badge-ok`:r.mode===`off`?``:`badge-warn`}">${s.up?`up ${e(s.ms)}`:r.mode===`off`?`off`:`down: ${s.error??`?`}`}</span>`:f}
                <button class="btn btn-sm" @click=${()=>this.testRole(n)} data-role-test=${n}>Test</button></span></div>
            ${this.roleTest[n]?t`<p class="small" role="status">${this.roleTest[n]}</p>`:f}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${r.endpoint} @input=${e=>i([n,`endpoint`],e.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${r.model} @input=${e=>i([n,`model`],e.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(r.timeout_ms)} @input=${e=>i([n,`timeout_ms`],Math.round(Number(e.target.value)))} /></label>
              <label class="field"><span>memory claim (GiB, its peak)${m(`memory_claim_gib`)}</span><input type="number" step="0.1" min="0" data-role-field="memory_claim_gib" .value=${String(r.memory_claim_gib)} @input=${e=>i([n,`memory_claim_gib`],Number(e.target.value))} /></label>
            </div>
            <div class="form-grid" data-role-contract=${n}>
              <label class="field"><span>kind${m(`kind`)}</span><select data-role-field="kind" @change=${e=>i([n,`kind`],e.target.value)}>${c.map(e=>t`<option value=${e} ?selected=${e===r.kind}>${e}</option>`)}</select></label>
              <label class="field"><span>window${m(`window`)}</span><select data-role-field="window" @change=${e=>i([n,`window`],e.target.value)}>${l.map(e=>t`<option value=${e} ?selected=${e===r.window} ?disabled=${e!==r.window&&!!Ae(r.kind,e)} title=${Xe[e]}>${e}</option>`)}</select></label>
              <label class="field"><span>latency budget (ms)${m(`latency_budget_ms`)}</span><input type="number" min="1" max="600000" data-role-field="latency_budget_ms" .value=${String(r.latency_budget_ms)} @input=${e=>i([n,`latency_budget_ms`],Math.round(Number(e.target.value)))} /></label>
              <label class="field"><span>fallback (off, slow, unsure or refused)${m(`fallback`)}</span><select data-role-field="fallback" @change=${e=>i([n,`fallback`],e.target.value)}>${u.map(e=>t`<option value=${e} ?selected=${e===r.fallback}>${e}</option>`)}</select></label>
              <label class="field"><span>writes${m(`writes`)}</span><select data-role-field="writes" @change=${e=>i([n,`writes`],e.target.value)}>${d.map(e=>t`<option value=${e} ?selected=${e===r.writes} ?disabled=${e!==r.writes&&!!oe(r.kind,e)}>${e.replace(/_/g,` `)}</option>`)}</select></label>
              <label class="field field-wide"><span>tokenizer sha256 (the engine records its own; none: text only)${m(`tokenizer_sha256`)}</span><input class="mono" data-role-field="tokenizer_sha256" spellcheck="false" .value=${r.tokenizer_sha256} @input=${e=>i([n,`tokenizer_sha256`],e.target.value.trim().toLowerCase()||`none`)} /></label>
            </div>
            <div class="toolbar" style="margin:0">
              <label class="switch"><input type="checkbox" data-role-field="preemptible" .checked=${r.preemptible} @change=${e=>i([n,`preemptible`],e.target.checked)} /> preemptible: a Qwen request cuts it, it resumes later${m(`preemptible`)}</label>
            </div>
            <div><span class="muted small">sharing with Qwen${m(`sharing`)}</span>
              <div class="pkg-picks" data-role-sharing=${n}>${g.map(e=>{let a=r.sharing.includes(e.form);return t`<button class="pick ${a?`is-on`:``}" aria-pressed=${a?`true`:`false`} ?disabled=${!a&&!e.allowed} title=${e.why??``} data-share=${e.form} @click=${()=>i([n,`sharing`],a?r.sharing.filter(t=>t!==e.form):[...r.sharing,e.form])}>${e.label}</button>`})}</div>
              ${g.filter(e=>!e.allowed).map(e=>t`<p class="muted small" data-share-why=${e.form}>${e.label}: ${e.why}</p>`)}
            </div>
            ${ee?t`<p class="warn-ink small" data-role-why="window">${_(`warn`)} ${ee}</p>`:f}
            ${h?t`<p class="warn-ink small" data-role-why="writes">${_(`warn`)} ${h}</p>`:f}
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${A([`off`,`shadow`,`on`],r.mode,e=>i([n,`mode`],e),{small:!0,label:`${n} mode`})}<span class="muted small mono">${r.contract}</span></div>
          </div>`})}
        ${O(n)}
        ${j(n,`the roles`)}
      </div>
    </details>`}renderResidency(){let e=this.residency;if(!e.draft)return e.loadError?t`<p class="err-ink small">${e.loadError}</p>`:f;let n=e.draft,r=(t,r)=>e.set(w(n,t,r)),i=(e,i=`0.1`,a=!1)=>t`<input type="number" step=${i} min="0" .value=${String(n[e[0]])} @input=${t=>{let n=Number(t.target.value);r(e,a?Math.round(n):n)}} data-res=${e[0]} />`,a=[`measured`,`peak`,`declared`];return t`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${i([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${i([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${i([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${n.order.join(`, `)} @change=${e=>r([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(n.claims).map(([e,i])=>t`<div class="row-card" data-claim=${e}>
            <div class="row-card-head"><strong>${e}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[e]:t,...i}=n.claims;r([`claims`],i)}} aria-label=${`remove the claim of ${e}`}>${_(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(i.gib)} @input=${t=>r([`claims`,e,`gib`],Number(t.target.value))} data-claim-gib=${e} /></label>
              <label class="field"><span>basis</span><select @change=${t=>r([`claims`,e,`basis`],t.target.value)}>${a.map(e=>t`<option value=${e} ?selected=${e===i.basis}>${e}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${i.note} @input=${t=>r([`claims`,e,`note`],t.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in n.claims} @click=${()=>{r([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${_(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${O(e)}
        ${j(e,`the residency limits`)}
      </div>
    </details>`}};l([o({attribute:!1})],B.prototype,`ids`,void 0),l([o({attribute:!1})],B.prototype,`meta`,void 0),l([o({attribute:!1})],B.prototype,`tiles`,void 0),l([o({attribute:!1})],B.prototype,`selected`,void 0),l([o({attribute:!1})],B.prototype,`onSaved`,void 0),l([a()],B.prototype,`router`,void 0),l([a()],B.prototype,`packages`,void 0),l([a()],B.prototype,`roleHealth`,void 0),l([a()],B.prototype,`roleContract`,void 0),l([a()],B.prototype,`roleTest`,void 0),l([a()],B.prototype,`newClaim`,void 0),B=l([s(`qse-cond-profiles`)],B);var V=[`package`,`task_skill`,`test_skill`,`card`,`library`],H={package:`package`,task_skill:`task skill`,test_skill:`test skill`,card:`card`,library:`library skill`},vt=3.5,yt={library:350,library_card:200},bt={q:``,category:`all`,cls:``,enabled:`all`,over:!1};function U(e){return Math.floor(e.length/vt)}function xt(e,t){return t?t.category:e.library?`library`:e.kind===`skill`?`task_skill`:`package`}function St(e,t,n,r){let i=r?.limits??yt,a=n===`library`?i.library:e===(r?.library.card??`tandem-library`)?i.library_card:null;return a!=null&&a<=t.max_tokens?{limit:a,source:`house`}:{limit:t.max_tokens,source:`max_tokens`}}function Ct(e,t){return(e?.classes??[]).filter(e=>(e.policy.packages??[]).includes(t)).map(e=>e.id)}function wt(e,t,n,r){return Object.entries(e?.packages??{}).map(([e,i])=>{let a=t?.packages[e]??null;return{name:e,pkg:i,entry:a,category:xt(i,a),classes:Ct(r,e),usage:n?.[e]??null,score:null}}).sort((e,t)=>V.indexOf(e.category)-V.indexOf(t.category)||e.name.localeCompare(t.name))}function Tt(e){return e.toLowerCase().split(/\s+/).filter(Boolean)}function Et(e,t){let n=`${e.name} ${e.pkg.title} ${e.pkg.description} ${e.pkg.source}`.toLowerCase();return Tt(t).every(e=>n.includes(e))}function Dt(e,t){return e.entry?e.entry.over:e.pkg.source===`inline`&&U(W(e.pkg.body,!!e.pkg.library)[0])>St(e.name,e.pkg,e.category,t).limit}function Ot(e,t,n,r=null){let i=new Map((n??[]).map(e=>[e.name,e.score])),a=t.q.trim(),o=e.filter(e=>t.category===`all`||e.category===t.category).filter(e=>!t.cls||e.classes.includes(t.cls)||e.category===`library`&&(e.entry?.fits??[]).includes(t.cls)).filter(e=>t.enabled===`all`||t.enabled===`on`===e.pkg.enabled).filter(e=>!t.over||Dt(e,r)).filter(e=>!a||Et(e,a)||e.category===`library`&&i.has(e.name)).map(e=>({...e,score:a&&e.category===`library`?i.get(e.name)??null:null}));return a?o.sort((e,t)=>V.indexOf(e.category)-V.indexOf(t.category)||(e.score!=null||t.score!=null)&&(t.score??-1)-(e.score??-1)||e.name.localeCompare(t.name)):o}function kt(e,t){let n=e=>!!t?.packages[e]?.general;return[...e.filter(n),...e.filter(e=>!n(e))]}function At(e,t,n,r){let i=e.classes.findIndex(e=>e.id===t);if(i<0)return{router:e,error:`no class ${t}`};if(r?.packages[n]?.library)return{router:e,error:`${n} is a library skill: the router searches the library, a class does not list it`};let a=e.classes[i].policy.packages??[];if(a.includes(n))return{router:e,error:null};if(a.length>=12)return{router:e,error:`${t} already lists 12 packages, the most a class may`};let o=e.classes.map((e,t)=>t===i?{...e,policy:{...e.policy,packages:kt([...a,n],r)}}:e);return{router:{...e,classes:o},error:null}}function jt(e,t,n){return{...e,classes:e.classes.map(e=>e.id===t?{...e,policy:{...e.policy,packages:(e.policy.packages??[]).filter(e=>e!==n)}}:e)}}var Mt=/^## Reference\b/m;function W(e,t){let n=t?Mt.exec(e):null;return n?[e.slice(0,n.index).trim(),e.slice(n.index).trim()]:[e.trim(),``]}function Nt(e){let t=1;for(;e.packages[`package-${t}`];)t++;return`package-${t}`}function Pt(e){return{title:`Package ${e}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0}}function Ft(e,t,n){return{...e,source:`inline`,body:n?`${t}\n\n${n}`:t}}function It(e,t){return t?e.filter(e=>e.profile===t):e}function Lt(e){return e===`on`?`the router puts the skill it picks into the prompt from the next request, and the library card on the coding classes with tools`:e===`shadow`?`the router searches and picks, and the journal shows the pick; nothing goes into the prompt`:`the router does not search the library`}var G=class extends d{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.catalog=null,this.usage=null,this.onChanged=()=>void 0,this.profile=``,this.busy=null,this.msg=null,this.error=null}async setMode(e,r){this.busy=e,[this.msg,this.error]=[null,null];try{let n=await k.doc(`profiles`,e),i={...n.doc,library:r},a=await k.check(`profiles`,e,i);if(a.errors.length){this.error=`${e}: ${a.errors.join(`; `)}`;return}if(!await D(`Library ${r} for ${e}?`,[`With ${r}, ${Lt(r)}.`,t`The server checked the change (live v${n.version}):${Le(a.changes)}`,`Saving makes a new version of the profile with the note “library to ${r}”.`],`Save the new version`))return;let o=await M(()=>k.put(`profiles`,e,i,`library to ${r}`,n.version));this.msg=`${e}: library ${r}, version ${o.version}`,this.onChanged()}catch(e){this.error=n(e).message}finally{this.busy=null}}render(){let e=this.catalog;if(!e||!this.usage)return m(5);let n=this.usage.library,r=Object.values(e.packages).filter(e=>e.category===`library`).length,i=e=>n.rows.filter(t=>t.profile===e&&t.chosen.length).length,a=Object.keys(e.library.profiles).sort();return t`
      <section class="panel" data-lib-profiles>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Retrieval per profile <span class="muted">· ${r} library skills</span></h2>
          <p class="panel-sub">On a new user message, and on a low-quality or stuck outcome, the router searches the library with the message, asks the package question among the best matches and, with <strong>on</strong>, puts the skill that fits at the prompt's tail. <strong>shadow</strong> only journals the pick. Judge the picks below before a profile goes on.</p></div></header>
        ${this.error?t`<p class="err-ink small" role="alert" data-lib-error>${this.error}</p>`:f}
        ${this.msg?t`<p class="status small" role="status" data-lib-saved>${_(`check`)} ${this.msg}</p>`:f}
        <div class="lib-profiles">
          ${a.map(n=>{let r=this.tiles.find(e=>e.name===n),a=e.library.profiles[n];return t`<div class="lib-profile" data-lib-profile=${n}>
              <span class="lib-profile-name"><strong>${n}</strong>${r?.locked?t` <span class="badge badge-lock">locked</span>`:f}<span class="muted small">${r?.clients.join(`, `)||`no client`}${r?` · profile ${r.mode}`:``}</span></span>
              ${A([`off`,`shadow`,`on`],a,e=>void this.setMode(n,e),{small:!0,label:`library of ${n}`,disabled:!!r?.locked||this.busy===n})}
              <span class="muted small num">${i(n)} pick${i(n)===1?``:`s`} in the rows below</span>
            </div>`})}
        </div>
      </section>
      <section class="panel" data-lib-shadow>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">What the harness picked</h2>
          <p class="panel-sub">The router's retrieval in the journal (extra.library), newest first: the candidates the search found, the skill the package question picked, and whether it went in (on) or would have (shadow). A pick opens the skill.</p></div>
          <div class="panel-tools"><label class="field"><span>profile</span><select data-lib-filter @change=${e=>this.profile=e.target.value}>
            <option value="" ?selected=${!this.profile}>all</option>${a.map(e=>t`<option value=${e} ?selected=${this.profile===e}>${e}</option>`)}
          </select></label></div></header>
        <div class="stats" data-lib-tiles>
          ${x({label:`messages searched`,value:u(n.searched.day),unit:`in 24 h`,sub:`${u(n.searched.week)} in 7 days`})}
          ${x({label:`with a pick`,value:u(n.picked.day),unit:`in 24 h`,sub:`${u(n.picked.week)} in 7 days`})}
          ${x({label:`pick rate`,value:n.searched.week?`${Math.round(100*n.picked.week/n.searched.week)}`:`—`,unit:n.searched.week?`%`:``,sub:`7 days`})}
          ${x({label:`most picked`,value:n.top[0]?.name??`—`,sub:n.top.length?n.top.slice(0,3).map(e=>`${e.name} ${e.picks}`).join(` · `):`no pick yet`})}
        </div>
        ${this.rows(It(n.rows,this.profile))}
      </section>`}rows(e){if(!e.length)return y(`No retrieval in the journal`,`A profile with the library in shadow or on journals each search here; with every profile off nothing is searched.`);let n=e=>S({ptab:null,pkg:e,pview:null});return t`<div class="lib-rows" data-lib-rows>${e.map(e=>t`<div class="lib-row" data-lib-row=${e.id}>
        <div class="lib-row-head">
          <span class="num small">${r(new Date(e.ts_ms).toISOString())}</span>
          <span class="badge ${e.mode===`on`?`badge-on`:`badge-shadow`}">${e.mode??`—`}</span>
          ${E(e.class_id,this.ids)}
          <span class="muted small">${e.profile??`—`} · ${e.trigger??`—`}${e.query_chars==null?``:` · ${u(e.query_chars)} characters searched`}${e.cached?` · asked before`:``}</span>
        </div>
        <div class="lib-cands">${e.candidates.length?e.candidates.map(n=>t`<span class="lib-cand ${n.name&&e.chosen.includes(n.name)?`is-chosen`:``}">${n.name}<span class="num muted">${n.score==null?``:n.score.toFixed(2)}</span></span>`):t`<span class="muted small">no candidate shares a word with the message</span>`}</div>
        <div class="lib-pick small">${e.chosen.length?t`${e.mode===`on`&&e.applied?.length?`went in:`:`would have gone in:`} ${e.chosen.map(e=>t`<button class="btn btn-sm lib-open" data-lib-open=${e} @click=${()=>n(e)}>${e}</button>`)} <span class="muted">(${e.how??`—`})</span>`:t`<span class="muted">no pick (${(e.how??`none`).replace(/_/g,` `)})</span>`}</div>
      </div>`)}</div>`}};l([o({attribute:!1})],G.prototype,`ids`,void 0),l([o({attribute:!1})],G.prototype,`tiles`,void 0),l([o({attribute:!1})],G.prototype,`catalog`,void 0),l([o({attribute:!1})],G.prototype,`usage`,void 0),l([o({attribute:!1})],G.prototype,`onChanged`,void 0),l([a()],G.prototype,`profile`,void 0),l([a()],G.prototype,`busy`,void 0),l([a()],G.prototype,`msg`,void 0),l([a()],G.prototype,`error`,void 0),G=l([s(`qse-cond-library`)],G);var Rt=[{id:`text`,label:`Text`},{id:`edit`,label:`Edit`},{id:`classes`,label:`Classes`},{id:`usage`,label:`Usage`}],K=class extends d{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.selected=null,this.tab=null,this.pane=null,this.catalog=null,this.catError=null,this.usage=null,this.filters=bt,this.hits=null,this.libError=null,this.item=null,this.itemError=null,this.recent=null,this.raw=!1,this.editView=`write`,this.counted=null,this.review=null,this.attachError=null,this.found=null,this.scanError=null,this.pk=new C(this,`packages`),this.rt=new C(this,`router`),this.seq=0,this.loadedFor=null,this.countTimer=null,this.countFor=null}connectedCallback(){super.connectedCallback(),this.pk.onSaved=()=>void this.refresh(),this.rt.onSaved=()=>void this.refresh(),this.pk.load(),this.rt.load(),this.refresh()}disconnectedCallback(){this.countTimer&&clearTimeout(this.countTimer),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.item=null,this.itemError=null,this.recent=null,this.attachError=null,this.selected&&this.loadItem(this.selected))}updated(e){let t=this.selected?this.pk.draft?.packages[this.selected]:null;if(!t||t.source!==`inline`)return;let n=W(t.body,!!t.library)[0];this.counted?.text!==n&&this.countFor!==n&&(this.countFor=n,this.countTimer&&clearTimeout(this.countTimer),this.countTimer=setTimeout(()=>{k.packagesTokens([n]).then(e=>this.counted={text:n,n:e.tokens[0],how:e.tokenizer}).catch(()=>void 0)},400))}async refresh(){try{let[e,t]=await Promise.all([k.packagesCatalog(),k.packagesUsage(null,1)]);[this.catalog,this.usage,this.catError]=[e,t,null]}catch(e){this.catError=n(e).message}this.selected&&await this.loadItem(this.selected)}async loadItem(e){try{let[t,n]=await Promise.all([this.pk.base?.doc.packages[e]||!this.pk.base?k.packageItem(e):Promise.resolve(null),k.packagesUsage(e,20)]);if(e!==this.selected)return;[this.item,this.recent,this.itemError]=[t,n.recent,null]}catch(t){e===this.selected&&(this.itemError=n(t).message)}}async search(e){this.filters={...this.filters,q:e};let t=++this.seq;if(!e.trim()||!Object.values(this.pk.draft?.packages??{}).some(e=>e.library)){[this.hits,this.libError]=[null,null];return}try{let n=await k.librarySearch(e.trim(),20);t===this.seq&&([this.hits,this.libError]=[n,null])}catch(e){t===this.seq&&([this.hits,this.libError]=[[],`the library's search: ${n(e).message}`])}}open(e,t=null){S({pkg:e,pview:t})}upd(e,t){this.pk.draft&&this.pk.set(w(this.pk.draft,e,t))}toggleAttach(e,t,n){let r=this.rt.draft;if(!r)return;if(!n){this.attachError=null,this.rt.set(jt(r,e,t));return}let i=At(r,e,t,this.pk.draft);this.attachError=i.error,i.error||this.rt.set(i.router)}addInline(){let e=this.pk.draft,t=Nt(e);this.pk.set(w(e,[`packages`,t],Pt(Number(t.slice(8))))),this.open(t,`edit`)}copyInline(e){let t=this.pk.draft,n=this.item;n&&n.name===e&&n.found&&(this.pk.set(w(t,[`packages`,e],Ft(t.packages[e],n.body,n.reference))),this.open(e,`edit`))}removePackage(e){let t=this.pk.draft,n={...t.packages};delete n[e],this.pk.set({...t,packages:n}),this.open(null)}async scan(){this.scanError=null;try{this.found=(await k.scan()).skills}catch(e){this.scanError=n(e).message}}register(e){let t={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.pk.set(w(this.pk.draft,[`packages`,e.name],t)),this.open(e.name)}async startReview(e,t){await e.check(),this.review=t}async confirmSave(e){await e.save()&&(this.review=null)}bar(e,n,r){let i=e.base?.meta,a=!!e.draft&&!!e.base&&JSON.stringify(e.draft)!==JSON.stringify(e.base.doc);return t`
      ${e.showVersions?rt(e,r):f}
      ${this.review===n?t`<section class="panel pk-review" data-review=${n}>
            <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e.errors.length?t`${_(`warn`)} The server refuses ${r}`:t`${_(`check`)} The server accepts ${r}`}</h2>
              <p class="panel-sub">POST /v1/conductor/${n}/check against live v${i?.version??0}: ${e.changes.length} change${e.changes.length===1?``:`s`}. Saving makes a new version with your note; the next request uses it.</p></div></header>
            ${O(e)}
            ${Le(e.changes)}
            <div class="toolbar pk-review-tools">
              <button class="btn btn-primary" data-review-save=${n} ?disabled=${e.errors.length>0||!e.note.trim()||e.saving||!e.changes.length} @click=${()=>this.confirmSave(e)}>${e.saving?`Saving…`:`Save as a new version`}</button>
              <button class="btn btn-ghost" @click=${()=>this.review=null}>Back to editing</button>
              ${e.note.trim()?f:t`<span class="muted small">Write the change note first.</span>`}
            </div>
          </section>`:f}
      <div class="savebar ${a||this.review===n?``:`is-idle`}" data-savebar=${n}>
        <span class="status num" title=${i?.note??``}>${r} · live v${i?.version??0}${a?t` · <strong>${e.changes.length||`…`} change${e.changes.length===1?``:`s`}</strong>`:f}</span>
        <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>{e.note=t.target.value,this.requestUpdate()}} aria-label=${`change note for ${r}`} maxlength="200" data-note=${n} />
        <button class="btn btn-primary" ?disabled=${!a||e.saving} @click=${()=>this.startReview(e,n)} data-check-save=${n}>Check and save…</button>
        <button class="btn btn-ghost" ?disabled=${!a} @click=${()=>{e.reset(),this.review=null}}>Reset</button>
        <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():this.requestUpdate()}} data-versions-btn=${n}>Versions</button>
        ${e.saved?t`<span class="status" role="status" data-saved=${n}>${_(`check`)} ${e.saved}</span>`:f}
        ${e.saveError?t`<span class="err-ink small" role="alert">${e.saveError}</span>`:f}
      </div>`}render(){let e=t`<nav class="mem-tabs pk-tabs-top" aria-label="packages tabs">
      ${[{id:null,label:`Packages and skills`},{id:`library`,label:`Library retrieval`}].map(e=>t`<a class="cond-tab ${this.tab===`library`==(e.id===`library`)?`is-on`:``}" href="#" data-pk-top=${e.id??`all`} aria-current=${this.tab===`library`==(e.id===`library`)?`page`:f} @click=${t=>{t.preventDefault(),S({ptab:e.id,pkg:null,pview:null})}}>${e.label}</a>`)}
    </nav>`;if(this.tab===`library`)return t`${e}<qse-cond-library .ids=${this.ids} .tiles=${this.tiles} .catalog=${this.catalog} .usage=${this.usage} .onChanged=${()=>this.refresh()}></qse-cond-library>`;let n=this.pk;if(n.loading&&!n.draft)return t`${e}${m(6)}`;if(n.loadError)return t`${e}<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${n.loadError}</div></div>`;let r=n.draft,i=wt(r,this.catalog,this.usage?.packages??null,this.rt.draft),a=Ot(i,this.filters,this.hits,this.catalog),o=this.selected&&r.packages[this.selected]?this.selected:null;return t`
      ${e}
      ${this.catError?t`<div class="guard-line" role="alert">${_(`warn`)} ${this.catError}</div>`:f}
      <div class="pk-layout ${o?`has-detail`:``}">
        ${this.renderList(a,i)}
        ${o?this.renderDetail(o,r.packages[o],i.find(e=>e.name===o)):this.selected?t`<section class="panel" data-pk-gone><p class="muted">No package ${this.selected} in this version.</p><button class="btn btn-sm" @click=${()=>this.open(null)}>Back to the list</button></section>`:f}
      </div>
      ${this.renderFound(r)}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${r.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${this.rt.draft&&(this.rt.showVersions||this.rt.saved||this.review===`router`||JSON.stringify(this.rt.draft)!==JSON.stringify(this.rt.base?.doc))?this.bar(this.rt,`router`,`the router`):f}
      ${this.bar(this.pk,`packages`,`the packages`)}
    `}renderList(e,n){let r=this.filters,i=n.filter(e=>Dt(e,this.catalog)).length,a=e=>this.filters={...this.filters,...e};return t`<section class="panel pk-list" data-pk-list>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills <span class="muted" data-pk-count-all>· ${e.length} of ${n.length}</span></h2>
        <p class="panel-sub">What the router may put into a prompt. Tokens are the text that goes in, ${this.catalog?.tokenizer===`estimate`?`estimated at 3.5 characters a token (no tokenizer on this server)`:`on the engine's tokenizer`}${i?t`; <span class="warn-ink" data-pk-over-count>${i} over the limit</span>`:f}.</p></div>
        <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-pk-new>${_(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${_(`search`)} Skills on the box</button></div></header>
      <div class="pk-filters">
        <label class="field field-search pk-search">${_(`search`)}<input type="search" data-pk-search placeholder="search packages, or the library's text" aria-label="search packages and skills" .value=${r.q} @input=${e=>void this.search(e.target.value)} /></label>
        <label class="field"><span>kind</span><select data-pk-filter="kind" @change=${e=>a({category:e.target.value})}>
          <option value="all" ?selected=${r.category===`all`}>all</option>
          ${V.map(e=>t`<option value=${e} ?selected=${r.category===e}>${H[e]}</option>`)}
        </select></label>
        <label class="field"><span>class</span><select data-pk-filter="class" @change=${e=>a({cls:e.target.value})}>
          <option value="" ?selected=${!r.cls}>any</option>
          ${this.ids.map(e=>t`<option value=${e} ?selected=${r.cls===e}>${e}</option>`)}
        </select></label>
        <label class="field"><span>enabled</span><select data-pk-filter="enabled" @change=${e=>a({enabled:e.target.value})}>
          ${[`all`,`on`,`off`].map(e=>t`<option value=${e} ?selected=${r.enabled===e}>${e}</option>`)}
        </select></label>
        <label class="switch"><input type="checkbox" data-pk-filter="over" .checked=${r.over} @change=${e=>a({over:e.target.checked})} /> over the limit</label>
      </div>
      ${this.libError?t`<p class="err-ink small">${this.libError}</p>`:f}
      <div class="pk-head" aria-hidden="true"><span>package</span><span>tokens / limit</span><span>used 24 h · 7 d</span></div>
      <div class="pk-rows" role="list">${e.map(e=>this.row(e))}</div>
      ${e.length?f:t`<p class="muted small" data-pk-empty>${r.q.trim()?`Nothing matches “${r.q.trim()}”${this.hits?`, and the library shares no word with it`:``}.`:`No package matches these filters.`}</p>`}
    </section>`}row(e){let n=St(e.name,e.pkg,e.category,this.catalog),r=e.entry?e.entry.tokens:U(W(e.pkg.body,!!e.pkg.library)[0]),i=Dt(e,this.catalog),a=e.usage,o=this.selected===e.name;return t`<button class="pk-row ${o?`is-open`:``} ${e.pkg.enabled?``:`is-off`}" role="listitem" data-pk-row=${e.name} data-category=${e.category} aria-current=${o?`true`:f} @click=${()=>this.open(e.name)}>
      <span class="pk-main">
        <span class="pk-name"><strong>${e.name}</strong>
          <span class="badge pk-kind pk-kind-${e.category}">${H[e.category]}</span>
          ${e.pkg.general?t`<span class="badge badge-on" title="goes in whole and first on every turn of a class that lists it">general</span>`:f}
          ${e.pkg.enabled?f:t`<span class="badge">off</span>`}
          ${e.entry?f:t`<span class="badge badge-shadow">not saved</span>`}
          ${e.entry&&!e.entry.available&&e.entry.error?t`<span class="badge badge-warn" title=${e.entry.path??``}>${e.entry.error.replace(/_/g,` `)}</span>`:f}
          ${e.score==null?f:t`<span class="pk-score num" data-score title="the library's BM25 score">${e.score.toFixed(2)}</span>`}
        </span>
        <span class="pk-desc muted small">${e.pkg.description}</span>
        <span class="pk-meta"><span class="mono small muted">${e.pkg.source}</span>${e.classes.map(e=>E(e,this.ids))}${e.category===`library`&&e.entry?.fits?.length?t`<span class="muted small">fits</span>${e.entry.fits.map(e=>E(e,this.ids))}`:f}</span>
      </span>
      <span class="pk-tokens num ${i?`is-over`:``}" data-pk-tokens title=${e.entry?`${e.entry.tokens} tokens go in; the whole text is ${e.entry.body_tokens}`:`estimated: not saved yet`}>
        <span>${e.entry?``:`≈`}${u(r)} <span class="muted">/ ${u(n.limit)}</span></span>
        <span class="pk-bar"><span style="width:${Math.min(100,100*r/Math.max(1,n.limit))}%"></span></span>
        ${i?t`<span class="badge badge-warn">over</span>`:f}
      </span>
      <span class="pk-usage num small" data-pk-usage>
        ${a&&(a.injected.week||a.shadow.week||a.pick.week)?t`${a.injected.week?t`<span>${a.injected.day} · ${a.injected.week} in</span>`:f}${a.shadow.week?t`<span class="muted">${a.shadow.day} · ${a.shadow.week} shadow</span>`:f}${a.pick.week?t`<span>${a.pick.day} · ${a.pick.week} picks</span>`:f}`:t`<span class="muted">—</span>`}
      </span>
    </button>`}renderFound(e){return this.scanError?t`<p class="err-ink small">${this.scanError}</p>`:this.found?t`<section class="panel" style="margin-top:16px" data-found>
      <header class="panel-head"><h2 class="panel-title">Skills in ${e.skill_dirs.join(`, `)}</h2><div class="panel-tools"><button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.found=null}>${_(`x`)}</button></div></header>
      ${this.found.length?t`<div class="table-scroll"><table class="table"><tbody>${this.found.map(n=>t`<tr><td><strong>${n.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${n.description}</div></td><td class="num">${u(n.chars)} chars</td>
              <td>${n.registered||e.packages[n.name]?t`<span class="badge badge-ok">registered</span>`:t`<button class="btn btn-sm" @click=${()=>this.register(n)} data-register=${n.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:t`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ spark:~/.config/opencode/skills/</code>).</p>`}
    </section>`:f}renderDetail(e,n,r){let i=r.entry,a=xt(n,i),o=St(e,n,a,this.catalog),s=Rt.some(e=>e.id===this.pane)?this.pane:`text`,c=n.source===`inline`,l=r.classes,d=this.tokensNow(e,n,i),p=d.n>o.limit||!!i?.cut&&!(c&&this.dirtyBody(e));return t`<section class="panel pk-detail" data-pk-detail=${e}>
      <header class="pk-detail-head">
        <div class="pk-title">
          <h2 class="panel-title">${e}</h2>
          <span class="muted">${n.title}</span>
        </div>
        <button class="btn btn-ghost btn-sm pk-close" @click=${()=>this.open(null)} data-pk-close aria-label="Close the package">${_(`x`)}<span class="pk-close-word">All packages</span></button>
      </header>
      <div class="pk-badges">
        <span class="badge pk-kind pk-kind-${a}" data-pk-category>${H[a]}</span>
        <span class="badge">${c?`inline`:n.source.startsWith(`skill:`)?`skill on the box`:`file on the box`}</span>
        ${n.general?t`<span class="badge badge-on">general</span>`:f}
        ${n.enabled?t`<span class="badge badge-ok">enabled</span>`:t`<span class="badge">disabled</span>`}
        ${i?f:t`<span class="badge badge-shadow">not saved</span>`}
        ${i&&!i.available&&i.error?t`<span class="badge badge-warn">${i.error.replace(/_/g,` `)}</span>`:f}
      </div>
      <div class="toolbar pk-switches">
        <label class="switch"><input type="checkbox" data-pk-enabled .checked=${n.enabled} @change=${t=>this.upd([`packages`,e,`enabled`],t.target.checked)} /> enabled</label>
        ${a===`library`?f:t`<label class="switch" title="goes in whole and first on every turn of a class that lists it (K1)"><input type="checkbox" data-pk-general .checked=${!!n.general} @change=${t=>this.upd([`packages`,e,`general`],t.target.checked)} /> general</label>`}
        ${c?f:t`<button class="btn btn-sm" data-pk-copy-inline ?disabled=${!this.item?.found} title=${this.item?.found?`an inline copy of the file overrides it`:`the file is not on the box`} @click=${()=>this.copyInline(e)}>${_(`copy`)} Copy to inline</button>`}
        <button class="btn btn-sm btn-ghost" data-pk-remove ?disabled=${l.length>0} title=${l.length?`${l.join(`, `)} list it; detach it first`:`remove it from the packages document`} @click=${()=>this.removePackage(e)}>${_(`trash`)} Remove</button>
      </div>
      <div class="pk-meter ${p?`is-over`:``}" data-pk-meter>
        <div class="pk-bar"><span style="width:${Math.min(100,100*d.n/Math.max(1,o.limit))}%"></span></div>
        <span class="num small" data-pk-count>${d.approx?`≈ `:``}${u(d.n)} tokens of ${u(o.limit)} <span class="muted">(${d.why}; limit: ${o.source===`house`?a===`library`?`the house rule for a library skill`:`the house rule for the library card`:`max_tokens, where the server cuts`})</span></span>
        ${p?t`<span class="badge badge-warn" data-pk-limit-warn>over the limit</span>`:f}
      </div>
      <nav class="mem-tabs pk-panes" aria-label="package detail">
        ${Rt.map(e=>t`<a class="cond-tab ${e.id===s?`is-on`:``}" href="#" data-pk-pane=${e.id} aria-current=${e.id===s?`page`:f} @click=${t=>{t.preventDefault(),S({pview:e.id===`text`?null:e.id})}}>${e.label}</a>`)}
      </nav>
      <div class="pk-pane" data-pk-pane-body=${s}>
        ${s===`text`?this.paneText(e,n,i):s===`edit`?this.paneEdit(e,n,a,o):s===`classes`?this.paneClasses(e,n,a):this.paneUsage(e,a)}
      </div>
    </section>`}dirtyBody(e){return this.pk.draft?.packages[e]?.body!==this.pk.base?.doc.packages[e]?.body}tokensNow(e,t,n){if(t.source===`inline`&&(this.dirtyBody(e)||!n)){let e=W(t.body,!!t.library)[0];return this.counted?.text===e?{n:this.counted.n,approx:this.counted.how===`estimate`,why:this.counted.how===`engine`?`the draft, on the engine's tokenizer`:`the draft, estimated`}:{n:U(e),approx:!0,why:`the draft, estimated until the engine counts it`}}return n?{n:n.body_tokens,approx:this.catalog?.tokenizer===`estimate`,why:n.cut?`the whole text; ${u(n.tokens)} go in after the cut`:this.catalog?.tokenizer===`estimate`?`estimated`:`on the engine's tokenizer`}:{n:0,approx:!0,why:`not counted yet`}}md(e){return e?this.raw?t`<pre class="note-text pk-raw" data-pk-raw>${e}</pre>`:t`<div class="pg-md pk-md" data-pk-md>${st(ct(e))}</div>`:t`<p class="muted small">(empty)</p>`}paneText(e,n,r){let i=n.source===`inline`,a=this.item?.name===e?this.item:null;if(!i&&!a)return this.itemError?t`<p class="err-ink small">${this.itemError}</p>`:m(4);let[o,s]=i?W(n.body,!!n.library):[a.body,a.reference],c=i?{}:a.frontmatter;return t`
      <div class="toolbar pk-text-tools">${A([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`text view`})}
        ${!i&&a?.path?t`<span class="mono small muted pk-path" data-pk-path>${a.path}</span>`:f}</div>
      ${!i&&a&&!a.found?t`<p class="note" data-pk-missing>${_(`warn`)} The file is not on the box (<code>${a.path??n.source}</code>): nothing goes in.</p>`:f}
      ${Object.keys(c).length?t`<h3 class="pk-sub">Frontmatter <span class="muted small">the client's metadata; never injected</span></h3>
            ${this.raw?t`<pre class="note-text pk-raw" data-pk-front>${a.frontmatter_raw}</pre>`:t`<dl class="kv pk-front" data-pk-front>${Object.entries(c).map(([e,n])=>t`<div><dt>${e}</dt><dd>${n}</dd></div>`)}</dl>`}`:f}
      <h3 class="pk-sub">The text that goes in ${o?re(o,`Copy the text`):f}</h3>
      ${this.md(o)}
      ${r?.cut&&!this.dirtyBody(e)?t`<p class="note" data-pk-cut>${_(`warn`)} The server cuts this text at ${u(n.max_tokens)} tokens × 3.5 characters, at a line end, and adds “[... cut to the package's max_tokens]”: ${u(r.tokens)} tokens go in of ${u(r.body_tokens)}.</p>`:f}
      ${s?t`<details class="sec pk-ref" data-pk-ref>
            <summary>Reference <span class="sec-sub">${u(a?.tokens.reference??U(s))} tokens; only a read gets it (GET /v1/library/${e})</span></summary>
            <div class="sec-body">${this.md(s)}</div>
          </details>`:f}`}paneEdit(e,n,r,i){let a=n.source===`inline`;return t`
      <div class="form-grid">
        <label class="field"><span>title</span><input data-pk-title .value=${n.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>max tokens</span><input type="number" min="16" max="32768" data-pk-max .value=${String(n.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">the server cuts a longer text at a line end</span></label>
        ${a?f:t`<label class="field field-wide"><span>source</span><input class="mono" .value=${n.source} readonly /><span class="hint">${this.item?.path??`a file on the box`}</span></label>`}
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" data-pk-description .value=${n.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${a?this.editor(e,n,r,i):t`<div class="note pk-readonly" data-pk-readonly>${_(`info`)} The text is the file on the box${this.item?.path?t` at <code>${this.item.path}</code>`:f}, read when first needed and again when it changes. Editing files on the box is not done here. <button class="btn btn-sm" ?disabled=${!this.item?.found} @click=${()=>this.copyInline(e)} data-pk-copy-inline-edit>Copy to inline</button> makes an inline copy of its text that overrides the file.</div>`}`}editor(e,n,r,i){let[a,o]=W(n.body,!!n.library),s=this.tokensNow(e,n,this.catalog?.packages[e]??null),c=s.n>i.limit,l=this.editView;return t`<div class="pk-editor pk-view-${l}" data-pk-editor>
      <div class="toolbar pk-editor-tools">
        ${A([`write`,`preview`,`split`],l,e=>this.editView=e,{small:!0,label:`editor view`})}
        <span class="pk-count num small ${c?`is-over`:``}" data-pk-editor-count>${s.approx?`≈ `:``}${u(s.n)} of ${u(i.limit)} tokens${n.library&&o?t` <span class="muted">· reference ≈ ${u(U(o))}</span>`:f}</span>
        ${c?t`<span class="badge badge-warn" data-pk-editor-warn>over ${i.source===`house`?`the ${H[r]}'s ${u(i.limit)}`:`max_tokens: the server will cut it`}</span>`:f}
      </div>
      <div class="pk-editor-panes">
        ${l===`preview`?f:t`<textarea class="code pk-body" rows="18" spellcheck="false" data-pk-body aria-label="the package's text (Markdown)" .value=${n.body} @input=${t=>this.upd([`packages`,e,`body`],t.target.value)}></textarea>`}
        ${l===`write`?f:t`<div class="pg-md pk-md pk-preview" data-pk-preview>${st(ct(a))}</div>`}
      </div>
      <p class="hint">${n.library?`A library skill: the text before “## Reference” goes in; the reference only on a read. `:``}${i.source===`house`?`The house rule keeps ${H[r]} texts at ${u(i.limit)} tokens or fewer.`:`The server cuts the text at max_tokens × 3.5 characters, at a line end.`} Markdown; the preview is the dashboard's renderer, the model gets the text as written.</p>
    </div>`}paneClasses(e,n,r){let i=this.catalog?.packages[e]??null;if(r===`library`)return t`<p class="muted small" data-pk-library-classes>A library skill is searched, not listed by a class: on a new user message the router searches the library and keeps the skills INDEX.md fits to the turn's class.</p>
        <div class="pk-meta">${i?.fits?.length?t`<span class="muted small">INDEX.md fits</span>${i.fits.map(e=>E(e,this.ids))}`:t`<span class="muted small">INDEX.md does not list it: it stays in for every class.</span>`}</div>
        <p class="small"><a href="#/conductor?screen=packages&ptab=library">Each profile's retrieval switch and its picks</a></p>`;let a=this.rt.draft;return a?t`
      <p class="muted small">A class's packages go in by K1's order: the general ones whole and first, then the router asks the package question among the rest and puts in the one that fits. A class lists at most ${12}. Attaching edits the router document; it is saved with its own check below.</p>
      ${this.attachError?t`<p class="err-ink small" role="alert" data-pk-attach-error>${this.attachError}</p>`:f}
      <div class="pk-classes" data-pk-classes>
        ${a.classes.map(n=>{let r=n.policy.packages??[],i=r.includes(e);return t`<div class="pk-class ${i?`is-on`:``}" data-pk-class=${n.id}>
            <label class="switch"><input type="checkbox" .checked=${i} data-pk-attach=${n.id} aria-label=${`attach ${e} to ${n.id}`} @change=${t=>this.toggleAttach(n.id,e,t.target.checked)} /> ${E(n.id,this.ids,n.label)}</label>
            <span class="pk-order small" data-pk-order=${n.id}>${r.length?r.map((n,r)=>t`<span class="pk-order-item ${n===e?`is-me`:``}">${r+1}. ${n}${this.pk.draft?.packages[n]?.general?t` <span class="muted">general</span>`:f}</span>`):t`<span class="muted">no packages</span>`}</span>
            <span class="muted small num">${r.length}/${12}</span>
          </div>`})}
      </div>
      ${i&&Object.keys(i.profiles).length?t`<p class="small muted" data-pk-profiles>A profile's own routing also lists it: ${Object.entries(i.profiles).map(([e,t])=>`${e} (${t.join(`, `)})`).join(`; `)}. A profile's routing replaces the router's policy for that class (the Profiles screen).</p>`:f}
      <div class="toolbar"><button class="btn btn-sm btn-ghost" data-pk-router-versions @click=${()=>{this.rt.showVersions=!0,this.rt.loadVersions()}}>Router versions</button></div>`:m(3)}paneUsage(e,n){let i=this.usage?.packages[e]??null,a={day:0,week:0},o=(e,t)=>x({label:e,value:u(t.day),unit:`in 24 h`,sub:`${u(t.week)} in 7 days`}),s=this.item?.name===e?this.item:null,c=n===`library`||(i?.pick.week??0)>0;return t`
      <div class="stats pk-stats" style="--pk-tiles:${c?3:2}" data-pk-usage-tiles>
        ${o(`injected`,i?.injected??a)}${o(`shadow (would have)`,i?.shadow??a)}${c?o(`library picks`,i?.pick??a):f}
      </div>
      <h3 class="pk-sub">Recent rows <span class="muted small">the journal, newest first (7 days)</span></h3>
      ${this.recent==null?m(3):this.recent.length?t`<div class="pk-uses" data-pk-recent>${this.recent.map(e=>t`<div class="pk-use" data-pk-use=${e.id}>
                <span class="num small">${r(new Date(e.ts_ms).toISOString())}</span>
                <span class="badge ${e.how===`injected`?`badge-on`:e.how===`shadow`?`badge-shadow`:`badge-warn`}">${e.how}</span>
                ${E(e.class_id,this.ids)}
                <span class="muted small">${e.profile??`—`}${e.client_kind?` · ${e.client_kind}`:``}</span>
                ${e.session_key?t`<a class="mono small pk-session" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${e.session_key.length>28?`${e.session_key.slice(0,28)}…`:e.session_key}</a>`:f}
              </div>`)}</div>`:t`<p class="muted small" data-pk-recent-empty>No router row names it in the last 7 days.</p>`}
      ${n===`library`?t`<h3 class="pk-sub">Neighbours <span class="muted small">the library's BM25 on this skill's name, description and text</span></h3>
            ${s?.neighbours?s.neighbours.length?t`<div class="pk-neighbours" data-pk-neighbours>${s.neighbours.map(e=>t`<button class="pk-neighbour" data-pk-neighbour=${e.name} @click=${()=>this.open(e.name)}><strong>${e.name}</strong> <span class="num muted small">${e.score.toFixed(2)}</span><span class="muted small pk-desc">${e.description}</span></button>`)}</div>`:t`<p class="muted small">No other skill shares a word with it.</p>`:m(2)}`:f}`}};l([o({attribute:!1})],K.prototype,`ids`,void 0),l([o({attribute:!1})],K.prototype,`tiles`,void 0),l([o({attribute:!1})],K.prototype,`selected`,void 0),l([o({attribute:!1})],K.prototype,`tab`,void 0),l([o({attribute:!1})],K.prototype,`pane`,void 0),l([a()],K.prototype,`catalog`,void 0),l([a()],K.prototype,`catError`,void 0),l([a()],K.prototype,`usage`,void 0),l([a()],K.prototype,`filters`,void 0),l([a()],K.prototype,`hits`,void 0),l([a()],K.prototype,`libError`,void 0),l([a()],K.prototype,`item`,void 0),l([a()],K.prototype,`itemError`,void 0),l([a()],K.prototype,`recent`,void 0),l([a()],K.prototype,`raw`,void 0),l([a()],K.prototype,`editView`,void 0),l([a()],K.prototype,`counted`,void 0),l([a()],K.prototype,`review`,void 0),l([a()],K.prototype,`attachError`,void 0),l([a()],K.prototype,`found`,void 0),l([a()],K.prototype,`scanError`,void 0),K=l([s(`qse-cond-packages`)],K);var zt={not_loaded:`not loaded`,loading:`loading the weights`,warming:`warming up`,loaded:`loaded`,unloading:`unloading`,failed:`failed`};function Bt(e){let t=e.role;return t?[t.kind,t.window.replace(`_`,` `),t.preemptible?`preemptible`:`not preemptible`,`${t.latency_budget_ms} ms then ${t.fallback}`,`writes ${t.writes.replace(/_/g,` `)}`,`shares ${t.sharing.join(`, `)||`nothing`}`,t.memory_claim_gib==null?`no claim`:`claims ${b(t.memory_claim_gib)}`].join(` · `):null}function Vt(e){return e===`loaded`?`badge-ok`:e===`failed`?`badge-warn`:e===`loading`||e===`warming`||e===`unloading`?`badge-shadow`:``}function Ht(e){if(!e||e.allowed)return null;let t=e.reason?ie[e.reason]??e.reason:`refused`;if(e.reason!==`refused_memory`)return t;let n=[`MemAvailable ${b(e.mem_available_gib)}`];return e.engine_growth_gib&&n.push(`engine growth ${b(e.engine_growth_gib)}`),n.push(`claim ${b(e.claim_gib)}`),e.after_gib!=null&&n.push(`${b(e.after_gib)} left`,`${b(e.need_gib)} needed`),`${t} (${n.join(`, `)})`}function Ut(e){return e?e.allowed?`fits, ${b(e.after_gib)} left after the claim`:Ht(e)??`refused`:`—`}function Wt(e){let t=Object.entries(e).sort((e,t)=>t[1]-e[1]);return t.length?t.map(([e,t])=>`${e} ${t}`).join(` · `):`no call yet`}function Gt(e){let t=e.limits;return[t.context_tokens,t.question_tokens,t.max_options].map(e=>e==null?`—`:String(e)).join(` / `)}function Kt(e,t){let n=e.trim();if(!n)return`Write the question.`;let r=t.split(/[,\n]/).map(e=>e.trim()).filter(Boolean);return r.length<2?`Give at least two options.`:r.length>20?`At most 20 options (the model card’s limit).`:new Set(r).size===r.length?{type:`choice`,instructions:n,criteria:Object.fromEntries(r.map(e=>[e,null]))}:`Two options have the same name.`}function qt(e){return e.type===`noul`?[[`true`,e.noul??0],[`false`,1-(e.noul??0)]]:Object.entries(e.probabilities??{}).sort((e,t)=>t[1]-e[1])}var Jt=[`on_start`,`on_demand`,`off`],Yt={mode:`on`,rate_per_min:240,max_texts:16,max_chars:1e5,deadline_ms:15e3,threshold:.8,drive_rate_per_s:10,drive_sessions:4,drive_idle_s:30,drive_max_texts:36e3},q=class extends d{constructor(...e){super(...e),this.st=null,this.error=null,this.busy=null,this.said={},this.tState=`The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.`,this.tQuestion=`What kind of work does this request ask for?`,this.tOptions=`debug, code_edit, plan, chat`,this.tModel=`auto`,this.tOut=null,this.tErr=null,this.asking=!1,this.ed=new C(this,`deciders`),this.poller=null}connectedCallback(){super.connectedCallback(),this.ed.load(),this.poller=h(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{this.st=await k.deciders(),this.error=null}catch(e){let t=n(e);this.error=`${t.endpoint}: ${t.message}`}}async act(e,t){if(t!==`unload`||await D(`Unload ${e.name}?`,[`Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).`,`Loading it again goes through the residency planner.`],`Unload`)){this.busy=e.name,this.said={...this.said,[e.name]:``};try{let n=await M(()=>t===`load`?k.deciderLoad(e.name):k.deciderUnload(e.name));this.said={...this.said,[e.name]:t===`load`?`load started (${n.state??`loading`})`:`unloaded`}}catch(r){let i=r instanceof ee&&r.status===409&&t===`load`?`refused by the residency planner: ${r.message}`:n(r).message;this.said={...this.said,[e.name]:i}}finally{this.busy=null,await this.load()}}}setLoad(e,t){let n=this.ed.draft;if(!n)return;let r=n.models.findIndex(t=>t.name===e),i=r>=0?n.models.map((e,n)=>n===r?{...e,load:t}:e):[...n.models,{name:e,load:t}];this.ed.set({...n,models:i})}async ask(){let e=Kt(this.tQuestion,this.tOptions);if(typeof e==`string`){this.tErr=e;return}this.asking=!0,this.tErr=null;try{this.tOut=await k.deciderTest(this.tState,{q:e},this.tModel)}catch(e){this.tOut=null,this.tErr=n(e).message}finally{this.asking=!1}}render(){let e=this.st;if(this.error&&!e)return t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${this.error}</div></div>`;if(!e)return m(6);if(!e.enabled)return g(`Decision models`,t`<div class="state" data-deciders-off>
          <div class="state-title">${_(`warn`)} This engine serves no decision model</div>
          <p class="muted small">${e.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port behind the admin token, and this screen loads, unloads and tests them.</p>
        </div>`,{id:`deciders-off`});let n=e.models.filter(e=>e.state===`loaded`);return t`
      <div class="stats" data-decider-tiles>
        ${x({label:`loaded`,value:`${n.length} / ${e.models.length}`,sub:n.length?n.map(e=>e.name).join(`, `):`none: questions fail open`,warn:!n.length})}
        ${x({label:`on the device`,value:ne(n.reduce((e,t)=>e+t.device_bytes,0)),sub:`${e.device??`no device`} · own stream and memory pool`})}
        ${x({label:`a load now`,value:e.planner?.allowed?`fits`:`refused`,sub:Ut(e.planner),warn:!e.planner?.allowed})}
        ${x({label:`when they run`,value:`idle`,sub:`only while no request holds the engine (stage A)`})}
      </div>
      ${this.renderModels(e)}
      ${this.renderMemory(e)}
      ${this.renderApi(e)}
      ${this.renderQuestions(e)}
      ${this.renderTest(e)}
    `}renderModels(n){let r=this.ed.draft,i=e=>r?.models.find(t=>t.name===e)?.load??`on_demand`;return g(`Models`,t`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${n.models.map(n=>{let r=Ht(n.refusal),a=this.said[n.name],o=n.state===`loaded`||n.state===`failed`||n.state===`warming`;return t`<tr data-decider-row=${n.name}>
              <td><strong>${n.name}</strong><div class="muted small">${n.family} · ${n.contract}${n.aliases.length?t` · ${n.aliases.join(`, `)}`:f}</div>${n.role?t`<div class="muted small" data-decider-role=${n.name} title="the role contract this model signs (SRV-133); the registry checks it before it serves the model">${Bt(n)}</div>`:f}</td>
              <td><span class="badge ${Vt(n.state)}" data-decider-state=${n.state}>${zt[n.state]}</span>${n.warm===!1?t` <span class="muted small">not warmed</span>`:f}</td>
              <td><select aria-label=${`load policy of ${n.name}`} data-decider-load=${n.name} @change=${e=>this.setLoad(n.name,e.target.value)}>
                ${Jt.map(e=>t`<option value=${e} ?selected=${i(n.name)===e}>${e.replace(`_`,` `)}</option>`)}
              </select></td>
              <td class="num">${Gt(n)}</td>
              <td class="small">${n.precision??t`<span class="muted">—</span>`}<div class="muted small mono" title=${n.weight_hash??``}>${n.weight_hash?n.weight_hash.slice(0,12):``}</div></td>
              <td class="num num-col">${u(n.questions)}</td>
              <td class="num num-col">${n.stats?.forward_ms_p50==null?`—`:`${e(n.stats.forward_ms_p50)} / ${e(n.stats.forward_ms_p95)}`}</td>
              <td class="num num-col ${n.stats?.errors?`err-ink`:``}">${n.stats?u(n.stats.errors+n.stats.deadline_missed):`—`}</td>
              <td>${o?t`<button class="btn btn-sm" ?disabled=${this.busy===n.name} @click=${()=>this.act(n,`unload`)} data-unload=${n.name}>Unload</button>`:t`<button class="btn btn-sm btn-primary" ?disabled=${this.busy===n.name||n.state===`loading`||n.state===`unloading`} @click=${()=>this.act(n,`load`)} data-load=${n.name}>Load</button>`}</td>
            </tr>
            ${r||n.error||a?t`<tr class="is-warn" data-decider-note=${n.name}><td colspan="9" class="small">
                  ${r?t`<span class="warn-ink" data-refusal=${n.name}>${_(`warn`)} ${r}</span> `:f}
                  ${n.error?t`<span class="err-ink">${n.error}</span> `:f}
                  ${a?t`<span role="status">${a}</span>`:f}
                </td></tr>`:f}`})}</tbody>
      </table></div>
      ${n.unserved?.length?t`<p class="muted small">In the document but not served by this engine: ${n.unserved.join(`, `)} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>`:f}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,{id:`deciders-models`,sub:`the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen`})}renderMemory(e){let n=e.planner,r=e.last_refusal;return g(`Memory`,t`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${b(n?.mem_available_gib)}</div></div>
          ${n?.engine_growth_gib==null?f:t`<div class="stat"><div class="stat-label">engine growth to its peak</div><div class="stat-value num">${b(n.engine_growth_gib)}</div></div>`}
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${b(n?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${b(n?.floor_gib)} + ${b(n?.margin_gib)}</div></div>
          <div class="stat ${n?.allowed?``:`is-warn`}"><div class="stat-label">a load now</div><div class="stat-value num small">${Ut(n)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${r?t`Last refused engine load: ${Ht(r)} <span class="muted">(${r.source??``})</span>`:t`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable, less what the engine may still take up to its declared peak, minus the claim stays at or above floor + margin, never during a hold, never beside another load. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,{id:`deciders-memory`})}renderApi(n){let r=this.ed.draft,i={...Yt,...r?.api??{}},a=n.api?.stats,o=(e,t)=>r&&this.ed.set(w(r,[`api`,e],t)),s=(e,n,r=1)=>t`<label class="field"><span>${n}</span><input type="number" step=${r} .value=${String(i[e])} data-api-field=${e} @change=${t=>o(e,r<1?Number(t.target.value):Math.round(Number(t.target.value)))} /></label>`;return g(`Laya for scripts`,r?t`<div data-decider-api>
            <div class="toolbar" style="margin:0"><span class="muted small">route</span>${A([`off`,`on`],i.mode,e=>o(`mode`,e),{small:!0,label:`Laya route mode`})}<code class="small">POST /v1/laya/ask</code></div>
            <div class="form-grid">
              ${s(`rate_per_min`,`texts a minute (every caller)`)}
              ${s(`max_texts`,`texts in one call`)}
              ${s(`deadline_ms`,`deadline (ms) while a request runs`)}
              ${s(`threshold`,`sure at or above`,.05)}
              ${s(`drive_rate_per_s`,`texts a second (a playtest session)`)}
              ${s(`drive_sessions`,`playtest sessions at once`)}
            </div>
            <p class="muted small" data-api-stats>${a?`${u(a.calls)} calls, ${u(a.texts)} texts, ${u(a.errors)} errors (${u(a.rate_limited)} over the rate) since the start${a.last_ms==null?``:`; the last took ${e(a.last_ms)}`}${a.drive_sessions?`; ${u(a.drive_sessions)} playtest sessions live`:``}`:`no calls yet`}</p>
            <p class="muted small">The model's own scripts ask Laya typed questions about their evidence (a log tail, an output, a state dump) through the provider, open like the chat API: a choice, a scale or a yes-no question; Laya reads about 1,100 characters of each text, its head and its tail. Every call is a journal row (<code>laya_api</code>). A playtest (Laya plays the build: <code>.tandem/laya_drive.py</code>) sends a <code>session</code>: its own rate a second, one <code>laya_drive</code> row when it ends.</p>
          </div>`:m(2),{id:`deciders-api`,sub:`the route the model’s scripts call: on or off, its rate and its limits`})}renderQuestions(e){let n=this.ed,r=n.draft,i=e.models.map(e=>e.name),a=r?Object.entries(r.questions):[];return t`${g(`Questions`,r?t`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${a.map(([a,o])=>{let s=e.questions.find(e=>e.id===a);return t`<tr data-question=${a}>
                  <td class="mono">${a}</td>
                  <td><select aria-label=${`model for ${a}`} data-question-model=${a} @change=${e=>n.set(w(r,[`questions`,a],e.target.value))}>
                    ${[`auto`,...i].map(e=>t`<option value=${e} ?selected=${e===o}>${e===`auto`?`auto (the role’s model; the router picks)`:e}</option>`)}
                    ${o!==`auto`&&!i.includes(o)?t`<option value=${o} selected>${o} (not served)</option>`:f}
                  </select></td>
                  <td class="small">${Wt(s?.picks??{})}</td>
                </tr>`})}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); <code>judge</code> is the test judge's questions on a run's evidence (the <code>tester</code> role), <code>laya_api</code> the model's own scripts through <code>/v1/laya/ask</code>; the session monitor's Tier 1 questions join this list when they land.</p>`:m(3),{id:`deciders-questions`,sub:`which model answers each question the conductor asks`})}
      ${O(n)}
      ${j(n,`the deciders document`)}`}renderTest(n){let r=this.tOut,i=r?.answers.q;return g(`Test box`,t`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${e=>this.tState=e.target.value} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${e=>this.tQuestion=e.target.value} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${e=>this.tOptions=e.target.value} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${e=>this.tModel=e.target.value} data-test-model>
            ${[`auto`,...n.models.map(e=>e.name)].map(e=>t`<option value=${e} ?selected=${e===this.tModel}>${e}</option>`)}
          </select></label>
        </div>
        <div class="savebar"><button class="btn btn-primary" ?disabled=${this.asking} @click=${()=>this.ask()} data-test-ask>${_(`send`)} ${this.asking?`Asking…`:`Ask`}</button>
          ${this.tErr?t`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>`:f}</div>
        ${r&&i?t`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${i.type===`choice`?i.choice:i.type===`noul`?(i.noul??0).toFixed(4):(i.score??0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${i.confidence.toFixed(2)} · ${r.model} · ${e(r.latency_ms)} · ${r.path} · ${u(r.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${qt(i).map(([e,n])=>t`<div class="prob-row" data-prob=${e}><span class="mono small">${e}</span><span class="conf-bar"><span style="width:${Math.round(n*100)}%"></span></span><span class="num small">${n.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${r.routing.reason}${r.routing.requested?` (wanted ${r.routing.requested})`:``}</p>
            </div>`:f}`,{id:`deciders-test`,sub:`one choice question, as /v1/systemone answers it (it runs in the idle window like any other)`})}};l([a()],q.prototype,`st`,void 0),l([a()],q.prototype,`error`,void 0),l([a()],q.prototype,`busy`,void 0),l([a()],q.prototype,`said`,void 0),l([a()],q.prototype,`tState`,void 0),l([a()],q.prototype,`tQuestion`,void 0),l([a()],q.prototype,`tOptions`,void 0),l([a()],q.prototype,`tModel`,void 0),l([a()],q.prototype,`tOut`,void 0),l([a()],q.prototype,`tErr`,void 0),l([a()],q.prototype,`asking`,void 0),q=l([s(`qse-cond-deciders`)],q);var J=class extends d{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=h(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await k.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let t=n(e);this.error=`${t.endpoint}: ${t.message}`}}async loadSession(e){try{let t=await k.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(t){e===this.selected&&(this.turns=[]),this.error=n(t).message}}render(){if(this.error&&!this.list)return t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${this.error}</div></div>`;if(!this.list)return m(6);let e=[...new Set(this.list.map(e=>e.client_kind??`other`))],n=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return t`
      ${this.selected?this.renderSession(this.selected):f}
      ${g(`Sessions, last 7 days`,n.length?t`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${n.map(e=>t`<tr class="req-row ${e.key===this.selected?`is-open`:``}" data-session=${e.key} @click=${()=>S({session:e.key})}>
                  <td class="mono small cell-wrap">${T(e.key,28)}</td><td>${e.client_kind??`—`}</td><td>${e.profile??`—`}</td>
                  <td>${E(e.class_id,this.ids)}</td><td class="num num-col">${u(e.turns)}</td>
                  <td class="num num-col">${e.notes?`${e.notes} (${u(e.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${e.acts?t`${u(e.acts)}${e.gave_ups?t` <span class="badge badge-warn">${e.gave_ups} gave up</span>`:f}`:`—`}</td>
                  <td class="num num-col" data-interventions>${u(e.interventions??0)}</td>
                  <td>${e.last_checker?t`<span class="badge ${e.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${e.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${p(new Date(e.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:y(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:t`<label class="field field-search">${_(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${e.map(e=>t`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?t`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:f}
    `}renderSession(e){let n=this.turns,i=this.live.find(t=>t.key===e);if(n==null)return m(4);let a=n.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,o=n.reduce((e,t)=>e+(t.injected?.length??0),0),s=this.events.filter(e=>ae(e)).length,l=Ce(n);return t`<section class="panel" style="margin-bottom:16px" data-timeline=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${T(e,44)} ${i?t`<span class="badge badge-on">live</span>`:f}</h2>
        <p class="panel-sub">${n.length} turns · ${o} hidden note${o===1?``:`s`} · ${s} monitor act${s===1?``:`s`}${i?` · class now ${i.class_id} · ${u(i.note_tokens)} note tokens replayed a turn`:``}${n.length?` · since ${r(new Date(n[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${k.exportUrl(e,`md`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${_(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${k.exportUrl(e,`json`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${_(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>S({session:null})}>${_(`x`)}</button></div></header>
      ${this.axis(fe(n,this.events,this.ids,e=>c(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${n.map(e=>{let n=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/a*900)),r=(e.injected?.length??0)>0,i=!!e.shadow,o=(e.events??[]).filter(e=>e.decision!==`PASS`),s=o.some(e=>ae(e))?`act`:o.length?`shadow`:null;return t`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${n}px; --cls: var(--series-${Ve(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${Ne(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${r?t`<span class="mark"></span>`:i?t`<span class="mark is-shadow"></span>`:f}
            ${s?t`<span class="mark is-monitor ${s===`shadow`?`is-shadow`:``}" data-monitor-mark=${s}></span>`:f}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${l.series.length?t`<qse-ribbon .series=${l.series} .labels=${l.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:f}
      <div class="feed" style="margin-top:10px">${n.slice().reverse().map(e=>Be(e,this.ids,e=>this.open=e,Ne(e)+((e.events??[]).some(e=>ae(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?t`<div style="margin-top:12px"><qse-cond-journal .session=${e}></qse-cond-journal></div>`:f}
    </section>`}axis(e){let n=e=>`${(e*100).toFixed(3)}%`;return t`<figure class="axis" data-axis aria-label="the session on one time axis">
      <div class="axis-lane axis-acts" aria-label=${`${e.acts.length} monitor decisions`}>
        ${e.acts.map(e=>t`<span class="axis-mark ${e.shadow?`is-shadow`:`is-act`}" style="left:${n(e.x)}" title=${e.title} data-axis-act=${e.shadow?`shadow`:`act`}></span>`)}
        ${e.notes.map(e=>t`<span class="axis-mark is-note ${e.shadow?`is-shadow`:``}" style="left:${n(e.x)}" title=${e.title} data-axis-note></span>`)}
      </div>
      <div class="axis-lane axis-turns" aria-label=${`${e.turns.length} turns`}>
        ${e.turns.map(e=>t`<span class="axis-turn ${e.compaction?`is-compaction`:``}" style="left:${n(e.x0)};width:${n(Math.max(0,e.x1-e.x0))};--cls:var(--series-${e.color})" title=${e.title} data-axis-turn></span>`)}
      </div>
      <div class="axis-lane axis-events" aria-label=${`${e.compactions.length} compactions, ${e.checkers.length} checker results`}>
        ${e.compactions.map(e=>t`<span class="axis-mark is-compaction" style="left:${n(e.x)}" title=${e.title} data-axis-compaction></span>`)}
        ${e.checkers.map(e=>t`<span class="axis-mark ${e.pass?`is-pass`:`is-fail`}" style="left:${n(e.x)}" title=${e.title} data-axis-checker=${e.pass?`pass`:`fail`}></span>`)}
      </div>
      <div class="axis-ticks" aria-hidden="true">${e.ticks.map((r,i)=>t`<span class="num ${i===0?`is-first`:i===e.ticks.length-1?`is-last`:``}" style="left:${n(r.x)}">${r.label.slice(0,5)}</span>`)}</div>
      <figcaption class="muted small">One time axis, ${e.ticks[0]?.label??``} to ${e.ticks[e.ticks.length-1]?.label??``}: ${e.turns.length} turns (the gaps are the client's time), ${e.acts.filter(e=>!e.shadow).length} monitor actions ◆ above (◇ shadow, ${e.acts.filter(e=>e.shadow).length}), ${e.notes.length} router note${e.notes.length===1?``:`s`}, ${e.compactions.length} compaction${e.compactions.length===1?``:`s`} ▲ and ${e.checkers.length} checker result${e.checkers.length===1?``:`s`} below (● pass, ○ fail; the gate numbers are in each one's title).</figcaption>
    </figure>`}tier1(){let e=ge(this.events);return e?t`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${e.agree} of ${e.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${e.decisions} shadow decisions)</span></p>`:f}};l([o({attribute:!1})],J.prototype,`ids`,void 0),l([o({attribute:!1})],J.prototype,`selected`,void 0),l([a()],J.prototype,`list`,void 0),l([a()],J.prototype,`live`,void 0),l([a()],J.prototype,`turns`,void 0),l([a()],J.prototype,`events`,void 0),l([a()],J.prototype,`error`,void 0),l([a()],J.prototype,`query`,void 0),l([a()],J.prototype,`client`,void 0),l([a()],J.prototype,`open`,void 0),J=l([s(`qse-cond-sessions`)],J);var Xt={scope:`tools`,retention_days:30,clip_chars:4e3,loop_chars:200},Zt=/^### (user|assistant|summary|runtime|tool result \((.*)\)|tool call \((.*)\)|reasoning \((.*)\))$/;function Qt(e){let t=e[1];return e[2]===void 0?e[3]===void 0?e[4]===void 0?{kind:t===`user`||t===`assistant`||t===`summary`||t===`runtime`?t:`other`,title:t,name:null,clipped:!1}:{kind:`reasoning`,title:t,name:null,clipped:!1}:{kind:`tool_call`,title:t,name:e[3].trim()||null,clipped:!1}:{kind:`tool_result`,title:t,name:e[2].split(`,`)[0].trim()||null,clipped:/clipped to/.test(e[2])}}function $t(e){let t=e.replace(/\n+$/,``).split(`
`),n=t[0]??``,r=null,i=[],a=null,o=()=>{a&&i.push({kind:a.kind,title:a.title,name:a.name,clipped:a.clipped,body:a.lines.join(`
`).replace(/^\n+|\n+$/g,``)})};for(let e of t.slice(1)){let t=e.match(Zt);t?(o(),a={...Qt(t),lines:[]}):a?a.lines.push(e):e.startsWith(`compaction: `)&&(r=e)}o();let s=i.find(e=>e.kind===`runtime`);return{head:n,compaction:r,sections:i.filter(e=>e.kind!==`runtime`),runtime:s?en(s.body):[]}}function en(e){return e.split(`
`).filter(e=>e.trim()).map(e=>{let t=[`hidden`,`replayed`,`acts`,`checker`,`snapshot`,`finish`,`redactions`].find(t=>e.startsWith(`${t}: `))??`other`,n=e.match(/ · journal ([A-Za-z0-9_-]{4,40})$/),r=t===`checker`?e.match(/ (PASS|FAIL)(?: <-|$)/):null;return{kind:t,text:e,journal:n?n[1]:null,verdict:r?r[1]:null}})}function tn(e){if(!e.counts)return`counts unknown (no meta.json)`;let t=e.counts,n=[`${t.blocks} turn${t.blocks===1?``:`s`}`];return t.untranscribed&&n.push(`${t.untranscribed} without tools (counted only)`),n.push(`${t.acts} act${t.acts===1?``:`s`}`),t.snapshots&&n.push(`${t.snapshots} snapshot${t.snapshots===1?``:`s`}`),t.compactions&&n.push(`${t.compactions} compaction${t.compactions===1?``:`s`}`),t.redactions&&n.push(`${t.redactions} redacted`),e.checkers?.last&&n.push(e.checkers.last),n.join(` · `)}function nn(e){return{...Xt,...e?.memory?.transcripts??{}}}function Y(e,t){return e?e.replace(/^[ch]:/,``):t}var rn={mode:`on`,budget_gb:50,ttl_days:7,admit_after:2,min_free_gb:50},an={"opencode-head":`opencode head`,head:`session head`,package:`package`,memory:`memory file`,prefix:`repeated prefix`};function on(e){return{...rn,...e?.memory?.disk_cache??{}}}function sn(e){let t=[],n=e=>Number.isFinite(e)&&Math.round(e)===e;return e.budget_gb>=1&&e.budget_gb<=4e3||t.push(`the budget is 1 to 4,000 GB`),n(e.ttl_days)&&e.ttl_days>=1&&e.ttl_days<=90||t.push(`an entry is kept 1 to 90 days`),n(e.admit_after)&&e.admit_after>=1&&e.admit_after<=10||t.push(`a prefix is written after 1 to 10 prefills`),e.min_free_gb>=50&&e.min_free_gb<=4e3||t.push(`the NVMe keeps at least 50 GB free`),t}function cn(e,t){if(t.free_bytes==null)return{usable_gb:null,short:!1};let n=((t.bytes??0)+Math.max(0,t.free_bytes-Math.max(50,e.min_free_gb)*1e9))/1e9;return{usable_gb:n,short:n<e.budget_gb}}function ln(e){let t=e.lookups??0;return t?`${Math.round((e.hit_rate??0)*100)} % of ${t.toLocaleString(`en-US`)} prefill${t===1?``:`s`} it could have shortened`:`no prefill the disk could have shortened yet`}function un(e){let t=e=>typeof e==`number`?e.toLocaleString(`en-US`):String(e),n=[];return e.tokens!=null&&n.push(`${t(e.tokens)} tokens`),e.kind===`hit`&&e.read!=null&&n.push(`${t(e.read)} read`,`${t(e.ms)} ms`),e.kind===`write`&&(n.push(`${t(e.chunks)} chunk${e.chunks===1?``:`s`}`,`${t(e.anchors)} anchor${e.anchors===1?``:`s`}`),e.dropped&&typeof e.dropped==`object`&&n.push(`dropped ${Object.entries(e.dropped).map(([e,t])=>`${t} ${e}`).join(`, `)}`)),e.why!=null&&n.push(String(e.why)),e.kind===`clear`&&n.push(`${t(e.entries)} entries`),e.error!=null&&n.push(String(e.error)),[e.kind,...n].join(` · `)}var dn=e=>r(new Date(e*1e3).toISOString()),fn=e=>Object.entries(e??{}).filter(([,e])=>e).map(([e,t])=>`${u(t)} ${e}`).join(`, `)||`none`,X=class extends d{constructor(...e){super(...e),this.disk=null,this.error=null,this.busy=!1,this.cleared=null,this.settings=new C(this,`settings`)}connectedCallback(){super.connectedCallback(),this.load(),this.settings.load(),this.settings.onSaved=()=>void this.load()}async load(){try{this.disk=await M(()=>k.memoryDisk()),this.error=null}catch(e){this.error=n(e).message}}async clear(){let e=this.disk;if(e&&await D(`Clear the disk cache?`,[`Every entry goes: ${u(e.entries??0)} prefixes in ${u(e.chunks??0)} chunk files, ${i(e.bytes??0)}, and the counts that decide what is written next.`,`Nothing a request needs is lost: the next prefill of each prefix runs cold, as without the tier, and writes it again.`],`Clear`)){this.busy=!0;try{let e=await M(()=>k.memoryDiskClear());this.cleared=`cleared ${u(e.entries)} entries, ${i(e.bytes)}`,this.disk=e.disk}catch(e){this.error=n(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.disk)return t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${this.error}</div></div>`;if(!this.disk)return m(5);let e=this.disk;return t`${e.running?this.renderRunning(e):this.renderOff(e)}${this.renderSettings(e)}`}renderOff(e){return g(`Disk cache`,y(`The disk tier does not run on this server`,t`It starts with the engine when <code>QSE_DISK_CACHE</code> in serve.env names its directory (the flag is <code class="mono">${e.flag}</code> now)${e.free_bytes==null?f:t`; the NVMe has <span class="num">${i(e.free_bytes,0)}</span> free`}. Its settings below apply once it runs.`),{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM`})}renderRunning(e){let n=e.bytes??0,r=e.budget??0,a=r?n/r:0,o=e.free_bytes??null,s=o!=null&&e.floor!=null&&o<e.floor+1e10,c=t`<div class="stats disk-stats" data-disk-tiles>
      ${x({label:`on disk`,value:i(n,1),sub:t`of <span class="num">${i(r,0)}</span> · ${u(e.entries??0)} prefix${e.entries===1?``:`es`}, ${u(e.chunks??0)} chunk files`,warn:a>.95})}
      ${x({label:`NVMe free`,value:o==null?`—`:i(o,0),sub:t`never under <span class="num">${i(e.floor??5e10,0)}</span>`,warn:s})}
      ${x({label:`hit rate`,value:e.hit_rate==null?`—`:v(e.hit_rate,0),sub:ln(e),na:e.hit_rate==null})}
      ${x({label:`restored`,value:u(e.tokens_restored??0),unit:`tokens`,sub:e.last_restore_ms==null?`no restore yet`:t`last <span class="num">${u(Math.round(e.last_restore_ms))}</span> ms${e.last_read_gbs==null?f:t` at <span class="num">${e.last_read_gbs.toFixed(1)}</span> GB/s`}`})}
    </div>`,l=t`<div class="meter-block disk-meter">
      <div class="meter-head"><span>budget</span><span class="num">${i(n,1)} of ${i(r,0)} · ${v(a,0)}</span></div>
      <div class="meter" role="img" aria-label=${`disk cache ${v(a,0)} of its budget`}><span class="meter-fill ${a>.95?`is-warn`:``}" style="width:${Math.min(100,a*100).toFixed(1)}%"></span></div>
    </div>`,d=e.top??[],p=t`${c}${l}
      <p class="small muted disk-line" data-disk-counters>
        ${e.on?t`<span class="badge badge-on">on</span>`:t`<span class="badge">off (settings)</span>`}
        written: <span class="num">${u(e.writes_chunk??0)}</span> chunks, <span class="num">${u(e.writes_anchor??0)}</span> anchors, <span class="num">${i(e.bytes_written??0,1)}</span> ·
        evicted: ${fn(e.evicted)} · dropped: ${fn(e.dropped)}${e.failed_restores?t` · <span class="warn-ink">${u(e.failed_restores)} failed restore${e.failed_restores===1?``:`s`}</span>`:f}
        · staging buffer <span class="num">${i(e.staging_bytes||e.staging_planned||0,2)}</span>${e.staging_bytes?``:` (when first used)`} · <span class="mono">${e.root??``}</span>
      </p>
      ${d.length?this.renderEntries(d):y(`Nothing on disk yet`,`A prefix is written after ${e.settings.admit_after===1?`its first prefill`:`${e.settings.admit_after} prefills`} of at least ${u(e.min_tokens??4096)} tokens, and the first long request of an opencode session is written where its user message starts.`)}
      ${this.renderEvents(e)}`;return g(`Disk cache`,p,{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM; a hit is read only past the rows already in place`,tools:t`${this.cleared?t`<span class="small muted" data-disk-cleared>${this.cleared}</span>`:f}
        <button class="btn btn-sm disk-clear" ?disabled=${this.busy||!(e.entries||e.chunks)} @click=${()=>this.clear()} data-disk-clear>${_(`x`)} Clear disk cache</button>
        <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the disk cache">${_(`refresh`)}</button>`})}renderEntries(e){let n=e=>t`<span class="badge disk-kind disk-kind-${e.kind}">${an[e.kind]}</span>${e.label?t` <span class="mono small">${e.label}</span>`:f}${e.complete?f:t` <span class="badge badge-warn" title="a chunk below it is not on disk: it is not read, and goes first">incomplete</span>`}`,r=e=>e.preview?t`<div class="small muted disk-preview" data-disk-preview>${e.preview}</div>`:f;return t`<h3 class="disk-sub">Most-hit prefixes</h3>
      <div class="table-scroll mem-table"><table class="table" data-disk-entries>
        <thead><tr><th class="num-col">tokens</th><th>what</th><th class="num-col">hits</th><th>last used</th><th class="num-col">a hit reads</th></tr></thead>
        <tbody>${e.map(e=>t`<tr data-disk-row=${`${e.kind}-${e.tokens}`}><td class="num num-col">${u(e.tokens)}</td><td class="cell-wrap">${n(e)}${r(e)}</td>
            <td class="num num-col">${u(e.hits)}</td><td class="num">${dn(e.used)}</td><td class="num num-col">${i(e.reads_bytes,1)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards" role="list">${e.map(e=>t`<div class="mem-card disk-card" role="listitem" data-disk-card=${`${e.kind}-${e.tokens}`}>
          <div class="mem-card-head"><span class="num">${u(e.tokens)} tokens</span><span class="num small">${u(e.hits)} hit${e.hits===1?``:`s`}</span></div>
          <div>${n(e)}</div>${r(e)}
          <div class="small muted">last used ${dn(e.used)} · a hit reads ${i(e.reads_bytes,1)}</div>
        </div>`)}</div>`}renderEvents(e){let n=(e.events??[]).slice(-10).reverse();return n.length?t`<details class="sec" style="margin-top:12px" data-disk-events>
      <summary>Last events <span class="sec-sub">what it read, wrote, dropped and evicted (the journal keeps them, kind memory)</span></summary>
      <ul class="disk-events">${n.map(e=>t`<li class="small"><span class="num muted">${dn(e.ts)}</span> <span class="disk-ev disk-ev-${e.kind}">${un(e)}</span></li>`)}</ul>
    </details>`:f}renderSettings(e){let n=this.settings;if(!n.draft)return n.loadError?t`<p class="err-ink small">${n.loadError}</p>`:f;let r=on(n.draft),a=(e,t)=>n.set(w(n.draft,[`memory`,`disk_cache`],{...r,[e]:t})),o=(e,n,i,o,s=1)=>t`<label class="field"><span>${n}</span><input type="number" inputmode="decimal" min=${i} max=${o} step=${s} .value=${String(r[e])} data-disk-setting=${e} @input=${t=>a(e,Number(t.target.value))} /></label>`,s=cn(r,e),c=sn(r);return t`<details class="sec" style="margin-top:16px" data-sec="disk" open>
      <summary>Disk cache settings <span class="sec-sub">on or off, its size on the NVMe, how long it keeps an entry, what it writes</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-disk-mode><span class="muted small">disk cache</span>${A([`on`,`off`],r.mode,e=>a(`mode`,e),{small:!0,label:`disk cache mode`})}<span class="small">${r.mode===`on`?`prefixes are read from and written to the disk`:`nothing is read or written; the files stay`}</span></div>
        <div class="form-grid">
          ${o(`budget_gb`,`cache budget (GB)`,1,4e3)}
          ${o(`ttl_days`,`drop an entry unused for (days)`,1,90)}
          ${o(`admit_after`,`write a prefix after it was prefilled (times)`,1,10)}
          ${o(`min_free_gb`,`keep free on the NVMe (GB, at least 50)`,50,4e3)}
        </div>
        <p class="small" data-disk-free>
          ${e.free_bytes==null?f:t`The NVMe has <span class="num">${i(e.free_bytes,0)}</span> free${e.bytes?t`, and the cache holds <span class="num">${i(e.bytes,1)}</span>`:f}. `}
          ${s.short?t`<span class="warn-ink" data-disk-short>${_(`warn`)} At most <span class="num">${s.usable_gb.toFixed(0)} GB</span> of the ${r.budget_gb} GB fit above the ${Math.max(50,r.min_free_gb)} GB it keeps free; the tier stops there.</span>`:f}
        </p>
        <p class="muted small">Least valuable first when it is full: hits, times the tokens an entry saves per byte it holds, halved for every day unused. The first long request of an opencode session is written whatever the count, at the lines below where its user message starts. Applied from the next request; the free space it keeps is never under ${50} GB, whatever is set here.</p>
        ${c.length?t`<ul class="errors" role="alert" data-disk-errors>${c.map(e=>t`<li>${e}</li>`)}</ul>`:f}
        ${O(n)}
        ${j(n,`the settings`)}
      </div>
    </details>`}};l([a()],X.prototype,`disk`,void 0),l([a()],X.prototype,`error`,void 0),l([a()],X.prototype,`busy`,void 0),l([a()],X.prototype,`cleared`,void 0),X=l([s(`qse-cond-disk`)],X);var pn={tools:`requests with tools in full (and compactions); chats without tools as counters only`,all:`every request in full`,none:`nothing`},Z=class extends d{constructor(...e){super(...e),this.ids=[],this.selected=null,this.page=null,this.tab=null,this.list=null,this.one=null,this.error=null,this.oneError=null,this.query=``,this.busy=null,this.row=null,this.ev=null,this.rowError=null,this.settings=new C(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.tab!==`disk`&&(this.loadList(),this.settings.load())}updated(e){e.get(`tab`)===`disk`&&this.tab!==`disk`&&!this.list&&!this.error&&(this.loadList(),this.settings.draft||this.settings.load())}willUpdate(){let e=this.selected?`${this.selected}#${this.page??``}`:null;e!==this.loadedFor&&(this.loadedFor=e,this.one=null,this.oneError=null,this.selected&&this.loadOne(this.selected,this.page))}async loadList(){try{this.list=await M(()=>k.memorySessions()),this.error=null}catch(e){this.error=n(e).message}}async loadOne(e,t){try{let n=await M(()=>k.memorySession(e,t));e===this.selected&&(this.one=n)}catch(t){e===this.selected&&(this.oneError=n(t).message)}}async keep(e,t){if(e.session){this.busy=e.dir;try{await M(()=>k.memoryKeep(e.session,t)),await this.loadList(),this.one&&this.one.session.session===e.session&&(this.one={...this.one,session:{...this.one.session,keep:t}})}catch(e){this.error=n(e).message}finally{this.busy=null}}}async download(e,t){try{let n=await M(()=>k.memoryExport(e,t)),r=document.createElement(`a`);r.href=URL.createObjectURL(n),r.download=`transcript-${e.replace(/[^a-z0-9]+/gi,`-`)}.${t}`,r.click(),setTimeout(()=>URL.revokeObjectURL(r.href),1e3)}catch(e){this.oneError=n(e).message}}async openRow(e){this.rowError=null;try{let t=(await k.decision(e)).decision;t.kind===`monitor`?this.ev=t:this.row=t}catch(t){this.rowError=`journal row ${e}: ${n(t).message} (the journal keeps 30 days)`}}render(){let e=this.tab===`disk`,n=(n,r)=>t`<a class="cond-tab ${n===`disk`===e?`is-on`:``}" href="#" aria-current=${n===`disk`===e?`page`:f} data-mem-tab=${n??`sessions`} @click=${e=>{e.preventDefault(),S({mtab:n,memory:null,mpage:null})}}>${r}</a>`,r=t`<nav class="mem-tabs" aria-label="memory tabs">${n(null,`Sessions`)}${n(`disk`,`Disk cache`)}<span class="cond-tab is-off" title="the notes tab comes with the memory files (VIS-50)">Notes</span></nav>`;return e?t`${r}<qse-cond-disk></qse-cond-disk>`:t`
      ${r}
      ${this.selected?this.renderOne(this.selected):f}
      ${this.renderList()}
      ${this.renderSettings()}
      ${this.row?t`<qse-cond-decision .row=${this.row} .ids=${this.ids} .close=${()=>this.row=null}></qse-cond-decision>`:f}
      ${this.ev?t`<qse-cond-event .ev=${this.ev} .close=${()=>this.ev=null}></qse-cond-event>`:f}
    `}renderList(){if(this.error&&!this.list)return t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${this.error}</div></div>`;if(!this.list)return m(5);let e=this.list.writer,n=this.query.toLowerCase(),a=this.list.sessions.filter(e=>!n||`${e.session??e.dir} ${e.client??``} ${e.project??``}`.toLowerCase().includes(n)),o=e=>e.last_ms?r(new Date(e.last_ms).toISOString()):`—`,s=n=>t`<label class="switch" title="kept sessions are exempt from the ${e.settings.retention_days}-day prune"><input type="checkbox" .checked=${n.keep} ?disabled=${!n.session||this.busy===n.dir} data-keep=${n.dir} @click=${e=>e.stopPropagation()} @change=${e=>this.keep(n,e.target.checked)} aria-label=${`keep ${Y(n.session,n.dir)}`} /> keep</label>`,c=e=>e.checkers?.last?t`<span class="badge ${/ PASS$/.test(e.checkers.last)?`badge-ok`:`badge-warn`}">${e.checkers.last}</span>${e.checkers.PASS||e.checkers.FAIL?t` <span class="muted small num">${e.checkers.PASS??0} pass · ${e.checkers.FAIL??0} fail</span>`:f}`:t`<span class="muted">—</span>`,l=e=>e.session&&S({memory:e.session,mpage:null});return g(`Session transcripts`,a.length?t`<div class="table-scroll mem-table"><table class="table" data-memory-sessions>
              <thead><tr><th>last request</th><th>session</th><th>client</th><th>profile</th><th>project</th><th class="num-col">turns</th><th class="num-col">acts</th><th>checker</th><th class="num-col">size</th><th>keep</th></tr></thead>
              <tbody>${a.map(e=>t`<tr class="req-row ${e.session&&e.session===this.selected?`is-open`:``}" data-memory-row=${e.dir} @click=${()=>l(e)}>
                  <td class="num">${o(e)}</td><td class="mono small cell-wrap">${Y(e.session,e.dir)}</td><td>${e.client??`—`}</td><td>${e.profile??`—`}</td>
                  <td class="small cell-wrap">${e.project??`—`}</td>
                  <td class="num num-col">${e.counts?u(e.counts.blocks):`?`}</td><td class="num num-col">${e.counts?u(e.counts.acts):`?`}</td>
                  <td>${c(e)}</td><td class="num num-col">${i(e.bytes??0)}</td><td>${s(e)}</td></tr>`)}</tbody></table></div>
            <div class="mem-cards" role="list">${a.map(e=>t`<div class="mem-card ${e.session&&e.session===this.selected?`is-open`:``}" role="listitem" data-memory-card=${e.dir} @click=${()=>l(e)}>
                <div class="mem-card-head"><span class="mono small">${Y(e.session,e.dir)}</span>${s(e)}</div>
                <div class="small">${o(e)} · ${e.client??`—`} · ${e.profile??`—`}</div>
                <div class="small muted">${tn(e)}</div>
                ${e.project?t`<div class="small muted cell-wrap">${e.project}</div>`:f}
                <div>${c(e)}</div>
              </div>`)}</div>`:y(`No transcripts yet`,e.refused?`The memory directory ${e.root} is inside a git work tree, so nothing is written there.`:`TandemLLM requests with tools are written to ${e.root} when their stream closes; chats without tools are counted only.`),{sub:t`<span data-writer>${u(e.blocks)} block${e.blocks===1?``:`s`} written since the start · ${u(e.redactions)} secret${e.redactions===1?``:`s`} redacted · ${u(e.failed)} failed · ${u(e.dropped)} dropped · ${i(e.bytes)} · <span class="mono">${e.root}</span></span>`,tools:t`<label class="field field-search">${_(`search`)}<input placeholder="search a session, client or project" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search transcripts" /></label>
          <button class="btn btn-sm" @click=${()=>this.loadList()} aria-label="reload the list">${_(`refresh`)}</button>`,id:`cond-memory`})}renderOne(e){if(this.oneError)return t`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${_(`warn`)} ${this.oneError}</div></div>`;let n=this.one;if(!n)return m(4);let i=n.session,a=n.pages??1,o=n.page??1,s=a>1?t`<div class="toolbar mem-pager" data-pager><button class="btn btn-sm" ?disabled=${o<=1} @click=${()=>S({mpage:String(o-1)})}>Older</button>
          <span class="small num">page ${o} of ${a} · ${u(n.total??n.blocks.length)} blocks, ${n.per_page} a page</span>
          <button class="btn btn-sm" ?disabled=${o>=a} @click=${()=>S({mpage:String(o+1)})}>Newer</button></div>`:f;return t`<section class="panel" style="margin-bottom:16px" data-transcript=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title mono">${Y(i.session,i.dir)}${i.keep?t` <span class="badge badge-on">kept</span>`:f}</h2>
        <p class="panel-sub">${i.client??`—`} · ${i.profile??`—`}${i.project?` · ${i.project}`:``} · ${tn(i)}${i.first_ms?` · since ${r(new Date(i.first_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <button class="btn btn-sm" @click=${()=>this.download(e,`md`)} data-mem-export="md">${_(`down`)} Markdown</button>
          <button class="btn btn-sm" @click=${()=>this.download(e,`json`)} data-mem-export="json">${_(`down`)} JSON</button>
          <a class="btn btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e)}`} data-mem-timeline>Timeline</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the transcript" @click=${()=>S({memory:null,mpage:null})}>${_(`x`)}</button></div></header>
      ${this.rowError?t`<p class="warn-ink small" role="alert">${_(`warn`)} ${this.rowError}</p>`:f}
      ${s}
      ${n.blocks.length?t`<div class="mem-blocks">${n.blocks.map(e=>this.renderBlock(e))}</div>`:y(`No blocks`,`This session was counted but not written: its requests carried no tools.`)}
      ${s}
    </section>`}renderBlock(e){let n=$t(e.text),r=n.runtime.filter(e=>e.kind===`hidden`).length,i=n.runtime.find(e=>e.kind===`acts`),a=t`<span class="mem-turn num">turn ${e.turn??`?`}</span>
      <span class="muted small num">${e.time??``}</span>
      ${e.class_id?E(e.class_id,this.ids):f}
      ${e.effort?t`<span class="badge">${e.effort}</span>`:f}
      ${e.reasoning_tokens?t`<span class="muted small num">${u(e.reasoning_tokens)} reasoning tokens</span>`:f}
      ${r?t`<span class="badge badge-shadow" data-hidden-count=${r}>${r} hidden</span>`:f}
      ${i?t`<span class="badge badge-warn">${i.text.replace(/^acts: /,`acts `)}</span>`:f}`;return e.compaction?t`<details class="mem-block is-compaction" data-block=${e.turn??``} data-compaction>
        <summary><span class="mem-block-head">${a}<span class="badge">compaction</span><span class="small">${n.compaction?.replace(/^compaction: /,``)??``}</span></span></summary>
        ${n.sections.map(e=>this.renderSection(e))}${this.renderRuntime(n.runtime)}
      </details>`:t`<article class="mem-block" data-block=${e.turn??``}>
      <div class="mem-block-head">${a}</div>
      ${n.sections.map(e=>this.renderSection(e))}
      ${this.renderRuntime(n.runtime)}
    </article>`}renderSection(e){let n=e.body.length>600||e.body.split(`
`).length>12,r=t`<pre class="mem-text">${e.body}</pre>`;return e.kind===`tool_result`||e.kind===`tool_call`&&n||e.kind===`reasoning`?t`<details class="mem-sec mem-${e.kind}" ?open=${!n&&e.kind!==`reasoning`} data-sec-kind=${e.kind}>
        <summary class="small">${e.title}${e.clipped?t` <span class="badge badge-warn">clipped</span>`:f}</summary>${r}</details>`:t`<div class="mem-sec mem-${e.kind}" data-sec-kind=${e.kind}><div class="small muted">${e.title}</div>${r}</div>`}renderRuntime(e){return e.length?t`<ul class="mem-runtime" aria-label="what the runtime did">${e.map(e=>t`<li class="mem-rt mem-rt-${e.kind} ${e.verdict?`is-${e.verdict.toLowerCase()}`:``}" data-rt=${e.kind}>
        ${e.kind===`hidden`?_(`info`):f}<span>${e.text.replace(/ · journal [A-Za-z0-9_-]+$/,``)}</span>
        ${e.journal?t`<button class="btn btn-ghost btn-sm" @click=${()=>this.openRow(e.journal)} data-journal=${e.journal} aria-label=${`open journal row ${e.journal}`}>journal row</button>`:f}
      </li>`)}</ul>`:f}renderSettings(){let e=this.settings;if(!e.draft)return e.loadError?t`<p class="err-ink small">${e.loadError}</p>`:f;let n=nn(e.draft),r=(t,r)=>e.set(w(e.draft,[`memory`,`transcripts`],{...n,[t]:r})),i=(e,i,a,o)=>t`<label class="field"><span>${i}</span><input type="number" min=${a} max=${o} .value=${String(n[e])} data-mem-setting=${e} @input=${t=>r(e,Math.round(Number(t.target.value)))} /></label>`;return t`<details class="sec" style="margin-top:16px" data-sec="transcripts">
      <summary>Transcript settings <span class="sec-sub">what the engine writes to the box, how long it keeps it, where tool output is clipped</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-mem-scope><span class="muted small">write</span>${A([`tools`,`all`,`none`],n.scope,e=>r(`scope`,e),{small:!0,label:`transcript scope`})}<span class="small">${pn[n.scope]}</span></div>
        <div class="form-grid">
          ${i(`retention_days`,`keep a session for (days; kept sessions are exempt)`,1,400)}
          ${i(`clip_chars`,`clip a tool result or a call at (characters)`,200,1e5)}
          ${i(`loop_chars`,`a loop the D1 guard cut keeps its first and last (characters)`,0,4e3)}
        </div>
        <p class="muted small">Every block passes the secret-pattern refusal before it is written: a key, token or password becomes <code>[redacted:kind]</code> and the journal counts it, never the value. There is no switch for it. The files are <code>0600</code> under the engine's home on the box, never in a repository, and the ledger's backup never holds them. Khaled's decision 2: chats without tools are counters only, unless the scope is widened here.</p>
        ${O(e)}
        ${j(e,`the settings`)}
      </div>
    </details>`}};l([o({attribute:!1})],Z.prototype,`ids`,void 0),l([o({attribute:!1})],Z.prototype,`selected`,void 0),l([o({attribute:!1})],Z.prototype,`page`,void 0),l([o({attribute:!1})],Z.prototype,`tab`,void 0),l([a()],Z.prototype,`list`,void 0),l([a()],Z.prototype,`one`,void 0),l([a()],Z.prototype,`error`,void 0),l([a()],Z.prototype,`oneError`,void 0),l([a()],Z.prototype,`query`,void 0),l([a()],Z.prototype,`busy`,void 0),l([a()],Z.prototype,`row`,void 0),l([a()],Z.prototype,`ev`,void 0),l([a()],Z.prototype,`rowError`,void 0),Z=l([s(`qse-cond-memory`)],Z);var Q=class extends d{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([k.stats(168),k.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let t=n(e);this.error=`${t.endpoint}: ${t.message}`}}async promote(e,t,r,i,a){if(await D(`Promote ${r} to ${t===`task`?`act`:`on`} in ${e}?`,[t===`task`?`The session monitor will act on ${r} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${r} decisions instead of only journaling them: ${r===`effort`?`budgets and thinking off go to the engine`:r===`packages`?`skills and packages go into the prompt`:r===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${i.toFixed(2)} over ${a} labelled decisions. A new profile version records it.`],`Promote`))try{let n=await k.doc(`profiles`,e),o=t===`task`?w(n.doc,[`tasks`,r,`mode`],`act`):w(n.doc,[`decisions`,r],`on`),s=await M(()=>k.put(`profiles`,e,o,`${r} to ${t===`task`?`act`:`on`}, precision ${i.toFixed(2)} / ${a}`,n.version));this.msg=`${e} v${s.version}: ${r} promoted`,this.onSaved()}catch(e){this.msg=n(e).message}}async loadAb(){try{this.abVersions=(await k.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await k.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await k.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=n(e).message}}render(){return this.error&&!this.stats?t`<div class="state state-error" role="alert"><div class="state-title">${_(`warn`)} ${this.error}</div></div>`:this.stats?t`
      ${this.msg?t`<div class="note note-info" role="status">${this.msg}</div>`:f}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:m(8)}renderPromotion(e){let n=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,r=(e.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return g(`Shadow precision and promotion`,t`<p class="muted small">${Ue}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${e.precision.length||r.length?t`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${e.precision.map(e=>{let r=ve(e),i=n(e.profile,e.decision);return t`<tr data-precision-row=${`${e.profile}/${e.decision}`}>
                    <td>${e.profile}</td><td class="mono">${e.decision}</td><td><span class="badge ${i===`on`?`badge-on`:i===`shadow`?`badge-shadow`:``}">${i}</span></td>
                    <td class="num num-col">${u(e.decisions)}</td><td class="num num-col">${u(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${r.allowed?``:`muted`}">${r.why}</td>
                    <td><button class="btn btn-sm ${r.allowed?`btn-primary`:``}" ?disabled=${!r.allowed||i!==`shadow`} title=${r.allowed?``:Ue} data-promote=${`${e.profile}/${e.decision}`} @click=${()=>this.promote(e.profile??``,`decision`,e.decision,e.precision??0,e.labelled)}>Promote to on</button></td></tr>`})}
                ${r.map(e=>{let n=ve(e);return t`<tr data-precision-row=${`task/${e.task}`}><td>—</td><td class="mono">${e.task}</td><td><span class="badge ${e.mode===`act`?`badge-on`:`badge-shadow`}">${e.mode}</span></td>
                    <td class="num num-col">${u(e.decisions)}</td><td class="num num-col">${u(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${n.allowed?``:`muted`}">${n.why}</td>
                    <td><button class="btn btn-sm ${n.allowed?`btn-primary`:``}" ?disabled=${!n.allowed||e.mode!==`shadow`} data-promote=${`task/${e.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,e.task,e.precision??0,e.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:y(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${e.monitor?f:t`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let n=this.reports.find(e=>e.kind===`classifier`);if(!n)return g(`Classifier accuracy`,y(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let i=n.stages??{},a=n.set??{},o=(i[this.stage]??Object.values(i)[0])?.confusion??{},s=a.classes??Object.keys(o);return g(t`Classifier accuracy <span class="muted small num">${a.name??``} · ${a.n??`?`} prompts · ${n.created_at?r(String(n.created_at)):``}</span>`,t`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(i).map(([n,r])=>t`<tr class="req-row ${n===this.stage?`is-open`:``}" @click=${()=>this.stage=n}><td class="mono">${n}</td><td class="num num-col">${v(r.accuracy,1)}</td><td class="num num-col">${r.answered} / ${r.n}</td><td class="num num-col">${r.accuracy_answered==null?`—`:v(r.accuracy_answered,1)}</td><td class="num num-col">${e(r.p50_ms)}</td><td class="num num-col">${e(r.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${s.length&&Object.keys(o).length?t`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...s,`—`].map(e=>t`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${s.map(e=>t`<tr><th>${E(e,this.ids)}</th>${[...s,`—`].map(n=>{let r=o[e]?.[n]??0;return t`<td class="cm-cell ${e===n?`cm-diag`:r?`cm-miss`:``}">${r||``}</td>`})}</tr>`)}</tbody></table></div>`:t`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}
        ${ke(n).length?t`<p class="muted small" style="margin-top:12px">Digest variants (SRV-153): <strong>today</strong> the 2,000-character digest the decider cut from the front, <strong>sized</strong> each stage's own head and tail, <strong>line</strong> sized plus the context line. <strong>router</strong> is the session's class after the stickiness rule.</p>
            <div class="table-scroll"><table class="table" data-classifier-variants>
              <thead><tr><th>digest</th><th>set</th><th class="num-col">n</th><th class="num-col">decider</th><th class="num-col">cascade</th><th class="num-col">router</th><th class="num-col">decider p50</th></tr></thead>
              <tbody>${ke(n).map(n=>t`<tr><td class="mono">${n.variant}</td><td class="mono small">${n.set}</td><td class="num num-col">${n.n}</td><td class="num num-col">${n.decider==null?`—`:v(n.decider,1)}</td><td class="num num-col">${n.cascade==null?`—`:v(n.cascade,1)}</td><td class="num num-col">${n.router==null?`—`:v(n.router,1)}</td><td class="num num-col">${e(n.p50)}</td></tr>`)}</tbody></table></div>`:f}`,{id:`cond-classifier`})}renderSuites(){let e=this.reports.filter(e=>e.kind===`suite`),n=se(this.reports,this.suiteFile),r=pe(n,e),i=n?.rows??[],a=Ee(this.reports,this.engine),o=De(i),s=n?.verdict,c=n?.comparison??[],l=n?.raw_row,d=n?.build,p=s?.indicative??it(i),m=(e,t=1)=>e==null?`—`:Number.isInteger(e)?u(e):e.toFixed(t);return g(t`TandemLLM against the raw engine ${n?t`<span class="muted small num">night ${n.night??`—`}${d?` · ${d.version??`?`} ${T(d.code_sha,12)}`:``}${n.profile?` · profile ${n.profile}`:``}</span>`:f}`,t`${a.tone===`ok`?t`<p class="small" data-guard-detail=${a.status}>${_(`check`)} ${a.line}${a.detail?t`: <span class="num">${a.detail}</span>`:f}.</p>`:t`<div class="guard-line ${a.tone===`note`?`guard-note`:``}" data-guard-detail=${a.status}>${_(a.tone===`note`?`info`:`warn`)}<span>${a.line}${a.detail?t`. <span class="small">${a.detail}</span>`:f}</span></div>`}
        ${e.length>1?t`<label class="field suite-pick"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${e.map(r=>t`<option value=${String(r.file)} ?selected=${r===n}>${String(r.night??r.file)} · ${String(r.file)}${pe(r,e)?` · superseded`:``}</option>`)}</select></label>`:f}
        ${r?t`<div class="guard-line guard-note" data-superseded>${_(`info`)}<span><span class="badge badge-warn">superseded</span> by <span class="mono">${r.file}</span>${r.why?t`: <span class="small">${r.why}</span>`:f}. Its numbers are kept as published; read the newer report.</span></div>`:f}
        ${s?t`<div class="note" data-eval-verdict>
              <span class="badge ${s.better?`badge-ok`:s.better===!1?`badge-warn`:``}"><strong>${s.better?`Better`:s.better===!1?`Not better`:`Not shown yet`}</strong></span>${p?t` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:f} — ${s.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${l?.status??`—`}${l?.source?` (${l.source})`:``}${n?.driver?` · driver ${n.driver.mode}`:``}${n?.isolation?` · ${n.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:f}
        ${o.length?t`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">turns</th><th class="num-col">thinking tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>efforts (turns)</th><th>build · night</th></tr></thead>
              <tbody>${o.flatMap(e=>{let n=c.find(t=>t.suite===e.suite||t.suite_id===(e.raw??e.tandem)?.suite_id);return[e.raw,e.tandem].filter(Boolean).map((r,i)=>t`<tr data-suite=${e.suite} data-arm=${r.arm}>
                    <td>${i===0?e.suite:``}${i===0&&!e.cmp.ok?t` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${e.cmp.why}</div>`:f}${i===0&&n?t` <span class="badge ${n.verdict===`better`?`badge-ok`:n.verdict===`worse`||n.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${n.verdict}>${n.verdict}</span>`:f}${i===0&&(r.indicative??(r.n??0)<10)?t` <span class="badge" data-row-indicative>indicative</span>`:f}</td>
                    <td class="mono">${qe(r.arm)}${r.pristine===!1?t` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:f}</td>
                    <td class="num num-col">${r.n??`—`}</td>
                    <td class="num num-col">${r.pass_rate==null?`—`:v(r.pass_rate,0)}</td>
                    <td class="num num-col">${r.gates_passed??`—`}${r.gates_total?` / ${r.gates_total}`:``}</td>
                    <td class="num num-col">${m(r.minutes_to_pass)}</td><td class="num num-col">${m(r.wall_minutes)}</td>
                    <td class="num num-col">${r.tokens==null?`—`:u(Math.round(r.tokens))}</td>
                    <td class="num num-col" data-turns>${m(r.turns)}</td><td class="num num-col" data-thinking>${r.reasoning_tokens==null?`—`:u(Math.round(r.reasoning_tokens))}</td>
                    <td class="num num-col">${m(r.decode_tps)}</td>
                    <td class="num num-col">${r.empty_stops??`—`}</td><td class="num num-col">${r.continuations??`—`}</td>
                    <td class="num num-col">${r.interventions??`—`}</td><td class="num num-col">${r.gave_ups??`—`}</td>
                    <td class="num num-col">${r.cache_hit==null?`—`:v(r.cache_hit,0)}</td>
                    <td class="small" data-efforts>${at(r)}</td>
                    <td class="small">${T(r.build,16)} · ${r.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts. Tokens, turns and thinking tokens are medians per run.</p>`:y(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return g(`A/B of two profile versions`,t`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>t`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>t`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(n=>t`<option value=${n.version} ?selected=${n.version===this[e]}>v${n.version} ${T(n.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?t`<div class="diff" data-ab>${this.abChanges.map(e=>t`<div class="diff-row"><span>${ye(e.path)}</span><span class="diff-before">${T(e.before,80)}</span><span class="diff-after">${T(e.after,80)}</span></div>`)}</div>`:t`<p class="muted small">The two versions are the same.</p>`:f}`,{id:`cond-ab`})}};l([o({attribute:!1})],Q.prototype,`ids`,void 0),l([o({attribute:!1})],Q.prototype,`tiles`,void 0),l([o({attribute:!1})],Q.prototype,`onSaved`,void 0),l([a()],Q.prototype,`stats`,void 0),l([a()],Q.prototype,`reports`,void 0),l([a()],Q.prototype,`engine`,void 0),l([a()],Q.prototype,`error`,void 0),l([a()],Q.prototype,`msg`,void 0),l([a()],Q.prototype,`stage`,void 0),l([a()],Q.prototype,`abProfile`,void 0),l([a()],Q.prototype,`abVersions`,void 0),l([a()],Q.prototype,`abA`,void 0),l([a()],Q.prototype,`abB`,void 0),l([a()],Q.prototype,`abChanges`,void 0),l([a()],Q.prototype,`abCounts`,void 0),l([a()],Q.prototype,`suiteFile`,void 0),Q=l([s(`qse-cond-compare`)],Q);var mn=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`deciders`,label:`Deciders`},{id:`sessions`,label:`Sessions`},{id:`memory`,label:`Memory`},{id:`compare`,label:`Compare`}],$=class extends d{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.guard=null,this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=h(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&mn.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r,i]=await Promise.all([k.profiles(),this.meta?Promise.resolve(this.meta):k.meta(),k.config(),k.doc(`router`),k.evals()]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.guard=Ee(i.reports,i.engine),this.error=null}catch(e){let t=n(e);this.error=t.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${t.endpoint}: ${t.message}`}}async setMode(e){let t=this.tiles.find(e=>e.name===this.active);if(t&&!t.locked&&(e!==`on`||await D(`Turn the ${t.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let n=await M(()=>k.setMode(t.name,e,`mode to ${e}`,t.version));this.reloaded=`v${n.version} · reloaded ${n.loaded_at?c(n.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=n(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),n=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return t`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${_(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>S({profile:e.target.value})}>
                ${this.tiles.map(e=>t`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?A([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):f}
            ${this.reloaded?t`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:f}
          </div>
        </div>
        ${this.switchError?t`<div class="guard-line" role="alert">${_(`warn`)} ${this.switchError}</div>`:f}
        ${this.error?t`<div class="guard-line" role="alert">${_(`warn`)} ${this.error}</div>`:f}
        ${this.guard&&this.guard.tone!==`ok`?t`<div class="guard-line ${this.guard.tone===`note`?`guard-note`:``}" data-guard=${this.guard.status}>${_(this.guard.tone===`note`?`info`:`warn`)}<span>${this.guard.line}${this.guard.detail?t`. <span class="small">${this.guard.detail}</span>`:f}</span></div>`:f}
        ${n.map(([e,n])=>t`<div class="guard-line" role="alert">${_(`warn`)} ${e}: ${n.error?.message} (kept version ${n.version})</div>`)}
        ${e?.locked?t`<div class="note" data-locked>${_(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:f}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${mn.map(e=>t`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:f}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return t`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return t`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return t`<qse-cond-monitor .meta=${e.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return t`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return t`<qse-cond-packages .ids=${e.ids} .tiles=${e.tiles} .selected=${this.params.get(`pkg`)} .tab=${this.params.get(`ptab`)} .pane=${this.params.get(`pview`)}></qse-cond-packages>`;case`deciders`:return t`<qse-cond-deciders></qse-cond-deciders>`;case`sessions`:return t`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`memory`:return t`<qse-cond-memory .ids=${e.ids} .selected=${this.params.get(`memory`)} .page=${Number(this.params.get(`mpage`))||null} .tab=${this.params.get(`mtab`)}></qse-cond-memory>`;case`compare`:return t`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};l([o({attribute:!1})],$.prototype,`params`,void 0),l([a()],$.prototype,`tiles`,void 0),l([a()],$.prototype,`defaultProfile`,void 0),l([a()],$.prototype,`meta`,void 0),l([a()],$.prototype,`config`,void 0),l([a()],$.prototype,`ids`,void 0),l([a()],$.prototype,`guard`,void 0),l([a()],$.prototype,`error`,void 0),l([a()],$.prototype,`switching`,void 0),l([a()],$.prototype,`reloaded`,void 0),l([a()],$.prototype,`switchError`,void 0),$=l([s(`qse-conductor`)],$);export{$ as QseConductor};