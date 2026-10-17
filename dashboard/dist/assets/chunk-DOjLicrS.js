import{a as e,c as t,d as n,f as r,g as i,i as a,l as o,n as s,o as c,p as l,s as u,t as d,u as f}from"./chunk-5VjiyoiJ.js";import{c as p,d as m,f as h,m as g,n as _,o as v,r as y}from"./chunk-BXmDkkzu.js";import{i as b,t as ee}from"./chunk-CI4nmvus.js";import{$ as te,B as ne,C as re,D as ie,E as ae,F as oe,G as se,H as ce,I as le,J as ue,K as de,L as fe,M as pe,N as me,Nt as x,O as he,Ot as ge,P as _e,Q as ve,St as S,T as ye,Tt as C,U as be,V as xe,W as Se,X as Ce,Y as we,Z as Te,_ as Ee,_t as De,at as w,b as Oe,bt as ke,ct as Ae,dt as je,et as Me,ft as Ne,gt as Pe,ht as Fe,it as Ie,j as Le,k as Re,kt as ze,lt as Be,mt as Ve,nt as He,ot as Ue,pt as We,q as Ge,rt as T,st as Ke,tt as qe,ut as Je,v as Ye,vt as Xe,w as Ze,x as Qe,xt as $e,y as E,yt as et,z as tt}from"./app-D2awEB7Z.js";import{n as nt,t as rt}from"./chunk-dBfjuPm3.js";import{DocEditor as D,clsChip as O,confView as it,confirmDialog as k,diffList as at,errorsList as A,eventRow as ot,feedRow as st,modeSeg as j,saveBar as M,sourceWords as ct,versionsPanel as lt}from"./chunk-DslnNd92.js";var N=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=C(()=>this.loadStats(),2e4),this.feedPoll=C(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await S.stats(24),this.error=null}catch(e){let t=b(e);this.error=`${t.endpoint}: ${t.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await S.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await S.event(e)).event}catch(e){this.error=b(e).message}}renderMonitor(t){let n=we(t);return i`<div class="grid-2 grid-2-wide">
      ${e(`Monitor acts an hour, last 24 h`,t.acts?i`<qse-ribbon .series=${n} .labels=${t.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:s(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${e(`Last actions of the monitor`,t.last_actions.length?i`<div class="feed" data-last-actions>${t.last_actions.map(e=>i`<button class="feed-row" data-action=${e.id} @click=${()=>this.openAction(e.id)}>
                <span class="feed-time">${g(new Date(e.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
                <span class="feed-conf muted small">${e.outcome?.kind??`running`}</span>
                <span class="feed-what">${ge({task:e.task,decision:e.decision,mode:`act`,trigger:e.trigger,cap:e.cap})}</span>
                <span class="feed-act">${e.session_key?i`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:l}</span>
              </button>`)}</div>`:i`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:i`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${a(`external`)}</a>`})}
    </div>`}render(){let t=this.stats;if(this.error&&!t)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${this.error}</div></div>`;if(!t)return c(6);let n=t.classifier,r=n.decider,o=n.qwen,d=r?.calls?r:o?.calls?o:r,f=Object.entries(t.roles),g=f.filter(([,e])=>e.up).length,_=t.monitor;return i`
      <div class="stats" data-tiles>
        ${u({label:`decisions, 24 h`,value:p(t.decisions),sub:i`<span class="num">${p(t.acted)}</span> acted · <span class="num">${p(t.shadow)}</span> in shadow`,title:`${t.injections} notes, ${t.injected_tokens} tokens injected`})}
        ${u({label:`monitor, 24 h`,value:_?p(_.acts):`—`,sub:_?i`acts · <span class="num">${p(_.shadow)}</span> in shadow · <span class="num" data-gave-up>${p(_.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!_,warn:(_?.gave_up??0)>0,title:_?`${p(_.hidden_tokens)} hidden tokens`:void 0})}
        ${u({label:`classifier p50 / p95${!r?.calls&&o?.calls?` (Qwen)`:``}`,value:d?.p50_ms==null?`—`:`${Math.round(d.p50_ms)} / ${Math.round(d.p95_ms??0)}`,unit:d?.p50_ms==null?void 0:`ms`,sub:i`decider ${p(r?.calls??0)} calls${o?i` · Qwen ${p(o.calls)} (${m(o.p50_ms)})`:l} · fail-open ${h(t.fail_open_rate,1)}`,warn:(t.fail_open_rate??0)>.1})}
        ${u({label:`roles online`,value:`${g} / ${f.length}`,sub:i`${f.map(([e,t])=>i`<span title=${t.error??``} class=${t.up?``:t.mode===`off`?`muted`:`err-ink`} data-role=${e}>${t.up?`●`:`○`} ${e} </span>`)}`,warn:f.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${e(`Decisions an hour, last 24 h`,t.decisions?i`<qse-ribbon .series=${Se(t,this.ids)} .labels=${t.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:s(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${e(`Task classes, 24 h`,i`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${t.classes.map(e=>i`<tr><td>${O(e.class_id,this.ids)}</td><td class="num num-col">${p(e.decisions)}</td><td class="num num-col">${e.labelled?`${e.agree} / ${e.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(t.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(t.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${e(i`Live decisions ${this.paused?i`<span class="badge">paused</span>`:i`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?i`<div class="feed" data-feed>${this.feed.map(e=>st(e,this.ids,e=>this.open=e,xe(e)))}</div>`:i`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:i`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${a(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${a(`external`)}</a>`})}
        ${e(`Profiles by client`,i`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(t=>i`<button class="pick" @click=${()=>x({screen:`profiles`,edit:e.name})}><span class="muted">${t}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(e=>i`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${_?this.renderMonitor(_):l}
      ${this.open?i`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:l}
      ${this.openEvent?i`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:l}
    `}};t([n({attribute:!1})],N.prototype,`ids`,void 0),t([n({attribute:!1})],N.prototype,`tiles`,void 0),t([n({attribute:!1})],N.prototype,`active`,void 0),t([f()],N.prototype,`stats`,void 0),t([f()],N.prototype,`feed`,void 0),t([f()],N.prototype,`error`,void 0),t([f()],N.prototype,`open`,void 0),t([f()],N.prototype,`openEvent`,void 0),t([f()],N.prototype,`paused`,void 0),N=t([r(`qse-cond-overview`)],N);var ut=[`off`,`low`,`medium`,`high`,`xhigh`],dt=[`small`,`medium`,`large`],P=class extends o{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new D(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),S.doc(`packages`).then(e=>this.packages=fe(e.doc)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(T(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await S.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=b(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return c(8);if(e.loadError)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=t.classes.map(e=>e.id);return i`
      ${this.renderTest(n)}
      <details class="sec" open data-sec="classes">
        <summary>Task classes <span class="sec-sub">${t.classes.length} classes; a classifier sees only the descriptions</span></summary>
        <div class="sec-body">
          ${t.classes.map((e,r)=>this.renderClass(e,r,t.classes.length,n))}
          <button class="btn" data-add-class @click=${()=>this.addClass()}>${a(`plus`)} Add a class</button>
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
          ${this.renderContextStage(t)}
          <div class="form-grid">
            ${this.text(`question to the decider`,t.classifier.question,e=>this.upd([`classifier`,`question`],e))}
            ${this.num(`digest (characters)`,t.classifier.digest_chars,e=>this.upd([`classifier`,`digest_chars`],Math.round(e)),{step:100,min:200,hint:`Qwen's digest of the user message, head and tail; the decider's is sized the same way to its own limits`})}
            <label class="switch" data-context-line><input type="checkbox" .checked=${t.classifier.context_line??!1} @change=${e=>this.upd([`classifier`,`context_line`],e.target.checked)} /> context line: the previous class, the last tools, the file types, the checker's status and a compaction, at most 120 characters out of the tail's share (no path, no file text)</label>
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
            <label class="field" data-tool-turn-effort><span>effort of a tool-result turn</span>
              <select aria-label="effort of a tool-result turn" @change=${e=>{let t=e.target.value;this.upd([`structural`,`tool_turn_effort`],t===`inherit`?void 0:t)}}>
                ${pe.map(e=>i`<option value=${e} ?selected=${e===(t.structural.tool_turn_effort??`inherit`)}>${e===`inherit`?`the class's (inherit)`:Pe(e,t,null)}</option>`)}
              </select><span class="hint">a turn whose newest message is a tool result (a screenshot a tool returned too) thinks at this level, its switch and cap; the first turn and every new user message keep the class's, the note stays the class's. A profile can set its own (Profiles › Thinking); a client's own budget wins</span></label>
            <label class="field" data-side-requests><span>a client's title or summary request</span>
              <select aria-label="a client's title or summary request" @change=${e=>this.upd([`structural`,`side_requests`],e.target.value)}>
                <option value="on" ?selected=${(t.structural.side_requests??`on`)===`on`}>a side request: no class, thinking off, short, first in the queue</option>
                <option value="off" ?selected=${t.structural.side_requests===`off`}>a turn, as any request</option>
              </select><span class="hint">known by its shape: no tools and the client's own fixed instruction (${(t.structural.side_shapes??ie).map(e=>`${e.client??`a client`}'s ${e.kind}, at most ${e.max_tokens} tokens`).join(`; `)}). It asks no classifier, gets no note and leaves the session's class alone; thinking only when the client asks for a number</span></label>
          </div>
        </div>
      </details>
      <details class="sec" data-sec="effort">
        <summary>Effort levels <span class="sec-sub">a budget and a sentence at the prompt's end; never the template's reasoning_effort</span></summary>
        <div class="sec-body">
          <p class="muted small">The template renders <code>reasoning_effort</code> as the first system tokens, so changing it per turn would cost a cold prefill. A level here sets the reasoning budget (0 = no cap below the backstop), thinking on or off (off uses <code>enable_thinking: false</code> only when the server's effort is medium, else a budget of 1), and a sentence in the tail note.</p>
          <p class="muted small">The budget is a real cap (SRV-159): when the reasoning reaches it, the server closes it with the session monitor's hidden close, so the client sees only <code>&lt;/think&gt;</code> and the answer. Defaults: low 1,024, medium 8,192, high 16,384, xhigh no cap. A profile can override a level (Profiles › Thinking); a client's own budget always wins.</p>
          ${ut.map(e=>i`<div class="row-card" data-effort=${e}>
              <div class="row-card-head"><strong class="mono">${e}</strong>
                <label class="switch"><input type="checkbox" .checked=${t.effort.levels[e].thinking} @change=${t=>this.upd([`effort`,`levels`,e,`thinking`],t.target.checked)} /> thinking</label></div>
              <div class="form-grid">${this.num(`budget (tokens, 0 = no cap)`,t.effort.levels[e].budget,t=>this.upd([`effort`,`levels`,e,`budget`],Math.round(t)),{step:256,min:0})}</div>
              <label class="field field-block"><span>sentence in the note</span><textarea rows="2" .value=${t.effort.levels[e].note} @input=${t=>this.upd([`effort`,`levels`,e,`note`],t.target.value)}></textarea></label>
            </div>`)}
          <label class="switch"><input type="checkbox" .checked=${t.effort.raise_on_fail} @change=${e=>this.upd([`effort`,`raise_on_fail`],e.target.checked)} /> a failing checker (RESULT: FAIL) raises the effort one level for that turn</label>
          ${this.renderSize(t.effort.size??he)}
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
      ${A(e)}
      ${M(e,`the router`)}
    `}renderTest(e){let t=this.result;return i`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?i`<span class="err-ink small" role="alert">${this.testError}</span>`:l}
      </div>
      ${t?t.active?i`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${O(t.class_id??null,e)} ${it(t.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${ct(t.source??null)}${t.kept?i` <span class="badge badge-warn">${t.kept}</span>`:l} <span class="muted num">${m(t.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${t.effort??`unchanged`} · thinking ${t.thinking??`on`}${t.budget==null?``:` · budget ${t.budget}`}</dd></div>
                  ${t.extra?.size?i`<div data-test-size><dt>task size</dt><dd>${Ke(t.extra.size)}</dd></div>`:l}
                  <div><dt>profile</dt><dd>${t.profile} (${t.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(t.stages??[]).map(e=>i`<div class="stage ${e.error?`is-error`:e.below?``:`is-accepted`}"><strong>${Je(e.stage)}</strong><span>${e.error??i`${e.label} ${it(e.confidence,e.accept??.6)}`}</span><span class="num muted">${m(e.ms)}</span></div>`)}</div>
              </div>
              <div>${t.note?i`<p class="muted small">The model would see this at the end of the last turn (${(t.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${t.note}</pre>`:i`<p class="muted small">No note: ${t.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:i`<p class="muted small" style="margin-top:12px">Profile ${t.profile} is raw (${t.mode}): the conductor does nothing for it.</p>`:l}
    </section>`}renderClass(e,t,n,r){let o=e.policy,s=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return i`<details class="row-card" data-class-row=${e.id} ?open=${s}>
      <summary>
        <span>${O(e.id,r,e.label)} <span class="muted small mono">${o.effort_by_size?dt.map(e=>Ue(o,e)).join(`/`):o.effort??`medium`}${(o.packages??[]).length?` · ${(o.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===0} @click=${()=>this.moveClass(t,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===n-1} @click=${()=>this.moveClass(t,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n<=1} @click=${()=>this.removeClass(t)} aria-label=${`remove ${e.id}`}>${a(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${e.id} ?readonly=${!s} @input=${e=>this.upd([`classes`,t,`id`],e.target.value)} data-class-id /><span class="hint">${s?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${e.label} @input=${e=>this.upd([`classes`,t,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${e.description} @input=${e=>this.upd([`classes`,t,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${j(ut,o.effort??`medium`,e=>this.upd([`classes`,t,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div class="toolbar size-map" style="margin:0" data-size-map=${e.id}><span class="muted small">by task size (the effort above is the ceiling)</span>
        ${dt.map(n=>i`<label class="field"><span>${n}</span><select aria-label=${`${e.id} effort when ${n}`} @change=${e=>this.setSizeLevel(t,o.effort_by_size,n,e.target.value)}>
            <option value="" ?selected=${!o.effort_by_size?.[n]}>the class's</option>
            ${ut.map(e=>i`<option value=${e} ?selected=${o.effort_by_size?.[n]===e}>${e}</option>`)}
          </select>${o.effort_by_size?.[n]&&Ue(o,n)!==o.effort_by_size[n]?i`<span class="hint">cut to ${Ue(o,n)}</span>`:l}</label>`)}
      </div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let n=(o.packages??[]).includes(e);return i`<button class="pick ${n?`is-on`:``}" aria-pressed=${n?`true`:`false`} @click=${()=>this.upd([`classes`,t,`policy`,`packages`],n?(o.packages??[]).filter(t=>t!==e):[...o.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(o.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,t,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}renderSize(e){let t=(t,n)=>this.upd([`effort`,`size`],T(e,t,n));return i`<div class="row-card" data-size-config>
      <div class="row-card-head"><strong>Task size</strong>
        <label class="switch"><input type="checkbox" .checked=${e.reset_on_compaction} @change=${e=>t([`reset_on_compaction`],e.target.checked)} /> a compaction starts the size over</label></div>
      <p class="muted small">Each signal is small below its first number, medium up to its second, large above. The task's size is the largest any counting signal reached, and it never falls within a task. A class's levels per size (in its row above) pick the effort under the class's effort; whether that acts is each profile's <em>effort by task size</em> (off, shadow, on).</p>
      ${Re.map(n=>{let r=e.signals[n.id]??he.signals[n.id];return i`<div class="size-signal" data-size-signal=${n.id}>
          <label class="switch"><input type="checkbox" .checked=${r.enabled} @change=${e=>t([`signals`,n.id,`enabled`],e.target.checked)} /> ${n.label}</label>
          <div class="form-grid">
            ${this.num(`small below (${n.unit})`,r.small,e=>t([`signals`,n.id],{...r,small:e}),{min:0,step:n.id===`session_tokens`?1e3:1})}
            ${this.num(`medium up to (${n.unit})`,r.medium,e=>t([`signals`,n.id],{...r,medium:e}),{min:0,step:n.id===`session_tokens`?1e3:1,hint:n.hint})}
          </div>
        </div>`})}
    </div>`}setSizeLevel(e,t,n,r){let i={...t??{}};r?i[n]=r:delete i[n],this.upd([`classes`,e,`policy`,`effort_by_size`],Object.keys(i).length?i:void 0)}renderContextStage(e){let t=tt(e),n=(n,r)=>this.upd([`classifier`,`context_stage`],{...t,...e.classifier.context_stage??{},[n]:r});return i`<div class="row-card" data-context-stage>
      <div class="row-card-head">
        <strong>Qwen in the conversation (in-context stage)</strong>
        ${j([`off`,`on`],t.mode,e=>n(`mode`,e),{small:!0,label:`in-context stage`})}
      </div>
      <p class="muted small">A follow-up or a long prompt is classified by the engine inside the conversation it already holds, right before the turn: its prompt up to the last message plus a hidden question, one letter per class, thinking off. The turn resumes from the anchors that prefill leaves. A short first message keeps the stages above; a request with images too.</p>
      <div class="form-grid">
        ${this.num(`long prompt from (tokens)`,t.min_tokens,e=>n(`min_tokens`,Math.round(e)),{step:1024,min:0,max:262144,hint:`a prompt at least this long goes to the stage`})}
        <label class="switch" data-context-followups><input type="checkbox" .checked=${t.followups} @change=${e=>n(`followups`,e.target.checked)} /> every follow-up (a user message after an earlier answer), whatever its length</label>
        ${this.num(`accept at (probability)`,t.accept,e=>n(`accept`,e),{step:.01,min:0,max:1,hint:`below it, the session's class`})}
      </div>
    </div>`}num(e,t,n,r={}){return i`<label class="field"><span>${e}</span><input type="number" .value=${String(t)} step=${r.step??1} min=${r.min??``} max=${r.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&n(t)}} />${r.hint?i`<span class="hint">${r.hint}</span>`:l}</label>`}text(e,t,n){return i`<label class="field"><span>${e}</span><input .value=${t} @input=${e=>n(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(T(n,[`classifier`,`stages`],r))}};t([n({attribute:!1})],P.prototype,`ids`,void 0),t([n({attribute:!1})],P.prototype,`meta`,void 0),t([n({attribute:!1})],P.prototype,`active`,void 0),t([n({attribute:!1})],P.prototype,`onSaved`,void 0),t([f()],P.prototype,`packages`,void 0),t([f()],P.prototype,`testText`,void 0),t([f()],P.prototype,`testProfile`,void 0),t([f()],P.prototype,`testTools`,void 0),t([f()],P.prototype,`testing`,void 0),t([f()],P.prototype,`result`,void 0),t([f()],P.prototype,`testError`,void 0),P=t([r(`qse-cond-routing`)],P);var ft=[`off`,`shadow`,`act`],pt=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,mt=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,F=class extends o{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new D(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),S.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(T(this.ed.draft,e,t))}render(){let e=this.ed,t=this.config?.server.monitor_mode;return i`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${e.loading&&!e.draft?c(8):e.loadError?i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${e.loadError}</div></div>`:i`
            ${t===`raw`?i`<div class="guard-line" data-monitor-off>${a(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:l}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(e.draft)}
            ${this.renderRest(e.draft)}
            ${A(e)}
            ${M(e,`the monitor`)}`}`}renderTasks(e){let t=Ce(e),n=t.filter(t=>e.tasks[t]&&/^C\d/.test(t)),r=t.filter(e=>de(e)),a=t.filter(e=>!n.includes(e)&&!r.includes(e)),o=r.map(t=>e.tasks[t]).find(e=>e.rule===`stall_watchdog`)?.notify;return i`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${n.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${n.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${r.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${r.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${o?.enabled?i`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:i`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, a design decision of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${r.map(t=>this.renderTask(t,e.tasks[t]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${a.join(`, `)}</span></summary>
        <div class="sec-body">${a.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>`}renderTask(e,t){let n=qe(this.meta?.schemas?.monitor,t.rule),r=this.meta?.tasks?.[e];return i`<div class="row-card" data-monitor-task=${e}>
      <div class="row-card-head">
        <strong>${e}${r?i` · ${r.label}`:l} <span class="muted small mono">${t.rule}</span></strong>
        ${j(ft,t.mode,t=>this.upd([`tasks`,e,`mode`],t),{small:!0,label:`${e} mode without a profile`})}
      </div>
      ${r?.why?i`<p class="muted small">${r.why}</p>`:l}
      ${n?.properties?i`<div class="form-stack">${Object.entries(n.properties).map(([n,r])=>this.field(n,r,t[n],[`tasks`,e,n]))}</div>`:i`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,t,[`tasks`,e])}`}
    </div>`}field(e,t,n,r){let o=Array.isArray(t.type)?t.type[0]:t.type,s=e.replace(/_/g,` `),c=t.description?i`<span class="hint">${t.description}</span>`:l;if(t.enum)return i`<label class="field"><span>${s}</span><select data-param=${e} @change=${e=>this.upd(r,e.target.value)}>${t.enum.map(e=>i`<option value=${String(e)} ?selected=${e===n}>${String(e)}</option>`)}</select>${c}</label>`;if(o===`boolean`)return i`<label class="switch" data-param=${e}><input type="checkbox" .checked=${!!n} @change=${e=>this.upd(r,e.target.checked)} /> ${s}</label>${t.description?i`<p class="hint">${t.description}</p>`:l}`;if(o===`integer`||o===`number`)return i`<label class="field"><span>${s}</span><input type="number" data-param=${e} step=${o===`integer`?`1`:`0.01`} min=${t.minimum??l} max=${t.maximum??l} .value=${n==null?``:String(n)} @input=${e=>{let t=e.target.value;this.upd(r,t===``?null:o===`integer`?Math.round(Number(t)):Number(t))}} />${c}</label>`;if(o===`string`){let t=String(n??``);return pt.test(e)||t.length>60?i`<label class="field field-block"><span>${s}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(t.length/90)))} class=${mt.test(e)?`mono`:``} data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)}></textarea>${c}</label>`:i`<label class="field"><span>${s}</span><input data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)} />${c}</label>`}if(o===`array`&&t.items&&(Array.isArray(t.items.type)?t.items.type[0]:t.items.type)===`string`){let a=!Array.isArray(n)&&Array.isArray(t.default),o=Array.isArray(n)?n:a?t.default:[];return i`<label class="field field-block"><span>${s} <span class="muted small">(one a line${a?`; the default, not saved yet`:``})</span></span><textarea rows=${Math.min(8,Math.max(2,o.length+1))} data-param=${e} .value=${o.join(`
`)} @input=${e=>this.upd(r,ue(e.target.value))}></textarea>${c}</label>`}if(o===`array`&&t.items?.properties){let o=Array.isArray(n)?n:[],l=t.items;return i`<fieldset class="fieldset" data-param=${e}><legend>${s} <span class="muted small">${o.length}</span></legend>${c}
        ${o.map((e,t)=>i`<div class="row-card row-card-sub" data-item=${String(e.id??t)}>
          <div class="row-card-head"><strong class="mono small">${String(e.id??`#${t+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(r,o.filter((e,n)=>n!==t))} aria-label=${`remove ${String(e.id??t+1)}`}>${a(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(l.properties).map(([n,i])=>this.field(n,i,e[n],[...r,t,n]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${e} @click=${()=>this.upd(r,[...o,ht(l)])}>${a(`plus`)} Add</button>
      </fieldset>`}if(o===`object`&&t.properties){let o=n??{};return i`<fieldset class="fieldset" data-param=${e}><legend>${s}</legend>${c}
        <div class="form-stack">${Object.entries(t.properties).map(([e,n])=>o[e]===void 0&&!(t.required??[]).includes(e)?i`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...r,e],ht(n))}>${a(`plus`)} ${e.replace(/_/g,` `)}</button>`:this.field(e,n,o[e],[...r,e]))}</div></fieldset>`}return this.jsonField(s,n,r)}jsonField(e,t,n){return i`<label class="field field-block"><span>${e} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(t,null,2)} @change=${t=>{try{let e=JSON.parse(t.target.value);if(n.length===2&&n[0]===`tasks`){let t=this.ed.draft.tasks[n[1]];this.upd(n,{...e,rule:t.rule,mode:t.mode})}else this.upd(n,e)}catch{this.ed.errors=[`${e}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(e){return i`
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
      </details>`}};t([n({attribute:!1})],F.prototype,`meta`,void 0),t([n({attribute:!1})],F.prototype,`onSaved`,void 0),t([f()],F.prototype,`config`,void 0),F=t([r(`qse-cond-monitor`)],F);function ht(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,ht(t)])):``}var I=class extends o{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=C(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await S.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let t=b(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let t=this.sessions;return e(i`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!t?i`<p class="err-ink small" role="alert">${this.error}</p>`:t==null?c(2):t.length?i`<div class="form-stack" data-watch>${t.map(e=>this.row(e))}</div>`:i`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(e){let t=e.gates.some(e=>e.noted);return i`<div class="row-card" data-watch-session=${e.session}>
      <div class="row-card-head">
        <strong class="mono small">${e.session}</strong>
        <span class="toolbar" style="margin:0">
          ${e.watch?i`<span class="badge ${e.watch.mode===`act`?`badge-on`:`badge-shadow`}">${e.watch.task} ${e.watch.mode}</span>`:l}
          ${e.client_gone?i`<span class="badge badge-warn" data-client-gone>client gone</span>`:e.running?i`<span class="badge badge-on">running</span>`:l}
          ${e.watch?.notify?i`<span class="badge badge-warn" data-notify>push on</span>`:l}
          ${t?i`<span class="badge badge-warn">gate noted</span>`:l}</span>
      </div>
      <p class="small" style="margin:4px 0">${Xe(e)}</p>
      <p class="muted small num" style="margin:0">${e.watch?`${e.watch.profile||`—`} · ${e.watch.client||`—`} · `:``}${e.requests} requests · last ${v(new Date(e.last_seen*1e3).toISOString())}${e.note_at?` · last note ${v(new Date(e.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};t([f()],I.prototype,`sessions`,void 0),t([f()],I.prototype,`monitor`,void 0),t([f()],I.prototype,`error`,void 0),I=t([r(`qse-cond-watch`)],I);var L=class extends o{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=C(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await S.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let t=b(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let t=this.events,n=[...new Set((t??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!n.includes(this.task)&&n.push(this.task),i`${e(i`Monitor journal ${this.paused?i`<span class="badge">paused</span>`:i`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!t?i`<p class="err-ink small" role="alert">${this.error}</p>`:t==null?c(4):t.length?i`<div class="feed" data-journal>${t.map(e=>ot(e,e=>this.open=e))}</div>`:i`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:i`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${n.map(e=>i`<option value=${e} ?selected=${e===this.task}>${e}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${a(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?i`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:l}`}};t([n({attribute:!1})],L.prototype,`session`,void 0),t([f()],L.prototype,`events`,void 0),t([f()],L.prototype,`monitor`,void 0),t([f()],L.prototype,`error`,void 0),t([f()],L.prototype,`task`,void 0),t([f()],L.prototype,`kind`,void 0),t([f()],L.prototype,`paused`,void 0),t([f()],L.prototype,`open`,void 0),L=t([r(`qse-cond-journal`)],L);var gt=[`off`,`low`,`medium`,`high`,`xhigh`],_t={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},vt=new Set([`S1`,`S4`]),yt=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],R=class extends o{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleContract=null,this.roleTest={},this.ed=null,this.roles=new D(this,`roles`),this.residency=new D(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),S.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),S.doc(`packages`).then(e=>this.packages=fe(e.doc)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new D(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{let e=await S.roles();this.roleHealth=e.health,this.roleContract=e.contract??null}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let t=await S.roleTest(e);this.roleTest={...this.roleTest,[e]:t.ok?`ok in ${m(t.ms)}: "${t.answer?.label}" at ${t.answer?.confidence.toFixed(2)}`:`failed: ${t.error}`}}catch(t){this.roleTest={...this.roleTest,[e]:b(t).message}}this.loadHealth()}render(){return i`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>i`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>x({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?i`<span class="badge badge-lock">${a(`warn`)} locked</span>`:i`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${v(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):i`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(T(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=T(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=T(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderUi(e,t){let n=e.ui??{states:`off`},r=n.kinds??{},a=t||n.states!==`on`,o=je(n),s=ve(n),c=(e,t,r,o,c)=>i`<label class="field"><span>${t}</span><input type="number" inputmode="numeric" min=${r} max=${o} step=${c} ?disabled=${a||s.mode!==`on`} data-ui-progress=${e} .value=${String(s[e])}
        @input=${t=>{let i=Math.round(Number(t.target.value));Number.isFinite(i)&&i>=r&&i<=o&&this.upd([`ui`],{...n,progress:{...n.progress??{},mode:s.mode,[e]:i}})}} /></label>`;return i`<details class="sec" data-sec="ui">
      <summary>What the user sees <span class="sec-sub">TandemLLM's own steps as lines the client shows</span> <span class="badge ${n.states===`on`?`badge-on`:``}" data-ui-badge>${n.states}</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0"><span class="muted small">state lines</span>${j([`off`,`on`],n.states,e=>this.upd([`ui`],{...n,states:e}),{disabled:t,small:!0,label:`state lines`})}</div>
        <p class="muted small">Each step goes out at the moment it happens as one line, <code>› deciding: code_edit, high effort</code>. The server takes them out of the reasoning and the content a client sends back, so Qwen never sees them; they are not tokens and not in the usage. Only in mode on.</p>
        <div class="toolbar" style="margin:0" data-ui-channel><span class="muted small">where</span>${j([`reasoning`,`content`,`inline`],o,e=>this.upd([`ui`],{...n,states_channel:e}),{disabled:a,small:!0,label:`where the lines go`})}</div>
        <p class="muted small">${o===`inline`?i`Each line opens a thinking block of its own and is its title, in time order: opencode shows <code>Thought: › thinking 2 min, 4,096 tokens</code> between Qwen's thinking blocks, and the answer after them. Needs <code>reasoning_content</code>; with <code>&lt;think&gt;</code> in the content (Hermes) the lines go as in the content.`:o===`content`?i`In the content, outside the thinking: before the reasoning, between its parts, after it. opencode shows them in its text part; with <code>&lt;think&gt;</code> in the content (Hermes) a line said inside the block goes out right after <code>&lt;/think&gt;</code>.`:i`In the reasoning opencode and Hermes show (a <code>reasoning_content</code> delta, or inside <code>&lt;think&gt;</code>). A profile saved before this choice existed keeps them here.`}</p>
        <div class="pkg-picks" data-ui-kinds>${Le.map(({id:e,example:t})=>{let o=r[e]!==!1;return i`<button class="pick ${o?`is-on`:``}" data-ui-kind=${e} title=${t} aria-pressed=${o?`true`:`false`} ?disabled=${a} @click=${()=>this.upd([`ui`],{...n,kinds:{...r,[e]:!o}})}>${e}</button>`})}</div>
        <div class="toolbar" style="margin:0" data-ui-progress-mode><span class="muted small">progress</span>${j([`off`,`on`],s.mode,e=>this.upd([`ui`],{...n,progress:{...n.progress??{},mode:e}}),{disabled:a,small:!0,label:`progress lines`})}</div>
        <p class="muted small">The numbers of a wait, templated: <code>waiting: opencode's title request runs first, position 1</code> while the request waits, <code>reading 35,189 tokens at ~750 tok/s, about 47 s left</code> during a long prefill, <code>thinking 4 min, 9,800 tokens</code> during a long thinking. Streams only; a request that waits opens its stream at its first waiting line.</p>
        <div class="form-grid">
          ${c(`prefill_every_s`,`a prefill line every (s)`,2,600,1)}
          ${c(`thinking_every_s`,`a thinking line every (s)`,10,3600,10)}
          ${c(`thinking_every_tokens`,`or every (tokens)`,256,262144,256)}
        </div>
        <div class="toolbar" style="margin:0" data-ui-narrator><span class="muted small">narrator</span>${j([`off`,`on`,`thoughts`,`live`],n.narrator??`off`,e=>this.upd([`ui`],{...n,narrator:e}),{disabled:a,small:!0,label:`narrator`})}</div>
        <p class="muted small">A small model inside the engine (the <code>narrator</code> role, Qwen3.5) writes in plain words. <b>live</b>: a running line of what Qwen is doing right now, every ~25 s of thinking (“thinking 3 min: how the enemies patrol …”), what a thought decided, what a long write is for and how far it is, a plan in the answer; the other lines in plain words and the user’s language; the facts (times, files, commands) come from the engine, never from the model. <b>thoughts</b>: after each stretch of thinking (and at the long-thinking marks) one line with what Qwen thought about and decided, e.g. “Thought 1m 43s: planned the level layout …”, read from the thought alone; the other lines stay templates. <b>on</b>: it writes every line instead of the template. Qwen waits while it writes, at most 3 % of its time (the rest at the turn’s end); a line it cannot write in time is the template. It loads on its first line; the Roles section below holds its budget and mode.</p>
      </div>
    </details>`}renderToolTurn(e,t){let n=e.thinking.tool_turn_effort,r=Fe(this.router,null).level,a=Fe(this.router,e);return i`<label class="field ${n?``:`is-inherited`}" data-tool-turn-effort><span>effort of a tool-result turn</span>
      <select aria-label="effort of a tool-result turn" ?disabled=${t} @change=${e=>{let t=e.target.value;if(!this.ed?.draft)return;let n={...this.ed.draft.thinking,tool_turn_effort:t};t||delete n.tool_turn_effort,this.ed.set(T(this.ed.draft,[`thinking`],n))}}>
        <option value="" ?selected=${!n}>Routing's: ${r===`inherit`?`the class's`:r}</option>
        ${pe.map(e=>i`<option value=${e} ?selected=${n===e}>${e===`inherit`?`the class's (inherit)`:e}</option>`)}
      </select>
      <span class="hint">${Pe(a.level,this.router,e)}${a.source===`profile`?` (this profile)`:a.source===`router`?` (Routing)`:``}: a turn whose newest message is a tool result thinks at this level, its switch and cap; the first turn and every new user message keep the class's</span></label>`}renderBudgets(e,t){return i`<div data-budgets>
      <div class="muted small">Thinking budget per level <span class="muted">(reasoning tokens; empty keeps Routing's level, 0 is no cap below the backstop)</span></div>
      <div class="form-grid">
        ${Oe.map(n=>{let r=e.thinking.budgets?.[n],a=Ge(n,this.router,null).cap,o=Ge(n,this.router,e);return i`<label class="field ${r==null?`is-inherited`:``}" data-budget=${n}><span class="mono">${n}</span>
            <input type="number" inputmode="numeric" min="0" max=${Qe} step="1024" ?disabled=${t} placeholder=${a==null?``:String(a)} .value=${r==null?``:String(r)}
              @change=${e=>{let t=e.target.value.trim();this.ed?.draft&&this.ed.set(T(this.ed.draft,[`thinking`],ke(this.ed.draft.thinking,n,t===``?null:Number(t))))}} />
            <span class="hint">${oe(o.cap)}${o.source===`router`?` (Routing)`:o.source===`profile`?` (this profile)`:``}</span></label>`})}
      </div>
    </div>`}renderEditor(e){if(e.loading&&!e.draft)return c(8);if(e.loadError)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=!!e.locked,r=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`hermes`,`curl`,`dashboard`,`chat-app`,`other`],o=this.router?.classes??[];return i`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${a(`playground`)} Try in Playground</a></div>
      ${n?i`<div class="note" data-lock-reason>${a(`warn`)} ${e.locked}. Every control below is read-only.</div>`:l}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${t.description} ?disabled=${n} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${j([`raw`,`shadow`,`on`],t.mode,e=>this.upd([`mode`],e),{disabled:n,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${r.map(e=>{let r=t.clients.includes(e);return i`<button class="pick ${r?`is-on`:``}" ?disabled=${n} aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`clients`],r?t.clients.filter(t=>t!==e):[...t.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${n} @change=${e=>this.upd([`default_class`],e.target.value)}>${o.map(e=>i`<option value=${e.id} ?selected=${e.id===t.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(_t).map(e=>i`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${j([`off`,`shadow`,`on`],t.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:n,small:!0,label:e})}<span class="small">${_t[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${o.map(e=>{let r=t.routing[e.id]??{},a=Object.keys(r).length>0;return i`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${O(e.id,this.ids,e.label)}
                ${a?i`<button class="btn btn-ghost btn-sm" ?disabled=${n} @click=${()=>{let n={...t.routing};delete n[e.id],this.upd([`routing`],n)}}>use the router's</button>`:i`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${j(gt,r.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...r,effort:t}),{disabled:n,small:!0,label:`effort override of ${e.id}`,cls:r.effort?``:`is-inherited`})}</div>
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
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${j(gt,t.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:n,small:!0,label:`default effort`})}</div>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.thinking.hints} @change=${e=>this.upd([`thinking`,`hints`],e.target.checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <div class="toolbar" style="margin:0" data-by-size><span class="muted small">effort by task size</span>${j([`off`,`shadow`,`on`],t.thinking.by_size??`shadow`,e=>this.upd([`thinking`,`by_size`],e),{disabled:n,small:!0,label:`effort by task size`,cls:t.thinking.by_size?``:`is-inherited`})}</div>
          <p class="muted small">Effort by task size (SRV-157): a class with levels per size (Routing › the class) thinks at its size's level, never above the class's effort; the size only rises within a task. <strong>off</strong>: the class's effort; <strong>shadow</strong>: the class's effort, and the journal shows the size and the effort it would set; <strong>on</strong>: the size picks.</p>
          ${this.renderToolTurn(t,n)}
          ${this.renderBudgets(t,n)}
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="library">
        <summary>Package library <span class="sec-sub">a searched skill for a new message or a stuck run (R1)</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0" data-library-mode><span class="muted small">library</span>${j([`off`,`shadow`,`on`],t.library??`off`,e=>this.upd([`library`],e),{disabled:n,small:!0,label:`package library`,cls:t.library?``:`is-inherited`})}</div>
          <p class="muted small">On a new user message, and on a low-quality or stuck outcome, the router searches the library (the <a href="#/conductor?screen=packages">Packages</a> screen) with the message, asks the package question among the best matches and puts the skill that fits at the prompt's tail. <strong>off</strong>: nothing; <strong>shadow</strong>: the journal shows the pick, nothing goes in; <strong>on</strong>: the skill and the card that tells the model how to search the library itself.</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113); a pack that is off turns its task off</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${yt.map(([e,r,a])=>i`<label class="switch" data-pack=${e}><input type="checkbox" ?disabled=${n} .checked=${t.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${a} <span class="muted small">(${r}, ${t.toolgate[e]?t.tasks[r]?.mode??`off`:`off`})</span></label>`)}
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
        <p class="muted small" data-notify-rule>A design decision of 2026-09-30: off by default, on in the overnight profile. Only S4 in act pushes, never in shadow, once a turn end; the token is a file on the box. S2 (re-read after a compaction) has no number here: its texts and state files are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      ${this.renderUi(t,n)}
      <details class="sec" data-sec="hints">
        <summary>Speed hints <span class="badge badge-warn">not gated</span></summary>
        <div class="sec-body">
          <p class="warn-ink small">No SPD measurement is in the journal yet (SPD-73); the router's ≤ 1 ulp text rule and the never-worse gate come first.</p>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.stair_width} @change=${e=>this.upd([`hints`,`stair_width`],e.target.checked)} /> StairCut width hint</label>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.prefix_retention} @change=${e=>this.upd([`hints`,`prefix_retention`],e.target.checked)} /> prefix retention hint</label>
        </div>
      </details>
      ${A(e)}
      ${M(e,`profile ${e.name}`)}
    </div>`}renderTask(e,t,n,r,o){let s=this.meta?.tasks[e],c=s?.role&&s.tier===`1`,u=!c||!!this.roleHealth[s.role]?.up;return i`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${s?.family??``}</div></div>
      <div>${s?.label??e}<div class="task-why">${s?.why??``}</div>${c&&!u?i`<div class="warn-ink small">${a(`warn`)} the ${s.role} role is offline: act is refused</div>`:l}</div>
      <div class="task-mode">${j([`off`,`shadow`,`act`],t,t=>this.upd([`tasks`,e,`mode`],t),{disabled:o,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${vt.has(e)&&Object.keys(r).length?i`<p class="muted small" data-in-context=${e}>${Object.entries(r).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:l}
        ${Object.entries(vt.has(e)?{}:r).map(([t,n])=>{let r=s?.thresholds[t];return i`<label class="field ${r&&r.default!==n?`is-changed`:``}"><span>${t.replace(/_/g,` `)} <span class="hint">${r?.unit??``}${r?`, default ${r.default}`:``}</span></span>
            <input type="number" ?disabled=${o} .value=${String(n)} step="any" @input=${n=>{let r=Number(n.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,t],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${o} min="0" max="10" .value=${String(n)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let e=this.roles;if(!e.draft)return e.loadError?i`<p class="err-ink small">${e.loadError}</p>`:l;let t=e.draft,n=(n,r)=>e.set(T(t,n,r)),r=ne(e.base?.meta.defaults),o=this.roleContract?.engine_tokenizer_sha256??null,s=this.meta?.roles,c=s?.kinds??Ze,u=s?.windows??ye,d=s?.fallbacks??[`PASS`,`stale`,`skip`,`next-role`],f=ae.filter(e=>!s||s.writes.includes(e));return i`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${e.base?.meta.defaults?.length?i`<div class="note" data-role-defaults>${a(`warn`)} Version ${e.base.version} was stored before the role contract and loaded with ${e.base.meta.defaults.length} default${e.base.meta.defaults.length===1?``:`s`} (marked <span class="badge">default</span> below; the journal has the list). Save the roles to make them explicit.</div>`:l}
        <p class="muted small">The contract (SRV-133): when a role may use the GPU, whether a Qwen request cuts it, how long its caller waits before the fallback, what it may write, and the tokenizer it reads, which decides what it may share with Qwen. Engine tokenizer: <code class="mono" data-engine-sha>${o?`${o.slice(0,12)}…`:`not recorded`}</code></p>
        ${Object.entries(t).map(([e,t])=>{let s=this.roleHealth[e],p=new Set((r[e]??[]).map(e=>e.split(`=`)[0])),h=t=>p.has(t)?i` <span class="badge" data-default=${`${e}.${t}`}>default</span>`:l,g=et(t.kind,t.window),_=$e(t.kind,t.writes),v=Ie(t,o);return i`<div class="row-card" data-role-row=${e}>
            <div class="row-card-head"><strong>${e}</strong>
              <span class="toolbar" style="margin:0">${s?i`<span class="badge ${s.up?`badge-ok`:t.mode===`off`?``:`badge-warn`}">${s.up?`up ${m(s.ms)}`:t.mode===`off`?`off`:`down: ${s.error??`?`}`}</span>`:l}
                <button class="btn btn-sm" @click=${()=>this.testRole(e)} data-role-test=${e}>Test</button></span></div>
            ${this.roleTest[e]?i`<p class="small" role="status">${this.roleTest[e]}</p>`:l}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${t.endpoint} @input=${t=>n([e,`endpoint`],t.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${t.model} @input=${t=>n([e,`model`],t.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(t.timeout_ms)} @input=${t=>n([e,`timeout_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>memory claim (GiB, its peak)${h(`memory_claim_gib`)}</span><input type="number" step="0.1" min="0" data-role-field="memory_claim_gib" .value=${String(t.memory_claim_gib)} @input=${t=>n([e,`memory_claim_gib`],Number(t.target.value))} /></label>
            </div>
            <div class="form-grid" data-role-contract=${e}>
              <label class="field"><span>kind${h(`kind`)}</span><select data-role-field="kind" @change=${t=>n([e,`kind`],t.target.value)}>${c.map(e=>i`<option value=${e} ?selected=${e===t.kind}>${e}</option>`)}</select></label>
              <label class="field"><span>window${h(`window`)}</span><select data-role-field="window" @change=${t=>n([e,`window`],t.target.value)}>${u.map(e=>i`<option value=${e} ?selected=${e===t.window} ?disabled=${e!==t.window&&!!et(t.kind,e)} title=${me[e]}>${e}</option>`)}</select></label>
              <label class="field"><span>latency budget (ms)${h(`latency_budget_ms`)}</span><input type="number" min="1" max="600000" data-role-field="latency_budget_ms" .value=${String(t.latency_budget_ms)} @input=${t=>n([e,`latency_budget_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>fallback (off, slow, unsure or refused)${h(`fallback`)}</span><select data-role-field="fallback" @change=${t=>n([e,`fallback`],t.target.value)}>${d.map(e=>i`<option value=${e} ?selected=${e===t.fallback}>${e}</option>`)}</select></label>
              <label class="field"><span>writes${h(`writes`)}</span><select data-role-field="writes" @change=${t=>n([e,`writes`],t.target.value)}>${f.map(e=>i`<option value=${e} ?selected=${e===t.writes} ?disabled=${e!==t.writes&&!!$e(t.kind,e)}>${e.replace(/_/g,` `)}</option>`)}</select></label>
              <label class="field field-wide"><span>tokenizer sha256 (the engine records its own; none: text only)${h(`tokenizer_sha256`)}</span><input class="mono" data-role-field="tokenizer_sha256" spellcheck="false" .value=${t.tokenizer_sha256} @input=${t=>n([e,`tokenizer_sha256`],t.target.value.trim().toLowerCase()||`none`)} /></label>
            </div>
            <div class="toolbar" style="margin:0">
              <label class="switch"><input type="checkbox" data-role-field="preemptible" .checked=${t.preemptible} @change=${t=>n([e,`preemptible`],t.target.checked)} /> preemptible: a Qwen request cuts it, it resumes later${h(`preemptible`)}</label>
            </div>
            <div><span class="muted small">sharing with Qwen${h(`sharing`)}</span>
              <div class="pkg-picks" data-role-sharing=${e}>${v.map(r=>{let a=t.sharing.includes(r.form);return i`<button class="pick ${a?`is-on`:``}" aria-pressed=${a?`true`:`false`} ?disabled=${!a&&!r.allowed} title=${r.why??``} data-share=${r.form} @click=${()=>n([e,`sharing`],a?t.sharing.filter(e=>e!==r.form):[...t.sharing,r.form])}>${r.label}</button>`})}</div>
              ${v.filter(e=>!e.allowed).map(e=>i`<p class="muted small" data-share-why=${e.form}>${e.label}: ${e.why}</p>`)}
            </div>
            ${g?i`<p class="warn-ink small" data-role-why="window">${a(`warn`)} ${g}</p>`:l}
            ${_?i`<p class="warn-ink small" data-role-why="writes">${a(`warn`)} ${_}</p>`:l}
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${j([`off`,`shadow`,`on`],t.mode,t=>n([e,`mode`],t),{small:!0,label:`${e} mode`})}<span class="muted small mono">${t.contract}</span></div>
          </div>`})}
        ${A(e)}
        ${M(e,`the roles`)}
      </div>
    </details>`}renderResidency(){let e=this.residency;if(!e.draft)return e.loadError?i`<p class="err-ink small">${e.loadError}</p>`:l;let t=e.draft,n=(n,r)=>e.set(T(t,n,r)),r=(e,r=`0.1`,a=!1)=>i`<input type="number" step=${r} min="0" .value=${String(t[e[0]])} @input=${t=>{let r=Number(t.target.value);n(e,a?Math.round(r):r)}} data-res=${e[0]} />`,o=[`measured`,`peak`,`declared`];return i`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${r([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${r([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${r([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${t.order.join(`, `)} @change=${e=>n([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(t.claims).map(([e,r])=>i`<div class="row-card" data-claim=${e}>
            <div class="row-card-head"><strong>${e}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[e]:r,...i}=t.claims;n([`claims`],i)}} aria-label=${`remove the claim of ${e}`}>${a(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(r.gib)} @input=${t=>n([`claims`,e,`gib`],Number(t.target.value))} data-claim-gib=${e} /></label>
              <label class="field"><span>basis</span><select @change=${t=>n([`claims`,e,`basis`],t.target.value)}>${o.map(e=>i`<option value=${e} ?selected=${e===r.basis}>${e}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${r.note} @input=${t=>n([`claims`,e,`note`],t.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in t.claims} @click=${()=>{n([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${a(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${A(e)}
        ${M(e,`the residency limits`)}
      </div>
    </details>`}};t([n({attribute:!1})],R.prototype,`ids`,void 0),t([n({attribute:!1})],R.prototype,`meta`,void 0),t([n({attribute:!1})],R.prototype,`tiles`,void 0),t([n({attribute:!1})],R.prototype,`selected`,void 0),t([n({attribute:!1})],R.prototype,`onSaved`,void 0),t([f()],R.prototype,`router`,void 0),t([f()],R.prototype,`packages`,void 0),t([f()],R.prototype,`roleHealth`,void 0),t([f()],R.prototype,`roleContract`,void 0),t([f()],R.prototype,`roleTest`,void 0),t([f()],R.prototype,`newClaim`,void 0),R=t([r(`qse-cond-profiles`)],R);var z=[`package`,`task_skill`,`test_skill`,`card`,`library`],B={package:`package`,task_skill:`task skill`,test_skill:`test skill`,card:`card`,library:`library skill`},bt=3.5,xt={library:350,library_card:200},St={q:``,category:`all`,cls:``,enabled:`all`,over:!1};function Ct(e){return Math.floor(e.length/bt)}function wt(e,t){return t?t.category:e.library?`library`:e.kind===`skill`?`task_skill`:`package`}function Tt(e,t,n,r){let i=r?.limits??xt,a=n===`library`?i.library:e===(r?.library.card??`tandem-library`)?i.library_card:null;return a!=null&&a<=t.max_tokens?{limit:a,source:`house`}:{limit:t.max_tokens,source:`max_tokens`}}function Et(e,t){return(e?.classes??[]).filter(e=>(e.policy.packages??[]).includes(t)).map(e=>e.id)}function Dt(e,t,n,r){return Object.entries(e?.packages??{}).map(([e,i])=>{let a=t?.packages[e]??null;return{name:e,pkg:i,entry:a,category:wt(i,a),classes:Et(r,e),usage:n?.[e]??null,score:null}}).sort((e,t)=>z.indexOf(e.category)-z.indexOf(t.category)||e.name.localeCompare(t.name))}function Ot(e){return e.toLowerCase().split(/\s+/).filter(Boolean)}function kt(e,t){let n=`${e.name} ${e.pkg.title} ${e.pkg.description} ${e.pkg.source}`.toLowerCase();return Ot(t).every(e=>n.includes(e))}function At(e,t){return e.entry?e.entry.over:e.pkg.source===`inline`&&Ct(V(e.pkg.body,!!e.pkg.library)[0])>Tt(e.name,e.pkg,e.category,t).limit}function jt(e,t,n,r=null){let i=new Map((n??[]).map(e=>[e.name,e.score])),a=t.q.trim(),o=e.filter(e=>t.category===`all`||e.category===t.category).filter(e=>!t.cls||e.classes.includes(t.cls)||e.category===`library`&&(e.entry?.fits??[]).includes(t.cls)).filter(e=>t.enabled===`all`||t.enabled===`on`===e.pkg.enabled).filter(e=>!t.over||At(e,r)).filter(e=>!a||kt(e,a)||e.category===`library`&&i.has(e.name)).map(e=>({...e,score:a&&e.category===`library`?i.get(e.name)??null:null}));return a?o.sort((e,t)=>z.indexOf(e.category)-z.indexOf(t.category)||(e.score!=null||t.score!=null)&&(t.score??-1)-(e.score??-1)||e.name.localeCompare(t.name)):o}function Mt(e,t){let n=e=>!!t?.packages[e]?.general;return[...e.filter(n),...e.filter(e=>!n(e))]}function Nt(e,t,n,r){let i=e.classes.findIndex(e=>e.id===t);if(i<0)return{router:e,error:`no class ${t}`};if(r?.packages[n]?.library)return{router:e,error:`${n} is a library skill: the router searches the library, a class does not list it`};let a=e.classes[i].policy.packages??[];if(a.includes(n))return{router:e,error:null};if(a.length>=12)return{router:e,error:`${t} already lists 12 packages, the most a class may`};let o=e.classes.map((e,t)=>t===i?{...e,policy:{...e.policy,packages:Mt([...a,n],r)}}:e);return{router:{...e,classes:o},error:null}}function Pt(e,t,n){return{...e,classes:e.classes.map(e=>e.id===t?{...e,policy:{...e.policy,packages:(e.policy.packages??[]).filter(e=>e!==n)}}:e)}}var Ft=/^## Reference\b/m;function V(e,t){let n=t?Ft.exec(e):null;return n?[e.slice(0,n.index).trim(),e.slice(n.index).trim()]:[e.trim(),``]}function It(e){let t=1;for(;e.packages[`package-${t}`];)t++;return`package-${t}`}function Lt(e){return{title:`Package ${e}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0}}function Rt(e,t,n){return{...e,source:`inline`,body:n?`${t}\n\n${n}`:t}}function zt(e,t){return t?e.filter(e=>e.profile===t):e}function Bt(e){return e===`on`?`the router puts the skill it picks into the prompt from the next request, and the library card on the coding classes with tools`:e===`shadow`?`the router searches and picks, and the journal shows the pick; nothing goes into the prompt`:`the router does not search the library`}var H=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.catalog=null,this.usage=null,this.onChanged=()=>void 0,this.profile=``,this.busy=null,this.msg=null,this.error=null}async setMode(e,t){this.busy=e,[this.msg,this.error]=[null,null];try{let n=await S.doc(`profiles`,e),r={...n.doc,library:t},a=await S.check(`profiles`,e,r);if(a.errors.length){this.error=`${e}: ${a.errors.join(`; `)}`;return}if(!await k(`Library ${t} for ${e}?`,[`With ${t}, ${Bt(t)}.`,i`The server checked the change (live v${n.version}):${at(a.changes)}`,`Saving makes a new version of the profile with the note “library to ${t}”.`],`Save the new version`))return;let o=await S.put(`profiles`,e,r,`library to ${t}`,n.version);this.msg=`${e}: library ${t}, version ${o.version}`,this.onChanged()}catch(e){this.error=b(e).message}finally{this.busy=null}}render(){let e=this.catalog;if(!e||!this.usage)return c(5);let t=this.usage.library,n=Object.values(e.packages).filter(e=>e.category===`library`).length,r=e=>t.rows.filter(t=>t.profile===e&&t.chosen.length).length,o=Object.keys(e.library.profiles).sort();return i`
      <section class="panel" data-lib-profiles>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Retrieval per profile <span class="muted">· ${n} library skills</span></h2>
          <p class="panel-sub">On a new user message, and on a low-quality or stuck outcome, the router searches the library with the message, asks the package question among the best matches and, with <strong>on</strong>, puts the skill that fits at the prompt's tail. <strong>shadow</strong> only journals the pick. Judge the picks below before a profile goes on.</p></div></header>
        ${this.error?i`<p class="err-ink small" role="alert" data-lib-error>${this.error}</p>`:l}
        ${this.msg?i`<p class="status small" role="status" data-lib-saved>${a(`check`)} ${this.msg}</p>`:l}
        <div class="lib-profiles">
          ${o.map(t=>{let n=this.tiles.find(e=>e.name===t),a=e.library.profiles[t];return i`<div class="lib-profile" data-lib-profile=${t}>
              <span class="lib-profile-name"><strong>${t}</strong>${n?.locked?i` <span class="badge badge-lock">locked</span>`:l}<span class="muted small">${n?.clients.join(`, `)||`no client`}${n?` · profile ${n.mode}`:``}</span></span>
              ${j([`off`,`shadow`,`on`],a,e=>void this.setMode(t,e),{small:!0,label:`library of ${t}`,disabled:!!n?.locked||this.busy===t})}
              <span class="muted small num">${r(t)} pick${r(t)===1?``:`s`} in the rows below</span>
            </div>`})}
        </div>
      </section>
      <section class="panel" data-lib-shadow>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">What the harness picked</h2>
          <p class="panel-sub">The router's retrieval in the journal (extra.library), newest first: the candidates the search found, the skill the package question picked, and whether it went in (on) or would have (shadow). A pick opens the skill.</p></div>
          <div class="panel-tools"><label class="field"><span>profile</span><select data-lib-filter @change=${e=>this.profile=e.target.value}>
            <option value="" ?selected=${!this.profile}>all</option>${o.map(e=>i`<option value=${e} ?selected=${this.profile===e}>${e}</option>`)}
          </select></label></div></header>
        <div class="stats" data-lib-tiles>
          ${u({label:`messages searched`,value:p(t.searched.day),unit:`in 24 h`,sub:`${p(t.searched.week)} in 7 days`})}
          ${u({label:`with a pick`,value:p(t.picked.day),unit:`in 24 h`,sub:`${p(t.picked.week)} in 7 days`})}
          ${u({label:`pick rate`,value:t.searched.week?`${Math.round(100*t.picked.week/t.searched.week)}`:`—`,unit:t.searched.week?`%`:``,sub:`7 days`})}
          ${u({label:`most picked`,value:t.top[0]?.name??`—`,sub:t.top.length?t.top.slice(0,3).map(e=>`${e.name} ${e.picks}`).join(` · `):`no pick yet`})}
        </div>
        ${this.rows(zt(t.rows,this.profile))}
      </section>`}rows(e){if(!e.length)return s(`No retrieval in the journal`,`A profile with the library in shadow or on journals each search here; with every profile off nothing is searched.`);let t=e=>x({ptab:null,pkg:e,pview:null});return i`<div class="lib-rows" data-lib-rows>${e.map(e=>i`<div class="lib-row" data-lib-row=${e.id}>
        <div class="lib-row-head">
          <span class="num small">${v(new Date(e.ts_ms).toISOString())}</span>
          <span class="badge ${e.mode===`on`?`badge-on`:`badge-shadow`}">${e.mode??`—`}</span>
          ${O(e.class_id,this.ids)}
          <span class="muted small">${e.profile??`—`} · ${e.trigger??`—`}${e.query_chars==null?``:` · ${p(e.query_chars)} characters searched`}${e.cached?` · asked before`:``}</span>
        </div>
        <div class="lib-cands">${e.candidates.length?e.candidates.map(t=>i`<span class="lib-cand ${t.name&&e.chosen.includes(t.name)?`is-chosen`:``}">${t.name}<span class="num muted">${t.score==null?``:t.score.toFixed(2)}</span></span>`):i`<span class="muted small">no candidate shares a word with the message</span>`}</div>
        <div class="lib-pick small">${e.chosen.length?i`${e.mode===`on`&&e.applied?.length?`went in:`:`would have gone in:`} ${e.chosen.map(e=>i`<button class="btn btn-sm lib-open" data-lib-open=${e} @click=${()=>t(e)}>${e}</button>`)} <span class="muted">(${e.how??`—`})</span>`:i`<span class="muted">no pick (${(e.how??`none`).replace(/_/g,` `)})</span>`}</div>
      </div>`)}</div>`}};t([n({attribute:!1})],H.prototype,`ids`,void 0),t([n({attribute:!1})],H.prototype,`tiles`,void 0),t([n({attribute:!1})],H.prototype,`catalog`,void 0),t([n({attribute:!1})],H.prototype,`usage`,void 0),t([n({attribute:!1})],H.prototype,`onChanged`,void 0),t([f()],H.prototype,`profile`,void 0),t([f()],H.prototype,`busy`,void 0),t([f()],H.prototype,`msg`,void 0),t([f()],H.prototype,`error`,void 0),H=t([r(`qse-cond-library`)],H);var Vt={mode:`off`,idle_s:180,due_idle_s:3,min_new_tokens:1500,max_tokens:600,max_blocks:4,gate:!0,after_compaction:`idle`,turn_deadline_s:30,existing:5},Ht={index_tokens:1500,body_tokens:600,pinned_tokens:300,bodies:3,min_score:0,min_words:2,dedupe_threshold:.6,intent:`shadow`,provisional:!0,digest_chars:2e3,writer:Vt},Ut={rule:`how I want things done`,ruling:`a decision that was taken`,gotcha:`what went wrong once`,runbook:`how to do a thing`,program_state:`where the work stands`,reference:`a fact to look up`,registry:`a list of things`,archive:`kept for the record`},Wt=Object.keys(Ut),U={"qwen-inline":`the model, mid-answer`,"qwen-writer":`the model, in the background`,api:`the model's script`,human:`you`},Gt={node:null,kind:``,status:``,writer:``,stale:!1,q:``};function Kt(e,t){let n=t.q.trim().toLowerCase();return e.filter(e=>(!t.node||e.node===t.node)&&(!t.kind||e.kind===t.kind)&&(!t.status||e.status===t.status)&&(!t.writer||e.writer===t.writer)&&(!t.stale||!!e.stale)&&(!n||`${e.title} ${e.tags.join(` `)} ${e.id}`.toLowerCase().includes(n)))}function qt(e){return e.scope===`user`?`About you`:e.scope===`project`?e.label??e.node.replace(/^projects\//,``):`session ${e.label??e.node.replace(/^sessions\//,``)}`}function Jt(e,t){let n=e.split(`
`),r=t.split(`
`),i=n.length,a=r.length;if(i*a>4e5)return[...n.map(e=>({op:`-`,text:e})),...r.map(e=>({op:`+`,text:e}))];let o=Array.from({length:i+1},()=>Array(a+1).fill(0));for(let e=i-1;e>=0;e--)for(let t=a-1;t>=0;t--)o[e][t]=n[e]===r[t]?o[e+1][t+1]+1:Math.max(o[e+1][t],o[e][t+1]);let s=[],c=0,l=0;for(;c<i&&l<a;)n[c]===r[l]?(s.push({op:`=`,text:n[c]}),c++,l++):o[c+1][l]>=o[c][l+1]?s.push({op:`-`,text:n[c++]}):s.push({op:`+`,text:r[l++]});for(;c<i;)s.push({op:`-`,text:n[c++]});for(;l<a;)s.push({op:`+`,text:r[l++]});return s}function Yt(e){let t=e.detail??{};if(e.task===`nightly`)return`nightly: ${t.deleted??0} archives deleted, ${t.duplicates??0} duplicates, ${t.stale??0} stale`;let n=typeof t.why==`string`?t.why:null,r=String(t.op??e.decision??`write`),i=e.task===`api_write`?`the model's script`:U[String(t.writer??`qwen-inline`)]??String(t.writer);return n||e.decision&&![`add`,`update`,`archive`,`propose`,`ok`].includes(e.decision)?`refused (${n??e.decision}): ${r} by ${i}`:`${r===`add`?t.revised?`revised`:`added`:r===`update`?`updated`:r===`archive`?`archived`:r===`propose`?`proposed`:r} ${[t.kind,t.scope?`(${t.scope})`:null].filter(Boolean).join(` `)} by ${i}${t.id?` · ${t.id}`:``}`.replace(/\s+/g,` `).trim()}function Xt(e){let t=e?.memory?.notes??{};return{...Ht,...t,writer:{...Vt,...t.writer??{}}}}function Zt(e){let t=e.detail??{},n=Array.isArray(t.ops)?t.ops:[],r=e=>n.filter(t=>t.op===e&&(e!==`add`||!t.revised)).length,i=n.filter(e=>e.revised).length,a=[r(`add`)?`${r(`add`)} added`:``,i?`${i} revised`:``,r(`update`)?`${r(`update`)} updated`:``,r(`archive`)?`${r(`archive`)} archived`:``].filter(Boolean),o=t.gate??{},s=n.length&&n.every(e=>e.stored===!1)?` (shadow: not stored)`:``;return{when:e.ts_ms,trigger:String(t.trigger??`—`),outcome:typeof t.skipped==`string`?t.skipped:n.length?`wrote`:t.none?`nothing worth keeping`:`no block`,gate:o.answer?String(o.answer):String(o.how??`—`),seconds:typeof t.ms==`number`?Math.round(t.ms/100)/10:null,tokens:typeof t.tokens==`number`?t.tokens:null,ops:(a.join(`, `)||`—`)+s,refused:Array.isArray(t.refused)?t.refused:[]}}function Qt(e,t){let n=Object.values(e).reduce((e,t)=>e+t,0);if(!n)return`not injected yet`;let r=e[t]??0;return`${n} injection${n===1?``:`s`}${r?` (${r} today)`:``}`}var $t=e=>e?v(e):`—`,W=class extends o{constructor(...e){super(...e),this.onChanged=null,this.list=null,this.open=null,this.error=null,this.flash=null,this.busy=!1,this.status=`open`}connectedCallback(){super.connectedCallback(),this.load()}async load(){try{this.list=(await S.proposals()).proposals,this.error=null}catch(e){this.error=b(e).message}}async show(e){try{this.open=(await S.proposal(e.name,e.n)).proposal}catch(e){this.error=b(e).message}}async apply(e){if(await k(`Apply the proposal to ${e.name}?`,[e.title,`The text becomes a new version of the packages document; every session gets it from its next note. The versions panel restores the old text.`],`Apply`)){this.busy=!0;try{let t=await S.proposalApply(e.name,e.n);this.flash=`${e.name}: packages version ${t.version} (was ${t.previous})`,this.onChanged?.(),await this.load(),await this.show(e)}catch(e){this.error=b(e).message}finally{this.busy=!1}}}async reject(e){let t=window.prompt(`Why is this proposal rejected?`,`not what the skill needs`);if(t!==null){this.busy=!0;try{await S.proposalReject(e.name,e.n,t||`rejected in the dashboard`),this.flash=`${e.name}#${e.n} rejected`,await this.load(),await this.show(e)}catch(e){this.error=b(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.list)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${this.error}</div></div>`;if(!this.list)return c(4);let t=this.list.filter(e=>this.status===`all`||e.status===`open`),n=e=>i`<span class="badge ${e===`open`?`badge-shadow`:e===`applied`?`badge-ok`:``}">${e}</span>`;return i`
      ${this.flash?i`<p class="ok-ink small" role="status" data-prop-flash>${a(`check`)} ${this.flash}</p>`:l}
      ${this.error?i`<p class="err-ink small" role="alert">${a(`warn`)} ${this.error}</p>`:l}
      ${this.open?this.renderOne(this.open):l}
      ${e(`Proposals`,t.length?i`<div class="table-scroll"><table class="table" data-proposals><thead><tr><th>package</th><th>change</th><th>mode</th><th>status</th><th>evidence</th><th>proposed</th></tr></thead><tbody>${t.map(e=>i`<tr class="req-row ${this.open&&this.open.name===e.name&&this.open.n===e.n?`is-open`:``}" data-proposal=${`${e.name}#${e.n}`} @click=${()=>this.show(e)}>
                <td class="mono small">${e.name}#${e.n}</td><td class="cell-wrap">${e.title}</td><td class="small">${e.mode}</td><td>${n(e.status)}</td>
                <td class="small">${e.evidence.length?`${e.evidence.length} outcome row${e.evidence.length===1?``:`s`}`:i`<span class="muted">no evidence</span>`}</td><td class="num small">${$t(e.created)}</td></tr>`)}</tbody></table></div>`:s(this.status===`open`?`No open proposal`:`No proposal yet`,`The model proposes a change to a package with a hidden block when a session showed a gap; it waits here for your click.`),{sub:`changes to packages and skills the model proposed; none goes live without Apply`,tools:i`<label class="switch small"><input type="checkbox" .checked=${this.status===`all`} @change=${e=>this.status=e.target.checked?`all`:`open`} data-prop-all /> applied and rejected too</label>
            <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the proposals">${a(`refresh`)}</button>`,id:`proposals`})}`}renderOne(e){let t=e.live===null?[]:Jt(e.live,e.proposed);return i`<section class="panel" style="margin-bottom:16px" data-proposal-open=${`${e.name}#${e.n}`}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e.name}#${e.n}: ${e.title}</h2>
        <p class="panel-sub">${e.mode===`append`?`adds a section`:`replaces the text`} · ${e.status}${e.applied_version?` as version ${e.applied_version}`:``}${e.rejected_reason?`: ${e.rejected_reason}`:``} · ${e.evidence_word}${e.session?i` · <a class="mono" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(e.session)}`}>${e.session}</a>`:l}</p></div>
        <div class="panel-tools">
          ${e.status===`open`?i`<button class="btn btn-primary btn-sm" ?disabled=${!e.applies||this.busy} @click=${()=>this.apply(e)} data-prop-apply>Apply</button>
            <button class="btn btn-sm" ?disabled=${this.busy} @click=${()=>this.reject(e)} data-prop-reject>Reject</button>`:l}
          <button class="btn btn-ghost btn-icon" aria-label="Close the proposal" @click=${()=>this.open=null}>${a(`x`)}</button></div></header>
      ${e.reason?i`<p class="small"><span class="muted">why:</span> ${e.reason}</p>`:l}
      ${e.stale&&e.status===`open`?i`<p class="warn-ink small">${a(`warn`)} ${e.name} changed since this was proposed${e.mode===`append`?`; the section is added to the text as it is now`:`; a replacement cannot be applied`}.</p>`:l}
      ${e.live===null?i`<p class="warn-ink small">No package ${e.name} with a text any more.</p>`:i`<pre class="notes-diff" data-prop-diff>${t.map(e=>i`<span class="diff-${e.op===`+`?`add`:e.op===`-`?`del`:`eq`}">${e.op===`=`?` `:e.op} ${e.text}\n</span>`)}</pre>`}
    </section>`}};t([f()],W.prototype,`list`,void 0),t([f()],W.prototype,`open`,void 0),t([f()],W.prototype,`error`,void 0),t([f()],W.prototype,`flash`,void 0),t([f()],W.prototype,`busy`,void 0),t([f()],W.prototype,`status`,void 0),W=t([r(`qse-cond-proposals`)],W);var en=[{id:`text`,label:`Text`},{id:`edit`,label:`Edit`},{id:`classes`,label:`Classes`},{id:`usage`,label:`Usage`}],G=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.selected=null,this.tab=null,this.pane=null,this.catalog=null,this.catError=null,this.usage=null,this.filters=St,this.hits=null,this.libError=null,this.item=null,this.itemError=null,this.recent=null,this.raw=!1,this.editView=`write`,this.counted=null,this.review=null,this.attachError=null,this.found=null,this.scanError=null,this.pk=new D(this,`packages`),this.rt=new D(this,`router`),this.seq=0,this.loadedFor=null,this.countTimer=null,this.countFor=null}connectedCallback(){super.connectedCallback(),this.pk.onSaved=()=>void this.refresh(),this.rt.onSaved=()=>void this.refresh(),this.pk.load(),this.rt.load(),this.refresh()}disconnectedCallback(){this.countTimer&&clearTimeout(this.countTimer),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.item=null,this.itemError=null,this.recent=null,this.attachError=null,this.selected&&this.loadItem(this.selected))}updated(e){let t=this.selected?this.pk.draft?.packages[this.selected]:null;if(!t||t.source!==`inline`)return;let n=V(t.body,!!t.library)[0];this.counted?.text!==n&&this.countFor!==n&&(this.countFor=n,this.countTimer&&clearTimeout(this.countTimer),this.countTimer=setTimeout(()=>{S.packagesTokens([n]).then(e=>this.counted={text:n,n:e.tokens[0],how:e.tokenizer}).catch(()=>void 0)},400))}async refresh(){try{let[e,t]=await Promise.all([S.packagesCatalog(),S.packagesUsage(null,1)]);[this.catalog,this.usage,this.catError]=[e,t,null]}catch(e){this.catError=b(e).message}this.selected&&await this.loadItem(this.selected)}async loadItem(e){try{let[t,n]=await Promise.all([this.pk.base?.doc.packages[e]||!this.pk.base?S.packageItem(e):Promise.resolve(null),S.packagesUsage(e,20)]);if(e!==this.selected)return;[this.item,this.recent,this.itemError]=[t,n.recent,null]}catch(t){e===this.selected&&(this.itemError=b(t).message)}}async search(e){this.filters={...this.filters,q:e};let t=++this.seq;if(!e.trim()||!Object.values(this.pk.draft?.packages??{}).some(e=>e.library)){[this.hits,this.libError]=[null,null];return}try{let n=await S.librarySearch(e.trim(),20);t===this.seq&&([this.hits,this.libError]=[n,null])}catch(e){t===this.seq&&([this.hits,this.libError]=[[],`the library's search: ${b(e).message}`])}}open(e,t=null){x({pkg:e,pview:t})}upd(e,t){this.pk.draft&&this.pk.set(T(this.pk.draft,e,t))}toggleAttach(e,t,n){let r=this.rt.draft;if(!r)return;if(!n){this.attachError=null,this.rt.set(Pt(r,e,t));return}let i=Nt(r,e,t,this.pk.draft);this.attachError=i.error,i.error||this.rt.set(i.router)}addInline(){let e=this.pk.draft,t=It(e);this.pk.set(T(e,[`packages`,t],Lt(Number(t.slice(8))))),this.open(t,`edit`)}copyInline(e){let t=this.pk.draft,n=this.item;n&&n.name===e&&n.found&&(this.pk.set(T(t,[`packages`,e],Rt(t.packages[e],n.body,n.reference))),this.open(e,`edit`))}removePackage(e){let t=this.pk.draft,n={...t.packages};delete n[e],this.pk.set({...t,packages:n}),this.open(null)}async scan(){this.scanError=null;try{this.found=(await S.scan()).skills}catch(e){this.scanError=b(e).message}}register(e){let t={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.pk.set(T(this.pk.draft,[`packages`,e.name],t)),this.open(e.name)}async startReview(e,t){await e.check(),this.review=t}async confirmSave(e){await e.save()&&(this.review=null)}bar(e,t,n){let r=e.base?.meta,o=!!e.draft&&!!e.base&&JSON.stringify(e.draft)!==JSON.stringify(e.base.doc);return i`
      ${e.showVersions?lt(e,n):l}
      ${this.review===t?i`<section class="panel pk-review" data-review=${t}>
            <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e.errors.length?i`${a(`warn`)} The server refuses ${n}`:i`${a(`check`)} The server accepts ${n}`}</h2>
              <p class="panel-sub">POST /v1/conductor/${t}/check against live v${r?.version??0}: ${e.changes.length} change${e.changes.length===1?``:`s`}. Saving makes a new version with your note; the next request uses it.</p></div></header>
            ${A(e)}
            ${at(e.changes)}
            <div class="toolbar pk-review-tools">
              <button class="btn btn-primary" data-review-save=${t} ?disabled=${e.errors.length>0||!e.note.trim()||e.saving||!e.changes.length} @click=${()=>this.confirmSave(e)}>${e.saving?`Saving…`:`Save as a new version`}</button>
              <button class="btn btn-ghost" @click=${()=>this.review=null}>Back to editing</button>
              ${e.note.trim()?l:i`<span class="muted small">Write the change note first.</span>`}
            </div>
          </section>`:l}
      <div class="savebar ${o||this.review===t?``:`is-idle`}" data-savebar=${t}>
        <span class="status num" title=${r?.note??``}>${n} · live v${r?.version??0}${o?i` · <strong>${e.changes.length||`…`} change${e.changes.length===1?``:`s`}</strong>`:l}</span>
        <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>{e.note=t.target.value,this.requestUpdate()}} aria-label=${`change note for ${n}`} maxlength="200" data-note=${t} />
        <button class="btn btn-primary" ?disabled=${!o||e.saving} @click=${()=>this.startReview(e,t)} data-check-save=${t}>Check and save…</button>
        <button class="btn btn-ghost" ?disabled=${!o} @click=${()=>{e.reset(),this.review=null}}>Reset</button>
        <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():this.requestUpdate()}} data-versions-btn=${t}>Versions</button>
        ${e.saved?i`<span class="status" role="status" data-saved=${t}>${a(`check`)} ${e.saved}</span>`:l}
        ${e.saveError?i`<span class="err-ink small" role="alert">${e.saveError}</span>`:l}
      </div>`}render(){let e=i`<nav class="mem-tabs pk-tabs-top" aria-label="packages tabs">
      ${[{id:null,label:`Packages and skills`},{id:`library`,label:`Library retrieval`},{id:`proposals`,label:`Proposals`}].map(e=>i`<a class="cond-tab ${(this.tab??null)===e.id?`is-on`:``}" href="#" data-pk-top=${e.id??`all`} aria-current=${(this.tab??null)===e.id?`page`:l} @click=${t=>{t.preventDefault(),x({ptab:e.id,pkg:null,pview:null})}}>${e.label}</a>`)}
    </nav>`;if(this.tab===`proposals`)return i`${e}<qse-cond-proposals .onChanged=${()=>this.refresh()}></qse-cond-proposals>`;if(this.tab===`library`)return i`${e}<qse-cond-library .ids=${this.ids} .tiles=${this.tiles} .catalog=${this.catalog} .usage=${this.usage} .onChanged=${()=>this.refresh()}></qse-cond-library>`;let t=this.pk;if(t.loading&&!t.draft)return i`${e}${c(6)}`;if(t.loadError)return i`${e}<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${t.loadError}</div></div>`;let n=t.draft,r=Dt(n,this.catalog,this.usage?.packages??null,this.rt.draft),o=jt(r,this.filters,this.hits,this.catalog),s=this.selected&&n.packages[this.selected]?this.selected:null;return i`
      ${e}
      ${this.catError?i`<div class="guard-line" role="alert">${a(`warn`)} ${this.catError}</div>`:l}
      <div class="pk-layout ${s?`has-detail`:``}">
        ${this.renderList(o,r)}
        ${s?this.renderDetail(s,n.packages[s],r.find(e=>e.name===s)):this.selected?i`<section class="panel" data-pk-gone><p class="muted">No package ${this.selected} in this version.</p><button class="btn btn-sm" @click=${()=>this.open(null)}>Back to the list</button></section>`:l}
      </div>
      ${this.renderFound(n)}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${n.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${this.rt.draft&&(this.rt.showVersions||this.rt.saved||this.review===`router`||JSON.stringify(this.rt.draft)!==JSON.stringify(this.rt.base?.doc))?this.bar(this.rt,`router`,`the router`):l}
      ${this.bar(this.pk,`packages`,`the packages`)}
    `}renderList(e,t){let n=this.filters,r=t.filter(e=>At(e,this.catalog)).length,o=e=>this.filters={...this.filters,...e};return i`<section class="panel pk-list" data-pk-list>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills <span class="muted" data-pk-count-all>· ${e.length} of ${t.length}</span></h2>
        <p class="panel-sub">What the router may put into a prompt. Tokens are the text that goes in, ${this.catalog?.tokenizer===`estimate`?`estimated at 3.5 characters a token (no tokenizer on this server)`:`on the engine's tokenizer`}${r?i`; <span class="warn-ink" data-pk-over-count>${r} over the limit</span>`:l}.</p></div>
        <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-pk-new>${a(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${a(`search`)} Skills on the box</button></div></header>
      <div class="pk-filters">
        <label class="field field-search pk-search">${a(`search`)}<input type="search" data-pk-search placeholder="search packages, or the library's text" aria-label="search packages and skills" .value=${n.q} @input=${e=>void this.search(e.target.value)} /></label>
        <label class="field"><span>kind</span><select data-pk-filter="kind" @change=${e=>o({category:e.target.value})}>
          <option value="all" ?selected=${n.category===`all`}>all</option>
          ${z.map(e=>i`<option value=${e} ?selected=${n.category===e}>${B[e]}</option>`)}
        </select></label>
        <label class="field"><span>class</span><select data-pk-filter="class" @change=${e=>o({cls:e.target.value})}>
          <option value="" ?selected=${!n.cls}>any</option>
          ${this.ids.map(e=>i`<option value=${e} ?selected=${n.cls===e}>${e}</option>`)}
        </select></label>
        <label class="field"><span>enabled</span><select data-pk-filter="enabled" @change=${e=>o({enabled:e.target.value})}>
          ${[`all`,`on`,`off`].map(e=>i`<option value=${e} ?selected=${n.enabled===e}>${e}</option>`)}
        </select></label>
        <label class="switch"><input type="checkbox" data-pk-filter="over" .checked=${n.over} @change=${e=>o({over:e.target.checked})} /> over the limit</label>
      </div>
      ${this.libError?i`<p class="err-ink small">${this.libError}</p>`:l}
      <div class="pk-head" aria-hidden="true"><span>package</span><span>tokens / limit</span><span>used 24 h · 7 d</span></div>
      <div class="pk-rows" role="list">${e.map(e=>this.row(e))}</div>
      ${e.length?l:i`<p class="muted small" data-pk-empty>${n.q.trim()?`Nothing matches “${n.q.trim()}”${this.hits?`, and the library shares no word with it`:``}.`:`No package matches these filters.`}</p>`}
    </section>`}row(e){let t=Tt(e.name,e.pkg,e.category,this.catalog),n=e.entry?e.entry.tokens:Ct(V(e.pkg.body,!!e.pkg.library)[0]),r=At(e,this.catalog),a=e.usage,o=this.selected===e.name;return i`<button class="pk-row ${o?`is-open`:``} ${e.pkg.enabled?``:`is-off`}" role="listitem" data-pk-row=${e.name} data-category=${e.category} aria-current=${o?`true`:l} @click=${()=>this.open(e.name)}>
      <span class="pk-main">
        <span class="pk-name"><strong>${e.name}</strong>
          <span class="badge pk-kind pk-kind-${e.category}">${B[e.category]}</span>
          ${e.pkg.general?i`<span class="badge badge-on" title="goes in whole and first on every turn of a class that lists it">general</span>`:l}
          ${e.pkg.enabled?l:i`<span class="badge">off</span>`}
          ${e.entry?l:i`<span class="badge badge-shadow">not saved</span>`}
          ${e.entry&&!e.entry.available&&e.entry.error?i`<span class="badge badge-warn" title=${e.entry.path??``}>${e.entry.error.replace(/_/g,` `)}</span>`:l}
          ${e.score==null?l:i`<span class="pk-score num" data-score title="the library's BM25 score">${e.score.toFixed(2)}</span>`}
        </span>
        <span class="pk-desc muted small">${e.pkg.description}</span>
        <span class="pk-meta"><span class="mono small muted">${e.pkg.source}</span>${e.classes.map(e=>O(e,this.ids))}${e.category===`library`&&e.entry?.fits?.length?i`<span class="muted small">fits</span>${e.entry.fits.map(e=>O(e,this.ids))}`:l}</span>
      </span>
      <span class="pk-tokens num ${r?`is-over`:``}" data-pk-tokens title=${e.entry?`${e.entry.tokens} tokens go in; the whole text is ${e.entry.body_tokens}`:`estimated: not saved yet`}>
        <span>${e.entry?``:`≈`}${p(n)} <span class="muted">/ ${p(t.limit)}</span></span>
        <span class="pk-bar"><span style="width:${Math.min(100,100*n/Math.max(1,t.limit))}%"></span></span>
        ${r?i`<span class="badge badge-warn">over</span>`:l}
      </span>
      <span class="pk-usage num small" data-pk-usage>
        ${a&&(a.injected.week||a.shadow.week||a.pick.week)?i`${a.injected.week?i`<span>${a.injected.day} · ${a.injected.week} in</span>`:l}${a.shadow.week?i`<span class="muted">${a.shadow.day} · ${a.shadow.week} shadow</span>`:l}${a.pick.week?i`<span>${a.pick.day} · ${a.pick.week} picks</span>`:l}`:i`<span class="muted">—</span>`}
      </span>
    </button>`}renderFound(e){return this.scanError?i`<p class="err-ink small">${this.scanError}</p>`:this.found?i`<section class="panel" style="margin-top:16px" data-found>
      <header class="panel-head"><h2 class="panel-title">Skills in ${e.skill_dirs.join(`, `)}</h2><div class="panel-tools"><button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.found=null}>${a(`x`)}</button></div></header>
      ${this.found.length?i`<div class="table-scroll"><table class="table"><tbody>${this.found.map(t=>i`<tr><td><strong>${t.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${t.description}</div></td><td class="num">${p(t.chars)} chars</td>
              <td>${t.registered||e.packages[t.name]?i`<span class="badge badge-ok">registered</span>`:i`<button class="btn btn-sm" @click=${()=>this.register(t)} data-register=${t.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:i`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ &lt;box&gt;:~/.config/opencode/skills/</code>).</p>`}
    </section>`:l}renderDetail(e,t,n){let r=n.entry,o=wt(t,r),s=Tt(e,t,o,this.catalog),c=en.some(e=>e.id===this.pane)?this.pane:`text`,u=t.source===`inline`,d=n.classes,f=this.tokensNow(e,t,r),m=f.n>s.limit||!!r?.cut&&!(u&&this.dirtyBody(e));return i`<section class="panel pk-detail" data-pk-detail=${e}>
      <header class="pk-detail-head">
        <div class="pk-title">
          <h2 class="panel-title">${e}</h2>
          <span class="muted">${t.title}</span>
        </div>
        <button class="btn btn-ghost btn-sm pk-close" @click=${()=>this.open(null)} data-pk-close aria-label="Close the package">${a(`x`)}<span class="pk-close-word">All packages</span></button>
      </header>
      <div class="pk-badges">
        <span class="badge pk-kind pk-kind-${o}" data-pk-category>${B[o]}</span>
        <span class="badge">${u?`inline`:t.source.startsWith(`skill:`)?`skill on the box`:`file on the box`}</span>
        ${t.general?i`<span class="badge badge-on">general</span>`:l}
        ${t.enabled?i`<span class="badge badge-ok">enabled</span>`:i`<span class="badge">disabled</span>`}
        ${r?l:i`<span class="badge badge-shadow">not saved</span>`}
        ${r&&!r.available&&r.error?i`<span class="badge badge-warn">${r.error.replace(/_/g,` `)}</span>`:l}
      </div>
      <div class="toolbar pk-switches">
        <label class="switch"><input type="checkbox" data-pk-enabled .checked=${t.enabled} @change=${t=>this.upd([`packages`,e,`enabled`],t.target.checked)} /> enabled</label>
        ${o===`library`?l:i`<label class="switch" title="goes in whole and first on every turn of a class that lists it (K1)"><input type="checkbox" data-pk-general .checked=${!!t.general} @change=${t=>this.upd([`packages`,e,`general`],t.target.checked)} /> general</label>`}
        ${u?l:i`<button class="btn btn-sm" data-pk-copy-inline ?disabled=${!this.item?.found} title=${this.item?.found?`an inline copy of the file overrides it`:`the file is not on the box`} @click=${()=>this.copyInline(e)}>${a(`copy`)} Copy to inline</button>`}
        <button class="btn btn-sm btn-ghost" data-pk-remove ?disabled=${d.length>0} title=${d.length?`${d.join(`, `)} list it; detach it first`:`remove it from the packages document`} @click=${()=>this.removePackage(e)}>${a(`trash`)} Remove</button>
      </div>
      <div class="pk-meter ${m?`is-over`:``}" data-pk-meter>
        <div class="pk-bar"><span style="width:${Math.min(100,100*f.n/Math.max(1,s.limit))}%"></span></div>
        <span class="num small" data-pk-count>${f.approx?`≈ `:``}${p(f.n)} tokens of ${p(s.limit)} <span class="muted">(${f.why}; limit: ${s.source===`house`?o===`library`?`the house rule for a library skill`:`the house rule for the library card`:`max_tokens, where the server cuts`})</span></span>
        ${m?i`<span class="badge badge-warn" data-pk-limit-warn>over the limit</span>`:l}
      </div>
      <nav class="mem-tabs pk-panes" aria-label="package detail">
        ${en.map(e=>i`<a class="cond-tab ${e.id===c?`is-on`:``}" href="#" data-pk-pane=${e.id} aria-current=${e.id===c?`page`:l} @click=${t=>{t.preventDefault(),x({pview:e.id===`text`?null:e.id})}}>${e.label}</a>`)}
      </nav>
      <div class="pk-pane" data-pk-pane-body=${c}>
        ${c===`text`?this.paneText(e,t,r):c===`edit`?this.paneEdit(e,t,o,s):c===`classes`?this.paneClasses(e,t,o):this.paneUsage(e,o)}
      </div>
    </section>`}dirtyBody(e){return this.pk.draft?.packages[e]?.body!==this.pk.base?.doc.packages[e]?.body}tokensNow(e,t,n){if(t.source===`inline`&&(this.dirtyBody(e)||!n)){let e=V(t.body,!!t.library)[0];return this.counted?.text===e?{n:this.counted.n,approx:this.counted.how===`estimate`,why:this.counted.how===`engine`?`the draft, on the engine's tokenizer`:`the draft, estimated`}:{n:Ct(e),approx:!0,why:`the draft, estimated until the engine counts it`}}return n?{n:n.body_tokens,approx:this.catalog?.tokenizer===`estimate`,why:n.cut?`the whole text; ${p(n.tokens)} go in after the cut`:this.catalog?.tokenizer===`estimate`?`estimated`:`on the engine's tokenizer`}:{n:0,approx:!0,why:`not counted yet`}}md(e){return e?this.raw?i`<pre class="note-text pk-raw" data-pk-raw>${e}</pre>`:i`<div class="pg-md pk-md" data-pk-md>${nt(rt(e))}</div>`:i`<p class="muted small">(empty)</p>`}paneText(e,t,n){let r=t.source===`inline`,o=this.item?.name===e?this.item:null;if(!r&&!o)return this.itemError?i`<p class="err-ink small">${this.itemError}</p>`:c(4);let[s,u]=r?V(t.body,!!t.library):[o.body,o.reference],f=r?{}:o.frontmatter;return i`
      <div class="toolbar pk-text-tools">${j([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`text view`})}
        ${!r&&o?.path?i`<span class="mono small muted pk-path" data-pk-path>${o.path}</span>`:l}</div>
      ${!r&&o&&!o.found?i`<p class="note" data-pk-missing>${a(`warn`)} The file is not on the box (<code>${o.path??t.source}</code>): nothing goes in.</p>`:l}
      ${Object.keys(f).length?i`<h3 class="pk-sub">Frontmatter <span class="muted small">the client's metadata; never injected</span></h3>
            ${this.raw?i`<pre class="note-text pk-raw" data-pk-front>${o.frontmatter_raw}</pre>`:i`<dl class="kv pk-front" data-pk-front>${Object.entries(f).map(([e,t])=>i`<div><dt>${e}</dt><dd>${t}</dd></div>`)}</dl>`}`:l}
      <h3 class="pk-sub">The text that goes in ${s?d(s,`Copy the text`):l}</h3>
      ${this.md(s)}
      ${n?.cut&&!this.dirtyBody(e)?i`<p class="note" data-pk-cut>${a(`warn`)} The server cuts this text at ${p(t.max_tokens)} tokens × 3.5 characters, at a line end, and adds “[... cut to the package's max_tokens]”: ${p(n.tokens)} tokens go in of ${p(n.body_tokens)}.</p>`:l}
      ${u?i`<details class="sec pk-ref" data-pk-ref>
            <summary>Reference <span class="sec-sub">${p(o?.tokens.reference??Ct(u))} tokens; only a read gets it (GET /v1/library/${e})</span></summary>
            <div class="sec-body">${this.md(u)}</div>
          </details>`:l}`}paneEdit(e,t,n,r){let o=t.source===`inline`;return i`
      <div class="form-grid">
        <label class="field"><span>title</span><input data-pk-title .value=${t.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>max tokens</span><input type="number" min="16" max="32768" data-pk-max .value=${String(t.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">the server cuts a longer text at a line end</span></label>
        ${o?l:i`<label class="field field-wide"><span>source</span><input class="mono" .value=${t.source} readonly /><span class="hint">${this.item?.path??`a file on the box`}</span></label>`}
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" data-pk-description .value=${t.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${o?this.editor(e,t,n,r):i`<div class="note pk-readonly" data-pk-readonly>${a(`info`)} The text is the file on the box${this.item?.path?i` at <code>${this.item.path}</code>`:l}, read when first needed and again when it changes. Editing files on the box is not done here. <button class="btn btn-sm" ?disabled=${!this.item?.found} @click=${()=>this.copyInline(e)} data-pk-copy-inline-edit>Copy to inline</button> makes an inline copy of its text that overrides the file.</div>`}`}editor(e,t,n,r){let[a,o]=V(t.body,!!t.library),s=this.tokensNow(e,t,this.catalog?.packages[e]??null),c=s.n>r.limit,u=this.editView;return i`<div class="pk-editor pk-view-${u}" data-pk-editor>
      <div class="toolbar pk-editor-tools">
        ${j([`write`,`preview`,`split`],u,e=>this.editView=e,{small:!0,label:`editor view`})}
        <span class="pk-count num small ${c?`is-over`:``}" data-pk-editor-count>${s.approx?`≈ `:``}${p(s.n)} of ${p(r.limit)} tokens${t.library&&o?i` <span class="muted">· reference ≈ ${p(Ct(o))}</span>`:l}</span>
        ${c?i`<span class="badge badge-warn" data-pk-editor-warn>over ${r.source===`house`?`the ${B[n]}'s ${p(r.limit)}`:`max_tokens: the server will cut it`}</span>`:l}
      </div>
      <div class="pk-editor-panes">
        ${u===`preview`?l:i`<textarea class="code pk-body" rows="18" spellcheck="false" data-pk-body aria-label="the package's text (Markdown)" .value=${t.body} @input=${t=>this.upd([`packages`,e,`body`],t.target.value)}></textarea>`}
        ${u===`write`?l:i`<div class="pg-md pk-md pk-preview" data-pk-preview>${nt(rt(a))}</div>`}
      </div>
      <p class="hint">${t.library?`A library skill: the text before “## Reference” goes in; the reference only on a read. `:``}${r.source===`house`?`The house rule keeps ${B[n]} texts at ${p(r.limit)} tokens or fewer.`:`The server cuts the text at max_tokens × 3.5 characters, at a line end.`} Markdown; the preview is the dashboard's renderer, the model gets the text as written.</p>
    </div>`}paneClasses(e,t,n){let r=this.catalog?.packages[e]??null;if(n===`library`)return i`<p class="muted small" data-pk-library-classes>A library skill is searched, not listed by a class: on a new user message the router searches the library and keeps the skills INDEX.md fits to the turn's class.</p>
        <div class="pk-meta">${r?.fits?.length?i`<span class="muted small">INDEX.md fits</span>${r.fits.map(e=>O(e,this.ids))}`:i`<span class="muted small">INDEX.md does not list it: it stays in for every class.</span>`}</div>
        <p class="small"><a href="#/conductor?screen=packages&ptab=library">Each profile's retrieval switch and its picks</a></p>`;let a=this.rt.draft;return a?i`
      <p class="muted small">A class's packages go in by K1's order: the general ones whole and first, then the router asks the package question among the rest and puts in the one that fits. A class lists at most ${12}. Attaching edits the router document; it is saved with its own check below.</p>
      ${this.attachError?i`<p class="err-ink small" role="alert" data-pk-attach-error>${this.attachError}</p>`:l}
      <div class="pk-classes" data-pk-classes>
        ${a.classes.map(t=>{let n=t.policy.packages??[],r=n.includes(e);return i`<div class="pk-class ${r?`is-on`:``}" data-pk-class=${t.id}>
            <label class="switch"><input type="checkbox" .checked=${r} data-pk-attach=${t.id} aria-label=${`attach ${e} to ${t.id}`} @change=${n=>this.toggleAttach(t.id,e,n.target.checked)} /> ${O(t.id,this.ids,t.label)}</label>
            <span class="pk-order small" data-pk-order=${t.id}>${n.length?n.map((t,n)=>i`<span class="pk-order-item ${t===e?`is-me`:``}">${n+1}. ${t}${this.pk.draft?.packages[t]?.general?i` <span class="muted">general</span>`:l}</span>`):i`<span class="muted">no packages</span>`}</span>
            <span class="muted small num">${n.length}/${12}</span>
          </div>`})}
      </div>
      ${r&&Object.keys(r.profiles).length?i`<p class="small muted" data-pk-profiles>A profile's own routing also lists it: ${Object.entries(r.profiles).map(([e,t])=>`${e} (${t.join(`, `)})`).join(`; `)}. A profile's routing replaces the router's policy for that class (the Profiles screen).</p>`:l}
      <div class="toolbar"><button class="btn btn-sm btn-ghost" data-pk-router-versions @click=${()=>{this.rt.showVersions=!0,this.rt.loadVersions()}}>Router versions</button></div>`:c(3)}paneUsage(e,t){let n=this.usage?.packages[e]??null,r={day:0,week:0},a=(e,t)=>u({label:e,value:p(t.day),unit:`in 24 h`,sub:`${p(t.week)} in 7 days`}),o=this.item?.name===e?this.item:null,s=t===`library`||(n?.pick.week??0)>0;return i`
      <div class="stats pk-stats" style="--pk-tiles:${s?3:2}" data-pk-usage-tiles>
        ${a(`injected`,n?.injected??r)}${a(`shadow (would have)`,n?.shadow??r)}${s?a(`library picks`,n?.pick??r):l}
      </div>
      <h3 class="pk-sub">Recent rows <span class="muted small">the journal, newest first (7 days)</span></h3>
      ${this.recent==null?c(3):this.recent.length?i`<div class="pk-uses" data-pk-recent>${this.recent.map(e=>i`<div class="pk-use" data-pk-use=${e.id}>
                <span class="num small">${v(new Date(e.ts_ms).toISOString())}</span>
                <span class="badge ${e.how===`injected`?`badge-on`:e.how===`shadow`?`badge-shadow`:`badge-warn`}">${e.how}</span>
                ${O(e.class_id,this.ids)}
                <span class="muted small">${e.profile??`—`}${e.client_kind?` · ${e.client_kind}`:``}</span>
                ${e.session_key?i`<a class="mono small pk-session" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${e.session_key.length>28?`${e.session_key.slice(0,28)}…`:e.session_key}</a>`:l}
              </div>`)}</div>`:i`<p class="muted small" data-pk-recent-empty>No router row names it in the last 7 days.</p>`}
      ${t===`library`?i`<h3 class="pk-sub">Neighbours <span class="muted small">the library's BM25 on this skill's name, description and text</span></h3>
            ${o?.neighbours?o.neighbours.length?i`<div class="pk-neighbours" data-pk-neighbours>${o.neighbours.map(e=>i`<button class="pk-neighbour" data-pk-neighbour=${e.name} @click=${()=>this.open(e.name)}><strong>${e.name}</strong> <span class="num muted small">${e.score.toFixed(2)}</span><span class="muted small pk-desc">${e.description}</span></button>`)}</div>`:i`<p class="muted small">No other skill shares a word with it.</p>`:c(2)}`:l}`}};t([n({attribute:!1})],G.prototype,`ids`,void 0),t([n({attribute:!1})],G.prototype,`tiles`,void 0),t([n({attribute:!1})],G.prototype,`selected`,void 0),t([n({attribute:!1})],G.prototype,`tab`,void 0),t([n({attribute:!1})],G.prototype,`pane`,void 0),t([f()],G.prototype,`catalog`,void 0),t([f()],G.prototype,`catError`,void 0),t([f()],G.prototype,`usage`,void 0),t([f()],G.prototype,`filters`,void 0),t([f()],G.prototype,`hits`,void 0),t([f()],G.prototype,`libError`,void 0),t([f()],G.prototype,`item`,void 0),t([f()],G.prototype,`itemError`,void 0),t([f()],G.prototype,`recent`,void 0),t([f()],G.prototype,`raw`,void 0),t([f()],G.prototype,`editView`,void 0),t([f()],G.prototype,`counted`,void 0),t([f()],G.prototype,`review`,void 0),t([f()],G.prototype,`attachError`,void 0),t([f()],G.prototype,`found`,void 0),t([f()],G.prototype,`scanError`,void 0),G=t([r(`qse-cond-packages`)],G);var tn={not_loaded:`not loaded`,loading:`loading the weights`,warming:`warming up`,loaded:`loaded`,unloading:`unloading`,failed:`failed`};function nn(e){let t=e.role;return t?[t.kind,t.window.replace(`_`,` `),t.preemptible?`preemptible`:`not preemptible`,`${t.latency_budget_ms} ms then ${t.fallback}`,`writes ${t.writes.replace(/_/g,` `)}`,`shares ${t.sharing.join(`, `)||`nothing`}`,t.memory_claim_gib==null?`no claim`:`claims ${E(t.memory_claim_gib)}`].join(` · `):null}function rn(e){return e===`loaded`?`badge-ok`:e===`failed`?`badge-warn`:e===`loading`||e===`warming`||e===`unloading`?`badge-shadow`:``}var an=`not loaded: kept free for requests`;function on(e){if(!e||e.allowed)return null;let t=e.reason?Ee[e.reason]??e.reason:`refused`;if(e.reason!==`refused_memory`)return t;if(e.mem_available_gib==null||e.after_gib==null)return`${an} (MemAvailable not read)`;let n=e.engine_growth_gib?`less ${E(e.engine_growth_gib)} the engine may still take for a long request and the ${E(e.claim_gib)} claim`:`less the ${E(e.claim_gib)} claim`,r=e.margin_gib?`${E(e.need_gib)} free (a ${E(e.floor_gib)} floor and a ${E(e.margin_gib)} margin)`:`${E(e.need_gib)} free`;return`${an} (MemAvailable ${E(e.mem_available_gib)}, ${n}, leaves ${E(e.after_gib)}; the box keeps ${r})`}function sn(e){return e?e.allowed?`fits, ${E(e.after_gib)} left after the claim`:on(e)??`refused`:`—`}function cn(e){return e?e.allowed?`fits`:e.reason===`refused_memory`?`kept free`:`refused`:`—`}function ln(e){return!e?.allowed&&e?.reason!==`refused_memory`}function un(e){if(!e)return``;let t=(e.source??``).replace(/^engine:/,``);return e.role===`shadow`?t?`the shadow reader (${t})`:`the shadow reader`:t}function dn(e){let t=Object.entries(e).sort((e,t)=>t[1]-e[1]);return t.length?t.map(([e,t])=>`${e} ${t}`).join(` · `):`no call yet`}function fn(e){let t=e.limits;return[t.context_tokens,t.question_tokens,t.max_options].map(e=>e==null?`—`:String(e)).join(` / `)}function pn(e,t){let n=e.trim();if(!n)return`Write the question.`;let r=t.split(/[,\n]/).map(e=>e.trim()).filter(Boolean);return r.length<2?`Give at least two options.`:r.length>20?`At most 20 options (the model card’s limit).`:new Set(r).size===r.length?{type:`choice`,instructions:n,criteria:Object.fromEntries(r.map(e=>[e,null]))}:`Two options have the same name.`}function mn(e){return e.type===`noul`?[[`true`,e.noul??0],[`false`,1-(e.noul??0)]]:Object.entries(e.probabilities??{}).sort((e,t)=>t[1]-e[1])}var hn=[`on_start`,`on_demand`,`off`],gn={mode:`on`,rate_per_min:240,max_texts:16,max_chars:1e5,deadline_ms:15e3,threshold:.8,drive_rate_per_s:10,drive_sessions:4,drive_idle_s:30,drive_max_texts:36e3},K=class extends o{constructor(...e){super(...e),this.st=null,this.error=null,this.busy=null,this.said={},this.tState=`The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.`,this.tQuestion=`What kind of work does this request ask for?`,this.tOptions=`debug, code_edit, plan, chat`,this.tModel=`auto`,this.tOut=null,this.tErr=null,this.asking=!1,this.ed=new D(this,`deciders`),this.poller=null}connectedCallback(){super.connectedCallback(),this.ed.load(),this.poller=C(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{this.st=await S.deciders(),this.error=null}catch(e){let t=b(e);this.error=`${t.endpoint}: ${t.message}`}}async act(e,t){if(t!==`unload`||await k(`Unload ${e.name}?`,[`Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).`,`Loading it again goes through the residency planner.`],`Unload`)){this.busy=e.name,this.said={...this.said,[e.name]:``};try{let n=await(t===`load`?S.deciderLoad(e.name):S.deciderUnload(e.name));this.said={...this.said,[e.name]:t===`load`?`load started (${n.state??`loading`})`:`unloaded`}}catch(n){let r=n instanceof ee&&n.status===409&&t===`load`?`refused by the residency planner: ${n.message}`:b(n).message;this.said={...this.said,[e.name]:r}}finally{this.busy=null,await this.load()}}}setLoad(e,t){let n=this.ed.draft;if(!n)return;let r=n.models.findIndex(t=>t.name===e),i=r>=0?n.models.map((e,n)=>n===r?{...e,load:t}:e):[...n.models,{name:e,load:t}];this.ed.set({...n,models:i})}async ask(){let e=pn(this.tQuestion,this.tOptions);if(typeof e==`string`){this.tErr=e;return}this.asking=!0,this.tErr=null;try{this.tOut=await S.deciderTest(this.tState,{q:e},this.tModel)}catch(e){this.tOut=null,this.tErr=b(e).message}finally{this.asking=!1}}render(){let t=this.st;if(this.error&&!t)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${this.error}</div></div>`;if(!t)return c(6);if(!t.enabled)return e(`Decision models`,i`<div class="state" data-deciders-off>
          <div class="state-title">${a(`warn`)} This engine serves no decision model</div>
          <p class="muted small">${t.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port (behind the admin password once one is set), and this screen loads, unloads and tests them.</p>
        </div>`,{id:`deciders-off`});let n=t.models.filter(e=>e.state===`loaded`);return i`
      <div class="stats" data-decider-tiles>
        ${u({label:`loaded`,value:`${n.length} / ${t.models.length}`,sub:n.length?n.map(e=>e.name).join(`, `):`none: questions fail open`,warn:!n.length})}
        ${u({label:`on the device`,value:Ye(n.reduce((e,t)=>e+t.device_bytes,0)),sub:`${t.device??`no device`} · own stream and memory pool`})}
        ${u({label:`a load now`,value:cn(t.planner),sub:sn(t.planner),warn:ln(t.planner)})}
        ${u({label:`when they run`,value:`idle`,sub:`only while no request holds the engine (stage A)`})}
      </div>
      ${this.renderModels(t)}
      ${this.renderMemory(t)}
      ${this.renderApi(t)}
      ${this.renderQuestions(t)}
      ${this.renderTest(t)}
    `}renderModels(t){let n=this.ed.draft,r=e=>n?.models.find(t=>t.name===e)?.load??`on_demand`;return e(`Models`,i`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${t.models.map(e=>{let t=on(e.refusal),n=this.said[e.name],o=e.state===`loaded`||e.state===`failed`||e.state===`warming`;return i`<tr data-decider-row=${e.name}>
              <td><strong>${e.name}</strong><div class="muted small">${e.family} · ${e.contract}${e.aliases.length?i` · ${e.aliases.join(`, `)}`:l}</div>${e.role?i`<div class="muted small" data-decider-role=${e.name} title="the role contract this model signs (SRV-133); the registry checks it before it serves the model">${nn(e)}</div>`:l}</td>
              <td><span class="badge ${rn(e.state)}" data-decider-state=${e.state}>${tn[e.state]}</span>${e.warm===!1?i` <span class="muted small">not warmed</span>`:l}</td>
              <td><select aria-label=${`load policy of ${e.name}`} data-decider-load=${e.name} @change=${t=>this.setLoad(e.name,t.target.value)}>
                ${hn.map(t=>i`<option value=${t} ?selected=${r(e.name)===t}>${t.replace(`_`,` `)}</option>`)}
              </select></td>
              <td class="num">${fn(e)}</td>
              <td class="small">${e.precision??i`<span class="muted">—</span>`}<div class="muted small mono" title=${e.weight_hash??``}>${e.weight_hash?e.weight_hash.slice(0,12):``}</div></td>
              <td class="num num-col">${p(e.questions)}</td>
              <td class="num num-col">${e.stats?.forward_ms_p50==null?`—`:`${m(e.stats.forward_ms_p50)} / ${m(e.stats.forward_ms_p95)}`}</td>
              <td class="num num-col ${e.stats?.errors?`err-ink`:``}">${e.stats?p(e.stats.errors+e.stats.deadline_missed):`—`}</td>
              <td>${o?i`<button class="btn btn-sm" ?disabled=${this.busy===e.name} @click=${()=>this.act(e,`unload`)} data-unload=${e.name}>Unload</button>`:i`<button class="btn btn-sm btn-primary" ?disabled=${this.busy===e.name||e.state===`loading`||e.state===`unloading`} @click=${()=>this.act(e,`load`)} data-load=${e.name}>Load</button>`}</td>
            </tr>
            ${t||e.error||n?i`<tr class="is-warn" data-decider-note=${e.name}><td colspan="9" class="small">
                  ${t?i`<span class="warn-ink" data-refusal=${e.name}>${a(`warn`)} ${t}</span> `:l}
                  ${e.error?i`<span class="err-ink">${e.error}</span> `:l}
                  ${n?i`<span role="status">${n}</span>`:l}
                </td></tr>`:l}`})}</tbody>
      </table></div>
      ${t.unserved?.length?i`<p class="muted small">In the document but not served by this engine: ${t.unserved.join(`, `)} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>`:l}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,{id:`deciders-models`,sub:`the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen`})}renderMemory(t){let n=t.planner,r=t.last_refusal;return e(`Memory`,i`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${E(n?.mem_available_gib)}</div></div>
          ${n?.engine_growth_gib==null?l:i`<div class="stat"><div class="stat-label">engine growth to its peak</div><div class="stat-value num">${E(n.engine_growth_gib)}</div></div>`}
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${E(n?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${E(n?.floor_gib)} + ${E(n?.margin_gib)}</div></div>
          <div class="stat ${ln(n)?`is-warn`:``}" data-decider-load-now><div class="stat-label">a load now</div><div class="stat-value num">${cn(n)}</div><div class="stat-sub">${sn(n)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${r?i`Last engine load not made: ${un(r)}, ${on(r)}`:i`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable, less what the engine may still take up to its declared peak, minus the claim stays at or above floor + margin, never during a hold, never beside another load. A model left out this way is not a fault: the room is kept for requests, because one long request is admitted by the engine's own peak, not by what is free, and the box must keep its floor at that peak. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,{id:`deciders-memory`})}renderApi(t){let n=this.ed.draft,r={...gn,...n?.api??{}},a=t.api?.stats,o=(e,t)=>n&&this.ed.set(T(n,[`api`,e],t)),s=(e,t,n=1)=>i`<label class="field"><span>${t}</span><input type="number" step=${n} .value=${String(r[e])} data-api-field=${e} @change=${t=>o(e,n<1?Number(t.target.value):Math.round(Number(t.target.value)))} /></label>`;return e(`Laya for scripts`,n?i`<div data-decider-api>
            <div class="toolbar" style="margin:0"><span class="muted small">route</span>${j([`off`,`on`],r.mode,e=>o(`mode`,e),{small:!0,label:`Laya route mode`})}<code class="small">POST /v1/laya/ask</code></div>
            <div class="form-grid">
              ${s(`rate_per_min`,`texts a minute (every caller)`)}
              ${s(`max_texts`,`texts in one call`)}
              ${s(`deadline_ms`,`deadline (ms) while a request runs`)}
              ${s(`threshold`,`sure at or above`,.05)}
              ${s(`drive_rate_per_s`,`texts a second (a playtest session)`)}
              ${s(`drive_sessions`,`playtest sessions at once`)}
            </div>
            <p class="muted small" data-api-stats>${a?`${p(a.calls)} calls, ${p(a.texts)} texts, ${p(a.errors)} errors (${p(a.rate_limited)} over the rate) since the start${a.last_ms==null?``:`; the last took ${m(a.last_ms)}`}${a.drive_sessions?`; ${p(a.drive_sessions)} playtest sessions live`:``}`:`no calls yet`}</p>
            <p class="muted small">The model's own scripts ask Laya typed questions about their evidence (a log tail, an output, a state dump) through the provider, open like the chat API: a choice, a scale or a yes-no question; Laya reads about 1,100 characters of each text, its head and its tail. Every call is a journal row (<code>laya_api</code>). A playtest (Laya plays the build: <code>.tandem/laya_drive.py</code>) sends a <code>session</code>: its own rate a second, one <code>laya_drive</code> row when it ends.</p>
          </div>`:c(2),{id:`deciders-api`,sub:`the route the model’s scripts call: on or off, its rate and its limits`})}renderQuestions(t){let n=this.ed,r=n.draft,a=t.models.map(e=>e.name),o=r?Object.entries(r.questions):[];return i`${e(`Questions`,r?i`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${o.map(([e,o])=>{let s=t.questions.find(t=>t.id===e);return i`<tr data-question=${e}>
                  <td class="mono">${e}</td>
                  <td><select aria-label=${`model for ${e}`} data-question-model=${e} @change=${t=>n.set(T(r,[`questions`,e],t.target.value))}>
                    ${[`auto`,...a].map(e=>i`<option value=${e} ?selected=${e===o}>${e===`auto`?`auto (the role’s model; the router picks)`:e}</option>`)}
                    ${o!==`auto`&&!a.includes(o)?i`<option value=${o} selected>${o} (not served)</option>`:l}
                  </select></td>
                  <td class="small">${dn(s?.picks??{})}</td>
                </tr>`})}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); <code>judge</code> is the test judge's questions on a run's evidence (the <code>tester</code> role), <code>laya_api</code> the model's own scripts through <code>/v1/laya/ask</code>; the session monitor's Tier 1 questions join this list when they land.</p>`:c(3),{id:`deciders-questions`,sub:`which model answers each question the conductor asks`})}
      ${A(n)}
      ${M(n,`the deciders document`)}`}renderTest(t){let n=this.tOut,r=n?.answers.q;return e(`Test box`,i`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${e=>this.tState=e.target.value} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${e=>this.tQuestion=e.target.value} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${e=>this.tOptions=e.target.value} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${e=>this.tModel=e.target.value} data-test-model>
            ${[`auto`,...t.models.map(e=>e.name)].map(e=>i`<option value=${e} ?selected=${e===this.tModel}>${e}</option>`)}
          </select></label>
        </div>
        <div class="savebar is-idle"><button class="btn btn-primary" ?disabled=${this.asking} @click=${()=>this.ask()} data-test-ask>${a(`send`)} ${this.asking?`Asking…`:`Ask`}</button>
          ${this.tErr?i`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>`:l}</div>
        ${n&&r?i`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${r.type===`choice`?r.choice:r.type===`noul`?(r.noul??0).toFixed(4):(r.score??0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${r.confidence.toFixed(2)} · ${n.model} · ${m(n.latency_ms)} · ${n.path} · ${p(n.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${mn(r).map(([e,t])=>i`<div class="prob-row" data-prob=${e}><span class="mono small">${e}</span><span class="conf-bar"><span style="width:${Math.round(t*100)}%"></span></span><span class="num small">${t.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${n.routing.reason}${n.routing.requested?` (wanted ${n.routing.requested})`:``}</p>
            </div>`:l}`,{id:`deciders-test`,sub:`one choice question, as /v1/systemone answers it (it runs in the idle window like any other)`})}};t([f()],K.prototype,`st`,void 0),t([f()],K.prototype,`error`,void 0),t([f()],K.prototype,`busy`,void 0),t([f()],K.prototype,`said`,void 0),t([f()],K.prototype,`tState`,void 0),t([f()],K.prototype,`tQuestion`,void 0),t([f()],K.prototype,`tOptions`,void 0),t([f()],K.prototype,`tModel`,void 0),t([f()],K.prototype,`tOut`,void 0),t([f()],K.prototype,`tErr`,void 0),t([f()],K.prototype,`asking`,void 0),K=t([r(`qse-cond-deciders`)],K);var q=class extends o{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=C(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await S.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let t=b(e);this.error=`${t.endpoint}: ${t.message}`}}async loadSession(e){try{let t=await S.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(t){e===this.selected&&(this.turns=[]),this.error=b(t).message}}render(){if(this.error&&!this.list)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${this.error}</div></div>`;if(!this.list)return c(6);let t=[...new Set(this.list.map(e=>e.client_kind??`other`))],n=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return i`
      ${this.selected?this.renderSession(this.selected):l}
      ${e(`Sessions, last 7 days`,n.length?i`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${n.map(e=>i`<tr class="req-row ${e.key===this.selected?`is-open`:``}" data-session=${e.key} @click=${()=>x({session:e.key})}>
                  <td class="mono small cell-wrap">${w(e.key,28)}</td><td>${e.client_kind??`—`}</td><td>${e.profile??`—`}</td>
                  <td>${O(e.class_id,this.ids)}</td><td class="num num-col">${p(e.turns)}</td>
                  <td class="num num-col">${e.notes?`${e.notes} (${p(e.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${e.acts?i`${p(e.acts)}${e.gave_ups?i` <span class="badge badge-warn">${e.gave_ups} gave up</span>`:l}`:`—`}</td>
                  <td class="num num-col" data-interventions>${p(e.interventions??0)}</td>
                  <td>${e.last_checker?i`<span class="badge ${e.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${e.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${_(new Date(e.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:s(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:i`<label class="field field-search">${a(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${t.map(e=>i`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?i`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:l}
    `}renderSession(e){let t=this.turns,n=this.live.find(t=>t.key===e);if(t==null)return c(4);let r=t.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,o=t.reduce((e,t)=>e+(t.injected?.length??0),0),s=this.events.filter(e=>ze(e)).length,u=be(t);return i`<section class="panel" style="margin-bottom:16px" data-timeline=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${w(e,44)} ${n?i`<span class="badge badge-on">live</span>`:l}</h2>
        <p class="panel-sub">${t.length} turns · ${o} hidden note${o===1?``:`s`} · ${s} monitor act${s===1?``:`s`}${n?` · class now ${n.class_id} · ${p(n.note_tokens)} note tokens replayed a turn`:``}${t.length?` · since ${v(new Date(t[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${S.exportUrl(e,`md`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${a(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${S.exportUrl(e,`json`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${a(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>x({session:null})}>${a(`x`)}</button></div></header>
      ${this.axis(He(t,this.events,this.ids,e=>g(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${t.map(e=>{let t=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/r*900)),n=(e.injected?.length??0)>0,a=!!e.shadow,o=(e.events??[]).filter(e=>e.decision!==`PASS`),s=o.some(e=>ze(e))?`act`:o.length?`shadow`:null;return i`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${t}px; --cls: var(--series-${le(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${xe(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${n?i`<span class="mark"></span>`:a?i`<span class="mark is-shadow"></span>`:l}
            ${s?i`<span class="mark is-monitor ${s===`shadow`?`is-shadow`:``}" data-monitor-mark=${s}></span>`:l}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${u.series.length?i`<qse-ribbon .series=${u.series} .labels=${u.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:l}
      <div class="feed" style="margin-top:10px">${t.slice().reverse().map(e=>st(e,this.ids,e=>this.open=e,xe(e)+((e.events??[]).some(e=>ze(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?i`<div style="margin-top:12px"><qse-cond-journal .session=${e}></qse-cond-journal></div>`:l}
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
    </figure>`}tier1(){let e=Ve(this.events);return e?i`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${e.agree} of ${e.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${e.decisions} shadow decisions)</span></p>`:l}};t([n({attribute:!1})],q.prototype,`ids`,void 0),t([n({attribute:!1})],q.prototype,`selected`,void 0),t([f()],q.prototype,`list`,void 0),t([f()],q.prototype,`live`,void 0),t([f()],q.prototype,`turns`,void 0),t([f()],q.prototype,`events`,void 0),t([f()],q.prototype,`error`,void 0),t([f()],q.prototype,`query`,void 0),t([f()],q.prototype,`client`,void 0),t([f()],q.prototype,`open`,void 0),q=t([r(`qse-cond-sessions`)],q);var _n={scope:`tools`,retention_days:30,clip_chars:4e3,loop_chars:200},vn=/^### (user|assistant|summary|runtime|tool result \((.*)\)|tool call \((.*)\)|reasoning \((.*)\))$/;function yn(e){let t=e[1];return e[2]===void 0?e[3]===void 0?e[4]===void 0?{kind:t===`user`||t===`assistant`||t===`summary`||t===`runtime`?t:`other`,title:t,name:null,clipped:!1}:{kind:`reasoning`,title:t,name:null,clipped:!1}:{kind:`tool_call`,title:t,name:e[3].trim()||null,clipped:!1}:{kind:`tool_result`,title:t,name:e[2].split(`,`)[0].trim()||null,clipped:/clipped to/.test(e[2])}}function bn(e){let t=e.replace(/\n+$/,``).split(`
`),n=t[0]??``,r=null,i=[],a=null,o=()=>{a&&i.push({kind:a.kind,title:a.title,name:a.name,clipped:a.clipped,body:a.lines.join(`
`).replace(/^\n+|\n+$/g,``)})};for(let e of t.slice(1)){let t=e.match(vn);t?(o(),a={...yn(t),lines:[]}):a?a.lines.push(e):e.startsWith(`compaction: `)&&(r=e)}o();let s=i.find(e=>e.kind===`runtime`);return{head:n,compaction:r,sections:i.filter(e=>e.kind!==`runtime`),runtime:s?xn(s.body):[]}}function xn(e){return e.split(`
`).filter(e=>e.trim()).map(e=>{let t=[`hidden`,`replayed`,`acts`,`checker`,`snapshot`,`finish`,`redactions`].find(t=>e.startsWith(`${t}: `))??`other`,n=e.match(/ · journal ([A-Za-z0-9_-]{4,40})$/),r=t===`checker`?e.match(/ (PASS|FAIL)(?: <-|$)/):null;return{kind:t,text:e,journal:n?n[1]:null,verdict:r?r[1]:null}})}function Sn(e){if(!e.counts)return`counts unknown (no meta.json)`;let t=e.counts,n=[`${t.blocks} turn${t.blocks===1?``:`s`}`];return t.untranscribed&&n.push(`${t.untranscribed} without tools (counted only)`),n.push(`${t.acts} act${t.acts===1?``:`s`}`),t.snapshots&&n.push(`${t.snapshots} snapshot${t.snapshots===1?``:`s`}`),t.compactions&&n.push(`${t.compactions} compaction${t.compactions===1?``:`s`}`),t.redactions&&n.push(`${t.redactions} redacted`),e.checkers?.last&&n.push(e.checkers.last),n.join(` · `)}function Cn(e){return{..._n,...e?.memory?.transcripts??{}}}function wn(e,t){return e?e.replace(/^[ch]:/,``):t}var Tn={mode:`on`,budget_gb:16,ttl_days:7,admit_after:2,min_free_gb:50},En={"opencode-head":`opencode head`,head:`session head`,package:`package`,memory:`memory file`,prefix:`repeated prefix`};function Dn(e){return{...Tn,...e?.memory?.disk_cache??{}}}function On(e){let t=[],n=e=>Number.isFinite(e)&&Math.round(e)===e;return e.budget_gb>=1&&e.budget_gb<=4e3||t.push(`the budget is 1 to 4,000 GB`),n(e.ttl_days)&&e.ttl_days>=1&&e.ttl_days<=90||t.push(`an entry is kept 1 to 90 days`),n(e.admit_after)&&e.admit_after>=1&&e.admit_after<=10||t.push(`a prefix is written after 1 to 10 prefills`),e.min_free_gb>=50&&e.min_free_gb<=4e3||t.push(`the NVMe keeps at least 50 GB free`),t}function kn(e,t){if(t.free_bytes==null)return{usable_gb:null,short:!1};let n=((t.bytes??0)+Math.max(0,t.free_bytes-Math.max(50,e.min_free_gb)*1e9))/1e9;return{usable_gb:n,short:n<e.budget_gb}}function An(e){let t=e.lookups??0;return t?`${Math.round((e.hit_rate??0)*100)} % of ${t.toLocaleString(`en-US`)} prefill${t===1?``:`s`} it could have shortened`:`no prefill the disk could have shortened yet`}function jn(e){let t=e=>typeof e==`number`?e.toLocaleString(`en-US`):String(e),n=[];return e.tokens!=null&&n.push(`${t(e.tokens)} tokens`),e.kind===`hit`&&e.read!=null&&n.push(`${t(e.read)} read`,`${t(e.ms)} ms`),e.kind===`write`&&(n.push(`${t(e.chunks)} chunk${e.chunks===1?``:`s`}`,`${t(e.anchors)} anchor${e.anchors===1?``:`s`}`),e.dropped&&typeof e.dropped==`object`&&n.push(`dropped ${Object.entries(e.dropped).map(([e,t])=>`${t} ${e}`).join(`, `)}`)),e.why!=null&&n.push(String(e.why)),e.kind===`clear`&&n.push(`${t(e.entries)} entries`),e.error!=null&&n.push(String(e.error)),[e.kind,...n].join(` · `)}var Mn=e=>v(new Date(e*1e3).toISOString()),Nn=e=>Object.entries(e??{}).filter(([,e])=>e).map(([e,t])=>`${p(t)} ${e}`).join(`, `)||`none`,J=class extends o{constructor(...e){super(...e),this.disk=null,this.error=null,this.busy=!1,this.cleared=null,this.settings=new D(this,`settings`)}connectedCallback(){super.connectedCallback(),this.load(),this.settings.load(),this.settings.onSaved=()=>void this.load()}async load(){try{this.disk=await S.memoryDisk(),this.error=null}catch(e){this.error=b(e).message}}async clear(){let e=this.disk;if(e&&await k(`Clear the disk cache?`,[`Every entry goes: ${p(e.entries??0)} prefixes in ${p(e.chunks??0)} chunk files, ${y(e.bytes??0)}, and the counts that decide what is written next.`,`Nothing a request needs is lost: the next prefill of each prefix runs cold, as without the tier, and writes it again.`],`Clear`)){this.busy=!0;try{let e=await S.memoryDiskClear();this.cleared=`cleared ${p(e.entries)} entries, ${y(e.bytes)}`,this.disk=e.disk}catch(e){this.error=b(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.disk)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${this.error}</div></div>`;if(!this.disk)return c(5);let e=this.disk;return i`${e.running?this.renderRunning(e):this.renderOff(e)}${this.renderSettings(e)}`}renderOff(t){return e(`Disk cache`,s(`The disk tier does not run on this server`,i`It starts with the engine when <code>QSE_DISK_CACHE</code> in serve.env names its directory (the flag is <code class="mono">${t.flag}</code> now)${t.free_bytes==null?l:i`; the NVMe has <span class="num">${y(t.free_bytes,0)}</span> free`}. Its settings below apply once it runs.`),{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM`})}renderRunning(t){let n=t.bytes??0,r=t.budget??0,o=r?n/r:0,c=t.free_bytes??null,d=c!=null&&t.floor!=null&&c<t.floor+1e10,f=i`<div class="stats disk-stats" data-disk-tiles>
      ${u({label:`on disk`,value:y(n,1),sub:i`of <span class="num">${y(r,0)}</span> · ${p(t.entries??0)} prefix${t.entries===1?``:`es`}, ${p(t.chunks??0)} chunk files`,warn:o>.95})}
      ${u({label:`NVMe free`,value:c==null?`—`:y(c,0),sub:i`never under <span class="num">${y(t.floor??5e10,0)}</span>`,warn:d})}
      ${u({label:`hit rate`,value:t.hit_rate==null?`—`:h(t.hit_rate,0),sub:An(t),na:t.hit_rate==null})}
      ${u({label:`restored`,value:p(t.tokens_restored??0),unit:`tokens`,sub:t.last_restore_ms==null?`no restore yet`:i`last <span class="num">${p(Math.round(t.last_restore_ms))}</span> ms${t.last_read_gbs==null?l:i` at <span class="num">${t.last_read_gbs.toFixed(1)}</span> GB/s`}`})}
    </div>`,m=i`<div class="meter-block disk-meter">
      <div class="meter-head"><span>budget</span><span class="num">${y(n,1)} of ${y(r,0)} · ${h(o,0)}</span></div>
      <div class="meter" role="img" aria-label=${`disk cache ${h(o,0)} of its budget`}><span class="meter-fill ${o>.95?`is-warn`:``}" style="width:${Math.min(100,o*100).toFixed(1)}%"></span></div>
    </div>`,g=t.top??[],_=i`${f}${m}
      <p class="small muted disk-line" data-disk-counters>
        ${t.on?i`<span class="badge badge-on">on</span>`:i`<span class="badge">off (settings)</span>`}
        written: <span class="num">${p(t.writes_chunk??0)}</span> chunks, <span class="num">${p(t.writes_anchor??0)}</span> anchors, <span class="num">${y(t.bytes_written??0,1)}</span> ·
        evicted: ${Nn(t.evicted)} · dropped: ${Nn(t.dropped)}${t.failed_restores?i` · <span class="warn-ink">${p(t.failed_restores)} failed restore${t.failed_restores===1?``:`s`}</span>`:l}
        · staging buffer <span class="num">${y(t.staging_bytes||t.staging_planned||0,2)}</span>${t.staging_bytes?``:` (when first used)`} · <span class="mono">${t.root??``}</span>
      </p>
      ${g.length?this.renderEntries(g):s(`Nothing on disk yet`,`A prefix is written after ${t.settings.admit_after===1?`its first prefill`:`${t.settings.admit_after} prefills`} of at least ${p(t.min_tokens??4096)} tokens, and the first long request of an opencode session is written where its user message starts.`)}
      ${this.renderEvents(t)}`;return e(`Disk cache`,_,{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM; a hit is read only past the rows already in place`,tools:i`${this.cleared?i`<span class="small muted" data-disk-cleared>${this.cleared}</span>`:l}
        <button class="btn btn-sm disk-clear" ?disabled=${this.busy||!(t.entries||t.chunks)} @click=${()=>this.clear()} data-disk-clear>${a(`x`)} Clear disk cache</button>
        <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the disk cache">${a(`refresh`)}</button>`})}renderEntries(e){let t=e=>i`<span class="badge disk-kind disk-kind-${e.kind}">${En[e.kind]}</span>${e.label?i` <span class="mono small">${e.label}</span>`:l}${e.complete?l:i` <span class="badge badge-warn" title="a chunk below it is not on disk: it is not read, and goes first">incomplete</span>`}`,n=e=>e.preview?i`<div class="small muted disk-preview" data-disk-preview>${e.preview}</div>`:l;return i`<h3 class="disk-sub">Most-hit prefixes</h3>
      <div class="table-scroll mem-table"><table class="table" data-disk-entries>
        <thead><tr><th class="num-col">tokens</th><th>what</th><th class="num-col">hits</th><th>last used</th><th class="num-col">a hit reads</th></tr></thead>
        <tbody>${e.map(e=>i`<tr data-disk-row=${`${e.kind}-${e.tokens}`}><td class="num num-col">${p(e.tokens)}</td><td class="cell-wrap">${t(e)}${n(e)}</td>
            <td class="num num-col">${p(e.hits)}</td><td class="num">${Mn(e.used)}</td><td class="num num-col">${y(e.reads_bytes,1)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards" role="list">${e.map(e=>i`<div class="mem-card disk-card" role="listitem" data-disk-card=${`${e.kind}-${e.tokens}`}>
          <div class="mem-card-head"><span class="num">${p(e.tokens)} tokens</span><span class="num small">${p(e.hits)} hit${e.hits===1?``:`s`}</span></div>
          <div>${t(e)}</div>${n(e)}
          <div class="small muted">last used ${Mn(e.used)} · a hit reads ${y(e.reads_bytes,1)}</div>
        </div>`)}</div>`}renderEvents(e){let t=(e.events??[]).slice(-10).reverse();return t.length?i`<details class="sec" style="margin-top:12px" data-disk-events>
      <summary>Last events <span class="sec-sub">what it read, wrote, dropped and evicted (the journal keeps them, kind memory)</span></summary>
      <ul class="disk-events">${t.map(e=>i`<li class="small"><span class="num muted">${Mn(e.ts)}</span> <span class="disk-ev disk-ev-${e.kind}">${jn(e)}</span></li>`)}</ul>
    </details>`:l}renderSettings(e){let t=this.settings;if(!t.draft)return t.loadError?i`<p class="err-ink small">${t.loadError}</p>`:l;let n=Dn(t.draft),r=(e,r)=>t.set(T(t.draft,[`memory`,`disk_cache`],{...n,[e]:r})),o=(e,t,a,o,s=1)=>i`<label class="field"><span>${t}</span><input type="number" inputmode="decimal" min=${a} max=${o} step=${s} .value=${String(n[e])} data-disk-setting=${e} @input=${t=>r(e,Number(t.target.value))} /></label>`,s=kn(n,e),c=On(n);return i`<details class="sec" style="margin-top:16px" data-sec="disk" open>
      <summary>Disk cache settings <span class="sec-sub">on or off, its size on the NVMe, how long it keeps an entry, what it writes</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-disk-mode><span class="muted small">disk cache</span>${j([`on`,`off`],n.mode,e=>r(`mode`,e),{small:!0,label:`disk cache mode`})}<span class="small">${n.mode===`on`?`prefixes are read from and written to the disk`:`nothing is read or written; the files stay`}</span></div>
        <div class="form-grid">
          ${o(`budget_gb`,`cache budget (GB)`,1,4e3)}
          ${o(`ttl_days`,`drop an entry unused for (days)`,1,90)}
          ${o(`admit_after`,`write a prefix after it was prefilled (times)`,1,10)}
          ${o(`min_free_gb`,`keep free on the NVMe (GB, at least 50)`,50,4e3)}
        </div>
        <p class="small" data-disk-free>
          ${e.free_bytes==null?l:i`The NVMe has <span class="num">${y(e.free_bytes,0)}</span> free${e.bytes?i`, and the cache holds <span class="num">${y(e.bytes,1)}</span>`:l}. `}
          ${s.short?i`<span class="warn-ink" data-disk-short>${a(`warn`)} At most <span class="num">${s.usable_gb.toFixed(0)} GB</span> of the ${n.budget_gb} GB fit above the ${Math.max(50,n.min_free_gb)} GB it keeps free; the tier stops there.</span>`:l}
        </p>
        <p class="muted small">Least valuable first when it is full: hits, times the tokens an entry saves per byte it holds, halved for every day unused. The first long request of an opencode session is written whatever the count, at the lines below where its user message starts. Applied from the next request; the free space it keeps is never under ${50} GB, whatever is set here.</p>
        ${c.length?i`<ul class="errors" role="alert" data-disk-errors>${c.map(e=>i`<li>${e}</li>`)}</ul>`:l}
        ${A(t)}
        ${M(t,`the settings`)}
      </div>
    </details>`}};t([f()],J.prototype,`disk`,void 0),t([f()],J.prototype,`error`,void 0),t([f()],J.prototype,`busy`,void 0),t([f()],J.prototype,`cleared`,void 0),J=t([r(`qse-cond-disk`)],J);var Y=e=>e?v(e):`—`,Pn=()=>new Date().toISOString().slice(0,10),X=class extends o{constructor(...e){super(...e),this.selected=null,this.node=null,this.user=`owner`,this.list=null,this.one=null,this.error=null,this.oneError=null,this.filter={...Gt},this.query=``,this.hits=null,this.injected=null,this.writes=null,this.busy=null,this.editing=null,this.raw=!1,this.diffAt=null,this.confirmUser=``,this.flash=null,this.settings=new D(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.reload(),this.settings.load()}willUpdate(e){e.has(`node`)&&(this.filter={...this.filter,node:this.node});let t=this.selected?`${this.user}#${this.selected}`:null;t!==this.loadedFor&&(this.loadedFor=t,this.one=null,this.oneError=null,this.editing=null,this.diffAt=null,this.selected&&this.loadOne(this.selected))}async reload(){try{let[e,t,n]=await Promise.all([S.notes(this.user),S.notesInjected(this.user,30),S.notesWrites(30)]);this.list=e,this.injected=t.rows,this.writes=n.rows,this.error=null}catch(e){this.error=b(e).message}}async loadOne(e){try{let t=await S.note(e,this.user);e===this.selected&&(this.one=t)}catch(t){e===this.selected&&(this.oneError=b(t).message)}}async act(e,t,n){this.busy=e,this.flash=null;try{await t(),n&&(this.flash=n),await this.reload(),this.selected&&await this.loadOne(this.selected)}catch(e){this.oneError=b(e).message}finally{this.busy=null}}async search(){let e=this.query.trim();if(!e){this.hits=null;return}try{this.hits=(await S.notesSearch(e,this.user,10)).hits}catch(e){this.error=b(e).message}}async download(){try{let e=await S.notesExport(this.user),t=document.createElement(`a`);t.href=URL.createObjectURL(e),t.download=`memory-${this.user}.zip`,t.click(),setTimeout(()=>URL.revokeObjectURL(t.href),1e3)}catch(e){this.error=b(e).message}}open(e){x({note:e})}render(){if(this.error&&!this.list)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${this.error}</div></div>`;if(!this.list)return c(6);let e=this.list;return i`
      ${this.renderTiles(e)}
      ${this.flash?i`<p class="ok-ink small" role="status" data-notes-flash>${a(`check`)} ${this.flash}</p>`:l}
      ${this.selected?this.renderOne():l}
      ${this.renderSearch()}
      ${this.renderBrowser(e)}
      ${this.renderInjected()}
      ${this.renderRuns()}
      ${this.renderWrites()}
      ${this.renderSettings()}
      ${this.renderDanger(e)}
    `}renderTiles(e){let t=new Map;for(let n of e.notes)t.set(n.kind,(t.get(n.kind)??0)+1);let n=e.notes.filter(e=>e.status===`provisional`).length,r=Pn(),a=(this.injected??[]).filter(e=>new Date(e.ts_ms).toISOString().slice(0,10)===r).length,o=(this.writes??[]).filter(e=>new Date(e.ts_ms).toISOString().slice(0,10)===r&&/^refused/.test(Yt(e))).length,s=[...t.entries()].sort((e,t)=>t[1]-e[1]).map(([e,t])=>`${t} ${e.replace(`_`,` `)}`).join(` · `);return i`<div class="stats notes-stats" data-notes-tiles>
      ${u({label:`live notes`,value:p(e.notes.length),sub:s||`none yet`})}
      ${u({label:`provisional`,value:p(n),sub:i`<span class="num">${p(e.notes.length-n)}</span> ratified by you`,warn:n>0&&n===e.notes.length&&e.notes.length>5})}
      ${u({label:`injected today`,value:p(a),unit:`requests`,sub:i`<span class="num">${p(e.memory.index_notes+e.memory.body_notes)}</span> notes since the start`})}
      ${u({label:`refused today`,value:p(o),sub:i`<span class="num">${p(e.memory.taken)}</span> blocks written, <span class="num">${p(e.memory.failed)}</span> refused since the start`})}
    </div>`}renderSearch(){return e(`Search`,this.hits===null?i`<p class="muted small">The runtime's own search (BM25 over title, tags and body, every scope of ${this.user}): the scores are the ones that decide what goes into a request.</p>`:this.hits.length?i`<ol class="notes-hits" data-notes-hits>${this.hits.map(e=>i`<li><button class="link-btn" @click=${()=>this.open(e.id)} data-hit=${e.id}>${e.title}</button>
                <span class="badge">${e.kind}</span> <span class="badge">${e.scope}</span> <span class="num small" data-score>${e.score.toFixed(2)}</span>
                ${e.status===`provisional`?i`<span class="badge badge-shadow">provisional</span>`:l}</li>`)}</ol>`:s(`No match`,`No note shares a word with the query.`),{tools:i`<form class="toolbar notes-search" style="margin:0" @submit=${e=>{e.preventDefault(),this.search()}}><label class="field field-search">${a(`search`)}<input placeholder="words, as the runtime would search them" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search the notes" data-notes-query /></label>
          <button class="btn btn-sm" type="submit" data-notes-search>Search</button>${this.hits?i`<button class="btn btn-sm btn-ghost" type="button" @click=${()=>(this.hits=null,this.query=``)}>Clear</button>`:l}</form>`,id:`notes-search`})}renderBrowser(t){let n=Kt(t.notes,this.filter),r=e=>x({nnode:e,note:null}),o=i`<nav class="notes-tree" aria-label="memory scopes" data-notes-tree>
      <button class="notes-node ${this.node?``:`is-on`}" @click=${()=>r(null)} data-node="all"><span>All notes</span><span class="num muted">${t.notes.length}</span></button>
      ${t.tree.map(e=>i`<button class="notes-node ${this.node===e.node?`is-on`:``}" @click=${()=>r(e.node)} data-node=${e.node} title=${e.label??e.node}>
          <span class="notes-node-label">${e.provisional?i`<span class="dot dot-shadow" aria-label="has provisional notes"></span>`:l}${qt(e)}</span>
          <span class="num muted">${e.count}</span></button>`)}
    </nav>`,c=i`<label class="field notes-picker"><span>scope</span><select @change=${e=>r(e.target.value||null)} aria-label="memory scope" data-notes-picker>
        <option value="" ?selected=${!this.node}>All notes (${t.notes.length})</option>
        ${t.tree.map(e=>i`<option value=${e.node} ?selected=${this.node===e.node}>${qt(e)} (${e.count})</option>`)}
      </select></label>`,u=this.filter,d=e=>this.filter={...this.filter,...e},f=i`<div class="toolbar notes-filters" style="margin:0 0 8px">
      <select aria-label="kind" @change=${e=>d({kind:e.target.value})} data-filter="kind"><option value="">every kind</option>${Wt.map(e=>i`<option value=${e} ?selected=${u.kind===e}>${e.replace(`_`,` `)}</option>`)}</select>
      <select aria-label="status" @change=${e=>d({status:e.target.value})} data-filter="status"><option value="">every status</option><option value="provisional">provisional</option><option value="ratified">ratified</option></select>
      <select aria-label="writer" @change=${e=>d({writer:e.target.value})} data-filter="writer"><option value="">every writer</option>${Object.entries(U).map(([e,t])=>i`<option value=${e}>${t}</option>`)}</select>
      <label class="switch small"><input type="checkbox" .checked=${u.stale} @change=${e=>d({stale:e.target.checked})} data-filter="stale" /> stale</label>
      <label class="field field-search">${a(`search`)}<input placeholder="filter by title or tag" .value=${u.q} @input=${e=>d({q:e.target.value})} aria-label="filter the list" /></label>
    </div>`,m=e=>i`<span class="badge ${e.status===`ratified`?`badge-ok`:`badge-shadow`}">${e.status}</span>${e.pinned?i` <span class="badge badge-on">pinned</span>`:l}${e.stale?i` <span class="badge badge-warn">stale</span>`:l}`,h=i`<div class="table-scroll notes-table"><table class="table" data-notes-list>
        <thead><tr><th>title</th><th>kind</th><th>status</th><th>updated</th><th>writer</th><th class="num-col">uses 7d</th></tr></thead>
        <tbody>${n.map(e=>i`<tr class="req-row ${e.id===this.selected?`is-open`:``}" data-note-row=${e.id} @click=${()=>this.open(e.id)}>
            <td class="cell-wrap">${e.title}</td><td><span class="badge" title=${Ut[e.kind]??``}>${e.kind.replace(`_`,` `)}</span></td><td>${m(e)}</td>
            <td class="num small">${Y(e.updated)}</td><td class="small">${U[e.writer]??e.writer}</td><td class="num num-col">${p(e.uses_7d??0)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards notes-cards" role="list">${n.map(e=>i`<div class="mem-card ${e.id===this.selected?`is-open`:``}" role="listitem" data-note-card=${e.id} @click=${()=>this.open(e.id)}>
          <div class="mem-card-head"><span class="small">${e.title}</span></div>
          <div class="small">${e.kind.replace(`_`,` `)} · ${m(e)}</div>
          <div class="small muted">${Y(e.updated)} · ${U[e.writer]??e.writer} · ${p(e.uses_7d??0)} uses in 7 days</div>
        </div>`)}</div>`,g=t.notes.length?s(`Nothing here`,`No note in this scope matches the filters.`):s(`No notes yet`,t.store.refused?`The memory directory is inside a git work tree (${t.store.refused}), so nothing is written.`:`The model writes a note when you ask it to remember something, correct it, or decide something; with the profile's memory switch on or in shadow.`);return e(`Notes`,i`<div class="notes-browser">${o}<div class="notes-main">${c}${f}${n.length?h:g}</div></div>`,{sub:i`<span data-notes-root>${p(t.notes.length)} live · ${p(t.archived.length)} archived${t.duplicates.length?i` · <span class="warn-ink">${t.duplicates.length} possible duplicate${t.duplicates.length===1?``:`s`}</span>`:l} · <span class="mono">${t.store.root}/${this.user}</span></span>`,tools:i`${t.users.length>1?i`<select aria-label="user" @change=${e=>(this.user=e.target.value,x({note:null,nnode:null}),void this.reload())} data-notes-user>${t.users.map(e=>i`<option ?selected=${e===this.user}>${e}</option>`)}</select>`:l}
          <button class="btn btn-sm" @click=${()=>this.download()} data-notes-export>${a(`down`)} Export</button>
          <button class="btn btn-sm" @click=${()=>this.reload()} aria-label="reload the notes">${a(`refresh`)}</button>`,id:`notes`})}renderOne(){if(this.oneError&&!this.one)return i`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${a(`warn`)} ${this.oneError}</div></div>`;let e=this.one;if(!e)return c(4);let t=e.note,n=!!t.archived,r=t.status===`ratified`&&(t.kind===`rule`||t.kind===`ruling`),o=e=>this.busy===e,s=t.id,u=n?i`<button class="btn btn-sm btn-danger" ?disabled=${o(`delete`)} @click=${()=>this.del(t)} data-note-act="delete">${a(`trash`)} Delete now</button>`:i`<button class="btn btn-sm" @click=${()=>this.startEdit(t)} data-note-act="edit">${a(`edit`)} Edit</button>
          ${t.status===`provisional`?i`<button class="btn btn-sm" ?disabled=${o(`ratify`)} @click=${()=>this.act(`ratify`,()=>S.noteStatus(s,{status:`ratified`},this.user),`ratified`)} data-note-act="ratify">${a(`check`)} Ratify</button>`:l}
          ${t.status===`provisional`?i`<button class="btn btn-sm" ?disabled=${o(`reject`)} @click=${()=>this.act(`reject`,()=>S.noteArchive(s,`rejected in the dashboard`,this.user),`rejected and archived`)} data-note-act="reject">Reject</button>`:l}
          ${r?i`<button class="btn btn-sm" ?disabled=${o(`pin`)} @click=${()=>this.act(`pin`,()=>S.noteStatus(s,{pinned:!t.pinned},this.user),t.pinned?`unpinned`:`pinned to every request`)} data-note-act="pin">${t.pinned?`Unpin`:`Pin to every request`}</button>`:l}
          <button class="btn btn-sm" ?disabled=${o(`archive`)} @click=${()=>this.archive(t)} data-note-act="archive">Archive</button>
          <button class="btn btn-sm btn-danger" ?disabled=${o(`delete`)} @click=${()=>this.del(t)} data-note-act="delete">${a(`trash`)} Delete</button>`,d=i`<dl class="notes-meta" data-note-meta>
      <dt>kind</dt><dd><span class="badge">${t.kind.replace(`_`,` `)}</span> <span class="muted small">${Ut[t.kind]??``}</span></dd>
      <dt>scope</dt><dd>${t.scope}${t.project?i` · <span class="mono small">${t.project}</span>`:l}${t.session?i` · <a class="mono small" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(t.session)}`}>${t.session}</a>`:l}</dd>
      <dt>status</dt><dd>${t.status}${t.pinned?` · pinned`:``}${t.stale?i` · <span class="warn-ink">stale since ${t.verify_by}</span>`:l}${n?i` · <span class="warn-ink">archived ${Y(t.archived)}: ${t.reason}</span>`:l}</dd>
      <dt>writer</dt><dd>${U[t.writer]??t.writer}</dd>
      <dt>tags</dt><dd>${t.tags.length?t.tags.map(e=>i`<span class="badge">${e}</span> `):i`<span class="muted">—</span>`}</dd>
      ${t.verify_by?i`<dt>verify by</dt><dd class="num">${t.verify_by}</dd>`:l}
      <dt>created</dt><dd class="num">${Y(t.created)}</dd><dt>updated</dt><dd class="num">${Y(t.updated)}</dd>
      <dt>sources</dt><dd>${t.sources.length?t.sources.map(e=>i`<span class="mono small">${e}</span> `):i`<span class="muted">—</span>`}</dd>
      <dt>used</dt><dd data-note-uses>${Qt(e.uses,Pn())}</dd>
    </dl>`,f=this.editing?this.renderEditor():i`<div class="notes-body">
        <div class="toolbar" style="margin:0 0 6px">${j([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`body view`})}</div>
        ${this.raw?i`<pre class="mem-text" data-note-raw>${t.body}</pre>`:i`<div class="notes-rendered" data-note-body>${Fn(t.body)}</div>`}
      </div>`;return i`<section class="panel notes-one" style="margin-bottom:16px" data-note=${s}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${t.title}</h2><p class="panel-sub mono">${s}</p></div>
        <div class="panel-tools notes-actions">${u}<button class="btn btn-ghost btn-icon" aria-label="Close the note" @click=${()=>x({note:null})}>${a(`x`)}</button></div></header>
      ${this.oneError?i`<p class="err-ink small" role="alert">${a(`warn`)} ${this.oneError}</p>`:l}
      <div class="notes-file">${d}${f}</div>
      ${this.renderHistory(e)}
    </section>`}startEdit(e){this.editing={title:e.title,body:e.body,kind:e.kind,tags:e.tags.join(`, `),verify_by:e.verify_by??``}}renderEditor(){let e=this.editing,t=t=>this.editing={...e,...t},n=e.body.trim()?e.body.trim().split(/\s+/).length:0,r=e.title.length>120||n>400||!e.body.trim();return i`<div class="notes-editor" data-note-editor>
      <label class="field"><span>title <span class="muted num">${e.title.length}/120</span></span><input .value=${e.title} @input=${e=>t({title:e.target.value})} data-edit="title" /></label>
      <div class="form-grid">
        <label class="field"><span>kind</span><select @change=${e=>t({kind:e.target.value})} data-edit="kind">${Wt.map(t=>i`<option value=${t} ?selected=${t===e.kind}>${t.replace(`_`,` `)}</option>`)}</select></label>
        <label class="field"><span>tags (comma)</span><input .value=${e.tags} @input=${e=>t({tags:e.target.value})} data-edit="tags" /></label>
        <label class="field"><span>verify by</span><input type="date" .value=${e.verify_by} @input=${e=>t({verify_by:e.target.value})} data-edit="verify_by" /></label>
      </div>
      <label class="field"><span>body <span class="muted num">${n}/400 words</span></span><textarea rows="8" .value=${e.body} @input=${e=>t({body:e.target.value})} data-edit="body"></textarea></label>
      <div class="toolbar" style="margin:0"><button class="btn btn-primary btn-sm" ?disabled=${r||this.busy===`edit`} @click=${()=>this.act(`edit`,async()=>{await S.noteEdit(this.selected,{title:e.title,body:e.body,kind:e.kind,tags:e.tags.split(`,`).map(e=>e.trim()).filter(Boolean),verify_by:e.verify_by||null},this.user),this.editing=null},`saved as a new revision, ratified`)} data-edit-save>Save as a revision</button><button class="btn btn-sm btn-ghost" @click=${()=>this.editing=null}>Cancel</button>
        ${r?i`<span class="warn-ink small">a title of at most 120 characters, a body of 1 to 400 words</span>`:i`<span class="muted small">Your edit ratifies the note; the old text stays in its history.</span>`}</div>
    </div>`}renderHistory(e){let t=e.revisions,n=e.note,r=[...t.map(e=>({label:`revision ${e.n} · ${Y(e.updated)} · ${U[e.writer]??e.writer}`,title:e.title,body:e.body})),{label:`now · ${Y(n.updated)}`,title:n.title,body:n.body}],a=this.diffAt,o=l;if(a!==null&&a+1<r.length){let e=r[a],t=r[a+1],n=Jt(`# ${e.title}\n${e.body}`,`# ${t.title}\n${t.body}`);o=i`<pre class="notes-diff" data-note-diff>${n.map(e=>i`<span class="diff-${e.op===`+`?`add`:e.op===`-`?`del`:`eq`}">${e.op===`=`?` `:e.op} ${e.text}\n</span>`)}</pre>`}return i`<details class="sec" ?open=${t.length>0} data-note-history>
      <summary>History <span class="sec-sub">${n.changelog.length} change${n.changelog.length===1?``:`s`}, ${t.length} earlier text${t.length===1?``:`s`}</span></summary>
      <div class="sec-body">
        <ul class="notes-changelog">${n.changelog.map(e=>i`<li class="small">${e}</li>`)}</ul>
        ${t.length?i`<ol class="notes-revs">${r.slice(0,-1).map((e,t)=>i`<li><button class="link-btn ${a===t?`is-on`:``}" @click=${()=>this.diffAt=a===t?null:t} data-rev=${t}>${e.label}</button> <span class="muted small">to the next</span></li>`)}</ol>${o}`:i`<p class="muted small">No earlier text: this is the note as it was written.</p>`}
      </div>
    </details>`}async archive(e){let t=window.prompt(`Why is this note archived? (kept with it for 30 days)`,`no longer true`);t!==null&&await this.act(`archive`,()=>S.noteArchive(e.id,t||`archived in the dashboard`,this.user),`archived`)}async del(e){await k(`Delete this note?`,[i`<strong>${e.title}</strong>`,`The file, its history and any archived copy go at once. This cannot be undone.`],`Delete`)&&(await this.act(`delete`,()=>S.noteDelete(e.id,this.user),`deleted`),x({note:null}))}renderInjected(){let e=this.injected??[];return i`<details class="sec" style="margin-top:16px" data-notes-injected>
      <summary>Injected <span class="sec-sub">the last requests the notes went into (the index once a session, the matching notes on a new message)</span></summary>
      <div class="sec-body">${e.length?i`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>session</th><th>what</th><th>notes</th></tr></thead><tbody>${e.map(e=>i`<tr><td class="num small">${v(new Date(e.ts_ms).toISOString())}</td>
              <td class="mono small cell-wrap">${e.session_key?i`<a href=${`#/conductor?screen=memory&memory=${encodeURIComponent(e.session_key)}`}>${e.session_key}</a>`:`—`}</td>
              <td class="small">${e.index?`index`:`bodies`}</td>
              <td class="cell-wrap">${e.ids.map(e=>i`<button class="link-btn mono small" @click=${()=>this.open(e)}>${e}</button> `)}</td></tr>`)}</tbody></table></div>`:i`<p class="muted small">No note went into a request yet.</p>`}</div>
    </details>`}renderRuns(){let e=(this.writes??[]).filter(e=>e.task===`writer`).map(Zt);return i`<details class="sec" data-notes-runs ?open=${e.length>0}>
      <summary>Writer runs <span class="sec-sub">the model's hidden turns that read a session and write its notes: after a compaction, after a give-up, at the session's end</span></summary>
      <div class="sec-body">${e.length?i`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>trigger</th><th>Laya's gate</th><th>outcome</th><th class="num-col">seconds</th><th class="num-col">tokens</th><th>notes</th><th>refused</th></tr></thead><tbody>${e.map(e=>i`<tr data-run=${e.trigger}><td class="num small">${v(new Date(e.when).toISOString())}</td><td>${e.trigger.replace(`_`,` `)}</td><td class="small">${e.gate}</td><td class="small">${e.outcome}</td>
              <td class="num num-col">${e.seconds??`—`}</td><td class="num num-col">${e.tokens??`—`}</td><td class="small">${e.ops}</td><td class="small warn-ink">${e.refused.join(`, `)}</td></tr>`)}</tbody></table></div>`:i`<p class="muted small">No run yet. The writer is ${this.list?.settings.writer?.mode??`off`} (settings below).</p>`}</div>
    </details>`}renderWrites(){let e=(this.writes??[]).filter(e=>e.task!==`writer`);return i`<details class="sec" data-notes-writes>
      <summary>Writes <span class="sec-sub">every block the model wrote, the script calls, your edits, the nightly job, and what was refused and why</span></summary>
      <div class="sec-body">${e.length?i`<ul class="notes-writes">${e.map(e=>i`<li class="small ${/^refused/.test(Yt(e))?`warn-ink`:``}"><span class="num muted">${v(new Date(e.ts_ms).toISOString())}</span> ${Yt(e)}</li>`)}</ul>`:i`<p class="muted small">Nothing written yet.</p>`}</div>
    </details>`}renderSettings(){let e=this.settings;if(!e.draft)return e.loadError?i`<p class="err-ink small">${e.loadError}</p>`:l;let t=Xt(e.draft),n=(n,r)=>e.set(T(e.draft,[`memory`,`notes`],{...t,[n]:r})),r=(e,r,a,o)=>i`<label class="field"><span>${r}</span><input type="number" min=${a} max=${o} .value=${String(t[e])} data-notes-setting=${e} @input=${t=>n(e,Math.round(Number(t.target.value)))} /></label>`;return i`<details class="sec" data-sec="notes-settings">
      <summary>Notes settings <span class="sec-sub">what goes into a request, the duplicate rule, Laya's remember question</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          ${r(`index_tokens`,`the index, once a session (tokens)`,0,8e3)}
          ${r(`body_tokens`,`the matching notes on a new message (tokens)`,0,4e3)}
          ${r(`pinned_tokens`,`pinned rules (tokens)`,0,2e3)}
          ${r(`bodies`,`matching notes at most`,0,10)}
          ${r(`min_words`,`words a note shares with the message at least`,1,10)}
          <label class="field"><span>a second note this similar is a revision</span><input type="number" min="0" max="1" step="0.05" .value=${String(t.dedupe_threshold)} data-notes-setting="dedupe_threshold" @input=${e=>n(`dedupe_threshold`,Number(e.target.value))} /></label>
        </div>
        <div class="toolbar" style="margin:0" data-notes-intent><span class="muted small">Laya asks "remember, forget or recall?"</span>${j([`off`,`shadow`,`on`],t.intent,e=>n(`intent`,e),{small:!0,label:`intent question`})}</div>
        <label class="switch small"><input type="checkbox" .checked=${t.provisional} @change=${e=>n(`provisional`,e.target.checked)} /> provisional notes go into requests too (marked)</label>
        ${this.renderWriterSettings(t,n)}
        <p class="muted small">A profile's <code>memory</code> switch turns the notes on (off, shadow: the model may write and nothing is read back, on). Every note passes the secret-pattern refusal; a rule about you needs your own message in the turn; the model never ratifies, pins or deletes.</p>
        ${A(e)}
        ${M(e,`the settings`)}
      </div>
    </details>`}renderWriterSettings(e,t){let n=e.writer,r=(e,r)=>t(`writer`,{...n,[e]:r}),a=(e,t,a,o)=>i`<label class="field"><span>${t}</span><input type="number" min=${a} max=${o} .value=${String(n[e])} data-writer-setting=${e} @input=${t=>r(e,Number(t.target.value))} /></label>`;return i`<h4 class="small" style="margin:12px 0 4px">The background writer</h4>
      <div class="toolbar" style="margin:0" data-writer-mode><span class="muted small">writer</span>${j([`off`,`shadow`,`on`],n.mode,e=>r(`mode`,e),{small:!0,label:`writer mode`})}
        <span class="muted small">after a compaction</span>${j([`off`,`idle`,`turn`],n.after_compaction,e=>r(`after_compaction`,e),{small:!0,label:`after a compaction`})}</div>
      <div class="form-grid">
        ${a(`idle_s`,`a session's end: seconds without a request`,10,86400)}
        ${a(`min_new_tokens`,`new tokens before the next pass`,0,1e6)}
        ${a(`max_tokens`,`the turn writes at most (tokens)`,64,4096)}
      </div>
      <label class="switch small"><input type="checkbox" .checked=${n.gate} @change=${e=>r(`gate`,e.target.checked)} /> Laya decides first whether the session has anything worth keeping (about 20 ms; a no skips the turn)</label>
      <p class="muted small">shadow: the turn runs and its notes are counted, nothing is stored. <code>turn</code> after a compaction writes before the next request gets the engine (the next turn waits up to ${n.turn_deadline_s} s); <code>idle</code> waits for an idle moment, which opencode rarely leaves right after a compaction.</p>`}renderDanger(e){let t=this.confirmUser===this.user;return i`<details class="sec" data-notes-danger>
      <summary>Delete ${this.user}'s memory <span class="sec-sub">every note, revision and archive of this user, at once</span></summary>
      <div class="sec-body">
        <p class="small">Type <code>${this.user}</code> to delete ${p(e.notes.length+e.archived.length)} notes. Export them first if you want a copy.</p>
        <div class="toolbar" style="margin:0"><label class="field"><span>user</span><input .value=${this.confirmUser} @input=${e=>this.confirmUser=e.target.value} data-notes-confirm aria-label="type the user's name" /></label>
          <button class="btn btn-sm btn-danger" ?disabled=${!t||this.busy===`user`} @click=${()=>this.act(`user`,()=>S.notesDeleteUser(this.user,this.confirmUser),`${this.user}'s memory deleted`).then(()=>this.confirmUser=``)} data-notes-delete-user>${a(`trash`)} Delete everything</button></div>
      </div>
    </details>`}};t([n({attribute:!1})],X.prototype,`selected`,void 0),t([n({attribute:!1})],X.prototype,`node`,void 0),t([f()],X.prototype,`user`,void 0),t([f()],X.prototype,`list`,void 0),t([f()],X.prototype,`one`,void 0),t([f()],X.prototype,`error`,void 0),t([f()],X.prototype,`oneError`,void 0),t([f()],X.prototype,`filter`,void 0),t([f()],X.prototype,`query`,void 0),t([f()],X.prototype,`hits`,void 0),t([f()],X.prototype,`injected`,void 0),t([f()],X.prototype,`writes`,void 0),t([f()],X.prototype,`busy`,void 0),t([f()],X.prototype,`editing`,void 0),t([f()],X.prototype,`raw`,void 0),t([f()],X.prototype,`diffAt`,void 0),t([f()],X.prototype,`confirmUser`,void 0),t([f()],X.prototype,`flash`,void 0),X=t([r(`qse-cond-notes`)],X);function Fn(e){let t=[],n=(e??``).split(`
`),r=0,a=e=>e.split(/(`[^`]+`)/).map(e=>e.startsWith("`")&&e.endsWith("`")&&e.length>1?i`<code>${e.slice(1,-1)}</code>`:e);for(;r<n.length;){let e=n[r];if(e.startsWith("```")){let e=[];for(r++;r<n.length&&!n[r].startsWith("```");)e.push(n[r++]);r++,t.push(i`<pre class="mem-text">${e.join(`
`)}</pre>`)}else if(/^\s*[-*] /.test(e)){let e=[];for(;r<n.length&&/^\s*[-*] /.test(n[r]);)e.push(n[r++].replace(/^\s*[-*] /,``));t.push(i`<ul>${e.map(e=>i`<li>${a(e)}</li>`)}</ul>`)}else if(!e.trim())r++;else{let e=[];for(;r<n.length&&n[r].trim()&&!n[r].startsWith("```")&&!/^\s*[-*] /.test(n[r]);)e.push(n[r++]);t.push(i`<p>${a(e.join(` `))}</p>`)}}return i`${t}`}var In={tools:`requests with tools in full (and compactions); chats without tools as counters only`,all:`every request in full`,none:`nothing`},Z=class extends o{constructor(...e){super(...e),this.ids=[],this.selected=null,this.page=null,this.tab=null,this.note=null,this.nnode=null,this.list=null,this.one=null,this.error=null,this.oneError=null,this.query=``,this.busy=null,this.row=null,this.ev=null,this.rowError=null,this.settings=new D(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.tab!==`disk`&&this.tab!==`notes`&&(this.loadList(),this.settings.load())}updated(e){let t=e=>e===`disk`||e===`notes`;t(e.get(`tab`))&&!t(this.tab)&&!this.list&&!this.error&&(this.loadList(),this.settings.draft||this.settings.load())}willUpdate(){let e=this.selected?`${this.selected}#${this.page??``}`:null;e!==this.loadedFor&&(this.loadedFor=e,this.one=null,this.oneError=null,this.selected&&this.loadOne(this.selected,this.page))}async loadList(){try{this.list=await S.memorySessions(),this.error=null}catch(e){this.error=b(e).message}}async loadOne(e,t){try{let n=await S.memorySession(e,t);e===this.selected&&(this.one=n)}catch(t){e===this.selected&&(this.oneError=b(t).message)}}async keep(e,t){if(e.session){this.busy=e.dir;try{await S.memoryKeep(e.session,t),await this.loadList(),this.one&&this.one.session.session===e.session&&(this.one={...this.one,session:{...this.one.session,keep:t}})}catch(e){this.error=b(e).message}finally{this.busy=null}}}async download(e,t){try{let n=await S.memoryExport(e,t),r=document.createElement(`a`);r.href=URL.createObjectURL(n),r.download=`transcript-${e.replace(/[^a-z0-9]+/gi,`-`)}.${t}`,r.click(),setTimeout(()=>URL.revokeObjectURL(r.href),1e3)}catch(e){this.oneError=b(e).message}}async openRow(e){this.rowError=null;try{let t=(await S.decision(e)).decision;t.kind===`monitor`?this.ev=t:this.row=t}catch(t){this.rowError=`journal row ${e}: ${b(t).message} (the journal keeps 30 days)`}}render(){let e=this.tab===`disk`,t=this.tab===`disk`||this.tab===`notes`?this.tab:null,n=(e,n)=>i`<a class="cond-tab ${e===t?`is-on`:``}" href="#" aria-current=${e===t?`page`:l} data-mem-tab=${e??`sessions`} @click=${t=>{t.preventDefault(),x({mtab:e,memory:null,mpage:null,note:null,nnode:null})}}>${n}</a>`,r=i`<nav class="mem-tabs" aria-label="memory tabs">${n(null,`Sessions`)}${n(`notes`,`Notes`)}${n(`disk`,`Disk cache`)}</nav>`;return e?i`${r}<qse-cond-disk></qse-cond-disk>`:t===`notes`?i`${r}<qse-cond-notes .selected=${this.note} .node=${this.nnode}></qse-cond-notes>`:i`
      ${r}
      ${this.selected?this.renderOne(this.selected):l}
      ${this.renderList()}
      ${this.renderSettings()}
      ${this.row?i`<qse-cond-decision .row=${this.row} .ids=${this.ids} .close=${()=>this.row=null}></qse-cond-decision>`:l}
      ${this.ev?i`<qse-cond-event .ev=${this.ev} .close=${()=>this.ev=null}></qse-cond-event>`:l}
    `}renderList(){if(this.error&&!this.list)return i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${this.error}</div></div>`;if(!this.list)return c(5);let t=this.list.writer,n=this.query.toLowerCase(),r=this.list.sessions.filter(e=>!n||`${e.session??e.dir} ${e.client??``} ${e.project??``}`.toLowerCase().includes(n)),o=e=>e.last_ms?v(new Date(e.last_ms).toISOString()):`—`,u=e=>i`<label class="switch" title="kept sessions are exempt from the ${t.settings.retention_days}-day prune"><input type="checkbox" .checked=${e.keep} ?disabled=${!e.session||this.busy===e.dir} data-keep=${e.dir} @click=${e=>e.stopPropagation()} @change=${t=>this.keep(e,t.target.checked)} aria-label=${`keep ${wn(e.session,e.dir)}`} /> keep</label>`,d=e=>e.checkers?.last?i`<span class="badge ${/ PASS$/.test(e.checkers.last)?`badge-ok`:`badge-warn`}">${e.checkers.last}</span>${e.checkers.PASS||e.checkers.FAIL?i` <span class="muted small num">${e.checkers.PASS??0} pass · ${e.checkers.FAIL??0} fail</span>`:l}`:i`<span class="muted">—</span>`,f=e=>e.session&&x({memory:e.session,mpage:null});return e(`Session transcripts`,r.length?i`<div class="table-scroll mem-table"><table class="table" data-memory-sessions>
              <thead><tr><th>last request</th><th>session</th><th>client</th><th>profile</th><th>project</th><th class="num-col">turns</th><th class="num-col">acts</th><th>checker</th><th class="num-col">size</th><th>keep</th></tr></thead>
              <tbody>${r.map(e=>i`<tr class="req-row ${e.session&&e.session===this.selected?`is-open`:``}" data-memory-row=${e.dir} @click=${()=>f(e)}>
                  <td class="num">${o(e)}</td><td class="mono small cell-wrap">${wn(e.session,e.dir)}</td><td>${e.client??`—`}</td><td>${e.profile??`—`}</td>
                  <td class="small cell-wrap">${e.project??`—`}</td>
                  <td class="num num-col">${e.counts?p(e.counts.blocks):`?`}</td><td class="num num-col">${e.counts?p(e.counts.acts):`?`}</td>
                  <td>${d(e)}</td><td class="num num-col">${y(e.bytes??0)}</td><td>${u(e)}</td></tr>`)}</tbody></table></div>
            <div class="mem-cards" role="list">${r.map(e=>i`<div class="mem-card ${e.session&&e.session===this.selected?`is-open`:``}" role="listitem" data-memory-card=${e.dir} @click=${()=>f(e)}>
                <div class="mem-card-head"><span class="mono small">${wn(e.session,e.dir)}</span>${u(e)}</div>
                <div class="small">${o(e)} · ${e.client??`—`} · ${e.profile??`—`}</div>
                <div class="small muted">${Sn(e)}</div>
                ${e.project?i`<div class="small muted cell-wrap">${e.project}</div>`:l}
                <div>${d(e)}</div>
              </div>`)}</div>`:s(`No transcripts yet`,t.refused?`The memory directory ${t.root} is inside a git work tree, so nothing is written there.`:`TandemLLM requests with tools are written to ${t.root} when their stream closes; chats without tools are counted only.`),{sub:i`<span data-writer>${p(t.blocks)} block${t.blocks===1?``:`s`} written since the start · ${p(t.redactions)} secret${t.redactions===1?``:`s`} redacted · ${p(t.failed)} failed · ${p(t.dropped)} dropped · ${y(t.bytes)} · <span class="mono">${t.root}</span></span>`,tools:i`<label class="field field-search">${a(`search`)}<input placeholder="search a session, client or project" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search transcripts" /></label>
          <button class="btn btn-sm" @click=${()=>this.loadList()} aria-label="reload the list">${a(`refresh`)}</button>`,id:`cond-memory`})}renderOne(e){if(this.oneError)return i`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${a(`warn`)} ${this.oneError}</div></div>`;let t=this.one;if(!t)return c(4);let n=t.session,r=t.pages??1,o=t.page??1,u=r>1?i`<div class="toolbar mem-pager" data-pager><button class="btn btn-sm" ?disabled=${o<=1} @click=${()=>x({mpage:String(o-1)})}>Older</button>
          <span class="small num">page ${o} of ${r} · ${p(t.total??t.blocks.length)} blocks, ${t.per_page} a page</span>
          <button class="btn btn-sm" ?disabled=${o>=r} @click=${()=>x({mpage:String(o+1)})}>Newer</button></div>`:l;return i`<section class="panel" style="margin-bottom:16px" data-transcript=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title mono">${wn(n.session,n.dir)}${n.keep?i` <span class="badge badge-on">kept</span>`:l}</h2>
        <p class="panel-sub">${n.client??`—`} · ${n.profile??`—`}${n.project?` · ${n.project}`:``} · ${Sn(n)}${n.first_ms?` · since ${v(new Date(n.first_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <button class="btn btn-sm" @click=${()=>this.download(e,`md`)} data-mem-export="md">${a(`down`)} Markdown</button>
          <button class="btn btn-sm" @click=${()=>this.download(e,`json`)} data-mem-export="json">${a(`down`)} JSON</button>
          <a class="btn btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e)}`} data-mem-timeline>Timeline</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the transcript" @click=${()=>x({memory:null,mpage:null})}>${a(`x`)}</button></div></header>
      ${this.rowError?i`<p class="warn-ink small" role="alert">${a(`warn`)} ${this.rowError}</p>`:l}
      ${u}
      ${t.blocks.length?i`<div class="mem-blocks">${t.blocks.map(e=>this.renderBlock(e))}</div>`:s(`No blocks`,`This session was counted but not written: its requests carried no tools.`)}
      ${u}
    </section>`}renderBlock(e){let t=bn(e.text),n=t.runtime.filter(e=>e.kind===`hidden`).length,r=t.runtime.find(e=>e.kind===`acts`),a=i`<span class="mem-turn num">turn ${e.turn??`?`}</span>
      <span class="muted small num">${e.time??``}</span>
      ${e.class_id?O(e.class_id,this.ids):l}
      ${e.effort?i`<span class="badge">${e.effort}</span>`:l}
      ${e.reasoning_tokens?i`<span class="muted small num">${p(e.reasoning_tokens)} reasoning tokens</span>`:l}
      ${n?i`<span class="badge badge-shadow" data-hidden-count=${n}>${n} hidden</span>`:l}
      ${r?i`<span class="badge badge-warn">${r.text.replace(/^acts: /,`acts `)}</span>`:l}`;return e.compaction?i`<details class="mem-block is-compaction" data-block=${e.turn??``} data-compaction>
        <summary><span class="mem-block-head">${a}<span class="badge">compaction</span><span class="small">${t.compaction?.replace(/^compaction: /,``)??``}</span></span></summary>
        ${t.sections.map(e=>this.renderSection(e))}${this.renderRuntime(t.runtime)}
      </details>`:i`<article class="mem-block" data-block=${e.turn??``}>
      <div class="mem-block-head">${a}</div>
      ${t.sections.map(e=>this.renderSection(e))}
      ${this.renderRuntime(t.runtime)}
    </article>`}renderSection(e){let t=e.body.length>600||e.body.split(`
`).length>12,n=i`<pre class="mem-text">${e.body}</pre>`;return e.kind===`tool_result`||e.kind===`tool_call`&&t||e.kind===`reasoning`?i`<details class="mem-sec mem-${e.kind}" ?open=${!t&&e.kind!==`reasoning`} data-sec-kind=${e.kind}>
        <summary class="small">${e.title}${e.clipped?i` <span class="badge badge-warn">clipped</span>`:l}</summary>${n}</details>`:i`<div class="mem-sec mem-${e.kind}" data-sec-kind=${e.kind}><div class="small muted">${e.title}</div>${n}</div>`}renderRuntime(e){return e.length?i`<ul class="mem-runtime" aria-label="what the runtime did">${e.map(e=>i`<li class="mem-rt mem-rt-${e.kind} ${e.verdict?`is-${e.verdict.toLowerCase()}`:``}" data-rt=${e.kind}>
        ${e.kind===`hidden`?a(`info`):l}<span>${e.text.replace(/ · journal [A-Za-z0-9_-]+$/,``)}</span>
        ${e.journal?i`<button class="btn btn-ghost btn-sm" @click=${()=>this.openRow(e.journal)} data-journal=${e.journal} aria-label=${`open journal row ${e.journal}`}>journal row</button>`:l}
      </li>`)}</ul>`:l}renderSettings(){let e=this.settings;if(!e.draft)return e.loadError?i`<p class="err-ink small">${e.loadError}</p>`:l;let t=Cn(e.draft),n=(n,r)=>e.set(T(e.draft,[`memory`,`transcripts`],{...t,[n]:r})),r=(e,r,a,o)=>i`<label class="field"><span>${r}</span><input type="number" min=${a} max=${o} .value=${String(t[e])} data-mem-setting=${e} @input=${t=>n(e,Math.round(Number(t.target.value)))} /></label>`;return i`<details class="sec" style="margin-top:16px" data-sec="transcripts">
      <summary>Transcript settings <span class="sec-sub">what the engine writes to the box, how long it keeps it, where tool output is clipped</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-mem-scope><span class="muted small">write</span>${j([`tools`,`all`,`none`],t.scope,e=>n(`scope`,e),{small:!0,label:`transcript scope`})}<span class="small">${In[t.scope]}</span></div>
        <div class="form-grid">
          ${r(`retention_days`,`keep a session for (days; kept sessions are exempt)`,1,400)}
          ${r(`clip_chars`,`clip a tool result or a call at (characters)`,200,1e5)}
          ${r(`loop_chars`,`a loop the D1 guard cut keeps its first and last (characters)`,0,4e3)}
        </div>
        <p class="muted small">Every block passes the secret-pattern refusal before it is written: a key, token or password becomes <code>[redacted:kind]</code> and the journal counts it, never the value. There is no switch for it. The files are <code>0600</code> under the engine's home on the box, never in a repository, and the ledger's backup never holds them. By design, chats without tools are counters only, unless the scope is widened here.</p>
        ${A(e)}
        ${M(e,`the settings`)}
      </div>
    </details>`}};t([n({attribute:!1})],Z.prototype,`ids`,void 0),t([n({attribute:!1})],Z.prototype,`selected`,void 0),t([n({attribute:!1})],Z.prototype,`page`,void 0),t([n({attribute:!1})],Z.prototype,`tab`,void 0),t([n({attribute:!1})],Z.prototype,`note`,void 0),t([n({attribute:!1})],Z.prototype,`nnode`,void 0),t([f()],Z.prototype,`list`,void 0),t([f()],Z.prototype,`one`,void 0),t([f()],Z.prototype,`error`,void 0),t([f()],Z.prototype,`oneError`,void 0),t([f()],Z.prototype,`query`,void 0),t([f()],Z.prototype,`busy`,void 0),t([f()],Z.prototype,`row`,void 0),t([f()],Z.prototype,`ev`,void 0),t([f()],Z.prototype,`rowError`,void 0),Z=t([r(`qse-cond-memory`)],Z);var Q=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([S.stats(168),S.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let t=b(e);this.error=`${t.endpoint}: ${t.message}`}}async promote(e,t,n,r,i){if(await k(`Promote ${n} to ${t===`task`?`act`:`on`} in ${e}?`,[t===`task`?`The session monitor will act on ${n} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${n} decisions instead of only journaling them: ${n===`effort`?`budgets and thinking off go to the engine`:n===`packages`?`skills and packages go into the prompt`:n===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${r.toFixed(2)} over ${i} labelled decisions. A new profile version records it.`],`Promote`))try{let a=await S.doc(`profiles`,e),o=t===`task`?T(a.doc,[`tasks`,n,`mode`],`act`):T(a.doc,[`decisions`,n],`on`),s=await S.put(`profiles`,e,o,`${n} to ${t===`task`?`act`:`on`}, precision ${r.toFixed(2)} / ${i}`,a.version);this.msg=`${e} v${s.version}: ${n} promoted`,this.onSaved()}catch(e){this.msg=b(e).message}}async loadAb(){try{this.abVersions=(await S.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await S.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await S.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=b(e).message}}render(){return this.error&&!this.stats?i`<div class="state state-error" role="alert"><div class="state-title">${a(`warn`)} ${this.error}</div></div>`:this.stats?i`
      ${this.msg?i`<div class="note note-info" role="status">${this.msg}</div>`:l}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:c(8)}renderPromotion(t){let n=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,r=(t.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return e(`Shadow precision and promotion`,i`<p class="muted small">${re}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${t.precision.length||r.length?i`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${t.precision.map(e=>{let t=te(e),r=n(e.profile,e.decision),a=e.backend?i`<div class="muted small mono" data-backend=${e.backend}>via ${e.backend}</div>`:l;return i`<tr data-precision-row=${`${e.profile}/${e.decision}${e.backend?`@${e.backend}`:``}`}>
                    <td>${e.profile}${a}</td><td class="mono">${e.decision}</td><td><span class="badge ${r===`on`?`badge-on`:r===`shadow`?`badge-shadow`:``}">${r}</span></td>
                    <td class="num num-col">${p(e.decisions)}</td><td class="num num-col">${p(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${t.allowed?``:`muted`}">${t.why}</td>
                    <td><button class="btn btn-sm ${t.allowed?`btn-primary`:``}" ?disabled=${!t.allowed||r!==`shadow`||!!e.backend} title=${e.backend?`a backend’s switches are on the Backends screen, per model`:t.allowed?``:re} data-promote=${`${e.profile}/${e.decision}`} @click=${()=>this.promote(e.profile??``,`decision`,e.decision,e.precision??0,e.labelled)}>Promote to on</button></td></tr>`})}
                ${r.map(e=>{let t=te(e);return i`<tr data-precision-row=${`task/${e.task}${e.backend?`@${e.backend}`:``}`}><td>${e.backend?i`<span class="muted small mono">via ${e.backend}</span>`:`—`}</td><td class="mono">${e.task}</td><td><span class="badge ${e.mode===`act`?`badge-on`:`badge-shadow`}">${e.mode}</span></td>
                    <td class="num num-col">${p(e.decisions)}</td><td class="num num-col">${p(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${t.allowed?``:`muted`}">${t.why}</td>
                    <td><button class="btn btn-sm ${t.allowed?`btn-primary`:``}" ?disabled=${!t.allowed||e.mode!==`shadow`||!!e.backend} data-promote=${`task/${e.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,e.task,e.precision??0,e.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:s(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${t.monitor?l:i`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let t=this.reports.find(e=>e.kind===`classifier`);if(!t)return e(`Classifier accuracy`,s(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let n=t.stages??{},r=t.set??{},a=(n[this.stage]??Object.values(n)[0])?.confusion??{},o=r.classes??Object.keys(a);return e(i`Classifier accuracy <span class="muted small num">${r.name??``} · ${r.n??`?`} prompts · ${t.created_at?v(String(t.created_at)):``}</span>`,i`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(n).map(([e,t])=>i`<tr class="req-row ${e===this.stage?`is-open`:``}" @click=${()=>this.stage=e}><td class="mono">${e}</td><td class="num num-col">${h(t.accuracy,1)}</td><td class="num num-col">${t.answered} / ${t.n}</td><td class="num num-col">${t.accuracy_answered==null?`—`:h(t.accuracy_answered,1)}</td><td class="num num-col">${m(t.p50_ms)}</td><td class="num num-col">${m(t.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${o.length&&Object.keys(a).length?i`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...o,`—`].map(e=>i`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${o.map(e=>i`<tr><th>${O(e,this.ids)}</th>${[...o,`—`].map(t=>{let n=a[e]?.[t]??0;return i`<td class="cm-cell ${e===t?`cm-diag`:n?`cm-miss`:``}">${n||``}</td>`})}</tr>`)}</tbody></table></div>`:i`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}
        ${De(t).length?i`<p class="muted small" style="margin-top:12px">Digest variants (SRV-153): <strong>today</strong> the 2,000-character digest the decider cut from the front, <strong>sized</strong> each stage's own head and tail, <strong>line</strong> sized plus the context line. <strong>router</strong> is the session's class after the stickiness rule.</p>
            <div class="table-scroll"><table class="table" data-classifier-variants>
              <thead><tr><th>digest</th><th>set</th><th class="num-col">n</th><th class="num-col">decider</th><th class="num-col">cascade</th><th class="num-col">router</th><th class="num-col">decider p50</th></tr></thead>
              <tbody>${De(t).map(e=>i`<tr><td class="mono">${e.variant}</td><td class="mono small">${e.set}</td><td class="num num-col">${e.n}</td><td class="num num-col">${e.decider==null?`—`:h(e.decider,1)}</td><td class="num num-col">${e.cascade==null?`—`:h(e.cascade,1)}</td><td class="num num-col">${e.router==null?`—`:h(e.router,1)}</td><td class="num num-col">${m(e.p50)}</td></tr>`)}</tbody></table></div>`:l}`,{id:`cond-classifier`})}renderSuites(){let t=this.reports.filter(e=>e.kind===`suite`),n=Te(this.reports,this.suiteFile),r=We(n,t),o=n?.rows??[],c=Be(this.reports,this.engine),u=Ae(this.reports,this.engine),d=Ne(o),f=n?.verdict,m=n?.comparison??[],g=n?.raw_row,_=n?.build,v=f?.indicative??se(o),y=(e,t=1)=>e==null?`—`:Number.isInteger(e)?p(e):e.toFixed(t);return e(i`TandemLLM against the raw engine ${n?i`<span class="muted small num">night ${n.night??`—`}${_?` · ${_.version??`?`} ${w(_.code_sha,12)}`:``}${n.profile?` · profile ${n.profile}`:``}</span>`:l}`,i`<p class="small muted" data-guard-detail=${c.status} data-speed-check=${u.state}>Speed check: ${u.text}. <span class="num">${c.line}${c.detail?i`. ${c.detail}`:l}</span></p>
        ${t.length>1?i`<label class="field suite-pick"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${t.map(e=>i`<option value=${String(e.file)} ?selected=${e===n}>${String(e.night??e.file)} · ${String(e.file)}${We(e,t)?` · superseded`:``}</option>`)}</select></label>`:l}
        ${r?i`<div class="guard-line guard-note" data-superseded>${a(`info`)}<span><span class="badge badge-warn">superseded</span> by <span class="mono">${r.file}</span>${r.why?i`: <span class="small">${r.why}</span>`:l}. Its numbers are kept as published; read the newer report.</span></div>`:l}
        ${f?i`<div class="note" data-eval-verdict>
              <span class="badge ${f.better?`badge-ok`:f.better===!1?`badge-warn`:``}"><strong>${f.better?`Better`:f.better===!1?`Not better`:`Not shown yet`}</strong></span>${v?i` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:l} — ${f.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${g?.status??`—`}${g?.source?` (${g.source})`:``}${n?.driver?` · driver ${n.driver.mode}`:``}${n?.isolation?` · ${n.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:l}
        ${d.length?i`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">turns</th><th class="num-col">thinking tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>efforts (turns)</th><th>build · night</th></tr></thead>
              <tbody>${d.flatMap(e=>{let t=m.find(t=>t.suite===e.suite||t.suite_id===(e.raw??e.tandem)?.suite_id);return[e.raw,e.tandem].filter(Boolean).map((n,r)=>i`<tr data-suite=${e.suite} data-arm=${n.arm}>
                    <td>${r===0?e.suite:``}${r===0&&!e.cmp.ok?i` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${e.cmp.why}</div>`:l}${r===0&&t?i` <span class="badge ${t.verdict===`better`?`badge-ok`:t.verdict===`worse`||t.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${t.verdict}>${t.verdict}</span>`:l}${r===0&&(n.indicative??(n.n??0)<10)?i` <span class="badge" data-row-indicative>indicative</span>`:l}</td>
                    <td class="mono">${_e(n.arm)}${n.pristine===!1?i` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:l}</td>
                    <td class="num num-col">${n.n??`—`}</td>
                    <td class="num num-col">${n.pass_rate==null?`—`:h(n.pass_rate,0)}</td>
                    <td class="num num-col">${n.gates_passed??`—`}${n.gates_total?` / ${n.gates_total}`:``}</td>
                    <td class="num num-col">${y(n.minutes_to_pass)}</td><td class="num num-col">${y(n.wall_minutes)}</td>
                    <td class="num num-col">${n.tokens==null?`—`:p(Math.round(n.tokens))}</td>
                    <td class="num num-col" data-turns>${y(n.turns)}</td><td class="num num-col" data-thinking>${n.reasoning_tokens==null?`—`:p(Math.round(n.reasoning_tokens))}</td>
                    <td class="num num-col">${y(n.decode_tps)}</td>
                    <td class="num num-col">${n.empty_stops??`—`}</td><td class="num num-col">${n.continuations??`—`}</td>
                    <td class="num num-col">${n.interventions??`—`}</td><td class="num num-col">${n.gave_ups??`—`}</td>
                    <td class="num num-col">${n.cache_hit==null?`—`:h(n.cache_hit,0)}</td>
                    <td class="small" data-efforts>${ce(n)}</td>
                    <td class="small">${w(n.build,16)} · ${n.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts. Tokens, turns and thinking tokens are medians per run.</p>`:s(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return e(`A/B of two profile versions`,i`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>i`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>i`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(t=>i`<option value=${t.version} ?selected=${t.version===this[e]}>v${t.version} ${w(t.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?i`<div class="diff" data-ab>${this.abChanges.map(e=>i`<div class="diff-row"><span>${Me(e.path)}</span><span class="diff-before">${w(e.before,80)}</span><span class="diff-after">${w(e.after,80)}</span></div>`)}</div>`:i`<p class="muted small">The two versions are the same.</p>`:l}`,{id:`cond-ab`})}};t([n({attribute:!1})],Q.prototype,`ids`,void 0),t([n({attribute:!1})],Q.prototype,`tiles`,void 0),t([n({attribute:!1})],Q.prototype,`onSaved`,void 0),t([f()],Q.prototype,`stats`,void 0),t([f()],Q.prototype,`reports`,void 0),t([f()],Q.prototype,`engine`,void 0),t([f()],Q.prototype,`error`,void 0),t([f()],Q.prototype,`msg`,void 0),t([f()],Q.prototype,`stage`,void 0),t([f()],Q.prototype,`abProfile`,void 0),t([f()],Q.prototype,`abVersions`,void 0),t([f()],Q.prototype,`abA`,void 0),t([f()],Q.prototype,`abB`,void 0),t([f()],Q.prototype,`abChanges`,void 0),t([f()],Q.prototype,`abCounts`,void 0),t([f()],Q.prototype,`suiteFile`,void 0),Q=t([r(`qse-cond-compare`)],Q);var Ln=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`deciders`,label:`Deciders`},{id:`sessions`,label:`Sessions`},{id:`memory`,label:`Memory`},{id:`compare`,label:`Compare`}],$=class extends o{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=C(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&Ln.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r]=await Promise.all([S.profiles(),this.meta?Promise.resolve(this.meta):S.meta(),S.config(),S.doc(`router`)]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.error=null}catch(e){let t=b(e);this.error=t.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${t.endpoint}: ${t.message}`}}async setMode(e){let t=this.tiles.find(e=>e.name===this.active);if(t&&!t.locked&&(e!==`on`||await k(`Turn the ${t.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let n=await S.setMode(t.name,e,`mode to ${e}`,t.version);this.reloaded=`v${n.version} · reloaded ${n.loaded_at?g(n.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=b(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),t=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return i`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${a(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>x({profile:e.target.value})}>
                ${this.tiles.map(e=>i`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?j([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):l}
            ${this.reloaded?i`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:l}
          </div>
        </div>
        ${this.switchError?i`<div class="guard-line" role="alert">${a(`warn`)} ${this.switchError}</div>`:l}
        ${this.error?i`<div class="guard-line" role="alert">${a(`warn`)} ${this.error}</div>`:l}
        ${t.map(([e,t])=>i`<div class="guard-line" role="alert">${a(`warn`)} ${e}: ${t.error?.message} (kept version ${t.version})</div>`)}
        ${e?.locked?i`<div class="note" data-locked>${a(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:l}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${Ln.map(e=>i`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:l}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return i`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return i`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return i`<qse-cond-monitor .meta=${e.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return i`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return i`<qse-cond-packages .ids=${e.ids} .tiles=${e.tiles} .selected=${this.params.get(`pkg`)} .tab=${this.params.get(`ptab`)} .pane=${this.params.get(`pview`)}></qse-cond-packages>`;case`deciders`:return i`<qse-cond-deciders></qse-cond-deciders>`;case`sessions`:return i`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`memory`:return i`<qse-cond-memory .ids=${e.ids} .selected=${this.params.get(`memory`)} .page=${Number(this.params.get(`mpage`))||null} .tab=${this.params.get(`mtab`)} .note=${this.params.get(`note`)} .nnode=${this.params.get(`nnode`)}></qse-cond-memory>`;case`compare`:return i`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};t([n({attribute:!1})],$.prototype,`params`,void 0),t([f()],$.prototype,`tiles`,void 0),t([f()],$.prototype,`defaultProfile`,void 0),t([f()],$.prototype,`meta`,void 0),t([f()],$.prototype,`config`,void 0),t([f()],$.prototype,`ids`,void 0),t([f()],$.prototype,`error`,void 0),t([f()],$.prototype,`switching`,void 0),t([f()],$.prototype,`reloaded`,void 0),t([f()],$.prototype,`switchError`,void 0),$=t([r(`qse-conductor`)],$);export{$ as QseConductor};