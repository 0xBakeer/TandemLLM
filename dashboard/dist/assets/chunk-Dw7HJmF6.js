import{A as e,B as t,D as n,E as r,F as i,I as a,L as o,N as s,P as c,R as l,U as u,V as d,_ as f,b as p,c as m,g as h,k as g,l as _,m as v,n as ee,p as te,r as y,t as ne,u as re,v as b,w as x,x as ie,y as S,z as C}from"./app-B2nIcfRo.js";import{$ as ae,A as oe,B as se,C as ce,D as le,DocEditor as w,E as ue,F as de,G as fe,H as pe,I as T,J as me,K as he,L as ge,M as _e,N as ve,O as ye,P as be,Q as xe,R as E,S as Se,T as Ce,U as we,V as Te,W as Ee,X as De,Y as Oe,Z as ke,_ as Ae,a as je,b as Me,c as Ne,clsChip as D,confView as Pe,confirmDialog as O,d as Fe,diffList as Ie,errorsList as k,et as A,eventRow as Le,f as Re,feedRow as ze,g as Be,h as Ve,i as He,j as Ue,k as We,l as Ge,m as Ke,modeSeg as j,n as qe,o as Je,p as Ye,q as Xe,s as Ze,saveBar as M,sourceWords as Qe,t as $e,u as et,v as tt,versionsPanel as nt,w as rt,withToken as N,x as it,y as at,z as ot}from"./chunk-CHHfio3O.js";import{n as st,t as ct}from"./chunk-CYQWl6dW.js";var P=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=m(()=>this.loadStats(),2e4),this.feedPoll=m(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await A.stats(24),this.error=null}catch(e){let t=x(e);this.error=`${t.endpoint}: ${t.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await A.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await A.event(e)).event}catch(e){this.error=x(e).message}}renderMonitor(e){let t=ye(e);return u`<div class="grid-2 grid-2-wide">
      ${f(`Monitor acts an hour, last 24 h`,e.acts?u`<qse-ribbon .series=${t} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:v(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${f(`Last actions of the monitor`,e.last_actions.length?u`<div class="feed" data-last-actions>${e.last_actions.map(e=>u`<button class="feed-row" data-action=${e.id} @click=${()=>this.openAction(e.id)}>
                <span class="feed-time">${i(new Date(e.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
                <span class="feed-conf muted small">${e.outcome?.kind??`running`}</span>
                <span class="feed-what">${_({task:e.task,decision:e.decision,mode:`act`,trigger:e.trigger,cap:e.cap})}</span>
                <span class="feed-act">${e.session_key?u`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:d}</span>
              </button>`)}</div>`:u`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:u`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${h(`external`)}</a>`})}
    </div>`}render(){let t=this.stats;if(this.error&&!t)return u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!t)return b(6);let n=t.classifier,r=n.decider,i=n.qwen,a=r?.calls?r:i?.calls?i:r,o=Object.entries(t.roles),l=o.filter(([,e])=>e.up).length,m=t.monitor;return u`
      <div class="stats" data-tiles>
        ${S({label:`decisions, 24 h`,value:e(t.decisions),sub:u`<span class="num">${e(t.acted)}</span> acted · <span class="num">${e(t.shadow)}</span> in shadow`,title:`${t.injections} notes, ${t.injected_tokens} tokens injected`})}
        ${S({label:`monitor, 24 h`,value:m?e(m.acts):`—`,sub:m?u`acts · <span class="num">${e(m.shadow)}</span> in shadow · <span class="num" data-gave-up>${e(m.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!m,warn:(m?.gave_up??0)>0,title:m?`${e(m.hidden_tokens)} hidden tokens`:void 0})}
        ${S({label:`classifier p50 / p95${!r?.calls&&i?.calls?` (Qwen)`:``}`,value:a?.p50_ms==null?`—`:`${Math.round(a.p50_ms)} / ${Math.round(a.p95_ms??0)}`,unit:a?.p50_ms==null?void 0:`ms`,sub:u`decider ${e(r?.calls??0)} calls${i?u` · Qwen ${e(i.calls)} (${s(i.p50_ms)})`:d} · fail-open ${c(t.fail_open_rate,1)}`,warn:(t.fail_open_rate??0)>.1})}
        ${S({label:`roles online`,value:`${l} / ${o.length}`,sub:u`${o.map(([e,t])=>u`<span title=${t.error??``} class=${t.up?``:t.mode===`off`?`muted`:`err-ink`} data-role=${e}>${t.up?`●`:`○`} ${e} </span>`)}`,warn:o.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${f(`Decisions an hour, last 24 h`,t.decisions?u`<qse-ribbon .series=${ce(t,this.ids)} .labels=${t.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:v(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${f(`Task classes, 24 h`,u`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${t.classes.map(t=>u`<tr><td>${D(t.class_id,this.ids)}</td><td class="num num-col">${e(t.decisions)}</td><td class="num num-col">${t.labelled?`${t.agree} / ${t.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(t.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(t.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${f(u`Live decisions ${this.paused?u`<span class="badge">paused</span>`:u`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?u`<div class="feed" data-feed>${this.feed.map(e=>ze(e,this.ids,e=>this.open=e,Me(e)))}</div>`:u`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:u`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${h(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${h(`external`)}</a>`})}
        ${f(`Profiles by client`,u`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(t=>u`<button class="pick" @click=${()=>p({screen:`profiles`,edit:e.name})}><span class="muted">${t}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(e=>u`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${m?this.renderMonitor(m):d}
      ${this.open?u`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:d}
      ${this.openEvent?u`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:d}
    `}};a([C({attribute:!1})],P.prototype,`ids`,void 0),a([C({attribute:!1})],P.prototype,`tiles`,void 0),a([C({attribute:!1})],P.prototype,`active`,void 0),a([l()],P.prototype,`stats`,void 0),a([l()],P.prototype,`feed`,void 0),a([l()],P.prototype,`error`,void 0),a([l()],P.prototype,`open`,void 0),a([l()],P.prototype,`openEvent`,void 0),a([l()],P.prototype,`paused`,void 0),P=a([t(`qse-cond-overview`)],P);var F=[`off`,`low`,`medium`,`high`,`xhigh`],lt=[`small`,`medium`,`large`],I=class extends o{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new w(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),A.doc(`packages`).then(e=>this.packages=Ae(e.doc)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(T(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await A.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=x(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return b(8);if(e.loadError)return u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=t.classes.map(e=>e.id);return u`
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
          ${t.classifier.stages.map((e,n)=>u`<div class="row-card" data-stage=${e.kind}>
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
                ${[`keep`,`classify`,...n].map(e=>u`<option value=${e} ?selected=${e===t.structural.compaction}>${e===`keep`?`keep the session class`:e===`classify`?`ask a classifier`:`class ${e}`}</option>`)}
              </select><span class="hint">the turn itself; the session keeps its class after it</span></label>
            <label class="field"><span>a turn that only carries tool results</span>
              <select @change=${e=>this.upd([`structural`,`tool_turn`],e.target.value)}>
                <option value="keep" ?selected=${t.structural.tool_turn===`keep`}>keep the session class (no model call)</option>
                <option value="classify" ?selected=${t.structural.tool_turn===`classify`}>ask a classifier every turn</option>
              </select></label>
            <label class="field" data-tool-turn-effort><span>effort of a tool-result turn</span>
              <select aria-label="effort of a tool-result turn" @change=${e=>{let t=e.target.value;this.upd([`structural`,`tool_turn_effort`],t===`inherit`?void 0:t)}}>
                ${Re.map(e=>u`<option value=${e} ?selected=${e===(t.structural.tool_turn_effort??`inherit`)}>${e===`inherit`?`the class's (inherit)`:me(e,t,null)}</option>`)}
              </select><span class="hint">a turn whose newest message is a tool result (a screenshot a tool returned too) thinks at this level, its switch and cap; the first turn and every new user message keep the class's, the note stays the class's. A profile can set its own (Profiles › Thinking); a client's own budget wins</span></label>
            <label class="field" data-side-requests><span>a client's title or summary request</span>
              <select aria-label="a client's title or summary request" @change=${e=>this.upd([`structural`,`side_requests`],e.target.value)}>
                <option value="on" ?selected=${(t.structural.side_requests??`on`)===`on`}>a side request: no class, thinking off, short, first in the queue</option>
                <option value="off" ?selected=${t.structural.side_requests===`off`}>a turn, as any request</option>
              </select><span class="hint">known by its shape: no tools and the client's own fixed instruction (${(t.structural.side_shapes??Ne).map(e=>`${e.client??`a client`}'s ${e.kind}, at most ${e.max_tokens} tokens`).join(`; `)}). It asks no classifier, gets no note and leaves the session's class alone; thinking only when the client asks for a number</span></label>
          </div>
        </div>
      </details>
      <details class="sec" data-sec="effort">
        <summary>Effort levels <span class="sec-sub">a budget and a sentence at the prompt's end; never the template's reasoning_effort</span></summary>
        <div class="sec-body">
          <p class="muted small">The template renders <code>reasoning_effort</code> as the first system tokens, so changing it per turn would cost a cold prefill. A level here sets the reasoning budget (0 = no cap below the backstop), thinking on or off (off uses <code>enable_thinking: false</code> only when the server's effort is medium, else a budget of 1), and a sentence in the tail note.</p>
          <p class="muted small">The budget is a real cap (SRV-159): when the reasoning reaches it, the server closes it with the session monitor's hidden close, so the client sees only <code>&lt;/think&gt;</code> and the answer. Defaults: low 1,024, medium 8,192, high 16,384, xhigh no cap. A profile can override a level (Profiles › Thinking); a client's own budget always wins.</p>
          ${F.map(e=>u`<div class="row-card" data-effort=${e}>
              <div class="row-card-head"><strong class="mono">${e}</strong>
                <label class="switch"><input type="checkbox" .checked=${t.effort.levels[e].thinking} @change=${t=>this.upd([`effort`,`levels`,e,`thinking`],t.target.checked)} /> thinking</label></div>
              <div class="form-grid">${this.num(`budget (tokens, 0 = no cap)`,t.effort.levels[e].budget,t=>this.upd([`effort`,`levels`,e,`budget`],Math.round(t)),{step:256,min:0})}</div>
              <label class="field field-block"><span>sentence in the note</span><textarea rows="2" .value=${t.effort.levels[e].note} @input=${t=>this.upd([`effort`,`levels`,e,`note`],t.target.value)}></textarea></label>
            </div>`)}
          <label class="switch"><input type="checkbox" .checked=${t.effort.raise_on_fail} @change=${e=>this.upd([`effort`,`raise_on_fail`],e.target.checked)} /> a failing checker (RESULT: FAIL) raises the effort one level for that turn</label>
          ${this.renderSize(t.effort.size??Ge)}
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
      ${k(e)}
      ${M(e,`the router`)}
    `}renderTest(e){let t=this.result;return u`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?u`<span class="err-ink small" role="alert">${this.testError}</span>`:d}
      </div>
      ${t?t.active?u`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${D(t.class_id??null,e)} ${Pe(t.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${Qe(t.source??null)}${t.kept?u` <span class="badge badge-warn">${t.kept}</span>`:d} <span class="muted num">${s(t.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${t.effort??`unchanged`} · thinking ${t.thinking??`on`}${t.budget==null?``:` · budget ${t.budget}`}</dd></div>
                  ${t.extra?.size?u`<div data-test-size><dt>task size</dt><dd>${se(t.extra.size)}</dd></div>`:d}
                  <div><dt>profile</dt><dd>${t.profile} (${t.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(t.stages??[]).map(e=>u`<div class="stage ${e.error?`is-error`:e.below?``:`is-accepted`}"><strong>${pe(e.stage)}</strong><span>${e.error??u`${e.label} ${Pe(e.confidence,e.accept??.6)}`}</span><span class="num muted">${s(e.ms)}</span></div>`)}</div>
              </div>
              <div>${t.note?u`<p class="muted small">The model would see this at the end of the last turn (${(t.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${t.note}</pre>`:u`<p class="muted small">No note: ${t.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:u`<p class="muted small" style="margin-top:12px">Profile ${t.profile} is raw (${t.mode}): the conductor does nothing for it.</p>`:d}
    </section>`}renderClass(e,t,n,r){let i=e.policy,a=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return u`<details class="row-card" data-class-row=${e.id} ?open=${a}>
      <summary>
        <span>${D(e.id,r,e.label)} <span class="muted small mono">${i.effort_by_size?lt.map(e=>ot(i,e)).join(`/`):i.effort??`medium`}${(i.packages??[]).length?` · ${(i.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===0} @click=${()=>this.moveClass(t,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===n-1} @click=${()=>this.moveClass(t,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n<=1} @click=${()=>this.removeClass(t)} aria-label=${`remove ${e.id}`}>${h(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${e.id} ?readonly=${!a} @input=${e=>this.upd([`classes`,t,`id`],e.target.value)} data-class-id /><span class="hint">${a?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${e.label} @input=${e=>this.upd([`classes`,t,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${e.description} @input=${e=>this.upd([`classes`,t,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${j(F,i.effort??`medium`,e=>this.upd([`classes`,t,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div class="toolbar size-map" style="margin:0" data-size-map=${e.id}><span class="muted small">by task size (the effort above is the ceiling)</span>
        ${lt.map(n=>u`<label class="field"><span>${n}</span><select aria-label=${`${e.id} effort when ${n}`} @change=${e=>this.setSizeLevel(t,i.effort_by_size,n,e.target.value)}>
            <option value="" ?selected=${!i.effort_by_size?.[n]}>the class's</option>
            ${F.map(e=>u`<option value=${e} ?selected=${i.effort_by_size?.[n]===e}>${e}</option>`)}
          </select>${i.effort_by_size?.[n]&&ot(i,n)!==i.effort_by_size[n]?u`<span class="hint">cut to ${ot(i,n)}</span>`:d}</label>`)}
      </div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let n=(i.packages??[]).includes(e);return u`<button class="pick ${n?`is-on`:``}" aria-pressed=${n?`true`:`false`} @click=${()=>this.upd([`classes`,t,`policy`,`packages`],n?(i.packages??[]).filter(t=>t!==e):[...i.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(i.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,t,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}renderSize(e){let t=(t,n)=>this.upd([`effort`,`size`],T(e,t,n));return u`<div class="row-card" data-size-config>
      <div class="row-card-head"><strong>Task size</strong>
        <label class="switch"><input type="checkbox" .checked=${e.reset_on_compaction} @change=${e=>t([`reset_on_compaction`],e.target.checked)} /> a compaction starts the size over</label></div>
      <p class="muted small">Each signal is small below its first number, medium up to its second, large above. The task's size is the largest any counting signal reached, and it never falls within a task. A class's levels per size (in its row above) pick the effort under the class's effort; whether that acts is each profile's <em>effort by task size</em> (off, shadow, on).</p>
      ${et.map(n=>{let r=e.signals[n.id]??Ge.signals[n.id];return u`<div class="size-signal" data-size-signal=${n.id}>
          <label class="switch"><input type="checkbox" .checked=${r.enabled} @change=${e=>t([`signals`,n.id,`enabled`],e.target.checked)} /> ${n.label}</label>
          <div class="form-grid">
            ${this.num(`small below (${n.unit})`,r.small,e=>t([`signals`,n.id],{...r,small:e}),{min:0,step:n.id===`session_tokens`?1e3:1})}
            ${this.num(`medium up to (${n.unit})`,r.medium,e=>t([`signals`,n.id],{...r,medium:e}),{min:0,step:n.id===`session_tokens`?1e3:1,hint:n.hint})}
          </div>
        </div>`})}
    </div>`}setSizeLevel(e,t,n,r){let i={...t??{}};r?i[n]=r:delete i[n],this.upd([`classes`,e,`policy`,`effort_by_size`],Object.keys(i).length?i:void 0)}renderContextStage(e){let t=tt(e),n=(n,r)=>this.upd([`classifier`,`context_stage`],{...t,...e.classifier.context_stage??{},[n]:r});return u`<div class="row-card" data-context-stage>
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
    </div>`}num(e,t,n,r={}){return u`<label class="field"><span>${e}</span><input type="number" .value=${String(t)} step=${r.step??1} min=${r.min??``} max=${r.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&n(t)}} />${r.hint?u`<span class="hint">${r.hint}</span>`:d}</label>`}text(e,t,n){return u`<label class="field"><span>${e}</span><input .value=${t} @input=${e=>n(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(T(n,[`classifier`,`stages`],r))}};a([C({attribute:!1})],I.prototype,`ids`,void 0),a([C({attribute:!1})],I.prototype,`meta`,void 0),a([C({attribute:!1})],I.prototype,`active`,void 0),a([C({attribute:!1})],I.prototype,`onSaved`,void 0),a([l()],I.prototype,`packages`,void 0),a([l()],I.prototype,`testText`,void 0),a([l()],I.prototype,`testProfile`,void 0),a([l()],I.prototype,`testTools`,void 0),a([l()],I.prototype,`testing`,void 0),a([l()],I.prototype,`result`,void 0),a([l()],I.prototype,`testError`,void 0),I=a([t(`qse-cond-routing`)],I);var ut=[`off`,`shadow`,`act`],dt=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,ft=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,L=class extends o{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new w(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),A.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(T(this.ed.draft,e,t))}render(){let e=this.ed,t=this.config?.server.monitor_mode;return u`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${e.loading&&!e.draft?b(8):e.loadError?u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`:u`
            ${t===`raw`?u`<div class="guard-line" data-monitor-off>${h(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:d}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(e.draft)}
            ${this.renderRest(e.draft)}
            ${k(e)}
            ${M(e,`the monitor`)}`}`}renderTasks(e){let t=We(e),n=t.filter(t=>e.tasks[t]&&/^C\d/.test(t)),r=t.filter(e=>Ce(e)),i=t.filter(e=>!n.includes(e)&&!r.includes(e)),a=r.map(t=>e.tasks[t]).find(e=>e.rule===`stall_watchdog`)?.notify;return u`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${n.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${n.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${r.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${r.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${a?.enabled?u`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:u`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, Khaled's ruling of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${r.map(t=>this.renderTask(t,e.tasks[t]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${i.join(`, `)}</span></summary>
        <div class="sec-body">${i.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>`}renderTask(e,t){let n=be(this.meta?.schemas?.monitor,t.rule),r=this.meta?.tasks?.[e];return u`<div class="row-card" data-monitor-task=${e}>
      <div class="row-card-head">
        <strong>${e}${r?u` · ${r.label}`:d} <span class="muted small mono">${t.rule}</span></strong>
        ${j(ut,t.mode,t=>this.upd([`tasks`,e,`mode`],t),{small:!0,label:`${e} mode without a profile`})}
      </div>
      ${r?.why?u`<p class="muted small">${r.why}</p>`:d}
      ${n?.properties?u`<div class="form-stack">${Object.entries(n.properties).map(([n,r])=>this.field(n,r,t[n],[`tasks`,e,n]))}</div>`:u`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,t,[`tasks`,e])}`}
    </div>`}field(e,t,n,r){let i=Array.isArray(t.type)?t.type[0]:t.type,a=e.replace(/_/g,` `),o=t.description?u`<span class="hint">${t.description}</span>`:d;if(t.enum)return u`<label class="field"><span>${a}</span><select data-param=${e} @change=${e=>this.upd(r,e.target.value)}>${t.enum.map(e=>u`<option value=${String(e)} ?selected=${e===n}>${String(e)}</option>`)}</select>${o}</label>`;if(i===`boolean`)return u`<label class="switch" data-param=${e}><input type="checkbox" .checked=${!!n} @change=${e=>this.upd(r,e.target.checked)} /> ${a}</label>${t.description?u`<p class="hint">${t.description}</p>`:d}`;if(i===`integer`||i===`number`)return u`<label class="field"><span>${a}</span><input type="number" data-param=${e} step=${i===`integer`?`1`:`0.01`} min=${t.minimum??d} max=${t.maximum??d} .value=${n==null?``:String(n)} @input=${e=>{let t=e.target.value;this.upd(r,t===``?null:i===`integer`?Math.round(Number(t)):Number(t))}} />${o}</label>`;if(i===`string`){let t=String(n??``);return dt.test(e)||t.length>60?u`<label class="field field-block"><span>${a}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(t.length/90)))} class=${ft.test(e)?`mono`:``} data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)}></textarea>${o}</label>`:u`<label class="field"><span>${a}</span><input data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)} />${o}</label>`}if(i===`array`&&t.items&&(Array.isArray(t.items.type)?t.items.type[0]:t.items.type)===`string`){let i=!Array.isArray(n)&&Array.isArray(t.default),s=Array.isArray(n)?n:i?t.default:[];return u`<label class="field field-block"><span>${a} <span class="muted small">(one a line${i?`; the default, not saved yet`:``})</span></span><textarea rows=${Math.min(8,Math.max(2,s.length+1))} data-param=${e} .value=${s.join(`
`)} @input=${e=>this.upd(r,le(e.target.value))}></textarea>${o}</label>`}if(i===`array`&&t.items?.properties){let i=Array.isArray(n)?n:[],s=t.items;return u`<fieldset class="fieldset" data-param=${e}><legend>${a} <span class="muted small">${i.length}</span></legend>${o}
        ${i.map((e,t)=>u`<div class="row-card row-card-sub" data-item=${String(e.id??t)}>
          <div class="row-card-head"><strong class="mono small">${String(e.id??`#${t+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(r,i.filter((e,n)=>n!==t))} aria-label=${`remove ${String(e.id??t+1)}`}>${h(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(s.properties).map(([n,i])=>this.field(n,i,e[n],[...r,t,n]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${e} @click=${()=>this.upd(r,[...i,pt(s)])}>${h(`plus`)} Add</button>
      </fieldset>`}if(i===`object`&&t.properties){let i=n??{};return u`<fieldset class="fieldset" data-param=${e}><legend>${a}</legend>${o}
        <div class="form-stack">${Object.entries(t.properties).map(([e,n])=>i[e]===void 0&&!(t.required??[]).includes(e)?u`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...r,e],pt(n))}>${h(`plus`)} ${e.replace(/_/g,` `)}</button>`:this.field(e,n,i[e],[...r,e]))}</div></fieldset>`}return this.jsonField(a,n,r)}jsonField(e,t,n){return u`<label class="field field-block"><span>${e} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(t,null,2)} @change=${t=>{try{let e=JSON.parse(t.target.value);if(n.length===2&&n[0]===`tasks`){let t=this.ed.draft.tasks[n[1]];this.upd(n,{...e,rule:t.rule,mode:t.mode})}else this.upd(n,e)}catch{this.ed.errors=[`${e}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(e){return u`
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
      </details>`}};a([C({attribute:!1})],L.prototype,`meta`,void 0),a([C({attribute:!1})],L.prototype,`onSaved`,void 0),a([l()],L.prototype,`config`,void 0),L=a([t(`qse-cond-monitor`)],L);function pt(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,pt(t)])):``}var R=class extends o{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=m(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await A.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let t=x(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let e=this.sessions;return f(u`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!e?u`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?b(2):e.length?u`<div class="form-stack" data-watch>${e.map(e=>this.row(e))}</div>`:u`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(e){let t=e.gates.some(e=>e.noted);return u`<div class="row-card" data-watch-session=${e.session}>
      <div class="row-card-head">
        <strong class="mono small">${e.session}</strong>
        <span class="toolbar" style="margin:0">
          ${e.watch?u`<span class="badge ${e.watch.mode===`act`?`badge-on`:`badge-shadow`}">${e.watch.task} ${e.watch.mode}</span>`:d}
          ${e.client_gone?u`<span class="badge badge-warn" data-client-gone>client gone</span>`:e.running?u`<span class="badge badge-on">running</span>`:d}
          ${e.watch?.notify?u`<span class="badge badge-warn" data-notify>push on</span>`:d}
          ${t?u`<span class="badge badge-warn">gate noted</span>`:d}</span>
      </div>
      <p class="small" style="margin:4px 0">${De(e)}</p>
      <p class="muted small num" style="margin:0">${e.watch?`${e.watch.profile||`—`} · ${e.watch.client||`—`} · `:``}${e.requests} requests · last ${g(new Date(e.last_seen*1e3).toISOString())}${e.note_at?` · last note ${g(new Date(e.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};a([l()],R.prototype,`sessions`,void 0),a([l()],R.prototype,`monitor`,void 0),a([l()],R.prototype,`error`,void 0),R=a([t(`qse-cond-watch`)],R);var z=class extends o{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=m(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await A.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let t=x(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let e=this.events,t=[...new Set((e??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!t.includes(this.task)&&t.push(this.task),u`${f(u`Monitor journal ${this.paused?u`<span class="badge">paused</span>`:u`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!e?u`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?b(4):e.length?u`<div class="feed" data-journal>${e.map(e=>Le(e,e=>this.open=e))}</div>`:u`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:u`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${t.map(e=>u`<option value=${e} ?selected=${e===this.task}>${e}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${h(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?u`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:d}`}};a([C({attribute:!1})],z.prototype,`session`,void 0),a([l()],z.prototype,`events`,void 0),a([l()],z.prototype,`monitor`,void 0),a([l()],z.prototype,`error`,void 0),a([l()],z.prototype,`task`,void 0),a([l()],z.prototype,`kind`,void 0),a([l()],z.prototype,`paused`,void 0),a([l()],z.prototype,`open`,void 0),z=a([t(`qse-cond-journal`)],z);var mt=[`off`,`low`,`medium`,`high`,`xhigh`],ht={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},gt=new Set([`S1`,`S4`]),_t=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],B=class extends o{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleContract=null,this.roleTest={},this.ed=null,this.roles=new w(this,`roles`),this.residency=new w(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),A.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),A.doc(`packages`).then(e=>this.packages=Ae(e.doc)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new w(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{let e=await A.roles();this.roleHealth=e.health,this.roleContract=e.contract??null}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let t=await A.roleTest(e);this.roleTest={...this.roleTest,[e]:t.ok?`ok in ${s(t.ms)}: "${t.answer?.label}" at ${t.answer?.confidence.toFixed(2)}`:`failed: ${t.error}`}}catch(t){this.roleTest={...this.roleTest,[e]:x(t).message}}this.loadHealth()}render(){return u`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>u`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>p({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?u`<span class="badge badge-lock">${h(`warn`)} locked</span>`:u`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${g(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):u`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(T(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=T(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=T(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderUi(e,t){let n=e.ui??{states:`off`},r=n.kinds??{},i=t||n.states!==`on`,a=we(n),o=Ue(n),s=(e,t,r,a,s)=>u`<label class="field"><span>${t}</span><input type="number" inputmode="numeric" min=${r} max=${a} step=${s} ?disabled=${i||o.mode!==`on`} data-ui-progress=${e} .value=${String(o[e])}
        @input=${t=>{let i=Math.round(Number(t.target.value));Number.isFinite(i)&&i>=r&&i<=a&&this.upd([`ui`],{...n,progress:{...n.progress??{},mode:o.mode,[e]:i}})}} /></label>`;return u`<details class="sec" data-sec="ui">
      <summary>What the user sees <span class="sec-sub">TandemLLM's own steps as lines the client shows</span> <span class="badge ${n.states===`on`?`badge-on`:``}" data-ui-badge>${n.states}</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0"><span class="muted small">state lines</span>${j([`off`,`on`],n.states,e=>this.upd([`ui`],{...n,states:e}),{disabled:t,small:!0,label:`state lines`})}</div>
        <p class="muted small">Each step goes out at the moment it happens as one line, <code>› deciding: code_edit, high effort</code>. The server takes them out of the reasoning and the content a client sends back, so Qwen never sees them; they are not tokens and not in the usage. Only in mode on.</p>
        <div class="toolbar" style="margin:0" data-ui-channel><span class="muted small">where</span>${j([`reasoning`,`content`,`inline`],a,e=>this.upd([`ui`],{...n,states_channel:e}),{disabled:i,small:!0,label:`where the lines go`})}</div>
        <p class="muted small">${a===`inline`?u`Each line opens a thinking block of its own and is its title, in time order: opencode shows <code>Thought: › thinking 2 min, 4,096 tokens</code> between Qwen's thinking blocks, and the answer after them. Needs <code>reasoning_content</code>; with <code>&lt;think&gt;</code> in the content (Hermes) the lines go as in the content.`:a===`content`?u`In the content, outside the thinking: before the reasoning, between its parts, after it. opencode shows them in its text part; with <code>&lt;think&gt;</code> in the content (Hermes) a line said inside the block goes out right after <code>&lt;/think&gt;</code>.`:u`In the reasoning opencode and Hermes show (a <code>reasoning_content</code> delta, or inside <code>&lt;think&gt;</code>). A profile saved before this choice existed keeps them here.`}</p>
        <div class="pkg-picks" data-ui-kinds>${Fe.map(({id:e,example:t})=>{let a=r[e]!==!1;return u`<button class="pick ${a?`is-on`:``}" data-ui-kind=${e} title=${t} aria-pressed=${a?`true`:`false`} ?disabled=${i} @click=${()=>this.upd([`ui`],{...n,kinds:{...r,[e]:!a}})}>${e}</button>`})}</div>
        <div class="toolbar" style="margin:0" data-ui-progress-mode><span class="muted small">progress</span>${j([`off`,`on`],o.mode,e=>this.upd([`ui`],{...n,progress:{...n.progress??{},mode:e}}),{disabled:i,small:!0,label:`progress lines`})}</div>
        <p class="muted small">The numbers of a wait, templated: <code>waiting: opencode's title request runs first, position 1</code> while the request waits, <code>reading 35,189 tokens at ~750 tok/s, about 47 s left</code> during a long prefill, <code>thinking 4 min, 9,800 tokens</code> during a long thinking. Streams only; a request that waits opens its stream at its first waiting line.</p>
        <div class="form-grid">
          ${s(`prefill_every_s`,`a prefill line every (s)`,2,600,1)}
          ${s(`thinking_every_s`,`a thinking line every (s)`,10,3600,10)}
          ${s(`thinking_every_tokens`,`or every (tokens)`,256,262144,256)}
        </div>
        <div class="toolbar" style="margin:0" data-ui-narrator><span class="muted small">narrator</span>${j([`off`,`on`,`thoughts`,`live`],n.narrator??`off`,e=>this.upd([`ui`],{...n,narrator:e}),{disabled:i,small:!0,label:`narrator`})}</div>
        <p class="muted small">A small model inside the engine (the <code>narrator</code> role, Qwen3.5) writes in plain words. <b>live</b>: a running line of what Qwen is doing right now, every ~25 s of thinking (“thinking 3 min: how the enemies patrol …”), what a thought decided, what a long write is for and how far it is, a plan in the answer; the other lines in plain words and the user’s language; the facts (times, files, commands) come from the engine, never from the model. <b>thoughts</b>: after each stretch of thinking (and at the long-thinking marks) one line with what Qwen thought about and decided, e.g. “Thought 1m 43s: planned the level layout …”, read from the thought alone; the other lines stay templates. <b>on</b>: it writes every line instead of the template. Qwen waits while it writes, at most 3 % of its time (the rest at the turn’s end); a line it cannot write in time is the template. It loads on its first line; the Roles section below holds its budget and mode.</p>
      </div>
    </details>`}renderToolTurn(e,t){let n=e.thinking.tool_turn_effort,r=Xe(this.router,null).level,i=Xe(this.router,e);return u`<label class="field ${n?``:`is-inherited`}" data-tool-turn-effort><span>effort of a tool-result turn</span>
      <select aria-label="effort of a tool-result turn" ?disabled=${t} @change=${e=>{let t=e.target.value;if(!this.ed?.draft)return;let n={...this.ed.draft.thinking,tool_turn_effort:t};t||delete n.tool_turn_effort,this.ed.set(T(this.ed.draft,[`thinking`],n))}}>
        <option value="" ?selected=${!n}>Routing's: ${r===`inherit`?`the class's`:r}</option>
        ${Re.map(e=>u`<option value=${e} ?selected=${n===e}>${e===`inherit`?`the class's (inherit)`:e}</option>`)}
      </select>
      <span class="hint">${me(i.level,this.router,e)}${i.source===`profile`?` (this profile)`:i.source===`router`?` (Routing)`:``}: a turn whose newest message is a tool result thinks at this level, its switch and cap; the first turn and every new user message keep the class's</span></label>`}renderBudgets(e,t){return u`<div data-budgets>
      <div class="muted small">Thinking budget per level <span class="muted">(reasoning tokens; empty keeps Routing's level, 0 is no cap below the backstop)</span></div>
      <div class="form-grid">
        ${$e.map(n=>{let r=e.thinking.budgets?.[n],i=ue(n,this.router,null).cap,a=ue(n,this.router,e);return u`<label class="field ${r==null?`is-inherited`:``}" data-budget=${n}><span class="mono">${n}</span>
            <input type="number" inputmode="numeric" min="0" max=${qe} step="1024" ?disabled=${t} placeholder=${i==null?``:String(i)} .value=${r==null?``:String(r)}
              @change=${e=>{let t=e.target.value.trim();this.ed?.draft&&this.ed.set(T(this.ed.draft,[`thinking`],xe(this.ed.draft.thinking,n,t===``?null:Number(t))))}} />
            <span class="hint">${Ve(a.cap)}${a.source===`router`?` (Routing)`:a.source===`profile`?` (this profile)`:``}</span></label>`})}
      </div>
    </div>`}renderEditor(e){if(e.loading&&!e.draft)return b(8);if(e.loadError)return u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=!!e.locked,r=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],i=this.router?.classes??[];return u`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${h(`playground`)} Try in Playground</a></div>
      ${n?u`<div class="note" data-lock-reason>${h(`warn`)} ${e.locked}. Every control below is read-only.</div>`:d}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${t.description} ?disabled=${n} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${j([`raw`,`shadow`,`on`],t.mode,e=>this.upd([`mode`],e),{disabled:n,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${r.map(e=>{let r=t.clients.includes(e);return u`<button class="pick ${r?`is-on`:``}" ?disabled=${n} aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`clients`],r?t.clients.filter(t=>t!==e):[...t.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${n} @change=${e=>this.upd([`default_class`],e.target.value)}>${i.map(e=>u`<option value=${e.id} ?selected=${e.id===t.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(ht).map(e=>u`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${j([`off`,`shadow`,`on`],t.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:n,small:!0,label:e})}<span class="small">${ht[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${i.map(e=>{let r=t.routing[e.id]??{},i=Object.keys(r).length>0;return u`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${D(e.id,this.ids,e.label)}
                ${i?u`<button class="btn btn-ghost btn-sm" ?disabled=${n} @click=${()=>{let n={...t.routing};delete n[e.id],this.upd([`routing`],n)}}>use the router's</button>`:u`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${j(mt,r.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...r,effort:t}),{disabled:n,small:!0,label:`effort override of ${e.id}`,cls:r.effort?``:`is-inherited`})}</div>
              <div class="pkg-picks">${this.packages.map(t=>{let i=r.packages??e.policy.packages??[],a=i.includes(t);return u`<button class="pick ${a?`is-on`:``}" ?disabled=${n} @click=${()=>this.upd([`routing`,e.id],{...r,packages:a?i.filter(e=>e!==t):[...i,t]})}>${t}</button>`})}</div>
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
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${j(mt,t.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:n,small:!0,label:`default effort`})}</div>
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
          ${_t.map(([e,r,i])=>u`<label class="switch" data-pack=${e}><input type="checkbox" ?disabled=${n} .checked=${t.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${i} <span class="muted small">(${r}, ${t.toolgate[e]?t.tasks[r]?.mode??`off`:`off`})</span></label>`)}
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
      ${this.renderUi(t,n)}
      <details class="sec" data-sec="hints">
        <summary>Speed hints <span class="badge badge-warn">not gated</span></summary>
        <div class="sec-body">
          <p class="warn-ink small">No SPD measurement is in the journal yet (SPD-73); the router's ≤ 1 ulp text rule and the never-worse gate come first.</p>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.stair_width} @change=${e=>this.upd([`hints`,`stair_width`],e.target.checked)} /> StairCut width hint</label>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.prefix_retention} @change=${e=>this.upd([`hints`,`prefix_retention`],e.target.checked)} /> prefix retention hint</label>
        </div>
      </details>
      ${k(e)}
      ${M(e,`profile ${e.name}`)}
    </div>`}renderTask(e,t,n,r,i){let a=this.meta?.tasks[e],o=a?.role&&a.tier===`1`,s=!o||!!this.roleHealth[a.role]?.up;return u`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${a?.family??``}</div></div>
      <div>${a?.label??e}<div class="task-why">${a?.why??``}</div>${o&&!s?u`<div class="warn-ink small">${h(`warn`)} the ${a.role} role is offline: act is refused</div>`:d}</div>
      <div class="task-mode">${j([`off`,`shadow`,`act`],t,t=>this.upd([`tasks`,e,`mode`],t),{disabled:i,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${gt.has(e)&&Object.keys(r).length?u`<p class="muted small" data-in-context=${e}>${Object.entries(r).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:d}
        ${Object.entries(gt.has(e)?{}:r).map(([t,n])=>{let r=a?.thresholds[t];return u`<label class="field ${r&&r.default!==n?`is-changed`:``}"><span>${t.replace(/_/g,` `)} <span class="hint">${r?.unit??``}${r?`, default ${r.default}`:``}</span></span>
            <input type="number" ?disabled=${i} .value=${String(n)} step="any" @input=${n=>{let r=Number(n.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,t],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${i} min="0" max="10" .value=${String(n)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let e=this.roles;if(!e.draft)return e.loadError?u`<p class="err-ink small">${e.loadError}</p>`:d;let t=e.draft,n=(n,r)=>e.set(T(t,n,r)),r=at(e.base?.meta.defaults),i=this.roleContract?.engine_tokenizer_sha256??null,a=this.meta?.roles,o=a?.kinds??je,c=a?.windows??Je,l=a?.fallbacks??[`PASS`,`stale`,`skip`,`next-role`],f=Ze.filter(e=>!a||a.writes.includes(e));return u`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${e.base?.meta.defaults?.length?u`<div class="note" data-role-defaults>${h(`warn`)} Version ${e.base.version} was stored before the role contract and loaded with ${e.base.meta.defaults.length} default${e.base.meta.defaults.length===1?``:`s`} (marked <span class="badge">default</span> below; the journal has the list). Save the roles to make them explicit.</div>`:d}
        <p class="muted small">The contract (SRV-133): when a role may use the GPU, whether a Qwen request cuts it, how long its caller waits before the fallback, what it may write, and the tokenizer it reads, which decides what it may share with Qwen. Engine tokenizer: <code class="mono" data-engine-sha>${i?`${i.slice(0,12)}…`:`not recorded`}</code></p>
        ${Object.entries(t).map(([e,t])=>{let a=this.roleHealth[e],p=new Set((r[e]??[]).map(e=>e.split(`=`)[0])),m=t=>p.has(t)?u` <span class="badge" data-default=${`${e}.${t}`}>default</span>`:d,g=ke(t.kind,t.window),_=ae(t.kind,t.writes),v=ge(t,i);return u`<div class="row-card" data-role-row=${e}>
            <div class="row-card-head"><strong>${e}</strong>
              <span class="toolbar" style="margin:0">${a?u`<span class="badge ${a.up?`badge-ok`:t.mode===`off`?``:`badge-warn`}">${a.up?`up ${s(a.ms)}`:t.mode===`off`?`off`:`down: ${a.error??`?`}`}</span>`:d}
                <button class="btn btn-sm" @click=${()=>this.testRole(e)} data-role-test=${e}>Test</button></span></div>
            ${this.roleTest[e]?u`<p class="small" role="status">${this.roleTest[e]}</p>`:d}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${t.endpoint} @input=${t=>n([e,`endpoint`],t.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${t.model} @input=${t=>n([e,`model`],t.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(t.timeout_ms)} @input=${t=>n([e,`timeout_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>memory claim (GiB, its peak)${m(`memory_claim_gib`)}</span><input type="number" step="0.1" min="0" data-role-field="memory_claim_gib" .value=${String(t.memory_claim_gib)} @input=${t=>n([e,`memory_claim_gib`],Number(t.target.value))} /></label>
            </div>
            <div class="form-grid" data-role-contract=${e}>
              <label class="field"><span>kind${m(`kind`)}</span><select data-role-field="kind" @change=${t=>n([e,`kind`],t.target.value)}>${o.map(e=>u`<option value=${e} ?selected=${e===t.kind}>${e}</option>`)}</select></label>
              <label class="field"><span>window${m(`window`)}</span><select data-role-field="window" @change=${t=>n([e,`window`],t.target.value)}>${c.map(e=>u`<option value=${e} ?selected=${e===t.window} ?disabled=${e!==t.window&&!!ke(t.kind,e)} title=${Ye[e]}>${e}</option>`)}</select></label>
              <label class="field"><span>latency budget (ms)${m(`latency_budget_ms`)}</span><input type="number" min="1" max="600000" data-role-field="latency_budget_ms" .value=${String(t.latency_budget_ms)} @input=${t=>n([e,`latency_budget_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>fallback (off, slow, unsure or refused)${m(`fallback`)}</span><select data-role-field="fallback" @change=${t=>n([e,`fallback`],t.target.value)}>${l.map(e=>u`<option value=${e} ?selected=${e===t.fallback}>${e}</option>`)}</select></label>
              <label class="field"><span>writes${m(`writes`)}</span><select data-role-field="writes" @change=${t=>n([e,`writes`],t.target.value)}>${f.map(e=>u`<option value=${e} ?selected=${e===t.writes} ?disabled=${e!==t.writes&&!!ae(t.kind,e)}>${e.replace(/_/g,` `)}</option>`)}</select></label>
              <label class="field field-wide"><span>tokenizer sha256 (the engine records its own; none: text only)${m(`tokenizer_sha256`)}</span><input class="mono" data-role-field="tokenizer_sha256" spellcheck="false" .value=${t.tokenizer_sha256} @input=${t=>n([e,`tokenizer_sha256`],t.target.value.trim().toLowerCase()||`none`)} /></label>
            </div>
            <div class="toolbar" style="margin:0">
              <label class="switch"><input type="checkbox" data-role-field="preemptible" .checked=${t.preemptible} @change=${t=>n([e,`preemptible`],t.target.checked)} /> preemptible: a Qwen request cuts it, it resumes later${m(`preemptible`)}</label>
            </div>
            <div><span class="muted small">sharing with Qwen${m(`sharing`)}</span>
              <div class="pkg-picks" data-role-sharing=${e}>${v.map(r=>{let i=t.sharing.includes(r.form);return u`<button class="pick ${i?`is-on`:``}" aria-pressed=${i?`true`:`false`} ?disabled=${!i&&!r.allowed} title=${r.why??``} data-share=${r.form} @click=${()=>n([e,`sharing`],i?t.sharing.filter(e=>e!==r.form):[...t.sharing,r.form])}>${r.label}</button>`})}</div>
              ${v.filter(e=>!e.allowed).map(e=>u`<p class="muted small" data-share-why=${e.form}>${e.label}: ${e.why}</p>`)}
            </div>
            ${g?u`<p class="warn-ink small" data-role-why="window">${h(`warn`)} ${g}</p>`:d}
            ${_?u`<p class="warn-ink small" data-role-why="writes">${h(`warn`)} ${_}</p>`:d}
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${j([`off`,`shadow`,`on`],t.mode,t=>n([e,`mode`],t),{small:!0,label:`${e} mode`})}<span class="muted small mono">${t.contract}</span></div>
          </div>`})}
        ${k(e)}
        ${M(e,`the roles`)}
      </div>
    </details>`}renderResidency(){let e=this.residency;if(!e.draft)return e.loadError?u`<p class="err-ink small">${e.loadError}</p>`:d;let t=e.draft,n=(n,r)=>e.set(T(t,n,r)),r=(e,r=`0.1`,i=!1)=>u`<input type="number" step=${r} min="0" .value=${String(t[e[0]])} @input=${t=>{let r=Number(t.target.value);n(e,i?Math.round(r):r)}} data-res=${e[0]} />`,i=[`measured`,`peak`,`declared`];return u`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${r([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${r([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${r([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${t.order.join(`, `)} @change=${e=>n([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(t.claims).map(([e,r])=>u`<div class="row-card" data-claim=${e}>
            <div class="row-card-head"><strong>${e}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[e]:r,...i}=t.claims;n([`claims`],i)}} aria-label=${`remove the claim of ${e}`}>${h(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(r.gib)} @input=${t=>n([`claims`,e,`gib`],Number(t.target.value))} data-claim-gib=${e} /></label>
              <label class="field"><span>basis</span><select @change=${t=>n([`claims`,e,`basis`],t.target.value)}>${i.map(e=>u`<option value=${e} ?selected=${e===r.basis}>${e}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${r.note} @input=${t=>n([`claims`,e,`note`],t.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in t.claims} @click=${()=>{n([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${h(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${k(e)}
        ${M(e,`the residency limits`)}
      </div>
    </details>`}};a([C({attribute:!1})],B.prototype,`ids`,void 0),a([C({attribute:!1})],B.prototype,`meta`,void 0),a([C({attribute:!1})],B.prototype,`tiles`,void 0),a([C({attribute:!1})],B.prototype,`selected`,void 0),a([C({attribute:!1})],B.prototype,`onSaved`,void 0),a([l()],B.prototype,`router`,void 0),a([l()],B.prototype,`packages`,void 0),a([l()],B.prototype,`roleHealth`,void 0),a([l()],B.prototype,`roleContract`,void 0),a([l()],B.prototype,`roleTest`,void 0),a([l()],B.prototype,`newClaim`,void 0),B=a([t(`qse-cond-profiles`)],B);var V=[`package`,`task_skill`,`test_skill`,`card`,`library`],H={package:`package`,task_skill:`task skill`,test_skill:`test skill`,card:`card`,library:`library skill`},vt=3.5,yt={library:350,library_card:200},bt={q:``,category:`all`,cls:``,enabled:`all`,over:!1};function U(e){return Math.floor(e.length/vt)}function xt(e,t){return t?t.category:e.library?`library`:e.kind===`skill`?`task_skill`:`package`}function St(e,t,n,r){let i=r?.limits??yt,a=n===`library`?i.library:e===(r?.library.card??`tandem-library`)?i.library_card:null;return a!=null&&a<=t.max_tokens?{limit:a,source:`house`}:{limit:t.max_tokens,source:`max_tokens`}}function Ct(e,t){return(e?.classes??[]).filter(e=>(e.policy.packages??[]).includes(t)).map(e=>e.id)}function wt(e,t,n,r){return Object.entries(e?.packages??{}).map(([e,i])=>{let a=t?.packages[e]??null;return{name:e,pkg:i,entry:a,category:xt(i,a),classes:Ct(r,e),usage:n?.[e]??null,score:null}}).sort((e,t)=>V.indexOf(e.category)-V.indexOf(t.category)||e.name.localeCompare(t.name))}function Tt(e){return e.toLowerCase().split(/\s+/).filter(Boolean)}function Et(e,t){let n=`${e.name} ${e.pkg.title} ${e.pkg.description} ${e.pkg.source}`.toLowerCase();return Tt(t).every(e=>n.includes(e))}function Dt(e,t){return e.entry?e.entry.over:e.pkg.source===`inline`&&U(W(e.pkg.body,!!e.pkg.library)[0])>St(e.name,e.pkg,e.category,t).limit}function Ot(e,t,n,r=null){let i=new Map((n??[]).map(e=>[e.name,e.score])),a=t.q.trim(),o=e.filter(e=>t.category===`all`||e.category===t.category).filter(e=>!t.cls||e.classes.includes(t.cls)||e.category===`library`&&(e.entry?.fits??[]).includes(t.cls)).filter(e=>t.enabled===`all`||t.enabled===`on`===e.pkg.enabled).filter(e=>!t.over||Dt(e,r)).filter(e=>!a||Et(e,a)||e.category===`library`&&i.has(e.name)).map(e=>({...e,score:a&&e.category===`library`?i.get(e.name)??null:null}));return a?o.sort((e,t)=>V.indexOf(e.category)-V.indexOf(t.category)||(e.score!=null||t.score!=null)&&(t.score??-1)-(e.score??-1)||e.name.localeCompare(t.name)):o}function kt(e,t){let n=e=>!!t?.packages[e]?.general;return[...e.filter(n),...e.filter(e=>!n(e))]}function At(e,t,n,r){let i=e.classes.findIndex(e=>e.id===t);if(i<0)return{router:e,error:`no class ${t}`};if(r?.packages[n]?.library)return{router:e,error:`${n} is a library skill: the router searches the library, a class does not list it`};let a=e.classes[i].policy.packages??[];if(a.includes(n))return{router:e,error:null};if(a.length>=12)return{router:e,error:`${t} already lists 12 packages, the most a class may`};let o=e.classes.map((e,t)=>t===i?{...e,policy:{...e.policy,packages:kt([...a,n],r)}}:e);return{router:{...e,classes:o},error:null}}function jt(e,t,n){return{...e,classes:e.classes.map(e=>e.id===t?{...e,policy:{...e.policy,packages:(e.policy.packages??[]).filter(e=>e!==n)}}:e)}}var Mt=/^## Reference\b/m;function W(e,t){let n=t?Mt.exec(e):null;return n?[e.slice(0,n.index).trim(),e.slice(n.index).trim()]:[e.trim(),``]}function Nt(e){let t=1;for(;e.packages[`package-${t}`];)t++;return`package-${t}`}function Pt(e){return{title:`Package ${e}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0}}function Ft(e,t,n){return{...e,source:`inline`,body:n?`${t}\n\n${n}`:t}}function It(e,t){return t?e.filter(e=>e.profile===t):e}function Lt(e){return e===`on`?`the router puts the skill it picks into the prompt from the next request, and the library card on the coding classes with tools`:e===`shadow`?`the router searches and picks, and the journal shows the pick; nothing goes into the prompt`:`the router does not search the library`}var G=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.catalog=null,this.usage=null,this.onChanged=()=>void 0,this.profile=``,this.busy=null,this.msg=null,this.error=null}async setMode(e,t){this.busy=e,[this.msg,this.error]=[null,null];try{let n=await A.doc(`profiles`,e),r={...n.doc,library:t},i=await A.check(`profiles`,e,r);if(i.errors.length){this.error=`${e}: ${i.errors.join(`; `)}`;return}if(!await O(`Library ${t} for ${e}?`,[`With ${t}, ${Lt(t)}.`,u`The server checked the change (live v${n.version}):${Ie(i.changes)}`,`Saving makes a new version of the profile with the note “library to ${t}”.`],`Save the new version`))return;let a=await N(()=>A.put(`profiles`,e,r,`library to ${t}`,n.version));this.msg=`${e}: library ${t}, version ${a.version}`,this.onChanged()}catch(e){this.error=x(e).message}finally{this.busy=null}}render(){let t=this.catalog;if(!t||!this.usage)return b(5);let n=this.usage.library,r=Object.values(t.packages).filter(e=>e.category===`library`).length,i=e=>n.rows.filter(t=>t.profile===e&&t.chosen.length).length,a=Object.keys(t.library.profiles).sort();return u`
      <section class="panel" data-lib-profiles>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Retrieval per profile <span class="muted">· ${r} library skills</span></h2>
          <p class="panel-sub">On a new user message, and on a low-quality or stuck outcome, the router searches the library with the message, asks the package question among the best matches and, with <strong>on</strong>, puts the skill that fits at the prompt's tail. <strong>shadow</strong> only journals the pick. Judge the picks below before a profile goes on.</p></div></header>
        ${this.error?u`<p class="err-ink small" role="alert" data-lib-error>${this.error}</p>`:d}
        ${this.msg?u`<p class="status small" role="status" data-lib-saved>${h(`check`)} ${this.msg}</p>`:d}
        <div class="lib-profiles">
          ${a.map(e=>{let n=this.tiles.find(t=>t.name===e),r=t.library.profiles[e];return u`<div class="lib-profile" data-lib-profile=${e}>
              <span class="lib-profile-name"><strong>${e}</strong>${n?.locked?u` <span class="badge badge-lock">locked</span>`:d}<span class="muted small">${n?.clients.join(`, `)||`no client`}${n?` · profile ${n.mode}`:``}</span></span>
              ${j([`off`,`shadow`,`on`],r,t=>void this.setMode(e,t),{small:!0,label:`library of ${e}`,disabled:!!n?.locked||this.busy===e})}
              <span class="muted small num">${i(e)} pick${i(e)===1?``:`s`} in the rows below</span>
            </div>`})}
        </div>
      </section>
      <section class="panel" data-lib-shadow>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">What the harness picked</h2>
          <p class="panel-sub">The router's retrieval in the journal (extra.library), newest first: the candidates the search found, the skill the package question picked, and whether it went in (on) or would have (shadow). A pick opens the skill.</p></div>
          <div class="panel-tools"><label class="field"><span>profile</span><select data-lib-filter @change=${e=>this.profile=e.target.value}>
            <option value="" ?selected=${!this.profile}>all</option>${a.map(e=>u`<option value=${e} ?selected=${this.profile===e}>${e}</option>`)}
          </select></label></div></header>
        <div class="stats" data-lib-tiles>
          ${S({label:`messages searched`,value:e(n.searched.day),unit:`in 24 h`,sub:`${e(n.searched.week)} in 7 days`})}
          ${S({label:`with a pick`,value:e(n.picked.day),unit:`in 24 h`,sub:`${e(n.picked.week)} in 7 days`})}
          ${S({label:`pick rate`,value:n.searched.week?`${Math.round(100*n.picked.week/n.searched.week)}`:`—`,unit:n.searched.week?`%`:``,sub:`7 days`})}
          ${S({label:`most picked`,value:n.top[0]?.name??`—`,sub:n.top.length?n.top.slice(0,3).map(e=>`${e.name} ${e.picks}`).join(` · `):`no pick yet`})}
        </div>
        ${this.rows(It(n.rows,this.profile))}
      </section>`}rows(t){if(!t.length)return v(`No retrieval in the journal`,`A profile with the library in shadow or on journals each search here; with every profile off nothing is searched.`);let n=e=>p({ptab:null,pkg:e,pview:null});return u`<div class="lib-rows" data-lib-rows>${t.map(t=>u`<div class="lib-row" data-lib-row=${t.id}>
        <div class="lib-row-head">
          <span class="num small">${g(new Date(t.ts_ms).toISOString())}</span>
          <span class="badge ${t.mode===`on`?`badge-on`:`badge-shadow`}">${t.mode??`—`}</span>
          ${D(t.class_id,this.ids)}
          <span class="muted small">${t.profile??`—`} · ${t.trigger??`—`}${t.query_chars==null?``:` · ${e(t.query_chars)} characters searched`}${t.cached?` · asked before`:``}</span>
        </div>
        <div class="lib-cands">${t.candidates.length?t.candidates.map(e=>u`<span class="lib-cand ${e.name&&t.chosen.includes(e.name)?`is-chosen`:``}">${e.name}<span class="num muted">${e.score==null?``:e.score.toFixed(2)}</span></span>`):u`<span class="muted small">no candidate shares a word with the message</span>`}</div>
        <div class="lib-pick small">${t.chosen.length?u`${t.mode===`on`&&t.applied?.length?`went in:`:`would have gone in:`} ${t.chosen.map(e=>u`<button class="btn btn-sm lib-open" data-lib-open=${e} @click=${()=>n(e)}>${e}</button>`)} <span class="muted">(${t.how??`—`})</span>`:u`<span class="muted">no pick (${(t.how??`none`).replace(/_/g,` `)})</span>`}</div>
      </div>`)}</div>`}};a([C({attribute:!1})],G.prototype,`ids`,void 0),a([C({attribute:!1})],G.prototype,`tiles`,void 0),a([C({attribute:!1})],G.prototype,`catalog`,void 0),a([C({attribute:!1})],G.prototype,`usage`,void 0),a([C({attribute:!1})],G.prototype,`onChanged`,void 0),a([l()],G.prototype,`profile`,void 0),a([l()],G.prototype,`busy`,void 0),a([l()],G.prototype,`msg`,void 0),a([l()],G.prototype,`error`,void 0),G=a([t(`qse-cond-library`)],G);var Rt=[{id:`text`,label:`Text`},{id:`edit`,label:`Edit`},{id:`classes`,label:`Classes`},{id:`usage`,label:`Usage`}],K=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.selected=null,this.tab=null,this.pane=null,this.catalog=null,this.catError=null,this.usage=null,this.filters=bt,this.hits=null,this.libError=null,this.item=null,this.itemError=null,this.recent=null,this.raw=!1,this.editView=`write`,this.counted=null,this.review=null,this.attachError=null,this.found=null,this.scanError=null,this.pk=new w(this,`packages`),this.rt=new w(this,`router`),this.seq=0,this.loadedFor=null,this.countTimer=null,this.countFor=null}connectedCallback(){super.connectedCallback(),this.pk.onSaved=()=>void this.refresh(),this.rt.onSaved=()=>void this.refresh(),this.pk.load(),this.rt.load(),this.refresh()}disconnectedCallback(){this.countTimer&&clearTimeout(this.countTimer),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.item=null,this.itemError=null,this.recent=null,this.attachError=null,this.selected&&this.loadItem(this.selected))}updated(e){let t=this.selected?this.pk.draft?.packages[this.selected]:null;if(!t||t.source!==`inline`)return;let n=W(t.body,!!t.library)[0];this.counted?.text!==n&&this.countFor!==n&&(this.countFor=n,this.countTimer&&clearTimeout(this.countTimer),this.countTimer=setTimeout(()=>{A.packagesTokens([n]).then(e=>this.counted={text:n,n:e.tokens[0],how:e.tokenizer}).catch(()=>void 0)},400))}async refresh(){try{let[e,t]=await Promise.all([A.packagesCatalog(),A.packagesUsage(null,1)]);[this.catalog,this.usage,this.catError]=[e,t,null]}catch(e){this.catError=x(e).message}this.selected&&await this.loadItem(this.selected)}async loadItem(e){try{let[t,n]=await Promise.all([this.pk.base?.doc.packages[e]||!this.pk.base?A.packageItem(e):Promise.resolve(null),A.packagesUsage(e,20)]);if(e!==this.selected)return;[this.item,this.recent,this.itemError]=[t,n.recent,null]}catch(t){e===this.selected&&(this.itemError=x(t).message)}}async search(e){this.filters={...this.filters,q:e};let t=++this.seq;if(!e.trim()||!Object.values(this.pk.draft?.packages??{}).some(e=>e.library)){[this.hits,this.libError]=[null,null];return}try{let n=await A.librarySearch(e.trim(),20);t===this.seq&&([this.hits,this.libError]=[n,null])}catch(e){t===this.seq&&([this.hits,this.libError]=[[],`the library's search: ${x(e).message}`])}}open(e,t=null){p({pkg:e,pview:t})}upd(e,t){this.pk.draft&&this.pk.set(T(this.pk.draft,e,t))}toggleAttach(e,t,n){let r=this.rt.draft;if(!r)return;if(!n){this.attachError=null,this.rt.set(jt(r,e,t));return}let i=At(r,e,t,this.pk.draft);this.attachError=i.error,i.error||this.rt.set(i.router)}addInline(){let e=this.pk.draft,t=Nt(e);this.pk.set(T(e,[`packages`,t],Pt(Number(t.slice(8))))),this.open(t,`edit`)}copyInline(e){let t=this.pk.draft,n=this.item;n&&n.name===e&&n.found&&(this.pk.set(T(t,[`packages`,e],Ft(t.packages[e],n.body,n.reference))),this.open(e,`edit`))}removePackage(e){let t=this.pk.draft,n={...t.packages};delete n[e],this.pk.set({...t,packages:n}),this.open(null)}async scan(){this.scanError=null;try{this.found=(await A.scan()).skills}catch(e){this.scanError=x(e).message}}register(e){let t={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.pk.set(T(this.pk.draft,[`packages`,e.name],t)),this.open(e.name)}async startReview(e,t){await e.check(),this.review=t}async confirmSave(e){await e.save()&&(this.review=null)}bar(e,t,n){let r=e.base?.meta,i=!!e.draft&&!!e.base&&JSON.stringify(e.draft)!==JSON.stringify(e.base.doc);return u`
      ${e.showVersions?nt(e,n):d}
      ${this.review===t?u`<section class="panel pk-review" data-review=${t}>
            <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e.errors.length?u`${h(`warn`)} The server refuses ${n}`:u`${h(`check`)} The server accepts ${n}`}</h2>
              <p class="panel-sub">POST /v1/conductor/${t}/check against live v${r?.version??0}: ${e.changes.length} change${e.changes.length===1?``:`s`}. Saving makes a new version with your note; the next request uses it.</p></div></header>
            ${k(e)}
            ${Ie(e.changes)}
            <div class="toolbar pk-review-tools">
              <button class="btn btn-primary" data-review-save=${t} ?disabled=${e.errors.length>0||!e.note.trim()||e.saving||!e.changes.length} @click=${()=>this.confirmSave(e)}>${e.saving?`Saving…`:`Save as a new version`}</button>
              <button class="btn btn-ghost" @click=${()=>this.review=null}>Back to editing</button>
              ${e.note.trim()?d:u`<span class="muted small">Write the change note first.</span>`}
            </div>
          </section>`:d}
      <div class="savebar ${i||this.review===t?``:`is-idle`}" data-savebar=${t}>
        <span class="status num" title=${r?.note??``}>${n} · live v${r?.version??0}${i?u` · <strong>${e.changes.length||`…`} change${e.changes.length===1?``:`s`}</strong>`:d}</span>
        <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>{e.note=t.target.value,this.requestUpdate()}} aria-label=${`change note for ${n}`} maxlength="200" data-note=${t} />
        <button class="btn btn-primary" ?disabled=${!i||e.saving} @click=${()=>this.startReview(e,t)} data-check-save=${t}>Check and save…</button>
        <button class="btn btn-ghost" ?disabled=${!i} @click=${()=>{e.reset(),this.review=null}}>Reset</button>
        <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():this.requestUpdate()}} data-versions-btn=${t}>Versions</button>
        ${e.saved?u`<span class="status" role="status" data-saved=${t}>${h(`check`)} ${e.saved}</span>`:d}
        ${e.saveError?u`<span class="err-ink small" role="alert">${e.saveError}</span>`:d}
      </div>`}render(){let e=u`<nav class="mem-tabs pk-tabs-top" aria-label="packages tabs">
      ${[{id:null,label:`Packages and skills`},{id:`library`,label:`Library retrieval`}].map(e=>u`<a class="cond-tab ${this.tab===`library`==(e.id===`library`)?`is-on`:``}" href="#" data-pk-top=${e.id??`all`} aria-current=${this.tab===`library`==(e.id===`library`)?`page`:d} @click=${t=>{t.preventDefault(),p({ptab:e.id,pkg:null,pview:null})}}>${e.label}</a>`)}
    </nav>`;if(this.tab===`library`)return u`${e}<qse-cond-library .ids=${this.ids} .tiles=${this.tiles} .catalog=${this.catalog} .usage=${this.usage} .onChanged=${()=>this.refresh()}></qse-cond-library>`;let t=this.pk;if(t.loading&&!t.draft)return u`${e}${b(6)}`;if(t.loadError)return u`${e}<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${t.loadError}</div></div>`;let n=t.draft,r=wt(n,this.catalog,this.usage?.packages??null,this.rt.draft),i=Ot(r,this.filters,this.hits,this.catalog),a=this.selected&&n.packages[this.selected]?this.selected:null;return u`
      ${e}
      ${this.catError?u`<div class="guard-line" role="alert">${h(`warn`)} ${this.catError}</div>`:d}
      <div class="pk-layout ${a?`has-detail`:``}">
        ${this.renderList(i,r)}
        ${a?this.renderDetail(a,n.packages[a],r.find(e=>e.name===a)):this.selected?u`<section class="panel" data-pk-gone><p class="muted">No package ${this.selected} in this version.</p><button class="btn btn-sm" @click=${()=>this.open(null)}>Back to the list</button></section>`:d}
      </div>
      ${this.renderFound(n)}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${n.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${this.rt.draft&&(this.rt.showVersions||this.rt.saved||this.review===`router`||JSON.stringify(this.rt.draft)!==JSON.stringify(this.rt.base?.doc))?this.bar(this.rt,`router`,`the router`):d}
      ${this.bar(this.pk,`packages`,`the packages`)}
    `}renderList(e,t){let n=this.filters,r=t.filter(e=>Dt(e,this.catalog)).length,i=e=>this.filters={...this.filters,...e};return u`<section class="panel pk-list" data-pk-list>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills <span class="muted" data-pk-count-all>· ${e.length} of ${t.length}</span></h2>
        <p class="panel-sub">What the router may put into a prompt. Tokens are the text that goes in, ${this.catalog?.tokenizer===`estimate`?`estimated at 3.5 characters a token (no tokenizer on this server)`:`on the engine's tokenizer`}${r?u`; <span class="warn-ink" data-pk-over-count>${r} over the limit</span>`:d}.</p></div>
        <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-pk-new>${h(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${h(`search`)} Skills on the box</button></div></header>
      <div class="pk-filters">
        <label class="field field-search pk-search">${h(`search`)}<input type="search" data-pk-search placeholder="search packages, or the library's text" aria-label="search packages and skills" .value=${n.q} @input=${e=>void this.search(e.target.value)} /></label>
        <label class="field"><span>kind</span><select data-pk-filter="kind" @change=${e=>i({category:e.target.value})}>
          <option value="all" ?selected=${n.category===`all`}>all</option>
          ${V.map(e=>u`<option value=${e} ?selected=${n.category===e}>${H[e]}</option>`)}
        </select></label>
        <label class="field"><span>class</span><select data-pk-filter="class" @change=${e=>i({cls:e.target.value})}>
          <option value="" ?selected=${!n.cls}>any</option>
          ${this.ids.map(e=>u`<option value=${e} ?selected=${n.cls===e}>${e}</option>`)}
        </select></label>
        <label class="field"><span>enabled</span><select data-pk-filter="enabled" @change=${e=>i({enabled:e.target.value})}>
          ${[`all`,`on`,`off`].map(e=>u`<option value=${e} ?selected=${n.enabled===e}>${e}</option>`)}
        </select></label>
        <label class="switch"><input type="checkbox" data-pk-filter="over" .checked=${n.over} @change=${e=>i({over:e.target.checked})} /> over the limit</label>
      </div>
      ${this.libError?u`<p class="err-ink small">${this.libError}</p>`:d}
      <div class="pk-head" aria-hidden="true"><span>package</span><span>tokens / limit</span><span>used 24 h · 7 d</span></div>
      <div class="pk-rows" role="list">${e.map(e=>this.row(e))}</div>
      ${e.length?d:u`<p class="muted small" data-pk-empty>${n.q.trim()?`Nothing matches “${n.q.trim()}”${this.hits?`, and the library shares no word with it`:``}.`:`No package matches these filters.`}</p>`}
    </section>`}row(t){let n=St(t.name,t.pkg,t.category,this.catalog),r=t.entry?t.entry.tokens:U(W(t.pkg.body,!!t.pkg.library)[0]),i=Dt(t,this.catalog),a=t.usage,o=this.selected===t.name;return u`<button class="pk-row ${o?`is-open`:``} ${t.pkg.enabled?``:`is-off`}" role="listitem" data-pk-row=${t.name} data-category=${t.category} aria-current=${o?`true`:d} @click=${()=>this.open(t.name)}>
      <span class="pk-main">
        <span class="pk-name"><strong>${t.name}</strong>
          <span class="badge pk-kind pk-kind-${t.category}">${H[t.category]}</span>
          ${t.pkg.general?u`<span class="badge badge-on" title="goes in whole and first on every turn of a class that lists it">general</span>`:d}
          ${t.pkg.enabled?d:u`<span class="badge">off</span>`}
          ${t.entry?d:u`<span class="badge badge-shadow">not saved</span>`}
          ${t.entry&&!t.entry.available&&t.entry.error?u`<span class="badge badge-warn" title=${t.entry.path??``}>${t.entry.error.replace(/_/g,` `)}</span>`:d}
          ${t.score==null?d:u`<span class="pk-score num" data-score title="the library's BM25 score">${t.score.toFixed(2)}</span>`}
        </span>
        <span class="pk-desc muted small">${t.pkg.description}</span>
        <span class="pk-meta"><span class="mono small muted">${t.pkg.source}</span>${t.classes.map(e=>D(e,this.ids))}${t.category===`library`&&t.entry?.fits?.length?u`<span class="muted small">fits</span>${t.entry.fits.map(e=>D(e,this.ids))}`:d}</span>
      </span>
      <span class="pk-tokens num ${i?`is-over`:``}" data-pk-tokens title=${t.entry?`${t.entry.tokens} tokens go in; the whole text is ${t.entry.body_tokens}`:`estimated: not saved yet`}>
        <span>${t.entry?``:`≈`}${e(r)} <span class="muted">/ ${e(n.limit)}</span></span>
        <span class="pk-bar"><span style="width:${Math.min(100,100*r/Math.max(1,n.limit))}%"></span></span>
        ${i?u`<span class="badge badge-warn">over</span>`:d}
      </span>
      <span class="pk-usage num small" data-pk-usage>
        ${a&&(a.injected.week||a.shadow.week||a.pick.week)?u`${a.injected.week?u`<span>${a.injected.day} · ${a.injected.week} in</span>`:d}${a.shadow.week?u`<span class="muted">${a.shadow.day} · ${a.shadow.week} shadow</span>`:d}${a.pick.week?u`<span>${a.pick.day} · ${a.pick.week} picks</span>`:d}`:u`<span class="muted">—</span>`}
      </span>
    </button>`}renderFound(t){return this.scanError?u`<p class="err-ink small">${this.scanError}</p>`:this.found?u`<section class="panel" style="margin-top:16px" data-found>
      <header class="panel-head"><h2 class="panel-title">Skills in ${t.skill_dirs.join(`, `)}</h2><div class="panel-tools"><button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.found=null}>${h(`x`)}</button></div></header>
      ${this.found.length?u`<div class="table-scroll"><table class="table"><tbody>${this.found.map(n=>u`<tr><td><strong>${n.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${n.description}</div></td><td class="num">${e(n.chars)} chars</td>
              <td>${n.registered||t.packages[n.name]?u`<span class="badge badge-ok">registered</span>`:u`<button class="btn btn-sm" @click=${()=>this.register(n)} data-register=${n.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:u`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ spark:~/.config/opencode/skills/</code>).</p>`}
    </section>`:d}renderDetail(t,n,r){let i=r.entry,a=xt(n,i),o=St(t,n,a,this.catalog),s=Rt.some(e=>e.id===this.pane)?this.pane:`text`,c=n.source===`inline`,l=r.classes,f=this.tokensNow(t,n,i),m=f.n>o.limit||!!i?.cut&&!(c&&this.dirtyBody(t));return u`<section class="panel pk-detail" data-pk-detail=${t}>
      <header class="pk-detail-head">
        <div class="pk-title">
          <h2 class="panel-title">${t}</h2>
          <span class="muted">${n.title}</span>
        </div>
        <button class="btn btn-ghost btn-sm pk-close" @click=${()=>this.open(null)} data-pk-close aria-label="Close the package">${h(`x`)}<span class="pk-close-word">All packages</span></button>
      </header>
      <div class="pk-badges">
        <span class="badge pk-kind pk-kind-${a}" data-pk-category>${H[a]}</span>
        <span class="badge">${c?`inline`:n.source.startsWith(`skill:`)?`skill on the box`:`file on the box`}</span>
        ${n.general?u`<span class="badge badge-on">general</span>`:d}
        ${n.enabled?u`<span class="badge badge-ok">enabled</span>`:u`<span class="badge">disabled</span>`}
        ${i?d:u`<span class="badge badge-shadow">not saved</span>`}
        ${i&&!i.available&&i.error?u`<span class="badge badge-warn">${i.error.replace(/_/g,` `)}</span>`:d}
      </div>
      <div class="toolbar pk-switches">
        <label class="switch"><input type="checkbox" data-pk-enabled .checked=${n.enabled} @change=${e=>this.upd([`packages`,t,`enabled`],e.target.checked)} /> enabled</label>
        ${a===`library`?d:u`<label class="switch" title="goes in whole and first on every turn of a class that lists it (K1)"><input type="checkbox" data-pk-general .checked=${!!n.general} @change=${e=>this.upd([`packages`,t,`general`],e.target.checked)} /> general</label>`}
        ${c?d:u`<button class="btn btn-sm" data-pk-copy-inline ?disabled=${!this.item?.found} title=${this.item?.found?`an inline copy of the file overrides it`:`the file is not on the box`} @click=${()=>this.copyInline(t)}>${h(`copy`)} Copy to inline</button>`}
        <button class="btn btn-sm btn-ghost" data-pk-remove ?disabled=${l.length>0} title=${l.length?`${l.join(`, `)} list it; detach it first`:`remove it from the packages document`} @click=${()=>this.removePackage(t)}>${h(`trash`)} Remove</button>
      </div>
      <div class="pk-meter ${m?`is-over`:``}" data-pk-meter>
        <div class="pk-bar"><span style="width:${Math.min(100,100*f.n/Math.max(1,o.limit))}%"></span></div>
        <span class="num small" data-pk-count>${f.approx?`≈ `:``}${e(f.n)} tokens of ${e(o.limit)} <span class="muted">(${f.why}; limit: ${o.source===`house`?a===`library`?`the house rule for a library skill`:`the house rule for the library card`:`max_tokens, where the server cuts`})</span></span>
        ${m?u`<span class="badge badge-warn" data-pk-limit-warn>over the limit</span>`:d}
      </div>
      <nav class="mem-tabs pk-panes" aria-label="package detail">
        ${Rt.map(e=>u`<a class="cond-tab ${e.id===s?`is-on`:``}" href="#" data-pk-pane=${e.id} aria-current=${e.id===s?`page`:d} @click=${t=>{t.preventDefault(),p({pview:e.id===`text`?null:e.id})}}>${e.label}</a>`)}
      </nav>
      <div class="pk-pane" data-pk-pane-body=${s}>
        ${s===`text`?this.paneText(t,n,i):s===`edit`?this.paneEdit(t,n,a,o):s===`classes`?this.paneClasses(t,n,a):this.paneUsage(t,a)}
      </div>
    </section>`}dirtyBody(e){return this.pk.draft?.packages[e]?.body!==this.pk.base?.doc.packages[e]?.body}tokensNow(t,n,r){if(n.source===`inline`&&(this.dirtyBody(t)||!r)){let e=W(n.body,!!n.library)[0];return this.counted?.text===e?{n:this.counted.n,approx:this.counted.how===`estimate`,why:this.counted.how===`engine`?`the draft, on the engine's tokenizer`:`the draft, estimated`}:{n:U(e),approx:!0,why:`the draft, estimated until the engine counts it`}}return r?{n:r.body_tokens,approx:this.catalog?.tokenizer===`estimate`,why:r.cut?`the whole text; ${e(r.tokens)} go in after the cut`:this.catalog?.tokenizer===`estimate`?`estimated`:`on the engine's tokenizer`}:{n:0,approx:!0,why:`not counted yet`}}md(e){return e?this.raw?u`<pre class="note-text pk-raw" data-pk-raw>${e}</pre>`:u`<div class="pg-md pk-md" data-pk-md>${st(ct(e))}</div>`:u`<p class="muted small">(empty)</p>`}paneText(t,n,r){let i=n.source===`inline`,a=this.item?.name===t?this.item:null;if(!i&&!a)return this.itemError?u`<p class="err-ink small">${this.itemError}</p>`:b(4);let[o,s]=i?W(n.body,!!n.library):[a.body,a.reference],c=i?{}:a.frontmatter;return u`
      <div class="toolbar pk-text-tools">${j([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`text view`})}
        ${!i&&a?.path?u`<span class="mono small muted pk-path" data-pk-path>${a.path}</span>`:d}</div>
      ${!i&&a&&!a.found?u`<p class="note" data-pk-missing>${h(`warn`)} The file is not on the box (<code>${a.path??n.source}</code>): nothing goes in.</p>`:d}
      ${Object.keys(c).length?u`<h3 class="pk-sub">Frontmatter <span class="muted small">the client's metadata; never injected</span></h3>
            ${this.raw?u`<pre class="note-text pk-raw" data-pk-front>${a.frontmatter_raw}</pre>`:u`<dl class="kv pk-front" data-pk-front>${Object.entries(c).map(([e,t])=>u`<div><dt>${e}</dt><dd>${t}</dd></div>`)}</dl>`}`:d}
      <h3 class="pk-sub">The text that goes in ${o?te(o,`Copy the text`):d}</h3>
      ${this.md(o)}
      ${r?.cut&&!this.dirtyBody(t)?u`<p class="note" data-pk-cut>${h(`warn`)} The server cuts this text at ${e(n.max_tokens)} tokens × 3.5 characters, at a line end, and adds “[... cut to the package's max_tokens]”: ${e(r.tokens)} tokens go in of ${e(r.body_tokens)}.</p>`:d}
      ${s?u`<details class="sec pk-ref" data-pk-ref>
            <summary>Reference <span class="sec-sub">${e(a?.tokens.reference??U(s))} tokens; only a read gets it (GET /v1/library/${t})</span></summary>
            <div class="sec-body">${this.md(s)}</div>
          </details>`:d}`}paneEdit(e,t,n,r){let i=t.source===`inline`;return u`
      <div class="form-grid">
        <label class="field"><span>title</span><input data-pk-title .value=${t.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>max tokens</span><input type="number" min="16" max="32768" data-pk-max .value=${String(t.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">the server cuts a longer text at a line end</span></label>
        ${i?d:u`<label class="field field-wide"><span>source</span><input class="mono" .value=${t.source} readonly /><span class="hint">${this.item?.path??`a file on the box`}</span></label>`}
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" data-pk-description .value=${t.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${i?this.editor(e,t,n,r):u`<div class="note pk-readonly" data-pk-readonly>${h(`info`)} The text is the file on the box${this.item?.path?u` at <code>${this.item.path}</code>`:d}, read when first needed and again when it changes. Editing files on the box is not done here. <button class="btn btn-sm" ?disabled=${!this.item?.found} @click=${()=>this.copyInline(e)} data-pk-copy-inline-edit>Copy to inline</button> makes an inline copy of its text that overrides the file.</div>`}`}editor(t,n,r,i){let[a,o]=W(n.body,!!n.library),s=this.tokensNow(t,n,this.catalog?.packages[t]??null),c=s.n>i.limit,l=this.editView;return u`<div class="pk-editor pk-view-${l}" data-pk-editor>
      <div class="toolbar pk-editor-tools">
        ${j([`write`,`preview`,`split`],l,e=>this.editView=e,{small:!0,label:`editor view`})}
        <span class="pk-count num small ${c?`is-over`:``}" data-pk-editor-count>${s.approx?`≈ `:``}${e(s.n)} of ${e(i.limit)} tokens${n.library&&o?u` <span class="muted">· reference ≈ ${e(U(o))}</span>`:d}</span>
        ${c?u`<span class="badge badge-warn" data-pk-editor-warn>over ${i.source===`house`?`the ${H[r]}'s ${e(i.limit)}`:`max_tokens: the server will cut it`}</span>`:d}
      </div>
      <div class="pk-editor-panes">
        ${l===`preview`?d:u`<textarea class="code pk-body" rows="18" spellcheck="false" data-pk-body aria-label="the package's text (Markdown)" .value=${n.body} @input=${e=>this.upd([`packages`,t,`body`],e.target.value)}></textarea>`}
        ${l===`write`?d:u`<div class="pg-md pk-md pk-preview" data-pk-preview>${st(ct(a))}</div>`}
      </div>
      <p class="hint">${n.library?`A library skill: the text before “## Reference” goes in; the reference only on a read. `:``}${i.source===`house`?`The house rule keeps ${H[r]} texts at ${e(i.limit)} tokens or fewer.`:`The server cuts the text at max_tokens × 3.5 characters, at a line end.`} Markdown; the preview is the dashboard's renderer, the model gets the text as written.</p>
    </div>`}paneClasses(e,t,n){let r=this.catalog?.packages[e]??null;if(n===`library`)return u`<p class="muted small" data-pk-library-classes>A library skill is searched, not listed by a class: on a new user message the router searches the library and keeps the skills INDEX.md fits to the turn's class.</p>
        <div class="pk-meta">${r?.fits?.length?u`<span class="muted small">INDEX.md fits</span>${r.fits.map(e=>D(e,this.ids))}`:u`<span class="muted small">INDEX.md does not list it: it stays in for every class.</span>`}</div>
        <p class="small"><a href="#/conductor?screen=packages&ptab=library">Each profile's retrieval switch and its picks</a></p>`;let i=this.rt.draft;return i?u`
      <p class="muted small">A class's packages go in by K1's order: the general ones whole and first, then the router asks the package question among the rest and puts in the one that fits. A class lists at most ${12}. Attaching edits the router document; it is saved with its own check below.</p>
      ${this.attachError?u`<p class="err-ink small" role="alert" data-pk-attach-error>${this.attachError}</p>`:d}
      <div class="pk-classes" data-pk-classes>
        ${i.classes.map(t=>{let n=t.policy.packages??[],r=n.includes(e);return u`<div class="pk-class ${r?`is-on`:``}" data-pk-class=${t.id}>
            <label class="switch"><input type="checkbox" .checked=${r} data-pk-attach=${t.id} aria-label=${`attach ${e} to ${t.id}`} @change=${n=>this.toggleAttach(t.id,e,n.target.checked)} /> ${D(t.id,this.ids,t.label)}</label>
            <span class="pk-order small" data-pk-order=${t.id}>${n.length?n.map((t,n)=>u`<span class="pk-order-item ${t===e?`is-me`:``}">${n+1}. ${t}${this.pk.draft?.packages[t]?.general?u` <span class="muted">general</span>`:d}</span>`):u`<span class="muted">no packages</span>`}</span>
            <span class="muted small num">${n.length}/${12}</span>
          </div>`})}
      </div>
      ${r&&Object.keys(r.profiles).length?u`<p class="small muted" data-pk-profiles>A profile's own routing also lists it: ${Object.entries(r.profiles).map(([e,t])=>`${e} (${t.join(`, `)})`).join(`; `)}. A profile's routing replaces the router's policy for that class (the Profiles screen).</p>`:d}
      <div class="toolbar"><button class="btn btn-sm btn-ghost" data-pk-router-versions @click=${()=>{this.rt.showVersions=!0,this.rt.loadVersions()}}>Router versions</button></div>`:b(3)}paneUsage(t,n){let r=this.usage?.packages[t]??null,i={day:0,week:0},a=(t,n)=>S({label:t,value:e(n.day),unit:`in 24 h`,sub:`${e(n.week)} in 7 days`}),o=this.item?.name===t?this.item:null,s=n===`library`||(r?.pick.week??0)>0;return u`
      <div class="stats pk-stats" style="--pk-tiles:${s?3:2}" data-pk-usage-tiles>
        ${a(`injected`,r?.injected??i)}${a(`shadow (would have)`,r?.shadow??i)}${s?a(`library picks`,r?.pick??i):d}
      </div>
      <h3 class="pk-sub">Recent rows <span class="muted small">the journal, newest first (7 days)</span></h3>
      ${this.recent==null?b(3):this.recent.length?u`<div class="pk-uses" data-pk-recent>${this.recent.map(e=>u`<div class="pk-use" data-pk-use=${e.id}>
                <span class="num small">${g(new Date(e.ts_ms).toISOString())}</span>
                <span class="badge ${e.how===`injected`?`badge-on`:e.how===`shadow`?`badge-shadow`:`badge-warn`}">${e.how}</span>
                ${D(e.class_id,this.ids)}
                <span class="muted small">${e.profile??`—`}${e.client_kind?` · ${e.client_kind}`:``}</span>
                ${e.session_key?u`<a class="mono small pk-session" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${e.session_key.length>28?`${e.session_key.slice(0,28)}…`:e.session_key}</a>`:d}
              </div>`)}</div>`:u`<p class="muted small" data-pk-recent-empty>No router row names it in the last 7 days.</p>`}
      ${n===`library`?u`<h3 class="pk-sub">Neighbours <span class="muted small">the library's BM25 on this skill's name, description and text</span></h3>
            ${o?.neighbours?o.neighbours.length?u`<div class="pk-neighbours" data-pk-neighbours>${o.neighbours.map(e=>u`<button class="pk-neighbour" data-pk-neighbour=${e.name} @click=${()=>this.open(e.name)}><strong>${e.name}</strong> <span class="num muted small">${e.score.toFixed(2)}</span><span class="muted small pk-desc">${e.description}</span></button>`)}</div>`:u`<p class="muted small">No other skill shares a word with it.</p>`:b(2)}`:d}`}};a([C({attribute:!1})],K.prototype,`ids`,void 0),a([C({attribute:!1})],K.prototype,`tiles`,void 0),a([C({attribute:!1})],K.prototype,`selected`,void 0),a([C({attribute:!1})],K.prototype,`tab`,void 0),a([C({attribute:!1})],K.prototype,`pane`,void 0),a([l()],K.prototype,`catalog`,void 0),a([l()],K.prototype,`catError`,void 0),a([l()],K.prototype,`usage`,void 0),a([l()],K.prototype,`filters`,void 0),a([l()],K.prototype,`hits`,void 0),a([l()],K.prototype,`libError`,void 0),a([l()],K.prototype,`item`,void 0),a([l()],K.prototype,`itemError`,void 0),a([l()],K.prototype,`recent`,void 0),a([l()],K.prototype,`raw`,void 0),a([l()],K.prototype,`editView`,void 0),a([l()],K.prototype,`counted`,void 0),a([l()],K.prototype,`review`,void 0),a([l()],K.prototype,`attachError`,void 0),a([l()],K.prototype,`found`,void 0),a([l()],K.prototype,`scanError`,void 0),K=a([t(`qse-cond-packages`)],K);var zt={not_loaded:`not loaded`,loading:`loading the weights`,warming:`warming up`,loaded:`loaded`,unloading:`unloading`,failed:`failed`};function Bt(e){let t=e.role;return t?[t.kind,t.window.replace(`_`,` `),t.preemptible?`preemptible`:`not preemptible`,`${t.latency_budget_ms} ms then ${t.fallback}`,`writes ${t.writes.replace(/_/g,` `)}`,`shares ${t.sharing.join(`, `)||`nothing`}`,t.memory_claim_gib==null?`no claim`:`claims ${y(t.memory_claim_gib)}`].join(` · `):null}function Vt(e){return e===`loaded`?`badge-ok`:e===`failed`?`badge-warn`:e===`loading`||e===`warming`||e===`unloading`?`badge-shadow`:``}function Ht(e){if(!e||e.allowed)return null;let t=e.reason?ne[e.reason]??e.reason:`refused`;if(e.reason!==`refused_memory`)return t;let n=[`MemAvailable ${y(e.mem_available_gib)}`];return e.engine_growth_gib&&n.push(`engine growth ${y(e.engine_growth_gib)}`),n.push(`claim ${y(e.claim_gib)}`),e.after_gib!=null&&n.push(`${y(e.after_gib)} left`,`${y(e.need_gib)} needed`),`${t} (${n.join(`, `)})`}function Ut(e){return e?e.allowed?`fits, ${y(e.after_gib)} left after the claim`:Ht(e)??`refused`:`—`}function Wt(e){let t=Object.entries(e).sort((e,t)=>t[1]-e[1]);return t.length?t.map(([e,t])=>`${e} ${t}`).join(` · `):`no call yet`}function Gt(e){let t=e.limits;return[t.context_tokens,t.question_tokens,t.max_options].map(e=>e==null?`—`:String(e)).join(` / `)}function Kt(e,t){let n=e.trim();if(!n)return`Write the question.`;let r=t.split(/[,\n]/).map(e=>e.trim()).filter(Boolean);return r.length<2?`Give at least two options.`:r.length>20?`At most 20 options (the model card’s limit).`:new Set(r).size===r.length?{type:`choice`,instructions:n,criteria:Object.fromEntries(r.map(e=>[e,null]))}:`Two options have the same name.`}function qt(e){return e.type===`noul`?[[`true`,e.noul??0],[`false`,1-(e.noul??0)]]:Object.entries(e.probabilities??{}).sort((e,t)=>t[1]-e[1])}var Jt=[`on_start`,`on_demand`,`off`],Yt={mode:`on`,rate_per_min:240,max_texts:16,max_chars:1e5,deadline_ms:15e3,threshold:.8,drive_rate_per_s:10,drive_sessions:4,drive_idle_s:30,drive_max_texts:36e3},q=class extends o{constructor(...e){super(...e),this.st=null,this.error=null,this.busy=null,this.said={},this.tState=`The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.`,this.tQuestion=`What kind of work does this request ask for?`,this.tOptions=`debug, code_edit, plan, chat`,this.tModel=`auto`,this.tOut=null,this.tErr=null,this.asking=!1,this.ed=new w(this,`deciders`),this.poller=null}connectedCallback(){super.connectedCallback(),this.ed.load(),this.poller=m(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{this.st=await A.deciders(),this.error=null}catch(e){let t=x(e);this.error=`${t.endpoint}: ${t.message}`}}async act(e,t){if(t!==`unload`||await O(`Unload ${e.name}?`,[`Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).`,`Loading it again goes through the residency planner.`],`Unload`)){this.busy=e.name,this.said={...this.said,[e.name]:``};try{let n=await N(()=>t===`load`?A.deciderLoad(e.name):A.deciderUnload(e.name));this.said={...this.said,[e.name]:t===`load`?`load started (${n.state??`loading`})`:`unloaded`}}catch(n){let r=n instanceof ie&&n.status===409&&t===`load`?`refused by the residency planner: ${n.message}`:x(n).message;this.said={...this.said,[e.name]:r}}finally{this.busy=null,await this.load()}}}setLoad(e,t){let n=this.ed.draft;if(!n)return;let r=n.models.findIndex(t=>t.name===e),i=r>=0?n.models.map((e,n)=>n===r?{...e,load:t}:e):[...n.models,{name:e,load:t}];this.ed.set({...n,models:i})}async ask(){let e=Kt(this.tQuestion,this.tOptions);if(typeof e==`string`){this.tErr=e;return}this.asking=!0,this.tErr=null;try{this.tOut=await A.deciderTest(this.tState,{q:e},this.tModel)}catch(e){this.tOut=null,this.tErr=x(e).message}finally{this.asking=!1}}render(){let e=this.st;if(this.error&&!e)return u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!e)return b(6);if(!e.enabled)return f(`Decision models`,u`<div class="state" data-deciders-off>
          <div class="state-title">${h(`warn`)} This engine serves no decision model</div>
          <p class="muted small">${e.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port behind the admin token, and this screen loads, unloads and tests them.</p>
        </div>`,{id:`deciders-off`});let t=e.models.filter(e=>e.state===`loaded`);return u`
      <div class="stats" data-decider-tiles>
        ${S({label:`loaded`,value:`${t.length} / ${e.models.length}`,sub:t.length?t.map(e=>e.name).join(`, `):`none: questions fail open`,warn:!t.length})}
        ${S({label:`on the device`,value:ee(t.reduce((e,t)=>e+t.device_bytes,0)),sub:`${e.device??`no device`} · own stream and memory pool`})}
        ${S({label:`a load now`,value:e.planner?.allowed?`fits`:`refused`,sub:Ut(e.planner),warn:!e.planner?.allowed})}
        ${S({label:`when they run`,value:`idle`,sub:`only while no request holds the engine (stage A)`})}
      </div>
      ${this.renderModels(e)}
      ${this.renderMemory(e)}
      ${this.renderApi(e)}
      ${this.renderQuestions(e)}
      ${this.renderTest(e)}
    `}renderModels(t){let n=this.ed.draft,r=e=>n?.models.find(t=>t.name===e)?.load??`on_demand`;return f(`Models`,u`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${t.models.map(t=>{let n=Ht(t.refusal),i=this.said[t.name],a=t.state===`loaded`||t.state===`failed`||t.state===`warming`;return u`<tr data-decider-row=${t.name}>
              <td><strong>${t.name}</strong><div class="muted small">${t.family} · ${t.contract}${t.aliases.length?u` · ${t.aliases.join(`, `)}`:d}</div>${t.role?u`<div class="muted small" data-decider-role=${t.name} title="the role contract this model signs (SRV-133); the registry checks it before it serves the model">${Bt(t)}</div>`:d}</td>
              <td><span class="badge ${Vt(t.state)}" data-decider-state=${t.state}>${zt[t.state]}</span>${t.warm===!1?u` <span class="muted small">not warmed</span>`:d}</td>
              <td><select aria-label=${`load policy of ${t.name}`} data-decider-load=${t.name} @change=${e=>this.setLoad(t.name,e.target.value)}>
                ${Jt.map(e=>u`<option value=${e} ?selected=${r(t.name)===e}>${e.replace(`_`,` `)}</option>`)}
              </select></td>
              <td class="num">${Gt(t)}</td>
              <td class="small">${t.precision??u`<span class="muted">—</span>`}<div class="muted small mono" title=${t.weight_hash??``}>${t.weight_hash?t.weight_hash.slice(0,12):``}</div></td>
              <td class="num num-col">${e(t.questions)}</td>
              <td class="num num-col">${t.stats?.forward_ms_p50==null?`—`:`${s(t.stats.forward_ms_p50)} / ${s(t.stats.forward_ms_p95)}`}</td>
              <td class="num num-col ${t.stats?.errors?`err-ink`:``}">${t.stats?e(t.stats.errors+t.stats.deadline_missed):`—`}</td>
              <td>${a?u`<button class="btn btn-sm" ?disabled=${this.busy===t.name} @click=${()=>this.act(t,`unload`)} data-unload=${t.name}>Unload</button>`:u`<button class="btn btn-sm btn-primary" ?disabled=${this.busy===t.name||t.state===`loading`||t.state===`unloading`} @click=${()=>this.act(t,`load`)} data-load=${t.name}>Load</button>`}</td>
            </tr>
            ${n||t.error||i?u`<tr class="is-warn" data-decider-note=${t.name}><td colspan="9" class="small">
                  ${n?u`<span class="warn-ink" data-refusal=${t.name}>${h(`warn`)} ${n}</span> `:d}
                  ${t.error?u`<span class="err-ink">${t.error}</span> `:d}
                  ${i?u`<span role="status">${i}</span>`:d}
                </td></tr>`:d}`})}</tbody>
      </table></div>
      ${t.unserved?.length?u`<p class="muted small">In the document but not served by this engine: ${t.unserved.join(`, `)} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>`:d}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,{id:`deciders-models`,sub:`the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen`})}renderMemory(e){let t=e.planner,n=e.last_refusal;return f(`Memory`,u`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${y(t?.mem_available_gib)}</div></div>
          ${t?.engine_growth_gib==null?d:u`<div class="stat"><div class="stat-label">engine growth to its peak</div><div class="stat-value num">${y(t.engine_growth_gib)}</div></div>`}
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${y(t?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${y(t?.floor_gib)} + ${y(t?.margin_gib)}</div></div>
          <div class="stat ${t?.allowed?``:`is-warn`}"><div class="stat-label">a load now</div><div class="stat-value num small">${Ut(t)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${n?u`Last refused engine load: ${Ht(n)} <span class="muted">(${n.source??``})</span>`:u`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable, less what the engine may still take up to its declared peak, minus the claim stays at or above floor + margin, never during a hold, never beside another load. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,{id:`deciders-memory`})}renderApi(t){let n=this.ed.draft,r={...Yt,...n?.api??{}},i=t.api?.stats,a=(e,t)=>n&&this.ed.set(T(n,[`api`,e],t)),o=(e,t,n=1)=>u`<label class="field"><span>${t}</span><input type="number" step=${n} .value=${String(r[e])} data-api-field=${e} @change=${t=>a(e,n<1?Number(t.target.value):Math.round(Number(t.target.value)))} /></label>`;return f(`Laya for scripts`,n?u`<div data-decider-api>
            <div class="toolbar" style="margin:0"><span class="muted small">route</span>${j([`off`,`on`],r.mode,e=>a(`mode`,e),{small:!0,label:`Laya route mode`})}<code class="small">POST /v1/laya/ask</code></div>
            <div class="form-grid">
              ${o(`rate_per_min`,`texts a minute (every caller)`)}
              ${o(`max_texts`,`texts in one call`)}
              ${o(`deadline_ms`,`deadline (ms) while a request runs`)}
              ${o(`threshold`,`sure at or above`,.05)}
              ${o(`drive_rate_per_s`,`texts a second (a playtest session)`)}
              ${o(`drive_sessions`,`playtest sessions at once`)}
            </div>
            <p class="muted small" data-api-stats>${i?`${e(i.calls)} calls, ${e(i.texts)} texts, ${e(i.errors)} errors (${e(i.rate_limited)} over the rate) since the start${i.last_ms==null?``:`; the last took ${s(i.last_ms)}`}${i.drive_sessions?`; ${e(i.drive_sessions)} playtest sessions live`:``}`:`no calls yet`}</p>
            <p class="muted small">The model's own scripts ask Laya typed questions about their evidence (a log tail, an output, a state dump) through the provider, open like the chat API: a choice, a scale or a yes-no question; Laya reads about 1,100 characters of each text, its head and its tail. Every call is a journal row (<code>laya_api</code>). A playtest (Laya plays the build: <code>.tandem/laya_drive.py</code>) sends a <code>session</code>: its own rate a second, one <code>laya_drive</code> row when it ends.</p>
          </div>`:b(2),{id:`deciders-api`,sub:`the route the model’s scripts call: on or off, its rate and its limits`})}renderQuestions(e){let t=this.ed,n=t.draft,r=e.models.map(e=>e.name),i=n?Object.entries(n.questions):[];return u`${f(`Questions`,n?u`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${i.map(([i,a])=>{let o=e.questions.find(e=>e.id===i);return u`<tr data-question=${i}>
                  <td class="mono">${i}</td>
                  <td><select aria-label=${`model for ${i}`} data-question-model=${i} @change=${e=>t.set(T(n,[`questions`,i],e.target.value))}>
                    ${[`auto`,...r].map(e=>u`<option value=${e} ?selected=${e===a}>${e===`auto`?`auto (the role’s model; the router picks)`:e}</option>`)}
                    ${a!==`auto`&&!r.includes(a)?u`<option value=${a} selected>${a} (not served)</option>`:d}
                  </select></td>
                  <td class="small">${Wt(o?.picks??{})}</td>
                </tr>`})}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); <code>judge</code> is the test judge's questions on a run's evidence (the <code>tester</code> role), <code>laya_api</code> the model's own scripts through <code>/v1/laya/ask</code>; the session monitor's Tier 1 questions join this list when they land.</p>`:b(3),{id:`deciders-questions`,sub:`which model answers each question the conductor asks`})}
      ${k(t)}
      ${M(t,`the deciders document`)}`}renderTest(t){let n=this.tOut,r=n?.answers.q;return f(`Test box`,u`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${e=>this.tState=e.target.value} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${e=>this.tQuestion=e.target.value} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${e=>this.tOptions=e.target.value} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${e=>this.tModel=e.target.value} data-test-model>
            ${[`auto`,...t.models.map(e=>e.name)].map(e=>u`<option value=${e} ?selected=${e===this.tModel}>${e}</option>`)}
          </select></label>
        </div>
        <div class="savebar"><button class="btn btn-primary" ?disabled=${this.asking} @click=${()=>this.ask()} data-test-ask>${h(`send`)} ${this.asking?`Asking…`:`Ask`}</button>
          ${this.tErr?u`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>`:d}</div>
        ${n&&r?u`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${r.type===`choice`?r.choice:r.type===`noul`?(r.noul??0).toFixed(4):(r.score??0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${r.confidence.toFixed(2)} · ${n.model} · ${s(n.latency_ms)} · ${n.path} · ${e(n.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${qt(r).map(([e,t])=>u`<div class="prob-row" data-prob=${e}><span class="mono small">${e}</span><span class="conf-bar"><span style="width:${Math.round(t*100)}%"></span></span><span class="num small">${t.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${n.routing.reason}${n.routing.requested?` (wanted ${n.routing.requested})`:``}</p>
            </div>`:d}`,{id:`deciders-test`,sub:`one choice question, as /v1/systemone answers it (it runs in the idle window like any other)`})}};a([l()],q.prototype,`st`,void 0),a([l()],q.prototype,`error`,void 0),a([l()],q.prototype,`busy`,void 0),a([l()],q.prototype,`said`,void 0),a([l()],q.prototype,`tState`,void 0),a([l()],q.prototype,`tQuestion`,void 0),a([l()],q.prototype,`tOptions`,void 0),a([l()],q.prototype,`tModel`,void 0),a([l()],q.prototype,`tOut`,void 0),a([l()],q.prototype,`tErr`,void 0),a([l()],q.prototype,`asking`,void 0),q=a([t(`qse-cond-deciders`)],q);var J=class extends o{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=m(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await A.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let t=x(e);this.error=`${t.endpoint}: ${t.message}`}}async loadSession(e){try{let t=await A.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(t){e===this.selected&&(this.turns=[]),this.error=x(t).message}}render(){if(this.error&&!this.list)return u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!this.list)return b(6);let t=[...new Set(this.list.map(e=>e.client_kind??`other`))],n=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return u`
      ${this.selected?this.renderSession(this.selected):d}
      ${f(`Sessions, last 7 days`,n.length?u`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${n.map(t=>u`<tr class="req-row ${t.key===this.selected?`is-open`:``}" data-session=${t.key} @click=${()=>p({session:t.key})}>
                  <td class="mono small cell-wrap">${E(t.key,28)}</td><td>${t.client_kind??`—`}</td><td>${t.profile??`—`}</td>
                  <td>${D(t.class_id,this.ids)}</td><td class="num num-col">${e(t.turns)}</td>
                  <td class="num num-col">${t.notes?`${t.notes} (${e(t.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${t.acts?u`${e(t.acts)}${t.gave_ups?u` <span class="badge badge-warn">${t.gave_ups} gave up</span>`:d}`:`—`}</td>
                  <td class="num num-col" data-interventions>${e(t.interventions??0)}</td>
                  <td>${t.last_checker?u`<span class="badge ${t.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${t.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${r(new Date(t.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:v(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:u`<label class="field field-search">${h(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${t.map(e=>u`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?u`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:d}
    `}renderSession(t){let n=this.turns,r=this.live.find(e=>e.key===t);if(n==null)return b(4);let a=n.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,o=n.reduce((e,t)=>e+(t.injected?.length??0),0),s=this.events.filter(e=>re(e)).length,c=Se(n);return u`<section class="panel" style="margin-bottom:16px" data-timeline=${t}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${E(t,44)} ${r?u`<span class="badge badge-on">live</span>`:d}</h2>
        <p class="panel-sub">${n.length} turns · ${o} hidden note${o===1?``:`s`} · ${s} monitor act${s===1?``:`s`}${r?` · class now ${r.class_id} · ${e(r.note_tokens)} note tokens replayed a turn`:``}${n.length?` · since ${g(new Date(n[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${A.exportUrl(t,`md`)} download=${`session-${t.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${h(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${A.exportUrl(t,`json`)} download=${`session-${t.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${h(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>p({session:null})}>${h(`x`)}</button></div></header>
      ${this.axis(de(n,this.events,this.ids,e=>i(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${n.map(e=>{let t=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/a*900)),n=(e.injected?.length??0)>0,r=!!e.shadow,i=(e.events??[]).filter(e=>e.decision!==`PASS`),o=i.some(e=>re(e))?`act`:i.length?`shadow`:null;return u`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${t}px; --cls: var(--series-${Be(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${Me(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${n?u`<span class="mark"></span>`:r?u`<span class="mark is-shadow"></span>`:d}
            ${o?u`<span class="mark is-monitor ${o===`shadow`?`is-shadow`:``}" data-monitor-mark=${o}></span>`:d}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${c.series.length?u`<qse-ribbon .series=${c.series} .labels=${c.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:d}
      <div class="feed" style="margin-top:10px">${n.slice().reverse().map(e=>ze(e,this.ids,e=>this.open=e,Me(e)+((e.events??[]).some(e=>re(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?u`<div style="margin-top:12px"><qse-cond-journal .session=${t}></qse-cond-journal></div>`:d}
    </section>`}axis(e){let t=e=>`${(e*100).toFixed(3)}%`;return u`<figure class="axis" data-axis aria-label="the session on one time axis">
      <div class="axis-lane axis-acts" aria-label=${`${e.acts.length} monitor decisions`}>
        ${e.acts.map(e=>u`<span class="axis-mark ${e.shadow?`is-shadow`:`is-act`}" style="left:${t(e.x)}" title=${e.title} data-axis-act=${e.shadow?`shadow`:`act`}></span>`)}
        ${e.notes.map(e=>u`<span class="axis-mark is-note ${e.shadow?`is-shadow`:``}" style="left:${t(e.x)}" title=${e.title} data-axis-note></span>`)}
      </div>
      <div class="axis-lane axis-turns" aria-label=${`${e.turns.length} turns`}>
        ${e.turns.map(e=>u`<span class="axis-turn ${e.compaction?`is-compaction`:``}" style="left:${t(e.x0)};width:${t(Math.max(0,e.x1-e.x0))};--cls:var(--series-${e.color})" title=${e.title} data-axis-turn></span>`)}
      </div>
      <div class="axis-lane axis-events" aria-label=${`${e.compactions.length} compactions, ${e.checkers.length} checker results`}>
        ${e.compactions.map(e=>u`<span class="axis-mark is-compaction" style="left:${t(e.x)}" title=${e.title} data-axis-compaction></span>`)}
        ${e.checkers.map(e=>u`<span class="axis-mark ${e.pass?`is-pass`:`is-fail`}" style="left:${t(e.x)}" title=${e.title} data-axis-checker=${e.pass?`pass`:`fail`}></span>`)}
      </div>
      <div class="axis-ticks" aria-hidden="true">${e.ticks.map((n,r)=>u`<span class="num ${r===0?`is-first`:r===e.ticks.length-1?`is-last`:``}" style="left:${t(n.x)}">${n.label.slice(0,5)}</span>`)}</div>
      <figcaption class="muted small">One time axis, ${e.ticks[0]?.label??``} to ${e.ticks[e.ticks.length-1]?.label??``}: ${e.turns.length} turns (the gaps are the client's time), ${e.acts.filter(e=>!e.shadow).length} monitor actions ◆ above (◇ shadow, ${e.acts.filter(e=>e.shadow).length}), ${e.notes.length} router note${e.notes.length===1?``:`s`}, ${e.compactions.length} compaction${e.compactions.length===1?``:`s`} ▲ and ${e.checkers.length} checker result${e.checkers.length===1?``:`s`} below (● pass, ○ fail; the gate numbers are in each one's title).</figcaption>
    </figure>`}tier1(){let e=he(this.events);return e?u`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${e.agree} of ${e.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${e.decisions} shadow decisions)</span></p>`:d}};a([C({attribute:!1})],J.prototype,`ids`,void 0),a([C({attribute:!1})],J.prototype,`selected`,void 0),a([l()],J.prototype,`list`,void 0),a([l()],J.prototype,`live`,void 0),a([l()],J.prototype,`turns`,void 0),a([l()],J.prototype,`events`,void 0),a([l()],J.prototype,`error`,void 0),a([l()],J.prototype,`query`,void 0),a([l()],J.prototype,`client`,void 0),a([l()],J.prototype,`open`,void 0),J=a([t(`qse-cond-sessions`)],J);var Xt={scope:`tools`,retention_days:30,clip_chars:4e3,loop_chars:200},Zt=/^### (user|assistant|summary|runtime|tool result \((.*)\)|tool call \((.*)\)|reasoning \((.*)\))$/;function Qt(e){let t=e[1];return e[2]===void 0?e[3]===void 0?e[4]===void 0?{kind:t===`user`||t===`assistant`||t===`summary`||t===`runtime`?t:`other`,title:t,name:null,clipped:!1}:{kind:`reasoning`,title:t,name:null,clipped:!1}:{kind:`tool_call`,title:t,name:e[3].trim()||null,clipped:!1}:{kind:`tool_result`,title:t,name:e[2].split(`,`)[0].trim()||null,clipped:/clipped to/.test(e[2])}}function $t(e){let t=e.replace(/\n+$/,``).split(`
`),n=t[0]??``,r=null,i=[],a=null,o=()=>{a&&i.push({kind:a.kind,title:a.title,name:a.name,clipped:a.clipped,body:a.lines.join(`
`).replace(/^\n+|\n+$/g,``)})};for(let e of t.slice(1)){let t=e.match(Zt);t?(o(),a={...Qt(t),lines:[]}):a?a.lines.push(e):e.startsWith(`compaction: `)&&(r=e)}o();let s=i.find(e=>e.kind===`runtime`);return{head:n,compaction:r,sections:i.filter(e=>e.kind!==`runtime`),runtime:s?en(s.body):[]}}function en(e){return e.split(`
`).filter(e=>e.trim()).map(e=>{let t=[`hidden`,`replayed`,`acts`,`checker`,`snapshot`,`finish`,`redactions`].find(t=>e.startsWith(`${t}: `))??`other`,n=e.match(/ · journal ([A-Za-z0-9_-]{4,40})$/),r=t===`checker`?e.match(/ (PASS|FAIL)(?: <-|$)/):null;return{kind:t,text:e,journal:n?n[1]:null,verdict:r?r[1]:null}})}function tn(e){if(!e.counts)return`counts unknown (no meta.json)`;let t=e.counts,n=[`${t.blocks} turn${t.blocks===1?``:`s`}`];return t.untranscribed&&n.push(`${t.untranscribed} without tools (counted only)`),n.push(`${t.acts} act${t.acts===1?``:`s`}`),t.snapshots&&n.push(`${t.snapshots} snapshot${t.snapshots===1?``:`s`}`),t.compactions&&n.push(`${t.compactions} compaction${t.compactions===1?``:`s`}`),t.redactions&&n.push(`${t.redactions} redacted`),e.checkers?.last&&n.push(e.checkers.last),n.join(` · `)}function nn(e){return{...Xt,...e?.memory?.transcripts??{}}}function Y(e,t){return e?e.replace(/^[ch]:/,``):t}var rn={mode:`on`,budget_gb:50,ttl_days:7,admit_after:2,min_free_gb:50},an={"opencode-head":`opencode head`,head:`session head`,package:`package`,memory:`memory file`,prefix:`repeated prefix`};function on(e){return{...rn,...e?.memory?.disk_cache??{}}}function sn(e){let t=[],n=e=>Number.isFinite(e)&&Math.round(e)===e;return e.budget_gb>=1&&e.budget_gb<=4e3||t.push(`the budget is 1 to 4,000 GB`),n(e.ttl_days)&&e.ttl_days>=1&&e.ttl_days<=90||t.push(`an entry is kept 1 to 90 days`),n(e.admit_after)&&e.admit_after>=1&&e.admit_after<=10||t.push(`a prefix is written after 1 to 10 prefills`),e.min_free_gb>=50&&e.min_free_gb<=4e3||t.push(`the NVMe keeps at least 50 GB free`),t}function cn(e,t){if(t.free_bytes==null)return{usable_gb:null,short:!1};let n=((t.bytes??0)+Math.max(0,t.free_bytes-Math.max(50,e.min_free_gb)*1e9))/1e9;return{usable_gb:n,short:n<e.budget_gb}}function ln(e){let t=e.lookups??0;return t?`${Math.round((e.hit_rate??0)*100)} % of ${t.toLocaleString(`en-US`)} prefill${t===1?``:`s`} it could have shortened`:`no prefill the disk could have shortened yet`}function un(e){let t=e=>typeof e==`number`?e.toLocaleString(`en-US`):String(e),n=[];return e.tokens!=null&&n.push(`${t(e.tokens)} tokens`),e.kind===`hit`&&e.read!=null&&n.push(`${t(e.read)} read`,`${t(e.ms)} ms`),e.kind===`write`&&(n.push(`${t(e.chunks)} chunk${e.chunks===1?``:`s`}`,`${t(e.anchors)} anchor${e.anchors===1?``:`s`}`),e.dropped&&typeof e.dropped==`object`&&n.push(`dropped ${Object.entries(e.dropped).map(([e,t])=>`${t} ${e}`).join(`, `)}`)),e.why!=null&&n.push(String(e.why)),e.kind===`clear`&&n.push(`${t(e.entries)} entries`),e.error!=null&&n.push(String(e.error)),[e.kind,...n].join(` · `)}var dn=e=>g(new Date(e*1e3).toISOString()),fn=t=>Object.entries(t??{}).filter(([,e])=>e).map(([t,n])=>`${e(n)} ${t}`).join(`, `)||`none`,X=class extends o{constructor(...e){super(...e),this.disk=null,this.error=null,this.busy=!1,this.cleared=null,this.settings=new w(this,`settings`)}connectedCallback(){super.connectedCallback(),this.load(),this.settings.load(),this.settings.onSaved=()=>void this.load()}async load(){try{this.disk=await N(()=>A.memoryDisk()),this.error=null}catch(e){this.error=x(e).message}}async clear(){let t=this.disk;if(t&&await O(`Clear the disk cache?`,[`Every entry goes: ${e(t.entries??0)} prefixes in ${e(t.chunks??0)} chunk files, ${n(t.bytes??0)}, and the counts that decide what is written next.`,`Nothing a request needs is lost: the next prefill of each prefix runs cold, as without the tier, and writes it again.`],`Clear`)){this.busy=!0;try{let t=await N(()=>A.memoryDiskClear());this.cleared=`cleared ${e(t.entries)} entries, ${n(t.bytes)}`,this.disk=t.disk}catch(e){this.error=x(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.disk)return u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!this.disk)return b(5);let e=this.disk;return u`${e.running?this.renderRunning(e):this.renderOff(e)}${this.renderSettings(e)}`}renderOff(e){return f(`Disk cache`,v(`The disk tier does not run on this server`,u`It starts with the engine when <code>QSE_DISK_CACHE</code> in serve.env names its directory (the flag is <code class="mono">${e.flag}</code> now)${e.free_bytes==null?d:u`; the NVMe has <span class="num">${n(e.free_bytes,0)}</span> free`}. Its settings below apply once it runs.`),{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM`})}renderRunning(t){let r=t.bytes??0,i=t.budget??0,a=i?r/i:0,o=t.free_bytes??null,s=o!=null&&t.floor!=null&&o<t.floor+1e10,l=u`<div class="stats disk-stats" data-disk-tiles>
      ${S({label:`on disk`,value:n(r,1),sub:u`of <span class="num">${n(i,0)}</span> · ${e(t.entries??0)} prefix${t.entries===1?``:`es`}, ${e(t.chunks??0)} chunk files`,warn:a>.95})}
      ${S({label:`NVMe free`,value:o==null?`—`:n(o,0),sub:u`never under <span class="num">${n(t.floor??5e10,0)}</span>`,warn:s})}
      ${S({label:`hit rate`,value:t.hit_rate==null?`—`:c(t.hit_rate,0),sub:ln(t),na:t.hit_rate==null})}
      ${S({label:`restored`,value:e(t.tokens_restored??0),unit:`tokens`,sub:t.last_restore_ms==null?`no restore yet`:u`last <span class="num">${e(Math.round(t.last_restore_ms))}</span> ms${t.last_read_gbs==null?d:u` at <span class="num">${t.last_read_gbs.toFixed(1)}</span> GB/s`}`})}
    </div>`,p=u`<div class="meter-block disk-meter">
      <div class="meter-head"><span>budget</span><span class="num">${n(r,1)} of ${n(i,0)} · ${c(a,0)}</span></div>
      <div class="meter" role="img" aria-label=${`disk cache ${c(a,0)} of its budget`}><span class="meter-fill ${a>.95?`is-warn`:``}" style="width:${Math.min(100,a*100).toFixed(1)}%"></span></div>
    </div>`,m=t.top??[],g=u`${l}${p}
      <p class="small muted disk-line" data-disk-counters>
        ${t.on?u`<span class="badge badge-on">on</span>`:u`<span class="badge">off (settings)</span>`}
        written: <span class="num">${e(t.writes_chunk??0)}</span> chunks, <span class="num">${e(t.writes_anchor??0)}</span> anchors, <span class="num">${n(t.bytes_written??0,1)}</span> ·
        evicted: ${fn(t.evicted)} · dropped: ${fn(t.dropped)}${t.failed_restores?u` · <span class="warn-ink">${e(t.failed_restores)} failed restore${t.failed_restores===1?``:`s`}</span>`:d}
        · staging buffer <span class="num">${n(t.staging_bytes||t.staging_planned||0,2)}</span>${t.staging_bytes?``:` (when first used)`} · <span class="mono">${t.root??``}</span>
      </p>
      ${m.length?this.renderEntries(m):v(`Nothing on disk yet`,`A prefix is written after ${t.settings.admit_after===1?`its first prefill`:`${t.settings.admit_after} prefills`} of at least ${e(t.min_tokens??4096)} tokens, and the first long request of an opencode session is written where its user message starts.`)}
      ${this.renderEvents(t)}`;return f(`Disk cache`,g,{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM; a hit is read only past the rows already in place`,tools:u`${this.cleared?u`<span class="small muted" data-disk-cleared>${this.cleared}</span>`:d}
        <button class="btn btn-sm disk-clear" ?disabled=${this.busy||!(t.entries||t.chunks)} @click=${()=>this.clear()} data-disk-clear>${h(`x`)} Clear disk cache</button>
        <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the disk cache">${h(`refresh`)}</button>`})}renderEntries(t){let r=e=>u`<span class="badge disk-kind disk-kind-${e.kind}">${an[e.kind]}</span>${e.label?u` <span class="mono small">${e.label}</span>`:d}${e.complete?d:u` <span class="badge badge-warn" title="a chunk below it is not on disk: it is not read, and goes first">incomplete</span>`}`,i=e=>e.preview?u`<div class="small muted disk-preview" data-disk-preview>${e.preview}</div>`:d;return u`<h3 class="disk-sub">Most-hit prefixes</h3>
      <div class="table-scroll mem-table"><table class="table" data-disk-entries>
        <thead><tr><th class="num-col">tokens</th><th>what</th><th class="num-col">hits</th><th>last used</th><th class="num-col">a hit reads</th></tr></thead>
        <tbody>${t.map(t=>u`<tr data-disk-row=${`${t.kind}-${t.tokens}`}><td class="num num-col">${e(t.tokens)}</td><td class="cell-wrap">${r(t)}${i(t)}</td>
            <td class="num num-col">${e(t.hits)}</td><td class="num">${dn(t.used)}</td><td class="num num-col">${n(t.reads_bytes,1)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards" role="list">${t.map(t=>u`<div class="mem-card disk-card" role="listitem" data-disk-card=${`${t.kind}-${t.tokens}`}>
          <div class="mem-card-head"><span class="num">${e(t.tokens)} tokens</span><span class="num small">${e(t.hits)} hit${t.hits===1?``:`s`}</span></div>
          <div>${r(t)}</div>${i(t)}
          <div class="small muted">last used ${dn(t.used)} · a hit reads ${n(t.reads_bytes,1)}</div>
        </div>`)}</div>`}renderEvents(e){let t=(e.events??[]).slice(-10).reverse();return t.length?u`<details class="sec" style="margin-top:12px" data-disk-events>
      <summary>Last events <span class="sec-sub">what it read, wrote, dropped and evicted (the journal keeps them, kind memory)</span></summary>
      <ul class="disk-events">${t.map(e=>u`<li class="small"><span class="num muted">${dn(e.ts)}</span> <span class="disk-ev disk-ev-${e.kind}">${un(e)}</span></li>`)}</ul>
    </details>`:d}renderSettings(e){let t=this.settings;if(!t.draft)return t.loadError?u`<p class="err-ink small">${t.loadError}</p>`:d;let r=on(t.draft),i=(e,n)=>t.set(T(t.draft,[`memory`,`disk_cache`],{...r,[e]:n})),a=(e,t,n,a,o=1)=>u`<label class="field"><span>${t}</span><input type="number" inputmode="decimal" min=${n} max=${a} step=${o} .value=${String(r[e])} data-disk-setting=${e} @input=${t=>i(e,Number(t.target.value))} /></label>`,o=cn(r,e),s=sn(r);return u`<details class="sec" style="margin-top:16px" data-sec="disk" open>
      <summary>Disk cache settings <span class="sec-sub">on or off, its size on the NVMe, how long it keeps an entry, what it writes</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-disk-mode><span class="muted small">disk cache</span>${j([`on`,`off`],r.mode,e=>i(`mode`,e),{small:!0,label:`disk cache mode`})}<span class="small">${r.mode===`on`?`prefixes are read from and written to the disk`:`nothing is read or written; the files stay`}</span></div>
        <div class="form-grid">
          ${a(`budget_gb`,`cache budget (GB)`,1,4e3)}
          ${a(`ttl_days`,`drop an entry unused for (days)`,1,90)}
          ${a(`admit_after`,`write a prefix after it was prefilled (times)`,1,10)}
          ${a(`min_free_gb`,`keep free on the NVMe (GB, at least 50)`,50,4e3)}
        </div>
        <p class="small" data-disk-free>
          ${e.free_bytes==null?d:u`The NVMe has <span class="num">${n(e.free_bytes,0)}</span> free${e.bytes?u`, and the cache holds <span class="num">${n(e.bytes,1)}</span>`:d}. `}
          ${o.short?u`<span class="warn-ink" data-disk-short>${h(`warn`)} At most <span class="num">${o.usable_gb.toFixed(0)} GB</span> of the ${r.budget_gb} GB fit above the ${Math.max(50,r.min_free_gb)} GB it keeps free; the tier stops there.</span>`:d}
        </p>
        <p class="muted small">Least valuable first when it is full: hits, times the tokens an entry saves per byte it holds, halved for every day unused. The first long request of an opencode session is written whatever the count, at the lines below where its user message starts. Applied from the next request; the free space it keeps is never under ${50} GB, whatever is set here.</p>
        ${s.length?u`<ul class="errors" role="alert" data-disk-errors>${s.map(e=>u`<li>${e}</li>`)}</ul>`:d}
        ${k(t)}
        ${M(t,`the settings`)}
      </div>
    </details>`}};a([l()],X.prototype,`disk`,void 0),a([l()],X.prototype,`error`,void 0),a([l()],X.prototype,`busy`,void 0),a([l()],X.prototype,`cleared`,void 0),X=a([t(`qse-cond-disk`)],X);var pn={tools:`requests with tools in full (and compactions); chats without tools as counters only`,all:`every request in full`,none:`nothing`},Z=class extends o{constructor(...e){super(...e),this.ids=[],this.selected=null,this.page=null,this.tab=null,this.list=null,this.one=null,this.error=null,this.oneError=null,this.query=``,this.busy=null,this.row=null,this.ev=null,this.rowError=null,this.settings=new w(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.tab!==`disk`&&(this.loadList(),this.settings.load())}updated(e){e.get(`tab`)===`disk`&&this.tab!==`disk`&&!this.list&&!this.error&&(this.loadList(),this.settings.draft||this.settings.load())}willUpdate(){let e=this.selected?`${this.selected}#${this.page??``}`:null;e!==this.loadedFor&&(this.loadedFor=e,this.one=null,this.oneError=null,this.selected&&this.loadOne(this.selected,this.page))}async loadList(){try{this.list=await N(()=>A.memorySessions()),this.error=null}catch(e){this.error=x(e).message}}async loadOne(e,t){try{let n=await N(()=>A.memorySession(e,t));e===this.selected&&(this.one=n)}catch(t){e===this.selected&&(this.oneError=x(t).message)}}async keep(e,t){if(e.session){this.busy=e.dir;try{await N(()=>A.memoryKeep(e.session,t)),await this.loadList(),this.one&&this.one.session.session===e.session&&(this.one={...this.one,session:{...this.one.session,keep:t}})}catch(e){this.error=x(e).message}finally{this.busy=null}}}async download(e,t){try{let n=await N(()=>A.memoryExport(e,t)),r=document.createElement(`a`);r.href=URL.createObjectURL(n),r.download=`transcript-${e.replace(/[^a-z0-9]+/gi,`-`)}.${t}`,r.click(),setTimeout(()=>URL.revokeObjectURL(r.href),1e3)}catch(e){this.oneError=x(e).message}}async openRow(e){this.rowError=null;try{let t=(await A.decision(e)).decision;t.kind===`monitor`?this.ev=t:this.row=t}catch(t){this.rowError=`journal row ${e}: ${x(t).message} (the journal keeps 30 days)`}}render(){let e=this.tab===`disk`,t=(t,n)=>u`<a class="cond-tab ${t===`disk`===e?`is-on`:``}" href="#" aria-current=${t===`disk`===e?`page`:d} data-mem-tab=${t??`sessions`} @click=${e=>{e.preventDefault(),p({mtab:t,memory:null,mpage:null})}}>${n}</a>`,n=u`<nav class="mem-tabs" aria-label="memory tabs">${t(null,`Sessions`)}${t(`disk`,`Disk cache`)}<span class="cond-tab is-off" title="the notes tab comes with the memory files (VIS-50)">Notes</span></nav>`;return e?u`${n}<qse-cond-disk></qse-cond-disk>`:u`
      ${n}
      ${this.selected?this.renderOne(this.selected):d}
      ${this.renderList()}
      ${this.renderSettings()}
      ${this.row?u`<qse-cond-decision .row=${this.row} .ids=${this.ids} .close=${()=>this.row=null}></qse-cond-decision>`:d}
      ${this.ev?u`<qse-cond-event .ev=${this.ev} .close=${()=>this.ev=null}></qse-cond-event>`:d}
    `}renderList(){if(this.error&&!this.list)return u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!this.list)return b(5);let t=this.list.writer,r=this.query.toLowerCase(),i=this.list.sessions.filter(e=>!r||`${e.session??e.dir} ${e.client??``} ${e.project??``}`.toLowerCase().includes(r)),a=e=>e.last_ms?g(new Date(e.last_ms).toISOString()):`—`,o=e=>u`<label class="switch" title="kept sessions are exempt from the ${t.settings.retention_days}-day prune"><input type="checkbox" .checked=${e.keep} ?disabled=${!e.session||this.busy===e.dir} data-keep=${e.dir} @click=${e=>e.stopPropagation()} @change=${t=>this.keep(e,t.target.checked)} aria-label=${`keep ${Y(e.session,e.dir)}`} /> keep</label>`,s=e=>e.checkers?.last?u`<span class="badge ${/ PASS$/.test(e.checkers.last)?`badge-ok`:`badge-warn`}">${e.checkers.last}</span>${e.checkers.PASS||e.checkers.FAIL?u` <span class="muted small num">${e.checkers.PASS??0} pass · ${e.checkers.FAIL??0} fail</span>`:d}`:u`<span class="muted">—</span>`,c=e=>e.session&&p({memory:e.session,mpage:null});return f(`Session transcripts`,i.length?u`<div class="table-scroll mem-table"><table class="table" data-memory-sessions>
              <thead><tr><th>last request</th><th>session</th><th>client</th><th>profile</th><th>project</th><th class="num-col">turns</th><th class="num-col">acts</th><th>checker</th><th class="num-col">size</th><th>keep</th></tr></thead>
              <tbody>${i.map(t=>u`<tr class="req-row ${t.session&&t.session===this.selected?`is-open`:``}" data-memory-row=${t.dir} @click=${()=>c(t)}>
                  <td class="num">${a(t)}</td><td class="mono small cell-wrap">${Y(t.session,t.dir)}</td><td>${t.client??`—`}</td><td>${t.profile??`—`}</td>
                  <td class="small cell-wrap">${t.project??`—`}</td>
                  <td class="num num-col">${t.counts?e(t.counts.blocks):`?`}</td><td class="num num-col">${t.counts?e(t.counts.acts):`?`}</td>
                  <td>${s(t)}</td><td class="num num-col">${n(t.bytes??0)}</td><td>${o(t)}</td></tr>`)}</tbody></table></div>
            <div class="mem-cards" role="list">${i.map(e=>u`<div class="mem-card ${e.session&&e.session===this.selected?`is-open`:``}" role="listitem" data-memory-card=${e.dir} @click=${()=>c(e)}>
                <div class="mem-card-head"><span class="mono small">${Y(e.session,e.dir)}</span>${o(e)}</div>
                <div class="small">${a(e)} · ${e.client??`—`} · ${e.profile??`—`}</div>
                <div class="small muted">${tn(e)}</div>
                ${e.project?u`<div class="small muted cell-wrap">${e.project}</div>`:d}
                <div>${s(e)}</div>
              </div>`)}</div>`:v(`No transcripts yet`,t.refused?`The memory directory ${t.root} is inside a git work tree, so nothing is written there.`:`TandemLLM requests with tools are written to ${t.root} when their stream closes; chats without tools are counted only.`),{sub:u`<span data-writer>${e(t.blocks)} block${t.blocks===1?``:`s`} written since the start · ${e(t.redactions)} secret${t.redactions===1?``:`s`} redacted · ${e(t.failed)} failed · ${e(t.dropped)} dropped · ${n(t.bytes)} · <span class="mono">${t.root}</span></span>`,tools:u`<label class="field field-search">${h(`search`)}<input placeholder="search a session, client or project" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search transcripts" /></label>
          <button class="btn btn-sm" @click=${()=>this.loadList()} aria-label="reload the list">${h(`refresh`)}</button>`,id:`cond-memory`})}renderOne(t){if(this.oneError)return u`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${h(`warn`)} ${this.oneError}</div></div>`;let n=this.one;if(!n)return b(4);let r=n.session,i=n.pages??1,a=n.page??1,o=i>1?u`<div class="toolbar mem-pager" data-pager><button class="btn btn-sm" ?disabled=${a<=1} @click=${()=>p({mpage:String(a-1)})}>Older</button>
          <span class="small num">page ${a} of ${i} · ${e(n.total??n.blocks.length)} blocks, ${n.per_page} a page</span>
          <button class="btn btn-sm" ?disabled=${a>=i} @click=${()=>p({mpage:String(a+1)})}>Newer</button></div>`:d;return u`<section class="panel" style="margin-bottom:16px" data-transcript=${t}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title mono">${Y(r.session,r.dir)}${r.keep?u` <span class="badge badge-on">kept</span>`:d}</h2>
        <p class="panel-sub">${r.client??`—`} · ${r.profile??`—`}${r.project?` · ${r.project}`:``} · ${tn(r)}${r.first_ms?` · since ${g(new Date(r.first_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <button class="btn btn-sm" @click=${()=>this.download(t,`md`)} data-mem-export="md">${h(`down`)} Markdown</button>
          <button class="btn btn-sm" @click=${()=>this.download(t,`json`)} data-mem-export="json">${h(`down`)} JSON</button>
          <a class="btn btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(t)}`} data-mem-timeline>Timeline</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the transcript" @click=${()=>p({memory:null,mpage:null})}>${h(`x`)}</button></div></header>
      ${this.rowError?u`<p class="warn-ink small" role="alert">${h(`warn`)} ${this.rowError}</p>`:d}
      ${o}
      ${n.blocks.length?u`<div class="mem-blocks">${n.blocks.map(e=>this.renderBlock(e))}</div>`:v(`No blocks`,`This session was counted but not written: its requests carried no tools.`)}
      ${o}
    </section>`}renderBlock(t){let n=$t(t.text),r=n.runtime.filter(e=>e.kind===`hidden`).length,i=n.runtime.find(e=>e.kind===`acts`),a=u`<span class="mem-turn num">turn ${t.turn??`?`}</span>
      <span class="muted small num">${t.time??``}</span>
      ${t.class_id?D(t.class_id,this.ids):d}
      ${t.effort?u`<span class="badge">${t.effort}</span>`:d}
      ${t.reasoning_tokens?u`<span class="muted small num">${e(t.reasoning_tokens)} reasoning tokens</span>`:d}
      ${r?u`<span class="badge badge-shadow" data-hidden-count=${r}>${r} hidden</span>`:d}
      ${i?u`<span class="badge badge-warn">${i.text.replace(/^acts: /,`acts `)}</span>`:d}`;return t.compaction?u`<details class="mem-block is-compaction" data-block=${t.turn??``} data-compaction>
        <summary><span class="mem-block-head">${a}<span class="badge">compaction</span><span class="small">${n.compaction?.replace(/^compaction: /,``)??``}</span></span></summary>
        ${n.sections.map(e=>this.renderSection(e))}${this.renderRuntime(n.runtime)}
      </details>`:u`<article class="mem-block" data-block=${t.turn??``}>
      <div class="mem-block-head">${a}</div>
      ${n.sections.map(e=>this.renderSection(e))}
      ${this.renderRuntime(n.runtime)}
    </article>`}renderSection(e){let t=e.body.length>600||e.body.split(`
`).length>12,n=u`<pre class="mem-text">${e.body}</pre>`;return e.kind===`tool_result`||e.kind===`tool_call`&&t||e.kind===`reasoning`?u`<details class="mem-sec mem-${e.kind}" ?open=${!t&&e.kind!==`reasoning`} data-sec-kind=${e.kind}>
        <summary class="small">${e.title}${e.clipped?u` <span class="badge badge-warn">clipped</span>`:d}</summary>${n}</details>`:u`<div class="mem-sec mem-${e.kind}" data-sec-kind=${e.kind}><div class="small muted">${e.title}</div>${n}</div>`}renderRuntime(e){return e.length?u`<ul class="mem-runtime" aria-label="what the runtime did">${e.map(e=>u`<li class="mem-rt mem-rt-${e.kind} ${e.verdict?`is-${e.verdict.toLowerCase()}`:``}" data-rt=${e.kind}>
        ${e.kind===`hidden`?h(`info`):d}<span>${e.text.replace(/ · journal [A-Za-z0-9_-]+$/,``)}</span>
        ${e.journal?u`<button class="btn btn-ghost btn-sm" @click=${()=>this.openRow(e.journal)} data-journal=${e.journal} aria-label=${`open journal row ${e.journal}`}>journal row</button>`:d}
      </li>`)}</ul>`:d}renderSettings(){let e=this.settings;if(!e.draft)return e.loadError?u`<p class="err-ink small">${e.loadError}</p>`:d;let t=nn(e.draft),n=(n,r)=>e.set(T(e.draft,[`memory`,`transcripts`],{...t,[n]:r})),r=(e,r,i,a)=>u`<label class="field"><span>${r}</span><input type="number" min=${i} max=${a} .value=${String(t[e])} data-mem-setting=${e} @input=${t=>n(e,Math.round(Number(t.target.value)))} /></label>`;return u`<details class="sec" style="margin-top:16px" data-sec="transcripts">
      <summary>Transcript settings <span class="sec-sub">what the engine writes to the box, how long it keeps it, where tool output is clipped</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-mem-scope><span class="muted small">write</span>${j([`tools`,`all`,`none`],t.scope,e=>n(`scope`,e),{small:!0,label:`transcript scope`})}<span class="small">${pn[t.scope]}</span></div>
        <div class="form-grid">
          ${r(`retention_days`,`keep a session for (days; kept sessions are exempt)`,1,400)}
          ${r(`clip_chars`,`clip a tool result or a call at (characters)`,200,1e5)}
          ${r(`loop_chars`,`a loop the D1 guard cut keeps its first and last (characters)`,0,4e3)}
        </div>
        <p class="muted small">Every block passes the secret-pattern refusal before it is written: a key, token or password becomes <code>[redacted:kind]</code> and the journal counts it, never the value. There is no switch for it. The files are <code>0600</code> under the engine's home on the box, never in a repository, and the ledger's backup never holds them. Khaled's decision 2: chats without tools are counters only, unless the scope is widened here.</p>
        ${k(e)}
        ${M(e,`the settings`)}
      </div>
    </details>`}};a([C({attribute:!1})],Z.prototype,`ids`,void 0),a([C({attribute:!1})],Z.prototype,`selected`,void 0),a([C({attribute:!1})],Z.prototype,`page`,void 0),a([C({attribute:!1})],Z.prototype,`tab`,void 0),a([l()],Z.prototype,`list`,void 0),a([l()],Z.prototype,`one`,void 0),a([l()],Z.prototype,`error`,void 0),a([l()],Z.prototype,`oneError`,void 0),a([l()],Z.prototype,`query`,void 0),a([l()],Z.prototype,`busy`,void 0),a([l()],Z.prototype,`row`,void 0),a([l()],Z.prototype,`ev`,void 0),a([l()],Z.prototype,`rowError`,void 0),Z=a([t(`qse-cond-memory`)],Z);var Q=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([A.stats(168),A.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let t=x(e);this.error=`${t.endpoint}: ${t.message}`}}async promote(e,t,n,r,i){if(await O(`Promote ${n} to ${t===`task`?`act`:`on`} in ${e}?`,[t===`task`?`The session monitor will act on ${n} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${n} decisions instead of only journaling them: ${n===`effort`?`budgets and thinking off go to the engine`:n===`packages`?`skills and packages go into the prompt`:n===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${r.toFixed(2)} over ${i} labelled decisions. A new profile version records it.`],`Promote`))try{let a=await A.doc(`profiles`,e),o=t===`task`?T(a.doc,[`tasks`,n,`mode`],`act`):T(a.doc,[`decisions`,n],`on`),s=await N(()=>A.put(`profiles`,e,o,`${n} to ${t===`task`?`act`:`on`}, precision ${r.toFixed(2)} / ${i}`,a.version));this.msg=`${e} v${s.version}: ${n} promoted`,this.onSaved()}catch(e){this.msg=x(e).message}}async loadAb(){try{this.abVersions=(await A.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await A.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await A.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=x(e).message}}render(){return this.error&&!this.stats?u`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`:this.stats?u`
      ${this.msg?u`<div class="note note-info" role="status">${this.msg}</div>`:d}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:b(8)}renderPromotion(t){let n=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,r=(t.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return f(`Shadow precision and promotion`,u`<p class="muted small">${He}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${t.precision.length||r.length?u`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${t.precision.map(t=>{let r=_e(t),i=n(t.profile,t.decision);return u`<tr data-precision-row=${`${t.profile}/${t.decision}`}>
                    <td>${t.profile}</td><td class="mono">${t.decision}</td><td><span class="badge ${i===`on`?`badge-on`:i===`shadow`?`badge-shadow`:``}">${i}</span></td>
                    <td class="num num-col">${e(t.decisions)}</td><td class="num num-col">${e(t.labelled)}</td><td class="num num-col">${t.precision==null?`—`:t.precision.toFixed(2)}</td>
                    <td class="small ${r.allowed?``:`muted`}">${r.why}</td>
                    <td><button class="btn btn-sm ${r.allowed?`btn-primary`:``}" ?disabled=${!r.allowed||i!==`shadow`} title=${r.allowed?``:He} data-promote=${`${t.profile}/${t.decision}`} @click=${()=>this.promote(t.profile??``,`decision`,t.decision,t.precision??0,t.labelled)}>Promote to on</button></td></tr>`})}
                ${r.map(t=>{let n=_e(t);return u`<tr data-precision-row=${`task/${t.task}`}><td>—</td><td class="mono">${t.task}</td><td><span class="badge ${t.mode===`act`?`badge-on`:`badge-shadow`}">${t.mode}</span></td>
                    <td class="num num-col">${e(t.decisions)}</td><td class="num num-col">${e(t.labelled)}</td><td class="num num-col">${t.precision==null?`—`:t.precision.toFixed(2)}</td>
                    <td class="small ${n.allowed?``:`muted`}">${n.why}</td>
                    <td><button class="btn btn-sm ${n.allowed?`btn-primary`:``}" ?disabled=${!n.allowed||t.mode!==`shadow`} data-promote=${`task/${t.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,t.task,t.precision??0,t.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:v(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${t.monitor?d:u`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let e=this.reports.find(e=>e.kind===`classifier`);if(!e)return f(`Classifier accuracy`,v(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let t=e.stages??{},n=e.set??{},r=(t[this.stage]??Object.values(t)[0])?.confusion??{},i=n.classes??Object.keys(r);return f(u`Classifier accuracy <span class="muted small num">${n.name??``} · ${n.n??`?`} prompts · ${e.created_at?g(String(e.created_at)):``}</span>`,u`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(t).map(([e,t])=>u`<tr class="req-row ${e===this.stage?`is-open`:``}" @click=${()=>this.stage=e}><td class="mono">${e}</td><td class="num num-col">${c(t.accuracy,1)}</td><td class="num num-col">${t.answered} / ${t.n}</td><td class="num num-col">${t.accuracy_answered==null?`—`:c(t.accuracy_answered,1)}</td><td class="num num-col">${s(t.p50_ms)}</td><td class="num num-col">${s(t.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${i.length&&Object.keys(r).length?u`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...i,`—`].map(e=>u`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${i.map(e=>u`<tr><th>${D(e,this.ids)}</th>${[...i,`—`].map(t=>{let n=r[e]?.[t]??0;return u`<td class="cm-cell ${e===t?`cm-diag`:n?`cm-miss`:``}">${n||``}</td>`})}</tr>`)}</tbody></table></div>`:u`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}
        ${Oe(e).length?u`<p class="muted small" style="margin-top:12px">Digest variants (SRV-153): <strong>today</strong> the 2,000-character digest the decider cut from the front, <strong>sized</strong> each stage's own head and tail, <strong>line</strong> sized plus the context line. <strong>router</strong> is the session's class after the stickiness rule.</p>
            <div class="table-scroll"><table class="table" data-classifier-variants>
              <thead><tr><th>digest</th><th>set</th><th class="num-col">n</th><th class="num-col">decider</th><th class="num-col">cascade</th><th class="num-col">router</th><th class="num-col">decider p50</th></tr></thead>
              <tbody>${Oe(e).map(e=>u`<tr><td class="mono">${e.variant}</td><td class="mono small">${e.set}</td><td class="num num-col">${e.n}</td><td class="num num-col">${e.decider==null?`—`:c(e.decider,1)}</td><td class="num num-col">${e.cascade==null?`—`:c(e.cascade,1)}</td><td class="num num-col">${e.router==null?`—`:c(e.router,1)}</td><td class="num num-col">${s(e.p50)}</td></tr>`)}</tbody></table></div>`:d}`,{id:`cond-classifier`})}renderSuites(){let t=this.reports.filter(e=>e.kind===`suite`),n=oe(this.reports,this.suiteFile),r=fe(n,t),i=n?.rows??[],a=Te(this.reports,this.engine),o=Ee(i),s=n?.verdict,l=n?.comparison??[],p=n?.raw_row,m=n?.build,g=s?.indicative??rt(i),_=(t,n=1)=>t==null?`—`:Number.isInteger(t)?e(t):t.toFixed(n);return f(u`TandemLLM against the raw engine ${n?u`<span class="muted small num">night ${n.night??`—`}${m?` · ${m.version??`?`} ${E(m.code_sha,12)}`:``}${n.profile?` · profile ${n.profile}`:``}</span>`:d}`,u`${a.tone===`ok`?u`<p class="small" data-guard-detail=${a.status}>${h(`check`)} ${a.line}${a.detail?u`: <span class="num">${a.detail}</span>`:d}.</p>`:u`<div class="guard-line ${a.tone===`note`?`guard-note`:``}" data-guard-detail=${a.status}>${h(a.tone===`note`?`info`:`warn`)}<span>${a.line}${a.detail?u`. <span class="small">${a.detail}</span>`:d}</span></div>`}
        ${t.length>1?u`<label class="field suite-pick"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${t.map(e=>u`<option value=${String(e.file)} ?selected=${e===n}>${String(e.night??e.file)} · ${String(e.file)}${fe(e,t)?` · superseded`:``}</option>`)}</select></label>`:d}
        ${r?u`<div class="guard-line guard-note" data-superseded>${h(`info`)}<span><span class="badge badge-warn">superseded</span> by <span class="mono">${r.file}</span>${r.why?u`: <span class="small">${r.why}</span>`:d}. Its numbers are kept as published; read the newer report.</span></div>`:d}
        ${s?u`<div class="note" data-eval-verdict>
              <span class="badge ${s.better?`badge-ok`:s.better===!1?`badge-warn`:``}"><strong>${s.better?`Better`:s.better===!1?`Not better`:`Not shown yet`}</strong></span>${g?u` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:d} — ${s.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${p?.status??`—`}${p?.source?` (${p.source})`:``}${n?.driver?` · driver ${n.driver.mode}`:``}${n?.isolation?` · ${n.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:d}
        ${o.length?u`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">turns</th><th class="num-col">thinking tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>efforts (turns)</th><th>build · night</th></tr></thead>
              <tbody>${o.flatMap(t=>{let n=l.find(e=>e.suite===t.suite||e.suite_id===(t.raw??t.tandem)?.suite_id);return[t.raw,t.tandem].filter(Boolean).map((r,i)=>u`<tr data-suite=${t.suite} data-arm=${r.arm}>
                    <td>${i===0?t.suite:``}${i===0&&!t.cmp.ok?u` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${t.cmp.why}</div>`:d}${i===0&&n?u` <span class="badge ${n.verdict===`better`?`badge-ok`:n.verdict===`worse`||n.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${n.verdict}>${n.verdict}</span>`:d}${i===0&&(r.indicative??(r.n??0)<10)?u` <span class="badge" data-row-indicative>indicative</span>`:d}</td>
                    <td class="mono">${Ke(r.arm)}${r.pristine===!1?u` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:d}</td>
                    <td class="num num-col">${r.n??`—`}</td>
                    <td class="num num-col">${r.pass_rate==null?`—`:c(r.pass_rate,0)}</td>
                    <td class="num num-col">${r.gates_passed??`—`}${r.gates_total?` / ${r.gates_total}`:``}</td>
                    <td class="num num-col">${_(r.minutes_to_pass)}</td><td class="num num-col">${_(r.wall_minutes)}</td>
                    <td class="num num-col">${r.tokens==null?`—`:e(Math.round(r.tokens))}</td>
                    <td class="num num-col" data-turns>${_(r.turns)}</td><td class="num num-col" data-thinking>${r.reasoning_tokens==null?`—`:e(Math.round(r.reasoning_tokens))}</td>
                    <td class="num num-col">${_(r.decode_tps)}</td>
                    <td class="num num-col">${r.empty_stops??`—`}</td><td class="num num-col">${r.continuations??`—`}</td>
                    <td class="num num-col">${r.interventions??`—`}</td><td class="num num-col">${r.gave_ups??`—`}</td>
                    <td class="num num-col">${r.cache_hit==null?`—`:c(r.cache_hit,0)}</td>
                    <td class="small" data-efforts>${it(r)}</td>
                    <td class="small">${E(r.build,16)} · ${r.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts. Tokens, turns and thinking tokens are medians per run.</p>`:v(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return f(`A/B of two profile versions`,u`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>u`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>u`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(t=>u`<option value=${t.version} ?selected=${t.version===this[e]}>v${t.version} ${E(t.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?u`<div class="diff" data-ab>${this.abChanges.map(e=>u`<div class="diff-row"><span>${ve(e.path)}</span><span class="diff-before">${E(e.before,80)}</span><span class="diff-after">${E(e.after,80)}</span></div>`)}</div>`:u`<p class="muted small">The two versions are the same.</p>`:d}`,{id:`cond-ab`})}};a([C({attribute:!1})],Q.prototype,`ids`,void 0),a([C({attribute:!1})],Q.prototype,`tiles`,void 0),a([C({attribute:!1})],Q.prototype,`onSaved`,void 0),a([l()],Q.prototype,`stats`,void 0),a([l()],Q.prototype,`reports`,void 0),a([l()],Q.prototype,`engine`,void 0),a([l()],Q.prototype,`error`,void 0),a([l()],Q.prototype,`msg`,void 0),a([l()],Q.prototype,`stage`,void 0),a([l()],Q.prototype,`abProfile`,void 0),a([l()],Q.prototype,`abVersions`,void 0),a([l()],Q.prototype,`abA`,void 0),a([l()],Q.prototype,`abB`,void 0),a([l()],Q.prototype,`abChanges`,void 0),a([l()],Q.prototype,`abCounts`,void 0),a([l()],Q.prototype,`suiteFile`,void 0),Q=a([t(`qse-cond-compare`)],Q);var mn=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`deciders`,label:`Deciders`},{id:`sessions`,label:`Sessions`},{id:`memory`,label:`Memory`},{id:`compare`,label:`Compare`}],$=class extends o{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.guard=null,this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=m(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&mn.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r,i]=await Promise.all([A.profiles(),this.meta?Promise.resolve(this.meta):A.meta(),A.config(),A.doc(`router`),A.evals()]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.guard=Te(i.reports,i.engine),this.error=null}catch(e){let t=x(e);this.error=t.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${t.endpoint}: ${t.message}`}}async setMode(e){let t=this.tiles.find(e=>e.name===this.active);if(t&&!t.locked&&(e!==`on`||await O(`Turn the ${t.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let n=await N(()=>A.setMode(t.name,e,`mode to ${e}`,t.version));this.reloaded=`v${n.version} · reloaded ${n.loaded_at?i(n.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=x(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),t=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return u`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${h(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>p({profile:e.target.value})}>
                ${this.tiles.map(e=>u`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?j([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):d}
            ${this.reloaded?u`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:d}
          </div>
        </div>
        ${this.switchError?u`<div class="guard-line" role="alert">${h(`warn`)} ${this.switchError}</div>`:d}
        ${this.error?u`<div class="guard-line" role="alert">${h(`warn`)} ${this.error}</div>`:d}
        ${this.guard&&this.guard.tone!==`ok`?u`<div class="guard-line ${this.guard.tone===`note`?`guard-note`:``}" data-guard=${this.guard.status}>${h(this.guard.tone===`note`?`info`:`warn`)}<span>${this.guard.line}${this.guard.detail?u`. <span class="small">${this.guard.detail}</span>`:d}</span></div>`:d}
        ${t.map(([e,t])=>u`<div class="guard-line" role="alert">${h(`warn`)} ${e}: ${t.error?.message} (kept version ${t.version})</div>`)}
        ${e?.locked?u`<div class="note" data-locked>${h(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:d}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${mn.map(e=>u`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:d}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return u`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return u`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return u`<qse-cond-monitor .meta=${e.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return u`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return u`<qse-cond-packages .ids=${e.ids} .tiles=${e.tiles} .selected=${this.params.get(`pkg`)} .tab=${this.params.get(`ptab`)} .pane=${this.params.get(`pview`)}></qse-cond-packages>`;case`deciders`:return u`<qse-cond-deciders></qse-cond-deciders>`;case`sessions`:return u`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`memory`:return u`<qse-cond-memory .ids=${e.ids} .selected=${this.params.get(`memory`)} .page=${Number(this.params.get(`mpage`))||null} .tab=${this.params.get(`mtab`)}></qse-cond-memory>`;case`compare`:return u`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};a([C({attribute:!1})],$.prototype,`params`,void 0),a([l()],$.prototype,`tiles`,void 0),a([l()],$.prototype,`defaultProfile`,void 0),a([l()],$.prototype,`meta`,void 0),a([l()],$.prototype,`config`,void 0),a([l()],$.prototype,`ids`,void 0),a([l()],$.prototype,`guard`,void 0),a([l()],$.prototype,`error`,void 0),a([l()],$.prototype,`switching`,void 0),a([l()],$.prototype,`reloaded`,void 0),a([l()],$.prototype,`switchError`,void 0),$=a([t(`qse-conductor`)],$);export{$ as QseConductor};