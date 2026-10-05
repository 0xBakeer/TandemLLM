import{C as e,E as t,F as n,G as r,I as i,J as a,K as o,L as s,M as c,O as l,S as u,W as d,Y as f,Z as p,b as m,c as h,d as ee,f as te,g,h as ne,j as _,k as v,n as re,q as y,r as b,t as ie,v as x,x as S,y as C}from"./app-DCveUBto.js";import{$ as ae,A as oe,B as se,C as ce,D as le,DocEditor as w,E as ue,F as de,G as fe,H as pe,I as T,J as me,K as he,L as ge,M as _e,N as ve,O as ye,P as be,Q as xe,R as E,S as Se,T as Ce,U as we,V as Te,W as Ee,X as De,Y as Oe,Z as ke,_ as Ae,a as je,b as Me,c as Ne,clsChip as D,confView as Pe,confirmDialog as O,d as Fe,diffList as Ie,errorsList as k,et as A,eventRow as Le,f as Re,feedRow as ze,g as Be,h as Ve,i as He,j as Ue,k as We,l as Ge,m as Ke,modeSeg as j,n as qe,o as Je,p as Ye,q as Xe,s as Ze,saveBar as M,sourceWords as Qe,t as $e,u as et,v as tt,versionsPanel as nt,w as rt,withToken as N,x as it,y as at,z as ot}from"./chunk-_QyA8NVF.js";import{n as st,t as ct}from"./chunk-BAAC9L3f.js";var P=class extends r{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=h(()=>this.loadStats(),2e4),this.feedPoll=h(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await A.stats(24),this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await A.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await A.event(e)).event}catch(e){this.error=t(e).message}}renderMonitor(e){let t=ye(e);return p`<div class="grid-2 grid-2-wide">
      ${C(`Monitor acts an hour, last 24 h`,e.acts?p`<qse-ribbon .series=${t} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:g(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${C(`Last actions of the monitor`,e.last_actions.length?p`<div class="feed" data-last-actions>${e.last_actions.map(e=>p`<button class="feed-row" data-action=${e.id} @click=${()=>this.openAction(e.id)}>
                <span class="feed-time">${s(new Date(e.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
                <span class="feed-conf muted small">${e.outcome?.kind??`running`}</span>
                <span class="feed-what">${ee({task:e.task,decision:e.decision,mode:`act`,trigger:e.trigger,cap:e.cap})}</span>
                <span class="feed-act">${e.session_key?p`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:f}</span>
              </button>`)}</div>`:p`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:p`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${x(`external`)}</a>`})}
    </div>`}render(){let e=this.stats;if(this.error&&!e)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!e)return m(6);let t=e.classifier,r=t.decider,a=t.qwen,o=r?.calls?r:a?.calls?a:r,s=Object.entries(e.roles),l=s.filter(([,e])=>e.up).length,d=e.monitor;return p`
      <div class="stats" data-tiles>
        ${S({label:`decisions, 24 h`,value:c(e.decisions),sub:p`<span class="num">${c(e.acted)}</span> acted · <span class="num">${c(e.shadow)}</span> in shadow`,title:`${e.injections} notes, ${e.injected_tokens} tokens injected`})}
        ${S({label:`monitor, 24 h`,value:d?c(d.acts):`—`,sub:d?p`acts · <span class="num">${c(d.shadow)}</span> in shadow · <span class="num" data-gave-up>${c(d.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!d,warn:(d?.gave_up??0)>0,title:d?`${c(d.hidden_tokens)} hidden tokens`:void 0})}
        ${S({label:`classifier p50 / p95${!r?.calls&&a?.calls?` (Qwen)`:``}`,value:o?.p50_ms==null?`—`:`${Math.round(o.p50_ms)} / ${Math.round(o.p95_ms??0)}`,unit:o?.p50_ms==null?void 0:`ms`,sub:p`decider ${c(r?.calls??0)} calls${a?p` · Qwen ${c(a.calls)} (${n(a.p50_ms)})`:f} · fail-open ${i(e.fail_open_rate,1)}`,warn:(e.fail_open_rate??0)>.1})}
        ${S({label:`roles online`,value:`${l} / ${s.length}`,sub:p`${s.map(([e,t])=>p`<span title=${t.error??``} class=${t.up?``:t.mode===`off`?`muted`:`err-ink`} data-role=${e}>${t.up?`●`:`○`} ${e} </span>`)}`,warn:s.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${C(`Decisions an hour, last 24 h`,e.decisions?p`<qse-ribbon .series=${ce(e,this.ids)} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:g(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${C(`Task classes, 24 h`,p`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${e.classes.map(e=>p`<tr><td>${D(e.class_id,this.ids)}</td><td class="num num-col">${c(e.decisions)}</td><td class="num num-col">${e.labelled?`${e.agree} / ${e.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(e.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(e.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${C(p`Live decisions ${this.paused?p`<span class="badge">paused</span>`:p`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?p`<div class="feed" data-feed>${this.feed.map(e=>ze(e,this.ids,e=>this.open=e,Me(e)))}</div>`:p`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:p`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${x(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${x(`external`)}</a>`})}
        ${C(`Profiles by client`,p`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(t=>p`<button class="pick" @click=${()=>u({screen:`profiles`,edit:e.name})}><span class="muted">${t}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(e=>p`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${d?this.renderMonitor(d):f}
      ${this.open?p`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:f}
      ${this.openEvent?p`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:f}
    `}};d([y({attribute:!1})],P.prototype,`ids`,void 0),d([y({attribute:!1})],P.prototype,`tiles`,void 0),d([y({attribute:!1})],P.prototype,`active`,void 0),d([o()],P.prototype,`stats`,void 0),d([o()],P.prototype,`feed`,void 0),d([o()],P.prototype,`error`,void 0),d([o()],P.prototype,`open`,void 0),d([o()],P.prototype,`openEvent`,void 0),d([o()],P.prototype,`paused`,void 0),P=d([a(`qse-cond-overview`)],P);var lt=[`off`,`low`,`medium`,`high`,`xhigh`],ut=[`small`,`medium`,`large`],F=class extends r{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new w(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),A.doc(`packages`).then(e=>this.packages=Ae(e.doc)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(T(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await A.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=t(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return m(8);if(e.loadError)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=t.classes.map(e=>e.id);return p`
      ${this.renderTest(n)}
      <details class="sec" open data-sec="classes">
        <summary>Task classes <span class="sec-sub">${t.classes.length} classes; a classifier sees only the descriptions</span></summary>
        <div class="sec-body">
          ${t.classes.map((e,r)=>this.renderClass(e,r,t.classes.length,n))}
          <button class="btn" data-add-class @click=${()=>this.addClass()}>${x(`plus`)} Add a class</button>
        </div>
      </details>
      <details class="sec" data-sec="classifier">
        <summary>Classifier stages <span class="sec-sub">asked in order until one answers at its threshold</span></summary>
        <div class="sec-body">
          ${t.classifier.stages.map((e,n)=>p`<div class="row-card" data-stage=${e.kind}>
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
                ${[`keep`,`classify`,...n].map(e=>p`<option value=${e} ?selected=${e===t.structural.compaction}>${e===`keep`?`keep the session class`:e===`classify`?`ask a classifier`:`class ${e}`}</option>`)}
              </select><span class="hint">the turn itself; the session keeps its class after it</span></label>
            <label class="field"><span>a turn that only carries tool results</span>
              <select @change=${e=>this.upd([`structural`,`tool_turn`],e.target.value)}>
                <option value="keep" ?selected=${t.structural.tool_turn===`keep`}>keep the session class (no model call)</option>
                <option value="classify" ?selected=${t.structural.tool_turn===`classify`}>ask a classifier every turn</option>
              </select></label>
            <label class="field" data-tool-turn-effort><span>effort of a tool-result turn</span>
              <select aria-label="effort of a tool-result turn" @change=${e=>{let t=e.target.value;this.upd([`structural`,`tool_turn_effort`],t===`inherit`?void 0:t)}}>
                ${Re.map(e=>p`<option value=${e} ?selected=${e===(t.structural.tool_turn_effort??`inherit`)}>${e===`inherit`?`the class's (inherit)`:me(e,t,null)}</option>`)}
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
          ${lt.map(e=>p`<div class="row-card" data-effort=${e}>
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
    `}renderTest(e){let t=this.result;return p`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?p`<span class="err-ink small" role="alert">${this.testError}</span>`:f}
      </div>
      ${t?t.active?p`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${D(t.class_id??null,e)} ${Pe(t.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${Qe(t.source??null)}${t.kept?p` <span class="badge badge-warn">${t.kept}</span>`:f} <span class="muted num">${n(t.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${t.effort??`unchanged`} · thinking ${t.thinking??`on`}${t.budget==null?``:` · budget ${t.budget}`}</dd></div>
                  ${t.extra?.size?p`<div data-test-size><dt>task size</dt><dd>${se(t.extra.size)}</dd></div>`:f}
                  <div><dt>profile</dt><dd>${t.profile} (${t.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(t.stages??[]).map(e=>p`<div class="stage ${e.error?`is-error`:e.below?``:`is-accepted`}"><strong>${pe(e.stage)}</strong><span>${e.error??p`${e.label} ${Pe(e.confidence,e.accept??.6)}`}</span><span class="num muted">${n(e.ms)}</span></div>`)}</div>
              </div>
              <div>${t.note?p`<p class="muted small">The model would see this at the end of the last turn (${(t.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${t.note}</pre>`:p`<p class="muted small">No note: ${t.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:p`<p class="muted small" style="margin-top:12px">Profile ${t.profile} is raw (${t.mode}): the conductor does nothing for it.</p>`:f}
    </section>`}renderClass(e,t,n,r){let i=e.policy,a=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return p`<details class="row-card" data-class-row=${e.id} ?open=${a}>
      <summary>
        <span>${D(e.id,r,e.label)} <span class="muted small mono">${i.effort_by_size?ut.map(e=>ot(i,e)).join(`/`):i.effort??`medium`}${(i.packages??[]).length?` · ${(i.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===0} @click=${()=>this.moveClass(t,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${t===n-1} @click=${()=>this.moveClass(t,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n<=1} @click=${()=>this.removeClass(t)} aria-label=${`remove ${e.id}`}>${x(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${e.id} ?readonly=${!a} @input=${e=>this.upd([`classes`,t,`id`],e.target.value)} data-class-id /><span class="hint">${a?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${e.label} @input=${e=>this.upd([`classes`,t,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${e.description} @input=${e=>this.upd([`classes`,t,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${j(lt,i.effort??`medium`,e=>this.upd([`classes`,t,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div class="toolbar size-map" style="margin:0" data-size-map=${e.id}><span class="muted small">by task size (the effort above is the ceiling)</span>
        ${ut.map(n=>p`<label class="field"><span>${n}</span><select aria-label=${`${e.id} effort when ${n}`} @change=${e=>this.setSizeLevel(t,i.effort_by_size,n,e.target.value)}>
            <option value="" ?selected=${!i.effort_by_size?.[n]}>the class's</option>
            ${lt.map(e=>p`<option value=${e} ?selected=${i.effort_by_size?.[n]===e}>${e}</option>`)}
          </select>${i.effort_by_size?.[n]&&ot(i,n)!==i.effort_by_size[n]?p`<span class="hint">cut to ${ot(i,n)}</span>`:f}</label>`)}
      </div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let n=(i.packages??[]).includes(e);return p`<button class="pick ${n?`is-on`:``}" aria-pressed=${n?`true`:`false`} @click=${()=>this.upd([`classes`,t,`policy`,`packages`],n?(i.packages??[]).filter(t=>t!==e):[...i.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(i.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,t,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}renderSize(e){let t=(t,n)=>this.upd([`effort`,`size`],T(e,t,n));return p`<div class="row-card" data-size-config>
      <div class="row-card-head"><strong>Task size</strong>
        <label class="switch"><input type="checkbox" .checked=${e.reset_on_compaction} @change=${e=>t([`reset_on_compaction`],e.target.checked)} /> a compaction starts the size over</label></div>
      <p class="muted small">Each signal is small below its first number, medium up to its second, large above. The task's size is the largest any counting signal reached, and it never falls within a task. A class's levels per size (in its row above) pick the effort under the class's effort; whether that acts is each profile's <em>effort by task size</em> (off, shadow, on).</p>
      ${et.map(n=>{let r=e.signals[n.id]??Ge.signals[n.id];return p`<div class="size-signal" data-size-signal=${n.id}>
          <label class="switch"><input type="checkbox" .checked=${r.enabled} @change=${e=>t([`signals`,n.id,`enabled`],e.target.checked)} /> ${n.label}</label>
          <div class="form-grid">
            ${this.num(`small below (${n.unit})`,r.small,e=>t([`signals`,n.id],{...r,small:e}),{min:0,step:n.id===`session_tokens`?1e3:1})}
            ${this.num(`medium up to (${n.unit})`,r.medium,e=>t([`signals`,n.id],{...r,medium:e}),{min:0,step:n.id===`session_tokens`?1e3:1,hint:n.hint})}
          </div>
        </div>`})}
    </div>`}setSizeLevel(e,t,n,r){let i={...t??{}};r?i[n]=r:delete i[n],this.upd([`classes`,e,`policy`,`effort_by_size`],Object.keys(i).length?i:void 0)}renderContextStage(e){let t=tt(e),n=(n,r)=>this.upd([`classifier`,`context_stage`],{...t,...e.classifier.context_stage??{},[n]:r});return p`<div class="row-card" data-context-stage>
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
    </div>`}num(e,t,n,r={}){return p`<label class="field"><span>${e}</span><input type="number" .value=${String(t)} step=${r.step??1} min=${r.min??``} max=${r.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&n(t)}} />${r.hint?p`<span class="hint">${r.hint}</span>`:f}</label>`}text(e,t,n){return p`<label class="field"><span>${e}</span><input .value=${t} @input=${e=>n(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(T(n,[`classifier`,`stages`],r))}};d([y({attribute:!1})],F.prototype,`ids`,void 0),d([y({attribute:!1})],F.prototype,`meta`,void 0),d([y({attribute:!1})],F.prototype,`active`,void 0),d([y({attribute:!1})],F.prototype,`onSaved`,void 0),d([o()],F.prototype,`packages`,void 0),d([o()],F.prototype,`testText`,void 0),d([o()],F.prototype,`testProfile`,void 0),d([o()],F.prototype,`testTools`,void 0),d([o()],F.prototype,`testing`,void 0),d([o()],F.prototype,`result`,void 0),d([o()],F.prototype,`testError`,void 0),F=d([a(`qse-cond-routing`)],F);var dt=[`off`,`shadow`,`act`],ft=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,pt=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,mt=class extends r{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new w(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),A.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(T(this.ed.draft,e,t))}render(){let e=this.ed,t=this.config?.server.monitor_mode;return p`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${e.loading&&!e.draft?m(8):e.loadError?p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${e.loadError}</div></div>`:p`
            ${t===`raw`?p`<div class="guard-line" data-monitor-off>${x(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:f}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(e.draft)}
            ${this.renderRest(e.draft)}
            ${k(e)}
            ${M(e,`the monitor`)}`}`}renderTasks(e){let t=We(e),n=t.filter(t=>e.tasks[t]&&/^C\d/.test(t)),r=t.filter(e=>Ce(e)),i=t.filter(e=>!n.includes(e)&&!r.includes(e)),a=r.map(t=>e.tasks[t]).find(e=>e.rule===`stall_watchdog`)?.notify;return p`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${n.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${n.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${r.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${r.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${a?.enabled?p`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:p`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, the owner's ruling of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${r.map(t=>this.renderTask(t,e.tasks[t]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${i.join(`, `)}</span></summary>
        <div class="sec-body">${i.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>`}renderTask(e,t){let n=be(this.meta?.schemas?.monitor,t.rule),r=this.meta?.tasks?.[e];return p`<div class="row-card" data-monitor-task=${e}>
      <div class="row-card-head">
        <strong>${e}${r?p` · ${r.label}`:f} <span class="muted small mono">${t.rule}</span></strong>
        ${j(dt,t.mode,t=>this.upd([`tasks`,e,`mode`],t),{small:!0,label:`${e} mode without a profile`})}
      </div>
      ${r?.why?p`<p class="muted small">${r.why}</p>`:f}
      ${n?.properties?p`<div class="form-stack">${Object.entries(n.properties).map(([n,r])=>this.field(n,r,t[n],[`tasks`,e,n]))}</div>`:p`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,t,[`tasks`,e])}`}
    </div>`}field(e,t,n,r){let i=Array.isArray(t.type)?t.type[0]:t.type,a=e.replace(/_/g,` `),o=t.description?p`<span class="hint">${t.description}</span>`:f;if(t.enum)return p`<label class="field"><span>${a}</span><select data-param=${e} @change=${e=>this.upd(r,e.target.value)}>${t.enum.map(e=>p`<option value=${String(e)} ?selected=${e===n}>${String(e)}</option>`)}</select>${o}</label>`;if(i===`boolean`)return p`<label class="switch" data-param=${e}><input type="checkbox" .checked=${!!n} @change=${e=>this.upd(r,e.target.checked)} /> ${a}</label>${t.description?p`<p class="hint">${t.description}</p>`:f}`;if(i===`integer`||i===`number`)return p`<label class="field"><span>${a}</span><input type="number" data-param=${e} step=${i===`integer`?`1`:`0.01`} min=${t.minimum??f} max=${t.maximum??f} .value=${n==null?``:String(n)} @input=${e=>{let t=e.target.value;this.upd(r,t===``?null:i===`integer`?Math.round(Number(t)):Number(t))}} />${o}</label>`;if(i===`string`){let t=String(n??``);return ft.test(e)||t.length>60?p`<label class="field field-block"><span>${a}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(t.length/90)))} class=${pt.test(e)?`mono`:``} data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)}></textarea>${o}</label>`:p`<label class="field"><span>${a}</span><input data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)} />${o}</label>`}if(i===`array`&&t.items&&(Array.isArray(t.items.type)?t.items.type[0]:t.items.type)===`string`){let i=!Array.isArray(n)&&Array.isArray(t.default),s=Array.isArray(n)?n:i?t.default:[];return p`<label class="field field-block"><span>${a} <span class="muted small">(one a line${i?`; the default, not saved yet`:``})</span></span><textarea rows=${Math.min(8,Math.max(2,s.length+1))} data-param=${e} .value=${s.join(`
`)} @input=${e=>this.upd(r,le(e.target.value))}></textarea>${o}</label>`}if(i===`array`&&t.items?.properties){let i=Array.isArray(n)?n:[],s=t.items;return p`<fieldset class="fieldset" data-param=${e}><legend>${a} <span class="muted small">${i.length}</span></legend>${o}
        ${i.map((e,t)=>p`<div class="row-card row-card-sub" data-item=${String(e.id??t)}>
          <div class="row-card-head"><strong class="mono small">${String(e.id??`#${t+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(r,i.filter((e,n)=>n!==t))} aria-label=${`remove ${String(e.id??t+1)}`}>${x(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(s.properties).map(([n,i])=>this.field(n,i,e[n],[...r,t,n]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${e} @click=${()=>this.upd(r,[...i,ht(s)])}>${x(`plus`)} Add</button>
      </fieldset>`}if(i===`object`&&t.properties){let i=n??{};return p`<fieldset class="fieldset" data-param=${e}><legend>${a}</legend>${o}
        <div class="form-stack">${Object.entries(t.properties).map(([e,n])=>i[e]===void 0&&!(t.required??[]).includes(e)?p`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...r,e],ht(n))}>${x(`plus`)} ${e.replace(/_/g,` `)}</button>`:this.field(e,n,i[e],[...r,e]))}</div></fieldset>`}return this.jsonField(a,n,r)}jsonField(e,t,n){return p`<label class="field field-block"><span>${e} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(t,null,2)} @change=${t=>{try{let e=JSON.parse(t.target.value);if(n.length===2&&n[0]===`tasks`){let t=this.ed.draft.tasks[n[1]];this.upd(n,{...e,rule:t.rule,mode:t.mode})}else this.upd(n,e)}catch{this.ed.errors=[`${e}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(e){return p`
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
      </details>`}};d([y({attribute:!1})],mt.prototype,`meta`,void 0),d([y({attribute:!1})],mt.prototype,`onSaved`,void 0),d([o()],mt.prototype,`config`,void 0),mt=d([a(`qse-cond-monitor`)],mt);function ht(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,ht(t)])):``}var gt=class extends r{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=h(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await A.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}render(){let e=this.sessions;return C(p`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!e?p`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?m(2):e.length?p`<div class="form-stack" data-watch>${e.map(e=>this.row(e))}</div>`:p`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(e){let t=e.gates.some(e=>e.noted);return p`<div class="row-card" data-watch-session=${e.session}>
      <div class="row-card-head">
        <strong class="mono small">${e.session}</strong>
        <span class="toolbar" style="margin:0">
          ${e.watch?p`<span class="badge ${e.watch.mode===`act`?`badge-on`:`badge-shadow`}">${e.watch.task} ${e.watch.mode}</span>`:f}
          ${e.client_gone?p`<span class="badge badge-warn" data-client-gone>client gone</span>`:e.running?p`<span class="badge badge-on">running</span>`:f}
          ${e.watch?.notify?p`<span class="badge badge-warn" data-notify>push on</span>`:f}
          ${t?p`<span class="badge badge-warn">gate noted</span>`:f}</span>
      </div>
      <p class="small" style="margin:4px 0">${De(e)}</p>
      <p class="muted small num" style="margin:0">${e.watch?`${e.watch.profile||`—`} · ${e.watch.client||`—`} · `:``}${e.requests} requests · last ${_(new Date(e.last_seen*1e3).toISOString())}${e.note_at?` · last note ${_(new Date(e.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};d([o()],gt.prototype,`sessions`,void 0),d([o()],gt.prototype,`monitor`,void 0),d([o()],gt.prototype,`error`,void 0),gt=d([a(`qse-cond-watch`)],gt);var I=class extends r{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=h(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await A.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}render(){let e=this.events,t=[...new Set((e??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!t.includes(this.task)&&t.push(this.task),p`${C(p`Monitor journal ${this.paused?p`<span class="badge">paused</span>`:p`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!e?p`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?m(4):e.length?p`<div class="feed" data-journal>${e.map(e=>Le(e,e=>this.open=e))}</div>`:p`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:p`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${t.map(e=>p`<option value=${e} ?selected=${e===this.task}>${e}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${x(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?p`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:f}`}};d([y({attribute:!1})],I.prototype,`session`,void 0),d([o()],I.prototype,`events`,void 0),d([o()],I.prototype,`monitor`,void 0),d([o()],I.prototype,`error`,void 0),d([o()],I.prototype,`task`,void 0),d([o()],I.prototype,`kind`,void 0),d([o()],I.prototype,`paused`,void 0),d([o()],I.prototype,`open`,void 0),I=d([a(`qse-cond-journal`)],I);var _t=[`off`,`low`,`medium`,`high`,`xhigh`],vt={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},yt=new Set([`S1`,`S4`]),bt=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],L=class extends r{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleContract=null,this.roleTest={},this.ed=null,this.roles=new w(this,`roles`),this.residency=new w(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),A.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),A.doc(`packages`).then(e=>this.packages=Ae(e.doc)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new w(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{let e=await A.roles();this.roleHealth=e.health,this.roleContract=e.contract??null}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let t=await A.roleTest(e);this.roleTest={...this.roleTest,[e]:t.ok?`ok in ${n(t.ms)}: "${t.answer?.label}" at ${t.answer?.confidence.toFixed(2)}`:`failed: ${t.error}`}}catch(n){this.roleTest={...this.roleTest,[e]:t(n).message}}this.loadHealth()}render(){return p`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>p`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>u({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?p`<span class="badge badge-lock">${x(`warn`)} locked</span>`:p`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${_(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):p`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(T(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=T(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=T(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderUi(e,t){let n=e.ui??{states:`off`},r=n.kinds??{},i=t||n.states!==`on`,a=we(n),o=Ue(n),s=(e,t,r,a,s)=>p`<label class="field"><span>${t}</span><input type="number" inputmode="numeric" min=${r} max=${a} step=${s} ?disabled=${i||o.mode!==`on`} data-ui-progress=${e} .value=${String(o[e])}
        @input=${t=>{let i=Math.round(Number(t.target.value));Number.isFinite(i)&&i>=r&&i<=a&&this.upd([`ui`],{...n,progress:{...n.progress??{},mode:o.mode,[e]:i}})}} /></label>`;return p`<details class="sec" data-sec="ui">
      <summary>What the user sees <span class="sec-sub">TandemLLM's own steps as lines the client shows</span> <span class="badge ${n.states===`on`?`badge-on`:``}" data-ui-badge>${n.states}</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0"><span class="muted small">state lines</span>${j([`off`,`on`],n.states,e=>this.upd([`ui`],{...n,states:e}),{disabled:t,small:!0,label:`state lines`})}</div>
        <p class="muted small">Each step goes out at the moment it happens as one line, <code>› deciding: code_edit, high effort</code>. The server takes them out of the reasoning and the content a client sends back, so Qwen never sees them; they are not tokens and not in the usage. Only in mode on.</p>
        <div class="toolbar" style="margin:0" data-ui-channel><span class="muted small">where</span>${j([`reasoning`,`content`,`inline`],a,e=>this.upd([`ui`],{...n,states_channel:e}),{disabled:i,small:!0,label:`where the lines go`})}</div>
        <p class="muted small">${a===`inline`?p`Each line opens a thinking block of its own and is its title, in time order: opencode shows <code>Thought: › thinking 2 min, 4,096 tokens</code> between Qwen's thinking blocks, and the answer after them. Needs <code>reasoning_content</code>; with <code>&lt;think&gt;</code> in the content (Hermes) the lines go as in the content.`:a===`content`?p`In the content, outside the thinking: before the reasoning, between its parts, after it. opencode shows them in its text part; with <code>&lt;think&gt;</code> in the content (Hermes) a line said inside the block goes out right after <code>&lt;/think&gt;</code>.`:p`In the reasoning opencode and Hermes show (a <code>reasoning_content</code> delta, or inside <code>&lt;think&gt;</code>). A profile saved before this choice existed keeps them here.`}</p>
        <div class="pkg-picks" data-ui-kinds>${Fe.map(({id:e,example:t})=>{let a=r[e]!==!1;return p`<button class="pick ${a?`is-on`:``}" data-ui-kind=${e} title=${t} aria-pressed=${a?`true`:`false`} ?disabled=${i} @click=${()=>this.upd([`ui`],{...n,kinds:{...r,[e]:!a}})}>${e}</button>`})}</div>
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
    </details>`}renderToolTurn(e,t){let n=e.thinking.tool_turn_effort,r=Xe(this.router,null).level,i=Xe(this.router,e);return p`<label class="field ${n?``:`is-inherited`}" data-tool-turn-effort><span>effort of a tool-result turn</span>
      <select aria-label="effort of a tool-result turn" ?disabled=${t} @change=${e=>{let t=e.target.value;if(!this.ed?.draft)return;let n={...this.ed.draft.thinking,tool_turn_effort:t};t||delete n.tool_turn_effort,this.ed.set(T(this.ed.draft,[`thinking`],n))}}>
        <option value="" ?selected=${!n}>Routing's: ${r===`inherit`?`the class's`:r}</option>
        ${Re.map(e=>p`<option value=${e} ?selected=${n===e}>${e===`inherit`?`the class's (inherit)`:e}</option>`)}
      </select>
      <span class="hint">${me(i.level,this.router,e)}${i.source===`profile`?` (this profile)`:i.source===`router`?` (Routing)`:``}: a turn whose newest message is a tool result thinks at this level, its switch and cap; the first turn and every new user message keep the class's</span></label>`}renderBudgets(e,t){return p`<div data-budgets>
      <div class="muted small">Thinking budget per level <span class="muted">(reasoning tokens; empty keeps Routing's level, 0 is no cap below the backstop)</span></div>
      <div class="form-grid">
        ${$e.map(n=>{let r=e.thinking.budgets?.[n],i=ue(n,this.router,null).cap,a=ue(n,this.router,e);return p`<label class="field ${r==null?`is-inherited`:``}" data-budget=${n}><span class="mono">${n}</span>
            <input type="number" inputmode="numeric" min="0" max=${qe} step="1024" ?disabled=${t} placeholder=${i==null?``:String(i)} .value=${r==null?``:String(r)}
              @change=${e=>{let t=e.target.value.trim();this.ed?.draft&&this.ed.set(T(this.ed.draft,[`thinking`],xe(this.ed.draft.thinking,n,t===``?null:Number(t))))}} />
            <span class="hint">${Ve(a.cap)}${a.source===`router`?` (Routing)`:a.source===`profile`?` (this profile)`:``}</span></label>`})}
      </div>
    </div>`}renderEditor(e){if(e.loading&&!e.draft)return m(8);if(e.loadError)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=!!e.locked,r=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],i=this.router?.classes??[];return p`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${x(`playground`)} Try in Playground</a></div>
      ${n?p`<div class="note" data-lock-reason>${x(`warn`)} ${e.locked}. Every control below is read-only.</div>`:f}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${t.description} ?disabled=${n} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${j([`raw`,`shadow`,`on`],t.mode,e=>this.upd([`mode`],e),{disabled:n,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${r.map(e=>{let r=t.clients.includes(e);return p`<button class="pick ${r?`is-on`:``}" ?disabled=${n} aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`clients`],r?t.clients.filter(t=>t!==e):[...t.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${n} @change=${e=>this.upd([`default_class`],e.target.value)}>${i.map(e=>p`<option value=${e.id} ?selected=${e.id===t.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(vt).map(e=>p`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${j([`off`,`shadow`,`on`],t.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:n,small:!0,label:e})}<span class="small">${vt[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${i.map(e=>{let r=t.routing[e.id]??{},i=Object.keys(r).length>0;return p`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${D(e.id,this.ids,e.label)}
                ${i?p`<button class="btn btn-ghost btn-sm" ?disabled=${n} @click=${()=>{let n={...t.routing};delete n[e.id],this.upd([`routing`],n)}}>use the router's</button>`:p`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${j(_t,r.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...r,effort:t}),{disabled:n,small:!0,label:`effort override of ${e.id}`,cls:r.effort?``:`is-inherited`})}</div>
              <div class="pkg-picks">${this.packages.map(t=>{let i=r.packages??e.policy.packages??[],a=i.includes(t);return p`<button class="pick ${a?`is-on`:``}" ?disabled=${n} @click=${()=>this.upd([`routing`,e.id],{...r,packages:a?i.filter(e=>e!==t):[...i,t]})}>${t}</button>`})}</div>
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
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${j(_t,t.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:n,small:!0,label:`default effort`})}</div>
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
          ${bt.map(([e,r,i])=>p`<label class="switch" data-pack=${e}><input type="checkbox" ?disabled=${n} .checked=${t.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${i} <span class="muted small">(${r}, ${t.toolgate[e]?t.tasks[r]?.mode??`off`:`off`})</span></label>`)}
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
        <p class="muted small" data-notify-rule>Allowed by the owner on 2026-09-30: off by default, on in the overnight profile. Only S4 in act pushes, never in shadow, once a turn end; the token is a file on the box. S2 (re-read after a compaction) has no number here: its texts and state files are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
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
    </div>`}renderTask(e,t,n,r,i){let a=this.meta?.tasks[e],o=a?.role&&a.tier===`1`,s=!o||!!this.roleHealth[a.role]?.up;return p`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${a?.family??``}</div></div>
      <div>${a?.label??e}<div class="task-why">${a?.why??``}</div>${o&&!s?p`<div class="warn-ink small">${x(`warn`)} the ${a.role} role is offline: act is refused</div>`:f}</div>
      <div class="task-mode">${j([`off`,`shadow`,`act`],t,t=>this.upd([`tasks`,e,`mode`],t),{disabled:i,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${yt.has(e)&&Object.keys(r).length?p`<p class="muted small" data-in-context=${e}>${Object.entries(r).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:f}
        ${Object.entries(yt.has(e)?{}:r).map(([t,n])=>{let r=a?.thresholds[t];return p`<label class="field ${r&&r.default!==n?`is-changed`:``}"><span>${t.replace(/_/g,` `)} <span class="hint">${r?.unit??``}${r?`, default ${r.default}`:``}</span></span>
            <input type="number" ?disabled=${i} .value=${String(n)} step="any" @input=${n=>{let r=Number(n.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,t],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${i} min="0" max="10" .value=${String(n)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let e=this.roles;if(!e.draft)return e.loadError?p`<p class="err-ink small">${e.loadError}</p>`:f;let t=e.draft,r=(n,r)=>e.set(T(t,n,r)),i=at(e.base?.meta.defaults),a=this.roleContract?.engine_tokenizer_sha256??null,o=this.meta?.roles,s=o?.kinds??je,c=o?.windows??Je,l=o?.fallbacks??[`PASS`,`stale`,`skip`,`next-role`],u=Ze.filter(e=>!o||o.writes.includes(e));return p`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${e.base?.meta.defaults?.length?p`<div class="note" data-role-defaults>${x(`warn`)} Version ${e.base.version} was stored before the role contract and loaded with ${e.base.meta.defaults.length} default${e.base.meta.defaults.length===1?``:`s`} (marked <span class="badge">default</span> below; the journal has the list). Save the roles to make them explicit.</div>`:f}
        <p class="muted small">The contract (SRV-133): when a role may use the GPU, whether a Qwen request cuts it, how long its caller waits before the fallback, what it may write, and the tokenizer it reads, which decides what it may share with Qwen. Engine tokenizer: <code class="mono" data-engine-sha>${a?`${a.slice(0,12)}…`:`not recorded`}</code></p>
        ${Object.entries(t).map(([e,t])=>{let o=this.roleHealth[e],d=new Set((i[e]??[]).map(e=>e.split(`=`)[0])),m=t=>d.has(t)?p` <span class="badge" data-default=${`${e}.${t}`}>default</span>`:f,h=ke(t.kind,t.window),ee=ae(t.kind,t.writes),te=ge(t,a);return p`<div class="row-card" data-role-row=${e}>
            <div class="row-card-head"><strong>${e}</strong>
              <span class="toolbar" style="margin:0">${o?p`<span class="badge ${o.up?`badge-ok`:t.mode===`off`?``:`badge-warn`}">${o.up?`up ${n(o.ms)}`:t.mode===`off`?`off`:`down: ${o.error??`?`}`}</span>`:f}
                <button class="btn btn-sm" @click=${()=>this.testRole(e)} data-role-test=${e}>Test</button></span></div>
            ${this.roleTest[e]?p`<p class="small" role="status">${this.roleTest[e]}</p>`:f}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${t.endpoint} @input=${t=>r([e,`endpoint`],t.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${t.model} @input=${t=>r([e,`model`],t.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(t.timeout_ms)} @input=${t=>r([e,`timeout_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>memory claim (GiB, its peak)${m(`memory_claim_gib`)}</span><input type="number" step="0.1" min="0" data-role-field="memory_claim_gib" .value=${String(t.memory_claim_gib)} @input=${t=>r([e,`memory_claim_gib`],Number(t.target.value))} /></label>
            </div>
            <div class="form-grid" data-role-contract=${e}>
              <label class="field"><span>kind${m(`kind`)}</span><select data-role-field="kind" @change=${t=>r([e,`kind`],t.target.value)}>${s.map(e=>p`<option value=${e} ?selected=${e===t.kind}>${e}</option>`)}</select></label>
              <label class="field"><span>window${m(`window`)}</span><select data-role-field="window" @change=${t=>r([e,`window`],t.target.value)}>${c.map(e=>p`<option value=${e} ?selected=${e===t.window} ?disabled=${e!==t.window&&!!ke(t.kind,e)} title=${Ye[e]}>${e}</option>`)}</select></label>
              <label class="field"><span>latency budget (ms)${m(`latency_budget_ms`)}</span><input type="number" min="1" max="600000" data-role-field="latency_budget_ms" .value=${String(t.latency_budget_ms)} @input=${t=>r([e,`latency_budget_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>fallback (off, slow, unsure or refused)${m(`fallback`)}</span><select data-role-field="fallback" @change=${t=>r([e,`fallback`],t.target.value)}>${l.map(e=>p`<option value=${e} ?selected=${e===t.fallback}>${e}</option>`)}</select></label>
              <label class="field"><span>writes${m(`writes`)}</span><select data-role-field="writes" @change=${t=>r([e,`writes`],t.target.value)}>${u.map(e=>p`<option value=${e} ?selected=${e===t.writes} ?disabled=${e!==t.writes&&!!ae(t.kind,e)}>${e.replace(/_/g,` `)}</option>`)}</select></label>
              <label class="field field-wide"><span>tokenizer sha256 (the engine records its own; none: text only)${m(`tokenizer_sha256`)}</span><input class="mono" data-role-field="tokenizer_sha256" spellcheck="false" .value=${t.tokenizer_sha256} @input=${t=>r([e,`tokenizer_sha256`],t.target.value.trim().toLowerCase()||`none`)} /></label>
            </div>
            <div class="toolbar" style="margin:0">
              <label class="switch"><input type="checkbox" data-role-field="preemptible" .checked=${t.preemptible} @change=${t=>r([e,`preemptible`],t.target.checked)} /> preemptible: a Qwen request cuts it, it resumes later${m(`preemptible`)}</label>
            </div>
            <div><span class="muted small">sharing with Qwen${m(`sharing`)}</span>
              <div class="pkg-picks" data-role-sharing=${e}>${te.map(n=>{let i=t.sharing.includes(n.form);return p`<button class="pick ${i?`is-on`:``}" aria-pressed=${i?`true`:`false`} ?disabled=${!i&&!n.allowed} title=${n.why??``} data-share=${n.form} @click=${()=>r([e,`sharing`],i?t.sharing.filter(e=>e!==n.form):[...t.sharing,n.form])}>${n.label}</button>`})}</div>
              ${te.filter(e=>!e.allowed).map(e=>p`<p class="muted small" data-share-why=${e.form}>${e.label}: ${e.why}</p>`)}
            </div>
            ${h?p`<p class="warn-ink small" data-role-why="window">${x(`warn`)} ${h}</p>`:f}
            ${ee?p`<p class="warn-ink small" data-role-why="writes">${x(`warn`)} ${ee}</p>`:f}
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${j([`off`,`shadow`,`on`],t.mode,t=>r([e,`mode`],t),{small:!0,label:`${e} mode`})}<span class="muted small mono">${t.contract}</span></div>
          </div>`})}
        ${k(e)}
        ${M(e,`the roles`)}
      </div>
    </details>`}renderResidency(){let e=this.residency;if(!e.draft)return e.loadError?p`<p class="err-ink small">${e.loadError}</p>`:f;let t=e.draft,n=(n,r)=>e.set(T(t,n,r)),r=(e,r=`0.1`,i=!1)=>p`<input type="number" step=${r} min="0" .value=${String(t[e[0]])} @input=${t=>{let r=Number(t.target.value);n(e,i?Math.round(r):r)}} data-res=${e[0]} />`,i=[`measured`,`peak`,`declared`];return p`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${r([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${r([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${r([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${t.order.join(`, `)} @change=${e=>n([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(t.claims).map(([e,r])=>p`<div class="row-card" data-claim=${e}>
            <div class="row-card-head"><strong>${e}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[e]:r,...i}=t.claims;n([`claims`],i)}} aria-label=${`remove the claim of ${e}`}>${x(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(r.gib)} @input=${t=>n([`claims`,e,`gib`],Number(t.target.value))} data-claim-gib=${e} /></label>
              <label class="field"><span>basis</span><select @change=${t=>n([`claims`,e,`basis`],t.target.value)}>${i.map(e=>p`<option value=${e} ?selected=${e===r.basis}>${e}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${r.note} @input=${t=>n([`claims`,e,`note`],t.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in t.claims} @click=${()=>{n([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${x(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${k(e)}
        ${M(e,`the residency limits`)}
      </div>
    </details>`}};d([y({attribute:!1})],L.prototype,`ids`,void 0),d([y({attribute:!1})],L.prototype,`meta`,void 0),d([y({attribute:!1})],L.prototype,`tiles`,void 0),d([y({attribute:!1})],L.prototype,`selected`,void 0),d([y({attribute:!1})],L.prototype,`onSaved`,void 0),d([o()],L.prototype,`router`,void 0),d([o()],L.prototype,`packages`,void 0),d([o()],L.prototype,`roleHealth`,void 0),d([o()],L.prototype,`roleContract`,void 0),d([o()],L.prototype,`roleTest`,void 0),d([o()],L.prototype,`newClaim`,void 0),L=d([a(`qse-cond-profiles`)],L);var R=[`package`,`task_skill`,`test_skill`,`card`,`library`],z={package:`package`,task_skill:`task skill`,test_skill:`test skill`,card:`card`,library:`library skill`},xt=3.5,St={library:350,library_card:200},Ct={q:``,category:`all`,cls:``,enabled:`all`,over:!1};function B(e){return Math.floor(e.length/xt)}function wt(e,t){return t?t.category:e.library?`library`:e.kind===`skill`?`task_skill`:`package`}function Tt(e,t,n,r){let i=r?.limits??St,a=n===`library`?i.library:e===(r?.library.card??`tandem-library`)?i.library_card:null;return a!=null&&a<=t.max_tokens?{limit:a,source:`house`}:{limit:t.max_tokens,source:`max_tokens`}}function Et(e,t){return(e?.classes??[]).filter(e=>(e.policy.packages??[]).includes(t)).map(e=>e.id)}function Dt(e,t,n,r){return Object.entries(e?.packages??{}).map(([e,i])=>{let a=t?.packages[e]??null;return{name:e,pkg:i,entry:a,category:wt(i,a),classes:Et(r,e),usage:n?.[e]??null,score:null}}).sort((e,t)=>R.indexOf(e.category)-R.indexOf(t.category)||e.name.localeCompare(t.name))}function Ot(e){return e.toLowerCase().split(/\s+/).filter(Boolean)}function kt(e,t){let n=`${e.name} ${e.pkg.title} ${e.pkg.description} ${e.pkg.source}`.toLowerCase();return Ot(t).every(e=>n.includes(e))}function At(e,t){return e.entry?e.entry.over:e.pkg.source===`inline`&&B(V(e.pkg.body,!!e.pkg.library)[0])>Tt(e.name,e.pkg,e.category,t).limit}function jt(e,t,n,r=null){let i=new Map((n??[]).map(e=>[e.name,e.score])),a=t.q.trim(),o=e.filter(e=>t.category===`all`||e.category===t.category).filter(e=>!t.cls||e.classes.includes(t.cls)||e.category===`library`&&(e.entry?.fits??[]).includes(t.cls)).filter(e=>t.enabled===`all`||t.enabled===`on`===e.pkg.enabled).filter(e=>!t.over||At(e,r)).filter(e=>!a||kt(e,a)||e.category===`library`&&i.has(e.name)).map(e=>({...e,score:a&&e.category===`library`?i.get(e.name)??null:null}));return a?o.sort((e,t)=>R.indexOf(e.category)-R.indexOf(t.category)||(e.score!=null||t.score!=null)&&(t.score??-1)-(e.score??-1)||e.name.localeCompare(t.name)):o}function Mt(e,t){let n=e=>!!t?.packages[e]?.general;return[...e.filter(n),...e.filter(e=>!n(e))]}function Nt(e,t,n,r){let i=e.classes.findIndex(e=>e.id===t);if(i<0)return{router:e,error:`no class ${t}`};if(r?.packages[n]?.library)return{router:e,error:`${n} is a library skill: the router searches the library, a class does not list it`};let a=e.classes[i].policy.packages??[];if(a.includes(n))return{router:e,error:null};if(a.length>=12)return{router:e,error:`${t} already lists 12 packages, the most a class may`};let o=e.classes.map((e,t)=>t===i?{...e,policy:{...e.policy,packages:Mt([...a,n],r)}}:e);return{router:{...e,classes:o},error:null}}function Pt(e,t,n){return{...e,classes:e.classes.map(e=>e.id===t?{...e,policy:{...e.policy,packages:(e.policy.packages??[]).filter(e=>e!==n)}}:e)}}var Ft=/^## Reference\b/m;function V(e,t){let n=t?Ft.exec(e):null;return n?[e.slice(0,n.index).trim(),e.slice(n.index).trim()]:[e.trim(),``]}function It(e){let t=1;for(;e.packages[`package-${t}`];)t++;return`package-${t}`}function Lt(e){return{title:`Package ${e}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0}}function Rt(e,t,n){return{...e,source:`inline`,body:n?`${t}\n\n${n}`:t}}function zt(e,t){return t?e.filter(e=>e.profile===t):e}function Bt(e){return e===`on`?`the router puts the skill it picks into the prompt from the next request, and the library card on the coding classes with tools`:e===`shadow`?`the router searches and picks, and the journal shows the pick; nothing goes into the prompt`:`the router does not search the library`}var H=class extends r{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.catalog=null,this.usage=null,this.onChanged=()=>void 0,this.profile=``,this.busy=null,this.msg=null,this.error=null}async setMode(e,n){this.busy=e,[this.msg,this.error]=[null,null];try{let t=await A.doc(`profiles`,e),r={...t.doc,library:n},i=await A.check(`profiles`,e,r);if(i.errors.length){this.error=`${e}: ${i.errors.join(`; `)}`;return}if(!await O(`Library ${n} for ${e}?`,[`With ${n}, ${Bt(n)}.`,p`The server checked the change (live v${t.version}):${Ie(i.changes)}`,`Saving makes a new version of the profile with the note “library to ${n}”.`],`Save the new version`))return;let a=await N(()=>A.put(`profiles`,e,r,`library to ${n}`,t.version));this.msg=`${e}: library ${n}, version ${a.version}`,this.onChanged()}catch(e){this.error=t(e).message}finally{this.busy=null}}render(){let e=this.catalog;if(!e||!this.usage)return m(5);let t=this.usage.library,n=Object.values(e.packages).filter(e=>e.category===`library`).length,r=e=>t.rows.filter(t=>t.profile===e&&t.chosen.length).length,i=Object.keys(e.library.profiles).sort();return p`
      <section class="panel" data-lib-profiles>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Retrieval per profile <span class="muted">· ${n} library skills</span></h2>
          <p class="panel-sub">On a new user message, and on a low-quality or stuck outcome, the router searches the library with the message, asks the package question among the best matches and, with <strong>on</strong>, puts the skill that fits at the prompt's tail. <strong>shadow</strong> only journals the pick. Judge the picks below before a profile goes on.</p></div></header>
        ${this.error?p`<p class="err-ink small" role="alert" data-lib-error>${this.error}</p>`:f}
        ${this.msg?p`<p class="status small" role="status" data-lib-saved>${x(`check`)} ${this.msg}</p>`:f}
        <div class="lib-profiles">
          ${i.map(t=>{let n=this.tiles.find(e=>e.name===t),i=e.library.profiles[t];return p`<div class="lib-profile" data-lib-profile=${t}>
              <span class="lib-profile-name"><strong>${t}</strong>${n?.locked?p` <span class="badge badge-lock">locked</span>`:f}<span class="muted small">${n?.clients.join(`, `)||`no client`}${n?` · profile ${n.mode}`:``}</span></span>
              ${j([`off`,`shadow`,`on`],i,e=>void this.setMode(t,e),{small:!0,label:`library of ${t}`,disabled:!!n?.locked||this.busy===t})}
              <span class="muted small num">${r(t)} pick${r(t)===1?``:`s`} in the rows below</span>
            </div>`})}
        </div>
      </section>
      <section class="panel" data-lib-shadow>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">What the harness picked</h2>
          <p class="panel-sub">The router's retrieval in the journal (extra.library), newest first: the candidates the search found, the skill the package question picked, and whether it went in (on) or would have (shadow). A pick opens the skill.</p></div>
          <div class="panel-tools"><label class="field"><span>profile</span><select data-lib-filter @change=${e=>this.profile=e.target.value}>
            <option value="" ?selected=${!this.profile}>all</option>${i.map(e=>p`<option value=${e} ?selected=${this.profile===e}>${e}</option>`)}
          </select></label></div></header>
        <div class="stats" data-lib-tiles>
          ${S({label:`messages searched`,value:c(t.searched.day),unit:`in 24 h`,sub:`${c(t.searched.week)} in 7 days`})}
          ${S({label:`with a pick`,value:c(t.picked.day),unit:`in 24 h`,sub:`${c(t.picked.week)} in 7 days`})}
          ${S({label:`pick rate`,value:t.searched.week?`${Math.round(100*t.picked.week/t.searched.week)}`:`—`,unit:t.searched.week?`%`:``,sub:`7 days`})}
          ${S({label:`most picked`,value:t.top[0]?.name??`—`,sub:t.top.length?t.top.slice(0,3).map(e=>`${e.name} ${e.picks}`).join(` · `):`no pick yet`})}
        </div>
        ${this.rows(zt(t.rows,this.profile))}
      </section>`}rows(e){if(!e.length)return g(`No retrieval in the journal`,`A profile with the library in shadow or on journals each search here; with every profile off nothing is searched.`);let t=e=>u({ptab:null,pkg:e,pview:null});return p`<div class="lib-rows" data-lib-rows>${e.map(e=>p`<div class="lib-row" data-lib-row=${e.id}>
        <div class="lib-row-head">
          <span class="num small">${_(new Date(e.ts_ms).toISOString())}</span>
          <span class="badge ${e.mode===`on`?`badge-on`:`badge-shadow`}">${e.mode??`—`}</span>
          ${D(e.class_id,this.ids)}
          <span class="muted small">${e.profile??`—`} · ${e.trigger??`—`}${e.query_chars==null?``:` · ${c(e.query_chars)} characters searched`}${e.cached?` · asked before`:``}</span>
        </div>
        <div class="lib-cands">${e.candidates.length?e.candidates.map(t=>p`<span class="lib-cand ${t.name&&e.chosen.includes(t.name)?`is-chosen`:``}">${t.name}<span class="num muted">${t.score==null?``:t.score.toFixed(2)}</span></span>`):p`<span class="muted small">no candidate shares a word with the message</span>`}</div>
        <div class="lib-pick small">${e.chosen.length?p`${e.mode===`on`&&e.applied?.length?`went in:`:`would have gone in:`} ${e.chosen.map(e=>p`<button class="btn btn-sm lib-open" data-lib-open=${e} @click=${()=>t(e)}>${e}</button>`)} <span class="muted">(${e.how??`—`})</span>`:p`<span class="muted">no pick (${(e.how??`none`).replace(/_/g,` `)})</span>`}</div>
      </div>`)}</div>`}};d([y({attribute:!1})],H.prototype,`ids`,void 0),d([y({attribute:!1})],H.prototype,`tiles`,void 0),d([y({attribute:!1})],H.prototype,`catalog`,void 0),d([y({attribute:!1})],H.prototype,`usage`,void 0),d([y({attribute:!1})],H.prototype,`onChanged`,void 0),d([o()],H.prototype,`profile`,void 0),d([o()],H.prototype,`busy`,void 0),d([o()],H.prototype,`msg`,void 0),d([o()],H.prototype,`error`,void 0),H=d([a(`qse-cond-library`)],H);var Vt={mode:`off`,idle_s:180,due_idle_s:3,min_new_tokens:1500,max_tokens:600,max_blocks:4,gate:!0,after_compaction:`idle`,turn_deadline_s:30,existing:5},Ht={index_tokens:1500,body_tokens:600,pinned_tokens:300,bodies:3,min_score:0,min_words:2,dedupe_threshold:.6,intent:`shadow`,provisional:!0,digest_chars:2e3,writer:Vt},Ut={rule:`how I want things done`,ruling:`a decision that was taken`,gotcha:`what went wrong once`,runbook:`how to do a thing`,program_state:`where the work stands`,reference:`a fact to look up`,registry:`a list of things`,archive:`kept for the record`},Wt=Object.keys(Ut),U={"qwen-inline":`the model, mid-answer`,"qwen-writer":`the model, in the background`,api:`the model's script`,human:`you`},Gt={node:null,kind:``,status:``,writer:``,stale:!1,q:``};function Kt(e,t){let n=t.q.trim().toLowerCase();return e.filter(e=>(!t.node||e.node===t.node)&&(!t.kind||e.kind===t.kind)&&(!t.status||e.status===t.status)&&(!t.writer||e.writer===t.writer)&&(!t.stale||!!e.stale)&&(!n||`${e.title} ${e.tags.join(` `)} ${e.id}`.toLowerCase().includes(n)))}function qt(e){return e.scope===`user`?`About you`:e.scope===`project`?e.label??e.node.replace(/^projects\//,``):`session ${e.label??e.node.replace(/^sessions\//,``)}`}function Jt(e,t){let n=e.split(`
`),r=t.split(`
`),i=n.length,a=r.length;if(i*a>4e5)return[...n.map(e=>({op:`-`,text:e})),...r.map(e=>({op:`+`,text:e}))];let o=Array.from({length:i+1},()=>Array(a+1).fill(0));for(let e=i-1;e>=0;e--)for(let t=a-1;t>=0;t--)o[e][t]=n[e]===r[t]?o[e+1][t+1]+1:Math.max(o[e+1][t],o[e][t+1]);let s=[],c=0,l=0;for(;c<i&&l<a;)n[c]===r[l]?(s.push({op:`=`,text:n[c]}),c++,l++):o[c+1][l]>=o[c][l+1]?s.push({op:`-`,text:n[c++]}):s.push({op:`+`,text:r[l++]});for(;c<i;)s.push({op:`-`,text:n[c++]});for(;l<a;)s.push({op:`+`,text:r[l++]});return s}function Yt(e){let t=e.detail??{};if(e.task===`nightly`)return`nightly: ${t.deleted??0} archives deleted, ${t.duplicates??0} duplicates, ${t.stale??0} stale`;let n=typeof t.why==`string`?t.why:null,r=String(t.op??e.decision??`write`),i=e.task===`api_write`?`the model's script`:U[String(t.writer??`qwen-inline`)]??String(t.writer);return n||e.decision&&![`add`,`update`,`archive`,`propose`,`ok`].includes(e.decision)?`refused (${n??e.decision}): ${r} by ${i}`:`${r===`add`?t.revised?`revised`:`added`:r===`update`?`updated`:r===`archive`?`archived`:r===`propose`?`proposed`:r} ${[t.kind,t.scope?`(${t.scope})`:null].filter(Boolean).join(` `)} by ${i}${t.id?` · ${t.id}`:``}`.replace(/\s+/g,` `).trim()}function Xt(e){let t=e?.memory?.notes??{};return{...Ht,...t,writer:{...Vt,...t.writer??{}}}}function Zt(e){let t=e.detail??{},n=Array.isArray(t.ops)?t.ops:[],r=e=>n.filter(t=>t.op===e&&(e!==`add`||!t.revised)).length,i=n.filter(e=>e.revised).length,a=[r(`add`)?`${r(`add`)} added`:``,i?`${i} revised`:``,r(`update`)?`${r(`update`)} updated`:``,r(`archive`)?`${r(`archive`)} archived`:``].filter(Boolean),o=t.gate??{},s=n.length&&n.every(e=>e.stored===!1)?` (shadow: not stored)`:``;return{when:e.ts_ms,trigger:String(t.trigger??`—`),outcome:typeof t.skipped==`string`?t.skipped:n.length?`wrote`:t.none?`nothing worth keeping`:`no block`,gate:o.answer?String(o.answer):String(o.how??`—`),seconds:typeof t.ms==`number`?Math.round(t.ms/100)/10:null,tokens:typeof t.tokens==`number`?t.tokens:null,ops:(a.join(`, `)||`—`)+s,refused:Array.isArray(t.refused)?t.refused:[]}}function Qt(e,t){let n=Object.values(e).reduce((e,t)=>e+t,0);if(!n)return`not injected yet`;let r=e[t]??0;return`${n} injection${n===1?``:`s`}${r?` (${r} today)`:``}`}var $t=e=>e?_(e):`—`,W=class extends r{constructor(...e){super(...e),this.onChanged=null,this.list=null,this.open=null,this.error=null,this.flash=null,this.busy=!1,this.status=`open`}connectedCallback(){super.connectedCallback(),this.load()}async load(){try{this.list=(await N(()=>A.proposals())).proposals,this.error=null}catch(e){this.error=t(e).message}}async show(e){try{this.open=(await N(()=>A.proposal(e.name,e.n))).proposal}catch(e){this.error=t(e).message}}async apply(e){if(await O(`Apply the proposal to ${e.name}?`,[e.title,`The text becomes a new version of the packages document; every session gets it from its next note. The versions panel restores the old text.`],`Apply`)){this.busy=!0;try{let t=await N(()=>A.proposalApply(e.name,e.n));this.flash=`${e.name}: packages version ${t.version} (was ${t.previous})`,this.onChanged?.(),await this.load(),await this.show(e)}catch(e){this.error=t(e).message}finally{this.busy=!1}}}async reject(e){let n=window.prompt(`Why is this proposal rejected?`,`not what the skill needs`);if(n!==null){this.busy=!0;try{await N(()=>A.proposalReject(e.name,e.n,n||`rejected in the dashboard`)),this.flash=`${e.name}#${e.n} rejected`,await this.load(),await this.show(e)}catch(e){this.error=t(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.list)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.list)return m(4);let e=this.list.filter(e=>this.status===`all`||e.status===`open`),t=e=>p`<span class="badge ${e===`open`?`badge-shadow`:e===`applied`?`badge-ok`:``}">${e}</span>`;return p`
      ${this.flash?p`<p class="ok-ink small" role="status" data-prop-flash>${x(`check`)} ${this.flash}</p>`:f}
      ${this.error?p`<p class="err-ink small" role="alert">${x(`warn`)} ${this.error}</p>`:f}
      ${this.open?this.renderOne(this.open):f}
      ${C(`Proposals`,e.length?p`<div class="table-scroll"><table class="table" data-proposals><thead><tr><th>package</th><th>change</th><th>mode</th><th>status</th><th>evidence</th><th>proposed</th></tr></thead><tbody>${e.map(e=>p`<tr class="req-row ${this.open&&this.open.name===e.name&&this.open.n===e.n?`is-open`:``}" data-proposal=${`${e.name}#${e.n}`} @click=${()=>this.show(e)}>
                <td class="mono small">${e.name}#${e.n}</td><td class="cell-wrap">${e.title}</td><td class="small">${e.mode}</td><td>${t(e.status)}</td>
                <td class="small">${e.evidence.length?`${e.evidence.length} outcome row${e.evidence.length===1?``:`s`}`:p`<span class="muted">no evidence</span>`}</td><td class="num small">${$t(e.created)}</td></tr>`)}</tbody></table></div>`:g(this.status===`open`?`No open proposal`:`No proposal yet`,`The model proposes a change to a package with a hidden block when a session showed a gap; it waits here for your click.`),{sub:`changes to packages and skills the model proposed; none goes live without Apply`,tools:p`<label class="switch small"><input type="checkbox" .checked=${this.status===`all`} @change=${e=>this.status=e.target.checked?`all`:`open`} data-prop-all /> applied and rejected too</label>
            <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the proposals">${x(`refresh`)}</button>`,id:`proposals`})}`}renderOne(e){let t=e.live===null?[]:Jt(e.live,e.proposed);return p`<section class="panel" style="margin-bottom:16px" data-proposal-open=${`${e.name}#${e.n}`}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e.name}#${e.n}: ${e.title}</h2>
        <p class="panel-sub">${e.mode===`append`?`adds a section`:`replaces the text`} · ${e.status}${e.applied_version?` as version ${e.applied_version}`:``}${e.rejected_reason?`: ${e.rejected_reason}`:``} · ${e.evidence_word}${e.session?p` · <a class="mono" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(e.session)}`}>${e.session}</a>`:f}</p></div>
        <div class="panel-tools">
          ${e.status===`open`?p`<button class="btn btn-primary btn-sm" ?disabled=${!e.applies||this.busy} @click=${()=>this.apply(e)} data-prop-apply>Apply</button>
            <button class="btn btn-sm" ?disabled=${this.busy} @click=${()=>this.reject(e)} data-prop-reject>Reject</button>`:f}
          <button class="btn btn-ghost btn-icon" aria-label="Close the proposal" @click=${()=>this.open=null}>${x(`x`)}</button></div></header>
      ${e.reason?p`<p class="small"><span class="muted">why:</span> ${e.reason}</p>`:f}
      ${e.stale&&e.status===`open`?p`<p class="warn-ink small">${x(`warn`)} ${e.name} changed since this was proposed${e.mode===`append`?`; the section is added to the text as it is now`:`; a replacement cannot be applied`}.</p>`:f}
      ${e.live===null?p`<p class="warn-ink small">No package ${e.name} with a text any more.</p>`:p`<pre class="notes-diff" data-prop-diff>${t.map(e=>p`<span class="diff-${e.op===`+`?`add`:e.op===`-`?`del`:`eq`}">${e.op===`=`?` `:e.op} ${e.text}\n</span>`)}</pre>`}
    </section>`}};d([o()],W.prototype,`list`,void 0),d([o()],W.prototype,`open`,void 0),d([o()],W.prototype,`error`,void 0),d([o()],W.prototype,`flash`,void 0),d([o()],W.prototype,`busy`,void 0),d([o()],W.prototype,`status`,void 0),W=d([a(`qse-cond-proposals`)],W);var en=[{id:`text`,label:`Text`},{id:`edit`,label:`Edit`},{id:`classes`,label:`Classes`},{id:`usage`,label:`Usage`}],G=class extends r{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.selected=null,this.tab=null,this.pane=null,this.catalog=null,this.catError=null,this.usage=null,this.filters=Ct,this.hits=null,this.libError=null,this.item=null,this.itemError=null,this.recent=null,this.raw=!1,this.editView=`write`,this.counted=null,this.review=null,this.attachError=null,this.found=null,this.scanError=null,this.pk=new w(this,`packages`),this.rt=new w(this,`router`),this.seq=0,this.loadedFor=null,this.countTimer=null,this.countFor=null}connectedCallback(){super.connectedCallback(),this.pk.onSaved=()=>void this.refresh(),this.rt.onSaved=()=>void this.refresh(),this.pk.load(),this.rt.load(),this.refresh()}disconnectedCallback(){this.countTimer&&clearTimeout(this.countTimer),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.item=null,this.itemError=null,this.recent=null,this.attachError=null,this.selected&&this.loadItem(this.selected))}updated(e){let t=this.selected?this.pk.draft?.packages[this.selected]:null;if(!t||t.source!==`inline`)return;let n=V(t.body,!!t.library)[0];this.counted?.text!==n&&this.countFor!==n&&(this.countFor=n,this.countTimer&&clearTimeout(this.countTimer),this.countTimer=setTimeout(()=>{A.packagesTokens([n]).then(e=>this.counted={text:n,n:e.tokens[0],how:e.tokenizer}).catch(()=>void 0)},400))}async refresh(){try{let[e,t]=await Promise.all([A.packagesCatalog(),A.packagesUsage(null,1)]);[this.catalog,this.usage,this.catError]=[e,t,null]}catch(e){this.catError=t(e).message}this.selected&&await this.loadItem(this.selected)}async loadItem(e){try{let[t,n]=await Promise.all([this.pk.base?.doc.packages[e]||!this.pk.base?A.packageItem(e):Promise.resolve(null),A.packagesUsage(e,20)]);if(e!==this.selected)return;[this.item,this.recent,this.itemError]=[t,n.recent,null]}catch(n){e===this.selected&&(this.itemError=t(n).message)}}async search(e){this.filters={...this.filters,q:e};let n=++this.seq;if(!e.trim()||!Object.values(this.pk.draft?.packages??{}).some(e=>e.library)){[this.hits,this.libError]=[null,null];return}try{let t=await A.librarySearch(e.trim(),20);n===this.seq&&([this.hits,this.libError]=[t,null])}catch(e){n===this.seq&&([this.hits,this.libError]=[[],`the library's search: ${t(e).message}`])}}open(e,t=null){u({pkg:e,pview:t})}upd(e,t){this.pk.draft&&this.pk.set(T(this.pk.draft,e,t))}toggleAttach(e,t,n){let r=this.rt.draft;if(!r)return;if(!n){this.attachError=null,this.rt.set(Pt(r,e,t));return}let i=Nt(r,e,t,this.pk.draft);this.attachError=i.error,i.error||this.rt.set(i.router)}addInline(){let e=this.pk.draft,t=It(e);this.pk.set(T(e,[`packages`,t],Lt(Number(t.slice(8))))),this.open(t,`edit`)}copyInline(e){let t=this.pk.draft,n=this.item;n&&n.name===e&&n.found&&(this.pk.set(T(t,[`packages`,e],Rt(t.packages[e],n.body,n.reference))),this.open(e,`edit`))}removePackage(e){let t=this.pk.draft,n={...t.packages};delete n[e],this.pk.set({...t,packages:n}),this.open(null)}async scan(){this.scanError=null;try{this.found=(await A.scan()).skills}catch(e){this.scanError=t(e).message}}register(e){let t={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.pk.set(T(this.pk.draft,[`packages`,e.name],t)),this.open(e.name)}async startReview(e,t){await e.check(),this.review=t}async confirmSave(e){await e.save()&&(this.review=null)}bar(e,t,n){let r=e.base?.meta,i=!!e.draft&&!!e.base&&JSON.stringify(e.draft)!==JSON.stringify(e.base.doc);return p`
      ${e.showVersions?nt(e,n):f}
      ${this.review===t?p`<section class="panel pk-review" data-review=${t}>
            <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e.errors.length?p`${x(`warn`)} The server refuses ${n}`:p`${x(`check`)} The server accepts ${n}`}</h2>
              <p class="panel-sub">POST /v1/conductor/${t}/check against live v${r?.version??0}: ${e.changes.length} change${e.changes.length===1?``:`s`}. Saving makes a new version with your note; the next request uses it.</p></div></header>
            ${k(e)}
            ${Ie(e.changes)}
            <div class="toolbar pk-review-tools">
              <button class="btn btn-primary" data-review-save=${t} ?disabled=${e.errors.length>0||!e.note.trim()||e.saving||!e.changes.length} @click=${()=>this.confirmSave(e)}>${e.saving?`Saving…`:`Save as a new version`}</button>
              <button class="btn btn-ghost" @click=${()=>this.review=null}>Back to editing</button>
              ${e.note.trim()?f:p`<span class="muted small">Write the change note first.</span>`}
            </div>
          </section>`:f}
      <div class="savebar ${i||this.review===t?``:`is-idle`}" data-savebar=${t}>
        <span class="status num" title=${r?.note??``}>${n} · live v${r?.version??0}${i?p` · <strong>${e.changes.length||`…`} change${e.changes.length===1?``:`s`}</strong>`:f}</span>
        <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>{e.note=t.target.value,this.requestUpdate()}} aria-label=${`change note for ${n}`} maxlength="200" data-note=${t} />
        <button class="btn btn-primary" ?disabled=${!i||e.saving} @click=${()=>this.startReview(e,t)} data-check-save=${t}>Check and save…</button>
        <button class="btn btn-ghost" ?disabled=${!i} @click=${()=>{e.reset(),this.review=null}}>Reset</button>
        <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():this.requestUpdate()}} data-versions-btn=${t}>Versions</button>
        ${e.saved?p`<span class="status" role="status" data-saved=${t}>${x(`check`)} ${e.saved}</span>`:f}
        ${e.saveError?p`<span class="err-ink small" role="alert">${e.saveError}</span>`:f}
      </div>`}render(){let e=p`<nav class="mem-tabs pk-tabs-top" aria-label="packages tabs">
      ${[{id:null,label:`Packages and skills`},{id:`library`,label:`Library retrieval`},{id:`proposals`,label:`Proposals`}].map(e=>p`<a class="cond-tab ${(this.tab??null)===e.id?`is-on`:``}" href="#" data-pk-top=${e.id??`all`} aria-current=${(this.tab??null)===e.id?`page`:f} @click=${t=>{t.preventDefault(),u({ptab:e.id,pkg:null,pview:null})}}>${e.label}</a>`)}
    </nav>`;if(this.tab===`proposals`)return p`${e}<qse-cond-proposals .onChanged=${()=>this.refresh()}></qse-cond-proposals>`;if(this.tab===`library`)return p`${e}<qse-cond-library .ids=${this.ids} .tiles=${this.tiles} .catalog=${this.catalog} .usage=${this.usage} .onChanged=${()=>this.refresh()}></qse-cond-library>`;let t=this.pk;if(t.loading&&!t.draft)return p`${e}${m(6)}`;if(t.loadError)return p`${e}<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${t.loadError}</div></div>`;let n=t.draft,r=Dt(n,this.catalog,this.usage?.packages??null,this.rt.draft),i=jt(r,this.filters,this.hits,this.catalog),a=this.selected&&n.packages[this.selected]?this.selected:null;return p`
      ${e}
      ${this.catError?p`<div class="guard-line" role="alert">${x(`warn`)} ${this.catError}</div>`:f}
      <div class="pk-layout ${a?`has-detail`:``}">
        ${this.renderList(i,r)}
        ${a?this.renderDetail(a,n.packages[a],r.find(e=>e.name===a)):this.selected?p`<section class="panel" data-pk-gone><p class="muted">No package ${this.selected} in this version.</p><button class="btn btn-sm" @click=${()=>this.open(null)}>Back to the list</button></section>`:f}
      </div>
      ${this.renderFound(n)}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${n.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${this.rt.draft&&(this.rt.showVersions||this.rt.saved||this.review===`router`||JSON.stringify(this.rt.draft)!==JSON.stringify(this.rt.base?.doc))?this.bar(this.rt,`router`,`the router`):f}
      ${this.bar(this.pk,`packages`,`the packages`)}
    `}renderList(e,t){let n=this.filters,r=t.filter(e=>At(e,this.catalog)).length,i=e=>this.filters={...this.filters,...e};return p`<section class="panel pk-list" data-pk-list>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills <span class="muted" data-pk-count-all>· ${e.length} of ${t.length}</span></h2>
        <p class="panel-sub">What the router may put into a prompt. Tokens are the text that goes in, ${this.catalog?.tokenizer===`estimate`?`estimated at 3.5 characters a token (no tokenizer on this server)`:`on the engine's tokenizer`}${r?p`; <span class="warn-ink" data-pk-over-count>${r} over the limit</span>`:f}.</p></div>
        <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-pk-new>${x(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${x(`search`)} Skills on the box</button></div></header>
      <div class="pk-filters">
        <label class="field field-search pk-search">${x(`search`)}<input type="search" data-pk-search placeholder="search packages, or the library's text" aria-label="search packages and skills" .value=${n.q} @input=${e=>void this.search(e.target.value)} /></label>
        <label class="field"><span>kind</span><select data-pk-filter="kind" @change=${e=>i({category:e.target.value})}>
          <option value="all" ?selected=${n.category===`all`}>all</option>
          ${R.map(e=>p`<option value=${e} ?selected=${n.category===e}>${z[e]}</option>`)}
        </select></label>
        <label class="field"><span>class</span><select data-pk-filter="class" @change=${e=>i({cls:e.target.value})}>
          <option value="" ?selected=${!n.cls}>any</option>
          ${this.ids.map(e=>p`<option value=${e} ?selected=${n.cls===e}>${e}</option>`)}
        </select></label>
        <label class="field"><span>enabled</span><select data-pk-filter="enabled" @change=${e=>i({enabled:e.target.value})}>
          ${[`all`,`on`,`off`].map(e=>p`<option value=${e} ?selected=${n.enabled===e}>${e}</option>`)}
        </select></label>
        <label class="switch"><input type="checkbox" data-pk-filter="over" .checked=${n.over} @change=${e=>i({over:e.target.checked})} /> over the limit</label>
      </div>
      ${this.libError?p`<p class="err-ink small">${this.libError}</p>`:f}
      <div class="pk-head" aria-hidden="true"><span>package</span><span>tokens / limit</span><span>used 24 h · 7 d</span></div>
      <div class="pk-rows" role="list">${e.map(e=>this.row(e))}</div>
      ${e.length?f:p`<p class="muted small" data-pk-empty>${n.q.trim()?`Nothing matches “${n.q.trim()}”${this.hits?`, and the library shares no word with it`:``}.`:`No package matches these filters.`}</p>`}
    </section>`}row(e){let t=Tt(e.name,e.pkg,e.category,this.catalog),n=e.entry?e.entry.tokens:B(V(e.pkg.body,!!e.pkg.library)[0]),r=At(e,this.catalog),i=e.usage,a=this.selected===e.name;return p`<button class="pk-row ${a?`is-open`:``} ${e.pkg.enabled?``:`is-off`}" role="listitem" data-pk-row=${e.name} data-category=${e.category} aria-current=${a?`true`:f} @click=${()=>this.open(e.name)}>
      <span class="pk-main">
        <span class="pk-name"><strong>${e.name}</strong>
          <span class="badge pk-kind pk-kind-${e.category}">${z[e.category]}</span>
          ${e.pkg.general?p`<span class="badge badge-on" title="goes in whole and first on every turn of a class that lists it">general</span>`:f}
          ${e.pkg.enabled?f:p`<span class="badge">off</span>`}
          ${e.entry?f:p`<span class="badge badge-shadow">not saved</span>`}
          ${e.entry&&!e.entry.available&&e.entry.error?p`<span class="badge badge-warn" title=${e.entry.path??``}>${e.entry.error.replace(/_/g,` `)}</span>`:f}
          ${e.score==null?f:p`<span class="pk-score num" data-score title="the library's BM25 score">${e.score.toFixed(2)}</span>`}
        </span>
        <span class="pk-desc muted small">${e.pkg.description}</span>
        <span class="pk-meta"><span class="mono small muted">${e.pkg.source}</span>${e.classes.map(e=>D(e,this.ids))}${e.category===`library`&&e.entry?.fits?.length?p`<span class="muted small">fits</span>${e.entry.fits.map(e=>D(e,this.ids))}`:f}</span>
      </span>
      <span class="pk-tokens num ${r?`is-over`:``}" data-pk-tokens title=${e.entry?`${e.entry.tokens} tokens go in; the whole text is ${e.entry.body_tokens}`:`estimated: not saved yet`}>
        <span>${e.entry?``:`≈`}${c(n)} <span class="muted">/ ${c(t.limit)}</span></span>
        <span class="pk-bar"><span style="width:${Math.min(100,100*n/Math.max(1,t.limit))}%"></span></span>
        ${r?p`<span class="badge badge-warn">over</span>`:f}
      </span>
      <span class="pk-usage num small" data-pk-usage>
        ${i&&(i.injected.week||i.shadow.week||i.pick.week)?p`${i.injected.week?p`<span>${i.injected.day} · ${i.injected.week} in</span>`:f}${i.shadow.week?p`<span class="muted">${i.shadow.day} · ${i.shadow.week} shadow</span>`:f}${i.pick.week?p`<span>${i.pick.day} · ${i.pick.week} picks</span>`:f}`:p`<span class="muted">—</span>`}
      </span>
    </button>`}renderFound(e){return this.scanError?p`<p class="err-ink small">${this.scanError}</p>`:this.found?p`<section class="panel" style="margin-top:16px" data-found>
      <header class="panel-head"><h2 class="panel-title">Skills in ${e.skill_dirs.join(`, `)}</h2><div class="panel-tools"><button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.found=null}>${x(`x`)}</button></div></header>
      ${this.found.length?p`<div class="table-scroll"><table class="table"><tbody>${this.found.map(t=>p`<tr><td><strong>${t.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${t.description}</div></td><td class="num">${c(t.chars)} chars</td>
              <td>${t.registered||e.packages[t.name]?p`<span class="badge badge-ok">registered</span>`:p`<button class="btn btn-sm" @click=${()=>this.register(t)} data-register=${t.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:p`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ &lt;box&gt;:~/.config/opencode/skills/</code>).</p>`}
    </section>`:f}renderDetail(e,t,n){let r=n.entry,i=wt(t,r),a=Tt(e,t,i,this.catalog),o=en.some(e=>e.id===this.pane)?this.pane:`text`,s=t.source===`inline`,l=n.classes,d=this.tokensNow(e,t,r),m=d.n>a.limit||!!r?.cut&&!(s&&this.dirtyBody(e));return p`<section class="panel pk-detail" data-pk-detail=${e}>
      <header class="pk-detail-head">
        <div class="pk-title">
          <h2 class="panel-title">${e}</h2>
          <span class="muted">${t.title}</span>
        </div>
        <button class="btn btn-ghost btn-sm pk-close" @click=${()=>this.open(null)} data-pk-close aria-label="Close the package">${x(`x`)}<span class="pk-close-word">All packages</span></button>
      </header>
      <div class="pk-badges">
        <span class="badge pk-kind pk-kind-${i}" data-pk-category>${z[i]}</span>
        <span class="badge">${s?`inline`:t.source.startsWith(`skill:`)?`skill on the box`:`file on the box`}</span>
        ${t.general?p`<span class="badge badge-on">general</span>`:f}
        ${t.enabled?p`<span class="badge badge-ok">enabled</span>`:p`<span class="badge">disabled</span>`}
        ${r?f:p`<span class="badge badge-shadow">not saved</span>`}
        ${r&&!r.available&&r.error?p`<span class="badge badge-warn">${r.error.replace(/_/g,` `)}</span>`:f}
      </div>
      <div class="toolbar pk-switches">
        <label class="switch"><input type="checkbox" data-pk-enabled .checked=${t.enabled} @change=${t=>this.upd([`packages`,e,`enabled`],t.target.checked)} /> enabled</label>
        ${i===`library`?f:p`<label class="switch" title="goes in whole and first on every turn of a class that lists it (K1)"><input type="checkbox" data-pk-general .checked=${!!t.general} @change=${t=>this.upd([`packages`,e,`general`],t.target.checked)} /> general</label>`}
        ${s?f:p`<button class="btn btn-sm" data-pk-copy-inline ?disabled=${!this.item?.found} title=${this.item?.found?`an inline copy of the file overrides it`:`the file is not on the box`} @click=${()=>this.copyInline(e)}>${x(`copy`)} Copy to inline</button>`}
        <button class="btn btn-sm btn-ghost" data-pk-remove ?disabled=${l.length>0} title=${l.length?`${l.join(`, `)} list it; detach it first`:`remove it from the packages document`} @click=${()=>this.removePackage(e)}>${x(`trash`)} Remove</button>
      </div>
      <div class="pk-meter ${m?`is-over`:``}" data-pk-meter>
        <div class="pk-bar"><span style="width:${Math.min(100,100*d.n/Math.max(1,a.limit))}%"></span></div>
        <span class="num small" data-pk-count>${d.approx?`≈ `:``}${c(d.n)} tokens of ${c(a.limit)} <span class="muted">(${d.why}; limit: ${a.source===`house`?i===`library`?`the house rule for a library skill`:`the house rule for the library card`:`max_tokens, where the server cuts`})</span></span>
        ${m?p`<span class="badge badge-warn" data-pk-limit-warn>over the limit</span>`:f}
      </div>
      <nav class="mem-tabs pk-panes" aria-label="package detail">
        ${en.map(e=>p`<a class="cond-tab ${e.id===o?`is-on`:``}" href="#" data-pk-pane=${e.id} aria-current=${e.id===o?`page`:f} @click=${t=>{t.preventDefault(),u({pview:e.id===`text`?null:e.id})}}>${e.label}</a>`)}
      </nav>
      <div class="pk-pane" data-pk-pane-body=${o}>
        ${o===`text`?this.paneText(e,t,r):o===`edit`?this.paneEdit(e,t,i,a):o===`classes`?this.paneClasses(e,t,i):this.paneUsage(e,i)}
      </div>
    </section>`}dirtyBody(e){return this.pk.draft?.packages[e]?.body!==this.pk.base?.doc.packages[e]?.body}tokensNow(e,t,n){if(t.source===`inline`&&(this.dirtyBody(e)||!n)){let e=V(t.body,!!t.library)[0];return this.counted?.text===e?{n:this.counted.n,approx:this.counted.how===`estimate`,why:this.counted.how===`engine`?`the draft, on the engine's tokenizer`:`the draft, estimated`}:{n:B(e),approx:!0,why:`the draft, estimated until the engine counts it`}}return n?{n:n.body_tokens,approx:this.catalog?.tokenizer===`estimate`,why:n.cut?`the whole text; ${c(n.tokens)} go in after the cut`:this.catalog?.tokenizer===`estimate`?`estimated`:`on the engine's tokenizer`}:{n:0,approx:!0,why:`not counted yet`}}md(e){return e?this.raw?p`<pre class="note-text pk-raw" data-pk-raw>${e}</pre>`:p`<div class="pg-md pk-md" data-pk-md>${st(ct(e))}</div>`:p`<p class="muted small">(empty)</p>`}paneText(e,t,n){let r=t.source===`inline`,i=this.item?.name===e?this.item:null;if(!r&&!i)return this.itemError?p`<p class="err-ink small">${this.itemError}</p>`:m(4);let[a,o]=r?V(t.body,!!t.library):[i.body,i.reference],s=r?{}:i.frontmatter;return p`
      <div class="toolbar pk-text-tools">${j([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`text view`})}
        ${!r&&i?.path?p`<span class="mono small muted pk-path" data-pk-path>${i.path}</span>`:f}</div>
      ${!r&&i&&!i.found?p`<p class="note" data-pk-missing>${x(`warn`)} The file is not on the box (<code>${i.path??t.source}</code>): nothing goes in.</p>`:f}
      ${Object.keys(s).length?p`<h3 class="pk-sub">Frontmatter <span class="muted small">the client's metadata; never injected</span></h3>
            ${this.raw?p`<pre class="note-text pk-raw" data-pk-front>${i.frontmatter_raw}</pre>`:p`<dl class="kv pk-front" data-pk-front>${Object.entries(s).map(([e,t])=>p`<div><dt>${e}</dt><dd>${t}</dd></div>`)}</dl>`}`:f}
      <h3 class="pk-sub">The text that goes in ${a?ne(a,`Copy the text`):f}</h3>
      ${this.md(a)}
      ${n?.cut&&!this.dirtyBody(e)?p`<p class="note" data-pk-cut>${x(`warn`)} The server cuts this text at ${c(t.max_tokens)} tokens × 3.5 characters, at a line end, and adds “[... cut to the package's max_tokens]”: ${c(n.tokens)} tokens go in of ${c(n.body_tokens)}.</p>`:f}
      ${o?p`<details class="sec pk-ref" data-pk-ref>
            <summary>Reference <span class="sec-sub">${c(i?.tokens.reference??B(o))} tokens; only a read gets it (GET /v1/library/${e})</span></summary>
            <div class="sec-body">${this.md(o)}</div>
          </details>`:f}`}paneEdit(e,t,n,r){let i=t.source===`inline`;return p`
      <div class="form-grid">
        <label class="field"><span>title</span><input data-pk-title .value=${t.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>max tokens</span><input type="number" min="16" max="32768" data-pk-max .value=${String(t.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">the server cuts a longer text at a line end</span></label>
        ${i?f:p`<label class="field field-wide"><span>source</span><input class="mono" .value=${t.source} readonly /><span class="hint">${this.item?.path??`a file on the box`}</span></label>`}
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" data-pk-description .value=${t.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${i?this.editor(e,t,n,r):p`<div class="note pk-readonly" data-pk-readonly>${x(`info`)} The text is the file on the box${this.item?.path?p` at <code>${this.item.path}</code>`:f}, read when first needed and again when it changes. Editing files on the box is not done here. <button class="btn btn-sm" ?disabled=${!this.item?.found} @click=${()=>this.copyInline(e)} data-pk-copy-inline-edit>Copy to inline</button> makes an inline copy of its text that overrides the file.</div>`}`}editor(e,t,n,r){let[i,a]=V(t.body,!!t.library),o=this.tokensNow(e,t,this.catalog?.packages[e]??null),s=o.n>r.limit,l=this.editView;return p`<div class="pk-editor pk-view-${l}" data-pk-editor>
      <div class="toolbar pk-editor-tools">
        ${j([`write`,`preview`,`split`],l,e=>this.editView=e,{small:!0,label:`editor view`})}
        <span class="pk-count num small ${s?`is-over`:``}" data-pk-editor-count>${o.approx?`≈ `:``}${c(o.n)} of ${c(r.limit)} tokens${t.library&&a?p` <span class="muted">· reference ≈ ${c(B(a))}</span>`:f}</span>
        ${s?p`<span class="badge badge-warn" data-pk-editor-warn>over ${r.source===`house`?`the ${z[n]}'s ${c(r.limit)}`:`max_tokens: the server will cut it`}</span>`:f}
      </div>
      <div class="pk-editor-panes">
        ${l===`preview`?f:p`<textarea class="code pk-body" rows="18" spellcheck="false" data-pk-body aria-label="the package's text (Markdown)" .value=${t.body} @input=${t=>this.upd([`packages`,e,`body`],t.target.value)}></textarea>`}
        ${l===`write`?f:p`<div class="pg-md pk-md pk-preview" data-pk-preview>${st(ct(i))}</div>`}
      </div>
      <p class="hint">${t.library?`A library skill: the text before “## Reference” goes in; the reference only on a read. `:``}${r.source===`house`?`The house rule keeps ${z[n]} texts at ${c(r.limit)} tokens or fewer.`:`The server cuts the text at max_tokens × 3.5 characters, at a line end.`} Markdown; the preview is the dashboard's renderer, the model gets the text as written.</p>
    </div>`}paneClasses(e,t,n){let r=this.catalog?.packages[e]??null;if(n===`library`)return p`<p class="muted small" data-pk-library-classes>A library skill is searched, not listed by a class: on a new user message the router searches the library and keeps the skills INDEX.md fits to the turn's class.</p>
        <div class="pk-meta">${r?.fits?.length?p`<span class="muted small">INDEX.md fits</span>${r.fits.map(e=>D(e,this.ids))}`:p`<span class="muted small">INDEX.md does not list it: it stays in for every class.</span>`}</div>
        <p class="small"><a href="#/conductor?screen=packages&ptab=library">Each profile's retrieval switch and its picks</a></p>`;let i=this.rt.draft;return i?p`
      <p class="muted small">A class's packages go in by K1's order: the general ones whole and first, then the router asks the package question among the rest and puts in the one that fits. A class lists at most ${12}. Attaching edits the router document; it is saved with its own check below.</p>
      ${this.attachError?p`<p class="err-ink small" role="alert" data-pk-attach-error>${this.attachError}</p>`:f}
      <div class="pk-classes" data-pk-classes>
        ${i.classes.map(t=>{let n=t.policy.packages??[],r=n.includes(e);return p`<div class="pk-class ${r?`is-on`:``}" data-pk-class=${t.id}>
            <label class="switch"><input type="checkbox" .checked=${r} data-pk-attach=${t.id} aria-label=${`attach ${e} to ${t.id}`} @change=${n=>this.toggleAttach(t.id,e,n.target.checked)} /> ${D(t.id,this.ids,t.label)}</label>
            <span class="pk-order small" data-pk-order=${t.id}>${n.length?n.map((t,n)=>p`<span class="pk-order-item ${t===e?`is-me`:``}">${n+1}. ${t}${this.pk.draft?.packages[t]?.general?p` <span class="muted">general</span>`:f}</span>`):p`<span class="muted">no packages</span>`}</span>
            <span class="muted small num">${n.length}/${12}</span>
          </div>`})}
      </div>
      ${r&&Object.keys(r.profiles).length?p`<p class="small muted" data-pk-profiles>A profile's own routing also lists it: ${Object.entries(r.profiles).map(([e,t])=>`${e} (${t.join(`, `)})`).join(`; `)}. A profile's routing replaces the router's policy for that class (the Profiles screen).</p>`:f}
      <div class="toolbar"><button class="btn btn-sm btn-ghost" data-pk-router-versions @click=${()=>{this.rt.showVersions=!0,this.rt.loadVersions()}}>Router versions</button></div>`:m(3)}paneUsage(e,t){let n=this.usage?.packages[e]??null,r={day:0,week:0},i=(e,t)=>S({label:e,value:c(t.day),unit:`in 24 h`,sub:`${c(t.week)} in 7 days`}),a=this.item?.name===e?this.item:null,o=t===`library`||(n?.pick.week??0)>0;return p`
      <div class="stats pk-stats" style="--pk-tiles:${o?3:2}" data-pk-usage-tiles>
        ${i(`injected`,n?.injected??r)}${i(`shadow (would have)`,n?.shadow??r)}${o?i(`library picks`,n?.pick??r):f}
      </div>
      <h3 class="pk-sub">Recent rows <span class="muted small">the journal, newest first (7 days)</span></h3>
      ${this.recent==null?m(3):this.recent.length?p`<div class="pk-uses" data-pk-recent>${this.recent.map(e=>p`<div class="pk-use" data-pk-use=${e.id}>
                <span class="num small">${_(new Date(e.ts_ms).toISOString())}</span>
                <span class="badge ${e.how===`injected`?`badge-on`:e.how===`shadow`?`badge-shadow`:`badge-warn`}">${e.how}</span>
                ${D(e.class_id,this.ids)}
                <span class="muted small">${e.profile??`—`}${e.client_kind?` · ${e.client_kind}`:``}</span>
                ${e.session_key?p`<a class="mono small pk-session" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${e.session_key.length>28?`${e.session_key.slice(0,28)}…`:e.session_key}</a>`:f}
              </div>`)}</div>`:p`<p class="muted small" data-pk-recent-empty>No router row names it in the last 7 days.</p>`}
      ${t===`library`?p`<h3 class="pk-sub">Neighbours <span class="muted small">the library's BM25 on this skill's name, description and text</span></h3>
            ${a?.neighbours?a.neighbours.length?p`<div class="pk-neighbours" data-pk-neighbours>${a.neighbours.map(e=>p`<button class="pk-neighbour" data-pk-neighbour=${e.name} @click=${()=>this.open(e.name)}><strong>${e.name}</strong> <span class="num muted small">${e.score.toFixed(2)}</span><span class="muted small pk-desc">${e.description}</span></button>`)}</div>`:p`<p class="muted small">No other skill shares a word with it.</p>`:m(2)}`:f}`}};d([y({attribute:!1})],G.prototype,`ids`,void 0),d([y({attribute:!1})],G.prototype,`tiles`,void 0),d([y({attribute:!1})],G.prototype,`selected`,void 0),d([y({attribute:!1})],G.prototype,`tab`,void 0),d([y({attribute:!1})],G.prototype,`pane`,void 0),d([o()],G.prototype,`catalog`,void 0),d([o()],G.prototype,`catError`,void 0),d([o()],G.prototype,`usage`,void 0),d([o()],G.prototype,`filters`,void 0),d([o()],G.prototype,`hits`,void 0),d([o()],G.prototype,`libError`,void 0),d([o()],G.prototype,`item`,void 0),d([o()],G.prototype,`itemError`,void 0),d([o()],G.prototype,`recent`,void 0),d([o()],G.prototype,`raw`,void 0),d([o()],G.prototype,`editView`,void 0),d([o()],G.prototype,`counted`,void 0),d([o()],G.prototype,`review`,void 0),d([o()],G.prototype,`attachError`,void 0),d([o()],G.prototype,`found`,void 0),d([o()],G.prototype,`scanError`,void 0),G=d([a(`qse-cond-packages`)],G);var tn={not_loaded:`not loaded`,loading:`loading the weights`,warming:`warming up`,loaded:`loaded`,unloading:`unloading`,failed:`failed`};function nn(e){let t=e.role;return t?[t.kind,t.window.replace(`_`,` `),t.preemptible?`preemptible`:`not preemptible`,`${t.latency_budget_ms} ms then ${t.fallback}`,`writes ${t.writes.replace(/_/g,` `)}`,`shares ${t.sharing.join(`, `)||`nothing`}`,t.memory_claim_gib==null?`no claim`:`claims ${b(t.memory_claim_gib)}`].join(` · `):null}function rn(e){return e===`loaded`?`badge-ok`:e===`failed`?`badge-warn`:e===`loading`||e===`warming`||e===`unloading`?`badge-shadow`:``}function an(e){if(!e||e.allowed)return null;let t=e.reason?ie[e.reason]??e.reason:`refused`;if(e.reason!==`refused_memory`)return t;let n=[`MemAvailable ${b(e.mem_available_gib)}`];return e.engine_growth_gib&&n.push(`engine growth ${b(e.engine_growth_gib)}`),n.push(`claim ${b(e.claim_gib)}`),e.after_gib!=null&&n.push(`${b(e.after_gib)} left`,`${b(e.need_gib)} needed`),`${t} (${n.join(`, `)})`}function on(e){return e?e.allowed?`fits, ${b(e.after_gib)} left after the claim`:an(e)??`refused`:`—`}function sn(e){let t=Object.entries(e).sort((e,t)=>t[1]-e[1]);return t.length?t.map(([e,t])=>`${e} ${t}`).join(` · `):`no call yet`}function cn(e){let t=e.limits;return[t.context_tokens,t.question_tokens,t.max_options].map(e=>e==null?`—`:String(e)).join(` / `)}function ln(e,t){let n=e.trim();if(!n)return`Write the question.`;let r=t.split(/[,\n]/).map(e=>e.trim()).filter(Boolean);return r.length<2?`Give at least two options.`:r.length>20?`At most 20 options (the model card’s limit).`:new Set(r).size===r.length?{type:`choice`,instructions:n,criteria:Object.fromEntries(r.map(e=>[e,null]))}:`Two options have the same name.`}function un(e){return e.type===`noul`?[[`true`,e.noul??0],[`false`,1-(e.noul??0)]]:Object.entries(e.probabilities??{}).sort((e,t)=>t[1]-e[1])}var dn=[`on_start`,`on_demand`,`off`],fn={mode:`on`,rate_per_min:240,max_texts:16,max_chars:1e5,deadline_ms:15e3,threshold:.8,drive_rate_per_s:10,drive_sessions:4,drive_idle_s:30,drive_max_texts:36e3},K=class extends r{constructor(...e){super(...e),this.st=null,this.error=null,this.busy=null,this.said={},this.tState=`The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.`,this.tQuestion=`What kind of work does this request ask for?`,this.tOptions=`debug, code_edit, plan, chat`,this.tModel=`auto`,this.tOut=null,this.tErr=null,this.asking=!1,this.ed=new w(this,`deciders`),this.poller=null}connectedCallback(){super.connectedCallback(),this.ed.load(),this.poller=h(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{this.st=await A.deciders(),this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async act(n,r){if(r!==`unload`||await O(`Unload ${n.name}?`,[`Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).`,`Loading it again goes through the residency planner.`],`Unload`)){this.busy=n.name,this.said={...this.said,[n.name]:``};try{let e=await N(()=>r===`load`?A.deciderLoad(n.name):A.deciderUnload(n.name));this.said={...this.said,[n.name]:r===`load`?`load started (${e.state??`loading`})`:`unloaded`}}catch(i){let a=i instanceof e&&i.status===409&&r===`load`?`refused by the residency planner: ${i.message}`:t(i).message;this.said={...this.said,[n.name]:a}}finally{this.busy=null,await this.load()}}}setLoad(e,t){let n=this.ed.draft;if(!n)return;let r=n.models.findIndex(t=>t.name===e),i=r>=0?n.models.map((e,n)=>n===r?{...e,load:t}:e):[...n.models,{name:e,load:t}];this.ed.set({...n,models:i})}async ask(){let e=ln(this.tQuestion,this.tOptions);if(typeof e==`string`){this.tErr=e;return}this.asking=!0,this.tErr=null;try{this.tOut=await A.deciderTest(this.tState,{q:e},this.tModel)}catch(e){this.tOut=null,this.tErr=t(e).message}finally{this.asking=!1}}render(){let e=this.st;if(this.error&&!e)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!e)return m(6);if(!e.enabled)return C(`Decision models`,p`<div class="state" data-deciders-off>
          <div class="state-title">${x(`warn`)} This engine serves no decision model</div>
          <p class="muted small">${e.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port behind the admin token, and this screen loads, unloads and tests them.</p>
        </div>`,{id:`deciders-off`});let t=e.models.filter(e=>e.state===`loaded`);return p`
      <div class="stats" data-decider-tiles>
        ${S({label:`loaded`,value:`${t.length} / ${e.models.length}`,sub:t.length?t.map(e=>e.name).join(`, `):`none: questions fail open`,warn:!t.length})}
        ${S({label:`on the device`,value:re(t.reduce((e,t)=>e+t.device_bytes,0)),sub:`${e.device??`no device`} · own stream and memory pool`})}
        ${S({label:`a load now`,value:e.planner?.allowed?`fits`:`refused`,sub:on(e.planner),warn:!e.planner?.allowed})}
        ${S({label:`when they run`,value:`idle`,sub:`only while no request holds the engine (stage A)`})}
      </div>
      ${this.renderModels(e)}
      ${this.renderMemory(e)}
      ${this.renderApi(e)}
      ${this.renderQuestions(e)}
      ${this.renderTest(e)}
    `}renderModels(e){let t=this.ed.draft,r=e=>t?.models.find(t=>t.name===e)?.load??`on_demand`;return C(`Models`,p`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${e.models.map(e=>{let t=an(e.refusal),i=this.said[e.name],a=e.state===`loaded`||e.state===`failed`||e.state===`warming`;return p`<tr data-decider-row=${e.name}>
              <td><strong>${e.name}</strong><div class="muted small">${e.family} · ${e.contract}${e.aliases.length?p` · ${e.aliases.join(`, `)}`:f}</div>${e.role?p`<div class="muted small" data-decider-role=${e.name} title="the role contract this model signs (SRV-133); the registry checks it before it serves the model">${nn(e)}</div>`:f}</td>
              <td><span class="badge ${rn(e.state)}" data-decider-state=${e.state}>${tn[e.state]}</span>${e.warm===!1?p` <span class="muted small">not warmed</span>`:f}</td>
              <td><select aria-label=${`load policy of ${e.name}`} data-decider-load=${e.name} @change=${t=>this.setLoad(e.name,t.target.value)}>
                ${dn.map(t=>p`<option value=${t} ?selected=${r(e.name)===t}>${t.replace(`_`,` `)}</option>`)}
              </select></td>
              <td class="num">${cn(e)}</td>
              <td class="small">${e.precision??p`<span class="muted">—</span>`}<div class="muted small mono" title=${e.weight_hash??``}>${e.weight_hash?e.weight_hash.slice(0,12):``}</div></td>
              <td class="num num-col">${c(e.questions)}</td>
              <td class="num num-col">${e.stats?.forward_ms_p50==null?`—`:`${n(e.stats.forward_ms_p50)} / ${n(e.stats.forward_ms_p95)}`}</td>
              <td class="num num-col ${e.stats?.errors?`err-ink`:``}">${e.stats?c(e.stats.errors+e.stats.deadline_missed):`—`}</td>
              <td>${a?p`<button class="btn btn-sm" ?disabled=${this.busy===e.name} @click=${()=>this.act(e,`unload`)} data-unload=${e.name}>Unload</button>`:p`<button class="btn btn-sm btn-primary" ?disabled=${this.busy===e.name||e.state===`loading`||e.state===`unloading`} @click=${()=>this.act(e,`load`)} data-load=${e.name}>Load</button>`}</td>
            </tr>
            ${t||e.error||i?p`<tr class="is-warn" data-decider-note=${e.name}><td colspan="9" class="small">
                  ${t?p`<span class="warn-ink" data-refusal=${e.name}>${x(`warn`)} ${t}</span> `:f}
                  ${e.error?p`<span class="err-ink">${e.error}</span> `:f}
                  ${i?p`<span role="status">${i}</span>`:f}
                </td></tr>`:f}`})}</tbody>
      </table></div>
      ${e.unserved?.length?p`<p class="muted small">In the document but not served by this engine: ${e.unserved.join(`, `)} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>`:f}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,{id:`deciders-models`,sub:`the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen`})}renderMemory(e){let t=e.planner,n=e.last_refusal;return C(`Memory`,p`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${b(t?.mem_available_gib)}</div></div>
          ${t?.engine_growth_gib==null?f:p`<div class="stat"><div class="stat-label">engine growth to its peak</div><div class="stat-value num">${b(t.engine_growth_gib)}</div></div>`}
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${b(t?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${b(t?.floor_gib)} + ${b(t?.margin_gib)}</div></div>
          <div class="stat ${t?.allowed?``:`is-warn`}"><div class="stat-label">a load now</div><div class="stat-value num small">${on(t)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${n?p`Last refused engine load: ${an(n)} <span class="muted">(${n.source??``})</span>`:p`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable, less what the engine may still take up to its declared peak, minus the claim stays at or above floor + margin, never during a hold, never beside another load. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,{id:`deciders-memory`})}renderApi(e){let t=this.ed.draft,r={...fn,...t?.api??{}},i=e.api?.stats,a=(e,n)=>t&&this.ed.set(T(t,[`api`,e],n)),o=(e,t,n=1)=>p`<label class="field"><span>${t}</span><input type="number" step=${n} .value=${String(r[e])} data-api-field=${e} @change=${t=>a(e,n<1?Number(t.target.value):Math.round(Number(t.target.value)))} /></label>`;return C(`Laya for scripts`,t?p`<div data-decider-api>
            <div class="toolbar" style="margin:0"><span class="muted small">route</span>${j([`off`,`on`],r.mode,e=>a(`mode`,e),{small:!0,label:`Laya route mode`})}<code class="small">POST /v1/laya/ask</code></div>
            <div class="form-grid">
              ${o(`rate_per_min`,`texts a minute (every caller)`)}
              ${o(`max_texts`,`texts in one call`)}
              ${o(`deadline_ms`,`deadline (ms) while a request runs`)}
              ${o(`threshold`,`sure at or above`,.05)}
              ${o(`drive_rate_per_s`,`texts a second (a playtest session)`)}
              ${o(`drive_sessions`,`playtest sessions at once`)}
            </div>
            <p class="muted small" data-api-stats>${i?`${c(i.calls)} calls, ${c(i.texts)} texts, ${c(i.errors)} errors (${c(i.rate_limited)} over the rate) since the start${i.last_ms==null?``:`; the last took ${n(i.last_ms)}`}${i.drive_sessions?`; ${c(i.drive_sessions)} playtest sessions live`:``}`:`no calls yet`}</p>
            <p class="muted small">The model's own scripts ask Laya typed questions about their evidence (a log tail, an output, a state dump) through the provider, open like the chat API: a choice, a scale or a yes-no question; Laya reads about 1,100 characters of each text, its head and its tail. Every call is a journal row (<code>laya_api</code>). A playtest (Laya plays the build: <code>.tandem/laya_drive.py</code>) sends a <code>session</code>: its own rate a second, one <code>laya_drive</code> row when it ends.</p>
          </div>`:m(2),{id:`deciders-api`,sub:`the route the model’s scripts call: on or off, its rate and its limits`})}renderQuestions(e){let t=this.ed,n=t.draft,r=e.models.map(e=>e.name),i=n?Object.entries(n.questions):[];return p`${C(`Questions`,n?p`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${i.map(([i,a])=>{let o=e.questions.find(e=>e.id===i);return p`<tr data-question=${i}>
                  <td class="mono">${i}</td>
                  <td><select aria-label=${`model for ${i}`} data-question-model=${i} @change=${e=>t.set(T(n,[`questions`,i],e.target.value))}>
                    ${[`auto`,...r].map(e=>p`<option value=${e} ?selected=${e===a}>${e===`auto`?`auto (the role’s model; the router picks)`:e}</option>`)}
                    ${a!==`auto`&&!r.includes(a)?p`<option value=${a} selected>${a} (not served)</option>`:f}
                  </select></td>
                  <td class="small">${sn(o?.picks??{})}</td>
                </tr>`})}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); <code>judge</code> is the test judge's questions on a run's evidence (the <code>tester</code> role), <code>laya_api</code> the model's own scripts through <code>/v1/laya/ask</code>; the session monitor's Tier 1 questions join this list when they land.</p>`:m(3),{id:`deciders-questions`,sub:`which model answers each question the conductor asks`})}
      ${k(t)}
      ${M(t,`the deciders document`)}`}renderTest(e){let t=this.tOut,r=t?.answers.q;return C(`Test box`,p`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${e=>this.tState=e.target.value} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${e=>this.tQuestion=e.target.value} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${e=>this.tOptions=e.target.value} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${e=>this.tModel=e.target.value} data-test-model>
            ${[`auto`,...e.models.map(e=>e.name)].map(e=>p`<option value=${e} ?selected=${e===this.tModel}>${e}</option>`)}
          </select></label>
        </div>
        <div class="savebar"><button class="btn btn-primary" ?disabled=${this.asking} @click=${()=>this.ask()} data-test-ask>${x(`send`)} ${this.asking?`Asking…`:`Ask`}</button>
          ${this.tErr?p`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>`:f}</div>
        ${t&&r?p`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${r.type===`choice`?r.choice:r.type===`noul`?(r.noul??0).toFixed(4):(r.score??0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${r.confidence.toFixed(2)} · ${t.model} · ${n(t.latency_ms)} · ${t.path} · ${c(t.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${un(r).map(([e,t])=>p`<div class="prob-row" data-prob=${e}><span class="mono small">${e}</span><span class="conf-bar"><span style="width:${Math.round(t*100)}%"></span></span><span class="num small">${t.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${t.routing.reason}${t.routing.requested?` (wanted ${t.routing.requested})`:``}</p>
            </div>`:f}`,{id:`deciders-test`,sub:`one choice question, as /v1/systemone answers it (it runs in the idle window like any other)`})}};d([o()],K.prototype,`st`,void 0),d([o()],K.prototype,`error`,void 0),d([o()],K.prototype,`busy`,void 0),d([o()],K.prototype,`said`,void 0),d([o()],K.prototype,`tState`,void 0),d([o()],K.prototype,`tQuestion`,void 0),d([o()],K.prototype,`tOptions`,void 0),d([o()],K.prototype,`tModel`,void 0),d([o()],K.prototype,`tOut`,void 0),d([o()],K.prototype,`tErr`,void 0),d([o()],K.prototype,`asking`,void 0),K=d([a(`qse-cond-deciders`)],K);var q=class extends r{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=h(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await A.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async loadSession(e){try{let t=await A.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(n){e===this.selected&&(this.turns=[]),this.error=t(n).message}}render(){if(this.error&&!this.list)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.list)return m(6);let e=[...new Set(this.list.map(e=>e.client_kind??`other`))],t=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return p`
      ${this.selected?this.renderSession(this.selected):f}
      ${C(`Sessions, last 7 days`,t.length?p`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${t.map(e=>p`<tr class="req-row ${e.key===this.selected?`is-open`:``}" data-session=${e.key} @click=${()=>u({session:e.key})}>
                  <td class="mono small cell-wrap">${E(e.key,28)}</td><td>${e.client_kind??`—`}</td><td>${e.profile??`—`}</td>
                  <td>${D(e.class_id,this.ids)}</td><td class="num num-col">${c(e.turns)}</td>
                  <td class="num num-col">${e.notes?`${e.notes} (${c(e.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${e.acts?p`${c(e.acts)}${e.gave_ups?p` <span class="badge badge-warn">${e.gave_ups} gave up</span>`:f}`:`—`}</td>
                  <td class="num num-col" data-interventions>${c(e.interventions??0)}</td>
                  <td>${e.last_checker?p`<span class="badge ${e.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${e.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${l(new Date(e.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:g(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:p`<label class="field field-search">${x(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${e.map(e=>p`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?p`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:f}
    `}renderSession(e){let t=this.turns,n=this.live.find(t=>t.key===e);if(t==null)return m(4);let r=t.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,i=t.reduce((e,t)=>e+(t.injected?.length??0),0),a=this.events.filter(e=>te(e)).length,o=Se(t);return p`<section class="panel" style="margin-bottom:16px" data-timeline=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${E(e,44)} ${n?p`<span class="badge badge-on">live</span>`:f}</h2>
        <p class="panel-sub">${t.length} turns · ${i} hidden note${i===1?``:`s`} · ${a} monitor act${a===1?``:`s`}${n?` · class now ${n.class_id} · ${c(n.note_tokens)} note tokens replayed a turn`:``}${t.length?` · since ${_(new Date(t[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${A.exportUrl(e,`md`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${x(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${A.exportUrl(e,`json`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${x(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>u({session:null})}>${x(`x`)}</button></div></header>
      ${this.axis(de(t,this.events,this.ids,e=>s(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${t.map(e=>{let t=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/r*900)),n=(e.injected?.length??0)>0,i=!!e.shadow,a=(e.events??[]).filter(e=>e.decision!==`PASS`),o=a.some(e=>te(e))?`act`:a.length?`shadow`:null;return p`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${t}px; --cls: var(--series-${Be(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${Me(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${n?p`<span class="mark"></span>`:i?p`<span class="mark is-shadow"></span>`:f}
            ${o?p`<span class="mark is-monitor ${o===`shadow`?`is-shadow`:``}" data-monitor-mark=${o}></span>`:f}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${o.series.length?p`<qse-ribbon .series=${o.series} .labels=${o.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:f}
      <div class="feed" style="margin-top:10px">${t.slice().reverse().map(e=>ze(e,this.ids,e=>this.open=e,Me(e)+((e.events??[]).some(e=>te(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?p`<div style="margin-top:12px"><qse-cond-journal .session=${e}></qse-cond-journal></div>`:f}
    </section>`}axis(e){let t=e=>`${(e*100).toFixed(3)}%`;return p`<figure class="axis" data-axis aria-label="the session on one time axis">
      <div class="axis-lane axis-acts" aria-label=${`${e.acts.length} monitor decisions`}>
        ${e.acts.map(e=>p`<span class="axis-mark ${e.shadow?`is-shadow`:`is-act`}" style="left:${t(e.x)}" title=${e.title} data-axis-act=${e.shadow?`shadow`:`act`}></span>`)}
        ${e.notes.map(e=>p`<span class="axis-mark is-note ${e.shadow?`is-shadow`:``}" style="left:${t(e.x)}" title=${e.title} data-axis-note></span>`)}
      </div>
      <div class="axis-lane axis-turns" aria-label=${`${e.turns.length} turns`}>
        ${e.turns.map(e=>p`<span class="axis-turn ${e.compaction?`is-compaction`:``}" style="left:${t(e.x0)};width:${t(Math.max(0,e.x1-e.x0))};--cls:var(--series-${e.color})" title=${e.title} data-axis-turn></span>`)}
      </div>
      <div class="axis-lane axis-events" aria-label=${`${e.compactions.length} compactions, ${e.checkers.length} checker results`}>
        ${e.compactions.map(e=>p`<span class="axis-mark is-compaction" style="left:${t(e.x)}" title=${e.title} data-axis-compaction></span>`)}
        ${e.checkers.map(e=>p`<span class="axis-mark ${e.pass?`is-pass`:`is-fail`}" style="left:${t(e.x)}" title=${e.title} data-axis-checker=${e.pass?`pass`:`fail`}></span>`)}
      </div>
      <div class="axis-ticks" aria-hidden="true">${e.ticks.map((n,r)=>p`<span class="num ${r===0?`is-first`:r===e.ticks.length-1?`is-last`:``}" style="left:${t(n.x)}">${n.label.slice(0,5)}</span>`)}</div>
      <figcaption class="muted small">One time axis, ${e.ticks[0]?.label??``} to ${e.ticks[e.ticks.length-1]?.label??``}: ${e.turns.length} turns (the gaps are the client's time), ${e.acts.filter(e=>!e.shadow).length} monitor actions ◆ above (◇ shadow, ${e.acts.filter(e=>e.shadow).length}), ${e.notes.length} router note${e.notes.length===1?``:`s`}, ${e.compactions.length} compaction${e.compactions.length===1?``:`s`} ▲ and ${e.checkers.length} checker result${e.checkers.length===1?``:`s`} below (● pass, ○ fail; the gate numbers are in each one's title).</figcaption>
    </figure>`}tier1(){let e=he(this.events);return e?p`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${e.agree} of ${e.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${e.decisions} shadow decisions)</span></p>`:f}};d([y({attribute:!1})],q.prototype,`ids`,void 0),d([y({attribute:!1})],q.prototype,`selected`,void 0),d([o()],q.prototype,`list`,void 0),d([o()],q.prototype,`live`,void 0),d([o()],q.prototype,`turns`,void 0),d([o()],q.prototype,`events`,void 0),d([o()],q.prototype,`error`,void 0),d([o()],q.prototype,`query`,void 0),d([o()],q.prototype,`client`,void 0),d([o()],q.prototype,`open`,void 0),q=d([a(`qse-cond-sessions`)],q);var pn={scope:`tools`,retention_days:30,clip_chars:4e3,loop_chars:200},mn=/^### (user|assistant|summary|runtime|tool result \((.*)\)|tool call \((.*)\)|reasoning \((.*)\))$/;function hn(e){let t=e[1];return e[2]===void 0?e[3]===void 0?e[4]===void 0?{kind:t===`user`||t===`assistant`||t===`summary`||t===`runtime`?t:`other`,title:t,name:null,clipped:!1}:{kind:`reasoning`,title:t,name:null,clipped:!1}:{kind:`tool_call`,title:t,name:e[3].trim()||null,clipped:!1}:{kind:`tool_result`,title:t,name:e[2].split(`,`)[0].trim()||null,clipped:/clipped to/.test(e[2])}}function gn(e){let t=e.replace(/\n+$/,``).split(`
`),n=t[0]??``,r=null,i=[],a=null,o=()=>{a&&i.push({kind:a.kind,title:a.title,name:a.name,clipped:a.clipped,body:a.lines.join(`
`).replace(/^\n+|\n+$/g,``)})};for(let e of t.slice(1)){let t=e.match(mn);t?(o(),a={...hn(t),lines:[]}):a?a.lines.push(e):e.startsWith(`compaction: `)&&(r=e)}o();let s=i.find(e=>e.kind===`runtime`);return{head:n,compaction:r,sections:i.filter(e=>e.kind!==`runtime`),runtime:s?_n(s.body):[]}}function _n(e){return e.split(`
`).filter(e=>e.trim()).map(e=>{let t=[`hidden`,`replayed`,`acts`,`checker`,`snapshot`,`finish`,`redactions`].find(t=>e.startsWith(`${t}: `))??`other`,n=e.match(/ · journal ([A-Za-z0-9_-]{4,40})$/),r=t===`checker`?e.match(/ (PASS|FAIL)(?: <-|$)/):null;return{kind:t,text:e,journal:n?n[1]:null,verdict:r?r[1]:null}})}function vn(e){if(!e.counts)return`counts unknown (no meta.json)`;let t=e.counts,n=[`${t.blocks} turn${t.blocks===1?``:`s`}`];return t.untranscribed&&n.push(`${t.untranscribed} without tools (counted only)`),n.push(`${t.acts} act${t.acts===1?``:`s`}`),t.snapshots&&n.push(`${t.snapshots} snapshot${t.snapshots===1?``:`s`}`),t.compactions&&n.push(`${t.compactions} compaction${t.compactions===1?``:`s`}`),t.redactions&&n.push(`${t.redactions} redacted`),e.checkers?.last&&n.push(e.checkers.last),n.join(` · `)}function yn(e){return{...pn,...e?.memory?.transcripts??{}}}function bn(e,t){return e?e.replace(/^[ch]:/,``):t}var xn={mode:`on`,budget_gb:50,ttl_days:7,admit_after:2,min_free_gb:50},Sn={"opencode-head":`opencode head`,head:`session head`,package:`package`,memory:`memory file`,prefix:`repeated prefix`};function Cn(e){return{...xn,...e?.memory?.disk_cache??{}}}function wn(e){let t=[],n=e=>Number.isFinite(e)&&Math.round(e)===e;return e.budget_gb>=1&&e.budget_gb<=4e3||t.push(`the budget is 1 to 4,000 GB`),n(e.ttl_days)&&e.ttl_days>=1&&e.ttl_days<=90||t.push(`an entry is kept 1 to 90 days`),n(e.admit_after)&&e.admit_after>=1&&e.admit_after<=10||t.push(`a prefix is written after 1 to 10 prefills`),e.min_free_gb>=50&&e.min_free_gb<=4e3||t.push(`the NVMe keeps at least 50 GB free`),t}function Tn(e,t){if(t.free_bytes==null)return{usable_gb:null,short:!1};let n=((t.bytes??0)+Math.max(0,t.free_bytes-Math.max(50,e.min_free_gb)*1e9))/1e9;return{usable_gb:n,short:n<e.budget_gb}}function En(e){let t=e.lookups??0;return t?`${Math.round((e.hit_rate??0)*100)} % of ${t.toLocaleString(`en-US`)} prefill${t===1?``:`s`} it could have shortened`:`no prefill the disk could have shortened yet`}function Dn(e){let t=e=>typeof e==`number`?e.toLocaleString(`en-US`):String(e),n=[];return e.tokens!=null&&n.push(`${t(e.tokens)} tokens`),e.kind===`hit`&&e.read!=null&&n.push(`${t(e.read)} read`,`${t(e.ms)} ms`),e.kind===`write`&&(n.push(`${t(e.chunks)} chunk${e.chunks===1?``:`s`}`,`${t(e.anchors)} anchor${e.anchors===1?``:`s`}`),e.dropped&&typeof e.dropped==`object`&&n.push(`dropped ${Object.entries(e.dropped).map(([e,t])=>`${t} ${e}`).join(`, `)}`)),e.why!=null&&n.push(String(e.why)),e.kind===`clear`&&n.push(`${t(e.entries)} entries`),e.error!=null&&n.push(String(e.error)),[e.kind,...n].join(` · `)}var On=e=>_(new Date(e*1e3).toISOString()),kn=e=>Object.entries(e??{}).filter(([,e])=>e).map(([e,t])=>`${c(t)} ${e}`).join(`, `)||`none`,J=class extends r{constructor(...e){super(...e),this.disk=null,this.error=null,this.busy=!1,this.cleared=null,this.settings=new w(this,`settings`)}connectedCallback(){super.connectedCallback(),this.load(),this.settings.load(),this.settings.onSaved=()=>void this.load()}async load(){try{this.disk=await N(()=>A.memoryDisk()),this.error=null}catch(e){this.error=t(e).message}}async clear(){let e=this.disk;if(e&&await O(`Clear the disk cache?`,[`Every entry goes: ${c(e.entries??0)} prefixes in ${c(e.chunks??0)} chunk files, ${v(e.bytes??0)}, and the counts that decide what is written next.`,`Nothing a request needs is lost: the next prefill of each prefix runs cold, as without the tier, and writes it again.`],`Clear`)){this.busy=!0;try{let e=await N(()=>A.memoryDiskClear());this.cleared=`cleared ${c(e.entries)} entries, ${v(e.bytes)}`,this.disk=e.disk}catch(e){this.error=t(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.disk)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.disk)return m(5);let e=this.disk;return p`${e.running?this.renderRunning(e):this.renderOff(e)}${this.renderSettings(e)}`}renderOff(e){return C(`Disk cache`,g(`The disk tier does not run on this server`,p`It starts with the engine when <code>QSE_DISK_CACHE</code> in serve.env names its directory (the flag is <code class="mono">${e.flag}</code> now)${e.free_bytes==null?f:p`; the NVMe has <span class="num">${v(e.free_bytes,0)}</span> free`}. Its settings below apply once it runs.`),{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM`})}renderRunning(e){let t=e.bytes??0,n=e.budget??0,r=n?t/n:0,a=e.free_bytes??null,o=a!=null&&e.floor!=null&&a<e.floor+1e10,s=p`<div class="stats disk-stats" data-disk-tiles>
      ${S({label:`on disk`,value:v(t,1),sub:p`of <span class="num">${v(n,0)}</span> · ${c(e.entries??0)} prefix${e.entries===1?``:`es`}, ${c(e.chunks??0)} chunk files`,warn:r>.95})}
      ${S({label:`NVMe free`,value:a==null?`—`:v(a,0),sub:p`never under <span class="num">${v(e.floor??5e10,0)}</span>`,warn:o})}
      ${S({label:`hit rate`,value:e.hit_rate==null?`—`:i(e.hit_rate,0),sub:En(e),na:e.hit_rate==null})}
      ${S({label:`restored`,value:c(e.tokens_restored??0),unit:`tokens`,sub:e.last_restore_ms==null?`no restore yet`:p`last <span class="num">${c(Math.round(e.last_restore_ms))}</span> ms${e.last_read_gbs==null?f:p` at <span class="num">${e.last_read_gbs.toFixed(1)}</span> GB/s`}`})}
    </div>`,l=p`<div class="meter-block disk-meter">
      <div class="meter-head"><span>budget</span><span class="num">${v(t,1)} of ${v(n,0)} · ${i(r,0)}</span></div>
      <div class="meter" role="img" aria-label=${`disk cache ${i(r,0)} of its budget`}><span class="meter-fill ${r>.95?`is-warn`:``}" style="width:${Math.min(100,r*100).toFixed(1)}%"></span></div>
    </div>`,u=e.top??[],d=p`${s}${l}
      <p class="small muted disk-line" data-disk-counters>
        ${e.on?p`<span class="badge badge-on">on</span>`:p`<span class="badge">off (settings)</span>`}
        written: <span class="num">${c(e.writes_chunk??0)}</span> chunks, <span class="num">${c(e.writes_anchor??0)}</span> anchors, <span class="num">${v(e.bytes_written??0,1)}</span> ·
        evicted: ${kn(e.evicted)} · dropped: ${kn(e.dropped)}${e.failed_restores?p` · <span class="warn-ink">${c(e.failed_restores)} failed restore${e.failed_restores===1?``:`s`}</span>`:f}
        · staging buffer <span class="num">${v(e.staging_bytes||e.staging_planned||0,2)}</span>${e.staging_bytes?``:` (when first used)`} · <span class="mono">${e.root??``}</span>
      </p>
      ${u.length?this.renderEntries(u):g(`Nothing on disk yet`,`A prefix is written after ${e.settings.admit_after===1?`its first prefill`:`${e.settings.admit_after} prefills`} of at least ${c(e.min_tokens??4096)} tokens, and the first long request of an opencode session is written where its user message starts.`)}
      ${this.renderEvents(e)}`;return C(`Disk cache`,d,{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM; a hit is read only past the rows already in place`,tools:p`${this.cleared?p`<span class="small muted" data-disk-cleared>${this.cleared}</span>`:f}
        <button class="btn btn-sm disk-clear" ?disabled=${this.busy||!(e.entries||e.chunks)} @click=${()=>this.clear()} data-disk-clear>${x(`x`)} Clear disk cache</button>
        <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the disk cache">${x(`refresh`)}</button>`})}renderEntries(e){let t=e=>p`<span class="badge disk-kind disk-kind-${e.kind}">${Sn[e.kind]}</span>${e.label?p` <span class="mono small">${e.label}</span>`:f}${e.complete?f:p` <span class="badge badge-warn" title="a chunk below it is not on disk: it is not read, and goes first">incomplete</span>`}`,n=e=>e.preview?p`<div class="small muted disk-preview" data-disk-preview>${e.preview}</div>`:f;return p`<h3 class="disk-sub">Most-hit prefixes</h3>
      <div class="table-scroll mem-table"><table class="table" data-disk-entries>
        <thead><tr><th class="num-col">tokens</th><th>what</th><th class="num-col">hits</th><th>last used</th><th class="num-col">a hit reads</th></tr></thead>
        <tbody>${e.map(e=>p`<tr data-disk-row=${`${e.kind}-${e.tokens}`}><td class="num num-col">${c(e.tokens)}</td><td class="cell-wrap">${t(e)}${n(e)}</td>
            <td class="num num-col">${c(e.hits)}</td><td class="num">${On(e.used)}</td><td class="num num-col">${v(e.reads_bytes,1)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards" role="list">${e.map(e=>p`<div class="mem-card disk-card" role="listitem" data-disk-card=${`${e.kind}-${e.tokens}`}>
          <div class="mem-card-head"><span class="num">${c(e.tokens)} tokens</span><span class="num small">${c(e.hits)} hit${e.hits===1?``:`s`}</span></div>
          <div>${t(e)}</div>${n(e)}
          <div class="small muted">last used ${On(e.used)} · a hit reads ${v(e.reads_bytes,1)}</div>
        </div>`)}</div>`}renderEvents(e){let t=(e.events??[]).slice(-10).reverse();return t.length?p`<details class="sec" style="margin-top:12px" data-disk-events>
      <summary>Last events <span class="sec-sub">what it read, wrote, dropped and evicted (the journal keeps them, kind memory)</span></summary>
      <ul class="disk-events">${t.map(e=>p`<li class="small"><span class="num muted">${On(e.ts)}</span> <span class="disk-ev disk-ev-${e.kind}">${Dn(e)}</span></li>`)}</ul>
    </details>`:f}renderSettings(e){let t=this.settings;if(!t.draft)return t.loadError?p`<p class="err-ink small">${t.loadError}</p>`:f;let n=Cn(t.draft),r=(e,r)=>t.set(T(t.draft,[`memory`,`disk_cache`],{...n,[e]:r})),i=(e,t,i,a,o=1)=>p`<label class="field"><span>${t}</span><input type="number" inputmode="decimal" min=${i} max=${a} step=${o} .value=${String(n[e])} data-disk-setting=${e} @input=${t=>r(e,Number(t.target.value))} /></label>`,a=Tn(n,e),o=wn(n);return p`<details class="sec" style="margin-top:16px" data-sec="disk" open>
      <summary>Disk cache settings <span class="sec-sub">on or off, its size on the NVMe, how long it keeps an entry, what it writes</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-disk-mode><span class="muted small">disk cache</span>${j([`on`,`off`],n.mode,e=>r(`mode`,e),{small:!0,label:`disk cache mode`})}<span class="small">${n.mode===`on`?`prefixes are read from and written to the disk`:`nothing is read or written; the files stay`}</span></div>
        <div class="form-grid">
          ${i(`budget_gb`,`cache budget (GB)`,1,4e3)}
          ${i(`ttl_days`,`drop an entry unused for (days)`,1,90)}
          ${i(`admit_after`,`write a prefix after it was prefilled (times)`,1,10)}
          ${i(`min_free_gb`,`keep free on the NVMe (GB, at least 50)`,50,4e3)}
        </div>
        <p class="small" data-disk-free>
          ${e.free_bytes==null?f:p`The NVMe has <span class="num">${v(e.free_bytes,0)}</span> free${e.bytes?p`, and the cache holds <span class="num">${v(e.bytes,1)}</span>`:f}. `}
          ${a.short?p`<span class="warn-ink" data-disk-short>${x(`warn`)} At most <span class="num">${a.usable_gb.toFixed(0)} GB</span> of the ${n.budget_gb} GB fit above the ${Math.max(50,n.min_free_gb)} GB it keeps free; the tier stops there.</span>`:f}
        </p>
        <p class="muted small">Least valuable first when it is full: hits, times the tokens an entry saves per byte it holds, halved for every day unused. The first long request of an opencode session is written whatever the count, at the lines below where its user message starts. Applied from the next request; the free space it keeps is never under ${50} GB, whatever is set here.</p>
        ${o.length?p`<ul class="errors" role="alert" data-disk-errors>${o.map(e=>p`<li>${e}</li>`)}</ul>`:f}
        ${k(t)}
        ${M(t,`the settings`)}
      </div>
    </details>`}};d([o()],J.prototype,`disk`,void 0),d([o()],J.prototype,`error`,void 0),d([o()],J.prototype,`busy`,void 0),d([o()],J.prototype,`cleared`,void 0),J=d([a(`qse-cond-disk`)],J);var Y=e=>e?_(e):`—`,An=()=>new Date().toISOString().slice(0,10),X=class extends r{constructor(...e){super(...e),this.selected=null,this.node=null,this.user=`owner`,this.list=null,this.one=null,this.error=null,this.oneError=null,this.filter={...Gt},this.query=``,this.hits=null,this.injected=null,this.writes=null,this.busy=null,this.editing=null,this.raw=!1,this.diffAt=null,this.confirmUser=``,this.flash=null,this.settings=new w(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.reload(),this.settings.load()}willUpdate(e){e.has(`node`)&&(this.filter={...this.filter,node:this.node});let t=this.selected?`${this.user}#${this.selected}`:null;t!==this.loadedFor&&(this.loadedFor=t,this.one=null,this.oneError=null,this.editing=null,this.diffAt=null,this.selected&&this.loadOne(this.selected))}async reload(){try{let[e,t,n]=await N(()=>Promise.all([A.notes(this.user),A.notesInjected(this.user,30),A.notesWrites(30)]));this.list=e,this.injected=t.rows,this.writes=n.rows,this.error=null}catch(e){this.error=t(e).message}}async loadOne(e){try{let t=await N(()=>A.note(e,this.user));e===this.selected&&(this.one=t)}catch(n){e===this.selected&&(this.oneError=t(n).message)}}async act(e,n,r){this.busy=e,this.flash=null;try{await N(n),r&&(this.flash=r),await this.reload(),this.selected&&await this.loadOne(this.selected)}catch(e){this.oneError=t(e).message}finally{this.busy=null}}async search(){let e=this.query.trim();if(!e){this.hits=null;return}try{this.hits=(await N(()=>A.notesSearch(e,this.user,10))).hits}catch(e){this.error=t(e).message}}async download(){try{let e=await N(()=>A.notesExport(this.user)),t=document.createElement(`a`);t.href=URL.createObjectURL(e),t.download=`memory-${this.user}.zip`,t.click(),setTimeout(()=>URL.revokeObjectURL(t.href),1e3)}catch(e){this.error=t(e).message}}open(e){u({note:e})}render(){if(this.error&&!this.list)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.list)return m(6);let e=this.list;return p`
      ${this.renderTiles(e)}
      ${this.flash?p`<p class="ok-ink small" role="status" data-notes-flash>${x(`check`)} ${this.flash}</p>`:f}
      ${this.selected?this.renderOne():f}
      ${this.renderSearch()}
      ${this.renderBrowser(e)}
      ${this.renderInjected()}
      ${this.renderRuns()}
      ${this.renderWrites()}
      ${this.renderSettings()}
      ${this.renderDanger(e)}
    `}renderTiles(e){let t=new Map;for(let n of e.notes)t.set(n.kind,(t.get(n.kind)??0)+1);let n=e.notes.filter(e=>e.status===`provisional`).length,r=An(),i=(this.injected??[]).filter(e=>new Date(e.ts_ms).toISOString().slice(0,10)===r).length,a=(this.writes??[]).filter(e=>new Date(e.ts_ms).toISOString().slice(0,10)===r&&/^refused/.test(Yt(e))).length,o=[...t.entries()].sort((e,t)=>t[1]-e[1]).map(([e,t])=>`${t} ${e.replace(`_`,` `)}`).join(` · `);return p`<div class="stats notes-stats" data-notes-tiles>
      ${S({label:`live notes`,value:c(e.notes.length),sub:o||`none yet`})}
      ${S({label:`provisional`,value:c(n),sub:p`<span class="num">${c(e.notes.length-n)}</span> ratified by you`,warn:n>0&&n===e.notes.length&&e.notes.length>5})}
      ${S({label:`injected today`,value:c(i),unit:`requests`,sub:p`<span class="num">${c(e.memory.index_notes+e.memory.body_notes)}</span> notes since the start`})}
      ${S({label:`refused today`,value:c(a),sub:p`<span class="num">${c(e.memory.taken)}</span> blocks written, <span class="num">${c(e.memory.failed)}</span> refused since the start`})}
    </div>`}renderSearch(){return C(`Search`,this.hits===null?p`<p class="muted small">The runtime's own search (BM25 over title, tags and body, every scope of ${this.user}): the scores are the ones that decide what goes into a request.</p>`:this.hits.length?p`<ol class="notes-hits" data-notes-hits>${this.hits.map(e=>p`<li><button class="link-btn" @click=${()=>this.open(e.id)} data-hit=${e.id}>${e.title}</button>
                <span class="badge">${e.kind}</span> <span class="badge">${e.scope}</span> <span class="num small" data-score>${e.score.toFixed(2)}</span>
                ${e.status===`provisional`?p`<span class="badge badge-shadow">provisional</span>`:f}</li>`)}</ol>`:g(`No match`,`No note shares a word with the query.`),{tools:p`<form class="toolbar notes-search" style="margin:0" @submit=${e=>{e.preventDefault(),this.search()}}><label class="field field-search">${x(`search`)}<input placeholder="words, as the runtime would search them" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search the notes" data-notes-query /></label>
          <button class="btn btn-sm" type="submit" data-notes-search>Search</button>${this.hits?p`<button class="btn btn-sm btn-ghost" type="button" @click=${()=>(this.hits=null,this.query=``)}>Clear</button>`:f}</form>`,id:`notes-search`})}renderBrowser(e){let t=Kt(e.notes,this.filter),n=e=>u({nnode:e,note:null}),r=p`<nav class="notes-tree" aria-label="memory scopes" data-notes-tree>
      <button class="notes-node ${this.node?``:`is-on`}" @click=${()=>n(null)} data-node="all"><span>All notes</span><span class="num muted">${e.notes.length}</span></button>
      ${e.tree.map(e=>p`<button class="notes-node ${this.node===e.node?`is-on`:``}" @click=${()=>n(e.node)} data-node=${e.node} title=${e.label??e.node}>
          <span class="notes-node-label">${e.provisional?p`<span class="dot dot-shadow" aria-label="has provisional notes"></span>`:f}${qt(e)}</span>
          <span class="num muted">${e.count}</span></button>`)}
    </nav>`,i=p`<label class="field notes-picker"><span>scope</span><select @change=${e=>n(e.target.value||null)} aria-label="memory scope" data-notes-picker>
        <option value="" ?selected=${!this.node}>All notes (${e.notes.length})</option>
        ${e.tree.map(e=>p`<option value=${e.node} ?selected=${this.node===e.node}>${qt(e)} (${e.count})</option>`)}
      </select></label>`,a=this.filter,o=e=>this.filter={...this.filter,...e},s=p`<div class="toolbar notes-filters" style="margin:0 0 8px">
      <select aria-label="kind" @change=${e=>o({kind:e.target.value})} data-filter="kind"><option value="">every kind</option>${Wt.map(e=>p`<option value=${e} ?selected=${a.kind===e}>${e.replace(`_`,` `)}</option>`)}</select>
      <select aria-label="status" @change=${e=>o({status:e.target.value})} data-filter="status"><option value="">every status</option><option value="provisional">provisional</option><option value="ratified">ratified</option></select>
      <select aria-label="writer" @change=${e=>o({writer:e.target.value})} data-filter="writer"><option value="">every writer</option>${Object.entries(U).map(([e,t])=>p`<option value=${e}>${t}</option>`)}</select>
      <label class="switch small"><input type="checkbox" .checked=${a.stale} @change=${e=>o({stale:e.target.checked})} data-filter="stale" /> stale</label>
      <label class="field field-search">${x(`search`)}<input placeholder="filter by title or tag" .value=${a.q} @input=${e=>o({q:e.target.value})} aria-label="filter the list" /></label>
    </div>`,l=e=>p`<span class="badge ${e.status===`ratified`?`badge-ok`:`badge-shadow`}">${e.status}</span>${e.pinned?p` <span class="badge badge-on">pinned</span>`:f}${e.stale?p` <span class="badge badge-warn">stale</span>`:f}`,d=p`<div class="table-scroll notes-table"><table class="table" data-notes-list>
        <thead><tr><th>title</th><th>kind</th><th>status</th><th>updated</th><th>writer</th><th class="num-col">uses 7d</th></tr></thead>
        <tbody>${t.map(e=>p`<tr class="req-row ${e.id===this.selected?`is-open`:``}" data-note-row=${e.id} @click=${()=>this.open(e.id)}>
            <td class="cell-wrap">${e.title}</td><td><span class="badge" title=${Ut[e.kind]??``}>${e.kind.replace(`_`,` `)}</span></td><td>${l(e)}</td>
            <td class="num small">${Y(e.updated)}</td><td class="small">${U[e.writer]??e.writer}</td><td class="num num-col">${c(e.uses_7d??0)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards notes-cards" role="list">${t.map(e=>p`<div class="mem-card ${e.id===this.selected?`is-open`:``}" role="listitem" data-note-card=${e.id} @click=${()=>this.open(e.id)}>
          <div class="mem-card-head"><span class="small">${e.title}</span></div>
          <div class="small">${e.kind.replace(`_`,` `)} · ${l(e)}</div>
          <div class="small muted">${Y(e.updated)} · ${U[e.writer]??e.writer} · ${c(e.uses_7d??0)} uses in 7 days</div>
        </div>`)}</div>`,m=e.notes.length?g(`Nothing here`,`No note in this scope matches the filters.`):g(`No notes yet`,e.store.refused?`The memory directory is inside a git work tree (${e.store.refused}), so nothing is written.`:`The model writes a note when you ask it to remember something, correct it, or decide something; with the profile's memory switch on or in shadow.`);return C(`Notes`,p`<div class="notes-browser">${r}<div class="notes-main">${i}${s}${t.length?d:m}</div></div>`,{sub:p`<span data-notes-root>${c(e.notes.length)} live · ${c(e.archived.length)} archived${e.duplicates.length?p` · <span class="warn-ink">${e.duplicates.length} possible duplicate${e.duplicates.length===1?``:`s`}</span>`:f} · <span class="mono">${e.store.root}/${this.user}</span></span>`,tools:p`${e.users.length>1?p`<select aria-label="user" @change=${e=>(this.user=e.target.value,u({note:null,nnode:null}),void this.reload())} data-notes-user>${e.users.map(e=>p`<option ?selected=${e===this.user}>${e}</option>`)}</select>`:f}
          <button class="btn btn-sm" @click=${()=>this.download()} data-notes-export>${x(`down`)} Export</button>
          <button class="btn btn-sm" @click=${()=>this.reload()} aria-label="reload the notes">${x(`refresh`)}</button>`,id:`notes`})}renderOne(){if(this.oneError&&!this.one)return p`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${x(`warn`)} ${this.oneError}</div></div>`;let e=this.one;if(!e)return m(4);let t=e.note,n=!!t.archived,r=t.status===`ratified`&&(t.kind===`rule`||t.kind===`ruling`),i=e=>this.busy===e,a=t.id,o=n?p`<button class="btn btn-sm btn-danger" ?disabled=${i(`delete`)} @click=${()=>this.del(t)} data-note-act="delete">${x(`trash`)} Delete now</button>`:p`<button class="btn btn-sm" @click=${()=>this.startEdit(t)} data-note-act="edit">${x(`edit`)} Edit</button>
          ${t.status===`provisional`?p`<button class="btn btn-sm" ?disabled=${i(`ratify`)} @click=${()=>this.act(`ratify`,()=>A.noteStatus(a,{status:`ratified`},this.user),`ratified`)} data-note-act="ratify">${x(`check`)} Ratify</button>`:f}
          ${t.status===`provisional`?p`<button class="btn btn-sm" ?disabled=${i(`reject`)} @click=${()=>this.act(`reject`,()=>A.noteArchive(a,`rejected in the dashboard`,this.user),`rejected and archived`)} data-note-act="reject">Reject</button>`:f}
          ${r?p`<button class="btn btn-sm" ?disabled=${i(`pin`)} @click=${()=>this.act(`pin`,()=>A.noteStatus(a,{pinned:!t.pinned},this.user),t.pinned?`unpinned`:`pinned to every request`)} data-note-act="pin">${t.pinned?`Unpin`:`Pin to every request`}</button>`:f}
          <button class="btn btn-sm" ?disabled=${i(`archive`)} @click=${()=>this.archive(t)} data-note-act="archive">Archive</button>
          <button class="btn btn-sm btn-danger" ?disabled=${i(`delete`)} @click=${()=>this.del(t)} data-note-act="delete">${x(`trash`)} Delete</button>`,s=p`<dl class="notes-meta" data-note-meta>
      <dt>kind</dt><dd><span class="badge">${t.kind.replace(`_`,` `)}</span> <span class="muted small">${Ut[t.kind]??``}</span></dd>
      <dt>scope</dt><dd>${t.scope}${t.project?p` · <span class="mono small">${t.project}</span>`:f}${t.session?p` · <a class="mono small" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(t.session)}`}>${t.session}</a>`:f}</dd>
      <dt>status</dt><dd>${t.status}${t.pinned?` · pinned`:``}${t.stale?p` · <span class="warn-ink">stale since ${t.verify_by}</span>`:f}${n?p` · <span class="warn-ink">archived ${Y(t.archived)}: ${t.reason}</span>`:f}</dd>
      <dt>writer</dt><dd>${U[t.writer]??t.writer}</dd>
      <dt>tags</dt><dd>${t.tags.length?t.tags.map(e=>p`<span class="badge">${e}</span> `):p`<span class="muted">—</span>`}</dd>
      ${t.verify_by?p`<dt>verify by</dt><dd class="num">${t.verify_by}</dd>`:f}
      <dt>created</dt><dd class="num">${Y(t.created)}</dd><dt>updated</dt><dd class="num">${Y(t.updated)}</dd>
      <dt>sources</dt><dd>${t.sources.length?t.sources.map(e=>p`<span class="mono small">${e}</span> `):p`<span class="muted">—</span>`}</dd>
      <dt>used</dt><dd data-note-uses>${Qt(e.uses,An())}</dd>
    </dl>`,c=this.editing?this.renderEditor():p`<div class="notes-body">
        <div class="toolbar" style="margin:0 0 6px">${j([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`body view`})}</div>
        ${this.raw?p`<pre class="mem-text" data-note-raw>${t.body}</pre>`:p`<div class="notes-rendered" data-note-body>${jn(t.body)}</div>`}
      </div>`;return p`<section class="panel notes-one" style="margin-bottom:16px" data-note=${a}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${t.title}</h2><p class="panel-sub mono">${a}</p></div>
        <div class="panel-tools notes-actions">${o}<button class="btn btn-ghost btn-icon" aria-label="Close the note" @click=${()=>u({note:null})}>${x(`x`)}</button></div></header>
      ${this.oneError?p`<p class="err-ink small" role="alert">${x(`warn`)} ${this.oneError}</p>`:f}
      <div class="notes-file">${s}${c}</div>
      ${this.renderHistory(e)}
    </section>`}startEdit(e){this.editing={title:e.title,body:e.body,kind:e.kind,tags:e.tags.join(`, `),verify_by:e.verify_by??``}}renderEditor(){let e=this.editing,t=t=>this.editing={...e,...t},n=e.body.trim()?e.body.trim().split(/\s+/).length:0,r=e.title.length>120||n>400||!e.body.trim();return p`<div class="notes-editor" data-note-editor>
      <label class="field"><span>title <span class="muted num">${e.title.length}/120</span></span><input .value=${e.title} @input=${e=>t({title:e.target.value})} data-edit="title" /></label>
      <div class="form-grid">
        <label class="field"><span>kind</span><select @change=${e=>t({kind:e.target.value})} data-edit="kind">${Wt.map(t=>p`<option value=${t} ?selected=${t===e.kind}>${t.replace(`_`,` `)}</option>`)}</select></label>
        <label class="field"><span>tags (comma)</span><input .value=${e.tags} @input=${e=>t({tags:e.target.value})} data-edit="tags" /></label>
        <label class="field"><span>verify by</span><input type="date" .value=${e.verify_by} @input=${e=>t({verify_by:e.target.value})} data-edit="verify_by" /></label>
      </div>
      <label class="field"><span>body <span class="muted num">${n}/400 words</span></span><textarea rows="8" .value=${e.body} @input=${e=>t({body:e.target.value})} data-edit="body"></textarea></label>
      <div class="toolbar" style="margin:0"><button class="btn btn-primary btn-sm" ?disabled=${r||this.busy===`edit`} @click=${()=>this.act(`edit`,async()=>{await A.noteEdit(this.selected,{title:e.title,body:e.body,kind:e.kind,tags:e.tags.split(`,`).map(e=>e.trim()).filter(Boolean),verify_by:e.verify_by||null},this.user),this.editing=null},`saved as a new revision, ratified`)} data-edit-save>Save as a revision</button><button class="btn btn-sm btn-ghost" @click=${()=>this.editing=null}>Cancel</button>
        ${r?p`<span class="warn-ink small">a title of at most 120 characters, a body of 1 to 400 words</span>`:p`<span class="muted small">Your edit ratifies the note; the old text stays in its history.</span>`}</div>
    </div>`}renderHistory(e){let t=e.revisions,n=e.note,r=[...t.map(e=>({label:`revision ${e.n} · ${Y(e.updated)} · ${U[e.writer]??e.writer}`,title:e.title,body:e.body})),{label:`now · ${Y(n.updated)}`,title:n.title,body:n.body}],i=this.diffAt,a=f;if(i!==null&&i+1<r.length){let e=r[i],t=r[i+1],n=Jt(`# ${e.title}\n${e.body}`,`# ${t.title}\n${t.body}`);a=p`<pre class="notes-diff" data-note-diff>${n.map(e=>p`<span class="diff-${e.op===`+`?`add`:e.op===`-`?`del`:`eq`}">${e.op===`=`?` `:e.op} ${e.text}\n</span>`)}</pre>`}return p`<details class="sec" ?open=${t.length>0} data-note-history>
      <summary>History <span class="sec-sub">${n.changelog.length} change${n.changelog.length===1?``:`s`}, ${t.length} earlier text${t.length===1?``:`s`}</span></summary>
      <div class="sec-body">
        <ul class="notes-changelog">${n.changelog.map(e=>p`<li class="small">${e}</li>`)}</ul>
        ${t.length?p`<ol class="notes-revs">${r.slice(0,-1).map((e,t)=>p`<li><button class="link-btn ${i===t?`is-on`:``}" @click=${()=>this.diffAt=i===t?null:t} data-rev=${t}>${e.label}</button> <span class="muted small">to the next</span></li>`)}</ol>${a}`:p`<p class="muted small">No earlier text: this is the note as it was written.</p>`}
      </div>
    </details>`}async archive(e){let t=window.prompt(`Why is this note archived? (kept with it for 30 days)`,`no longer true`);t!==null&&await this.act(`archive`,()=>A.noteArchive(e.id,t||`archived in the dashboard`,this.user),`archived`)}async del(e){await O(`Delete this note?`,[p`<strong>${e.title}</strong>`,`The file, its history and any archived copy go at once. This cannot be undone.`],`Delete`)&&(await this.act(`delete`,()=>A.noteDelete(e.id,this.user),`deleted`),u({note:null}))}renderInjected(){let e=this.injected??[];return p`<details class="sec" style="margin-top:16px" data-notes-injected>
      <summary>Injected <span class="sec-sub">the last requests the notes went into (the index once a session, the matching notes on a new message)</span></summary>
      <div class="sec-body">${e.length?p`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>session</th><th>what</th><th>notes</th></tr></thead><tbody>${e.map(e=>p`<tr><td class="num small">${_(new Date(e.ts_ms).toISOString())}</td>
              <td class="mono small cell-wrap">${e.session_key?p`<a href=${`#/conductor?screen=memory&memory=${encodeURIComponent(e.session_key)}`}>${e.session_key}</a>`:`—`}</td>
              <td class="small">${e.index?`index`:`bodies`}</td>
              <td class="cell-wrap">${e.ids.map(e=>p`<button class="link-btn mono small" @click=${()=>this.open(e)}>${e}</button> `)}</td></tr>`)}</tbody></table></div>`:p`<p class="muted small">No note went into a request yet.</p>`}</div>
    </details>`}renderRuns(){let e=(this.writes??[]).filter(e=>e.task===`writer`).map(Zt);return p`<details class="sec" data-notes-runs ?open=${e.length>0}>
      <summary>Writer runs <span class="sec-sub">the model's hidden turns that read a session and write its notes: after a compaction, after a give-up, at the session's end</span></summary>
      <div class="sec-body">${e.length?p`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>trigger</th><th>Laya's gate</th><th>outcome</th><th class="num-col">seconds</th><th class="num-col">tokens</th><th>notes</th><th>refused</th></tr></thead><tbody>${e.map(e=>p`<tr data-run=${e.trigger}><td class="num small">${_(new Date(e.when).toISOString())}</td><td>${e.trigger.replace(`_`,` `)}</td><td class="small">${e.gate}</td><td class="small">${e.outcome}</td>
              <td class="num num-col">${e.seconds??`—`}</td><td class="num num-col">${e.tokens??`—`}</td><td class="small">${e.ops}</td><td class="small warn-ink">${e.refused.join(`, `)}</td></tr>`)}</tbody></table></div>`:p`<p class="muted small">No run yet. The writer is ${this.list?.settings.writer?.mode??`off`} (settings below).</p>`}</div>
    </details>`}renderWrites(){let e=(this.writes??[]).filter(e=>e.task!==`writer`);return p`<details class="sec" data-notes-writes>
      <summary>Writes <span class="sec-sub">every block the model wrote, the script calls, your edits, the nightly job, and what was refused and why</span></summary>
      <div class="sec-body">${e.length?p`<ul class="notes-writes">${e.map(e=>p`<li class="small ${/^refused/.test(Yt(e))?`warn-ink`:``}"><span class="num muted">${_(new Date(e.ts_ms).toISOString())}</span> ${Yt(e)}</li>`)}</ul>`:p`<p class="muted small">Nothing written yet.</p>`}</div>
    </details>`}renderSettings(){let e=this.settings;if(!e.draft)return e.loadError?p`<p class="err-ink small">${e.loadError}</p>`:f;let t=Xt(e.draft),n=(n,r)=>e.set(T(e.draft,[`memory`,`notes`],{...t,[n]:r})),r=(e,r,i,a)=>p`<label class="field"><span>${r}</span><input type="number" min=${i} max=${a} .value=${String(t[e])} data-notes-setting=${e} @input=${t=>n(e,Math.round(Number(t.target.value)))} /></label>`;return p`<details class="sec" data-sec="notes-settings">
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
        ${k(e)}
        ${M(e,`the settings`)}
      </div>
    </details>`}renderWriterSettings(e,t){let n=e.writer,r=(e,r)=>t(`writer`,{...n,[e]:r}),i=(e,t,i,a)=>p`<label class="field"><span>${t}</span><input type="number" min=${i} max=${a} .value=${String(n[e])} data-writer-setting=${e} @input=${t=>r(e,Number(t.target.value))} /></label>`;return p`<h4 class="small" style="margin:12px 0 4px">The background writer</h4>
      <div class="toolbar" style="margin:0" data-writer-mode><span class="muted small">writer</span>${j([`off`,`shadow`,`on`],n.mode,e=>r(`mode`,e),{small:!0,label:`writer mode`})}
        <span class="muted small">after a compaction</span>${j([`off`,`idle`,`turn`],n.after_compaction,e=>r(`after_compaction`,e),{small:!0,label:`after a compaction`})}</div>
      <div class="form-grid">
        ${i(`idle_s`,`a session's end: seconds without a request`,10,86400)}
        ${i(`min_new_tokens`,`new tokens before the next pass`,0,1e6)}
        ${i(`max_tokens`,`the turn writes at most (tokens)`,64,4096)}
      </div>
      <label class="switch small"><input type="checkbox" .checked=${n.gate} @change=${e=>r(`gate`,e.target.checked)} /> Laya decides first whether the session has anything worth keeping (about 20 ms; a no skips the turn)</label>
      <p class="muted small">shadow: the turn runs and its notes are counted, nothing is stored. <code>turn</code> after a compaction writes before the next request gets the engine (the next turn waits up to ${n.turn_deadline_s} s); <code>idle</code> waits for an idle moment, which opencode rarely leaves right after a compaction.</p>`}renderDanger(e){let t=this.confirmUser===this.user;return p`<details class="sec" data-notes-danger>
      <summary>Delete ${this.user}'s memory <span class="sec-sub">every note, revision and archive of this user, at once</span></summary>
      <div class="sec-body">
        <p class="small">Type <code>${this.user}</code> to delete ${c(e.notes.length+e.archived.length)} notes. Export them first if you want a copy.</p>
        <div class="toolbar" style="margin:0"><label class="field"><span>user</span><input .value=${this.confirmUser} @input=${e=>this.confirmUser=e.target.value} data-notes-confirm aria-label="type the user's name" /></label>
          <button class="btn btn-sm btn-danger" ?disabled=${!t||this.busy===`user`} @click=${()=>this.act(`user`,()=>A.notesDeleteUser(this.user,this.confirmUser),`${this.user}'s memory deleted`).then(()=>this.confirmUser=``)} data-notes-delete-user>${x(`trash`)} Delete everything</button></div>
      </div>
    </details>`}};d([y({attribute:!1})],X.prototype,`selected`,void 0),d([y({attribute:!1})],X.prototype,`node`,void 0),d([o()],X.prototype,`user`,void 0),d([o()],X.prototype,`list`,void 0),d([o()],X.prototype,`one`,void 0),d([o()],X.prototype,`error`,void 0),d([o()],X.prototype,`oneError`,void 0),d([o()],X.prototype,`filter`,void 0),d([o()],X.prototype,`query`,void 0),d([o()],X.prototype,`hits`,void 0),d([o()],X.prototype,`injected`,void 0),d([o()],X.prototype,`writes`,void 0),d([o()],X.prototype,`busy`,void 0),d([o()],X.prototype,`editing`,void 0),d([o()],X.prototype,`raw`,void 0),d([o()],X.prototype,`diffAt`,void 0),d([o()],X.prototype,`confirmUser`,void 0),d([o()],X.prototype,`flash`,void 0),X=d([a(`qse-cond-notes`)],X);function jn(e){let t=[],n=(e??``).split(`
`),r=0,i=e=>e.split(/(`[^`]+`)/).map(e=>e.startsWith("`")&&e.endsWith("`")&&e.length>1?p`<code>${e.slice(1,-1)}</code>`:e);for(;r<n.length;){let e=n[r];if(e.startsWith("```")){let e=[];for(r++;r<n.length&&!n[r].startsWith("```");)e.push(n[r++]);r++,t.push(p`<pre class="mem-text">${e.join(`
`)}</pre>`)}else if(/^\s*[-*] /.test(e)){let e=[];for(;r<n.length&&/^\s*[-*] /.test(n[r]);)e.push(n[r++].replace(/^\s*[-*] /,``));t.push(p`<ul>${e.map(e=>p`<li>${i(e)}</li>`)}</ul>`)}else if(!e.trim())r++;else{let e=[];for(;r<n.length&&n[r].trim()&&!n[r].startsWith("```")&&!/^\s*[-*] /.test(n[r]);)e.push(n[r++]);t.push(p`<p>${i(e.join(` `))}</p>`)}}return p`${t}`}var Mn={tools:`requests with tools in full (and compactions); chats without tools as counters only`,all:`every request in full`,none:`nothing`},Z=class extends r{constructor(...e){super(...e),this.ids=[],this.selected=null,this.page=null,this.tab=null,this.note=null,this.nnode=null,this.list=null,this.one=null,this.error=null,this.oneError=null,this.query=``,this.busy=null,this.row=null,this.ev=null,this.rowError=null,this.settings=new w(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.tab!==`disk`&&this.tab!==`notes`&&(this.loadList(),this.settings.load())}updated(e){let t=e=>e===`disk`||e===`notes`;t(e.get(`tab`))&&!t(this.tab)&&!this.list&&!this.error&&(this.loadList(),this.settings.draft||this.settings.load())}willUpdate(){let e=this.selected?`${this.selected}#${this.page??``}`:null;e!==this.loadedFor&&(this.loadedFor=e,this.one=null,this.oneError=null,this.selected&&this.loadOne(this.selected,this.page))}async loadList(){try{this.list=await N(()=>A.memorySessions()),this.error=null}catch(e){this.error=t(e).message}}async loadOne(e,n){try{let t=await N(()=>A.memorySession(e,n));e===this.selected&&(this.one=t)}catch(n){e===this.selected&&(this.oneError=t(n).message)}}async keep(e,n){if(e.session){this.busy=e.dir;try{await N(()=>A.memoryKeep(e.session,n)),await this.loadList(),this.one&&this.one.session.session===e.session&&(this.one={...this.one,session:{...this.one.session,keep:n}})}catch(e){this.error=t(e).message}finally{this.busy=null}}}async download(e,n){try{let t=await N(()=>A.memoryExport(e,n)),r=document.createElement(`a`);r.href=URL.createObjectURL(t),r.download=`transcript-${e.replace(/[^a-z0-9]+/gi,`-`)}.${n}`,r.click(),setTimeout(()=>URL.revokeObjectURL(r.href),1e3)}catch(e){this.oneError=t(e).message}}async openRow(e){this.rowError=null;try{let t=(await A.decision(e)).decision;t.kind===`monitor`?this.ev=t:this.row=t}catch(n){this.rowError=`journal row ${e}: ${t(n).message} (the journal keeps 30 days)`}}render(){let e=this.tab===`disk`,t=this.tab===`disk`||this.tab===`notes`?this.tab:null,n=(e,n)=>p`<a class="cond-tab ${e===t?`is-on`:``}" href="#" aria-current=${e===t?`page`:f} data-mem-tab=${e??`sessions`} @click=${t=>{t.preventDefault(),u({mtab:e,memory:null,mpage:null,note:null,nnode:null})}}>${n}</a>`,r=p`<nav class="mem-tabs" aria-label="memory tabs">${n(null,`Sessions`)}${n(`notes`,`Notes`)}${n(`disk`,`Disk cache`)}</nav>`;return e?p`${r}<qse-cond-disk></qse-cond-disk>`:t===`notes`?p`${r}<qse-cond-notes .selected=${this.note} .node=${this.nnode}></qse-cond-notes>`:p`
      ${r}
      ${this.selected?this.renderOne(this.selected):f}
      ${this.renderList()}
      ${this.renderSettings()}
      ${this.row?p`<qse-cond-decision .row=${this.row} .ids=${this.ids} .close=${()=>this.row=null}></qse-cond-decision>`:f}
      ${this.ev?p`<qse-cond-event .ev=${this.ev} .close=${()=>this.ev=null}></qse-cond-event>`:f}
    `}renderList(){if(this.error&&!this.list)return p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.list)return m(5);let e=this.list.writer,t=this.query.toLowerCase(),n=this.list.sessions.filter(e=>!t||`${e.session??e.dir} ${e.client??``} ${e.project??``}`.toLowerCase().includes(t)),r=e=>e.last_ms?_(new Date(e.last_ms).toISOString()):`—`,i=t=>p`<label class="switch" title="kept sessions are exempt from the ${e.settings.retention_days}-day prune"><input type="checkbox" .checked=${t.keep} ?disabled=${!t.session||this.busy===t.dir} data-keep=${t.dir} @click=${e=>e.stopPropagation()} @change=${e=>this.keep(t,e.target.checked)} aria-label=${`keep ${bn(t.session,t.dir)}`} /> keep</label>`,a=e=>e.checkers?.last?p`<span class="badge ${/ PASS$/.test(e.checkers.last)?`badge-ok`:`badge-warn`}">${e.checkers.last}</span>${e.checkers.PASS||e.checkers.FAIL?p` <span class="muted small num">${e.checkers.PASS??0} pass · ${e.checkers.FAIL??0} fail</span>`:f}`:p`<span class="muted">—</span>`,o=e=>e.session&&u({memory:e.session,mpage:null});return C(`Session transcripts`,n.length?p`<div class="table-scroll mem-table"><table class="table" data-memory-sessions>
              <thead><tr><th>last request</th><th>session</th><th>client</th><th>profile</th><th>project</th><th class="num-col">turns</th><th class="num-col">acts</th><th>checker</th><th class="num-col">size</th><th>keep</th></tr></thead>
              <tbody>${n.map(e=>p`<tr class="req-row ${e.session&&e.session===this.selected?`is-open`:``}" data-memory-row=${e.dir} @click=${()=>o(e)}>
                  <td class="num">${r(e)}</td><td class="mono small cell-wrap">${bn(e.session,e.dir)}</td><td>${e.client??`—`}</td><td>${e.profile??`—`}</td>
                  <td class="small cell-wrap">${e.project??`—`}</td>
                  <td class="num num-col">${e.counts?c(e.counts.blocks):`?`}</td><td class="num num-col">${e.counts?c(e.counts.acts):`?`}</td>
                  <td>${a(e)}</td><td class="num num-col">${v(e.bytes??0)}</td><td>${i(e)}</td></tr>`)}</tbody></table></div>
            <div class="mem-cards" role="list">${n.map(e=>p`<div class="mem-card ${e.session&&e.session===this.selected?`is-open`:``}" role="listitem" data-memory-card=${e.dir} @click=${()=>o(e)}>
                <div class="mem-card-head"><span class="mono small">${bn(e.session,e.dir)}</span>${i(e)}</div>
                <div class="small">${r(e)} · ${e.client??`—`} · ${e.profile??`—`}</div>
                <div class="small muted">${vn(e)}</div>
                ${e.project?p`<div class="small muted cell-wrap">${e.project}</div>`:f}
                <div>${a(e)}</div>
              </div>`)}</div>`:g(`No transcripts yet`,e.refused?`The memory directory ${e.root} is inside a git work tree, so nothing is written there.`:`TandemLLM requests with tools are written to ${e.root} when their stream closes; chats without tools are counted only.`),{sub:p`<span data-writer>${c(e.blocks)} block${e.blocks===1?``:`s`} written since the start · ${c(e.redactions)} secret${e.redactions===1?``:`s`} redacted · ${c(e.failed)} failed · ${c(e.dropped)} dropped · ${v(e.bytes)} · <span class="mono">${e.root}</span></span>`,tools:p`<label class="field field-search">${x(`search`)}<input placeholder="search a session, client or project" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search transcripts" /></label>
          <button class="btn btn-sm" @click=${()=>this.loadList()} aria-label="reload the list">${x(`refresh`)}</button>`,id:`cond-memory`})}renderOne(e){if(this.oneError)return p`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${x(`warn`)} ${this.oneError}</div></div>`;let t=this.one;if(!t)return m(4);let n=t.session,r=t.pages??1,i=t.page??1,a=r>1?p`<div class="toolbar mem-pager" data-pager><button class="btn btn-sm" ?disabled=${i<=1} @click=${()=>u({mpage:String(i-1)})}>Older</button>
          <span class="small num">page ${i} of ${r} · ${c(t.total??t.blocks.length)} blocks, ${t.per_page} a page</span>
          <button class="btn btn-sm" ?disabled=${i>=r} @click=${()=>u({mpage:String(i+1)})}>Newer</button></div>`:f;return p`<section class="panel" style="margin-bottom:16px" data-transcript=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title mono">${bn(n.session,n.dir)}${n.keep?p` <span class="badge badge-on">kept</span>`:f}</h2>
        <p class="panel-sub">${n.client??`—`} · ${n.profile??`—`}${n.project?` · ${n.project}`:``} · ${vn(n)}${n.first_ms?` · since ${_(new Date(n.first_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <button class="btn btn-sm" @click=${()=>this.download(e,`md`)} data-mem-export="md">${x(`down`)} Markdown</button>
          <button class="btn btn-sm" @click=${()=>this.download(e,`json`)} data-mem-export="json">${x(`down`)} JSON</button>
          <a class="btn btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e)}`} data-mem-timeline>Timeline</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the transcript" @click=${()=>u({memory:null,mpage:null})}>${x(`x`)}</button></div></header>
      ${this.rowError?p`<p class="warn-ink small" role="alert">${x(`warn`)} ${this.rowError}</p>`:f}
      ${a}
      ${t.blocks.length?p`<div class="mem-blocks">${t.blocks.map(e=>this.renderBlock(e))}</div>`:g(`No blocks`,`This session was counted but not written: its requests carried no tools.`)}
      ${a}
    </section>`}renderBlock(e){let t=gn(e.text),n=t.runtime.filter(e=>e.kind===`hidden`).length,r=t.runtime.find(e=>e.kind===`acts`),i=p`<span class="mem-turn num">turn ${e.turn??`?`}</span>
      <span class="muted small num">${e.time??``}</span>
      ${e.class_id?D(e.class_id,this.ids):f}
      ${e.effort?p`<span class="badge">${e.effort}</span>`:f}
      ${e.reasoning_tokens?p`<span class="muted small num">${c(e.reasoning_tokens)} reasoning tokens</span>`:f}
      ${n?p`<span class="badge badge-shadow" data-hidden-count=${n}>${n} hidden</span>`:f}
      ${r?p`<span class="badge badge-warn">${r.text.replace(/^acts: /,`acts `)}</span>`:f}`;return e.compaction?p`<details class="mem-block is-compaction" data-block=${e.turn??``} data-compaction>
        <summary><span class="mem-block-head">${i}<span class="badge">compaction</span><span class="small">${t.compaction?.replace(/^compaction: /,``)??``}</span></span></summary>
        ${t.sections.map(e=>this.renderSection(e))}${this.renderRuntime(t.runtime)}
      </details>`:p`<article class="mem-block" data-block=${e.turn??``}>
      <div class="mem-block-head">${i}</div>
      ${t.sections.map(e=>this.renderSection(e))}
      ${this.renderRuntime(t.runtime)}
    </article>`}renderSection(e){let t=e.body.length>600||e.body.split(`
`).length>12,n=p`<pre class="mem-text">${e.body}</pre>`;return e.kind===`tool_result`||e.kind===`tool_call`&&t||e.kind===`reasoning`?p`<details class="mem-sec mem-${e.kind}" ?open=${!t&&e.kind!==`reasoning`} data-sec-kind=${e.kind}>
        <summary class="small">${e.title}${e.clipped?p` <span class="badge badge-warn">clipped</span>`:f}</summary>${n}</details>`:p`<div class="mem-sec mem-${e.kind}" data-sec-kind=${e.kind}><div class="small muted">${e.title}</div>${n}</div>`}renderRuntime(e){return e.length?p`<ul class="mem-runtime" aria-label="what the runtime did">${e.map(e=>p`<li class="mem-rt mem-rt-${e.kind} ${e.verdict?`is-${e.verdict.toLowerCase()}`:``}" data-rt=${e.kind}>
        ${e.kind===`hidden`?x(`info`):f}<span>${e.text.replace(/ · journal [A-Za-z0-9_-]+$/,``)}</span>
        ${e.journal?p`<button class="btn btn-ghost btn-sm" @click=${()=>this.openRow(e.journal)} data-journal=${e.journal} aria-label=${`open journal row ${e.journal}`}>journal row</button>`:f}
      </li>`)}</ul>`:f}renderSettings(){let e=this.settings;if(!e.draft)return e.loadError?p`<p class="err-ink small">${e.loadError}</p>`:f;let t=yn(e.draft),n=(n,r)=>e.set(T(e.draft,[`memory`,`transcripts`],{...t,[n]:r})),r=(e,r,i,a)=>p`<label class="field"><span>${r}</span><input type="number" min=${i} max=${a} .value=${String(t[e])} data-mem-setting=${e} @input=${t=>n(e,Math.round(Number(t.target.value)))} /></label>`;return p`<details class="sec" style="margin-top:16px" data-sec="transcripts">
      <summary>Transcript settings <span class="sec-sub">what the engine writes to the box, how long it keeps it, where tool output is clipped</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-mem-scope><span class="muted small">write</span>${j([`tools`,`all`,`none`],t.scope,e=>n(`scope`,e),{small:!0,label:`transcript scope`})}<span class="small">${Mn[t.scope]}</span></div>
        <div class="form-grid">
          ${r(`retention_days`,`keep a session for (days; kept sessions are exempt)`,1,400)}
          ${r(`clip_chars`,`clip a tool result or a call at (characters)`,200,1e5)}
          ${r(`loop_chars`,`a loop the D1 guard cut keeps its first and last (characters)`,0,4e3)}
        </div>
        <p class="muted small">Every block passes the secret-pattern refusal before it is written: a key, token or password becomes <code>[redacted:kind]</code> and the journal counts it, never the value. There is no switch for it. The files are <code>0600</code> under the engine's home on the box, never in a repository, and the ledger's backup never holds them. The owner's decision 2: chats without tools are counters only, unless the scope is widened here.</p>
        ${k(e)}
        ${M(e,`the settings`)}
      </div>
    </details>`}};d([y({attribute:!1})],Z.prototype,`ids`,void 0),d([y({attribute:!1})],Z.prototype,`selected`,void 0),d([y({attribute:!1})],Z.prototype,`page`,void 0),d([y({attribute:!1})],Z.prototype,`tab`,void 0),d([y({attribute:!1})],Z.prototype,`note`,void 0),d([y({attribute:!1})],Z.prototype,`nnode`,void 0),d([o()],Z.prototype,`list`,void 0),d([o()],Z.prototype,`one`,void 0),d([o()],Z.prototype,`error`,void 0),d([o()],Z.prototype,`oneError`,void 0),d([o()],Z.prototype,`query`,void 0),d([o()],Z.prototype,`busy`,void 0),d([o()],Z.prototype,`row`,void 0),d([o()],Z.prototype,`ev`,void 0),d([o()],Z.prototype,`rowError`,void 0),Z=d([a(`qse-cond-memory`)],Z);var Q=class extends r{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([A.stats(168),A.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async promote(e,n,r,i,a){if(await O(`Promote ${r} to ${n===`task`?`act`:`on`} in ${e}?`,[n===`task`?`The session monitor will act on ${r} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${r} decisions instead of only journaling them: ${r===`effort`?`budgets and thinking off go to the engine`:r===`packages`?`skills and packages go into the prompt`:r===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${i.toFixed(2)} over ${a} labelled decisions. A new profile version records it.`],`Promote`))try{let t=await A.doc(`profiles`,e),o=n===`task`?T(t.doc,[`tasks`,r,`mode`],`act`):T(t.doc,[`decisions`,r],`on`),s=await N(()=>A.put(`profiles`,e,o,`${r} to ${n===`task`?`act`:`on`}, precision ${i.toFixed(2)} / ${a}`,t.version));this.msg=`${e} v${s.version}: ${r} promoted`,this.onSaved()}catch(e){this.msg=t(e).message}}async loadAb(){try{this.abVersions=(await A.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await A.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await A.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=t(e).message}}render(){return this.error&&!this.stats?p`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`:this.stats?p`
      ${this.msg?p`<div class="note note-info" role="status">${this.msg}</div>`:f}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:m(8)}renderPromotion(e){let t=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,n=(e.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return C(`Shadow precision and promotion`,p`<p class="muted small">${He}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${e.precision.length||n.length?p`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${e.precision.map(e=>{let n=_e(e),r=t(e.profile,e.decision),i=e.backend?p`<div class="muted small mono" data-backend=${e.backend}>via ${e.backend}</div>`:f;return p`<tr data-precision-row=${`${e.profile}/${e.decision}${e.backend?`@${e.backend}`:``}`}>
                    <td>${e.profile}${i}</td><td class="mono">${e.decision}</td><td><span class="badge ${r===`on`?`badge-on`:r===`shadow`?`badge-shadow`:``}">${r}</span></td>
                    <td class="num num-col">${c(e.decisions)}</td><td class="num num-col">${c(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${n.allowed?``:`muted`}">${n.why}</td>
                    <td><button class="btn btn-sm ${n.allowed?`btn-primary`:``}" ?disabled=${!n.allowed||r!==`shadow`||!!e.backend} title=${e.backend?`a backend’s switches are on the Backends screen, per model`:n.allowed?``:He} data-promote=${`${e.profile}/${e.decision}`} @click=${()=>this.promote(e.profile??``,`decision`,e.decision,e.precision??0,e.labelled)}>Promote to on</button></td></tr>`})}
                ${n.map(e=>{let t=_e(e);return p`<tr data-precision-row=${`task/${e.task}${e.backend?`@${e.backend}`:``}`}><td>${e.backend?p`<span class="muted small mono">via ${e.backend}</span>`:`—`}</td><td class="mono">${e.task}</td><td><span class="badge ${e.mode===`act`?`badge-on`:`badge-shadow`}">${e.mode}</span></td>
                    <td class="num num-col">${c(e.decisions)}</td><td class="num num-col">${c(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${t.allowed?``:`muted`}">${t.why}</td>
                    <td><button class="btn btn-sm ${t.allowed?`btn-primary`:``}" ?disabled=${!t.allowed||e.mode!==`shadow`||!!e.backend} data-promote=${`task/${e.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,e.task,e.precision??0,e.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:g(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${e.monitor?f:p`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let e=this.reports.find(e=>e.kind===`classifier`);if(!e)return C(`Classifier accuracy`,g(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let t=e.stages??{},r=e.set??{},a=(t[this.stage]??Object.values(t)[0])?.confusion??{},o=r.classes??Object.keys(a);return C(p`Classifier accuracy <span class="muted small num">${r.name??``} · ${r.n??`?`} prompts · ${e.created_at?_(String(e.created_at)):``}</span>`,p`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(t).map(([e,t])=>p`<tr class="req-row ${e===this.stage?`is-open`:``}" @click=${()=>this.stage=e}><td class="mono">${e}</td><td class="num num-col">${i(t.accuracy,1)}</td><td class="num num-col">${t.answered} / ${t.n}</td><td class="num num-col">${t.accuracy_answered==null?`—`:i(t.accuracy_answered,1)}</td><td class="num num-col">${n(t.p50_ms)}</td><td class="num num-col">${n(t.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${o.length&&Object.keys(a).length?p`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...o,`—`].map(e=>p`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${o.map(e=>p`<tr><th>${D(e,this.ids)}</th>${[...o,`—`].map(t=>{let n=a[e]?.[t]??0;return p`<td class="cm-cell ${e===t?`cm-diag`:n?`cm-miss`:``}">${n||``}</td>`})}</tr>`)}</tbody></table></div>`:p`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}
        ${Oe(e).length?p`<p class="muted small" style="margin-top:12px">Digest variants (SRV-153): <strong>today</strong> the 2,000-character digest the decider cut from the front, <strong>sized</strong> each stage's own head and tail, <strong>line</strong> sized plus the context line. <strong>router</strong> is the session's class after the stickiness rule.</p>
            <div class="table-scroll"><table class="table" data-classifier-variants>
              <thead><tr><th>digest</th><th>set</th><th class="num-col">n</th><th class="num-col">decider</th><th class="num-col">cascade</th><th class="num-col">router</th><th class="num-col">decider p50</th></tr></thead>
              <tbody>${Oe(e).map(e=>p`<tr><td class="mono">${e.variant}</td><td class="mono small">${e.set}</td><td class="num num-col">${e.n}</td><td class="num num-col">${e.decider==null?`—`:i(e.decider,1)}</td><td class="num num-col">${e.cascade==null?`—`:i(e.cascade,1)}</td><td class="num num-col">${e.router==null?`—`:i(e.router,1)}</td><td class="num num-col">${n(e.p50)}</td></tr>`)}</tbody></table></div>`:f}`,{id:`cond-classifier`})}renderSuites(){let e=this.reports.filter(e=>e.kind===`suite`),t=oe(this.reports,this.suiteFile),n=fe(t,e),r=t?.rows??[],a=Te(this.reports,this.engine),o=Ee(r),s=t?.verdict,l=t?.comparison??[],u=t?.raw_row,d=t?.build,m=s?.indicative??rt(r),h=(e,t=1)=>e==null?`—`:Number.isInteger(e)?c(e):e.toFixed(t);return C(p`TandemLLM against the raw engine ${t?p`<span class="muted small num">night ${t.night??`—`}${d?` · ${d.version??`?`} ${E(d.code_sha,12)}`:``}${t.profile?` · profile ${t.profile}`:``}</span>`:f}`,p`${a.tone===`ok`?p`<p class="small" data-guard-detail=${a.status}>${x(`check`)} ${a.line}${a.detail?p`: <span class="num">${a.detail}</span>`:f}.</p>`:p`<div class="guard-line ${a.tone===`note`?`guard-note`:``}" data-guard-detail=${a.status}>${x(a.tone===`note`?`info`:`warn`)}<span>${a.line}${a.detail?p`. <span class="small">${a.detail}</span>`:f}</span></div>`}
        ${e.length>1?p`<label class="field suite-pick"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${e.map(n=>p`<option value=${String(n.file)} ?selected=${n===t}>${String(n.night??n.file)} · ${String(n.file)}${fe(n,e)?` · superseded`:``}</option>`)}</select></label>`:f}
        ${n?p`<div class="guard-line guard-note" data-superseded>${x(`info`)}<span><span class="badge badge-warn">superseded</span> by <span class="mono">${n.file}</span>${n.why?p`: <span class="small">${n.why}</span>`:f}. Its numbers are kept as published; read the newer report.</span></div>`:f}
        ${s?p`<div class="note" data-eval-verdict>
              <span class="badge ${s.better?`badge-ok`:s.better===!1?`badge-warn`:``}"><strong>${s.better?`Better`:s.better===!1?`Not better`:`Not shown yet`}</strong></span>${m?p` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:f} — ${s.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${u?.status??`—`}${u?.source?` (${u.source})`:``}${t?.driver?` · driver ${t.driver.mode}`:``}${t?.isolation?` · ${t.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:f}
        ${o.length?p`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">turns</th><th class="num-col">thinking tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>efforts (turns)</th><th>build · night</th></tr></thead>
              <tbody>${o.flatMap(e=>{let t=l.find(t=>t.suite===e.suite||t.suite_id===(e.raw??e.tandem)?.suite_id);return[e.raw,e.tandem].filter(Boolean).map((n,r)=>p`<tr data-suite=${e.suite} data-arm=${n.arm}>
                    <td>${r===0?e.suite:``}${r===0&&!e.cmp.ok?p` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${e.cmp.why}</div>`:f}${r===0&&t?p` <span class="badge ${t.verdict===`better`?`badge-ok`:t.verdict===`worse`||t.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${t.verdict}>${t.verdict}</span>`:f}${r===0&&(n.indicative??(n.n??0)<10)?p` <span class="badge" data-row-indicative>indicative</span>`:f}</td>
                    <td class="mono">${Ke(n.arm)}${n.pristine===!1?p` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:f}</td>
                    <td class="num num-col">${n.n??`—`}</td>
                    <td class="num num-col">${n.pass_rate==null?`—`:i(n.pass_rate,0)}</td>
                    <td class="num num-col">${n.gates_passed??`—`}${n.gates_total?` / ${n.gates_total}`:``}</td>
                    <td class="num num-col">${h(n.minutes_to_pass)}</td><td class="num num-col">${h(n.wall_minutes)}</td>
                    <td class="num num-col">${n.tokens==null?`—`:c(Math.round(n.tokens))}</td>
                    <td class="num num-col" data-turns>${h(n.turns)}</td><td class="num num-col" data-thinking>${n.reasoning_tokens==null?`—`:c(Math.round(n.reasoning_tokens))}</td>
                    <td class="num num-col">${h(n.decode_tps)}</td>
                    <td class="num num-col">${n.empty_stops??`—`}</td><td class="num num-col">${n.continuations??`—`}</td>
                    <td class="num num-col">${n.interventions??`—`}</td><td class="num num-col">${n.gave_ups??`—`}</td>
                    <td class="num num-col">${n.cache_hit==null?`—`:i(n.cache_hit,0)}</td>
                    <td class="small" data-efforts>${it(n)}</td>
                    <td class="small">${E(n.build,16)} · ${n.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts. Tokens, turns and thinking tokens are medians per run.</p>`:g(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return C(`A/B of two profile versions`,p`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>p`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>p`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(t=>p`<option value=${t.version} ?selected=${t.version===this[e]}>v${t.version} ${E(t.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?p`<div class="diff" data-ab>${this.abChanges.map(e=>p`<div class="diff-row"><span>${ve(e.path)}</span><span class="diff-before">${E(e.before,80)}</span><span class="diff-after">${E(e.after,80)}</span></div>`)}</div>`:p`<p class="muted small">The two versions are the same.</p>`:f}`,{id:`cond-ab`})}};d([y({attribute:!1})],Q.prototype,`ids`,void 0),d([y({attribute:!1})],Q.prototype,`tiles`,void 0),d([y({attribute:!1})],Q.prototype,`onSaved`,void 0),d([o()],Q.prototype,`stats`,void 0),d([o()],Q.prototype,`reports`,void 0),d([o()],Q.prototype,`engine`,void 0),d([o()],Q.prototype,`error`,void 0),d([o()],Q.prototype,`msg`,void 0),d([o()],Q.prototype,`stage`,void 0),d([o()],Q.prototype,`abProfile`,void 0),d([o()],Q.prototype,`abVersions`,void 0),d([o()],Q.prototype,`abA`,void 0),d([o()],Q.prototype,`abB`,void 0),d([o()],Q.prototype,`abChanges`,void 0),d([o()],Q.prototype,`abCounts`,void 0),d([o()],Q.prototype,`suiteFile`,void 0),Q=d([a(`qse-cond-compare`)],Q);var Nn=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`deciders`,label:`Deciders`},{id:`sessions`,label:`Sessions`},{id:`memory`,label:`Memory`},{id:`compare`,label:`Compare`}],$=class extends r{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.guard=null,this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=h(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&Nn.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r,i]=await Promise.all([A.profiles(),this.meta?Promise.resolve(this.meta):A.meta(),A.config(),A.doc(`router`),A.evals()]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.guard=Te(i.reports,i.engine),this.error=null}catch(e){let n=t(e);this.error=n.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${n.endpoint}: ${n.message}`}}async setMode(e){let n=this.tiles.find(e=>e.name===this.active);if(n&&!n.locked&&(e!==`on`||await O(`Turn the ${n.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let t=await N(()=>A.setMode(n.name,e,`mode to ${e}`,n.version));this.reloaded=`v${t.version} · reloaded ${t.loaded_at?s(t.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=t(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),t=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return p`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${x(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>u({profile:e.target.value})}>
                ${this.tiles.map(e=>p`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?j([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):f}
            ${this.reloaded?p`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:f}
          </div>
        </div>
        ${this.switchError?p`<div class="guard-line" role="alert">${x(`warn`)} ${this.switchError}</div>`:f}
        ${this.error?p`<div class="guard-line" role="alert">${x(`warn`)} ${this.error}</div>`:f}
        ${this.guard&&this.guard.tone!==`ok`?p`<div class="guard-line ${this.guard.tone===`note`?`guard-note`:``}" data-guard=${this.guard.status}>${x(this.guard.tone===`note`?`info`:`warn`)}<span>${this.guard.line}${this.guard.detail?p`. <span class="small">${this.guard.detail}</span>`:f}</span></div>`:f}
        ${t.map(([e,t])=>p`<div class="guard-line" role="alert">${x(`warn`)} ${e}: ${t.error?.message} (kept version ${t.version})</div>`)}
        ${e?.locked?p`<div class="note" data-locked>${x(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:f}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${Nn.map(e=>p`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:f}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return p`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return p`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return p`<qse-cond-monitor .meta=${e.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return p`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return p`<qse-cond-packages .ids=${e.ids} .tiles=${e.tiles} .selected=${this.params.get(`pkg`)} .tab=${this.params.get(`ptab`)} .pane=${this.params.get(`pview`)}></qse-cond-packages>`;case`deciders`:return p`<qse-cond-deciders></qse-cond-deciders>`;case`sessions`:return p`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`memory`:return p`<qse-cond-memory .ids=${e.ids} .selected=${this.params.get(`memory`)} .page=${Number(this.params.get(`mpage`))||null} .tab=${this.params.get(`mtab`)} .note=${this.params.get(`note`)} .nnode=${this.params.get(`nnode`)}></qse-cond-memory>`;case`compare`:return p`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};d([y({attribute:!1})],$.prototype,`params`,void 0),d([o()],$.prototype,`tiles`,void 0),d([o()],$.prototype,`defaultProfile`,void 0),d([o()],$.prototype,`meta`,void 0),d([o()],$.prototype,`config`,void 0),d([o()],$.prototype,`ids`,void 0),d([o()],$.prototype,`guard`,void 0),d([o()],$.prototype,`error`,void 0),d([o()],$.prototype,`switching`,void 0),d([o()],$.prototype,`reloaded`,void 0),d([o()],$.prototype,`switchError`,void 0),$=d([a(`qse-conductor`)],$);export{$ as QseConductor};