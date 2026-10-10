import{$ as e,$t as t,A as n,At as r,B as i,C as a,Ct as o,D as s,Dt as c,E as l,Et as u,F as d,G as f,H as p,I as ee,J as te,Jt as m,K as h,Kt as g,L as ne,Lt as _,M as re,N as ie,Nt as v,O as ae,P as oe,Pt as y,Q as se,R as ce,Rt as b,S as le,St as x,T as ue,Tt as S,U as C,V as de,W as fe,X as pe,Xt as w,Y as me,Yt as T,Z as he,Zt as E,_ as ge,_t as _e,a as D,at as ve,b as ye,bt as O,ct as k,d as be,et as xe,f as Se,gt as Ce,h as we,i as Te,it as Ee,j as De,jt as A,k as Oe,kt as j,l as ke,m as Ae,nt as je,o as Me,ot as Ne,p as Pe,pt as M,q as Fe,qt as N,r as Ie,rt as Le,s as Re,st as ze,tt as Be,u as Ve,v as He,wt as P,x as Ue,y as We,yt as Ge,z as Ke,zt as qe}from"./app-BwclxT7O.js";import{DocEditor as F,clsChip as I,confView as Je,confirmDialog as L,diffList as Ye,errorsList as R,eventRow as Xe,feedRow as Ze,modeSeg as z,saveBar as B,sourceWords as Qe,versionsPanel as $e}from"./chunk-CDQdC6iM.js";import{n as et,t as tt}from"./chunk-BJ8Pf1_I.js";var V=class extends N{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=M(()=>this.loadStats(),2e4),this.feedPoll=M(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await k.stats(24),this.error=null}catch(e){let t=j(e);this.error=`${t.endpoint}: ${t.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await k.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await k.event(e)).event}catch(e){this.error=j(e).message}}renderMonitor(e){let n=d(e);return t`<div class="grid-2 grid-2-wide">
      ${o(`Monitor acts an hour, last 24 h`,e.acts?t`<qse-ribbon .series=${n} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:O(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${o(`Last actions of the monitor`,e.last_actions.length?t`<div class="feed" data-last-actions>${e.last_actions.map(e=>t`<button class="feed-row" data-action=${e.id} @click=${()=>this.openAction(e.id)}>
                <span class="feed-time">${qe(new Date(e.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
                <span class="feed-conf muted small">${e.outcome?.kind??`running`}</span>
                <span class="feed-what">${Ce({task:e.task,decision:e.decision,mode:`act`,trigger:e.trigger,cap:e.cap})}</span>
                <span class="feed-act">${e.session_key?t`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:E}</span>
              </button>`)}</div>`:t`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:t`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${x(`external`)}</a>`})}
    </div>`}render(){let e=this.stats;if(this.error&&!e)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!e)return P(6);let r=e.classifier,i=r.decider,a=r.qwen,c=i?.calls?i:a?.calls?a:i,l=Object.entries(e.roles),d=l.filter(([,e])=>e.up).length,f=e.monitor;return t`
      <div class="stats" data-tiles>
        ${S({label:`decisions, 24 h`,value:y(e.decisions),sub:t`<span class="num">${y(e.acted)}</span> acted · <span class="num">${y(e.shadow)}</span> in shadow`,title:`${e.injections} notes, ${e.injected_tokens} tokens injected`})}
        ${S({label:`monitor, 24 h`,value:f?y(f.acts):`—`,sub:f?t`acts · <span class="num">${y(f.shadow)}</span> in shadow · <span class="num" data-gave-up>${y(f.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!f,warn:(f?.gave_up??0)>0,title:f?`${y(f.hidden_tokens)} hidden tokens`:void 0})}
        ${S({label:`classifier p50 / p95${!i?.calls&&a?.calls?` (Qwen)`:``}`,value:c?.p50_ms==null?`—`:`${Math.round(c.p50_ms)} / ${Math.round(c.p95_ms??0)}`,unit:c?.p50_ms==null?void 0:`ms`,sub:t`decider ${y(i?.calls??0)} calls${a?t` · Qwen ${y(a.calls)} (${_(a.p50_ms)})`:E} · fail-open ${b(e.fail_open_rate,1)}`,warn:(e.fail_open_rate??0)>.1})}
        ${S({label:`roles online`,value:`${d} / ${l.length}`,sub:t`${l.map(([e,n])=>t`<span title=${n.error??``} class=${n.up?``:n.mode===`off`?`muted`:`err-ink`} data-role=${e}>${n.up?`●`:`○`} ${e} </span>`)}`,warn:l.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${o(`Decisions an hour, last 24 h`,e.decisions?t`<qse-ribbon .series=${n(e,this.ids)} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:O(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${o(`Task classes, 24 h`,t`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${e.classes.map(e=>t`<tr><td>${I(e.class_id,this.ids)}</td><td class="num num-col">${y(e.decisions)}</td><td class="num num-col">${e.labelled?`${e.agree} / ${e.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(e.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(e.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${o(t`Live decisions ${this.paused?t`<span class="badge">paused</span>`:t`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?t`<div class="feed" data-feed>${this.feed.map(e=>Ze(e,this.ids,e=>this.open=e,s(e)))}</div>`:t`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:t`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${x(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${x(`external`)}</a>`})}
        ${o(`Profiles by client`,t`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(n=>t`<button class="pick" @click=${()=>u({screen:`profiles`,edit:e.name})}><span class="muted">${n}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(e=>t`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${f?this.renderMonitor(f):E}
      ${this.open?t`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:E}
      ${this.openEvent?t`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:E}
    `}};g([T({attribute:!1})],V.prototype,`ids`,void 0),g([T({attribute:!1})],V.prototype,`tiles`,void 0),g([T({attribute:!1})],V.prototype,`active`,void 0),g([m()],V.prototype,`stats`,void 0),g([m()],V.prototype,`feed`,void 0),g([m()],V.prototype,`error`,void 0),g([m()],V.prototype,`open`,void 0),g([m()],V.prototype,`openEvent`,void 0),g([m()],V.prototype,`paused`,void 0),V=g([w(`qse-cond-overview`)],V);var nt=[`off`,`low`,`medium`,`high`,`xhigh`],rt=[`small`,`medium`,`large`],H=class extends N{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new F(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),k.doc(`packages`).then(e=>this.packages=a(e.doc)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(C(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await k.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=j(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return P(8);if(e.loadError)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${e.loadError}</div></div>`;let n=e.draft,r=n.classes.map(e=>e.id);return t`
      ${this.renderTest(r)}
      <details class="sec" open data-sec="classes">
        <summary>Task classes <span class="sec-sub">${n.classes.length} classes; a classifier sees only the descriptions</span></summary>
        <div class="sec-body">
          ${n.classes.map((e,t)=>this.renderClass(e,t,n.classes.length,r))}
          <button class="btn" data-add-class @click=${()=>this.addClass()}>${x(`plus`)} Add a class</button>
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
                ${He.map(e=>t`<option value=${e} ?selected=${e===(n.structural.tool_turn_effort??`inherit`)}>${e===`inherit`?`the class's (inherit)`:je(e,n,null)}</option>`)}
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
          ${nt.map(e=>t`<div class="row-card" data-effort=${e}>
              <div class="row-card-head"><strong class="mono">${e}</strong>
                <label class="switch"><input type="checkbox" .checked=${n.effort.levels[e].thinking} @change=${t=>this.upd([`effort`,`levels`,e,`thinking`],t.target.checked)} /> thinking</label></div>
              <div class="form-grid">${this.num(`budget (tokens, 0 = no cap)`,n.effort.levels[e].budget,t=>this.upd([`effort`,`levels`,e,`budget`],Math.round(t)),{step:256,min:0})}</div>
              <label class="field field-block"><span>sentence in the note</span><textarea rows="2" .value=${n.effort.levels[e].note} @input=${t=>this.upd([`effort`,`levels`,e,`note`],t.target.value)}></textarea></label>
            </div>`)}
          <label class="switch"><input type="checkbox" .checked=${n.effort.raise_on_fail} @change=${e=>this.upd([`effort`,`raise_on_fail`],e.target.checked)} /> a failing checker (RESULT: FAIL) raises the effort one level for that turn</label>
          ${this.renderSize(n.effort.size??Ae)}
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
      ${R(e)}
      ${B(e,`the router`)}
    `}renderTest(e){let n=this.result;return t`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?t`<span class="err-ink small" role="alert">${this.testError}</span>`:E}
      </div>
      ${n?n.active?t`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${I(n.class_id??null,e)} ${Je(n.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${Qe(n.source??null)}${n.kept?t` <span class="badge badge-warn">${n.kept}</span>`:E} <span class="muted num">${_(n.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${n.effort??`unchanged`} · thinking ${n.thinking??`on`}${n.budget==null?``:` · budget ${n.budget}`}</dd></div>
                  ${n.extra?.size?t`<div data-test-size><dt>task size</dt><dd>${Fe(n.extra.size)}</dd></div>`:E}
                  <div><dt>profile</dt><dd>${n.profile} (${n.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(n.stages??[]).map(e=>t`<div class="stage ${e.error?`is-error`:e.below?``:`is-accepted`}"><strong>${pe(e.stage)}</strong><span>${e.error??t`${e.label} ${Je(e.confidence,e.accept??.6)}`}</span><span class="num muted">${_(e.ms)}</span></div>`)}</div>
              </div>
              <div>${n.note?t`<p class="muted small">The model would see this at the end of the last turn (${(n.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${n.note}</pre>`:t`<p class="muted small">No note: ${n.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:t`<p class="muted small" style="margin-top:12px">Profile ${n.profile} is raw (${n.mode}): the conductor does nothing for it.</p>`:E}
    </section>`}renderClass(e,n,r,i){let a=e.policy,o=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return t`<details class="row-card" data-class-row=${e.id} ?open=${o}>
      <summary>
        <span>${I(e.id,i,e.label)} <span class="muted small mono">${a.effort_by_size?rt.map(e=>h(a,e)).join(`/`):a.effort??`medium`}${(a.packages??[]).length?` · ${(a.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${n===0} @click=${()=>this.moveClass(n,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n===r-1} @click=${()=>this.moveClass(n,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${r<=1} @click=${()=>this.removeClass(n)} aria-label=${`remove ${e.id}`}>${x(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${e.id} ?readonly=${!o} @input=${e=>this.upd([`classes`,n,`id`],e.target.value)} data-class-id /><span class="hint">${o?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${e.label} @input=${e=>this.upd([`classes`,n,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${e.description} @input=${e=>this.upd([`classes`,n,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${z(nt,a.effort??`medium`,e=>this.upd([`classes`,n,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div class="toolbar size-map" style="margin:0" data-size-map=${e.id}><span class="muted small">by task size (the effort above is the ceiling)</span>
        ${rt.map(r=>t`<label class="field"><span>${r}</span><select aria-label=${`${e.id} effort when ${r}`} @change=${e=>this.setSizeLevel(n,a.effort_by_size,r,e.target.value)}>
            <option value="" ?selected=${!a.effort_by_size?.[r]}>the class's</option>
            ${nt.map(e=>t`<option value=${e} ?selected=${a.effort_by_size?.[r]===e}>${e}</option>`)}
          </select>${a.effort_by_size?.[r]&&h(a,r)!==a.effort_by_size[r]?t`<span class="hint">cut to ${h(a,r)}</span>`:E}</label>`)}
      </div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let r=(a.packages??[]).includes(e);return t`<button class="pick ${r?`is-on`:``}" aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`classes`,n,`policy`,`packages`],r?(a.packages??[]).filter(t=>t!==e):[...a.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(a.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,n,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}renderSize(e){let n=(t,n)=>this.upd([`effort`,`size`],C(e,t,n));return t`<div class="row-card" data-size-config>
      <div class="row-card-head"><strong>Task size</strong>
        <label class="switch"><input type="checkbox" .checked=${e.reset_on_compaction} @change=${e=>n([`reset_on_compaction`],e.target.checked)} /> a compaction starts the size over</label></div>
      <p class="muted small">Each signal is small below its first number, medium up to its second, large above. The task's size is the largest any counting signal reached, and it never falls within a task. A class's levels per size (in its row above) pick the effort under the class's effort; whether that acts is each profile's <em>effort by task size</em> (off, shadow, on).</p>
      ${we.map(r=>{let i=e.signals[r.id]??Ae.signals[r.id];return t`<div class="size-signal" data-size-signal=${r.id}>
          <label class="switch"><input type="checkbox" .checked=${i.enabled} @change=${e=>n([`signals`,r.id,`enabled`],e.target.checked)} /> ${r.label}</label>
          <div class="form-grid">
            ${this.num(`small below (${r.unit})`,i.small,e=>n([`signals`,r.id],{...i,small:e}),{min:0,step:r.id===`session_tokens`?1e3:1})}
            ${this.num(`medium up to (${r.unit})`,i.medium,e=>n([`signals`,r.id],{...i,medium:e}),{min:0,step:r.id===`session_tokens`?1e3:1,hint:r.hint})}
          </div>
        </div>`})}
    </div>`}setSizeLevel(e,t,n,r){let i={...t??{}};r?i[n]=r:delete i[n],this.upd([`classes`,e,`policy`,`effort_by_size`],Object.keys(i).length?i:void 0)}renderContextStage(e){let n=ue(e),r=(t,r)=>this.upd([`classifier`,`context_stage`],{...n,...e.classifier.context_stage??{},[t]:r});return t`<div class="row-card" data-context-stage>
      <div class="row-card-head">
        <strong>Qwen in the conversation (in-context stage)</strong>
        ${z([`off`,`on`],n.mode,e=>r(`mode`,e),{small:!0,label:`in-context stage`})}
      </div>
      <p class="muted small">A follow-up or a long prompt is classified by the engine inside the conversation it already holds, right before the turn: its prompt up to the last message plus a hidden question, one letter per class, thinking off. The turn resumes from the anchors that prefill leaves. A short first message keeps the stages above; a request with images too.</p>
      <div class="form-grid">
        ${this.num(`long prompt from (tokens)`,n.min_tokens,e=>r(`min_tokens`,Math.round(e)),{step:1024,min:0,max:262144,hint:`a prompt at least this long goes to the stage`})}
        <label class="switch" data-context-followups><input type="checkbox" .checked=${n.followups} @change=${e=>r(`followups`,e.target.checked)} /> every follow-up (a user message after an earlier answer), whatever its length</label>
        ${this.num(`accept at (probability)`,n.accept,e=>r(`accept`,e),{step:.01,min:0,max:1,hint:`below it, the session's class`})}
      </div>
    </div>`}num(e,n,r,i={}){return t`<label class="field"><span>${e}</span><input type="number" .value=${String(n)} step=${i.step??1} min=${i.min??``} max=${i.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&r(t)}} />${i.hint?t`<span class="hint">${i.hint}</span>`:E}</label>`}text(e,n,r){return t`<label class="field"><span>${e}</span><input .value=${n} @input=${e=>r(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(C(n,[`classifier`,`stages`],r))}};g([T({attribute:!1})],H.prototype,`ids`,void 0),g([T({attribute:!1})],H.prototype,`meta`,void 0),g([T({attribute:!1})],H.prototype,`active`,void 0),g([T({attribute:!1})],H.prototype,`onSaved`,void 0),g([m()],H.prototype,`packages`,void 0),g([m()],H.prototype,`testText`,void 0),g([m()],H.prototype,`testProfile`,void 0),g([m()],H.prototype,`testTools`,void 0),g([m()],H.prototype,`testing`,void 0),g([m()],H.prototype,`result`,void 0),g([m()],H.prototype,`testError`,void 0),H=g([w(`qse-cond-routing`)],H);var it=[`off`,`shadow`,`act`],at=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,ot=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,st=class extends N{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new F(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),k.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(C(this.ed.draft,e,t))}render(){let e=this.ed,n=this.config?.server.monitor_mode;return t`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${e.loading&&!e.draft?P(8):e.loadError?t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${e.loadError}</div></div>`:t`
            ${n===`raw`?t`<div class="guard-line" data-monitor-off>${x(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:E}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(e.draft)}
            ${this.renderRest(e.draft)}
            ${R(e)}
            ${B(e,`the monitor`)}`}`}renderTasks(e){let n=ee(e),r=n.filter(t=>e.tasks[t]&&/^C\d/.test(t)),i=n.filter(e=>re(e)),a=n.filter(e=>!r.includes(e)&&!i.includes(e)),o=i.map(t=>e.tasks[t]).find(e=>e.rule===`stall_watchdog`)?.notify;return t`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${r.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${r.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${i.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${i.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${o?.enabled?t`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:t`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, the owner's ruling of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${i.map(t=>this.renderTask(t,e.tasks[t]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${a.join(`, `)}</span></summary>
        <div class="sec-body">${a.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>`}renderTask(e,n){let r=de(this.meta?.schemas?.monitor,n.rule),i=this.meta?.tasks?.[e];return t`<div class="row-card" data-monitor-task=${e}>
      <div class="row-card-head">
        <strong>${e}${i?t` · ${i.label}`:E} <span class="muted small mono">${n.rule}</span></strong>
        ${z(it,n.mode,t=>this.upd([`tasks`,e,`mode`],t),{small:!0,label:`${e} mode without a profile`})}
      </div>
      ${i?.why?t`<p class="muted small">${i.why}</p>`:E}
      ${r?.properties?t`<div class="form-stack">${Object.entries(r.properties).map(([t,r])=>this.field(t,r,n[t],[`tasks`,e,t]))}</div>`:t`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,n,[`tasks`,e])}`}
    </div>`}field(e,n,r,i){let a=Array.isArray(n.type)?n.type[0]:n.type,o=e.replace(/_/g,` `),s=n.description?t`<span class="hint">${n.description}</span>`:E;if(n.enum)return t`<label class="field"><span>${o}</span><select data-param=${e} @change=${e=>this.upd(i,e.target.value)}>${n.enum.map(e=>t`<option value=${String(e)} ?selected=${e===r}>${String(e)}</option>`)}</select>${s}</label>`;if(a===`boolean`)return t`<label class="switch" data-param=${e}><input type="checkbox" .checked=${!!r} @change=${e=>this.upd(i,e.target.checked)} /> ${o}</label>${n.description?t`<p class="hint">${n.description}</p>`:E}`;if(a===`integer`||a===`number`)return t`<label class="field"><span>${o}</span><input type="number" data-param=${e} step=${a===`integer`?`1`:`0.01`} min=${n.minimum??E} max=${n.maximum??E} .value=${r==null?``:String(r)} @input=${e=>{let t=e.target.value;this.upd(i,t===``?null:a===`integer`?Math.round(Number(t)):Number(t))}} />${s}</label>`;if(a===`string`){let n=String(r??``);return at.test(e)||n.length>60?t`<label class="field field-block"><span>${o}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(n.length/90)))} class=${ot.test(e)?`mono`:``} data-param=${e} .value=${n} @input=${e=>this.upd(i,e.target.value)}></textarea>${s}</label>`:t`<label class="field"><span>${o}</span><input data-param=${e} .value=${n} @input=${e=>this.upd(i,e.target.value)} />${s}</label>`}if(a===`array`&&n.items&&(Array.isArray(n.items.type)?n.items.type[0]:n.items.type)===`string`){let a=!Array.isArray(r)&&Array.isArray(n.default),c=Array.isArray(r)?r:a?n.default:[];return t`<label class="field field-block"><span>${o} <span class="muted small">(one a line${a?`; the default, not saved yet`:``})</span></span><textarea rows=${Math.min(8,Math.max(2,c.length+1))} data-param=${e} .value=${c.join(`
`)} @input=${e=>this.upd(i,oe(e.target.value))}></textarea>${s}</label>`}if(a===`array`&&n.items?.properties){let a=Array.isArray(r)?r:[],c=n.items;return t`<fieldset class="fieldset" data-param=${e}><legend>${o} <span class="muted small">${a.length}</span></legend>${s}
        ${a.map((e,n)=>t`<div class="row-card row-card-sub" data-item=${String(e.id??n)}>
          <div class="row-card-head"><strong class="mono small">${String(e.id??`#${n+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(i,a.filter((e,t)=>t!==n))} aria-label=${`remove ${String(e.id??n+1)}`}>${x(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(c.properties).map(([t,r])=>this.field(t,r,e[t],[...i,n,t]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${e} @click=${()=>this.upd(i,[...a,ct(c)])}>${x(`plus`)} Add</button>
      </fieldset>`}if(a===`object`&&n.properties){let a=r??{};return t`<fieldset class="fieldset" data-param=${e}><legend>${o}</legend>${s}
        <div class="form-stack">${Object.entries(n.properties).map(([e,r])=>a[e]===void 0&&!(n.required??[]).includes(e)?t`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...i,e],ct(r))}>${x(`plus`)} ${e.replace(/_/g,` `)}</button>`:this.field(e,r,a[e],[...i,e]))}</div></fieldset>`}return this.jsonField(o,r,i)}jsonField(e,n,r){return t`<label class="field field-block"><span>${e} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(n,null,2)} @change=${t=>{try{let e=JSON.parse(t.target.value);if(r.length===2&&r[0]===`tasks`){let t=this.ed.draft.tasks[r[1]];this.upd(r,{...e,rule:t.rule,mode:t.mode})}else this.upd(r,e)}catch{this.ed.errors=[`${e}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(e){return t`
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
      </details>`}};g([T({attribute:!1})],st.prototype,`meta`,void 0),g([T({attribute:!1})],st.prototype,`onSaved`,void 0),g([m()],st.prototype,`config`,void 0),st=g([w(`qse-cond-monitor`)],st);function ct(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,ct(t)])):``}var lt=class extends N{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=M(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await k.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let t=j(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let e=this.sessions;return o(t`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!e?t`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?P(2):e.length?t`<div class="form-stack" data-watch>${e.map(e=>this.row(e))}</div>`:t`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(e){let n=e.gates.some(e=>e.noted);return t`<div class="row-card" data-watch-session=${e.session}>
      <div class="row-card-head">
        <strong class="mono small">${e.session}</strong>
        <span class="toolbar" style="margin:0">
          ${e.watch?t`<span class="badge ${e.watch.mode===`act`?`badge-on`:`badge-shadow`}">${e.watch.task} ${e.watch.mode}</span>`:E}
          ${e.client_gone?t`<span class="badge badge-warn" data-client-gone>client gone</span>`:e.running?t`<span class="badge badge-on">running</span>`:E}
          ${e.watch?.notify?t`<span class="badge badge-warn" data-notify>push on</span>`:E}
          ${n?t`<span class="badge badge-warn">gate noted</span>`:E}</span>
      </div>
      <p class="small" style="margin:4px 0">${Ee(e)}</p>
      <p class="muted small num" style="margin:0">${e.watch?`${e.watch.profile||`—`} · ${e.watch.client||`—`} · `:``}${e.requests} requests · last ${v(new Date(e.last_seen*1e3).toISOString())}${e.note_at?` · last note ${v(new Date(e.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};g([m()],lt.prototype,`sessions`,void 0),g([m()],lt.prototype,`monitor`,void 0),g([m()],lt.prototype,`error`,void 0),lt=g([w(`qse-cond-watch`)],lt);var U=class extends N{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=M(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await k.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let t=j(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let e=this.events,n=[...new Set((e??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!n.includes(this.task)&&n.push(this.task),t`${o(t`Monitor journal ${this.paused?t`<span class="badge">paused</span>`:t`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!e?t`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?P(4):e.length?t`<div class="feed" data-journal>${e.map(e=>Xe(e,e=>this.open=e))}</div>`:t`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:t`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${n.map(e=>t`<option value=${e} ?selected=${e===this.task}>${e}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${x(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?t`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:E}`}};g([T({attribute:!1})],U.prototype,`session`,void 0),g([m()],U.prototype,`events`,void 0),g([m()],U.prototype,`monitor`,void 0),g([m()],U.prototype,`error`,void 0),g([m()],U.prototype,`task`,void 0),g([m()],U.prototype,`kind`,void 0),g([m()],U.prototype,`paused`,void 0),g([m()],U.prototype,`open`,void 0),U=g([w(`qse-cond-journal`)],U);var ut=[`off`,`low`,`medium`,`high`,`xhigh`],dt={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},ft=new Set([`S1`,`S4`]),pt=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],W=class extends N{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleContract=null,this.roleTest={},this.ed=null,this.roles=new F(this,`roles`),this.residency=new F(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),k.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),k.doc(`packages`).then(e=>this.packages=a(e.doc)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new F(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{let e=await k.roles();this.roleHealth=e.health,this.roleContract=e.contract??null}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let t=await k.roleTest(e);this.roleTest={...this.roleTest,[e]:t.ok?`ok in ${_(t.ms)}: "${t.answer?.label}" at ${t.answer?.confidence.toFixed(2)}`:`failed: ${t.error}`}}catch(t){this.roleTest={...this.roleTest,[e]:j(t).message}}this.loadHealth()}render(){return t`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>t`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>u({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?t`<span class="badge badge-lock">${x(`warn`)} locked</span>`:t`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${v(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):t`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(C(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=C(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=C(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderUi(e,n){let r=e.ui??{states:`off`},i=r.kinds??{},a=n||r.states!==`on`,o=he(r),s=ce(r),c=(e,n,i,o,c)=>t`<label class="field"><span>${n}</span><input type="number" inputmode="numeric" min=${i} max=${o} step=${c} ?disabled=${a||s.mode!==`on`} data-ui-progress=${e} .value=${String(s[e])}
        @input=${t=>{let n=Math.round(Number(t.target.value));Number.isFinite(n)&&n>=i&&n<=o&&this.upd([`ui`],{...r,progress:{...r.progress??{},mode:s.mode,[e]:n}})}} /></label>`;return t`<details class="sec" data-sec="ui">
      <summary>What the user sees <span class="sec-sub">TandemLLM's own steps as lines the client shows</span> <span class="badge ${r.states===`on`?`badge-on`:``}" data-ui-badge>${r.states}</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0"><span class="muted small">state lines</span>${z([`off`,`on`],r.states,e=>this.upd([`ui`],{...r,states:e}),{disabled:n,small:!0,label:`state lines`})}</div>
        <p class="muted small">Each step goes out at the moment it happens as one line, <code>› deciding: code_edit, high effort</code>. The server takes them out of the reasoning and the content a client sends back, so Qwen never sees them; they are not tokens and not in the usage. Only in mode on.</p>
        <div class="toolbar" style="margin:0" data-ui-channel><span class="muted small">where</span>${z([`reasoning`,`content`,`inline`],o,e=>this.upd([`ui`],{...r,states_channel:e}),{disabled:a,small:!0,label:`where the lines go`})}</div>
        <p class="muted small">${o===`inline`?t`Each line opens a thinking block of its own and is its title, in time order: opencode shows <code>Thought: › thinking 2 min, 4,096 tokens</code> between Qwen's thinking blocks, and the answer after them. Needs <code>reasoning_content</code>; with <code>&lt;think&gt;</code> in the content (Hermes) the lines go as in the content.`:o===`content`?t`In the content, outside the thinking: before the reasoning, between its parts, after it. opencode shows them in its text part; with <code>&lt;think&gt;</code> in the content (Hermes) a line said inside the block goes out right after <code>&lt;/think&gt;</code>.`:t`In the reasoning opencode and Hermes show (a <code>reasoning_content</code> delta, or inside <code>&lt;think&gt;</code>). A profile saved before this choice existed keeps them here.`}</p>
        <div class="pkg-picks" data-ui-kinds>${ge.map(({id:e,example:n})=>{let o=i[e]!==!1;return t`<button class="pick ${o?`is-on`:``}" data-ui-kind=${e} title=${n} aria-pressed=${o?`true`:`false`} ?disabled=${a} @click=${()=>this.upd([`ui`],{...r,kinds:{...i,[e]:!o}})}>${e}</button>`})}</div>
        <div class="toolbar" style="margin:0" data-ui-progress-mode><span class="muted small">progress</span>${z([`off`,`on`],s.mode,e=>this.upd([`ui`],{...r,progress:{...r.progress??{},mode:e}}),{disabled:a,small:!0,label:`progress lines`})}</div>
        <p class="muted small">The numbers of a wait, templated: <code>waiting: opencode's title request runs first, position 1</code> while the request waits, <code>reading 35,189 tokens at ~750 tok/s, about 47 s left</code> during a long prefill, <code>thinking 4 min, 9,800 tokens</code> during a long thinking. Streams only; a request that waits opens its stream at its first waiting line.</p>
        <div class="form-grid">
          ${c(`prefill_every_s`,`a prefill line every (s)`,2,600,1)}
          ${c(`thinking_every_s`,`a thinking line every (s)`,10,3600,10)}
          ${c(`thinking_every_tokens`,`or every (tokens)`,256,262144,256)}
        </div>
        <div class="toolbar" style="margin:0" data-ui-narrator><span class="muted small">narrator</span>${z([`off`,`on`,`thoughts`,`live`],r.narrator??`off`,e=>this.upd([`ui`],{...r,narrator:e}),{disabled:a,small:!0,label:`narrator`})}</div>
        <p class="muted small">A small model inside the engine (the <code>narrator</code> role, Qwen3.5) writes in plain words. <b>live</b>: a running line of what Qwen is doing right now, every ~25 s of thinking (“thinking 3 min: how the enemies patrol …”), what a thought decided, what a long write is for and how far it is, a plan in the answer; the other lines in plain words and the user’s language; the facts (times, files, commands) come from the engine, never from the model. <b>thoughts</b>: after each stretch of thinking (and at the long-thinking marks) one line with what Qwen thought about and decided, e.g. “Thought 1m 43s: planned the level layout …”, read from the thought alone; the other lines stay templates. <b>on</b>: it writes every line instead of the template. Qwen waits while it writes, at most 3 % of its time (the rest at the turn’s end); a line it cannot write in time is the template. It loads on its first line; the Roles section below holds its budget and mode.</p>
      </div>
    </details>`}renderToolTurn(e,n){let r=e.thinking.tool_turn_effort,i=Be(this.router,null).level,a=Be(this.router,e);return t`<label class="field ${r?``:`is-inherited`}" data-tool-turn-effort><span>effort of a tool-result turn</span>
      <select aria-label="effort of a tool-result turn" ?disabled=${n} @change=${e=>{let t=e.target.value;if(!this.ed?.draft)return;let n={...this.ed.draft.thinking,tool_turn_effort:t};t||delete n.tool_turn_effort,this.ed.set(C(this.ed.draft,[`thinking`],n))}}>
        <option value="" ?selected=${!r}>Routing's: ${i===`inherit`?`the class's`:i}</option>
        ${He.map(e=>t`<option value=${e} ?selected=${r===e}>${e===`inherit`?`the class's (inherit)`:e}</option>`)}
      </select>
      <span class="hint">${je(a.level,this.router,e)}${a.source===`profile`?` (this profile)`:a.source===`router`?` (Routing)`:``}: a turn whose newest message is a tool result thinks at this level, its switch and cap; the first turn and every new user message keep the class's</span></label>`}renderBudgets(e,n){return t`<div data-budgets>
      <div class="muted small">Thinking budget per level <span class="muted">(reasoning tokens; empty keeps Routing's level, 0 is no cap below the backstop)</span></div>
      <div class="form-grid">
        ${Me.map(r=>{let i=e.thinking.budgets?.[r],a=ie(r,this.router,null).cap,o=ie(r,this.router,e);return t`<label class="field ${i==null?`is-inherited`:``}" data-budget=${r}><span class="mono">${r}</span>
            <input type="number" inputmode="numeric" min="0" max=${Re} step="1024" ?disabled=${n} placeholder=${a==null?``:String(a)} .value=${i==null?``:String(i)}
              @change=${e=>{let t=e.target.value.trim();this.ed?.draft&&this.ed.set(C(this.ed.draft,[`thinking`],Ne(this.ed.draft.thinking,r,t===``?null:Number(t))))}} />
            <span class="hint">${Ue(o.cap)}${o.source===`router`?` (Routing)`:o.source===`profile`?` (this profile)`:``}</span></label>`})}
      </div>
    </div>`}renderEditor(e){if(e.loading&&!e.draft)return P(8);if(e.loadError)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${e.loadError}</div></div>`;let n=e.draft,r=!!e.locked,i=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],a=this.router?.classes??[];return t`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${x(`playground`)} Try in Playground</a></div>
      ${r?t`<div class="note" data-lock-reason>${x(`warn`)} ${e.locked}. Every control below is read-only.</div>`:E}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${n.description} ?disabled=${r} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${z([`raw`,`shadow`,`on`],n.mode,e=>this.upd([`mode`],e),{disabled:r,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${i.map(e=>{let i=n.clients.includes(e);return t`<button class="pick ${i?`is-on`:``}" ?disabled=${r} aria-pressed=${i?`true`:`false`} @click=${()=>this.upd([`clients`],i?n.clients.filter(t=>t!==e):[...n.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${r} @change=${e=>this.upd([`default_class`],e.target.value)}>${a.map(e=>t`<option value=${e.id} ?selected=${e.id===n.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(dt).map(e=>t`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${z([`off`,`shadow`,`on`],n.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:r,small:!0,label:e})}<span class="small">${dt[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${a.map(e=>{let i=n.routing[e.id]??{},a=Object.keys(i).length>0;return t`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${I(e.id,this.ids,e.label)}
                ${a?t`<button class="btn btn-ghost btn-sm" ?disabled=${r} @click=${()=>{let t={...n.routing};delete t[e.id],this.upd([`routing`],t)}}>use the router's</button>`:t`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${z(ut,i.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...i,effort:t}),{disabled:r,small:!0,label:`effort override of ${e.id}`,cls:i.effort?``:`is-inherited`})}</div>
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
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${z(ut,n.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:r,small:!0,label:`default effort`})}</div>
          <label class="switch"><input type="checkbox" ?disabled=${r} .checked=${n.thinking.hints} @change=${e=>this.upd([`thinking`,`hints`],e.target.checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <div class="toolbar" style="margin:0" data-by-size><span class="muted small">effort by task size</span>${z([`off`,`shadow`,`on`],n.thinking.by_size??`shadow`,e=>this.upd([`thinking`,`by_size`],e),{disabled:r,small:!0,label:`effort by task size`,cls:n.thinking.by_size?``:`is-inherited`})}</div>
          <p class="muted small">Effort by task size (SRV-157): a class with levels per size (Routing › the class) thinks at its size's level, never above the class's effort; the size only rises within a task. <strong>off</strong>: the class's effort; <strong>shadow</strong>: the class's effort, and the journal shows the size and the effort it would set; <strong>on</strong>: the size picks.</p>
          ${this.renderToolTurn(n,r)}
          ${this.renderBudgets(n,r)}
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="library">
        <summary>Package library <span class="sec-sub">a searched skill for a new message or a stuck run (R1)</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0" data-library-mode><span class="muted small">library</span>${z([`off`,`shadow`,`on`],n.library??`off`,e=>this.upd([`library`],e),{disabled:r,small:!0,label:`package library`,cls:n.library?``:`is-inherited`})}</div>
          <p class="muted small">On a new user message, and on a low-quality or stuck outcome, the router searches the library (the <a href="#/conductor?screen=packages">Packages</a> screen) with the message, asks the package question among the best matches and puts the skill that fits at the prompt's tail. <strong>off</strong>: nothing; <strong>shadow</strong>: the journal shows the pick, nothing goes in; <strong>on</strong>: the skill and the card that tells the model how to search the library itself.</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113); a pack that is off turns its task off</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${pt.map(([e,i,a])=>t`<label class="switch" data-pack=${e}><input type="checkbox" ?disabled=${r} .checked=${n.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${a} <span class="muted small">(${i}, ${n.toolgate[e]?n.tasks[i]?.mode??`off`:`off`})</span></label>`)}
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
        <p class="muted small" data-notify-rule>Allowed by the owner on 2026-09-30: off by default, on in the overnight profile. Only S4 in act pushes, never in shadow, once a turn end; the token is a file on the box. S2 (re-read after a compaction) has no number here: its texts and state files are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
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
      ${R(e)}
      ${B(e,`profile ${e.name}`)}
    </div>`}renderTask(e,n,r,i,a){let o=this.meta?.tasks[e],s=o?.role&&o.tier===`1`,c=!s||!!this.roleHealth[o.role]?.up;return t`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${o?.family??``}</div></div>
      <div>${o?.label??e}<div class="task-why">${o?.why??``}</div>${s&&!c?t`<div class="warn-ink small">${x(`warn`)} the ${o.role} role is offline: act is refused</div>`:E}</div>
      <div class="task-mode">${z([`off`,`shadow`,`act`],n,t=>this.upd([`tasks`,e,`mode`],t),{disabled:a,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${ft.has(e)&&Object.keys(i).length?t`<p class="muted small" data-in-context=${e}>${Object.entries(i).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:E}
        ${Object.entries(ft.has(e)?{}:i).map(([n,r])=>{let i=o?.thresholds[n];return t`<label class="field ${i&&i.default!==r?`is-changed`:``}"><span>${n.replace(/_/g,` `)} <span class="hint">${i?.unit??``}${i?`, default ${i.default}`:``}</span></span>
            <input type="number" ?disabled=${a} .value=${String(r)} step="any" @input=${t=>{let r=Number(t.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,n],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${a} min="0" max="10" .value=${String(r)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let e=this.roles;if(!e.draft)return e.loadError?t`<p class="err-ink small">${e.loadError}</p>`:E;let n=e.draft,r=(t,r)=>e.set(C(n,t,r)),i=l(e.base?.meta.defaults),a=this.roleContract?.engine_tokenizer_sha256??null,o=this.meta?.roles,s=o?.kinds??Ve,c=o?.windows??be,u=o?.fallbacks??[`PASS`,`stale`,`skip`,`next-role`],d=Se.filter(e=>!o||o.writes.includes(e));return t`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${e.base?.meta.defaults?.length?t`<div class="note" data-role-defaults>${x(`warn`)} Version ${e.base.version} was stored before the role contract and loaded with ${e.base.meta.defaults.length} default${e.base.meta.defaults.length===1?``:`s`} (marked <span class="badge">default</span> below; the journal has the list). Save the roles to make them explicit.</div>`:E}
        <p class="muted small">The contract (SRV-133): when a role may use the GPU, whether a Qwen request cuts it, how long its caller waits before the fallback, what it may write, and the tokenizer it reads, which decides what it may share with Qwen. Engine tokenizer: <code class="mono" data-engine-sha>${a?`${a.slice(0,12)}…`:`not recorded`}</code></p>
        ${Object.entries(n).map(([e,n])=>{let o=this.roleHealth[e],l=new Set((i[e]??[]).map(e=>e.split(`=`)[0])),f=n=>l.has(n)?t` <span class="badge" data-default=${`${e}.${n}`}>default</span>`:E,p=ve(n.kind,n.window),ee=ze(n.kind,n.writes),te=fe(n,a);return t`<div class="row-card" data-role-row=${e}>
            <div class="row-card-head"><strong>${e}</strong>
              <span class="toolbar" style="margin:0">${o?t`<span class="badge ${o.up?`badge-ok`:n.mode===`off`?``:`badge-warn`}">${o.up?`up ${_(o.ms)}`:n.mode===`off`?`off`:`down: ${o.error??`?`}`}</span>`:E}
                <button class="btn btn-sm" @click=${()=>this.testRole(e)} data-role-test=${e}>Test</button></span></div>
            ${this.roleTest[e]?t`<p class="small" role="status">${this.roleTest[e]}</p>`:E}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${n.endpoint} @input=${t=>r([e,`endpoint`],t.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${n.model} @input=${t=>r([e,`model`],t.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(n.timeout_ms)} @input=${t=>r([e,`timeout_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>memory claim (GiB, its peak)${f(`memory_claim_gib`)}</span><input type="number" step="0.1" min="0" data-role-field="memory_claim_gib" .value=${String(n.memory_claim_gib)} @input=${t=>r([e,`memory_claim_gib`],Number(t.target.value))} /></label>
            </div>
            <div class="form-grid" data-role-contract=${e}>
              <label class="field"><span>kind${f(`kind`)}</span><select data-role-field="kind" @change=${t=>r([e,`kind`],t.target.value)}>${s.map(e=>t`<option value=${e} ?selected=${e===n.kind}>${e}</option>`)}</select></label>
              <label class="field"><span>window${f(`window`)}</span><select data-role-field="window" @change=${t=>r([e,`window`],t.target.value)}>${c.map(e=>t`<option value=${e} ?selected=${e===n.window} ?disabled=${e!==n.window&&!!ve(n.kind,e)} title=${We[e]}>${e}</option>`)}</select></label>
              <label class="field"><span>latency budget (ms)${f(`latency_budget_ms`)}</span><input type="number" min="1" max="600000" data-role-field="latency_budget_ms" .value=${String(n.latency_budget_ms)} @input=${t=>r([e,`latency_budget_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>fallback (off, slow, unsure or refused)${f(`fallback`)}</span><select data-role-field="fallback" @change=${t=>r([e,`fallback`],t.target.value)}>${u.map(e=>t`<option value=${e} ?selected=${e===n.fallback}>${e}</option>`)}</select></label>
              <label class="field"><span>writes${f(`writes`)}</span><select data-role-field="writes" @change=${t=>r([e,`writes`],t.target.value)}>${d.map(e=>t`<option value=${e} ?selected=${e===n.writes} ?disabled=${e!==n.writes&&!!ze(n.kind,e)}>${e.replace(/_/g,` `)}</option>`)}</select></label>
              <label class="field field-wide"><span>tokenizer sha256 (the engine records its own; none: text only)${f(`tokenizer_sha256`)}</span><input class="mono" data-role-field="tokenizer_sha256" spellcheck="false" .value=${n.tokenizer_sha256} @input=${t=>r([e,`tokenizer_sha256`],t.target.value.trim().toLowerCase()||`none`)} /></label>
            </div>
            <div class="toolbar" style="margin:0">
              <label class="switch"><input type="checkbox" data-role-field="preemptible" .checked=${n.preemptible} @change=${t=>r([e,`preemptible`],t.target.checked)} /> preemptible: a Qwen request cuts it, it resumes later${f(`preemptible`)}</label>
            </div>
            <div><span class="muted small">sharing with Qwen${f(`sharing`)}</span>
              <div class="pkg-picks" data-role-sharing=${e}>${te.map(i=>{let a=n.sharing.includes(i.form);return t`<button class="pick ${a?`is-on`:``}" aria-pressed=${a?`true`:`false`} ?disabled=${!a&&!i.allowed} title=${i.why??``} data-share=${i.form} @click=${()=>r([e,`sharing`],a?n.sharing.filter(e=>e!==i.form):[...n.sharing,i.form])}>${i.label}</button>`})}</div>
              ${te.filter(e=>!e.allowed).map(e=>t`<p class="muted small" data-share-why=${e.form}>${e.label}: ${e.why}</p>`)}
            </div>
            ${p?t`<p class="warn-ink small" data-role-why="window">${x(`warn`)} ${p}</p>`:E}
            ${ee?t`<p class="warn-ink small" data-role-why="writes">${x(`warn`)} ${ee}</p>`:E}
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${z([`off`,`shadow`,`on`],n.mode,t=>r([e,`mode`],t),{small:!0,label:`${e} mode`})}<span class="muted small mono">${n.contract}</span></div>
          </div>`})}
        ${R(e)}
        ${B(e,`the roles`)}
      </div>
    </details>`}renderResidency(){let e=this.residency;if(!e.draft)return e.loadError?t`<p class="err-ink small">${e.loadError}</p>`:E;let n=e.draft,r=(t,r)=>e.set(C(n,t,r)),i=(e,i=`0.1`,a=!1)=>t`<input type="number" step=${i} min="0" .value=${String(n[e[0]])} @input=${t=>{let n=Number(t.target.value);r(e,a?Math.round(n):n)}} data-res=${e[0]} />`,a=[`measured`,`peak`,`declared`];return t`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${i([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${i([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${i([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${n.order.join(`, `)} @change=${e=>r([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(n.claims).map(([e,i])=>t`<div class="row-card" data-claim=${e}>
            <div class="row-card-head"><strong>${e}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[e]:t,...i}=n.claims;r([`claims`],i)}} aria-label=${`remove the claim of ${e}`}>${x(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(i.gib)} @input=${t=>r([`claims`,e,`gib`],Number(t.target.value))} data-claim-gib=${e} /></label>
              <label class="field"><span>basis</span><select @change=${t=>r([`claims`,e,`basis`],t.target.value)}>${a.map(e=>t`<option value=${e} ?selected=${e===i.basis}>${e}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${i.note} @input=${t=>r([`claims`,e,`note`],t.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in n.claims} @click=${()=>{r([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${x(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${R(e)}
        ${B(e,`the residency limits`)}
      </div>
    </details>`}};g([T({attribute:!1})],W.prototype,`ids`,void 0),g([T({attribute:!1})],W.prototype,`meta`,void 0),g([T({attribute:!1})],W.prototype,`tiles`,void 0),g([T({attribute:!1})],W.prototype,`selected`,void 0),g([T({attribute:!1})],W.prototype,`onSaved`,void 0),g([m()],W.prototype,`router`,void 0),g([m()],W.prototype,`packages`,void 0),g([m()],W.prototype,`roleHealth`,void 0),g([m()],W.prototype,`roleContract`,void 0),g([m()],W.prototype,`roleTest`,void 0),g([m()],W.prototype,`newClaim`,void 0),W=g([w(`qse-cond-profiles`)],W);var mt=[`package`,`task_skill`,`test_skill`,`card`,`library`],ht={package:`package`,task_skill:`task skill`,test_skill:`test skill`,card:`card`,library:`library skill`},gt=3.5,_t={library:350,library_card:200},vt={q:``,category:`all`,cls:``,enabled:`all`,over:!1};function yt(e){return Math.floor(e.length/gt)}function bt(e,t){return t?t.category:e.library?`library`:e.kind===`skill`?`task_skill`:`package`}function xt(e,t,n,r){let i=r?.limits??_t,a=n===`library`?i.library:e===(r?.library.card??`tandem-library`)?i.library_card:null;return a!=null&&a<=t.max_tokens?{limit:a,source:`house`}:{limit:t.max_tokens,source:`max_tokens`}}function St(e,t){return(e?.classes??[]).filter(e=>(e.policy.packages??[]).includes(t)).map(e=>e.id)}function Ct(e,t,n,r){return Object.entries(e?.packages??{}).map(([e,i])=>{let a=t?.packages[e]??null;return{name:e,pkg:i,entry:a,category:bt(i,a),classes:St(r,e),usage:n?.[e]??null,score:null}}).sort((e,t)=>mt.indexOf(e.category)-mt.indexOf(t.category)||e.name.localeCompare(t.name))}function wt(e){return e.toLowerCase().split(/\s+/).filter(Boolean)}function Tt(e,t){let n=`${e.name} ${e.pkg.title} ${e.pkg.description} ${e.pkg.source}`.toLowerCase();return wt(t).every(e=>n.includes(e))}function Et(e,t){return e.entry?e.entry.over:e.pkg.source===`inline`&&yt(Mt(e.pkg.body,!!e.pkg.library)[0])>xt(e.name,e.pkg,e.category,t).limit}function Dt(e,t,n,r=null){let i=new Map((n??[]).map(e=>[e.name,e.score])),a=t.q.trim(),o=e.filter(e=>t.category===`all`||e.category===t.category).filter(e=>!t.cls||e.classes.includes(t.cls)||e.category===`library`&&(e.entry?.fits??[]).includes(t.cls)).filter(e=>t.enabled===`all`||t.enabled===`on`===e.pkg.enabled).filter(e=>!t.over||Et(e,r)).filter(e=>!a||Tt(e,a)||e.category===`library`&&i.has(e.name)).map(e=>({...e,score:a&&e.category===`library`?i.get(e.name)??null:null}));return a?o.sort((e,t)=>mt.indexOf(e.category)-mt.indexOf(t.category)||(e.score!=null||t.score!=null)&&(t.score??-1)-(e.score??-1)||e.name.localeCompare(t.name)):o}function Ot(e,t){let n=e=>!!t?.packages[e]?.general;return[...e.filter(n),...e.filter(e=>!n(e))]}function kt(e,t,n,r){let i=e.classes.findIndex(e=>e.id===t);if(i<0)return{router:e,error:`no class ${t}`};if(r?.packages[n]?.library)return{router:e,error:`${n} is a library skill: the router searches the library, a class does not list it`};let a=e.classes[i].policy.packages??[];if(a.includes(n))return{router:e,error:null};if(a.length>=12)return{router:e,error:`${t} already lists 12 packages, the most a class may`};let o=e.classes.map((e,t)=>t===i?{...e,policy:{...e.policy,packages:Ot([...a,n],r)}}:e);return{router:{...e,classes:o},error:null}}function At(e,t,n){return{...e,classes:e.classes.map(e=>e.id===t?{...e,policy:{...e.policy,packages:(e.policy.packages??[]).filter(e=>e!==n)}}:e)}}var jt=/^## Reference\b/m;function Mt(e,t){let n=t?jt.exec(e):null;return n?[e.slice(0,n.index).trim(),e.slice(n.index).trim()]:[e.trim(),``]}function Nt(e){let t=1;for(;e.packages[`package-${t}`];)t++;return`package-${t}`}function Pt(e){return{title:`Package ${e}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0}}function Ft(e,t,n){return{...e,source:`inline`,body:n?`${t}\n\n${n}`:t}}function It(e,t){return t?e.filter(e=>e.profile===t):e}function Lt(e){return e===`on`?`the router puts the skill it picks into the prompt from the next request, and the library card on the coding classes with tools`:e===`shadow`?`the router searches and picks, and the journal shows the pick; nothing goes into the prompt`:`the router does not search the library`}var G=class extends N{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.catalog=null,this.usage=null,this.onChanged=()=>void 0,this.profile=``,this.busy=null,this.msg=null,this.error=null}async setMode(e,n){this.busy=e,[this.msg,this.error]=[null,null];try{let r=await k.doc(`profiles`,e),i={...r.doc,library:n},a=await k.check(`profiles`,e,i);if(a.errors.length){this.error=`${e}: ${a.errors.join(`; `)}`;return}if(!await L(`Library ${n} for ${e}?`,[`With ${n}, ${Lt(n)}.`,t`The server checked the change (live v${r.version}):${Ye(a.changes)}`,`Saving makes a new version of the profile with the note “library to ${n}”.`],`Save the new version`))return;let o=await k.put(`profiles`,e,i,`library to ${n}`,r.version);this.msg=`${e}: library ${n}, version ${o.version}`,this.onChanged()}catch(e){this.error=j(e).message}finally{this.busy=null}}render(){let e=this.catalog;if(!e||!this.usage)return P(5);let n=this.usage.library,r=Object.values(e.packages).filter(e=>e.category===`library`).length,i=e=>n.rows.filter(t=>t.profile===e&&t.chosen.length).length,a=Object.keys(e.library.profiles).sort();return t`
      <section class="panel" data-lib-profiles>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Retrieval per profile <span class="muted">· ${r} library skills</span></h2>
          <p class="panel-sub">On a new user message, and on a low-quality or stuck outcome, the router searches the library with the message, asks the package question among the best matches and, with <strong>on</strong>, puts the skill that fits at the prompt's tail. <strong>shadow</strong> only journals the pick. Judge the picks below before a profile goes on.</p></div></header>
        ${this.error?t`<p class="err-ink small" role="alert" data-lib-error>${this.error}</p>`:E}
        ${this.msg?t`<p class="status small" role="status" data-lib-saved>${x(`check`)} ${this.msg}</p>`:E}
        <div class="lib-profiles">
          ${a.map(n=>{let r=this.tiles.find(e=>e.name===n),a=e.library.profiles[n];return t`<div class="lib-profile" data-lib-profile=${n}>
              <span class="lib-profile-name"><strong>${n}</strong>${r?.locked?t` <span class="badge badge-lock">locked</span>`:E}<span class="muted small">${r?.clients.join(`, `)||`no client`}${r?` · profile ${r.mode}`:``}</span></span>
              ${z([`off`,`shadow`,`on`],a,e=>void this.setMode(n,e),{small:!0,label:`library of ${n}`,disabled:!!r?.locked||this.busy===n})}
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
          ${S({label:`messages searched`,value:y(n.searched.day),unit:`in 24 h`,sub:`${y(n.searched.week)} in 7 days`})}
          ${S({label:`with a pick`,value:y(n.picked.day),unit:`in 24 h`,sub:`${y(n.picked.week)} in 7 days`})}
          ${S({label:`pick rate`,value:n.searched.week?`${Math.round(100*n.picked.week/n.searched.week)}`:`—`,unit:n.searched.week?`%`:``,sub:`7 days`})}
          ${S({label:`most picked`,value:n.top[0]?.name??`—`,sub:n.top.length?n.top.slice(0,3).map(e=>`${e.name} ${e.picks}`).join(` · `):`no pick yet`})}
        </div>
        ${this.rows(It(n.rows,this.profile))}
      </section>`}rows(e){if(!e.length)return O(`No retrieval in the journal`,`A profile with the library in shadow or on journals each search here; with every profile off nothing is searched.`);let n=e=>u({ptab:null,pkg:e,pview:null});return t`<div class="lib-rows" data-lib-rows>${e.map(e=>t`<div class="lib-row" data-lib-row=${e.id}>
        <div class="lib-row-head">
          <span class="num small">${v(new Date(e.ts_ms).toISOString())}</span>
          <span class="badge ${e.mode===`on`?`badge-on`:`badge-shadow`}">${e.mode??`—`}</span>
          ${I(e.class_id,this.ids)}
          <span class="muted small">${e.profile??`—`} · ${e.trigger??`—`}${e.query_chars==null?``:` · ${y(e.query_chars)} characters searched`}${e.cached?` · asked before`:``}</span>
        </div>
        <div class="lib-cands">${e.candidates.length?e.candidates.map(n=>t`<span class="lib-cand ${n.name&&e.chosen.includes(n.name)?`is-chosen`:``}">${n.name}<span class="num muted">${n.score==null?``:n.score.toFixed(2)}</span></span>`):t`<span class="muted small">no candidate shares a word with the message</span>`}</div>
        <div class="lib-pick small">${e.chosen.length?t`${e.mode===`on`&&e.applied?.length?`went in:`:`would have gone in:`} ${e.chosen.map(e=>t`<button class="btn btn-sm lib-open" data-lib-open=${e} @click=${()=>n(e)}>${e}</button>`)} <span class="muted">(${e.how??`—`})</span>`:t`<span class="muted">no pick (${(e.how??`none`).replace(/_/g,` `)})</span>`}</div>
      </div>`)}</div>`}};g([T({attribute:!1})],G.prototype,`ids`,void 0),g([T({attribute:!1})],G.prototype,`tiles`,void 0),g([T({attribute:!1})],G.prototype,`catalog`,void 0),g([T({attribute:!1})],G.prototype,`usage`,void 0),g([T({attribute:!1})],G.prototype,`onChanged`,void 0),g([m()],G.prototype,`profile`,void 0),g([m()],G.prototype,`busy`,void 0),g([m()],G.prototype,`msg`,void 0),g([m()],G.prototype,`error`,void 0),G=g([w(`qse-cond-library`)],G);var Rt={mode:`off`,idle_s:180,due_idle_s:3,min_new_tokens:1500,max_tokens:600,max_blocks:4,gate:!0,after_compaction:`idle`,turn_deadline_s:30,existing:5},zt={index_tokens:1500,body_tokens:600,pinned_tokens:300,bodies:3,min_score:0,min_words:2,dedupe_threshold:.6,intent:`shadow`,provisional:!0,digest_chars:2e3,writer:Rt},Bt={rule:`how I want things done`,ruling:`a decision that was taken`,gotcha:`what went wrong once`,runbook:`how to do a thing`,program_state:`where the work stands`,reference:`a fact to look up`,registry:`a list of things`,archive:`kept for the record`},Vt=Object.keys(Bt),Ht={"qwen-inline":`the model, mid-answer`,"qwen-writer":`the model, in the background`,api:`the model's script`,human:`you`},Ut={node:null,kind:``,status:``,writer:``,stale:!1,q:``};function Wt(e,t){let n=t.q.trim().toLowerCase();return e.filter(e=>(!t.node||e.node===t.node)&&(!t.kind||e.kind===t.kind)&&(!t.status||e.status===t.status)&&(!t.writer||e.writer===t.writer)&&(!t.stale||!!e.stale)&&(!n||`${e.title} ${e.tags.join(` `)} ${e.id}`.toLowerCase().includes(n)))}function Gt(e){return e.scope===`user`?`About you`:e.scope===`project`?e.label??e.node.replace(/^projects\//,``):`session ${e.label??e.node.replace(/^sessions\//,``)}`}function Kt(e,t){let n=e.split(`
`),r=t.split(`
`),i=n.length,a=r.length;if(i*a>4e5)return[...n.map(e=>({op:`-`,text:e})),...r.map(e=>({op:`+`,text:e}))];let o=Array.from({length:i+1},()=>Array(a+1).fill(0));for(let e=i-1;e>=0;e--)for(let t=a-1;t>=0;t--)o[e][t]=n[e]===r[t]?o[e+1][t+1]+1:Math.max(o[e+1][t],o[e][t+1]);let s=[],c=0,l=0;for(;c<i&&l<a;)n[c]===r[l]?(s.push({op:`=`,text:n[c]}),c++,l++):o[c+1][l]>=o[c][l+1]?s.push({op:`-`,text:n[c++]}):s.push({op:`+`,text:r[l++]});for(;c<i;)s.push({op:`-`,text:n[c++]});for(;l<a;)s.push({op:`+`,text:r[l++]});return s}function qt(e){let t=e.detail??{};if(e.task===`nightly`)return`nightly: ${t.deleted??0} archives deleted, ${t.duplicates??0} duplicates, ${t.stale??0} stale`;let n=typeof t.why==`string`?t.why:null,r=String(t.op??e.decision??`write`),i=e.task===`api_write`?`the model's script`:Ht[String(t.writer??`qwen-inline`)]??String(t.writer);return n||e.decision&&![`add`,`update`,`archive`,`propose`,`ok`].includes(e.decision)?`refused (${n??e.decision}): ${r} by ${i}`:`${r===`add`?t.revised?`revised`:`added`:r===`update`?`updated`:r===`archive`?`archived`:r===`propose`?`proposed`:r} ${[t.kind,t.scope?`(${t.scope})`:null].filter(Boolean).join(` `)} by ${i}${t.id?` · ${t.id}`:``}`.replace(/\s+/g,` `).trim()}function Jt(e){let t=e?.memory?.notes??{};return{...zt,...t,writer:{...Rt,...t.writer??{}}}}function Yt(e){let t=e.detail??{},n=Array.isArray(t.ops)?t.ops:[],r=e=>n.filter(t=>t.op===e&&(e!==`add`||!t.revised)).length,i=n.filter(e=>e.revised).length,a=[r(`add`)?`${r(`add`)} added`:``,i?`${i} revised`:``,r(`update`)?`${r(`update`)} updated`:``,r(`archive`)?`${r(`archive`)} archived`:``].filter(Boolean),o=t.gate??{},s=n.length&&n.every(e=>e.stored===!1)?` (shadow: not stored)`:``;return{when:e.ts_ms,trigger:String(t.trigger??`—`),outcome:typeof t.skipped==`string`?t.skipped:n.length?`wrote`:t.none?`nothing worth keeping`:`no block`,gate:o.answer?String(o.answer):String(o.how??`—`),seconds:typeof t.ms==`number`?Math.round(t.ms/100)/10:null,tokens:typeof t.tokens==`number`?t.tokens:null,ops:(a.join(`, `)||`—`)+s,refused:Array.isArray(t.refused)?t.refused:[]}}function Xt(e,t){let n=Object.values(e).reduce((e,t)=>e+t,0);if(!n)return`not injected yet`;let r=e[t]??0;return`${n} injection${n===1?``:`s`}${r?` (${r} today)`:``}`}var Zt=e=>e?v(e):`—`,K=class extends N{constructor(...e){super(...e),this.onChanged=null,this.list=null,this.open=null,this.error=null,this.flash=null,this.busy=!1,this.status=`open`}connectedCallback(){super.connectedCallback(),this.load()}async load(){try{this.list=(await k.proposals()).proposals,this.error=null}catch(e){this.error=j(e).message}}async show(e){try{this.open=(await k.proposal(e.name,e.n)).proposal}catch(e){this.error=j(e).message}}async apply(e){if(await L(`Apply the proposal to ${e.name}?`,[e.title,`The text becomes a new version of the packages document; every session gets it from its next note. The versions panel restores the old text.`],`Apply`)){this.busy=!0;try{let t=await k.proposalApply(e.name,e.n);this.flash=`${e.name}: packages version ${t.version} (was ${t.previous})`,this.onChanged?.(),await this.load(),await this.show(e)}catch(e){this.error=j(e).message}finally{this.busy=!1}}}async reject(e){let t=window.prompt(`Why is this proposal rejected?`,`not what the skill needs`);if(t!==null){this.busy=!0;try{await k.proposalReject(e.name,e.n,t||`rejected in the dashboard`),this.flash=`${e.name}#${e.n} rejected`,await this.load(),await this.show(e)}catch(e){this.error=j(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.list)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.list)return P(4);let e=this.list.filter(e=>this.status===`all`||e.status===`open`),n=e=>t`<span class="badge ${e===`open`?`badge-shadow`:e===`applied`?`badge-ok`:``}">${e}</span>`;return t`
      ${this.flash?t`<p class="ok-ink small" role="status" data-prop-flash>${x(`check`)} ${this.flash}</p>`:E}
      ${this.error?t`<p class="err-ink small" role="alert">${x(`warn`)} ${this.error}</p>`:E}
      ${this.open?this.renderOne(this.open):E}
      ${o(`Proposals`,e.length?t`<div class="table-scroll"><table class="table" data-proposals><thead><tr><th>package</th><th>change</th><th>mode</th><th>status</th><th>evidence</th><th>proposed</th></tr></thead><tbody>${e.map(e=>t`<tr class="req-row ${this.open&&this.open.name===e.name&&this.open.n===e.n?`is-open`:``}" data-proposal=${`${e.name}#${e.n}`} @click=${()=>this.show(e)}>
                <td class="mono small">${e.name}#${e.n}</td><td class="cell-wrap">${e.title}</td><td class="small">${e.mode}</td><td>${n(e.status)}</td>
                <td class="small">${e.evidence.length?`${e.evidence.length} outcome row${e.evidence.length===1?``:`s`}`:t`<span class="muted">no evidence</span>`}</td><td class="num small">${Zt(e.created)}</td></tr>`)}</tbody></table></div>`:O(this.status===`open`?`No open proposal`:`No proposal yet`,`The model proposes a change to a package with a hidden block when a session showed a gap; it waits here for your click.`),{sub:`changes to packages and skills the model proposed; none goes live without Apply`,tools:t`<label class="switch small"><input type="checkbox" .checked=${this.status===`all`} @change=${e=>this.status=e.target.checked?`all`:`open`} data-prop-all /> applied and rejected too</label>
            <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the proposals">${x(`refresh`)}</button>`,id:`proposals`})}`}renderOne(e){let n=e.live===null?[]:Kt(e.live,e.proposed);return t`<section class="panel" style="margin-bottom:16px" data-proposal-open=${`${e.name}#${e.n}`}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e.name}#${e.n}: ${e.title}</h2>
        <p class="panel-sub">${e.mode===`append`?`adds a section`:`replaces the text`} · ${e.status}${e.applied_version?` as version ${e.applied_version}`:``}${e.rejected_reason?`: ${e.rejected_reason}`:``} · ${e.evidence_word}${e.session?t` · <a class="mono" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(e.session)}`}>${e.session}</a>`:E}</p></div>
        <div class="panel-tools">
          ${e.status===`open`?t`<button class="btn btn-primary btn-sm" ?disabled=${!e.applies||this.busy} @click=${()=>this.apply(e)} data-prop-apply>Apply</button>
            <button class="btn btn-sm" ?disabled=${this.busy} @click=${()=>this.reject(e)} data-prop-reject>Reject</button>`:E}
          <button class="btn btn-ghost btn-icon" aria-label="Close the proposal" @click=${()=>this.open=null}>${x(`x`)}</button></div></header>
      ${e.reason?t`<p class="small"><span class="muted">why:</span> ${e.reason}</p>`:E}
      ${e.stale&&e.status===`open`?t`<p class="warn-ink small">${x(`warn`)} ${e.name} changed since this was proposed${e.mode===`append`?`; the section is added to the text as it is now`:`; a replacement cannot be applied`}.</p>`:E}
      ${e.live===null?t`<p class="warn-ink small">No package ${e.name} with a text any more.</p>`:t`<pre class="notes-diff" data-prop-diff>${n.map(e=>t`<span class="diff-${e.op===`+`?`add`:e.op===`-`?`del`:`eq`}">${e.op===`=`?` `:e.op} ${e.text}\n</span>`)}</pre>`}
    </section>`}};g([m()],K.prototype,`list`,void 0),g([m()],K.prototype,`open`,void 0),g([m()],K.prototype,`error`,void 0),g([m()],K.prototype,`flash`,void 0),g([m()],K.prototype,`busy`,void 0),g([m()],K.prototype,`status`,void 0),K=g([w(`qse-cond-proposals`)],K);var Qt=[{id:`text`,label:`Text`},{id:`edit`,label:`Edit`},{id:`classes`,label:`Classes`},{id:`usage`,label:`Usage`}],q=class extends N{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.selected=null,this.tab=null,this.pane=null,this.catalog=null,this.catError=null,this.usage=null,this.filters=vt,this.hits=null,this.libError=null,this.item=null,this.itemError=null,this.recent=null,this.raw=!1,this.editView=`write`,this.counted=null,this.review=null,this.attachError=null,this.found=null,this.scanError=null,this.pk=new F(this,`packages`),this.rt=new F(this,`router`),this.seq=0,this.loadedFor=null,this.countTimer=null,this.countFor=null}connectedCallback(){super.connectedCallback(),this.pk.onSaved=()=>void this.refresh(),this.rt.onSaved=()=>void this.refresh(),this.pk.load(),this.rt.load(),this.refresh()}disconnectedCallback(){this.countTimer&&clearTimeout(this.countTimer),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.item=null,this.itemError=null,this.recent=null,this.attachError=null,this.selected&&this.loadItem(this.selected))}updated(e){let t=this.selected?this.pk.draft?.packages[this.selected]:null;if(!t||t.source!==`inline`)return;let n=Mt(t.body,!!t.library)[0];this.counted?.text!==n&&this.countFor!==n&&(this.countFor=n,this.countTimer&&clearTimeout(this.countTimer),this.countTimer=setTimeout(()=>{k.packagesTokens([n]).then(e=>this.counted={text:n,n:e.tokens[0],how:e.tokenizer}).catch(()=>void 0)},400))}async refresh(){try{let[e,t]=await Promise.all([k.packagesCatalog(),k.packagesUsage(null,1)]);[this.catalog,this.usage,this.catError]=[e,t,null]}catch(e){this.catError=j(e).message}this.selected&&await this.loadItem(this.selected)}async loadItem(e){try{let[t,n]=await Promise.all([this.pk.base?.doc.packages[e]||!this.pk.base?k.packageItem(e):Promise.resolve(null),k.packagesUsage(e,20)]);if(e!==this.selected)return;[this.item,this.recent,this.itemError]=[t,n.recent,null]}catch(t){e===this.selected&&(this.itemError=j(t).message)}}async search(e){this.filters={...this.filters,q:e};let t=++this.seq;if(!e.trim()||!Object.values(this.pk.draft?.packages??{}).some(e=>e.library)){[this.hits,this.libError]=[null,null];return}try{let n=await k.librarySearch(e.trim(),20);t===this.seq&&([this.hits,this.libError]=[n,null])}catch(e){t===this.seq&&([this.hits,this.libError]=[[],`the library's search: ${j(e).message}`])}}open(e,t=null){u({pkg:e,pview:t})}upd(e,t){this.pk.draft&&this.pk.set(C(this.pk.draft,e,t))}toggleAttach(e,t,n){let r=this.rt.draft;if(!r)return;if(!n){this.attachError=null,this.rt.set(At(r,e,t));return}let i=kt(r,e,t,this.pk.draft);this.attachError=i.error,i.error||this.rt.set(i.router)}addInline(){let e=this.pk.draft,t=Nt(e);this.pk.set(C(e,[`packages`,t],Pt(Number(t.slice(8))))),this.open(t,`edit`)}copyInline(e){let t=this.pk.draft,n=this.item;n&&n.name===e&&n.found&&(this.pk.set(C(t,[`packages`,e],Ft(t.packages[e],n.body,n.reference))),this.open(e,`edit`))}removePackage(e){let t=this.pk.draft,n={...t.packages};delete n[e],this.pk.set({...t,packages:n}),this.open(null)}async scan(){this.scanError=null;try{this.found=(await k.scan()).skills}catch(e){this.scanError=j(e).message}}register(e){let t={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.pk.set(C(this.pk.draft,[`packages`,e.name],t)),this.open(e.name)}async startReview(e,t){await e.check(),this.review=t}async confirmSave(e){await e.save()&&(this.review=null)}bar(e,n,r){let i=e.base?.meta,a=!!e.draft&&!!e.base&&JSON.stringify(e.draft)!==JSON.stringify(e.base.doc);return t`
      ${e.showVersions?$e(e,r):E}
      ${this.review===n?t`<section class="panel pk-review" data-review=${n}>
            <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e.errors.length?t`${x(`warn`)} The server refuses ${r}`:t`${x(`check`)} The server accepts ${r}`}</h2>
              <p class="panel-sub">POST /v1/conductor/${n}/check against live v${i?.version??0}: ${e.changes.length} change${e.changes.length===1?``:`s`}. Saving makes a new version with your note; the next request uses it.</p></div></header>
            ${R(e)}
            ${Ye(e.changes)}
            <div class="toolbar pk-review-tools">
              <button class="btn btn-primary" data-review-save=${n} ?disabled=${e.errors.length>0||!e.note.trim()||e.saving||!e.changes.length} @click=${()=>this.confirmSave(e)}>${e.saving?`Saving…`:`Save as a new version`}</button>
              <button class="btn btn-ghost" @click=${()=>this.review=null}>Back to editing</button>
              ${e.note.trim()?E:t`<span class="muted small">Write the change note first.</span>`}
            </div>
          </section>`:E}
      <div class="savebar ${a||this.review===n?``:`is-idle`}" data-savebar=${n}>
        <span class="status num" title=${i?.note??``}>${r} · live v${i?.version??0}${a?t` · <strong>${e.changes.length||`…`} change${e.changes.length===1?``:`s`}</strong>`:E}</span>
        <input class="note-input" placeholder="one-line change note (required)" .value=${e.note} @input=${t=>{e.note=t.target.value,this.requestUpdate()}} aria-label=${`change note for ${r}`} maxlength="200" data-note=${n} />
        <button class="btn btn-primary" ?disabled=${!a||e.saving} @click=${()=>this.startReview(e,n)} data-check-save=${n}>Check and save…</button>
        <button class="btn btn-ghost" ?disabled=${!a} @click=${()=>{e.reset(),this.review=null}}>Reset</button>
        <button class="btn btn-ghost ${e.showVersions?`is-on`:``}" @click=${()=>{e.showVersions=!e.showVersions,e.showVersions?e.loadVersions():this.requestUpdate()}} data-versions-btn=${n}>Versions</button>
        ${e.saved?t`<span class="status" role="status" data-saved=${n}>${x(`check`)} ${e.saved}</span>`:E}
        ${e.saveError?t`<span class="err-ink small" role="alert">${e.saveError}</span>`:E}
      </div>`}render(){let e=t`<nav class="mem-tabs pk-tabs-top" aria-label="packages tabs">
      ${[{id:null,label:`Packages and skills`},{id:`library`,label:`Library retrieval`},{id:`proposals`,label:`Proposals`}].map(e=>t`<a class="cond-tab ${(this.tab??null)===e.id?`is-on`:``}" href="#" data-pk-top=${e.id??`all`} aria-current=${(this.tab??null)===e.id?`page`:E} @click=${t=>{t.preventDefault(),u({ptab:e.id,pkg:null,pview:null})}}>${e.label}</a>`)}
    </nav>`;if(this.tab===`proposals`)return t`${e}<qse-cond-proposals .onChanged=${()=>this.refresh()}></qse-cond-proposals>`;if(this.tab===`library`)return t`${e}<qse-cond-library .ids=${this.ids} .tiles=${this.tiles} .catalog=${this.catalog} .usage=${this.usage} .onChanged=${()=>this.refresh()}></qse-cond-library>`;let n=this.pk;if(n.loading&&!n.draft)return t`${e}${P(6)}`;if(n.loadError)return t`${e}<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${n.loadError}</div></div>`;let r=n.draft,i=Ct(r,this.catalog,this.usage?.packages??null,this.rt.draft),a=Dt(i,this.filters,this.hits,this.catalog),o=this.selected&&r.packages[this.selected]?this.selected:null;return t`
      ${e}
      ${this.catError?t`<div class="guard-line" role="alert">${x(`warn`)} ${this.catError}</div>`:E}
      <div class="pk-layout ${o?`has-detail`:``}">
        ${this.renderList(a,i)}
        ${o?this.renderDetail(o,r.packages[o],i.find(e=>e.name===o)):this.selected?t`<section class="panel" data-pk-gone><p class="muted">No package ${this.selected} in this version.</p><button class="btn btn-sm" @click=${()=>this.open(null)}>Back to the list</button></section>`:E}
      </div>
      ${this.renderFound(r)}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${r.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${this.rt.draft&&(this.rt.showVersions||this.rt.saved||this.review===`router`||JSON.stringify(this.rt.draft)!==JSON.stringify(this.rt.base?.doc))?this.bar(this.rt,`router`,`the router`):E}
      ${this.bar(this.pk,`packages`,`the packages`)}
    `}renderList(e,n){let r=this.filters,i=n.filter(e=>Et(e,this.catalog)).length,a=e=>this.filters={...this.filters,...e};return t`<section class="panel pk-list" data-pk-list>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills <span class="muted" data-pk-count-all>· ${e.length} of ${n.length}</span></h2>
        <p class="panel-sub">What the router may put into a prompt. Tokens are the text that goes in, ${this.catalog?.tokenizer===`estimate`?`estimated at 3.5 characters a token (no tokenizer on this server)`:`on the engine's tokenizer`}${i?t`; <span class="warn-ink" data-pk-over-count>${i} over the limit</span>`:E}.</p></div>
        <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-pk-new>${x(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${x(`search`)} Skills on the box</button></div></header>
      <div class="pk-filters">
        <label class="field field-search pk-search">${x(`search`)}<input type="search" data-pk-search placeholder="search packages, or the library's text" aria-label="search packages and skills" .value=${r.q} @input=${e=>void this.search(e.target.value)} /></label>
        <label class="field"><span>kind</span><select data-pk-filter="kind" @change=${e=>a({category:e.target.value})}>
          <option value="all" ?selected=${r.category===`all`}>all</option>
          ${mt.map(e=>t`<option value=${e} ?selected=${r.category===e}>${ht[e]}</option>`)}
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
      ${this.libError?t`<p class="err-ink small">${this.libError}</p>`:E}
      <div class="pk-head" aria-hidden="true"><span>package</span><span>tokens / limit</span><span>used 24 h · 7 d</span></div>
      <div class="pk-rows" role="list">${e.map(e=>this.row(e))}</div>
      ${e.length?E:t`<p class="muted small" data-pk-empty>${r.q.trim()?`Nothing matches “${r.q.trim()}”${this.hits?`, and the library shares no word with it`:``}.`:`No package matches these filters.`}</p>`}
    </section>`}row(e){let n=xt(e.name,e.pkg,e.category,this.catalog),r=e.entry?e.entry.tokens:yt(Mt(e.pkg.body,!!e.pkg.library)[0]),i=Et(e,this.catalog),a=e.usage,o=this.selected===e.name;return t`<button class="pk-row ${o?`is-open`:``} ${e.pkg.enabled?``:`is-off`}" role="listitem" data-pk-row=${e.name} data-category=${e.category} aria-current=${o?`true`:E} @click=${()=>this.open(e.name)}>
      <span class="pk-main">
        <span class="pk-name"><strong>${e.name}</strong>
          <span class="badge pk-kind pk-kind-${e.category}">${ht[e.category]}</span>
          ${e.pkg.general?t`<span class="badge badge-on" title="goes in whole and first on every turn of a class that lists it">general</span>`:E}
          ${e.pkg.enabled?E:t`<span class="badge">off</span>`}
          ${e.entry?E:t`<span class="badge badge-shadow">not saved</span>`}
          ${e.entry&&!e.entry.available&&e.entry.error?t`<span class="badge badge-warn" title=${e.entry.path??``}>${e.entry.error.replace(/_/g,` `)}</span>`:E}
          ${e.score==null?E:t`<span class="pk-score num" data-score title="the library's BM25 score">${e.score.toFixed(2)}</span>`}
        </span>
        <span class="pk-desc muted small">${e.pkg.description}</span>
        <span class="pk-meta"><span class="mono small muted">${e.pkg.source}</span>${e.classes.map(e=>I(e,this.ids))}${e.category===`library`&&e.entry?.fits?.length?t`<span class="muted small">fits</span>${e.entry.fits.map(e=>I(e,this.ids))}`:E}</span>
      </span>
      <span class="pk-tokens num ${i?`is-over`:``}" data-pk-tokens title=${e.entry?`${e.entry.tokens} tokens go in; the whole text is ${e.entry.body_tokens}`:`estimated: not saved yet`}>
        <span>${e.entry?``:`≈`}${y(r)} <span class="muted">/ ${y(n.limit)}</span></span>
        <span class="pk-bar"><span style="width:${Math.min(100,100*r/Math.max(1,n.limit))}%"></span></span>
        ${i?t`<span class="badge badge-warn">over</span>`:E}
      </span>
      <span class="pk-usage num small" data-pk-usage>
        ${a&&(a.injected.week||a.shadow.week||a.pick.week)?t`${a.injected.week?t`<span>${a.injected.day} · ${a.injected.week} in</span>`:E}${a.shadow.week?t`<span class="muted">${a.shadow.day} · ${a.shadow.week} shadow</span>`:E}${a.pick.week?t`<span>${a.pick.day} · ${a.pick.week} picks</span>`:E}`:t`<span class="muted">—</span>`}
      </span>
    </button>`}renderFound(e){return this.scanError?t`<p class="err-ink small">${this.scanError}</p>`:this.found?t`<section class="panel" style="margin-top:16px" data-found>
      <header class="panel-head"><h2 class="panel-title">Skills in ${e.skill_dirs.join(`, `)}</h2><div class="panel-tools"><button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.found=null}>${x(`x`)}</button></div></header>
      ${this.found.length?t`<div class="table-scroll"><table class="table"><tbody>${this.found.map(n=>t`<tr><td><strong>${n.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${n.description}</div></td><td class="num">${y(n.chars)} chars</td>
              <td>${n.registered||e.packages[n.name]?t`<span class="badge badge-ok">registered</span>`:t`<button class="btn btn-sm" @click=${()=>this.register(n)} data-register=${n.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:t`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ &lt;box&gt;:~/.config/opencode/skills/</code>).</p>`}
    </section>`:E}renderDetail(e,n,r){let i=r.entry,a=bt(n,i),o=xt(e,n,a,this.catalog),s=Qt.some(e=>e.id===this.pane)?this.pane:`text`,c=n.source===`inline`,l=r.classes,d=this.tokensNow(e,n,i),f=d.n>o.limit||!!i?.cut&&!(c&&this.dirtyBody(e));return t`<section class="panel pk-detail" data-pk-detail=${e}>
      <header class="pk-detail-head">
        <div class="pk-title">
          <h2 class="panel-title">${e}</h2>
          <span class="muted">${n.title}</span>
        </div>
        <button class="btn btn-ghost btn-sm pk-close" @click=${()=>this.open(null)} data-pk-close aria-label="Close the package">${x(`x`)}<span class="pk-close-word">All packages</span></button>
      </header>
      <div class="pk-badges">
        <span class="badge pk-kind pk-kind-${a}" data-pk-category>${ht[a]}</span>
        <span class="badge">${c?`inline`:n.source.startsWith(`skill:`)?`skill on the box`:`file on the box`}</span>
        ${n.general?t`<span class="badge badge-on">general</span>`:E}
        ${n.enabled?t`<span class="badge badge-ok">enabled</span>`:t`<span class="badge">disabled</span>`}
        ${i?E:t`<span class="badge badge-shadow">not saved</span>`}
        ${i&&!i.available&&i.error?t`<span class="badge badge-warn">${i.error.replace(/_/g,` `)}</span>`:E}
      </div>
      <div class="toolbar pk-switches">
        <label class="switch"><input type="checkbox" data-pk-enabled .checked=${n.enabled} @change=${t=>this.upd([`packages`,e,`enabled`],t.target.checked)} /> enabled</label>
        ${a===`library`?E:t`<label class="switch" title="goes in whole and first on every turn of a class that lists it (K1)"><input type="checkbox" data-pk-general .checked=${!!n.general} @change=${t=>this.upd([`packages`,e,`general`],t.target.checked)} /> general</label>`}
        ${c?E:t`<button class="btn btn-sm" data-pk-copy-inline ?disabled=${!this.item?.found} title=${this.item?.found?`an inline copy of the file overrides it`:`the file is not on the box`} @click=${()=>this.copyInline(e)}>${x(`copy`)} Copy to inline</button>`}
        <button class="btn btn-sm btn-ghost" data-pk-remove ?disabled=${l.length>0} title=${l.length?`${l.join(`, `)} list it; detach it first`:`remove it from the packages document`} @click=${()=>this.removePackage(e)}>${x(`trash`)} Remove</button>
      </div>
      <div class="pk-meter ${f?`is-over`:``}" data-pk-meter>
        <div class="pk-bar"><span style="width:${Math.min(100,100*d.n/Math.max(1,o.limit))}%"></span></div>
        <span class="num small" data-pk-count>${d.approx?`≈ `:``}${y(d.n)} tokens of ${y(o.limit)} <span class="muted">(${d.why}; limit: ${o.source===`house`?a===`library`?`the house rule for a library skill`:`the house rule for the library card`:`max_tokens, where the server cuts`})</span></span>
        ${f?t`<span class="badge badge-warn" data-pk-limit-warn>over the limit</span>`:E}
      </div>
      <nav class="mem-tabs pk-panes" aria-label="package detail">
        ${Qt.map(e=>t`<a class="cond-tab ${e.id===s?`is-on`:``}" href="#" data-pk-pane=${e.id} aria-current=${e.id===s?`page`:E} @click=${t=>{t.preventDefault(),u({pview:e.id===`text`?null:e.id})}}>${e.label}</a>`)}
      </nav>
      <div class="pk-pane" data-pk-pane-body=${s}>
        ${s===`text`?this.paneText(e,n,i):s===`edit`?this.paneEdit(e,n,a,o):s===`classes`?this.paneClasses(e,n,a):this.paneUsage(e,a)}
      </div>
    </section>`}dirtyBody(e){return this.pk.draft?.packages[e]?.body!==this.pk.base?.doc.packages[e]?.body}tokensNow(e,t,n){if(t.source===`inline`&&(this.dirtyBody(e)||!n)){let e=Mt(t.body,!!t.library)[0];return this.counted?.text===e?{n:this.counted.n,approx:this.counted.how===`estimate`,why:this.counted.how===`engine`?`the draft, on the engine's tokenizer`:`the draft, estimated`}:{n:yt(e),approx:!0,why:`the draft, estimated until the engine counts it`}}return n?{n:n.body_tokens,approx:this.catalog?.tokenizer===`estimate`,why:n.cut?`the whole text; ${y(n.tokens)} go in after the cut`:this.catalog?.tokenizer===`estimate`?`estimated`:`on the engine's tokenizer`}:{n:0,approx:!0,why:`not counted yet`}}md(e){return e?this.raw?t`<pre class="note-text pk-raw" data-pk-raw>${e}</pre>`:t`<div class="pg-md pk-md" data-pk-md>${et(tt(e))}</div>`:t`<p class="muted small">(empty)</p>`}paneText(e,n,r){let i=n.source===`inline`,a=this.item?.name===e?this.item:null;if(!i&&!a)return this.itemError?t`<p class="err-ink small">${this.itemError}</p>`:P(4);let[o,s]=i?Mt(n.body,!!n.library):[a.body,a.reference],c=i?{}:a.frontmatter;return t`
      <div class="toolbar pk-text-tools">${z([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`text view`})}
        ${!i&&a?.path?t`<span class="mono small muted pk-path" data-pk-path>${a.path}</span>`:E}</div>
      ${!i&&a&&!a.found?t`<p class="note" data-pk-missing>${x(`warn`)} The file is not on the box (<code>${a.path??n.source}</code>): nothing goes in.</p>`:E}
      ${Object.keys(c).length?t`<h3 class="pk-sub">Frontmatter <span class="muted small">the client's metadata; never injected</span></h3>
            ${this.raw?t`<pre class="note-text pk-raw" data-pk-front>${a.frontmatter_raw}</pre>`:t`<dl class="kv pk-front" data-pk-front>${Object.entries(c).map(([e,n])=>t`<div><dt>${e}</dt><dd>${n}</dd></div>`)}</dl>`}`:E}
      <h3 class="pk-sub">The text that goes in ${o?Ge(o,`Copy the text`):E}</h3>
      ${this.md(o)}
      ${r?.cut&&!this.dirtyBody(e)?t`<p class="note" data-pk-cut>${x(`warn`)} The server cuts this text at ${y(n.max_tokens)} tokens × 3.5 characters, at a line end, and adds “[... cut to the package's max_tokens]”: ${y(r.tokens)} tokens go in of ${y(r.body_tokens)}.</p>`:E}
      ${s?t`<details class="sec pk-ref" data-pk-ref>
            <summary>Reference <span class="sec-sub">${y(a?.tokens.reference??yt(s))} tokens; only a read gets it (GET /v1/library/${e})</span></summary>
            <div class="sec-body">${this.md(s)}</div>
          </details>`:E}`}paneEdit(e,n,r,i){let a=n.source===`inline`;return t`
      <div class="form-grid">
        <label class="field"><span>title</span><input data-pk-title .value=${n.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>max tokens</span><input type="number" min="16" max="32768" data-pk-max .value=${String(n.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">the server cuts a longer text at a line end</span></label>
        ${a?E:t`<label class="field field-wide"><span>source</span><input class="mono" .value=${n.source} readonly /><span class="hint">${this.item?.path??`a file on the box`}</span></label>`}
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" data-pk-description .value=${n.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${a?this.editor(e,n,r,i):t`<div class="note pk-readonly" data-pk-readonly>${x(`info`)} The text is the file on the box${this.item?.path?t` at <code>${this.item.path}</code>`:E}, read when first needed and again when it changes. Editing files on the box is not done here. <button class="btn btn-sm" ?disabled=${!this.item?.found} @click=${()=>this.copyInline(e)} data-pk-copy-inline-edit>Copy to inline</button> makes an inline copy of its text that overrides the file.</div>`}`}editor(e,n,r,i){let[a,o]=Mt(n.body,!!n.library),s=this.tokensNow(e,n,this.catalog?.packages[e]??null),c=s.n>i.limit,l=this.editView;return t`<div class="pk-editor pk-view-${l}" data-pk-editor>
      <div class="toolbar pk-editor-tools">
        ${z([`write`,`preview`,`split`],l,e=>this.editView=e,{small:!0,label:`editor view`})}
        <span class="pk-count num small ${c?`is-over`:``}" data-pk-editor-count>${s.approx?`≈ `:``}${y(s.n)} of ${y(i.limit)} tokens${n.library&&o?t` <span class="muted">· reference ≈ ${y(yt(o))}</span>`:E}</span>
        ${c?t`<span class="badge badge-warn" data-pk-editor-warn>over ${i.source===`house`?`the ${ht[r]}'s ${y(i.limit)}`:`max_tokens: the server will cut it`}</span>`:E}
      </div>
      <div class="pk-editor-panes">
        ${l===`preview`?E:t`<textarea class="code pk-body" rows="18" spellcheck="false" data-pk-body aria-label="the package's text (Markdown)" .value=${n.body} @input=${t=>this.upd([`packages`,e,`body`],t.target.value)}></textarea>`}
        ${l===`write`?E:t`<div class="pg-md pk-md pk-preview" data-pk-preview>${et(tt(a))}</div>`}
      </div>
      <p class="hint">${n.library?`A library skill: the text before “## Reference” goes in; the reference only on a read. `:``}${i.source===`house`?`The house rule keeps ${ht[r]} texts at ${y(i.limit)} tokens or fewer.`:`The server cuts the text at max_tokens × 3.5 characters, at a line end.`} Markdown; the preview is the dashboard's renderer, the model gets the text as written.</p>
    </div>`}paneClasses(e,n,r){let i=this.catalog?.packages[e]??null;if(r===`library`)return t`<p class="muted small" data-pk-library-classes>A library skill is searched, not listed by a class: on a new user message the router searches the library and keeps the skills INDEX.md fits to the turn's class.</p>
        <div class="pk-meta">${i?.fits?.length?t`<span class="muted small">INDEX.md fits</span>${i.fits.map(e=>I(e,this.ids))}`:t`<span class="muted small">INDEX.md does not list it: it stays in for every class.</span>`}</div>
        <p class="small"><a href="#/conductor?screen=packages&ptab=library">Each profile's retrieval switch and its picks</a></p>`;let a=this.rt.draft;return a?t`
      <p class="muted small">A class's packages go in by K1's order: the general ones whole and first, then the router asks the package question among the rest and puts in the one that fits. A class lists at most ${12}. Attaching edits the router document; it is saved with its own check below.</p>
      ${this.attachError?t`<p class="err-ink small" role="alert" data-pk-attach-error>${this.attachError}</p>`:E}
      <div class="pk-classes" data-pk-classes>
        ${a.classes.map(n=>{let r=n.policy.packages??[],i=r.includes(e);return t`<div class="pk-class ${i?`is-on`:``}" data-pk-class=${n.id}>
            <label class="switch"><input type="checkbox" .checked=${i} data-pk-attach=${n.id} aria-label=${`attach ${e} to ${n.id}`} @change=${t=>this.toggleAttach(n.id,e,t.target.checked)} /> ${I(n.id,this.ids,n.label)}</label>
            <span class="pk-order small" data-pk-order=${n.id}>${r.length?r.map((n,r)=>t`<span class="pk-order-item ${n===e?`is-me`:``}">${r+1}. ${n}${this.pk.draft?.packages[n]?.general?t` <span class="muted">general</span>`:E}</span>`):t`<span class="muted">no packages</span>`}</span>
            <span class="muted small num">${r.length}/${12}</span>
          </div>`})}
      </div>
      ${i&&Object.keys(i.profiles).length?t`<p class="small muted" data-pk-profiles>A profile's own routing also lists it: ${Object.entries(i.profiles).map(([e,t])=>`${e} (${t.join(`, `)})`).join(`; `)}. A profile's routing replaces the router's policy for that class (the Profiles screen).</p>`:E}
      <div class="toolbar"><button class="btn btn-sm btn-ghost" data-pk-router-versions @click=${()=>{this.rt.showVersions=!0,this.rt.loadVersions()}}>Router versions</button></div>`:P(3)}paneUsage(e,n){let r=this.usage?.packages[e]??null,i={day:0,week:0},a=(e,t)=>S({label:e,value:y(t.day),unit:`in 24 h`,sub:`${y(t.week)} in 7 days`}),o=this.item?.name===e?this.item:null,s=n===`library`||(r?.pick.week??0)>0;return t`
      <div class="stats pk-stats" style="--pk-tiles:${s?3:2}" data-pk-usage-tiles>
        ${a(`injected`,r?.injected??i)}${a(`shadow (would have)`,r?.shadow??i)}${s?a(`library picks`,r?.pick??i):E}
      </div>
      <h3 class="pk-sub">Recent rows <span class="muted small">the journal, newest first (7 days)</span></h3>
      ${this.recent==null?P(3):this.recent.length?t`<div class="pk-uses" data-pk-recent>${this.recent.map(e=>t`<div class="pk-use" data-pk-use=${e.id}>
                <span class="num small">${v(new Date(e.ts_ms).toISOString())}</span>
                <span class="badge ${e.how===`injected`?`badge-on`:e.how===`shadow`?`badge-shadow`:`badge-warn`}">${e.how}</span>
                ${I(e.class_id,this.ids)}
                <span class="muted small">${e.profile??`—`}${e.client_kind?` · ${e.client_kind}`:``}</span>
                ${e.session_key?t`<a class="mono small pk-session" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`}>${e.session_key.length>28?`${e.session_key.slice(0,28)}…`:e.session_key}</a>`:E}
              </div>`)}</div>`:t`<p class="muted small" data-pk-recent-empty>No router row names it in the last 7 days.</p>`}
      ${n===`library`?t`<h3 class="pk-sub">Neighbours <span class="muted small">the library's BM25 on this skill's name, description and text</span></h3>
            ${o?.neighbours?o.neighbours.length?t`<div class="pk-neighbours" data-pk-neighbours>${o.neighbours.map(e=>t`<button class="pk-neighbour" data-pk-neighbour=${e.name} @click=${()=>this.open(e.name)}><strong>${e.name}</strong> <span class="num muted small">${e.score.toFixed(2)}</span><span class="muted small pk-desc">${e.description}</span></button>`)}</div>`:t`<p class="muted small">No other skill shares a word with it.</p>`:P(2)}`:E}`}};g([T({attribute:!1})],q.prototype,`ids`,void 0),g([T({attribute:!1})],q.prototype,`tiles`,void 0),g([T({attribute:!1})],q.prototype,`selected`,void 0),g([T({attribute:!1})],q.prototype,`tab`,void 0),g([T({attribute:!1})],q.prototype,`pane`,void 0),g([m()],q.prototype,`catalog`,void 0),g([m()],q.prototype,`catError`,void 0),g([m()],q.prototype,`usage`,void 0),g([m()],q.prototype,`filters`,void 0),g([m()],q.prototype,`hits`,void 0),g([m()],q.prototype,`libError`,void 0),g([m()],q.prototype,`item`,void 0),g([m()],q.prototype,`itemError`,void 0),g([m()],q.prototype,`recent`,void 0),g([m()],q.prototype,`raw`,void 0),g([m()],q.prototype,`editView`,void 0),g([m()],q.prototype,`counted`,void 0),g([m()],q.prototype,`review`,void 0),g([m()],q.prototype,`attachError`,void 0),g([m()],q.prototype,`found`,void 0),g([m()],q.prototype,`scanError`,void 0),q=g([w(`qse-cond-packages`)],q);var $t={not_loaded:`not loaded`,loading:`loading the weights`,warming:`warming up`,loaded:`loaded`,unloading:`unloading`,failed:`failed`};function en(e){let t=e.role;return t?[t.kind,t.window.replace(`_`,` `),t.preemptible?`preemptible`:`not preemptible`,`${t.latency_budget_ms} ms then ${t.fallback}`,`writes ${t.writes.replace(/_/g,` `)}`,`shares ${t.sharing.join(`, `)||`nothing`}`,t.memory_claim_gib==null?`no claim`:`claims ${D(t.memory_claim_gib)}`].join(` · `):null}function tn(e){return e===`loaded`?`badge-ok`:e===`failed`?`badge-warn`:e===`loading`||e===`warming`||e===`unloading`?`badge-shadow`:``}var nn=`not loaded: kept free for requests`;function rn(e){if(!e||e.allowed)return null;let t=e.reason?Ie[e.reason]??e.reason:`refused`;if(e.reason!==`refused_memory`)return t;if(e.mem_available_gib==null||e.after_gib==null)return`${nn} (MemAvailable not read)`;let n=e.engine_growth_gib?`less ${D(e.engine_growth_gib)} the engine may still take for a long request and the ${D(e.claim_gib)} claim`:`less the ${D(e.claim_gib)} claim`,r=e.margin_gib?`${D(e.need_gib)} free (a ${D(e.floor_gib)} floor and a ${D(e.margin_gib)} margin)`:`${D(e.need_gib)} free`;return`${nn} (MemAvailable ${D(e.mem_available_gib)}, ${n}, leaves ${D(e.after_gib)}; the box keeps ${r})`}function an(e){return e?e.allowed?`fits, ${D(e.after_gib)} left after the claim`:rn(e)??`refused`:`—`}function on(e){return e?e.allowed?`fits`:e.reason===`refused_memory`?`kept free`:`refused`:`—`}function sn(e){return!e?.allowed&&e?.reason!==`refused_memory`}function cn(e){if(!e)return``;let t=(e.source??``).replace(/^engine:/,``);return e.role===`shadow`?t?`the shadow reader (${t})`:`the shadow reader`:t}function ln(e){let t=Object.entries(e).sort((e,t)=>t[1]-e[1]);return t.length?t.map(([e,t])=>`${e} ${t}`).join(` · `):`no call yet`}function un(e){let t=e.limits;return[t.context_tokens,t.question_tokens,t.max_options].map(e=>e==null?`—`:String(e)).join(` / `)}function dn(e,t){let n=e.trim();if(!n)return`Write the question.`;let r=t.split(/[,\n]/).map(e=>e.trim()).filter(Boolean);return r.length<2?`Give at least two options.`:r.length>20?`At most 20 options (the model card’s limit).`:new Set(r).size===r.length?{type:`choice`,instructions:n,criteria:Object.fromEntries(r.map(e=>[e,null]))}:`Two options have the same name.`}function fn(e){return e.type===`noul`?[[`true`,e.noul??0],[`false`,1-(e.noul??0)]]:Object.entries(e.probabilities??{}).sort((e,t)=>t[1]-e[1])}var pn=[`on_start`,`on_demand`,`off`],mn={mode:`on`,rate_per_min:240,max_texts:16,max_chars:1e5,deadline_ms:15e3,threshold:.8,drive_rate_per_s:10,drive_sessions:4,drive_idle_s:30,drive_max_texts:36e3},J=class extends N{constructor(...e){super(...e),this.st=null,this.error=null,this.busy=null,this.said={},this.tState=`The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.`,this.tQuestion=`What kind of work does this request ask for?`,this.tOptions=`debug, code_edit, plan, chat`,this.tModel=`auto`,this.tOut=null,this.tErr=null,this.asking=!1,this.ed=new F(this,`deciders`),this.poller=null}connectedCallback(){super.connectedCallback(),this.ed.load(),this.poller=M(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{this.st=await k.deciders(),this.error=null}catch(e){let t=j(e);this.error=`${t.endpoint}: ${t.message}`}}async act(e,t){if(t!==`unload`||await L(`Unload ${e.name}?`,[`Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).`,`Loading it again goes through the residency planner.`],`Unload`)){this.busy=e.name,this.said={...this.said,[e.name]:``};try{let n=await(t===`load`?k.deciderLoad(e.name):k.deciderUnload(e.name));this.said={...this.said,[e.name]:t===`load`?`load started (${n.state??`loading`})`:`unloaded`}}catch(n){let r=n instanceof c&&n.status===409&&t===`load`?`refused by the residency planner: ${n.message}`:j(n).message;this.said={...this.said,[e.name]:r}}finally{this.busy=null,await this.load()}}}setLoad(e,t){let n=this.ed.draft;if(!n)return;let r=n.models.findIndex(t=>t.name===e),i=r>=0?n.models.map((e,n)=>n===r?{...e,load:t}:e):[...n.models,{name:e,load:t}];this.ed.set({...n,models:i})}async ask(){let e=dn(this.tQuestion,this.tOptions);if(typeof e==`string`){this.tErr=e;return}this.asking=!0,this.tErr=null;try{this.tOut=await k.deciderTest(this.tState,{q:e},this.tModel)}catch(e){this.tOut=null,this.tErr=j(e).message}finally{this.asking=!1}}render(){let e=this.st;if(this.error&&!e)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!e)return P(6);if(!e.enabled)return o(`Decision models`,t`<div class="state" data-deciders-off>
          <div class="state-title">${x(`warn`)} This engine serves no decision model</div>
          <p class="muted small">${e.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port (behind the admin password once one is set), and this screen loads, unloads and tests them.</p>
        </div>`,{id:`deciders-off`});let n=e.models.filter(e=>e.state===`loaded`);return t`
      <div class="stats" data-decider-tiles>
        ${S({label:`loaded`,value:`${n.length} / ${e.models.length}`,sub:n.length?n.map(e=>e.name).join(`, `):`none: questions fail open`,warn:!n.length})}
        ${S({label:`on the device`,value:Te(n.reduce((e,t)=>e+t.device_bytes,0)),sub:`${e.device??`no device`} · own stream and memory pool`})}
        ${S({label:`a load now`,value:on(e.planner),sub:an(e.planner),warn:sn(e.planner)})}
        ${S({label:`when they run`,value:`idle`,sub:`only while no request holds the engine (stage A)`})}
      </div>
      ${this.renderModels(e)}
      ${this.renderMemory(e)}
      ${this.renderApi(e)}
      ${this.renderQuestions(e)}
      ${this.renderTest(e)}
    `}renderModels(e){let n=this.ed.draft,r=e=>n?.models.find(t=>t.name===e)?.load??`on_demand`;return o(`Models`,t`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${e.models.map(e=>{let n=rn(e.refusal),i=this.said[e.name],a=e.state===`loaded`||e.state===`failed`||e.state===`warming`;return t`<tr data-decider-row=${e.name}>
              <td><strong>${e.name}</strong><div class="muted small">${e.family} · ${e.contract}${e.aliases.length?t` · ${e.aliases.join(`, `)}`:E}</div>${e.role?t`<div class="muted small" data-decider-role=${e.name} title="the role contract this model signs (SRV-133); the registry checks it before it serves the model">${en(e)}</div>`:E}</td>
              <td><span class="badge ${tn(e.state)}" data-decider-state=${e.state}>${$t[e.state]}</span>${e.warm===!1?t` <span class="muted small">not warmed</span>`:E}</td>
              <td><select aria-label=${`load policy of ${e.name}`} data-decider-load=${e.name} @change=${t=>this.setLoad(e.name,t.target.value)}>
                ${pn.map(n=>t`<option value=${n} ?selected=${r(e.name)===n}>${n.replace(`_`,` `)}</option>`)}
              </select></td>
              <td class="num">${un(e)}</td>
              <td class="small">${e.precision??t`<span class="muted">—</span>`}<div class="muted small mono" title=${e.weight_hash??``}>${e.weight_hash?e.weight_hash.slice(0,12):``}</div></td>
              <td class="num num-col">${y(e.questions)}</td>
              <td class="num num-col">${e.stats?.forward_ms_p50==null?`—`:`${_(e.stats.forward_ms_p50)} / ${_(e.stats.forward_ms_p95)}`}</td>
              <td class="num num-col ${e.stats?.errors?`err-ink`:``}">${e.stats?y(e.stats.errors+e.stats.deadline_missed):`—`}</td>
              <td>${a?t`<button class="btn btn-sm" ?disabled=${this.busy===e.name} @click=${()=>this.act(e,`unload`)} data-unload=${e.name}>Unload</button>`:t`<button class="btn btn-sm btn-primary" ?disabled=${this.busy===e.name||e.state===`loading`||e.state===`unloading`} @click=${()=>this.act(e,`load`)} data-load=${e.name}>Load</button>`}</td>
            </tr>
            ${n||e.error||i?t`<tr class="is-warn" data-decider-note=${e.name}><td colspan="9" class="small">
                  ${n?t`<span class="warn-ink" data-refusal=${e.name}>${x(`warn`)} ${n}</span> `:E}
                  ${e.error?t`<span class="err-ink">${e.error}</span> `:E}
                  ${i?t`<span role="status">${i}</span>`:E}
                </td></tr>`:E}`})}</tbody>
      </table></div>
      ${e.unserved?.length?t`<p class="muted small">In the document but not served by this engine: ${e.unserved.join(`, `)} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>`:E}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,{id:`deciders-models`,sub:`the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen`})}renderMemory(e){let n=e.planner,r=e.last_refusal;return o(`Memory`,t`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${D(n?.mem_available_gib)}</div></div>
          ${n?.engine_growth_gib==null?E:t`<div class="stat"><div class="stat-label">engine growth to its peak</div><div class="stat-value num">${D(n.engine_growth_gib)}</div></div>`}
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${D(n?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${D(n?.floor_gib)} + ${D(n?.margin_gib)}</div></div>
          <div class="stat ${sn(n)?`is-warn`:``}" data-decider-load-now><div class="stat-label">a load now</div><div class="stat-value num">${on(n)}</div><div class="stat-sub">${an(n)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${r?t`Last engine load not made: ${cn(r)}, ${rn(r)}`:t`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable, less what the engine may still take up to its declared peak, minus the claim stays at or above floor + margin, never during a hold, never beside another load. A model left out this way is not a fault: the room is kept for requests, because one long request is admitted by the engine's own peak, not by what is free, and the box must keep its floor at that peak. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,{id:`deciders-memory`})}renderApi(e){let n=this.ed.draft,r={...mn,...n?.api??{}},i=e.api?.stats,a=(e,t)=>n&&this.ed.set(C(n,[`api`,e],t)),s=(e,n,i=1)=>t`<label class="field"><span>${n}</span><input type="number" step=${i} .value=${String(r[e])} data-api-field=${e} @change=${t=>a(e,i<1?Number(t.target.value):Math.round(Number(t.target.value)))} /></label>`;return o(`Laya for scripts`,n?t`<div data-decider-api>
            <div class="toolbar" style="margin:0"><span class="muted small">route</span>${z([`off`,`on`],r.mode,e=>a(`mode`,e),{small:!0,label:`Laya route mode`})}<code class="small">POST /v1/laya/ask</code></div>
            <div class="form-grid">
              ${s(`rate_per_min`,`texts a minute (every caller)`)}
              ${s(`max_texts`,`texts in one call`)}
              ${s(`deadline_ms`,`deadline (ms) while a request runs`)}
              ${s(`threshold`,`sure at or above`,.05)}
              ${s(`drive_rate_per_s`,`texts a second (a playtest session)`)}
              ${s(`drive_sessions`,`playtest sessions at once`)}
            </div>
            <p class="muted small" data-api-stats>${i?`${y(i.calls)} calls, ${y(i.texts)} texts, ${y(i.errors)} errors (${y(i.rate_limited)} over the rate) since the start${i.last_ms==null?``:`; the last took ${_(i.last_ms)}`}${i.drive_sessions?`; ${y(i.drive_sessions)} playtest sessions live`:``}`:`no calls yet`}</p>
            <p class="muted small">The model's own scripts ask Laya typed questions about their evidence (a log tail, an output, a state dump) through the provider, open like the chat API: a choice, a scale or a yes-no question; Laya reads about 1,100 characters of each text, its head and its tail. Every call is a journal row (<code>laya_api</code>). A playtest (Laya plays the build: <code>.tandem/laya_drive.py</code>) sends a <code>session</code>: its own rate a second, one <code>laya_drive</code> row when it ends.</p>
          </div>`:P(2),{id:`deciders-api`,sub:`the route the model’s scripts call: on or off, its rate and its limits`})}renderQuestions(e){let n=this.ed,r=n.draft,i=e.models.map(e=>e.name),a=r?Object.entries(r.questions):[];return t`${o(`Questions`,r?t`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${a.map(([a,o])=>{let s=e.questions.find(e=>e.id===a);return t`<tr data-question=${a}>
                  <td class="mono">${a}</td>
                  <td><select aria-label=${`model for ${a}`} data-question-model=${a} @change=${e=>n.set(C(r,[`questions`,a],e.target.value))}>
                    ${[`auto`,...i].map(e=>t`<option value=${e} ?selected=${e===o}>${e===`auto`?`auto (the role’s model; the router picks)`:e}</option>`)}
                    ${o!==`auto`&&!i.includes(o)?t`<option value=${o} selected>${o} (not served)</option>`:E}
                  </select></td>
                  <td class="small">${ln(s?.picks??{})}</td>
                </tr>`})}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); <code>judge</code> is the test judge's questions on a run's evidence (the <code>tester</code> role), <code>laya_api</code> the model's own scripts through <code>/v1/laya/ask</code>; the session monitor's Tier 1 questions join this list when they land.</p>`:P(3),{id:`deciders-questions`,sub:`which model answers each question the conductor asks`})}
      ${R(n)}
      ${B(n,`the deciders document`)}`}renderTest(e){let n=this.tOut,r=n?.answers.q;return o(`Test box`,t`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${e=>this.tState=e.target.value} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${e=>this.tQuestion=e.target.value} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${e=>this.tOptions=e.target.value} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${e=>this.tModel=e.target.value} data-test-model>
            ${[`auto`,...e.models.map(e=>e.name)].map(e=>t`<option value=${e} ?selected=${e===this.tModel}>${e}</option>`)}
          </select></label>
        </div>
        <div class="savebar is-idle"><button class="btn btn-primary" ?disabled=${this.asking} @click=${()=>this.ask()} data-test-ask>${x(`send`)} ${this.asking?`Asking…`:`Ask`}</button>
          ${this.tErr?t`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>`:E}</div>
        ${n&&r?t`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${r.type===`choice`?r.choice:r.type===`noul`?(r.noul??0).toFixed(4):(r.score??0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${r.confidence.toFixed(2)} · ${n.model} · ${_(n.latency_ms)} · ${n.path} · ${y(n.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${fn(r).map(([e,n])=>t`<div class="prob-row" data-prob=${e}><span class="mono small">${e}</span><span class="conf-bar"><span style="width:${Math.round(n*100)}%"></span></span><span class="num small">${n.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${n.routing.reason}${n.routing.requested?` (wanted ${n.routing.requested})`:``}</p>
            </div>`:E}`,{id:`deciders-test`,sub:`one choice question, as /v1/systemone answers it (it runs in the idle window like any other)`})}};g([m()],J.prototype,`st`,void 0),g([m()],J.prototype,`error`,void 0),g([m()],J.prototype,`busy`,void 0),g([m()],J.prototype,`said`,void 0),g([m()],J.prototype,`tState`,void 0),g([m()],J.prototype,`tQuestion`,void 0),g([m()],J.prototype,`tOptions`,void 0),g([m()],J.prototype,`tModel`,void 0),g([m()],J.prototype,`tOut`,void 0),g([m()],J.prototype,`tErr`,void 0),g([m()],J.prototype,`asking`,void 0),J=g([w(`qse-cond-deciders`)],J);var Y=class extends N{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=M(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await k.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let t=j(e);this.error=`${t.endpoint}: ${t.message}`}}async loadSession(e){try{let t=await k.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(t){e===this.selected&&(this.turns=[]),this.error=j(t).message}}render(){if(this.error&&!this.list)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.list)return P(6);let e=[...new Set(this.list.map(e=>e.client_kind??`other`))],n=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return t`
      ${this.selected?this.renderSession(this.selected):E}
      ${o(`Sessions, last 7 days`,n.length?t`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${n.map(e=>t`<tr class="req-row ${e.key===this.selected?`is-open`:``}" data-session=${e.key} @click=${()=>u({session:e.key})}>
                  <td class="mono small cell-wrap">${f(e.key,28)}</td><td>${e.client_kind??`—`}</td><td>${e.profile??`—`}</td>
                  <td>${I(e.class_id,this.ids)}</td><td class="num num-col">${y(e.turns)}</td>
                  <td class="num num-col">${e.notes?`${e.notes} (${y(e.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${e.acts?t`${y(e.acts)}${e.gave_ups?t` <span class="badge badge-warn">${e.gave_ups} gave up</span>`:E}`:`—`}</td>
                  <td class="num num-col" data-interventions>${y(e.interventions??0)}</td>
                  <td>${e.last_checker?t`<span class="badge ${e.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${e.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${r(new Date(e.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:O(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:t`<label class="field field-search">${x(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${e.map(e=>t`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?t`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:E}
    `}renderSession(e){let n=this.turns,r=this.live.find(t=>t.key===e);if(n==null)return P(4);let i=n.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,a=n.reduce((e,t)=>e+(t.injected?.length??0),0),o=this.events.filter(e=>_e(e)).length,c=Oe(n);return t`<section class="panel" style="margin-bottom:16px" data-timeline=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${f(e,44)} ${r?t`<span class="badge badge-on">live</span>`:E}</h2>
        <p class="panel-sub">${n.length} turns · ${a} hidden note${a===1?``:`s`} · ${o} monitor act${o===1?``:`s`}${r?` · class now ${r.class_id} · ${y(r.note_tokens)} note tokens replayed a turn`:``}${n.length?` · since ${v(new Date(n[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${k.exportUrl(e,`md`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${x(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${k.exportUrl(e,`json`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${x(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>u({session:null})}>${x(`x`)}</button></div></header>
      ${this.axis(p(n,this.events,this.ids,e=>qe(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${n.map(e=>{let n=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/i*900)),r=(e.injected?.length??0)>0,a=!!e.shadow,o=(e.events??[]).filter(e=>e.decision!==`PASS`),c=o.some(e=>_e(e))?`act`:o.length?`shadow`:null;return t`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${n}px; --cls: var(--series-${le(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${s(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${r?t`<span class="mark"></span>`:a?t`<span class="mark is-shadow"></span>`:E}
            ${c?t`<span class="mark is-monitor ${c===`shadow`?`is-shadow`:``}" data-monitor-mark=${c}></span>`:E}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${c.series.length?t`<qse-ribbon .series=${c.series} .labels=${c.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:E}
      <div class="feed" style="margin-top:10px">${n.slice().reverse().map(e=>Ze(e,this.ids,e=>this.open=e,s(e)+((e.events??[]).some(e=>_e(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?t`<div style="margin-top:12px"><qse-cond-journal .session=${e}></qse-cond-journal></div>`:E}
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
    </figure>`}tier1(){let e=xe(this.events);return e?t`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${e.agree} of ${e.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${e.decisions} shadow decisions)</span></p>`:E}};g([T({attribute:!1})],Y.prototype,`ids`,void 0),g([T({attribute:!1})],Y.prototype,`selected`,void 0),g([m()],Y.prototype,`list`,void 0),g([m()],Y.prototype,`live`,void 0),g([m()],Y.prototype,`turns`,void 0),g([m()],Y.prototype,`events`,void 0),g([m()],Y.prototype,`error`,void 0),g([m()],Y.prototype,`query`,void 0),g([m()],Y.prototype,`client`,void 0),g([m()],Y.prototype,`open`,void 0),Y=g([w(`qse-cond-sessions`)],Y);var hn={scope:`tools`,retention_days:30,clip_chars:4e3,loop_chars:200},gn=/^### (user|assistant|summary|runtime|tool result \((.*)\)|tool call \((.*)\)|reasoning \((.*)\))$/;function _n(e){let t=e[1];return e[2]===void 0?e[3]===void 0?e[4]===void 0?{kind:t===`user`||t===`assistant`||t===`summary`||t===`runtime`?t:`other`,title:t,name:null,clipped:!1}:{kind:`reasoning`,title:t,name:null,clipped:!1}:{kind:`tool_call`,title:t,name:e[3].trim()||null,clipped:!1}:{kind:`tool_result`,title:t,name:e[2].split(`,`)[0].trim()||null,clipped:/clipped to/.test(e[2])}}function vn(e){let t=e.replace(/\n+$/,``).split(`
`),n=t[0]??``,r=null,i=[],a=null,o=()=>{a&&i.push({kind:a.kind,title:a.title,name:a.name,clipped:a.clipped,body:a.lines.join(`
`).replace(/^\n+|\n+$/g,``)})};for(let e of t.slice(1)){let t=e.match(gn);t?(o(),a={..._n(t),lines:[]}):a?a.lines.push(e):e.startsWith(`compaction: `)&&(r=e)}o();let s=i.find(e=>e.kind===`runtime`);return{head:n,compaction:r,sections:i.filter(e=>e.kind!==`runtime`),runtime:s?yn(s.body):[]}}function yn(e){return e.split(`
`).filter(e=>e.trim()).map(e=>{let t=[`hidden`,`replayed`,`acts`,`checker`,`snapshot`,`finish`,`redactions`].find(t=>e.startsWith(`${t}: `))??`other`,n=e.match(/ · journal ([A-Za-z0-9_-]{4,40})$/),r=t===`checker`?e.match(/ (PASS|FAIL)(?: <-|$)/):null;return{kind:t,text:e,journal:n?n[1]:null,verdict:r?r[1]:null}})}function bn(e){if(!e.counts)return`counts unknown (no meta.json)`;let t=e.counts,n=[`${t.blocks} turn${t.blocks===1?``:`s`}`];return t.untranscribed&&n.push(`${t.untranscribed} without tools (counted only)`),n.push(`${t.acts} act${t.acts===1?``:`s`}`),t.snapshots&&n.push(`${t.snapshots} snapshot${t.snapshots===1?``:`s`}`),t.compactions&&n.push(`${t.compactions} compaction${t.compactions===1?``:`s`}`),t.redactions&&n.push(`${t.redactions} redacted`),e.checkers?.last&&n.push(e.checkers.last),n.join(` · `)}function xn(e){return{...hn,...e?.memory?.transcripts??{}}}function Sn(e,t){return e?e.replace(/^[ch]:/,``):t}var Cn={mode:`on`,budget_gb:50,ttl_days:7,admit_after:2,min_free_gb:50},wn={"opencode-head":`opencode head`,head:`session head`,package:`package`,memory:`memory file`,prefix:`repeated prefix`};function Tn(e){return{...Cn,...e?.memory?.disk_cache??{}}}function En(e){let t=[],n=e=>Number.isFinite(e)&&Math.round(e)===e;return e.budget_gb>=1&&e.budget_gb<=4e3||t.push(`the budget is 1 to 4,000 GB`),n(e.ttl_days)&&e.ttl_days>=1&&e.ttl_days<=90||t.push(`an entry is kept 1 to 90 days`),n(e.admit_after)&&e.admit_after>=1&&e.admit_after<=10||t.push(`a prefix is written after 1 to 10 prefills`),e.min_free_gb>=50&&e.min_free_gb<=4e3||t.push(`the NVMe keeps at least 50 GB free`),t}function Dn(e,t){if(t.free_bytes==null)return{usable_gb:null,short:!1};let n=((t.bytes??0)+Math.max(0,t.free_bytes-Math.max(50,e.min_free_gb)*1e9))/1e9;return{usable_gb:n,short:n<e.budget_gb}}function On(e){let t=e.lookups??0;return t?`${Math.round((e.hit_rate??0)*100)} % of ${t.toLocaleString(`en-US`)} prefill${t===1?``:`s`} it could have shortened`:`no prefill the disk could have shortened yet`}function kn(e){let t=e=>typeof e==`number`?e.toLocaleString(`en-US`):String(e),n=[];return e.tokens!=null&&n.push(`${t(e.tokens)} tokens`),e.kind===`hit`&&e.read!=null&&n.push(`${t(e.read)} read`,`${t(e.ms)} ms`),e.kind===`write`&&(n.push(`${t(e.chunks)} chunk${e.chunks===1?``:`s`}`,`${t(e.anchors)} anchor${e.anchors===1?``:`s`}`),e.dropped&&typeof e.dropped==`object`&&n.push(`dropped ${Object.entries(e.dropped).map(([e,t])=>`${t} ${e}`).join(`, `)}`)),e.why!=null&&n.push(String(e.why)),e.kind===`clear`&&n.push(`${t(e.entries)} entries`),e.error!=null&&n.push(String(e.error)),[e.kind,...n].join(` · `)}var An=e=>v(new Date(e*1e3).toISOString()),jn=e=>Object.entries(e??{}).filter(([,e])=>e).map(([e,t])=>`${y(t)} ${e}`).join(`, `)||`none`,Mn=class extends N{constructor(...e){super(...e),this.disk=null,this.error=null,this.busy=!1,this.cleared=null,this.settings=new F(this,`settings`)}connectedCallback(){super.connectedCallback(),this.load(),this.settings.load(),this.settings.onSaved=()=>void this.load()}async load(){try{this.disk=await k.memoryDisk(),this.error=null}catch(e){this.error=j(e).message}}async clear(){let e=this.disk;if(e&&await L(`Clear the disk cache?`,[`Every entry goes: ${y(e.entries??0)} prefixes in ${y(e.chunks??0)} chunk files, ${A(e.bytes??0)}, and the counts that decide what is written next.`,`Nothing a request needs is lost: the next prefill of each prefix runs cold, as without the tier, and writes it again.`],`Clear`)){this.busy=!0;try{let e=await k.memoryDiskClear();this.cleared=`cleared ${y(e.entries)} entries, ${A(e.bytes)}`,this.disk=e.disk}catch(e){this.error=j(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.disk)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.disk)return P(5);let e=this.disk;return t`${e.running?this.renderRunning(e):this.renderOff(e)}${this.renderSettings(e)}`}renderOff(e){return o(`Disk cache`,O(`The disk tier does not run on this server`,t`It starts with the engine when <code>QSE_DISK_CACHE</code> in serve.env names its directory (the flag is <code class="mono">${e.flag}</code> now)${e.free_bytes==null?E:t`; the NVMe has <span class="num">${A(e.free_bytes,0)}</span> free`}. Its settings below apply once it runs.`),{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM`})}renderRunning(e){let n=e.bytes??0,r=e.budget??0,i=r?n/r:0,a=e.free_bytes??null,s=a!=null&&e.floor!=null&&a<e.floor+1e10,c=t`<div class="stats disk-stats" data-disk-tiles>
      ${S({label:`on disk`,value:A(n,1),sub:t`of <span class="num">${A(r,0)}</span> · ${y(e.entries??0)} prefix${e.entries===1?``:`es`}, ${y(e.chunks??0)} chunk files`,warn:i>.95})}
      ${S({label:`NVMe free`,value:a==null?`—`:A(a,0),sub:t`never under <span class="num">${A(e.floor??5e10,0)}</span>`,warn:s})}
      ${S({label:`hit rate`,value:e.hit_rate==null?`—`:b(e.hit_rate,0),sub:On(e),na:e.hit_rate==null})}
      ${S({label:`restored`,value:y(e.tokens_restored??0),unit:`tokens`,sub:e.last_restore_ms==null?`no restore yet`:t`last <span class="num">${y(Math.round(e.last_restore_ms))}</span> ms${e.last_read_gbs==null?E:t` at <span class="num">${e.last_read_gbs.toFixed(1)}</span> GB/s`}`})}
    </div>`,l=t`<div class="meter-block disk-meter">
      <div class="meter-head"><span>budget</span><span class="num">${A(n,1)} of ${A(r,0)} · ${b(i,0)}</span></div>
      <div class="meter" role="img" aria-label=${`disk cache ${b(i,0)} of its budget`}><span class="meter-fill ${i>.95?`is-warn`:``}" style="width:${Math.min(100,i*100).toFixed(1)}%"></span></div>
    </div>`,u=e.top??[],d=t`${c}${l}
      <p class="small muted disk-line" data-disk-counters>
        ${e.on?t`<span class="badge badge-on">on</span>`:t`<span class="badge">off (settings)</span>`}
        written: <span class="num">${y(e.writes_chunk??0)}</span> chunks, <span class="num">${y(e.writes_anchor??0)}</span> anchors, <span class="num">${A(e.bytes_written??0,1)}</span> ·
        evicted: ${jn(e.evicted)} · dropped: ${jn(e.dropped)}${e.failed_restores?t` · <span class="warn-ink">${y(e.failed_restores)} failed restore${e.failed_restores===1?``:`s`}</span>`:E}
        · staging buffer <span class="num">${A(e.staging_bytes||e.staging_planned||0,2)}</span>${e.staging_bytes?``:` (when first used)`} · <span class="mono">${e.root??``}</span>
      </p>
      ${u.length?this.renderEntries(u):O(`Nothing on disk yet`,`A prefix is written after ${e.settings.admit_after===1?`its first prefill`:`${e.settings.admit_after} prefills`} of at least ${y(e.min_tokens??4096)} tokens, and the first long request of an opencode session is written where its user message starts.`)}
      ${this.renderEvents(e)}`;return o(`Disk cache`,d,{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM; a hit is read only past the rows already in place`,tools:t`${this.cleared?t`<span class="small muted" data-disk-cleared>${this.cleared}</span>`:E}
        <button class="btn btn-sm disk-clear" ?disabled=${this.busy||!(e.entries||e.chunks)} @click=${()=>this.clear()} data-disk-clear>${x(`x`)} Clear disk cache</button>
        <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the disk cache">${x(`refresh`)}</button>`})}renderEntries(e){let n=e=>t`<span class="badge disk-kind disk-kind-${e.kind}">${wn[e.kind]}</span>${e.label?t` <span class="mono small">${e.label}</span>`:E}${e.complete?E:t` <span class="badge badge-warn" title="a chunk below it is not on disk: it is not read, and goes first">incomplete</span>`}`,r=e=>e.preview?t`<div class="small muted disk-preview" data-disk-preview>${e.preview}</div>`:E;return t`<h3 class="disk-sub">Most-hit prefixes</h3>
      <div class="table-scroll mem-table"><table class="table" data-disk-entries>
        <thead><tr><th class="num-col">tokens</th><th>what</th><th class="num-col">hits</th><th>last used</th><th class="num-col">a hit reads</th></tr></thead>
        <tbody>${e.map(e=>t`<tr data-disk-row=${`${e.kind}-${e.tokens}`}><td class="num num-col">${y(e.tokens)}</td><td class="cell-wrap">${n(e)}${r(e)}</td>
            <td class="num num-col">${y(e.hits)}</td><td class="num">${An(e.used)}</td><td class="num num-col">${A(e.reads_bytes,1)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards" role="list">${e.map(e=>t`<div class="mem-card disk-card" role="listitem" data-disk-card=${`${e.kind}-${e.tokens}`}>
          <div class="mem-card-head"><span class="num">${y(e.tokens)} tokens</span><span class="num small">${y(e.hits)} hit${e.hits===1?``:`s`}</span></div>
          <div>${n(e)}</div>${r(e)}
          <div class="small muted">last used ${An(e.used)} · a hit reads ${A(e.reads_bytes,1)}</div>
        </div>`)}</div>`}renderEvents(e){let n=(e.events??[]).slice(-10).reverse();return n.length?t`<details class="sec" style="margin-top:12px" data-disk-events>
      <summary>Last events <span class="sec-sub">what it read, wrote, dropped and evicted (the journal keeps them, kind memory)</span></summary>
      <ul class="disk-events">${n.map(e=>t`<li class="small"><span class="num muted">${An(e.ts)}</span> <span class="disk-ev disk-ev-${e.kind}">${kn(e)}</span></li>`)}</ul>
    </details>`:E}renderSettings(e){let n=this.settings;if(!n.draft)return n.loadError?t`<p class="err-ink small">${n.loadError}</p>`:E;let r=Tn(n.draft),i=(e,t)=>n.set(C(n.draft,[`memory`,`disk_cache`],{...r,[e]:t})),a=(e,n,a,o,s=1)=>t`<label class="field"><span>${n}</span><input type="number" inputmode="decimal" min=${a} max=${o} step=${s} .value=${String(r[e])} data-disk-setting=${e} @input=${t=>i(e,Number(t.target.value))} /></label>`,o=Dn(r,e),s=En(r);return t`<details class="sec" style="margin-top:16px" data-sec="disk" open>
      <summary>Disk cache settings <span class="sec-sub">on or off, its size on the NVMe, how long it keeps an entry, what it writes</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-disk-mode><span class="muted small">disk cache</span>${z([`on`,`off`],r.mode,e=>i(`mode`,e),{small:!0,label:`disk cache mode`})}<span class="small">${r.mode===`on`?`prefixes are read from and written to the disk`:`nothing is read or written; the files stay`}</span></div>
        <div class="form-grid">
          ${a(`budget_gb`,`cache budget (GB)`,1,4e3)}
          ${a(`ttl_days`,`drop an entry unused for (days)`,1,90)}
          ${a(`admit_after`,`write a prefix after it was prefilled (times)`,1,10)}
          ${a(`min_free_gb`,`keep free on the NVMe (GB, at least 50)`,50,4e3)}
        </div>
        <p class="small" data-disk-free>
          ${e.free_bytes==null?E:t`The NVMe has <span class="num">${A(e.free_bytes,0)}</span> free${e.bytes?t`, and the cache holds <span class="num">${A(e.bytes,1)}</span>`:E}. `}
          ${o.short?t`<span class="warn-ink" data-disk-short>${x(`warn`)} At most <span class="num">${o.usable_gb.toFixed(0)} GB</span> of the ${r.budget_gb} GB fit above the ${Math.max(50,r.min_free_gb)} GB it keeps free; the tier stops there.</span>`:E}
        </p>
        <p class="muted small">Least valuable first when it is full: hits, times the tokens an entry saves per byte it holds, halved for every day unused. The first long request of an opencode session is written whatever the count, at the lines below where its user message starts. Applied from the next request; the free space it keeps is never under ${50} GB, whatever is set here.</p>
        ${s.length?t`<ul class="errors" role="alert" data-disk-errors>${s.map(e=>t`<li>${e}</li>`)}</ul>`:E}
        ${R(n)}
        ${B(n,`the settings`)}
      </div>
    </details>`}};g([m()],Mn.prototype,`disk`,void 0),g([m()],Mn.prototype,`error`,void 0),g([m()],Mn.prototype,`busy`,void 0),g([m()],Mn.prototype,`cleared`,void 0),Mn=g([w(`qse-cond-disk`)],Mn);var Nn=e=>e?v(e):`—`,Pn=()=>new Date().toISOString().slice(0,10),X=class extends N{constructor(...e){super(...e),this.selected=null,this.node=null,this.user=`owner`,this.list=null,this.one=null,this.error=null,this.oneError=null,this.filter={...Ut},this.query=``,this.hits=null,this.injected=null,this.writes=null,this.busy=null,this.editing=null,this.raw=!1,this.diffAt=null,this.confirmUser=``,this.flash=null,this.settings=new F(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.reload(),this.settings.load()}willUpdate(e){e.has(`node`)&&(this.filter={...this.filter,node:this.node});let t=this.selected?`${this.user}#${this.selected}`:null;t!==this.loadedFor&&(this.loadedFor=t,this.one=null,this.oneError=null,this.editing=null,this.diffAt=null,this.selected&&this.loadOne(this.selected))}async reload(){try{let[e,t,n]=await Promise.all([k.notes(this.user),k.notesInjected(this.user,30),k.notesWrites(30)]);this.list=e,this.injected=t.rows,this.writes=n.rows,this.error=null}catch(e){this.error=j(e).message}}async loadOne(e){try{let t=await k.note(e,this.user);e===this.selected&&(this.one=t)}catch(t){e===this.selected&&(this.oneError=j(t).message)}}async act(e,t,n){this.busy=e,this.flash=null;try{await t(),n&&(this.flash=n),await this.reload(),this.selected&&await this.loadOne(this.selected)}catch(e){this.oneError=j(e).message}finally{this.busy=null}}async search(){let e=this.query.trim();if(!e){this.hits=null;return}try{this.hits=(await k.notesSearch(e,this.user,10)).hits}catch(e){this.error=j(e).message}}async download(){try{let e=await k.notesExport(this.user),t=document.createElement(`a`);t.href=URL.createObjectURL(e),t.download=`memory-${this.user}.zip`,t.click(),setTimeout(()=>URL.revokeObjectURL(t.href),1e3)}catch(e){this.error=j(e).message}}open(e){u({note:e})}render(){if(this.error&&!this.list)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.list)return P(6);let e=this.list;return t`
      ${this.renderTiles(e)}
      ${this.flash?t`<p class="ok-ink small" role="status" data-notes-flash>${x(`check`)} ${this.flash}</p>`:E}
      ${this.selected?this.renderOne():E}
      ${this.renderSearch()}
      ${this.renderBrowser(e)}
      ${this.renderInjected()}
      ${this.renderRuns()}
      ${this.renderWrites()}
      ${this.renderSettings()}
      ${this.renderDanger(e)}
    `}renderTiles(e){let n=new Map;for(let t of e.notes)n.set(t.kind,(n.get(t.kind)??0)+1);let r=e.notes.filter(e=>e.status===`provisional`).length,i=Pn(),a=(this.injected??[]).filter(e=>new Date(e.ts_ms).toISOString().slice(0,10)===i).length,o=(this.writes??[]).filter(e=>new Date(e.ts_ms).toISOString().slice(0,10)===i&&/^refused/.test(qt(e))).length,s=[...n.entries()].sort((e,t)=>t[1]-e[1]).map(([e,t])=>`${t} ${e.replace(`_`,` `)}`).join(` · `);return t`<div class="stats notes-stats" data-notes-tiles>
      ${S({label:`live notes`,value:y(e.notes.length),sub:s||`none yet`})}
      ${S({label:`provisional`,value:y(r),sub:t`<span class="num">${y(e.notes.length-r)}</span> ratified by you`,warn:r>0&&r===e.notes.length&&e.notes.length>5})}
      ${S({label:`injected today`,value:y(a),unit:`requests`,sub:t`<span class="num">${y(e.memory.index_notes+e.memory.body_notes)}</span> notes since the start`})}
      ${S({label:`refused today`,value:y(o),sub:t`<span class="num">${y(e.memory.taken)}</span> blocks written, <span class="num">${y(e.memory.failed)}</span> refused since the start`})}
    </div>`}renderSearch(){return o(`Search`,this.hits===null?t`<p class="muted small">The runtime's own search (BM25 over title, tags and body, every scope of ${this.user}): the scores are the ones that decide what goes into a request.</p>`:this.hits.length?t`<ol class="notes-hits" data-notes-hits>${this.hits.map(e=>t`<li><button class="link-btn" @click=${()=>this.open(e.id)} data-hit=${e.id}>${e.title}</button>
                <span class="badge">${e.kind}</span> <span class="badge">${e.scope}</span> <span class="num small" data-score>${e.score.toFixed(2)}</span>
                ${e.status===`provisional`?t`<span class="badge badge-shadow">provisional</span>`:E}</li>`)}</ol>`:O(`No match`,`No note shares a word with the query.`),{tools:t`<form class="toolbar notes-search" style="margin:0" @submit=${e=>{e.preventDefault(),this.search()}}><label class="field field-search">${x(`search`)}<input placeholder="words, as the runtime would search them" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search the notes" data-notes-query /></label>
          <button class="btn btn-sm" type="submit" data-notes-search>Search</button>${this.hits?t`<button class="btn btn-sm btn-ghost" type="button" @click=${()=>(this.hits=null,this.query=``)}>Clear</button>`:E}</form>`,id:`notes-search`})}renderBrowser(e){let n=Wt(e.notes,this.filter),r=e=>u({nnode:e,note:null}),i=t`<nav class="notes-tree" aria-label="memory scopes" data-notes-tree>
      <button class="notes-node ${this.node?``:`is-on`}" @click=${()=>r(null)} data-node="all"><span>All notes</span><span class="num muted">${e.notes.length}</span></button>
      ${e.tree.map(e=>t`<button class="notes-node ${this.node===e.node?`is-on`:``}" @click=${()=>r(e.node)} data-node=${e.node} title=${e.label??e.node}>
          <span class="notes-node-label">${e.provisional?t`<span class="dot dot-shadow" aria-label="has provisional notes"></span>`:E}${Gt(e)}</span>
          <span class="num muted">${e.count}</span></button>`)}
    </nav>`,a=t`<label class="field notes-picker"><span>scope</span><select @change=${e=>r(e.target.value||null)} aria-label="memory scope" data-notes-picker>
        <option value="" ?selected=${!this.node}>All notes (${e.notes.length})</option>
        ${e.tree.map(e=>t`<option value=${e.node} ?selected=${this.node===e.node}>${Gt(e)} (${e.count})</option>`)}
      </select></label>`,s=this.filter,c=e=>this.filter={...this.filter,...e},l=t`<div class="toolbar notes-filters" style="margin:0 0 8px">
      <select aria-label="kind" @change=${e=>c({kind:e.target.value})} data-filter="kind"><option value="">every kind</option>${Vt.map(e=>t`<option value=${e} ?selected=${s.kind===e}>${e.replace(`_`,` `)}</option>`)}</select>
      <select aria-label="status" @change=${e=>c({status:e.target.value})} data-filter="status"><option value="">every status</option><option value="provisional">provisional</option><option value="ratified">ratified</option></select>
      <select aria-label="writer" @change=${e=>c({writer:e.target.value})} data-filter="writer"><option value="">every writer</option>${Object.entries(Ht).map(([e,n])=>t`<option value=${e}>${n}</option>`)}</select>
      <label class="switch small"><input type="checkbox" .checked=${s.stale} @change=${e=>c({stale:e.target.checked})} data-filter="stale" /> stale</label>
      <label class="field field-search">${x(`search`)}<input placeholder="filter by title or tag" .value=${s.q} @input=${e=>c({q:e.target.value})} aria-label="filter the list" /></label>
    </div>`,d=e=>t`<span class="badge ${e.status===`ratified`?`badge-ok`:`badge-shadow`}">${e.status}</span>${e.pinned?t` <span class="badge badge-on">pinned</span>`:E}${e.stale?t` <span class="badge badge-warn">stale</span>`:E}`,f=t`<div class="table-scroll notes-table"><table class="table" data-notes-list>
        <thead><tr><th>title</th><th>kind</th><th>status</th><th>updated</th><th>writer</th><th class="num-col">uses 7d</th></tr></thead>
        <tbody>${n.map(e=>t`<tr class="req-row ${e.id===this.selected?`is-open`:``}" data-note-row=${e.id} @click=${()=>this.open(e.id)}>
            <td class="cell-wrap">${e.title}</td><td><span class="badge" title=${Bt[e.kind]??``}>${e.kind.replace(`_`,` `)}</span></td><td>${d(e)}</td>
            <td class="num small">${Nn(e.updated)}</td><td class="small">${Ht[e.writer]??e.writer}</td><td class="num num-col">${y(e.uses_7d??0)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards notes-cards" role="list">${n.map(e=>t`<div class="mem-card ${e.id===this.selected?`is-open`:``}" role="listitem" data-note-card=${e.id} @click=${()=>this.open(e.id)}>
          <div class="mem-card-head"><span class="small">${e.title}</span></div>
          <div class="small">${e.kind.replace(`_`,` `)} · ${d(e)}</div>
          <div class="small muted">${Nn(e.updated)} · ${Ht[e.writer]??e.writer} · ${y(e.uses_7d??0)} uses in 7 days</div>
        </div>`)}</div>`,p=e.notes.length?O(`Nothing here`,`No note in this scope matches the filters.`):O(`No notes yet`,e.store.refused?`The memory directory is inside a git work tree (${e.store.refused}), so nothing is written.`:`The model writes a note when you ask it to remember something, correct it, or decide something; with the profile's memory switch on or in shadow.`);return o(`Notes`,t`<div class="notes-browser">${i}<div class="notes-main">${a}${l}${n.length?f:p}</div></div>`,{sub:t`<span data-notes-root>${y(e.notes.length)} live · ${y(e.archived.length)} archived${e.duplicates.length?t` · <span class="warn-ink">${e.duplicates.length} possible duplicate${e.duplicates.length===1?``:`s`}</span>`:E} · <span class="mono">${e.store.root}/${this.user}</span></span>`,tools:t`${e.users.length>1?t`<select aria-label="user" @change=${e=>(this.user=e.target.value,u({note:null,nnode:null}),void this.reload())} data-notes-user>${e.users.map(e=>t`<option ?selected=${e===this.user}>${e}</option>`)}</select>`:E}
          <button class="btn btn-sm" @click=${()=>this.download()} data-notes-export>${x(`down`)} Export</button>
          <button class="btn btn-sm" @click=${()=>this.reload()} aria-label="reload the notes">${x(`refresh`)}</button>`,id:`notes`})}renderOne(){if(this.oneError&&!this.one)return t`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${x(`warn`)} ${this.oneError}</div></div>`;let e=this.one;if(!e)return P(4);let n=e.note,r=!!n.archived,i=n.status===`ratified`&&(n.kind===`rule`||n.kind===`ruling`),a=e=>this.busy===e,o=n.id,s=r?t`<button class="btn btn-sm btn-danger" ?disabled=${a(`delete`)} @click=${()=>this.del(n)} data-note-act="delete">${x(`trash`)} Delete now</button>`:t`<button class="btn btn-sm" @click=${()=>this.startEdit(n)} data-note-act="edit">${x(`edit`)} Edit</button>
          ${n.status===`provisional`?t`<button class="btn btn-sm" ?disabled=${a(`ratify`)} @click=${()=>this.act(`ratify`,()=>k.noteStatus(o,{status:`ratified`},this.user),`ratified`)} data-note-act="ratify">${x(`check`)} Ratify</button>`:E}
          ${n.status===`provisional`?t`<button class="btn btn-sm" ?disabled=${a(`reject`)} @click=${()=>this.act(`reject`,()=>k.noteArchive(o,`rejected in the dashboard`,this.user),`rejected and archived`)} data-note-act="reject">Reject</button>`:E}
          ${i?t`<button class="btn btn-sm" ?disabled=${a(`pin`)} @click=${()=>this.act(`pin`,()=>k.noteStatus(o,{pinned:!n.pinned},this.user),n.pinned?`unpinned`:`pinned to every request`)} data-note-act="pin">${n.pinned?`Unpin`:`Pin to every request`}</button>`:E}
          <button class="btn btn-sm" ?disabled=${a(`archive`)} @click=${()=>this.archive(n)} data-note-act="archive">Archive</button>
          <button class="btn btn-sm btn-danger" ?disabled=${a(`delete`)} @click=${()=>this.del(n)} data-note-act="delete">${x(`trash`)} Delete</button>`,c=t`<dl class="notes-meta" data-note-meta>
      <dt>kind</dt><dd><span class="badge">${n.kind.replace(`_`,` `)}</span> <span class="muted small">${Bt[n.kind]??``}</span></dd>
      <dt>scope</dt><dd>${n.scope}${n.project?t` · <span class="mono small">${n.project}</span>`:E}${n.session?t` · <a class="mono small" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(n.session)}`}>${n.session}</a>`:E}</dd>
      <dt>status</dt><dd>${n.status}${n.pinned?` · pinned`:``}${n.stale?t` · <span class="warn-ink">stale since ${n.verify_by}</span>`:E}${r?t` · <span class="warn-ink">archived ${Nn(n.archived)}: ${n.reason}</span>`:E}</dd>
      <dt>writer</dt><dd>${Ht[n.writer]??n.writer}</dd>
      <dt>tags</dt><dd>${n.tags.length?n.tags.map(e=>t`<span class="badge">${e}</span> `):t`<span class="muted">—</span>`}</dd>
      ${n.verify_by?t`<dt>verify by</dt><dd class="num">${n.verify_by}</dd>`:E}
      <dt>created</dt><dd class="num">${Nn(n.created)}</dd><dt>updated</dt><dd class="num">${Nn(n.updated)}</dd>
      <dt>sources</dt><dd>${n.sources.length?n.sources.map(e=>t`<span class="mono small">${e}</span> `):t`<span class="muted">—</span>`}</dd>
      <dt>used</dt><dd data-note-uses>${Xt(e.uses,Pn())}</dd>
    </dl>`,l=this.editing?this.renderEditor():t`<div class="notes-body">
        <div class="toolbar" style="margin:0 0 6px">${z([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`body view`})}</div>
        ${this.raw?t`<pre class="mem-text" data-note-raw>${n.body}</pre>`:t`<div class="notes-rendered" data-note-body>${Fn(n.body)}</div>`}
      </div>`;return t`<section class="panel notes-one" style="margin-bottom:16px" data-note=${o}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${n.title}</h2><p class="panel-sub mono">${o}</p></div>
        <div class="panel-tools notes-actions">${s}<button class="btn btn-ghost btn-icon" aria-label="Close the note" @click=${()=>u({note:null})}>${x(`x`)}</button></div></header>
      ${this.oneError?t`<p class="err-ink small" role="alert">${x(`warn`)} ${this.oneError}</p>`:E}
      <div class="notes-file">${c}${l}</div>
      ${this.renderHistory(e)}
    </section>`}startEdit(e){this.editing={title:e.title,body:e.body,kind:e.kind,tags:e.tags.join(`, `),verify_by:e.verify_by??``}}renderEditor(){let e=this.editing,n=t=>this.editing={...e,...t},r=e.body.trim()?e.body.trim().split(/\s+/).length:0,i=e.title.length>120||r>400||!e.body.trim();return t`<div class="notes-editor" data-note-editor>
      <label class="field"><span>title <span class="muted num">${e.title.length}/120</span></span><input .value=${e.title} @input=${e=>n({title:e.target.value})} data-edit="title" /></label>
      <div class="form-grid">
        <label class="field"><span>kind</span><select @change=${e=>n({kind:e.target.value})} data-edit="kind">${Vt.map(n=>t`<option value=${n} ?selected=${n===e.kind}>${n.replace(`_`,` `)}</option>`)}</select></label>
        <label class="field"><span>tags (comma)</span><input .value=${e.tags} @input=${e=>n({tags:e.target.value})} data-edit="tags" /></label>
        <label class="field"><span>verify by</span><input type="date" .value=${e.verify_by} @input=${e=>n({verify_by:e.target.value})} data-edit="verify_by" /></label>
      </div>
      <label class="field"><span>body <span class="muted num">${r}/400 words</span></span><textarea rows="8" .value=${e.body} @input=${e=>n({body:e.target.value})} data-edit="body"></textarea></label>
      <div class="toolbar" style="margin:0"><button class="btn btn-primary btn-sm" ?disabled=${i||this.busy===`edit`} @click=${()=>this.act(`edit`,async()=>{await k.noteEdit(this.selected,{title:e.title,body:e.body,kind:e.kind,tags:e.tags.split(`,`).map(e=>e.trim()).filter(Boolean),verify_by:e.verify_by||null},this.user),this.editing=null},`saved as a new revision, ratified`)} data-edit-save>Save as a revision</button><button class="btn btn-sm btn-ghost" @click=${()=>this.editing=null}>Cancel</button>
        ${i?t`<span class="warn-ink small">a title of at most 120 characters, a body of 1 to 400 words</span>`:t`<span class="muted small">Your edit ratifies the note; the old text stays in its history.</span>`}</div>
    </div>`}renderHistory(e){let n=e.revisions,r=e.note,i=[...n.map(e=>({label:`revision ${e.n} · ${Nn(e.updated)} · ${Ht[e.writer]??e.writer}`,title:e.title,body:e.body})),{label:`now · ${Nn(r.updated)}`,title:r.title,body:r.body}],a=this.diffAt,o=E;if(a!==null&&a+1<i.length){let e=i[a],n=i[a+1],r=Kt(`# ${e.title}\n${e.body}`,`# ${n.title}\n${n.body}`);o=t`<pre class="notes-diff" data-note-diff>${r.map(e=>t`<span class="diff-${e.op===`+`?`add`:e.op===`-`?`del`:`eq`}">${e.op===`=`?` `:e.op} ${e.text}\n</span>`)}</pre>`}return t`<details class="sec" ?open=${n.length>0} data-note-history>
      <summary>History <span class="sec-sub">${r.changelog.length} change${r.changelog.length===1?``:`s`}, ${n.length} earlier text${n.length===1?``:`s`}</span></summary>
      <div class="sec-body">
        <ul class="notes-changelog">${r.changelog.map(e=>t`<li class="small">${e}</li>`)}</ul>
        ${n.length?t`<ol class="notes-revs">${i.slice(0,-1).map((e,n)=>t`<li><button class="link-btn ${a===n?`is-on`:``}" @click=${()=>this.diffAt=a===n?null:n} data-rev=${n}>${e.label}</button> <span class="muted small">to the next</span></li>`)}</ol>${o}`:t`<p class="muted small">No earlier text: this is the note as it was written.</p>`}
      </div>
    </details>`}async archive(e){let t=window.prompt(`Why is this note archived? (kept with it for 30 days)`,`no longer true`);t!==null&&await this.act(`archive`,()=>k.noteArchive(e.id,t||`archived in the dashboard`,this.user),`archived`)}async del(e){await L(`Delete this note?`,[t`<strong>${e.title}</strong>`,`The file, its history and any archived copy go at once. This cannot be undone.`],`Delete`)&&(await this.act(`delete`,()=>k.noteDelete(e.id,this.user),`deleted`),u({note:null}))}renderInjected(){let e=this.injected??[];return t`<details class="sec" style="margin-top:16px" data-notes-injected>
      <summary>Injected <span class="sec-sub">the last requests the notes went into (the index once a session, the matching notes on a new message)</span></summary>
      <div class="sec-body">${e.length?t`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>session</th><th>what</th><th>notes</th></tr></thead><tbody>${e.map(e=>t`<tr><td class="num small">${v(new Date(e.ts_ms).toISOString())}</td>
              <td class="mono small cell-wrap">${e.session_key?t`<a href=${`#/conductor?screen=memory&memory=${encodeURIComponent(e.session_key)}`}>${e.session_key}</a>`:`—`}</td>
              <td class="small">${e.index?`index`:`bodies`}</td>
              <td class="cell-wrap">${e.ids.map(e=>t`<button class="link-btn mono small" @click=${()=>this.open(e)}>${e}</button> `)}</td></tr>`)}</tbody></table></div>`:t`<p class="muted small">No note went into a request yet.</p>`}</div>
    </details>`}renderRuns(){let e=(this.writes??[]).filter(e=>e.task===`writer`).map(Yt);return t`<details class="sec" data-notes-runs ?open=${e.length>0}>
      <summary>Writer runs <span class="sec-sub">the model's hidden turns that read a session and write its notes: after a compaction, after a give-up, at the session's end</span></summary>
      <div class="sec-body">${e.length?t`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>trigger</th><th>Laya's gate</th><th>outcome</th><th class="num-col">seconds</th><th class="num-col">tokens</th><th>notes</th><th>refused</th></tr></thead><tbody>${e.map(e=>t`<tr data-run=${e.trigger}><td class="num small">${v(new Date(e.when).toISOString())}</td><td>${e.trigger.replace(`_`,` `)}</td><td class="small">${e.gate}</td><td class="small">${e.outcome}</td>
              <td class="num num-col">${e.seconds??`—`}</td><td class="num num-col">${e.tokens??`—`}</td><td class="small">${e.ops}</td><td class="small warn-ink">${e.refused.join(`, `)}</td></tr>`)}</tbody></table></div>`:t`<p class="muted small">No run yet. The writer is ${this.list?.settings.writer?.mode??`off`} (settings below).</p>`}</div>
    </details>`}renderWrites(){let e=(this.writes??[]).filter(e=>e.task!==`writer`);return t`<details class="sec" data-notes-writes>
      <summary>Writes <span class="sec-sub">every block the model wrote, the script calls, your edits, the nightly job, and what was refused and why</span></summary>
      <div class="sec-body">${e.length?t`<ul class="notes-writes">${e.map(e=>t`<li class="small ${/^refused/.test(qt(e))?`warn-ink`:``}"><span class="num muted">${v(new Date(e.ts_ms).toISOString())}</span> ${qt(e)}</li>`)}</ul>`:t`<p class="muted small">Nothing written yet.</p>`}</div>
    </details>`}renderSettings(){let e=this.settings;if(!e.draft)return e.loadError?t`<p class="err-ink small">${e.loadError}</p>`:E;let n=Jt(e.draft),r=(t,r)=>e.set(C(e.draft,[`memory`,`notes`],{...n,[t]:r})),i=(e,i,a,o)=>t`<label class="field"><span>${i}</span><input type="number" min=${a} max=${o} .value=${String(n[e])} data-notes-setting=${e} @input=${t=>r(e,Math.round(Number(t.target.value)))} /></label>`;return t`<details class="sec" data-sec="notes-settings">
      <summary>Notes settings <span class="sec-sub">what goes into a request, the duplicate rule, Laya's remember question</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          ${i(`index_tokens`,`the index, once a session (tokens)`,0,8e3)}
          ${i(`body_tokens`,`the matching notes on a new message (tokens)`,0,4e3)}
          ${i(`pinned_tokens`,`pinned rules (tokens)`,0,2e3)}
          ${i(`bodies`,`matching notes at most`,0,10)}
          ${i(`min_words`,`words a note shares with the message at least`,1,10)}
          <label class="field"><span>a second note this similar is a revision</span><input type="number" min="0" max="1" step="0.05" .value=${String(n.dedupe_threshold)} data-notes-setting="dedupe_threshold" @input=${e=>r(`dedupe_threshold`,Number(e.target.value))} /></label>
        </div>
        <div class="toolbar" style="margin:0" data-notes-intent><span class="muted small">Laya asks "remember, forget or recall?"</span>${z([`off`,`shadow`,`on`],n.intent,e=>r(`intent`,e),{small:!0,label:`intent question`})}</div>
        <label class="switch small"><input type="checkbox" .checked=${n.provisional} @change=${e=>r(`provisional`,e.target.checked)} /> provisional notes go into requests too (marked)</label>
        ${this.renderWriterSettings(n,r)}
        <p class="muted small">A profile's <code>memory</code> switch turns the notes on (off, shadow: the model may write and nothing is read back, on). Every note passes the secret-pattern refusal; a rule about you needs your own message in the turn; the model never ratifies, pins or deletes.</p>
        ${R(e)}
        ${B(e,`the settings`)}
      </div>
    </details>`}renderWriterSettings(e,n){let r=e.writer,i=(e,t)=>n(`writer`,{...r,[e]:t}),a=(e,n,a,o)=>t`<label class="field"><span>${n}</span><input type="number" min=${a} max=${o} .value=${String(r[e])} data-writer-setting=${e} @input=${t=>i(e,Number(t.target.value))} /></label>`;return t`<h4 class="small" style="margin:12px 0 4px">The background writer</h4>
      <div class="toolbar" style="margin:0" data-writer-mode><span class="muted small">writer</span>${z([`off`,`shadow`,`on`],r.mode,e=>i(`mode`,e),{small:!0,label:`writer mode`})}
        <span class="muted small">after a compaction</span>${z([`off`,`idle`,`turn`],r.after_compaction,e=>i(`after_compaction`,e),{small:!0,label:`after a compaction`})}</div>
      <div class="form-grid">
        ${a(`idle_s`,`a session's end: seconds without a request`,10,86400)}
        ${a(`min_new_tokens`,`new tokens before the next pass`,0,1e6)}
        ${a(`max_tokens`,`the turn writes at most (tokens)`,64,4096)}
      </div>
      <label class="switch small"><input type="checkbox" .checked=${r.gate} @change=${e=>i(`gate`,e.target.checked)} /> Laya decides first whether the session has anything worth keeping (about 20 ms; a no skips the turn)</label>
      <p class="muted small">shadow: the turn runs and its notes are counted, nothing is stored. <code>turn</code> after a compaction writes before the next request gets the engine (the next turn waits up to ${r.turn_deadline_s} s); <code>idle</code> waits for an idle moment, which opencode rarely leaves right after a compaction.</p>`}renderDanger(e){let n=this.confirmUser===this.user;return t`<details class="sec" data-notes-danger>
      <summary>Delete ${this.user}'s memory <span class="sec-sub">every note, revision and archive of this user, at once</span></summary>
      <div class="sec-body">
        <p class="small">Type <code>${this.user}</code> to delete ${y(e.notes.length+e.archived.length)} notes. Export them first if you want a copy.</p>
        <div class="toolbar" style="margin:0"><label class="field"><span>user</span><input .value=${this.confirmUser} @input=${e=>this.confirmUser=e.target.value} data-notes-confirm aria-label="type the user's name" /></label>
          <button class="btn btn-sm btn-danger" ?disabled=${!n||this.busy===`user`} @click=${()=>this.act(`user`,()=>k.notesDeleteUser(this.user,this.confirmUser),`${this.user}'s memory deleted`).then(()=>this.confirmUser=``)} data-notes-delete-user>${x(`trash`)} Delete everything</button></div>
      </div>
    </details>`}};g([T({attribute:!1})],X.prototype,`selected`,void 0),g([T({attribute:!1})],X.prototype,`node`,void 0),g([m()],X.prototype,`user`,void 0),g([m()],X.prototype,`list`,void 0),g([m()],X.prototype,`one`,void 0),g([m()],X.prototype,`error`,void 0),g([m()],X.prototype,`oneError`,void 0),g([m()],X.prototype,`filter`,void 0),g([m()],X.prototype,`query`,void 0),g([m()],X.prototype,`hits`,void 0),g([m()],X.prototype,`injected`,void 0),g([m()],X.prototype,`writes`,void 0),g([m()],X.prototype,`busy`,void 0),g([m()],X.prototype,`editing`,void 0),g([m()],X.prototype,`raw`,void 0),g([m()],X.prototype,`diffAt`,void 0),g([m()],X.prototype,`confirmUser`,void 0),g([m()],X.prototype,`flash`,void 0),X=g([w(`qse-cond-notes`)],X);function Fn(e){let n=[],r=(e??``).split(`
`),i=0,a=e=>e.split(/(`[^`]+`)/).map(e=>e.startsWith("`")&&e.endsWith("`")&&e.length>1?t`<code>${e.slice(1,-1)}</code>`:e);for(;i<r.length;){let e=r[i];if(e.startsWith("```")){let e=[];for(i++;i<r.length&&!r[i].startsWith("```");)e.push(r[i++]);i++,n.push(t`<pre class="mem-text">${e.join(`
`)}</pre>`)}else if(/^\s*[-*] /.test(e)){let e=[];for(;i<r.length&&/^\s*[-*] /.test(r[i]);)e.push(r[i++].replace(/^\s*[-*] /,``));n.push(t`<ul>${e.map(e=>t`<li>${a(e)}</li>`)}</ul>`)}else if(!e.trim())i++;else{let e=[];for(;i<r.length&&r[i].trim()&&!r[i].startsWith("```")&&!/^\s*[-*] /.test(r[i]);)e.push(r[i++]);n.push(t`<p>${a(e.join(` `))}</p>`)}}return t`${n}`}var In={tools:`requests with tools in full (and compactions); chats without tools as counters only`,all:`every request in full`,none:`nothing`},Z=class extends N{constructor(...e){super(...e),this.ids=[],this.selected=null,this.page=null,this.tab=null,this.note=null,this.nnode=null,this.list=null,this.one=null,this.error=null,this.oneError=null,this.query=``,this.busy=null,this.row=null,this.ev=null,this.rowError=null,this.settings=new F(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.tab!==`disk`&&this.tab!==`notes`&&(this.loadList(),this.settings.load())}updated(e){let t=e=>e===`disk`||e===`notes`;t(e.get(`tab`))&&!t(this.tab)&&!this.list&&!this.error&&(this.loadList(),this.settings.draft||this.settings.load())}willUpdate(){let e=this.selected?`${this.selected}#${this.page??``}`:null;e!==this.loadedFor&&(this.loadedFor=e,this.one=null,this.oneError=null,this.selected&&this.loadOne(this.selected,this.page))}async loadList(){try{this.list=await k.memorySessions(),this.error=null}catch(e){this.error=j(e).message}}async loadOne(e,t){try{let n=await k.memorySession(e,t);e===this.selected&&(this.one=n)}catch(t){e===this.selected&&(this.oneError=j(t).message)}}async keep(e,t){if(e.session){this.busy=e.dir;try{await k.memoryKeep(e.session,t),await this.loadList(),this.one&&this.one.session.session===e.session&&(this.one={...this.one,session:{...this.one.session,keep:t}})}catch(e){this.error=j(e).message}finally{this.busy=null}}}async download(e,t){try{let n=await k.memoryExport(e,t),r=document.createElement(`a`);r.href=URL.createObjectURL(n),r.download=`transcript-${e.replace(/[^a-z0-9]+/gi,`-`)}.${t}`,r.click(),setTimeout(()=>URL.revokeObjectURL(r.href),1e3)}catch(e){this.oneError=j(e).message}}async openRow(e){this.rowError=null;try{let t=(await k.decision(e)).decision;t.kind===`monitor`?this.ev=t:this.row=t}catch(t){this.rowError=`journal row ${e}: ${j(t).message} (the journal keeps 30 days)`}}render(){let e=this.tab===`disk`,n=this.tab===`disk`||this.tab===`notes`?this.tab:null,r=(e,r)=>t`<a class="cond-tab ${e===n?`is-on`:``}" href="#" aria-current=${e===n?`page`:E} data-mem-tab=${e??`sessions`} @click=${t=>{t.preventDefault(),u({mtab:e,memory:null,mpage:null,note:null,nnode:null})}}>${r}</a>`,i=t`<nav class="mem-tabs" aria-label="memory tabs">${r(null,`Sessions`)}${r(`notes`,`Notes`)}${r(`disk`,`Disk cache`)}</nav>`;return e?t`${i}<qse-cond-disk></qse-cond-disk>`:n===`notes`?t`${i}<qse-cond-notes .selected=${this.note} .node=${this.nnode}></qse-cond-notes>`:t`
      ${i}
      ${this.selected?this.renderOne(this.selected):E}
      ${this.renderList()}
      ${this.renderSettings()}
      ${this.row?t`<qse-cond-decision .row=${this.row} .ids=${this.ids} .close=${()=>this.row=null}></qse-cond-decision>`:E}
      ${this.ev?t`<qse-cond-event .ev=${this.ev} .close=${()=>this.ev=null}></qse-cond-event>`:E}
    `}renderList(){if(this.error&&!this.list)return t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`;if(!this.list)return P(5);let e=this.list.writer,n=this.query.toLowerCase(),r=this.list.sessions.filter(e=>!n||`${e.session??e.dir} ${e.client??``} ${e.project??``}`.toLowerCase().includes(n)),i=e=>e.last_ms?v(new Date(e.last_ms).toISOString()):`—`,a=n=>t`<label class="switch" title="kept sessions are exempt from the ${e.settings.retention_days}-day prune"><input type="checkbox" .checked=${n.keep} ?disabled=${!n.session||this.busy===n.dir} data-keep=${n.dir} @click=${e=>e.stopPropagation()} @change=${e=>this.keep(n,e.target.checked)} aria-label=${`keep ${Sn(n.session,n.dir)}`} /> keep</label>`,s=e=>e.checkers?.last?t`<span class="badge ${/ PASS$/.test(e.checkers.last)?`badge-ok`:`badge-warn`}">${e.checkers.last}</span>${e.checkers.PASS||e.checkers.FAIL?t` <span class="muted small num">${e.checkers.PASS??0} pass · ${e.checkers.FAIL??0} fail</span>`:E}`:t`<span class="muted">—</span>`,c=e=>e.session&&u({memory:e.session,mpage:null});return o(`Session transcripts`,r.length?t`<div class="table-scroll mem-table"><table class="table" data-memory-sessions>
              <thead><tr><th>last request</th><th>session</th><th>client</th><th>profile</th><th>project</th><th class="num-col">turns</th><th class="num-col">acts</th><th>checker</th><th class="num-col">size</th><th>keep</th></tr></thead>
              <tbody>${r.map(e=>t`<tr class="req-row ${e.session&&e.session===this.selected?`is-open`:``}" data-memory-row=${e.dir} @click=${()=>c(e)}>
                  <td class="num">${i(e)}</td><td class="mono small cell-wrap">${Sn(e.session,e.dir)}</td><td>${e.client??`—`}</td><td>${e.profile??`—`}</td>
                  <td class="small cell-wrap">${e.project??`—`}</td>
                  <td class="num num-col">${e.counts?y(e.counts.blocks):`?`}</td><td class="num num-col">${e.counts?y(e.counts.acts):`?`}</td>
                  <td>${s(e)}</td><td class="num num-col">${A(e.bytes??0)}</td><td>${a(e)}</td></tr>`)}</tbody></table></div>
            <div class="mem-cards" role="list">${r.map(e=>t`<div class="mem-card ${e.session&&e.session===this.selected?`is-open`:``}" role="listitem" data-memory-card=${e.dir} @click=${()=>c(e)}>
                <div class="mem-card-head"><span class="mono small">${Sn(e.session,e.dir)}</span>${a(e)}</div>
                <div class="small">${i(e)} · ${e.client??`—`} · ${e.profile??`—`}</div>
                <div class="small muted">${bn(e)}</div>
                ${e.project?t`<div class="small muted cell-wrap">${e.project}</div>`:E}
                <div>${s(e)}</div>
              </div>`)}</div>`:O(`No transcripts yet`,e.refused?`The memory directory ${e.root} is inside a git work tree, so nothing is written there.`:`TandemLLM requests with tools are written to ${e.root} when their stream closes; chats without tools are counted only.`),{sub:t`<span data-writer>${y(e.blocks)} block${e.blocks===1?``:`s`} written since the start · ${y(e.redactions)} secret${e.redactions===1?``:`s`} redacted · ${y(e.failed)} failed · ${y(e.dropped)} dropped · ${A(e.bytes)} · <span class="mono">${e.root}</span></span>`,tools:t`<label class="field field-search">${x(`search`)}<input placeholder="search a session, client or project" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search transcripts" /></label>
          <button class="btn btn-sm" @click=${()=>this.loadList()} aria-label="reload the list">${x(`refresh`)}</button>`,id:`cond-memory`})}renderOne(e){if(this.oneError)return t`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${x(`warn`)} ${this.oneError}</div></div>`;let n=this.one;if(!n)return P(4);let r=n.session,i=n.pages??1,a=n.page??1,o=i>1?t`<div class="toolbar mem-pager" data-pager><button class="btn btn-sm" ?disabled=${a<=1} @click=${()=>u({mpage:String(a-1)})}>Older</button>
          <span class="small num">page ${a} of ${i} · ${y(n.total??n.blocks.length)} blocks, ${n.per_page} a page</span>
          <button class="btn btn-sm" ?disabled=${a>=i} @click=${()=>u({mpage:String(a+1)})}>Newer</button></div>`:E;return t`<section class="panel" style="margin-bottom:16px" data-transcript=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title mono">${Sn(r.session,r.dir)}${r.keep?t` <span class="badge badge-on">kept</span>`:E}</h2>
        <p class="panel-sub">${r.client??`—`} · ${r.profile??`—`}${r.project?` · ${r.project}`:``} · ${bn(r)}${r.first_ms?` · since ${v(new Date(r.first_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <button class="btn btn-sm" @click=${()=>this.download(e,`md`)} data-mem-export="md">${x(`down`)} Markdown</button>
          <button class="btn btn-sm" @click=${()=>this.download(e,`json`)} data-mem-export="json">${x(`down`)} JSON</button>
          <a class="btn btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e)}`} data-mem-timeline>Timeline</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the transcript" @click=${()=>u({memory:null,mpage:null})}>${x(`x`)}</button></div></header>
      ${this.rowError?t`<p class="warn-ink small" role="alert">${x(`warn`)} ${this.rowError}</p>`:E}
      ${o}
      ${n.blocks.length?t`<div class="mem-blocks">${n.blocks.map(e=>this.renderBlock(e))}</div>`:O(`No blocks`,`This session was counted but not written: its requests carried no tools.`)}
      ${o}
    </section>`}renderBlock(e){let n=vn(e.text),r=n.runtime.filter(e=>e.kind===`hidden`).length,i=n.runtime.find(e=>e.kind===`acts`),a=t`<span class="mem-turn num">turn ${e.turn??`?`}</span>
      <span class="muted small num">${e.time??``}</span>
      ${e.class_id?I(e.class_id,this.ids):E}
      ${e.effort?t`<span class="badge">${e.effort}</span>`:E}
      ${e.reasoning_tokens?t`<span class="muted small num">${y(e.reasoning_tokens)} reasoning tokens</span>`:E}
      ${r?t`<span class="badge badge-shadow" data-hidden-count=${r}>${r} hidden</span>`:E}
      ${i?t`<span class="badge badge-warn">${i.text.replace(/^acts: /,`acts `)}</span>`:E}`;return e.compaction?t`<details class="mem-block is-compaction" data-block=${e.turn??``} data-compaction>
        <summary><span class="mem-block-head">${a}<span class="badge">compaction</span><span class="small">${n.compaction?.replace(/^compaction: /,``)??``}</span></span></summary>
        ${n.sections.map(e=>this.renderSection(e))}${this.renderRuntime(n.runtime)}
      </details>`:t`<article class="mem-block" data-block=${e.turn??``}>
      <div class="mem-block-head">${a}</div>
      ${n.sections.map(e=>this.renderSection(e))}
      ${this.renderRuntime(n.runtime)}
    </article>`}renderSection(e){let n=e.body.length>600||e.body.split(`
`).length>12,r=t`<pre class="mem-text">${e.body}</pre>`;return e.kind===`tool_result`||e.kind===`tool_call`&&n||e.kind===`reasoning`?t`<details class="mem-sec mem-${e.kind}" ?open=${!n&&e.kind!==`reasoning`} data-sec-kind=${e.kind}>
        <summary class="small">${e.title}${e.clipped?t` <span class="badge badge-warn">clipped</span>`:E}</summary>${r}</details>`:t`<div class="mem-sec mem-${e.kind}" data-sec-kind=${e.kind}><div class="small muted">${e.title}</div>${r}</div>`}renderRuntime(e){return e.length?t`<ul class="mem-runtime" aria-label="what the runtime did">${e.map(e=>t`<li class="mem-rt mem-rt-${e.kind} ${e.verdict?`is-${e.verdict.toLowerCase()}`:``}" data-rt=${e.kind}>
        ${e.kind===`hidden`?x(`info`):E}<span>${e.text.replace(/ · journal [A-Za-z0-9_-]+$/,``)}</span>
        ${e.journal?t`<button class="btn btn-ghost btn-sm" @click=${()=>this.openRow(e.journal)} data-journal=${e.journal} aria-label=${`open journal row ${e.journal}`}>journal row</button>`:E}
      </li>`)}</ul>`:E}renderSettings(){let e=this.settings;if(!e.draft)return e.loadError?t`<p class="err-ink small">${e.loadError}</p>`:E;let n=xn(e.draft),r=(t,r)=>e.set(C(e.draft,[`memory`,`transcripts`],{...n,[t]:r})),i=(e,i,a,o)=>t`<label class="field"><span>${i}</span><input type="number" min=${a} max=${o} .value=${String(n[e])} data-mem-setting=${e} @input=${t=>r(e,Math.round(Number(t.target.value)))} /></label>`;return t`<details class="sec" style="margin-top:16px" data-sec="transcripts">
      <summary>Transcript settings <span class="sec-sub">what the engine writes to the box, how long it keeps it, where tool output is clipped</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-mem-scope><span class="muted small">write</span>${z([`tools`,`all`,`none`],n.scope,e=>r(`scope`,e),{small:!0,label:`transcript scope`})}<span class="small">${In[n.scope]}</span></div>
        <div class="form-grid">
          ${i(`retention_days`,`keep a session for (days; kept sessions are exempt)`,1,400)}
          ${i(`clip_chars`,`clip a tool result or a call at (characters)`,200,1e5)}
          ${i(`loop_chars`,`a loop the D1 guard cut keeps its first and last (characters)`,0,4e3)}
        </div>
        <p class="muted small">Every block passes the secret-pattern refusal before it is written: a key, token or password becomes <code>[redacted:kind]</code> and the journal counts it, never the value. There is no switch for it. The files are <code>0600</code> under the engine's home on the box, never in a repository, and the ledger's backup never holds them. The owner's decision 2: chats without tools are counters only, unless the scope is widened here.</p>
        ${R(e)}
        ${B(e,`the settings`)}
      </div>
    </details>`}};g([T({attribute:!1})],Z.prototype,`ids`,void 0),g([T({attribute:!1})],Z.prototype,`selected`,void 0),g([T({attribute:!1})],Z.prototype,`page`,void 0),g([T({attribute:!1})],Z.prototype,`tab`,void 0),g([T({attribute:!1})],Z.prototype,`note`,void 0),g([T({attribute:!1})],Z.prototype,`nnode`,void 0),g([m()],Z.prototype,`list`,void 0),g([m()],Z.prototype,`one`,void 0),g([m()],Z.prototype,`error`,void 0),g([m()],Z.prototype,`oneError`,void 0),g([m()],Z.prototype,`query`,void 0),g([m()],Z.prototype,`busy`,void 0),g([m()],Z.prototype,`row`,void 0),g([m()],Z.prototype,`ev`,void 0),g([m()],Z.prototype,`rowError`,void 0),Z=g([w(`qse-cond-memory`)],Z);var Q=class extends N{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([k.stats(168),k.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let t=j(e);this.error=`${t.endpoint}: ${t.message}`}}async promote(e,t,n,r,i){if(await L(`Promote ${n} to ${t===`task`?`act`:`on`} in ${e}?`,[t===`task`?`The session monitor will act on ${n} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${n} decisions instead of only journaling them: ${n===`effort`?`budgets and thinking off go to the engine`:n===`packages`?`skills and packages go into the prompt`:n===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${r.toFixed(2)} over ${i} labelled decisions. A new profile version records it.`],`Promote`))try{let a=await k.doc(`profiles`,e),o=t===`task`?C(a.doc,[`tasks`,n,`mode`],`act`):C(a.doc,[`decisions`,n],`on`),s=await k.put(`profiles`,e,o,`${n} to ${t===`task`?`act`:`on`}, precision ${r.toFixed(2)} / ${i}`,a.version);this.msg=`${e} v${s.version}: ${n} promoted`,this.onSaved()}catch(e){this.msg=j(e).message}}async loadAb(){try{this.abVersions=(await k.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await k.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await k.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=j(e).message}}render(){return this.error&&!this.stats?t`<div class="state state-error" role="alert"><div class="state-title">${x(`warn`)} ${this.error}</div></div>`:this.stats?t`
      ${this.msg?t`<div class="note note-info" role="status">${this.msg}</div>`:E}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:P(8)}renderPromotion(e){let n=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,r=(e.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return o(`Shadow precision and promotion`,t`<p class="muted small">${ke}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${e.precision.length||r.length?t`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${e.precision.map(e=>{let r=Ke(e),i=n(e.profile,e.decision),a=e.backend?t`<div class="muted small mono" data-backend=${e.backend}>via ${e.backend}</div>`:E;return t`<tr data-precision-row=${`${e.profile}/${e.decision}${e.backend?`@${e.backend}`:``}`}>
                    <td>${e.profile}${a}</td><td class="mono">${e.decision}</td><td><span class="badge ${i===`on`?`badge-on`:i===`shadow`?`badge-shadow`:``}">${i}</span></td>
                    <td class="num num-col">${y(e.decisions)}</td><td class="num num-col">${y(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${r.allowed?``:`muted`}">${r.why}</td>
                    <td><button class="btn btn-sm ${r.allowed?`btn-primary`:``}" ?disabled=${!r.allowed||i!==`shadow`||!!e.backend} title=${e.backend?`a backend’s switches are on the Backends screen, per model`:r.allowed?``:ke} data-promote=${`${e.profile}/${e.decision}`} @click=${()=>this.promote(e.profile??``,`decision`,e.decision,e.precision??0,e.labelled)}>Promote to on</button></td></tr>`})}
                ${r.map(e=>{let n=Ke(e);return t`<tr data-precision-row=${`task/${e.task}${e.backend?`@${e.backend}`:``}`}><td>${e.backend?t`<span class="muted small mono">via ${e.backend}</span>`:`—`}</td><td class="mono">${e.task}</td><td><span class="badge ${e.mode===`act`?`badge-on`:`badge-shadow`}">${e.mode}</span></td>
                    <td class="num num-col">${y(e.decisions)}</td><td class="num num-col">${y(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${n.allowed?``:`muted`}">${n.why}</td>
                    <td><button class="btn btn-sm ${n.allowed?`btn-primary`:``}" ?disabled=${!n.allowed||e.mode!==`shadow`||!!e.backend} data-promote=${`task/${e.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,e.task,e.precision??0,e.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:O(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${e.monitor?E:t`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let e=this.reports.find(e=>e.kind===`classifier`);if(!e)return o(`Classifier accuracy`,O(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let n=e.stages??{},r=e.set??{},i=(n[this.stage]??Object.values(n)[0])?.confusion??{},a=r.classes??Object.keys(i);return o(t`Classifier accuracy <span class="muted small num">${r.name??``} · ${r.n??`?`} prompts · ${e.created_at?v(String(e.created_at)):``}</span>`,t`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(n).map(([e,n])=>t`<tr class="req-row ${e===this.stage?`is-open`:``}" @click=${()=>this.stage=e}><td class="mono">${e}</td><td class="num num-col">${b(n.accuracy,1)}</td><td class="num num-col">${n.answered} / ${n.n}</td><td class="num num-col">${n.accuracy_answered==null?`—`:b(n.accuracy_answered,1)}</td><td class="num num-col">${_(n.p50_ms)}</td><td class="num num-col">${_(n.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${a.length&&Object.keys(i).length?t`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...a,`—`].map(e=>t`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${a.map(e=>t`<tr><th>${I(e,this.ids)}</th>${[...a,`—`].map(n=>{let r=i[e]?.[n]??0;return t`<td class="cm-cell ${e===n?`cm-diag`:r?`cm-miss`:``}">${r||``}</td>`})}</tr>`)}</tbody></table></div>`:t`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}
        ${Le(e).length?t`<p class="muted small" style="margin-top:12px">Digest variants (SRV-153): <strong>today</strong> the 2,000-character digest the decider cut from the front, <strong>sized</strong> each stage's own head and tail, <strong>line</strong> sized plus the context line. <strong>router</strong> is the session's class after the stickiness rule.</p>
            <div class="table-scroll"><table class="table" data-classifier-variants>
              <thead><tr><th>digest</th><th>set</th><th class="num-col">n</th><th class="num-col">decider</th><th class="num-col">cascade</th><th class="num-col">router</th><th class="num-col">decider p50</th></tr></thead>
              <tbody>${Le(e).map(e=>t`<tr><td class="mono">${e.variant}</td><td class="mono small">${e.set}</td><td class="num num-col">${e.n}</td><td class="num num-col">${e.decider==null?`—`:b(e.decider,1)}</td><td class="num num-col">${e.cascade==null?`—`:b(e.cascade,1)}</td><td class="num num-col">${e.router==null?`—`:b(e.router,1)}</td><td class="num num-col">${_(e.p50)}</td></tr>`)}</tbody></table></div>`:E}`,{id:`cond-classifier`})}renderSuites(){let n=this.reports.filter(e=>e.kind===`suite`),r=ne(this.reports,this.suiteFile),i=e(r,n),a=r?.rows??[],s=me(this.reports,this.engine),c=te(this.reports,this.engine),l=se(a),u=r?.verdict,d=r?.comparison??[],p=r?.raw_row,ee=r?.build,m=u?.indicative??De(a),h=(e,t=1)=>e==null?`—`:Number.isInteger(e)?y(e):e.toFixed(t);return o(t`TandemLLM against the raw engine ${r?t`<span class="muted small num">night ${r.night??`—`}${ee?` · ${ee.version??`?`} ${f(ee.code_sha,12)}`:``}${r.profile?` · profile ${r.profile}`:``}</span>`:E}`,t`<p class="small muted" data-guard-detail=${s.status} data-speed-check=${c.state}>Speed check: ${c.text}. <span class="num">${s.line}${s.detail?t`. ${s.detail}`:E}</span></p>
        ${n.length>1?t`<label class="field suite-pick"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${n.map(i=>t`<option value=${String(i.file)} ?selected=${i===r}>${String(i.night??i.file)} · ${String(i.file)}${e(i,n)?` · superseded`:``}</option>`)}</select></label>`:E}
        ${i?t`<div class="guard-line guard-note" data-superseded>${x(`info`)}<span><span class="badge badge-warn">superseded</span> by <span class="mono">${i.file}</span>${i.why?t`: <span class="small">${i.why}</span>`:E}. Its numbers are kept as published; read the newer report.</span></div>`:E}
        ${u?t`<div class="note" data-eval-verdict>
              <span class="badge ${u.better?`badge-ok`:u.better===!1?`badge-warn`:``}"><strong>${u.better?`Better`:u.better===!1?`Not better`:`Not shown yet`}</strong></span>${m?t` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:E} — ${u.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${p?.status??`—`}${p?.source?` (${p.source})`:``}${r?.driver?` · driver ${r.driver.mode}`:``}${r?.isolation?` · ${r.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:E}
        ${l.length?t`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">turns</th><th class="num-col">thinking tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>efforts (turns)</th><th>build · night</th></tr></thead>
              <tbody>${l.flatMap(e=>{let n=d.find(t=>t.suite===e.suite||t.suite_id===(e.raw??e.tandem)?.suite_id);return[e.raw,e.tandem].filter(Boolean).map((r,i)=>t`<tr data-suite=${e.suite} data-arm=${r.arm}>
                    <td>${i===0?e.suite:``}${i===0&&!e.cmp.ok?t` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${e.cmp.why}</div>`:E}${i===0&&n?t` <span class="badge ${n.verdict===`better`?`badge-ok`:n.verdict===`worse`||n.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${n.verdict}>${n.verdict}</span>`:E}${i===0&&(r.indicative??(r.n??0)<10)?t` <span class="badge" data-row-indicative>indicative</span>`:E}</td>
                    <td class="mono">${ye(r.arm)}${r.pristine===!1?t` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:E}</td>
                    <td class="num num-col">${r.n??`—`}</td>
                    <td class="num num-col">${r.pass_rate==null?`—`:b(r.pass_rate,0)}</td>
                    <td class="num num-col">${r.gates_passed??`—`}${r.gates_total?` / ${r.gates_total}`:``}</td>
                    <td class="num num-col">${h(r.minutes_to_pass)}</td><td class="num num-col">${h(r.wall_minutes)}</td>
                    <td class="num num-col">${r.tokens==null?`—`:y(Math.round(r.tokens))}</td>
                    <td class="num num-col" data-turns>${h(r.turns)}</td><td class="num num-col" data-thinking>${r.reasoning_tokens==null?`—`:y(Math.round(r.reasoning_tokens))}</td>
                    <td class="num num-col">${h(r.decode_tps)}</td>
                    <td class="num num-col">${r.empty_stops??`—`}</td><td class="num num-col">${r.continuations??`—`}</td>
                    <td class="num num-col">${r.interventions??`—`}</td><td class="num num-col">${r.gave_ups??`—`}</td>
                    <td class="num num-col">${r.cache_hit==null?`—`:b(r.cache_hit,0)}</td>
                    <td class="small" data-efforts>${ae(r)}</td>
                    <td class="small">${f(r.build,16)} · ${r.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts. Tokens, turns and thinking tokens are medians per run.</p>`:O(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return o(`A/B of two profile versions`,t`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>t`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>t`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(n=>t`<option value=${n.version} ?selected=${n.version===this[e]}>v${n.version} ${f(n.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?t`<div class="diff" data-ab>${this.abChanges.map(e=>t`<div class="diff-row"><span>${i(e.path)}</span><span class="diff-before">${f(e.before,80)}</span><span class="diff-after">${f(e.after,80)}</span></div>`)}</div>`:t`<p class="muted small">The two versions are the same.</p>`:E}`,{id:`cond-ab`})}};g([T({attribute:!1})],Q.prototype,`ids`,void 0),g([T({attribute:!1})],Q.prototype,`tiles`,void 0),g([T({attribute:!1})],Q.prototype,`onSaved`,void 0),g([m()],Q.prototype,`stats`,void 0),g([m()],Q.prototype,`reports`,void 0),g([m()],Q.prototype,`engine`,void 0),g([m()],Q.prototype,`error`,void 0),g([m()],Q.prototype,`msg`,void 0),g([m()],Q.prototype,`stage`,void 0),g([m()],Q.prototype,`abProfile`,void 0),g([m()],Q.prototype,`abVersions`,void 0),g([m()],Q.prototype,`abA`,void 0),g([m()],Q.prototype,`abB`,void 0),g([m()],Q.prototype,`abChanges`,void 0),g([m()],Q.prototype,`abCounts`,void 0),g([m()],Q.prototype,`suiteFile`,void 0),Q=g([w(`qse-cond-compare`)],Q);var Ln=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`deciders`,label:`Deciders`},{id:`sessions`,label:`Sessions`},{id:`memory`,label:`Memory`},{id:`compare`,label:`Compare`}],$=class extends N{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=M(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&Ln.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r]=await Promise.all([k.profiles(),this.meta?Promise.resolve(this.meta):k.meta(),k.config(),k.doc(`router`)]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.error=null}catch(e){let t=j(e);this.error=t.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${t.endpoint}: ${t.message}`}}async setMode(e){let t=this.tiles.find(e=>e.name===this.active);if(t&&!t.locked&&(e!==`on`||await L(`Turn the ${t.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let n=await k.setMode(t.name,e,`mode to ${e}`,t.version);this.reloaded=`v${n.version} · reloaded ${n.loaded_at?qe(n.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=j(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),n=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return t`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${x(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>u({profile:e.target.value})}>
                ${this.tiles.map(e=>t`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?z([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):E}
            ${this.reloaded?t`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:E}
          </div>
        </div>
        ${this.switchError?t`<div class="guard-line" role="alert">${x(`warn`)} ${this.switchError}</div>`:E}
        ${this.error?t`<div class="guard-line" role="alert">${x(`warn`)} ${this.error}</div>`:E}
        ${n.map(([e,n])=>t`<div class="guard-line" role="alert">${x(`warn`)} ${e}: ${n.error?.message} (kept version ${n.version})</div>`)}
        ${e?.locked?t`<div class="note" data-locked>${x(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:E}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${Ln.map(e=>t`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:E}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return t`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return t`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return t`<qse-cond-monitor .meta=${e.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return t`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return t`<qse-cond-packages .ids=${e.ids} .tiles=${e.tiles} .selected=${this.params.get(`pkg`)} .tab=${this.params.get(`ptab`)} .pane=${this.params.get(`pview`)}></qse-cond-packages>`;case`deciders`:return t`<qse-cond-deciders></qse-cond-deciders>`;case`sessions`:return t`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`memory`:return t`<qse-cond-memory .ids=${e.ids} .selected=${this.params.get(`memory`)} .page=${Number(this.params.get(`mpage`))||null} .tab=${this.params.get(`mtab`)} .note=${this.params.get(`note`)} .nnode=${this.params.get(`nnode`)}></qse-cond-memory>`;case`compare`:return t`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};g([T({attribute:!1})],$.prototype,`params`,void 0),g([m()],$.prototype,`tiles`,void 0),g([m()],$.prototype,`defaultProfile`,void 0),g([m()],$.prototype,`meta`,void 0),g([m()],$.prototype,`config`,void 0),g([m()],$.prototype,`ids`,void 0),g([m()],$.prototype,`error`,void 0),g([m()],$.prototype,`switching`,void 0),g([m()],$.prototype,`reloaded`,void 0),g([m()],$.prototype,`switchError`,void 0),$=g([w(`qse-conductor`)],$);export{$ as QseConductor};