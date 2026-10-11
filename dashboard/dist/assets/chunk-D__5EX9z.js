import{C as e,_ as t,a as n,b as r,c as i,d as a,g as o,h as s,i as c,n as l,o as u,s as d,t as f,v as p,y as m}from"./chunk-CkiyBafN.js";import{c as h,d as g,f as _,m as v,n as ee,o as y,r as b}from"./chunk-BXmDkkzu.js";import{$ as te,B as ne,C as re,Ct as x,D as ie,E as ae,F as oe,G as se,H as ce,I as le,J as ue,K as de,L as fe,M as pe,N as me,O as he,P as ge,Q as _e,R as ve,T as ye,U as be,V as xe,W as Se,X as Ce,Y as S,Z as C,_ as we,at as Te,b as Ee,bt as De,ct as Oe,d as w,dt as ke,et as Ae,f as je,ft as Me,g as Ne,gt as T,h as Pe,it as Fe,j as Ie,k as Le,l as Re,lt as ze,mt as E,nt as Be,ot as Ve,p as He,pt as Ue,q as We,rt as Ge,st as Ke,tt as qe,u as Je,ut as Ye,v as Xe,w as Ze,x as Qe,y as $e,yt as et,z as tt}from"./app-DTFfdS-y.js";import{n as nt,t as rt}from"./chunk-DiaLOnD7.js";import{DocEditor as D,clsChip as O,confView as it,confirmDialog as k,diffList as at,errorsList as A,eventRow as ot,feedRow as st,modeSeg as j,saveBar as M,sourceWords as ct,versionsPanel as lt}from"./chunk-BdeAb3SG.js";var N=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=T(()=>this.loadStats(),2e4),this.feedPoll=T(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await E.stats(24),this.error=null}catch(e){let t=a(e);this.error=`${t.endpoint}: ${t.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await E.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await E.event(e)).event}catch(e){this.error=a(e).message}}renderMonitor(t){let i=xe(t);return e`<div class="grid-2 grid-2-wide">
      ${n(`Monitor acts an hour, last 24 h`,t.acts?e`<qse-ribbon .series=${i} .labels=${t.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:l(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${n(`Last actions of the monitor`,t.last_actions.length?e`<div class="feed" data-last-actions>${t.last_actions.map(t=>e`<button class="feed-row" data-action=${t.id} @click=${()=>this.openAction(t.id)}>
                <span class="feed-time">${v(new Date(t.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${t.task??`—`}</span></span>
                <span class="feed-conf muted small">${t.outcome?.kind??`running`}</span>
                <span class="feed-what">${et({task:t.task,decision:t.decision,mode:`act`,trigger:t.trigger,cap:t.cap})}</span>
                <span class="feed-act">${t.session_key?e`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(t.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:r}</span>
              </button>`)}</div>`:e`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:e`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${c(`external`)}</a>`})}
    </div>`}render(){let t=this.stats;if(this.error&&!t)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${this.error}</div></div>`;if(!t)return u(6);let i=t.classifier,a=i.decider,o=i.qwen,s=a?.calls?a:o?.calls?o:a,f=Object.entries(t.roles),p=f.filter(([,e])=>e.up).length,m=t.monitor;return e`
      <div class="stats" data-tiles>
        ${d({label:`decisions, 24 h`,value:h(t.decisions),sub:e`<span class="num">${h(t.acted)}</span> acted · <span class="num">${h(t.shadow)}</span> in shadow`,title:`${t.injections} notes, ${t.injected_tokens} tokens injected`})}
        ${d({label:`monitor, 24 h`,value:m?h(m.acts):`—`,sub:m?e`acts · <span class="num">${h(m.shadow)}</span> in shadow · <span class="num" data-gave-up>${h(m.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!m,warn:(m?.gave_up??0)>0,title:m?`${h(m.hidden_tokens)} hidden tokens`:void 0})}
        ${d({label:`classifier p50 / p95${!a?.calls&&o?.calls?` (Qwen)`:``}`,value:s?.p50_ms==null?`—`:`${Math.round(s.p50_ms)} / ${Math.round(s.p95_ms??0)}`,unit:s?.p50_ms==null?void 0:`ms`,sub:e`decider ${h(a?.calls??0)} calls${o?e` · Qwen ${h(o.calls)} (${g(o.p50_ms)})`:r} · fail-open ${_(t.fail_open_rate,1)}`,warn:(t.fail_open_rate??0)>.1})}
        ${d({label:`roles online`,value:`${p} / ${f.length}`,sub:e`${f.map(([t,n])=>e`<span title=${n.error??``} class=${n.up?``:n.mode===`off`?`muted`:`err-ink`} data-role=${t}>${n.up?`●`:`○`} ${t} </span>`)}`,warn:f.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${n(`Decisions an hour, last 24 h`,t.decisions?e`<qse-ribbon .series=${le(t,this.ids)} .labels=${t.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:l(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${n(`Task classes, 24 h`,e`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${t.classes.map(t=>e`<tr><td>${O(t.class_id,this.ids)}</td><td class="num num-col">${h(t.decisions)}</td><td class="num num-col">${t.labelled?`${t.agree} / ${t.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(t.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(t.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${n(e`Live decisions ${this.paused?e`<span class="badge">paused</span>`:e`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?e`<div class="feed" data-feed>${this.feed.map(e=>st(e,this.ids,e=>this.open=e,me(e)))}</div>`:e`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:e`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${c(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${c(`external`)}</a>`})}
        ${n(`Profiles by client`,e`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(t=>t.clients.map(n=>e`<button class="pick" @click=${()=>x({screen:`profiles`,edit:t.name})}><span class="muted">${n}</span> → ${t.name} <span class="badge ${t.mode===`on`?`badge-on`:t.mode===`shadow`?`badge-shadow`:``}">${t.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(t=>e`<div><dt>${t.name}</dt><dd><span class="badge ${t.mode===`on`?`badge-on`:t.mode===`shadow`?`badge-shadow`:``}">${t.mode}</span> <span class="muted small">v${t.version}${t.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${m?this.renderMonitor(m):r}
      ${this.open?e`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:r}
      ${this.openEvent?e`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:r}
    `}};s([p({attribute:!1})],N.prototype,`ids`,void 0),s([p({attribute:!1})],N.prototype,`tiles`,void 0),s([p({attribute:!1})],N.prototype,`active`,void 0),s([t()],N.prototype,`stats`,void 0),s([t()],N.prototype,`feed`,void 0),s([t()],N.prototype,`error`,void 0),s([t()],N.prototype,`open`,void 0),s([t()],N.prototype,`openEvent`,void 0),s([t()],N.prototype,`paused`,void 0),N=s([m(`qse-cond-overview`)],N);var ut=[`off`,`low`,`medium`,`high`,`xhigh`],dt=[`small`,`medium`,`large`],P=class extends o{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new D(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),E.doc(`packages`).then(e=>this.packages=Le(e.doc)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(S(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await E.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=a(e).message}finally{this.testing=!1}}}render(){let t=this.ed;if(t.loading&&!t.draft)return u(8);if(t.loadError)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${t.loadError}</div></div>`;let n=t.draft,r=n.classes.map(e=>e.id);return e`
      ${this.renderTest(r)}
      <details class="sec" open data-sec="classes">
        <summary>Task classes <span class="sec-sub">${n.classes.length} classes; a classifier sees only the descriptions</span></summary>
        <div class="sec-body">
          ${n.classes.map((e,t)=>this.renderClass(e,t,n.classes.length,r))}
          <button class="btn" data-add-class @click=${()=>this.addClass()}>${c(`plus`)} Add a class</button>
        </div>
      </details>
      <details class="sec" data-sec="classifier">
        <summary>Classifier stages <span class="sec-sub">asked in order until one answers at its threshold</span></summary>
        <div class="sec-body">
          ${n.classifier.stages.map((t,r)=>e`<div class="row-card" data-stage=${t.kind}>
              <div class="row-card-head">
                <strong>${r+1}. ${t.kind===`decider`?`decider (role decider: any model behind the Jev question shape)`:`Qwen, thinking off (role classifier)`}</strong>
                <span class="toolbar" style="margin:0">
                  <label class="switch"><input type="checkbox" .checked=${t.enabled} @change=${e=>this.upd([`classifier`,`stages`,r,`enabled`],e.target.checked)} /> enabled</label>
                  <button class="btn btn-ghost btn-sm" ?disabled=${r===0} @click=${()=>this.moveStage(r,-1)} aria-label="earlier">↑</button>
                  <button class="btn btn-ghost btn-sm" ?disabled=${r===n.classifier.stages.length-1} @click=${()=>this.moveStage(r,1)} aria-label="later">↓</button>
                </span>
              </div>
              <div class="form-grid">
                ${this.num(`accept at (probability)`,t.accept,e=>this.upd([`classifier`,`stages`,r,`accept`],e),{step:.01,min:0,max:1,hint:`below it, the next stage is asked`})}
                ${this.num(`timeout (ms)`,t.timeout_ms,e=>this.upd([`classifier`,`stages`,r,`timeout_ms`],Math.round(e)),{step:50,min:1,hint:`a slower answer counts as no answer`})}
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
                ${[`keep`,`classify`,...r].map(t=>e`<option value=${t} ?selected=${t===n.structural.compaction}>${t===`keep`?`keep the session class`:t===`classify`?`ask a classifier`:`class ${t}`}</option>`)}
              </select><span class="hint">the turn itself; the session keeps its class after it</span></label>
            <label class="field"><span>a turn that only carries tool results</span>
              <select @change=${e=>this.upd([`structural`,`tool_turn`],e.target.value)}>
                <option value="keep" ?selected=${n.structural.tool_turn===`keep`}>keep the session class (no model call)</option>
                <option value="classify" ?selected=${n.structural.tool_turn===`classify`}>ask a classifier every turn</option>
              </select></label>
            <label class="field" data-tool-turn-effort><span>effort of a tool-result turn</span>
              <select aria-label="effort of a tool-result turn" @change=${e=>{let t=e.target.value;this.upd([`structural`,`tool_turn_effort`],t===`inherit`?void 0:t)}}>
                ${Ze.map(t=>e`<option value=${t} ?selected=${t===(n.structural.tool_turn_effort??`inherit`)}>${t===`inherit`?`the class's (inherit)`:Oe(t,n,null)}</option>`)}
              </select><span class="hint">a turn whose newest message is a tool result (a screenshot a tool returned too) thinks at this level, its switch and cap; the first turn and every new user message keep the class's, the note stays the class's. A profile can set its own (Profiles › Thinking); a client's own budget wins</span></label>
            <label class="field" data-side-requests><span>a client's title or summary request</span>
              <select aria-label="a client's title or summary request" @change=${e=>this.upd([`structural`,`side_requests`],e.target.value)}>
                <option value="on" ?selected=${(n.structural.side_requests??`on`)===`on`}>a side request: no class, thinking off, short, first in the queue</option>
                <option value="off" ?selected=${n.structural.side_requests===`off`}>a turn, as any request</option>
              </select><span class="hint">known by its shape: no tools and the client's own fixed instruction (${(n.structural.side_shapes??$e).map(e=>`${e.client??`a client`}'s ${e.kind}, at most ${e.max_tokens} tokens`).join(`; `)}). It asks no classifier, gets no note and leaves the session's class alone; thinking only when the client asks for a number</span></label>
          </div>
        </div>
      </details>
      <details class="sec" data-sec="effort">
        <summary>Effort levels <span class="sec-sub">a budget and a sentence at the prompt's end; never the template's reasoning_effort</span></summary>
        <div class="sec-body">
          <p class="muted small">The template renders <code>reasoning_effort</code> as the first system tokens, so changing it per turn would cost a cold prefill. A level here sets the reasoning budget (0 = no cap below the backstop), thinking on or off (off uses <code>enable_thinking: false</code> only when the server's effort is medium, else a budget of 1), and a sentence in the tail note.</p>
          <p class="muted small">The budget is a real cap (SRV-159): when the reasoning reaches it, the server closes it with the session monitor's hidden close, so the client sees only <code>&lt;/think&gt;</code> and the answer. Defaults: low 1,024, medium 8,192, high 16,384, xhigh no cap. A profile can override a level (Profiles › Thinking); a client's own budget always wins.</p>
          ${ut.map(t=>e`<div class="row-card" data-effort=${t}>
              <div class="row-card-head"><strong class="mono">${t}</strong>
                <label class="switch"><input type="checkbox" .checked=${n.effort.levels[t].thinking} @change=${e=>this.upd([`effort`,`levels`,t,`thinking`],e.target.checked)} /> thinking</label></div>
              <div class="form-grid">${this.num(`budget (tokens, 0 = no cap)`,n.effort.levels[t].budget,e=>this.upd([`effort`,`levels`,t,`budget`],Math.round(e)),{step:256,min:0})}</div>
              <label class="field field-block"><span>sentence in the note</span><textarea rows="2" .value=${n.effort.levels[t].note} @input=${e=>this.upd([`effort`,`levels`,t,`note`],e.target.value)}></textarea></label>
            </div>`)}
          <label class="switch"><input type="checkbox" .checked=${n.effort.raise_on_fail} @change=${e=>this.upd([`effort`,`raise_on_fail`],e.target.checked)} /> a failing checker (RESULT: FAIL) raises the effort one level for that turn</label>
          ${this.renderSize(n.effort.size??Ee)}
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
      ${A(t)}
      ${M(t,`the router`)}
    `}renderTest(t){let n=this.result;return e`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?e`<span class="err-ink small" role="alert">${this.testError}</span>`:r}
      </div>
      ${n?n.active?e`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${O(n.class_id??null,t)} ${it(n.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${ct(n.source??null)}${n.kept?e` <span class="badge badge-warn">${n.kept}</span>`:r} <span class="muted num">${g(n.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${n.effort??`unchanged`} · thinking ${n.thinking??`on`}${n.budget==null?``:` · budget ${n.budget}`}</dd></div>
                  ${n.extra?.size?e`<div data-test-size><dt>task size</dt><dd>${te(n.extra.size)}</dd></div>`:r}
                  <div><dt>profile</dt><dd>${n.profile} (${n.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(n.stages??[]).map(t=>e`<div class="stage ${t.error?`is-error`:t.below?``:`is-accepted`}"><strong>${Be(t.stage)}</strong><span>${t.error??e`${t.label} ${it(t.confidence,t.accept??.6)}`}</span><span class="num muted">${g(t.ms)}</span></div>`)}</div>
              </div>
              <div>${n.note?e`<p class="muted small">The model would see this at the end of the last turn (${(n.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${n.note}</pre>`:e`<p class="muted small">No note: ${n.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:e`<p class="muted small" style="margin-top:12px">Profile ${n.profile} is raw (${n.mode}): the conductor does nothing for it.</p>`:r}
    </section>`}renderClass(t,n,i,a){let o=t.policy,s=!this.ed.base?.doc.classes.some(e=>e.id===t.id);return e`<details class="row-card" data-class-row=${t.id} ?open=${s}>
      <summary>
        <span>${O(t.id,a,t.label)} <span class="muted small mono">${o.effort_by_size?dt.map(e=>_e(o,e)).join(`/`):o.effort??`medium`}${(o.packages??[]).length?` · ${(o.packages??[]).join(`, `)}`:``}</span></span>
        <span class="toolbar" style="margin:0" @click=${e=>e.preventDefault()}>
          <button class="btn btn-ghost btn-sm" ?disabled=${n===0} @click=${()=>this.moveClass(n,-1)} aria-label="up">↑</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${n===i-1} @click=${()=>this.moveClass(n,1)} aria-label="down">↓</button>
          <button class="btn btn-ghost btn-sm" ?disabled=${i<=1} @click=${()=>this.removeClass(n)} aria-label=${`remove ${t.id}`}>${c(`trash`)}</button>
        </span>
      </summary>
      <div class="form-grid">
        <label class="field"><span>id</span><input .value=${t.id} ?readonly=${!s} @input=${e=>this.upd([`classes`,n,`id`],e.target.value)} data-class-id /><span class="hint">${s?`lowercase, digits, _ and -`:`fixed once saved; the journal keys on it`}</span></label>
        <label class="field"><span>label</span><input .value=${t.label} @input=${e=>this.upd([`classes`,n,`label`],e.target.value)} /></label>
      </div>
      <label class="field field-block"><span>description (what a classifier reads)</span><textarea rows="2" .value=${t.description} @input=${e=>this.upd([`classes`,n,`description`],e.target.value)} data-class-desc></textarea></label>
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${j(ut,o.effort??`medium`,e=>this.upd([`classes`,n,`policy`,`effort`],e),{small:!0,label:`effort of ${t.id}`})}</div>
      <div class="toolbar size-map" style="margin:0" data-size-map=${t.id}><span class="muted small">by task size (the effort above is the ceiling)</span>
        ${dt.map(i=>e`<label class="field"><span>${i}</span><select aria-label=${`${t.id} effort when ${i}`} @change=${e=>this.setSizeLevel(n,o.effort_by_size,i,e.target.value)}>
            <option value="" ?selected=${!o.effort_by_size?.[i]}>the class's</option>
            ${ut.map(t=>e`<option value=${t} ?selected=${o.effort_by_size?.[i]===t}>${t}</option>`)}
          </select>${o.effort_by_size?.[i]&&_e(o,i)!==o.effort_by_size[i]?e`<span class="hint">cut to ${_e(o,i)}</span>`:r}</label>`)}
      </div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(t=>{let r=(o.packages??[]).includes(t);return e`<button class="pick ${r?`is-on`:``}" aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`classes`,n,`policy`,`packages`],r?(o.packages??[]).filter(e=>e!==t):[...o.packages??[],t])}>${t}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(o.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,n,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}renderSize(t){let n=(e,n)=>this.upd([`effort`,`size`],S(t,e,n));return e`<div class="row-card" data-size-config>
      <div class="row-card-head"><strong>Task size</strong>
        <label class="switch"><input type="checkbox" .checked=${t.reset_on_compaction} @change=${e=>n([`reset_on_compaction`],e.target.checked)} /> a compaction starts the size over</label></div>
      <p class="muted small">Each signal is small below its first number, medium up to its second, large above. The task's size is the largest any counting signal reached, and it never falls within a task. A class's levels per size (in its row above) pick the effort under the class's effort; whether that acts is each profile's <em>effort by task size</em> (off, shadow, on).</p>
      ${Qe.map(r=>{let i=t.signals[r.id]??Ee.signals[r.id];return e`<div class="size-signal" data-size-signal=${r.id}>
          <label class="switch"><input type="checkbox" .checked=${i.enabled} @change=${e=>n([`signals`,r.id,`enabled`],e.target.checked)} /> ${r.label}</label>
          <div class="form-grid">
            ${this.num(`small below (${r.unit})`,i.small,e=>n([`signals`,r.id],{...i,small:e}),{min:0,step:r.id===`session_tokens`?1e3:1})}
            ${this.num(`medium up to (${r.unit})`,i.medium,e=>n([`signals`,r.id],{...i,medium:e}),{min:0,step:r.id===`session_tokens`?1e3:1,hint:r.hint})}
          </div>
        </div>`})}
    </div>`}setSizeLevel(e,t,n,r){let i={...t??{}};r?i[n]=r:delete i[n],this.upd([`classes`,e,`policy`,`effort_by_size`],Object.keys(i).length?i:void 0)}renderContextStage(t){let n=Ie(t),r=(e,r)=>this.upd([`classifier`,`context_stage`],{...n,...t.classifier.context_stage??{},[e]:r});return e`<div class="row-card" data-context-stage>
      <div class="row-card-head">
        <strong>Qwen in the conversation (in-context stage)</strong>
        ${j([`off`,`on`],n.mode,e=>r(`mode`,e),{small:!0,label:`in-context stage`})}
      </div>
      <p class="muted small">A follow-up or a long prompt is classified by the engine inside the conversation it already holds, right before the turn: its prompt up to the last message plus a hidden question, one letter per class, thinking off. The turn resumes from the anchors that prefill leaves. A short first message keeps the stages above; a request with images too.</p>
      <div class="form-grid">
        ${this.num(`long prompt from (tokens)`,n.min_tokens,e=>r(`min_tokens`,Math.round(e)),{step:1024,min:0,max:262144,hint:`a prompt at least this long goes to the stage`})}
        <label class="switch" data-context-followups><input type="checkbox" .checked=${n.followups} @change=${e=>r(`followups`,e.target.checked)} /> every follow-up (a user message after an earlier answer), whatever its length</label>
        ${this.num(`accept at (probability)`,n.accept,e=>r(`accept`,e),{step:.01,min:0,max:1,hint:`below it, the session's class`})}
      </div>
    </div>`}num(t,n,i,a={}){return e`<label class="field"><span>${t}</span><input type="number" .value=${String(n)} step=${a.step??1} min=${a.min??``} max=${a.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&i(t)}} />${a.hint?e`<span class="hint">${a.hint}</span>`:r}</label>`}text(t,n,r){return e`<label class="field"><span>${t}</span><input .value=${n} @input=${e=>r(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(S(n,[`classifier`,`stages`],r))}};s([p({attribute:!1})],P.prototype,`ids`,void 0),s([p({attribute:!1})],P.prototype,`meta`,void 0),s([p({attribute:!1})],P.prototype,`active`,void 0),s([p({attribute:!1})],P.prototype,`onSaved`,void 0),s([t()],P.prototype,`packages`,void 0),s([t()],P.prototype,`testText`,void 0),s([t()],P.prototype,`testProfile`,void 0),s([t()],P.prototype,`testTools`,void 0),s([t()],P.prototype,`testing`,void 0),s([t()],P.prototype,`result`,void 0),s([t()],P.prototype,`testError`,void 0),P=s([m(`qse-cond-routing`)],P);var ft=[`off`,`shadow`,`act`],pt=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,mt=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,F=class extends o{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new D(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),E.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(S(this.ed.draft,e,t))}render(){let t=this.ed,n=this.config?.server.monitor_mode;return e`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${t.loading&&!t.draft?u(8):t.loadError?e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${t.loadError}</div></div>`:e`
            ${n===`raw`?e`<div class="guard-line" data-monitor-off>${c(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:r}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(t.draft)}
            ${this.renderRest(t.draft)}
            ${A(t)}
            ${M(t,`the monitor`)}`}`}renderTasks(t){let n=ce(t),r=n.filter(e=>t.tasks[e]&&/^C\d/.test(e)),i=n.filter(e=>ve(e)),a=n.filter(e=>!r.includes(e)&&!i.includes(e)),o=i.map(e=>t.tasks[e]).find(e=>e.rule===`stall_watchdog`)?.notify;return e`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${r.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${r.map(e=>this.renderTask(e,t.tasks[e]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${i.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${i.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${o?.enabled?e`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:e`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, the owner's ruling of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${i.map(e=>this.renderTask(e,t.tasks[e]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${a.join(`, `)}</span></summary>
        <div class="sec-body">${a.map(e=>this.renderTask(e,t.tasks[e]))}</div>
      </details>`}renderTask(t,n){let i=We(this.meta?.schemas?.monitor,n.rule),a=this.meta?.tasks?.[t];return e`<div class="row-card" data-monitor-task=${t}>
      <div class="row-card-head">
        <strong>${t}${a?e` · ${a.label}`:r} <span class="muted small mono">${n.rule}</span></strong>
        ${j(ft,n.mode,e=>this.upd([`tasks`,t,`mode`],e),{small:!0,label:`${t} mode without a profile`})}
      </div>
      ${a?.why?e`<p class="muted small">${a.why}</p>`:r}
      ${i?.properties?e`<div class="form-stack">${Object.entries(i.properties).map(([e,r])=>this.field(e,r,n[e],[`tasks`,t,e]))}</div>`:e`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,n,[`tasks`,t])}`}
    </div>`}field(t,n,i,a){let o=Array.isArray(n.type)?n.type[0]:n.type,s=t.replace(/_/g,` `),l=n.description?e`<span class="hint">${n.description}</span>`:r;if(n.enum)return e`<label class="field"><span>${s}</span><select data-param=${t} @change=${e=>this.upd(a,e.target.value)}>${n.enum.map(t=>e`<option value=${String(t)} ?selected=${t===i}>${String(t)}</option>`)}</select>${l}</label>`;if(o===`boolean`)return e`<label class="switch" data-param=${t}><input type="checkbox" .checked=${!!i} @change=${e=>this.upd(a,e.target.checked)} /> ${s}</label>${n.description?e`<p class="hint">${n.description}</p>`:r}`;if(o===`integer`||o===`number`)return e`<label class="field"><span>${s}</span><input type="number" data-param=${t} step=${o===`integer`?`1`:`0.01`} min=${n.minimum??r} max=${n.maximum??r} .value=${i==null?``:String(i)} @input=${e=>{let t=e.target.value;this.upd(a,t===``?null:o===`integer`?Math.round(Number(t)):Number(t))}} />${l}</label>`;if(o===`string`){let n=String(i??``);return pt.test(t)||n.length>60?e`<label class="field field-block"><span>${s}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(n.length/90)))} class=${mt.test(t)?`mono`:``} data-param=${t} .value=${n} @input=${e=>this.upd(a,e.target.value)}></textarea>${l}</label>`:e`<label class="field"><span>${s}</span><input data-param=${t} .value=${n} @input=${e=>this.upd(a,e.target.value)} />${l}</label>`}if(o===`array`&&n.items&&(Array.isArray(n.items.type)?n.items.type[0]:n.items.type)===`string`){let r=!Array.isArray(i)&&Array.isArray(n.default),o=Array.isArray(i)?i:r?n.default:[];return e`<label class="field field-block"><span>${s} <span class="muted small">(one a line${r?`; the default, not saved yet`:``})</span></span><textarea rows=${Math.min(8,Math.max(2,o.length+1))} data-param=${t} .value=${o.join(`
`)} @input=${e=>this.upd(a,ne(e.target.value))}></textarea>${l}</label>`}if(o===`array`&&n.items?.properties){let r=Array.isArray(i)?i:[],o=n.items;return e`<fieldset class="fieldset" data-param=${t}><legend>${s} <span class="muted small">${r.length}</span></legend>${l}
        ${r.map((t,n)=>e`<div class="row-card row-card-sub" data-item=${String(t.id??n)}>
          <div class="row-card-head"><strong class="mono small">${String(t.id??`#${n+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(a,r.filter((e,t)=>t!==n))} aria-label=${`remove ${String(t.id??n+1)}`}>${c(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(o.properties).map(([e,r])=>this.field(e,r,t[e],[...a,n,e]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${t} @click=${()=>this.upd(a,[...r,ht(o)])}>${c(`plus`)} Add</button>
      </fieldset>`}if(o===`object`&&n.properties){let r=i??{};return e`<fieldset class="fieldset" data-param=${t}><legend>${s}</legend>${l}
        <div class="form-stack">${Object.entries(n.properties).map(([t,i])=>r[t]===void 0&&!(n.required??[]).includes(t)?e`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...a,t],ht(i))}>${c(`plus`)} ${t.replace(/_/g,` `)}</button>`:this.field(t,i,r[t],[...a,t]))}</div></fieldset>`}return this.jsonField(s,i,a)}jsonField(t,n,r){return e`<label class="field field-block"><span>${t} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(n,null,2)} @change=${e=>{try{let t=JSON.parse(e.target.value);if(r.length===2&&r[0]===`tasks`){let e=this.ed.draft.tasks[r[1]];this.upd(r,{...t,rule:e.rule,mode:e.mode})}else this.upd(r,t)}catch{this.ed.errors=[`${t}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(t){return e`
      <details class="sec" data-sec="caps">
        <summary>Caps, texts and the journal <span class="sec-sub">beyond a cap an act becomes GIVE_UP; a profile sets its own caps</span></summary>
        <div class="sec-body">
          <div class="form-grid">
            ${this.field(`acts_per_request`,{type:`integer`,minimum:0},t.caps.acts_per_request,[`caps`,`acts_per_request`])}
            ${this.field(`acts_per_session_hour`,{type:`integer`,minimum:0},t.caps.acts_per_session_hour,[`caps`,`acts_per_session_hour`])}
            ${this.field(`same_action_in_a_row`,{type:`boolean`,description:`off: the same hidden action twice in a row on one turn ends in GIVE_UP`},t.caps.same_action_in_a_row,[`caps`,`same_action_in_a_row`])}
            ${this.field(`tool_passes`,{type:`boolean`,description:`one journal row for every answer whose calls passed every check (findings are always journaled, passes always counted)`},t.journal.tool_passes,[`journal`,`tool_passes`])}
          </div>
          ${Object.entries(t.texts).map(([e,t])=>this.field(e,{type:`string`,description:e===`give_up`?`the one visible sentence of a GIVE_UP`:void 0},t,[`texts`,e]))}
          <div class="form-grid">
            ${this.field(`lineage_tokens`,{type:`integer`,minimum:1,description:`prompt tokens hashed into the session key when the client sends no id`},t.session.lineage_tokens,[`session`,`lineage_tokens`])}
            ${this.field(`join_window_s`,{type:`number`,minimum:0},t.session.join_window_s,[`session`,`join_window_s`])}
            ${this.field(`max_sessions`,{type:`integer`,minimum:1},t.session.max_sessions,[`session`,`max_sessions`])}
          </div>
        </div>
      </details>
      <details class="sec" data-sec="templates">
        <summary>Chat template shapes <span class="sec-sub">how a hidden turn, a reminder and a note are written in the model's own format</span></summary>
        <div class="sec-body">${Object.entries(t.templates).map(([e,t])=>this.field(e,{type:`string`},t,[`templates`,e]))}</div>
      </details>`}};s([p({attribute:!1})],F.prototype,`meta`,void 0),s([p({attribute:!1})],F.prototype,`onSaved`,void 0),s([t()],F.prototype,`config`,void 0),F=s([m(`qse-cond-monitor`)],F);function ht(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,ht(t)])):``}var I=class extends o{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=T(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await E.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let t=a(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let t=this.sessions;return n(e`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!t?e`<p class="err-ink small" role="alert">${this.error}</p>`:t==null?u(2):t.length?e`<div class="form-stack" data-watch>${t.map(e=>this.row(e))}</div>`:e`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(t){let n=t.gates.some(e=>e.noted);return e`<div class="row-card" data-watch-session=${t.session}>
      <div class="row-card-head">
        <strong class="mono small">${t.session}</strong>
        <span class="toolbar" style="margin:0">
          ${t.watch?e`<span class="badge ${t.watch.mode===`act`?`badge-on`:`badge-shadow`}">${t.watch.task} ${t.watch.mode}</span>`:r}
          ${t.client_gone?e`<span class="badge badge-warn" data-client-gone>client gone</span>`:t.running?e`<span class="badge badge-on">running</span>`:r}
          ${t.watch?.notify?e`<span class="badge badge-warn" data-notify>push on</span>`:r}
          ${n?e`<span class="badge badge-warn">gate noted</span>`:r}</span>
      </div>
      <p class="small" style="margin:4px 0">${Ye(t)}</p>
      <p class="muted small num" style="margin:0">${t.watch?`${t.watch.profile||`—`} · ${t.watch.client||`—`} · `:``}${t.requests} requests · last ${y(new Date(t.last_seen*1e3).toISOString())}${t.note_at?` · last note ${y(new Date(t.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};s([t()],I.prototype,`sessions`,void 0),s([t()],I.prototype,`monitor`,void 0),s([t()],I.prototype,`error`,void 0),I=s([m(`qse-cond-watch`)],I);var L=class extends o{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=T(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await E.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let t=a(e);this.error=`${t.endpoint}: ${t.message}`}}render(){let t=this.events,i=[...new Set((t??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!i.includes(this.task)&&i.push(this.task),e`${n(e`Monitor journal ${this.paused?e`<span class="badge">paused</span>`:e`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!t?e`<p class="err-ink small" role="alert">${this.error}</p>`:t==null?u(4):t.length?e`<div class="feed" data-journal>${t.map(e=>ot(e,e=>this.open=e))}</div>`:e`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:e`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${i.map(t=>e`<option value=${t} ?selected=${t===this.task}>${t}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${c(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?e`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:r}`}};s([p({attribute:!1})],L.prototype,`session`,void 0),s([t()],L.prototype,`events`,void 0),s([t()],L.prototype,`monitor`,void 0),s([t()],L.prototype,`error`,void 0),s([t()],L.prototype,`task`,void 0),s([t()],L.prototype,`kind`,void 0),s([t()],L.prototype,`paused`,void 0),s([t()],L.prototype,`open`,void 0),L=s([m(`qse-cond-journal`)],L);var gt=[`off`,`low`,`medium`,`high`,`xhigh`],_t={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},vt=new Set([`S1`,`S4`]),yt=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],R=class extends o{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleContract=null,this.roleTest={},this.ed=null,this.roles=new D(this,`roles`),this.residency=new D(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),E.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),E.doc(`packages`).then(e=>this.packages=Le(e.doc)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new D(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{let e=await E.roles();this.roleHealth=e.health,this.roleContract=e.contract??null}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let t=await E.roleTest(e);this.roleTest={...this.roleTest,[e]:t.ok?`ok in ${g(t.ms)}: "${t.answer?.label}" at ${t.answer?.confidence.toFixed(2)}`:`failed: ${t.error}`}}catch(t){this.roleTest={...this.roleTest,[e]:a(t).message}}this.loadHealth()}render(){return e`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(t=>e`<button class="tile ${t.name===this.selected?`is-on`:``}" data-profile=${t.name} @click=${()=>x({edit:t.name})}>
            <div class="tile-head"><span class="tile-name">${t.name}</span>${t.locked?e`<span class="badge badge-lock">${c(`warn`)} locked</span>`:e`<span class="badge ${t.mode===`on`?`badge-on`:t.mode===`shadow`?`badge-shadow`:``}">${t.mode}</span>`}</div>
            <div class="tile-meta">${t.locked??t.description}</div>
            <div class="tile-meta">clients: ${t.clients.length?t.clients.join(`, `):`— (header or default)`} · default class ${t.default_class}</div>
            <div class="tile-meta num">decisions on ${t.counts.decisions.on??0} · shadow ${t.counts.decisions.shadow??0} · tasks act ${t.counts.tasks.act??0} · shadow ${t.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${t.version} · ${t.last_change.note??``}${t.last_change.created_at?` · ${y(t.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):e`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(S(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=S(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=S(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderUi(t,n){let r=t.ui??{states:`off`},i=r.kinds??{},a=n||r.states!==`on`,o=Ge(r),s=Se(r),c=(t,n,i,o,c)=>e`<label class="field"><span>${n}</span><input type="number" inputmode="numeric" min=${i} max=${o} step=${c} ?disabled=${a||s.mode!==`on`} data-ui-progress=${t} .value=${String(s[t])}
        @input=${e=>{let n=Math.round(Number(e.target.value));Number.isFinite(n)&&n>=i&&n<=o&&this.upd([`ui`],{...r,progress:{...r.progress??{},mode:s.mode,[t]:n}})}} /></label>`;return e`<details class="sec" data-sec="ui">
      <summary>What the user sees <span class="sec-sub">TandemLLM's own steps as lines the client shows</span> <span class="badge ${r.states===`on`?`badge-on`:``}" data-ui-badge>${r.states}</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0"><span class="muted small">state lines</span>${j([`off`,`on`],r.states,e=>this.upd([`ui`],{...r,states:e}),{disabled:n,small:!0,label:`state lines`})}</div>
        <p class="muted small">Each step goes out at the moment it happens as one line, <code>› deciding: code_edit, high effort</code>. The server takes them out of the reasoning and the content a client sends back, so Qwen never sees them; they are not tokens and not in the usage. Only in mode on.</p>
        <div class="toolbar" style="margin:0" data-ui-channel><span class="muted small">where</span>${j([`reasoning`,`content`,`inline`],o,e=>this.upd([`ui`],{...r,states_channel:e}),{disabled:a,small:!0,label:`where the lines go`})}</div>
        <p class="muted small">${o===`inline`?e`Each line opens a thinking block of its own and is its title, in time order: opencode shows <code>Thought: › thinking 2 min, 4,096 tokens</code> between Qwen's thinking blocks, and the answer after them. Needs <code>reasoning_content</code>; with <code>&lt;think&gt;</code> in the content (Hermes) the lines go as in the content.`:o===`content`?e`In the content, outside the thinking: before the reasoning, between its parts, after it. opencode shows them in its text part; with <code>&lt;think&gt;</code> in the content (Hermes) a line said inside the block goes out right after <code>&lt;/think&gt;</code>.`:e`In the reasoning opencode and Hermes show (a <code>reasoning_content</code> delta, or inside <code>&lt;think&gt;</code>). A profile saved before this choice existed keeps them here.`}</p>
        <div class="pkg-picks" data-ui-kinds>${re.map(({id:t,example:n})=>{let o=i[t]!==!1;return e`<button class="pick ${o?`is-on`:``}" data-ui-kind=${t} title=${n} aria-pressed=${o?`true`:`false`} ?disabled=${a} @click=${()=>this.upd([`ui`],{...r,kinds:{...i,[t]:!o}})}>${t}</button>`})}</div>
        <div class="toolbar" style="margin:0" data-ui-progress-mode><span class="muted small">progress</span>${j([`off`,`on`],s.mode,e=>this.upd([`ui`],{...r,progress:{...r.progress??{},mode:e}}),{disabled:a,small:!0,label:`progress lines`})}</div>
        <p class="muted small">The numbers of a wait, templated: <code>waiting: opencode's title request runs first, position 1</code> while the request waits, <code>reading 35,189 tokens at ~750 tok/s, about 47 s left</code> during a long prefill, <code>thinking 4 min, 9,800 tokens</code> during a long thinking. Streams only; a request that waits opens its stream at its first waiting line.</p>
        <div class="form-grid">
          ${c(`prefill_every_s`,`a prefill line every (s)`,2,600,1)}
          ${c(`thinking_every_s`,`a thinking line every (s)`,10,3600,10)}
          ${c(`thinking_every_tokens`,`or every (tokens)`,256,262144,256)}
        </div>
        <div class="toolbar" style="margin:0" data-ui-narrator><span class="muted small">narrator</span>${j([`off`,`on`,`thoughts`,`live`],r.narrator??`off`,e=>this.upd([`ui`],{...r,narrator:e}),{disabled:a,small:!0,label:`narrator`})}</div>
        <p class="muted small">A small model inside the engine (the <code>narrator</code> role, Qwen3.5) writes in plain words. <b>live</b>: a running line of what Qwen is doing right now, every ~25 s of thinking (“thinking 3 min: how the enemies patrol …”), what a thought decided, what a long write is for and how far it is, a plan in the answer; the other lines in plain words and the user’s language; the facts (times, files, commands) come from the engine, never from the model. <b>thoughts</b>: after each stretch of thinking (and at the long-thinking marks) one line with what Qwen thought about and decided, e.g. “Thought 1m 43s: planned the level layout …”, read from the thought alone; the other lines stay templates. <b>on</b>: it writes every line instead of the template. Qwen waits while it writes, at most 3 % of its time (the rest at the turn’s end); a line it cannot write in time is the template. It loads on its first line; the Roles section below holds its budget and mode.</p>
      </div>
    </details>`}renderToolTurn(t,n){let r=t.thinking.tool_turn_effort,i=Ke(this.router,null).level,a=Ke(this.router,t);return e`<label class="field ${r?``:`is-inherited`}" data-tool-turn-effort><span>effort of a tool-result turn</span>
      <select aria-label="effort of a tool-result turn" ?disabled=${n} @change=${e=>{let t=e.target.value;if(!this.ed?.draft)return;let n={...this.ed.draft.thinking,tool_turn_effort:t};t||delete n.tool_turn_effort,this.ed.set(S(this.ed.draft,[`thinking`],n))}}>
        <option value="" ?selected=${!r}>Routing's: ${i===`inherit`?`the class's`:i}</option>
        ${Ze.map(t=>e`<option value=${t} ?selected=${r===t}>${t===`inherit`?`the class's (inherit)`:t}</option>`)}
      </select>
      <span class="hint">${Oe(a.level,this.router,t)}${a.source===`profile`?` (this profile)`:a.source===`router`?` (Routing)`:``}: a turn whose newest message is a tool result thinks at this level, its switch and cap; the first turn and every new user message keep the class's</span></label>`}renderBudgets(t,n){return e`<div data-budgets>
      <div class="muted small">Thinking budget per level <span class="muted">(reasoning tokens; empty keeps Routing's level, 0 is no cap below the backstop)</span></div>
      <div class="form-grid">
        ${je.map(r=>{let i=t.thinking.budgets?.[r],a=tt(r,this.router,null).cap,o=tt(r,this.router,t);return e`<label class="field ${i==null?`is-inherited`:``}" data-budget=${r}><span class="mono">${r}</span>
            <input type="number" inputmode="numeric" min="0" max=${He} step="1024" ?disabled=${n} placeholder=${a==null?``:String(a)} .value=${i==null?``:String(i)}
              @change=${e=>{let t=e.target.value.trim();this.ed?.draft&&this.ed.set(S(this.ed.draft,[`thinking`],Me(this.ed.draft.thinking,r,t===``?null:Number(t))))}} />
            <span class="hint">${ie(o.cap)}${o.source===`router`?` (Routing)`:o.source===`profile`?` (this profile)`:``}</span></label>`})}
      </div>
    </div>`}renderEditor(t){if(t.loading&&!t.draft)return u(8);if(t.loadError)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${t.loadError}</div></div>`;let n=t.draft,i=!!t.locked,a=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],o=this.router?.classes??[];return e`<div style="margin-top:16px" data-editor=${t.name}>
      <div class="toolbar"><h2 class="panel-title">${t.name} <span class="muted small num">v${t.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(t.name??``)}`} data-try>${c(`playground`)} Try in Playground</a></div>
      ${i?e`<div class="note" data-lock-reason>${c(`warn`)} ${t.locked}. Every control below is read-only.</div>`:r}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${n.description} ?disabled=${i} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${j([`raw`,`shadow`,`on`],n.mode,e=>this.upd([`mode`],e),{disabled:i,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${a.map(t=>{let r=n.clients.includes(t);return e`<button class="pick ${r?`is-on`:``}" ?disabled=${i} aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`clients`],r?n.clients.filter(e=>e!==t):[...n.clients,t])}>${t}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${i} @change=${e=>this.upd([`default_class`],e.target.value)}>${o.map(t=>e`<option value=${t.id} ?selected=${t.id===n.default_class}>${t.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(_t).map(t=>e`<div class="toolbar" style="margin:0" data-decision-mode=${t}>${j([`off`,`shadow`,`on`],n.decisions[t],e=>this.upd([`decisions`,t],e),{disabled:i,small:!0,label:t})}<span class="small">${_t[t]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${o.map(t=>{let r=n.routing[t.id]??{},a=Object.keys(r).length>0;return e`<div class="row-card" data-override=${t.id}>
              <div class="row-card-head">${O(t.id,this.ids,t.label)}
                ${a?e`<button class="btn btn-ghost btn-sm" ?disabled=${i} @click=${()=>{let e={...n.routing};delete e[t.id],this.upd([`routing`],e)}}>use the router's</button>`:e`<span class="muted small">router: ${t.policy.effort??`medium`}${(t.policy.packages??[]).length?` · ${(t.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${j(gt,r.effort??t.policy.effort??`medium`,e=>this.upd([`routing`,t.id],{...r,effort:e}),{disabled:i,small:!0,label:`effort override of ${t.id}`,cls:r.effort?``:`is-inherited`})}</div>
              <div class="pkg-picks">${this.packages.map(n=>{let a=r.packages??t.policy.packages??[],o=a.includes(n);return e`<button class="pick ${o?`is-on`:``}" ?disabled=${i} @click=${()=>this.upd([`routing`,t.id],{...r,packages:o?a.filter(e=>e!==n):[...a,n]})}>${n}</button>`})}</div>
            </div>`})}
        </div>
      </details>
      <details class="sec" data-sec="tasks">
        <summary>Session monitor tasks <span class="sec-sub">the frame (SRV-109) runs these; act needs the role online</span></summary>
        <div class="sec-body">${Object.entries(n.tasks).map(([e,t])=>this.renderTask(e,t.mode,t.cap,t.thresholds,i))}</div>
      </details>
      <details class="sec" data-sec="thinking">
        <summary>Thinking <span class="sec-sub">the effort when no class sets one</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${j(gt,n.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:i,small:!0,label:`default effort`})}</div>
          <label class="switch"><input type="checkbox" ?disabled=${i} .checked=${n.thinking.hints} @change=${e=>this.upd([`thinking`,`hints`],e.target.checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <div class="toolbar" style="margin:0" data-by-size><span class="muted small">effort by task size</span>${j([`off`,`shadow`,`on`],n.thinking.by_size??`shadow`,e=>this.upd([`thinking`,`by_size`],e),{disabled:i,small:!0,label:`effort by task size`,cls:n.thinking.by_size?``:`is-inherited`})}</div>
          <p class="muted small">Effort by task size (SRV-157): a class with levels per size (Routing › the class) thinks at its size's level, never above the class's effort; the size only rises within a task. <strong>off</strong>: the class's effort; <strong>shadow</strong>: the class's effort, and the journal shows the size and the effort it would set; <strong>on</strong>: the size picks.</p>
          ${this.renderToolTurn(n,i)}
          ${this.renderBudgets(n,i)}
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="library">
        <summary>Package library <span class="sec-sub">a searched skill for a new message or a stuck run (R1)</span></summary>
        <div class="sec-body">
          <div class="toolbar" style="margin:0" data-library-mode><span class="muted small">library</span>${j([`off`,`shadow`,`on`],n.library??`off`,e=>this.upd([`library`],e),{disabled:i,small:!0,label:`package library`,cls:n.library?``:`is-inherited`})}</div>
          <p class="muted small">On a new user message, and on a low-quality or stuck outcome, the router searches the library (the <a href="#/conductor?screen=packages">Packages</a> screen) with the message, asks the package question among the best matches and puts the skill that fits at the prompt's tail. <strong>off</strong>: nothing; <strong>shadow</strong>: the journal shows the pick, nothing goes in; <strong>on</strong>: the skill and the card that tells the model how to search the library itself.</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113); a pack that is off turns its task off</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${yt.map(([t,r,a])=>e`<label class="switch" data-pack=${t}><input type="checkbox" ?disabled=${i} .checked=${n.toolgate[t]} @change=${e=>this.upd([`toolgate`,t],e.target.checked)} /> ${a} <span class="muted small">(${r}, ${n.toolgate[t]?n.tasks[r]?.mode??`off`:`off`})</span></label>`)}
          <label class="field"><span>Tier 1 timeout (ms)</span><input type="number" ?disabled=${i} .value=${String(n.toolgate.tier1_timeout_ms)} @input=${e=>this.upd([`toolgate`,`tier1_timeout_ms`],Math.round(Number(e.target.value)))} /></label>
        </div>
        <p class="muted small">With a pack on, its task runs in the mode set above; with one of them able to act, an answer's tool calls are held to its end and released whole. The floors, rules and reminders are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      <details class="sec" data-sec="context">
        <summary>Context <span class="sec-sub">the session tasks S1, S2, S4 (SRV-121): their numbers and the push</span></summary>
        <div class="sec-body"><div class="form-grid">
          <label class="field"><span>S1: failed runs of one gate before the blocked-stage note <span class="hint">count, default 3</span></span><input type="number" min="1" max="20" ?disabled=${i} data-context="s1_runs" .value=${String(n.context.s1_runs)} @input=${e=>this.ctx(`s1_runs`,`S1`,`fails`,Math.round(Number(e.target.value)))} /></label>
          <label class="field"><span>S4: minutes without progress before the note, and of silence before the client counts as gone <span class="hint">minutes, default 20</span></span><input type="number" min="1" max="600" ?disabled=${i} data-context="s4_minutes" .value=${String(n.context.s4_minutes)} @input=${e=>this.ctx(`s4_minutes`,`S4`,`minutes`,Math.round(Number(e.target.value)))} /></label>
          <label class="switch" data-context="notify"><input type="checkbox" ?disabled=${i} .checked=${n.context.notify} @change=${e=>this.upd([`context`,`notify`],e.target.checked)} /> NOTIFY: a push through ntfy to the phone when the client has gone <span class="badge ${n.context.notify?`badge-warn`:``}">${n.context.notify?`on`:`off`}</span></label>
        </div>
        <p class="muted small" data-notify-rule>Allowed by the owner on 2026-09-30: off by default, on in the overnight profile. Only S4 in act pushes, never in shadow, once a turn end; the token is a file on the box. S2 (re-read after a compaction) has no number here: its texts and state files are on the <a href="#/conductor?screen=monitor">Monitor</a> screen.</p></div>
      </details>
      ${this.renderUi(n,i)}
      <details class="sec" data-sec="hints">
        <summary>Speed hints <span class="badge badge-warn">not gated</span></summary>
        <div class="sec-body">
          <p class="warn-ink small">No SPD measurement is in the journal yet (SPD-73); the router's ≤ 1 ulp text rule and the never-worse gate come first.</p>
          <label class="switch"><input type="checkbox" ?disabled=${i} .checked=${n.hints.stair_width} @change=${e=>this.upd([`hints`,`stair_width`],e.target.checked)} /> StairCut width hint</label>
          <label class="switch"><input type="checkbox" ?disabled=${i} .checked=${n.hints.prefix_retention} @change=${e=>this.upd([`hints`,`prefix_retention`],e.target.checked)} /> prefix retention hint</label>
        </div>
      </details>
      ${A(t)}
      ${M(t,`profile ${t.name}`)}
    </div>`}renderTask(t,n,i,a,o){let s=this.meta?.tasks[t],l=s?.role&&s.tier===`1`,u=!l||!!this.roleHealth[s.role]?.up;return e`<div class="task-row" data-task=${t}>
      <div><div class="task-id">${t}</div><div class="muted small">${s?.family??``}</div></div>
      <div>${s?.label??t}<div class="task-why">${s?.why??``}</div>${l&&!u?e`<div class="warn-ink small">${c(`warn`)} the ${s.role} role is offline: act is refused</div>`:r}</div>
      <div class="task-mode">${j([`off`,`shadow`,`act`],n,e=>this.upd([`tasks`,t,`mode`],e),{disabled:o,small:!0,label:`${t} mode`})}</div>
      <div class="thresholds">
        ${vt.has(t)&&Object.keys(a).length?e`<p class="muted small" data-in-context=${t}>${Object.entries(a).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:r}
        ${Object.entries(vt.has(t)?{}:a).map(([n,r])=>{let i=s?.thresholds[n];return e`<label class="field ${i&&i.default!==r?`is-changed`:``}"><span>${n.replace(/_/g,` `)} <span class="hint">${i?.unit??``}${i?`, default ${i.default}`:``}</span></span>
            <input type="number" ?disabled=${o} .value=${String(r)} step="any" @input=${e=>{let r=Number(e.target.value);Number.isFinite(r)&&this.upd([`tasks`,t,`thresholds`,n],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${o} min="0" max="10" .value=${String(i)} @input=${e=>this.upd([`tasks`,t,`cap`],Math.round(Number(e.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let t=this.roles;if(!t.draft)return t.loadError?e`<p class="err-ink small">${t.loadError}</p>`:r;let n=t.draft,i=(e,r)=>t.set(S(n,e,r)),a=pe(t.base?.meta.defaults),o=this.roleContract?.engine_tokenizer_sha256??null,s=this.meta?.roles,l=s?.kinds??Ne,u=s?.windows??we,d=s?.fallbacks??[`PASS`,`stale`,`skip`,`next-role`],f=Xe.filter(e=>!s||s.writes.includes(e));return e`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${t.base?.meta.defaults?.length?e`<div class="note" data-role-defaults>${c(`warn`)} Version ${t.base.version} was stored before the role contract and loaded with ${t.base.meta.defaults.length} default${t.base.meta.defaults.length===1?``:`s`} (marked <span class="badge">default</span> below; the journal has the list). Save the roles to make them explicit.</div>`:r}
        <p class="muted small">The contract (SRV-133): when a role may use the GPU, whether a Qwen request cuts it, how long its caller waits before the fallback, what it may write, and the tokenizer it reads, which decides what it may share with Qwen. Engine tokenizer: <code class="mono" data-engine-sha>${o?`${o.slice(0,12)}…`:`not recorded`}</code></p>
        ${Object.entries(n).map(([t,n])=>{let s=this.roleHealth[t],p=new Set((a[t]??[]).map(e=>e.split(`=`)[0])),m=n=>p.has(n)?e` <span class="badge" data-default=${`${t}.${n}`}>default</span>`:r,h=ke(n.kind,n.window),_=Ue(n.kind,n.writes),v=Ce(n,o);return e`<div class="row-card" data-role-row=${t}>
            <div class="row-card-head"><strong>${t}</strong>
              <span class="toolbar" style="margin:0">${s?e`<span class="badge ${s.up?`badge-ok`:n.mode===`off`?``:`badge-warn`}">${s.up?`up ${g(s.ms)}`:n.mode===`off`?`off`:`down: ${s.error??`?`}`}</span>`:r}
                <button class="btn btn-sm" @click=${()=>this.testRole(t)} data-role-test=${t}>Test</button></span></div>
            ${this.roleTest[t]?e`<p class="small" role="status">${this.roleTest[t]}</p>`:r}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${n.endpoint} @input=${e=>i([t,`endpoint`],e.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${n.model} @input=${e=>i([t,`model`],e.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(n.timeout_ms)} @input=${e=>i([t,`timeout_ms`],Math.round(Number(e.target.value)))} /></label>
              <label class="field"><span>memory claim (GiB, its peak)${m(`memory_claim_gib`)}</span><input type="number" step="0.1" min="0" data-role-field="memory_claim_gib" .value=${String(n.memory_claim_gib)} @input=${e=>i([t,`memory_claim_gib`],Number(e.target.value))} /></label>
            </div>
            <div class="form-grid" data-role-contract=${t}>
              <label class="field"><span>kind${m(`kind`)}</span><select data-role-field="kind" @change=${e=>i([t,`kind`],e.target.value)}>${l.map(t=>e`<option value=${t} ?selected=${t===n.kind}>${t}</option>`)}</select></label>
              <label class="field"><span>window${m(`window`)}</span><select data-role-field="window" @change=${e=>i([t,`window`],e.target.value)}>${u.map(t=>e`<option value=${t} ?selected=${t===n.window} ?disabled=${t!==n.window&&!!ke(n.kind,t)} title=${ye[t]}>${t}</option>`)}</select></label>
              <label class="field"><span>latency budget (ms)${m(`latency_budget_ms`)}</span><input type="number" min="1" max="600000" data-role-field="latency_budget_ms" .value=${String(n.latency_budget_ms)} @input=${e=>i([t,`latency_budget_ms`],Math.round(Number(e.target.value)))} /></label>
              <label class="field"><span>fallback (off, slow, unsure or refused)${m(`fallback`)}</span><select data-role-field="fallback" @change=${e=>i([t,`fallback`],e.target.value)}>${d.map(t=>e`<option value=${t} ?selected=${t===n.fallback}>${t}</option>`)}</select></label>
              <label class="field"><span>writes${m(`writes`)}</span><select data-role-field="writes" @change=${e=>i([t,`writes`],e.target.value)}>${f.map(t=>e`<option value=${t} ?selected=${t===n.writes} ?disabled=${t!==n.writes&&!!Ue(n.kind,t)}>${t.replace(/_/g,` `)}</option>`)}</select></label>
              <label class="field field-wide"><span>tokenizer sha256 (the engine records its own; none: text only)${m(`tokenizer_sha256`)}</span><input class="mono" data-role-field="tokenizer_sha256" spellcheck="false" .value=${n.tokenizer_sha256} @input=${e=>i([t,`tokenizer_sha256`],e.target.value.trim().toLowerCase()||`none`)} /></label>
            </div>
            <div class="toolbar" style="margin:0">
              <label class="switch"><input type="checkbox" data-role-field="preemptible" .checked=${n.preemptible} @change=${e=>i([t,`preemptible`],e.target.checked)} /> preemptible: a Qwen request cuts it, it resumes later${m(`preemptible`)}</label>
            </div>
            <div><span class="muted small">sharing with Qwen${m(`sharing`)}</span>
              <div class="pkg-picks" data-role-sharing=${t}>${v.map(r=>{let a=n.sharing.includes(r.form);return e`<button class="pick ${a?`is-on`:``}" aria-pressed=${a?`true`:`false`} ?disabled=${!a&&!r.allowed} title=${r.why??``} data-share=${r.form} @click=${()=>i([t,`sharing`],a?n.sharing.filter(e=>e!==r.form):[...n.sharing,r.form])}>${r.label}</button>`})}</div>
              ${v.filter(e=>!e.allowed).map(t=>e`<p class="muted small" data-share-why=${t.form}>${t.label}: ${t.why}</p>`)}
            </div>
            ${h?e`<p class="warn-ink small" data-role-why="window">${c(`warn`)} ${h}</p>`:r}
            ${_?e`<p class="warn-ink small" data-role-why="writes">${c(`warn`)} ${_}</p>`:r}
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${j([`off`,`shadow`,`on`],n.mode,e=>i([t,`mode`],e),{small:!0,label:`${t} mode`})}<span class="muted small mono">${n.contract}</span></div>
          </div>`})}
        ${A(t)}
        ${M(t,`the roles`)}
      </div>
    </details>`}renderResidency(){let t=this.residency;if(!t.draft)return t.loadError?e`<p class="err-ink small">${t.loadError}</p>`:r;let n=t.draft,i=(e,r)=>t.set(S(n,e,r)),a=(t,r=`0.1`,a=!1)=>e`<input type="number" step=${r} min="0" .value=${String(n[t[0]])} @input=${e=>{let n=Number(e.target.value);i(t,a?Math.round(n):n)}} data-res=${t[0]} />`,o=[`measured`,`peak`,`declared`];return e`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${a([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${a([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${a([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${n.order.join(`, `)} @change=${e=>i([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(n.claims).map(([t,r])=>e`<div class="row-card" data-claim=${t}>
            <div class="row-card-head"><strong>${t}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[t]:e,...r}=n.claims;i([`claims`],r)}} aria-label=${`remove the claim of ${t}`}>${c(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(r.gib)} @input=${e=>i([`claims`,t,`gib`],Number(e.target.value))} data-claim-gib=${t} /></label>
              <label class="field"><span>basis</span><select @change=${e=>i([`claims`,t,`basis`],e.target.value)}>${o.map(t=>e`<option value=${t} ?selected=${t===r.basis}>${t}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${r.note} @input=${e=>i([`claims`,t,`note`],e.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in n.claims} @click=${()=>{i([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${c(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${A(t)}
        ${M(t,`the residency limits`)}
      </div>
    </details>`}};s([p({attribute:!1})],R.prototype,`ids`,void 0),s([p({attribute:!1})],R.prototype,`meta`,void 0),s([p({attribute:!1})],R.prototype,`tiles`,void 0),s([p({attribute:!1})],R.prototype,`selected`,void 0),s([p({attribute:!1})],R.prototype,`onSaved`,void 0),s([t()],R.prototype,`router`,void 0),s([t()],R.prototype,`packages`,void 0),s([t()],R.prototype,`roleHealth`,void 0),s([t()],R.prototype,`roleContract`,void 0),s([t()],R.prototype,`roleTest`,void 0),s([t()],R.prototype,`newClaim`,void 0),R=s([m(`qse-cond-profiles`)],R);var z=[`package`,`task_skill`,`test_skill`,`card`,`library`],B={package:`package`,task_skill:`task skill`,test_skill:`test skill`,card:`card`,library:`library skill`},bt=3.5,xt={library:350,library_card:200},St={q:``,category:`all`,cls:``,enabled:`all`,over:!1};function Ct(e){return Math.floor(e.length/bt)}function wt(e,t){return t?t.category:e.library?`library`:e.kind===`skill`?`task_skill`:`package`}function Tt(e,t,n,r){let i=r?.limits??xt,a=n===`library`?i.library:e===(r?.library.card??`tandem-library`)?i.library_card:null;return a!=null&&a<=t.max_tokens?{limit:a,source:`house`}:{limit:t.max_tokens,source:`max_tokens`}}function Et(e,t){return(e?.classes??[]).filter(e=>(e.policy.packages??[]).includes(t)).map(e=>e.id)}function Dt(e,t,n,r){return Object.entries(e?.packages??{}).map(([e,i])=>{let a=t?.packages[e]??null;return{name:e,pkg:i,entry:a,category:wt(i,a),classes:Et(r,e),usage:n?.[e]??null,score:null}}).sort((e,t)=>z.indexOf(e.category)-z.indexOf(t.category)||e.name.localeCompare(t.name))}function Ot(e){return e.toLowerCase().split(/\s+/).filter(Boolean)}function kt(e,t){let n=`${e.name} ${e.pkg.title} ${e.pkg.description} ${e.pkg.source}`.toLowerCase();return Ot(t).every(e=>n.includes(e))}function At(e,t){return e.entry?e.entry.over:e.pkg.source===`inline`&&Ct(V(e.pkg.body,!!e.pkg.library)[0])>Tt(e.name,e.pkg,e.category,t).limit}function jt(e,t,n,r=null){let i=new Map((n??[]).map(e=>[e.name,e.score])),a=t.q.trim(),o=e.filter(e=>t.category===`all`||e.category===t.category).filter(e=>!t.cls||e.classes.includes(t.cls)||e.category===`library`&&(e.entry?.fits??[]).includes(t.cls)).filter(e=>t.enabled===`all`||t.enabled===`on`===e.pkg.enabled).filter(e=>!t.over||At(e,r)).filter(e=>!a||kt(e,a)||e.category===`library`&&i.has(e.name)).map(e=>({...e,score:a&&e.category===`library`?i.get(e.name)??null:null}));return a?o.sort((e,t)=>z.indexOf(e.category)-z.indexOf(t.category)||(e.score!=null||t.score!=null)&&(t.score??-1)-(e.score??-1)||e.name.localeCompare(t.name)):o}function Mt(e,t){let n=e=>!!t?.packages[e]?.general;return[...e.filter(n),...e.filter(e=>!n(e))]}function Nt(e,t,n,r){let i=e.classes.findIndex(e=>e.id===t);if(i<0)return{router:e,error:`no class ${t}`};if(r?.packages[n]?.library)return{router:e,error:`${n} is a library skill: the router searches the library, a class does not list it`};let a=e.classes[i].policy.packages??[];if(a.includes(n))return{router:e,error:null};if(a.length>=12)return{router:e,error:`${t} already lists 12 packages, the most a class may`};let o=e.classes.map((e,t)=>t===i?{...e,policy:{...e.policy,packages:Mt([...a,n],r)}}:e);return{router:{...e,classes:o},error:null}}function Pt(e,t,n){return{...e,classes:e.classes.map(e=>e.id===t?{...e,policy:{...e.policy,packages:(e.policy.packages??[]).filter(e=>e!==n)}}:e)}}var Ft=/^## Reference\b/m;function V(e,t){let n=t?Ft.exec(e):null;return n?[e.slice(0,n.index).trim(),e.slice(n.index).trim()]:[e.trim(),``]}function It(e){let t=1;for(;e.packages[`package-${t}`];)t++;return`package-${t}`}function Lt(e){return{title:`Package ${e}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0}}function Rt(e,t,n){return{...e,source:`inline`,body:n?`${t}\n\n${n}`:t}}function zt(e,t){return t?e.filter(e=>e.profile===t):e}function Bt(e){return e===`on`?`the router puts the skill it picks into the prompt from the next request, and the library card on the coding classes with tools`:e===`shadow`?`the router searches and picks, and the journal shows the pick; nothing goes into the prompt`:`the router does not search the library`}var H=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.catalog=null,this.usage=null,this.onChanged=()=>void 0,this.profile=``,this.busy=null,this.msg=null,this.error=null}async setMode(t,n){this.busy=t,[this.msg,this.error]=[null,null];try{let r=await E.doc(`profiles`,t),i={...r.doc,library:n},a=await E.check(`profiles`,t,i);if(a.errors.length){this.error=`${t}: ${a.errors.join(`; `)}`;return}if(!await k(`Library ${n} for ${t}?`,[`With ${n}, ${Bt(n)}.`,e`The server checked the change (live v${r.version}):${at(a.changes)}`,`Saving makes a new version of the profile with the note “library to ${n}”.`],`Save the new version`))return;let o=await E.put(`profiles`,t,i,`library to ${n}`,r.version);this.msg=`${t}: library ${n}, version ${o.version}`,this.onChanged()}catch(e){this.error=a(e).message}finally{this.busy=null}}render(){let t=this.catalog;if(!t||!this.usage)return u(5);let n=this.usage.library,i=Object.values(t.packages).filter(e=>e.category===`library`).length,a=e=>n.rows.filter(t=>t.profile===e&&t.chosen.length).length,o=Object.keys(t.library.profiles).sort();return e`
      <section class="panel" data-lib-profiles>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Retrieval per profile <span class="muted">· ${i} library skills</span></h2>
          <p class="panel-sub">On a new user message, and on a low-quality or stuck outcome, the router searches the library with the message, asks the package question among the best matches and, with <strong>on</strong>, puts the skill that fits at the prompt's tail. <strong>shadow</strong> only journals the pick. Judge the picks below before a profile goes on.</p></div></header>
        ${this.error?e`<p class="err-ink small" role="alert" data-lib-error>${this.error}</p>`:r}
        ${this.msg?e`<p class="status small" role="status" data-lib-saved>${c(`check`)} ${this.msg}</p>`:r}
        <div class="lib-profiles">
          ${o.map(n=>{let i=this.tiles.find(e=>e.name===n),o=t.library.profiles[n];return e`<div class="lib-profile" data-lib-profile=${n}>
              <span class="lib-profile-name"><strong>${n}</strong>${i?.locked?e` <span class="badge badge-lock">locked</span>`:r}<span class="muted small">${i?.clients.join(`, `)||`no client`}${i?` · profile ${i.mode}`:``}</span></span>
              ${j([`off`,`shadow`,`on`],o,e=>void this.setMode(n,e),{small:!0,label:`library of ${n}`,disabled:!!i?.locked||this.busy===n})}
              <span class="muted small num">${a(n)} pick${a(n)===1?``:`s`} in the rows below</span>
            </div>`})}
        </div>
      </section>
      <section class="panel" data-lib-shadow>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">What the harness picked</h2>
          <p class="panel-sub">The router's retrieval in the journal (extra.library), newest first: the candidates the search found, the skill the package question picked, and whether it went in (on) or would have (shadow). A pick opens the skill.</p></div>
          <div class="panel-tools"><label class="field"><span>profile</span><select data-lib-filter @change=${e=>this.profile=e.target.value}>
            <option value="" ?selected=${!this.profile}>all</option>${o.map(t=>e`<option value=${t} ?selected=${this.profile===t}>${t}</option>`)}
          </select></label></div></header>
        <div class="stats" data-lib-tiles>
          ${d({label:`messages searched`,value:h(n.searched.day),unit:`in 24 h`,sub:`${h(n.searched.week)} in 7 days`})}
          ${d({label:`with a pick`,value:h(n.picked.day),unit:`in 24 h`,sub:`${h(n.picked.week)} in 7 days`})}
          ${d({label:`pick rate`,value:n.searched.week?`${Math.round(100*n.picked.week/n.searched.week)}`:`—`,unit:n.searched.week?`%`:``,sub:`7 days`})}
          ${d({label:`most picked`,value:n.top[0]?.name??`—`,sub:n.top.length?n.top.slice(0,3).map(e=>`${e.name} ${e.picks}`).join(` · `):`no pick yet`})}
        </div>
        ${this.rows(zt(n.rows,this.profile))}
      </section>`}rows(t){if(!t.length)return l(`No retrieval in the journal`,`A profile with the library in shadow or on journals each search here; with every profile off nothing is searched.`);let n=e=>x({ptab:null,pkg:e,pview:null});return e`<div class="lib-rows" data-lib-rows>${t.map(t=>e`<div class="lib-row" data-lib-row=${t.id}>
        <div class="lib-row-head">
          <span class="num small">${y(new Date(t.ts_ms).toISOString())}</span>
          <span class="badge ${t.mode===`on`?`badge-on`:`badge-shadow`}">${t.mode??`—`}</span>
          ${O(t.class_id,this.ids)}
          <span class="muted small">${t.profile??`—`} · ${t.trigger??`—`}${t.query_chars==null?``:` · ${h(t.query_chars)} characters searched`}${t.cached?` · asked before`:``}</span>
        </div>
        <div class="lib-cands">${t.candidates.length?t.candidates.map(n=>e`<span class="lib-cand ${n.name&&t.chosen.includes(n.name)?`is-chosen`:``}">${n.name}<span class="num muted">${n.score==null?``:n.score.toFixed(2)}</span></span>`):e`<span class="muted small">no candidate shares a word with the message</span>`}</div>
        <div class="lib-pick small">${t.chosen.length?e`${t.mode===`on`&&t.applied?.length?`went in:`:`would have gone in:`} ${t.chosen.map(t=>e`<button class="btn btn-sm lib-open" data-lib-open=${t} @click=${()=>n(t)}>${t}</button>`)} <span class="muted">(${t.how??`—`})</span>`:e`<span class="muted">no pick (${(t.how??`none`).replace(/_/g,` `)})</span>`}</div>
      </div>`)}</div>`}};s([p({attribute:!1})],H.prototype,`ids`,void 0),s([p({attribute:!1})],H.prototype,`tiles`,void 0),s([p({attribute:!1})],H.prototype,`catalog`,void 0),s([p({attribute:!1})],H.prototype,`usage`,void 0),s([p({attribute:!1})],H.prototype,`onChanged`,void 0),s([t()],H.prototype,`profile`,void 0),s([t()],H.prototype,`busy`,void 0),s([t()],H.prototype,`msg`,void 0),s([t()],H.prototype,`error`,void 0),H=s([m(`qse-cond-library`)],H);var Vt={mode:`off`,idle_s:180,due_idle_s:3,min_new_tokens:1500,max_tokens:600,max_blocks:4,gate:!0,after_compaction:`idle`,turn_deadline_s:30,existing:5},Ht={index_tokens:1500,body_tokens:600,pinned_tokens:300,bodies:3,min_score:0,min_words:2,dedupe_threshold:.6,intent:`shadow`,provisional:!0,digest_chars:2e3,writer:Vt},Ut={rule:`how I want things done`,ruling:`a decision that was taken`,gotcha:`what went wrong once`,runbook:`how to do a thing`,program_state:`where the work stands`,reference:`a fact to look up`,registry:`a list of things`,archive:`kept for the record`},Wt=Object.keys(Ut),U={"qwen-inline":`the model, mid-answer`,"qwen-writer":`the model, in the background`,api:`the model's script`,human:`you`},Gt={node:null,kind:``,status:``,writer:``,stale:!1,q:``};function Kt(e,t){let n=t.q.trim().toLowerCase();return e.filter(e=>(!t.node||e.node===t.node)&&(!t.kind||e.kind===t.kind)&&(!t.status||e.status===t.status)&&(!t.writer||e.writer===t.writer)&&(!t.stale||!!e.stale)&&(!n||`${e.title} ${e.tags.join(` `)} ${e.id}`.toLowerCase().includes(n)))}function qt(e){return e.scope===`user`?`About you`:e.scope===`project`?e.label??e.node.replace(/^projects\//,``):`session ${e.label??e.node.replace(/^sessions\//,``)}`}function Jt(e,t){let n=e.split(`
`),r=t.split(`
`),i=n.length,a=r.length;if(i*a>4e5)return[...n.map(e=>({op:`-`,text:e})),...r.map(e=>({op:`+`,text:e}))];let o=Array.from({length:i+1},()=>Array(a+1).fill(0));for(let e=i-1;e>=0;e--)for(let t=a-1;t>=0;t--)o[e][t]=n[e]===r[t]?o[e+1][t+1]+1:Math.max(o[e+1][t],o[e][t+1]);let s=[],c=0,l=0;for(;c<i&&l<a;)n[c]===r[l]?(s.push({op:`=`,text:n[c]}),c++,l++):o[c+1][l]>=o[c][l+1]?s.push({op:`-`,text:n[c++]}):s.push({op:`+`,text:r[l++]});for(;c<i;)s.push({op:`-`,text:n[c++]});for(;l<a;)s.push({op:`+`,text:r[l++]});return s}function Yt(e){let t=e.detail??{};if(e.task===`nightly`)return`nightly: ${t.deleted??0} archives deleted, ${t.duplicates??0} duplicates, ${t.stale??0} stale`;let n=typeof t.why==`string`?t.why:null,r=String(t.op??e.decision??`write`),i=e.task===`api_write`?`the model's script`:U[String(t.writer??`qwen-inline`)]??String(t.writer);return n||e.decision&&![`add`,`update`,`archive`,`propose`,`ok`].includes(e.decision)?`refused (${n??e.decision}): ${r} by ${i}`:`${r===`add`?t.revised?`revised`:`added`:r===`update`?`updated`:r===`archive`?`archived`:r===`propose`?`proposed`:r} ${[t.kind,t.scope?`(${t.scope})`:null].filter(Boolean).join(` `)} by ${i}${t.id?` · ${t.id}`:``}`.replace(/\s+/g,` `).trim()}function Xt(e){let t=e?.memory?.notes??{};return{...Ht,...t,writer:{...Vt,...t.writer??{}}}}function Zt(e){let t=e.detail??{},n=Array.isArray(t.ops)?t.ops:[],r=e=>n.filter(t=>t.op===e&&(e!==`add`||!t.revised)).length,i=n.filter(e=>e.revised).length,a=[r(`add`)?`${r(`add`)} added`:``,i?`${i} revised`:``,r(`update`)?`${r(`update`)} updated`:``,r(`archive`)?`${r(`archive`)} archived`:``].filter(Boolean),o=t.gate??{},s=n.length&&n.every(e=>e.stored===!1)?` (shadow: not stored)`:``;return{when:e.ts_ms,trigger:String(t.trigger??`—`),outcome:typeof t.skipped==`string`?t.skipped:n.length?`wrote`:t.none?`nothing worth keeping`:`no block`,gate:o.answer?String(o.answer):String(o.how??`—`),seconds:typeof t.ms==`number`?Math.round(t.ms/100)/10:null,tokens:typeof t.tokens==`number`?t.tokens:null,ops:(a.join(`, `)||`—`)+s,refused:Array.isArray(t.refused)?t.refused:[]}}function Qt(e,t){let n=Object.values(e).reduce((e,t)=>e+t,0);if(!n)return`not injected yet`;let r=e[t]??0;return`${n} injection${n===1?``:`s`}${r?` (${r} today)`:``}`}var $t=e=>e?y(e):`—`,W=class extends o{constructor(...e){super(...e),this.onChanged=null,this.list=null,this.open=null,this.error=null,this.flash=null,this.busy=!1,this.status=`open`}connectedCallback(){super.connectedCallback(),this.load()}async load(){try{this.list=(await E.proposals()).proposals,this.error=null}catch(e){this.error=a(e).message}}async show(e){try{this.open=(await E.proposal(e.name,e.n)).proposal}catch(e){this.error=a(e).message}}async apply(e){if(await k(`Apply the proposal to ${e.name}?`,[e.title,`The text becomes a new version of the packages document; every session gets it from its next note. The versions panel restores the old text.`],`Apply`)){this.busy=!0;try{let t=await E.proposalApply(e.name,e.n);this.flash=`${e.name}: packages version ${t.version} (was ${t.previous})`,this.onChanged?.(),await this.load(),await this.show(e)}catch(e){this.error=a(e).message}finally{this.busy=!1}}}async reject(e){let t=window.prompt(`Why is this proposal rejected?`,`not what the skill needs`);if(t!==null){this.busy=!0;try{await E.proposalReject(e.name,e.n,t||`rejected in the dashboard`),this.flash=`${e.name}#${e.n} rejected`,await this.load(),await this.show(e)}catch(e){this.error=a(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.list)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${this.error}</div></div>`;if(!this.list)return u(4);let t=this.list.filter(e=>this.status===`all`||e.status===`open`),i=t=>e`<span class="badge ${t===`open`?`badge-shadow`:t===`applied`?`badge-ok`:``}">${t}</span>`;return e`
      ${this.flash?e`<p class="ok-ink small" role="status" data-prop-flash>${c(`check`)} ${this.flash}</p>`:r}
      ${this.error?e`<p class="err-ink small" role="alert">${c(`warn`)} ${this.error}</p>`:r}
      ${this.open?this.renderOne(this.open):r}
      ${n(`Proposals`,t.length?e`<div class="table-scroll"><table class="table" data-proposals><thead><tr><th>package</th><th>change</th><th>mode</th><th>status</th><th>evidence</th><th>proposed</th></tr></thead><tbody>${t.map(t=>e`<tr class="req-row ${this.open&&this.open.name===t.name&&this.open.n===t.n?`is-open`:``}" data-proposal=${`${t.name}#${t.n}`} @click=${()=>this.show(t)}>
                <td class="mono small">${t.name}#${t.n}</td><td class="cell-wrap">${t.title}</td><td class="small">${t.mode}</td><td>${i(t.status)}</td>
                <td class="small">${t.evidence.length?`${t.evidence.length} outcome row${t.evidence.length===1?``:`s`}`:e`<span class="muted">no evidence</span>`}</td><td class="num small">${$t(t.created)}</td></tr>`)}</tbody></table></div>`:l(this.status===`open`?`No open proposal`:`No proposal yet`,`The model proposes a change to a package with a hidden block when a session showed a gap; it waits here for your click.`),{sub:`changes to packages and skills the model proposed; none goes live without Apply`,tools:e`<label class="switch small"><input type="checkbox" .checked=${this.status===`all`} @change=${e=>this.status=e.target.checked?`all`:`open`} data-prop-all /> applied and rejected too</label>
            <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the proposals">${c(`refresh`)}</button>`,id:`proposals`})}`}renderOne(t){let n=t.live===null?[]:Jt(t.live,t.proposed);return e`<section class="panel" style="margin-bottom:16px" data-proposal-open=${`${t.name}#${t.n}`}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${t.name}#${t.n}: ${t.title}</h2>
        <p class="panel-sub">${t.mode===`append`?`adds a section`:`replaces the text`} · ${t.status}${t.applied_version?` as version ${t.applied_version}`:``}${t.rejected_reason?`: ${t.rejected_reason}`:``} · ${t.evidence_word}${t.session?e` · <a class="mono" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(t.session)}`}>${t.session}</a>`:r}</p></div>
        <div class="panel-tools">
          ${t.status===`open`?e`<button class="btn btn-primary btn-sm" ?disabled=${!t.applies||this.busy} @click=${()=>this.apply(t)} data-prop-apply>Apply</button>
            <button class="btn btn-sm" ?disabled=${this.busy} @click=${()=>this.reject(t)} data-prop-reject>Reject</button>`:r}
          <button class="btn btn-ghost btn-icon" aria-label="Close the proposal" @click=${()=>this.open=null}>${c(`x`)}</button></div></header>
      ${t.reason?e`<p class="small"><span class="muted">why:</span> ${t.reason}</p>`:r}
      ${t.stale&&t.status===`open`?e`<p class="warn-ink small">${c(`warn`)} ${t.name} changed since this was proposed${t.mode===`append`?`; the section is added to the text as it is now`:`; a replacement cannot be applied`}.</p>`:r}
      ${t.live===null?e`<p class="warn-ink small">No package ${t.name} with a text any more.</p>`:e`<pre class="notes-diff" data-prop-diff>${n.map(t=>e`<span class="diff-${t.op===`+`?`add`:t.op===`-`?`del`:`eq`}">${t.op===`=`?` `:t.op} ${t.text}\n</span>`)}</pre>`}
    </section>`}};s([t()],W.prototype,`list`,void 0),s([t()],W.prototype,`open`,void 0),s([t()],W.prototype,`error`,void 0),s([t()],W.prototype,`flash`,void 0),s([t()],W.prototype,`busy`,void 0),s([t()],W.prototype,`status`,void 0),W=s([m(`qse-cond-proposals`)],W);var en=[{id:`text`,label:`Text`},{id:`edit`,label:`Edit`},{id:`classes`,label:`Classes`},{id:`usage`,label:`Usage`}],G=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.selected=null,this.tab=null,this.pane=null,this.catalog=null,this.catError=null,this.usage=null,this.filters=St,this.hits=null,this.libError=null,this.item=null,this.itemError=null,this.recent=null,this.raw=!1,this.editView=`write`,this.counted=null,this.review=null,this.attachError=null,this.found=null,this.scanError=null,this.pk=new D(this,`packages`),this.rt=new D(this,`router`),this.seq=0,this.loadedFor=null,this.countTimer=null,this.countFor=null}connectedCallback(){super.connectedCallback(),this.pk.onSaved=()=>void this.refresh(),this.rt.onSaved=()=>void this.refresh(),this.pk.load(),this.rt.load(),this.refresh()}disconnectedCallback(){this.countTimer&&clearTimeout(this.countTimer),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.item=null,this.itemError=null,this.recent=null,this.attachError=null,this.selected&&this.loadItem(this.selected))}updated(e){let t=this.selected?this.pk.draft?.packages[this.selected]:null;if(!t||t.source!==`inline`)return;let n=V(t.body,!!t.library)[0];this.counted?.text!==n&&this.countFor!==n&&(this.countFor=n,this.countTimer&&clearTimeout(this.countTimer),this.countTimer=setTimeout(()=>{E.packagesTokens([n]).then(e=>this.counted={text:n,n:e.tokens[0],how:e.tokenizer}).catch(()=>void 0)},400))}async refresh(){try{let[e,t]=await Promise.all([E.packagesCatalog(),E.packagesUsage(null,1)]);[this.catalog,this.usage,this.catError]=[e,t,null]}catch(e){this.catError=a(e).message}this.selected&&await this.loadItem(this.selected)}async loadItem(e){try{let[t,n]=await Promise.all([this.pk.base?.doc.packages[e]||!this.pk.base?E.packageItem(e):Promise.resolve(null),E.packagesUsage(e,20)]);if(e!==this.selected)return;[this.item,this.recent,this.itemError]=[t,n.recent,null]}catch(t){e===this.selected&&(this.itemError=a(t).message)}}async search(e){this.filters={...this.filters,q:e};let t=++this.seq;if(!e.trim()||!Object.values(this.pk.draft?.packages??{}).some(e=>e.library)){[this.hits,this.libError]=[null,null];return}try{let n=await E.librarySearch(e.trim(),20);t===this.seq&&([this.hits,this.libError]=[n,null])}catch(e){t===this.seq&&([this.hits,this.libError]=[[],`the library's search: ${a(e).message}`])}}open(e,t=null){x({pkg:e,pview:t})}upd(e,t){this.pk.draft&&this.pk.set(S(this.pk.draft,e,t))}toggleAttach(e,t,n){let r=this.rt.draft;if(!r)return;if(!n){this.attachError=null,this.rt.set(Pt(r,e,t));return}let i=Nt(r,e,t,this.pk.draft);this.attachError=i.error,i.error||this.rt.set(i.router)}addInline(){let e=this.pk.draft,t=It(e);this.pk.set(S(e,[`packages`,t],Lt(Number(t.slice(8))))),this.open(t,`edit`)}copyInline(e){let t=this.pk.draft,n=this.item;n&&n.name===e&&n.found&&(this.pk.set(S(t,[`packages`,e],Rt(t.packages[e],n.body,n.reference))),this.open(e,`edit`))}removePackage(e){let t=this.pk.draft,n={...t.packages};delete n[e],this.pk.set({...t,packages:n}),this.open(null)}async scan(){this.scanError=null;try{this.found=(await E.scan()).skills}catch(e){this.scanError=a(e).message}}register(e){let t={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.pk.set(S(this.pk.draft,[`packages`,e.name],t)),this.open(e.name)}async startReview(e,t){await e.check(),this.review=t}async confirmSave(e){await e.save()&&(this.review=null)}bar(t,n,i){let a=t.base?.meta,o=!!t.draft&&!!t.base&&JSON.stringify(t.draft)!==JSON.stringify(t.base.doc);return e`
      ${t.showVersions?lt(t,i):r}
      ${this.review===n?e`<section class="panel pk-review" data-review=${n}>
            <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${t.errors.length?e`${c(`warn`)} The server refuses ${i}`:e`${c(`check`)} The server accepts ${i}`}</h2>
              <p class="panel-sub">POST /v1/conductor/${n}/check against live v${a?.version??0}: ${t.changes.length} change${t.changes.length===1?``:`s`}. Saving makes a new version with your note; the next request uses it.</p></div></header>
            ${A(t)}
            ${at(t.changes)}
            <div class="toolbar pk-review-tools">
              <button class="btn btn-primary" data-review-save=${n} ?disabled=${t.errors.length>0||!t.note.trim()||t.saving||!t.changes.length} @click=${()=>this.confirmSave(t)}>${t.saving?`Saving…`:`Save as a new version`}</button>
              <button class="btn btn-ghost" @click=${()=>this.review=null}>Back to editing</button>
              ${t.note.trim()?r:e`<span class="muted small">Write the change note first.</span>`}
            </div>
          </section>`:r}
      <div class="savebar ${o||this.review===n?``:`is-idle`}" data-savebar=${n}>
        <span class="status num" title=${a?.note??``}>${i} · live v${a?.version??0}${o?e` · <strong>${t.changes.length||`…`} change${t.changes.length===1?``:`s`}</strong>`:r}</span>
        <input class="note-input" placeholder="one-line change note (required)" .value=${t.note} @input=${e=>{t.note=e.target.value,this.requestUpdate()}} aria-label=${`change note for ${i}`} maxlength="200" data-note=${n} />
        <button class="btn btn-primary" ?disabled=${!o||t.saving} @click=${()=>this.startReview(t,n)} data-check-save=${n}>Check and save…</button>
        <button class="btn btn-ghost" ?disabled=${!o} @click=${()=>{t.reset(),this.review=null}}>Reset</button>
        <button class="btn btn-ghost ${t.showVersions?`is-on`:``}" @click=${()=>{t.showVersions=!t.showVersions,t.showVersions?t.loadVersions():this.requestUpdate()}} data-versions-btn=${n}>Versions</button>
        ${t.saved?e`<span class="status" role="status" data-saved=${n}>${c(`check`)} ${t.saved}</span>`:r}
        ${t.saveError?e`<span class="err-ink small" role="alert">${t.saveError}</span>`:r}
      </div>`}render(){let t=e`<nav class="mem-tabs pk-tabs-top" aria-label="packages tabs">
      ${[{id:null,label:`Packages and skills`},{id:`library`,label:`Library retrieval`},{id:`proposals`,label:`Proposals`}].map(t=>e`<a class="cond-tab ${(this.tab??null)===t.id?`is-on`:``}" href="#" data-pk-top=${t.id??`all`} aria-current=${(this.tab??null)===t.id?`page`:r} @click=${e=>{e.preventDefault(),x({ptab:t.id,pkg:null,pview:null})}}>${t.label}</a>`)}
    </nav>`;if(this.tab===`proposals`)return e`${t}<qse-cond-proposals .onChanged=${()=>this.refresh()}></qse-cond-proposals>`;if(this.tab===`library`)return e`${t}<qse-cond-library .ids=${this.ids} .tiles=${this.tiles} .catalog=${this.catalog} .usage=${this.usage} .onChanged=${()=>this.refresh()}></qse-cond-library>`;let n=this.pk;if(n.loading&&!n.draft)return e`${t}${u(6)}`;if(n.loadError)return e`${t}<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${n.loadError}</div></div>`;let i=n.draft,a=Dt(i,this.catalog,this.usage?.packages??null,this.rt.draft),o=jt(a,this.filters,this.hits,this.catalog),s=this.selected&&i.packages[this.selected]?this.selected:null;return e`
      ${t}
      ${this.catError?e`<div class="guard-line" role="alert">${c(`warn`)} ${this.catError}</div>`:r}
      <div class="pk-layout ${s?`has-detail`:``}">
        ${this.renderList(o,a)}
        ${s?this.renderDetail(s,i.packages[s],a.find(e=>e.name===s)):this.selected?e`<section class="panel" data-pk-gone><p class="muted">No package ${this.selected} in this version.</p><button class="btn btn-sm" @click=${()=>this.open(null)}>Back to the list</button></section>`:r}
      </div>
      ${this.renderFound(i)}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${i.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${this.rt.draft&&(this.rt.showVersions||this.rt.saved||this.review===`router`||JSON.stringify(this.rt.draft)!==JSON.stringify(this.rt.base?.doc))?this.bar(this.rt,`router`,`the router`):r}
      ${this.bar(this.pk,`packages`,`the packages`)}
    `}renderList(t,n){let i=this.filters,a=n.filter(e=>At(e,this.catalog)).length,o=e=>this.filters={...this.filters,...e};return e`<section class="panel pk-list" data-pk-list>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills <span class="muted" data-pk-count-all>· ${t.length} of ${n.length}</span></h2>
        <p class="panel-sub">What the router may put into a prompt. Tokens are the text that goes in, ${this.catalog?.tokenizer===`estimate`?`estimated at 3.5 characters a token (no tokenizer on this server)`:`on the engine's tokenizer`}${a?e`; <span class="warn-ink" data-pk-over-count>${a} over the limit</span>`:r}.</p></div>
        <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-pk-new>${c(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${c(`search`)} Skills on the box</button></div></header>
      <div class="pk-filters">
        <label class="field field-search pk-search">${c(`search`)}<input type="search" data-pk-search placeholder="search packages, or the library's text" aria-label="search packages and skills" .value=${i.q} @input=${e=>void this.search(e.target.value)} /></label>
        <label class="field"><span>kind</span><select data-pk-filter="kind" @change=${e=>o({category:e.target.value})}>
          <option value="all" ?selected=${i.category===`all`}>all</option>
          ${z.map(t=>e`<option value=${t} ?selected=${i.category===t}>${B[t]}</option>`)}
        </select></label>
        <label class="field"><span>class</span><select data-pk-filter="class" @change=${e=>o({cls:e.target.value})}>
          <option value="" ?selected=${!i.cls}>any</option>
          ${this.ids.map(t=>e`<option value=${t} ?selected=${i.cls===t}>${t}</option>`)}
        </select></label>
        <label class="field"><span>enabled</span><select data-pk-filter="enabled" @change=${e=>o({enabled:e.target.value})}>
          ${[`all`,`on`,`off`].map(t=>e`<option value=${t} ?selected=${i.enabled===t}>${t}</option>`)}
        </select></label>
        <label class="switch"><input type="checkbox" data-pk-filter="over" .checked=${i.over} @change=${e=>o({over:e.target.checked})} /> over the limit</label>
      </div>
      ${this.libError?e`<p class="err-ink small">${this.libError}</p>`:r}
      <div class="pk-head" aria-hidden="true"><span>package</span><span>tokens / limit</span><span>used 24 h · 7 d</span></div>
      <div class="pk-rows" role="list">${t.map(e=>this.row(e))}</div>
      ${t.length?r:e`<p class="muted small" data-pk-empty>${i.q.trim()?`Nothing matches “${i.q.trim()}”${this.hits?`, and the library shares no word with it`:``}.`:`No package matches these filters.`}</p>`}
    </section>`}row(t){let n=Tt(t.name,t.pkg,t.category,this.catalog),i=t.entry?t.entry.tokens:Ct(V(t.pkg.body,!!t.pkg.library)[0]),a=At(t,this.catalog),o=t.usage,s=this.selected===t.name;return e`<button class="pk-row ${s?`is-open`:``} ${t.pkg.enabled?``:`is-off`}" role="listitem" data-pk-row=${t.name} data-category=${t.category} aria-current=${s?`true`:r} @click=${()=>this.open(t.name)}>
      <span class="pk-main">
        <span class="pk-name"><strong>${t.name}</strong>
          <span class="badge pk-kind pk-kind-${t.category}">${B[t.category]}</span>
          ${t.pkg.general?e`<span class="badge badge-on" title="goes in whole and first on every turn of a class that lists it">general</span>`:r}
          ${t.pkg.enabled?r:e`<span class="badge">off</span>`}
          ${t.entry?r:e`<span class="badge badge-shadow">not saved</span>`}
          ${t.entry&&!t.entry.available&&t.entry.error?e`<span class="badge badge-warn" title=${t.entry.path??``}>${t.entry.error.replace(/_/g,` `)}</span>`:r}
          ${t.score==null?r:e`<span class="pk-score num" data-score title="the library's BM25 score">${t.score.toFixed(2)}</span>`}
        </span>
        <span class="pk-desc muted small">${t.pkg.description}</span>
        <span class="pk-meta"><span class="mono small muted">${t.pkg.source}</span>${t.classes.map(e=>O(e,this.ids))}${t.category===`library`&&t.entry?.fits?.length?e`<span class="muted small">fits</span>${t.entry.fits.map(e=>O(e,this.ids))}`:r}</span>
      </span>
      <span class="pk-tokens num ${a?`is-over`:``}" data-pk-tokens title=${t.entry?`${t.entry.tokens} tokens go in; the whole text is ${t.entry.body_tokens}`:`estimated: not saved yet`}>
        <span>${t.entry?``:`≈`}${h(i)} <span class="muted">/ ${h(n.limit)}</span></span>
        <span class="pk-bar"><span style="width:${Math.min(100,100*i/Math.max(1,n.limit))}%"></span></span>
        ${a?e`<span class="badge badge-warn">over</span>`:r}
      </span>
      <span class="pk-usage num small" data-pk-usage>
        ${o&&(o.injected.week||o.shadow.week||o.pick.week)?e`${o.injected.week?e`<span>${o.injected.day} · ${o.injected.week} in</span>`:r}${o.shadow.week?e`<span class="muted">${o.shadow.day} · ${o.shadow.week} shadow</span>`:r}${o.pick.week?e`<span>${o.pick.day} · ${o.pick.week} picks</span>`:r}`:e`<span class="muted">—</span>`}
      </span>
    </button>`}renderFound(t){return this.scanError?e`<p class="err-ink small">${this.scanError}</p>`:this.found?e`<section class="panel" style="margin-top:16px" data-found>
      <header class="panel-head"><h2 class="panel-title">Skills in ${t.skill_dirs.join(`, `)}</h2><div class="panel-tools"><button class="btn btn-ghost btn-icon" aria-label="Close" @click=${()=>this.found=null}>${c(`x`)}</button></div></header>
      ${this.found.length?e`<div class="table-scroll"><table class="table"><tbody>${this.found.map(n=>e`<tr><td><strong>${n.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${n.description}</div></td><td class="num">${h(n.chars)} chars</td>
              <td>${n.registered||t.packages[n.name]?e`<span class="badge badge-ok">registered</span>`:e`<button class="btn btn-sm" @click=${()=>this.register(n)} data-register=${n.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:e`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ &lt;box&gt;:~/.config/opencode/skills/</code>).</p>`}
    </section>`:r}renderDetail(t,n,i){let a=i.entry,o=wt(n,a),s=Tt(t,n,o,this.catalog),l=en.some(e=>e.id===this.pane)?this.pane:`text`,u=n.source===`inline`,d=i.classes,f=this.tokensNow(t,n,a),p=f.n>s.limit||!!a?.cut&&!(u&&this.dirtyBody(t));return e`<section class="panel pk-detail" data-pk-detail=${t}>
      <header class="pk-detail-head">
        <div class="pk-title">
          <h2 class="panel-title">${t}</h2>
          <span class="muted">${n.title}</span>
        </div>
        <button class="btn btn-ghost btn-sm pk-close" @click=${()=>this.open(null)} data-pk-close aria-label="Close the package">${c(`x`)}<span class="pk-close-word">All packages</span></button>
      </header>
      <div class="pk-badges">
        <span class="badge pk-kind pk-kind-${o}" data-pk-category>${B[o]}</span>
        <span class="badge">${u?`inline`:n.source.startsWith(`skill:`)?`skill on the box`:`file on the box`}</span>
        ${n.general?e`<span class="badge badge-on">general</span>`:r}
        ${n.enabled?e`<span class="badge badge-ok">enabled</span>`:e`<span class="badge">disabled</span>`}
        ${a?r:e`<span class="badge badge-shadow">not saved</span>`}
        ${a&&!a.available&&a.error?e`<span class="badge badge-warn">${a.error.replace(/_/g,` `)}</span>`:r}
      </div>
      <div class="toolbar pk-switches">
        <label class="switch"><input type="checkbox" data-pk-enabled .checked=${n.enabled} @change=${e=>this.upd([`packages`,t,`enabled`],e.target.checked)} /> enabled</label>
        ${o===`library`?r:e`<label class="switch" title="goes in whole and first on every turn of a class that lists it (K1)"><input type="checkbox" data-pk-general .checked=${!!n.general} @change=${e=>this.upd([`packages`,t,`general`],e.target.checked)} /> general</label>`}
        ${u?r:e`<button class="btn btn-sm" data-pk-copy-inline ?disabled=${!this.item?.found} title=${this.item?.found?`an inline copy of the file overrides it`:`the file is not on the box`} @click=${()=>this.copyInline(t)}>${c(`copy`)} Copy to inline</button>`}
        <button class="btn btn-sm btn-ghost" data-pk-remove ?disabled=${d.length>0} title=${d.length?`${d.join(`, `)} list it; detach it first`:`remove it from the packages document`} @click=${()=>this.removePackage(t)}>${c(`trash`)} Remove</button>
      </div>
      <div class="pk-meter ${p?`is-over`:``}" data-pk-meter>
        <div class="pk-bar"><span style="width:${Math.min(100,100*f.n/Math.max(1,s.limit))}%"></span></div>
        <span class="num small" data-pk-count>${f.approx?`≈ `:``}${h(f.n)} tokens of ${h(s.limit)} <span class="muted">(${f.why}; limit: ${s.source===`house`?o===`library`?`the house rule for a library skill`:`the house rule for the library card`:`max_tokens, where the server cuts`})</span></span>
        ${p?e`<span class="badge badge-warn" data-pk-limit-warn>over the limit</span>`:r}
      </div>
      <nav class="mem-tabs pk-panes" aria-label="package detail">
        ${en.map(t=>e`<a class="cond-tab ${t.id===l?`is-on`:``}" href="#" data-pk-pane=${t.id} aria-current=${t.id===l?`page`:r} @click=${e=>{e.preventDefault(),x({pview:t.id===`text`?null:t.id})}}>${t.label}</a>`)}
      </nav>
      <div class="pk-pane" data-pk-pane-body=${l}>
        ${l===`text`?this.paneText(t,n,a):l===`edit`?this.paneEdit(t,n,o,s):l===`classes`?this.paneClasses(t,n,o):this.paneUsage(t,o)}
      </div>
    </section>`}dirtyBody(e){return this.pk.draft?.packages[e]?.body!==this.pk.base?.doc.packages[e]?.body}tokensNow(e,t,n){if(t.source===`inline`&&(this.dirtyBody(e)||!n)){let e=V(t.body,!!t.library)[0];return this.counted?.text===e?{n:this.counted.n,approx:this.counted.how===`estimate`,why:this.counted.how===`engine`?`the draft, on the engine's tokenizer`:`the draft, estimated`}:{n:Ct(e),approx:!0,why:`the draft, estimated until the engine counts it`}}return n?{n:n.body_tokens,approx:this.catalog?.tokenizer===`estimate`,why:n.cut?`the whole text; ${h(n.tokens)} go in after the cut`:this.catalog?.tokenizer===`estimate`?`estimated`:`on the engine's tokenizer`}:{n:0,approx:!0,why:`not counted yet`}}md(t){return t?this.raw?e`<pre class="note-text pk-raw" data-pk-raw>${t}</pre>`:e`<div class="pg-md pk-md" data-pk-md>${nt(rt(t))}</div>`:e`<p class="muted small">(empty)</p>`}paneText(t,n,i){let a=n.source===`inline`,o=this.item?.name===t?this.item:null;if(!a&&!o)return this.itemError?e`<p class="err-ink small">${this.itemError}</p>`:u(4);let[s,l]=a?V(n.body,!!n.library):[o.body,o.reference],d=a?{}:o.frontmatter;return e`
      <div class="toolbar pk-text-tools">${j([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`text view`})}
        ${!a&&o?.path?e`<span class="mono small muted pk-path" data-pk-path>${o.path}</span>`:r}</div>
      ${!a&&o&&!o.found?e`<p class="note" data-pk-missing>${c(`warn`)} The file is not on the box (<code>${o.path??n.source}</code>): nothing goes in.</p>`:r}
      ${Object.keys(d).length?e`<h3 class="pk-sub">Frontmatter <span class="muted small">the client's metadata; never injected</span></h3>
            ${this.raw?e`<pre class="note-text pk-raw" data-pk-front>${o.frontmatter_raw}</pre>`:e`<dl class="kv pk-front" data-pk-front>${Object.entries(d).map(([t,n])=>e`<div><dt>${t}</dt><dd>${n}</dd></div>`)}</dl>`}`:r}
      <h3 class="pk-sub">The text that goes in ${s?f(s,`Copy the text`):r}</h3>
      ${this.md(s)}
      ${i?.cut&&!this.dirtyBody(t)?e`<p class="note" data-pk-cut>${c(`warn`)} The server cuts this text at ${h(n.max_tokens)} tokens × 3.5 characters, at a line end, and adds “[... cut to the package's max_tokens]”: ${h(i.tokens)} tokens go in of ${h(i.body_tokens)}.</p>`:r}
      ${l?e`<details class="sec pk-ref" data-pk-ref>
            <summary>Reference <span class="sec-sub">${h(o?.tokens.reference??Ct(l))} tokens; only a read gets it (GET /v1/library/${t})</span></summary>
            <div class="sec-body">${this.md(l)}</div>
          </details>`:r}`}paneEdit(t,n,i,a){let o=n.source===`inline`;return e`
      <div class="form-grid">
        <label class="field"><span>title</span><input data-pk-title .value=${n.title} @input=${e=>this.upd([`packages`,t,`title`],e.target.value)} /></label>
        <label class="field"><span>max tokens</span><input type="number" min="16" max="32768" data-pk-max .value=${String(n.max_tokens)} @input=${e=>this.upd([`packages`,t,`max_tokens`],Math.round(Number(e.target.value)))} /><span class="hint">the server cuts a longer text at a line end</span></label>
        ${o?r:e`<label class="field field-wide"><span>source</span><input class="mono" .value=${n.source} readonly /><span class="hint">${this.item?.path??`a file on the box`}</span></label>`}
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" data-pk-description .value=${n.description} @input=${e=>this.upd([`packages`,t,`description`],e.target.value)}></textarea></label>
      ${o?this.editor(t,n,i,a):e`<div class="note pk-readonly" data-pk-readonly>${c(`info`)} The text is the file on the box${this.item?.path?e` at <code>${this.item.path}</code>`:r}, read when first needed and again when it changes. Editing files on the box is not done here. <button class="btn btn-sm" ?disabled=${!this.item?.found} @click=${()=>this.copyInline(t)} data-pk-copy-inline-edit>Copy to inline</button> makes an inline copy of its text that overrides the file.</div>`}`}editor(t,n,i,a){let[o,s]=V(n.body,!!n.library),c=this.tokensNow(t,n,this.catalog?.packages[t]??null),l=c.n>a.limit,u=this.editView;return e`<div class="pk-editor pk-view-${u}" data-pk-editor>
      <div class="toolbar pk-editor-tools">
        ${j([`write`,`preview`,`split`],u,e=>this.editView=e,{small:!0,label:`editor view`})}
        <span class="pk-count num small ${l?`is-over`:``}" data-pk-editor-count>${c.approx?`≈ `:``}${h(c.n)} of ${h(a.limit)} tokens${n.library&&s?e` <span class="muted">· reference ≈ ${h(Ct(s))}</span>`:r}</span>
        ${l?e`<span class="badge badge-warn" data-pk-editor-warn>over ${a.source===`house`?`the ${B[i]}'s ${h(a.limit)}`:`max_tokens: the server will cut it`}</span>`:r}
      </div>
      <div class="pk-editor-panes">
        ${u===`preview`?r:e`<textarea class="code pk-body" rows="18" spellcheck="false" data-pk-body aria-label="the package's text (Markdown)" .value=${n.body} @input=${e=>this.upd([`packages`,t,`body`],e.target.value)}></textarea>`}
        ${u===`write`?r:e`<div class="pg-md pk-md pk-preview" data-pk-preview>${nt(rt(o))}</div>`}
      </div>
      <p class="hint">${n.library?`A library skill: the text before “## Reference” goes in; the reference only on a read. `:``}${a.source===`house`?`The house rule keeps ${B[i]} texts at ${h(a.limit)} tokens or fewer.`:`The server cuts the text at max_tokens × 3.5 characters, at a line end.`} Markdown; the preview is the dashboard's renderer, the model gets the text as written.</p>
    </div>`}paneClasses(t,n,i){let a=this.catalog?.packages[t]??null;if(i===`library`)return e`<p class="muted small" data-pk-library-classes>A library skill is searched, not listed by a class: on a new user message the router searches the library and keeps the skills INDEX.md fits to the turn's class.</p>
        <div class="pk-meta">${a?.fits?.length?e`<span class="muted small">INDEX.md fits</span>${a.fits.map(e=>O(e,this.ids))}`:e`<span class="muted small">INDEX.md does not list it: it stays in for every class.</span>`}</div>
        <p class="small"><a href="#/conductor?screen=packages&ptab=library">Each profile's retrieval switch and its picks</a></p>`;let o=this.rt.draft;return o?e`
      <p class="muted small">A class's packages go in by K1's order: the general ones whole and first, then the router asks the package question among the rest and puts in the one that fits. A class lists at most ${12}. Attaching edits the router document; it is saved with its own check below.</p>
      ${this.attachError?e`<p class="err-ink small" role="alert" data-pk-attach-error>${this.attachError}</p>`:r}
      <div class="pk-classes" data-pk-classes>
        ${o.classes.map(n=>{let i=n.policy.packages??[],a=i.includes(t);return e`<div class="pk-class ${a?`is-on`:``}" data-pk-class=${n.id}>
            <label class="switch"><input type="checkbox" .checked=${a} data-pk-attach=${n.id} aria-label=${`attach ${t} to ${n.id}`} @change=${e=>this.toggleAttach(n.id,t,e.target.checked)} /> ${O(n.id,this.ids,n.label)}</label>
            <span class="pk-order small" data-pk-order=${n.id}>${i.length?i.map((n,i)=>e`<span class="pk-order-item ${n===t?`is-me`:``}">${i+1}. ${n}${this.pk.draft?.packages[n]?.general?e` <span class="muted">general</span>`:r}</span>`):e`<span class="muted">no packages</span>`}</span>
            <span class="muted small num">${i.length}/${12}</span>
          </div>`})}
      </div>
      ${a&&Object.keys(a.profiles).length?e`<p class="small muted" data-pk-profiles>A profile's own routing also lists it: ${Object.entries(a.profiles).map(([e,t])=>`${e} (${t.join(`, `)})`).join(`; `)}. A profile's routing replaces the router's policy for that class (the Profiles screen).</p>`:r}
      <div class="toolbar"><button class="btn btn-sm btn-ghost" data-pk-router-versions @click=${()=>{this.rt.showVersions=!0,this.rt.loadVersions()}}>Router versions</button></div>`:u(3)}paneUsage(t,n){let i=this.usage?.packages[t]??null,a={day:0,week:0},o=(e,t)=>d({label:e,value:h(t.day),unit:`in 24 h`,sub:`${h(t.week)} in 7 days`}),s=this.item?.name===t?this.item:null,c=n===`library`||(i?.pick.week??0)>0;return e`
      <div class="stats pk-stats" style="--pk-tiles:${c?3:2}" data-pk-usage-tiles>
        ${o(`injected`,i?.injected??a)}${o(`shadow (would have)`,i?.shadow??a)}${c?o(`library picks`,i?.pick??a):r}
      </div>
      <h3 class="pk-sub">Recent rows <span class="muted small">the journal, newest first (7 days)</span></h3>
      ${this.recent==null?u(3):this.recent.length?e`<div class="pk-uses" data-pk-recent>${this.recent.map(t=>e`<div class="pk-use" data-pk-use=${t.id}>
                <span class="num small">${y(new Date(t.ts_ms).toISOString())}</span>
                <span class="badge ${t.how===`injected`?`badge-on`:t.how===`shadow`?`badge-shadow`:`badge-warn`}">${t.how}</span>
                ${O(t.class_id,this.ids)}
                <span class="muted small">${t.profile??`—`}${t.client_kind?` · ${t.client_kind}`:``}</span>
                ${t.session_key?e`<a class="mono small pk-session" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(t.session_key)}`}>${t.session_key.length>28?`${t.session_key.slice(0,28)}…`:t.session_key}</a>`:r}
              </div>`)}</div>`:e`<p class="muted small" data-pk-recent-empty>No router row names it in the last 7 days.</p>`}
      ${n===`library`?e`<h3 class="pk-sub">Neighbours <span class="muted small">the library's BM25 on this skill's name, description and text</span></h3>
            ${s?.neighbours?s.neighbours.length?e`<div class="pk-neighbours" data-pk-neighbours>${s.neighbours.map(t=>e`<button class="pk-neighbour" data-pk-neighbour=${t.name} @click=${()=>this.open(t.name)}><strong>${t.name}</strong> <span class="num muted small">${t.score.toFixed(2)}</span><span class="muted small pk-desc">${t.description}</span></button>`)}</div>`:e`<p class="muted small">No other skill shares a word with it.</p>`:u(2)}`:r}`}};s([p({attribute:!1})],G.prototype,`ids`,void 0),s([p({attribute:!1})],G.prototype,`tiles`,void 0),s([p({attribute:!1})],G.prototype,`selected`,void 0),s([p({attribute:!1})],G.prototype,`tab`,void 0),s([p({attribute:!1})],G.prototype,`pane`,void 0),s([t()],G.prototype,`catalog`,void 0),s([t()],G.prototype,`catError`,void 0),s([t()],G.prototype,`usage`,void 0),s([t()],G.prototype,`filters`,void 0),s([t()],G.prototype,`hits`,void 0),s([t()],G.prototype,`libError`,void 0),s([t()],G.prototype,`item`,void 0),s([t()],G.prototype,`itemError`,void 0),s([t()],G.prototype,`recent`,void 0),s([t()],G.prototype,`raw`,void 0),s([t()],G.prototype,`editView`,void 0),s([t()],G.prototype,`counted`,void 0),s([t()],G.prototype,`review`,void 0),s([t()],G.prototype,`attachError`,void 0),s([t()],G.prototype,`found`,void 0),s([t()],G.prototype,`scanError`,void 0),G=s([m(`qse-cond-packages`)],G);var tn={not_loaded:`not loaded`,loading:`loading the weights`,warming:`warming up`,loaded:`loaded`,unloading:`unloading`,failed:`failed`};function nn(e){let t=e.role;return t?[t.kind,t.window.replace(`_`,` `),t.preemptible?`preemptible`:`not preemptible`,`${t.latency_budget_ms} ms then ${t.fallback}`,`writes ${t.writes.replace(/_/g,` `)}`,`shares ${t.sharing.join(`, `)||`nothing`}`,t.memory_claim_gib==null?`no claim`:`claims ${w(t.memory_claim_gib)}`].join(` · `):null}function rn(e){return e===`loaded`?`badge-ok`:e===`failed`?`badge-warn`:e===`loading`||e===`warming`||e===`unloading`?`badge-shadow`:``}var an=`not loaded: kept free for requests`;function on(e){if(!e||e.allowed)return null;let t=e.reason?Re[e.reason]??e.reason:`refused`;if(e.reason!==`refused_memory`)return t;if(e.mem_available_gib==null||e.after_gib==null)return`${an} (MemAvailable not read)`;let n=e.engine_growth_gib?`less ${w(e.engine_growth_gib)} the engine may still take for a long request and the ${w(e.claim_gib)} claim`:`less the ${w(e.claim_gib)} claim`,r=e.margin_gib?`${w(e.need_gib)} free (a ${w(e.floor_gib)} floor and a ${w(e.margin_gib)} margin)`:`${w(e.need_gib)} free`;return`${an} (MemAvailable ${w(e.mem_available_gib)}, ${n}, leaves ${w(e.after_gib)}; the box keeps ${r})`}function sn(e){return e?e.allowed?`fits, ${w(e.after_gib)} left after the claim`:on(e)??`refused`:`—`}function cn(e){return e?e.allowed?`fits`:e.reason===`refused_memory`?`kept free`:`refused`:`—`}function ln(e){return!e?.allowed&&e?.reason!==`refused_memory`}function un(e){if(!e)return``;let t=(e.source??``).replace(/^engine:/,``);return e.role===`shadow`?t?`the shadow reader (${t})`:`the shadow reader`:t}function dn(e){let t=Object.entries(e).sort((e,t)=>t[1]-e[1]);return t.length?t.map(([e,t])=>`${e} ${t}`).join(` · `):`no call yet`}function fn(e){let t=e.limits;return[t.context_tokens,t.question_tokens,t.max_options].map(e=>e==null?`—`:String(e)).join(` / `)}function pn(e,t){let n=e.trim();if(!n)return`Write the question.`;let r=t.split(/[,\n]/).map(e=>e.trim()).filter(Boolean);return r.length<2?`Give at least two options.`:r.length>20?`At most 20 options (the model card’s limit).`:new Set(r).size===r.length?{type:`choice`,instructions:n,criteria:Object.fromEntries(r.map(e=>[e,null]))}:`Two options have the same name.`}function mn(e){return e.type===`noul`?[[`true`,e.noul??0],[`false`,1-(e.noul??0)]]:Object.entries(e.probabilities??{}).sort((e,t)=>t[1]-e[1])}var hn=[`on_start`,`on_demand`,`off`],gn={mode:`on`,rate_per_min:240,max_texts:16,max_chars:1e5,deadline_ms:15e3,threshold:.8,drive_rate_per_s:10,drive_sessions:4,drive_idle_s:30,drive_max_texts:36e3},K=class extends o{constructor(...e){super(...e),this.st=null,this.error=null,this.busy=null,this.said={},this.tState=`The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.`,this.tQuestion=`What kind of work does this request ask for?`,this.tOptions=`debug, code_edit, plan, chat`,this.tModel=`auto`,this.tOut=null,this.tErr=null,this.asking=!1,this.ed=new D(this,`deciders`),this.poller=null}connectedCallback(){super.connectedCallback(),this.ed.load(),this.poller=T(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{this.st=await E.deciders(),this.error=null}catch(e){let t=a(e);this.error=`${t.endpoint}: ${t.message}`}}async act(e,t){if(t!==`unload`||await k(`Unload ${e.name}?`,[`Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).`,`Loading it again goes through the residency planner.`],`Unload`)){this.busy=e.name,this.said={...this.said,[e.name]:``};try{let n=await(t===`load`?E.deciderLoad(e.name):E.deciderUnload(e.name));this.said={...this.said,[e.name]:t===`load`?`load started (${n.state??`loading`})`:`unloaded`}}catch(n){let r=n instanceof i&&n.status===409&&t===`load`?`refused by the residency planner: ${n.message}`:a(n).message;this.said={...this.said,[e.name]:r}}finally{this.busy=null,await this.load()}}}setLoad(e,t){let n=this.ed.draft;if(!n)return;let r=n.models.findIndex(t=>t.name===e),i=r>=0?n.models.map((e,n)=>n===r?{...e,load:t}:e):[...n.models,{name:e,load:t}];this.ed.set({...n,models:i})}async ask(){let e=pn(this.tQuestion,this.tOptions);if(typeof e==`string`){this.tErr=e;return}this.asking=!0,this.tErr=null;try{this.tOut=await E.deciderTest(this.tState,{q:e},this.tModel)}catch(e){this.tOut=null,this.tErr=a(e).message}finally{this.asking=!1}}render(){let t=this.st;if(this.error&&!t)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${this.error}</div></div>`;if(!t)return u(6);if(!t.enabled)return n(`Decision models`,e`<div class="state" data-deciders-off>
          <div class="state-title">${c(`warn`)} This engine serves no decision model</div>
          <p class="muted small">${t.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port (behind the admin password once one is set), and this screen loads, unloads and tests them.</p>
        </div>`,{id:`deciders-off`});let r=t.models.filter(e=>e.state===`loaded`);return e`
      <div class="stats" data-decider-tiles>
        ${d({label:`loaded`,value:`${r.length} / ${t.models.length}`,sub:r.length?r.map(e=>e.name).join(`, `):`none: questions fail open`,warn:!r.length})}
        ${d({label:`on the device`,value:Je(r.reduce((e,t)=>e+t.device_bytes,0)),sub:`${t.device??`no device`} · own stream and memory pool`})}
        ${d({label:`a load now`,value:cn(t.planner),sub:sn(t.planner),warn:ln(t.planner)})}
        ${d({label:`when they run`,value:`idle`,sub:`only while no request holds the engine (stage A)`})}
      </div>
      ${this.renderModels(t)}
      ${this.renderMemory(t)}
      ${this.renderApi(t)}
      ${this.renderQuestions(t)}
      ${this.renderTest(t)}
    `}renderModels(t){let i=this.ed.draft,a=e=>i?.models.find(t=>t.name===e)?.load??`on_demand`;return n(`Models`,e`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${t.models.map(t=>{let n=on(t.refusal),i=this.said[t.name],o=t.state===`loaded`||t.state===`failed`||t.state===`warming`;return e`<tr data-decider-row=${t.name}>
              <td><strong>${t.name}</strong><div class="muted small">${t.family} · ${t.contract}${t.aliases.length?e` · ${t.aliases.join(`, `)}`:r}</div>${t.role?e`<div class="muted small" data-decider-role=${t.name} title="the role contract this model signs (SRV-133); the registry checks it before it serves the model">${nn(t)}</div>`:r}</td>
              <td><span class="badge ${rn(t.state)}" data-decider-state=${t.state}>${tn[t.state]}</span>${t.warm===!1?e` <span class="muted small">not warmed</span>`:r}</td>
              <td><select aria-label=${`load policy of ${t.name}`} data-decider-load=${t.name} @change=${e=>this.setLoad(t.name,e.target.value)}>
                ${hn.map(n=>e`<option value=${n} ?selected=${a(t.name)===n}>${n.replace(`_`,` `)}</option>`)}
              </select></td>
              <td class="num">${fn(t)}</td>
              <td class="small">${t.precision??e`<span class="muted">—</span>`}<div class="muted small mono" title=${t.weight_hash??``}>${t.weight_hash?t.weight_hash.slice(0,12):``}</div></td>
              <td class="num num-col">${h(t.questions)}</td>
              <td class="num num-col">${t.stats?.forward_ms_p50==null?`—`:`${g(t.stats.forward_ms_p50)} / ${g(t.stats.forward_ms_p95)}`}</td>
              <td class="num num-col ${t.stats?.errors?`err-ink`:``}">${t.stats?h(t.stats.errors+t.stats.deadline_missed):`—`}</td>
              <td>${o?e`<button class="btn btn-sm" ?disabled=${this.busy===t.name} @click=${()=>this.act(t,`unload`)} data-unload=${t.name}>Unload</button>`:e`<button class="btn btn-sm btn-primary" ?disabled=${this.busy===t.name||t.state===`loading`||t.state===`unloading`} @click=${()=>this.act(t,`load`)} data-load=${t.name}>Load</button>`}</td>
            </tr>
            ${n||t.error||i?e`<tr class="is-warn" data-decider-note=${t.name}><td colspan="9" class="small">
                  ${n?e`<span class="warn-ink" data-refusal=${t.name}>${c(`warn`)} ${n}</span> `:r}
                  ${t.error?e`<span class="err-ink">${t.error}</span> `:r}
                  ${i?e`<span role="status">${i}</span>`:r}
                </td></tr>`:r}`})}</tbody>
      </table></div>
      ${t.unserved?.length?e`<p class="muted small">In the document but not served by this engine: ${t.unserved.join(`, `)} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>`:r}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,{id:`deciders-models`,sub:`the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen`})}renderMemory(t){let i=t.planner,a=t.last_refusal;return n(`Memory`,e`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${w(i?.mem_available_gib)}</div></div>
          ${i?.engine_growth_gib==null?r:e`<div class="stat"><div class="stat-label">engine growth to its peak</div><div class="stat-value num">${w(i.engine_growth_gib)}</div></div>`}
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${w(i?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${w(i?.floor_gib)} + ${w(i?.margin_gib)}</div></div>
          <div class="stat ${ln(i)?`is-warn`:``}" data-decider-load-now><div class="stat-label">a load now</div><div class="stat-value num">${cn(i)}</div><div class="stat-sub">${sn(i)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${a?e`Last engine load not made: ${un(a)}, ${on(a)}`:e`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable, less what the engine may still take up to its declared peak, minus the claim stays at or above floor + margin, never during a hold, never beside another load. A model left out this way is not a fault: the room is kept for requests, because one long request is admitted by the engine's own peak, not by what is free, and the box must keep its floor at that peak. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,{id:`deciders-memory`})}renderApi(t){let r=this.ed.draft,i={...gn,...r?.api??{}},a=t.api?.stats,o=(e,t)=>r&&this.ed.set(S(r,[`api`,e],t)),s=(t,n,r=1)=>e`<label class="field"><span>${n}</span><input type="number" step=${r} .value=${String(i[t])} data-api-field=${t} @change=${e=>o(t,r<1?Number(e.target.value):Math.round(Number(e.target.value)))} /></label>`;return n(`Laya for scripts`,r?e`<div data-decider-api>
            <div class="toolbar" style="margin:0"><span class="muted small">route</span>${j([`off`,`on`],i.mode,e=>o(`mode`,e),{small:!0,label:`Laya route mode`})}<code class="small">POST /v1/laya/ask</code></div>
            <div class="form-grid">
              ${s(`rate_per_min`,`texts a minute (every caller)`)}
              ${s(`max_texts`,`texts in one call`)}
              ${s(`deadline_ms`,`deadline (ms) while a request runs`)}
              ${s(`threshold`,`sure at or above`,.05)}
              ${s(`drive_rate_per_s`,`texts a second (a playtest session)`)}
              ${s(`drive_sessions`,`playtest sessions at once`)}
            </div>
            <p class="muted small" data-api-stats>${a?`${h(a.calls)} calls, ${h(a.texts)} texts, ${h(a.errors)} errors (${h(a.rate_limited)} over the rate) since the start${a.last_ms==null?``:`; the last took ${g(a.last_ms)}`}${a.drive_sessions?`; ${h(a.drive_sessions)} playtest sessions live`:``}`:`no calls yet`}</p>
            <p class="muted small">The model's own scripts ask Laya typed questions about their evidence (a log tail, an output, a state dump) through the provider, open like the chat API: a choice, a scale or a yes-no question; Laya reads about 1,100 characters of each text, its head and its tail. Every call is a journal row (<code>laya_api</code>). A playtest (Laya plays the build: <code>.tandem/laya_drive.py</code>) sends a <code>session</code>: its own rate a second, one <code>laya_drive</code> row when it ends.</p>
          </div>`:u(2),{id:`deciders-api`,sub:`the route the model’s scripts call: on or off, its rate and its limits`})}renderQuestions(t){let i=this.ed,a=i.draft,o=t.models.map(e=>e.name),s=a?Object.entries(a.questions):[];return e`${n(`Questions`,a?e`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${s.map(([n,s])=>{let c=t.questions.find(e=>e.id===n);return e`<tr data-question=${n}>
                  <td class="mono">${n}</td>
                  <td><select aria-label=${`model for ${n}`} data-question-model=${n} @change=${e=>i.set(S(a,[`questions`,n],e.target.value))}>
                    ${[`auto`,...o].map(t=>e`<option value=${t} ?selected=${t===s}>${t===`auto`?`auto (the role’s model; the router picks)`:t}</option>`)}
                    ${s!==`auto`&&!o.includes(s)?e`<option value=${s} selected>${s} (not served)</option>`:r}
                  </select></td>
                  <td class="small">${dn(c?.picks??{})}</td>
                </tr>`})}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); <code>judge</code> is the test judge's questions on a run's evidence (the <code>tester</code> role), <code>laya_api</code> the model's own scripts through <code>/v1/laya/ask</code>; the session monitor's Tier 1 questions join this list when they land.</p>`:u(3),{id:`deciders-questions`,sub:`which model answers each question the conductor asks`})}
      ${A(i)}
      ${M(i,`the deciders document`)}`}renderTest(t){let i=this.tOut,a=i?.answers.q;return n(`Test box`,e`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${e=>this.tState=e.target.value} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${e=>this.tQuestion=e.target.value} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${e=>this.tOptions=e.target.value} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${e=>this.tModel=e.target.value} data-test-model>
            ${[`auto`,...t.models.map(e=>e.name)].map(t=>e`<option value=${t} ?selected=${t===this.tModel}>${t}</option>`)}
          </select></label>
        </div>
        <div class="savebar is-idle"><button class="btn btn-primary" ?disabled=${this.asking} @click=${()=>this.ask()} data-test-ask>${c(`send`)} ${this.asking?`Asking…`:`Ask`}</button>
          ${this.tErr?e`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>`:r}</div>
        ${i&&a?e`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${a.type===`choice`?a.choice:a.type===`noul`?(a.noul??0).toFixed(4):(a.score??0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${a.confidence.toFixed(2)} · ${i.model} · ${g(i.latency_ms)} · ${i.path} · ${h(i.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${mn(a).map(([t,n])=>e`<div class="prob-row" data-prob=${t}><span class="mono small">${t}</span><span class="conf-bar"><span style="width:${Math.round(n*100)}%"></span></span><span class="num small">${n.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${i.routing.reason}${i.routing.requested?` (wanted ${i.routing.requested})`:``}</p>
            </div>`:r}`,{id:`deciders-test`,sub:`one choice question, as /v1/systemone answers it (it runs in the idle window like any other)`})}};s([t()],K.prototype,`st`,void 0),s([t()],K.prototype,`error`,void 0),s([t()],K.prototype,`busy`,void 0),s([t()],K.prototype,`said`,void 0),s([t()],K.prototype,`tState`,void 0),s([t()],K.prototype,`tQuestion`,void 0),s([t()],K.prototype,`tOptions`,void 0),s([t()],K.prototype,`tModel`,void 0),s([t()],K.prototype,`tOut`,void 0),s([t()],K.prototype,`tErr`,void 0),s([t()],K.prototype,`asking`,void 0),K=s([m(`qse-cond-deciders`)],K);var q=class extends o{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=T(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await E.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let t=a(e);this.error=`${t.endpoint}: ${t.message}`}}async loadSession(e){try{let t=await E.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(t){e===this.selected&&(this.turns=[]),this.error=a(t).message}}render(){if(this.error&&!this.list)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${this.error}</div></div>`;if(!this.list)return u(6);let t=[...new Set(this.list.map(e=>e.client_kind??`other`))],i=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return e`
      ${this.selected?this.renderSession(this.selected):r}
      ${n(`Sessions, last 7 days`,i.length?e`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${i.map(t=>e`<tr class="req-row ${t.key===this.selected?`is-open`:``}" data-session=${t.key} @click=${()=>x({session:t.key})}>
                  <td class="mono small cell-wrap">${C(t.key,28)}</td><td>${t.client_kind??`—`}</td><td>${t.profile??`—`}</td>
                  <td>${O(t.class_id,this.ids)}</td><td class="num num-col">${h(t.turns)}</td>
                  <td class="num num-col">${t.notes?`${t.notes} (${h(t.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${t.acts?e`${h(t.acts)}${t.gave_ups?e` <span class="badge badge-warn">${t.gave_ups} gave up</span>`:r}`:`—`}</td>
                  <td class="num num-col" data-interventions>${h(t.interventions??0)}</td>
                  <td>${t.last_checker?e`<span class="badge ${t.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${t.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${ee(new Date(t.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:l(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:e`<label class="field field-search">${c(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${t.map(t=>e`<option value=${t}>${t}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?e`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:r}
    `}renderSession(t){let n=this.turns,i=this.live.find(e=>e.key===t);if(n==null)return u(4);let a=n.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,o=n.reduce((e,t)=>e+(t.injected?.length??0),0),s=this.events.filter(e=>De(e)).length,l=oe(n);return e`<section class="panel" style="margin-bottom:16px" data-timeline=${t}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${C(t,44)} ${i?e`<span class="badge badge-on">live</span>`:r}</h2>
        <p class="panel-sub">${n.length} turns · ${o} hidden note${o===1?``:`s`} · ${s} monitor act${s===1?``:`s`}${i?` · class now ${i.class_id} · ${h(i.note_tokens)} note tokens replayed a turn`:``}${n.length?` · since ${y(new Date(n[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${E.exportUrl(t,`md`)} download=${`session-${t.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${c(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${E.exportUrl(t,`json`)} download=${`session-${t.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${c(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>x({session:null})}>${c(`x`)}</button></div></header>
      ${this.axis(ue(n,this.events,this.ids,e=>v(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${n.map(t=>{let n=Math.max(18,Math.round((t.outcome?.total_ms??1e3)/a*900)),i=(t.injected?.length??0)>0,o=!!t.shadow,s=(t.events??[]).filter(e=>e.decision!==`PASS`),c=s.some(e=>De(e))?`act`:s.length?`shadow`:null;return e`<button class="strip-turn ${this.open?.id===t.id?`is-sel`:``}" role="listitem" style="width:${n}px; --cls: var(--series-${he(t.class_id,this.ids)})" title=${`${t.class_id} · ${t.outcome?.finish_reason??`running`} · ${me(t)}`} @click=${()=>this.open=t} data-turn=${t.id}>
            ${i?e`<span class="mark"></span>`:o?e`<span class="mark is-shadow"></span>`:r}
            ${c?e`<span class="mark is-monitor ${c===`shadow`?`is-shadow`:``}" data-monitor-mark=${c}></span>`:r}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${l.series.length?e`<qse-ribbon .series=${l.series} .labels=${l.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:r}
      <div class="feed" style="margin-top:10px">${n.slice().reverse().map(e=>st(e,this.ids,e=>this.open=e,me(e)+((e.events??[]).some(e=>De(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?e`<div style="margin-top:12px"><qse-cond-journal .session=${t}></qse-cond-journal></div>`:r}
    </section>`}axis(t){let n=e=>`${(e*100).toFixed(3)}%`;return e`<figure class="axis" data-axis aria-label="the session on one time axis">
      <div class="axis-lane axis-acts" aria-label=${`${t.acts.length} monitor decisions`}>
        ${t.acts.map(t=>e`<span class="axis-mark ${t.shadow?`is-shadow`:`is-act`}" style="left:${n(t.x)}" title=${t.title} data-axis-act=${t.shadow?`shadow`:`act`}></span>`)}
        ${t.notes.map(t=>e`<span class="axis-mark is-note ${t.shadow?`is-shadow`:``}" style="left:${n(t.x)}" title=${t.title} data-axis-note></span>`)}
      </div>
      <div class="axis-lane axis-turns" aria-label=${`${t.turns.length} turns`}>
        ${t.turns.map(t=>e`<span class="axis-turn ${t.compaction?`is-compaction`:``}" style="left:${n(t.x0)};width:${n(Math.max(0,t.x1-t.x0))};--cls:var(--series-${t.color})" title=${t.title} data-axis-turn></span>`)}
      </div>
      <div class="axis-lane axis-events" aria-label=${`${t.compactions.length} compactions, ${t.checkers.length} checker results`}>
        ${t.compactions.map(t=>e`<span class="axis-mark is-compaction" style="left:${n(t.x)}" title=${t.title} data-axis-compaction></span>`)}
        ${t.checkers.map(t=>e`<span class="axis-mark ${t.pass?`is-pass`:`is-fail`}" style="left:${n(t.x)}" title=${t.title} data-axis-checker=${t.pass?`pass`:`fail`}></span>`)}
      </div>
      <div class="axis-ticks" aria-hidden="true">${t.ticks.map((r,i)=>e`<span class="num ${i===0?`is-first`:i===t.ticks.length-1?`is-last`:``}" style="left:${n(r.x)}">${r.label.slice(0,5)}</span>`)}</div>
      <figcaption class="muted small">One time axis, ${t.ticks[0]?.label??``} to ${t.ticks[t.ticks.length-1]?.label??``}: ${t.turns.length} turns (the gaps are the client's time), ${t.acts.filter(e=>!e.shadow).length} monitor actions ◆ above (◇ shadow, ${t.acts.filter(e=>e.shadow).length}), ${t.notes.length} router note${t.notes.length===1?``:`s`}, ${t.compactions.length} compaction${t.compactions.length===1?``:`s`} ▲ and ${t.checkers.length} checker result${t.checkers.length===1?``:`s`} below (● pass, ○ fail; the gate numbers are in each one's title).</figcaption>
    </figure>`}tier1(){let t=Ve(this.events);return t?e`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${t.agree} of ${t.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${t.decisions} shadow decisions)</span></p>`:r}};s([p({attribute:!1})],q.prototype,`ids`,void 0),s([p({attribute:!1})],q.prototype,`selected`,void 0),s([t()],q.prototype,`list`,void 0),s([t()],q.prototype,`live`,void 0),s([t()],q.prototype,`turns`,void 0),s([t()],q.prototype,`events`,void 0),s([t()],q.prototype,`error`,void 0),s([t()],q.prototype,`query`,void 0),s([t()],q.prototype,`client`,void 0),s([t()],q.prototype,`open`,void 0),q=s([m(`qse-cond-sessions`)],q);var _n={scope:`tools`,retention_days:30,clip_chars:4e3,loop_chars:200},vn=/^### (user|assistant|summary|runtime|tool result \((.*)\)|tool call \((.*)\)|reasoning \((.*)\))$/;function yn(e){let t=e[1];return e[2]===void 0?e[3]===void 0?e[4]===void 0?{kind:t===`user`||t===`assistant`||t===`summary`||t===`runtime`?t:`other`,title:t,name:null,clipped:!1}:{kind:`reasoning`,title:t,name:null,clipped:!1}:{kind:`tool_call`,title:t,name:e[3].trim()||null,clipped:!1}:{kind:`tool_result`,title:t,name:e[2].split(`,`)[0].trim()||null,clipped:/clipped to/.test(e[2])}}function bn(e){let t=e.replace(/\n+$/,``).split(`
`),n=t[0]??``,r=null,i=[],a=null,o=()=>{a&&i.push({kind:a.kind,title:a.title,name:a.name,clipped:a.clipped,body:a.lines.join(`
`).replace(/^\n+|\n+$/g,``)})};for(let e of t.slice(1)){let t=e.match(vn);t?(o(),a={...yn(t),lines:[]}):a?a.lines.push(e):e.startsWith(`compaction: `)&&(r=e)}o();let s=i.find(e=>e.kind===`runtime`);return{head:n,compaction:r,sections:i.filter(e=>e.kind!==`runtime`),runtime:s?xn(s.body):[]}}function xn(e){return e.split(`
`).filter(e=>e.trim()).map(e=>{let t=[`hidden`,`replayed`,`acts`,`checker`,`snapshot`,`finish`,`redactions`].find(t=>e.startsWith(`${t}: `))??`other`,n=e.match(/ · journal ([A-Za-z0-9_-]{4,40})$/),r=t===`checker`?e.match(/ (PASS|FAIL)(?: <-|$)/):null;return{kind:t,text:e,journal:n?n[1]:null,verdict:r?r[1]:null}})}function Sn(e){if(!e.counts)return`counts unknown (no meta.json)`;let t=e.counts,n=[`${t.blocks} turn${t.blocks===1?``:`s`}`];return t.untranscribed&&n.push(`${t.untranscribed} without tools (counted only)`),n.push(`${t.acts} act${t.acts===1?``:`s`}`),t.snapshots&&n.push(`${t.snapshots} snapshot${t.snapshots===1?``:`s`}`),t.compactions&&n.push(`${t.compactions} compaction${t.compactions===1?``:`s`}`),t.redactions&&n.push(`${t.redactions} redacted`),e.checkers?.last&&n.push(e.checkers.last),n.join(` · `)}function Cn(e){return{..._n,...e?.memory?.transcripts??{}}}function wn(e,t){return e?e.replace(/^[ch]:/,``):t}var Tn={mode:`on`,budget_gb:50,ttl_days:7,admit_after:2,min_free_gb:50},En={"opencode-head":`opencode head`,head:`session head`,package:`package`,memory:`memory file`,prefix:`repeated prefix`};function Dn(e){return{...Tn,...e?.memory?.disk_cache??{}}}function On(e){let t=[],n=e=>Number.isFinite(e)&&Math.round(e)===e;return e.budget_gb>=1&&e.budget_gb<=4e3||t.push(`the budget is 1 to 4,000 GB`),n(e.ttl_days)&&e.ttl_days>=1&&e.ttl_days<=90||t.push(`an entry is kept 1 to 90 days`),n(e.admit_after)&&e.admit_after>=1&&e.admit_after<=10||t.push(`a prefix is written after 1 to 10 prefills`),e.min_free_gb>=50&&e.min_free_gb<=4e3||t.push(`the NVMe keeps at least 50 GB free`),t}function kn(e,t){if(t.free_bytes==null)return{usable_gb:null,short:!1};let n=((t.bytes??0)+Math.max(0,t.free_bytes-Math.max(50,e.min_free_gb)*1e9))/1e9;return{usable_gb:n,short:n<e.budget_gb}}function An(e){let t=e.lookups??0;return t?`${Math.round((e.hit_rate??0)*100)} % of ${t.toLocaleString(`en-US`)} prefill${t===1?``:`s`} it could have shortened`:`no prefill the disk could have shortened yet`}function jn(e){let t=e=>typeof e==`number`?e.toLocaleString(`en-US`):String(e),n=[];return e.tokens!=null&&n.push(`${t(e.tokens)} tokens`),e.kind===`hit`&&e.read!=null&&n.push(`${t(e.read)} read`,`${t(e.ms)} ms`),e.kind===`write`&&(n.push(`${t(e.chunks)} chunk${e.chunks===1?``:`s`}`,`${t(e.anchors)} anchor${e.anchors===1?``:`s`}`),e.dropped&&typeof e.dropped==`object`&&n.push(`dropped ${Object.entries(e.dropped).map(([e,t])=>`${t} ${e}`).join(`, `)}`)),e.why!=null&&n.push(String(e.why)),e.kind===`clear`&&n.push(`${t(e.entries)} entries`),e.error!=null&&n.push(String(e.error)),[e.kind,...n].join(` · `)}var Mn=e=>y(new Date(e*1e3).toISOString()),Nn=e=>Object.entries(e??{}).filter(([,e])=>e).map(([e,t])=>`${h(t)} ${e}`).join(`, `)||`none`,J=class extends o{constructor(...e){super(...e),this.disk=null,this.error=null,this.busy=!1,this.cleared=null,this.settings=new D(this,`settings`)}connectedCallback(){super.connectedCallback(),this.load(),this.settings.load(),this.settings.onSaved=()=>void this.load()}async load(){try{this.disk=await E.memoryDisk(),this.error=null}catch(e){this.error=a(e).message}}async clear(){let e=this.disk;if(e&&await k(`Clear the disk cache?`,[`Every entry goes: ${h(e.entries??0)} prefixes in ${h(e.chunks??0)} chunk files, ${b(e.bytes??0)}, and the counts that decide what is written next.`,`Nothing a request needs is lost: the next prefill of each prefix runs cold, as without the tier, and writes it again.`],`Clear`)){this.busy=!0;try{let e=await E.memoryDiskClear();this.cleared=`cleared ${h(e.entries)} entries, ${b(e.bytes)}`,this.disk=e.disk}catch(e){this.error=a(e).message}finally{this.busy=!1}}}render(){if(this.error&&!this.disk)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${this.error}</div></div>`;if(!this.disk)return u(5);let t=this.disk;return e`${t.running?this.renderRunning(t):this.renderOff(t)}${this.renderSettings(t)}`}renderOff(t){return n(`Disk cache`,l(`The disk tier does not run on this server`,e`It starts with the engine when <code>QSE_DISK_CACHE</code> in serve.env names its directory (the flag is <code class="mono">${t.flag}</code> now)${t.free_bytes==null?r:e`; the NVMe has <span class="num">${b(t.free_bytes,0)}</span> free`}. Its settings below apply once it runs.`),{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM`})}renderRunning(t){let i=t.bytes??0,a=t.budget??0,o=a?i/a:0,s=t.free_bytes??null,u=s!=null&&t.floor!=null&&s<t.floor+1e10,f=e`<div class="stats disk-stats" data-disk-tiles>
      ${d({label:`on disk`,value:b(i,1),sub:e`of <span class="num">${b(a,0)}</span> · ${h(t.entries??0)} prefix${t.entries===1?``:`es`}, ${h(t.chunks??0)} chunk files`,warn:o>.95})}
      ${d({label:`NVMe free`,value:s==null?`—`:b(s,0),sub:e`never under <span class="num">${b(t.floor??5e10,0)}</span>`,warn:u})}
      ${d({label:`hit rate`,value:t.hit_rate==null?`—`:_(t.hit_rate,0),sub:An(t),na:t.hit_rate==null})}
      ${d({label:`restored`,value:h(t.tokens_restored??0),unit:`tokens`,sub:t.last_restore_ms==null?`no restore yet`:e`last <span class="num">${h(Math.round(t.last_restore_ms))}</span> ms${t.last_read_gbs==null?r:e` at <span class="num">${t.last_read_gbs.toFixed(1)}</span> GB/s`}`})}
    </div>`,p=e`<div class="meter-block disk-meter">
      <div class="meter-head"><span>budget</span><span class="num">${b(i,1)} of ${b(a,0)} · ${_(o,0)}</span></div>
      <div class="meter" role="img" aria-label=${`disk cache ${_(o,0)} of its budget`}><span class="meter-fill ${o>.95?`is-warn`:``}" style="width:${Math.min(100,o*100).toFixed(1)}%"></span></div>
    </div>`,m=t.top??[],g=e`${f}${p}
      <p class="small muted disk-line" data-disk-counters>
        ${t.on?e`<span class="badge badge-on">on</span>`:e`<span class="badge">off (settings)</span>`}
        written: <span class="num">${h(t.writes_chunk??0)}</span> chunks, <span class="num">${h(t.writes_anchor??0)}</span> anchors, <span class="num">${b(t.bytes_written??0,1)}</span> ·
        evicted: ${Nn(t.evicted)} · dropped: ${Nn(t.dropped)}${t.failed_restores?e` · <span class="warn-ink">${h(t.failed_restores)} failed restore${t.failed_restores===1?``:`s`}</span>`:r}
        · staging buffer <span class="num">${b(t.staging_bytes||t.staging_planned||0,2)}</span>${t.staging_bytes?``:` (when first used)`} · <span class="mono">${t.root??``}</span>
      </p>
      ${m.length?this.renderEntries(m):l(`Nothing on disk yet`,`A prefix is written after ${t.settings.admit_after===1?`its first prefill`:`${t.settings.admit_after} prefills`} of at least ${h(t.min_tokens??4096)} tokens, and the first long request of an opencode session is written where its user message starts.`)}
      ${this.renderEvents(t)}`;return n(`Disk cache`,g,{id:`cond-disk`,sub:`prompt-prefix state on the NVMe, under the caches in RAM; a hit is read only past the rows already in place`,tools:e`${this.cleared?e`<span class="small muted" data-disk-cleared>${this.cleared}</span>`:r}
        <button class="btn btn-sm disk-clear" ?disabled=${this.busy||!(t.entries||t.chunks)} @click=${()=>this.clear()} data-disk-clear>${c(`x`)} Clear disk cache</button>
        <button class="btn btn-sm" @click=${()=>this.load()} aria-label="reload the disk cache">${c(`refresh`)}</button>`})}renderEntries(t){let n=t=>e`<span class="badge disk-kind disk-kind-${t.kind}">${En[t.kind]}</span>${t.label?e` <span class="mono small">${t.label}</span>`:r}${t.complete?r:e` <span class="badge badge-warn" title="a chunk below it is not on disk: it is not read, and goes first">incomplete</span>`}`,i=t=>t.preview?e`<div class="small muted disk-preview" data-disk-preview>${t.preview}</div>`:r;return e`<h3 class="disk-sub">Most-hit prefixes</h3>
      <div class="table-scroll mem-table"><table class="table" data-disk-entries>
        <thead><tr><th class="num-col">tokens</th><th>what</th><th class="num-col">hits</th><th>last used</th><th class="num-col">a hit reads</th></tr></thead>
        <tbody>${t.map(t=>e`<tr data-disk-row=${`${t.kind}-${t.tokens}`}><td class="num num-col">${h(t.tokens)}</td><td class="cell-wrap">${n(t)}${i(t)}</td>
            <td class="num num-col">${h(t.hits)}</td><td class="num">${Mn(t.used)}</td><td class="num num-col">${b(t.reads_bytes,1)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards" role="list">${t.map(t=>e`<div class="mem-card disk-card" role="listitem" data-disk-card=${`${t.kind}-${t.tokens}`}>
          <div class="mem-card-head"><span class="num">${h(t.tokens)} tokens</span><span class="num small">${h(t.hits)} hit${t.hits===1?``:`s`}</span></div>
          <div>${n(t)}</div>${i(t)}
          <div class="small muted">last used ${Mn(t.used)} · a hit reads ${b(t.reads_bytes,1)}</div>
        </div>`)}</div>`}renderEvents(t){let n=(t.events??[]).slice(-10).reverse();return n.length?e`<details class="sec" style="margin-top:12px" data-disk-events>
      <summary>Last events <span class="sec-sub">what it read, wrote, dropped and evicted (the journal keeps them, kind memory)</span></summary>
      <ul class="disk-events">${n.map(t=>e`<li class="small"><span class="num muted">${Mn(t.ts)}</span> <span class="disk-ev disk-ev-${t.kind}">${jn(t)}</span></li>`)}</ul>
    </details>`:r}renderSettings(t){let n=this.settings;if(!n.draft)return n.loadError?e`<p class="err-ink small">${n.loadError}</p>`:r;let i=Dn(n.draft),a=(e,t)=>n.set(S(n.draft,[`memory`,`disk_cache`],{...i,[e]:t})),o=(t,n,r,o,s=1)=>e`<label class="field"><span>${n}</span><input type="number" inputmode="decimal" min=${r} max=${o} step=${s} .value=${String(i[t])} data-disk-setting=${t} @input=${e=>a(t,Number(e.target.value))} /></label>`,s=kn(i,t),l=On(i);return e`<details class="sec" style="margin-top:16px" data-sec="disk" open>
      <summary>Disk cache settings <span class="sec-sub">on or off, its size on the NVMe, how long it keeps an entry, what it writes</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-disk-mode><span class="muted small">disk cache</span>${j([`on`,`off`],i.mode,e=>a(`mode`,e),{small:!0,label:`disk cache mode`})}<span class="small">${i.mode===`on`?`prefixes are read from and written to the disk`:`nothing is read or written; the files stay`}</span></div>
        <div class="form-grid">
          ${o(`budget_gb`,`cache budget (GB)`,1,4e3)}
          ${o(`ttl_days`,`drop an entry unused for (days)`,1,90)}
          ${o(`admit_after`,`write a prefix after it was prefilled (times)`,1,10)}
          ${o(`min_free_gb`,`keep free on the NVMe (GB, at least 50)`,50,4e3)}
        </div>
        <p class="small" data-disk-free>
          ${t.free_bytes==null?r:e`The NVMe has <span class="num">${b(t.free_bytes,0)}</span> free${t.bytes?e`, and the cache holds <span class="num">${b(t.bytes,1)}</span>`:r}. `}
          ${s.short?e`<span class="warn-ink" data-disk-short>${c(`warn`)} At most <span class="num">${s.usable_gb.toFixed(0)} GB</span> of the ${i.budget_gb} GB fit above the ${Math.max(50,i.min_free_gb)} GB it keeps free; the tier stops there.</span>`:r}
        </p>
        <p class="muted small">Least valuable first when it is full: hits, times the tokens an entry saves per byte it holds, halved for every day unused. The first long request of an opencode session is written whatever the count, at the lines below where its user message starts. Applied from the next request; the free space it keeps is never under ${50} GB, whatever is set here.</p>
        ${l.length?e`<ul class="errors" role="alert" data-disk-errors>${l.map(t=>e`<li>${t}</li>`)}</ul>`:r}
        ${A(n)}
        ${M(n,`the settings`)}
      </div>
    </details>`}};s([t()],J.prototype,`disk`,void 0),s([t()],J.prototype,`error`,void 0),s([t()],J.prototype,`busy`,void 0),s([t()],J.prototype,`cleared`,void 0),J=s([m(`qse-cond-disk`)],J);var Y=e=>e?y(e):`—`,Pn=()=>new Date().toISOString().slice(0,10),X=class extends o{constructor(...e){super(...e),this.selected=null,this.node=null,this.user=`owner`,this.list=null,this.one=null,this.error=null,this.oneError=null,this.filter={...Gt},this.query=``,this.hits=null,this.injected=null,this.writes=null,this.busy=null,this.editing=null,this.raw=!1,this.diffAt=null,this.confirmUser=``,this.flash=null,this.settings=new D(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.reload(),this.settings.load()}willUpdate(e){e.has(`node`)&&(this.filter={...this.filter,node:this.node});let t=this.selected?`${this.user}#${this.selected}`:null;t!==this.loadedFor&&(this.loadedFor=t,this.one=null,this.oneError=null,this.editing=null,this.diffAt=null,this.selected&&this.loadOne(this.selected))}async reload(){try{let[e,t,n]=await Promise.all([E.notes(this.user),E.notesInjected(this.user,30),E.notesWrites(30)]);this.list=e,this.injected=t.rows,this.writes=n.rows,this.error=null}catch(e){this.error=a(e).message}}async loadOne(e){try{let t=await E.note(e,this.user);e===this.selected&&(this.one=t)}catch(t){e===this.selected&&(this.oneError=a(t).message)}}async act(e,t,n){this.busy=e,this.flash=null;try{await t(),n&&(this.flash=n),await this.reload(),this.selected&&await this.loadOne(this.selected)}catch(e){this.oneError=a(e).message}finally{this.busy=null}}async search(){let e=this.query.trim();if(!e){this.hits=null;return}try{this.hits=(await E.notesSearch(e,this.user,10)).hits}catch(e){this.error=a(e).message}}async download(){try{let e=await E.notesExport(this.user),t=document.createElement(`a`);t.href=URL.createObjectURL(e),t.download=`memory-${this.user}.zip`,t.click(),setTimeout(()=>URL.revokeObjectURL(t.href),1e3)}catch(e){this.error=a(e).message}}open(e){x({note:e})}render(){if(this.error&&!this.list)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${this.error}</div></div>`;if(!this.list)return u(6);let t=this.list;return e`
      ${this.renderTiles(t)}
      ${this.flash?e`<p class="ok-ink small" role="status" data-notes-flash>${c(`check`)} ${this.flash}</p>`:r}
      ${this.selected?this.renderOne():r}
      ${this.renderSearch()}
      ${this.renderBrowser(t)}
      ${this.renderInjected()}
      ${this.renderRuns()}
      ${this.renderWrites()}
      ${this.renderSettings()}
      ${this.renderDanger(t)}
    `}renderTiles(t){let n=new Map;for(let e of t.notes)n.set(e.kind,(n.get(e.kind)??0)+1);let r=t.notes.filter(e=>e.status===`provisional`).length,i=Pn(),a=(this.injected??[]).filter(e=>new Date(e.ts_ms).toISOString().slice(0,10)===i).length,o=(this.writes??[]).filter(e=>new Date(e.ts_ms).toISOString().slice(0,10)===i&&/^refused/.test(Yt(e))).length,s=[...n.entries()].sort((e,t)=>t[1]-e[1]).map(([e,t])=>`${t} ${e.replace(`_`,` `)}`).join(` · `);return e`<div class="stats notes-stats" data-notes-tiles>
      ${d({label:`live notes`,value:h(t.notes.length),sub:s||`none yet`})}
      ${d({label:`provisional`,value:h(r),sub:e`<span class="num">${h(t.notes.length-r)}</span> ratified by you`,warn:r>0&&r===t.notes.length&&t.notes.length>5})}
      ${d({label:`injected today`,value:h(a),unit:`requests`,sub:e`<span class="num">${h(t.memory.index_notes+t.memory.body_notes)}</span> notes since the start`})}
      ${d({label:`refused today`,value:h(o),sub:e`<span class="num">${h(t.memory.taken)}</span> blocks written, <span class="num">${h(t.memory.failed)}</span> refused since the start`})}
    </div>`}renderSearch(){return n(`Search`,this.hits===null?e`<p class="muted small">The runtime's own search (BM25 over title, tags and body, every scope of ${this.user}): the scores are the ones that decide what goes into a request.</p>`:this.hits.length?e`<ol class="notes-hits" data-notes-hits>${this.hits.map(t=>e`<li><button class="link-btn" @click=${()=>this.open(t.id)} data-hit=${t.id}>${t.title}</button>
                <span class="badge">${t.kind}</span> <span class="badge">${t.scope}</span> <span class="num small" data-score>${t.score.toFixed(2)}</span>
                ${t.status===`provisional`?e`<span class="badge badge-shadow">provisional</span>`:r}</li>`)}</ol>`:l(`No match`,`No note shares a word with the query.`),{tools:e`<form class="toolbar notes-search" style="margin:0" @submit=${e=>{e.preventDefault(),this.search()}}><label class="field field-search">${c(`search`)}<input placeholder="words, as the runtime would search them" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search the notes" data-notes-query /></label>
          <button class="btn btn-sm" type="submit" data-notes-search>Search</button>${this.hits?e`<button class="btn btn-sm btn-ghost" type="button" @click=${()=>(this.hits=null,this.query=``)}>Clear</button>`:r}</form>`,id:`notes-search`})}renderBrowser(t){let i=Kt(t.notes,this.filter),a=e=>x({nnode:e,note:null}),o=e`<nav class="notes-tree" aria-label="memory scopes" data-notes-tree>
      <button class="notes-node ${this.node?``:`is-on`}" @click=${()=>a(null)} data-node="all"><span>All notes</span><span class="num muted">${t.notes.length}</span></button>
      ${t.tree.map(t=>e`<button class="notes-node ${this.node===t.node?`is-on`:``}" @click=${()=>a(t.node)} data-node=${t.node} title=${t.label??t.node}>
          <span class="notes-node-label">${t.provisional?e`<span class="dot dot-shadow" aria-label="has provisional notes"></span>`:r}${qt(t)}</span>
          <span class="num muted">${t.count}</span></button>`)}
    </nav>`,s=e`<label class="field notes-picker"><span>scope</span><select @change=${e=>a(e.target.value||null)} aria-label="memory scope" data-notes-picker>
        <option value="" ?selected=${!this.node}>All notes (${t.notes.length})</option>
        ${t.tree.map(t=>e`<option value=${t.node} ?selected=${this.node===t.node}>${qt(t)} (${t.count})</option>`)}
      </select></label>`,u=this.filter,d=e=>this.filter={...this.filter,...e},f=e`<div class="toolbar notes-filters" style="margin:0 0 8px">
      <select aria-label="kind" @change=${e=>d({kind:e.target.value})} data-filter="kind"><option value="">every kind</option>${Wt.map(t=>e`<option value=${t} ?selected=${u.kind===t}>${t.replace(`_`,` `)}</option>`)}</select>
      <select aria-label="status" @change=${e=>d({status:e.target.value})} data-filter="status"><option value="">every status</option><option value="provisional">provisional</option><option value="ratified">ratified</option></select>
      <select aria-label="writer" @change=${e=>d({writer:e.target.value})} data-filter="writer"><option value="">every writer</option>${Object.entries(U).map(([t,n])=>e`<option value=${t}>${n}</option>`)}</select>
      <label class="switch small"><input type="checkbox" .checked=${u.stale} @change=${e=>d({stale:e.target.checked})} data-filter="stale" /> stale</label>
      <label class="field field-search">${c(`search`)}<input placeholder="filter by title or tag" .value=${u.q} @input=${e=>d({q:e.target.value})} aria-label="filter the list" /></label>
    </div>`,p=t=>e`<span class="badge ${t.status===`ratified`?`badge-ok`:`badge-shadow`}">${t.status}</span>${t.pinned?e` <span class="badge badge-on">pinned</span>`:r}${t.stale?e` <span class="badge badge-warn">stale</span>`:r}`,m=e`<div class="table-scroll notes-table"><table class="table" data-notes-list>
        <thead><tr><th>title</th><th>kind</th><th>status</th><th>updated</th><th>writer</th><th class="num-col">uses 7d</th></tr></thead>
        <tbody>${i.map(t=>e`<tr class="req-row ${t.id===this.selected?`is-open`:``}" data-note-row=${t.id} @click=${()=>this.open(t.id)}>
            <td class="cell-wrap">${t.title}</td><td><span class="badge" title=${Ut[t.kind]??``}>${t.kind.replace(`_`,` `)}</span></td><td>${p(t)}</td>
            <td class="num small">${Y(t.updated)}</td><td class="small">${U[t.writer]??t.writer}</td><td class="num num-col">${h(t.uses_7d??0)}</td></tr>`)}</tbody></table></div>
      <div class="mem-cards notes-cards" role="list">${i.map(t=>e`<div class="mem-card ${t.id===this.selected?`is-open`:``}" role="listitem" data-note-card=${t.id} @click=${()=>this.open(t.id)}>
          <div class="mem-card-head"><span class="small">${t.title}</span></div>
          <div class="small">${t.kind.replace(`_`,` `)} · ${p(t)}</div>
          <div class="small muted">${Y(t.updated)} · ${U[t.writer]??t.writer} · ${h(t.uses_7d??0)} uses in 7 days</div>
        </div>`)}</div>`,g=t.notes.length?l(`Nothing here`,`No note in this scope matches the filters.`):l(`No notes yet`,t.store.refused?`The memory directory is inside a git work tree (${t.store.refused}), so nothing is written.`:`The model writes a note when you ask it to remember something, correct it, or decide something; with the profile's memory switch on or in shadow.`);return n(`Notes`,e`<div class="notes-browser">${o}<div class="notes-main">${s}${f}${i.length?m:g}</div></div>`,{sub:e`<span data-notes-root>${h(t.notes.length)} live · ${h(t.archived.length)} archived${t.duplicates.length?e` · <span class="warn-ink">${t.duplicates.length} possible duplicate${t.duplicates.length===1?``:`s`}</span>`:r} · <span class="mono">${t.store.root}/${this.user}</span></span>`,tools:e`${t.users.length>1?e`<select aria-label="user" @change=${e=>(this.user=e.target.value,x({note:null,nnode:null}),void this.reload())} data-notes-user>${t.users.map(t=>e`<option ?selected=${t===this.user}>${t}</option>`)}</select>`:r}
          <button class="btn btn-sm" @click=${()=>this.download()} data-notes-export>${c(`down`)} Export</button>
          <button class="btn btn-sm" @click=${()=>this.reload()} aria-label="reload the notes">${c(`refresh`)}</button>`,id:`notes`})}renderOne(){if(this.oneError&&!this.one)return e`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${c(`warn`)} ${this.oneError}</div></div>`;let t=this.one;if(!t)return u(4);let n=t.note,i=!!n.archived,a=n.status===`ratified`&&(n.kind===`rule`||n.kind===`ruling`),o=e=>this.busy===e,s=n.id,l=i?e`<button class="btn btn-sm btn-danger" ?disabled=${o(`delete`)} @click=${()=>this.del(n)} data-note-act="delete">${c(`trash`)} Delete now</button>`:e`<button class="btn btn-sm" @click=${()=>this.startEdit(n)} data-note-act="edit">${c(`edit`)} Edit</button>
          ${n.status===`provisional`?e`<button class="btn btn-sm" ?disabled=${o(`ratify`)} @click=${()=>this.act(`ratify`,()=>E.noteStatus(s,{status:`ratified`},this.user),`ratified`)} data-note-act="ratify">${c(`check`)} Ratify</button>`:r}
          ${n.status===`provisional`?e`<button class="btn btn-sm" ?disabled=${o(`reject`)} @click=${()=>this.act(`reject`,()=>E.noteArchive(s,`rejected in the dashboard`,this.user),`rejected and archived`)} data-note-act="reject">Reject</button>`:r}
          ${a?e`<button class="btn btn-sm" ?disabled=${o(`pin`)} @click=${()=>this.act(`pin`,()=>E.noteStatus(s,{pinned:!n.pinned},this.user),n.pinned?`unpinned`:`pinned to every request`)} data-note-act="pin">${n.pinned?`Unpin`:`Pin to every request`}</button>`:r}
          <button class="btn btn-sm" ?disabled=${o(`archive`)} @click=${()=>this.archive(n)} data-note-act="archive">Archive</button>
          <button class="btn btn-sm btn-danger" ?disabled=${o(`delete`)} @click=${()=>this.del(n)} data-note-act="delete">${c(`trash`)} Delete</button>`,d=e`<dl class="notes-meta" data-note-meta>
      <dt>kind</dt><dd><span class="badge">${n.kind.replace(`_`,` `)}</span> <span class="muted small">${Ut[n.kind]??``}</span></dd>
      <dt>scope</dt><dd>${n.scope}${n.project?e` · <span class="mono small">${n.project}</span>`:r}${n.session?e` · <a class="mono small" href=${`#/conductor?screen=memory&memory=${encodeURIComponent(n.session)}`}>${n.session}</a>`:r}</dd>
      <dt>status</dt><dd>${n.status}${n.pinned?` · pinned`:``}${n.stale?e` · <span class="warn-ink">stale since ${n.verify_by}</span>`:r}${i?e` · <span class="warn-ink">archived ${Y(n.archived)}: ${n.reason}</span>`:r}</dd>
      <dt>writer</dt><dd>${U[n.writer]??n.writer}</dd>
      <dt>tags</dt><dd>${n.tags.length?n.tags.map(t=>e`<span class="badge">${t}</span> `):e`<span class="muted">—</span>`}</dd>
      ${n.verify_by?e`<dt>verify by</dt><dd class="num">${n.verify_by}</dd>`:r}
      <dt>created</dt><dd class="num">${Y(n.created)}</dd><dt>updated</dt><dd class="num">${Y(n.updated)}</dd>
      <dt>sources</dt><dd>${n.sources.length?n.sources.map(t=>e`<span class="mono small">${t}</span> `):e`<span class="muted">—</span>`}</dd>
      <dt>used</dt><dd data-note-uses>${Qt(t.uses,Pn())}</dd>
    </dl>`,f=this.editing?this.renderEditor():e`<div class="notes-body">
        <div class="toolbar" style="margin:0 0 6px">${j([`rendered`,`raw`],this.raw?`raw`:`rendered`,e=>this.raw=e===`raw`,{small:!0,label:`body view`})}</div>
        ${this.raw?e`<pre class="mem-text" data-note-raw>${n.body}</pre>`:e`<div class="notes-rendered" data-note-body>${Fn(n.body)}</div>`}
      </div>`;return e`<section class="panel notes-one" style="margin-bottom:16px" data-note=${s}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${n.title}</h2><p class="panel-sub mono">${s}</p></div>
        <div class="panel-tools notes-actions">${l}<button class="btn btn-ghost btn-icon" aria-label="Close the note" @click=${()=>x({note:null})}>${c(`x`)}</button></div></header>
      ${this.oneError?e`<p class="err-ink small" role="alert">${c(`warn`)} ${this.oneError}</p>`:r}
      <div class="notes-file">${d}${f}</div>
      ${this.renderHistory(t)}
    </section>`}startEdit(e){this.editing={title:e.title,body:e.body,kind:e.kind,tags:e.tags.join(`, `),verify_by:e.verify_by??``}}renderEditor(){let t=this.editing,n=e=>this.editing={...t,...e},r=t.body.trim()?t.body.trim().split(/\s+/).length:0,i=t.title.length>120||r>400||!t.body.trim();return e`<div class="notes-editor" data-note-editor>
      <label class="field"><span>title <span class="muted num">${t.title.length}/120</span></span><input .value=${t.title} @input=${e=>n({title:e.target.value})} data-edit="title" /></label>
      <div class="form-grid">
        <label class="field"><span>kind</span><select @change=${e=>n({kind:e.target.value})} data-edit="kind">${Wt.map(n=>e`<option value=${n} ?selected=${n===t.kind}>${n.replace(`_`,` `)}</option>`)}</select></label>
        <label class="field"><span>tags (comma)</span><input .value=${t.tags} @input=${e=>n({tags:e.target.value})} data-edit="tags" /></label>
        <label class="field"><span>verify by</span><input type="date" .value=${t.verify_by} @input=${e=>n({verify_by:e.target.value})} data-edit="verify_by" /></label>
      </div>
      <label class="field"><span>body <span class="muted num">${r}/400 words</span></span><textarea rows="8" .value=${t.body} @input=${e=>n({body:e.target.value})} data-edit="body"></textarea></label>
      <div class="toolbar" style="margin:0"><button class="btn btn-primary btn-sm" ?disabled=${i||this.busy===`edit`} @click=${()=>this.act(`edit`,async()=>{await E.noteEdit(this.selected,{title:t.title,body:t.body,kind:t.kind,tags:t.tags.split(`,`).map(e=>e.trim()).filter(Boolean),verify_by:t.verify_by||null},this.user),this.editing=null},`saved as a new revision, ratified`)} data-edit-save>Save as a revision</button><button class="btn btn-sm btn-ghost" @click=${()=>this.editing=null}>Cancel</button>
        ${i?e`<span class="warn-ink small">a title of at most 120 characters, a body of 1 to 400 words</span>`:e`<span class="muted small">Your edit ratifies the note; the old text stays in its history.</span>`}</div>
    </div>`}renderHistory(t){let n=t.revisions,i=t.note,a=[...n.map(e=>({label:`revision ${e.n} · ${Y(e.updated)} · ${U[e.writer]??e.writer}`,title:e.title,body:e.body})),{label:`now · ${Y(i.updated)}`,title:i.title,body:i.body}],o=this.diffAt,s=r;if(o!==null&&o+1<a.length){let t=a[o],n=a[o+1],r=Jt(`# ${t.title}\n${t.body}`,`# ${n.title}\n${n.body}`);s=e`<pre class="notes-diff" data-note-diff>${r.map(t=>e`<span class="diff-${t.op===`+`?`add`:t.op===`-`?`del`:`eq`}">${t.op===`=`?` `:t.op} ${t.text}\n</span>`)}</pre>`}return e`<details class="sec" ?open=${n.length>0} data-note-history>
      <summary>History <span class="sec-sub">${i.changelog.length} change${i.changelog.length===1?``:`s`}, ${n.length} earlier text${n.length===1?``:`s`}</span></summary>
      <div class="sec-body">
        <ul class="notes-changelog">${i.changelog.map(t=>e`<li class="small">${t}</li>`)}</ul>
        ${n.length?e`<ol class="notes-revs">${a.slice(0,-1).map((t,n)=>e`<li><button class="link-btn ${o===n?`is-on`:``}" @click=${()=>this.diffAt=o===n?null:n} data-rev=${n}>${t.label}</button> <span class="muted small">to the next</span></li>`)}</ol>${s}`:e`<p class="muted small">No earlier text: this is the note as it was written.</p>`}
      </div>
    </details>`}async archive(e){let t=window.prompt(`Why is this note archived? (kept with it for 30 days)`,`no longer true`);t!==null&&await this.act(`archive`,()=>E.noteArchive(e.id,t||`archived in the dashboard`,this.user),`archived`)}async del(t){await k(`Delete this note?`,[e`<strong>${t.title}</strong>`,`The file, its history and any archived copy go at once. This cannot be undone.`],`Delete`)&&(await this.act(`delete`,()=>E.noteDelete(t.id,this.user),`deleted`),x({note:null}))}renderInjected(){let t=this.injected??[];return e`<details class="sec" style="margin-top:16px" data-notes-injected>
      <summary>Injected <span class="sec-sub">the last requests the notes went into (the index once a session, the matching notes on a new message)</span></summary>
      <div class="sec-body">${t.length?e`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>session</th><th>what</th><th>notes</th></tr></thead><tbody>${t.map(t=>e`<tr><td class="num small">${y(new Date(t.ts_ms).toISOString())}</td>
              <td class="mono small cell-wrap">${t.session_key?e`<a href=${`#/conductor?screen=memory&memory=${encodeURIComponent(t.session_key)}`}>${t.session_key}</a>`:`—`}</td>
              <td class="small">${t.index?`index`:`bodies`}</td>
              <td class="cell-wrap">${t.ids.map(t=>e`<button class="link-btn mono small" @click=${()=>this.open(t)}>${t}</button> `)}</td></tr>`)}</tbody></table></div>`:e`<p class="muted small">No note went into a request yet.</p>`}</div>
    </details>`}renderRuns(){let t=(this.writes??[]).filter(e=>e.task===`writer`).map(Zt);return e`<details class="sec" data-notes-runs ?open=${t.length>0}>
      <summary>Writer runs <span class="sec-sub">the model's hidden turns that read a session and write its notes: after a compaction, after a give-up, at the session's end</span></summary>
      <div class="sec-body">${t.length?e`<div class="table-scroll"><table class="table"><thead><tr><th>when</th><th>trigger</th><th>Laya's gate</th><th>outcome</th><th class="num-col">seconds</th><th class="num-col">tokens</th><th>notes</th><th>refused</th></tr></thead><tbody>${t.map(t=>e`<tr data-run=${t.trigger}><td class="num small">${y(new Date(t.when).toISOString())}</td><td>${t.trigger.replace(`_`,` `)}</td><td class="small">${t.gate}</td><td class="small">${t.outcome}</td>
              <td class="num num-col">${t.seconds??`—`}</td><td class="num num-col">${t.tokens??`—`}</td><td class="small">${t.ops}</td><td class="small warn-ink">${t.refused.join(`, `)}</td></tr>`)}</tbody></table></div>`:e`<p class="muted small">No run yet. The writer is ${this.list?.settings.writer?.mode??`off`} (settings below).</p>`}</div>
    </details>`}renderWrites(){let t=(this.writes??[]).filter(e=>e.task!==`writer`);return e`<details class="sec" data-notes-writes>
      <summary>Writes <span class="sec-sub">every block the model wrote, the script calls, your edits, the nightly job, and what was refused and why</span></summary>
      <div class="sec-body">${t.length?e`<ul class="notes-writes">${t.map(t=>e`<li class="small ${/^refused/.test(Yt(t))?`warn-ink`:``}"><span class="num muted">${y(new Date(t.ts_ms).toISOString())}</span> ${Yt(t)}</li>`)}</ul>`:e`<p class="muted small">Nothing written yet.</p>`}</div>
    </details>`}renderSettings(){let t=this.settings;if(!t.draft)return t.loadError?e`<p class="err-ink small">${t.loadError}</p>`:r;let n=Xt(t.draft),i=(e,r)=>t.set(S(t.draft,[`memory`,`notes`],{...n,[e]:r})),a=(t,r,a,o)=>e`<label class="field"><span>${r}</span><input type="number" min=${a} max=${o} .value=${String(n[t])} data-notes-setting=${t} @input=${e=>i(t,Math.round(Number(e.target.value)))} /></label>`;return e`<details class="sec" data-sec="notes-settings">
      <summary>Notes settings <span class="sec-sub">what goes into a request, the duplicate rule, Laya's remember question</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          ${a(`index_tokens`,`the index, once a session (tokens)`,0,8e3)}
          ${a(`body_tokens`,`the matching notes on a new message (tokens)`,0,4e3)}
          ${a(`pinned_tokens`,`pinned rules (tokens)`,0,2e3)}
          ${a(`bodies`,`matching notes at most`,0,10)}
          ${a(`min_words`,`words a note shares with the message at least`,1,10)}
          <label class="field"><span>a second note this similar is a revision</span><input type="number" min="0" max="1" step="0.05" .value=${String(n.dedupe_threshold)} data-notes-setting="dedupe_threshold" @input=${e=>i(`dedupe_threshold`,Number(e.target.value))} /></label>
        </div>
        <div class="toolbar" style="margin:0" data-notes-intent><span class="muted small">Laya asks "remember, forget or recall?"</span>${j([`off`,`shadow`,`on`],n.intent,e=>i(`intent`,e),{small:!0,label:`intent question`})}</div>
        <label class="switch small"><input type="checkbox" .checked=${n.provisional} @change=${e=>i(`provisional`,e.target.checked)} /> provisional notes go into requests too (marked)</label>
        ${this.renderWriterSettings(n,i)}
        <p class="muted small">A profile's <code>memory</code> switch turns the notes on (off, shadow: the model may write and nothing is read back, on). Every note passes the secret-pattern refusal; a rule about you needs your own message in the turn; the model never ratifies, pins or deletes.</p>
        ${A(t)}
        ${M(t,`the settings`)}
      </div>
    </details>`}renderWriterSettings(t,n){let r=t.writer,i=(e,t)=>n(`writer`,{...r,[e]:t}),a=(t,n,a,o)=>e`<label class="field"><span>${n}</span><input type="number" min=${a} max=${o} .value=${String(r[t])} data-writer-setting=${t} @input=${e=>i(t,Number(e.target.value))} /></label>`;return e`<h4 class="small" style="margin:12px 0 4px">The background writer</h4>
      <div class="toolbar" style="margin:0" data-writer-mode><span class="muted small">writer</span>${j([`off`,`shadow`,`on`],r.mode,e=>i(`mode`,e),{small:!0,label:`writer mode`})}
        <span class="muted small">after a compaction</span>${j([`off`,`idle`,`turn`],r.after_compaction,e=>i(`after_compaction`,e),{small:!0,label:`after a compaction`})}</div>
      <div class="form-grid">
        ${a(`idle_s`,`a session's end: seconds without a request`,10,86400)}
        ${a(`min_new_tokens`,`new tokens before the next pass`,0,1e6)}
        ${a(`max_tokens`,`the turn writes at most (tokens)`,64,4096)}
      </div>
      <label class="switch small"><input type="checkbox" .checked=${r.gate} @change=${e=>i(`gate`,e.target.checked)} /> Laya decides first whether the session has anything worth keeping (about 20 ms; a no skips the turn)</label>
      <p class="muted small">shadow: the turn runs and its notes are counted, nothing is stored. <code>turn</code> after a compaction writes before the next request gets the engine (the next turn waits up to ${r.turn_deadline_s} s); <code>idle</code> waits for an idle moment, which opencode rarely leaves right after a compaction.</p>`}renderDanger(t){let n=this.confirmUser===this.user;return e`<details class="sec" data-notes-danger>
      <summary>Delete ${this.user}'s memory <span class="sec-sub">every note, revision and archive of this user, at once</span></summary>
      <div class="sec-body">
        <p class="small">Type <code>${this.user}</code> to delete ${h(t.notes.length+t.archived.length)} notes. Export them first if you want a copy.</p>
        <div class="toolbar" style="margin:0"><label class="field"><span>user</span><input .value=${this.confirmUser} @input=${e=>this.confirmUser=e.target.value} data-notes-confirm aria-label="type the user's name" /></label>
          <button class="btn btn-sm btn-danger" ?disabled=${!n||this.busy===`user`} @click=${()=>this.act(`user`,()=>E.notesDeleteUser(this.user,this.confirmUser),`${this.user}'s memory deleted`).then(()=>this.confirmUser=``)} data-notes-delete-user>${c(`trash`)} Delete everything</button></div>
      </div>
    </details>`}};s([p({attribute:!1})],X.prototype,`selected`,void 0),s([p({attribute:!1})],X.prototype,`node`,void 0),s([t()],X.prototype,`user`,void 0),s([t()],X.prototype,`list`,void 0),s([t()],X.prototype,`one`,void 0),s([t()],X.prototype,`error`,void 0),s([t()],X.prototype,`oneError`,void 0),s([t()],X.prototype,`filter`,void 0),s([t()],X.prototype,`query`,void 0),s([t()],X.prototype,`hits`,void 0),s([t()],X.prototype,`injected`,void 0),s([t()],X.prototype,`writes`,void 0),s([t()],X.prototype,`busy`,void 0),s([t()],X.prototype,`editing`,void 0),s([t()],X.prototype,`raw`,void 0),s([t()],X.prototype,`diffAt`,void 0),s([t()],X.prototype,`confirmUser`,void 0),s([t()],X.prototype,`flash`,void 0),X=s([m(`qse-cond-notes`)],X);function Fn(t){let n=[],r=(t??``).split(`
`),i=0,a=t=>t.split(/(`[^`]+`)/).map(t=>t.startsWith("`")&&t.endsWith("`")&&t.length>1?e`<code>${t.slice(1,-1)}</code>`:t);for(;i<r.length;){let t=r[i];if(t.startsWith("```")){let t=[];for(i++;i<r.length&&!r[i].startsWith("```");)t.push(r[i++]);i++,n.push(e`<pre class="mem-text">${t.join(`
`)}</pre>`)}else if(/^\s*[-*] /.test(t)){let t=[];for(;i<r.length&&/^\s*[-*] /.test(r[i]);)t.push(r[i++].replace(/^\s*[-*] /,``));n.push(e`<ul>${t.map(t=>e`<li>${a(t)}</li>`)}</ul>`)}else if(!t.trim())i++;else{let t=[];for(;i<r.length&&r[i].trim()&&!r[i].startsWith("```")&&!/^\s*[-*] /.test(r[i]);)t.push(r[i++]);n.push(e`<p>${a(t.join(` `))}</p>`)}}return e`${n}`}var In={tools:`requests with tools in full (and compactions); chats without tools as counters only`,all:`every request in full`,none:`nothing`},Z=class extends o{constructor(...e){super(...e),this.ids=[],this.selected=null,this.page=null,this.tab=null,this.note=null,this.nnode=null,this.list=null,this.one=null,this.error=null,this.oneError=null,this.query=``,this.busy=null,this.row=null,this.ev=null,this.rowError=null,this.settings=new D(this,`settings`),this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.tab!==`disk`&&this.tab!==`notes`&&(this.loadList(),this.settings.load())}updated(e){let t=e=>e===`disk`||e===`notes`;t(e.get(`tab`))&&!t(this.tab)&&!this.list&&!this.error&&(this.loadList(),this.settings.draft||this.settings.load())}willUpdate(){let e=this.selected?`${this.selected}#${this.page??``}`:null;e!==this.loadedFor&&(this.loadedFor=e,this.one=null,this.oneError=null,this.selected&&this.loadOne(this.selected,this.page))}async loadList(){try{this.list=await E.memorySessions(),this.error=null}catch(e){this.error=a(e).message}}async loadOne(e,t){try{let n=await E.memorySession(e,t);e===this.selected&&(this.one=n)}catch(t){e===this.selected&&(this.oneError=a(t).message)}}async keep(e,t){if(e.session){this.busy=e.dir;try{await E.memoryKeep(e.session,t),await this.loadList(),this.one&&this.one.session.session===e.session&&(this.one={...this.one,session:{...this.one.session,keep:t}})}catch(e){this.error=a(e).message}finally{this.busy=null}}}async download(e,t){try{let n=await E.memoryExport(e,t),r=document.createElement(`a`);r.href=URL.createObjectURL(n),r.download=`transcript-${e.replace(/[^a-z0-9]+/gi,`-`)}.${t}`,r.click(),setTimeout(()=>URL.revokeObjectURL(r.href),1e3)}catch(e){this.oneError=a(e).message}}async openRow(e){this.rowError=null;try{let t=(await E.decision(e)).decision;t.kind===`monitor`?this.ev=t:this.row=t}catch(t){this.rowError=`journal row ${e}: ${a(t).message} (the journal keeps 30 days)`}}render(){let t=this.tab===`disk`,n=this.tab===`disk`||this.tab===`notes`?this.tab:null,i=(t,i)=>e`<a class="cond-tab ${t===n?`is-on`:``}" href="#" aria-current=${t===n?`page`:r} data-mem-tab=${t??`sessions`} @click=${e=>{e.preventDefault(),x({mtab:t,memory:null,mpage:null,note:null,nnode:null})}}>${i}</a>`,a=e`<nav class="mem-tabs" aria-label="memory tabs">${i(null,`Sessions`)}${i(`notes`,`Notes`)}${i(`disk`,`Disk cache`)}</nav>`;return t?e`${a}<qse-cond-disk></qse-cond-disk>`:n===`notes`?e`${a}<qse-cond-notes .selected=${this.note} .node=${this.nnode}></qse-cond-notes>`:e`
      ${a}
      ${this.selected?this.renderOne(this.selected):r}
      ${this.renderList()}
      ${this.renderSettings()}
      ${this.row?e`<qse-cond-decision .row=${this.row} .ids=${this.ids} .close=${()=>this.row=null}></qse-cond-decision>`:r}
      ${this.ev?e`<qse-cond-event .ev=${this.ev} .close=${()=>this.ev=null}></qse-cond-event>`:r}
    `}renderList(){if(this.error&&!this.list)return e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${this.error}</div></div>`;if(!this.list)return u(5);let t=this.list.writer,i=this.query.toLowerCase(),a=this.list.sessions.filter(e=>!i||`${e.session??e.dir} ${e.client??``} ${e.project??``}`.toLowerCase().includes(i)),o=e=>e.last_ms?y(new Date(e.last_ms).toISOString()):`—`,s=n=>e`<label class="switch" title="kept sessions are exempt from the ${t.settings.retention_days}-day prune"><input type="checkbox" .checked=${n.keep} ?disabled=${!n.session||this.busy===n.dir} data-keep=${n.dir} @click=${e=>e.stopPropagation()} @change=${e=>this.keep(n,e.target.checked)} aria-label=${`keep ${wn(n.session,n.dir)}`} /> keep</label>`,d=t=>t.checkers?.last?e`<span class="badge ${/ PASS$/.test(t.checkers.last)?`badge-ok`:`badge-warn`}">${t.checkers.last}</span>${t.checkers.PASS||t.checkers.FAIL?e` <span class="muted small num">${t.checkers.PASS??0} pass · ${t.checkers.FAIL??0} fail</span>`:r}`:e`<span class="muted">—</span>`,f=e=>e.session&&x({memory:e.session,mpage:null});return n(`Session transcripts`,a.length?e`<div class="table-scroll mem-table"><table class="table" data-memory-sessions>
              <thead><tr><th>last request</th><th>session</th><th>client</th><th>profile</th><th>project</th><th class="num-col">turns</th><th class="num-col">acts</th><th>checker</th><th class="num-col">size</th><th>keep</th></tr></thead>
              <tbody>${a.map(t=>e`<tr class="req-row ${t.session&&t.session===this.selected?`is-open`:``}" data-memory-row=${t.dir} @click=${()=>f(t)}>
                  <td class="num">${o(t)}</td><td class="mono small cell-wrap">${wn(t.session,t.dir)}</td><td>${t.client??`—`}</td><td>${t.profile??`—`}</td>
                  <td class="small cell-wrap">${t.project??`—`}</td>
                  <td class="num num-col">${t.counts?h(t.counts.blocks):`?`}</td><td class="num num-col">${t.counts?h(t.counts.acts):`?`}</td>
                  <td>${d(t)}</td><td class="num num-col">${b(t.bytes??0)}</td><td>${s(t)}</td></tr>`)}</tbody></table></div>
            <div class="mem-cards" role="list">${a.map(t=>e`<div class="mem-card ${t.session&&t.session===this.selected?`is-open`:``}" role="listitem" data-memory-card=${t.dir} @click=${()=>f(t)}>
                <div class="mem-card-head"><span class="mono small">${wn(t.session,t.dir)}</span>${s(t)}</div>
                <div class="small">${o(t)} · ${t.client??`—`} · ${t.profile??`—`}</div>
                <div class="small muted">${Sn(t)}</div>
                ${t.project?e`<div class="small muted cell-wrap">${t.project}</div>`:r}
                <div>${d(t)}</div>
              </div>`)}</div>`:l(`No transcripts yet`,t.refused?`The memory directory ${t.root} is inside a git work tree, so nothing is written there.`:`TandemLLM requests with tools are written to ${t.root} when their stream closes; chats without tools are counted only.`),{sub:e`<span data-writer>${h(t.blocks)} block${t.blocks===1?``:`s`} written since the start · ${h(t.redactions)} secret${t.redactions===1?``:`s`} redacted · ${h(t.failed)} failed · ${h(t.dropped)} dropped · ${b(t.bytes)} · <span class="mono">${t.root}</span></span>`,tools:e`<label class="field field-search">${c(`search`)}<input placeholder="search a session, client or project" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search transcripts" /></label>
          <button class="btn btn-sm" @click=${()=>this.loadList()} aria-label="reload the list">${c(`refresh`)}</button>`,id:`cond-memory`})}renderOne(t){if(this.oneError)return e`<div class="state state-error" role="alert" style="margin-bottom:16px"><div class="state-title">${c(`warn`)} ${this.oneError}</div></div>`;let n=this.one;if(!n)return u(4);let i=n.session,a=n.pages??1,o=n.page??1,s=a>1?e`<div class="toolbar mem-pager" data-pager><button class="btn btn-sm" ?disabled=${o<=1} @click=${()=>x({mpage:String(o-1)})}>Older</button>
          <span class="small num">page ${o} of ${a} · ${h(n.total??n.blocks.length)} blocks, ${n.per_page} a page</span>
          <button class="btn btn-sm" ?disabled=${o>=a} @click=${()=>x({mpage:String(o+1)})}>Newer</button></div>`:r;return e`<section class="panel" style="margin-bottom:16px" data-transcript=${t}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title mono">${wn(i.session,i.dir)}${i.keep?e` <span class="badge badge-on">kept</span>`:r}</h2>
        <p class="panel-sub">${i.client??`—`} · ${i.profile??`—`}${i.project?` · ${i.project}`:``} · ${Sn(i)}${i.first_ms?` · since ${y(new Date(i.first_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <button class="btn btn-sm" @click=${()=>this.download(t,`md`)} data-mem-export="md">${c(`down`)} Markdown</button>
          <button class="btn btn-sm" @click=${()=>this.download(t,`json`)} data-mem-export="json">${c(`down`)} JSON</button>
          <a class="btn btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(t)}`} data-mem-timeline>Timeline</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the transcript" @click=${()=>x({memory:null,mpage:null})}>${c(`x`)}</button></div></header>
      ${this.rowError?e`<p class="warn-ink small" role="alert">${c(`warn`)} ${this.rowError}</p>`:r}
      ${s}
      ${n.blocks.length?e`<div class="mem-blocks">${n.blocks.map(e=>this.renderBlock(e))}</div>`:l(`No blocks`,`This session was counted but not written: its requests carried no tools.`)}
      ${s}
    </section>`}renderBlock(t){let n=bn(t.text),i=n.runtime.filter(e=>e.kind===`hidden`).length,a=n.runtime.find(e=>e.kind===`acts`),o=e`<span class="mem-turn num">turn ${t.turn??`?`}</span>
      <span class="muted small num">${t.time??``}</span>
      ${t.class_id?O(t.class_id,this.ids):r}
      ${t.effort?e`<span class="badge">${t.effort}</span>`:r}
      ${t.reasoning_tokens?e`<span class="muted small num">${h(t.reasoning_tokens)} reasoning tokens</span>`:r}
      ${i?e`<span class="badge badge-shadow" data-hidden-count=${i}>${i} hidden</span>`:r}
      ${a?e`<span class="badge badge-warn">${a.text.replace(/^acts: /,`acts `)}</span>`:r}`;return t.compaction?e`<details class="mem-block is-compaction" data-block=${t.turn??``} data-compaction>
        <summary><span class="mem-block-head">${o}<span class="badge">compaction</span><span class="small">${n.compaction?.replace(/^compaction: /,``)??``}</span></span></summary>
        ${n.sections.map(e=>this.renderSection(e))}${this.renderRuntime(n.runtime)}
      </details>`:e`<article class="mem-block" data-block=${t.turn??``}>
      <div class="mem-block-head">${o}</div>
      ${n.sections.map(e=>this.renderSection(e))}
      ${this.renderRuntime(n.runtime)}
    </article>`}renderSection(t){let n=t.body.length>600||t.body.split(`
`).length>12,i=e`<pre class="mem-text">${t.body}</pre>`;return t.kind===`tool_result`||t.kind===`tool_call`&&n||t.kind===`reasoning`?e`<details class="mem-sec mem-${t.kind}" ?open=${!n&&t.kind!==`reasoning`} data-sec-kind=${t.kind}>
        <summary class="small">${t.title}${t.clipped?e` <span class="badge badge-warn">clipped</span>`:r}</summary>${i}</details>`:e`<div class="mem-sec mem-${t.kind}" data-sec-kind=${t.kind}><div class="small muted">${t.title}</div>${i}</div>`}renderRuntime(t){return t.length?e`<ul class="mem-runtime" aria-label="what the runtime did">${t.map(t=>e`<li class="mem-rt mem-rt-${t.kind} ${t.verdict?`is-${t.verdict.toLowerCase()}`:``}" data-rt=${t.kind}>
        ${t.kind===`hidden`?c(`info`):r}<span>${t.text.replace(/ · journal [A-Za-z0-9_-]+$/,``)}</span>
        ${t.journal?e`<button class="btn btn-ghost btn-sm" @click=${()=>this.openRow(t.journal)} data-journal=${t.journal} aria-label=${`open journal row ${t.journal}`}>journal row</button>`:r}
      </li>`)}</ul>`:r}renderSettings(){let t=this.settings;if(!t.draft)return t.loadError?e`<p class="err-ink small">${t.loadError}</p>`:r;let n=Cn(t.draft),i=(e,r)=>t.set(S(t.draft,[`memory`,`transcripts`],{...n,[e]:r})),a=(t,r,a,o)=>e`<label class="field"><span>${r}</span><input type="number" min=${a} max=${o} .value=${String(n[t])} data-mem-setting=${t} @input=${e=>i(t,Math.round(Number(e.target.value)))} /></label>`;return e`<details class="sec" style="margin-top:16px" data-sec="transcripts">
      <summary>Transcript settings <span class="sec-sub">what the engine writes to the box, how long it keeps it, where tool output is clipped</span></summary>
      <div class="sec-body">
        <div class="toolbar" style="margin:0" data-mem-scope><span class="muted small">write</span>${j([`tools`,`all`,`none`],n.scope,e=>i(`scope`,e),{small:!0,label:`transcript scope`})}<span class="small">${In[n.scope]}</span></div>
        <div class="form-grid">
          ${a(`retention_days`,`keep a session for (days; kept sessions are exempt)`,1,400)}
          ${a(`clip_chars`,`clip a tool result or a call at (characters)`,200,1e5)}
          ${a(`loop_chars`,`a loop the D1 guard cut keeps its first and last (characters)`,0,4e3)}
        </div>
        <p class="muted small">Every block passes the secret-pattern refusal before it is written: a key, token or password becomes <code>[redacted:kind]</code> and the journal counts it, never the value. There is no switch for it. The files are <code>0600</code> under the engine's home on the box, never in a repository, and the ledger's backup never holds them. The owner's decision 2: chats without tools are counters only, unless the scope is widened here.</p>
        ${A(t)}
        ${M(t,`the settings`)}
      </div>
    </details>`}};s([p({attribute:!1})],Z.prototype,`ids`,void 0),s([p({attribute:!1})],Z.prototype,`selected`,void 0),s([p({attribute:!1})],Z.prototype,`page`,void 0),s([p({attribute:!1})],Z.prototype,`tab`,void 0),s([p({attribute:!1})],Z.prototype,`note`,void 0),s([p({attribute:!1})],Z.prototype,`nnode`,void 0),s([t()],Z.prototype,`list`,void 0),s([t()],Z.prototype,`one`,void 0),s([t()],Z.prototype,`error`,void 0),s([t()],Z.prototype,`oneError`,void 0),s([t()],Z.prototype,`query`,void 0),s([t()],Z.prototype,`busy`,void 0),s([t()],Z.prototype,`row`,void 0),s([t()],Z.prototype,`ev`,void 0),s([t()],Z.prototype,`rowError`,void 0),Z=s([m(`qse-cond-memory`)],Z);var Q=class extends o{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([E.stats(168),E.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let t=a(e);this.error=`${t.endpoint}: ${t.message}`}}async promote(e,t,n,r,i){if(await k(`Promote ${n} to ${t===`task`?`act`:`on`} in ${e}?`,[t===`task`?`The session monitor will act on ${n} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${n} decisions instead of only journaling them: ${n===`effort`?`budgets and thinking off go to the engine`:n===`packages`?`skills and packages go into the prompt`:n===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${r.toFixed(2)} over ${i} labelled decisions. A new profile version records it.`],`Promote`))try{let a=await E.doc(`profiles`,e),o=t===`task`?S(a.doc,[`tasks`,n,`mode`],`act`):S(a.doc,[`decisions`,n],`on`),s=await E.put(`profiles`,e,o,`${n} to ${t===`task`?`act`:`on`}, precision ${r.toFixed(2)} / ${i}`,a.version);this.msg=`${e} v${s.version}: ${n} promoted`,this.onSaved()}catch(e){this.msg=a(e).message}}async loadAb(){try{this.abVersions=(await E.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await E.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await E.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=a(e).message}}render(){return this.error&&!this.stats?e`<div class="state state-error" role="alert"><div class="state-title">${c(`warn`)} ${this.error}</div></div>`:this.stats?e`
      ${this.msg?e`<div class="note note-info" role="status">${this.msg}</div>`:r}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:u(8)}renderPromotion(t){let i=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,a=(t.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return n(`Shadow precision and promotion`,e`<p class="muted small">${Pe}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${t.precision.length||a.length?e`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${t.precision.map(t=>{let n=se(t),a=i(t.profile,t.decision),o=t.backend?e`<div class="muted small mono" data-backend=${t.backend}>via ${t.backend}</div>`:r;return e`<tr data-precision-row=${`${t.profile}/${t.decision}${t.backend?`@${t.backend}`:``}`}>
                    <td>${t.profile}${o}</td><td class="mono">${t.decision}</td><td><span class="badge ${a===`on`?`badge-on`:a===`shadow`?`badge-shadow`:``}">${a}</span></td>
                    <td class="num num-col">${h(t.decisions)}</td><td class="num num-col">${h(t.labelled)}</td><td class="num num-col">${t.precision==null?`—`:t.precision.toFixed(2)}</td>
                    <td class="small ${n.allowed?``:`muted`}">${n.why}</td>
                    <td><button class="btn btn-sm ${n.allowed?`btn-primary`:``}" ?disabled=${!n.allowed||a!==`shadow`||!!t.backend} title=${t.backend?`a backend’s switches are on the Backends screen, per model`:n.allowed?``:Pe} data-promote=${`${t.profile}/${t.decision}`} @click=${()=>this.promote(t.profile??``,`decision`,t.decision,t.precision??0,t.labelled)}>Promote to on</button></td></tr>`})}
                ${a.map(t=>{let n=se(t);return e`<tr data-precision-row=${`task/${t.task}${t.backend?`@${t.backend}`:``}`}><td>${t.backend?e`<span class="muted small mono">via ${t.backend}</span>`:`—`}</td><td class="mono">${t.task}</td><td><span class="badge ${t.mode===`act`?`badge-on`:`badge-shadow`}">${t.mode}</span></td>
                    <td class="num num-col">${h(t.decisions)}</td><td class="num num-col">${h(t.labelled)}</td><td class="num num-col">${t.precision==null?`—`:t.precision.toFixed(2)}</td>
                    <td class="small ${n.allowed?``:`muted`}">${n.why}</td>
                    <td><button class="btn btn-sm ${n.allowed?`btn-primary`:``}" ?disabled=${!n.allowed||t.mode!==`shadow`||!!t.backend} data-promote=${`task/${t.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,t.task,t.precision??0,t.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:l(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${t.monitor?r:e`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let t=this.reports.find(e=>e.kind===`classifier`);if(!t)return n(`Classifier accuracy`,l(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let i=t.stages??{},a=t.set??{},o=(i[this.stage]??Object.values(i)[0])?.confusion??{},s=a.classes??Object.keys(o);return n(e`Classifier accuracy <span class="muted small num">${a.name??``} · ${a.n??`?`} prompts · ${t.created_at?y(String(t.created_at)):``}</span>`,e`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(i).map(([t,n])=>e`<tr class="req-row ${t===this.stage?`is-open`:``}" @click=${()=>this.stage=t}><td class="mono">${t}</td><td class="num num-col">${_(n.accuracy,1)}</td><td class="num num-col">${n.answered} / ${n.n}</td><td class="num num-col">${n.accuracy_answered==null?`—`:_(n.accuracy_answered,1)}</td><td class="num num-col">${g(n.p50_ms)}</td><td class="num num-col">${g(n.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${s.length&&Object.keys(o).length?e`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...s,`—`].map(t=>e`<th class="mono small">${t}</th>`)}</tr></thead>
              <tbody>${s.map(t=>e`<tr><th>${O(t,this.ids)}</th>${[...s,`—`].map(n=>{let r=o[t]?.[n]??0;return e`<td class="cm-cell ${t===n?`cm-diag`:r?`cm-miss`:``}">${r||``}</td>`})}</tr>`)}</tbody></table></div>`:e`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}
        ${ze(t).length?e`<p class="muted small" style="margin-top:12px">Digest variants (SRV-153): <strong>today</strong> the 2,000-character digest the decider cut from the front, <strong>sized</strong> each stage's own head and tail, <strong>line</strong> sized plus the context line. <strong>router</strong> is the session's class after the stickiness rule.</p>
            <div class="table-scroll"><table class="table" data-classifier-variants>
              <thead><tr><th>digest</th><th>set</th><th class="num-col">n</th><th class="num-col">decider</th><th class="num-col">cascade</th><th class="num-col">router</th><th class="num-col">decider p50</th></tr></thead>
              <tbody>${ze(t).map(t=>e`<tr><td class="mono">${t.variant}</td><td class="mono small">${t.set}</td><td class="num num-col">${t.n}</td><td class="num num-col">${t.decider==null?`—`:_(t.decider,1)}</td><td class="num num-col">${t.cascade==null?`—`:_(t.cascade,1)}</td><td class="num num-col">${t.router==null?`—`:_(t.router,1)}</td><td class="num num-col">${g(t.p50)}</td></tr>`)}</tbody></table></div>`:r}`,{id:`cond-classifier`})}renderSuites(){let t=this.reports.filter(e=>e.kind===`suite`),i=be(this.reports,this.suiteFile),a=Te(i,t),o=i?.rows??[],s=qe(this.reports,this.engine),u=Ae(this.reports,this.engine),d=Fe(o),f=i?.verdict,p=i?.comparison??[],m=i?.raw_row,g=i?.build,v=f?.indicative??fe(o),ee=(e,t=1)=>e==null?`—`:Number.isInteger(e)?h(e):e.toFixed(t);return n(e`TandemLLM against the raw engine ${i?e`<span class="muted small num">night ${i.night??`—`}${g?` · ${g.version??`?`} ${C(g.code_sha,12)}`:``}${i.profile?` · profile ${i.profile}`:``}</span>`:r}`,e`<p class="small muted" data-guard-detail=${s.status} data-speed-check=${u.state}>Speed check: ${u.text}. <span class="num">${s.line}${s.detail?e`. ${s.detail}`:r}</span></p>
        ${t.length>1?e`<label class="field suite-pick"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${t.map(n=>e`<option value=${String(n.file)} ?selected=${n===i}>${String(n.night??n.file)} · ${String(n.file)}${Te(n,t)?` · superseded`:``}</option>`)}</select></label>`:r}
        ${a?e`<div class="guard-line guard-note" data-superseded>${c(`info`)}<span><span class="badge badge-warn">superseded</span> by <span class="mono">${a.file}</span>${a.why?e`: <span class="small">${a.why}</span>`:r}. Its numbers are kept as published; read the newer report.</span></div>`:r}
        ${f?e`<div class="note" data-eval-verdict>
              <span class="badge ${f.better?`badge-ok`:f.better===!1?`badge-warn`:``}"><strong>${f.better?`Better`:f.better===!1?`Not better`:`Not shown yet`}</strong></span>${v?e` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:r} — ${f.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${m?.status??`—`}${m?.source?` (${m.source})`:``}${i?.driver?` · driver ${i.driver.mode}`:``}${i?.isolation?` · ${i.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:r}
        ${d.length?e`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">turns</th><th class="num-col">thinking tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>efforts (turns)</th><th>build · night</th></tr></thead>
              <tbody>${d.flatMap(t=>{let n=p.find(e=>e.suite===t.suite||e.suite_id===(t.raw??t.tandem)?.suite_id);return[t.raw,t.tandem].filter(Boolean).map((i,a)=>e`<tr data-suite=${t.suite} data-arm=${i.arm}>
                    <td>${a===0?t.suite:``}${a===0&&!t.cmp.ok?e` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${t.cmp.why}</div>`:r}${a===0&&n?e` <span class="badge ${n.verdict===`better`?`badge-ok`:n.verdict===`worse`||n.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${n.verdict}>${n.verdict}</span>`:r}${a===0&&(i.indicative??(i.n??0)<10)?e` <span class="badge" data-row-indicative>indicative</span>`:r}</td>
                    <td class="mono">${ae(i.arm)}${i.pristine===!1?e` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:r}</td>
                    <td class="num num-col">${i.n??`—`}</td>
                    <td class="num num-col">${i.pass_rate==null?`—`:_(i.pass_rate,0)}</td>
                    <td class="num num-col">${i.gates_passed??`—`}${i.gates_total?` / ${i.gates_total}`:``}</td>
                    <td class="num num-col">${ee(i.minutes_to_pass)}</td><td class="num num-col">${ee(i.wall_minutes)}</td>
                    <td class="num num-col">${i.tokens==null?`—`:h(Math.round(i.tokens))}</td>
                    <td class="num num-col" data-turns>${ee(i.turns)}</td><td class="num num-col" data-thinking>${i.reasoning_tokens==null?`—`:h(Math.round(i.reasoning_tokens))}</td>
                    <td class="num num-col">${ee(i.decode_tps)}</td>
                    <td class="num num-col">${i.empty_stops??`—`}</td><td class="num num-col">${i.continuations??`—`}</td>
                    <td class="num num-col">${i.interventions??`—`}</td><td class="num num-col">${i.gave_ups??`—`}</td>
                    <td class="num num-col">${i.cache_hit==null?`—`:_(i.cache_hit,0)}</td>
                    <td class="small" data-efforts>${ge(i)}</td>
                    <td class="small">${C(i.build,16)} · ${i.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts. Tokens, turns and thinking tokens are medians per run.</p>`:l(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return n(`A/B of two profile versions`,e`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(t=>e`<option value=${t.name} ?selected=${t.name===this.abProfile}>${t.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(t=>e`<label class="field"><span>${t===`abA`?`A`:`B`}</span><select @change=${e=>{this[t]=Number(e.target.value),this.runAb()}}>${this.abVersions.map(n=>e`<option value=${n.version} ?selected=${n.version===this[t]}>v${n.version} ${C(n.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?e`<div class="diff" data-ab>${this.abChanges.map(t=>e`<div class="diff-row"><span>${de(t.path)}</span><span class="diff-before">${C(t.before,80)}</span><span class="diff-after">${C(t.after,80)}</span></div>`)}</div>`:e`<p class="muted small">The two versions are the same.</p>`:r}`,{id:`cond-ab`})}};s([p({attribute:!1})],Q.prototype,`ids`,void 0),s([p({attribute:!1})],Q.prototype,`tiles`,void 0),s([p({attribute:!1})],Q.prototype,`onSaved`,void 0),s([t()],Q.prototype,`stats`,void 0),s([t()],Q.prototype,`reports`,void 0),s([t()],Q.prototype,`engine`,void 0),s([t()],Q.prototype,`error`,void 0),s([t()],Q.prototype,`msg`,void 0),s([t()],Q.prototype,`stage`,void 0),s([t()],Q.prototype,`abProfile`,void 0),s([t()],Q.prototype,`abVersions`,void 0),s([t()],Q.prototype,`abA`,void 0),s([t()],Q.prototype,`abB`,void 0),s([t()],Q.prototype,`abChanges`,void 0),s([t()],Q.prototype,`abCounts`,void 0),s([t()],Q.prototype,`suiteFile`,void 0),Q=s([m(`qse-cond-compare`)],Q);var Ln=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`deciders`,label:`Deciders`},{id:`sessions`,label:`Sessions`},{id:`memory`,label:`Memory`},{id:`compare`,label:`Compare`}],$=class extends o{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=T(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&Ln.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r]=await Promise.all([E.profiles(),this.meta?Promise.resolve(this.meta):E.meta(),E.config(),E.doc(`router`)]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.error=null}catch(e){let t=a(e);this.error=t.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${t.endpoint}: ${t.message}`}}async setMode(e){let t=this.tiles.find(e=>e.name===this.active);if(t&&!t.locked&&(e!==`on`||await k(`Turn the ${t.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let n=await E.setMode(t.name,e,`mode to ${e}`,t.version);this.reloaded=`v${n.version} · reloaded ${n.loaded_at?v(n.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=a(e).message}finally{this.switching=!1}}}render(){let t=this.tiles.find(e=>e.name===this.active),n=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return e`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${c(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>x({profile:e.target.value})}>
                ${this.tiles.map(t=>e`<option value=${t.name} ?selected=${t.name===this.active}>${t.name}${t.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${t?j([`raw`,`shadow`,`on`],t.mode,e=>void this.setMode(e),{disabled:!!t.locked||this.switching,label:`mode of ${t.name}`}):r}
            ${this.reloaded?e`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:r}
          </div>
        </div>
        ${this.switchError?e`<div class="guard-line" role="alert">${c(`warn`)} ${this.switchError}</div>`:r}
        ${this.error?e`<div class="guard-line" role="alert">${c(`warn`)} ${this.error}</div>`:r}
        ${n.map(([t,n])=>e`<div class="guard-line" role="alert">${c(`warn`)} ${t}: ${n.error?.message} (kept version ${n.version})</div>`)}
        ${t?.locked?e`<div class="note" data-locked>${c(`warn`)} ${t.name} is locked: ${t.locked}.</div>`:r}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${Ln.map(t=>e`<a class="cond-tab ${t.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${t.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${t.id===this.screen?`page`:r}>${t.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let t={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return e`<qse-cond-overview .ids=${t.ids} .tiles=${t.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return e`<qse-cond-routing .ids=${t.ids} .meta=${t.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return e`<qse-cond-monitor .meta=${t.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return e`<qse-cond-profiles .ids=${t.ids} .meta=${t.meta} .tiles=${t.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return e`<qse-cond-packages .ids=${t.ids} .tiles=${t.tiles} .selected=${this.params.get(`pkg`)} .tab=${this.params.get(`ptab`)} .pane=${this.params.get(`pview`)}></qse-cond-packages>`;case`deciders`:return e`<qse-cond-deciders></qse-cond-deciders>`;case`sessions`:return e`<qse-cond-sessions .ids=${t.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`memory`:return e`<qse-cond-memory .ids=${t.ids} .selected=${this.params.get(`memory`)} .page=${Number(this.params.get(`mpage`))||null} .tab=${this.params.get(`mtab`)} .note=${this.params.get(`note`)} .nnode=${this.params.get(`nnode`)}></qse-cond-memory>`;case`compare`:return e`<qse-cond-compare .ids=${t.ids} .tiles=${t.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};s([p({attribute:!1})],$.prototype,`params`,void 0),s([t()],$.prototype,`tiles`,void 0),s([t()],$.prototype,`defaultProfile`,void 0),s([t()],$.prototype,`meta`,void 0),s([t()],$.prototype,`config`,void 0),s([t()],$.prototype,`ids`,void 0),s([t()],$.prototype,`error`,void 0),s([t()],$.prototype,`switching`,void 0),s([t()],$.prototype,`reloaded`,void 0),s([t()],$.prototype,`switchError`,void 0),$=s([m(`qse-conductor`)],$);export{$ as QseConductor};