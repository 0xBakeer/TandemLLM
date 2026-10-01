import{A as e,C as t,D as n,E as r,F as i,I as a,L as o,M as s,N as c,P as l,T as u,_ as d,b as f,c as p,g as m,h,j as g,k as _,l as ee,m as v,n as te,r as y,t as ne,u as b,v as x,y as S,z as C}from"./app-BKeabXie.js";import{A as re,C as ie,D as w,DocEditor as T,E,F as ae,I as D,M as oe,N as se,O as ce,P as le,S as ue,T as O,_ as de,a as fe,b as pe,c as me,clsChip as k,confView as he,confirmDialog as A,d as ge,errorsList as j,eventRow as _e,f as ve,feedRow as ye,g as M,h as be,i as N,j as P,k as F,l as xe,m as Se,modeSeg as I,n as Ce,o as we,p as Te,s as Ee,saveBar as L,sourceWords as De,t as Oe,u as R,v as ke,w as Ae,withToken as z,x as je,y as Me}from"./chunk-K1Zcj945.js";var B=class extends c{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.active=`chat`,this.stats=null,this.feed=[],this.error=null,this.open=null,this.openEvent=null,this.paused=!1,this.statsPoll=null,this.feedPoll=null}connectedCallback(){super.connectedCallback(),this.statsPoll=p(()=>this.loadStats(),2e4),this.feedPoll=p(()=>this.loadFeed(),3e3),this.statsPoll.start(),this.feedPoll.start()}disconnectedCallback(){this.statsPoll?.stop(),this.feedPoll?.stop(),super.disconnectedCallback()}async loadStats(){try{this.stats=await D.stats(24),this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async loadFeed(){if(!this.paused)try{this.feed=(await D.decisions({limit:20})).decisions}catch{}}async openAction(e){try{this.openEvent=(await D.event(e)).event}catch(e){this.error=t(e).message}}renderMonitor(e){let t=ke(e);return C`<div class="grid-2 grid-2-wide">
      ${m(`Monitor acts an hour, last 24 h`,e.acts?C`<qse-ribbon .series=${t} .labels=${e.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${180} aria-label="the session monitor's acts per hour by task family"></qse-ribbon>`:v(`No act in 24 hours`,`Shadow decisions and passes are counted, not drawn here.`),{sub:`one colour per task family: thinking, end of turn, tool calls, session`,id:`cond-monitor-ribbon`})}
      ${m(`Last actions of the monitor`,e.last_actions.length?C`<div class="feed" data-last-actions>${e.last_actions.map(e=>C`<button class="feed-row" data-action=${e.id} @click=${()=>this.openAction(e.id)}>
                <span class="feed-time">${g(new Date(e.ts_ms).toISOString())}</span>
                <span class="feed-cls"><span class="badge mono">${e.task??`—`}</span></span>
                <span class="feed-conf muted small">${e.outcome?.kind??`running`}</span>
                <span class="feed-what">${ee({task:e.task,decision:e.decision,mode:`act`,trigger:e.trigger,cap:e.cap})}</span>
                <span class="feed-act">${e.session_key?C`<a class="btn btn-ghost btn-sm" href=${`#/conductor?screen=sessions&session=${encodeURIComponent(e.session_key)}`} @click=${e=>e.stopPropagation()}>session</a>`:o}</span>
              </button>`)}</div>`:C`<div class="feed-empty">No act yet. The Monitor screen's journal shows every decision, shadow and passes of the tool checks included.</div>`,{id:`cond-last-actions`,tools:C`<a class="btn btn-ghost btn-sm" href="#/conductor?screen=monitor">journal ${h(`external`)}</a>`})}
    </div>`}render(){let t=this.stats;if(this.error&&!t)return C`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!t)return d(6);let r=t.classifier,i=r.decider,a=r.qwen,s=i?.calls?i:a?.calls?a:i,c=Object.entries(t.roles),l=c.filter(([,e])=>e.up).length,u=t.monitor;return C`
      <div class="stats" data-tiles>
        ${x({label:`decisions, 24 h`,value:n(t.decisions),sub:C`<span class="num">${n(t.acted)}</span> acted · <span class="num">${n(t.shadow)}</span> in shadow`,title:`${t.injections} notes, ${t.injected_tokens} tokens injected`})}
        ${x({label:`monitor, 24 h`,value:u?n(u.acts):`—`,sub:u?C`acts · <span class="num">${n(u.shadow)}</span> in shadow · <span class="num" data-gave-up>${n(u.gave_up)}</span> gave up`:`the session monitor does not run on this server (--conductor raw)`,na:!u,warn:(u?.gave_up??0)>0,title:u?`${n(u.hidden_tokens)} hidden tokens`:void 0})}
        ${x({label:`classifier p50 / p95${!i?.calls&&a?.calls?` (Qwen)`:``}`,value:s?.p50_ms==null?`—`:`${Math.round(s.p50_ms)} / ${Math.round(s.p95_ms??0)}`,unit:s?.p50_ms==null?void 0:`ms`,sub:C`decider ${n(i?.calls??0)} calls${a?C` · Qwen ${n(a.calls)} (${_(a.p50_ms)})`:o} · fail-open ${e(t.fail_open_rate,1)}`,warn:(t.fail_open_rate??0)>.1})}
        ${x({label:`roles online`,value:`${l} / ${c.length}`,sub:C`${c.map(([e,t])=>C`<span title=${t.error??``} class=${t.up?``:t.mode===`off`?`muted`:`err-ink`} data-role=${e}>${t.up?`●`:`○`} ${e} </span>`)}`,warn:c.some(([,e])=>!e.up&&e.mode!==`off`)})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${m(`Decisions an hour, last 24 h`,t.decisions?C`<qse-ribbon .series=${Te(t,this.ids)} .labels=${t.hourly.map(e=>new Date(e.start_ms).toISOString().slice(11,13)+`:00`)} unit="/h" .height=${200} aria-label="conductor decisions per hour by task class"></qse-ribbon>`:v(`No decisions yet`,`Send a request to the model TandemLLM; the raw engine is never routed.`),{sub:`one colour per task class; tool-result turns count too (they keep the class)`,id:`cond-ribbon`})}
        ${m(`Task classes, 24 h`,C`<div class="table-scroll"><table class="table" data-class-mix>
            <thead><tr><th>class</th><th class="num-col">decisions</th><th class="num-col">labelled right</th></tr></thead>
            <tbody>${t.classes.map(e=>C`<tr><td>${k(e.class_id,this.ids)}</td><td class="num num-col">${n(e.decisions)}</td><td class="num num-col">${e.labelled?`${e.agree} / ${e.labelled}`:`—`}</td></tr>`)}</tbody>
          </table></div>
          <p class="panel-foot">By source: ${Object.entries(t.by_source).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}. By effort: ${Object.entries(t.by_effort).map(([e,t])=>`${e} ${t}`).join(` · `)||`—`}.</p>`,{id:`cond-classes`})}
      </div>
      <div class="grid-2 grid-2-wide">
        ${m(C`Live decisions ${this.paused?C`<span class="badge">paused</span>`:C`<span class="badge badge-on">every 3 s</span>`}`,this.feed.length?C`<div class="feed" data-feed>${this.feed.map(e=>ye(e,this.ids,e=>this.open=e,R(e)))}</div>`:C`<div class="feed-empty">No decisions yet. Every TandemLLM request shows up here: the class, how sure, who decided, what was set and injected.</div>`,{id:`cond-feed`,tools:C`<button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${h(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>
              <a class="btn btn-ghost btn-sm" href="#/conductor?screen=sessions">sessions ${h(`external`)}</a>`})}
        ${m(`Profiles by client`,C`<div class="chips" data-client-chips>
              ${this.tiles.flatMap(e=>e.clients.map(t=>C`<button class="pick" @click=${()=>S({screen:`profiles`,edit:e.name})}><span class="muted">${t}</span> → ${e.name} <span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span></button>`))}
            </div>
            <p class="panel-foot">Other clients get the default profile. A request picks a profile with <code>X-Tandem-Profile</code>; <code>X-Tandem: raw</code> (or the locked <code>bench</code> / <code>raw</code> profile) always serves the engine untouched. Every model name, the old raw one included, is TandemLLM.</p>
            <dl class="kv">${this.tiles.map(e=>C`<div><dt>${e.name}</dt><dd><span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span> <span class="muted small">v${e.version}${e.locked?` · locked`:``}</span></dd></div>`)}</dl>`,{id:`cond-clients`})}
      </div>
      ${u?this.renderMonitor(u):o}
      ${this.open?C`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:o}
      ${this.openEvent?C`<qse-cond-event .ev=${this.openEvent} .close=${()=>this.openEvent=null}></qse-cond-event>`:o}
    `}};s([i({attribute:!1})],B.prototype,`ids`,void 0),s([i({attribute:!1})],B.prototype,`tiles`,void 0),s([i({attribute:!1})],B.prototype,`active`,void 0),s([l()],B.prototype,`stats`,void 0),s([l()],B.prototype,`feed`,void 0),s([l()],B.prototype,`error`,void 0),s([l()],B.prototype,`open`,void 0),s([l()],B.prototype,`openEvent`,void 0),s([l()],B.prototype,`paused`,void 0),B=s([a(`qse-cond-overview`)],B);var V=[`off`,`low`,`medium`,`high`,`xhigh`],Ne=[`small`,`medium`,`large`],H=class extends c{constructor(...e){super(...e),this.ids=[],this.meta=null,this.active=`coding`,this.onSaved=()=>void 0,this.packages=[],this.testText=``,this.testProfile=null,this.testTools=!0,this.testing=!1,this.result=null,this.testError=null,this.ed=new T(this,`router`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),D.doc(`packages`).then(e=>this.packages=Object.keys(e.doc.packages)).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(O(this.ed.draft,e,t))}async test(){if(this.testText.trim()){this.testing=!0,this.testError=null;try{this.result=await D.classify(this.testText,this.testProfile??this.active,this.testTools)}catch(e){this.testError=t(e).message}finally{this.testing=!1}}}render(){let e=this.ed;if(e.loading&&!e.draft)return d(8);if(e.loadError)return C`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=t.classes.map(e=>e.id);return C`
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
          ${t.classifier.stages.map((e,n)=>C`<div class="row-card" data-stage=${e.kind}>
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
                ${[`keep`,`classify`,...n].map(e=>C`<option value=${e} ?selected=${e===t.structural.compaction}>${e===`keep`?`keep the session class`:e===`classify`?`ask a classifier`:`class ${e}`}</option>`)}
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
          <p class="muted small">The budget is a real cap (SRV-159): when the reasoning reaches it, the server closes it with the session monitor's hidden close, so the client sees only <code>&lt;/think&gt;</code> and the answer. Defaults: low 1,024, medium 8,192, high 16,384, xhigh no cap. A profile can override a level (Profiles › Thinking); a client's own budget always wins.</p>
          ${V.map(e=>C`<div class="row-card" data-effort=${e}>
              <div class="row-card-head"><strong class="mono">${e}</strong>
                <label class="switch"><input type="checkbox" .checked=${t.effort.levels[e].thinking} @change=${t=>this.upd([`effort`,`levels`,e,`thinking`],t.target.checked)} /> thinking</label></div>
              <div class="form-grid">${this.num(`budget (tokens, 0 = no cap)`,t.effort.levels[e].budget,t=>this.upd([`effort`,`levels`,e,`budget`],Math.round(t)),{step:256,min:0})}</div>
              <label class="field field-block"><span>sentence in the note</span><textarea rows="2" .value=${t.effort.levels[e].note} @input=${t=>this.upd([`effort`,`levels`,e,`note`],t.target.value)}></textarea></label>
            </div>`)}
          <label class="switch"><input type="checkbox" .checked=${t.effort.raise_on_fail} @change=${e=>this.upd([`effort`,`raise_on_fail`],e.target.checked)} /> a failing checker (RESULT: FAIL) raises the effort one level for that turn</label>
          ${this.renderSize(t.effort.size??fe)}
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
      ${j(e)}
      ${L(e,`the router`)}
    `}renderTest(e){let t=this.result;return C`<section class="panel" data-testbox>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Test a prompt</h2><p class="panel-sub">Runs the live cascade and policy on the text, as a new session, and shows the exact note. Nothing is journaled; the Qwen stage uses the engine for about a second.</p></div></header>
      <label class="field field-block"><span>prompt</span><textarea rows="3" data-test-text .value=${this.testText} @input=${e=>this.testText=e.target.value} placeholder="Refactor the parser so every token carries its line number."></textarea></label>
      <div class="toolbar" style="margin:10px 0 0">
        <label class="field"><span>profile</span><input style="width:120px" .value=${this.testProfile??this.active} @input=${e=>this.testProfile=e.target.value||null} /></label>
        <label class="switch"><input type="checkbox" .checked=${this.testTools} @change=${e=>this.testTools=e.target.checked} /> the request offers tools</label>
        <button class="btn btn-primary" ?disabled=${this.testing||!this.testText.trim()} @click=${()=>this.test()} data-run-test>${this.testing?`Running…`:`Run`}</button>
        ${this.testError?C`<span class="err-ink small" role="alert">${this.testError}</span>`:o}
      </div>
      ${t?t.active?C`<div class="grid-2 grid-2-tight" style="margin-top:14px" data-test-result>
              <div>
                <dl class="kv">
                  <div><dt>class</dt><dd>${k(t.class_id??null,e)} ${he(t.confidence)}</dd></div>
                  <div><dt>decided by</dt><dd>${De(t.source??null)}${t.kept?C` <span class="badge badge-warn">${t.kept}</span>`:o} <span class="muted num">${_(t.classifier_ms)}</span></dd></div>
                  <div><dt>effort</dt><dd>${t.effort??`unchanged`} · thinking ${t.thinking??`on`}${t.budget==null?``:` · budget ${t.budget}`}</dd></div>
                  ${t.extra?.size?C`<div data-test-size><dt>task size</dt><dd>${ce(t.extra.size)}</dd></div>`:o}
                  <div><dt>profile</dt><dd>${t.profile} (${t.mode})</dd></div>
                </dl>
                <div class="stage-list" style="margin-top:10px">${(t.stages??[]).map(e=>C`<div class="stage ${e.error?`is-error`:e.below?``:`is-accepted`}"><strong>${e.stage===`decider`?`decider`:`Qwen`}</strong><span>${e.error??C`${e.label} ${he(e.confidence,e.accept??.6)}`}</span><span class="num muted">${_(e.ms)}</span></div>`)}</div>
              </div>
              <div>${t.note?C`<p class="muted small">The model would see this at the end of the last turn (${(t.note_parts??[]).map(e=>`${e.kind} ${e.name}`).join(`, `)}):</p><pre class="note-text" data-test-note>${t.note}</pre>`:C`<p class="muted small">No note: ${t.shadow?`the decisions are in shadow`:`this class adds nothing`}.</p>`}</div>
            </div>`:C`<p class="muted small" style="margin-top:12px">Profile ${t.profile} is raw (${t.mode}): the conductor does nothing for it.</p>`:o}
    </section>`}renderClass(e,t,n,r){let i=e.policy,a=!this.ed.base?.doc.classes.some(t=>t.id===e.id);return C`<details class="row-card" data-class-row=${e.id} ?open=${a}>
      <summary>
        <span>${k(e.id,r,e.label)} <span class="muted small mono">${i.effort_by_size?Ne.map(e=>w(i,e)).join(`/`):i.effort??`medium`}${(i.packages??[]).length?` · ${(i.packages??[]).join(`, `)}`:``}</span></span>
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
      <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${I(V,i.effort??`medium`,e=>this.upd([`classes`,t,`policy`,`effort`],e),{small:!0,label:`effort of ${e.id}`})}</div>
      <div class="toolbar size-map" style="margin:0" data-size-map=${e.id}><span class="muted small">by task size (the effort above is the ceiling)</span>
        ${Ne.map(n=>C`<label class="field"><span>${n}</span><select aria-label=${`${e.id} effort when ${n}`} @change=${e=>this.setSizeLevel(t,i.effort_by_size,n,e.target.value)}>
            <option value="" ?selected=${!i.effort_by_size?.[n]}>the class's</option>
            ${V.map(e=>C`<option value=${e} ?selected=${i.effort_by_size?.[n]===e}>${e}</option>`)}
          </select>${i.effort_by_size?.[n]&&w(i,n)!==i.effort_by_size[n]?C`<span class="hint">cut to ${w(i,n)}</span>`:o}</label>`)}
      </div>
      <div><span class="muted small">packages (a class with several asks a second question and injects the one that fits)</span>
        <div class="pkg-picks">${this.packages.map(e=>{let n=(i.packages??[]).includes(e);return C`<button class="pick ${n?`is-on`:``}" aria-pressed=${n?`true`:`false`} @click=${()=>this.upd([`classes`,t,`policy`,`packages`],n?(i.packages??[]).filter(t=>t!==e):[...i.packages??[],e])}>${e}</button>`})}</div></div>
      <label class="field field-block"><span>rules (one per line; they go into the note)</span><textarea rows="2" .value=${(i.rules??[]).join(`
`)} @input=${e=>this.upd([`classes`,t,`policy`,`rules`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label>
    </details>`}renderSize(e){let t=(t,n)=>this.upd([`effort`,`size`],O(e,t,n));return C`<div class="row-card" data-size-config>
      <div class="row-card-head"><strong>Task size</strong>
        <label class="switch"><input type="checkbox" .checked=${e.reset_on_compaction} @change=${e=>t([`reset_on_compaction`],e.target.checked)} /> a compaction starts the size over</label></div>
      <p class="muted small">Each signal is small below its first number, medium up to its second, large above. The task's size is the largest any counting signal reached, and it never falls within a task. A class's levels per size (in its row above) pick the effort under the class's effort; whether that acts is each profile's <em>effort by task size</em> (off, shadow, on).</p>
      ${we.map(n=>{let r=e.signals[n.id]??fe.signals[n.id];return C`<div class="size-signal" data-size-signal=${n.id}>
          <label class="switch"><input type="checkbox" .checked=${r.enabled} @change=${e=>t([`signals`,n.id,`enabled`],e.target.checked)} /> ${n.label}</label>
          <div class="form-grid">
            ${this.num(`small below (${n.unit})`,r.small,e=>t([`signals`,n.id],{...r,small:e}),{min:0,step:n.id===`session_tokens`?1e3:1})}
            ${this.num(`medium up to (${n.unit})`,r.medium,e=>t([`signals`,n.id],{...r,medium:e}),{min:0,step:n.id===`session_tokens`?1e3:1,hint:n.hint})}
          </div>
        </div>`})}
    </div>`}setSizeLevel(e,t,n,r){let i={...t??{}};r?i[n]=r:delete i[n],this.upd([`classes`,e,`policy`,`effort_by_size`],Object.keys(i).length?i:void 0)}num(e,t,n,r={}){return C`<label class="field"><span>${e}</span><input type="number" .value=${String(t)} step=${r.step??1} min=${r.min??``} max=${r.max??``} @input=${e=>{let t=Number(e.target.value);Number.isFinite(t)&&n(t)}} />${r.hint?C`<span class="hint">${r.hint}</span>`:o}</label>`}text(e,t,n){return C`<label class="field"><span>${e}</span><input .value=${t} @input=${e=>n(e.target.value)} /></label>`}addClass(){let e=this.ed.draft,t=1;for(;e.classes.some(e=>e.id===`class_${t}`);)t++;let n={id:`class_${t}`,label:`Class ${t}`,description:`Describe the kind of request, in the words a user would use.`,policy:{effort:`medium`,packages:[],rules:[]}};this.ed.set({...e,classes:[...e.classes,n]})}removeClass(e){let t=this.ed.draft;this.ed.set({...t,classes:t.classes.filter((t,n)=>n!==e)})}moveClass(e,t){let n=this.ed.draft,r=[...n.classes],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set({...n,classes:r})}moveStage(e,t){let n=this.ed.draft,r=[...n.classifier.stages],[i]=r.splice(e,1);r.splice(e+t,0,i),this.ed.set(O(n,[`classifier`,`stages`],r))}};s([i({attribute:!1})],H.prototype,`ids`,void 0),s([i({attribute:!1})],H.prototype,`meta`,void 0),s([i({attribute:!1})],H.prototype,`active`,void 0),s([i({attribute:!1})],H.prototype,`onSaved`,void 0),s([l()],H.prototype,`packages`,void 0),s([l()],H.prototype,`testText`,void 0),s([l()],H.prototype,`testProfile`,void 0),s([l()],H.prototype,`testTools`,void 0),s([l()],H.prototype,`testing`,void 0),s([l()],H.prototype,`result`,void 0),s([l()],H.prototype,`testError`,void 0),H=s([a(`qse-cond-routing`)],H);var Pe=[`off`,`shadow`,`act`],Fe=/text|texts|close|nudge|nudges|turn|note|pattern|root|home|requires|suffix|give_up|message|header|reading|gate|ok/,Ie=/pattern|root|home|requires|suffix|header|reading|gate|ok|args/,U=class extends c{constructor(...e){super(...e),this.meta=null,this.onSaved=()=>void 0,this.config=null,this.ed=new T(this,`monitor`)}connectedCallback(){super.connectedCallback(),this.ed.onSaved=()=>this.onSaved(),this.ed.load(),D.config().then(e=>this.config=e).catch(()=>void 0)}upd(e,t){this.ed.draft&&this.ed.set(O(this.ed.draft,e,t))}render(){let e=this.ed,t=this.config?.server.monitor_mode;return C`
      <qse-cond-journal></qse-cond-journal>
      <qse-cond-watch></qse-cond-watch>
      ${e.loading&&!e.draft?d(8):e.loadError?C`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`:C`
            ${t===`raw`?C`<div class="guard-line" data-monitor-off>${h(`warn`)} The session monitor does not run on this server (<code>--conductor raw</code>): the rules below are kept and take effect once it runs.</div>`:o}
            <p class="muted small" style="margin:12px 0">The rules of every task. Each profile sets a task's mode (off, shadow, act), the tool-gate packs and the thresholds on top, on the Profiles screen; the mode here is what a server without a profile uses.</p>
            ${this.renderTasks(e.draft)}
            ${this.renderRest(e.draft)}
            ${j(e)}
            ${L(e,`the monitor`)}`}`}renderTasks(e){let t=Me(e),n=t.filter(t=>e.tasks[t]&&/^C\d/.test(t)),r=t.filter(e=>be(e)),i=t.filter(e=>!n.includes(e)&&!r.includes(e)),a=r.map(t=>e.tasks[t]).find(e=>e.rule===`stall_watchdog`)?.notify;return C`
      <details class="sec" open data-sec="toolchecks">
        <summary>Tool checks <span class="sec-sub">${n.join(`, `)||`none`}: held calls are checked at the end of the answer, before any goes out</span></summary>
        <div class="sec-body">${n.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>
      <details class="sec" data-sec="sessiontasks">
        <summary>Session tasks <span class="sec-sub">${r.join(`, `)||`none`}: across turns, one hidden note at the end of the prompt</span></summary>
        <div class="sec-body">
          <p class="muted small" data-session-order>A request carries at most one session note, in this order: ${r.join(`, `)||`—`}; the others keep their state for the next request. They stay silent on a request without tools and in a project whose system prompt does not match <code>requires</code>.</p>
          <p class="small" data-notify-state>${a?.enabled?C`<span class="badge badge-warn">NOTIFY on</span> without a profile, a push goes out when a client has gone.`:C`<span class="badge">NOTIFY off</span> without a profile nothing is pushed; a profile's Context section turns it on (the overnight profile does, Khaled's ruling of 2026-09-30). The token is a file on the box, never in this document.`}</p>
          ${r.map(t=>this.renderTask(t,e.tasks[t]))}
        </div>
      </details>
      <details class="sec" data-sec="othertasks">
        <summary>Thinking and end of turn <span class="sec-sub">${i.join(`, `)}</span></summary>
        <div class="sec-body">${i.map(t=>this.renderTask(t,e.tasks[t]))}</div>
      </details>`}renderTask(e,t){let n=ie(this.meta?.schemas?.monitor,t.rule),r=this.meta?.tasks?.[e];return C`<div class="row-card" data-monitor-task=${e}>
      <div class="row-card-head">
        <strong>${e}${r?C` · ${r.label}`:o} <span class="muted small mono">${t.rule}</span></strong>
        ${I(Pe,t.mode,t=>this.upd([`tasks`,e,`mode`],t),{small:!0,label:`${e} mode without a profile`})}
      </div>
      ${r?.why?C`<p class="muted small">${r.why}</p>`:o}
      ${n?.properties?C`<div class="form-stack">${Object.entries(n.properties).map(([n,r])=>this.field(n,r,t[n],[`tasks`,e,n]))}</div>`:C`<p class="muted small">This rule's parameters have no schema here; edit them as JSON.</p>${this.jsonField(`parameters`,t,[`tasks`,e])}`}
    </div>`}field(e,t,n,r){let i=Array.isArray(t.type)?t.type[0]:t.type,a=e.replace(/_/g,` `),s=t.description?C`<span class="hint">${t.description}</span>`:o;if(t.enum)return C`<label class="field"><span>${a}</span><select data-param=${e} @change=${e=>this.upd(r,e.target.value)}>${t.enum.map(e=>C`<option value=${String(e)} ?selected=${e===n}>${String(e)}</option>`)}</select>${s}</label>`;if(i===`boolean`)return C`<label class="switch" data-param=${e}><input type="checkbox" .checked=${!!n} @change=${e=>this.upd(r,e.target.checked)} /> ${a}</label>${t.description?C`<p class="hint">${t.description}</p>`:o}`;if(i===`integer`||i===`number`)return C`<label class="field"><span>${a}</span><input type="number" data-param=${e} step=${i===`integer`?`1`:`0.01`} min=${t.minimum??o} max=${t.maximum??o} .value=${n==null?``:String(n)} @input=${e=>{let t=e.target.value;this.upd(r,t===``?null:i===`integer`?Math.round(Number(t)):Number(t))}} />${s}</label>`;if(i===`string`){let t=String(n??``);return Fe.test(e)||t.length>60?C`<label class="field field-block"><span>${a}</span><textarea rows=${Math.min(6,Math.max(2,Math.ceil(t.length/90)))} class=${Ie.test(e)?`mono`:``} data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)}></textarea>${s}</label>`:C`<label class="field"><span>${a}</span><input data-param=${e} .value=${t} @input=${e=>this.upd(r,e.target.value)} />${s}</label>`}if(i===`array`&&t.items&&(Array.isArray(t.items.type)?t.items.type[0]:t.items.type)===`string`){let i=!Array.isArray(n)&&Array.isArray(t.default),o=Array.isArray(n)?n:i?t.default:[];return C`<label class="field field-block"><span>${a} <span class="muted small">(one a line${i?`; the default, not saved yet`:``})</span></span><textarea rows=${Math.min(8,Math.max(2,o.length+1))} data-param=${e} .value=${o.join(`
`)} @input=${e=>this.upd(r,de(e.target.value))}></textarea>${s}</label>`}if(i===`array`&&t.items?.properties){let i=Array.isArray(n)?n:[],o=t.items;return C`<fieldset class="fieldset" data-param=${e}><legend>${a} <span class="muted small">${i.length}</span></legend>${s}
        ${i.map((e,t)=>C`<div class="row-card row-card-sub" data-item=${String(e.id??t)}>
          <div class="row-card-head"><strong class="mono small">${String(e.id??`#${t+1}`)}</strong>
            <button class="btn btn-ghost btn-sm" @click=${()=>this.upd(r,i.filter((e,n)=>n!==t))} aria-label=${`remove ${String(e.id??t+1)}`}>${h(`trash`)}</button></div>
          <div class="form-stack">${Object.entries(o.properties).map(([n,i])=>this.field(n,i,e[n],[...r,t,n]))}</div></div>`)}
        <button class="btn btn-sm" data-add-item=${e} @click=${()=>this.upd(r,[...i,W(o)])}>${h(`plus`)} Add</button>
      </fieldset>`}if(i===`object`&&t.properties){let i=n??{};return C`<fieldset class="fieldset" data-param=${e}><legend>${a}</legend>${s}
        <div class="form-stack">${Object.entries(t.properties).map(([e,n])=>i[e]===void 0&&!(t.required??[]).includes(e)?C`<button class="btn btn-ghost btn-sm" @click=${()=>this.upd([...r,e],W(n))}>${h(`plus`)} ${e.replace(/_/g,` `)}</button>`:this.field(e,n,i[e],[...r,e]))}</div></fieldset>`}return this.jsonField(a,n,r)}jsonField(e,t,n){return C`<label class="field field-block"><span>${e} (JSON)</span><textarea class="mono" rows="6" .value=${JSON.stringify(t,null,2)} @change=${t=>{try{let e=JSON.parse(t.target.value);if(n.length===2&&n[0]===`tasks`){let t=this.ed.draft.tasks[n[1]];this.upd(n,{...e,rule:t.rule,mode:t.mode})}else this.upd(n,e)}catch{this.ed.errors=[`${e}: not JSON`],this.requestUpdate()}}}></textarea></label>`}renderRest(e){return C`
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
      </details>`}};s([i({attribute:!1})],U.prototype,`meta`,void 0),s([i({attribute:!1})],U.prototype,`onSaved`,void 0),s([l()],U.prototype,`config`,void 0),U=s([a(`qse-cond-monitor`)],U);function W(e){let t=Array.isArray(e.type)?e.type[0]:e.type;return e.enum?e.enum[0]:t===`boolean`?!1:t===`integer`||t===`number`?e.minimum??0:t===`array`?[]:t===`object`?Object.fromEntries(Object.entries(e.properties??{}).filter(([t])=>(e.required??[]).includes(t)).map(([e,t])=>[e,W(t)])):``}var G=class extends c{constructor(...e){super(...e),this.sessions=null,this.monitor=!0,this.error=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=p(()=>this.load(),5e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=await D.watch();this.sessions=e.sessions,this.monitor=e.monitor,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}render(){let e=this.sessions;return m(C`Session tasks now <span class="badge badge-on">every 5 s</span>`,this.error&&!e?C`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?d(2):e.length?C`<div class="form-stack" data-watch>${e.map(e=>this.row(e))}</div>`:C`<div class="feed-empty">${this.monitor?`No session the session tasks watch yet: they start with the first request that offers tools under a profile where S1, S2 or S4 runs.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-watch`,sub:`what S1, S2 and S4 hold per session: gate readings, a compaction being watched, minutes since progress, an idle client`})}row(e){let t=e.gates.some(e=>e.noted);return C`<div class="row-card" data-watch-session=${e.session}>
      <div class="row-card-head">
        <strong class="mono small">${e.session}</strong>
        <span class="toolbar" style="margin:0">
          ${e.watch?C`<span class="badge ${e.watch.mode===`act`?`badge-on`:`badge-shadow`}">${e.watch.task} ${e.watch.mode}</span>`:o}
          ${e.client_gone?C`<span class="badge badge-warn" data-client-gone>client gone</span>`:e.running?C`<span class="badge badge-on">running</span>`:o}
          ${e.watch?.notify?C`<span class="badge badge-warn" data-notify>push on</span>`:o}
          ${t?C`<span class="badge badge-warn">gate noted</span>`:o}</span>
      </div>
      <p class="small" style="margin:4px 0">${le(e)}</p>
      <p class="muted small num" style="margin:0">${e.watch?`${e.watch.profile||`—`} · ${e.watch.client||`—`} · `:``}${e.requests} requests · last ${r(new Date(e.last_seen*1e3).toISOString())}${e.note_at?` · last note ${r(new Date(e.note_at*1e3).toISOString())}`:``}</p>
    </div>`}};s([l()],G.prototype,`sessions`,void 0),s([l()],G.prototype,`monitor`,void 0),s([l()],G.prototype,`error`,void 0),G=s([a(`qse-cond-watch`)],G);var K=class extends c{constructor(...e){super(...e),this.session=null,this.events=null,this.monitor=!0,this.error=null,this.task=``,this.kind=``,this.paused=!1,this.open=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=p(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){if(!this.paused)try{let e=await D.events({limit:100,session:this.session,task:this.task||null,mode:this.kind===`shadow`?`shadow`:null,acted:this.kind===`acted`});this.events=e.events,this.monitor=e.monitor,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}render(){let e=this.events,t=[...new Set((e??[]).map(e=>e.task??``).filter(Boolean))].sort();return this.task&&!t.includes(this.task)&&t.push(this.task),C`${m(C`Monitor journal ${this.paused?C`<span class="badge">paused</span>`:C`<span class="badge badge-on">every 3 s</span>`}`,this.error&&!e?C`<p class="err-ink small" role="alert">${this.error}</p>`:e==null?d(4):e.length?C`<div class="feed" data-journal>${e.map(e=>_e(e,e=>this.open=e))}</div>`:C`<div class="feed-empty">${this.monitor?`No decision of the session monitor yet. Every check that finds something, every act and every shadow decision shows up here with its exact hidden text.`:`The session monitor does not run on this server (--conductor raw).`}</div>`,{id:`cond-journal`,sub:`the session monitor's decisions, newest first; the router's are on the Overview, both in a session's timeline and export`,tools:C`<label class="field"><span>task</span><select aria-label="journal task" @change=${e=>{this.task=e.target.value,this.load()}}><option value="">all</option>${t.map(e=>C`<option value=${e} ?selected=${e===this.task}>${e}</option>`)}</select></label>
          <label class="field"><span>show</span><select aria-label="journal decisions" @change=${e=>{this.kind=e.target.value,this.load()}}><option value="">every decision</option><option value="acted" ?selected=${this.kind===`acted`}>acted</option><option value="shadow" ?selected=${this.kind===`shadow`}>shadow</option></select></label>
          <button class="btn btn-ghost btn-sm" @click=${()=>this.paused=!this.paused}>${h(this.paused?`play`:`pause`)} ${this.paused?`resume`:`pause`}</button>`})}
    ${this.open?C`<qse-cond-event .ev=${this.open} .close=${()=>this.open=null}></qse-cond-event>`:o}`}};s([i({attribute:!1})],K.prototype,`session`,void 0),s([l()],K.prototype,`events`,void 0),s([l()],K.prototype,`monitor`,void 0),s([l()],K.prototype,`error`,void 0),s([l()],K.prototype,`task`,void 0),s([l()],K.prototype,`kind`,void 0),s([l()],K.prototype,`paused`,void 0),s([l()],K.prototype,`open`,void 0),K=s([a(`qse-cond-journal`)],K);var Le=[`off`,`low`,`medium`,`high`,`xhigh`],Re={classify:`classify: ask a model which task class the turn is`,effort:`effort: the thinking budget and the effort sentence`,packages:`packages: inject the class's skill or package, hidden`,rules:`rules: the class's working rules in the note`},ze=new Set([`S1`,`S4`]),Be=[[`risk`,`C1`,`risk floors`],[`project`,`C2`,`project rules`],[`same_call`,`C3`,`same call thrice`],[`malformed`,`C4`,`malformed calls`]],q=class extends c{constructor(...e){super(...e),this.ids=[],this.meta=null,this.tiles=[],this.selected=null,this.onSaved=()=>void 0,this.router=null,this.packages=[],this.roleHealth={},this.roleTest={},this.ed=null,this.roles=new T(this,`roles`),this.residency=new T(this,`residency`),this.newClaim=``,this.editing=null}connectedCallback(){super.connectedCallback(),this.roles.onSaved=()=>void this.loadHealth(),this.roles.load(),this.residency.load(),this.loadHealth(),D.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0),D.doc(`packages`).then(e=>this.packages=Object.keys(e.doc.packages)).catch(()=>void 0)}willUpdate(){this.selected!==this.editing&&(this.editing=this.selected,this.ed=this.selected?new T(this,`profiles`,this.selected):null,this.ed&&(this.ed.onSaved=()=>this.onSaved(),this.ed.load()))}async loadHealth(){try{this.roleHealth=(await D.roles()).health}catch{}}async testRole(e){this.roleTest={...this.roleTest,[e]:`testing…`};try{let t=await D.roleTest(e);this.roleTest={...this.roleTest,[e]:t.ok?`ok in ${_(t.ms)}: "${t.answer?.label}" at ${t.answer?.confidence.toFixed(2)}`:`failed: ${t.error}`}}catch(n){this.roleTest={...this.roleTest,[e]:t(n).message}}this.loadHealth()}render(){return C`
      <div class="tiles" data-profile-tiles>
        ${this.tiles.map(e=>C`<button class="tile ${e.name===this.selected?`is-on`:``}" data-profile=${e.name} @click=${()=>S({edit:e.name})}>
            <div class="tile-head"><span class="tile-name">${e.name}</span>${e.locked?C`<span class="badge badge-lock">${h(`warn`)} locked</span>`:C`<span class="badge ${e.mode===`on`?`badge-on`:e.mode===`shadow`?`badge-shadow`:``}">${e.mode}</span>`}</div>
            <div class="tile-meta">${e.locked??e.description}</div>
            <div class="tile-meta">clients: ${e.clients.length?e.clients.join(`, `):`— (header or default)`} · default class ${e.default_class}</div>
            <div class="tile-meta num">decisions on ${e.counts.decisions.on??0} · shadow ${e.counts.decisions.shadow??0} · tasks act ${e.counts.tasks.act??0} · shadow ${e.counts.tasks.shadow??0}</div>
            <div class="tile-meta num">v${e.version} · ${e.last_change.note??``}${e.last_change.created_at?` · ${r(e.last_change.created_at)}`:``}</div>
          </button>`)}
      </div>
      ${this.ed?this.renderEditor(this.ed):C`<p class="muted small" style="margin-top:14px">Pick a profile to edit it.</p>`}
      ${this.renderRoles()}
      ${this.renderResidency()}
    `}upd(e,t){this.ed?.draft&&this.ed.set(O(this.ed.draft,e,t))}ctx(e,t,n,r){if(!this.ed?.draft||!Number.isFinite(r))return;let i=O(this.ed.draft,[`context`,e],r);i.tasks[t]?.thresholds&&n in i.tasks[t].thresholds&&(i=O(i,[`tasks`,t,`thresholds`,n],r)),this.ed.set(i)}renderBudgets(e,t){return C`<div data-budgets>
      <div class="muted small">Thinking budget per level <span class="muted">(reasoning tokens; empty keeps Routing's level, 0 is no cap below the backstop)</span></div>
      <div class="form-grid">
        ${Oe.map(n=>{let r=e.thinking.budgets?.[n],i=M(n,this.router,null).cap,a=M(n,this.router,e);return C`<label class="field ${r==null?`is-inherited`:``}" data-budget=${n}><span class="mono">${n}</span>
            <input type="number" inputmode="numeric" min="0" max=${Ce} step="1024" ?disabled=${t} placeholder=${i==null?``:String(i)} .value=${r==null?``:String(r)}
              @change=${e=>{let t=e.target.value.trim();this.ed?.draft&&this.ed.set(O(this.ed.draft,[`thinking`],ae(this.ed.draft.thinking,n,t===``?null:Number(t))))}} />
            <span class="hint">${me(a.cap)}${a.source===`router`?` (Routing)`:a.source===`profile`?` (this profile)`:``}</span></label>`})}
      </div>
    </div>`}renderEditor(e){if(e.loading&&!e.draft)return d(8);if(e.loadError)return C`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,n=!!e.locked,r=this.meta?.client_kinds??[`open-webui`,`openai-sdk`,`opencode`,`curl`,`dashboard`,`other`],i=this.router?.classes??[];return C`<div style="margin-top:16px" data-editor=${e.name}>
      <div class="toolbar"><h2 class="panel-title">${e.name} <span class="muted small num">v${e.base?.version}</span></h2>
        <a class="btn btn-sm" href=${`#/playground?profile=${encodeURIComponent(e.name??``)}`} data-try>${h(`playground`)} Try in Playground</a></div>
      ${n?C`<div class="note" data-lock-reason>${h(`warn`)} ${e.locked}. Every control below is read-only.</div>`:o}
      <details class="sec" open data-sec="mapping">
        <summary>Mapping <span class="sec-sub">who gets this profile, and the class it starts from</span></summary>
        <div class="sec-body">
          <label class="field field-block"><span>description</span><input .value=${t.description} ?disabled=${n} @input=${e=>this.upd([`description`],e.target.value)} /></label>
          <div class="toolbar" style="margin:0"><span class="muted small">mode (the header switch sets this too)</span>${I([`raw`,`shadow`,`on`],t.mode,e=>this.upd([`mode`],e),{disabled:n,small:!0,label:`profile mode`})}</div>
          <div><span class="muted small">client kinds that resolve to this profile (model <code>TandemLLM</code>, no <code>X-Tandem-Profile</code> header)</span>
            <div class="pkg-picks">${r.map(e=>{let r=t.clients.includes(e);return C`<button class="pick ${r?`is-on`:``}" ?disabled=${n} aria-pressed=${r?`true`:`false`} @click=${()=>this.upd([`clients`],r?t.clients.filter(t=>t!==e):[...t.clients,e])}>${e}</button>`})}</div></div>
          <label class="field"><span>default class (a new session nobody could classify)</span>
            <select ?disabled=${n} @change=${e=>this.upd([`default_class`],e.target.value)}>${i.map(e=>C`<option value=${e.id} ?selected=${e.id===t.default_class}>${e.id}</option>`)}</select></label>
        </div>
      </details>
      <details class="sec" open data-sec="decisions">
        <summary>Routing decisions <span class="sec-sub">off, shadow (journal only) or on, each on its own</span></summary>
        <div class="sec-body">
          ${Object.keys(Re).map(e=>C`<div class="toolbar" style="margin:0" data-decision-mode=${e}>${I([`off`,`shadow`,`on`],t.decisions[e],t=>this.upd([`decisions`,e],t),{disabled:n,small:!0,label:e})}<span class="small">${Re[e]}</span></div>`)}
        </div>
      </details>
      <details class="sec" data-sec="overrides">
        <summary>Per-class overrides <span class="sec-sub">this profile's own effort, packages or rules for a class; grey = the router's</span></summary>
        <div class="sec-body">
          ${i.map(e=>{let r=t.routing[e.id]??{},i=Object.keys(r).length>0;return C`<div class="row-card" data-override=${e.id}>
              <div class="row-card-head">${k(e.id,this.ids,e.label)}
                ${i?C`<button class="btn btn-ghost btn-sm" ?disabled=${n} @click=${()=>{let n={...t.routing};delete n[e.id],this.upd([`routing`],n)}}>use the router's</button>`:C`<span class="muted small">router: ${e.policy.effort??`medium`}${(e.policy.packages??[]).length?` · ${(e.policy.packages??[]).join(`, `)}`:``}</span>`}</div>
              <div class="toolbar" style="margin:0"><span class="muted small">effort</span>${I(Le,r.effort??e.policy.effort??`medium`,t=>this.upd([`routing`,e.id],{...r,effort:t}),{disabled:n,small:!0,label:`effort override of ${e.id}`,cls:r.effort?``:`is-inherited`})}</div>
              <div class="pkg-picks">${this.packages.map(t=>{let i=r.packages??e.policy.packages??[],a=i.includes(t);return C`<button class="pick ${a?`is-on`:``}" ?disabled=${n} @click=${()=>this.upd([`routing`,e.id],{...r,packages:a?i.filter(e=>e!==t):[...i,t]})}>${t}</button>`})}</div>
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
          <div class="toolbar" style="margin:0"><span class="muted small">default</span>${I(Le,t.thinking.default,e=>this.upd([`thinking`,`default`],e),{disabled:n,small:!0,label:`default effort`})}</div>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.thinking.hints} @change=${e=>this.upd([`thinking`,`hints`],e.target.checked)} /> hints from the last tool result (a checker FAIL raises the effort)</label>
          <div class="toolbar" style="margin:0" data-by-size><span class="muted small">effort by task size</span>${I([`off`,`shadow`,`on`],t.thinking.by_size??`shadow`,e=>this.upd([`thinking`,`by_size`],e),{disabled:n,small:!0,label:`effort by task size`,cls:t.thinking.by_size?``:`is-inherited`})}</div>
          <p class="muted small">Effort by task size (SRV-157): a class with levels per size (Routing › the class) thinks at its size's level, never above the class's effort; the size only rises within a task. <strong>off</strong>: the class's effort; <strong>shadow</strong>: the class's effort, and the journal shows the size and the effort it would set; <strong>on</strong>: the size picks.</p>
          ${this.renderBudgets(t,n)}
          <p class="muted small">A client's own <code>max_reasoning_tokens</code> or <code>thinking_budget</code> always wins (fixed).</p>
        </div>
      </details>
      <details class="sec" data-sec="toolgate">
        <summary>Tool gate <span class="sec-sub">rule packs on held tool calls (SRV-113); a pack that is off turns its task off</span></summary>
        <div class="sec-body"><div class="form-grid">
          ${Be.map(([e,r,i])=>C`<label class="switch" data-pack=${e}><input type="checkbox" ?disabled=${n} .checked=${t.toolgate[e]} @change=${t=>this.upd([`toolgate`,e],t.target.checked)} /> ${i} <span class="muted small">(${r}, ${t.toolgate[e]?t.tasks[r]?.mode??`off`:`off`})</span></label>`)}
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
      <details class="sec" data-sec="hints">
        <summary>Speed hints <span class="badge badge-warn">not gated</span></summary>
        <div class="sec-body">
          <p class="warn-ink small">No SPD measurement is in the journal yet (SPD-73); the router's ≤ 1 ulp text rule and the never-worse gate come first.</p>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.stair_width} @change=${e=>this.upd([`hints`,`stair_width`],e.target.checked)} /> StairCut width hint</label>
          <label class="switch"><input type="checkbox" ?disabled=${n} .checked=${t.hints.prefix_retention} @change=${e=>this.upd([`hints`,`prefix_retention`],e.target.checked)} /> prefix retention hint</label>
        </div>
      </details>
      ${j(e)}
      ${L(e,`profile ${e.name}`)}
    </div>`}renderTask(e,t,n,r,i){let a=this.meta?.tasks[e],s=a?.role&&a.tier===`1`,c=!s||!!this.roleHealth[a.role]?.up;return C`<div class="task-row" data-task=${e}>
      <div><div class="task-id">${e}</div><div class="muted small">${a?.family??``}</div></div>
      <div>${a?.label??e}<div class="task-why">${a?.why??``}</div>${s&&!c?C`<div class="warn-ink small">${h(`warn`)} the ${a.role} role is offline: act is refused</div>`:o}</div>
      <div class="task-mode">${I([`off`,`shadow`,`act`],t,t=>this.upd([`tasks`,e,`mode`],t),{disabled:i,small:!0,label:`${e} mode`})}</div>
      <div class="thresholds">
        ${ze.has(e)&&Object.keys(r).length?C`<p class="muted small" data-in-context=${e}>${Object.entries(r).map(([e,t])=>`${e.replace(/_/g,` `)} ${t}`).join(` · `)}: set in Context below</p>`:o}
        ${Object.entries(ze.has(e)?{}:r).map(([t,n])=>{let r=a?.thresholds[t];return C`<label class="field ${r&&r.default!==n?`is-changed`:``}"><span>${t.replace(/_/g,` `)} <span class="hint">${r?.unit??``}${r?`, default ${r.default}`:``}</span></span>
            <input type="number" ?disabled=${i} .value=${String(n)} step="any" @input=${n=>{let r=Number(n.target.value);Number.isFinite(r)&&this.upd([`tasks`,e,`thresholds`,t],r)}} /></label>`})}
        <label class="field"><span>cap <span class="hint">acts a request</span></span><input type="number" ?disabled=${i} min="0" max="10" .value=${String(n)} @input=${t=>this.upd([`tasks`,e,`cap`],Math.round(Number(t.target.value)))} /></label>
      </div>
    </div>`}renderRoles(){let e=this.roles;if(!e.draft)return e.loadError?C`<p class="err-ink small">${e.loadError}</p>`:o;let t=e.draft,n=(n,r)=>e.set(O(t,n,r));return C`<details class="sec" style="margin-top:16px" data-sec="roles">
      <summary>Roles <span class="sec-sub">shared by every profile; a role is a kind of question, only this names models</span></summary>
      <div class="sec-body">
        ${Object.entries(t).map(([e,t])=>{let r=this.roleHealth[e];return C`<div class="row-card" data-role-row=${e}>
            <div class="row-card-head"><strong>${e}</strong>
              <span class="toolbar" style="margin:0">${r?C`<span class="badge ${r.up?`badge-ok`:t.mode===`off`?``:`badge-warn`}">${r.up?`up ${_(r.ms)}`:t.mode===`off`?`off`:`down: ${r.error??`?`}`}</span>`:o}
                <button class="btn btn-sm" @click=${()=>this.testRole(e)} data-role-test=${e}>Test</button></span></div>
            ${this.roleTest[e]?C`<p class="small" role="status">${this.roleTest[e]}</p>`:o}
            <div class="form-grid">
              <label class="field"><span>endpoint (self = this server)</span><input .value=${t.endpoint} @input=${t=>n([e,`endpoint`],t.target.value)} /></label>
              <label class="field"><span>model</span><input .value=${t.model} @input=${t=>n([e,`model`],t.target.value)} /></label>
              <label class="field"><span>timeout (ms)</span><input type="number" .value=${String(t.timeout_ms)} @input=${t=>n([e,`timeout_ms`],Math.round(Number(t.target.value)))} /></label>
              <label class="field"><span>memory claim (GB)</span><input type="number" step="0.1" .value=${String(t.memory_gb)} @input=${t=>n([e,`memory_gb`],Number(t.target.value))} /></label>
            </div>
            <div class="toolbar" style="margin:0"><span class="muted small">mode</span>${I([`off`,`shadow`,`on`],t.mode,t=>n([e,`mode`],t),{small:!0,label:`${e} mode`})}<span class="muted small mono">${t.contract}</span></div>
          </div>`})}
        ${j(e)}
        ${L(e,`the roles`)}
      </div>
    </details>`}renderResidency(){let e=this.residency;if(!e.draft)return e.loadError?C`<p class="err-ink small">${e.loadError}</p>`:o;let t=e.draft,n=(n,r)=>e.set(O(t,n,r)),r=(e,r=`0.1`,i=!1)=>C`<input type="number" step=${r} min="0" .value=${String(t[e[0]])} @input=${t=>{let r=Number(t.target.value);n(e,i?Math.round(r):r)}} data-res=${e[0]} />`,i=[`measured`,`peak`,`declared`];return C`<details class="sec" style="margin-top:16px" data-sec="residency">
      <summary>Residency <span class="sec-sub">what may load beside the engine: a role loads when MemAvailable minus its claim stays at or above floor + margin, one at a time, never during a hold</span></summary>
      <div class="sec-body">
        <div class="form-grid">
          <label class="field"><span>floor (GiB of MemAvailable)</span>${r([`floor_gib`])}</label>
          <label class="field"><span>margin above the floor (GiB)</span>${r([`margin_gib`])}</label>
          <label class="field"><span>settle after a stop (s)</span>${r([`settle_s`],`1`,!0)}</label>
          <label class="field"><span>load order (roles, first loads first)</span><input .value=${t.order.join(`, `)} @change=${e=>n([`order`],e.target.value.split(/[\s,]+/).filter(Boolean))} data-res="order" /></label>
        </div>
        ${Object.entries(t.claims).map(([e,r])=>C`<div class="row-card" data-claim=${e}>
            <div class="row-card-head"><strong>${e}</strong><button class="btn btn-ghost btn-sm" @click=${()=>{let{[e]:r,...i}=t.claims;n([`claims`],i)}} aria-label=${`remove the claim of ${e}`}>${h(`trash`)}</button></div>
            <div class="form-grid">
              <label class="field"><span>claim (GiB, its peak)</span><input type="number" step="0.1" min="0" .value=${String(r.gib)} @input=${t=>n([`claims`,e,`gib`],Number(t.target.value))} data-claim-gib=${e} /></label>
              <label class="field"><span>basis</span><select @change=${t=>n([`claims`,e,`basis`],t.target.value)}>${i.map(e=>C`<option value=${e} ?selected=${e===r.basis}>${e}</option>`)}</select></label>
              <label class="field field-wide"><span>note (where the number comes from)</span><input .value=${r.note} @input=${t=>n([`claims`,e,`note`],t.target.value)} /></label>
            </div>
          </div>`)}
        <div class="toolbar"><label class="field"><span>a claim for another role</span><input placeholder="role id, e.g. librarian" .value=${this.newClaim} @input=${e=>this.newClaim=e.target.value.trim()} /></label>
          <button class="btn btn-sm" ?disabled=${!/^[a-z][a-z0-9_-]{0,39}$/.test(this.newClaim)||this.newClaim in t.claims} @click=${()=>{n([`claims`,this.newClaim],{gib:0,basis:`declared`,note:``}),this.newClaim=``}}>${h(`plus`)} Add</button></div>
        <p class="muted small">The claims are peaks, not idle figures, and MemAvailable, not nvidia-smi (arbiter: 5.1 GB there, 8.2 GiB of MemAvailable). What fits now, the engine's own split and the last refusal are on the <a href="#/system">System view</a>.</p>
        ${j(e)}
        ${L(e,`the residency limits`)}
      </div>
    </details>`}};s([i({attribute:!1})],q.prototype,`ids`,void 0),s([i({attribute:!1})],q.prototype,`meta`,void 0),s([i({attribute:!1})],q.prototype,`tiles`,void 0),s([i({attribute:!1})],q.prototype,`selected`,void 0),s([i({attribute:!1})],q.prototype,`onSaved`,void 0),s([l()],q.prototype,`router`,void 0),s([l()],q.prototype,`packages`,void 0),s([l()],q.prototype,`roleHealth`,void 0),s([l()],q.prototype,`roleTest`,void 0),s([l()],q.prototype,`newClaim`,void 0),q=s([a(`qse-cond-profiles`)],q);var J=class extends c{constructor(...e){super(...e),this.ids=[],this.router=null,this.sel=null,this.found=null,this.scanError=null,this.ed=new T(this,`packages`)}connectedCallback(){super.connectedCallback(),this.ed.load(),D.doc(`router`).then(e=>this.router=e.doc).catch(()=>void 0)}usedBy(e){return(this.router?.classes??[]).filter(t=>(t.policy.packages??[]).includes(e)).map(e=>e.id)}upd(e,t){this.ed.draft&&this.ed.set(O(this.ed.draft,e,t))}async scan(){this.scanError=null;try{this.found=(await D.scan()).skills}catch(e){this.scanError=t(e).message}}register(e){let t=this.ed.draft,n={title:e.name.replace(/-/g,` `),kind:`skill`,source:`skill:${e.name}`,description:e.description.slice(0,600)||`The ${e.name} skill.`,body:``,max_tokens:4e3,enabled:!0};this.ed.set(O(t,[`packages`,e.name],n)),this.sel=e.name}addInline(){let e=this.ed.draft,t=1;for(;e.packages[`package-${t}`];)t++;let n=`package-${t}`;this.ed.set(O(e,[`packages`,n],{title:`Package ${t}`,kind:`package`,source:`inline`,description:`What this package is for, in one sentence.`,body:`# Instructions

`,max_tokens:3e3,enabled:!0})),this.sel=n}removePackage(e){let t=this.ed.draft,n={...t.packages};delete n[e],this.ed.set({...t,packages:n}),this.sel=null}render(){let e=this.ed;if(e.loading&&!e.draft)return d(6);if(e.loadError)return C`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${e.loadError}</div></div>`;let t=e.draft,r=e.base?.status??{};return C`
      <section class="panel" data-packages>
        <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">Packages and skills</h2><p class="panel-sub">The router injects one into the prompt when the turn's class names it and the client did not load it itself.</p></div>
          <div class="panel-tools"><button class="btn btn-sm" @click=${()=>this.addInline()} data-add-package>${h(`plus`)} New package</button><button class="btn btn-sm" @click=${()=>this.scan()} data-scan>${h(`search`)} Skills on the box</button></div></header>
        <div class="table-scroll"><table class="table">
          <thead><tr><th>name</th><th>source</th><th>on the box</th><th class="num-col">tokens</th><th>classes</th><th>enabled</th></tr></thead>
          <tbody>${Object.entries(t.packages).map(([e,t])=>{let i=r[e];return C`<tr class="req-row ${this.sel===e?`is-open`:``}" data-package-row=${e} @click=${()=>this.sel=e}>
              <td><strong>${e}</strong><div class="muted small">${t.title}</div></td>
              <td class="mono small">${t.source}</td>
              <td>${i?i.available?C`<span class="badge badge-ok">found</span>`:C`<span class="badge badge-warn" title=${i.path??``}>${i.error??`missing`}</span>`:C`<span class="badge">not saved</span>`}</td>
              <td class="num num-col">${i?n(i.tokens_est):`—`}</td>
              <td>${this.usedBy(e).map(e=>k(e,this.ids))}</td>
              <td>${t.enabled?`yes`:`no`}</td>
            </tr>`})}</tbody></table></div>
      </section>
      ${this.found?C`<section class="panel" style="margin-top:16px" data-found>
            <header class="panel-head"><h2 class="panel-title">Skills in ${t.skill_dirs.join(`, `)}</h2></header>
            ${this.found.length?C`<div class="table-scroll"><table class="table"><tbody>${this.found.map(e=>C`<tr><td><strong>${e.name}</strong><div class="muted small cell-wrap" style="white-space:normal">${e.description}</div></td><td class="num">${n(e.chars)} chars</td>
                    <td>${e.registered||t.packages[e.name]?C`<span class="badge badge-ok">registered</span>`:C`<button class="btn btn-sm" @click=${()=>this.register(e)} data-register=${e.name}>Register</button>`}</td></tr>`)}</tbody></table></div>`:C`<p class="muted small">No SKILL.md in those directories on the box. Copy the skills there (for opencode's: <code>rsync -a ~/.config/opencode/skills/ spark:~/.config/opencode/skills/</code>).</p>`}
          </section>`:this.scanError?C`<p class="err-ink small">${this.scanError}</p>`:o}
      ${this.sel&&t.packages[this.sel]?this.renderEditor(this.sel,t.packages[this.sel]):o}
      <details class="sec" style="margin-top:16px" data-sec="dirs">
        <summary>Skill directories on the box <span class="sec-sub">searched in order for <code>skill:&lt;name&gt;</code></span></summary>
        <div class="sec-body"><label class="field field-block"><span>one directory per line</span><textarea rows="3" .value=${t.skill_dirs.join(`
`)} @input=${e=>this.upd([`skill_dirs`],e.target.value.split(`
`).map(e=>e.trim()).filter(Boolean))}></textarea></label></div>
      </details>
      ${j(e)}
      ${L(e,`the packages`)}
    `}renderEditor(e,t){let n=t.source===`inline`,r=this.usedBy(e);return C`<section class="panel" style="margin-top:16px" data-package-editor=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${e}</h2><p class="panel-sub">${r.length?C`used by ${r.join(`, `)}`:`no class uses it yet (add it on the Routing screen)`}</p></div>
        <div class="panel-tools"><button class="btn btn-sm btn-ghost" ?disabled=${r.length>0} title=${r.length?`a class uses it`:``} @click=${()=>this.removePackage(e)}>${h(`trash`)} Remove</button></div></header>
      <div class="form-grid">
        <label class="field"><span>title</span><input .value=${t.title} @input=${t=>this.upd([`packages`,e,`title`],t.target.value)} /></label>
        <label class="field"><span>source</span><input .value=${t.source} @input=${t=>this.upd([`packages`,e,`source`],t.target.value)} /><span class="hint">inline, skill:&lt;name&gt; or file:&lt;path on the box&gt;</span></label>
        <label class="field"><span>max tokens</span><input type="number" .value=${String(t.max_tokens)} @input=${t=>this.upd([`packages`,e,`max_tokens`],Math.round(Number(t.target.value)))} /><span class="hint">longer text is cut at a line end</span></label>
        <label class="switch"><input type="checkbox" .checked=${t.enabled} @change=${t=>this.upd([`packages`,e,`enabled`],t.target.checked)} /> enabled</label>
      </div>
      <label class="field field-block"><span>description (the package question reads it)</span><textarea rows="2" .value=${t.description} @input=${t=>this.upd([`packages`,e,`description`],t.target.value)}></textarea></label>
      ${n?C`<label class="field field-block"><span>text the model gets</span><textarea rows="14" data-package-body .value=${t.body} @input=${t=>this.upd([`packages`,e,`body`],t.target.value)}></textarea></label>`:C`<p class="muted small">The text is the file on the box, read when first needed and again when it changes; its YAML front matter is left out.</p>`}
    </section>`}};s([i({attribute:!1})],J.prototype,`ids`,void 0),s([l()],J.prototype,`router`,void 0),s([l()],J.prototype,`sel`,void 0),s([l()],J.prototype,`found`,void 0),s([l()],J.prototype,`scanError`,void 0),J=s([a(`qse-cond-packages`)],J);var Ve={not_loaded:`not loaded`,loading:`loading the weights`,warming:`warming up`,loaded:`loaded`,unloading:`unloading`,failed:`failed`};function He(e){return e===`loaded`?`badge-ok`:e===`failed`?`badge-warn`:e===`loading`||e===`warming`||e===`unloading`?`badge-shadow`:``}function Y(e){if(!e||e.allowed)return null;let t=e.reason?ne[e.reason]??e.reason:`refused`;if(e.reason!==`refused_memory`)return t;let n=[`MemAvailable ${y(e.mem_available_gib)}`];return e.engine_growth_gib&&n.push(`engine growth ${y(e.engine_growth_gib)}`),n.push(`claim ${y(e.claim_gib)}`),e.after_gib!=null&&n.push(`${y(e.after_gib)} left`,`${y(e.need_gib)} needed`),`${t} (${n.join(`, `)})`}function Ue(e){return e?e.allowed?`fits, ${y(e.after_gib)} left after the claim`:Y(e)??`refused`:`—`}function We(e){let t=Object.entries(e).sort((e,t)=>t[1]-e[1]);return t.length?t.map(([e,t])=>`${e} ${t}`).join(` · `):`no call yet`}function Ge(e){let t=e.limits;return[t.context_tokens,t.question_tokens,t.max_options].map(e=>e==null?`—`:String(e)).join(` / `)}function Ke(e,t){let n=e.trim();if(!n)return`Write the question.`;let r=t.split(/[,\n]/).map(e=>e.trim()).filter(Boolean);return r.length<2?`Give at least two options.`:r.length>20?`At most 20 options (the model card’s limit).`:new Set(r).size===r.length?{type:`choice`,instructions:n,criteria:Object.fromEntries(r.map(e=>[e,null]))}:`Two options have the same name.`}function qe(e){return e.type===`noul`?[[`true`,e.noul??0],[`false`,1-(e.noul??0)]]:Object.entries(e.probabilities??{}).sort((e,t)=>t[1]-e[1])}var Je=[`on_start`,`on_demand`,`off`],X=class extends c{constructor(...e){super(...e),this.st=null,this.error=null,this.busy=null,this.said={},this.tState=`The build fails since this morning: TypeError in src/parser.ts line 41 after the refactor.`,this.tQuestion=`What kind of work does this request ask for?`,this.tOptions=`debug, code_edit, plan, chat`,this.tModel=`auto`,this.tOut=null,this.tErr=null,this.asking=!1,this.ed=new T(this,`deciders`),this.poller=null}connectedCallback(){super.connectedCallback(),this.ed.load(),this.poller=p(()=>this.load(),3e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{this.st=await D.deciders(),this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async act(e,n){if(n!==`unload`||await A(`Unload ${e.name}?`,[`Its weights and its memory pool go. Questions routed to it fall back to another loaded model of its family, or fail open (the conductor passes to Qwen).`,`Loading it again goes through the residency planner.`],`Unload`)){this.busy=e.name,this.said={...this.said,[e.name]:``};try{let t=await z(()=>n===`load`?D.deciderLoad(e.name):D.deciderUnload(e.name));this.said={...this.said,[e.name]:n===`load`?`load started (${t.state??`loading`})`:`unloaded`}}catch(r){let i=r instanceof f&&r.status===409&&n===`load`?`refused by the residency planner: ${r.message}`:t(r).message;this.said={...this.said,[e.name]:i}}finally{this.busy=null,await this.load()}}}setLoad(e,t){let n=this.ed.draft;if(!n)return;let r=n.models.findIndex(t=>t.name===e),i=r>=0?n.models.map((e,n)=>n===r?{...e,load:t}:e):[...n.models,{name:e,load:t}];this.ed.set({...n,models:i})}async ask(){let e=Ke(this.tQuestion,this.tOptions);if(typeof e==`string`){this.tErr=e;return}this.asking=!0,this.tErr=null;try{this.tOut=await D.deciderTest(this.tState,{q:e},this.tModel)}catch(e){this.tOut=null,this.tErr=t(e).message}finally{this.asking=!1}}render(){let e=this.st;if(this.error&&!e)return C`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!e)return d(6);if(!e.enabled)return m(`Decision models`,C`<div class="state" data-deciders-off>
          <div class="state-title">${h(`warn`)} This engine serves no decision model</div>
          <p class="muted small">${e.why}</p>
          <p class="muted small">With models named, the conductor's decider role (endpoint <code>self</code>) asks them in process, <code>/v1/systemone</code> answers on the engine port behind the admin token, and this screen loads, unloads and tests them.</p>
        </div>`,{id:`deciders-off`});let t=e.models.filter(e=>e.state===`loaded`);return C`
      <div class="stats" data-decider-tiles>
        ${x({label:`loaded`,value:`${t.length} / ${e.models.length}`,sub:t.length?t.map(e=>e.name).join(`, `):`none: questions fail open`,warn:!t.length})}
        ${x({label:`on the device`,value:te(t.reduce((e,t)=>e+t.device_bytes,0)),sub:`${e.device??`no device`} · own stream and memory pool`})}
        ${x({label:`a load now`,value:e.planner?.allowed?`fits`:`refused`,sub:Ue(e.planner),warn:!e.planner?.allowed})}
        ${x({label:`when they run`,value:`idle`,sub:`only while no request holds the engine (stage A)`})}
      </div>
      ${this.renderModels(e)}
      ${this.renderMemory(e)}
      ${this.renderQuestions(e)}
      ${this.renderTest(e)}
    `}renderModels(e){let t=this.ed.draft,r=e=>t?.models.find(t=>t.name===e)?.load??`on_demand`;return m(`Models`,C`<div class="table-scroll"><table class="table" data-deciders>
        <thead><tr><th>model</th><th>state</th><th>on start</th><th>limits <span class="muted small">ctx / question / options</span></th><th>precision</th><th class="num-col">questions</th><th class="num-col">forward p50 / p95</th><th class="num-col">errors</th><th></th></tr></thead>
        <tbody>${e.models.map(e=>{let t=Y(e.refusal),i=this.said[e.name],a=e.state===`loaded`||e.state===`failed`||e.state===`warming`;return C`<tr data-decider-row=${e.name}>
              <td><strong>${e.name}</strong><div class="muted small">${e.family} · ${e.contract}${e.aliases.length?C` · ${e.aliases.join(`, `)}`:o}</div></td>
              <td><span class="badge ${He(e.state)}" data-decider-state=${e.state}>${Ve[e.state]}</span>${e.warm===!1?C` <span class="muted small">not warmed</span>`:o}</td>
              <td><select aria-label=${`load policy of ${e.name}`} data-decider-load=${e.name} @change=${t=>this.setLoad(e.name,t.target.value)}>
                ${Je.map(t=>C`<option value=${t} ?selected=${r(e.name)===t}>${t.replace(`_`,` `)}</option>`)}
              </select></td>
              <td class="num">${Ge(e)}</td>
              <td class="small">${e.precision??C`<span class="muted">—</span>`}<div class="muted small mono" title=${e.weight_hash??``}>${e.weight_hash?e.weight_hash.slice(0,12):``}</div></td>
              <td class="num num-col">${n(e.questions)}</td>
              <td class="num num-col">${e.stats?.forward_ms_p50==null?`—`:`${_(e.stats.forward_ms_p50)} / ${_(e.stats.forward_ms_p95)}`}</td>
              <td class="num num-col ${e.stats?.errors?`err-ink`:``}">${e.stats?n(e.stats.errors+e.stats.deadline_missed):`—`}</td>
              <td>${a?C`<button class="btn btn-sm" ?disabled=${this.busy===e.name} @click=${()=>this.act(e,`unload`)} data-unload=${e.name}>Unload</button>`:C`<button class="btn btn-sm btn-primary" ?disabled=${this.busy===e.name||e.state===`loading`||e.state===`unloading`} @click=${()=>this.act(e,`load`)} data-load=${e.name}>Load</button>`}</td>
            </tr>
            ${t||e.error||i?C`<tr class="is-warn" data-decider-note=${e.name}><td colspan="9" class="small">
                  ${t?C`<span class="warn-ink" data-refusal=${e.name}>${h(`warn`)} ${t}</span> `:o}
                  ${e.error?C`<span class="err-ink">${e.error}</span> `:o}
                  ${i?C`<span role="status">${i}</span>`:o}
                </td></tr>`:o}`})}</tbody>
      </table></div>
      ${e.unserved?.length?C`<p class="muted small">In the document but not served by this engine: ${e.unserved.join(`, `)} (<code>QWEN38_DECIDER_MODELS</code> names what it serves).</p>`:o}
      <p class="muted small">"On start": loaded after the engine is healthy and its page cache is dropped, one at a time through the planner; "on demand": when a question first needs it; "off": only by Load here. Saved with the question mapping below.</p>`,{id:`deciders-models`,sub:`the engine’s own decision models: own CUDA stream and memory pool, nothing shared with Qwen`})}renderMemory(e){let t=e.planner,n=e.last_refusal;return m(`Memory`,C`<div class="stats residency-stats" data-decider-memory>
          <div class="stat"><div class="stat-label">MemAvailable</div><div class="stat-value num">${y(t?.mem_available_gib)}</div></div>
          ${t?.engine_growth_gib==null?o:C`<div class="stat"><div class="stat-label">engine growth to its peak</div><div class="stat-value num">${y(t.engine_growth_gib)}</div></div>`}
          <div class="stat"><div class="stat-label">decider claim</div><div class="stat-value num">${y(t?.claim_gib)}</div></div>
          <div class="stat"><div class="stat-label">floor + margin</div><div class="stat-value num">${y(t?.floor_gib)} + ${y(t?.margin_gib)}</div></div>
          <div class="stat ${t?.allowed?``:`is-warn`}"><div class="stat-label">a load now</div><div class="stat-value num small">${Ue(t)}</div></div>
        </div>
        <p class="small" data-decider-last-refusal>${n?C`Last refused engine load: ${Y(n)} <span class="muted">(${n.source??``})</span>`:C`<span class="muted">No refused engine load in the planner's journal.</span>`}</p>
        <p class="muted small">A load is allowed when MemAvailable, less what the engine may still take up to its declared peak, minus the claim stays at or above floor + margin, never during a hold, never beside another load. <a href="#/conductor?screen=profiles">Edit the claim and the limits</a> (Profiles, Residency).</p>`,{id:`deciders-memory`})}renderQuestions(e){let t=this.ed,n=t.draft,r=e.models.map(e=>e.name),i=n?Object.entries(n.questions):[];return C`${m(`Questions`,n?C`<div class="table-scroll"><table class="table" data-decider-questions>
              <thead><tr><th>question id</th><th>answered by</th><th>last calls</th></tr></thead>
              <tbody>${i.map(([i,a])=>{let s=e.questions.find(e=>e.id===i);return C`<tr data-question=${i}>
                  <td class="mono">${i}</td>
                  <td><select aria-label=${`model for ${i}`} data-question-model=${i} @change=${e=>t.set(O(n,[`questions`,i],e.target.value))}>
                    ${[`auto`,...r].map(e=>C`<option value=${e} ?selected=${e===a}>${e===`auto`?`auto (the role’s model; the router picks)`:e}</option>`)}
                    ${a!==`auto`&&!r.includes(a)?C`<option value=${a} selected>${a} (not served)</option>`:o}
                  </select></td>
                  <td class="small">${We(s?.picks??{})}</td>
                </tr>`})}</tbody></table></div>
              <p class="muted small">The conductor asks <code>classify</code> (the task class of a new message) and <code>packages</code> (which skill fits); the session monitor's Tier 1 questions join this list when they land.</p>`:d(3),{id:`deciders-questions`,sub:`which model answers each question the conductor asks`})}
      ${j(t)}
      ${L(t,`the deciders document`)}`}renderTest(e){let t=this.tOut,r=t?.answers.q;return m(`Test box`,C`<div class="form-grid" data-decider-test>
          <label class="field field-block"><span>state (the text the model reads)</span><textarea rows="3" .value=${this.tState} @input=${e=>this.tState=e.target.value} data-test-state></textarea></label>
          <label class="field"><span>question</span><input .value=${this.tQuestion} @input=${e=>this.tQuestion=e.target.value} data-test-question /></label>
          <label class="field"><span>options, comma separated</span><input .value=${this.tOptions} @input=${e=>this.tOptions=e.target.value} data-test-options /></label>
          <label class="field"><span>model</span><select @change=${e=>this.tModel=e.target.value} data-test-model>
            ${[`auto`,...e.models.map(e=>e.name)].map(e=>C`<option value=${e} ?selected=${e===this.tModel}>${e}</option>`)}
          </select></label>
        </div>
        <div class="savebar"><button class="btn btn-primary" ?disabled=${this.asking} @click=${()=>this.ask()} data-test-ask>${h(`send`)} ${this.asking?`Asking…`:`Ask`}</button>
          ${this.tErr?C`<span class="err-ink small" role="alert" data-test-error>${this.tErr}</span>`:o}</div>
        ${t&&r?C`<div class="decider-answer" data-test-out>
              <p><strong data-test-choice>${r.type===`choice`?r.choice:r.type===`noul`?(r.noul??0).toFixed(4):(r.score??0).toFixed(4)}</strong>
                <span class="muted small num">confidence ${r.confidence.toFixed(2)} · ${t.model} · ${_(t.latency_ms)} · ${t.path} · ${n(t.usage.input_tokens)} tokens</span></p>
              <div class="prob-list">${qe(r).map(([e,t])=>C`<div class="prob-row" data-prob=${e}><span class="mono small">${e}</span><span class="conf-bar"><span style="width:${Math.round(t*100)}%"></span></span><span class="num small">${t.toFixed(4)}</span></div>`)}</div>
              <p class="muted small" data-test-routing>routing: ${t.routing.reason}${t.routing.requested?` (wanted ${t.routing.requested})`:``}</p>
            </div>`:o}`,{id:`deciders-test`,sub:`one choice question, as /v1/systemone answers it (it runs in the idle window like any other)`})}};s([l()],X.prototype,`st`,void 0),s([l()],X.prototype,`error`,void 0),s([l()],X.prototype,`busy`,void 0),s([l()],X.prototype,`said`,void 0),s([l()],X.prototype,`tState`,void 0),s([l()],X.prototype,`tQuestion`,void 0),s([l()],X.prototype,`tOptions`,void 0),s([l()],X.prototype,`tModel`,void 0),s([l()],X.prototype,`tOut`,void 0),s([l()],X.prototype,`tErr`,void 0),s([l()],X.prototype,`asking`,void 0),X=s([a(`qse-cond-deciders`)],X);var Z=class extends c{constructor(...e){super(...e),this.ids=[],this.selected=null,this.list=null,this.live=[],this.turns=null,this.events=[],this.error=null,this.query=``,this.client=``,this.open=null,this.listPoll=null,this.loadedFor=null}connectedCallback(){super.connectedCallback(),this.listPoll=p(()=>this.loadList(),1e4),this.listPoll.start()}disconnectedCallback(){this.listPoll?.stop(),super.disconnectedCallback()}willUpdate(){this.selected!==this.loadedFor&&(this.loadedFor=this.selected,this.turns=null,this.selected&&this.loadSession(this.selected))}async loadList(){try{let e=await D.sessions();this.list=e.sessions,this.live=e.live,this.error=null,this.selected&&this.loadSession(this.selected)}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async loadSession(e){try{let t=await D.session(e);e===this.selected&&(this.turns=t.turns,this.events=t.events??[])}catch(n){e===this.selected&&(this.turns=[]),this.error=t(n).message}}render(){if(this.error&&!this.list)return C`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`;if(!this.list)return d(6);let e=[...new Set(this.list.map(e=>e.client_kind??`other`))],t=this.list.filter(e=>(!this.query||e.key.toLowerCase().includes(this.query.toLowerCase()))&&(!this.client||e.client_kind===this.client));return C`
      ${this.selected?this.renderSession(this.selected):o}
      ${m(`Sessions, last 7 days`,t.length?C`<div class="table-scroll"><table class="table" data-sessions>
              <thead><tr><th>session</th><th>client</th><th>profile</th><th>class now</th><th class="num-col">turns</th><th class="num-col">notes</th><th class="num-col">monitor acts</th><th class="num-col">interventions</th><th>last checker</th><th>last turn</th></tr></thead>
              <tbody>${t.map(e=>C`<tr class="req-row ${e.key===this.selected?`is-open`:``}" data-session=${e.key} @click=${()=>S({session:e.key})}>
                  <td class="mono small cell-wrap">${E(e.key,28)}</td><td>${e.client_kind??`—`}</td><td>${e.profile??`—`}</td>
                  <td>${k(e.class_id,this.ids)}</td><td class="num num-col">${n(e.turns)}</td>
                  <td class="num num-col">${e.notes?`${e.notes} (${n(e.note_tokens)} tok)`:`—`}</td>
                  <td class="num num-col">${e.acts?C`${n(e.acts)}${e.gave_ups?C` <span class="badge badge-warn">${e.gave_ups} gave up</span>`:o}`:`—`}</td>
                  <td class="num num-col" data-interventions>${n(e.interventions??0)}</td>
                  <td>${e.last_checker?C`<span class="badge ${e.last_checker===`PASS`?`badge-ok`:`badge-warn`}">${e.last_checker}</span>`:`—`}</td>
                  <td class="num muted">${u(new Date(e.last_ms).toISOString())}</td></tr>`)}</tbody></table></div>`:v(`No sessions`,`Every TandemLLM request belongs to one: opencode sends its session id, other clients are grouped by their first two messages.`),{tools:C`<label class="field field-search">${h(`search`)}<input placeholder="search a key" .value=${this.query} @input=${e=>this.query=e.target.value} aria-label="search sessions" /></label>
            <label class="field"><span>client</span><select @change=${e=>this.client=e.target.value}><option value="">all</option>${e.map(e=>C`<option value=${e}>${e}</option>`)}</select></label>`,id:`cond-sessions`})}
      ${this.open?C`<qse-cond-decision .row=${this.open} .ids=${this.ids} .close=${()=>this.open=null}></qse-cond-decision>`:o}
    `}renderSession(e){let t=this.turns,i=this.live.find(t=>t.key===e);if(t==null)return d(4);let a=t.reduce((e,t)=>e+(t.outcome?.total_ms??1e3),0)||1,s=t.reduce((e,t)=>e+(t.injected?.length??0),0),c=this.events.filter(e=>b(e)).length,l=ve(t);return C`<section class="panel" style="margin-bottom:16px" data-timeline=${e}>
      <header class="panel-head"><div class="panel-titles"><h2 class="panel-title">${E(e,44)} ${i?C`<span class="badge badge-on">live</span>`:o}</h2>
        <p class="panel-sub">${t.length} turns · ${s} hidden note${s===1?``:`s`} · ${c} monitor act${c===1?``:`s`}${i?` · class now ${i.class_id} · ${n(i.note_tokens)} note tokens replayed a turn`:``}${t.length?` · since ${r(new Date(t[0].ts_ms).toISOString())}`:``}</p></div>
        <div class="panel-tools">
          <a class="btn btn-sm" href=${D.exportUrl(e,`md`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.md`} data-export="md">${h(`down`)} Markdown</a>
          <a class="btn btn-sm" href=${D.exportUrl(e,`json`)} download=${`session-${e.replace(/[^a-z0-9]+/gi,`-`)}.json`} data-export="json">${h(`down`)} JSON</a>
          <button class="btn btn-ghost btn-icon" aria-label="Close the session" @click=${()=>S({session:null})}>${h(`x`)}</button></div></header>
      ${this.axis(Ae(t,this.events,this.ids,e=>g(new Date(e).toISOString())))}
      ${this.tier1()}
      <div class="strip" role="list" aria-label="turns">
        ${t.map(e=>{let t=Math.max(18,Math.round((e.outcome?.total_ms??1e3)/a*900)),n=(e.injected?.length??0)>0,r=!!e.shadow,i=(e.events??[]).filter(e=>e.decision!==`PASS`),s=i.some(e=>b(e))?`act`:i.length?`shadow`:null;return C`<button class="strip-turn ${this.open?.id===e.id?`is-sel`:``}" role="listitem" style="width:${t}px; --cls: var(--series-${xe(e.class_id,this.ids)})" title=${`${e.class_id} · ${e.outcome?.finish_reason??`running`} · ${R(e)}`} @click=${()=>this.open=e} data-turn=${e.id}>
            ${n?C`<span class="mark"></span>`:r?C`<span class="mark is-shadow"></span>`:o}
            ${s?C`<span class="mark is-monitor ${s===`shadow`?`is-shadow`:``}" data-monitor-mark=${s}></span>`:o}
          </button>`})}
      </div>
      <p class="muted small">Width is the turn's time, colour its class. Below: ◆ the router put a hidden note in, ◇ shadow would have. Above, orange: the session monitor acted (◆) or would have (◇).</p>
      ${l.series.length?C`<qse-ribbon .series=${l.series} .labels=${l.labels} .height=${160} aria-label="checker gate values per turn"></qse-ribbon>`:o}
      <div class="feed" style="margin-top:10px">${t.slice().reverse().map(e=>ye(e,this.ids,e=>this.open=e,R(e)+((e.events??[]).some(e=>b(e))?` · monitor acted`:``)))}</div>
      ${this.events.length?C`<div style="margin-top:12px"><qse-cond-journal .session=${e}></qse-cond-journal></div>`:o}
    </section>`}axis(e){let t=e=>`${(e*100).toFixed(3)}%`;return C`<figure class="axis" data-axis aria-label="the session on one time axis">
      <div class="axis-lane axis-acts" aria-label=${`${e.acts.length} monitor decisions`}>
        ${e.acts.map(e=>C`<span class="axis-mark ${e.shadow?`is-shadow`:`is-act`}" style="left:${t(e.x)}" title=${e.title} data-axis-act=${e.shadow?`shadow`:`act`}></span>`)}
        ${e.notes.map(e=>C`<span class="axis-mark is-note ${e.shadow?`is-shadow`:``}" style="left:${t(e.x)}" title=${e.title} data-axis-note></span>`)}
      </div>
      <div class="axis-lane axis-turns" aria-label=${`${e.turns.length} turns`}>
        ${e.turns.map(e=>C`<span class="axis-turn ${e.compaction?`is-compaction`:``}" style="left:${t(e.x0)};width:${t(Math.max(0,e.x1-e.x0))};--cls:var(--series-${e.color})" title=${e.title} data-axis-turn></span>`)}
      </div>
      <div class="axis-lane axis-events" aria-label=${`${e.compactions.length} compactions, ${e.checkers.length} checker results`}>
        ${e.compactions.map(e=>C`<span class="axis-mark is-compaction" style="left:${t(e.x)}" title=${e.title} data-axis-compaction></span>`)}
        ${e.checkers.map(e=>C`<span class="axis-mark ${e.pass?`is-pass`:`is-fail`}" style="left:${t(e.x)}" title=${e.title} data-axis-checker=${e.pass?`pass`:`fail`}></span>`)}
      </div>
      <div class="axis-ticks" aria-hidden="true">${e.ticks.map((n,r)=>C`<span class="num ${r===0?`is-first`:r===e.ticks.length-1?`is-last`:``}" style="left:${t(n.x)}">${n.label.slice(0,5)}</span>`)}</div>
      <figcaption class="muted small">One time axis, ${e.ticks[0]?.label??``} to ${e.ticks[e.ticks.length-1]?.label??``}: ${e.turns.length} turns (the gaps are the client's time), ${e.acts.filter(e=>!e.shadow).length} monitor actions ◆ above (◇ shadow, ${e.acts.filter(e=>e.shadow).length}), ${e.notes.length} router note${e.notes.length===1?``:`s`}, ${e.compactions.length} compaction${e.compactions.length===1?``:`s`} ▲ and ${e.checkers.length} checker result${e.checkers.length===1?``:`s`} below (● pass, ○ fail; the gate numbers are in each one's title).</figcaption>
    </figure>`}tier1(){let e=oe(this.events);return e?C`<p class="small" data-tier1>Tier 1 shadow agreement: <b class="num">${e.agree} of ${e.labelled}</b> labelled decisions confirmed by their outcome <span class="muted">(${e.decisions} shadow decisions)</span></p>`:o}};s([i({attribute:!1})],Z.prototype,`ids`,void 0),s([i({attribute:!1})],Z.prototype,`selected`,void 0),s([l()],Z.prototype,`list`,void 0),s([l()],Z.prototype,`live`,void 0),s([l()],Z.prototype,`turns`,void 0),s([l()],Z.prototype,`events`,void 0),s([l()],Z.prototype,`error`,void 0),s([l()],Z.prototype,`query`,void 0),s([l()],Z.prototype,`client`,void 0),s([l()],Z.prototype,`open`,void 0),Z=s([a(`qse-cond-sessions`)],Z);var Q=class extends c{constructor(...e){super(...e),this.ids=[],this.tiles=[],this.onSaved=()=>void 0,this.stats=null,this.reports=[],this.engine={version:null,code_sha:null},this.error=null,this.msg=null,this.stage=`cascade`,this.abProfile=`coding`,this.abVersions=[],this.abA=null,this.abB=null,this.abChanges=null,this.abCounts={},this.suiteFile=null}connectedCallback(){super.connectedCallback(),this.load(),this.loadAb()}async load(){try{let[e,t]=await Promise.all([D.stats(168),D.evals()]);this.stats=e,this.reports=t.reports,this.engine=t.engine,this.error=null}catch(e){let n=t(e);this.error=`${n.endpoint}: ${n.message}`}}async promote(e,n,r,i,a){if(await A(`Promote ${r} to ${n===`task`?`act`:`on`} in ${e}?`,[n===`task`?`The session monitor will act on ${r} from the next request of this profile.`:`From the next request of ${e}, the router applies its ${r} decisions instead of only journaling them: ${r===`effort`?`budgets and thinking off go to the engine`:r===`packages`?`skills and packages go into the prompt`:r===`rules`?`rules go into the note`:`the class it picks sets the policy`}.`,`Precision ${i.toFixed(2)} over ${a} labelled decisions. A new profile version records it.`],`Promote`))try{let t=await D.doc(`profiles`,e),o=n===`task`?O(t.doc,[`tasks`,r,`mode`],`act`):O(t.doc,[`decisions`,r],`on`),s=await z(()=>D.put(`profiles`,e,o,`${r} to ${n===`task`?`act`:`on`}, precision ${i.toFixed(2)} / ${a}`,t.version));this.msg=`${e} v${s.version}: ${r} promoted`,this.onSaved()}catch(e){this.msg=t(e).message}}async loadAb(){try{this.abVersions=(await D.versions(`profiles`,this.abProfile)).versions,this.abA=this.abVersions[1]?.version??this.abVersions[0]?.version??0,this.abB=this.abVersions[0]?.version??0;let e=(await D.decisions({limit:500})).decisions.filter(e=>e.profile===this.abProfile),t={};for(let n of e)n.profile_version!=null&&(t[n.profile_version]=(t[n.profile_version]??0)+1);this.abCounts=t,await this.runAb()}catch{this.abVersions=[]}}async runAb(){if(this.abA!=null&&this.abB!=null)try{this.abChanges=(await D.diff(`profiles`,this.abProfile,this.abA,this.abB)).changes}catch(e){this.msg=t(e).message}}render(){return this.error&&!this.stats?C`<div class="state state-error" role="alert"><div class="state-title">${h(`warn`)} ${this.error}</div></div>`:this.stats?C`
      ${this.msg?C`<div class="note note-info" role="status">${this.msg}</div>`:o}
      ${this.renderPromotion(this.stats)}
      ${this.renderClassifier()}
      ${this.renderSuites()}
      ${this.renderAb()}
    `:d(8)}renderPromotion(e){let t=(e,t)=>this.tiles.find(t=>t.name===e)?.decisions[t]??`—`,r=(e.monitor?.precision??[]).filter(e=>e.mode===`shadow`);return m(`Shadow precision and promotion`,C`<p class="muted small">${N}. Precision is the share of labelled decisions a person marked right in the drawer; label them from the Overview feed or a session.</p>
        ${e.precision.length||r.length?C`<div class="table-scroll"><table class="table" data-precision>
              <thead><tr><th>profile</th><th>decision</th><th>mode now</th><th class="num-col">decisions</th><th class="num-col">labelled</th><th class="num-col">precision</th><th>rule</th><th></th></tr></thead>
              <tbody>
                ${e.precision.map(e=>{let r=je(e),i=t(e.profile,e.decision);return C`<tr data-precision-row=${`${e.profile}/${e.decision}`}>
                    <td>${e.profile}</td><td class="mono">${e.decision}</td><td><span class="badge ${i===`on`?`badge-on`:i===`shadow`?`badge-shadow`:``}">${i}</span></td>
                    <td class="num num-col">${n(e.decisions)}</td><td class="num num-col">${n(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${r.allowed?``:`muted`}">${r.why}</td>
                    <td><button class="btn btn-sm ${r.allowed?`btn-primary`:``}" ?disabled=${!r.allowed||i!==`shadow`} title=${r.allowed?``:N} data-promote=${`${e.profile}/${e.decision}`} @click=${()=>this.promote(e.profile??``,`decision`,e.decision,e.precision??0,e.labelled)}>Promote to on</button></td></tr>`})}
                ${r.map(e=>{let t=je(e);return C`<tr data-precision-row=${`task/${e.task}`}><td>—</td><td class="mono">${e.task}</td><td><span class="badge ${e.mode===`act`?`badge-on`:`badge-shadow`}">${e.mode}</span></td>
                    <td class="num num-col">${n(e.decisions)}</td><td class="num num-col">${n(e.labelled)}</td><td class="num num-col">${e.precision==null?`—`:e.precision.toFixed(2)}</td>
                    <td class="small ${t.allowed?``:`muted`}">${t.why}</td>
                    <td><button class="btn btn-sm ${t.allowed?`btn-primary`:``}" ?disabled=${!t.allowed||e.mode!==`shadow`} data-promote=${`task/${e.task}`} @click=${()=>this.promote(this.tiles.find(e=>!e.locked&&e.name===`coding`)?.name??`coding`,`task`,e.task,e.precision??0,e.labelled)}>Promote to act</button></td></tr>`})}
              </tbody></table></div>`:v(`Nothing in shadow this week`,`Set a decision type to shadow in a profile; its decisions show up here with their labels.`)}
        ${e.monitor?o:C`<p class="muted small">The session monitor's tasks (D1, E1, C3…) appear here once its journal runs on this server (SRV-115).</p>`}`,{id:`cond-precision`})}renderClassifier(){let t=this.reports.find(e=>e.kind===`classifier`);if(!t)return m(`Classifier accuracy`,v(`No report yet`,`tools/conductor_eval.py runs the labelled set against each stage and writes the report the box keeps under the conductor directory (evals/).`),{id:`cond-classifier`});let n=t.stages??{},i=t.set??{},a=(n[this.stage]??Object.values(n)[0])?.confusion??{},s=i.classes??Object.keys(a);return m(C`Classifier accuracy <span class="muted small num">${i.name??``} · ${i.n??`?`} prompts · ${t.created_at?r(String(t.created_at)):``}</span>`,C`<div class="table-scroll"><table class="table" data-classifier-report>
          <thead><tr><th>stage</th><th class="num-col">accuracy</th><th class="num-col">answered</th><th class="num-col">accuracy when it answered</th><th class="num-col">p50</th><th class="num-col">p95</th></tr></thead>
          <tbody>${Object.entries(n).map(([t,n])=>C`<tr class="req-row ${t===this.stage?`is-open`:``}" @click=${()=>this.stage=t}><td class="mono">${t}</td><td class="num num-col">${e(n.accuracy,1)}</td><td class="num num-col">${n.answered} / ${n.n}</td><td class="num num-col">${n.accuracy_answered==null?`—`:e(n.accuracy_answered,1)}</td><td class="num num-col">${_(n.p50_ms)}</td><td class="num num-col">${_(n.p95_ms)}</td></tr>`)}</tbody></table></div>
        ${s.length&&Object.keys(a).length?C`<p class="muted small" style="margin-top:12px">Confusion of <strong>${this.stage}</strong>: rows are the true class, columns what the stage answered (— = no answer).</p>
            <div class="table-scroll"><table class="table cm">
              <thead><tr><th></th>${[...s,`—`].map(e=>C`<th class="mono small">${e}</th>`)}</tr></thead>
              <tbody>${s.map(e=>C`<tr><th>${k(e,this.ids)}</th>${[...s,`—`].map(t=>{let n=a[e]?.[t]??0;return C`<td class="cm-cell ${e===t?`cm-diag`:n?`cm-miss`:``}">${n||``}</td>`})}</tr>`)}</tbody></table></div>`:C`<p class="muted small" style="margin-top:12px">No confusion table for <strong>${this.stage}</strong>; pick another stage.</p>`}
        ${se(t).length?C`<p class="muted small" style="margin-top:12px">Digest variants (SRV-153): <strong>today</strong> the 2,000-character digest the decider cut from the front, <strong>sized</strong> each stage's own head and tail, <strong>line</strong> sized plus the context line. <strong>router</strong> is the session's class after the stickiness rule.</p>
            <div class="table-scroll"><table class="table" data-classifier-variants>
              <thead><tr><th>digest</th><th>set</th><th class="num-col">n</th><th class="num-col">decider</th><th class="num-col">cascade</th><th class="num-col">router</th><th class="num-col">decider p50</th></tr></thead>
              <tbody>${se(t).map(t=>C`<tr><td class="mono">${t.variant}</td><td class="mono small">${t.set}</td><td class="num num-col">${t.n}</td><td class="num num-col">${t.decider==null?`—`:e(t.decider,1)}</td><td class="num num-col">${t.cascade==null?`—`:e(t.cascade,1)}</td><td class="num num-col">${t.router==null?`—`:e(t.router,1)}</td><td class="num num-col">${_(t.p50)}</td></tr>`)}</tbody></table></div>`:o}`,{id:`cond-classifier`})}renderSuites(){let t=this.reports.filter(e=>e.kind===`suite`),r=pe(this.reports,this.suiteFile),i=P(r,t),a=r?.rows??[],s=F(this.reports,this.engine),c=re(a),l=r?.verdict,u=r?.comparison??[],d=r?.raw_row,f=r?.build,p=l?.indicative??Se(a),g=(e,t=1)=>e==null?`—`:Number.isInteger(e)?n(e):e.toFixed(t);return m(C`TandemLLM against the raw engine ${r?C`<span class="muted small num">night ${r.night??`—`}${f?` · ${f.version??`?`} ${E(f.code_sha,12)}`:``}${r.profile?` · profile ${r.profile}`:``}</span>`:o}`,C`${s.tone===`ok`?C`<p class="small" data-guard-detail=${s.status}>${h(`check`)} ${s.line}${s.detail?C`: <span class="num">${s.detail}</span>`:o}.</p>`:C`<div class="guard-line ${s.tone===`note`?`guard-note`:``}" data-guard-detail=${s.status}>${h(s.tone===`note`?`info`:`warn`)}<span>${s.line}${s.detail?C`. <span class="small">${s.detail}</span>`:o}</span></div>`}
        ${t.length>1?C`<label class="field suite-pick"><span>report</span><select aria-label="suite report" data-suite-report @change=${e=>this.suiteFile=e.target.value}>${t.map(e=>C`<option value=${String(e.file)} ?selected=${e===r}>${String(e.night??e.file)} · ${String(e.file)}${P(e,t)?` · superseded`:``}</option>`)}</select></label>`:o}
        ${i?C`<div class="guard-line guard-note" data-superseded>${h(`info`)}<span><span class="badge badge-warn">superseded</span> by <span class="mono">${i.file}</span>${i.why?C`: <span class="small">${i.why}</span>`:o}. Its numbers are kept as published; read the newer report.</span></div>`:o}
        ${l?C`<div class="note" data-eval-verdict>
              <span class="badge ${l.better?`badge-ok`:l.better===!1?`badge-warn`:``}"><strong>${l.better?`Better`:l.better===!1?`Not better`:`Not shown yet`}</strong></span>${p?C` <span class="badge badge-warn" data-indicative>indicative: n &lt; 10</span>`:o} — ${l.why}
              <div class="muted small" data-raw-row>raw never-worse row: ${d?.status??`—`}${d?.source?` (${d.source})`:``}${r?.driver?` · driver ${r.driver.mode}`:``}${r?.isolation?` · ${r.isolation.store_clears.length} store clears between arms`:``}</div>
            </div>`:o}
        ${c.length?C`<div class="table-scroll" style="margin-top:12px"><table class="table" data-suites>
              <thead><tr><th>suite</th><th>arm</th><th class="num-col">n</th><th class="num-col">pass rate</th><th class="num-col">gates passed</th><th class="num-col">minutes to PASS</th><th class="num-col">wall min</th><th class="num-col">tokens</th><th class="num-col">turns</th><th class="num-col">thinking tokens</th><th class="num-col">decode tok/s</th><th class="num-col">empty stops</th><th class="num-col">continuations</th><th class="num-col">interventions</th><th class="num-col">gave up</th><th class="num-col">cached</th><th>efforts (turns)</th><th>build · night</th></tr></thead>
              <tbody>${c.flatMap(t=>{let r=u.find(e=>e.suite===t.suite||e.suite_id===(t.raw??t.tandem)?.suite_id);return[t.raw,t.tandem].filter(Boolean).map((i,a)=>C`<tr data-suite=${t.suite} data-arm=${i.arm}>
                    <td>${a===0?t.suite:``}${a===0&&!t.cmp.ok?C` <span class="badge badge-warn" data-not-comparable>not comparable</span><div class="warn-ink small">${t.cmp.why}</div>`:o}${a===0&&r?C` <span class="badge ${r.verdict===`better`?`badge-ok`:r.verdict===`worse`||r.verdict===`mixed`?`badge-warn`:``}" data-suite-verdict=${r.verdict}>${r.verdict}</span>`:o}${a===0&&(i.indicative??(i.n??0)<10)?C` <span class="badge" data-row-indicative>indicative</span>`:o}</td>
                    <td class="mono">${Ee(i.arm)}${i.pristine===!1?C` <span class="badge badge-warn" title="a checker or reference file changed during a run">not pristine</span>`:o}</td>
                    <td class="num num-col">${i.n??`—`}</td>
                    <td class="num num-col">${i.pass_rate==null?`—`:e(i.pass_rate,0)}</td>
                    <td class="num num-col">${i.gates_passed??`—`}${i.gates_total?` / ${i.gates_total}`:``}</td>
                    <td class="num num-col">${g(i.minutes_to_pass)}</td><td class="num num-col">${g(i.wall_minutes)}</td>
                    <td class="num num-col">${i.tokens==null?`—`:n(Math.round(i.tokens))}</td>
                    <td class="num num-col" data-turns>${g(i.turns)}</td><td class="num num-col" data-thinking>${i.reasoning_tokens==null?`—`:n(Math.round(i.reasoning_tokens))}</td>
                    <td class="num num-col">${g(i.decode_tps)}</td>
                    <td class="num num-col">${i.empty_stops??`—`}</td><td class="num num-col">${i.continuations??`—`}</td>
                    <td class="num num-col">${i.interventions??`—`}</td><td class="num num-col">${i.gave_ups??`—`}</td>
                    <td class="num num-col">${i.cache_hit==null?`—`:e(i.cache_hit,0)}</td>
                    <td class="small" data-efforts>${ge(i)}</td>
                    <td class="small">${E(i.build,16)} · ${i.night??`—`}</td></tr>`)})}</tbody></table></div>
              <p class="muted small">Better means a pass rate not lower and minutes to PASS not higher on every suite, with the raw row unchanged; n under 10 is indicative only. Continuations are the ones the watch-only driver would have sent; interventions are the conductor's acts. Tokens, turns and thinking tokens are medians per run.</p>`:v(`No suite runs yet`,`OPS-43 (tools/tandem_eval.py) writes both arms per suite and keeps the report in the box's evals/ directory (--publish).`)}`,{id:`cond-suites`})}renderAb(){return m(`A/B of two profile versions`,C`<div class="toolbar" style="margin-top:0">
          <label class="field"><span>profile</span><select @change=${e=>{this.abProfile=e.target.value,this.loadAb()}}>${this.tiles.filter(e=>!e.locked).map(e=>C`<option value=${e.name} ?selected=${e.name===this.abProfile}>${e.name}</option>`)}</select></label>
          ${[`abA`,`abB`].map(e=>C`<label class="field"><span>${e===`abA`?`A`:`B`}</span><select @change=${t=>{this[e]=Number(t.target.value),this.runAb()}}>${this.abVersions.map(t=>C`<option value=${t.version} ?selected=${t.version===this[e]}>v${t.version} ${E(t.note,30)}</option>`)}</select></label>`)}
        </div>
        <p class="small num">decisions in the last 500: A ${this.abCounts[this.abA??-1]??0} · B ${this.abCounts[this.abB??-1]??0}</p>
        ${this.abChanges?this.abChanges.length?C`<div class="diff" data-ab>${this.abChanges.map(e=>C`<div class="diff-row"><span>${ue(e.path)}</span><span class="diff-before">${E(e.before,80)}</span><span class="diff-after">${E(e.after,80)}</span></div>`)}</div>`:C`<p class="muted small">The two versions are the same.</p>`:o}`,{id:`cond-ab`})}};s([i({attribute:!1})],Q.prototype,`ids`,void 0),s([i({attribute:!1})],Q.prototype,`tiles`,void 0),s([i({attribute:!1})],Q.prototype,`onSaved`,void 0),s([l()],Q.prototype,`stats`,void 0),s([l()],Q.prototype,`reports`,void 0),s([l()],Q.prototype,`engine`,void 0),s([l()],Q.prototype,`error`,void 0),s([l()],Q.prototype,`msg`,void 0),s([l()],Q.prototype,`stage`,void 0),s([l()],Q.prototype,`abProfile`,void 0),s([l()],Q.prototype,`abVersions`,void 0),s([l()],Q.prototype,`abA`,void 0),s([l()],Q.prototype,`abB`,void 0),s([l()],Q.prototype,`abChanges`,void 0),s([l()],Q.prototype,`abCounts`,void 0),s([l()],Q.prototype,`suiteFile`,void 0),Q=s([a(`qse-cond-compare`)],Q);var Ye=[{id:`overview`,label:`Overview`},{id:`routing`,label:`Routing`},{id:`monitor`,label:`Monitor`},{id:`profiles`,label:`Profiles`},{id:`packages`,label:`Packages`},{id:`deciders`,label:`Deciders`},{id:`sessions`,label:`Sessions`},{id:`compare`,label:`Compare`}],$=class extends c{constructor(...e){super(...e),this.params=new URLSearchParams,this.tiles=[],this.defaultProfile=`chat`,this.meta=null,this.config=null,this.ids=[],this.guard=null,this.error=null,this.switching=!1,this.reloaded=null,this.switchError=null,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=p(()=>this.load(),15e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}get screen(){let e=this.params.get(`screen`);return e&&Ye.some(t=>t.id===e)?e:`overview`}get active(){return this.params.get(`profile`)??this.defaultProfile}async load(){try{let[e,t,n,r,i]=await Promise.all([D.profiles(),this.meta?Promise.resolve(this.meta):D.meta(),D.config(),D.doc(`router`),D.evals()]);this.tiles=e.profiles,this.defaultProfile=e.default_profile,this.meta=t,this.config=n,this.ids=r.doc.classes.map(e=>e.id),this.guard=F(i.reports,i.engine),this.error=null}catch(e){let n=t(e);this.error=n.status===404?`The conductor is off on this server (start it with --conductor-config).`:`${n.endpoint}: ${n.message}`}}async setMode(e){let n=this.tiles.find(e=>e.name===this.active);if(n&&!n.locked&&(e!==`on`||await A(`Turn the ${n.name} profile on?`,[`Every decision type set to on in this profile will change what the model sees on its next request: effort, hidden skills and packages, rules.`,`Shadow decisions stay in shadow. Requests sent with X-Tandem: raw (benchmarks) are not affected.`],`Turn on`))){this.switching=!0,this.switchError=null;try{let t=await z(()=>D.setMode(n.name,e,`mode to ${e}`,n.version));this.reloaded=`v${t.version} · reloaded ${t.loaded_at?g(t.loaded_at):`now`}`,await this.load()}catch(e){this.switchError=t(e).message}finally{this.switching=!1}}}render(){let e=this.tiles.find(e=>e.name===this.active),t=this.config?Object.entries(this.config.store.documents).filter(([,e])=>e.error):[];return C`<div class="view view-conductor">
      <div class="cond-head">
        <div class="cond-head-row">
          <div class="cond-title">${h(`conductor`)} Conductor <span class="muted small num" title=${`the one model; ${this.config?.server.raw_model??`the raw engine name`} is a hidden alias, and X-Tandem: raw serves the raw path`}>TandemLLM, the one model · raw by X-Tandem: raw</span></div>
          <div class="mode-switch" data-mode-switch>
            <label class="field"><span>profile</span>
              <select aria-label="active profile" @change=${e=>S({profile:e.target.value})}>
                ${this.tiles.map(e=>C`<option value=${e.name} ?selected=${e.name===this.active}>${e.name}${e.locked?` (locked)`:``}</option>`)}
              </select>
            </label>
            ${e?I([`raw`,`shadow`,`on`],e.mode,e=>void this.setMode(e),{disabled:!!e.locked||this.switching,label:`mode of ${e.name}`}):o}
            ${this.reloaded?C`<span class="reload" role="status" data-reloaded>${this.reloaded}</span>`:o}
          </div>
        </div>
        ${this.switchError?C`<div class="guard-line" role="alert">${h(`warn`)} ${this.switchError}</div>`:o}
        ${this.error?C`<div class="guard-line" role="alert">${h(`warn`)} ${this.error}</div>`:o}
        ${this.guard&&this.guard.tone!==`ok`?C`<div class="guard-line ${this.guard.tone===`note`?`guard-note`:``}" data-guard=${this.guard.status}>${h(this.guard.tone===`note`?`info`:`warn`)}<span>${this.guard.line}${this.guard.detail?C`. <span class="small">${this.guard.detail}</span>`:o}</span></div>`:o}
        ${t.map(([e,t])=>C`<div class="guard-line" role="alert">${h(`warn`)} ${e}: ${t.error?.message} (kept version ${t.version})</div>`)}
        ${e?.locked?C`<div class="note" data-locked>${h(`warn`)} ${e.name} is locked: ${e.locked}.</div>`:o}
        <nav class="cond-tabs" aria-label="conductor screens">
          ${Ye.map(e=>C`<a class="cond-tab ${e.id===this.screen?`is-on`:``}" href=${`#/conductor?screen=${e.id}${this.params.get(`profile`)?`&profile=${encodeURIComponent(this.active)}`:``}`} aria-current=${e.id===this.screen?`page`:o}>${e.label}</a>`)}
        </nav>
      </div>
      ${this.renderScreen()}
    </div>`}renderScreen(){let e={ids:this.ids,meta:this.meta,tiles:this.tiles};switch(this.screen){case`overview`:return C`<qse-cond-overview .ids=${e.ids} .tiles=${e.tiles} .active=${this.active}></qse-cond-overview>`;case`routing`:return C`<qse-cond-routing .ids=${e.ids} .meta=${e.meta} .active=${this.active} .onSaved=${()=>this.load()}></qse-cond-routing>`;case`monitor`:return C`<qse-cond-monitor .meta=${e.meta} .onSaved=${()=>this.load()}></qse-cond-monitor>`;case`profiles`:return C`<qse-cond-profiles .ids=${e.ids} .meta=${e.meta} .tiles=${e.tiles} .selected=${this.params.get(`edit`)} .onSaved=${()=>this.load()}></qse-cond-profiles>`;case`packages`:return C`<qse-cond-packages .ids=${e.ids}></qse-cond-packages>`;case`deciders`:return C`<qse-cond-deciders></qse-cond-deciders>`;case`sessions`:return C`<qse-cond-sessions .ids=${e.ids} .selected=${this.params.get(`session`)}></qse-cond-sessions>`;case`compare`:return C`<qse-cond-compare .ids=${e.ids} .tiles=${e.tiles} .onSaved=${()=>this.load()}></qse-cond-compare>`}}};s([i({attribute:!1})],$.prototype,`params`,void 0),s([l()],$.prototype,`tiles`,void 0),s([l()],$.prototype,`defaultProfile`,void 0),s([l()],$.prototype,`meta`,void 0),s([l()],$.prototype,`config`,void 0),s([l()],$.prototype,`ids`,void 0),s([l()],$.prototype,`guard`,void 0),s([l()],$.prototype,`error`,void 0),s([l()],$.prototype,`switching`,void 0),s([l()],$.prototype,`reloaded`,void 0),s([l()],$.prototype,`switchError`,void 0),$=s([a(`qse-conductor`)],$);export{$ as QseConductor};