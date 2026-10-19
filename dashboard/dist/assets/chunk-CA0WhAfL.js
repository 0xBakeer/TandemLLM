import{c as e,d as t,f as n,g as r,l as i,o as a,p as o,r as s,u as c,v as l}from"./chunk-5VjiyoiJ.js";import{c as u,d,f,l as p,r as m,u as h}from"./chunk-D1RW2Riv.js";import{c as g,d as _,f as v,i as y,u as b}from"./chunk-BXmDkkzu.js";import{i as x,n as S}from"./chunk-CI4nmvus.js";import{Ct as C,Tt as w}from"./app-CRhkfnNj.js";var T=[`cobalt`,`aqua`,`orange`,`violet`,`yellow`,`red`,`muted`],E=class{constructor(){this.map=new Map}of(e){let t=this.map.get(e);if(!t){if(e===`engine`)t=`cobalt`;else{let e=new Set(this.map.values());t=T.find(t=>t!==`cobalt`&&!e.has(t))??T[this.map.size%(T.length-1)+1]}this.map.set(e,t)}return t}seed(e){for(let t of[...e].sort((e,t)=>e===`engine`?-1:t===`engine`?1:e.localeCompare(t)))this.of(t)}},D=class{constructor(e=12e3,t=3e3){this.trimMs=e,this.holdMs=t,this.slots=[],this.where=new Map}update(e,t,n){let r=new Map;for(let e of t)(e.ended_ms??0)<=this.holdMs&&r.set(e.id,e);for(let t of e)r.set(t.id,t);for(let[e,t]of[...this.where])r.has(e)||(this.where.delete(e),this.slots[t]={lane:null,freeSince:n});for(let[e,t]of this.where)this.slots[t]={lane:r.get(e),freeSince:null};let i=[...r.values()].filter(e=>!this.where.has(e.id)&&e.state!==`done`);i.sort((e,t)=>(t.elapsed_ms??0)-(e.elapsed_ms??0));for(let e of i){let t=this.slots.findIndex(e=>e.lane==null);t<0&&(t=this.slots.length,this.slots.push({lane:null,freeSince:null})),this.slots[t]={lane:e,freeSince:null},this.where.set(e.id,t)}for(;this.slots.length;){let e=this.slots[this.slots.length-1];if(e.lane==null&&e.freeSince!=null&&n-e.freeSince>=this.trimMs)this.slots.pop();else break}return this.slots}};function O(e,t,n=300){let r=e.length?e[e.length-1].t:-1/0,i=e.concat(t.filter(e=>e.t>r)),a=i.length?i[i.length-1].t:0,o=0;for(;o<i.length&&a-i[o].t>n;)o++;return o?i.slice(o):i}function k(e,t){let n=e.map(()=>0);return t.map(t=>({id:t,bands:e.map((e,r)=>{let i=e.by[t]??0,a=n[r];return n[r]=a+i,[a,a+i]})}))}function A(e,t){let n=new Set(t.map(e=>e.id));for(let t of e)for(let e of Object.keys(t.by))n.add(e);return[...n].sort((e,t)=>e===`engine`?-1:t===`engine`?1:e.localeCompare(t))}var j={queued:`Queued`,waiting_memory:`Waiting for memory`,prefilling:`Prefilling`,waiting:`Waiting for the first token`,replaying:`Replaying`,thinking:`Thinking`,closing_reasoning:`Closing the reasoning`,tool_call:`Tool call`,writing:`Writing`,finishing:`Finishing`,done:`Done`};function M(e){switch(e.state){case`queued`:case`waiting`:case`waiting_memory`:return`wait`;case`prefilling`:case`replaying`:return`prefill`;case`thinking`:case`closing_reasoning`:return`think`;case`tool_call`:return`tool`;case`done`:return e.finish===`error`||e.finish===`timeout`||(e.status??200)>=500?`error`:`done`;default:return`write`}}function N(e){return e.state===`tool_call`?e.tool?`Tool call: ${e.tool}`:`Tool call`:e.state===`prefilling`&&e.prefill?.pct!=null?`Prefilling ${Math.round(e.prefill.pct)} %`:e.state===`done`?e.finish&&e.finish!==`stop`?`Done: ${e.finish.replace(`_`,` `)}`:`Done`:j[e.state]??e.state}function P(e,t){let n=e.filter(e=>e.state!==`done`);return`${n.filter(e=>M(e)===`wait`).length} waiting · ${n.filter(e=>M(e)===`prefill`).length} prefilling · ${t} decoding`}function F(e,t,n=2){if(e.decode_tps!==0||!e.decoding)return null;let r=t.find(e=>e.state===`prefilling`&&e.backend===`engine`);if(!r)return`${e.decoding} decoding, no token in the last ${n} s`;let i=r.prefill,a=i?.done!=null&&i.total?` (${Math.round(100*i.done/i.total)} % of ${i.total.toLocaleString(`en-US`)})`:``;return`${e.decoding} decoding, stalled behind a prefill${a}`}function I(e){return{tandemllm:`TandemLLM`,vllm:`vLLM`,sglang:`SGLang`,deepseek:`DeepSeek`,openai:`OpenAI`,generic:`OpenAI-compatible`}[e]??e}function L(e,t){if(e.tokens==null||t<=0)return null;let n=Math.min(e.thinking_tokens??0,e.tokens);return{think:Math.min(1,n/t),out:Math.min(1,(e.tokens-n)/t)}}function R(e,t=0){let n=Math.max(256,...e.map(e=>e.tokens??0));return t>=n&&t<=n*4?t:Math.max(256,2**Math.ceil(Math.log2(n)))}function z(e){switch(e.tokens_source){case`engine`:return`counted by the engine`;case`frames`:return`stream frames, checked to carry one token each on this backend`;case`usage`:return`the backend's usage`;default:return e.state===`done`?`the backend sent no usage`:`not reported live: this backend's frames are not checked as one token each yet`}}function B(e){return e.busy?e.phase===`decode`?`decoding`:e.phase===`solo`?`decoding alone`:`busy`:`free`}function V(e){return e>=1024?`${Math.round(e/1024)}k`:String(e)}function H(e){if(!e)return`none yet`;let t=Object.entries(e).filter(([,e])=>e>0).sort(([e],[t])=>Number(e)-Number(t)).map(([e,t])=>`${e}-way ${t}`);return t.length?t.join(` · `):`none yet`}var U=class extends h{constructor(...e){super(...e),this.series=[],this.order=[],this.color=()=>`cobalt`,this.label=e=>e,this.span=300,this.now=0,this.hover=null,this.glowId=f(`st`)}onMove(e,t){let n=this.getBoundingClientRect(),r=`touches`in e?e.touches[0]?.clientX:e.clientX;if(r==null||!t.length)return;let i=r-n.left,a=0;for(let e=1;e<t.length;e++)Math.abs(t[e]-i)<Math.abs(t[a]-i)&&(a=e);this.hover=a}render(){let e=this.width||600,t=this.height,n={l:46,r:10,t:10,b:24},i=n.l,a=e-n.r,s=this.now||(this.series.length?this.series[this.series.length-1].t:Date.now()/1e3),c=s-this.span,f=this.series.filter(e=>e.t>=c-2),h=[];for(let e=0;e<f.length;e++){let t=f[e],n=h[h.length-1];n&&t.t-n.t>3&&h.push({t:n.t+1,by:{},in_flight:0},{t:t.t-1,by:{},in_flight:0}),h.push(t)}let g=k(h,this.order),_=h.map((e,t)=>g.length?g[g.length-1].bands[t][1]:0),v=u(Math.max(10,..._)),y=m([0,v],[t-n.b,n.t]),b=m([c,s],[i,a]),x=p(0,v,4),S=h.map(e=>b(e.t)),C=e=>{if(!e.length)return``;let t=`M${S[0].toFixed(1)},${y(e[0][1]).toFixed(1)}`;for(let n=1;n<e.length;n++)t+=`L${S[n].toFixed(1)},${y(e[n][1]).toFixed(1)}`;for(let n=e.length-1;n>=0;n--)t+=`L${S[n].toFixed(1)},${y(e[n][0]).toFixed(1)}`;return t+`Z`},w=e=>e.map((e,t)=>`${t?`L`:`M`}${S[t].toFixed(1)},${y(e).toFixed(1)}`).join(``),T=this.hover!=null&&this.hover<h.length?this.hover:null,E=this.span>120?60:30,D=Array.from({length:Math.floor(this.span/E)+1},(e,t)=>t*E);return r`<div
      class="chart chart-stack"
      tabindex="0"
      role="img"
      aria-label="decode tokens per second by backend, stacked, the last ${this.span} seconds"
      @mousemove=${e=>this.onMove(e,S)}
      @mouseleave=${()=>this.hover=null}
      @touchstart=${e=>this.onMove(e,S)}
      @touchmove=${e=>this.onMove(e,S)}
    >
      <svg width=${e} height=${t} viewBox="0 0 ${e} ${t}">
        <defs>
          <filter id="${this.glowId}-glow" x="-5%" y="-40%" width="110%" height="180%"><feGaussianBlur stdDeviation="4" /></filter>
        </defs>
        ${x.map(e=>l`<line class="grid" x1=${i} x2=${a} y1=${y(e)} y2=${y(e)} /><text class="axis" x=${i-6} y=${y(e)+3.5} text-anchor="end">${e>=1e3?`${(e/1e3).toFixed(e%1e3?1:0)}k`:e}</text>`)}
        ${D.map(e=>l`<text class="axis" x=${b(s-e)} y=${t-7} text-anchor=${e===0?`end`:e===this.span?`start`:`middle`}>${e===0?`now`:e%60?`−${e} s`:`−${e/60} min`}</text>`)}
        ${h.length?o:l`<text class="axis chart-nodata" x=${(i+a)/2} y=${t/2} text-anchor="middle">nothing decoded in this window</text>`}
        ${g.map(e=>{let t=d(this.color(e.id));return l`<path class="stack-area" data-layer=${e.id} d=${C(e.bands)} fill=${t} /><path d=${w(e.bands.map(e=>e[1]))} fill="none" stroke=${t} stroke-width="1.4" stroke-linejoin="round" />`})}
        ${h.length?l`<path d=${w(_)} fill="none" stroke="var(--ink)" stroke-width="7" opacity="0.10" filter="url(#${this.glowId}-glow)" /><path d=${w(_)} fill="none" stroke="var(--ink)" stroke-width="1.1" opacity="0.55" stroke-linejoin="round" />`:o}
        ${T==null?o:l`<line class="crosshair" x1=${S[T]} x2=${S[T]} y1=${n.t} y2=${t-n.b} />`}
      </svg>
      ${T==null?o:r`<div class="tip" style="left:${Math.min(Math.max(S[T],110),e-110)}px" role="status">
            <div class="tip-title">${Math.max(0,Math.round(s-h[T].t))} s ago · ${Math.round(_[T])} tok/s</div>
            ${[...this.order].reverse().map(e=>{let t=h[T].by[e];return t?r`<div class="tip-row"><span class="swatch" style="background:${d(this.color(e))}"></span><span>${this.label(e)}</span><span class="num">${Math.round(t)} tok/s</span></div>`:o})}
          </div>`}
    </div>`}};e([t({attribute:!1})],U.prototype,`series`,void 0),e([t({attribute:!1})],U.prototype,`order`,void 0),e([t({attribute:!1})],U.prototype,`color`,void 0),e([t({attribute:!1})],U.prototype,`label`,void 0),e([t({type:Number})],U.prototype,`span`,void 0),e([t({type:Number})],U.prototype,`now`,void 0),e([c()],U.prototype,`hover`,void 0),U=e([n(`qse-stack`)],U);var W=`not reported`,G=[`wait`,`prefill`,`think`,`tool`,`write`],K={wait:`waiting`,prefill:`prefilling`,think:`thinking`,tool:`tool call`,write:`writing`};function q(e){return e==null?`—`:e>=100?Math.round(e).toLocaleString(`en-US`):e.toFixed(1)}var J=class extends i{constructor(...e){super(...e),this.doc=null,this.series=[],this.error=null,this.bytes=0,this.slots=new D,this.colors=new E,this.scale=0,this.span=300,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=w(()=>this.load(),1e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=this.series.length?this.series[this.series.length-1].t:null,t=await S.parallel(e);this.bytes=JSON.stringify(t).length,this.series=O(this.series,t.series??[]),this.colors.seed(A(this.series,t.backends)),this.slots.update(t.lanes,t.recent,Date.now()),this.doc=t,this.error=null}catch(e){let t=x(e);this.error={endpoint:t.endpoint||`/v1/dashboard/parallel`,message:t.message}}}render(){let e=this.doc;if(!e)return r`<div class="par">${this.error?s(this.error.endpoint,this.error.message,()=>this.poller?.refresh()):a(6,`skeleton-live`)}</div>`;let t=A(this.series,e.backends),n=this.series.length?this.series[0].t:e.at;return this.span=Math.min(300,Math.max(60,Math.ceil((e.at-n+5)/30)*30)),r`<div class="par" data-in-flight=${e.totals.in_flight}>
      ${this.renderHero(e,t)}
      ${this.renderKpis(e)}
      ${this.renderBackends(e)}
      ${this.renderLanes(e)}
      ${this.error?r`<p class="par-foot warn-ink" role="status">${this.error.message} (the last numbers stay until the next poll answers)</p>`:o}
    </div>`}renderHero(e,t){let n=e.totals,i=e.lanes,a=Object.fromEntries(G.map(e=>[e,i.filter(t=>M(t)===e).length])),s=new Map(e.backends.map(e=>[e.id,e])),c=F(n,i,e.windows.now_s);return r`<section class="panel par-hero" aria-label="aggregate decode">
      <div class="par-hero-num">
        <div class="par-kicker">Aggregate decode, now</div>
        <div class="par-big ${n.decode_tps==null?`is-na`:``}" data-total-tps>
          <span class="num">${n.decode_tps==null?n.decoding?W:`0`:q(n.decode_tps)}</span><span class="par-big-unit">tok/s</span>
        </div>
        <div class="par-hero-sub">
          <span><b class="num">${n.in_flight}</b> in flight</span>
          <span><b class="num">${n.decoding}</b> decoding</span>
          <span><b class="num">${n.backends}</b> backend${n.backends===1?``:`s`}</span>
          <span class="par-nr ${n.decode_unreported?``:`is-hidden`}" title="no live count for these (server/parallel.py): the usage at the end has the exact one">${n.decode_unreported} not counted live</span>
          <span class=${n.prefill_tps==null?`is-hidden`:``}>prefill <b class="num">${q(n.prefill_tps)}</b> tok/s</span>
        </div>
        <div class="par-stall warn-ink" data-stall role="status" title="no decoding lane wrote a token over the last ${e.windows.now_s} s: the rates count tokens, so this 0 is a real stall">${c??`\xA0`}</div>
        <div class="par-mix" role="img" aria-label="requests in flight by state">
          <div class="par-mix-bar">
            ${G.map(e=>r`<span class="par-mix-seg tone-${e}" style="flex-grow:${a[e]};flex-basis:0;${a[e]?``:`padding:0;`}" title="${a[e]} ${K[e]}"></span>`)}
            ${i.length?o:r`<span class="par-mix-seg tone-idle" style="flex-grow:1"></span>`}
          </div>
          <div class="par-mix-legend">
            ${G.map(e=>r`<span class="par-mix-item ${a[e]?``:`is-zero`}" data-tone=${e}><span class="par-dot tone-${e}"></span>${K[e]} <b class="num">${a[e]}</b></span>`)}
          </div>
        </div>
      </div>
      <div class="par-hero-chart">
        <qse-stack .series=${this.series} .order=${t} .color=${e=>this.colors.of(e)} .label=${e=>e} .now=${e.at} .span=${this.span} height="196"></qse-stack>
        <div class="par-legend">
          ${t.map(e=>{let t=s.get(e);return r`<span class="par-legend-item" data-backend=${e}><span class="swatch" style="background:${d(this.colors.of(e))}"></span>${e}<span class="num">${t?.decode_tps==null?t?.decoding?W:t?.in_flight?`waiting`:`idle`:`${q(t.decode_tps)} tok/s`}</span></span>`})}
        </div>
      </div>
    </section>`}renderKpis(e){let t=e.totals,n=Math.round(e.windows.stats_s/60),i=(e,t,n,i,a=!1)=>r`<div class="stat ${a?`is-na`:``}" data-kpi=${i}><div class="stat-label">${e}</div><div class="stat-value">${t}</div><div class="stat-sub">${n}</div></div>`;return r`<div class="stats par-kpis">
      ${i(`In flight`,r`<span class="num">${t.in_flight}</span>`,P(e.lanes,t.decoding),`in-flight`)}
      ${i(`TTFT p50 · p90`,t.ttft_p50==null?`none yet`:r`<span class="num">${_(t.ttft_p50)}</span><span class="stat-unit">· ${_(t.ttft_p90)}</span>`,`${t.ttft_n} requests, last ${n} min`,`ttft`,t.ttft_p50==null)}
      ${i(`Cache hit`,t.cache_hit==null?W:r`<span class="num">${v(t.cache_hit,1)}</span>`,t.cache_hit==null?`no cached-token counts in the last ${n} min`:r`${y(t.cached_tokens_window)} of ${y(t.prompt_tokens_window)} prompt tokens · ${t.cache_reported} requests`,`cache`,t.cache_hit==null)}
      ${i(`Cost per minute`,t.cost_per_min==null?`none`:r`<span class="num">${b(t.cost_per_min)}</span>`,r`${b(t.cost_window)} in ${n} min${t.cost_in_flight==null?o:r` · <span title="completion tokens so far at the output price; the prompt's part comes with the usage">${b(t.cost_in_flight)} running</span>`}`,`cost`,t.cost_per_min==null)}
      ${i(`Tokens out`,r`<span class="num">${y(t.tokens_window)}</span>`,`${t.finished} requests finished, last ${n} min`,`tokens`)}
      ${i(`Errors`,r`<span class="num">${t.errors}</span>`,`${t.refused} refused, last ${n} min`,`errors`)}
    </div>`}renderBackends(e){if(!e.backends.length)return o;let t=new Map;for(let n of e.lanes)t.set(n.backend,[...t.get(n.backend)??[],n]);return r`<div class="par-backends">
      ${C(e.backends,e=>e.id,n=>this.renderBackend(n,t.get(n.id)??[],e))}
    </div>`}renderBackend(e,t,n){let i=d(this.colors.of(e.id)),a=e.frames,s=e.engine_view?`the engine's own count, read from its board`:a?a.verified?a.tools_verified===!1?`text frames = tokens (checked); tool-call frames carry several`:`frames = tokens (${a.passed}/${a.checked} answers checked)`:a.passed===a.checked?`checking frames against the usage: ${a.passed} of 2 answers so far`:`frames ≠ tokens (${a.checked-a.passed} of ${a.checked} answers): no live count`:e.kind===`tandemllm`&&e.id===`engine`?`tokens counted by the engine`:`frames not checked yet`,c=Math.round(n.windows.stats_s/60);return r`<article class="par-bk" data-backend=${e.id} style="--bk:${i}">
      <header class="par-bk-head">
        <span class="swatch" style="background:${i}"></span>
        <span class="par-bk-id">${e.id}</span>
        <span class="par-kind">${I(e.kind)}</span>
      </header>
      <div class="par-bk-models" title=${e.models.join(`, `)}>${e.models.join(`, `)||r`<span class="ink-3">idle</span>`}</div>
      <div class="par-bk-rate ${e.decode_tps==null&&e.decoding?`is-na`:``}">
        <span class="num">${e.decode_tps==null?e.decoding?W:`0`:q(e.decode_tps)}</span><span class="par-bk-unit">tok/s</span>
      </div>
      <div class="par-bk-dots" aria-label="${e.in_flight} in flight">
        <span class="par-bk-dotrow">${t.map(e=>r`<span class="par-dot tone-${M(e)}" title="${N(e)}"></span>`)}</span>
        <span class="par-bk-count">${e.in_flight} in flight${e.waiting?` · ${e.waiting} waiting`:``}</span>
      </div>
      <dl class="par-bk-figs">
        <div><dt>TTFT p50 · p90</dt><dd class="num">${e.ttft_p50==null?`—`:`${_(e.ttft_p50)} · ${_(e.ttft_p90)}`}</dd></div>
        <div><dt>cache hit</dt><dd class="num ${e.cache_hit==null?`par-nr`:``}">${e.cache_hit==null?e.finished?W:`—`:v(e.cache_hit,1)}</dd></div>
        <div><dt>cost / min</dt><dd class="num">${e.cost_per_min==null?e.cost_window==null?`no price`:`$0`:b(e.cost_per_min)}</dd></div>
        <div><dt>done, ${c} min</dt><dd class="num">${e.finished}${e.errors?r` <span class="err-ink">· ${e.errors} err</span>`:o}</dd></div>
      </dl>
      <div class="par-bk-frames" title="how live tokens are counted for this backend">${s}</div>
      ${e.id===`engine`&&n.engine_slots?this.renderSlots(n.engine_slots):o}
    </article>`}renderSlots(e){return r`<div class="par-slots" data-slots=${e.n} aria-label="${e.n} sequence slots, ${e.running} busy">
      <div class="par-slots-row">
        ${e.slots.map(e=>r`<span class="par-slot ${e.busy?`is-busy`:``} ${e.phase===`decode`?`is-decode`:``}" data-slot=${e.index} title="slot ${e.index}: window ${g(e.window)} tokens">
            <span class="par-slot-i num">${e.index}</span><span class="par-slot-w">${V(e.window)}</span><span class="par-slot-s">${B(e)}</span>
          </span>`)}
      </div>
      <div class="par-slots-foot" title="verify passes the decoding requests shared (one read of the weights for all of them), by how many blocks were in the pass">
        shared passes: <span class="num">${H(e.joint_hist)}</span>${e.admission&&e.admission.waits?r` · <span title="requests that waited for memory (admission by bytes)">${e.admission.waits} waited for memory</span>`:o}
      </div>
    </div>`}renderLanes(e){let t=this.slots.slots,n=t.map(e=>e.lane).filter(e=>e!=null);return this.scale=R(n,this.scale),r`<section class="panel par-lanes" aria-label="requests in flight">
      <header class="panel-head">
        <div class="panel-titles">
          <h2 class="panel-title">Lanes <span class="par-count num">${e.totals.in_flight}</span></h2>
          <p class="panel-sub">one row per request, kept in place; a finished one stays 3 s. Bars share one scale (${g(this.scale)} tokens); light: thinking</p>
        </div>
      </header>
      <div class="par-table" role="table">
        <div class="par-row par-row-head" role="row">
          <span role="columnheader">backend · model</span><span role="columnheader">state</span><span role="columnheader">tokens</span>
          <span role="columnheader" class="r">now</span><span role="columnheader" class="r">avg</span><span role="columnheader" class="r">TTFT</span>
          <span role="columnheader" class="r">prefill</span><span role="columnheader" class="r">cache</span><span role="columnheader" class="r">cost</span><span role="columnheader" class="r">elapsed</span>
        </div>
        ${t.length?o:r`<div class="par-empty">Nothing in flight. Requests appear here the moment they arrive, on every backend.</div>`}
        ${C(t.map((e,t)=>[e,t]),([,e])=>e,([e])=>e.lane?this.renderLane(e.lane):r`<div class="par-row par-row-free" role="row" aria-hidden="true"><span class="par-free">free</span></div>`)}
      </div>
      <p class="par-foot">
        Server: ${e.build_us?r`built in ${(e.build_us.last/1e3).toFixed(2)} ms (max ${(e.build_us.max/1e3).toFixed(1)} ms)`:o} · ${(this.bytes/1024).toFixed(1)} KB a poll · one poll a second.
        Live tokens are the engine's own count, or a backend's stream frames once they were checked against its usage; the rest is "${W}" until the usage arrives.
      </p>
    </section>`}renderLane(e){let t=M(e),n=d(this.colors.of(e.backend)),i=L(e,this.scale),a=e.state===`done`,s=e.client.session??e.client.user??e.client.kind,c=!a,l=e.decode_now==null?c&&(t===`think`||t===`write`||t===`tool`)?W:`—`:q(e.decode_now),u=e.decode_avg==null?e.tokens==null&&(c||e.tokens_source==null)&&e.frames?W:`—`:q(e.decode_avg),f=e.ttft_ms==null?`—`:_(e.ttft_ms);e.ttft_ms==null&&(e.state===`waiting`||e.state===`waiting_memory`||e.state===`queued`||e.state===`prefilling`)&&e.waiting_ms!=null&&(f=r`<span class="par-wait">${_(e.waiting_ms)}…</span>`);let p=`—`,m=e.prefill?.tps??e.prefill?.tps_avg;m==null?e.prefill||(p=r`<span class="par-nr" title="the proxy sees the time to the first token, not the backend's prefill">${W}</span>`):p=r`<span title=${e.prefill?.tps==null?`tokens a second over the whole prefill`:`tokens a second, now`}>${q(m)}</span>`;let h;h=e.cache_hit==null?c&&e.backend!==`engine`?r`<span class="par-nr" title="cached tokens come with the usage at the end">at the end</span>`:a&&e.prompt_tokens!=null&&e.cached_tokens==null?r`<span class="par-nr">${W}</span>`:`—`:r`<span title="${g(e.cached_tokens)} of ${g(e.prompt_tokens)} prompt tokens cached">${v(e.cache_hit,0)}</span>`;let x=`—`;e.cost_micros!=null&&!(e.cost_partial&&e.cost_micros===0)&&(x=e.cost_partial?r`<span title="completion tokens so far at the output price; the prompt's part comes with the usage">${b(e.cost_micros)}+</span>`:b(e.cost_micros));let S=e.tokens==null?e.frames?r`<span class="par-nr" title=${z(e)}>${g(e.frames)} frames</span>`:`0`:g(e.tokens);return r`<div class="par-row tone-${t} ${a?`is-done`:``}" role="row" data-lane=${e.id} data-backend=${e.backend} data-state=${e.state} style="--bk:${n}">
      <span class="par-who" role="cell">
        <span class="par-who-top"><span class="par-bk-tag">${e.backend}</span><span class="par-model">${e.model??``}</span></span>
        <span class="par-who-sub">${e.slot==null?o:r`<span class="par-slot-tag" title="the engine's sequence slot">slot ${e.slot}</span> · `}${e.via===`engine`?r`<span class="par-via" title="the engine's own live count, state and prefill (it files this request under the proxy's id)">engine view</span> · `:o}${e.client.kind}${s&&s!==e.client.kind?r` · <span class="par-sess">${s}</span>`:o}</span>
      </span>
      <span class="par-state" role="cell"><span class="par-dot tone-${t} ${c&&t!==`wait`?`is-live`:``}"></span><span class="par-state-word">${e.label??N(e)}</span></span>
      <span class="par-tokens" role="cell" title=${z(e)}>
        <span class="par-bar">${i?r`<span class="par-bar-think" style="width:${(i.think*100).toFixed(2)}%"></span><span class="par-bar-out" style="width:${(i.out*100).toFixed(2)}%"></span>`:r`<span class="par-bar-unknown"></span>`}</span>
        <span class="par-tok-num num">${S}${e.thinking_tokens?r`<span class="par-think-n"> · ${g(e.thinking_tokens)} think</span>`:o}${e.tool_chars?r`<span class="par-think-n"> · ${y(e.tool_chars)} chars of call</span>`:o}</span>
      </span>
      <span class="par-fig r num ${l===W?`par-nr`:``}" role="cell" data-cell="now"><span class="par-fig-k">now</span>${l}</span>
      <span class="par-fig r num ${u===W?`par-nr`:``}" role="cell" data-cell="avg"><span class="par-fig-k">avg</span>${u}</span>
      <span class="par-fig r num" role="cell" data-cell="ttft"><span class="par-fig-k">TTFT</span>${f}</span>
      <span class="par-fig r num" role="cell" data-cell="prefill"><span class="par-fig-k">prefill</span>${p}</span>
      <span class="par-fig r num" role="cell" data-cell="cache"><span class="par-fig-k">cache</span>${h}</span>
      <span class="par-fig r num" role="cell" data-cell="cost"><span class="par-fig-k">cost</span>${x}</span>
      <span class="par-fig r num" role="cell" data-cell="elapsed"><span class="par-fig-k">elapsed</span>${_(e.elapsed_ms)}</span>
    </div>`}};e([c()],J.prototype,`doc`,void 0),e([c()],J.prototype,`series`,void 0),e([c()],J.prototype,`error`,void 0),e([c()],J.prototype,`bytes`,void 0),J=e([n(`qse-parallel`)],J);export{J as QseParallel};