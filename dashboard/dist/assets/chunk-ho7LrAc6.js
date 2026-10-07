import{$ as e,B as t,E as n,G as r,H as i,I as a,J as o,K as s,L as c,M as l,O as u,P as d,R as f,S as p,U as m,V as h,W as g,X as _,Y as v,Z as y,et as b,o as x,q as S,u as C,y as w}from"./app-BW_G8HM1.js";var T=[`cobalt`,`aqua`,`orange`,`violet`,`yellow`,`red`,`muted`],E=class{constructor(){this.map=new Map}of(e){let t=this.map.get(e);if(!t){if(e===`engine`)t=`cobalt`;else{let e=new Set(this.map.values());t=T.find(t=>t!==`cobalt`&&!e.has(t))??T[this.map.size%(T.length-1)+1]}this.map.set(e,t)}return t}seed(e){for(let t of[...e].sort((e,t)=>e===`engine`?-1:t===`engine`?1:e.localeCompare(t)))this.of(t)}},D=class{constructor(e=12e3,t=3e3){this.trimMs=e,this.holdMs=t,this.slots=[],this.where=new Map}update(e,t,n){let r=new Map;for(let e of t)(e.ended_ms??0)<=this.holdMs&&r.set(e.id,e);for(let t of e)r.set(t.id,t);for(let[e,t]of[...this.where])r.has(e)||(this.where.delete(e),this.slots[t]={lane:null,freeSince:n});for(let[e,t]of this.where)this.slots[t]={lane:r.get(e),freeSince:null};let i=[...r.values()].filter(e=>!this.where.has(e.id)&&e.state!==`done`);i.sort((e,t)=>(t.elapsed_ms??0)-(e.elapsed_ms??0));for(let e of i){let t=this.slots.findIndex(e=>e.lane==null);t<0&&(t=this.slots.length,this.slots.push({lane:null,freeSince:null})),this.slots[t]={lane:e,freeSince:null},this.where.set(e.id,t)}for(;this.slots.length;){let e=this.slots[this.slots.length-1];if(e.lane==null&&e.freeSince!=null&&n-e.freeSince>=this.trimMs)this.slots.pop();else break}return this.slots}};function O(e,t,n=300){let r=e.length?e[e.length-1].t:-1/0,i=e.concat(t.filter(e=>e.t>r)),a=i.length?i[i.length-1].t:0,o=0;for(;o<i.length&&a-i[o].t>n;)o++;return o?i.slice(o):i}function k(e,t){let n=e.map(()=>0);return t.map(t=>({id:t,bands:e.map((e,r)=>{let i=e.by[t]??0,a=n[r];return n[r]=a+i,[a,a+i]})}))}function A(e,t){let n=new Set(t.map(e=>e.id));for(let t of e)for(let e of Object.keys(t.by))n.add(e);return[...n].sort((e,t)=>e===`engine`?-1:t===`engine`?1:e.localeCompare(t))}var j={queued:`Queued`,prefilling:`Prefilling`,waiting:`Waiting for the first token`,replaying:`Replaying`,thinking:`Thinking`,closing_reasoning:`Closing the reasoning`,tool_call:`Tool call`,writing:`Writing`,finishing:`Finishing`,done:`Done`};function M(e){switch(e.state){case`queued`:case`waiting`:return`wait`;case`prefilling`:case`replaying`:return`prefill`;case`thinking`:case`closing_reasoning`:return`think`;case`tool_call`:return`tool`;case`done`:return e.finish===`error`||e.finish===`timeout`||(e.status??200)>=500?`error`:`done`;default:return`write`}}function N(e){return e.state===`tool_call`?e.tool?`Tool call: ${e.tool}`:`Tool call`:e.state===`prefilling`&&e.prefill?.pct!=null?`Prefilling ${Math.round(e.prefill.pct)} %`:e.state===`done`?e.finish&&e.finish!==`stop`?`Done: ${e.finish.replace(`_`,` `)}`:`Done`:j[e.state]??e.state}function P(e){return{tandemllm:`TandemLLM`,vllm:`vLLM`,sglang:`SGLang`,deepseek:`DeepSeek`,openai:`OpenAI`,generic:`OpenAI-compatible`}[e]??e}function F(e,t){if(e.tokens==null||t<=0)return null;let n=Math.min(e.thinking_tokens??0,e.tokens);return{think:Math.min(1,n/t),out:Math.min(1,(e.tokens-n)/t)}}function I(e,t=0){let n=Math.max(256,...e.map(e=>e.tokens??0));return t>=n&&t<=n*4?t:Math.max(256,2**Math.ceil(Math.log2(n)))}function L(e){switch(e.tokens_source){case`engine`:return`counted by the engine`;case`frames`:return`stream frames, checked to carry one token each on this backend`;case`usage`:return`the backend's usage`;default:return e.state===`done`?`the backend sent no usage`:`not reported live: this backend's frames are not checked as one token each yet`}}function R(e){return e.busy?e.phase===`decode`?`decoding`:e.phase===`solo`?`decoding alone`:`busy`:`free`}function z(e){return e>=1024?`${Math.round(e/1024)}k`:String(e)}function B(e){if(!e)return`none yet`;let t=Object.entries(e).filter(([,e])=>e>0).sort(([e],[t])=>Number(e)-Number(t)).map(([e,t])=>`${e}-way ${t}`);return t.length?t.join(` · `):`none yet`}var V=class extends m{constructor(...e){super(...e),this.series=[],this.order=[],this.color=()=>`cobalt`,this.label=e=>e,this.span=300,this.now=0,this.hover=null,this.glowId=r(`st`)}onMove(e,t){let n=this.getBoundingClientRect(),r=`touches`in e?e.touches[0]?.clientX:e.clientX;if(r==null||!t.length)return;let i=r-n.left,a=0;for(let e=1;e<t.length;e++)Math.abs(t[e]-i)<Math.abs(t[a]-i)&&(a=e);this.hover=a}render(){let n=this.width||600,r=this.height,a={l:46,r:10,t:10,b:24},o=a.l,s=n-a.r,c=this.now||(this.series.length?this.series[this.series.length-1].t:Date.now()/1e3),l=c-this.span,u=this.series.filter(e=>e.t>=l-2),d=[];for(let e=0;e<u.length;e++){let t=u[e],n=d[d.length-1];n&&t.t-n.t>3&&d.push({t:n.t+1,by:{},in_flight:0},{t:t.t-1,by:{},in_flight:0}),d.push(t)}let f=k(d,this.order),p=d.map((e,t)=>f.length?f[f.length-1].bands[t][1]:0),m=h(Math.max(10,...p)),_=t([0,m],[r-a.b,a.t]),v=t([l,c],[o,s]),x=i(0,m,4),S=d.map(e=>v(e.t)),C=e=>{if(!e.length)return``;let t=`M${S[0].toFixed(1)},${_(e[0][1]).toFixed(1)}`;for(let n=1;n<e.length;n++)t+=`L${S[n].toFixed(1)},${_(e[n][1]).toFixed(1)}`;for(let n=e.length-1;n>=0;n--)t+=`L${S[n].toFixed(1)},${_(e[n][0]).toFixed(1)}`;return t+`Z`},w=e=>e.map((e,t)=>`${t?`L`:`M`}${S[t].toFixed(1)},${_(e).toFixed(1)}`).join(``),T=this.hover!=null&&this.hover<d.length?this.hover:null,E=this.span>120?60:30,D=Array.from({length:Math.floor(this.span/E)+1},(e,t)=>t*E);return e`<div
      class="chart chart-stack"
      tabindex="0"
      role="img"
      aria-label="decode tokens per second by backend, stacked, the last ${this.span} seconds"
      @mousemove=${e=>this.onMove(e,S)}
      @mouseleave=${()=>this.hover=null}
      @touchstart=${e=>this.onMove(e,S)}
      @touchmove=${e=>this.onMove(e,S)}
    >
      <svg width=${n} height=${r} viewBox="0 0 ${n} ${r}">
        <defs>
          <filter id="${this.glowId}-glow" x="-5%" y="-40%" width="110%" height="180%"><feGaussianBlur stdDeviation="4" /></filter>
        </defs>
        ${x.map(e=>b`<line class="grid" x1=${o} x2=${s} y1=${_(e)} y2=${_(e)} /><text class="axis" x=${o-6} y=${_(e)+3.5} text-anchor="end">${e>=1e3?`${(e/1e3).toFixed(e%1e3?1:0)}k`:e}</text>`)}
        ${D.map(e=>b`<text class="axis" x=${v(c-e)} y=${r-7} text-anchor=${e===0?`end`:e===this.span?`start`:`middle`}>${e===0?`now`:e%60?`−${e} s`:`−${e/60} min`}</text>`)}
        ${d.length?y:b`<text class="axis chart-nodata" x=${(o+s)/2} y=${r/2} text-anchor="middle">nothing decoded in this window</text>`}
        ${f.map(e=>{let t=g(this.color(e.id));return b`<path class="stack-area" data-layer=${e.id} d=${C(e.bands)} fill=${t} /><path d=${w(e.bands.map(e=>e[1]))} fill="none" stroke=${t} stroke-width="1.4" stroke-linejoin="round" />`})}
        ${d.length?b`<path d=${w(p)} fill="none" stroke="var(--ink)" stroke-width="7" opacity="0.10" filter="url(#${this.glowId}-glow)" /><path d=${w(p)} fill="none" stroke="var(--ink)" stroke-width="1.1" opacity="0.55" stroke-linejoin="round" />`:y}
        ${T==null?y:b`<line class="crosshair" x1=${S[T]} x2=${S[T]} y1=${a.t} y2=${r-a.b} />`}
      </svg>
      ${T==null?y:e`<div class="tip" style="left:${Math.min(Math.max(S[T],110),n-110)}px" role="status">
            <div class="tip-title">${Math.max(0,Math.round(c-d[T].t))} s ago · ${Math.round(p[T])} tok/s</div>
            ${[...this.order].reverse().map(t=>{let n=d[T].by[t];return n?e`<div class="tip-row"><span class="swatch" style="background:${g(this.color(t))}"></span><span>${this.label(t)}</span><span class="num">${Math.round(n)} tok/s</span></div>`:y})}
          </div>`}
    </div>`}};s([v({attribute:!1})],V.prototype,`series`,void 0),s([v({attribute:!1})],V.prototype,`order`,void 0),s([v({attribute:!1})],V.prototype,`color`,void 0),s([v({attribute:!1})],V.prototype,`label`,void 0),s([v({type:Number})],V.prototype,`span`,void 0),s([v({type:Number})],V.prototype,`now`,void 0),s([o()],V.prototype,`hover`,void 0),V=s([_(`qse-stack`)],V);var H=`not reported`,U=[`wait`,`prefill`,`think`,`tool`,`write`],W={wait:`waiting`,prefill:`prefilling`,think:`thinking`,tool:`tool call`,write:`writing`};function G(e){return e==null?`—`:e>=100?Math.round(e).toLocaleString(`en-US`):e.toFixed(1)}var K=class extends S{constructor(...e){super(...e),this.doc=null,this.series=[],this.error=null,this.bytes=0,this.slots=new D,this.colors=new E,this.scale=0,this.span=300,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=C(()=>this.load(),1e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=this.series.length?this.series[this.series.length-1].t:null,t=await n.parallel(e);this.bytes=JSON.stringify(t).length,this.series=O(this.series,t.series??[]),this.colors.seed(A(this.series,t.backends)),this.slots.update(t.lanes,t.recent,Date.now()),this.doc=t,this.error=null}catch(e){let t=u(e);this.error={endpoint:t.endpoint||`/v1/dashboard/parallel`,message:t.message}}}render(){let t=this.doc;if(!t)return e`<div class="par">${this.error?w(this.error.endpoint,this.error.message,()=>this.poller?.refresh()):p(6,`skeleton-live`)}</div>`;let n=A(this.series,t.backends),r=this.series.length?this.series[0].t:t.at;return this.span=Math.min(300,Math.max(60,Math.ceil((t.at-r+5)/30)*30)),e`<div class="par" data-in-flight=${t.totals.in_flight}>
      ${this.renderHero(t,n)}
      ${this.renderKpis(t)}
      ${this.renderBackends(t)}
      ${this.renderLanes(t)}
      ${this.error?e`<p class="par-foot warn-ink" role="status">${this.error.message} (the last numbers stay until the next poll answers)</p>`:y}
    </div>`}renderHero(t,n){let r=t.totals,i=t.lanes,a=Object.fromEntries(U.map(e=>[e,i.filter(t=>M(t)===e).length])),o=new Map(t.backends.map(e=>[e.id,e]));return e`<section class="panel par-hero" aria-label="aggregate decode">
      <div class="par-hero-num">
        <div class="par-kicker">Aggregate decode, now</div>
        <div class="par-big ${r.decode_tps==null?`is-na`:``}" data-total-tps>
          <span class="num">${r.decode_tps==null?r.decoding?H:`0`:G(r.decode_tps)}</span><span class="par-big-unit">tok/s</span>
        </div>
        <div class="par-hero-sub">
          <span><b class="num">${r.in_flight}</b> in flight</span>
          <span><b class="num">${r.decoding}</b> decoding</span>
          <span><b class="num">${r.backends}</b> backend${r.backends===1?``:`s`}</span>
          <span class="par-nr ${r.decode_unreported?``:`is-hidden`}" title="no live count for these (server/parallel.py): the usage at the end has the exact one">${r.decode_unreported} not counted live</span>
          <span class=${r.prefill_tps==null?`is-hidden`:``}>prefill <b class="num">${G(r.prefill_tps)}</b> tok/s</span>
        </div>
        <div class="par-mix" role="img" aria-label="requests in flight by state">
          <div class="par-mix-bar">
            ${U.map(t=>e`<span class="par-mix-seg tone-${t}" style="flex-grow:${a[t]};flex-basis:0;${a[t]?``:`padding:0;`}" title="${a[t]} ${W[t]}"></span>`)}
            ${i.length?y:e`<span class="par-mix-seg tone-idle" style="flex-grow:1"></span>`}
          </div>
          <div class="par-mix-legend">
            ${U.map(t=>e`<span class="par-mix-item ${a[t]?``:`is-zero`}" data-tone=${t}><span class="par-dot tone-${t}"></span>${W[t]} <b class="num">${a[t]}</b></span>`)}
          </div>
        </div>
      </div>
      <div class="par-hero-chart">
        <qse-stack .series=${this.series} .order=${n} .color=${e=>this.colors.of(e)} .label=${e=>e} .now=${t.at} .span=${this.span} height="196"></qse-stack>
        <div class="par-legend">
          ${n.map(t=>{let n=o.get(t);return e`<span class="par-legend-item" data-backend=${t}><span class="swatch" style="background:${g(this.colors.of(t))}"></span>${t}<span class="num">${n?.decode_tps==null?n?.decoding?H:n?.in_flight?`waiting`:`idle`:`${G(n.decode_tps)} tok/s`}</span></span>`})}
        </div>
      </div>
    </section>`}renderKpis(t){let n=t.totals,r=Math.round(t.windows.stats_s/60),i=(t,n,r,i,a=!1)=>e`<div class="stat ${a?`is-na`:``}" data-kpi=${i}><div class="stat-label">${t}</div><div class="stat-value">${n}</div><div class="stat-sub">${r}</div></div>`;return e`<div class="stats par-kpis">
      ${i(`In flight`,e`<span class="num">${n.in_flight}</span>`,e`${n.waiting} waiting · ${n.decoding} decoding`,`in-flight`)}
      ${i(`TTFT p50 · p90`,n.ttft_p50==null?`none yet`:e`<span class="num">${c(n.ttft_p50)}</span><span class="stat-unit">· ${c(n.ttft_p90)}</span>`,`${n.ttft_n} requests, last ${r} min`,`ttft`,n.ttft_p50==null)}
      ${i(`Cache hit`,n.cache_hit==null?H:e`<span class="num">${f(n.cache_hit,1)}</span>`,n.cache_hit==null?`no cached-token counts in the last ${r} min`:e`${l(n.cached_tokens_window)} of ${l(n.prompt_tokens_window)} prompt tokens · ${n.cache_reported} requests`,`cache`,n.cache_hit==null)}
      ${i(`Cost per minute`,n.cost_per_min==null?`none`:e`<span class="num">${a(n.cost_per_min)}</span>`,e`${a(n.cost_window)} in ${r} min${n.cost_in_flight==null?y:e` · <span title="completion tokens so far at the output price; the prompt's part comes with the usage">${a(n.cost_in_flight)} running</span>`}`,`cost`,n.cost_per_min==null)}
      ${i(`Tokens out`,e`<span class="num">${l(n.tokens_window)}</span>`,`${n.finished} requests finished, last ${r} min`,`tokens`)}
      ${i(`Errors`,e`<span class="num">${n.errors}</span>`,`${n.refused} refused, last ${r} min`,`errors`)}
    </div>`}renderBackends(t){if(!t.backends.length)return y;let n=new Map;for(let e of t.lanes)n.set(e.backend,[...n.get(e.backend)??[],e]);return e`<div class="par-backends">
      ${x(t.backends,e=>e.id,e=>this.renderBackend(e,n.get(e.id)??[],t))}
    </div>`}renderBackend(t,n,r){let i=g(this.colors.of(t.id)),o=t.frames,s=t.engine_view?`the engine's own count, read from its board`:o?o.verified?o.tools_verified===!1?`text frames = tokens (checked); tool-call frames carry several`:`frames = tokens (${o.passed}/${o.checked} answers checked)`:o.passed===o.checked?`checking frames against the usage: ${o.passed} of 2 answers so far`:`frames ≠ tokens (${o.checked-o.passed} of ${o.checked} answers): no live count`:t.kind===`tandemllm`&&t.id===`engine`?`tokens counted by the engine`:`frames not checked yet`,l=Math.round(r.windows.stats_s/60);return e`<article class="par-bk" data-backend=${t.id} style="--bk:${i}">
      <header class="par-bk-head">
        <span class="swatch" style="background:${i}"></span>
        <span class="par-bk-id">${t.id}</span>
        <span class="par-kind">${P(t.kind)}</span>
      </header>
      <div class="par-bk-models" title=${t.models.join(`, `)}>${t.models.join(`, `)||e`<span class="ink-3">idle</span>`}</div>
      <div class="par-bk-rate ${t.decode_tps==null&&t.decoding?`is-na`:``}">
        <span class="num">${t.decode_tps==null?t.decoding?H:`0`:G(t.decode_tps)}</span><span class="par-bk-unit">tok/s</span>
      </div>
      <div class="par-bk-dots" aria-label="${t.in_flight} in flight">
        <span class="par-bk-dotrow">${n.map(t=>e`<span class="par-dot tone-${M(t)}" title="${N(t)}"></span>`)}</span>
        <span class="par-bk-count">${t.in_flight} in flight${t.waiting?` · ${t.waiting} waiting`:``}</span>
      </div>
      <dl class="par-bk-figs">
        <div><dt>TTFT p50 · p90</dt><dd class="num">${t.ttft_p50==null?`—`:`${c(t.ttft_p50)} · ${c(t.ttft_p90)}`}</dd></div>
        <div><dt>cache hit</dt><dd class="num ${t.cache_hit==null?`par-nr`:``}">${t.cache_hit==null?t.finished?H:`—`:f(t.cache_hit,1)}</dd></div>
        <div><dt>cost / min</dt><dd class="num">${t.cost_per_min==null?t.cost_window==null?`no price`:`$0`:a(t.cost_per_min)}</dd></div>
        <div><dt>done, ${l} min</dt><dd class="num">${t.finished}${t.errors?e` <span class="err-ink">· ${t.errors} err</span>`:y}</dd></div>
      </dl>
      <div class="par-bk-frames" title="how live tokens are counted for this backend">${s}</div>
      ${t.id===`engine`&&r.engine_slots?this.renderSlots(r.engine_slots):y}
    </article>`}renderSlots(t){return e`<div class="par-slots" data-slots=${t.n} aria-label="${t.n} sequence slots, ${t.running} busy">
      <div class="par-slots-row">
        ${t.slots.map(t=>e`<span class="par-slot ${t.busy?`is-busy`:``} ${t.phase===`decode`?`is-decode`:``}" data-slot=${t.index} title="slot ${t.index}: window ${d(t.window)} tokens">
            <span class="par-slot-i num">${t.index}</span><span class="par-slot-w">${z(t.window)}</span><span class="par-slot-s">${R(t)}</span>
          </span>`)}
      </div>
      <div class="par-slots-foot" title="verify passes the decoding requests shared (one read of the weights for all of them), by how many blocks were in the pass">
        shared passes: <span class="num">${B(t.joint_hist)}</span>${t.admission&&t.admission.waits?e` · <span title="requests that waited for memory (admission by bytes)">${t.admission.waits} waited for memory</span>`:y}
      </div>
    </div>`}renderLanes(t){let n=this.slots.slots,r=n.map(e=>e.lane).filter(e=>e!=null);return this.scale=I(r,this.scale),e`<section class="panel par-lanes" aria-label="requests in flight">
      <header class="panel-head">
        <div class="panel-titles">
          <h2 class="panel-title">Lanes <span class="par-count num">${t.totals.in_flight}</span></h2>
          <p class="panel-sub">one row per request, kept in place; a finished one stays 3 s. Bars share one scale (${d(this.scale)} tokens); light: thinking</p>
        </div>
      </header>
      <div class="par-table" role="table">
        <div class="par-row par-row-head" role="row">
          <span role="columnheader">backend · model</span><span role="columnheader">state</span><span role="columnheader">tokens</span>
          <span role="columnheader" class="r">now</span><span role="columnheader" class="r">avg</span><span role="columnheader" class="r">TTFT</span>
          <span role="columnheader" class="r">prefill</span><span role="columnheader" class="r">cache</span><span role="columnheader" class="r">cost</span><span role="columnheader" class="r">elapsed</span>
        </div>
        ${n.length?y:e`<div class="par-empty">Nothing in flight. Requests appear here the moment they arrive, on every backend.</div>`}
        ${x(n.map((e,t)=>[e,t]),([,e])=>e,([t])=>t.lane?this.renderLane(t.lane):e`<div class="par-row par-row-free" role="row" aria-hidden="true"><span class="par-free">free</span></div>`)}
      </div>
      <p class="par-foot">
        Server: ${t.build_us?e`built in ${(t.build_us.last/1e3).toFixed(2)} ms (max ${(t.build_us.max/1e3).toFixed(1)} ms)`:y} · ${(this.bytes/1024).toFixed(1)} KB a poll · one poll a second.
        Live tokens are the engine's own count, or a backend's stream frames once they were checked against its usage; the rest is "${H}" until the usage arrives.
      </p>
    </section>`}renderLane(t){let n=M(t),r=g(this.colors.of(t.backend)),i=F(t,this.scale),o=t.state===`done`,s=t.client.session??t.client.user??t.client.kind,u=!o,p=t.decode_now==null?u&&(n===`think`||n===`write`||n===`tool`)?H:`—`:G(t.decode_now),m=t.decode_avg==null?t.tokens==null&&(u||t.tokens_source==null)&&t.frames?H:`—`:G(t.decode_avg),h=t.ttft_ms==null?`—`:c(t.ttft_ms);t.ttft_ms==null&&(t.state===`waiting`||t.state===`queued`||t.state===`prefilling`)&&t.waiting_ms!=null&&(h=e`<span class="par-wait">${c(t.waiting_ms)}…</span>`);let _=`—`,v=t.prefill?.tps??t.prefill?.tps_avg;v==null?t.prefill||(_=e`<span class="par-nr" title="the proxy sees the time to the first token, not the backend's prefill">${H}</span>`):_=e`<span title=${t.prefill?.tps==null?`tokens a second over the whole prefill`:`tokens a second, now`}>${G(v)}</span>`;let b;b=t.cache_hit==null?u&&t.backend!==`engine`?e`<span class="par-nr" title="cached tokens come with the usage at the end">at the end</span>`:o&&t.prompt_tokens!=null&&t.cached_tokens==null?e`<span class="par-nr">${H}</span>`:`—`:e`<span title="${d(t.cached_tokens)} of ${d(t.prompt_tokens)} prompt tokens cached">${f(t.cache_hit,0)}</span>`;let x=`—`;t.cost_micros!=null&&!(t.cost_partial&&t.cost_micros===0)&&(x=t.cost_partial?e`<span title="completion tokens so far at the output price; the prompt's part comes with the usage">${a(t.cost_micros)}+</span>`:a(t.cost_micros));let S=t.tokens==null?t.frames?e`<span class="par-nr" title=${L(t)}>${d(t.frames)} frames</span>`:`0`:d(t.tokens);return e`<div class="par-row tone-${n} ${o?`is-done`:``}" role="row" data-lane=${t.id} data-backend=${t.backend} data-state=${t.state} style="--bk:${r}">
      <span class="par-who" role="cell">
        <span class="par-who-top"><span class="par-bk-tag">${t.backend}</span><span class="par-model">${t.model??``}</span></span>
        <span class="par-who-sub">${t.slot==null?y:e`<span class="par-slot-tag" title="the engine's sequence slot">slot ${t.slot}</span> · `}${t.via===`engine`?e`<span class="par-via" title="the engine's own live count, state and prefill (it files this request under the proxy's id)">engine view</span> · `:y}${t.client.kind}${s&&s!==t.client.kind?e` · <span class="par-sess">${s}</span>`:y}</span>
      </span>
      <span class="par-state" role="cell"><span class="par-dot tone-${n} ${u&&n!==`wait`?`is-live`:``}"></span><span class="par-state-word">${t.label??N(t)}</span></span>
      <span class="par-tokens" role="cell" title=${L(t)}>
        <span class="par-bar">${i?e`<span class="par-bar-think" style="width:${(i.think*100).toFixed(2)}%"></span><span class="par-bar-out" style="width:${(i.out*100).toFixed(2)}%"></span>`:e`<span class="par-bar-unknown"></span>`}</span>
        <span class="par-tok-num num">${S}${t.thinking_tokens?e`<span class="par-think-n"> · ${d(t.thinking_tokens)} think</span>`:y}${t.tool_chars?e`<span class="par-think-n"> · ${l(t.tool_chars)} chars of call</span>`:y}</span>
      </span>
      <span class="par-fig r num ${p===H?`par-nr`:``}" role="cell" data-cell="now"><span class="par-fig-k">now</span>${p}</span>
      <span class="par-fig r num ${m===H?`par-nr`:``}" role="cell" data-cell="avg"><span class="par-fig-k">avg</span>${m}</span>
      <span class="par-fig r num" role="cell" data-cell="ttft"><span class="par-fig-k">TTFT</span>${h}</span>
      <span class="par-fig r num" role="cell" data-cell="prefill"><span class="par-fig-k">prefill</span>${_}</span>
      <span class="par-fig r num" role="cell" data-cell="cache"><span class="par-fig-k">cache</span>${b}</span>
      <span class="par-fig r num" role="cell" data-cell="cost"><span class="par-fig-k">cost</span>${x}</span>
      <span class="par-fig r num" role="cell" data-cell="elapsed"><span class="par-fig-k">elapsed</span>${c(t.elapsed_ms)}</span>
    </div>`}};s([o()],K.prototype,`doc`,void 0),s([o()],K.prototype,`series`,void 0),s([o()],K.prototype,`error`,void 0),s([o()],K.prototype,`bytes`,void 0),K=s([_(`qse-parallel`)],K);export{K as QseParallel};