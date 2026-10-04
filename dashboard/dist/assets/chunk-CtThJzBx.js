import{A as e,B as t,E as n,F as r,G as i,H as a,I as o,J as s,K as c,M as l,P as u,Q as d,R as f,U as p,V as m,W as h,Y as g,Z as _,_ as v,b as y,c as b,i as x,q as S,w as C,z as w}from"./app-DwE6ScUY.js";var T=[`cobalt`,`aqua`,`orange`,`violet`,`yellow`,`red`,`muted`],E=class{constructor(){this.map=new Map}of(e){let t=this.map.get(e);if(!t){if(e===`engine`)t=`cobalt`;else{let e=new Set(this.map.values());t=T.find(t=>t!==`cobalt`&&!e.has(t))??T[this.map.size%(T.length-1)+1]}this.map.set(e,t)}return t}seed(e){for(let t of[...e].sort((e,t)=>e===`engine`?-1:t===`engine`?1:e.localeCompare(t)))this.of(t)}},D=class{constructor(e=12e3,t=3e3){this.trimMs=e,this.holdMs=t,this.slots=[],this.where=new Map}update(e,t,n){let r=new Map;for(let e of t)(e.ended_ms??0)<=this.holdMs&&r.set(e.id,e);for(let t of e)r.set(t.id,t);for(let[e,t]of[...this.where])r.has(e)||(this.where.delete(e),this.slots[t]={lane:null,freeSince:n});for(let[e,t]of this.where)this.slots[t]={lane:r.get(e),freeSince:null};let i=[...r.values()].filter(e=>!this.where.has(e.id)&&e.state!==`done`);i.sort((e,t)=>(t.elapsed_ms??0)-(e.elapsed_ms??0));for(let e of i){let t=this.slots.findIndex(e=>e.lane==null);t<0&&(t=this.slots.length,this.slots.push({lane:null,freeSince:null})),this.slots[t]={lane:e,freeSince:null},this.where.set(e.id,t)}for(;this.slots.length;){let e=this.slots[this.slots.length-1];if(e.lane==null&&e.freeSince!=null&&n-e.freeSince>=this.trimMs)this.slots.pop();else break}return this.slots}};function O(e,t,n=300){let r=e.length?e[e.length-1].t:-1/0,i=e.concat(t.filter(e=>e.t>r)),a=i.length?i[i.length-1].t:0,o=0;for(;o<i.length&&a-i[o].t>n;)o++;return o?i.slice(o):i}function k(e,t){let n=e.map(()=>0);return t.map(t=>({id:t,bands:e.map((e,r)=>{let i=e.by[t]??0,a=n[r];return n[r]=a+i,[a,a+i]})}))}function A(e,t){let n=new Set(t.map(e=>e.id));for(let t of e)for(let e of Object.keys(t.by))n.add(e);return[...n].sort((e,t)=>e===`engine`?-1:t===`engine`?1:e.localeCompare(t))}var j={queued:`Queued`,prefilling:`Prefilling`,waiting:`Waiting for the first token`,replaying:`Replaying`,thinking:`Thinking`,closing_reasoning:`Closing the reasoning`,tool_call:`Tool call`,writing:`Writing`,finishing:`Finishing`,done:`Done`};function M(e){switch(e.state){case`queued`:case`waiting`:return`wait`;case`prefilling`:case`replaying`:return`prefill`;case`thinking`:case`closing_reasoning`:return`think`;case`tool_call`:return`tool`;case`done`:return e.finish===`error`||e.finish===`timeout`||(e.status??200)>=500?`error`:`done`;default:return`write`}}function N(e){return e.state===`tool_call`?e.tool?`Tool call: ${e.tool}`:`Tool call`:e.state===`prefilling`&&e.prefill?.pct!=null?`Prefilling ${Math.round(e.prefill.pct)} %`:e.state===`done`?e.finish&&e.finish!==`stop`?`Done: ${e.finish.replace(`_`,` `)}`:`Done`:j[e.state]??e.state}function P(e){return{tandemllm:`TandemLLM`,vllm:`vLLM`,sglang:`SGLang`,deepseek:`DeepSeek`,openai:`OpenAI`,generic:`OpenAI-compatible`}[e]??e}function F(e,t){if(e.tokens==null||t<=0)return null;let n=Math.min(e.thinking_tokens??0,e.tokens);return{think:Math.min(1,n/t),out:Math.min(1,(e.tokens-n)/t)}}function I(e,t=0){let n=Math.max(256,...e.map(e=>e.tokens??0));return t>=n&&t<=n*4?t:Math.max(256,2**Math.ceil(Math.log2(n)))}function L(e){switch(e.tokens_source){case`engine`:return`counted by the engine`;case`frames`:return`stream frames, checked to carry one token each on this backend`;case`usage`:return`the backend's usage`;default:return e.state===`done`?`the backend sent no usage`:`not reported live: this backend's frames are not checked as one token each yet`}}var R=class extends m{constructor(...e){super(...e),this.series=[],this.order=[],this.color=()=>`cobalt`,this.label=e=>e,this.span=300,this.now=0,this.hover=null,this.glowId=p(`st`)}onMove(e,t){let n=this.getBoundingClientRect(),r=`touches`in e?e.touches[0]?.clientX:e.clientX;if(r==null||!t.length)return;let i=r-n.left,a=0;for(let e=1;e<t.length;e++)Math.abs(t[e]-i)<Math.abs(t[a]-i)&&(a=e);this.hover=a}render(){let e=this.width||600,n=this.height,r={l:46,r:10,t:10,b:24},i=r.l,o=e-r.r,s=this.now||(this.series.length?this.series[this.series.length-1].t:Date.now()/1e3),c=s-this.span,l=this.series.filter(e=>e.t>=c-2),u=[];for(let e=0;e<l.length;e++){let t=l[e],n=u[u.length-1];n&&t.t-n.t>3&&u.push({t:n.t+1,by:{},in_flight:0},{t:t.t-1,by:{},in_flight:0}),u.push(t)}let p=k(u,this.order),m=u.map((e,t)=>p.length?p[p.length-1].bands[t][1]:0),h=w(Math.max(10,...m)),v=f([0,h],[n-r.b,r.t]),y=f([c,s],[i,o]),b=t(0,h,4),x=u.map(e=>y(e.t)),S=e=>{if(!e.length)return``;let t=`M${x[0].toFixed(1)},${v(e[0][1]).toFixed(1)}`;for(let n=1;n<e.length;n++)t+=`L${x[n].toFixed(1)},${v(e[n][1]).toFixed(1)}`;for(let n=e.length-1;n>=0;n--)t+=`L${x[n].toFixed(1)},${v(e[n][0]).toFixed(1)}`;return t+`Z`},C=e=>e.map((e,t)=>`${t?`L`:`M`}${x[t].toFixed(1)},${v(e).toFixed(1)}`).join(``),T=this.hover!=null&&this.hover<u.length?this.hover:null,E=this.span>120?60:30,D=Array.from({length:Math.floor(this.span/E)+1},(e,t)=>t*E);return _`<div
      class="chart chart-stack"
      tabindex="0"
      role="img"
      aria-label="decode tokens per second by backend, stacked, the last ${this.span} seconds"
      @mousemove=${e=>this.onMove(e,x)}
      @mouseleave=${()=>this.hover=null}
      @touchstart=${e=>this.onMove(e,x)}
      @touchmove=${e=>this.onMove(e,x)}
    >
      <svg width=${e} height=${n} viewBox="0 0 ${e} ${n}">
        <defs>
          <filter id="${this.glowId}-glow" x="-5%" y="-40%" width="110%" height="180%"><feGaussianBlur stdDeviation="4" /></filter>
        </defs>
        ${b.map(e=>d`<line class="grid" x1=${i} x2=${o} y1=${v(e)} y2=${v(e)} /><text class="axis" x=${i-6} y=${v(e)+3.5} text-anchor="end">${e>=1e3?`${(e/1e3).toFixed(e%1e3?1:0)}k`:e}</text>`)}
        ${D.map(e=>d`<text class="axis" x=${y(s-e)} y=${n-7} text-anchor=${e===0?`end`:e===this.span?`start`:`middle`}>${e===0?`now`:e%60?`−${e} s`:`−${e/60} min`}</text>`)}
        ${u.length?g:d`<text class="axis chart-nodata" x=${(i+o)/2} y=${n/2} text-anchor="middle">nothing decoded in this window</text>`}
        ${p.map(e=>{let t=a(this.color(e.id));return d`<path class="stack-area" data-layer=${e.id} d=${S(e.bands)} fill=${t} /><path d=${C(e.bands.map(e=>e[1]))} fill="none" stroke=${t} stroke-width="1.4" stroke-linejoin="round" />`})}
        ${u.length?d`<path d=${C(m)} fill="none" stroke="var(--ink)" stroke-width="7" opacity="0.10" filter="url(#${this.glowId}-glow)" /><path d=${C(m)} fill="none" stroke="var(--ink)" stroke-width="1.1" opacity="0.55" stroke-linejoin="round" />`:g}
        ${T==null?g:d`<line class="crosshair" x1=${x[T]} x2=${x[T]} y1=${r.t} y2=${n-r.b} />`}
      </svg>
      ${T==null?g:_`<div class="tip" style="left:${Math.min(Math.max(x[T],110),e-110)}px" role="status">
            <div class="tip-title">${Math.max(0,Math.round(s-u[T].t))} s ago · ${Math.round(m[T])} tok/s</div>
            ${[...this.order].reverse().map(e=>{let t=u[T].by[e];return t?_`<div class="tip-row"><span class="swatch" style="background:${a(this.color(e))}"></span><span>${this.label(e)}</span><span class="num">${Math.round(t)} tok/s</span></div>`:g})}
          </div>`}
    </div>`}};h([S({attribute:!1})],R.prototype,`series`,void 0),h([S({attribute:!1})],R.prototype,`order`,void 0),h([S({attribute:!1})],R.prototype,`color`,void 0),h([S({attribute:!1})],R.prototype,`label`,void 0),h([S({type:Number})],R.prototype,`span`,void 0),h([S({type:Number})],R.prototype,`now`,void 0),h([c()],R.prototype,`hover`,void 0),R=h([s(`qse-stack`)],R);var z=`not reported`,B=[`wait`,`prefill`,`think`,`tool`,`write`],V={wait:`waiting`,prefill:`prefilling`,think:`thinking`,tool:`tool call`,write:`writing`};function H(e){return e==null?`—`:e>=100?Math.round(e).toLocaleString(`en-US`):e.toFixed(1)}var U=class extends i{constructor(...e){super(...e),this.doc=null,this.series=[],this.error=null,this.bytes=0,this.slots=new D,this.colors=new E,this.scale=0,this.span=300,this.poller=null}connectedCallback(){super.connectedCallback(),this.poller=b(()=>this.load(),1e3),this.poller.start()}disconnectedCallback(){this.poller?.stop(),super.disconnectedCallback()}async load(){try{let e=this.series.length?this.series[this.series.length-1].t:null,t=await C.parallel(e);this.bytes=JSON.stringify(t).length,this.series=O(this.series,t.series??[]),this.colors.seed(A(this.series,t.backends)),this.slots.update(t.lanes,t.recent,Date.now()),this.doc=t,this.error=null}catch(e){let t=n(e);this.error={endpoint:t.endpoint||`/v1/dashboard/parallel`,message:t.message}}}render(){let e=this.doc;if(!e)return _`<div class="par">${this.error?v(this.error.endpoint,this.error.message,()=>this.poller?.refresh()):y(6,`skeleton-live`)}</div>`;let t=A(this.series,e.backends),n=this.series.length?this.series[0].t:e.at;return this.span=Math.min(300,Math.max(60,Math.ceil((e.at-n+5)/30)*30)),_`<div class="par" data-in-flight=${e.totals.in_flight}>
      ${this.renderHero(e,t)}
      ${this.renderKpis(e)}
      ${this.renderBackends(e)}
      ${this.renderLanes(e)}
      ${this.error?_`<p class="par-foot warn-ink" role="status">${this.error.message} (the last numbers stay until the next poll answers)</p>`:g}
    </div>`}renderHero(e,t){let n=e.totals,r=e.lanes,i=Object.fromEntries(B.map(e=>[e,r.filter(t=>M(t)===e).length])),o=new Map(e.backends.map(e=>[e.id,e]));return _`<section class="panel par-hero" aria-label="aggregate decode">
      <div class="par-hero-num">
        <div class="par-kicker">Aggregate decode, now</div>
        <div class="par-big ${n.decode_tps==null?`is-na`:``}" data-total-tps>
          <span class="num">${n.decode_tps==null?n.decoding?z:`0`:H(n.decode_tps)}</span><span class="par-big-unit">tok/s</span>
        </div>
        <div class="par-hero-sub">
          <span><b class="num">${n.in_flight}</b> in flight</span>
          <span><b class="num">${n.decoding}</b> decoding</span>
          <span><b class="num">${n.backends}</b> backend${n.backends===1?``:`s`}</span>
          <span class="par-nr ${n.decode_unreported?``:`is-hidden`}" title="no live count for these (server/parallel.py): the usage at the end has the exact one">${n.decode_unreported} not counted live</span>
          <span class=${n.prefill_tps==null?`is-hidden`:``}>prefill <b class="num">${H(n.prefill_tps)}</b> tok/s</span>
        </div>
        <div class="par-mix" role="img" aria-label="requests in flight by state">
          <div class="par-mix-bar">
            ${B.map(e=>_`<span class="par-mix-seg tone-${e}" style="flex-grow:${i[e]};flex-basis:0;${i[e]?``:`padding:0;`}" title="${i[e]} ${V[e]}"></span>`)}
            ${r.length?g:_`<span class="par-mix-seg tone-idle" style="flex-grow:1"></span>`}
          </div>
          <div class="par-mix-legend">
            ${B.map(e=>_`<span class="par-mix-item ${i[e]?``:`is-zero`}" data-tone=${e}><span class="par-dot tone-${e}"></span>${V[e]} <b class="num">${i[e]}</b></span>`)}
          </div>
        </div>
      </div>
      <div class="par-hero-chart">
        <qse-stack .series=${this.series} .order=${t} .color=${e=>this.colors.of(e)} .label=${e=>e} .now=${e.at} .span=${this.span} height="196"></qse-stack>
        <div class="par-legend">
          ${t.map(e=>{let t=o.get(e);return _`<span class="par-legend-item" data-backend=${e}><span class="swatch" style="background:${a(this.colors.of(e))}"></span>${e}<span class="num">${t?.decode_tps==null?t?.decoding?z:t?.in_flight?`waiting`:`idle`:`${H(t.decode_tps)} tok/s`}</span></span>`})}
        </div>
      </div>
    </section>`}renderKpis(t){let n=t.totals,i=Math.round(t.windows.stats_s/60),a=(e,t,n,r,i=!1)=>_`<div class="stat ${i?`is-na`:``}" data-kpi=${r}><div class="stat-label">${e}</div><div class="stat-value">${t}</div><div class="stat-sub">${n}</div></div>`;return _`<div class="stats par-kpis">
      ${a(`In flight`,_`<span class="num">${n.in_flight}</span>`,_`${n.waiting} waiting · ${n.decoding} decoding`,`in-flight`)}
      ${a(`TTFT p50 · p90`,n.ttft_p50==null?`none yet`:_`<span class="num">${r(n.ttft_p50)}</span><span class="stat-unit">· ${r(n.ttft_p90)}</span>`,`${n.ttft_n} requests, last ${i} min`,`ttft`,n.ttft_p50==null)}
      ${a(`Cache hit`,n.cache_hit==null?z:_`<span class="num">${o(n.cache_hit,1)}</span>`,n.cache_hit==null?`no cached-token counts in the last ${i} min`:_`${e(n.cached_tokens_window)} of ${e(n.prompt_tokens_window)} prompt tokens · ${n.cache_reported} requests`,`cache`,n.cache_hit==null)}
      ${a(`Cost per minute`,n.cost_per_min==null?`none`:_`<span class="num">${u(n.cost_per_min)}</span>`,_`${u(n.cost_window)} in ${i} min${n.cost_in_flight==null?g:_` · <span title="completion tokens so far at the output price; the prompt's part comes with the usage">${u(n.cost_in_flight)} running</span>`}`,`cost`,n.cost_per_min==null)}
      ${a(`Tokens out`,_`<span class="num">${e(n.tokens_window)}</span>`,`${n.finished} requests finished, last ${i} min`,`tokens`)}
      ${a(`Errors`,_`<span class="num">${n.errors}</span>`,`${n.refused} refused, last ${i} min`,`errors`)}
    </div>`}renderBackends(e){if(!e.backends.length)return g;let t=new Map;for(let n of e.lanes)t.set(n.backend,[...t.get(n.backend)??[],n]);return _`<div class="par-backends">
      ${x(e.backends,e=>e.id,n=>this.renderBackend(n,t.get(n.id)??[],e))}
    </div>`}renderBackend(e,t,n){let i=a(this.colors.of(e.id)),s=e.frames,c=e.engine_view?`the engine's own count, read from its board`:s?s.verified?s.tools_verified===!1?`text frames = tokens (checked); tool-call frames carry several`:`frames = tokens (${s.passed}/${s.checked} answers checked)`:s.passed===s.checked?`checking frames against the usage: ${s.passed} of 2 answers so far`:`frames ≠ tokens (${s.checked-s.passed} of ${s.checked} answers): no live count`:e.kind===`tandemllm`&&e.id===`engine`?`tokens counted by the engine`:`frames not checked yet`,l=Math.round(n.windows.stats_s/60);return _`<article class="par-bk" data-backend=${e.id} style="--bk:${i}">
      <header class="par-bk-head">
        <span class="swatch" style="background:${i}"></span>
        <span class="par-bk-id">${e.id}</span>
        <span class="par-kind">${P(e.kind)}</span>
      </header>
      <div class="par-bk-models" title=${e.models.join(`, `)}>${e.models.join(`, `)||_`<span class="ink-3">idle</span>`}</div>
      <div class="par-bk-rate ${e.decode_tps==null&&e.decoding?`is-na`:``}">
        <span class="num">${e.decode_tps==null?e.decoding?z:`0`:H(e.decode_tps)}</span><span class="par-bk-unit">tok/s</span>
      </div>
      <div class="par-bk-dots" aria-label="${e.in_flight} in flight">
        <span class="par-bk-dotrow">${t.map(e=>_`<span class="par-dot tone-${M(e)}" title="${N(e)}"></span>`)}</span>
        <span class="par-bk-count">${e.in_flight} in flight${e.waiting?` · ${e.waiting} waiting`:``}</span>
      </div>
      <dl class="par-bk-figs">
        <div><dt>TTFT p50 · p90</dt><dd class="num">${e.ttft_p50==null?`—`:`${r(e.ttft_p50)} · ${r(e.ttft_p90)}`}</dd></div>
        <div><dt>cache hit</dt><dd class="num ${e.cache_hit==null?`par-nr`:``}">${e.cache_hit==null?e.finished?z:`—`:o(e.cache_hit,1)}</dd></div>
        <div><dt>cost / min</dt><dd class="num">${e.cost_per_min==null?e.cost_window==null?`no price`:`$0`:u(e.cost_per_min)}</dd></div>
        <div><dt>done, ${l} min</dt><dd class="num">${e.finished}${e.errors?_` <span class="err-ink">· ${e.errors} err</span>`:g}</dd></div>
      </dl>
      <div class="par-bk-frames" title="how live tokens are counted for this backend">${c}</div>
    </article>`}renderLanes(e){let t=this.slots.slots,n=t.map(e=>e.lane).filter(e=>e!=null);return this.scale=I(n,this.scale),_`<section class="panel par-lanes" aria-label="requests in flight">
      <header class="panel-head">
        <div class="panel-titles">
          <h2 class="panel-title">Lanes <span class="par-count num">${e.totals.in_flight}</span></h2>
          <p class="panel-sub">one row per request, kept in place; a finished one stays 3 s. Bars share one scale (${l(this.scale)} tokens); light: thinking</p>
        </div>
      </header>
      <div class="par-table" role="table">
        <div class="par-row par-row-head" role="row">
          <span role="columnheader">backend · model</span><span role="columnheader">state</span><span role="columnheader">tokens</span>
          <span role="columnheader" class="r">now</span><span role="columnheader" class="r">avg</span><span role="columnheader" class="r">TTFT</span>
          <span role="columnheader" class="r">prefill</span><span role="columnheader" class="r">cache</span><span role="columnheader" class="r">cost</span><span role="columnheader" class="r">elapsed</span>
        </div>
        ${t.length?g:_`<div class="par-empty">Nothing in flight. Requests appear here the moment they arrive, on every backend.</div>`}
        ${x(t.map((e,t)=>[e,t]),([,e])=>e,([e])=>e.lane?this.renderLane(e.lane):_`<div class="par-row par-row-free" role="row" aria-hidden="true"><span class="par-free">free</span></div>`)}
      </div>
      <p class="par-foot">
        Server: ${e.build_us?_`built in ${(e.build_us.last/1e3).toFixed(2)} ms (max ${(e.build_us.max/1e3).toFixed(1)} ms)`:g} · ${(this.bytes/1024).toFixed(1)} KB a poll · one poll a second.
        Live tokens are the engine's own count, or a backend's stream frames once they were checked against its usage; the rest is "${z}" until the usage arrives.
      </p>
    </section>`}renderLane(t){let n=M(t),i=a(this.colors.of(t.backend)),s=F(t,this.scale),c=t.state===`done`,d=t.client.session??t.client.user??t.client.kind,f=!c,p=t.decode_now==null?f&&(n===`think`||n===`write`||n===`tool`)?z:`—`:H(t.decode_now),m=t.decode_avg==null?t.tokens==null&&(f||t.tokens_source==null)&&t.frames?z:`—`:H(t.decode_avg),h=t.ttft_ms==null?`—`:r(t.ttft_ms);t.ttft_ms==null&&(t.state===`waiting`||t.state===`queued`||t.state===`prefilling`)&&t.waiting_ms!=null&&(h=_`<span class="par-wait">${r(t.waiting_ms)}…</span>`);let v=`—`,y=t.prefill?.tps??t.prefill?.tps_avg;y==null?t.prefill||(v=_`<span class="par-nr" title="the proxy sees the time to the first token, not the backend's prefill">${z}</span>`):v=_`<span title=${t.prefill?.tps==null?`tokens a second over the whole prefill`:`tokens a second, now`}>${H(y)}</span>`;let b;b=t.cache_hit==null?f&&t.backend!==`engine`?_`<span class="par-nr" title="cached tokens come with the usage at the end">at the end</span>`:c&&t.prompt_tokens!=null&&t.cached_tokens==null?_`<span class="par-nr">${z}</span>`:`—`:_`<span title="${l(t.cached_tokens)} of ${l(t.prompt_tokens)} prompt tokens cached">${o(t.cache_hit,0)}</span>`;let x=`—`;t.cost_micros!=null&&!(t.cost_partial&&t.cost_micros===0)&&(x=t.cost_partial?_`<span title="completion tokens so far at the output price; the prompt's part comes with the usage">${u(t.cost_micros)}+</span>`:u(t.cost_micros));let S=t.tokens==null?t.frames?_`<span class="par-nr" title=${L(t)}>${l(t.frames)} frames</span>`:`0`:l(t.tokens);return _`<div class="par-row tone-${n} ${c?`is-done`:``}" role="row" data-lane=${t.id} data-backend=${t.backend} data-state=${t.state} style="--bk:${i}">
      <span class="par-who" role="cell">
        <span class="par-who-top"><span class="par-bk-tag">${t.backend}</span><span class="par-model">${t.model??``}</span></span>
        <span class="par-who-sub">${t.via===`engine`?_`<span class="par-via" title="the engine's own live count, state and prefill (it files this request under the proxy's id)">engine view</span> · `:g}${t.client.kind}${d&&d!==t.client.kind?_` · <span class="par-sess">${d}</span>`:g}</span>
      </span>
      <span class="par-state" role="cell"><span class="par-dot tone-${n} ${f&&n!==`wait`?`is-live`:``}"></span><span class="par-state-word">${t.label??N(t)}</span></span>
      <span class="par-tokens" role="cell" title=${L(t)}>
        <span class="par-bar">${s?_`<span class="par-bar-think" style="width:${(s.think*100).toFixed(2)}%"></span><span class="par-bar-out" style="width:${(s.out*100).toFixed(2)}%"></span>`:_`<span class="par-bar-unknown"></span>`}</span>
        <span class="par-tok-num num">${S}${t.thinking_tokens?_`<span class="par-think-n"> · ${l(t.thinking_tokens)} think</span>`:g}${t.tool_chars?_`<span class="par-think-n"> · ${e(t.tool_chars)} chars of call</span>`:g}</span>
      </span>
      <span class="par-fig r num ${p===z?`par-nr`:``}" role="cell" data-cell="now"><span class="par-fig-k">now</span>${p}</span>
      <span class="par-fig r num ${m===z?`par-nr`:``}" role="cell" data-cell="avg"><span class="par-fig-k">avg</span>${m}</span>
      <span class="par-fig r num" role="cell" data-cell="ttft"><span class="par-fig-k">TTFT</span>${h}</span>
      <span class="par-fig r num" role="cell" data-cell="prefill"><span class="par-fig-k">prefill</span>${v}</span>
      <span class="par-fig r num" role="cell" data-cell="cache"><span class="par-fig-k">cache</span>${b}</span>
      <span class="par-fig r num" role="cell" data-cell="cost"><span class="par-fig-k">cost</span>${x}</span>
      <span class="par-fig r num" role="cell" data-cell="elapsed"><span class="par-fig-k">elapsed</span>${r(t.elapsed_ms)}</span>
    </div>`}};h([c()],U.prototype,`doc`,void 0),h([c()],U.prototype,`series`,void 0),h([c()],U.prototype,`error`,void 0),h([c()],U.prototype,`bytes`,void 0),U=h([s(`qse-parallel`)],U);export{U as QseParallel};