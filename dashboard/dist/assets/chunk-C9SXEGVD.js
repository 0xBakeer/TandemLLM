import{c as e,d as t,f as n,g as r,i,l as a,m as o,n as s,o as c,p as l,r as u,u as d,v as f}from"./chunk-5VjiyoiJ.js";import{C as p,M as m,S as h,T as g,_,a as ee,c as te,d as ne,f as re,h as ie,i as v,j as ae,k as y,l as oe,m as se,n as b,o as x,p as ce,r as S,s as le,t as ue,u as de,v as fe,w as C,x as pe,y as me}from"./chunk-DFmeUL_g.js";import{a as w,n as he}from"./chunk-BXmDkkzu.js";import{i as T,t as E}from"./chunk-CI4nmvus.js";import{_t as ge,c as D,l as _e,o as O,s as k,u as A,vt as ve}from"./app-Dvw5yJdR.js";import{a as ye,c as j,d as be,f as M,i as xe,l as N,n as Se,o as Ce,p as we,r as Te,s as Ee,t as De,u as P}from"./chunk-D7LC9lvJ.js";import{n as Oe,t as ke}from"./chunk-dBfjuPm3.js";import{a as Ae,c as je,i as Me,n as Ne,o as F,r as Pe,s as Fe,t as Ie}from"./chunk-pq1PeqZe.js";var I=O.community,Le={text:f`<path d="M4 5.5h12M4 9.5h12M4 13.5h7" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>`,mic:f`<rect x="7.25" y="3" width="5.5" height="9" rx="2.75" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M4.5 10a5.5 5.5 0 0 0 11 0M10 15.5v2M7.5 17.5h5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,video:f`<rect x="2.5" y="5" width="10.5" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m13 9 4.5-2.5v7L13 11z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>`,drafters:f`<circle cx="4.5" cy="10" r="1.8" fill="currentColor"/><circle cx="10" cy="10" r="1.8" fill="currentColor" opacity=".7"/><circle cx="15.5" cy="10" r="1.8" fill="currentColor" opacity=".45"/><path d="M6.3 10h1.9M11.8 10h1.9" stroke="currentColor" stroke-width="1.4"/>`,x:f`<path d="M4 3.5h3.6l8.4 13H12.4z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M15.5 3.5 4.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,huggingface:f`<circle cx="10" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="7.6" cy="9" r="1" fill="currentColor"/><circle cx="12.4" cy="9" r="1" fill="currentColor"/><path d="M7 12.2c.8 1.2 1.8 1.8 3 1.8s2.2-.6 3-1.8" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M4.2 7.2 3 6.2M15.8 7.2 17 6.2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,bluesky:f`<path d="M10 8.6C9 6.8 6.8 4.2 4.6 3.7 3.3 3.5 2.8 4.3 2.8 5.4c0 1.6.6 4.3 1.9 5 .9.4 2 .2 2.7 0-2 .5-2.9 1.4-2.1 2.9.9 1.8 3.2 2.5 4.7-.4 1.5 2.9 3.8 2.2 4.7.4.8-1.5-.1-2.4-2.1-2.9.7.2 1.8.4 2.7 0 1.3-.7 1.9-3.4 1.9-5 0-1.1-.5-1.9-1.8-1.7C13.2 4.2 11 6.8 10 8.6Z" fill="currentColor"/>`,mastodon:f`<path d="M4 15.5c-.9-1-1.5-2.6-1.5-5V8.3C2.5 5 4.6 3.5 10 3.5s7.5 1.5 7.5 4.8v1.5c0 2.6-1.4 3.9-4.2 4.1-1.9.2-3.5 0-4.3-.3v.9c0 1.4 2.3 1.5 5 1" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M6.3 11.8V8.4c0-1.1.6-1.7 1.6-1.7s1.6.6 1.6 1.7v1.9M13.7 11.8V8.4c0-1.1-.6-1.7-1.6-1.7s-1.6.6-1.6 1.7v1.9" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,website:f`<circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M3 10h14M10 3c2.2 2.1 3.3 4.4 3.3 7s-1.1 4.9-3.3 7c-2.2-2.1-3.3-4.4-3.3-7S7.8 5.1 10 3Z" fill="none" stroke="currentColor" stroke-width="1.4"/>`},L=(e,t=16)=>r`<svg class="icon" viewBox="0 0 20 20" aria-hidden="true" width=${t} height=${t}>${Le[e]}</svg>`,Re=e=>e===`text`?L(`text`):e===`image`?i(`image`):L(e===`audio`?`mic`:`video`),ze=e=>e.toUpperCase(),R=class extends a{constructor(...e){super(...e),this.model=null,this.sync=null,this.selected=null,this.device=null,this.syncing=!1,this.onUse=()=>{},this.onRefresh=()=>{},this.query=``,this.sort=`newest`,this.droppedOpen=!1,this.noAvatar=new Set}render(){let e=this.model;if(!e)return l;let t=re(e),n=this.sync;if(!t.length&&!n)return l;let a=h(pe(t,this.query),this.sort),o=te,s=ie(n?.add_url)??o;return r`<section class="cw" aria-labelledby="cw-h" data-community data-syncing=${this.syncing?``:l}>
      <header class="cw-head">
        <div class="cw-title">
          <h2 class="cw-h" id="cw-h">${I.heading}</h2>
          ${this.renderSource()}
        </div>
        <p class="cw-intro">${I.intro} <a href=${o} target="_blank" rel="noopener noreferrer">${I.addYoursHow} ${i(`external`)}</a></p>
        ${t.length?r`<div class="cw-tools">
              <label class="cw-search">
                ${i(`search`)}
                <input type="search" placeholder=${I.search} aria-label=${I.searchLabel} .value=${this.query} autocomplete="off" autocapitalize="off" spellcheck="false" @input=${e=>this.query=e.target.value} />
              </label>
              <label class="cw-sort"><span class="muted small">Sort</span>
                <select class="sel" aria-label=${I.sortLabel} @change=${e=>this.setSort(e.target.value)}>${oe.map(e=>r`<option value=${e} ?selected=${this.sort===e}>${de[e]}</option>`)}</select>
              </label>
              <span class="cw-count num muted small" data-count>${se(a.length,t.length)}</span>
              <a class="btn btn-sm cw-add" href=${s} target="_blank" rel="noopener noreferrer">${i(`plus`)} ${I.addYours}</a>
            </div>`:l}
      </header>
      ${n?.dropped.length?this.renderDropped(n.dropped):l}
      ${t.length?a.length?r`<div class="cw-grid" role="list">${a.map(e=>this.renderCard(e))}</div>`:r`<div class="cw-empty" data-community-nomatch role="status"><p class="cw-empty-t">${I.noMatch(this.query.trim())}</p><p><button class="btn btn-sm" @click=${()=>this.query=``}>${i(`x`)} ${I.clearSearch}</button></p></div>`:r`<div class="cw-empty" data-community-empty>
            <p class="cw-empty-t">${I.empty}</p>
            <p class="muted">${I.emptyBody}</p>
            <p class="cw-empty-acts"><a class="btn btn-primary btn-sm" href=${s} target="_blank" rel="noopener noreferrer">${i(`plus`)} ${I.addYours}</a> <a class="btn btn-sm" href=${o} target="_blank" rel="noopener noreferrer">${I.addYoursHow} ${i(`external`)}</a></p>
          </div>`}
    </section>`}setSort(e){_(e)&&(this.sort=e)}renderSource(){let e=this.sync,t=e?.source??`bundled`,n=e=>e&&Date.now()-Date.parse(e)<1e4?`just now`:he(e),a=this.syncing?I.syncing:t===`github`?I.fromGithub(n(e?.fetched_at)):t===`cache`?I.fromCache(n(e?.fetched_at)):I.bundled;return r`<div class="cw-src" data-source=${this.syncing?`syncing`:t}>
      <span class="cw-src-dot" aria-hidden="true"></span>
      <span class="cw-src-text small" role="status">${a}</span>
      <button class="btn btn-ghost btn-sm cw-refresh" ?disabled=${this.syncing} aria-busy=${this.syncing} @click=${()=>this.onRefresh()}>${i(`refresh`)} ${I.refresh}</button>
      ${e?.error&&!this.syncing?r`<span class="cw-src-err small" role="alert">${i(`warn`)} ${I.syncFailed(e.error.replace(/\.?$/,`.`))}</span>`:l}
      ${e?.note&&!e.error&&!this.syncing?r`<span class="cw-src-note small muted">${e.note}</span>`:l}
    </div>`}renderDropped(e){return r`<div class="cw-dropped small" data-dropped>
      <p>${i(`info`)} <span>${I.dropped(e.length)}</span> <button class="linkish" aria-expanded=${this.droppedOpen} @click=${()=>this.droppedOpen=!this.droppedOpen}>${this.droppedOpen?I.droppedHide:I.droppedDetails}</button></p>
      ${this.droppedOpen?r`<ul>${e.map(e=>r`<li><code>${e.path}</code> <span class="muted">${e.reason}</span></li>`)}</ul>`:l}
    </div>`}renderCard(e){let t=this.selected===e.id,n=ne(e.profile,this.device),a=e.status===`broken`,o=e.author,s=o.avatar&&!this.noAvatar.has(o.login);return r`<article class="ccard ${t?`is-selected`:``} ${a?`is-broken`:``}" role="listitem" data-community-card=${e.slug} data-field=${t?`weights.profile`:l} aria-labelledby=${`cc-${e.slug}`}>
      <header class="ccard-maker">
        <span class="ccard-avatar" aria-hidden="true">
          ${s?r`<img src=${o.avatar} alt="" width="40" height="40" loading="lazy" decoding="async" referrerpolicy="no-referrer" @error=${()=>this.avatarFailed(o.login)} />`:r`<span class="ccard-initials">${o.initials}</span>`}
        </span>
        <span class="ccard-who">
          <span class="ccard-handles">
            ${o.url?r`<a class="ccard-login" href=${o.url} target="_blank" rel="noopener noreferrer" title=${I.github(o.login)}>@${o.login}</a>`:r`<span class="ccard-login">@${o.login}</span>`}
            ${o.social.length?r`<span class="ccard-social">${o.social.map(e=>r`<a class="ccard-net tip-host" href=${e.href} target="_blank" rel="noopener noreferrer" aria-label=${e.label} data-tip=${e.label} data-net=${e.kind}>${L(e.kind)}</a>`)}</span>`:l}
          </span>
          ${o.name&&o.name.toLowerCase()!==o.login.toLowerCase()?r`<span class="ccard-author">${o.name}</span>`:l}
        </span>
        <span class="ccard-badges">
          ${n===`running`?r`<span class="sbadge sbadge-now">${O.activeNow}</span>`:l}
          ${this.statusBadge(e)}
        </span>
      </header>
      <h3 class="ccard-name" id=${`cc-${e.slug}`}>${e.name}</h3>
      <p class="ccard-base">${e.finetuneOf?r`${I.finetuneOf} ${e.finetuneUrl?r`<a href=${e.finetuneUrl} target="_blank" rel="noopener noreferrer">${e.finetuneOf}</a>`:r`<span>${e.finetuneOf}</span>`}`:I.stock}</p>
      ${e.description?r`<p class="ccard-desc">${e.description}</p>`:l}
      <dl class="ccard-readout num">
        <div><dt>format</dt><dd>${ze(e.format)}</dd></div>
        <div><dt>${I.bpw}</dt><dd>${e.bpw==null?`–`:e.bpw.toFixed(2)}</dd></div>
        <div><dt>${I.download}</dt><dd>${n===`download`?`${e.sizeGb.toFixed(1)} GB`:O.onDisk}</dd></div>
      </dl>
      <ul class="ccard-modes" aria-label="Modalities">
        ${e.modalities.map(e=>r`<li class="ccard-mode tip-host ${e.runs?`is-on`:`is-off`}" tabindex="0" data-tip=${e.why} aria-label=${e.why} data-modality=${e.id}>${Re(e.id)}<span>${e.label}</span></li>`)}
      </ul>
      <p class="ccard-drafters">${L(`drafters`)} <span>${e.drafters}</span></p>
      ${e.verified&&e.verified.speeds.length?r`<div class="ccard-speed" data-speed>
            ${e.verified.speeds.map(e=>r`<span class="ccard-sp"><span class="ccard-sp-n num">${e.tok_s.toFixed(1)}</span><span class="ccard-sp-k">${e.task}</span></span>`)}
            <span class="ccard-sp-u"><span class="num">tok/s</span><span class="ccard-sp-note">${I.speedNote(e.verified.device,e.verified.build,w(e.verified.date))}</span></span>
          </div>`:r`<p class="ccard-speed-none muted small">${I.notMeasured}</p>`}
      <footer class="ccard-foot">
        <span class="ccard-links">
          ${e.licenseUrl?r`<a class="sbadge ccard-license" href=${e.licenseUrl} target="_blank" rel="noopener noreferrer" title=${`${I.license}: ${e.license}`}>${e.license}</a>`:r`<span class="sbadge ccard-license" title=${`${I.license}: ${e.license}`}>${e.license}</span>`}
          ${e.hfUrl?r`<a class="ccard-hf" href=${e.hfUrl} target="_blank" rel="noopener noreferrer" title=${e.hfRepo}>${L(`huggingface`)} ${I.hf} ${i(`external`)}</a>`:l}
        </span>
        <button class="btn ${t?``:`btn-primary`} ccard-use" ?disabled=${a} aria-pressed=${t} title=${a?I.brokenWhy:l} @click=${()=>!t&&this.onUse(e.id)}>${t?r`${i(`check`)} ${I.selected}`:I.use}</button>
      </footer>
    </article>`}statusBadge(e){return e.status===`verified`&&e.verified?r`<span class="sbadge sbadge-ok tip-host" tabindex="0" data-tip=${I.verifiedWhy(e.verified.device,e.verified.build,w(e.verified.date))}>${i(`check`)} ${I.verified}</span>`:e.status===`broken`?r`<span class="sbadge sbadge-err tip-host" tabindex="0" data-tip=${I.brokenWhy}>${i(`warn`)} ${I.broken}</span>`:r`<span class="sbadge tip-host" tabindex="0" data-tip=${I.submittedWhy}>${I.submitted}</span>`}avatarFailed(e){let t=new Set(this.noAvatar);t.add(e),this.noAvatar=t}};e([t({attribute:!1})],R.prototype,`model`,void 0),e([t({attribute:!1})],R.prototype,`sync`,void 0),e([t({attribute:!1})],R.prototype,`selected`,void 0),e([t({attribute:!1})],R.prototype,`device`,void 0),e([t({type:Boolean})],R.prototype,`syncing`,void 0),e([t({attribute:!1})],R.prototype,`onUse`,void 0),e([t({attribute:!1})],R.prototype,`onRefresh`,void 0),e([d()],R.prototype,`query`,void 0),e([d()],R.prototype,`sort`,void 0),e([d()],R.prototype,`droppedOpen`,void 0),e([d()],R.prototype,`noAvatar`,void 0),R=e([n(`qse-community`)],R);function z(e){return e.group===`long`?e.mode===`greedy`?`long_greedy`:`long_sampled`:`${e.mode}_${e.len}`}function B(e){return e===`everyday`?[`profile`,`prose`,`chat`,`code`,`ttft8`,`kl`,`lossless`]:[`profile`,`long32k`,`long64k`,`ttft32`,`kl`]}var V={profile:O.colWeights,prose:O.colProse,chat:O.colChat,code:O.colCode,long32k:O.colLong32,long64k:O.colLong64,ttft8:O.colTtft8,ttft32:O.colTtft32,kl:O.colKl,lossless:O.colLossless};function H(e,t,n,r){let i=t.cells[z(r)]??{};switch(n){case`profile`:{let n=e.profiles.find(e=>e.id===t.profile);return n?.bpw??n?.weights_gib??null}case`ttft8`:return t.ttft_s[8192]??null;case`ttft32`:return t.ttft_s[32768]??null;case`kl`:return e.profiles.find(e=>e.id===t.profile)?.quality.kl_vs_bf16??null;case`lossless`:return t.lossless?t.lossless.identical:null;default:return i[n]?.tok_s??null}}function Be(e,t){let n=e.measurements.rows,r=n.filter(e=>e.comparable),i=n.filter(e=>!e.comparable),a=n=>H(e,n,t.sort,t);return r.sort((e,n)=>{let r=a(e),i=a(n);return r==null&&i==null?0:r==null?1:i==null?-1:t.dir===`asc`?r-i:i-r}),{main:r,apart:i}}var U=[`prose`,`chat`,`code`,`long32k`,`long64k`],W=class extends a{constructor(...e){super(...e),this.model=null,this.selected=null,this.view={mode:`greedy`,len:`512`,group:`everyday`,sort:`code`,dir:`desc`}}set(e){let t={...this.view,...e};e.group&&!B(e.group).includes(t.sort)&&(t.sort=e.group===`long`?`long32k`:`code`),this.view=t}sortBy(e){this.view.sort===e?this.set({dir:this.view.dir===`asc`?`desc`:`asc`}):this.set({sort:e,dir:U.includes(e)||e===`lossless`?`desc`:`asc`})}render(){let e=this.model;if(!e)return l;let t=this.view,n=B(t.group),{main:i,apart:a}=Be(e,t),o={};for(let n of U)o[n]=Math.max(0,...i.map(r=>H(e,r,n,t)??0));let s=(e,t,n,i)=>r`<div class="seg" role="radiogroup" aria-label=${i}>${t.map(([t,i])=>r`<button class="seg-btn ${e===t?`is-on`:``}" role="radio" aria-checked=${e===t} data-val=${t} @click=${()=>n(t)}>${i}</button>`)}</div>`;return r`<div class="mt">
      <div class="mt-controls">
        ${s(t.mode,[[`greedy`,O.modeGreedy],[`sampled`,O.modeSampled]],e=>this.set({mode:e}),`Decoding`)}
        ${t.group===`everyday`?s(t.len,[[`512`,O.tokens512],[`128`,O.tokens128]],e=>this.set({len:e}),`Answer length`):l}
        ${s(t.group,[[`everyday`,O.groupEveryday],[`long`,O.groupLong]],e=>this.set({group:e}),`Tasks`)}
      </div>
      <div class="table-scroll mt-scroll">
        <table class="table mt-table" data-mode=${z(t)}>
          <thead>
            <tr>
              ${n.map(e=>{let n=t.sort===e,i=U.includes(e)?O.tokPerS:e===`ttft8`||e===`ttft32`?O.seconds:``;return r`<th scope="col" class=${e===`profile`?`mt-name`:`num-col`} aria-sort=${n?t.dir===`asc`?`ascending`:`descending`:`none`}>
                  <button class="mt-sort ${n?`is-on`:``}" data-col=${e} @click=${()=>this.sortBy(e)} aria-label=${O.sortBy(V[e])}>
                    <span>${V[e]}</span>${i?r`<span class="mt-unit">${i}</span>`:l}<span class="mt-arrow" aria-hidden="true">${n?t.dir===`asc`?`▴`:`▾`:``}</span>
                  </button>
                </th>`})}
            </tr>
          </thead>
          <tbody>
            ${i.map(r=>this.row(e,r,n,t,o))}
            ${a.length?r`<tr class="mt-apart-head"><td colspan=${n.length}>${O.notComparable}</td></tr>
                  ${a.map(r=>this.row(e,r,n,t,o,!0))}`:l}
          </tbody>
        </table>
      </div>
      <div class="mt-foot muted small">${O.measuredOn(e.measurements.build,e.measurements.date)} <details class="mt-method"><summary>${O.howMeasured}</summary><p>${e.measurements.method}</p></details></div>
    </div>`}row(e,t,n,i,a,o=!1){let s=e.profiles.find(e=>e.id===t.profile),c=t.profile===this.selected;return r`<tr class="mt-row ${c?`is-selected`:``} ${o?`is-apart`:``}" data-profile=${t.profile} aria-current=${c?`true`:l}>
      ${n.map(n=>{if(n===`profile`)return r`<th scope="row" class="mt-name">
            <span class="mt-label">${s?.label??t.profile}</span>
            ${c?r`<span class="mt-mine">${O.selectedRow}</span>`:l}
            ${o?r`<span class="mt-note" title=${t.note??``}>${O.notComparable}</span>`:l}
          </th>`;let u=H(e,t,n,i);if(u==null)return r`<td class="num-col mt-na">–</td>`;if(n===`kl`)return r`<td class="num-col num">${u.toFixed(4)}</td>`;if(n===`lossless`){let e=t.lossless;return r`<td class="num-col num" title=${O.losslessTitle(e.identical,e.near_tie)}>${e.identical} + ${e.near_tie}</td>`}if(n===`ttft8`||n===`ttft32`)return r`<td class="num-col num">${u.toFixed(1)}</td>`;let d=a[n]?u/a[n]*100:0;return r`<td class="num-col num mt-speed"><span class="mt-bar" style="--w:${d.toFixed(1)}" aria-hidden="true"></span><span class="mt-val">${u.toFixed(1)}</span></td>`})}
    </tr>`}};e([t({attribute:!1})],W.prototype,`model`,void 0),e([t({attribute:!1})],W.prototype,`selected`,void 0),e([d()],W.prototype,`view`,void 0),W=e([n(`qse-measure-table`)],W);var G=[`model`,`helpers`,`caches`,`engine`,`system`,`chats`],K=class extends a{constructor(...e){super(...e),this.model=null,this.estimate=null,this.available=null,this.local=!1,this.overText=null,this.open=!1,this.focusPart=null}render(){let e=this.estimate,t=this.model;if(!e||!t)return r`<section class="membar is-empty" aria-label=${O.memTitle}><div class="membar-line"><span class="membar-title">${O.memTitle}</span></div><div class="membar-track"></div></section>`;let n=e.budget_gb??e.device_gb-e.floor_gb,i=N(e),a=e.parts.find(e=>e.kind===`kv`),o=Math.max(e.device_gb,e.total_gb),s=e=>`${(e/o*100).toFixed(3)}%`,c=e=>G.indexOf(Ee(t,e)),u=[...e.parts].sort((e,t)=>c(e.id)-c(t.id)),d=0,f=u.map(e=>{let n={p:e,left:d,cat:Ee(t,e.id)};return d+=e.gb,n}),p=this.available==null?null:Math.max(0,e.device_gb-this.available),m=new Map;for(let e of f)m.set(e.cat,(m.get(e.cat)??0)+e.p.gb);let h=f.find(e=>e.p.id===this.focusPart),g=h?`${h.p.label}: ${h.p.gb.toFixed(2)} GiB${h.p.worst_gb!=null&&h.p.worst_gb!==h.p.gb?` (${O.memoryUpTo(h.p.worst_gb)})`:``}`:null,_=`${O.memTitle}: ${O.memFits(e.total_gb,n)}. ${i>0?this.overText??``:O.memLeft(n-e.total_gb)}`;return r`<section class="membar ${i>0?`is-over`:``} ${this.open?`is-open`:``}" aria-label=${O.memTitle} data-fits=${e.fits} data-total=${e.total_gb.toFixed(2)}>
      <div class="membar-line">
        <span class="membar-title">${O.memTitle}</span>
        <span class="membar-fig num" data-membar-total><b>${e.total_gb.toFixed(1)}</b><span class="membar-of"> / ${n.toFixed(1)} GiB</span></span>
        <span class="membar-say ${i>0?`is-over`:``}" data-membar-say role=${i>0?`alert`:l}>${i>0?this.overText??O.overSentence(i,null):O.memLeft(n-e.total_gb)}</span>
        <span class="membar-now num">${p==null?l:r`<i class="membar-now-dot" aria-hidden="true"></i>${O.memNow(p)}`}</span>
        <span class="membar-device num">${O.memDevice(e.device_gb)}</span>
        <button class="btn btn-ghost btn-sm membar-toggle" aria-expanded=${this.open} aria-controls="membar-more" aria-label=${this.open?O.memCollapse:O.memExpand} @click=${()=>this.open=!this.open}>
          <svg class="icon" viewBox="0 0 20 20" aria-hidden="true"><path d="m5 8 5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
        </button>
      </div>
      <div class="membar-track" role="img" aria-label=${_}>
        <span class="membar-free" style="left:${s(n)};width:${s(Math.max(0,e.device_gb-n))}" title=${`${O.memFloor}: ${e.floor_gb} GiB`}><span class="membar-floor-label">${O.memFloor} ${e.floor_gb} GiB</span></span>
        ${f.map(e=>r`<span
            class="membar-seg cat-${e.cat} ${e.p.kind===`kv`?`is-kv`:``} ${this.focusPart===e.p.id?`is-focus`:``}"
            style="left:${s(e.left)};width:${s(e.p.gb)}"
            data-part=${e.p.id}
            tabindex="0"
            aria-label=${`${e.p.label} ${e.p.gb.toFixed(1)} GiB`}
            @mouseenter=${()=>this.focusPart=e.p.id}
            @mouseleave=${()=>this.focusPart=null}
            @focus=${()=>this.focusPart=e.p.id}
            @blur=${()=>this.focusPart=null}
          ></span>`)}
        ${a&&a.worst_gb!=null&&a.worst_gb>a.gb?r`<span class="membar-worst" style="left:${s(e.total_gb)};width:${s(Math.min(a.worst_gb-a.gb,o-e.total_gb))}" title=${`${O.memWorst}: ${a.worst_gb.toFixed(1)} GiB`}></span>`:l}
        ${i>0?r`<span class="membar-over" style="left:${s(n)};width:${s(e.total_gb-n)}"></span>`:l}
        <span class="membar-floor" style="left:${s(n)}"></span>
        ${p==null?l:r`<span class="membar-nowtick" style="left:${s(Math.min(p,o))}" title=${O.memNow(p)}></span>`}
      </div>
      <div class="membar-more" id="membar-more">
        <ul class="membar-legend">
          ${G.filter(e=>m.has(e)).map(e=>r`<li class="cat-${e}"><i class="membar-sw" aria-hidden="true"></i>${O.memCat[e]} <span class="num">${(m.get(e)??0).toFixed(1)}</span></li>`)}
          <li class="cat-free"><i class="membar-sw" aria-hidden="true"></i>${O.memFloor} <span class="num">${e.floor_gb}</span></li>
        </ul>
        <p class="membar-detail" aria-live="polite">${g??(this.local?O.memLocal:e.notes[0]??`\xA0`)}</p>
      </div>
    </section>`}};e([t({attribute:!1})],K.prototype,`model`,void 0),e([t({attribute:!1})],K.prototype,`estimate`,void 0),e([t({attribute:!1})],K.prototype,`available`,void 0),e([t({type:Boolean})],K.prototype,`local`,void 0),e([t({attribute:!1})],K.prototype,`overText`,void 0),e([d()],K.prototype,`open`,void 0),e([d()],K.prototype,`focusPart`,void 0),K=e([n(`qse-memory-bar`)],K);var Ve=[`set_field`,`toggle`,`select_profile`,`go_to_step`,`next`,`previous`,`explain`,`read_device`],q=(e,t=!1)=>({ok:!1,error:e,...t?{stale:t}:{}});function J(e,t){return`${e}.${t}`}function He(e){if(e==null)return{};if(typeof e==`string`){if(!e.trim())return{};try{let t=JSON.parse(e);return t&&typeof t==`object`&&!Array.isArray(t)?t:null}catch{return null}}return typeof e==`object`&&!Array.isArray(e)?e:null}function Y(e){if(typeof e==`number`&&Number.isInteger(e)&&e>=0&&e<b.length)return b[e];if(typeof e==`string`){if(/^\d+$/.test(e))return Y(Number(e));if(v(e))return e}return null}function Ue(e,t){return(e===`int`||e===`number`)&&typeof t==`string`&&t.trim()!==``&&Number.isFinite(Number(t))?Number(t):e===`bool`&&(t===`true`||t===`false`)?t===`true`:t}function We(e){return typeof e==`boolean`?e?`on`:`off`:typeof e==`number`?e.toLocaleString(`en-US`):e}function Ge(e,t){let n=String(e.name??``);if(!Ve.includes(n))return/finish|apply|install|reset|delete|unload|load|confirm|submit/i.test(n)?q(O.tool.unknown(n)+` `+O.tool.userOnly(n)):q(O.tool.unknown(n));let r=He(e.args);if(!r)return q(O.tool.badArgs(`expected a JSON object`));if(n===`read_device`)return{ok:!0,kind:`read_device`};let i=t.model,a=t.selection,o=(e,n)=>{let r=t.touchedAt?.get(J(e,n));return r!=null&&t.turnStartedAt!=null&&r>t.turnStartedAt};switch(n){case`go_to_step`:{let e=r.n??r.step??r.index,n=Y(e);return n?n===`install`?q(O.tool.installStep):n!==`models`&&!i?q(O.tool.noModel):{ok:!0,kind:`action`,actions:[{type:n===`models`?`tiles`:`go`,step:n}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:O.toolDone.step(A[n]),result:O.tool.ok}:q(O.tool.noStep(String(e)))}case`next`:{if(!i)return q(O.tool.noModel);let e=x(t.step);if(e>=x(`review`))return q(O.tool.atEnd);let n=b[e+1];return{ok:!0,kind:`action`,actions:[{type:`next`}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:O.toolDone.step(A[n]),result:O.tool.ok}}case`previous`:{let e=x(t.step);if(e<=0)return q(O.tool.atStart);let n=b[e-1];return{ok:!0,kind:`action`,actions:[{type:`previous`}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:O.toolDone.step(A[n]),result:O.tool.ok}}}if(!i||!a)return q(O.tool.noModel);if(n===`explain`){let e=String(r.field??r.id??``),t=i.components.find(t=>t.id===e);if(t)return{ok:!0,kind:`explain`,target:{step:`components`,field:t.id},result:`${t.label}: ${t.what}${t.note?` `+t.note:``}`};let n=ee(i,e);if(n)return{ok:!0,kind:`explain`,target:{step:`methods`,field:n.field.id},result:`${n.field.label}: ${n.field.what}`};let a=i.profiles.find(t=>t.id===e);if(a)return{ok:!0,kind:`explain`,target:{step:`weights`,field:`profile`},result:`${a.label}: KL ${a.quality.kl_vs_bf16??`?`}, ${a.weights_gib} GiB in memory${a.note?`. `+a.note:``}`};let o={slots:O.slotsWhat,window:O.windowWhat,side_window:O.sideWindowWhat,kv_pool:O.kvPoolWhat,floor_gb:O.floorWhat};return o[e]?{ok:!0,kind:`explain`,target:{step:`parallelism`,field:e},result:o[e]}:q(O.tool.noField(`any`,e))}if(n===`select_profile`){let e=String(r.id??r.profile??``),t=i.profiles.find(t=>t.id===e);return t?o(`weights`,`profile`)?q(O.tool.stale,!0):a.profile===e?{ok:!0,kind:`action`,actions:[],undo:[],target:{step:`weights`,field:`profile`},summary:O.toolDone.profile(t.label),result:O.tool.same}:{ok:!0,kind:`action`,actions:[{type:`profile`,id:e}],undo:[{type:`profile`,id:a.profile}],target:{step:`weights`,field:`profile`},summary:O.toolDone.profile(t.label),result:O.tool.ok}:q(O.tool.noProfile(e))}if(n===`toggle`){let e=String(r.component??r.id??``),t=Ue(`bool`,r.on);return typeof t==`boolean`?Ke(i,a,e,t,o(`components`,e)):q(O.tool.badArgs(`on must be true or false`))}let s=Y(r.step);if(!s)return q(O.tool.noStep(String(r.step)));let c=String(r.field??``),l=le(i).find(e=>e.id===s)?.fields.find(e=>e.field===c);if(!l)return q(O.tool.noField(s,c));if(ue.has(c))return q(O.tool.secret(l.label));let u=Ue(l.type,r.value),d=S(l,u);if(d)return q(O.tool.badArgs(`${c}: ${d}`));if(s===`components`)return Ke(i,a,c,u,o(s,c));if(o(s,c))return q(O.tool.stale,!0);let f={step:s,field:c},p=O.toolDone.field(l.label,We(u));if(s===`weights`)return Ge({name:`select_profile`,args:{id:u}},t);if(s===`parallelism`){let e=c,t=e===`side_window`?a.side_window??a.window:a[e];return t===u?{ok:!0,kind:`action`,actions:[],undo:[],target:f,summary:p,result:O.tool.same}:{ok:!0,kind:`action`,actions:[{type:`parallel`,key:e,value:u}],undo:[{type:`parallel`,key:e,value:t}],target:f,summary:p,result:O.tool.ok}}if(s===`methods`){let e=ee(i,c);if(!e)return q(O.tool.noField(s,c));let t=a[e.group][c]??e.field.recommended;return t===u?{ok:!0,kind:`action`,actions:[],undo:[],target:f,summary:p,result:O.tool.same}:{ok:!0,kind:`action`,actions:[{type:`field`,group:e.group,id:c,value:u}],undo:[{type:`field`,group:e.group,id:c,value:t}],target:f,summary:p,result:O.tool.ok}}return q(O.tool.noField(s,c))}function Ke(e,t,n,r,i){let a=e.components.find(e=>e.id===n);if(!a)return q(O.tool.noComponent(n));if(a.required&&!r)return q(O.tool.required(a.label));let o=C(e,t.components)[n];if(r&&o.disabled&&!o.required)return q(o.reason??O.tool.noComponent(n));if(i)return q(O.tool.stale,!0);let s={step:`components`,field:n},c=r?O.toolDone.toggleOn(a.label):O.toolDone.toggleOff(a.label);return(a.required||(t.components[n]??a.default))===r?{ok:!0,kind:`action`,actions:[],undo:[],target:s,summary:c,result:O.tool.same}:{ok:!0,kind:`action`,actions:[{type:`component`,id:n,on:r}],undo:[{type:`components`,set:{...t.components}}],target:s,summary:c,result:O.tool.ok}}var X=new Set([`explain`,`read_device`]);function qe(e){if(!e.tools||!e.calls.length)return null;let t=e.round+1>=4;return e.calls.some(e=>!e.ok&&!X.has(e.name))?e.retried?{tools:!1,retried:!0}:{tools:!t,retried:!0}:e.calls.every(e=>X.has(e.name)&&e.name!==`read_device`)?e.text.trim()?null:{tools:!1,retried:e.retried}:{tools:!t,retried:e.retried}}function Je(e){let t=new Map,n=0,r=r=>{let i=new Map,a=null;for(let t=n;t<r;t++){let n=e[t];if(n.role===`assistant`)for(let e of n.tool_calls??[])i.set(e.id,e.name);if(n.role!==`tool`||X.has(i.get(n.tool_call_id??``)??``))continue;let r={};try{r=JSON.parse(n.content)}catch{}r.ok===!1?a=r.stale||a===`stale`?`stale`:`unchanged`:r.ok===!0&&(a=null)}a&&r>n&&t.set(r-1,a)};return e.forEach((e,t)=>{e.role===`user`&&t!==0&&(r(t),n=t)}),r(e.length),t}function Ye(e,t){return t&&e.ok&&e.summary?e.summary:null}var Xe=0;function Ze(e){return e.charAt(0).toUpperCase()+e.slice(1)}var Z=class extends a{constructor(...e){super(...e),this.host=null,this.step=`models`,this.chat=[],this.finished=!1,this.phase=`checking`,this.status=null,this.figure=`sleeping`,this.progress=null,this.streaming=!1,this.partial=``,this.composer=``,this.error=null,this.notice=null,this.sources=[],this.open=!1,this.toast=null,this.chips=new Map,this.ctrl=null,this.loadPoll=null,this.ping=null,this.loadStartedAt=0,this.lastUserText=``,this.opener=null,this.toastTimer=null,this.onKey=e=>{if(!this.open||!this.sheetMode())return;if(e.key===`Escape`){e.preventDefault(),this.close();return}if(e.key!==`Tab`)return;let t=this.querySelector(`.wren-panel`);if(!t)return;let n=[...t.querySelectorAll(`button:not([disabled]), textarea:not([disabled]), a[href], input:not([disabled])`)].filter(e=>e.offsetParent!==null);if(!n.length)return;let r=n[0],i=n[n.length-1];e.shiftKey&&document.activeElement===r?(e.preventDefault(),i.focus()):!e.shiftKey&&document.activeElement===i?(e.preventDefault(),r.focus()):t.contains(document.activeElement)||(e.preventDefault(),r.focus())}}connectedCallback(){super.connectedCallback(),this.checkStatus(),this.ping=setInterval(()=>{this.phase===`ready`&&document.visibilityState===`visible`&&this.checkStatus(!0)},3e4),document.addEventListener(`keydown`,this.onKey)}disconnectedCallback(){this.ctrl?.abort(),this.loadPoll&&clearTimeout(this.loadPoll),this.ping&&clearInterval(this.ping),this.toastTimer&&clearTimeout(this.toastTimer),document.removeEventListener(`keydown`,this.onKey),this.lockScroll(!1),super.disconnectedCallback()}updated(e){e.has(`finished`)&&this.finished&&this.phase!==`finished`&&this.sleepForInstall()}emit(e){this.figure=we(this.figure,e)}async checkStatus(e=!1){if(this.finished){this.phase=`finished`,this.figure=`sleeping`;return}try{let e=await y.assistant.status();this.applyStatus(e)}catch(t){if(e)return;this.phase=`failed`,this.error=T(t).status===0?O.wrenOffline:T(t).message,this.emit(`error`)}}applyStatus(e){switch(this.status=e,e.state){case`loaded`:this.phase!==`ready`&&(this.phase=`ready`,this.progress=null,this.figure=`loading`,this.emit(`loaded`));break;case`loading`:this.phase=`loading`,this.progress=this.timeProgress(e.loading_s??null),this.emit(`load-progress`),this.scheduleLoadPoll();break;case`refused`:this.phase=`refused`,this.emit(`error`);break;case`failed`:this.phase=`failed`,this.error=e.reason??O.wrenFailed,this.emit(`error`);break;default:this.phase=e.model==null?`refused`:`asleep`,this.figure=`sleeping`}}timeProgress(e){let t=e??(Date.now()-this.loadStartedAt)/1e3;return Math.min(.95,1-Math.exp(-t/8))}scheduleLoadPoll(){this.loadPoll&&clearTimeout(this.loadPoll),this.loadPoll=setTimeout(()=>void this.checkStatus(!0),500)}async wake(){this.error=null,this.loadStartedAt=Date.now(),this.phase=`loading`,this.progress=0,this.emit(`load-start`);try{this.applyStatus(await y.assistant.load())}catch(e){let t=T(e),n=e instanceof E?e.body:void 0;n?.assistant&&(this.status=n.assistant),t.status===409?(this.phase=`refused`,n?.assistant?.reason&&(this.status={...this.status,reason:n.assistant.reason})):(this.phase=`failed`,this.error=t.status===0?O.wrenOffline:t.message),this.emit(`error`)}}async sleep(){this.stop();try{this.applyStatus(await y.assistant.unload())}catch{}this.phase=`asleep`,this.emit(`unload`)}async sleepForInstall(){this.ctrl?.abort(),this.phase=`finished`,this.emit(`unload`),this.close();try{await y.assistant.unload()}catch{}}stop(){this.ctrl&&this.ctrl.abort()}async send(e){let t=e.trim();if(!t||this.streaming||!this.host)return;this.lastUserText=t,this.composer=``,this.error=null,this.notice=null,this.toast=null;let n=[...this.chat,{role:`user`,content:t}];this.setChat(n),await this.turn(n)}async retry(){if(!this.lastUserText)return;this.error=null;let e=this.chat[this.chat.length-1]?.role===`user`?this.chat:[...this.chat,{role:`user`,content:this.lastUserText}];this.setChat(e),await this.turn(e)}setChat(e){this.chat=e,this.host?.saveChat(e),this.updateComplete.then(()=>this.scrollLog())}async turn(e){let t=this.host,n=Date.now(),r=e;this.streaming=!0,this.emit(`send`),this.ctrl=new AbortController;let i=this.ctrl.signal,a=!0,o=!1,s=!1;try{for(let e=1;;e++){let c=``,l=[],u=[],d=[],f=!0,p=null;if(this.partial=``,await ae({messages:r,page:t.pageState(),...a?{}:{tools:!1}},e=>{e.type===`state`?this.emit(e.state===`thinking`?`state-thinking`:e.state===`loading`?`load-progress`:`first-token`):e.type===`text`?(c+=e.delta,this.partial=c,f?(f=!1,this.emit(`first-token`)):this.emit(`delta`),this.fig?.blip(),this.fabFig?.blip(),this.scrollLog()):e.type===`tool_call`?l.push({id:e.id||`call_${++Xe}`,name:e.name,args:e.args}):e.type===`done`?(this.sources=e.sources??[],e.unchanged&&(s=!0)):e.type===`error`&&(p=e.message)},i),p&&/another tab/i.test(p)){this.notice=O.wrenBusy;break}if(p)throw new E(`/v1/setup/assistant/chat`,500,p,`stream`);for(let e of l){let r=await t.execute(e,n);r.ok&&(this.emit(`tool-call`),this.fig?.hop(),this.emit(`acted`)),u.push({role:`tool`,tool_call_id:e.id,content:JSON.stringify(r.ok?{ok:!0,result:r.result,...r.summary?{summary:r.summary}:{}}:{ok:!1,error:r.result,...r.stale?{stale:!0}:{}})}),this.chips=new Map(this.chips).set(e.id,{summary:r.summary,ok:r.ok,stale:r.stale,undo:r.undo,undone:!1}),d.push({name:e.name,ok:r.ok,changed:!!r.summary});let i=Ye(r,this.open&&this.sheetMode());i&&this.showToast(e.id,i,r.reveal??null)}let m={role:`assistant`,content:c,...l.length?{tool_calls:l}:{}};r=[...r,m,...u],this.partial=``,this.setChat(r);let h=qe({round:e,tools:a,text:c,calls:d,retried:o});if(!h)break;({tools:a,retried:o}=h)}s&&!Je(r).has(r.length-1)&&(this.notice=O.wrenNotChanged),this.emit(`done`)}catch(e){if(e.name===`AbortError`)this.partial&&this.setChat([...r,{role:`assistant`,content:`${this.partial} ${O.wrenStopped}`}]),this.partial=``,this.emit(`stop`);else{let t=T(e);t.status===409?this.notice=O.wrenBusy:t.status===0?this.error=O.wrenOffline:t.status===503||/unload|asleep|not loaded/i.test(t.message)?(this.error=O.wrenWentToSleep,this.checkStatus(!0)):this.error=t.message||O.wrenError,this.partial&&this.setChat([...r,{role:`assistant`,content:this.partial}]),this.partial=``,this.emit(t.status===409?`done`:`error`)}}finally{this.streaming=!1,this.ctrl=null}}undoChip(e){let t=this.chips.get(e);t?.undo&&!t.undone&&(t.undo(),this.chips=new Map(this.chips).set(e,{...t,undone:!0}))}scrollLog(){let e=this.log;e&&(e.scrollTop=e.scrollHeight)}sheetMode(){return typeof matchMedia==`function`&&matchMedia(`(max-width: 1100px)`).matches}toggleOpen(){this.open?this.close():(this.opener=document.activeElement,this.open=!0,this.lockScroll(!0),this.updateComplete.then(()=>(this.querySelector(`.wren-panel .wren-input, .wren-panel .btn-primary, .wren-close`)??null)?.focus()))}close(){this.open&&(this.open=!1,this.lockScroll(!1),this.opener?.focus?.())}lockScroll(e){document.documentElement.classList.toggle(`wren-sheet-open`,e&&this.sheetMode())}showToast(e,t,n){this.toast={id:e,summary:t,reveal:n},this.toastTimer&&clearTimeout(this.toastTimer),this.toastTimer=setTimeout(()=>this.toast=null,12e3)}async showMe(){let e=this.toast?.reveal;this.toast=null,this.close(),await this.updateComplete,requestAnimationFrame(()=>e?.())}statusWord(){return this.phase===`finished`?O.wrenFinished:this.phase===`asleep`?O.wrenAsleep:this.phase===`refused`?O.wrenCannotLoad:this.phase===`failed`?O.wrenFailed:this.phase===`loading`?O.wrenWaking(this.progress):O.wrenStatus[this.figure]??``}render(){let e=this.figure;return r`
      <button class="wren-fab ${this.open?`is-open`:``}" aria-label=${this.open?O.wrenClose:O.wrenOpen} aria-expanded=${this.open} aria-controls="wren-panel" @click=${()=>this.toggleOpen()} data-figure=${e}>
        <qse-wren-figure class="wren-fab-fig" .state=${e} .size=${50} .progress=${this.progress}></qse-wren-figure>
      </button>
      <div class="wren-scrim ${this.open?`is-open`:``}" @click=${()=>this.close()}></div>
      <aside id="wren-panel" class="wren-panel ${this.open?`is-open`:``}" aria-label=${O.wrenName} data-phase=${this.phase} data-figure=${e} role="complementary">
        <div class="wren-handle" aria-hidden="true"></div>
        <header class="wren-head">
          <qse-wren-figure class="wren-main" .state=${e} .size=${76} .progress=${this.progress}></qse-wren-figure>
          <div class="wren-titles">
            <h2 class="wren-name">${O.wrenName}</h2>
            <p class="wren-status" role="status" aria-live="polite" data-wren-status>${this.statusWord()}</p>
            ${this.status?.model&&this.phase===`ready`?r`<p class="wren-model">${this.status.model}${this.status.fallback?r` <span title=${O.wrenFallback}>(${O.wrenFallback})</span>`:l}</p>`:l}
          </div>
          <div class="wren-head-tools">
            ${this.phase===`ready`&&!this.streaming?r`<button class="btn btn-ghost btn-icon" title=${O.wrenSleep} aria-label=${O.wrenSleep} @click=${()=>void this.sleep()}>${i(`moon`)}</button>`:l}
            <button class="btn btn-ghost btn-icon wren-close" aria-label=${O.wrenClose} title=${O.wrenClose} @click=${()=>this.close()}>${i(`x`)}</button>
          </div>
        </header>
        ${this.renderBody()}
      </aside>
    `}hero(){return r`<qse-wren-figure class="wren-hero" .state=${this.figure} .size=${148} .progress=${this.progress}></qse-wren-figure>`}renderBody(){switch(this.phase){case`checking`:return r`<div class="wren-body wren-center">${this.hero()}</div>`;case`finished`:return r`<div class="wren-body wren-center" data-wren-finished>${this.hero()}<p>${O.wrenFinished}</p></div>${this.renderComposer(!0)}`;case`asleep`:return r`<div class="wren-body wren-center">
            ${this.hero()}
            <p class="wren-lead">${O.wrenAsleep}</p>
            <p class="muted small">${O.wrenAsleepBody}</p>
            <button class="btn btn-primary wren-wake" @click=${()=>void this.wake()}>${O.wrenWake}</button>
          </div>`;case`loading`:return r`<div class="wren-body wren-center">
            ${this.hero()}
            <p class="wren-lead">${O.wrenWaking(this.progress)}</p>
            <div class="wren-loadbar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${this.progress==null?l:Math.round(this.progress*100)} aria-label=${O.wrenWaking(null)}><span style="width:${Math.round((this.progress??0)*100)}%"></span></div>
          </div>`;case`refused`:{let e=this.status,t=e?.model==null&&e!=null?O.wrenNoModel:O.wrenCannotLoadMemory(e?.claim_gib??null,this.host?.memAvailable()??null);return r`<div class="wren-body wren-center" data-wren-refused>
            ${this.hero()}
            <p class="wren-lead">${O.wrenCannotLoad}</p>
            <p class="muted small">${e?.reason?Ze(e.reason):t}</p>
            ${e?.reason?r`<p class="muted small">${t}</p>`:l}
            ${e?.model==null?l:r`<button class="btn wren-wake" @click=${()=>void this.wake()}>${i(`refresh`)} ${O.wrenRetry}</button>`}
          </div>`}case`failed`:return r`<div class="wren-body wren-center" data-wren-failed>
            ${this.hero()}
            <p class="wren-lead">${O.wrenFailed}</p>
            <p class="muted small">${this.error??``}</p>
            <button class="btn wren-wake" @click=${()=>void this.wake()}>${i(`refresh`)} ${O.wrenRetry}</button>
          </div>`;case`ready`:return this.renderChat()}}renderChat(){let e=this.chat,t=Je(e);return r`
      <div class="wren-body wren-log" aria-live="off">
        ${e.length===0?r`<p class="wren-msg is-assistant wren-hello">${O.wrenHello}</p>`:l}
        ${ge(e.map((e,t)=>({m:e,i:t})),({i:e})=>e,({m:n,i})=>r`${this.renderMessage(n,i===e.length-1)}${t.has(i)?r`<p class="wren-note" data-tool-rejected>${t.get(i)===`stale`?O.tool.stale:O.wrenNotChanged}</p>`:l}`)}
        ${this.partial?r`<div class="wren-msg is-assistant is-streaming">${Oe(ke(this.partial))}</div>`:l}
        ${this.streaming&&!this.partial?r`<div class="wren-msg is-assistant wren-typing" aria-hidden="true"><span></span><span></span><span></span></div>`:l}
        ${this.notice?r`<p class="wren-note">${this.notice}</p>`:l}
        ${this.error?r`<div class="wren-error" role="alert" data-wren-error>
              <span>${this.error}</span>
              <button class="btn btn-sm" @click=${()=>void this.retry()}>${i(`refresh`)} ${O.wrenRetry}</button>
            </div>`:l}
      </div>
      ${this.toast?r`<div class="wren-toast" role="status" data-wren-toast>
            <span class="wren-chip-dot" aria-hidden="true"></span>
            <span class="wren-toast-text">${this.toast.summary}</span>
            ${this.toast.reveal?r`<button class="btn btn-sm" @click=${()=>void this.showMe()}>${O.wrenShowMe}</button>`:l}
          </div>`:l}
      ${this.renderChips()} ${this.renderComposer(!1)}
    `}renderMessage(e,t){if(e.role===`user`)return r`<div class="wren-msg is-user">${e.content}</div>`;if(e.role===`assistant`)return e.content.trim()?r`<div class="wren-msg is-assistant">${Oe(ke(e.content))}
        ${t&&this.sources.length?r`<p class="wren-sources">${O.wrenSources} ${this.sources.map((e,t)=>r`${t?`, `:``}<span title=${e.path}>${e.title||e.path}</span>`)}</p>`:l}
      </div>`:l;let n={};try{n=JSON.parse(e.content)}catch{}let i=e.tool_call_id?this.chips.get(e.tool_call_id):void 0;if(n.ok===!1||i?.ok===!1)return l;let a=i?.summary??n.summary;return a?r`<div class="wren-chip ${i?.undone?`is-undone`:``} ${this.toast?.id===e.tool_call_id?`is-fresh`:``}" data-tool-chip>
      <span class="wren-chip-dot" aria-hidden="true"></span>
      <span>${a}</span>
      ${i?.undo?r`<button class="btn btn-ghost btn-sm" ?disabled=${i.undone} @click=${()=>this.undoChip(e.tool_call_id)}>${i.undone?O.wrenUndone:O.wrenUndo}</button>`:l}
    </div>`:l}renderChips(){if(this.streaming)return l;let e=O.wrenChips[this.step]??[];return r`<div class="wren-suggest">${e.map(e=>r`<button class="wren-suggestion" @click=${()=>void this.send(e)}>${e}</button>`)}</div>`}renderComposer(e){return r`<form
      class="wren-composer"
      @submit=${e=>{e.preventDefault(),this.send(this.composer)}}
    >
      <label class="sr-only" for="wren-input">${O.wrenPlaceholder}</label>
      <textarea
        id="wren-input"
        class="wren-input"
        rows="1"
        .value=${this.composer}
        ?disabled=${e}
        placeholder=${e?O.wrenFinished:O.wrenPlaceholder}
        @focus=${()=>this.emit(`focus`)}
        @blur=${()=>!this.composer&&this.emit(`blur`)}
        @input=${e=>{this.composer=e.target.value,this.emit(`input`)}}
        @keydown=${e=>{e.key===`Enter`&&!e.shiftKey&&!e.isComposing&&(e.preventDefault(),this.send(this.composer))}}
      ></textarea>
      ${this.streaming?r`<button type="button" class="btn wren-stop" @click=${()=>this.stop()} aria-label=${O.wrenStop}>${i(`stop`)}<span>${O.wrenStop}</span></button>`:r`<button type="submit" class="btn btn-primary wren-send" ?disabled=${e||!this.composer.trim()} aria-label=${O.wrenSend}>${i(`send`)}</button>`}
    </form>`}};e([t({attribute:!1})],Z.prototype,`host`,void 0),e([t({attribute:!1})],Z.prototype,`step`,void 0),e([t({attribute:!1})],Z.prototype,`chat`,void 0),e([t({type:Boolean})],Z.prototype,`finished`,void 0),e([d()],Z.prototype,`phase`,void 0),e([d()],Z.prototype,`status`,void 0),e([d()],Z.prototype,`figure`,void 0),e([d()],Z.prototype,`progress`,void 0),e([d()],Z.prototype,`streaming`,void 0),e([d()],Z.prototype,`partial`,void 0),e([d()],Z.prototype,`composer`,void 0),e([d()],Z.prototype,`error`,void 0),e([d()],Z.prototype,`notice`,void 0),e([d()],Z.prototype,`sources`,void 0),e([d()],Z.prototype,`open`,void 0),e([d()],Z.prototype,`toast`,void 0),e([d()],Z.prototype,`chips`,void 0),e([m(`qse-wren-figure.wren-main`)],Z.prototype,`fig`,void 0),e([m(`qse-wren-figure.wren-fab-fig`)],Z.prototype,`fabFig`,void 0),e([m(`.wren-log`)],Z.prototype,`log`,void 0),e([m(`.wren-input`)],Z.prototype,`input`,void 0),Z=e([n(`qse-wren`)],Z);var Qe=[8192,32768,65536,131072,262144],Q=e=>e>=1024?`${Math.round(e/1024)}k`:String(e),$=class extends a{constructor(...e){super(...e),this.path=[],this.wiz=Ne(),this.storageProblem=null,this.storageFailed=!1,this.list={state:`loading`},this.model=null,this.device=null,this.serverEst=null,this.estLocal=!1,this.plan=null,this.confirmOpen=!1,this.applying=!1,this.applyError=null,this.jobInitial=null,this.previousProfile=null,this.resetOpen=!1,this.unauthorized=!1,this.winNarrow=!1,this.communitySyncing=!1,this.store=null,this.devPoll=null,this.estTimer=null,this.estCtrl=null,this.planTimer=null,this.planCtrl=null,this.commCtrl=null,this.warnedDiff=!1,this.touchedAt=new Map,this.loadingModel=null,this.onUnauthorized=()=>{this.unauthorized=!0},this.onSignedIn=()=>{this.unauthorized=!1,this.list.state!==`ready`&&this.loadList();let e=this.path[0];e&&this.model?.state!==`ready`&&this.loadModel(e),this.loadDevice(),this.wiz.selection&&this.queueEstimate(),this.wiz.step===`review`&&this.queuePlan(),this.wiz.step===`weights`&&this.syncCommunity()},this.onReset=()=>{this.wiz=Ne(),this.plan=null,this.jobInitial=null,location.hash!==`#/setup`&&(location.hash=`#/setup`)},this.host={pageState:()=>De(this.m,this.wiz.step,this.wiz.selection,this.currentEstimate(),{errors:this.visibleErrors()}),execute:(e,t)=>this.executeTool(e,t),saveChat:e=>this.dispatch({type:`chat`,chat:e}),memAvailable:()=>this.device?.mem_available_gb??null}}connectedCallback(){super.connectedCallback(),this.store=Me();let e=Pe(this.store);this.wiz=e.state,this.storageProblem=e.problem,(e.problem===`corrupt`||e.problem===`version`)&&Ie(this.store),this.loadList(),this.devPoll=ve(()=>this.loadDevice(),5e3),this.devPoll.start(),window.addEventListener(`qse-setup-reset`,this.onReset),window.addEventListener(`qse-setup-unauthorized`,this.onUnauthorized),window.addEventListener(`qse-signed-in`,this.onSignedIn)}disconnectedCallback(){this.devPoll?.stop(),this.estCtrl?.abort(),this.planCtrl?.abort(),this.commCtrl?.abort(),window.removeEventListener(`qse-setup-reset`,this.onReset),window.removeEventListener(`qse-setup-unauthorized`,this.onUnauthorized),window.removeEventListener(`qse-signed-in`,this.onSignedIn),super.disconnectedCallback()}async loadList(){this.list={state:`loading`};try{let e=await y.models();this.list=e.models.length||e.errors.length?{state:`ready`,data:e}:{state:`empty`,note:O.modelsEmpty}}catch(e){let t=T(e);this.list={state:`error`,endpoint:t.endpoint,message:t.status===401?O.unauthorized:t.message}}}async loadModel(e){if(this.loadingModel!==e){this.loadingModel=e,this.model={state:`loading`};try{let t=p(await y.model(e));this.model={state:`ready`,data:t},await this.openModel(t,this.routeStep()??void 0)}catch(t){let n=T(t);this.model={state:`error`,endpoint:n.endpoint,message:n.status===404?`${O.modelsError} (${e})`:n.message}}finally{this.loadingModel=null}}}async loadDevice(){try{this.device=await y.device()}catch{}}async openModel(e,t){let n=null;if(!(this.wiz.model===e.id&&this.wiz.selection))try{n=(await y.current(e.id)).selection}catch{n=null}this.dispatch({type:`open`,model:e,step:t,seed:n},!1),this.wiz.step===`weights`&&this.syncCommunity()}get m(){return this.model?.state===`ready`?this.model.data:null}routeStep(){let e=this.path[1];return v(e)?e:null}willUpdate(e){e.has(`path`)&&this.syncFromRoute()}syncFromRoute(){let[e,t]=this.path;if(!e){this.wiz.step!==`models`&&(this.wiz={...this.wiz,step:`models`});return}let n=this.m;if(!n||n.id!==e){this.loadModel(e);return}if(this.wiz.model!==e||!this.wiz.selection){this.openModel(n,v(t)?t:void 0);return}let r=v(t)?t:`overview`;r===`install`&&!this.wiz.job&&(r=`review`),r===`models`&&(r=`overview`),r!==this.wiz.step&&this.dispatch({type:`go`,step:r},!1)}hashFor(e){return e.step===`models`||!e.model?`#/setup`:`#/setup/${e.model}/${e.step}`}dispatch(e,t=!0){let n=Ae(this.wiz,e,this.m);if(n===this.wiz)return;let r=n.step!==this.wiz.step;this.wiz=n,this.store&&(this.storageFailed=!Fe(this.store,n));let i=this.hashFor(n);location.hash!==i&&(t?location.hash=i:history.replaceState(null,``,i)),r&&(t&&window.scrollTo({top:0,behavior:M()?`auto`:`smooth`}),n.step===`review`&&this.queuePlan(),n.step===`weights`&&this.syncCommunity()),e.type!==`go`&&e.type!==`next`&&e.type!==`previous`&&e.type!==`chat`&&e.type!==`dismiss`&&(this.queueEstimate(),n.step===`review`&&this.queuePlan())}user(e,t,n){this.touchedAt.set(J(e,t),Date.now()),this.dispatch(n)}selKey(e){return JSON.stringify(e)}queueEstimate(){this.estTimer&&clearTimeout(this.estTimer),this.estTimer=setTimeout(()=>void this.confirmEstimate(),150)}async confirmEstimate(){let e=this.wiz.selection;if(!e)return;this.estCtrl?.abort();let t=this.estCtrl=new AbortController,n=this.selKey(e);try{let r=await y.estimate(e,t.signal);if(t.signal.aborted||n!==this.selKey(this.wiz.selection))return;this.serverEst={key:n,est:r},this.estLocal=!1;let i=this.m?j(this.m,e):null;i&&Math.abs(i.total_gb-r.total_gb)>.05&&!this.warnedDiff&&(this.warnedDiff=!0,console.warn(`setup: the server's estimate (${r.total_gb} GiB) and the page's (${i.total_gb} GiB) differ; showing the server's`))}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;this.estLocal=!0}}currentEstimate(){let e=this.m,t=this.wiz.selection;if(!e||!t)return null;let n=j(e,t,this.device?.mem_available_gb??null),r=this.serverEst;return r&&r.key===this.selKey(t)&&(!n||Math.abs(r.est.total_gb-n.total_gb)>.05)?{...r.est,available_gb:this.device?.mem_available_gb??r.est.available_gb}:n}queuePlan(){this.planTimer&&clearTimeout(this.planTimer),this.planTimer=setTimeout(()=>void this.loadPlan(),200)}async loadPlan(){let e=this.wiz.selection;if(!e)return;this.planCtrl?.abort();let t=this.planCtrl=new AbortController;(!this.plan||this.plan.state!==`ready`)&&(this.plan={state:`loading`});try{let n=await y.plan(e,t.signal);t.signal.aborted||(this.plan={state:`ready`,data:n})}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;let n=T(e);this.plan={state:`error`,endpoint:n.endpoint,message:n.message}}}async syncCommunity(){let e=this.m;if(!e)return;this.commCtrl?.abort();let t=this.commCtrl=new AbortController;this.communitySyncing=!0;try{let n=await y.communitySync(t.signal);if(t.signal.aborted||this.m?.id!==e.id)return;let r=this.m,i={...n.sync,dropped:n.sync.dropped.filter(t=>t.model===e.id),add_url:r.community_sync?.add_url||n.sync.add_url},a=p({...r,community_sync:i},n.models[e.id]??[]);this.model={state:`ready`,data:a};let o=this.wiz.selection;o&&!a.profiles.some(e=>e.id===o.profile)?this.dispatch({type:`open`,model:a},!1):this.queueEstimate()}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;let n=this.m;n?.community_sync&&!(e instanceof E&&e.status===404)&&(this.model={state:`ready`,data:{...n,community_sync:{...n.community_sync,error:T(e).message,checked_at:new Date().toISOString()}}})}finally{t.signal.aborted||(this.communitySyncing=!1)}}visibleErrors(){let e=this.issues()[this.wiz.step]??[];return this.wiz.step!==`review`||this.plan?.state!==`ready`?e:[...e,...this.plan.data.errors.filter(t=>!e.includes(t))]}async executeTool(e,t){let n=Ge(e,{model:this.m,selection:this.wiz.selection,step:this.wiz.step,touchedAt:this.touchedAt,turnStartedAt:t});if(!n.ok)return{ok:!1,result:n.error,summary:null,undo:null,stale:n.stale};if(n.kind===`read_device`)try{let e=await y.device();this.device=e;let t={mode:e.mode,mem_total_gb:e.mem_total_gb,mem_available_gb:e.mem_available_gb,gpu:e.gpu,disk:e.disk,engine:e.engine,active_profile:e.active_profile,profiles_on_disk:e.profiles_on_disk,password_set:e.password_set};return{ok:!0,result:JSON.stringify(t),summary:null,undo:null}}catch(e){return{ok:!1,result:T(e).message,summary:null,undo:null}}if(n.kind===`explain`)return n.target.step===this.wiz.step&&this.highlight(n.target.step,n.target.field),{ok:!0,result:n.result,summary:null,undo:null};let{target:r}=n;r.field&&r.step!==this.wiz.step&&this.dispatch({type:`go`,step:r.step});for(let e of n.actions)this.dispatch(e);await this.updateComplete,r.field&&this.highlight(r.step,r.field);let i=n.undo.length?()=>{for(let e of n.undo)this.dispatch(e);r.field&&this.highlight(r.step,r.field)}:null,a=r.field?()=>{r.step!==this.wiz.step&&this.dispatch({type:`go`,step:r.step}),this.highlight(r.step,r.field)}:null;return{ok:!0,result:n.result,summary:n.actions.length?n.summary:null,undo:i,reveal:a}}async highlight(e,t){if(!t)return;await this.updateComplete;let n=this.querySelector(`[data-field="${CSS.escape(J(e,t))}"]`);if(!n)return;let r=n.getBoundingClientRect();(r.top<120||r.bottom>innerHeight-80)&&n.scrollIntoView({block:`center`,behavior:M()?`auto`:`smooth`}),n.setAttribute(`data-wren-acting`,``),n.querySelector(`:scope > .wren-mini`)?.remove();let i=document.createElement(`span`);i.className=`wren-mini`,i.setAttribute(`aria-hidden`,`true`),n.append(i),o(be,i),setTimeout(()=>{n.removeAttribute(`data-wren-acting`),i.remove()},1600)}issues(){let e=this.m,t=this.wiz.selection;return!e||!t?{}:ye(e,t,this.device,this.currentEstimate())}finishBlockers(){let e=this.m,t=this.wiz.selection;if(!e||!t)return[];let n=[],r=this.currentEstimate();r&&!r.fits&&n.push(O.reviewOver(N(r))),Ce(e,t,this.device)&&n.push(O.blockedToken);let i=Object.keys(this.issues()).filter(e=>e!==`review`);return i.length&&n.push(O.blockedInvalid(i.map(e=>A[e].toLowerCase()))),this.plan?.state===`ready`&&!this.plan.data.ok&&!n.length&&n.push(...this.plan.data.errors),n}async finish(){let e=this.wiz.selection;if(!e||this.applying||this.finishBlockers().length)return;this.applying=!0,this.applyError=null,this.previousProfile=this.device?.active_profile??null,this.querySelector(`qse-wren`)?.sleepForInstall();let t=e;try{let e=await y.apply({selection:t});this.jobInitial=e.job,this.confirmOpen=!1,this.dispatch({type:`job`,id:e.job.id})}catch(e){this.applyError=this.applyErrorText(e)}finally{this.applying=!1}}applyErrorText(e){let t=T(e);if(t.status===401)return null;let n=e instanceof E?e.body:void 0;return n?.plan?(this.plan={state:`ready`,data:n.plan},this.confirmOpen=!1,n.plan.errors.join(` `)||t.message):e instanceof E&&e.type===`job_running`?O.blockedBusy:e instanceof E&&e.type===`busy`?O.blockedRequests(n?.error?.running??1):`${O.installError} ${t.message}`}render(){let e=this.wiz.step,t=this.m,n=this.path.length>0;return r`<div class="view view-setup" data-step=${n?e:`models`}>
      <div class="setup">
        <div class="setup-main">
          ${this.renderBanners()}
          ${this.unauthorized?r`<qse-setup-signin></qse-setup-signin>`:n?this.renderModelArea():this.renderTiles()}
          <footer class="setup-foot">
            <button class="btn btn-ghost btn-sm" aria-expanded=${this.resetOpen} aria-controls="setup-reset" @click=${()=>this.resetOpen=!this.resetOpen}>${i(`trash`)} ${O.resetLink}</button>
          </footer>
          ${this.resetOpen?r`<section class="panel setup-reset" id="setup-reset"><header class="panel-head"><h2 class="panel-title">${O.resetHeading}</h2></header><qse-reset inSetup></qse-reset></section>`:l}
        </div>
        <qse-wren class="setup-wren" .host=${this.host} .step=${n?e:`models`} .chat=${this.wiz.chat} .finished=${e===`install`&&!!this.wiz.job}></qse-wren>
      </div>
      ${t?this.renderConfirm(t):l}
    </div>`}renderBanners(){let e=this.storageProblem,t=[];return(e===`unavailable`||this.storageFailed)&&t.push(O.storageOff),e===`corrupt`&&t.push(O.storageCorrupt),e===`version`&&t.push(O.storageOld),this.wiz.notice&&this.wiz.step!==`components`&&this.wiz.step!==`parallelism`&&t.push(this.wiz.notice),r`${t.map(e=>r`<div class="banner setup-banner" role="status" data-setup-notice>${i(`info`)} <span>${e}</span></div>`)}`}renderTiles(){let e=this.list,t=this.device,n=this.wiz.model&&this.wiz.selection?this.wiz:null;return r`<section class="setup-tiles">
      <header class="setup-hero">
        <h1 class="setup-h1">${A.models}</h1>
        <p class="setup-lead">${k.models}</p>
        ${t?.mem_total_gb==null?l:r`<p class="setup-device muted num" data-device-line>${O.deviceLine(t.mem_total_gb,t.mem_available_gb)}</p>`}
      </header>
      ${e.state===`loading`?c(4):l}
      ${e.state===`error`?u(e.endpoint,`${O.modelsError} ${e.message}`,()=>void this.loadList()):l}
      ${e.state===`empty`?s(O.modelsEmpty):l}
      ${e.state===`ready`?r`<div class="mtiles">
              ${e.data.models.map(e=>{let t=n?.model===e.id,i=t?`#/setup/${e.id}/${n.step===`models`?`overview`:n.step}`:`#/setup/${e.id}/overview`;return r`<a class="mtile ${e.recommended?`is-rec`:``}" href=${i} data-model=${e.id}>
                  <span class="mtile-top"><span class="mtile-family">${e.family}</span>${e.recommended?r`<span class="sbadge sbadge-rec">${O.tileRecommended}</span>`:l}</span>
                  <span class="mtile-name">${e.name}</span>
                  <span class="mtile-params">${O.tileParams(e.params)}</span>
                  <span class="mtile-pitch">${e.pitch}</span>
                  ${e.community_count?r`<span class="mtile-comm">${O.community.tileCount(e.community_count)}</span>`:l}
                  ${e.best_speed?r`<span class="mtile-speed" title=${O.tileSpeedTitle}><span class="mtile-speed-n num">${e.best_speed.tok_s.toFixed(1)}</span><span class="mtile-speed-u">tok/s</span><span class="mtile-speed-note">${e.best_speed.task}, ${e.best_speed.profile}</span></span>`:l}
                  <span class="mtile-go"><span class="btn btn-primary">${t?`${O.tileOpen}: ${D[n.step]}`:O.tileOpen}</span></span>
                </a>`})}
            </div>
            ${e.data.errors.length?r`<div class="setup-load-errors" role="status"><p>${O.modelLoadErrors(e.data.errors.length)}</p><ul>${e.data.errors.map(e=>r`<li><code>${e.path}</code> <span class="muted">${e.message}</span></li>`)}</ul></div>`:l}`:l}
      <section class="guide" aria-labelledby="guide-h">
        <h2 class="guide-h" id="guide-h">${O.guideSteps}</h2>
        <ol class="guide-list">${b.slice(1).map(e=>r`<li><span class="guide-t">${D[e]}</span><span class="guide-d">${_e[e]}</span></li>`)}</ol>
      </section>
    </section>`}renderModelArea(){let e=this.model;if(!e||e.state===`loading`)return c(8);if(e.state===`error`)return u(e.endpoint,e.message,()=>void this.loadModel(this.path[0]));if(e.state!==`ready`||!this.wiz.selection)return c(8);let t=e.data,n=this.wiz.selection,i=this.currentEstimate(),a=this.wiz.step,o=i&&!i.fits?Te(t,n,i):null;return r`
      <div class="setup-sticky">
        <qse-memory-bar .model=${t} .estimate=${i} .available=${this.device?.mem_available_gb??null} .local=${this.estLocal} .overText=${o}></qse-memory-bar>
        ${this.estLocal?r`<p class="setup-local" role="status" data-local-estimate>${O.memLocal}</p>`:l}
      </div>
      ${this.renderRail(t)}
      <section class="setup-step" data-step=${a} aria-labelledby="setup-step-title">
        <header class="step-head">
          <p class="step-kicker muted num">${O.stepOf(x(a)+1,b.length)}</p>
          <h1 class="step-title" id="setup-step-title" tabindex="-1">${A[a]}</h1>
          <p class="step-intro">${k[a]}</p>
        </header>
        ${this.renderStep(t,n,i)}
      </section>
      ${this.renderNav()}
    `}renderRail(e){let t=this.wiz.step,n=this.issues(),a=x(t);return r`<nav class="rail-steps" aria-label="Setup steps">
      <div class="rail-phone" aria-hidden="true">
        <span class="num">${O.stepOf(a+1,b.length)}</span><span class="rail-phone-label">${A[t]}</span>
        <span class="rail-phone-bar"><span style="width:${(a+1)/b.length*100}%"></span></span>
      </div>
      <ol class="steps">
        ${b.map((o,s)=>{let c=this.wiz.visited.includes(o),u=c&&(n[o]?.length??0)>0,d=o===t,f=c&&!u&&s<a,p=o!==`install`||!!this.wiz.job,m=o===`models`?`#/setup`:`#/setup/${e.id}/${o}`;return r`<li class="steps-item ${d?`is-current`:``} ${f?`is-done`:``} ${u?`is-bad`:``} ${c?`is-visited`:``}" data-rail=${o}>
            ${p?r`<a href=${m} aria-current=${d?`step`:l} title=${u?n[o].join(` `):A[o]}>
                  <span class="steps-node num" aria-hidden="true">${u?`!`:f?i(`check`):s+1}</span>
                  <span class="steps-label">${D[o]}</span>
                  ${u?r`<span class="sr-only">(needs a fix)</span>`:l}
                </a>`:r`<span class="steps-link is-locked" aria-disabled="true"><span class="steps-node num" aria-hidden="true">${s+1}</span><span class="steps-label">${D[o]}</span></span>`}
          </li>`})}
      </ol>
    </nav>`}renderNav(){let e=this.wiz.step;if(e===`install`)return l;let t=x(e);return r`<nav class="setup-nav" aria-label="Step navigation">
      <button class="btn setup-prev" @click=${()=>t<=1?this.dispatch({type:`tiles`}):this.dispatch({type:`previous`})}>${O.previous}</button>
      <span class="setup-nav-mid muted small num">${O.stepOf(t+1,b.length)}</span>
      ${e===`review`?r`<button class="btn btn-primary setup-finish" ?disabled=${this.finishBlockers().length>0||this.plan?.state!==`ready`} @click=${()=>this.confirmOpen=!0}>${O.finish}</button>`:r`<button class="btn btn-primary setup-next" @click=${()=>this.dispatch({type:`next`})}><span>${O.next}<span class="setup-next-to">: ${D[b[t+1]]}</span></span></button>`}
    </nav>`}renderStep(e,t,n){switch(this.wiz.step){case`overview`:return this.stepOverview(e,t);case`weights`:return this.stepWeights(e,t);case`components`:return this.stepComponents(e,t);case`parallelism`:return this.stepParallel(e,t,n);case`methods`:return this.stepMethods(e,t);case`review`:return this.stepReview(e,t,n);case`install`:return this.stepInstall();default:return r``}}stepOverview(e,t){let n=e.profiles.find(e=>e.id===t.profile),a=F(e,t.profile);return r`<div class="ov">
      <div class="ov-facts">
        <p class="ov-arch">${e.architecture}</p>
        <dl class="ov-dl">
          <div><dt>${O.colWeights}</dt><dd>${e.profiles.length} sizes, ${e.profiles.map(e=>e.label.replace(/ bpw$/,``)).join(`, `)}</dd></div>
          <div><dt>${O.window}</dt><dd class="num">${n?.max_window===null?O.windowNone(n.label):`${a.toLocaleString(`en-US`)} ${O.tokensUnit}`}</dd></div>
          <div><dt>${O.license}</dt><dd>${e.license}</dd></div>
        </dl>
        <ul class="ov-links">${e.links.map(e=>r`<li><a href=${e.url} target="_blank" rel="noopener noreferrer">${e.label} ${i(`external`)}</a></li>`)}</ul>
      </div>
      <section class="panel ov-table" aria-label="Measured speed">
        <qse-measure-table .model=${e} .selected=${t.profile}></qse-measure-table>
      </section>
    </div>`}stepWeights(e,t){let n=this.device,a=Ce(e,t,n),o=e.measurements.rows,s=Math.max(...o.filter(e=>e.comparable).map(e=>e.cells.greedy_512?.code?.tok_s??0));return r`
      ${a?r`<div class="banner banner-offline" role="alert" data-token-blocked>${i(`warn`)} ${O.privateNoToken}</div>`:l}
      ${!a&&n?.hf_token_present&&e.profiles.some(e=>e.hf_private&&!n.profiles_on_disk?.[e.id])?r`<p class="wnote muted small">${i(`info`)} ${O.privateRepo}</p>`:l}
      <div class="wcards" role="radiogroup" aria-label=${A.weights}>
        ${fe(e).map(e=>{let a=t.profile===e.id,c=o.find(t=>t.profile===e.id),u=c?.cells.greedy_512,d=n?.profiles_on_disk?.[e.id],f=n?.active_profile===e.id,p=e.quality.top1_pct;return r`<label class="wcard ${a?`is-selected`:``} ${e.recommended?`is-rec`:``}" data-profile-card=${e.id} data-field=${a?`weights.profile`:l}>
            <input class="sr-only" type="radio" name="profile" .checked=${a} value=${e.id} @change=${()=>this.user(`weights`,`profile`,{type:`profile`,id:e.id})} />
            <span class="wcard-head">
              <span class="wcard-radio" aria-hidden="true"></span>
              <span class="wcard-name">${e.label}</span>
              ${f?r`<span class="sbadge sbadge-now">${O.activeNow}</span>`:l}
            </span>
            ${e.badges.length?r`<span class="wcard-badges">${e.badges.map(e=>r`<span class="sbadge ${e===`recommended`?`sbadge-rec`:``}">${e===`recommended`?O.recommended:e}</span>`)}</span>`:l}
            <span class="wcard-meters">
              <span class="wmeter" title=${e.quality.source}>
                <span class="wmeter-k">${O.weightsTop1}</span>
                <span class="wmeter-v num">${p==null?`–`:`${p.toFixed(1)} %`}</span>
                <span class="wmeter-bar q"><span style="width:${p==null?0:Math.max(4,(p-80)/20*100)}%"></span></span>
              </span>
              <span class="wmeter">
                <span class="wmeter-k">${O.weightsSpeed}</span>
                <span class="wmeter-v num">${u?.prose&&u?.code?r`${u.prose.tok_s.toFixed(1)} / ${u.code.tok_s.toFixed(1)}`:r`<span class="muted">${c&&!c.comparable?O.notComparable:O.notMeasured}</span>`}</span>
                <span class="wmeter-bar s"><span style="width:${u?.code&&c?.comparable?u.code.tok_s/s*100:0}%"></span></span>
              </span>
            </span>
            <span class="wcard-facts num">
              <span><span class="wcard-k">${O.weightsKl}</span> ${e.quality.kl_vs_bf16==null?`–`:e.quality.kl_vs_bf16.toFixed(4)}</span>
              <span><span class="wcard-k">${O.weightsMemory}</span> ${e.weights_gib.toFixed(1)} GiB</span>
              <span><span class="wcard-k">${O.weightsDisk}</span> ${d?O.onDisk:O.toDownload(e.disk_gb)}</span>
            </span>
            ${e.note?r`<span class="wcard-note">${e.note}</span>`:l}
            ${e.hf_private&&!d&&n?.hf_token_present===!1?r`<span class="wcard-private is-blocked" data-private>${i(`warn`)} ${O.privateNoTokenShort}</span>`:l}
          </label>`})}
      </div>
      ${ce(e).length||e.community?r`<qse-community .model=${e} .sync=${e.community_sync??null} .selected=${t.profile} .device=${n} .syncing=${this.communitySyncing} .onUse=${e=>this.user(`weights`,`profile`,{type:`profile`,id:e})} .onRefresh=${()=>void this.syncCommunity()}></qse-community>`:l}`}stepComponents(e,t){let n=C(e,t.components);n.vision&&me(e,t.profile)&&(n.vision={disabled:!0,reason:O.community.visionOff,required:!1});let a=[...new Set(e.components.map(e=>e.group))],o=P(e,t.profile);return r`
      <div class="step-tools">
        <button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`components`})}>${i(`refresh`)} ${O.resetToRecommended}</button>
      </div>
      <p class="comp-notice" role="status" aria-live="polite" data-cascade>${this.wiz.notice??``}</p>
      ${a.map(a=>r`<section class="comp-group" aria-labelledby=${`cg-${a}`}>
          <h2 class="comp-group-title" id=${`cg-${a}`}>${O.groups[a]??a}</h2>
          <ul class="comp-list">
            ${e.components.filter(e=>e.group===a).map(e=>{let a=g(e,t.components),s=n[e.id],c=e.id===`drafters`&&o.drafters_gib!=null?o.drafters_gib:e.memory_gib;return r`<li>
                  <label class="comp ${a?`is-on`:``} ${s.disabled?`is-disabled`:``} ${s.required?`is-required`:``}" data-field=${`components.${e.id}`} data-component=${e.id}>
                    <input type="checkbox" class="comp-input" .checked=${a} ?disabled=${s.disabled} aria-describedby=${`cw-${e.id}${s.reason?` cr-${e.id}`:``}`} @change=${t=>this.user(`components`,e.id,{type:`component`,id:e.id,on:t.target.checked})} />
                    <span class="comp-box" aria-hidden="true">${i(`check`)}</span>
                    <span class="comp-main">
                      <span class="comp-label">${e.label}${e.default&&!e.required?r` <span class="comp-rec">${O.recommendedValue}</span>`:l}</span>
                      <span class="comp-what" id=${`cw-${e.id}`}>${e.what}</span>
                      ${s.reason?r`<span class="comp-reason" id=${`cr-${e.id}`} data-reason>${s.reason}</span>`:l}
                    </span>
                    <span class="comp-cost num">
                      <span class="comp-mem">${O.memoryCost(c)}</span>
                      ${e.memory_max_gib==null?l:r`<span class="muted">${O.memoryUpTo(e.memory_max_gib)}</span>`}
                      ${e.disk_gb>0?r`<span class="muted">${O.diskCost(e.disk_gb)}</span>`:l}
                    </span>
                  </label>
                </li>`})}
          </ul>
        </section>`)}`}stepParallel(e,t,n){let a=e.parallelism,o=t.side_window??t.window,s=n?.context_tokens_fit??0,c=Math.floor(s/t.slots),u=(e,t)=>this.user(`parallelism`,e,{type:`parallel`,key:e,value:t}),d=e.profiles.find(e=>e.id===t.profile),f=(e,t,n)=>r`<div class="win-ctl">
      <div class="seg seg-wrap" role="radiogroup" aria-label=${e===`window`?O.window:O.sideWindow}>
        ${Qe.filter(t=>t>=a[e].min&&t<=(e===`window`?a.window.max:n)).map(i=>{let a=i>n;return r`<button class="seg-btn ${t===i?`is-on`:``}" role="radio" aria-checked=${t===i} ?disabled=${a} title=${a&&d?O.windowAboveProfile(d.label,n):l} @click=${()=>u(e,i)}>${Q(i)}</button>`})}
      </div>
      <input class="num-input num" type="number" inputmode="numeric" min=${a[e].min} max=${n} step=${a[e].step} .value=${String(t)} aria-label=${`${e===`window`?O.window:O.sideWindow} (${O.tokensUnit})`} @change=${t=>u(e,Number(t.target.value))} />
      <span class="muted small">${O.tokensUnit}</span>
    </div>`;return r`<div class="spar">
      <div class="spar-fields">
        <div class="pfield" data-field="parallelism.slots">
          <div class="pfield-k"><span class="pfield-label" id="pf-slots">${O.slots}</span><span class="pfield-what">${O.slotsWhat}</span></div>
          <div class="seg" role="radiogroup" aria-labelledby="pf-slots">
            ${Array.from({length:a.slots.max-a.slots.min+1},(e,t)=>a.slots.min+t).map(e=>r`<button class="seg-btn ${t.slots===e?`is-on`:``}" role="radio" aria-checked=${t.slots===e} @click=${()=>u(`slots`,e)}>${e}${e===a.slots.default?r`<span class="seg-rec" title=${O.recommended}>•</span>`:l}</button>`)}
          </div>
        </div>
        <div class="pfield" data-field="parallelism.window">
          <div class="pfield-k"><span class="pfield-label">${O.window}</span><span class="pfield-what">${O.windowWhat}</span></div>
          ${f(`window`,t.window,F(e,t.profile))}
        </div>
        ${t.slots>1?r`<div class="pfield" data-field="parallelism.side_window">
              <div class="pfield-k"><span class="pfield-label">${O.sideWindow}</span><span class="pfield-what">${O.sideWindowWhat}</span></div>
              ${f(`side_window`,o,t.window)}
            </div>`:l}
        <div class="pfield" data-field="parallelism.kv_pool">
          <div class="pfield-k"><span class="pfield-label" id="pf-pool">${O.kvPool}</span><span class="pfield-what">${O.kvPoolWhat}</span></div>
          <label class="switch"><input type="checkbox" role="switch" .checked=${t.kv_pool} aria-labelledby="pf-pool" @change=${e=>u(`kv_pool`,e.target.checked)} /><span>${t.kv_pool?`on`:`off`}</span></label>
        </div>
        <div class="pfield" data-field="parallelism.floor_gb">
          <div class="pfield-k"><span class="pfield-label" id="pf-floor">${O.floor}</span><span class="pfield-what">${O.floorWhat}</span></div>
          <div class="floor-ctl">
            <input type="range" min=${a.floor_gb.min} max=${a.floor_gb.max} step="1" .value=${String(t.floor_gb)} aria-labelledby="pf-floor" @input=${e=>u(`floor_gb`,Number(e.target.value))} />
            <output class="num">${t.floor_gb} GiB</output>
          </div>
        </div>
        <div class="step-tools"><button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`parallelism`})}>${i(`refresh`)} ${O.resetToRecommended}</button></div>
        ${this.wiz.notice?r`<p class="comp-notice" role="status">${this.wiz.notice}</p>`:l}
      </div>
      <aside class="spar-readout panel" aria-live="polite" data-fit-tokens=${s}>
        <p class="spar-big num">${s.toLocaleString(`en-US`)}</p>
        <p class="spar-big-u">${O.fitsUnit}</p>
        <div class="spar-slots" aria-hidden="true">
          ${Array.from({length:t.slots},(e,n)=>{let i=n===0?t.window:o;return r`<div class="spar-slot">
              <span class="spar-slot-n num">${n+1}</span>
              <span class="spar-slot-track"><span class="spar-slot-win" style="width:${i/a.window.max*100}%"></span><span class="spar-slot-fit" style="width:${Math.min(i,c)/a.window.max*100}%"></span></span>
              <span class="spar-slot-k num">${Q(i)}</span>
            </div>`})}
        </div>
        <p class="small">${O.fitsPerSlot(c,t.slots)}</p>
        ${n?r`<p class="small muted">${O.worstCase(n.kv_worst_gb,n.fits_worst??!1)}</p>`:l}
      </aside>
    </div>`}fieldControl(e,t,n){let i=n=>this.user(`methods`,t.id,{type:`field`,group:e,id:t.id,value:n}),a=`mf-${t.id}`;if(t.type===`enum`){let e=t.options??[];return e.length<=3?r`<div class="seg seg-wrap" role="radiogroup" aria-labelledby=${`${a}-l`}>${e.map(e=>r`<button class="seg-btn ${n===e.value?`is-on`:``}" role="radio" aria-checked=${n===e.value} @click=${()=>i(e.value)}>${e.label}</button>`)}</div>`:r`<select id=${a} class="sel" @change=${e=>i(e.target.value)}>${e.map(e=>r`<option value=${e.value} ?selected=${n===e.value}>${e.label}</option>`)}</select>`}return t.type===`bool`?r`<label class="switch"><input id=${a} type="checkbox" role="switch" .checked=${n===!0} @change=${e=>i(e.target.checked)} /><span>${n?`on`:`off`}</span></label>`:t.type===`int`||t.type===`number`?r`<input id=${a} class="num-input num" type="number" inputmode=${t.type===`int`?`numeric`:`decimal`} min=${t.min??l} max=${t.max??l} step=${t.type===`int`?1:.05} .value=${String(n)} @change=${e=>{let t=e.target.value;i(t===``?NaN:Number(t))}} />`:r`<input id=${a} class="text-input" type="text" .value=${String(n)} spellcheck="false" autocapitalize="off" @change=${e=>i(e.target.value)} />`}fieldRow(e,t,n){let i=n[e][t.id]??t.recommended,a=S(t,i),o=t.options?.find(e=>e.value===t.recommended)?.label??String(t.recommended);return r`<div class="mfield ${a?`is-bad`:``}" data-field=${`methods.${t.id}`}>
      <div class="pfield-k">
        <label class="pfield-label" id=${`mf-${t.id}-l`} for=${`mf-${t.id}`}>${t.label}</label>
        <span class="pfield-what">${t.what}</span>
        <span class="mfield-rec muted small">${O.recommendedValue}: <span class="num">${o}</span>${i===t.recommended?l:r` <button class="linkish" @click=${()=>this.user(`methods`,t.id,{type:`field`,group:e,id:t.id,value:t.recommended})}>${O.resetToRecommended.toLowerCase()}</button>`}</span>
      </div>
      <div class="mfield-ctl">${this.fieldControl(e,t,i)}${a?r`<p class="mfield-err" role="alert">${O.invalidValue(a)}</p>`:l}</div>
    </div>`}stepMethods(e,t){let n=e.advanced.some(e=>S(e,t.advanced[e.id]??e.recommended));return r`<div class="methods">
      ${e.methods.map(e=>this.fieldRow(`methods`,e,t))}
      <details class="adv" ?open=${n}>
        <summary><span class="adv-title">${O.advanced}</span> <span class="muted small">${O.advancedNote}</span></summary>
        ${e.advanced.map(e=>this.fieldRow(`advanced`,e,t))}
      </details>
      <div class="step-tools"><button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`methods`})}>${i(`refresh`)} ${O.resetToRecommended}</button></div>
    </div>`}stepReview(e,t,n){let a=this.plan,o=this.wiz.step,s=n&&!n.fits?Se(e,t):[],u=this.finishBlockers(),d=this.device,f=P(e,t.profile),p=a?.state===`ready`?a.data:null;return r`<div class="review">
      <section class="rv-card rv-memory ${n&&!n.fits?`is-over`:`is-ok`}" data-review-memory>
        <h2 class="rv-h">${O.reviewMemory}</h2>
        ${n?r`<p class="rv-say">${n.fits?O.reviewFits(n.total_gb,n.budget_gb??n.device_gb-n.floor_gb):Te(e,t,n)}</p>`:l}
        ${s.length?r`<h3 class="rv-sub">${O.howToFix}</h3>
              <ul class="rv-fixes">${s.map(t=>r`<li><span>${t.text}</span><span class="muted small num">−${t.saves.toFixed(1)} GiB</span><a class="btn btn-sm" href=${`#/setup/${e.id}/${t.step}`}>${O.fixGo}: ${D[t.step]}</a></li>`)}</ul>`:l}
      </section>

      <section class="rv-card">
        <h2 class="rv-h">${O.reviewWeights}</h2>
        <p>${d?.active_profile===f.id?O.reviewWeightsSame(f.label):O.reviewWeightsSwitch(d?.active_profile?e.profiles.find(e=>e.id===d.active_profile)?.label??d.active_profile:null,f.label)}</p>
        ${p&&!p.profile.on_disk?r`<p class="muted">${O.reviewDownload(p.profile.download_gb)}</p>`:l}
        ${p?.downloads?.length?r`<ul class="rv-downloads" data-review-downloads>${p.downloads.map(e=>r`<li data-download=${e.id}><span>${e.label}</span> <span class="muted small">${e.for}</span> <span class="num">${e.gb.toFixed(2)} GB</span></li>`)}</ul>`:l}
        ${p?r`<p class="muted small">${p.restart?O.reviewRestart:O.reviewNoRestart}</p>`:l}
      </section>

      <section class="rv-card rv-changes">
        <h2 class="rv-h">${O.reviewChanges}</h2>
        ${a==null||a.state===`loading`?c(4):l}
        ${a?.state===`error`?r`<p class="rv-warn" role="alert">${i(`warn`)} ${O.planError} <span class="muted">${a.message}</span> <button class="btn btn-sm" @click=${()=>void this.loadPlan()}>${O.retry}</button></p>`:l}
        ${p?p.changes.length?r`<div class="table-scroll"><table class="table rv-table">
                <thead><tr><th scope="col">${O.reviewKey}</th><th scope="col">${O.reviewNow}</th><th scope="col">${O.reviewAfter}</th><th scope="col">${O.reviewFrom}</th></tr></thead>
                <tbody>${p.changes.map(e=>r`<tr data-env=${e.key}><th scope="row"><code>${e.key}</code></th><td class="rv-old">${e.current??r`<span class="muted">${O.reviewUnset}</span>`}</td><td class="rv-new">${e.removed?r`<span class="rv-removed" data-removed>${je(e,O.reviewUnset).to}</span>`:e.next??r`<span class="muted">${O.reviewUnset}</span>`}</td><td class="muted small">${e.from}</td></tr>`)}</tbody>
              </table></div>`:r`<p class="muted">${O.reviewNoChanges}</p>`:l}
        ${p&&(p.errors.length||p.warnings.length)?r`<ul class="rv-notes">
              ${p.errors.map((t,n)=>{let a=p.error_codes?.[n],s=xe(a);return r`<li class="is-err" data-error-code=${a??l}>${i(`warn`)} <span>${t}</span>${s&&s!==o?r` <a class="btn btn-sm" href=${`#/setup/${e.id}/${s}`}>${O.planFix}: ${D[s]}</a>`:l}</li>`})}
              ${p.warnings.map(e=>r`<li>${i(`info`)} ${e}</li>`)}
            </ul>`:l}
      </section>

      ${p?.commands.length?r`<section class="rv-card">
            <h2 class="rv-h">${O.reviewCommands}</h2>
            <p class="muted small">${O.reviewCommandsNote}</p>
            <pre class="rv-cmds">${p.commands.join(`
`)}</pre>
          </section>`:l}

      ${d&&!d.password_set?this.securityCard():d?.password_set?r`<p class="rv-secure muted">${i(`check`)} ${O.securitySet}</p>`:l}

      ${u.length?r`<ul class="rv-blockers" role="status" data-finish-blocked>${u.map(e=>r`<li>${i(`warn`)} ${e}</li>`)}</ul>`:l}
      ${this.applyError?r`<p class="rv-warn" role="alert">${i(`warn`)} ${this.applyError}</p>`:l}
    </div>`}get sec(){return this.wiz.selection?.advanced.admin_password===`generate`?`generate`:`keep`}loginOff(){let e=this.m?.advanced.find(e=>e.id===`dashboard_login`);return!!e&&(this.wiz.selection?.advanced.dashboard_login??e.recommended)!==`on`}securityCard(){let e=(e,t,n)=>r`<label class="sec-opt ${this.sec===e?`is-on`:``}">
      <input type="radio" name="sec" .checked=${this.sec===e} @change=${()=>this.user(`methods`,`admin_password`,{type:`field`,group:`advanced`,id:`admin_password`,value:e})} />
      <span><span class="sec-opt-label">${t}</span>${n?r`<span class="muted small">${n}</span>`:l}</span>
    </label>`;return r`<section class="rv-card rv-security" data-security-warning role="note">
      <h2 class="rv-h">${i(`warn`)} ${O.securityTitle}</h2>
      <p>${O.securityBody}</p>
      <div class="sec-opts" role="radiogroup" aria-label=${O.securityChoice}>
        ${e(`keep`,O.securityKeep)} ${e(`generate`,O.securityGenerate,O.securityGenerateNote)}
      </div>
      ${this.sec===`generate`&&this.loginOff()?r`<p class="sec-login" data-login-off>${i(`info`)} <span>${O.securityLoginOff}</span>
            <button class="btn btn-sm" @click=${()=>this.user(`methods`,`dashboard_login`,{type:`field`,group:`advanced`,id:`dashboard_login`,value:`on`})}>${O.securityLoginOn}</button></p>`:l}
    </section>`}renderConfirm(e){let t=this.plan?.state!==`ready`||this.plan.data.restart;return r`<dialog
      class="setup-confirm"
      aria-labelledby="sc-t"
      aria-describedby="sc-b"
      @cancel=${e=>this.applying&&e.preventDefault()}
      @close=${()=>this.confirmOpen=!1}
    >
      ${this.confirmOpen?r`<h2 id="sc-t">${O.confirmTitle}</h2>
            <p id="sc-b">${O.confirmBody(t)}</p>
            <p class="muted small">${P(e,this.wiz.selection.profile).label}</p>
            ${this.applyError?r`<p class="rv-warn" role="alert">${this.applyError}</p>`:l}
            <div class="setup-confirm-foot">
              <button class="btn" ?disabled=${this.applying} @click=${()=>this.confirmOpen=!1}>${O.cancel}</button>
              <button class="btn btn-primary setup-confirm-go" autofocus ?disabled=${this.applying} @click=${()=>void this.finish()}>${this.applying?O.installStarting:O.confirmGo}</button>
            </div>`:l}
    </dialog>`}updated(){let e=this.querySelector(`dialog.setup-confirm`);e&&(this.confirmOpen&&!e.open?e.showModal():!this.confirmOpen&&e.open&&e.close())}stepInstall(){let e=this.wiz.job;return e?r`<qse-install-progress .jobId=${e} .initial=${this.jobInitial} .previousProfile=${this.previousProfile} .onBack=${()=>this.dispatch({type:`job`,id:null})}></qse-install-progress>`:r`<p class="muted">${O.installStarting}</p>`}};e([t({attribute:!1})],$.prototype,`path`,void 0),e([d()],$.prototype,`wiz`,void 0),e([d()],$.prototype,`storageProblem`,void 0),e([d()],$.prototype,`storageFailed`,void 0),e([d()],$.prototype,`list`,void 0),e([d()],$.prototype,`model`,void 0),e([d()],$.prototype,`device`,void 0),e([d()],$.prototype,`serverEst`,void 0),e([d()],$.prototype,`estLocal`,void 0),e([d()],$.prototype,`plan`,void 0),e([d()],$.prototype,`confirmOpen`,void 0),e([d()],$.prototype,`applying`,void 0),e([d()],$.prototype,`applyError`,void 0),e([d()],$.prototype,`jobInitial`,void 0),e([d()],$.prototype,`previousProfile`,void 0),e([d()],$.prototype,`resetOpen`,void 0),e([d()],$.prototype,`unauthorized`,void 0),e([d()],$.prototype,`winNarrow`,void 0),e([d()],$.prototype,`communitySyncing`,void 0),$=e([n(`qse-setup`)],$);export{$ as QseSetup};