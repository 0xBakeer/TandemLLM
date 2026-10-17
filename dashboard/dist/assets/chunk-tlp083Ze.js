import{c as e,d as t,f as n,g as r,i,l as a,m as o,n as s,o as c,p as l,r as u,u as d,v as f}from"./chunk-5VjiyoiJ.js";import{A as p,C as m,E as h,M as g,N as _,S as ee,T as v,_ as te,a as y,c as ne,d as re,f as ie,h as ae,i as b,l as oe,m as se,n as x,o as S,p as ce,r as C,s as le,t as ue,u as de,v as fe,w,x as pe,y as me}from"./chunk-Z785vX97.js";import{a as he,n as ge}from"./chunk-BXmDkkzu.js";import{i as T,t as E}from"./chunk-CI4nmvus.js";import{Ct as _e,Tt as ve,f as D,g as O,h as ye,m as k,p as A}from"./app-B3ASbeuN.js";import{_ as j,a as be,c as xe,d as Se,f as M,g as Ce,h as N,i as we,l as P,m as F,n as Te,o as I,p as L,r as R,s as Ee,t as De,u as Oe,v as ke}from"./chunk-Bt_Ijxru.js";import{n as Ae,t as je}from"./chunk-dBfjuPm3.js";import{a as Me,c as Ne,d as Pe,f as Fe,i as Ie,l as Le,n as Re,o as ze,r as z,s as Be,t as Ve,u as He}from"./chunk-BkZ9FFqZ.js";var B=D.community,Ue={text:f`<path d="M4 5.5h12M4 9.5h12M4 13.5h7" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>`,mic:f`<rect x="7.25" y="3" width="5.5" height="9" rx="2.75" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M4.5 10a5.5 5.5 0 0 0 11 0M10 15.5v2M7.5 17.5h5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,video:f`<rect x="2.5" y="5" width="10.5" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m13 9 4.5-2.5v7L13 11z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>`,drafters:f`<circle cx="4.5" cy="10" r="1.8" fill="currentColor"/><circle cx="10" cy="10" r="1.8" fill="currentColor" opacity=".7"/><circle cx="15.5" cy="10" r="1.8" fill="currentColor" opacity=".45"/><path d="M6.3 10h1.9M11.8 10h1.9" stroke="currentColor" stroke-width="1.4"/>`,x:f`<path d="M4 3.5h3.6l8.4 13H12.4z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M15.5 3.5 4.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,huggingface:f`<circle cx="10" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="7.6" cy="9" r="1" fill="currentColor"/><circle cx="12.4" cy="9" r="1" fill="currentColor"/><path d="M7 12.2c.8 1.2 1.8 1.8 3 1.8s2.2-.6 3-1.8" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M4.2 7.2 3 6.2M15.8 7.2 17 6.2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,bluesky:f`<path d="M10 8.6C9 6.8 6.8 4.2 4.6 3.7 3.3 3.5 2.8 4.3 2.8 5.4c0 1.6.6 4.3 1.9 5 .9.4 2 .2 2.7 0-2 .5-2.9 1.4-2.1 2.9.9 1.8 3.2 2.5 4.7-.4 1.5 2.9 3.8 2.2 4.7.4.8-1.5-.1-2.4-2.1-2.9.7.2 1.8.4 2.7 0 1.3-.7 1.9-3.4 1.9-5 0-1.1-.5-1.9-1.8-1.7C13.2 4.2 11 6.8 10 8.6Z" fill="currentColor"/>`,mastodon:f`<path d="M4 15.5c-.9-1-1.5-2.6-1.5-5V8.3C2.5 5 4.6 3.5 10 3.5s7.5 1.5 7.5 4.8v1.5c0 2.6-1.4 3.9-4.2 4.1-1.9.2-3.5 0-4.3-.3v.9c0 1.4 2.3 1.5 5 1" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M6.3 11.8V8.4c0-1.1.6-1.7 1.6-1.7s1.6.6 1.6 1.7v1.9M13.7 11.8V8.4c0-1.1-.6-1.7-1.6-1.7s-1.6.6-1.6 1.7v1.9" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,website:f`<circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M3 10h14M10 3c2.2 2.1 3.3 4.4 3.3 7s-1.1 4.9-3.3 7c-2.2-2.1-3.3-4.4-3.3-7S7.8 5.1 10 3Z" fill="none" stroke="currentColor" stroke-width="1.4"/>`},V=(e,t=16)=>r`<svg class="icon" viewBox="0 0 20 20" aria-hidden="true" width=${t} height=${t}>${Ue[e]}</svg>`,We=e=>e===`text`?V(`text`):e===`image`?i(`image`):V(e===`audio`?`mic`:`video`),Ge=e=>e.toUpperCase(),H=class extends a{constructor(...e){super(...e),this.model=null,this.sync=null,this.selected=null,this.device=null,this.syncing=!1,this.onUse=()=>{},this.onRefresh=()=>{},this.query=``,this.sort=`newest`,this.droppedOpen=!1,this.noAvatar=new Set}render(){let e=this.model;if(!e)return l;let t=ie(e),n=this.sync;if(!t.length&&!n)return l;let a=ee(pe(t,this.query),this.sort),o=ne,s=ae(n?.add_url)??o;return r`<section class="cw" aria-labelledby="cw-h" data-community data-syncing=${this.syncing?``:l}>
      <header class="cw-head">
        <div class="cw-title">
          <h2 class="cw-h" id="cw-h">${B.heading}</h2>
          ${this.renderSource()}
        </div>
        <p class="cw-intro">${B.intro} <a href=${o} target="_blank" rel="noopener noreferrer">${B.addYoursHow} ${i(`external`)}</a></p>
        ${t.length?r`<div class="cw-tools">
              <label class="cw-search">
                ${i(`search`)}
                <input type="search" placeholder=${B.search} aria-label=${B.searchLabel} .value=${this.query} autocomplete="off" autocapitalize="off" spellcheck="false" @input=${e=>this.query=e.target.value} />
              </label>
              <label class="cw-sort"><span class="muted small">Sort</span>
                <select class="sel" aria-label=${B.sortLabel} @change=${e=>this.setSort(e.target.value)}>${oe.map(e=>r`<option value=${e} ?selected=${this.sort===e}>${de[e]}</option>`)}</select>
              </label>
              <span class="cw-count num muted small" data-count>${se(a.length,t.length)}</span>
              <a class="btn btn-sm cw-add" href=${s} target="_blank" rel="noopener noreferrer">${i(`plus`)} ${B.addYours}</a>
            </div>`:l}
      </header>
      ${n?.dropped.length?this.renderDropped(n.dropped):l}
      ${t.length?a.length?r`<div class="cw-grid" role="list">${a.map(e=>this.renderCard(e))}</div>`:r`<div class="cw-empty" data-community-nomatch role="status"><p class="cw-empty-t">${B.noMatch(this.query.trim())}</p><p><button class="btn btn-sm" @click=${()=>this.query=``}>${i(`x`)} ${B.clearSearch}</button></p></div>`:r`<div class="cw-empty" data-community-empty>
            <p class="cw-empty-t">${B.empty}</p>
            <p class="muted">${B.emptyBody}</p>
            <p class="cw-empty-acts"><a class="btn btn-primary btn-sm" href=${s} target="_blank" rel="noopener noreferrer">${i(`plus`)} ${B.addYours}</a> <a class="btn btn-sm" href=${o} target="_blank" rel="noopener noreferrer">${B.addYoursHow} ${i(`external`)}</a></p>
          </div>`}
    </section>`}setSort(e){te(e)&&(this.sort=e)}renderSource(){let e=this.sync,t=e?.source??`bundled`,n=e=>e&&Date.now()-Date.parse(e)<1e4?`just now`:ge(e),a=this.syncing?B.syncing:t===`github`?B.fromGithub(n(e?.fetched_at)):t===`cache`?B.fromCache(n(e?.fetched_at)):B.bundled;return r`<div class="cw-src" data-source=${this.syncing?`syncing`:t}>
      <span class="cw-src-dot" aria-hidden="true"></span>
      <span class="cw-src-text small" role="status">${a}</span>
      <button class="btn btn-ghost btn-sm cw-refresh" ?disabled=${this.syncing} aria-busy=${this.syncing} @click=${()=>this.onRefresh()}>${i(`refresh`)} ${B.refresh}</button>
      ${e?.error&&!this.syncing?r`<span class="cw-src-err small" role="alert">${i(`warn`)} ${B.syncFailed(e.error.replace(/\.?$/,`.`))}</span>`:l}
      ${e?.note&&!e.error&&!this.syncing?r`<span class="cw-src-note small muted">${e.note}</span>`:l}
    </div>`}renderDropped(e){return r`<div class="cw-dropped small" data-dropped>
      <p>${i(`info`)} <span>${B.dropped(e.length)}</span> <button class="linkish" aria-expanded=${this.droppedOpen} @click=${()=>this.droppedOpen=!this.droppedOpen}>${this.droppedOpen?B.droppedHide:B.droppedDetails}</button></p>
      ${this.droppedOpen?r`<ul>${e.map(e=>r`<li><code>${e.path}</code> <span class="muted">${e.reason}</span></li>`)}</ul>`:l}
    </div>`}renderCard(e){let t=this.selected===e.id,n=re(e.profile,this.device),a=e.status===`broken`,o=e.author,s=o.avatar&&!this.noAvatar.has(o.login);return r`<article class="ccard ${t?`is-selected`:``} ${a?`is-broken`:``}" role="listitem" data-community-card=${e.slug} data-field=${t?`weights.profile`:l} aria-labelledby=${`cc-${e.slug}`}>
      <header class="ccard-maker">
        <span class="ccard-avatar" aria-hidden="true">
          ${s?r`<img src=${o.avatar} alt="" width="40" height="40" loading="lazy" decoding="async" referrerpolicy="no-referrer" @error=${()=>this.avatarFailed(o.login)} />`:r`<span class="ccard-initials">${o.initials}</span>`}
        </span>
        <span class="ccard-who">
          <span class="ccard-handles">
            ${o.url?r`<a class="ccard-login" href=${o.url} target="_blank" rel="noopener noreferrer" title=${B.github(o.login)}>@${o.login}</a>`:r`<span class="ccard-login">@${o.login}</span>`}
            ${o.social.length?r`<span class="ccard-social">${o.social.map(e=>r`<a class="ccard-net tip-host" href=${e.href} target="_blank" rel="noopener noreferrer" aria-label=${e.label} data-tip=${e.label} data-net=${e.kind}>${V(e.kind)}</a>`)}</span>`:l}
          </span>
          ${o.name&&o.name.toLowerCase()!==o.login.toLowerCase()?r`<span class="ccard-author">${o.name}</span>`:l}
        </span>
        <span class="ccard-badges">
          ${n===`running`?r`<span class="sbadge sbadge-now">${D.activeNow}</span>`:l}
          ${this.statusBadge(e)}
        </span>
      </header>
      <h3 class="ccard-name" id=${`cc-${e.slug}`}>${e.name}</h3>
      <p class="ccard-base">${e.finetuneOf?r`${B.finetuneOf} ${e.finetuneUrl?r`<a href=${e.finetuneUrl} target="_blank" rel="noopener noreferrer">${e.finetuneOf}</a>`:r`<span>${e.finetuneOf}</span>`}`:B.stock}</p>
      ${e.description?r`<p class="ccard-desc">${e.description}</p>`:l}
      <dl class="ccard-readout num">
        <div><dt>format</dt><dd>${Ge(e.format)}</dd></div>
        <div><dt>${B.bpw}</dt><dd>${e.bpw==null?`–`:e.bpw.toFixed(2)}</dd></div>
        <div><dt>${B.download}</dt><dd>${n===`download`?`${e.sizeGb.toFixed(1)} GB`:D.onDisk}</dd></div>
      </dl>
      <ul class="ccard-modes" aria-label="Modalities">
        ${e.modalities.map(e=>r`<li class="ccard-mode tip-host ${e.runs?`is-on`:`is-off`}" tabindex="0" data-tip=${e.why} aria-label=${e.why} data-modality=${e.id}>${We(e.id)}<span>${e.label}</span></li>`)}
      </ul>
      <p class="ccard-drafters">${V(`drafters`)} <span>${e.drafters}</span></p>
      ${e.verified&&e.verified.speeds.length?r`<div class="ccard-speed" data-speed>
            ${e.verified.speeds.map(e=>r`<span class="ccard-sp"><span class="ccard-sp-n num">${e.tok_s.toFixed(1)}</span><span class="ccard-sp-k">${e.task}</span></span>`)}
            <span class="ccard-sp-u"><span class="num">tok/s</span><span class="ccard-sp-note">${B.speedNote(e.verified.device,e.verified.build,he(e.verified.date))}</span></span>
          </div>`:r`<p class="ccard-speed-none muted small">${B.notMeasured}</p>`}
      <footer class="ccard-foot">
        <span class="ccard-links">
          ${e.licenseUrl?r`<a class="sbadge ccard-license" href=${e.licenseUrl} target="_blank" rel="noopener noreferrer" title=${`${B.license}: ${e.license}`}>${e.license}</a>`:r`<span class="sbadge ccard-license" title=${`${B.license}: ${e.license}`}>${e.license}</span>`}
          ${e.hfUrl?r`<a class="ccard-hf" href=${e.hfUrl} target="_blank" rel="noopener noreferrer" title=${B.onHf(e.hfRepo)} data-hf-repo>${V(`huggingface`)} <span class="ccard-repo">${e.hfRepo}</span> ${i(`external`)}</a>`:r`<span class="ccard-hf" data-hf-repo>${V(`huggingface`)} <span class="ccard-repo">${e.hfRepo}</span></span>`}
        </span>
        <button class="btn ${t?``:`btn-primary`} ccard-use" ?disabled=${a} aria-pressed=${t} title=${a?B.brokenWhy:l} @click=${()=>!t&&this.onUse(e.id)}>${t?r`${i(`check`)} ${B.selected}`:B.use}</button>
      </footer>
    </article>`}statusBadge(e){return e.status===`verified`&&e.verified?r`<span class="sbadge sbadge-ok tip-host" tabindex="0" data-tip=${B.verifiedWhy(e.verified.device,e.verified.build,he(e.verified.date))}>${i(`check`)} ${B.verified}</span>`:e.status===`broken`?r`<span class="sbadge sbadge-err tip-host" tabindex="0" data-tip=${B.brokenWhy}>${i(`warn`)} ${B.broken}</span>`:r`<span class="sbadge tip-host" tabindex="0" data-tip=${B.submittedWhy}>${B.submitted}</span>`}avatarFailed(e){let t=new Set(this.noAvatar);t.add(e),this.noAvatar=t}};e([t({attribute:!1})],H.prototype,`model`,void 0),e([t({attribute:!1})],H.prototype,`sync`,void 0),e([t({attribute:!1})],H.prototype,`selected`,void 0),e([t({attribute:!1})],H.prototype,`device`,void 0),e([t({type:Boolean})],H.prototype,`syncing`,void 0),e([t({attribute:!1})],H.prototype,`onUse`,void 0),e([t({attribute:!1})],H.prototype,`onRefresh`,void 0),e([d()],H.prototype,`query`,void 0),e([d()],H.prototype,`sort`,void 0),e([d()],H.prototype,`droppedOpen`,void 0),e([d()],H.prototype,`noAvatar`,void 0),H=e([n(`qse-community`)],H);function Ke(e){return{title:D.removeTitle(e.label,e.gb),body:e.shared_with.length?D.removeShared(e.shared_with):D.removeBody(e.model_name),again:`bash ops/use-profile.sh ${e.profile} --fetch-only`}}function qe(e){if(!e)return[];let t=new Set,n=[];for(let r of[...e.weights].sort((e,t)=>Number(t.active)-Number(e.active)||t.gb-e.gb))t.has(r.removes)||(t.add(r.removes),n.push(r));return n}var U=class extends a{constructor(...e){super(...e),this.data=null,this.error=null,this.asking=null,this.busy=!1,this.done=null}connectedCallback(){super.connectedCallback(),this.load()}async load(){try{this.data=await p.weights(),this.error=null}catch(e){this.error=T(e).message}}async removeRow(e){this.busy=!0;try{let t=await p.remove([e.profile]);this.done=D.removed(e.label,t.freed_gb,t.log??`run/weights-removed.txt`),this.asking=null,this.dispatchEvent(new CustomEvent(`weights-removed`,{detail:t,bubbles:!0})),await this.load()}catch(e){this.error=T(e).message}finally{this.busy=!1}}render(){let e=qe(this.data);return!this.data&&!this.error?l:r`<section class="iw" aria-labelledby="iw-h" data-installed-weights>
      <header class="iw-head">
        <h2 class="iw-title" id="iw-h">${D.installedTitle}</h2>
        ${this.data?.free_gb==null?l:r`<span class="muted small num">${D.diskFree(this.data.free_gb)}</span>`}
      </header>
      ${this.error?r`<p class="rv-warn" role="alert">${i(`warn`)} ${this.error}</p>`:l}
      ${this.done?r`<p class="iw-done" role="status" data-removed>${i(`check`)} ${this.done}</p>`:l}
      ${e.length===0&&this.data?r`<p class="muted small">${D.installedNone}</p>`:l}
      <ul class="iw-list">
        ${e.map(e=>{let t=Ke(e);return r`<li class="iw-row ${e.active?`is-active`:``}" data-installed=${e.profile}>
            <div class="iw-main">
              <span class="iw-name">${e.label}</span>
              <span class="muted small">${e.model_name}${e.shared_with.length?` · ${D.sharedWith(e.shared_with)}`:``}</span>
            </div>
            <span class="iw-gb num">${e.gb.toFixed(1)} GB</span>
            ${e.active?r`<span class="sbadge sbadge-now" title=${D.runningNoRemove}>${D.activeNow}</span>`:r`<button class="btn btn-sm btn-ghost" data-remove=${e.profile} ?disabled=${!e.removable||this.busy} @click=${()=>this.asking=e.profile}>${i(`trash`)} ${D.removeGo}</button>`}
            ${this.asking===e.profile?r`<div class="iw-ask" role="alertdialog" aria-label=${t.title} data-remove-confirm>
                  <p class="iw-ask-t">${t.title}</p>
                  <p class="small">${t.body}</p>
                  <p class="small muted">${D.removeAgain} <code>${t.again}</code></p>
                  <div class="inst-actions">
                    <button class="btn btn-sm btn-danger" ?disabled=${this.busy} data-remove-yes @click=${()=>void this.removeRow(e)}>${this.busy?D.removing:D.removeYes}</button>
                    <button class="btn btn-sm" ?disabled=${this.busy} @click=${()=>this.asking=null}>${D.cancel}</button>
                  </div>
                </div>`:l}
          </li>`})}
      </ul>
    </section>`}};e([d()],U.prototype,`data`,void 0),e([d()],U.prototype,`error`,void 0),e([d()],U.prototype,`asking`,void 0),e([d()],U.prototype,`busy`,void 0),e([d()],U.prototype,`done`,void 0),U=e([n(`qse-installed-weights`)],U);function Je(e){return e.group===`long`?e.mode===`greedy`?`long_greedy`:`long_sampled`:`${e.mode}_${e.len}`}function Ye(e){return e===`everyday`?[`profile`,`prose`,`chat`,`code`,`ttft8`,`kl`,`lossless`]:[`profile`,`long32k`,`long64k`,`ttft32`,`kl`]}var Xe={profile:D.colWeights,prose:D.colProse,chat:D.colChat,code:D.colCode,long32k:D.colLong32,long64k:D.colLong64,ttft8:D.colTtft8,ttft32:D.colTtft32,kl:D.colKl,lossless:D.colLossless};function W(e,t,n,r){let i=t.cells[Je(r)]??{};switch(n){case`profile`:{let n=e.profiles.find(e=>e.id===t.profile);return n?.bpw??n?.weights_gib??null}case`ttft8`:return t.ttft_s[8192]??null;case`ttft32`:return t.ttft_s[32768]??null;case`kl`:return e.profiles.find(e=>e.id===t.profile)?.quality.kl_vs_bf16??null;case`lossless`:return t.lossless?t.lossless.identical:null;default:return i[n]?.tok_s??null}}function Ze(e,t){let n=e.measurements.rows,r=n.filter(e=>e.comparable),i=n.filter(e=>!e.comparable),a=n=>W(e,n,t.sort,t);return r.sort((e,n)=>{let r=a(e),i=a(n);return r==null&&i==null?0:r==null?1:i==null?-1:t.dir===`asc`?r-i:i-r}),{main:r,apart:i}}var G=[`prose`,`chat`,`code`,`long32k`,`long64k`],K=class extends a{constructor(...e){super(...e),this.model=null,this.selected=null,this.view={mode:`greedy`,len:`512`,group:`everyday`,sort:`code`,dir:`desc`}}set(e){let t={...this.view,...e};e.group&&!Ye(e.group).includes(t.sort)&&(t.sort=e.group===`long`?`long32k`:`code`),this.view=t}sortBy(e){this.view.sort===e?this.set({dir:this.view.dir===`asc`?`desc`:`asc`}):this.set({sort:e,dir:G.includes(e)||e===`lossless`?`desc`:`asc`})}render(){let e=this.model;if(!e)return l;let t=this.view,n=Ye(t.group),{main:i,apart:a}=Ze(e,t),o={};for(let n of G)o[n]=Math.max(0,...i.map(r=>W(e,r,n,t)??0));let s=(e,t,n,i)=>r`<div class="seg" role="radiogroup" aria-label=${i}>${t.map(([t,i])=>r`<button class="seg-btn ${e===t?`is-on`:``}" role="radio" aria-checked=${e===t} data-val=${t} @click=${()=>n(t)}>${i}</button>`)}</div>`;return r`<div class="mt">
      <div class="mt-controls">
        ${s(t.mode,[[`greedy`,D.modeGreedy],[`sampled`,D.modeSampled]],e=>this.set({mode:e}),`Decoding`)}
        ${t.group===`everyday`?s(t.len,[[`512`,D.tokens512],[`128`,D.tokens128]],e=>this.set({len:e}),`Answer length`):l}
        ${s(t.group,[[`everyday`,D.groupEveryday],[`long`,D.groupLong]],e=>this.set({group:e}),`Tasks`)}
      </div>
      <div class="table-scroll mt-scroll">
        <table class="table mt-table" data-mode=${Je(t)}>
          <thead>
            <tr>
              ${n.map(e=>{let n=t.sort===e,i=G.includes(e)?D.tokPerS:e===`ttft8`||e===`ttft32`?D.seconds:``;return r`<th scope="col" class=${e===`profile`?`mt-name`:`num-col`} aria-sort=${n?t.dir===`asc`?`ascending`:`descending`:`none`}>
                  <button class="mt-sort ${n?`is-on`:``}" data-col=${e} @click=${()=>this.sortBy(e)} aria-label=${D.sortBy(Xe[e])}>
                    <span>${Xe[e]}</span>${i?r`<span class="mt-unit">${i}</span>`:l}<span class="mt-arrow" aria-hidden="true">${n?t.dir===`asc`?`▴`:`▾`:``}</span>
                  </button>
                </th>`})}
            </tr>
          </thead>
          <tbody>
            ${i.map(r=>this.row(e,r,n,t,o))}
            ${a.length?r`<tr class="mt-apart-head"><td colspan=${n.length}>${D.notComparable}</td></tr>
                  ${a.map(r=>this.row(e,r,n,t,o,!0))}`:l}
          </tbody>
        </table>
      </div>
      <div class="mt-foot muted small">${D.measuredOn(e.measurements.build,e.measurements.date)} <details class="mt-method"><summary>${D.howMeasured}</summary><p>${e.measurements.method}</p></details></div>
    </div>`}row(e,t,n,i,a,o=!1){let s=e.profiles.find(e=>e.id===t.profile),c=t.profile===this.selected;return r`<tr class="mt-row ${c?`is-selected`:``} ${o?`is-apart`:``}" data-profile=${t.profile} aria-current=${c?`true`:l}>
      ${n.map(n=>{if(n===`profile`)return r`<th scope="row" class="mt-name">
            <span class="mt-label">${s?.label??t.profile}</span>
            ${c?r`<span class="mt-mine">${D.selectedRow}</span>`:l}
            ${o?r`<span class="mt-note" title=${t.note??``}>${D.notComparable}</span>`:l}
          </th>`;let u=W(e,t,n,i);if(u==null)return r`<td class="num-col mt-na">–</td>`;if(n===`kl`)return r`<td class="num-col num">${u.toFixed(4)}</td>`;if(n===`lossless`){let e=t.lossless;return r`<td class="num-col num" title=${D.losslessTitle(e.identical,e.near_tie)}>${e.identical} + ${e.near_tie}</td>`}if(n===`ttft8`||n===`ttft32`)return r`<td class="num-col num">${u.toFixed(1)}</td>`;let d=a[n]?u/a[n]*100:0;return r`<td class="num-col num mt-speed"><span class="mt-bar" style="--w:${d.toFixed(1)}" aria-hidden="true"></span><span class="mt-val">${u.toFixed(1)}</span></td>`})}
    </tr>`}};e([t({attribute:!1})],K.prototype,`model`,void 0),e([t({attribute:!1})],K.prototype,`selected`,void 0),e([d()],K.prototype,`view`,void 0),K=e([n(`qse-measure-table`)],K);var Qe=[`model`,`helpers`,`caches`,`engine`,`system`,`chats`],q=class extends a{constructor(...e){super(...e),this.model=null,this.estimate=null,this.available=null,this.local=!1,this.overText=null,this.open=!1,this.focusPart=null}render(){let e=this.estimate,t=this.model;if(!e||!t)return r`<section class="membar is-empty" aria-label=${D.memTitle}><div class="membar-line"><span class="membar-title">${D.memTitle}</span></div><div class="membar-track"></div></section>`;let n=e.budget_gb??e.device_gb-e.floor_gb,i=F(e),a=e.parts.find(e=>e.kind===`kv`),o=Math.max(e.device_gb,e.total_gb),s=e=>`${(e/o*100).toFixed(3)}%`,c=e=>Qe.indexOf(M(t,e)),u=[...e.parts].sort((e,t)=>c(e.id)-c(t.id)),d=0,f=u.map(e=>{let n={p:e,left:d,cat:M(t,e.id)};return d+=e.gb,n}),p=this.available==null?null:Math.max(0,e.device_gb-this.available),m=new Map;for(let e of f)m.set(e.cat,(m.get(e.cat)??0)+e.p.gb);let h=f.find(e=>e.p.id===this.focusPart),g=h?`${h.p.label}: ${h.p.gb.toFixed(2)} GiB${h.p.worst_gb!=null&&h.p.worst_gb!==h.p.gb?` (${D.memoryUpTo(h.p.worst_gb)})`:``}`:null,_=`${D.memTitle}: ${D.memFits(e.total_gb,n)}. ${i>0?this.overText??``:D.memLeft(n-e.total_gb)}`;return r`<section class="membar ${i>0?`is-over`:``} ${this.open?`is-open`:``}" aria-label=${D.memTitle} data-fits=${e.fits} data-total=${e.total_gb.toFixed(2)}>
      <div class="membar-line">
        <span class="membar-title">${D.memTitle}</span>
        <span class="membar-fig num" data-membar-total><b>${e.total_gb.toFixed(1)}</b><span class="membar-of"> / ${n.toFixed(1)} GiB</span></span>
        <span class="membar-say ${i>0?`is-over`:``}" data-membar-say role=${i>0?`alert`:l}>${i>0?this.overText??D.overSentence(i,null):D.memLeft(n-e.total_gb)}</span>
        <span class="membar-now num">${p==null?l:r`<i class="membar-now-dot" aria-hidden="true"></i>${D.memNow(p)}`}</span>
        <span class="membar-device num">${D.memDevice(e.device_gb)}</span>
        <button class="btn btn-ghost btn-sm membar-toggle" aria-expanded=${this.open} aria-controls="membar-more" aria-label=${this.open?D.memCollapse:D.memExpand} @click=${()=>this.open=!this.open}>
          <svg class="icon" viewBox="0 0 20 20" aria-hidden="true"><path d="m5 8 5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
        </button>
      </div>
      <div class="membar-track" role="img" aria-label=${_}>
        <span class="membar-free" style="left:${s(n)};width:${s(Math.max(0,e.device_gb-n))}" title=${`${D.memFloor}: ${e.floor_gb} GiB`}><span class="membar-floor-label">${D.memFloor} ${e.floor_gb} GiB</span></span>
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
        ${a&&a.worst_gb!=null&&a.worst_gb>a.gb?r`<span class="membar-worst" style="left:${s(e.total_gb)};width:${s(Math.min(a.worst_gb-a.gb,o-e.total_gb))}" title=${`${D.memWorst}: ${a.worst_gb.toFixed(1)} GiB`}></span>`:l}
        ${i>0?r`<span class="membar-over" style="left:${s(n)};width:${s(e.total_gb-n)}"></span>`:l}
        <span class="membar-floor" style="left:${s(n)}"></span>
        ${p==null?l:r`<span class="membar-nowtick" style="left:${s(Math.min(p,o))}" title=${D.memNow(p)}></span>`}
      </div>
      <div class="membar-more" id="membar-more">
        <ul class="membar-legend">
          ${Qe.filter(e=>m.has(e)).map(e=>r`<li class="cat-${e}"><i class="membar-sw" aria-hidden="true"></i>${D.memCat[e]} <span class="num">${(m.get(e)??0).toFixed(1)}</span></li>`)}
          <li class="cat-free"><i class="membar-sw" aria-hidden="true"></i>${D.memFloor} <span class="num">${e.floor_gb}</span></li>
        </ul>
        <p class="membar-detail" aria-live="polite">${g??(this.local?D.memLocal:e.notes[0]??`\xA0`)}</p>
      </div>
    </section>`}};e([t({attribute:!1})],q.prototype,`model`,void 0),e([t({attribute:!1})],q.prototype,`estimate`,void 0),e([t({attribute:!1})],q.prototype,`available`,void 0),e([t({type:Boolean})],q.prototype,`local`,void 0),e([t({attribute:!1})],q.prototype,`overText`,void 0),e([d()],q.prototype,`open`,void 0),e([d()],q.prototype,`focusPart`,void 0),q=e([n(`qse-memory-bar`)],q);var $e=[`set_field`,`toggle`,`select_profile`,`go_to_step`,`next`,`previous`,`explain`,`read_device`],J=(e,t=!1)=>({ok:!1,error:e,...t?{stale:t}:{}});function Y(e,t){return`${e}.${t}`}function et(e){if(e==null)return{};if(typeof e==`string`){if(!e.trim())return{};try{let t=JSON.parse(e);return t&&typeof t==`object`&&!Array.isArray(t)?t:null}catch{return null}}return typeof e==`object`&&!Array.isArray(e)?e:null}function X(e){if(typeof e==`number`&&Number.isInteger(e)&&e>=0&&e<x.length)return x[e];if(typeof e==`string`){if(/^\d+$/.test(e))return X(Number(e));if(b(e))return e}return null}function tt(e,t){return(e===`int`||e===`number`)&&typeof t==`string`&&t.trim()!==``&&Number.isFinite(Number(t))?Number(t):e===`bool`&&(t===`true`||t===`false`)?t===`true`:t}function nt(e){return typeof e==`boolean`?e?`on`:`off`:typeof e==`number`?e.toLocaleString(`en-US`):e}function rt(e,t){let n=String(e.name??``);if(!$e.includes(n))return/finish|apply|install|reset|delete|unload|load|confirm|submit/i.test(n)?J(D.tool.unknown(n)+` `+D.tool.userOnly(n)):J(D.tool.unknown(n));let r=et(e.args);if(!r)return J(D.tool.badArgs(`expected a JSON object`));if(n===`read_device`)return{ok:!0,kind:`read_device`};let i=t.model,a=t.selection,o=(e,n)=>{let r=t.touchedAt?.get(Y(e,n));return r!=null&&t.turnStartedAt!=null&&r>t.turnStartedAt};switch(n){case`go_to_step`:{let e=r.n??r.step??r.index,n=X(e);return n?n===`install`?J(D.tool.installStep):n!==`models`&&!i?J(D.tool.noModel):{ok:!0,kind:`action`,actions:[{type:n===`models`?`tiles`:`go`,step:n}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:D.toolDone.step(O[n]),result:D.tool.ok}:J(D.tool.noStep(String(e)))}case`next`:{if(!i)return J(D.tool.noModel);let e=S(t.step);if(e>=S(`review`))return J(D.tool.atEnd);let n=x[e+1];return{ok:!0,kind:`action`,actions:[{type:`next`}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:D.toolDone.step(O[n]),result:D.tool.ok}}case`previous`:{let e=S(t.step);if(e<=0)return J(D.tool.atStart);let n=x[e-1];return{ok:!0,kind:`action`,actions:[{type:`previous`}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:D.toolDone.step(O[n]),result:D.tool.ok}}}if(!i||!a)return J(D.tool.noModel);if(n===`explain`){let e=String(r.field??r.id??``),t=i.components.find(t=>t.id===e);if(t)return{ok:!0,kind:`explain`,target:{step:`components`,field:t.id},result:`${t.label}: ${t.what}${t.note?` `+t.note:``}`};let n=y(i,e);if(n)return{ok:!0,kind:`explain`,target:{step:`methods`,field:n.field.id},result:`${n.field.label}: ${n.field.what}`};let o=i.profiles.find(t=>t.id===e);if(o)return{ok:!0,kind:`explain`,target:{step:`weights`,field:`profile`},result:`${o.label}: KL ${o.quality.kl_vs_bf16??`?`}, ${o.weights_gib} GiB in memory${o.note?`. `+o.note:``}`};let s={slots:D.slotsWhat,window:D.windowWhat,side_window:D.sideWindowWhat(a.slots),kv_pool:D.kvPoolWhat,floor_gb:D.floorWhat};return s[e]?{ok:!0,kind:`explain`,target:{step:`parallelism`,field:e},result:s[e]}:J(D.tool.noField(`any`,e))}if(n===`select_profile`){let e=String(r.id??r.profile??``),t=i.profiles.find(t=>t.id===e);return t?o(`weights`,`profile`)?J(D.tool.stale,!0):a.profile===e?{ok:!0,kind:`action`,actions:[],undo:[],target:{step:`weights`,field:`profile`},summary:D.toolDone.profile(t.label),result:D.tool.same}:{ok:!0,kind:`action`,actions:[{type:`profile`,id:e}],undo:[{type:`profile`,id:a.profile}],target:{step:`weights`,field:`profile`},summary:D.toolDone.profile(t.label),result:D.tool.ok}:J(D.tool.noProfile(e))}if(n===`toggle`){let e=String(r.component??r.id??``),t=tt(`bool`,r.on);return typeof t==`boolean`?it(i,a,e,t,o(`components`,e)):J(D.tool.badArgs(`on must be true or false`))}let s=X(r.step);if(!s)return J(D.tool.noStep(String(r.step)));let c=String(r.field??``),l=le(i).find(e=>e.id===s)?.fields.find(e=>e.field===c);if(!l)return J(D.tool.noField(s,c));if(ue.has(c))return J(D.tool.secret(l.label));let u=tt(l.type,r.value),d=C(l,u);if(d)return J(D.tool.badArgs(`${c}: ${d}`));if(s===`components`)return it(i,a,c,u,o(s,c));if(o(s,c))return J(D.tool.stale,!0);let f={step:s,field:c},p=D.toolDone.field(l.label,nt(u));if(s===`weights`)return rt({name:`select_profile`,args:{id:u}},t);if(s===`parallelism`){let e=c,t=e===`side_window`?a.side_window??a.window:a[e];return t===u?{ok:!0,kind:`action`,actions:[],undo:[],target:f,summary:p,result:D.tool.same}:{ok:!0,kind:`action`,actions:[{type:`parallel`,key:e,value:u}],undo:[{type:`parallel`,key:e,value:t}],target:f,summary:p,result:D.tool.ok}}if(s===`methods`){let e=y(i,c);if(!e)return J(D.tool.noField(s,c));let t=a[e.group][c]??e.field.recommended;return t===u?{ok:!0,kind:`action`,actions:[],undo:[],target:f,summary:p,result:D.tool.same}:{ok:!0,kind:`action`,actions:[{type:`field`,group:e.group,id:c,value:u}],undo:[{type:`field`,group:e.group,id:c,value:t}],target:f,summary:p,result:D.tool.ok}}return J(D.tool.noField(s,c))}function it(e,t,n,r,i){let a=e.components.find(e=>e.id===n);if(!a)return J(D.tool.noComponent(n));if(a.required&&!r)return J(D.tool.required(a.label));let o=v(e,t.components)[n];if(r&&o.disabled&&!o.required)return J(o.reason??D.tool.noComponent(n));if(i)return J(D.tool.stale,!0);let s={step:`components`,field:n},c=r?D.toolDone.toggleOn(a.label):D.toolDone.toggleOff(a.label);return(a.required||(t.components[n]??a.default))===r?{ok:!0,kind:`action`,actions:[],undo:[],target:s,summary:c,result:D.tool.same}:{ok:!0,kind:`action`,actions:[{type:`component`,id:n,on:r}],undo:[{type:`components`,set:{...t.components}}],target:s,summary:c,result:D.tool.ok}}var Z=new Set([`explain`,`read_device`]);function at(e){if(!e.tools||!e.calls.length)return null;let t=e.round+1>=4;return e.calls.some(e=>!e.ok&&!Z.has(e.name))?e.retried?{tools:!1,retried:!0}:{tools:!t,retried:!0}:e.calls.every(e=>Z.has(e.name)&&e.name!==`read_device`)?e.text.trim()?null:{tools:!1,retried:e.retried}:{tools:!t,retried:e.retried}}function ot(e){let t=new Map,n=0,r=r=>{let i=new Map,a=null;for(let t=n;t<r;t++){let n=e[t];if(n.role===`assistant`)for(let e of n.tool_calls??[])i.set(e.id,e.name);if(n.role!==`tool`||Z.has(i.get(n.tool_call_id??``)??``))continue;let r={};try{r=JSON.parse(n.content)}catch{}r.ok===!1?a=r.stale||a===`stale`?`stale`:`unchanged`:r.ok===!0&&(a=null)}a&&r>n&&t.set(r-1,a)};return e.forEach((e,t)=>{e.role===`user`&&t!==0&&(r(t),n=t)}),r(e.length),t}function st(e,t){return t&&e.ok&&e.summary?e.summary:null}var ct=0;function lt(e){return e.charAt(0).toUpperCase()+e.slice(1)}var Q=class extends a{constructor(...e){super(...e),this.host=null,this.step=`models`,this.chat=[],this.finished=!1,this.phase=`checking`,this.status=null,this.figure=`sleeping`,this.progress=null,this.streaming=!1,this.partial=``,this.composer=``,this.error=null,this.notice=null,this.sources=[],this.open=!1,this.toast=null,this.chips=new Map,this.ctrl=null,this.loadPoll=null,this.ping=null,this.loadStartedAt=0,this.lastUserText=``,this.opener=null,this.toastTimer=null,this.onKey=e=>{if(!this.open||!this.sheetMode())return;if(e.key===`Escape`){e.preventDefault(),this.close();return}if(e.key!==`Tab`)return;let t=this.querySelector(`.wren-panel`);if(!t)return;let n=[...t.querySelectorAll(`button:not([disabled]), textarea:not([disabled]), a[href], input:not([disabled])`)].filter(e=>e.offsetParent!==null);if(!n.length)return;let r=n[0],i=n[n.length-1];e.shiftKey&&document.activeElement===r?(e.preventDefault(),i.focus()):!e.shiftKey&&document.activeElement===i?(e.preventDefault(),r.focus()):t.contains(document.activeElement)||(e.preventDefault(),r.focus())}}connectedCallback(){super.connectedCallback(),this.checkStatus(),this.ping=setInterval(()=>{this.phase===`ready`&&document.visibilityState===`visible`&&this.checkStatus(!0)},3e4),document.addEventListener(`keydown`,this.onKey)}disconnectedCallback(){this.ctrl?.abort(),this.loadPoll&&clearTimeout(this.loadPoll),this.ping&&clearInterval(this.ping),this.toastTimer&&clearTimeout(this.toastTimer),document.removeEventListener(`keydown`,this.onKey),this.lockScroll(!1),super.disconnectedCallback()}updated(e){e.has(`finished`)&&this.finished&&this.phase!==`finished`&&this.sleepForInstall()}emit(e){this.figure=ke(this.figure,e)}async checkStatus(e=!1){if(this.finished){this.phase=`finished`,this.figure=`sleeping`;return}try{let e=await p.assistant.status();this.applyStatus(e)}catch(t){if(e)return;this.phase=`failed`,this.error=T(t).status===0?D.wrenOffline:T(t).message,this.emit(`error`)}}applyStatus(e){switch(this.status=e,e.state){case`loaded`:this.phase!==`ready`&&(this.phase=`ready`,this.progress=null,this.figure=`loading`,this.emit(`loaded`));break;case`loading`:this.phase=`loading`,this.progress=this.timeProgress(e.loading_s??null),this.emit(`load-progress`),this.scheduleLoadPoll();break;case`refused`:this.phase=`refused`,this.emit(`error`);break;case`failed`:this.phase=`failed`,this.error=e.reason??D.wrenFailed,this.emit(`error`);break;default:this.phase=e.model==null?`refused`:`asleep`,this.figure=`sleeping`}}timeProgress(e){let t=e??(Date.now()-this.loadStartedAt)/1e3;return Math.min(.95,1-Math.exp(-t/8))}scheduleLoadPoll(){this.loadPoll&&clearTimeout(this.loadPoll),this.loadPoll=setTimeout(()=>void this.checkStatus(!0),500)}async wake(){this.error=null,this.loadStartedAt=Date.now(),this.phase=`loading`,this.progress=0,this.emit(`load-start`);try{this.applyStatus(await p.assistant.load())}catch(e){let t=T(e),n=e instanceof E?e.body:void 0;n?.assistant&&(this.status=n.assistant),t.status===409?(this.phase=`refused`,n?.assistant?.reason&&(this.status={...this.status,reason:n.assistant.reason})):(this.phase=`failed`,this.error=t.status===0?D.wrenOffline:t.message),this.emit(`error`)}}async sleep(){this.stop();try{this.applyStatus(await p.assistant.unload())}catch{}this.phase=`asleep`,this.emit(`unload`)}async sleepForInstall(){this.ctrl?.abort(),this.phase=`finished`,this.emit(`unload`),this.close();try{await p.assistant.unload()}catch{}}stop(){this.ctrl&&this.ctrl.abort()}async send(e){let t=e.trim();if(!t||this.streaming||!this.host)return;this.lastUserText=t,this.composer=``,this.error=null,this.notice=null,this.toast=null;let n=[...this.chat,{role:`user`,content:t}];this.setChat(n),await this.turn(n)}async retry(){if(!this.lastUserText)return;this.error=null;let e=this.chat[this.chat.length-1]?.role===`user`?this.chat:[...this.chat,{role:`user`,content:this.lastUserText}];this.setChat(e),await this.turn(e)}setChat(e){this.chat=e,this.host?.saveChat(e),this.updateComplete.then(()=>this.scrollLog())}async turn(e){let t=this.host,n=Date.now(),r=e;this.streaming=!0,this.emit(`send`),this.ctrl=new AbortController;let i=this.ctrl.signal,a=!0,o=!1,s=!1;try{for(let e=1;;e++){let c=``,l=[],u=[],d=[],f=!0,p=null;if(this.partial=``,await g({messages:r,page:t.pageState(),...a?{}:{tools:!1}},e=>{e.type===`state`?this.emit(e.state===`thinking`?`state-thinking`:e.state===`loading`?`load-progress`:`first-token`):e.type===`text`?(c+=e.delta,this.partial=c,f?(f=!1,this.emit(`first-token`)):this.emit(`delta`),this.fig?.blip(),this.fabFig?.blip(),this.scrollLog()):e.type===`tool_call`?l.push({id:e.id||`call_${++ct}`,name:e.name,args:e.args}):e.type===`done`?(this.sources=e.sources??[],e.unchanged&&(s=!0)):e.type===`error`&&(p=e.message)},i),p&&/another tab/i.test(p)){this.notice=D.wrenBusy;break}if(p)throw new E(`/v1/setup/assistant/chat`,500,p,`stream`);for(let e of l){let r=await t.execute(e,n);r.ok&&(this.emit(`tool-call`),this.fig?.hop(),this.emit(`acted`)),u.push({role:`tool`,tool_call_id:e.id,content:JSON.stringify(r.ok?{ok:!0,result:r.result,...r.summary?{summary:r.summary}:{}}:{ok:!1,error:r.result,...r.stale?{stale:!0}:{}})}),this.chips=new Map(this.chips).set(e.id,{summary:r.summary,ok:r.ok,stale:r.stale,undo:r.undo,undone:!1}),d.push({name:e.name,ok:r.ok,changed:!!r.summary});let i=st(r,this.open&&this.sheetMode());i&&this.showToast(e.id,i,r.reveal??null)}let m={role:`assistant`,content:c,...l.length?{tool_calls:l}:{}};r=[...r,m,...u],this.partial=``,this.setChat(r);let h=at({round:e,tools:a,text:c,calls:d,retried:o});if(!h)break;({tools:a,retried:o}=h)}s&&!ot(r).has(r.length-1)&&(this.notice=D.wrenNotChanged),this.emit(`done`)}catch(e){if(e.name===`AbortError`)this.partial&&this.setChat([...r,{role:`assistant`,content:`${this.partial} ${D.wrenStopped}`}]),this.partial=``,this.emit(`stop`);else{let t=T(e);t.status===409?this.notice=D.wrenBusy:t.status===0?this.error=D.wrenOffline:t.status===503||/unload|asleep|not loaded/i.test(t.message)?(this.error=D.wrenWentToSleep,this.checkStatus(!0)):this.error=t.message||D.wrenError,this.partial&&this.setChat([...r,{role:`assistant`,content:this.partial}]),this.partial=``,this.emit(t.status===409?`done`:`error`)}}finally{this.streaming=!1,this.ctrl=null}}undoChip(e){let t=this.chips.get(e);t?.undo&&!t.undone&&(t.undo(),this.chips=new Map(this.chips).set(e,{...t,undone:!0}))}scrollLog(){let e=this.log;e&&(e.scrollTop=e.scrollHeight)}sheetMode(){return typeof matchMedia==`function`&&matchMedia(`(max-width: 1100px)`).matches}toggleOpen(){this.open?this.close():(this.opener=document.activeElement,this.open=!0,this.lockScroll(!0),this.updateComplete.then(()=>(this.querySelector(`.wren-panel .wren-input, .wren-panel .btn-primary, .wren-close`)??null)?.focus()))}close(){this.open&&(this.open=!1,this.lockScroll(!1),this.opener?.focus?.())}lockScroll(e){document.documentElement.classList.toggle(`wren-sheet-open`,e&&this.sheetMode())}showToast(e,t,n){this.toast={id:e,summary:t,reveal:n},this.toastTimer&&clearTimeout(this.toastTimer),this.toastTimer=setTimeout(()=>this.toast=null,12e3)}async showMe(){let e=this.toast?.reveal;this.toast=null,this.close(),await this.updateComplete,requestAnimationFrame(()=>e?.())}statusWord(){return this.phase===`finished`?D.wrenFinished:this.phase===`asleep`?D.wrenAsleep:this.phase===`refused`?D.wrenCannotLoad:this.phase===`failed`?D.wrenFailed:this.phase===`loading`?D.wrenWaking(this.progress):D.wrenStatus[this.figure]??``}render(){let e=this.figure;return r`
      <button class="wren-fab ${this.open?`is-open`:``}" aria-label=${this.open?D.wrenClose:D.wrenOpen} aria-expanded=${this.open} aria-controls="wren-panel" @click=${()=>this.toggleOpen()} data-figure=${e}>
        <qse-wren-figure class="wren-fab-fig" .state=${e} .size=${50} .progress=${this.progress}></qse-wren-figure>
      </button>
      <div class="wren-scrim ${this.open?`is-open`:``}" @click=${()=>this.close()}></div>
      <aside id="wren-panel" class="wren-panel ${this.open?`is-open`:``}" aria-label=${D.wrenName} data-phase=${this.phase} data-figure=${e} role="complementary">
        <div class="wren-handle" aria-hidden="true"></div>
        <header class="wren-head">
          <qse-wren-figure class="wren-main" .state=${e} .size=${76} .progress=${this.progress}></qse-wren-figure>
          <div class="wren-titles">
            <h2 class="wren-name">${D.wrenName}</h2>
            <p class="wren-status" role="status" aria-live="polite" data-wren-status>${this.statusWord()}</p>
            ${this.status?.model&&this.phase===`ready`?r`<p class="wren-model">${this.status.model}${this.status.fallback?r` <span title=${D.wrenFallback}>(${D.wrenFallback})</span>`:l}</p>`:l}
          </div>
          <div class="wren-head-tools">
            ${this.phase===`ready`&&!this.streaming?r`<button class="btn btn-ghost btn-icon" title=${D.wrenSleep} aria-label=${D.wrenSleep} @click=${()=>void this.sleep()}>${i(`moon`)}</button>`:l}
            <button class="btn btn-ghost btn-icon wren-close" aria-label=${D.wrenClose} title=${D.wrenClose} @click=${()=>this.close()}>${i(`x`)}</button>
          </div>
        </header>
        ${this.renderBody()}
      </aside>
    `}hero(){return r`<qse-wren-figure class="wren-hero" .state=${this.figure} .size=${148} .progress=${this.progress}></qse-wren-figure>`}renderBody(){switch(this.phase){case`checking`:return r`<div class="wren-body wren-center">${this.hero()}</div>`;case`finished`:return r`<div class="wren-body wren-center" data-wren-finished>${this.hero()}<p>${D.wrenFinished}</p></div>${this.renderComposer(!0)}`;case`asleep`:return r`<div class="wren-body wren-center">
            ${this.hero()}
            <p class="wren-lead">${D.wrenAsleep}</p>
            <p class="muted small">${D.wrenAsleepBody}</p>
            <button class="btn btn-primary wren-wake" @click=${()=>void this.wake()}>${D.wrenWake}</button>
          </div>`;case`loading`:return r`<div class="wren-body wren-center">
            ${this.hero()}
            <p class="wren-lead">${D.wrenWaking(this.progress)}</p>
            <div class="wren-loadbar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${this.progress==null?l:Math.round(this.progress*100)} aria-label=${D.wrenWaking(null)}><span style="width:${Math.round((this.progress??0)*100)}%"></span></div>
          </div>`;case`refused`:{let e=this.status,t=e?.model==null&&e!=null?D.wrenNoModel:D.wrenCannotLoadMemory(e?.claim_gib??null,this.host?.memAvailable()??null);return r`<div class="wren-body wren-center" data-wren-refused>
            ${this.hero()}
            <p class="wren-lead">${D.wrenCannotLoad}</p>
            <p class="muted small">${e?.reason?lt(e.reason):t}</p>
            ${e?.reason?r`<p class="muted small">${t}</p>`:l}
            ${e?.model==null?l:r`<button class="btn wren-wake" @click=${()=>void this.wake()}>${i(`refresh`)} ${D.wrenRetry}</button>`}
          </div>`}case`failed`:return r`<div class="wren-body wren-center" data-wren-failed>
            ${this.hero()}
            <p class="wren-lead">${D.wrenFailed}</p>
            <p class="muted small">${this.error??``}</p>
            <button class="btn wren-wake" @click=${()=>void this.wake()}>${i(`refresh`)} ${D.wrenRetry}</button>
          </div>`;case`ready`:return this.renderChat()}}renderChat(){let e=this.chat,t=ot(e);return r`
      <div class="wren-body wren-log" aria-live="off">
        ${e.length===0?r`<p class="wren-msg is-assistant wren-hello">${D.wrenHello}</p>`:l}
        ${_e(e.map((e,t)=>({m:e,i:t})),({i:e})=>e,({m:n,i})=>r`${this.renderMessage(n,i===e.length-1)}${t.has(i)?r`<p class="wren-note" data-tool-rejected>${t.get(i)===`stale`?D.tool.stale:D.wrenNotChanged}</p>`:l}`)}
        ${this.partial?r`<div class="wren-msg is-assistant is-streaming">${Ae(je(this.partial))}</div>`:l}
        ${this.streaming&&!this.partial?r`<div class="wren-msg is-assistant wren-typing" aria-hidden="true"><span></span><span></span><span></span></div>`:l}
        ${this.notice?r`<p class="wren-note">${this.notice}</p>`:l}
        ${this.error?r`<div class="wren-error" role="alert" data-wren-error>
              <span>${this.error}</span>
              <button class="btn btn-sm" @click=${()=>void this.retry()}>${i(`refresh`)} ${D.wrenRetry}</button>
            </div>`:l}
      </div>
      ${this.toast?r`<div class="wren-toast" role="status" data-wren-toast>
            <span class="wren-chip-dot" aria-hidden="true"></span>
            <span class="wren-toast-text">${this.toast.summary}</span>
            ${this.toast.reveal?r`<button class="btn btn-sm" @click=${()=>void this.showMe()}>${D.wrenShowMe}</button>`:l}
          </div>`:l}
      ${this.renderChips()} ${this.renderComposer(!1)}
    `}renderMessage(e,t){if(e.role===`user`)return r`<div class="wren-msg is-user">${e.content}</div>`;if(e.role===`assistant`)return e.content.trim()?r`<div class="wren-msg is-assistant">${Ae(je(e.content))}
        ${t&&this.sources.length?r`<p class="wren-sources">${D.wrenSources} ${this.sources.map((e,t)=>r`${t?`, `:``}<span title=${e.path}>${e.title||e.path}</span>`)}</p>`:l}
      </div>`:l;let n={};try{n=JSON.parse(e.content)}catch{}let i=e.tool_call_id?this.chips.get(e.tool_call_id):void 0;if(n.ok===!1||i?.ok===!1)return l;let a=i?.summary??n.summary;return a?r`<div class="wren-chip ${i?.undone?`is-undone`:``} ${this.toast?.id===e.tool_call_id?`is-fresh`:``}" data-tool-chip>
      <span class="wren-chip-dot" aria-hidden="true"></span>
      <span>${a}</span>
      ${i?.undo?r`<button class="btn btn-ghost btn-sm" ?disabled=${i.undone} @click=${()=>this.undoChip(e.tool_call_id)}>${i.undone?D.wrenUndone:D.wrenUndo}</button>`:l}
    </div>`:l}renderChips(){if(this.streaming)return l;let e=D.wrenChips[this.step]??[];return r`<div class="wren-suggest">${e.map(e=>r`<button class="wren-suggestion" @click=${()=>void this.send(e)}>${e}</button>`)}</div>`}renderComposer(e){return r`<form
      class="wren-composer"
      @submit=${e=>{e.preventDefault(),this.send(this.composer)}}
    >
      <label class="sr-only" for="wren-input">${D.wrenPlaceholder}</label>
      <textarea
        id="wren-input"
        class="wren-input"
        rows="1"
        .value=${this.composer}
        ?disabled=${e}
        placeholder=${e?D.wrenFinished:D.wrenPlaceholder}
        @focus=${()=>this.emit(`focus`)}
        @blur=${()=>!this.composer&&this.emit(`blur`)}
        @input=${e=>{this.composer=e.target.value,this.emit(`input`)}}
        @keydown=${e=>{e.key===`Enter`&&!e.shiftKey&&!e.isComposing&&(e.preventDefault(),this.send(this.composer))}}
      ></textarea>
      ${this.streaming?r`<button type="button" class="btn wren-stop" @click=${()=>this.stop()} aria-label=${D.wrenStop}>${i(`stop`)}<span>${D.wrenStop}</span></button>`:r`<button type="submit" class="btn btn-primary wren-send" ?disabled=${e||!this.composer.trim()} aria-label=${D.wrenSend}>${i(`send`)}</button>`}
    </form>`}};e([t({attribute:!1})],Q.prototype,`host`,void 0),e([t({attribute:!1})],Q.prototype,`step`,void 0),e([t({attribute:!1})],Q.prototype,`chat`,void 0),e([t({type:Boolean})],Q.prototype,`finished`,void 0),e([d()],Q.prototype,`phase`,void 0),e([d()],Q.prototype,`status`,void 0),e([d()],Q.prototype,`figure`,void 0),e([d()],Q.prototype,`progress`,void 0),e([d()],Q.prototype,`streaming`,void 0),e([d()],Q.prototype,`partial`,void 0),e([d()],Q.prototype,`composer`,void 0),e([d()],Q.prototype,`error`,void 0),e([d()],Q.prototype,`notice`,void 0),e([d()],Q.prototype,`sources`,void 0),e([d()],Q.prototype,`open`,void 0),e([d()],Q.prototype,`toast`,void 0),e([d()],Q.prototype,`chips`,void 0),e([_(`qse-wren-figure.wren-main`)],Q.prototype,`fig`,void 0),e([_(`qse-wren-figure.wren-fab-fig`)],Q.prototype,`fabFig`,void 0),e([_(`.wren-log`)],Q.prototype,`log`,void 0),e([_(`.wren-input`)],Q.prototype,`input`,void 0),Q=e([n(`qse-wren`)],Q);var $=class extends a{constructor(...e){super(...e),this.path=[],this.wiz=ze(),this.storageProblem=null,this.storageFailed=!1,this.list={state:`loading`},this.model=null,this.device=null,this.serverEst=null,this.estLocal=!1,this.plan=null,this.confirmOpen=!1,this.applying=!1,this.removeAfter=!1,this.applyError=null,this.jobInitial=null,this.previousProfile=null,this.resetOpen=!1,this.unauthorized=!1,this.winNarrow=!1,this.slotsDraft=null,this.winDraft={window:null,side_window:null},this.nextRefused=!1,this.communitySyncing=!1,this.store=null,this.devPoll=null,this.estTimer=null,this.estCtrl=null,this.planTimer=null,this.planCtrl=null,this.commCtrl=null,this.warnedDiff=!1,this.touchedAt=new Map,this.loadingModel=null,this.onUnauthorized=()=>{this.unauthorized=!0},this.onSignedIn=()=>{this.unauthorized=!1,this.list.state!==`ready`&&this.loadList();let e=this.path[0];e&&this.model?.state!==`ready`&&this.loadModel(e),this.loadDevice(),this.wiz.selection&&this.queueEstimate(),this.wiz.step===`review`&&this.queuePlan(),this.wiz.step===`weights`&&this.syncCommunity()},this.onReset=()=>{this.wiz=ze(),this.plan=null,this.jobInitial=null,location.hash!==`#/setup`&&(location.hash=`#/setup`)},this.host={pageState:()=>De(this.m,this.wiz.step,this.wiz.selection,this.currentEstimate(),{errors:this.visibleErrors()}),execute:(e,t)=>this.executeTool(e,t),saveChat:e=>this.dispatch({type:`chat`,chat:e}),memAvailable:()=>this.device?.mem_available_gb??null}}connectedCallback(){super.connectedCallback(),this.store=Ne();let e=Be(this.store);this.wiz=e.state,this.storageProblem=e.problem,(e.problem===`corrupt`||e.problem===`version`)&&Me(this.store),this.loadList(),this.devPoll=ve(()=>this.loadDevice(),5e3),this.devPoll.start(),window.addEventListener(`qse-setup-reset`,this.onReset),window.addEventListener(`qse-setup-unauthorized`,this.onUnauthorized),window.addEventListener(`qse-signed-in`,this.onSignedIn)}disconnectedCallback(){this.devPoll?.stop(),this.estCtrl?.abort(),this.planCtrl?.abort(),this.commCtrl?.abort(),window.removeEventListener(`qse-setup-reset`,this.onReset),window.removeEventListener(`qse-setup-unauthorized`,this.onUnauthorized),window.removeEventListener(`qse-signed-in`,this.onSignedIn),super.disconnectedCallback()}async loadList(){this.list={state:`loading`};try{let e=await p.models();this.list=e.models.length||e.errors.length?{state:`ready`,data:e}:{state:`empty`,note:D.modelsEmpty}}catch(e){let t=T(e);this.list={state:`error`,endpoint:t.endpoint,message:t.status===401?D.unauthorized:t.message}}}async loadModel(e){if(this.loadingModel!==e){this.loadingModel=e,this.model={state:`loading`};try{let t=w(await p.model(e));this.model={state:`ready`,data:t},await this.openModel(t,this.routeStep()??void 0)}catch(t){let n=T(t);this.model={state:`error`,endpoint:n.endpoint,message:n.status===404?`${D.modelsError} (${e})`:n.message}}finally{this.loadingModel=null}}}async loadDevice(){try{this.device=await p.device()}catch{}}async openModel(e,t){let n=null;if(!(this.wiz.model===e.id&&this.wiz.selection))try{n=(await p.current(e.id)).selection}catch{n=null}this.dispatch({type:`open`,model:e,step:t,seed:n},!1),this.wiz.step===`weights`&&this.syncCommunity()}get m(){return this.model?.state===`ready`?this.model.data:null}routeStep(){let e=this.path[1];return b(e)?e:null}willUpdate(e){e.has(`path`)&&this.syncFromRoute()}syncFromRoute(){let[e,t]=this.path;if(!e){this.wiz.step!==`models`&&(this.wiz={...this.wiz,step:`models`});return}let n=this.m;if(!n||n.id!==e){this.loadModel(e);return}if(this.wiz.model!==e||!this.wiz.selection){this.openModel(n,b(t)?t:void 0);return}let r=b(t)?t:`overview`;r===`install`&&!this.wiz.job&&(r=`review`),r===`models`&&(r=`overview`),r!==this.wiz.step&&this.dispatch({type:`go`,step:r},!1)}hashFor(e){return e.step===`models`||!e.model?`#/setup`:`#/setup/${e.model}/${e.step}`}dispatch(e,t=!0){let n=Le(this.wiz,e,this.m);if(n===this.wiz)return;let r=n.step!==this.wiz.step;this.wiz=n,this.store&&(this.storageFailed=!Pe(this.store,n));let i=this.hashFor(n);location.hash!==i&&(t?location.hash=i:history.replaceState(null,``,i)),(r||e.type===`recommended`||e.type===`parallel`&&e.key===`slots`)&&(this.slotsDraft=null,this.nextRefused=!1),(r||e.type===`recommended`||e.type===`profile`)&&(this.winDraft={window:null,side_window:null}),r&&(t&&window.scrollTo({top:0,behavior:j()?`auto`:`smooth`}),n.step===`review`&&this.queuePlan(),n.step===`weights`&&this.syncCommunity()),e.type!==`go`&&e.type!==`next`&&e.type!==`previous`&&e.type!==`chat`&&e.type!==`dismiss`&&(this.queueEstimate(),n.step===`review`&&this.queuePlan())}user(e,t,n){this.touchedAt.set(Y(e,t),Date.now()),this.dispatch(n)}selKey(e){return JSON.stringify(e)}queueEstimate(){this.estTimer&&clearTimeout(this.estTimer),this.estTimer=setTimeout(()=>void this.confirmEstimate(),150)}async confirmEstimate(){let e=this.wiz.selection;if(!e)return;this.estCtrl?.abort();let t=this.estCtrl=new AbortController,n=this.selKey(e);try{let r=await p.estimate(e,t.signal);if(t.signal.aborted||n!==this.selKey(this.wiz.selection))return;this.serverEst={key:n,est:r},this.estLocal=!1;let i=this.m?L(this.m,e):null;i&&Math.abs(i.total_gb-r.total_gb)>.05&&!this.warnedDiff&&(this.warnedDiff=!0,console.warn(`setup: the server's estimate (${r.total_gb} GiB) and the page's (${i.total_gb} GiB) differ; showing the server's`))}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;this.estLocal=!0}}currentEstimate(){let e=this.m,t=this.wiz.selection;if(!e||!t)return null;let n=L(e,t,this.device?.mem_available_gb??null),r=this.serverEst;return r&&r.key===this.selKey(t)&&(!n||Math.abs(r.est.total_gb-n.total_gb)>.05)?{...r.est,available_gb:this.device?.mem_available_gb??r.est.available_gb}:n}queuePlan(){this.planTimer&&clearTimeout(this.planTimer),this.planTimer=setTimeout(()=>void this.loadPlan(),200)}async loadPlan(){let e=this.wiz.selection;if(!e)return;this.planCtrl?.abort();let t=this.planCtrl=new AbortController;(!this.plan||this.plan.state!==`ready`)&&(this.plan={state:`loading`});try{let n=await p.plan(e,t.signal);t.signal.aborted||(this.plan={state:`ready`,data:n})}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;let n=T(e);this.plan={state:`error`,endpoint:n.endpoint,message:n.message}}}async syncCommunity(){let e=this.m;if(!e)return;this.commCtrl?.abort();let t=this.commCtrl=new AbortController;this.communitySyncing=!0;try{let n=await p.communitySync(t.signal);if(t.signal.aborted||this.m?.id!==e.id)return;let r=this.m,i={...n.sync,dropped:n.sync.dropped.filter(t=>t.model===e.id),add_url:r.community_sync?.add_url||n.sync.add_url},a=w({...r,community_sync:i},n.models[e.id]??[]);this.model={state:`ready`,data:a};let o=this.wiz.selection;o&&!a.profiles.some(e=>e.id===o.profile)?this.dispatch({type:`open`,model:a},!1):this.queueEstimate()}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;let n=this.m;n?.community_sync&&!(e instanceof E&&e.status===404)&&(this.model={state:`ready`,data:{...n,community_sync:{...n.community_sync,error:T(e).message,checked_at:new Date().toISOString()}}})}finally{t.signal.aborted||(this.communitySyncing=!1)}}visibleErrors(){let e=this.issues()[this.wiz.step]??[];return this.wiz.step!==`review`||this.plan?.state!==`ready`?e:[...e,...this.plan.data.errors.filter(t=>!e.includes(t))]}async executeTool(e,t){let n=rt(e,{model:this.m,selection:this.wiz.selection,step:this.wiz.step,touchedAt:this.touchedAt,turnStartedAt:t});if(!n.ok)return{ok:!1,result:n.error,summary:null,undo:null,stale:n.stale};if(n.kind===`read_device`)try{let e=await p.device();this.device=e;let t={mode:e.mode,mem_total_gb:e.mem_total_gb,mem_available_gb:e.mem_available_gb,gpu:e.gpu,disk:e.disk,engine:e.engine,active_profile:e.active_profile,profiles_on_disk:e.profiles_on_disk,password_set:e.password_set};return{ok:!0,result:JSON.stringify(t),summary:null,undo:null}}catch(e){return{ok:!1,result:T(e).message,summary:null,undo:null}}if(n.kind===`explain`)return n.target.step===this.wiz.step&&this.highlight(n.target.step,n.target.field),{ok:!0,result:n.result,summary:null,undo:null};let{target:r}=n;r.field&&r.step!==this.wiz.step&&this.dispatch({type:`go`,step:r.step});for(let e of n.actions)this.dispatch(e);await this.updateComplete,r.field&&this.highlight(r.step,r.field);let i=n.undo.length?()=>{for(let e of n.undo)this.dispatch(e);r.field&&this.highlight(r.step,r.field)}:null,a=r.field?()=>{r.step!==this.wiz.step&&this.dispatch({type:`go`,step:r.step}),this.highlight(r.step,r.field)}:null;return{ok:!0,result:n.result,summary:n.actions.length?n.summary:null,undo:i,reveal:a}}async highlight(e,t){if(!t)return;await this.updateComplete;let n=this.querySelector(`[data-field="${CSS.escape(Y(e,t))}"]`);if(!n)return;let r=n.getBoundingClientRect();(r.top<120||r.bottom>innerHeight-80)&&n.scrollIntoView({block:`center`,behavior:j()?`auto`:`smooth`}),n.setAttribute(`data-wren-acting`,``),n.querySelector(`:scope > .wren-mini`)?.remove();let i=document.createElement(`span`);i.className=`wren-mini`,i.setAttribute(`aria-hidden`,`true`),n.append(i),o(Ce,i),setTimeout(()=>{n.removeAttribute(`data-wren-acting`),i.remove()},1600)}issues(){let e=this.m,t=this.wiz.selection;return!e||!t?{}:be(e,t,this.device,this.currentEstimate())}finishBlockers(){let e=this.m,t=this.wiz.selection;if(!e||!t)return[];let n=[],r=this.currentEstimate();r&&!r.fits&&n.push(D.reviewOver(F(r))),I(e,t,this.device)&&n.push(D.blockedToken);let i=Object.keys(this.issues()).filter(e=>e!==`review`);return i.length&&n.push(D.blockedInvalid(i.map(e=>O[e].toLowerCase()))),this.plan?.state===`ready`&&!this.plan.data.ok&&!n.length&&n.push(...this.plan.data.errors),n}async finish(){let e=this.wiz.selection;if(!e||this.applying||this.finishBlockers().length)return;this.applying=!0,this.applyError=null,this.previousProfile=this.device?.active_profile??null,this.querySelector(`qse-wren`)?.sleepForInstall();let t=e;try{let e=this.device?.active_profile,n=await p.apply({selection:t,...this.removeAfter&&e&&e!==t.profile?{remove_after:[e]}:{}});this.jobInitial=n.job,this.confirmOpen=!1,this.dispatch({type:`job`,id:n.job.id})}catch(e){this.applyError=this.applyErrorText(e)}finally{this.applying=!1}}applyErrorText(e){let t=T(e);if(t.status===401)return null;let n=e instanceof E?e.body:void 0;return n?.plan?(this.plan={state:`ready`,data:n.plan},this.confirmOpen=!1,n.plan.errors.join(` `)||t.message):e instanceof E&&e.type===`job_running`?D.blockedBusy:e instanceof E&&e.type===`busy`?D.blockedRequests(n?.error?.running??1):`${D.installError} ${t.message}`}render(){let e=this.wiz.step,t=this.m,n=this.path.length>0;return r`<div class="view view-setup" data-step=${n?e:`models`}>
      <div class="setup">
        <div class="setup-main">
          ${this.renderBanners()}
          ${this.unauthorized?r`<qse-setup-signin></qse-setup-signin>`:n?this.renderModelArea():this.renderTiles()}
          <footer class="setup-foot">
            <button class="btn btn-ghost btn-sm" aria-expanded=${this.resetOpen} aria-controls="setup-reset" @click=${()=>this.resetOpen=!this.resetOpen}>${i(`trash`)} ${D.resetLink}</button>
          </footer>
          ${this.resetOpen?r`<section class="panel setup-reset" id="setup-reset"><header class="panel-head"><h2 class="panel-title">${D.resetHeading}</h2></header><qse-reset inSetup></qse-reset></section>`:l}
        </div>
        <qse-wren class="setup-wren" .host=${this.host} .step=${n?e:`models`} .chat=${this.wiz.chat} .finished=${e===`install`&&!!this.wiz.job}></qse-wren>
      </div>
      ${t?this.renderConfirm(t):l}
    </div>`}renderBanners(){let e=this.storageProblem,t=[];return(e===`unavailable`||this.storageFailed)&&t.push(D.storageOff),e===`corrupt`&&t.push(D.storageCorrupt),e===`version`&&t.push(D.storageOld),this.wiz.notice&&this.wiz.step!==`components`&&this.wiz.step!==`parallelism`&&t.push(this.wiz.notice),r`${t.map(e=>r`<div class="banner setup-banner" role="status" data-setup-notice>${i(`info`)} <span>${e}</span></div>`)}`}renderTiles(){let e=this.list,t=this.device,n=this.wiz.model&&this.wiz.selection?this.wiz:null;return r`<section class="setup-tiles">
      <header class="setup-hero">
        <h1 class="setup-h1">${O.models}</h1>
        <p class="setup-lead">${A.models}</p>
        ${t?.mem_total_gb==null?l:r`<p class="setup-device muted num" data-device-line>${D.deviceLine(t.mem_total_gb,t.mem_available_gb)}</p>`}
      </header>
      ${e.state===`loading`?c(4):l}
      ${e.state===`error`?u(e.endpoint,`${D.modelsError} ${e.message}`,()=>void this.loadList()):l}
      ${e.state===`empty`?s(D.modelsEmpty):l}
      ${e.state===`ready`?r`<div class="mtiles">
              ${e.data.models.map(e=>{let t=n?.model===e.id,i=t?`#/setup/${e.id}/${n.step===`models`?`overview`:n.step}`:`#/setup/${e.id}/overview`;return r`<a class="mtile ${e.recommended?`is-rec`:``}" href=${i} data-model=${e.id}>
                  <span class="mtile-top"><span class="mtile-family">${e.family}</span>${e.recommended?r`<span class="sbadge sbadge-rec">${D.tileRecommended}</span>`:l}</span>
                  <span class="mtile-name">${e.name}</span>
                  <span class="mtile-params">${D.tileParams(e.params)}</span>
                  <span class="mtile-pitch">${e.pitch}</span>
                  ${e.community_count?r`<span class="mtile-comm">${D.community.tileCount(e.community_count)}</span>`:l}
                  ${e.best_speed?r`<span class="mtile-speed" title=${D.tileSpeedTitle}><span class="mtile-speed-n num">${e.best_speed.tok_s.toFixed(1)}</span><span class="mtile-speed-u">tok/s</span><span class="mtile-speed-note">${e.best_speed.task}, ${e.best_speed.profile}</span></span>`:l}
                  <span class="mtile-go"><span class="btn btn-primary">${t?`${D.tileOpen}: ${k[n.step]}`:D.tileOpen}</span></span>
                </a>`})}
            </div>
            ${e.data.errors.length?r`<div class="setup-load-errors" role="status"><p>${D.modelLoadErrors(e.data.errors.length)}</p><ul>${e.data.errors.map(e=>r`<li><code>${e.path}</code> <span class="muted">${e.message}</span></li>`)}</ul></div>`:l}`:l}
      <section class="guide" aria-labelledby="guide-h">
        <h2 class="guide-h" id="guide-h">${D.guideSteps}</h2>
        <ol class="guide-list">${x.slice(1).map(e=>r`<li><span class="guide-t">${k[e]}</span><span class="guide-d">${ye[e]}</span></li>`)}</ol>
      </section>
    </section>`}renderModelArea(){let e=this.model;if(this.wiz.step===`install`&&this.wiz.job&&e?.state!==`ready`)return r`<section class="setup-step" data-step="install" aria-labelledby="setup-step-title">
        <header class="step-head"><h1 class="step-title" id="setup-step-title" tabindex="-1">${O.install}</h1></header>
        ${this.stepInstall()}
      </section>`;if(!e||e.state===`loading`)return c(8);if(e.state===`error`)return u(e.endpoint,e.message,()=>void this.loadModel(this.path[0]));if(e.state!==`ready`||!this.wiz.selection)return c(8);let t=e.data,n=this.wiz.selection,i=this.currentEstimate(),a=this.wiz.step,o=i&&!i.fits?R(t,n,i):null;return r`
      <div class="setup-sticky">
        <qse-memory-bar .model=${t} .estimate=${i} .available=${this.device?.mem_available_gb??null} .local=${this.estLocal} .overText=${o}></qse-memory-bar>
        ${this.estLocal?r`<p class="setup-local" role="status" data-local-estimate>${D.memLocal}</p>`:l}
      </div>
      ${this.renderRail(t)}
      <section class="setup-step" data-step=${a} aria-labelledby="setup-step-title">
        <header class="step-head">
          <p class="step-kicker muted num">${D.stepOf(S(a)+1,x.length)}</p>
          <h1 class="step-title" id="setup-step-title" tabindex="-1">${O[a]}</h1>
          <p class="step-intro">${A[a]}</p>
        </header>
        ${this.renderStep(t,n,i)}
      </section>
      ${this.renderNav()}
    `}renderRail(e){let t=this.wiz.step,n=this.issues(),a=S(t);return r`<nav class="rail-steps" aria-label="Setup steps">
      <div class="rail-phone" aria-hidden="true">
        <span class="num">${D.stepOf(a+1,x.length)}</span><span class="rail-phone-label">${O[t]}</span>
        <span class="rail-phone-bar"><span style="width:${(a+1)/x.length*100}%"></span></span>
      </div>
      <ol class="steps">
        ${x.map((o,s)=>{let c=this.wiz.visited.includes(o),u=c&&(n[o]?.length??0)>0,d=o===t,f=c&&!u&&s<a,p=o!==`install`||!!this.wiz.job,m=o===`models`?`#/setup`:`#/setup/${e.id}/${o}`;return r`<li class="steps-item ${d?`is-current`:``} ${f?`is-done`:``} ${u?`is-bad`:``} ${c?`is-visited`:``}" data-rail=${o}>
            ${p?r`<a href=${m} aria-current=${d?`step`:l} title=${u?n[o].join(` `):O[o]}>
                  <span class="steps-node num" aria-hidden="true">${u?`!`:f?i(`check`):s+1}</span>
                  <span class="steps-label">${k[o]}</span>
                  ${u?r`<span class="sr-only">(needs a fix)</span>`:l}
                </a>`:r`<span class="steps-link is-locked" aria-disabled="true"><span class="steps-node num" aria-hidden="true">${s+1}</span><span class="steps-label">${k[o]}</span></span>`}
          </li>`})}
      </ol>
    </nav>`}renderNav(){let e=this.wiz.step;if(e===`install`)return l;let t=S(e),n=this.nextRefused?this.nextBlocker():null;return r`<nav class="setup-nav ${n?`has-why`:``}" aria-label="Step navigation">
      ${n?r`<p class="setup-nav-why" id="setup-nav-why" role="alert" data-next-refused>${i(`info`)} <span>${D.nextRefused(n)}</span></p>`:l}
      <button class="btn setup-prev" @click=${()=>t<=1?this.dispatch({type:`tiles`}):this.dispatch({type:`previous`})}>${D.previous}</button>
      <span class="setup-nav-mid muted small num">${D.stepOf(t+1,x.length)}</span>
      ${e===`review`?r`<button class="btn btn-primary setup-finish" ?disabled=${this.finishBlockers().length>0||this.plan?.state!==`ready`} @click=${()=>this.confirmOpen=!0}>${D.finish}</button>`:r`<button class="btn btn-primary setup-next" aria-describedby=${n?`setup-nav-why`:l} @click=${()=>this.next()}><span>${D.next}<span class="setup-next-to">: ${k[x[t+1]]}</span></span></button>`}
    </nav>`}nextBlocker(){let e=this.m,t=this.wiz.selection;if(this.wiz.step!==`parallelism`||!e||!t)return null;let n=xe(Oe(e,t),this.slotsParse(e));if(n)return n;for(let n of[`window`,`side_window`]){if(n===`side_window`&&t.slots<2)continue;let r=this.winParse(e,t,n);if(r&&!r.ok)return`${n===`window`?D.window:D.sideWindow}: ${r.why}`}return null}winParse(e,t,n){let r=this.winDraft[n];return r==null?null:Ve(r,Ie(e,t,n))}slotsParse(e){return this.slotsDraft==null?null:Ee(this.slotsDraft,P(e))}next(){if(this.nextBlocker()){this.nextRefused=!0,this.querySelector(`[data-field^="parallelism."].is-bad input`)?.focus();return}this.nextRefused=!1,this.dispatch({type:`next`})}renderStep(e,t,n){switch(this.wiz.step){case`overview`:return this.stepOverview(e,t);case`weights`:return this.stepWeights(e,t);case`components`:return this.stepComponents(e,t);case`parallelism`:return this.stepParallel(e,t,n);case`methods`:return this.stepMethods(e,t);case`review`:return this.stepReview(e,t,n);case`install`:return this.stepInstall();default:return r``}}stepOverview(e,t){let n=e.profiles.find(e=>e.id===t.profile),a=He(e,t.profile);return r`<div class="ov">
      <div class="ov-facts">
        <p class="ov-arch">${e.architecture}</p>
        <dl class="ov-dl">
          <div><dt>${D.colWeights}</dt><dd>${e.profiles.length} sizes, ${e.profiles.map(e=>e.label.replace(/ bpw$/,``)).join(`, `)}</dd></div>
          <div><dt>${D.window}</dt><dd class="num">${n?.max_window===null?D.windowNone(n.label):`${a.toLocaleString(`en-US`)} ${D.tokensUnit}`}</dd></div>
          <div><dt>${D.license}</dt><dd>${e.license}</dd></div>
        </dl>
        <ul class="ov-links">${e.links.map(e=>r`<li><a href=${e.url} target="_blank" rel="noopener noreferrer">${e.label} ${i(`external`)}</a></li>`)}</ul>
      </div>
      <section class="panel ov-table" aria-label="Measured speed">
        <qse-measure-table .model=${e} .selected=${t.profile}></qse-measure-table>
      </section>
    </div>`}stepWeights(e,t){let n=this.device,a=I(e,t,n),o=e.measurements.rows,s=Math.max(...o.filter(e=>e.comparable).map(e=>e.cells.greedy_512?.code?.tok_s??0));return r`
      ${a?r`<div class="banner banner-offline" role="alert" data-token-blocked>${i(`warn`)} ${D.privateNoToken}</div>`:l}
      ${!a&&n?.hf_token_present&&e.profiles.some(e=>e.hf_private&&!n.profiles_on_disk?.[e.id])?r`<p class="wnote muted small">${i(`info`)} ${D.privateRepo}</p>`:l}
      <div class="wcards" role="radiogroup" aria-label=${O.weights}>
        ${fe(e).map(e=>{let a=t.profile===e.id,c=o.find(t=>t.profile===e.id),u=c?.cells.greedy_512,d=n?.profiles_on_disk?.[e.id],f=n?.active_profile===e.id,p=e.quality.top1_pct;return r`<label class="wcard ${a?`is-selected`:``} ${e.recommended?`is-rec`:``}" data-profile-card=${e.id} data-field=${a?`weights.profile`:l}>
            <input class="sr-only" type="radio" name="profile" .checked=${a} value=${e.id} @change=${()=>this.user(`weights`,`profile`,{type:`profile`,id:e.id})} />
            <span class="wcard-head">
              <span class="wcard-radio" aria-hidden="true"></span>
              <span class="wcard-name">${e.label}</span>
              ${f?r`<span class="sbadge sbadge-now">${D.activeNow}</span>`:l}
            </span>
            ${e.badges.length?r`<span class="wcard-badges">${e.badges.map(e=>r`<span class="sbadge ${e===`recommended`?`sbadge-rec`:``}">${e===`recommended`?D.recommended:e}</span>`)}</span>`:l}
            <span class="wcard-meters">
              <span class="wmeter" title=${e.quality.source}>
                <span class="wmeter-k">${D.weightsTop1}</span>
                <span class="wmeter-v num">${p==null?`–`:`${p.toFixed(1)} %`}</span>
                <span class="wmeter-bar q"><span style="width:${p==null?0:Math.max(4,(p-80)/20*100)}%"></span></span>
              </span>
              <span class="wmeter">
                <span class="wmeter-k">${D.weightsSpeed}</span>
                <span class="wmeter-v num">${u?.prose&&u?.code?r`${u.prose.tok_s.toFixed(1)} / ${u.code.tok_s.toFixed(1)}`:r`<span class="muted">${c&&!c.comparable?D.notComparable:D.notMeasured}</span>`}</span>
                <span class="wmeter-bar s"><span style="width:${u?.code&&c?.comparable?u.code.tok_s/s*100:0}%"></span></span>
              </span>
            </span>
            <span class="wcard-facts num">
              <span><span class="wcard-k">${D.weightsKl}</span> ${e.quality.kl_vs_bf16==null?`–`:e.quality.kl_vs_bf16.toFixed(4)}</span>
              <span><span class="wcard-k">${D.weightsMemory}</span> ${e.weights_gib.toFixed(1)} GiB</span>
              <span><span class="wcard-k">${D.weightsDisk}</span> ${d?D.onDisk:D.toDownload(e.disk_gb)}</span>
            </span>
            ${this.renderSource(e)}
            ${e.note?r`<span class="wcard-note">${e.note}</span>`:l}
            ${e.hf_private&&!d&&n?.hf_token_present===!1?r`<span class="wcard-private is-blocked" data-private>${i(`warn`)} ${D.privateNoTokenShort}</span>`:l}
          </label>`})}
      </div>
      ${ce(e).length||e.community?r`<qse-community .model=${e} .sync=${e.community_sync??null} .selected=${t.profile} .device=${n} .syncing=${this.communitySyncing} .onUse=${e=>this.user(`weights`,`profile`,{type:`profile`,id:e})} .onRefresh=${()=>void this.syncCommunity()}></qse-community>`:l}
      <qse-installed-weights @weights-removed=${()=>void this.loadDevice()}></qse-installed-weights>`}renderSource(e){let t=m(e);return r`<span class="wcard-src" data-weights-src=${t.kind}>
      <span class="sbadge wsrc-kind wsrc-${t.kind}">${t.label}</span>
      <span class="wsrc-by">${D.madeBy} ${t.byUrl?r`<a href=${t.byUrl} target="_blank" rel="noopener noreferrer" title=${D.community.github(t.by.slice(1))}>${t.by}</a>`:t.by}</span>
      ${t.repoUrl?r`<a class="wsrc-repo" href=${t.repoUrl} target="_blank" rel="noopener noreferrer" title=${D.community.onHf(t.repo)}>${t.repo} ${i(`external`)}</a>`:r`<span class="wsrc-repo">${t.repo}</span>`}
    </span>`}stepComponents(e,t){let n=v(e,t.components);n.vision&&me(e,t.profile)&&(n.vision={disabled:!0,reason:D.community.visionOff,required:!1});let a=[...new Set(e.components.map(e=>e.group))],o=N(e,t.profile);return r`
      <div class="step-tools">
        <button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`components`})}>${i(`refresh`)} ${D.resetToRecommended}</button>
      </div>
      <p class="comp-notice" role="status" aria-live="polite" data-cascade>${this.wiz.notice??``}</p>
      ${a.map(a=>r`<section class="comp-group" aria-labelledby=${`cg-${a}`}>
          <h2 class="comp-group-title" id=${`cg-${a}`}>${D.groups[a]??a}</h2>
          <ul class="comp-list">
            ${e.components.filter(e=>e.group===a).map(e=>{let a=h(e,t.components),s=n[e.id],c=e.id===`drafters`&&o.drafters_gib!=null?o.drafters_gib:e.memory_gib;return r`<li>
                  <label class="comp ${a?`is-on`:``} ${s.disabled?`is-disabled`:``} ${s.required?`is-required`:``}" data-field=${`components.${e.id}`} data-component=${e.id}>
                    <input type="checkbox" class="comp-input" .checked=${a} ?disabled=${s.disabled} aria-describedby=${`cw-${e.id}${s.reason?` cr-${e.id}`:``}`} @change=${t=>this.user(`components`,e.id,{type:`component`,id:e.id,on:t.target.checked})} />
                    <span class="comp-box" aria-hidden="true">${i(`check`)}</span>
                    <span class="comp-main">
                      <span class="comp-label">${e.label}${e.default&&!e.required?r` <span class="comp-rec">${D.recommendedValue}</span>`:l}</span>
                      <span class="comp-what" id=${`cw-${e.id}`}>${e.what}</span>
                      ${s.reason?r`<span class="comp-reason" id=${`cr-${e.id}`} data-reason>${s.reason}</span>`:l}
                    </span>
                    <span class="comp-cost num">
                      <span class="comp-mem">${D.memoryCost(c)}</span>
                      ${e.memory_max_gib==null?l:r`<span class="muted">${D.memoryUpTo(e.memory_max_gib)}</span>`}
                      ${e.disk_gb>0?r`<span class="muted">${D.diskCost(e.disk_gb)}</span>`:l}
                    </span>
                  </label>
                </li>`})}
          </ul>
        </section>`)}`}stepParallel(e,t,n){let a=e.parallelism,o=t.side_window??t.window,s=n?.context_tokens_fit??0,c=Math.floor(s/t.slots),u=(e,t)=>this.user(`parallelism`,e,{type:`parallel`,key:e,value:t});return r`<div class="spar">
      <div class="spar-fields">
        ${this.slotsField(e,t)}
        ${this.windowField(e,t,`window`)}
        ${t.slots>1?this.windowField(e,t,`side_window`):l}
        <div class="pfield" data-field="parallelism.kv_pool">
          <div class="pfield-k"><span class="pfield-label" id="pf-pool">${D.kvPool}</span><span class="pfield-what">${D.kvPoolWhat}</span></div>
          <label class="switch"><input type="checkbox" role="switch" .checked=${t.kv_pool} aria-labelledby="pf-pool" @change=${e=>u(`kv_pool`,e.target.checked)} /><span>${t.kv_pool?`on`:`off`}</span></label>
        </div>
        <div class="pfield" data-field="parallelism.floor_gb">
          <div class="pfield-k"><span class="pfield-label" id="pf-floor">${D.floor}</span><span class="pfield-what">${D.floorWhat}</span></div>
          <div class="floor-ctl">
            <input type="range" min=${a.floor_gb.min} max=${a.floor_gb.max} step="1" .value=${String(t.floor_gb)} aria-labelledby="pf-floor" @input=${e=>u(`floor_gb`,Number(e.target.value))} />
            <output class="num">${t.floor_gb} GiB</output>
          </div>
        </div>
        <div class="step-tools"><button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`parallelism`})}>${i(`refresh`)} ${D.resetToRecommended}</button></div>
        ${this.wiz.notice?r`<p class="comp-notice" role="status">${this.wiz.notice}</p>`:l}
      </div>
      <aside class="spar-readout panel" aria-live="polite" data-fit-tokens=${s}>
        <p class="spar-big num">${s.toLocaleString(`en-US`)}</p>
        <p class="spar-big-u">${D.fitsUnit}</p>
        <div class="spar-slots" aria-hidden="true">
          ${[{n:`1`,w:t.window,on:!0},{n:t.slots>2?`2–${t.slots}`:`2`,w:o,on:t.slots>1}].map(e=>r`<div class="spar-slot ${e.on?``:`is-off`}">
              <span class="spar-slot-n num">${e.n}</span>
              <span class="spar-slot-track"><span class="spar-slot-win" style="width:${e.w/a.window.max*100}%"></span><span class="spar-slot-fit" style="width:${Math.min(e.w,c)/a.window.max*100}%"></span></span>
              <span class="spar-slot-k num">${Re(e.w)}</span>
            </div>`)}
        </div>
        <p class="small">${D.fitsPerSlot(c,t.slots)}</p>
        ${n?r`<p class="small muted">${D.worstCase(n.kv_worst_gb,n.fits_worst??!1)}</p>`:l}
      </aside>
    </div>`}slotsField(e,t){let n=P(e),i=Oe(e,t),a=this.slotsParse(e),o=a&&!a.ok?a.why:null,s=o??(i.over?D.slotsOver(i.n,i.overGb,i.fitMax):null),c=e=>this.user(`parallelism`,`slots`,{type:`parallel`,key:`slots`,value:e}),u=e=>{let r=a?.ok?a.n:t.slots,i=Se(r,e,n);i===t.slots?this.slotsDraft=null:c(i)},d=e=>{let r=e.target.value,i=Ee(r,n);i.ok?(this.slotsDraft=null,i.n!==t.slots&&c(i.n)):this.slotsDraft=r},f=e=>{(e.key===`ArrowUp`||e.key===`ArrowDown`)&&(e.preventDefault(),u(e.key===`ArrowUp`?1:-1))},p=this.slotsDraft??String(t.slots),m=t.slots>=n.max&&!o;return r`<div class="pfield pfield-slots ${s?`is-bad`:``}" data-field="parallelism.slots">
      <div class="pfield-k"><label class="pfield-label" id="pf-slots" for="pf-slots-in">${D.slots}</label><span class="pfield-what">${D.slotsWhat}</span></div>
      <div class="stepper ${s?`is-bad`:``}" role="group" aria-labelledby="pf-slots">
        <button class="stepper-btn" type="button" aria-label=${D.slotsFewer} aria-controls="pf-slots-in" ?disabled=${t.slots<=n.min&&!o} @click=${()=>u(-1)}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9" /></svg>
        </button>
        <input
          id="pf-slots-in"
          class="stepper-in num"
          type="text"
          inputmode="numeric"
          enterkeyhint="done"
          autocomplete="off"
          spellcheck="false"
          role="spinbutton"
          aria-valuemin=${n.min}
          aria-valuemax=${n.max}
          aria-valuenow=${a&&!a.ok?l:t.slots}
          aria-valuetext=${a&&!a.ok?l:`${t.slots} ${t.slots===1?`chat`:`chats`}`}
          aria-invalid=${s?`true`:`false`}
          aria-describedby="pf-slots-say"
          maxlength="3"
          .value=${p}
          @input=${d}
          @keydown=${f}
          @focus=${e=>e.target.select()}
        />
        <button class="stepper-btn" type="button" aria-label=${D.slotsMore} aria-controls="pf-slots-in" ?disabled=${m} title=${m?D.slotsAtMax(n.max):l} @click=${()=>u(1)}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9M8 3.5v9" /></svg>
        </button>
      </div>
      <div class="slots-say" id="pf-slots-say">
        ${s?r`<p class="slots-err" role="alert" data-slots-error>${s}</p>`:l}
        <p class="slots-cost" data-slots-cost>${D.slotsCost(i.cost.now,i.cost.full,i.cost.window)}</p>
        ${i.untested&&!o?r`<p class="slots-note" data-slots-untested>${D.slotsUntested(n.measured)}</p>`:l}
      </div>
    </div>`}windowField(e,t,n){let i=Ie(e,t,n),a=n===`window`?t.window:Math.min(t.side_window??t.window,t.window),o=n===`window`?D.window:D.sideWindow,s=n===`window`?`pf-win`:`pf-side`,c=this.winParse(e,t,n),u=c&&!c.ok?c.why:null,d=c?.ok?c.n:a,f=e=>this.user(`parallelism`,n,{type:`parallel`,key:n,value:e}),p=e=>{this.winDraft={...this.winDraft,[n]:null},e!==a&&f(e)},m=e=>{let t=e.target.value;this.winDraft={...this.winDraft,[n]:t};let r=Ve(t,i);r.ok&&r.n!==a&&f(r.n)},h=e=>{let t=e.key;if(t===`ArrowUp`||t===`ArrowDown`)p(z(d,t===`ArrowUp`?1:-1,i));else if(t===`PageUp`||t===`PageDown`){let e=t===`PageUp`?1:-1;p(z(z(d,e,i),e,i))}else if(t===`Home`)p(i.min);else if(t===`End`)p(i.max);else if(t===`Enter`)e.target.blur();else return;e.preventDefault()},g=()=>{c?.ok&&(this.winDraft={...this.winDraft,[n]:null})},_=this.winDraft[n]??a.toLocaleString(`en-US`),ee=d<=i.min&&!u,v=d>=i.max&&!u;return r`<div class="pfield pfield-win ${u?`is-bad`:``}" data-field=${`parallelism.${n}`}>
      <div class="pfield-k"><label class="pfield-label" id=${s} for=${`${s}-in`}>${o}</label><span class="pfield-what">${n===`window`?D.windowWhat:D.sideWindowWhat(t.slots)}</span></div>
      <div class="win-ctl">
        <div class="stepper stepper-wide ${u?`is-bad`:``}" role="group" aria-labelledby=${s}>
          <button class="stepper-btn" type="button" aria-label=${D.windowShorter(o)} aria-controls=${`${s}-in`} ?disabled=${ee} @click=${()=>p(z(d,-1,i))}>
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9" /></svg>
          </button>
          <input
            id=${`${s}-in`}
            class="stepper-in num"
            type="text"
            inputmode="numeric"
            enterkeyhint="done"
            autocomplete="off"
            spellcheck="false"
            role="spinbutton"
            aria-valuemin=${i.min}
            aria-valuemax=${i.max}
            aria-valuenow=${u?l:d}
            aria-valuetext=${u?l:`${d.toLocaleString(`en-US`)} ${D.tokensUnit}`}
            aria-invalid=${u?`true`:`false`}
            aria-describedby=${`${s}-say`}
            maxlength="12"
            .value=${_}
            @input=${m}
            @keydown=${h}
            @blur=${g}
            @focus=${e=>e.target.select()}
          />
          <button class="stepper-btn" type="button" aria-label=${D.windowLonger(o)} aria-controls=${`${s}-in`} ?disabled=${v} title=${v?i.maxWhy:l} @click=${()=>p(z(d,1,i))}>
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9M8 3.5v9" /></svg>
          </button>
        </div>
        <span class="stepper-hint num" aria-hidden="true" data-win-short>${u?``:Re(d)}</span>
      </div>
      <div class="slots-say" id=${`${s}-say`}>
        ${u?r`<p class="slots-err" role="alert" data-win-error>${u}</p>`:l}
        ${!u&&v&&i.max<e.parallelism[n].max?r`<p class="slots-note" data-win-max>${i.maxWhy}</p>`:l}
      </div>
    </div>`}fieldControl(e,t,n){let i=n=>this.user(`methods`,t.id,{type:`field`,group:e,id:t.id,value:n}),a=`mf-${t.id}`;if(t.type===`enum`){let e=t.options??[];return e.length<=3?r`<div class="seg seg-wrap" role="radiogroup" aria-labelledby=${`${a}-l`}>${e.map(e=>r`<button class="seg-btn ${n===e.value?`is-on`:``}" role="radio" aria-checked=${n===e.value} @click=${()=>i(e.value)}>${e.label}</button>`)}</div>`:r`<select id=${a} class="sel" @change=${e=>i(e.target.value)}>${e.map(e=>r`<option value=${e.value} ?selected=${n===e.value}>${e.label}</option>`)}</select>`}return t.type===`bool`?r`<label class="switch"><input id=${a} type="checkbox" role="switch" .checked=${n===!0} @change=${e=>i(e.target.checked)} /><span>${n?`on`:`off`}</span></label>`:t.type===`int`||t.type===`number`?r`<input id=${a} class="num-input num" type="number" inputmode=${t.type===`int`?`numeric`:`decimal`} min=${t.min??l} max=${t.max??l} step=${t.type===`int`?1:.05} .value=${String(n)} @change=${e=>{let t=e.target.value;i(t===``?NaN:Number(t))}} />`:r`<input id=${a} class="text-input" type="text" .value=${String(n)} spellcheck="false" autocapitalize="off" @change=${e=>i(e.target.value)} />`}fieldRow(e,t,n){let i=n[e][t.id]??t.recommended,a=C(t,i),o=t.options?.find(e=>e.value===t.recommended)?.label??String(t.recommended);return r`<div class="mfield ${a?`is-bad`:``}" data-field=${`methods.${t.id}`}>
      <div class="pfield-k">
        <label class="pfield-label" id=${`mf-${t.id}-l`} for=${`mf-${t.id}`}>${t.label}</label>
        <span class="pfield-what">${t.what}</span>
        <span class="mfield-rec muted small">${D.recommendedValue}: <span class="num">${o}</span>${i===t.recommended?l:r` <button class="linkish" @click=${()=>this.user(`methods`,t.id,{type:`field`,group:e,id:t.id,value:t.recommended})}>${D.resetToRecommended.toLowerCase()}</button>`}</span>
      </div>
      <div class="mfield-ctl">${this.fieldControl(e,t,i)}${a?r`<p class="mfield-err" role="alert">${D.invalidValue(a)}</p>`:l}</div>
    </div>`}stepMethods(e,t){let n=e.advanced.some(e=>C(e,t.advanced[e.id]??e.recommended));return r`<div class="methods">
      ${e.methods.map(e=>this.fieldRow(`methods`,e,t))}
      <details class="adv" ?open=${n}>
        <summary><span class="adv-title">${D.advanced}</span> <span class="muted small">${D.advancedNote}</span></summary>
        ${e.advanced.map(e=>this.fieldRow(`advanced`,e,t))}
      </details>
      <div class="step-tools"><button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`methods`})}>${i(`refresh`)} ${D.resetToRecommended}</button></div>
    </div>`}stepReview(e,t,n){let a=this.plan,o=this.wiz.step,s=n&&!n.fits?Te(e,t):[],u=this.finishBlockers(),d=this.device,f=N(e,t.profile),p=a?.state===`ready`?a.data:null;return r`<div class="review">
      <section class="rv-card rv-memory ${n&&!n.fits?`is-over`:`is-ok`}" data-review-memory>
        <h2 class="rv-h">${D.reviewMemory}</h2>
        ${n?r`<p class="rv-say">${n.fits?D.reviewFits(n.total_gb,n.budget_gb??n.device_gb-n.floor_gb):R(e,t,n)}</p>`:l}
        ${s.length?r`<h3 class="rv-sub">${D.howToFix}</h3>
              <ul class="rv-fixes">${s.map(t=>r`<li><span>${t.text}</span><span class="muted small num">−${t.saves.toFixed(1)} GiB</span><a class="btn btn-sm" href=${`#/setup/${e.id}/${t.step}`}>${D.fixGo}: ${k[t.step]}</a></li>`)}</ul>`:l}
      </section>

      <section class="rv-card">
        <h2 class="rv-h">${D.reviewWeights}</h2>
        <p>${d?.active_profile===f.id?D.reviewWeightsSame(f.label):D.reviewWeightsSwitch(d?.active_profile?e.profiles.find(e=>e.id===d.active_profile)?.label??d.active_profile:null,f.label)}</p>
        ${p&&!p.profile.on_disk?r`<p class="muted">${D.reviewDownload(p.profile.download_gb)}</p>`:l}
        ${p?.downloads?.length?r`<ul class="rv-downloads" data-review-downloads>${p.downloads.map(e=>r`<li data-download=${e.id}><span>${e.label}</span> <span class="muted small">${e.for}</span> <span class="num">${e.gb.toFixed(2)} GB</span></li>`)}</ul>`:l}
        ${p?r`<p class="muted small">${p.restart?D.reviewRestart:D.reviewNoRestart}</p>`:l}
        ${d?.active_profile&&d.active_profile!==f.id&&d.profiles_on_disk?.[d.active_profile]?r`<label class="rv-remove-after" data-remove-after><input type="checkbox" .checked=${this.removeAfter} @change=${e=>this.removeAfter=e.target.checked} />
              <span>${D.removeAfter(e.profiles.find(e=>e.id===d.active_profile)?.label??d.active_profile)}</span></label>`:l}
      </section>

      <section class="rv-card rv-changes">
        <h2 class="rv-h">${D.reviewChanges}</h2>
        ${a==null||a.state===`loading`?c(4):l}
        ${a?.state===`error`?r`<p class="rv-warn" role="alert">${i(`warn`)} ${D.planError} <span class="muted">${a.message}</span> <button class="btn btn-sm" @click=${()=>void this.loadPlan()}>${D.retry}</button></p>`:l}
        ${p?p.changes.length?r`<div class="table-scroll"><table class="table rv-table">
                <thead><tr><th scope="col">${D.reviewKey}</th><th scope="col">${D.reviewNow}</th><th scope="col">${D.reviewAfter}</th><th scope="col">${D.reviewFrom}</th></tr></thead>
                <tbody>${p.changes.map(e=>r`<tr data-env=${e.key}><th scope="row"><code>${e.key}</code></th><td class="rv-old">${e.current??r`<span class="muted">${D.reviewUnset}</span>`}</td><td class="rv-new">${e.removed?r`<span class="rv-removed" data-removed>${Fe(e,D.reviewUnset).to}</span>`:e.next??r`<span class="muted">${D.reviewUnset}</span>`}</td><td class="muted small">${e.from}</td></tr>`)}</tbody>
              </table></div>`:r`<p class="muted">${D.reviewNoChanges}</p>`:l}
        ${p&&(p.errors.length||p.warnings.length)?r`<ul class="rv-notes">
              ${p.errors.map((t,n)=>{let a=p.error_codes?.[n],s=we(a);return r`<li class="is-err" data-error-code=${a??l}>${i(`warn`)} <span>${t}</span>${s&&s!==o?r` <a class="btn btn-sm" href=${`#/setup/${e.id}/${s}`}>${D.planFix}: ${k[s]}</a>`:l}</li>`})}
              ${p.warnings.map(e=>r`<li>${i(`info`)} ${e}</li>`)}
            </ul>`:l}
      </section>

      ${p?.commands.length?r`<section class="rv-card">
            <h2 class="rv-h">${D.reviewCommands}</h2>
            <p class="muted small">${D.reviewCommandsNote}</p>
            <pre class="rv-cmds">${p.commands.join(`
`)}</pre>
          </section>`:l}

      ${d&&!d.password_set?this.securityCard():d?.password_set?r`<p class="rv-secure muted">${i(`check`)} ${D.securitySet}</p>`:l}

      ${u.length?r`<ul class="rv-blockers" role="status" data-finish-blocked>${u.map(e=>r`<li>${i(`warn`)} ${e}</li>`)}</ul>`:l}
      ${this.applyError?r`<p class="rv-warn" role="alert">${i(`warn`)} ${this.applyError}</p>`:l}
    </div>`}get sec(){return this.wiz.selection?.advanced.admin_password===`generate`?`generate`:`keep`}loginOff(){let e=this.m?.advanced.find(e=>e.id===`dashboard_login`);return!!e&&(this.wiz.selection?.advanced.dashboard_login??e.recommended)!==`on`}securityCard(){let e=(e,t,n)=>r`<label class="sec-opt ${this.sec===e?`is-on`:``}">
      <input type="radio" name="sec" .checked=${this.sec===e} @change=${()=>this.user(`methods`,`admin_password`,{type:`field`,group:`advanced`,id:`admin_password`,value:e})} />
      <span><span class="sec-opt-label">${t}</span>${n?r`<span class="muted small">${n}</span>`:l}</span>
    </label>`;return r`<section class="rv-card rv-security" data-security-warning role="note">
      <h2 class="rv-h">${i(`warn`)} ${D.securityTitle}</h2>
      <p>${D.securityBody}</p>
      <div class="sec-opts" role="radiogroup" aria-label=${D.securityChoice}>
        ${e(`keep`,D.securityKeep)} ${e(`generate`,D.securityGenerate,D.securityGenerateNote)}
      </div>
      ${this.sec===`generate`&&this.loginOff()?r`<p class="sec-login" data-login-off>${i(`info`)} <span>${D.securityLoginOff}</span>
            <button class="btn btn-sm" @click=${()=>this.user(`methods`,`dashboard_login`,{type:`field`,group:`advanced`,id:`dashboard_login`,value:`on`})}>${D.securityLoginOn}</button></p>`:l}
    </section>`}renderConfirm(e){let t=this.plan?.state!==`ready`||this.plan.data.restart;return r`<dialog
      class="setup-confirm"
      aria-labelledby="sc-t"
      aria-describedby="sc-b"
      @cancel=${e=>this.applying&&e.preventDefault()}
      @close=${()=>this.confirmOpen=!1}
    >
      ${this.confirmOpen?r`<h2 id="sc-t">${D.confirmTitle}</h2>
            <p id="sc-b">${D.confirmBody(t)}</p>
            <p class="muted small">${N(e,this.wiz.selection.profile).label}</p>
            ${this.applyError?r`<p class="rv-warn" role="alert">${this.applyError}</p>`:l}
            <div class="setup-confirm-foot">
              <button class="btn" ?disabled=${this.applying} @click=${()=>this.confirmOpen=!1}>${D.cancel}</button>
              <button class="btn btn-primary setup-confirm-go" autofocus ?disabled=${this.applying} @click=${()=>void this.finish()}>${this.applying?D.installStarting:D.confirmGo}</button>
            </div>`:l}
    </dialog>`}updated(){let e=this.querySelector(`dialog.setup-confirm`);e&&(this.confirmOpen&&!e.open?e.showModal():!this.confirmOpen&&e.open&&e.close())}stepInstall(){let e=this.wiz.job;return e?r`<qse-install-progress .jobId=${e} .initial=${this.jobInitial} .previousProfile=${this.previousProfile} .onBack=${()=>this.dispatch({type:`job`,id:null})} @job-retry=${e=>{this.jobInitial=e.detail,this.dispatch({type:`job`,id:e.detail.id})}}></qse-install-progress>`:r`<p class="muted">${D.installStarting}</p>`}};e([t({attribute:!1})],$.prototype,`path`,void 0),e([d()],$.prototype,`wiz`,void 0),e([d()],$.prototype,`storageProblem`,void 0),e([d()],$.prototype,`storageFailed`,void 0),e([d()],$.prototype,`list`,void 0),e([d()],$.prototype,`model`,void 0),e([d()],$.prototype,`device`,void 0),e([d()],$.prototype,`serverEst`,void 0),e([d()],$.prototype,`estLocal`,void 0),e([d()],$.prototype,`plan`,void 0),e([d()],$.prototype,`confirmOpen`,void 0),e([d()],$.prototype,`applying`,void 0),e([d()],$.prototype,`removeAfter`,void 0),e([d()],$.prototype,`applyError`,void 0),e([d()],$.prototype,`jobInitial`,void 0),e([d()],$.prototype,`previousProfile`,void 0),e([d()],$.prototype,`resetOpen`,void 0),e([d()],$.prototype,`unauthorized`,void 0),e([d()],$.prototype,`winNarrow`,void 0),e([d()],$.prototype,`slotsDraft`,void 0),e([d()],$.prototype,`winDraft`,void 0),e([d()],$.prototype,`nextRefused`,void 0),e([d()],$.prototype,`communitySyncing`,void 0),$=e([n(`qse-setup`)],$);export{$ as QseSetup};