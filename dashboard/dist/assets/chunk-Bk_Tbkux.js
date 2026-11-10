import{c as e,d as t,f as n,g as r,i,l as a,m as o,n as s,o as c,p as l,r as u,u as d,v as f}from"./chunk-5VjiyoiJ.js";import{C as p,D as m,E as h,N as g,P as _,S as ee,T as v,a as te,b as ne,c as re,d as ie,f as ae,g as oe,h as se,i as y,j as b,l as ce,m as le,n as x,p as ue,r as S,s as C,t as de,u as fe,v as pe,w as me,y as he}from"./chunk-TX5oitx2.js";import{a as ge,n as _e}from"./chunk-BXmDkkzu.js";import{i as w,t as T}from"./chunk-C-rLp5B5.js";import{Et as ve,_ as E,f as D,g as ye,h as O,i as be,m as xe,p as k,wt as Se}from"./app-aJ2XSYW4.js";import{_ as Ce,a as we,c as Te,d as Ee,f as De,g as Oe,h as A,i as ke,l as Ae,m as je,n as Me,o as Ne,p as j,r as Pe,s as Fe,t as Ie,u as Le,v as Re}from"./chunk-CkRGlP_L.js";import{n as ze,t as Be}from"./chunk-dBfjuPm3.js";import{a as Ve,c as M,d as He,f as Ue,h as We,i as Ge,l as Ke,m as qe,n as Je,o as Ye,p as Xe,r as N,s as Ze,t as Qe,u as $e}from"./chunk-CAih7RBx.js";var P=[`1`,`2`,`4`,`8`],F=[`2048`,`8192`,`32768`];function et(e){let t=e.parallel?.rows??[],n=t=>{let n=e.profiles.findIndex(e=>e.id===t.profile);return n<0?e.profiles.length:n};return[...t].sort((e,t)=>n(e)-n(t)||e.profile.localeCompare(t.profile))}function tt(e,t){return e?.parallel?.rows.find(e=>e.profile===t)??null}function I(e){let t=e?.parallelism?.slots?.default??null;return t!=null&&t>0?D.parallelSlots(t):null}function nt(e,t){let n=e.profiles.find(e=>e.id===t.profile);return n?n.label:(e.community??[]).find(e=>e.profile===t.profile||`community/${e.id}`===t.profile)?.name??t.profile}function L(e,t){let n=e.levels[t];return n&&n.at_once===Number(t)?n.wall_tok_s:null}function rt(e){let t=e.setup,n=t.sampling;return D.parallelSetup(t.lanes,t.window,t.max_tokens,`T ${n.temperature}, top-k ${n.top_k}, top-p ${n.top_p}`,t.suffix_store,t.response_cache,e.rounds,t.prompts.set)}var R=e=>e==null?`–`:e.toFixed(1),it=e=>e==null?`–`:e.toFixed(2),z=class extends a{constructor(...e){super(...e),this.model=null,this.selected=null}render(){let e=this.model,t=e?et(e):[];return!e||!t.length?l:r`<div class="mt pt">
      <h3 class="pt-title">${D.parallelTitle}</h3>
      <div class="table-scroll mt-scroll">
        <table class="table mt-table pt-table">
          <thead>
            <tr>
              <th scope="col" class="mt-name" rowspan="2">${D.colWeights}</th>
              <th scope="colgroup" colspan="2">${D.parallelDecode}<span class="mt-unit">${D.tokPerS}</span></th>
              <th scope="colgroup" colspan=${F.length}>${D.parallelPrefill}<span class="mt-unit">${D.seconds}</span></th>
              <th scope="colgroup" colspan=${P.length}>${D.parallelAtOnce}<span class="mt-unit">${D.tokPerS}</span></th>
            </tr>
            <tr>
              <th scope="col" class="num-col">${D.colCode}</th>
              <th scope="col" class="num-col">${D.colProse}</th>
              ${F.map(e=>r`<th scope="col" class="num-col">${Number(e)/1024}k</th>`)}
              ${P.map(e=>r`<th scope="col" class="num-col">${e}</th>`)}
            </tr>
          </thead>
          <tbody>${t.map(t=>this.row(e,t))}</tbody>
        </table>
      </div>
      <div class="mt-foot muted small">
        ${I(e)?r`<p class="pt-slots" data-slots-line>${I(e)}</p>`:l}${D.parallelFoot}
        <details class="mt-method"><summary>${D.howMeasured}</summary><p>${e.parallel?.method??``}</p>
          <ul class="pt-setups">${t.map(t=>r`<li data-setup=${t.profile}><strong>${nt(e,t)}</strong>: ${D.parallelBuild(t.build,t.commit,t.date)} ${rt(t)}${t.note?r` ${t.note}`:l}</li>`)}</ul>
        </details>
      </div>
    </div>`}row(e,t){let n=t.profile===this.selected;return r`<tr class="mt-row ${n?`is-selected`:``}" data-parallel=${t.profile} aria-current=${n?`true`:l}>
      <th scope="row" class="mt-name"><span class="mt-label">${nt(e,t)}</span>${n?r`<span class="mt-mine">${D.selectedRow}</span>`:l}</th>
      <td class="num-col num">${R(t.decode?.code)}</td>
      <td class="num-col num">${R(t.decode?.prose)}</td>
      ${F.map(e=>{let n=t.prefill?.[e];return n?r`<td class="num-col num" title=${D.parallelPrefillTitle(n.prompt_tokens,n.tok_s)}>${it(n.ttft_s)}</td>`:r`<td class="num-col mt-na">–</td>`})}
      ${P.map(e=>{let n=t.levels[e],i=L(t,e);return i==null||!n?r`<td class="num-col mt-na">–</td>`:r`<td class="num-col num" title=${D.parallelLevelTitle(Number(e),n.decode_tok_s,n.ttft_s)}>${R(i)}</td>`})}
    </tr>`}};e([t({attribute:!1})],z.prototype,`model`,void 0),e([t({attribute:!1})],z.prototype,`selected`,void 0),z=e([n(`qse-parallel-table`)],z);function B(e,t){let n=tt(e,t);if(!n)return null;let r=L(n,`1`),i=[`8`,`4`,`2`].find(e=>L(n,e)!=null);return i?D.parallelCard(r,Number(i),L(n,i)):r==null?null:D.parallelCard(r,1,r)}var V=D.community,at={text:f`<path d="M4 5.5h12M4 9.5h12M4 13.5h7" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>`,mic:f`<rect x="7.25" y="3" width="5.5" height="9" rx="2.75" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M4.5 10a5.5 5.5 0 0 0 11 0M10 15.5v2M7.5 17.5h5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,video:f`<rect x="2.5" y="5" width="10.5" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m13 9 4.5-2.5v7L13 11z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>`,drafters:f`<circle cx="4.5" cy="10" r="1.8" fill="currentColor"/><circle cx="10" cy="10" r="1.8" fill="currentColor" opacity=".7"/><circle cx="15.5" cy="10" r="1.8" fill="currentColor" opacity=".45"/><path d="M6.3 10h1.9M11.8 10h1.9" stroke="currentColor" stroke-width="1.4"/>`,x:f`<path d="M4 3.5h3.6l8.4 13H12.4z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M15.5 3.5 4.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,huggingface:f`<circle cx="10" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="7.6" cy="9" r="1" fill="currentColor"/><circle cx="12.4" cy="9" r="1" fill="currentColor"/><path d="M7 12.2c.8 1.2 1.8 1.8 3 1.8s2.2-.6 3-1.8" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M4.2 7.2 3 6.2M15.8 7.2 17 6.2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,bluesky:f`<path d="M10 8.6C9 6.8 6.8 4.2 4.6 3.7 3.3 3.5 2.8 4.3 2.8 5.4c0 1.6.6 4.3 1.9 5 .9.4 2 .2 2.7 0-2 .5-2.9 1.4-2.1 2.9.9 1.8 3.2 2.5 4.7-.4 1.5 2.9 3.8 2.2 4.7.4.8-1.5-.1-2.4-2.1-2.9.7.2 1.8.4 2.7 0 1.3-.7 1.9-3.4 1.9-5 0-1.1-.5-1.9-1.8-1.7C13.2 4.2 11 6.8 10 8.6Z" fill="currentColor"/>`,mastodon:f`<path d="M4 15.5c-.9-1-1.5-2.6-1.5-5V8.3C2.5 5 4.6 3.5 10 3.5s7.5 1.5 7.5 4.8v1.5c0 2.6-1.4 3.9-4.2 4.1-1.9.2-3.5 0-4.3-.3v.9c0 1.4 2.3 1.5 5 1" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M6.3 11.8V8.4c0-1.1.6-1.7 1.6-1.7s1.6.6 1.6 1.7v1.9M13.7 11.8V8.4c0-1.1-.6-1.7-1.6-1.7s-1.6.6-1.6 1.7v1.9" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`,website:f`<circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M3 10h14M10 3c2.2 2.1 3.3 4.4 3.3 7s-1.1 4.9-3.3 7c-2.2-2.1-3.3-4.4-3.3-7S7.8 5.1 10 3Z" fill="none" stroke="currentColor" stroke-width="1.4"/>`},H=(e,t=16)=>r`<svg class="icon" viewBox="0 0 20 20" aria-hidden="true" width=${t} height=${t}>${at[e]}</svg>`,ot=e=>e===`text`?H(`text`):e===`image`?i(`image`):H(e===`audio`?`mic`:`video`),st=e=>e.toUpperCase(),U=class extends a{constructor(...e){super(...e),this.model=null,this.sync=null,this.selected=null,this.device=null,this.syncing=!1,this.onUse=()=>{},this.onRefresh=()=>{},this.query=``,this.sort=`newest`,this.droppedOpen=!1,this.noAvatar=new Set}render(){let e=this.model;if(!e)return l;let t=ue(e),n=this.sync;if(!t.length&&!n)return l;let a=p(ee(t,this.query),this.sort),o=ce,s=oe(n?.add_url)??o;return r`<section class="cw" aria-labelledby="cw-h" data-community data-syncing=${this.syncing?``:l}>
      <header class="cw-head">
        <div class="cw-title">
          <h2 class="cw-h" id="cw-h">${V.heading}</h2>
          ${this.renderSource()}
        </div>
        <p class="cw-intro">${V.intro} <a href=${o} target="_blank" rel="noopener noreferrer">${V.addYoursHow} ${i(`external`)}</a></p>
        ${t.length?r`<div class="cw-tools">
              <label class="cw-search">
                ${i(`search`)}
                <input type="search" placeholder=${V.search} aria-label=${V.searchLabel} .value=${this.query} autocomplete="off" autocapitalize="off" spellcheck="false" @input=${e=>this.query=e.target.value} />
              </label>
              <label class="cw-sort"><span class="muted small">Sort</span>
                <select class="sel" aria-label=${V.sortLabel} @change=${e=>this.setSort(e.target.value)}>${fe.map(e=>r`<option value=${e} ?selected=${this.sort===e}>${ie[e]}</option>`)}</select>
              </label>
              <span class="cw-count num muted small" data-count>${se(a.length,t.length)}</span>
              <a class="btn btn-sm cw-add" href=${s} target="_blank" rel="noopener noreferrer">${i(`plus`)} ${V.addYours}</a>
            </div>`:l}
      </header>
      ${n?.dropped.length?this.renderDropped(n.dropped):l}
      ${t.length?a.length?r`<div class="cw-grid" role="list">${a.map(e=>this.renderCard(e))}</div>`:r`<div class="cw-empty" data-community-nomatch role="status"><p class="cw-empty-t">${V.noMatch(this.query.trim())}</p><p><button class="btn btn-sm" @click=${()=>this.query=``}>${i(`x`)} ${V.clearSearch}</button></p></div>`:r`<div class="cw-empty" data-community-empty>
            <p class="cw-empty-t">${V.empty}</p>
            <p class="muted">${V.emptyBody}</p>
            <p class="cw-empty-acts"><a class="btn btn-primary btn-sm" href=${s} target="_blank" rel="noopener noreferrer">${i(`plus`)} ${V.addYours}</a> <a class="btn btn-sm" href=${o} target="_blank" rel="noopener noreferrer">${V.addYoursHow} ${i(`external`)}</a></p>
          </div>`}
    </section>`}setSort(e){pe(e)&&(this.sort=e)}renderSource(){let e=this.sync,t=e?.source??`bundled`,n=e=>e&&Date.now()-Date.parse(e)<1e4?`just now`:_e(e),a=this.syncing?V.syncing:t===`github`?V.fromGithub(n(e?.fetched_at)):t===`cache`?V.fromCache(n(e?.fetched_at)):V.bundled;return r`<div class="cw-src" data-source=${this.syncing?`syncing`:t}>
      <span class="cw-src-dot" aria-hidden="true"></span>
      <span class="cw-src-text small" role="status">${a}</span>
      <button class="btn btn-ghost btn-sm cw-refresh" ?disabled=${this.syncing} aria-busy=${this.syncing} @click=${()=>this.onRefresh()}>${i(`refresh`)} ${V.refresh}</button>
      ${e?.error&&!this.syncing?r`<span class="cw-src-err small" role="alert">${i(`warn`)} ${V.syncFailed(e.error.replace(/\.?$/,`.`))}</span>`:l}
      ${e?.note&&!e.error&&!this.syncing?r`<span class="cw-src-note small muted">${e.note}</span>`:l}
    </div>`}renderDropped(e){return r`<div class="cw-dropped small" data-dropped>
      <p>${i(`info`)} <span>${V.dropped(e.length)}</span> <button class="linkish" aria-expanded=${this.droppedOpen} @click=${()=>this.droppedOpen=!this.droppedOpen}>${this.droppedOpen?V.droppedHide:V.droppedDetails}</button></p>
      ${this.droppedOpen?r`<ul>${e.map(e=>r`<li><code>${e.path}</code> <span class="muted">${e.reason}</span></li>`)}</ul>`:l}
    </div>`}renderCard(e){let t=this.selected===e.id,n=ae(e.profile,this.device),a=e.status===`broken`,o=e.author,s=o.avatar&&!this.noAvatar.has(o.login);return r`<article class="ccard ${t?`is-selected`:``} ${a?`is-broken`:``}" role="listitem" data-community-card=${e.slug} data-field=${t?`weights.profile`:l} aria-labelledby=${`cc-${e.slug}`}>
      <header class="ccard-maker">
        <span class="ccard-avatar" aria-hidden="true">
          ${s?r`<img src=${o.avatar} alt="" width="40" height="40" loading="lazy" decoding="async" referrerpolicy="no-referrer" @error=${()=>this.avatarFailed(o.login)} />`:r`<span class="ccard-initials">${o.initials}</span>`}
        </span>
        <span class="ccard-who">
          <span class="ccard-handles">
            ${o.url?r`<a class="ccard-login" href=${o.url} target="_blank" rel="noopener noreferrer" title=${V.github(o.login)}>@${o.login}</a>`:r`<span class="ccard-login">@${o.login}</span>`}
            ${o.social.length?r`<span class="ccard-social">${o.social.map(e=>r`<a class="ccard-net tip-host" href=${e.href} target="_blank" rel="noopener noreferrer" aria-label=${e.label} data-tip=${e.label} data-net=${e.kind}>${H(e.kind)}</a>`)}</span>`:l}
          </span>
          ${o.name&&o.name.toLowerCase()!==o.login.toLowerCase()?r`<span class="ccard-author">${o.name}</span>`:l}
        </span>
        <span class="ccard-badges">
          ${n===`running`?r`<span class="sbadge sbadge-now">${D.activeNow}</span>`:l}
          ${this.statusBadge(e)}
        </span>
      </header>
      <h3 class="ccard-name" id=${`cc-${e.slug}`}>${e.name}</h3>
      <p class="ccard-base">${e.finetuneOf?r`${V.finetuneOf} ${e.finetuneUrl?r`<a href=${e.finetuneUrl} target="_blank" rel="noopener noreferrer">${e.finetuneOf}</a>`:r`<span>${e.finetuneOf}</span>`}`:V.stock}</p>
      ${e.description?r`<p class="ccard-desc">${e.description}</p>`:l}
      <dl class="ccard-readout num">
        <div><dt>format</dt><dd>${st(e.format)}</dd></div>
        <div><dt>${V.bpw}</dt><dd>${e.bpw==null?`–`:e.bpw.toFixed(2)}</dd></div>
        <div><dt>${V.download}</dt><dd>${n===`download`?`${e.sizeGb.toFixed(1)} GB`:D.onDisk}</dd></div>
      </dl>
      <ul class="ccard-modes" aria-label="Modalities">
        ${e.modalities.map(e=>r`<li class="ccard-mode tip-host ${e.runs?`is-on`:`is-off`}" tabindex="0" data-tip=${e.why} aria-label=${e.why} data-modality=${e.id}>${ot(e.id)}<span>${e.label}</span></li>`)}
      </ul>
      <p class="ccard-drafters">${H(`drafters`)} <span>${e.drafters}</span></p>
      ${e.verified&&e.verified.speeds.length?r`<div class="ccard-speed" data-speed>
            ${e.verified.speeds.map(e=>r`<span class="ccard-sp"><span class="ccard-sp-n num">${e.tok_s.toFixed(1)}</span><span class="ccard-sp-k">${e.task}</span></span>`)}
            <span class="ccard-sp-u"><span class="num">tok/s</span><span class="ccard-sp-note">${V.speedNote(e.verified.device,e.verified.build,ge(e.verified.date))}</span></span>
          </div>`:r`<p class="ccard-speed-none muted small">${V.notMeasured}</p>`}
      ${B(this.model,e.id)?r`<p class="ccard-par muted small num" data-parallel-line>${D.weightsAtOnce}: ${B(this.model,e.id)}</p>`:l}
      ${I(this.model)?r`<p class="ccard-par muted small num" data-slots-line>${D.weightsSlots}: ${I(this.model)}</p>`:l}
      <footer class="ccard-foot">
        <span class="ccard-links">
          ${e.licenseUrl?r`<a class="sbadge ccard-license" href=${e.licenseUrl} target="_blank" rel="noopener noreferrer" title=${`${V.license}: ${e.license}`}>${e.license}</a>`:r`<span class="sbadge ccard-license" title=${`${V.license}: ${e.license}`}>${e.license}</span>`}
          ${e.hfUrl?r`<a class="ccard-hf" href=${e.hfUrl} target="_blank" rel="noopener noreferrer" title=${V.onHf(e.hfRepo)} data-hf-repo>${H(`huggingface`)} <span class="ccard-repo">${e.hfRepo}</span> ${i(`external`)}</a>`:r`<span class="ccard-hf" data-hf-repo>${H(`huggingface`)} <span class="ccard-repo">${e.hfRepo}</span></span>`}
        </span>
        <button class="btn ${t?``:`btn-primary`} ccard-use" ?disabled=${a} aria-pressed=${t} title=${a?V.brokenWhy:l} @click=${()=>!t&&this.onUse(e.id)}>${t?r`${i(`check`)} ${V.selected}`:V.use}</button>
      </footer>
    </article>`}statusBadge(e){return e.status===`verified`&&e.verified?r`<span class="sbadge sbadge-ok tip-host" tabindex="0" data-tip=${V.verifiedWhy(e.verified.device,e.verified.build,ge(e.verified.date))}>${i(`check`)} ${V.verified}</span>`:e.status===`broken`?r`<span class="sbadge sbadge-err tip-host" tabindex="0" data-tip=${V.brokenWhy}>${i(`warn`)} ${V.broken}</span>`:r`<span class="sbadge tip-host" tabindex="0" data-tip=${V.submittedWhy}>${V.submitted}</span>`}avatarFailed(e){let t=new Set(this.noAvatar);t.add(e),this.noAvatar=t}};e([t({attribute:!1})],U.prototype,`model`,void 0),e([t({attribute:!1})],U.prototype,`sync`,void 0),e([t({attribute:!1})],U.prototype,`selected`,void 0),e([t({attribute:!1})],U.prototype,`device`,void 0),e([t({type:Boolean})],U.prototype,`syncing`,void 0),e([t({attribute:!1})],U.prototype,`onUse`,void 0),e([t({attribute:!1})],U.prototype,`onRefresh`,void 0),e([d()],U.prototype,`query`,void 0),e([d()],U.prototype,`sort`,void 0),e([d()],U.prototype,`droppedOpen`,void 0),e([d()],U.prototype,`noAvatar`,void 0),U=e([n(`qse-community`)],U);function ct(e){return{title:D.removeTitle(e.label,e.gb),body:e.shared_with.length?D.removeShared(e.shared_with):D.removeBody(e.model_name),again:`bash ops/use-profile.sh ${e.profile} --fetch-only`}}function lt(e){if(!e)return[];let t=new Set,n=[];for(let r of[...e.weights].sort((e,t)=>Number(t.active)-Number(e.active)||t.gb-e.gb))t.has(r.removes)||(t.add(r.removes),n.push(r));return n}var W=class extends a{constructor(...e){super(...e),this.data=null,this.error=null,this.asking=null,this.busy=!1,this.done=null}connectedCallback(){super.connectedCallback(),this.load()}async load(){try{this.data=await b.weights(),this.error=null}catch(e){this.error=w(e).message}}async removeRow(e){this.busy=!0;try{let t=await b.remove([e.profile]);this.done=D.removed(e.label,t.freed_gb,t.log??`run/weights-removed.txt`),this.asking=null,this.dispatchEvent(new CustomEvent(`weights-removed`,{detail:t,bubbles:!0})),await this.load()}catch(e){this.error=w(e).message}finally{this.busy=!1}}render(){let e=lt(this.data);return!this.data&&!this.error?l:r`<section class="iw" aria-labelledby="iw-h" data-installed-weights>
      <header class="iw-head">
        <h2 class="iw-title" id="iw-h">${D.installedTitle}</h2>
        ${this.data?.free_gb==null?l:r`<span class="muted small num">${D.diskFree(this.data.free_gb)}</span>`}
      </header>
      ${this.error?r`<p class="rv-warn" role="alert">${i(`warn`)} ${this.error}</p>`:l}
      ${this.done?r`<p class="iw-done" role="status" data-removed>${i(`check`)} ${this.done}</p>`:l}
      ${e.length===0&&this.data?r`<p class="muted small">${D.installedNone}</p>`:l}
      <ul class="iw-list">
        ${e.map(e=>{let t=ct(e);return r`<li class="iw-row ${e.active?`is-active`:``}" data-installed=${e.profile}>
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
    </section>`}};e([d()],W.prototype,`data`,void 0),e([d()],W.prototype,`error`,void 0),e([d()],W.prototype,`asking`,void 0),e([d()],W.prototype,`busy`,void 0),e([d()],W.prototype,`done`,void 0),W=e([n(`qse-installed-weights`)],W);function ut(e){return e.group===`long`?e.mode===`greedy`?`long_greedy`:`long_sampled`:`${e.mode}_${e.len}`}function dt(e){return e===`everyday`?[`profile`,`prose`,`chat`,`code`,`ttft8`,`kl`,`lossless`]:[`profile`,`long32k`,`long64k`,`ttft32`,`kl`]}var ft={profile:D.colWeights,prose:D.colProse,chat:D.colChat,code:D.colCode,long32k:D.colLong32,long64k:D.colLong64,ttft8:D.colTtft8,ttft32:D.colTtft32,kl:D.colKl,lossless:D.colLossless};function G(e,t,n,r){let i=t.cells[ut(r)]??{};switch(n){case`profile`:{let n=e.profiles.find(e=>e.id===t.profile);return n?.bpw??n?.weights_gib??null}case`ttft8`:return t.ttft_s[8192]??null;case`ttft32`:return t.ttft_s[32768]??null;case`kl`:return e.profiles.find(e=>e.id===t.profile)?.quality.kl_vs_bf16??null;case`lossless`:return t.lossless?t.lossless.identical:null;default:return i[n]?.tok_s??null}}function pt(e,t){let n=e.measurements.rows,r=n.filter(e=>e.comparable),i=n.filter(e=>!e.comparable),a=n=>G(e,n,t.sort,t);return r.sort((e,n)=>{let r=a(e),i=a(n);return r==null&&i==null?0:r==null?1:i==null?-1:t.dir===`asc`?r-i:i-r}),{main:r,apart:i}}var K=[`prose`,`chat`,`code`,`long32k`,`long64k`],q=class extends a{constructor(...e){super(...e),this.model=null,this.selected=null,this.view={mode:`greedy`,len:`512`,group:`everyday`,sort:`code`,dir:`desc`}}set(e){let t={...this.view,...e};e.group&&!dt(e.group).includes(t.sort)&&(t.sort=e.group===`long`?`long32k`:`code`),this.view=t}sortBy(e){this.view.sort===e?this.set({dir:this.view.dir===`asc`?`desc`:`asc`}):this.set({sort:e,dir:K.includes(e)||e===`lossless`?`desc`:`asc`})}render(){let e=this.model;if(!e)return l;let t=this.view,n=dt(t.group),{main:i,apart:a}=pt(e,t),o={};for(let n of K)o[n]=Math.max(0,...i.map(r=>G(e,r,n,t)??0));let s=(e,t,n,i)=>r`<div class="seg" role="radiogroup" aria-label=${i}>${t.map(([t,i])=>r`<button class="seg-btn ${e===t?`is-on`:``}" role="radio" aria-checked=${e===t} data-val=${t} @click=${()=>n(t)}>${i}</button>`)}</div>`;return r`<div class="mt">
      <div class="mt-controls">
        ${s(t.mode,[[`greedy`,D.modeGreedy],[`sampled`,D.modeSampled]],e=>this.set({mode:e}),`Decoding`)}
        ${t.group===`everyday`?s(t.len,[[`512`,D.tokens512],[`128`,D.tokens128]],e=>this.set({len:e}),`Answer length`):l}
        ${s(t.group,[[`everyday`,D.groupEveryday],[`long`,D.groupLong]],e=>this.set({group:e}),`Tasks`)}
      </div>
      <div class="table-scroll mt-scroll">
        <table class="table mt-table" data-mode=${ut(t)}>
          <thead>
            <tr>
              ${n.map(e=>{let n=t.sort===e,i=K.includes(e)?D.tokPerS:e===`ttft8`||e===`ttft32`?D.seconds:``;return r`<th scope="col" class=${e===`profile`?`mt-name`:`num-col`} aria-sort=${n?t.dir===`asc`?`ascending`:`descending`:`none`}>
                  <button class="mt-sort ${n?`is-on`:``}" data-col=${e} @click=${()=>this.sortBy(e)} aria-label=${D.sortBy(ft[e])}>
                    <span>${ft[e]}</span>${i?r`<span class="mt-unit">${i}</span>`:l}<span class="mt-arrow" aria-hidden="true">${n?t.dir===`asc`?`▴`:`▾`:``}</span>
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
          </th>`;let u=G(e,t,n,i);if(u==null)return r`<td class="num-col mt-na">–</td>`;if(n===`kl`)return r`<td class="num-col num">${u.toFixed(4)}</td>`;if(n===`lossless`){let e=t.lossless;return r`<td class="num-col num" title=${D.losslessTitle(e.identical,e.near_tie)}>${e.identical} + ${e.near_tie}</td>`}if(n===`ttft8`||n===`ttft32`)return r`<td class="num-col num">${u.toFixed(1)}</td>`;let d=a[n]?u/a[n]*100:0;return r`<td class="num-col num mt-speed"><span class="mt-bar" style="--w:${d.toFixed(1)}" aria-hidden="true"></span><span class="mt-val">${u.toFixed(1)}</span></td>`})}
    </tr>`}};e([t({attribute:!1})],q.prototype,`model`,void 0),e([t({attribute:!1})],q.prototype,`selected`,void 0),e([d()],q.prototype,`view`,void 0),q=e([n(`qse-measure-table`)],q);var mt=[`model`,`helpers`,`caches`,`engine`,`system`,`chats`],J=class extends a{constructor(...e){super(...e),this.model=null,this.estimate=null,this.available=null,this.local=!1,this.overText=null,this.open=!1,this.focusPart=null}render(){let e=this.estimate,t=this.model;if(!e||!t)return r`<section class="membar is-empty" aria-label=${D.memTitle}><div class="membar-line"><span class="membar-title">${D.memTitle}</span></div><div class="membar-track"></div></section>`;let n=e.budget_gb??e.device_gb-e.floor_gb,i=je(e),a=e.parts.find(e=>e.kind===`kv`),o=Math.max(e.device_gb,e.total_gb),s=e=>`${(e/o*100).toFixed(3)}%`,c=e=>mt.indexOf(De(t,e)),u=[...e.parts].sort((e,t)=>c(e.id)-c(t.id)),d=0,f=u.map(e=>{let n={p:e,left:d,cat:De(t,e.id)};return d+=e.gb,n}),p=this.available==null?null:Math.max(0,e.device_gb-this.available),m=new Map;for(let e of f)m.set(e.cat,(m.get(e.cat)??0)+e.p.gb);let h=f.find(e=>e.p.id===this.focusPart),g=h?`${h.p.label}: ${h.p.gb.toFixed(2)} GiB${h.p.worst_gb!=null&&h.p.worst_gb!==h.p.gb?` (${D.memoryUpTo(h.p.worst_gb)})`:``}`:null,_=`${D.memTitle}: ${D.memFits(e.total_gb,n)}. ${i>0?this.overText??``:D.memLeft(n-e.total_gb)}`;return r`<section class="membar ${i>0?`is-over`:``} ${this.open?`is-open`:``}" aria-label=${D.memTitle} data-fits=${e.fits} data-total=${e.total_gb.toFixed(2)}>
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
          ${mt.filter(e=>m.has(e)).map(e=>r`<li class="cat-${e}"><i class="membar-sw" aria-hidden="true"></i>${D.memCat[e]} <span class="num">${(m.get(e)??0).toFixed(1)}</span></li>`)}
          <li class="cat-free"><i class="membar-sw" aria-hidden="true"></i>${D.memFloor} <span class="num">${e.floor_gb}</span></li>
        </ul>
        <p class="membar-detail" aria-live="polite">${g??(this.local?D.memLocal:e.notes[0]??`\xA0`)}</p>
      </div>
    </section>`}};e([t({attribute:!1})],J.prototype,`model`,void 0),e([t({attribute:!1})],J.prototype,`estimate`,void 0),e([t({attribute:!1})],J.prototype,`available`,void 0),e([t({type:Boolean})],J.prototype,`local`,void 0),e([t({attribute:!1})],J.prototype,`overText`,void 0),e([d()],J.prototype,`open`,void 0),e([d()],J.prototype,`focusPart`,void 0),J=e([n(`qse-memory-bar`)],J);var ht=`tandem.setup.simple`;function gt(){try{let e=localStorage.getItem(ht);return e===null||e===`1`}catch{return!0}}function _t(e){try{localStorage.setItem(ht,e?`1`:`0`)}catch{}}function vt(e,t){let n=[],r=Xe(e,t.profile);t.window!==r&&n.push({type:`parallel`,key:`window`,value:r});let i=Math.max(1,Math.min(4,e.parallelism.slots?.max??4)),a=(n,i)=>j(e,{...t,window:r,kv_pool:n,slots:i})?.fits??!1,o=i;for(;o>1&&!a(!1,o);)o--;let s=i;for(;s>1&&!a(!0,s);)s--;let c=s>o?s:o,l=s>o||t.kv_pool;return t.kv_pool!==l&&n.push({type:`parallel`,key:`kv_pool`,value:l}),t.slots!==c&&n.push({type:`parallel`,key:`slots`,value:c}),(e.components??[]).some(e=>e.id===`response_cache`)&&t.components?.response_cache!==!0&&n.push({type:`component`,id:`response_cache`,on:!0}),n}function yt(e,t){return r`<div class="setup-mode-switch" role="group" aria-label=${k.switchLabel}>
    <button class="btn btn-sm ${e?`btn-primary`:`btn-ghost`}" aria-pressed=${e} @click=${()=>t(!0)}>${k.simple}</button>
    <button class="btn btn-sm ${e?``:`btn-primary`} ${e?`btn-ghost`:``}" aria-pressed=${!e} @click=${()=>t(!1)}>${k.advanced}</button>
    <span class="muted setup-mode-note">${e?k.modeNoteSimple:k.modeNoteAdvanced}</span>
  </div>`}function Y(e,t,n){return r`<details class="setup-fold" data-fold=${t} ?open=${t===`review`}>
    <summary class="setup-fold-head">${e}</summary>
    <div class="setup-fold-body">${n}</div>
  </details>`}var bt=[`set_field`,`toggle`,`select_profile`,`go_to_step`,`next`,`previous`,`explain`,`read_device`],X=(e,t=!1)=>({ok:!1,error:e,...t?{stale:t}:{}});function Z(e,t){return`${e}.${t}`}function xt(e){if(e==null)return{};if(typeof e==`string`){if(!e.trim())return{};try{let t=JSON.parse(e);return t&&typeof t==`object`&&!Array.isArray(t)?t:null}catch{return null}}return typeof e==`object`&&!Array.isArray(e)?e:null}function St(e){if(typeof e==`number`&&Number.isInteger(e)&&e>=0&&e<x.length)return x[e];if(typeof e==`string`){if(/^\d+$/.test(e))return St(Number(e));if(y(e))return e}return null}function Ct(e,t){return(e===`int`||e===`number`)&&typeof t==`string`&&t.trim()!==``&&Number.isFinite(Number(t))?Number(t):e===`bool`&&(t===`true`||t===`false`)?t===`true`:t}function wt(e){return typeof e==`boolean`?e?`on`:`off`:typeof e==`number`?e.toLocaleString(`en-US`):e}function Tt(e,t){let n=String(e.name??``);if(!bt.includes(n))return/finish|apply|install|reset|delete|unload|load|confirm|submit/i.test(n)?X(D.tool.unknown(n)+` `+D.tool.userOnly(n)):X(D.tool.unknown(n));let r=xt(e.args);if(!r)return X(D.tool.badArgs(`expected a JSON object`));if(n===`read_device`)return{ok:!0,kind:`read_device`};let i=t.model,a=t.selection,o=(e,n)=>{let r=t.touchedAt?.get(Z(e,n));return r!=null&&t.turnStartedAt!=null&&r>t.turnStartedAt};switch(n){case`go_to_step`:{let e=r.n??r.step??r.index,n=St(e);return n?n===`install`?X(D.tool.installStep):n!==`models`&&!i?X(D.tool.noModel):{ok:!0,kind:`action`,actions:[{type:n===`models`?`tiles`:`go`,step:n}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:D.toolDone.step(E[n]),result:D.tool.ok}:X(D.tool.noStep(String(e)))}case`next`:{if(!i)return X(D.tool.noModel);let e=C(t.step);if(e>=C(`review`))return X(D.tool.atEnd);let n=x[e+1];return{ok:!0,kind:`action`,actions:[{type:`next`}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:D.toolDone.step(E[n]),result:D.tool.ok}}case`previous`:{let e=C(t.step);if(e<=0)return X(D.tool.atStart);let n=x[e-1];return{ok:!0,kind:`action`,actions:[{type:`previous`}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:D.toolDone.step(E[n]),result:D.tool.ok}}}if(!i||!a)return X(D.tool.noModel);if(n===`explain`){let e=String(r.field??r.id??``),t=i.components.find(t=>t.id===e);if(t)return{ok:!0,kind:`explain`,target:{step:`components`,field:t.id},result:`${t.label}: ${t.what}${t.note?` `+t.note:``}`};let n=te(i,e);if(n)return{ok:!0,kind:`explain`,target:{step:`methods`,field:n.field.id},result:`${n.field.label}: ${n.field.what}`};let a=i.profiles.find(t=>t.id===e);if(a)return{ok:!0,kind:`explain`,target:{step:`weights`,field:`profile`},result:`${a.label}: KL ${a.quality.kl_vs_bf16??`?`}, ${a.weights_gib} GiB in memory${a.note?`. `+a.note:``}`};let o={slots:D.slotsWhat,window:D.windowWhat,kv_pool:D.kvPoolWhat,floor_gb:D.floorWhat};return o[e]?{ok:!0,kind:`explain`,target:{step:`parallelism`,field:e},result:o[e]}:X(D.tool.noField(`any`,e))}if(n===`select_profile`){let e=String(r.id??r.profile??``),t=i.profiles.find(t=>t.id===e);return t?o(`weights`,`profile`)?X(D.tool.stale,!0):a.profile===e?{ok:!0,kind:`action`,actions:[],undo:[],target:{step:`weights`,field:`profile`},summary:D.toolDone.profile(t.label),result:D.tool.same}:{ok:!0,kind:`action`,actions:[{type:`profile`,id:e}],undo:[{type:`profile`,id:a.profile}],target:{step:`weights`,field:`profile`},summary:D.toolDone.profile(t.label),result:D.tool.ok}:X(D.tool.noProfile(e))}if(n===`toggle`){let e=String(r.component??r.id??``),t=Ct(`bool`,r.on);return typeof t==`boolean`?Et(i,a,e,t,o(`components`,e)):X(D.tool.badArgs(`on must be true or false`))}let s=St(r.step);if(!s)return X(D.tool.noStep(String(r.step)));let c=String(r.field??``),l=re(i).find(e=>e.id===s)?.fields.find(e=>e.field===c);if(!l)return X(D.tool.noField(s,c));if(de.has(c))return X(D.tool.secret(l.label));let u=Ct(l.type,r.value),d=S(l,u);if(d)return X(D.tool.badArgs(`${c}: ${d}`));if(s===`components`)return Et(i,a,c,u,o(s,c));if(o(s,c))return X(D.tool.stale,!0);let f={step:s,field:c},p=D.toolDone.field(l.label,wt(u));if(s===`weights`)return Tt({name:`select_profile`,args:{id:u}},t);if(s===`parallelism`){let e=c,t=a[e];return t===u?{ok:!0,kind:`action`,actions:[],undo:[],target:f,summary:p,result:D.tool.same}:{ok:!0,kind:`action`,actions:[{type:`parallel`,key:e,value:u}],undo:[{type:`parallel`,key:e,value:t}],target:f,summary:p,result:D.tool.ok}}if(s===`methods`){let e=te(i,c);if(!e)return X(D.tool.noField(s,c));let t=a[e.group][c]??e.field.recommended;return t===u?{ok:!0,kind:`action`,actions:[],undo:[],target:f,summary:p,result:D.tool.same}:{ok:!0,kind:`action`,actions:[{type:`field`,group:e.group,id:c,value:u}],undo:[{type:`field`,group:e.group,id:c,value:t}],target:f,summary:p,result:D.tool.ok}}return X(D.tool.noField(s,c))}function Et(e,t,n,r,i){let a=e.components.find(e=>e.id===n);if(!a)return X(D.tool.noComponent(n));if(a.required&&!r)return X(D.tool.required(a.label));let o=h(e,t.components)[n];if(r&&o.disabled&&!o.required)return X(o.reason??D.tool.noComponent(n));if(i)return X(D.tool.stale,!0);let s={step:`components`,field:n},c=r?D.toolDone.toggleOn(a.label):D.toolDone.toggleOff(a.label);return(a.required||(t.components[n]??a.default))===r?{ok:!0,kind:`action`,actions:[],undo:[],target:s,summary:c,result:D.tool.same}:{ok:!0,kind:`action`,actions:[{type:`component`,id:n,on:r}],undo:[{type:`components`,set:{...t.components}}],target:s,summary:c,result:D.tool.ok}}var Dt=new Set([`explain`,`read_device`]);function Ot(e){if(!e.tools||!e.calls.length)return null;let t=e.round+1>=4;return e.calls.some(e=>!e.ok&&!Dt.has(e.name))?e.retried?{tools:!1,retried:!0}:{tools:!t,retried:!0}:e.calls.every(e=>Dt.has(e.name)&&e.name!==`read_device`)?e.text.trim()?null:{tools:!1,retried:e.retried}:{tools:!t,retried:e.retried}}function kt(e){let t=new Map,n=0,r=r=>{let i=new Map,a=null;for(let t=n;t<r;t++){let n=e[t];if(n.role===`assistant`)for(let e of n.tool_calls??[])i.set(e.id,e.name);if(n.role!==`tool`||Dt.has(i.get(n.tool_call_id??``)??``))continue;let r={};try{r=JSON.parse(n.content)}catch{}r.ok===!1?a=r.stale||a===`stale`?`stale`:`unchanged`:r.ok===!0&&(a=null)}a&&r>n&&t.set(r-1,a)};return e.forEach((e,t)=>{e.role===`user`&&t!==0&&(r(t),n=t)}),r(e.length),t}function At(e,t){return t&&e.ok&&e.summary?e.summary:null}var jt=0;function Mt(e){return e.charAt(0).toUpperCase()+e.slice(1)}var Q=class extends a{constructor(...e){super(...e),this.host=null,this.step=`models`,this.chat=[],this.finished=!1,this.phase=`checking`,this.status=null,this.figure=`sleeping`,this.progress=null,this.streaming=!1,this.partial=``,this.composer=``,this.error=null,this.notice=null,this.sources=[],this.open=!1,this.toast=null,this.chips=new Map,this.ctrl=null,this.loadPoll=null,this.ping=null,this.loadStartedAt=0,this.lastUserText=``,this.opener=null,this.toastTimer=null,this.onKey=e=>{if(!this.open||!this.sheetMode())return;if(e.key===`Escape`){e.preventDefault(),this.close();return}if(e.key!==`Tab`)return;let t=this.querySelector(`.wren-panel`);if(!t)return;let n=[...t.querySelectorAll(`button:not([disabled]), textarea:not([disabled]), a[href], input:not([disabled])`)].filter(e=>e.offsetParent!==null);if(!n.length)return;let r=n[0],i=n[n.length-1];e.shiftKey&&document.activeElement===r?(e.preventDefault(),i.focus()):!e.shiftKey&&document.activeElement===i?(e.preventDefault(),r.focus()):t.contains(document.activeElement)||(e.preventDefault(),r.focus())}}connectedCallback(){super.connectedCallback(),this.checkStatus(),this.ping=setInterval(()=>{this.phase===`ready`&&document.visibilityState===`visible`&&this.checkStatus(!0)},3e4),document.addEventListener(`keydown`,this.onKey)}disconnectedCallback(){this.ctrl?.abort(),this.loadPoll&&clearTimeout(this.loadPoll),this.ping&&clearInterval(this.ping),this.toastTimer&&clearTimeout(this.toastTimer),document.removeEventListener(`keydown`,this.onKey),this.lockScroll(!1),super.disconnectedCallback()}updated(e){e.has(`finished`)&&this.finished&&this.phase!==`finished`&&this.sleepForInstall()}emit(e){this.figure=Re(this.figure,e)}async checkStatus(e=!1){if(this.finished){this.phase=`finished`,this.figure=`sleeping`;return}try{let e=await b.assistant.status();this.applyStatus(e)}catch(t){if(e)return;this.phase=`failed`,this.error=w(t).status===0?D.wrenOffline:w(t).message,this.emit(`error`)}}applyStatus(e){switch(this.status=e,e.state){case`loaded`:this.phase!==`ready`&&(this.phase=`ready`,this.progress=null,this.figure=`loading`,this.emit(`loaded`));break;case`loading`:this.phase=`loading`,this.progress=this.timeProgress(e.loading_s??null),this.emit(`load-progress`),this.scheduleLoadPoll();break;case`refused`:this.phase=`refused`,this.emit(`error`);break;case`failed`:this.phase=`failed`,this.error=e.reason??D.wrenFailed,this.emit(`error`);break;default:this.phase=e.model==null?`refused`:`asleep`,this.figure=`sleeping`}}timeProgress(e){let t=e??(Date.now()-this.loadStartedAt)/1e3;return Math.min(.95,1-Math.exp(-t/8))}scheduleLoadPoll(){this.loadPoll&&clearTimeout(this.loadPoll),this.loadPoll=setTimeout(()=>void this.checkStatus(!0),500)}async wake(){this.error=null,this.loadStartedAt=Date.now(),this.phase=`loading`,this.progress=0,this.emit(`load-start`);try{this.applyStatus(await b.assistant.load())}catch(e){let t=w(e),n=e instanceof T?e.body:void 0;n?.assistant&&(this.status=n.assistant),t.status===409?(this.phase=`refused`,n?.assistant?.reason&&(this.status={...this.status,reason:n.assistant.reason})):(this.phase=`failed`,this.error=t.status===0?D.wrenOffline:t.message),this.emit(`error`)}}async sleep(){this.stop();try{this.applyStatus(await b.assistant.unload())}catch{}this.phase=`asleep`,this.emit(`unload`)}async sleepForInstall(){this.ctrl?.abort(),this.phase=`finished`,this.emit(`unload`),this.close();try{await b.assistant.unload()}catch{}}stop(){this.ctrl&&this.ctrl.abort()}async send(e){let t=e.trim();if(!t||this.streaming||!this.host)return;this.lastUserText=t,this.composer=``,this.error=null,this.notice=null,this.toast=null;let n=[...this.chat,{role:`user`,content:t}];this.setChat(n),await this.turn(n)}async retry(){if(!this.lastUserText)return;this.error=null;let e=this.chat[this.chat.length-1]?.role===`user`?this.chat:[...this.chat,{role:`user`,content:this.lastUserText}];this.setChat(e),await this.turn(e)}setChat(e){this.chat=e,this.host?.saveChat(e),this.updateComplete.then(()=>this.scrollLog())}async turn(e){let t=this.host,n=Date.now(),r=e;this.streaming=!0,this.emit(`send`),this.ctrl=new AbortController;let i=this.ctrl.signal,a=!0,o=!1,s=!1;try{for(let e=1;;e++){let c=``,l=[],u=[],d=[],f=!0,p=null;if(this.partial=``,await g({messages:r,page:t.pageState(),...a?{}:{tools:!1}},e=>{e.type===`state`?this.emit(e.state===`thinking`?`state-thinking`:e.state===`loading`?`load-progress`:`first-token`):e.type===`text`?(c+=e.delta,this.partial=c,f?(f=!1,this.emit(`first-token`)):this.emit(`delta`),this.fig?.blip(),this.fabFig?.blip(),this.scrollLog()):e.type===`tool_call`?l.push({id:e.id||`call_${++jt}`,name:e.name,args:e.args}):e.type===`done`?(this.sources=e.sources??[],e.unchanged&&(s=!0)):e.type===`error`&&(p=e.message)},i),p&&/another tab/i.test(p)){this.notice=D.wrenBusy;break}if(p)throw new T(`/v1/setup/assistant/chat`,500,p,`stream`);for(let e of l){if(i.aborted)break;let r=await t.execute(e,n);r.ok&&(this.emit(`tool-call`),this.fig?.hop(),this.emit(`acted`)),u.push({role:`tool`,tool_call_id:e.id,content:JSON.stringify(r.ok?{ok:!0,result:r.result,...r.summary?{summary:r.summary}:{}}:{ok:!1,error:r.result,...r.stale?{stale:!0}:{}})}),this.chips=new Map(this.chips).set(e.id,{summary:r.summary,ok:r.ok,stale:r.stale,undo:r.undo,undone:!1}),d.push({name:e.name,ok:r.ok,changed:!!r.summary});let a=At(r,this.open&&this.sheetMode());a&&this.showToast(e.id,a,r.reveal??null)}let m={role:`assistant`,content:c,...l.length?{tool_calls:l}:{}};r=[...r,m,...u],this.partial=``,this.setChat(r);let h=Ot({round:e,tools:a,text:c,calls:d,retried:o});if(!h)break;({tools:a,retried:o}=h)}s&&!kt(r).has(r.length-1)&&(this.notice=D.wrenNotChanged),this.emit(`done`)}catch(e){if(e.name===`AbortError`)this.partial&&this.setChat([...r,{role:`assistant`,content:`${this.partial} ${D.wrenStopped}`}]),this.partial=``,this.emit(`stop`);else{let t=w(e);t.status===409?this.notice=D.wrenBusy:t.status===0?this.error=D.wrenOffline:t.status===503||/unload|asleep|not loaded/i.test(t.message)?(this.error=D.wrenWentToSleep,this.checkStatus(!0)):this.error=t.message||D.wrenError,this.partial&&this.setChat([...r,{role:`assistant`,content:this.partial}]),this.partial=``,this.emit(t.status===409?`done`:`error`)}}finally{this.streaming=!1,this.ctrl=null}}undoChip(e){let t=this.chips.get(e);t?.undo&&!t.undone&&(t.undo(),this.chips=new Map(this.chips).set(e,{...t,undone:!0}))}scrollLog(){let e=this.log;e&&(e.scrollTop=e.scrollHeight)}sheetMode(){return typeof matchMedia==`function`&&matchMedia(`(max-width: 1100px)`).matches}toggleOpen(){this.open?this.close():(this.opener=document.activeElement,this.open=!0,this.lockScroll(!0),this.updateComplete.then(()=>(this.querySelector(`.wren-panel .wren-input, .wren-panel .btn-primary, .wren-close`)??null)?.focus()))}close(){this.open&&(this.open=!1,this.lockScroll(!1),this.opener?.focus?.())}lockScroll(e){document.documentElement.classList.toggle(`wren-sheet-open`,e&&this.sheetMode())}showToast(e,t,n){this.toast={id:e,summary:t,reveal:n},this.toastTimer&&clearTimeout(this.toastTimer),this.toastTimer=setTimeout(()=>this.toast=null,12e3)}async showMe(){let e=this.toast?.reveal;this.toast=null,this.close(),await this.updateComplete,requestAnimationFrame(()=>e?.())}statusWord(){return this.phase===`finished`?D.wrenFinished:this.phase===`asleep`?D.wrenAsleep:this.phase===`refused`?D.wrenCannotLoad:this.phase===`failed`?D.wrenFailed:this.phase===`loading`?D.wrenWaking(this.progress):D.wrenStatus[this.figure]??``}render(){let e=this.figure;return r`
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
            <p class="muted small">${e?.reason?Mt(e.reason):t}</p>
            ${e?.reason?r`<p class="muted small">${t}</p>`:l}
            ${e?.model==null?l:r`<button class="btn wren-wake" @click=${()=>void this.wake()}>${i(`refresh`)} ${D.wrenRetry}</button>`}
          </div>`}case`failed`:return r`<div class="wren-body wren-center" data-wren-failed>
            ${this.hero()}
            <p class="wren-lead">${D.wrenFailed}</p>
            <p class="muted small">${this.error??``}</p>
            <button class="btn wren-wake" @click=${()=>void this.wake()}>${i(`refresh`)} ${D.wrenRetry}</button>
          </div>`;case`ready`:return this.renderChat()}}renderChat(){let e=this.chat,t=kt(e);return r`
      <div class="wren-body wren-log" aria-live="off">
        ${e.length===0?r`<p class="wren-msg is-assistant wren-hello">${D.wrenHello}</p>`:l}
        ${Se(e.map((e,t)=>({m:e,i:t})),({i:e})=>e,({m:n,i})=>r`${this.renderMessage(n,i===e.length-1)}${t.has(i)?r`<p class="wren-note" data-tool-rejected>${t.get(i)===`stale`?D.tool.stale:D.wrenNotChanged}</p>`:l}`)}
        ${this.partial?r`<div class="wren-msg is-assistant is-streaming">${ze(Be(this.partial))}</div>`:l}
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
    `}renderMessage(e,t){if(e.role===`user`)return r`<div class="wren-msg is-user">${e.content}</div>`;if(e.role===`assistant`)return e.content.trim()?r`<div class="wren-msg is-assistant">${ze(Be(e.content))}
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
    </form>`}};e([t({attribute:!1})],Q.prototype,`host`,void 0),e([t({attribute:!1})],Q.prototype,`step`,void 0),e([t({attribute:!1})],Q.prototype,`chat`,void 0),e([t({type:Boolean})],Q.prototype,`finished`,void 0),e([d()],Q.prototype,`phase`,void 0),e([d()],Q.prototype,`status`,void 0),e([d()],Q.prototype,`figure`,void 0),e([d()],Q.prototype,`progress`,void 0),e([d()],Q.prototype,`streaming`,void 0),e([d()],Q.prototype,`partial`,void 0),e([d()],Q.prototype,`composer`,void 0),e([d()],Q.prototype,`error`,void 0),e([d()],Q.prototype,`notice`,void 0),e([d()],Q.prototype,`sources`,void 0),e([d()],Q.prototype,`open`,void 0),e([d()],Q.prototype,`toast`,void 0),e([d()],Q.prototype,`chips`,void 0),e([_(`qse-wren-figure.wren-main`)],Q.prototype,`fig`,void 0),e([_(`qse-wren-figure.wren-fab-fig`)],Q.prototype,`fabFig`,void 0),e([_(`.wren-log`)],Q.prototype,`log`,void 0),e([_(`.wren-input`)],Q.prototype,`input`,void 0),Q=e([n(`qse-wren`)],Q);var $=class extends a{constructor(...e){super(...e),this.path=[],this.wiz=Ze(),this.storageProblem=null,this.storageFailed=!1,this.list={state:`loading`},this.model=null,this.device=null,this.serverEst=null,this.estLocal=!1,this.plan=null,this.confirmOpen=!1,this.applying=!1,this.removeAfter=!1,this.applyError=null,this.jobInitial=null,this.previousProfile=null,this.resetOpen=!1,this.unauthorized=!1,this.winNarrow=!1,this.simple=gt(),this.slotsDraft=null,this.winDraft={window:null},this.nextRefused=!1,this.communitySyncing=!1,this.store=null,this.devPoll=null,this.estTimer=null,this.estCtrl=null,this.planTimer=null,this.planCtrl=null,this.commCtrl=null,this.warnedDiff=!1,this.touchedAt=new Map,this.loadingModel=null,this.modelSeq=0,this.onUnauthorized=()=>{this.unauthorized=!0},this.onSignedIn=()=>{this.unauthorized=!1,this.list.state!==`ready`&&this.loadList();let e=this.path[0];e&&this.model?.state!==`ready`&&this.loadModel(e),this.loadDevice(),this.wiz.selection&&this.queueEstimate(),this.wiz.step===`review`&&this.queuePlan(),this.wiz.step===`weights`&&this.syncCommunity()},this.onReset=()=>{this.wiz=Ze(),this.plan=null,this.jobInitial=null,location.hash!==`#/setup`&&(location.hash=`#/setup`)},this.host={pageState:()=>Ie(this.m,this.wiz.step,this.wiz.selection,this.currentEstimate(),{errors:this.visibleErrors()}),execute:(e,t)=>this.executeTool(e,t),saveChat:e=>this.dispatch({type:`chat`,chat:e}),memAvailable:()=>this.device?.mem_available_gb??null}}connectedCallback(){super.connectedCallback(),this.store=$e();let e=Ke(this.store);this.wiz=e.state,this.storageProblem=e.problem,(e.problem===`corrupt`||e.problem===`version`)&&Ye(this.store),this.loadList(),this.devPoll=ve(()=>this.loadDevice(),5e3),this.devPoll.start(),window.addEventListener(`qse-setup-reset`,this.onReset),window.addEventListener(`qse-setup-unauthorized`,this.onUnauthorized),window.addEventListener(`qse-signed-in`,this.onSignedIn)}disconnectedCallback(){this.devPoll?.stop(),this.estTimer&&clearTimeout(this.estTimer),this.planTimer&&clearTimeout(this.planTimer),this.estCtrl?.abort(),this.planCtrl?.abort(),this.commCtrl?.abort(),window.removeEventListener(`qse-setup-reset`,this.onReset),window.removeEventListener(`qse-setup-unauthorized`,this.onUnauthorized),window.removeEventListener(`qse-signed-in`,this.onSignedIn),super.disconnectedCallback()}async loadList(){this.list={state:`loading`};try{let e=await b.models();this.list=e.models.length||e.errors.length?{state:`ready`,data:e}:{state:`empty`,note:D.modelsEmpty}}catch(e){let t=w(e);this.list={state:`error`,endpoint:t.endpoint,message:t.status===401?D.unauthorized:t.message}}}async loadModel(e){if(this.loadingModel===e)return;this.loadingModel=e;let t=++this.modelSeq;this.model={state:`loading`};try{let n=v(await b.model(e));if(t!==this.modelSeq)return;this.model={state:`ready`,data:n},await this.openModel(n,this.routeStep()??void 0)}catch(n){if(t!==this.modelSeq)return;let r=w(n);this.model={state:`error`,endpoint:r.endpoint,message:r.status===404?`${D.modelsError} (${e})`:r.message}}finally{t===this.modelSeq&&(this.loadingModel=null)}}async loadDevice(){try{this.device=await b.device()}catch{}}async openModel(e,t){let n=!(this.wiz.model===e.id&&this.wiz.selection);this.simple&&(t===void 0||t===`overview`)&&(t=`weights`);let r=null;if(n)try{r=(await b.current(e.id)).selection}catch{r=null}if(this.model?.state===`ready`&&this.model.data.id===e.id&&(this.dispatch({type:`open`,model:e,step:t,seed:r},!1),this.wiz.step===`weights`&&this.syncCommunity(),this.simple&&n&&!r&&this.wiz.selection))for(let t of vt(e,this.wiz.selection))this.dispatch(t,!1)}get m(){return this.model?.state===`ready`?this.model.data:null}routeStep(){let e=this.path[1];return y(e)?e:null}willUpdate(e){e.has(`path`)&&this.syncFromRoute()}syncFromRoute(){let[e,t]=this.path;if(!e){this.wiz.step!==`models`&&this.dispatch({type:`tiles`},!1);return}let n=this.m;if(!n||n.id!==e){this.loadModel(e);return}if(this.wiz.model!==e||!this.wiz.selection){this.openModel(n,y(t)?t:void 0);return}let r=y(t)?t:`overview`;r===`install`&&(!this.wiz.job||this.wiz.job_model!==this.wiz.model)&&(r=`review`),r===`models`&&(r=`overview`),r!==this.wiz.step&&this.dispatch({type:`go`,step:r},!1)}hashFor(e){return e.step===`models`||!e.model?`#/setup`:`#/setup/${e.model}/${e.step}`}dispatch(e,t=!0){let n=He(this.wiz,e,this.m,this.simple);if(n===this.wiz)return;let r=n.step!==this.wiz.step;this.wiz=n,this.store&&(this.storageFailed=!qe(this.store,n));let i=this.hashFor(n);location.hash!==i&&(t?location.hash=i:history.replaceState(null,``,i)),(r||e.type===`recommended`||e.type===`parallel`&&e.key===`slots`)&&(this.slotsDraft=null,this.nextRefused=!1),(r||e.type===`recommended`||e.type===`profile`)&&(this.winDraft={window:null}),r&&(t&&window.scrollTo({top:0,behavior:Ce()?`auto`:`smooth`}),n.step===`review`&&this.queuePlan(),n.step===`weights`&&this.syncCommunity()),e.type!==`go`&&e.type!==`next`&&e.type!==`previous`&&e.type!==`chat`&&e.type!==`dismiss`&&(this.queueEstimate(),n.step===`review`&&this.queuePlan())}user(e,t,n){this.touchedAt.set(Z(e,t),Date.now()),this.dispatch(n)}selKey(e){return JSON.stringify(e)}queueEstimate(){this.estTimer&&clearTimeout(this.estTimer),this.estTimer=setTimeout(()=>void this.confirmEstimate(),150)}async confirmEstimate(){let e=this.wiz.selection;if(!e)return;this.estCtrl?.abort();let t=this.estCtrl=new AbortController,n=this.selKey(e);try{let r=await b.estimate(e,t.signal);if(t.signal.aborted||n!==this.selKey(this.wiz.selection))return;this.serverEst={key:n,est:r},this.estLocal=!1;let i=this.m?j(this.m,e):null;i&&Math.abs(i.total_gb-r.total_gb)>.05&&!this.warnedDiff&&(this.warnedDiff=!0,console.warn(`setup: the server's estimate (${r.total_gb} GiB) and the page's (${i.total_gb} GiB) differ; showing the server's`))}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;this.estLocal=!0}}currentEstimate(){let e=this.m,t=this.wiz.selection;if(!e||!t)return null;let n=j(e,t,this.device?.mem_available_gb??null),r=this.serverEst;return r&&r.key===this.selKey(t)&&(!n||Math.abs(r.est.total_gb-n.total_gb)>.05)?{...r.est,available_gb:this.device?.mem_available_gb??r.est.available_gb}:n}queuePlan(){this.planTimer&&clearTimeout(this.planTimer),this.planTimer=setTimeout(()=>void this.loadPlan(),200)}async loadPlan(){let e=this.wiz.selection;if(!e)return;this.planCtrl?.abort();let t=this.planCtrl=new AbortController;(!this.plan||this.plan.state!==`ready`)&&(this.plan={state:`loading`});try{let n=await b.plan(e,t.signal);t.signal.aborted||(this.plan={state:`ready`,data:n})}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;let n=w(e);this.plan={state:`error`,endpoint:n.endpoint,message:n.message}}}async syncCommunity(){let e=this.m;if(!e)return;this.commCtrl?.abort();let t=this.commCtrl=new AbortController;this.communitySyncing=!0;try{let n=await b.communitySync(t.signal);if(t.signal.aborted||this.m?.id!==e.id)return;let r=this.m,i={...n.sync,dropped:n.sync.dropped.filter(t=>t.model===e.id),add_url:r.community_sync?.add_url||n.sync.add_url},a=v({...r,community_sync:i},n.models[e.id]??[]);this.model={state:`ready`,data:a};let o=this.wiz.selection;o&&!a.profiles.some(e=>e.id===o.profile)?this.dispatch({type:`open`,model:a},!1):this.queueEstimate()}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;let n=this.m;n?.community_sync&&!(e instanceof T&&e.status===404)&&(this.model={state:`ready`,data:{...n,community_sync:{...n.community_sync,error:w(e).message,checked_at:new Date().toISOString()}}})}finally{t.signal.aborted||(this.communitySyncing=!1)}}visibleErrors(){let e=this.issues()[this.wiz.step]??[];return this.wiz.step!==`review`||this.plan?.state!==`ready`?e:[...e,...this.plan.data.errors.filter(t=>!e.includes(t))]}async executeTool(e,t){let n=Tt(e,{model:this.m,selection:this.wiz.selection,step:this.wiz.step,touchedAt:this.touchedAt,turnStartedAt:t});if(!n.ok)return{ok:!1,result:n.error,summary:null,undo:null,stale:n.stale};if(n.kind===`read_device`)try{let e=await b.device();this.device=e;let t={mode:e.mode,mem_total_gb:e.mem_total_gb,mem_available_gb:e.mem_available_gb,gpu:e.gpu,disk:e.disk,engine:e.engine,active_profile:e.active_profile,profiles_on_disk:e.profiles_on_disk,password_set:e.password_set};return{ok:!0,result:JSON.stringify(t),summary:null,undo:null}}catch(e){return{ok:!1,result:w(e).message,summary:null,undo:null}}if(n.kind===`explain`)return M(n.target.step,this.simple)===M(this.wiz.step,this.simple)&&this.highlight(n.target.step,n.target.field),{ok:!0,result:n.result,summary:null,undo:null};let{target:r}=n;r.field&&r.step!==this.wiz.step&&this.dispatch({type:`go`,step:r.step});for(let e of n.actions)this.dispatch(e);await this.updateComplete,r.field&&this.highlight(r.step,r.field);let i=n.undo.length?()=>{for(let e of n.undo)this.dispatch(e);r.field&&this.highlight(r.step,r.field)}:null,a=r.field?()=>{r.step!==this.wiz.step&&this.dispatch({type:`go`,step:r.step}),this.highlight(r.step,r.field)}:null;return{ok:!0,result:n.result,summary:n.actions.length?n.summary:null,undo:i,reveal:a}}async highlight(e,t){if(!t)return;await this.updateComplete;let n=this.querySelector(`details.setup-fold[data-fold="${CSS.escape(e)}"]`);n&&!n.open&&(n.open=!0,await this.updateComplete);let r=this.querySelector(`[data-field="${CSS.escape(Z(e,t))}"]`);if(!r)return;let i=r.getBoundingClientRect();(i.top<120||i.bottom>innerHeight-80)&&r.scrollIntoView({block:`center`,behavior:Ce()?`auto`:`smooth`}),r.setAttribute(`data-wren-acting`,``),r.querySelector(`:scope > .wren-mini`)?.remove();let a=document.createElement(`span`);a.className=`wren-mini`,a.setAttribute(`aria-hidden`,`true`),r.append(a),o(Oe,a),setTimeout(()=>{r.removeAttribute(`data-wren-acting`),a.remove()},1600)}issues(){let e=this.m,t=this.wiz.selection;return!e||!t?{}:we(e,t,this.device,this.currentEstimate())}finishBlockers(){let e=this.m,t=this.wiz.selection;if(!e||!t)return[];let n=[],r=this.currentEstimate();r&&!r.fits&&n.push(D.reviewOver(je(r))),Ne(e,t,this.device)&&n.push(D.blockedToken);let i=Object.keys(this.issues()).filter(e=>e!==`review`);if(i.length&&n.push(D.blockedInvalid(i.map(e=>E[e].toLowerCase()))),this.plan?.state===`ready`&&!this.plan.data.ok)for(let e of this.plan.data.errors)n.includes(e)||n.push(e);return n}async finish(){let e=this.wiz.selection;if(!e||this.applying||this.finishBlockers().length)return;this.applying=!0,this.applyError=null,this.previousProfile=this.device?.active_profile??null,this.querySelector(`qse-wren`)?.sleepForInstall();let t=e;try{let e=this.device?.active_profile,n=await b.apply({selection:t,...this.removeAfter&&e&&e!==t.profile?{remove_after:[e]}:{}});if(this.confirmOpen=!1,!this.isConnected||this.path[0]!==t.model||this.wiz.step!==`review`){be(n.job);return}this.jobInitial=n.job,this.dispatch({type:`job`,id:n.job.id})}catch(e){this.applyError=this.applyErrorText(e)}finally{this.applying=!1}}setMode(e){this.simple=e,_t(e),this.dispatch({type:`mode`,simple:e})}applyErrorText(e){let t=w(e);if(t.status===401)return null;let n=e instanceof T?e.body:void 0;return n?.plan?(this.plan={state:`ready`,data:n.plan},this.confirmOpen=!1,n.plan.errors.join(` `)||t.message):e instanceof T&&e.type===`job_running`?D.blockedBusy:e instanceof T&&e.type===`busy`?D.blockedRequests(n?.error?.running??1):`${D.installError} ${t.message}`}render(){let e=this.wiz.step,t=this.m,n=this.path.length>0;return r`<div class="view view-setup" data-step=${n?e:`models`}>
      <div class="setup">
        <div class="setup-main">
          ${this.renderBanners()}
          ${yt(this.simple,e=>this.setMode(e))}
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
        <h1 class="setup-h1">${E.models}</h1>
        <p class="setup-lead">${xe.models}</p>
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
                  <span class="mtile-go"><span class="btn btn-primary">${t?`${D.tileOpen}: ${O[n.step]}`:D.tileOpen}</span></span>
                </a>`})}
            </div>
            ${e.data.errors.length?r`<div class="setup-load-errors" role="status"><p>${D.modelLoadErrors(e.data.errors.length)}</p><ul>${e.data.errors.map(e=>r`<li><code>${e.path}</code> <span class="muted">${e.message}</span></li>`)}</ul></div>`:l}`:l}
      <section class="guide" aria-labelledby="guide-h">
        <h2 class="guide-h" id="guide-h">${D.guideSteps}</h2>
        <ol class="guide-list">${x.slice(1).map(e=>r`<li><span class="guide-t">${O[e]}</span><span class="guide-d">${ye[e]}</span></li>`)}</ol>
      </section>
    </section>`}renderModelArea(){let e=this.model;if(this.wiz.step===`install`&&this.wiz.job&&this.wiz.job_model===this.wiz.model&&e?.state!==`ready`)return r`<section class="setup-step" data-step="install" aria-labelledby="setup-step-title">
        <header class="step-head"><h1 class="step-title" id="setup-step-title" tabindex="-1">${E.install}</h1></header>
        ${this.stepInstall()}
      </section>`;if(!e||e.state===`loading`)return c(8);if(e.state===`error`)return u(e.endpoint,e.message,()=>void this.loadModel(this.path[0]));if(e.state!==`ready`||!this.wiz.selection)return c(8);let t=e.data,n=this.wiz.selection,a=this.currentEstimate(),o=this.wiz.step,s=Ue(o,this.simple),d=a&&!a.fits?Pe(t,n,a):null;return r`
      <div class="setup-sticky">
        <qse-memory-bar .model=${t} .estimate=${a} .available=${this.device?.mem_available_gb??null} .local=${this.estLocal} .overText=${d}></qse-memory-bar>
        ${this.estLocal?r`<p class="setup-local" role="status" data-local-estimate>${D.memLocal}</p>`:l}
      </div>
      ${this.renderRail(t)}
      <section class="setup-step" data-step=${o} aria-labelledby="setup-step-title">
        <header class="step-head">
          <p class="step-kicker muted num">${this.simple?k.of3(s.n):D.stepOf(s.n,x.length)}</p>
          ${this.simple&&o===`review`?r`<h1 class="step-title" id="setup-step-title" tabindex="-1">${k.settingsTitle}</h1>
                <p class="step-intro">${k.settingsIntro}</p>`:r`<h1 class="step-title" id="setup-step-title" tabindex="-1">${E[o]}</h1>
                <p class="step-intro">${xe[o]}</p>`}
        </header>
        ${this.simple&&o===`review`?this.stepSimpleSettings(t,n,a):this.renderStep(t,n,a)}
      </section>
      ${this.renderNav()}
      ${this.simple?r`<p class="setup-simple-advanced"><button class="btn btn-ghost btn-sm" data-advanced-here @click=${()=>this.setMode(!1)}>${i(`sliders`)} ${k.showAll}</button></p>`:l}
    `}renderRail(e){let t=this.wiz.step,n=this.issues(),a=C(t);return this.simple?l:r`<nav class="rail-steps" aria-label="Setup steps">
      <div class="rail-phone" aria-hidden="true">
        <span class="num">${D.stepOf(a+1,x.length)}</span><span class="rail-phone-label">${E[t]}</span>
        <span class="rail-phone-bar"><span style="width:${(a+1)/x.length*100}%"></span></span>
      </div>
      <ol class="steps">
        ${x.map((o,s)=>{let c=this.wiz.visited.includes(o),u=c&&(n[o]?.length??0)>0,d=o===t,f=c&&!u&&s<a,p=o!==`install`||!!this.wiz.job&&this.wiz.job_model===this.wiz.model,m=o===`models`?`#/setup`:`#/setup/${e.id}/${o}`;return r`<li class="steps-item ${d?`is-current`:``} ${f?`is-done`:``} ${u?`is-bad`:``} ${c?`is-visited`:``}" data-rail=${o}>
            ${p?r`<a href=${m} aria-current=${d?`step`:l} title=${u?n[o].join(` `):E[o]}>
                  <span class="steps-node num" aria-hidden="true">${u?`!`:f?i(`check`):s+1}</span>
                  <span class="steps-label">${O[o]}</span>
                  ${u?r`<span class="sr-only">(needs a fix)</span>`:l}
                </a>`:r`<span class="steps-link is-locked" aria-disabled="true"><span class="steps-node num" aria-hidden="true">${s+1}</span><span class="steps-label">${O[o]}</span></span>`}
          </li>`})}
      </ol>
    </nav>`}renderNav(){let e=this.wiz.step;if(e===`install`)return l;let t=Ve(this.simple),n=Ue(e,this.simple),a=Math.max(0,t.indexOf(M(e,this.simple))),o=this.nextRefused?this.nextBlocker():null;return r`<nav class="setup-nav ${o?`has-why`:``}" aria-label="Step navigation">
      ${o?r`<p class="setup-nav-why" id="setup-nav-why" role="alert" data-next-refused>${i(`info`)} <span>${D.nextRefused(o)}</span></p>`:l}
      <button class="btn setup-prev" @click=${()=>a<=0?this.dispatch({type:`tiles`}):this.dispatch({type:`previous`})}>${D.previous}</button>
      <span class="setup-nav-mid muted small num">${this.simple?k.of3(n.n):D.stepOf(n.n,x.length)}</span>
      ${e===`review`&&(this.finishBlockers().length>0||this.plan?.state!==`ready`)?r`<p class="setup-nav-why" role="alert" data-finish-why>${i(`warn`)} <span>${this.finishBlockers()[0]??(this.plan?.state===`error`?D.finishPlanFailed:D.finishWaiting)}</span></p>`:l}
      ${e===`review`?r`<button class="btn btn-primary setup-finish" ?disabled=${this.finishBlockers().length>0||this.plan?.state!==`ready`} @click=${()=>this.confirmOpen=!0}>${D.finish}</button>`:r`<button class="btn btn-primary setup-next" aria-describedby=${o?`setup-nav-why`:l} @click=${()=>this.next()}><span>${D.next}<span class="setup-next-to">: ${this.simple?k.stepSettings:O[t[a+1]]}</span></span></button>`}
    </nav>`}nextBlocker(){let e=this.m,t=this.wiz.selection;if(this.wiz.step!==`parallelism`||!e||!t)return null;let n=Te(Le(e,t),this.slotsParse(e));if(n)return n;let r=this.winParse(e,t,`window`);return r&&!r.ok?`${D.window}: ${r.why}`:null}winParse(e,t,n){let r=this.winDraft[n];return r==null?null:Qe(r,Ge(e,t,n))}slotsParse(e){return this.slotsDraft==null?null:Fe(this.slotsDraft,Ae(e))}next(){if(this.nextBlocker()){this.nextRefused=!0,this.querySelector(`[data-field^="parallelism."].is-bad input`)?.focus();return}this.nextRefused=!1,this.dispatch({type:`next`})}stepSimpleSettings(e,t,n){return r`<div class="setup-folds" data-simple-settings>
      ${Y(O.components,`components`,this.stepComponents(e,t))}
      ${Y(O.parallelism,`parallelism`,this.stepParallel(e,t,n))}
      ${Y(O.methods,`methods`,this.stepMethods(e,t))}
      ${Y(O.review,`review`,this.stepReview(e,t,n))}
    </div>`}renderStep(e,t,n){switch(this.wiz.step){case`overview`:return this.stepOverview(e,t);case`weights`:return this.stepWeights(e,t);case`components`:return this.stepComponents(e,t);case`parallelism`:return this.stepParallel(e,t,n);case`methods`:return this.stepMethods(e,t);case`review`:return this.stepReview(e,t,n);case`install`:return this.stepInstall();default:return r``}}stepOverview(e,t){let n=e.profiles.find(e=>e.id===t.profile),a=Xe(e,t.profile);return r`<div class="ov">
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
      ${e.parallel?.rows.length?r`<section class="panel ov-table" aria-label="Speed on this build"><qse-parallel-table .model=${e} .selected=${t.profile}></qse-parallel-table></section>`:l}
    </div>`}stepWeights(e,t){let n=this.device,a=Ne(e,t,n),o=e.measurements.rows,s=Math.max(...o.filter(e=>e.comparable).map(e=>e.cells.greedy_512?.code?.tok_s??0));return r`
      ${a?r`<div class="banner banner-offline" role="alert" data-token-blocked>${i(`warn`)} ${D.privateNoToken}</div>`:l}
      ${!a&&n?.hf_token_present&&e.profiles.some(e=>e.hf_private&&!n.profiles_on_disk?.[e.id])?r`<p class="wnote muted small">${i(`info`)} ${D.privateRepo}</p>`:l}
      <div class="wcards" role="radiogroup" aria-label=${E.weights}>
        ${he(e).map(a=>{let c=t.profile===a.id,u=o.find(e=>e.profile===a.id),d=u?.cells.greedy_512,f=n?.profiles_on_disk?.[a.id],p=n?.active_profile===a.id,m=a.quality.top1_pct;return r`<label class="wcard ${c?`is-selected`:``} ${a.recommended?`is-rec`:``}" data-profile-card=${a.id} data-field=${c?`weights.profile`:l}>
            <input class="sr-only" type="radio" name="profile" .checked=${c} value=${a.id} @change=${()=>this.user(`weights`,`profile`,{type:`profile`,id:a.id})} />
            <span class="wcard-head">
              <span class="wcard-radio" aria-hidden="true"></span>
              <span class="wcard-name">${a.label}</span>
              ${p?r`<span class="sbadge sbadge-now">${D.activeNow}</span>`:l}
            </span>
            ${a.badges.length?r`<span class="wcard-badges">${a.badges.map(e=>r`<span class="sbadge ${e===`recommended`?`sbadge-rec`:``}">${e===`recommended`?D.recommended:e}</span>`)}</span>`:l}
            <span class="wcard-meters">
              <span class="wmeter" title=${a.quality.source}>
                <span class="wmeter-k">${D.weightsTop1}</span>
                <span class="wmeter-v num">${m==null?`–`:`${m.toFixed(1)} %`}</span>
                <span class="wmeter-bar q"><span style="width:${m==null?0:Math.max(4,(m-80)/20*100)}%"></span></span>
              </span>
              <span class="wmeter">
                <span class="wmeter-k">${D.weightsSpeed}</span>
                <span class="wmeter-v num">${d?.prose&&d?.code?r`${d.prose.tok_s.toFixed(1)} / ${d.code.tok_s.toFixed(1)}`:r`<span class="muted">${u&&!u.comparable?D.notComparable:D.notMeasured}</span>`}</span>
                <span class="wmeter-bar s"><span style="width:${d?.code&&u?.comparable?d.code.tok_s/s*100:0}%"></span></span>
              </span>
            </span>
            <span class="wcard-facts num">
              <span><span class="wcard-k">${D.weightsKl}</span> ${a.quality.kl_vs_bf16==null?`–`:a.quality.kl_vs_bf16.toFixed(4)}</span>
              <span><span class="wcard-k">${D.weightsMemory}</span> ${a.weights_gib.toFixed(1)} GiB</span>
              <span><span class="wcard-k">${D.weightsDisk}</span> ${f?D.onDisk:D.toDownload(a.disk_gb)}</span>
              ${B(e,a.id)?r`<span data-parallel-line><span class="wcard-k">${D.weightsAtOnce}</span> ${B(e,a.id)}</span>`:l}
              ${I(e)?r`<span data-slots-line><span class="wcard-k">${D.weightsSlots}</span> ${I(e)}</span>`:l}
            </span>
            ${this.renderSource(a)}
            ${a.note&&!this.simple?r`<span class="wcard-note">${a.note}</span>`:l}
            ${a.hf_private&&!f&&n?.hf_token_present===!1?r`<span class="wcard-private is-blocked" data-private>${i(`warn`)} ${D.privateNoTokenShort}</span>`:l}
          </label>`})}
      </div>
      ${le(e).length||e.community?r`<qse-community .model=${e} .sync=${e.community_sync??null} .selected=${t.profile} .device=${n} .syncing=${this.communitySyncing} .onUse=${e=>this.user(`weights`,`profile`,{type:`profile`,id:e})} .onRefresh=${()=>void this.syncCommunity()}></qse-community>`:l}
      <qse-installed-weights @weights-removed=${()=>void this.loadDevice()}></qse-installed-weights>`}renderSource(e){let t=me(e);return r`<span class="wcard-src" data-weights-src=${t.kind}>
      <span class="sbadge wsrc-kind wsrc-${t.kind}">${t.label}</span>
      <span class="wsrc-by">${D.madeBy} ${t.byUrl?r`<a href=${t.byUrl} target="_blank" rel="noopener noreferrer" title=${D.community.github(t.by.slice(1))}>${t.by}</a>`:t.by}</span>
      ${t.repoUrl?r`<a class="wsrc-repo" href=${t.repoUrl} target="_blank" rel="noopener noreferrer" title=${D.community.onHf(t.repo)}>${t.repo} ${i(`external`)}</a>`:r`<span class="wsrc-repo">${t.repo}</span>`}
    </span>`}stepComponents(e,t){let n=h(e,t.components);n.vision&&ne(e,t.profile)&&(n.vision={disabled:!0,reason:D.community.visionOff,required:!1});let a=[...new Set(e.components.map(e=>e.group))],o=A(e,t.profile);return r`
      <div class="step-tools">
        <button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`components`})}>${i(`refresh`)} ${D.resetToRecommended}</button>
      </div>
      <p class="comp-notice" role="status" aria-live="polite" data-cascade>${this.wiz.notice??``}</p>
      ${a.map(a=>r`<section class="comp-group" aria-labelledby=${`cg-${a}`}>
          <h2 class="comp-group-title" id=${`cg-${a}`}>${D.groups[a]??a}</h2>
          <ul class="comp-list">
            ${e.components.filter(e=>e.group===a).map(e=>{let a=m(e,t.components),s=n[e.id],c=e.id===`drafters`&&o.drafters_gib!=null?o.drafters_gib:e.memory_gib;return r`<li>
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
        </section>`)}`}stepParallel(e,t,n){let a=e.parallelism,o=n?.context_tokens_fit??0,s=Math.floor(o/t.slots),c=(e,t)=>this.user(`parallelism`,e,{type:`parallel`,key:e,value:t});return r`<div class="spar">
      <div class="spar-fields">
        ${this.slotsField(e,t)}
        ${this.windowField(e,t,`window`)}
        <div class="pfield" data-field="parallelism.kv_pool">
          <div class="pfield-k"><span class="pfield-label" id="pf-pool">${D.kvPool}</span><span class="pfield-what">${D.kvPoolWhat}</span></div>
          <label class="switch"><input type="checkbox" role="switch" .checked=${t.kv_pool} aria-labelledby="pf-pool" @change=${e=>c(`kv_pool`,e.target.checked)} /><span>${t.kv_pool?`on`:`off`}</span></label>
        </div>
        <div class="pfield" data-field="parallelism.floor_gb">
          <div class="pfield-k"><span class="pfield-label" id="pf-floor">${D.floor}</span><span class="pfield-what">${D.floorWhat}</span></div>
          <div class="floor-ctl">
            <input type="range" min=${a.floor_gb.min} max=${a.floor_gb.max} step="1" .value=${String(t.floor_gb)} aria-labelledby="pf-floor" @input=${e=>c(`floor_gb`,Number(e.target.value))} />
            <output class="num">${t.floor_gb} GiB</output>
          </div>
        </div>
        <div class="step-tools"><button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`parallelism`})}>${i(`refresh`)} ${D.resetToRecommended}</button></div>
        ${this.wiz.notice?r`<p class="comp-notice" role="status">${this.wiz.notice}</p>`:l}
      </div>
      <aside class="spar-readout panel" aria-live="polite" data-fit-tokens=${o}>
        <p class="spar-big num">${o.toLocaleString(`en-US`)}</p>
        <p class="spar-big-u">${D.fitsUnit}</p>
        <div class="spar-slots" aria-hidden="true">
          ${[{n:t.slots>1?`1–${t.slots}`:`1`,w:t.window,on:!0}].map(e=>r`<div class="spar-slot ${e.on?``:`is-off`}">
              <span class="spar-slot-n num">${e.n}</span>
              <span class="spar-slot-track"><span class="spar-slot-win" style="width:${e.w/a.window.max*100}%"></span><span class="spar-slot-fit" style="width:${Math.min(e.w,s)/a.window.max*100}%"></span></span>
              <span class="spar-slot-k num">${Je(e.w)}</span>
            </div>`)}
        </div>
        <p class="small">${D.fitsPerSlot(s,t.slots)}</p>
        ${n&&t.slots>1?r`<p class="small" data-lanes-full=${n.lanes_full}>${D.lanesFull(n.lanes_full,t.slots,t.window,n.lane_full_gb,t.kv_pool)}</p>`:l}
        ${n?r`<p class="small muted">${D.worstCase(n.kv_worst_gb,n.fits_worst??!1)}</p>`:l}
      </aside>
    </div>`}slotsField(e,t){let n=Ae(e),i=Le(e,t),a=this.slotsParse(e),o=a&&!a.ok?a.why:null,s=o??(i.over?D.slotsOver(i.n,i.overGb,i.fitMax):null),c=e=>this.user(`parallelism`,`slots`,{type:`parallel`,key:`slots`,value:e}),u=e=>{let r=a?.ok?a.n:t.slots,i=Ee(r,e,n);i===t.slots?this.slotsDraft=null:c(i)},d=e=>{let r=e.target.value,i=Fe(r,n);i.ok?(this.slotsDraft=null,i.n!==t.slots&&c(i.n)):this.slotsDraft=r},f=e=>{(e.key===`ArrowUp`||e.key===`ArrowDown`)&&(e.preventDefault(),u(e.key===`ArrowUp`?1:-1))},p=this.slotsDraft??String(t.slots),m=t.slots>=n.max&&!o;return r`<div class="pfield pfield-slots ${s?`is-bad`:``}" data-field="parallelism.slots">
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
        ${n.max>1?r`<p class="slots-cost" data-slots-cost>${D.slotsCost(i.cost.now,i.cost.full,i.cost.window)}</p>`:l}
        ${i.untested&&!o?r`<p class="slots-note" data-slots-untested>${D.slotsUntested(n.measured)}</p>`:l}
        ${i.passes>1&&!o?r`<p class="slots-note" data-slots-passes>${D.slotsPasses(i.passes)}</p>`:l}
        ${e.parallelism.slots.engine_max_reason?r`<p class="slots-note" data-slots-engine>${e.parallelism.slots.engine_max_reason}</p>`:l}
      </div>
    </div>`}windowField(e,t,n){let i=Ge(e,t,n),a=t.window,o=D.window,s=`pf-win`,c=this.winParse(e,t,n),u=c&&!c.ok?c.why:null,d=c?.ok?c.n:a,f=e=>this.user(`parallelism`,n,{type:`parallel`,key:n,value:e}),p=e=>{this.winDraft={...this.winDraft,[n]:null},e!==a&&f(e)},m=e=>{let t=e.target.value;this.winDraft={...this.winDraft,[n]:t};let r=Qe(t,i);r.ok&&r.n!==a&&f(r.n)},h=e=>{let t=e.key;if(t===`ArrowUp`||t===`ArrowDown`)p(N(d,t===`ArrowUp`?1:-1,i));else if(t===`PageUp`||t===`PageDown`){let e=t===`PageUp`?1:-1;p(N(N(d,e,i),e,i))}else if(t===`Home`)p(i.min);else if(t===`End`)p(i.max);else if(t===`Enter`)e.target.blur();else return;e.preventDefault()},g=()=>{c?.ok&&(this.winDraft={...this.winDraft,[n]:null})},_=this.winDraft[n]??a.toLocaleString(`en-US`),ee=d<=i.min&&!u,v=d>=i.max&&!u;return r`<div class="pfield pfield-win ${u?`is-bad`:``}" data-field=${`parallelism.${n}`}>
      <div class="pfield-k"><label class="pfield-label" id=${s} for=${`${s}-in`}>${o}</label><span class="pfield-what">${D.windowWhat}</span></div>
      <div class="win-ctl">
        <div class="stepper stepper-wide ${u?`is-bad`:``}" role="group" aria-labelledby=${s}>
          <button class="stepper-btn" type="button" aria-label=${D.windowShorter(o)} aria-controls=${`${s}-in`} ?disabled=${ee} @click=${()=>p(N(d,-1,i))}>
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
          <button class="stepper-btn" type="button" aria-label=${D.windowLonger(o)} aria-controls=${`${s}-in`} ?disabled=${v} title=${v?i.maxWhy:l} @click=${()=>p(N(d,1,i))}>
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9M8 3.5v9" /></svg>
          </button>
        </div>
        <span class="stepper-hint num" aria-hidden="true" data-win-short>${u?``:Je(d)}</span>
      </div>
      <div class="slots-say" id=${`${s}-say`}>
        ${u?r`<p class="slots-err" role="alert" data-win-error>${u}</p>`:l}
        ${!u&&v&&i.max<e.parallelism[n].max?r`<p class="slots-note" data-win-max>${i.maxWhy}</p>`:l}
      </div>
    </div>`}fieldControl(e,t,n){let i=n=>this.user(`methods`,t.id,{type:`field`,group:e,id:t.id,value:n}),a=`mf-${t.id}`;if(t.type===`enum`){let e=t.options??[];return e.length<=3?r`<div class="seg seg-wrap" role="radiogroup" aria-labelledby=${`${a}-l`}>${e.map(e=>r`<button class="seg-btn ${n===e.value?`is-on`:``}" role="radio" aria-checked=${n===e.value} @click=${()=>i(e.value)}>${e.label}</button>`)}</div>`:r`<select id=${a} class="sel" @change=${e=>i(e.target.value)}>${e.map(e=>r`<option value=${e.value} ?selected=${n===e.value}>${e.label}</option>`)}</select>`}return t.type===`bool`?r`<label class="switch"><input id=${a} type="checkbox" role="switch" .checked=${n===!0} @change=${e=>i(e.target.checked)} /><span>${n?`on`:`off`}</span></label>`:t.type===`int`||t.type===`number`?r`<input id=${a} class="num-input num" type="number" inputmode=${t.type===`int`?`numeric`:`decimal`} min=${t.min??l} max=${t.max??l} step=${t.type===`int`?1:.05} .value=${String(n)} @change=${e=>{let t=e.target.value;i(t===``?NaN:Number(t))}} />`:r`<input id=${a} class="text-input" type="text" .value=${String(n)} spellcheck="false" autocapitalize="off" @change=${e=>i(e.target.value)} />`}fieldRow(e,t,n){let i=n[e][t.id]??t.recommended,a=S(t,i),o=t.options?.find(e=>e.value===t.recommended)?.label??String(t.recommended);return r`<div class="mfield ${a?`is-bad`:``}" data-field=${`methods.${t.id}`}>
      <div class="pfield-k">
        <label class="pfield-label" id=${`mf-${t.id}-l`} for=${`mf-${t.id}`}>${t.label}</label>
        <span class="pfield-what">${t.what}</span>
        <span class="mfield-rec muted small">${D.recommendedValue}: <span class="num">${o}</span>${i===t.recommended?l:r` <button class="linkish" @click=${()=>this.user(`methods`,t.id,{type:`field`,group:e,id:t.id,value:t.recommended})}>${D.resetToRecommended.toLowerCase()}</button>`}</span>
      </div>
      <div class="mfield-ctl">${this.fieldControl(e,t,i)}${a?r`<p class="mfield-err" role="alert">${D.invalidValue(a)}</p>`:l}</div>
    </div>`}stepMethods(e,t){let n=e.advanced.some(e=>S(e,t.advanced[e.id]??e.recommended));return r`<div class="methods">
      ${e.methods.map(e=>this.fieldRow(`methods`,e,t))}
      <details class="adv" ?open=${n}>
        <summary><span class="adv-title">${D.advanced}</span> <span class="muted small">${D.advancedNote}</span></summary>
        ${e.advanced.map(e=>this.fieldRow(`advanced`,e,t))}
      </details>
      <div class="step-tools"><button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`methods`})}>${i(`refresh`)} ${D.resetToRecommended}</button></div>
    </div>`}stepReview(e,t,n){let a=this.plan,o=this.wiz.step,s=n&&!n.fits?Me(e,t):[],u=this.finishBlockers(),d=this.device,f=A(e,t.profile),p=a?.state===`ready`?a.data:null;return r`<div class="review">
      <section class="rv-card rv-memory ${n&&!n.fits?`is-over`:`is-ok`}" data-review-memory>
        <h2 class="rv-h">${D.reviewMemory}</h2>
        ${n?r`<p class="rv-say">${n.fits?D.reviewFits(n.total_gb,n.budget_gb??n.device_gb-n.floor_gb):Pe(e,t,n)}</p>`:l}
        ${s.length?r`<h3 class="rv-sub">${D.howToFix}</h3>
              <ul class="rv-fixes">${s.map(t=>r`<li><span>${t.text}</span><span class="muted small num">−${t.saves.toFixed(1)} GiB</span><a class="btn btn-sm" href=${`#/setup/${e.id}/${t.step}`}>${D.fixGo}: ${O[t.step]}</a></li>`)}</ul>`:l}
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
                <tbody>${p.changes.map(e=>r`<tr data-env=${e.key}><th scope="row"><code>${e.key}</code></th><td class="rv-old">${e.current??r`<span class="muted">${D.reviewUnset}</span>`}</td><td class="rv-new">${e.removed?r`<span class="rv-removed" data-removed>${We(e,D.reviewUnset).to}</span>`:e.next??r`<span class="muted">${D.reviewUnset}</span>`}</td><td class="muted small">${e.from}</td></tr>`)}</tbody>
              </table></div>`:r`<p class="muted">${D.reviewNoChanges}</p>`:l}
        ${p&&(p.errors.length||p.warnings.length)?r`<ul class="rv-notes">
              ${p.errors.map((t,n)=>{let a=p.error_codes?.[n],s=ke(a);return r`<li class="is-err" data-error-code=${a??l}>${i(`warn`)} <span>${t}</span>${s&&s!==o?r` <a class="btn btn-sm" href=${`#/setup/${e.id}/${s}`}>${D.planFix}: ${O[s]}</a>`:l}</li>`})}
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
            <p class="muted small">${A(e,this.wiz.selection.profile).label}</p>
            ${this.applyError?r`<p class="rv-warn" role="alert">${this.applyError}</p>`:l}
            <div class="setup-confirm-foot">
              <button class="btn" ?disabled=${this.applying} @click=${()=>this.confirmOpen=!1}>${D.cancel}</button>
              <button class="btn btn-primary setup-confirm-go" autofocus ?disabled=${this.applying} @click=${()=>void this.finish()}>${this.applying?D.installStarting:D.confirmGo}</button>
            </div>`:l}
    </dialog>`}updated(){let e=this.querySelector(`dialog.setup-confirm`);e&&(this.confirmOpen&&!e.open?e.showModal():!this.confirmOpen&&e.open&&e.close())}stepInstall(){let e=this.wiz.job;return e?r`<qse-install-progress .jobId=${e} .initial=${this.jobInitial} .previousProfile=${this.previousProfile} .onBack=${()=>this.dispatch({type:`job`,id:null})} @job-retry=${e=>{this.jobInitial=e.detail,this.dispatch({type:`job`,id:e.detail.id})}}></qse-install-progress>`:r`<p class="muted">${D.installStarting}</p>`}};e([t({attribute:!1})],$.prototype,`path`,void 0),e([d()],$.prototype,`wiz`,void 0),e([d()],$.prototype,`storageProblem`,void 0),e([d()],$.prototype,`storageFailed`,void 0),e([d()],$.prototype,`list`,void 0),e([d()],$.prototype,`model`,void 0),e([d()],$.prototype,`device`,void 0),e([d()],$.prototype,`serverEst`,void 0),e([d()],$.prototype,`estLocal`,void 0),e([d()],$.prototype,`plan`,void 0),e([d()],$.prototype,`confirmOpen`,void 0),e([d()],$.prototype,`applying`,void 0),e([d()],$.prototype,`removeAfter`,void 0),e([d()],$.prototype,`applyError`,void 0),e([d()],$.prototype,`jobInitial`,void 0),e([d()],$.prototype,`previousProfile`,void 0),e([d()],$.prototype,`resetOpen`,void 0),e([d()],$.prototype,`unauthorized`,void 0),e([d()],$.prototype,`winNarrow`,void 0),e([d()],$.prototype,`simple`,void 0),e([d()],$.prototype,`slotsDraft`,void 0),e([d()],$.prototype,`winDraft`,void 0),e([d()],$.prototype,`nextRefused`,void 0),e([d()],$.prototype,`communitySyncing`,void 0),$=e([n(`qse-setup`)],$);export{$ as QseSetup};