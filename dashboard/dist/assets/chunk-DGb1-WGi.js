import{_ as e,a as t,b as n,c as r,d as i,f as a,g as o,h as s,i as c,l,m as u,n as d,o as f,p,r as m,s as h,u as g,v as _,y as v}from"./chunk-DsvZC2dt.js";import{C as y,T as b,_ as x,b as S,c as C,d as w,g as T,h as E,i as D,n as ee,o as O,r as te,v as k,x as ne,y as A}from"./chunk-CkiyBafN.js";import{a as re,c as j,gt as ie,ht as ae,i as M,o as N,s as oe}from"./app-DTFfdS-y.js";import{n as se,t as ce}from"./chunk-DiaLOnD7.js";var P=2**30,le=[`model`,`profile`,`components`,`slots`,`window`,`side_window`,`kv_pool`,`floor_gb`,`methods`,`advanced`,`device`],F=e=>Math.round(e*100)/100+0,I=class extends Error{constructor(e,t,n){super(n),this.status=e,this.code=t}};function ue(e,t){return e.devices.find(e=>e.id===t.device)??e.devices[0]}function L(e,t){return e.profiles.find(e=>e.id===t)??e.profiles.find(e=>e.recommended)??e.profiles[0]}var R=e=>String(Number.isInteger(e)?e:+e.toPrecision(6));function z(e,t,n,r){if(typeof t!=`number`||!Number.isInteger(t))throw new I(422,`out_of_range`,`${e}: a whole number from ${R(n)} to ${R(r)}`);if(t<n||t>r)throw new I(422,`out_of_range`,`${e}: ${R(t)} is outside ${R(n)} to ${R(r)}`);return t}function de(e,t,n){if(n&&t%n)throw new I(422,`off_step`,`${e}: a multiple of ${B(n)}, not ${B(t)}`);return t}function fe(e,t){if(!t||typeof t!=`object`||Array.isArray(t))throw new I(422,`invalid_selection`,`the selection must be a JSON object`);let n=t,r=Object.keys(n).filter(e=>!le.includes(e)).sort();if(r.length)throw new I(422,`unknown_field`,`unknown field '${r[0]}'; a selection has ${le.join(`, `)}`);if(n.model!==e.id)throw new I(400,`unknown_model`,`no model '${n.model}' here`);let i=n.profile??(e.profiles.find(e=>e.recommended)??e.profiles[0]).id;if(typeof i!=`string`||!e.profiles.some(e=>e.id===i))throw new I(400,`unknown_profile`,`no weight profile '${i}' for ${e.id}`);let a={};for(let t of e.components)a[t.id]=t.default;let o=n.components??{};if(!o||typeof o!=`object`||Array.isArray(o))throw new I(422,`invalid_selection`,`components: an object of id -> true/false`);for(let[t,n]of Object.entries(o)){if(!(t in a))throw new I(400,`unknown_component`,`no tool '${t}' for ${e.id}`);if(typeof n!=`boolean`)throw new I(422,`invalid_selection`,`components.${t}: true or false`);a[t]=n}let s=[],c=t=>e.components.find(e=>e.id===t)?.label??t;for(let t of e.components)if(t.required&&!a[t.id]&&s.push(`${t.label} cannot be switched off`),a[t.id]){for(let e of t.depends_on)a[e]||s.push(`${t.label} needs ${c(e)} (${e}) on`);for(let e of t.conflicts)a[e]&&s.push(`${t.label} and ${c(e)} (${e}) cannot both be on`)}if(s.length)throw new I(422,`invalid_combination`,s.join(`; `));let l=e.parallelism,u=z(`slots`,n.slots??l.slots.default,l.slots.min,l.slots.max),d=de(`window`,z(`window`,n.window??l.window.default,l.window.min,l.window.max),l.window.step),f=Math.min(d,de(`side_window`,z(`side_window`,n.side_window??d,l.side_window.min,l.side_window.max),l.side_window.step)),p=n.kv_pool??l.kv_pool.default;if(typeof p!=`boolean`)throw new I(422,`invalid_selection`,`kv_pool: true or false`);let m=n.floor_gb??l.floor_gb.default;if(typeof m!=`number`||!(m>=l.floor_gb.min&&m<=l.floor_gb.max))throw new I(422,`out_of_range`,`floor_gb: a number from ${R(l.floor_gb.min)} to ${R(l.floor_gb.max)}`);let h={methods:{},advanced:{}};for(let t of[`methods`,`advanced`]){let r={},i=e[t];for(let e of i??[])r[e.id]=e.default;let a=n[t]??{};if(!a||typeof a!=`object`||Array.isArray(a))throw new I(422,`invalid_selection`,`${t}: an object of id -> value`);for(let[e,n]of Object.entries(a)){if(!i){r[e]=n;continue}let a=i.find(t=>t.id===e);if(!a)throw new I(422,`unknown_field`,`${t}.${e}: no such field`);let o=g(a,n);if(o)throw new I(422,`invalid_value`,`${t}.${e}: ${o}`);r[e]=n}h[t]=r}let _=n.device??e.devices[0].id;if(typeof _!=`string`||!e.devices.some(e=>e.id===_))throw new I(400,`unknown_device`,`no device preset '${_}' for ${e.id}`);return{model:e.id,profile:i,components:a,slots:u,window:d,side_window:f,kv_pool:p,floor_gb:m,methods:h.methods,advanced:h.advanced,device:_}}function pe(e,t){if(t===`weights`||t===`drafters`)return`model`;if(t===`kv`)return`chats`;if(t===`os`)return`system`;let n=e.components.find(e=>e.id===t);return n?n.group===`roles`?`helpers`:n.group===`speed`?`model`:`caches`:`engine`}var B=e=>e.toLocaleString(`en-US`);function me(e,t,n=null){let r=fe(e,t),i=e.parallelism,a=L(e,r.profile),o=ue(e,r),s=i.kv_bytes_per_token,c=i.typical_fill_tokens??32768,{slots:l,window:u}=r,d=r.side_window,f=[{id:`os`,label:`The OS and other programs`,gb:o.reserved_gib,kind:`fixed`},{id:`weights`,label:`Weights (${a.label})`,gb:a.weights_gib,kind:`fixed`}];for(let t of e.memory.parts)f.push({id:t.id,label:t.label,gb:t.gib,kind:t.kind});l>1&&f.push({id:`side_slots`,label:`${l-1} more slot(s), beyond their KV`,gb:(l-1)*i.side_slot_const_gib,kind:`fixed`});for(let t of e.components){if(!r.components[t.id])continue;let e=t.id===`drafters`?a.drafters_gib??t.memory_gib:t.memory_gib;if(e<=0&&t.memory_max_gib==null)continue;let n={id:t.id,label:t.label,gb:e,kind:`fixed`};t.memory_max_gib!=null&&(n.kind=`reserve`,n.worst_gb=t.memory_max_gib),f.push(n)}let p=u+(l-1)*d,m=Math.min(u,c)+(l-1)*Math.min(d,c),h=p*s/P,g=(r.kv_pool?m:p)*s/P,_=f.reduce((e,t)=>e+t.gb,0);f.push({id:`kv`,label:`Conversations (KV)`+(r.kv_pool?`, committed as they grow`:`, every window at load`),gb:g,kind:`kv`,worst_gb:h});let v=_+g,y=o.total_gib-r.floor_gb,b=y-_,x=F(v)<=F(y),S=F(b)>=F(h),C=Math.max(0,Math.floor(b*P/s)),w=[];if(x)S||w.push(`With every slot at its full window the conversations would need ${F(h-b).toFixed(2)} GiB more than there is. When memory runs short, a new request waits for room instead of the engine running out of memory.`);else{let e=[...f].sort((e,t)=>t.gb-e.gb).slice(0,3);w.push(`Over the budget by ${F(v-y).toFixed(2)} GiB. The largest parts: `+e.map(e=>`${e.label} ${e.gb.toFixed(2)} GiB`).join(`, `)+`.`),r.kv_pool||w.push(`Turn the shared KV pool on: without it every window takes its memory at load.`),w.push(`Fewer slots, a shorter window, a smaller weight profile or fewer tools bring it down.`)}if(b>0){let e=l?Math.min(u,Math.floor(C/l)):0;w.push(`About ${B(C)} tokens of conversation fit in all, ${B(e)} a slot when all ${l} are full (the window is ${B(u)}).`)}return{contract_version:`1.0`,total_gb:F(v),available_gb:n,device_gb:F(o.total_gib),floor_gb:F(r.floor_gb),budget_gb:F(y),parts:f.map(e=>({...e,gb:F(e.gb),...e.worst_gb==null?{}:{worst_gb:F(e.worst_gb)}})),fits:x,fits_worst:S,context_tokens_fit:C,kv_bytes_per_token:s,typical_fill_tokens:c,kv_worst_gb:F(h),room_gb:F(b),notes:w}}function V(e){let t=e.budget_gb??e.device_gb-e.floor_gb;return Math.max(0,F(e.total_gb-t))}function H(e,t,n=null){try{return me(e,t,n)}catch(r){if(!(r instanceof I))throw r;try{return me(e,{...t,methods:{},advanced:{}},n)}catch{return null}}}var he={over_budget:`parallelism`,missing_token:`weights`,disk_full:`weights`,missing_assets:`components`,secrets_exist:`methods`,port_in_use:`methods`};function ge(e){return e&&he[e]||null}function _e(e,t){return e===401?M.signIn.refused:e===429?M.signIn.tooMany:e===404?M.signIn.loginOff:e===0?M.signIn.offline:`${M.signIn.failed} ${t}`}function ve(e,t,n,r){let i={},a=(e,t)=>(i[e]??=[]).push(t);e.profiles.find(e=>e.id===t.profile)?U(e,t,n)&&a(`weights`,M.privateNoToken):a(`weights`,M.tool.noProfile(t.profile));for(let n of[...e.methods,...e.advanced]){let r=e.methods.includes(n)?t.methods:t.advanced,i=g(n,r[n.id]??n.recommended);i&&a(`methods`,`${n.label}: ${i}`)}let o=e.parallelism;return(t.slots<o.slots.min||t.slots>o.slots.max)&&a(`parallelism`,`${M.slots}: ${t.slots}`),r&&!r.fits&&a(`review`,M.reviewOver(V(r))),i}function U(e,t,n){let r=e.profiles.find(e=>e.id===t.profile);return!r||!r.hf_private||!n||n.profiles_on_disk?.[r.id]?!1:n.hf_token_present===!1}var ye=[262144,196608,131072,65536,32768,16384,8192];function be(t,n){let r=H(t,n);if(!r||r.fits)return[];let i=[],a={fits:!1,total_gb:r.total_gb},s=e=>H(t,e)??a;if(!n.kv_pool){let e=s({...n,kv_pool:!0});i.push({text:M.fixPool,step:`parallelism`,saves:r.total_gb-e.total_gb})}for(let e of ye.filter(e=>e<n.window&&e>=t.parallelism.window.min)){let t=s({...n,window:e,side_window:Math.min(n.side_window??n.window,e)});if(t.fits){i.push({text:M.fixWindow(e),step:`parallelism`,saves:r.total_gb-t.total_gb});break}}for(let e=n.slots-1;e>=t.parallelism.slots.min;e--){let a=s({...n,slots:e});if(a.fits||e===t.parallelism.slots.min){i.push({text:M.fixSlots(e),step:`parallelism`,saves:r.total_gb-a.total_gb});break}}let c=L(t,n.profile),l=t.profiles.filter(e=>e.weights_gib<c.weights_gib).sort((e,t)=>t.weights_gib-e.weights_gib);for(let e of l){let t=s({...n,profile:e.id});if(t.fits||e===l[l.length-1]){i.push({text:M.fixProfile(e.label),step:`weights`,saves:r.total_gb-t.total_gb});break}}let u=t.components.filter(e=>!e.required&&o(e,n.components)&&e.memory_gib>=.5).sort((e,t)=>t.memory_gib-e.memory_gib).slice(0,2);for(let t of u)i.push({text:M.fixComponent(e(t.label),t.memory_gib),step:`components`,saves:t.memory_gib});let d=V(r);return i.sort((e,t)=>Number(t.saves>=d)-Number(e.saves>=d)||t.saves-e.saves)}function xe(e,t,n){let r=be(e,t),i=r.length?M.overHint(r[0].text,r[1]?.text??null):null;return M.overSentence(V(n),i)}var Se=6144,Ce=/password|secret|token|api_?key/i;function we(e,t,n,r,i={}){let a={model:e?.id??null,step:t,step_index:p(t),selection:{}};if(!e||!n)return a.visible={title:j[t]},a;let o=e=>Object.fromEntries(Object.entries(e).map(([e,t])=>[e,Ce.test(e)&&t!==`keep`&&t!==`generate`?`<set>`:t]));a.selection={model:n.model,profile:n.profile,components:n.components,slots:n.slots,window:n.window,side_window:n.side_window,kv_pool:n.kv_pool,floor_gb:n.floor_gb,methods:o(n.methods),advanced:o(n.advanced)};let s=u(e).find(e=>e.id===t),c=e=>{if(t===`weights`)return n.profile;if(t===`components`)return n.components[e];if(t===`parallelism`)return n[e];if(t===`methods`)return e===`admin_password`?n.advanced[e]===`generate`?`generate`:`keep`:n.methods[e]??n.advanced[e]},l=(s?.fields??[]).map(e=>{let t={field:e.field,label:e.label,type:e.type,value:c(e.field)};return e.options&&(t.options=e.options),(e.min!=null||e.max!=null)&&(t.range=[e.min??null,e.max??null]),t});if(a.visible={title:j[t],fields:l,...i.errors?.length?{errors:i.errors}:{}},r){let e=[...r.parts].sort((e,t)=>t.gb-e.gb).slice(0,3).map(e=>({id:e.id,gb:e.gb}));a.estimate={total_gb:r.total_gb,budget_gb:r.budget_gb,device_gb:r.device_gb,available_gb:r.available_gb,floor_gb:r.floor_gb,fits:r.fits,fits_worst:r.fits_worst,kv_worst_gb:r.kv_worst_gb,context_tokens_fit:r.context_tokens_fit,parts:e}}let d=JSON.stringify(a);return d.length>Se&&a.visible&&(a.visible={title:j[t],fields:l.map(({options:e,...t})=>t)},d=JSON.stringify(a),d.length>Se&&(a.visible={title:j[t]})),a}function Te(e,t){if(t===`error`)return`error`;if(t===`unload`)return`sleeping`;if(t===`load-start`||t===`load-progress`)return`loading`;if(e===`sleeping`)return`sleeping`;if(e===`loading`)return t===`loaded`?`idle`:`loading`;switch(t){case`loaded`:case`done`:case`stop`:case`reset`:return`idle`;case`focus`:case`input`:return e===`idle`||e===`listening`||e===`error`?`listening`:e;case`blur`:return e===`listening`?`idle`:e;case`send`:case`state-thinking`:case`acted`:return`thinking`;case`first-token`:case`delta`:return e===`acting`?`acting`:`talking`;case`tool-call`:return`acting`}return e}var Ee={idle:{eyes:`dot`,mouth:`smile`,extra:`none`,led:`breathe`,motion:`breathe`},listening:{eyes:`wide`,mouth:`o`,extra:`none`,led:`steady`,motion:`lean`},thinking:{eyes:`up`,mouth:`flat`,extra:`question`,led:`amber`,motion:`wiggle`},loading:{eyes:`lid`,mouth:`flat`,extra:`none`,led:`steady`,motion:`slow`},talking:{eyes:`dot`,mouth:`open`,extra:`none`,led:`steady`,motion:`bounce`},acting:{eyes:`side`,mouth:`wide`,extra:`spark`,led:`flash`,motion:`hop`},error:{eyes:`small`,mouth:`wavy`,extra:`alarm`,led:`blink`,motion:`shake`},sleeping:{eyes:`shut`,mouth:`tiny`,extra:`z`,led:`off`,motion:`sleep`}};function De(e){return Ee[e]??Ee.idle}function Oe(e){return e==null||!Number.isFinite(e)?null:Math.max(.02,Math.min(1,e))}function ke(e,t=0){let n=.55+.45*Math.max(0,Math.min(1,Number.isFinite(e)?e:0));return Math.abs(n-t)<.12?n>.78?n-.25:n+.2:n}function W(){return typeof matchMedia==`function`&&matchMedia(`(prefers-reduced-motion: reduce)`).matches}var Ae=`M16.2 29.4L83.6 29Q85.2 29 86 30.4L89.6 38.6Q90.4 40 90.3 43.4L89.8 79.2Q89.7 82.2 86.8 82.2L13.4 82.6Q10.4 82.6 10.4 79.6L10.2 43Q10.1 40.2 10.9 38.8L14.4 30.6Q15 29.4 16.2 29.4Z`,je=`M12.5 40L87.5 40.3Q90.3 40.4 90.3 43.4L89.8 79.2Q89.7 82.2 86.8 82.2L13.4 82.6Q10.4 82.6 10.4 79.6L10.2 43Q10.2 40 12.5 40Z`,Me=[16.6,21.6,26.6,73.4,78.4,83.4],Ne=[47,55,63,71],Pe=Me.flatMap(e=>Ne.map(t=>`M${e-1.3} ${t}a1.3 1.3 0 1 0 2.6 0a1.3 1.3 0 1 0-2.6 0`)).join(``),Fe=b`<g class="gb-lit">${Me.flatMap((e,t)=>Ne.map(n=>b`<circle cx=${e} cy=${n} r="1.5" style=${`animation-delay:${(t*.12).toFixed(2)}s`} />`))}</g>`,Ie=(e,t)=>b`<g class=${`gb-eye gb-eye-${e}`}>
  <ellipse cx=${t} cy="56.4" rx="4.3" ry="5.5" />
  <circle class="gb-glint" cx=${t+1.5} cy="54.3" r="1.25" />
  <rect class="gb-lid" x=${t-4.8} y="50.4" width="9.6" height="5.8" />
</g>`,Le=e=>b`
  <ellipse class="gb-ground" cx="50" cy="88.6" rx="36" ry="2.6" />
  <g class="gb-all">
    <g class="gb-feet"><rect x="21" y="79.6" width="10.5" height="8" rx="2.2" /><rect x="68.5" y="79.6" width="10.5" height="8" rx="2.2" /></g>
    <g class="gb-body">
      <path class="gb-cut" d=${Ae} transform="translate(1.4 1.8)" />
      <path class="gb-edge" d=${Ae} />
      <path class="gb-front" d=${je} />
      <path class="gb-chin" d="M10.4 77.6L89.9 77.2 89.8 79.2Q89.7 82.2 86.8 82.2L13.4 82.6Q10.4 82.6 10.4 79.6Z" />
      <path class="gb-top" d="M16.2 29.4L83.6 29Q85.2 29 86 30.4L89.8 39.4Q89.6 40.3 87.5 40.3L12.5 40Q10.6 40 10.9 39L14.4 30.6Q15 29.4 16.2 29.4Z" />
      <path class="gb-seam" d="M11.4 40L89.4 40.3" />
      <path class="gb-dots" d=${Pe} />
      ${e}
      <circle class="gb-led-halo" cx="86.4" cy="43.4" r="2.9" />
      <circle class="gb-led" cx="86.4" cy="43.4" r="1.25" />
      <g class="gb-face">
        <ellipse class="gb-blush" cx="28.4" cy="66" rx="4.6" ry="2.8" />
        <ellipse class="gb-blush" cx="71.6" cy="66" rx="4.6" ry="2.8" />
        <g class="gb-eyes">${Ie(`l`,38.6)}${Ie(`r`,61.4)}</g>
        <path class="gb-shut" d="M34.4 56.6Q38.6 60.8 42.8 56.6M57.2 56.6Q61.4 60.8 65.6 56.6" />
        <path class="gb-m gb-m-smile" d="M45.8 67.8L47.7 70.4H52.3L54.2 67.8" />
        <ellipse class="gb-m gb-m-o gb-fill" cx="50" cy="69.8" rx="1.8" ry="2.2" />
        <path class="gb-m gb-m-flat" d="M47 69.6H53" />
        <ellipse class="gb-m gb-m-open gb-fill" cx="50" cy="70.4" rx="3.3" ry="3.4" />
        <g class="gb-m gb-m-wide"><path class="gb-fill" d="M44.6 67.2H55.4Q55.4 73 50 73Q44.6 73 44.6 67.2Z" /><ellipse class="gb-tongue" cx="50" cy="71.4" rx="2.6" ry="1.3" /></g>
        <path class="gb-m gb-m-wavy" d="M45.4 70.4L47.7 68.4 50 70.4 52.3 68.4 54.6 70.4" />
        <path class="gb-m gb-m-tiny" d="M48.4 69.8H51.6" />
      </g>
    </g>
  </g>
  <g class="gb-question"><text x="68" y="23">?</text><text x="77" y="15">?</text><text x="86" y="7">?</text></g>
  <g class="gb-alarm"><text x="70" y="22">?!</text><path class="gb-sweat" d="M87 32.4Q84 36.6 84 38.6A3 3 0 0 0 90 38.6Q90 36.6 87 32.4Z" /></g>
  <g class="gb-z"><text x="70" y="22">z</text><text x="78" y="14">z</text><text x="86" y="6">z</text></g>
  <g class="gb-spark"><path d="M10 22l0-7M6.5 18.5l7 0" /><path d="M20 12l0-5M17.5 9.5l5 0" /></g>
`,Re=56,G=0,K=class extends T{constructor(...e){super(...e),this.state=`idle`,this.size=72,this.progress=null,this.lastBlip=0}blip(){let e=performance.now();e-this.lastBlip<140||W()||(this.lastBlip=e,G=ke(Math.random(),G),this.querySelector(`.gb-m-open`)?.animate([{transform:`scaleY(.5)`},{transform:`scaleY(${G})`,offset:.4},{transform:`scaleY(.5)`}],{duration:180,easing:`ease-out`}))}hop(){let e=this.querySelector(`.gb-all`);if(!e||W())return Promise.resolve();let t={duration:1100,easing:`cubic-bezier(.3,.7,.4,1)`},n=e.animate([{transform:`translate(0,0) scale(1,1)`},{transform:`translate(0,0) scale(1.1,.86)`,offset:.1},{transform:`translate(-5px,-16px) scale(.92,1.1)`,offset:.28},{transform:`translate(-9px,0) scale(1.12,.84)`,offset:.44},{transform:`translate(-12px,-10px) scale(.94,1.07)`,offset:.62},{transform:`translate(-14px,0) scale(1.08,.9)`,offset:.78},{transform:`translate(0,0) scale(1,1)`}],t);return this.querySelector(`.gb-ground`)?.animate([{transform:`translateX(0) scaleX(1)`},{transform:`translateX(-5px) scaleX(.6)`,offset:.28},{transform:`translateX(-9px) scaleX(1.06)`,offset:.44},{transform:`translateX(-12px) scaleX(.75)`,offset:.62},{transform:`translateX(-14px) scaleX(1)`,offset:.78},{transform:`translateX(0) scaleX(1)`}],t),this.querySelector(`.gb-spark`)?.animate([{opacity:0,transform:`scale(.4)`},{opacity:1,transform:`scale(1.1)`,offset:.3},{opacity:0,transform:`scale(1.3)`}],{duration:900,delay:300,easing:`ease-out`}),n.finished.then(()=>void 0).catch(()=>void 0)}render(){let e=De(this.state),t=this.state===`loading`,n=t?Oe(this.progress):null,r=this.size<=Re?Math.round(this.size*108/84):this.size,i=(r-this.size)/2;return y`<svg
      class="goldbox"
      data-state=${this.state}
      data-eyes=${e.eyes}
      data-mouth=${e.mouth}
      data-extra=${e.extra}
      data-led=${e.led}
      data-motion=${e.motion}
      ?data-small=${i>0}
      viewBox="-4 -4 108 108"
      width=${r}
      height=${r}
      style=${i?`margin:${-i}px`:S}
      aria-hidden="true"
      focusable="false"
    >
      ${t?b`<g class="w-ring"><circle class="w-ring-track" cx="50" cy="58" r="48" pathLength="100" />${n==null?b`<circle class="w-ring-spin" cx="50" cy="58" r="48" pathLength="100" stroke-dasharray="18 82" />`:b`<circle class="w-ring-fill" cx="50" cy="58" r="48" pathLength="100" stroke-dasharray=${`${(n*100).toFixed(1)} 100`} />`}</g>`:S}
      ${Le(t?Fe:S)}
    </svg>`}};E([k({reflect:!0})],K.prototype,`state`,void 0),E([k({type:Number})],K.prototype,`size`,void 0),E([k({attribute:!1})],K.prototype,`progress`,void 0),K=E([A(`qse-wren-figure`)],K);var ze=b`<svg class="wren-mini-box" viewBox="0 0 30 22" width="32" height="24" aria-hidden="true" focusable="false">
  <rect class="mw-foot" x="6.4" y="16.4" width="3.4" height="4" rx="1" /><rect class="mw-foot" x="20.2" y="16.4" width="3.4" height="4" rx="1" />
  <path class="mw-edge" d="M6.2 2.6L23.8 2.4 27 6.6Q27.6 7.2 27.6 8.2V17.2Q27.6 18.4 26.4 18.4L3.6 18.6Q2.4 18.6 2.4 17.4V8.2Q2.4 7.2 3 6.6Z" />
  <path class="mw-front" d="M3.6 7L26.4 7.1Q27.6 7.1 27.6 8.3V17.2Q27.6 18.4 26.4 18.4L3.6 18.6Q2.4 18.6 2.4 17.4V8.2Q2.4 7 3.6 7Z" />
  <path class="mw-top" d="M6.2 2.6L23.8 2.4 27 7.1 3 7Z" />
  <ellipse class="mw-blush" cx="8.8" cy="14.4" rx="1.7" ry="1" /><ellipse class="mw-blush" cx="21.2" cy="14.4" rx="1.7" ry="1" />
  <g class="mw-eyes"><ellipse cx="12" cy="11.4" rx="1.25" ry="1.6" /><ellipse cx="18" cy="11.4" rx="1.25" ry="1.6" /></g>
  <ellipse class="mw-o" cx="15" cy="15" rx=".75" ry=".9" />
  <path class="mw-smile" d="M13.6 14.4L14.3 15.3H15.7L16.4 14.4" />
</svg>`;function Be(e){return e.group===`long`?e.mode===`greedy`?`long_greedy`:`long_sampled`:`${e.mode}_${e.len}`}function Ve(e){return e===`everyday`?[`profile`,`prose`,`chat`,`code`,`ttft8`,`kl`,`lossless`]:[`profile`,`long32k`,`long64k`,`ttft32`,`kl`]}var He={profile:M.colWeights,prose:M.colProse,chat:M.colChat,code:M.colCode,long32k:M.colLong32,long64k:M.colLong64,ttft8:M.colTtft8,ttft32:M.colTtft32,kl:M.colKl,lossless:M.colLossless};function q(e,t,n,r){let i=t.cells[Be(r)]??{};switch(n){case`profile`:{let n=e.profiles.find(e=>e.id===t.profile);return n?.bpw??n?.weights_gib??null}case`ttft8`:return t.ttft_s[8192]??null;case`ttft32`:return t.ttft_s[32768]??null;case`kl`:return e.profiles.find(e=>e.id===t.profile)?.quality.kl_vs_bf16??null;case`lossless`:return t.lossless?t.lossless.identical:null;default:return i[n]?.tok_s??null}}function Ue(e,t){let n=e.measurements.rows,r=n.filter(e=>e.comparable),i=n.filter(e=>!e.comparable),a=n=>q(e,n,t.sort,t);return r.sort((e,n)=>{let r=a(e),i=a(n);return r==null&&i==null?0:r==null?1:i==null?-1:t.dir===`asc`?r-i:i-r}),{main:r,apart:i}}var We=[`prose`,`chat`,`code`,`long32k`,`long64k`],J=class extends T{constructor(...e){super(...e),this.model=null,this.selected=null,this.view={mode:`greedy`,len:`512`,group:`everyday`,sort:`code`,dir:`desc`}}set(e){let t={...this.view,...e};e.group&&!Ve(e.group).includes(t.sort)&&(t.sort=e.group===`long`?`long32k`:`code`),this.view=t}sortBy(e){this.view.sort===e?this.set({dir:this.view.dir===`asc`?`desc`:`asc`}):this.set({sort:e,dir:We.includes(e)||e===`lossless`?`desc`:`asc`})}render(){let e=this.model;if(!e)return S;let t=this.view,n=Ve(t.group),{main:r,apart:i}=Ue(e,t),a={};for(let n of We)a[n]=Math.max(0,...r.map(r=>q(e,r,n,t)??0));let o=(e,t,n,r)=>y`<div class="seg" role="radiogroup" aria-label=${r}>${t.map(([t,r])=>y`<button class="seg-btn ${e===t?`is-on`:``}" role="radio" aria-checked=${e===t} data-val=${t} @click=${()=>n(t)}>${r}</button>`)}</div>`;return y`<div class="mt">
      <div class="mt-controls">
        ${o(t.mode,[[`greedy`,M.modeGreedy],[`sampled`,M.modeSampled]],e=>this.set({mode:e}),`Decoding`)}
        ${t.group===`everyday`?o(t.len,[[`512`,M.tokens512],[`128`,M.tokens128]],e=>this.set({len:e}),`Answer length`):S}
        ${o(t.group,[[`everyday`,M.groupEveryday],[`long`,M.groupLong]],e=>this.set({group:e}),`Tasks`)}
      </div>
      <div class="table-scroll mt-scroll">
        <table class="table mt-table" data-mode=${Be(t)}>
          <thead>
            <tr>
              ${n.map(e=>{let n=t.sort===e,r=We.includes(e)?M.tokPerS:e===`ttft8`||e===`ttft32`?M.seconds:``;return y`<th scope="col" class=${e===`profile`?`mt-name`:`num-col`} aria-sort=${n?t.dir===`asc`?`ascending`:`descending`:`none`}>
                  <button class="mt-sort ${n?`is-on`:``}" data-col=${e} @click=${()=>this.sortBy(e)} aria-label=${M.sortBy(He[e])}>
                    <span>${He[e]}</span>${r?y`<span class="mt-unit">${r}</span>`:S}<span class="mt-arrow" aria-hidden="true">${n?t.dir===`asc`?`▴`:`▾`:``}</span>
                  </button>
                </th>`})}
            </tr>
          </thead>
          <tbody>
            ${r.map(r=>this.row(e,r,n,t,a))}
            ${i.length?y`<tr class="mt-apart-head"><td colspan=${n.length}>${M.notComparable}</td></tr>
                  ${i.map(r=>this.row(e,r,n,t,a,!0))}`:S}
          </tbody>
        </table>
      </div>
      <div class="mt-foot muted small">${M.measuredOn(e.measurements.build,e.measurements.date)} <details class="mt-method"><summary>${M.howMeasured}</summary><p>${e.measurements.method}</p></details></div>
    </div>`}row(e,t,n,r,i,a=!1){let o=e.profiles.find(e=>e.id===t.profile),s=t.profile===this.selected;return y`<tr class="mt-row ${s?`is-selected`:``} ${a?`is-apart`:``}" data-profile=${t.profile} aria-current=${s?`true`:S}>
      ${n.map(n=>{if(n===`profile`)return y`<th scope="row" class="mt-name">
            <span class="mt-label">${o?.label??t.profile}</span>
            ${s?y`<span class="mt-mine">${M.selectedRow}</span>`:S}
            ${a?y`<span class="mt-note" title=${t.note??``}>${M.notComparable}</span>`:S}
          </th>`;let c=q(e,t,n,r);if(c==null)return y`<td class="num-col mt-na">–</td>`;if(n===`kl`)return y`<td class="num-col num">${c.toFixed(4)}</td>`;if(n===`lossless`){let e=t.lossless;return y`<td class="num-col num" title=${M.losslessTitle(e.identical,e.near_tie)}>${e.identical} + ${e.near_tie}</td>`}if(n===`ttft8`||n===`ttft32`)return y`<td class="num-col num">${c.toFixed(1)}</td>`;let l=i[n]?c/i[n]*100:0;return y`<td class="num-col num mt-speed"><span class="mt-bar" style="--w:${l.toFixed(1)}" aria-hidden="true"></span><span class="mt-val">${c.toFixed(1)}</span></td>`})}
    </tr>`}};E([k({attribute:!1})],J.prototype,`model`,void 0),E([k({attribute:!1})],J.prototype,`selected`,void 0),E([x()],J.prototype,`view`,void 0),J=E([A(`qse-measure-table`)],J);var Ge=[`model`,`helpers`,`caches`,`engine`,`system`,`chats`],Y=class extends T{constructor(...e){super(...e),this.model=null,this.estimate=null,this.available=null,this.local=!1,this.overText=null,this.open=!1,this.focusPart=null}render(){let e=this.estimate,t=this.model;if(!e||!t)return y`<section class="membar is-empty" aria-label=${M.memTitle}><div class="membar-line"><span class="membar-title">${M.memTitle}</span></div><div class="membar-track"></div></section>`;let n=e.budget_gb??e.device_gb-e.floor_gb,r=V(e),i=e.parts.find(e=>e.kind===`kv`),a=Math.max(e.device_gb,e.total_gb),o=e=>`${(e/a*100).toFixed(3)}%`,s=e=>Ge.indexOf(pe(t,e)),c=[...e.parts].sort((e,t)=>s(e.id)-s(t.id)),l=0,u=c.map(e=>{let n={p:e,left:l,cat:pe(t,e.id)};return l+=e.gb,n}),d=this.available==null?null:Math.max(0,e.device_gb-this.available),f=new Map;for(let e of u)f.set(e.cat,(f.get(e.cat)??0)+e.p.gb);let p=u.find(e=>e.p.id===this.focusPart),m=p?`${p.p.label}: ${p.p.gb.toFixed(2)} GiB${p.p.worst_gb!=null&&p.p.worst_gb!==p.p.gb?` (${M.memoryUpTo(p.p.worst_gb)})`:``}`:null,h=`${M.memTitle}: ${M.memFits(e.total_gb,n)}. ${r>0?this.overText??``:M.memLeft(n-e.total_gb)}`;return y`<section class="membar ${r>0?`is-over`:``} ${this.open?`is-open`:``}" aria-label=${M.memTitle} data-fits=${e.fits} data-total=${e.total_gb.toFixed(2)}>
      <div class="membar-line">
        <span class="membar-title">${M.memTitle}</span>
        <span class="membar-fig num" data-membar-total><b>${e.total_gb.toFixed(1)}</b><span class="membar-of"> / ${n.toFixed(1)} GiB</span></span>
        <span class="membar-say ${r>0?`is-over`:``}" data-membar-say role=${r>0?`alert`:S}>${r>0?this.overText??M.overSentence(r,null):M.memLeft(n-e.total_gb)}</span>
        <span class="membar-now num">${d==null?S:y`<i class="membar-now-dot" aria-hidden="true"></i>${M.memNow(d)}`}</span>
        <span class="membar-device num">${M.memDevice(e.device_gb)}</span>
        <button class="btn btn-ghost btn-sm membar-toggle" aria-expanded=${this.open} aria-controls="membar-more" aria-label=${this.open?M.memCollapse:M.memExpand} @click=${()=>this.open=!this.open}>
          <svg class="icon" viewBox="0 0 20 20" aria-hidden="true"><path d="m5 8 5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
        </button>
      </div>
      <div class="membar-track" role="img" aria-label=${h}>
        <span class="membar-free" style="left:${o(n)};width:${o(Math.max(0,e.device_gb-n))}" title=${`${M.memFloor}: ${e.floor_gb} GiB`}><span class="membar-floor-label">${M.memFloor} ${e.floor_gb} GiB</span></span>
        ${u.map(e=>y`<span
            class="membar-seg cat-${e.cat} ${e.p.kind===`kv`?`is-kv`:``} ${this.focusPart===e.p.id?`is-focus`:``}"
            style="left:${o(e.left)};width:${o(e.p.gb)}"
            data-part=${e.p.id}
            tabindex="0"
            aria-label=${`${e.p.label} ${e.p.gb.toFixed(1)} GiB`}
            @mouseenter=${()=>this.focusPart=e.p.id}
            @mouseleave=${()=>this.focusPart=null}
            @focus=${()=>this.focusPart=e.p.id}
            @blur=${()=>this.focusPart=null}
          ></span>`)}
        ${i&&i.worst_gb!=null&&i.worst_gb>i.gb?y`<span class="membar-worst" style="left:${o(e.total_gb)};width:${o(Math.min(i.worst_gb-i.gb,a-e.total_gb))}" title=${`${M.memWorst}: ${i.worst_gb.toFixed(1)} GiB`}></span>`:S}
        ${r>0?y`<span class="membar-over" style="left:${o(n)};width:${o(e.total_gb-n)}"></span>`:S}
        <span class="membar-floor" style="left:${o(n)}"></span>
        ${d==null?S:y`<span class="membar-nowtick" style="left:${o(Math.min(d,a))}" title=${M.memNow(d)}></span>`}
      </div>
      <div class="membar-more" id="membar-more">
        <ul class="membar-legend">
          ${Ge.filter(e=>f.has(e)).map(e=>y`<li class="cat-${e}"><i class="membar-sw" aria-hidden="true"></i>${M.memCat[e]} <span class="num">${(f.get(e)??0).toFixed(1)}</span></li>`)}
          <li class="cat-free"><i class="membar-sw" aria-hidden="true"></i>${M.memFloor} <span class="num">${e.floor_gb}</span></li>
        </ul>
        <p class="membar-detail" aria-live="polite">${m??(this.local?M.memLocal:e.notes[0]??`\xA0`)}</p>
      </div>
    </section>`}};E([k({attribute:!1})],Y.prototype,`model`,void 0),E([k({attribute:!1})],Y.prototype,`estimate`,void 0),E([k({attribute:!1})],Y.prototype,`available`,void 0),E([k({type:Boolean})],Y.prototype,`local`,void 0),E([k({attribute:!1})],Y.prototype,`overText`,void 0),E([x()],Y.prototype,`open`,void 0),E([x()],Y.prototype,`focusPart`,void 0),Y=E([A(`qse-memory-bar`)],Y);var X=class extends T{constructor(...e){super(...e),this.busy=!1,this.error=null}async submit(e){e.preventDefault();let t=this.querySelector(`.signin-input`),n=t?.value??``;if(n&&!this.busy){this.busy=!0,this.error=null;try{await _.signIn(n),t&&(t.value=``),window.dispatchEvent(new CustomEvent(`qse-signed-in`))}catch(e){let t=w(e);this.error=_e(t.status,t.message)}finally{this.busy=!1}}}render(){return y`<form class="panel signin-card" data-setup-signin @submit=${e=>void this.submit(e)}>
      <h2 class="signin-title">${D(`warn`)} ${M.signIn.title}</h2>
      <p class="signin-body">${M.signIn.body}</p>
      <label class="signin-field"
        ><span>${M.signIn.label}</span>
        <input class="signin-input text-input" type="password" name="password" autocomplete="current-password" required spellcheck="false" autocapitalize="off" />
      </label>
      ${this.error?y`<p class="login-error" role="alert" data-signin-error>${this.error}</p>`:S}
      <div class="signin-foot">
        <button class="btn btn-primary" type="submit" ?disabled=${this.busy}>${this.busy?M.signIn.checking:M.signIn.go}</button>
        <span class="muted small">${M.signIn.kept}</span>
      </div>
    </form>`}};E([x()],X.prototype,`busy`,void 0),E([x()],X.prototype,`error`,void 0),X=E([A(`qse-setup-signin`)],X);var Ke=[`set_field`,`toggle`,`select_profile`,`go_to_step`,`next`,`previous`,`explain`,`read_device`],Z=(e,t=!1)=>({ok:!1,error:e,...t?{stale:t}:{}});function qe(e,t){return`${e}.${t}`}function Je(e){if(e==null)return{};if(typeof e==`string`){if(!e.trim())return{};try{let t=JSON.parse(e);return t&&typeof t==`object`&&!Array.isArray(t)?t:null}catch{return null}}return typeof e==`object`&&!Array.isArray(e)?e:null}function Ye(e){if(typeof e==`number`&&Number.isInteger(e)&&e>=0&&e<l.length)return l[e];if(typeof e==`string`){if(/^\d+$/.test(e))return Ye(Number(e));if(i(e))return e}return null}function Xe(e,t){return(e===`int`||e===`number`)&&typeof t==`string`&&t.trim()!==``&&Number.isFinite(Number(t))?Number(t):e===`bool`&&(t===`true`||t===`false`)?t===`true`:t}function Ze(e){return typeof e==`boolean`?e?`on`:`off`:typeof e==`number`?e.toLocaleString(`en-US`):e}function Qe(e,t){let n=String(e.name??``);if(!Ke.includes(n))return/finish|apply|install|reset|delete|unload|load|confirm|submit/i.test(n)?Z(M.tool.unknown(n)+` `+M.tool.userOnly(n)):Z(M.tool.unknown(n));let i=Je(e.args);if(!i)return Z(M.tool.badArgs(`expected a JSON object`));if(n===`read_device`)return{ok:!0,kind:`read_device`};let o=t.model,s=t.selection,c=(e,n)=>{let r=t.touchedAt?.get(qe(e,n));return r!=null&&t.turnStartedAt!=null&&r>t.turnStartedAt};switch(n){case`go_to_step`:{let e=i.n??i.step??i.index,n=Ye(e);return n?n===`install`?Z(M.tool.installStep):n!==`models`&&!o?Z(M.tool.noModel):{ok:!0,kind:`action`,actions:[{type:n===`models`?`tiles`:`go`,step:n}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:M.toolDone.step(j[n]),result:M.tool.ok}:Z(M.tool.noStep(String(e)))}case`next`:{if(!o)return Z(M.tool.noModel);let e=p(t.step);if(e>=p(`review`))return Z(M.tool.atEnd);let n=l[e+1];return{ok:!0,kind:`action`,actions:[{type:`next`}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:M.toolDone.step(j[n]),result:M.tool.ok}}case`previous`:{let e=p(t.step);if(e<=0)return Z(M.tool.atStart);let n=l[e-1];return{ok:!0,kind:`action`,actions:[{type:`previous`}],undo:[{type:`go`,step:t.step}],target:{step:n,field:null},summary:M.toolDone.step(j[n]),result:M.tool.ok}}}if(!o||!s)return Z(M.tool.noModel);if(n===`explain`){let e=String(i.field??i.id??``),t=o.components.find(t=>t.id===e);if(t)return{ok:!0,kind:`explain`,target:{step:`components`,field:t.id},result:`${t.label}: ${t.what}${t.note?` `+t.note:``}`};let n=a(o,e);if(n)return{ok:!0,kind:`explain`,target:{step:`methods`,field:n.field.id},result:`${n.field.label}: ${n.field.what}`};let r=o.profiles.find(t=>t.id===e);if(r)return{ok:!0,kind:`explain`,target:{step:`weights`,field:`profile`},result:`${r.label}: KL ${r.quality.kl_vs_bf16??`?`}, ${r.weights_gib} GiB in memory${r.note?`. `+r.note:``}`};let s={slots:M.slotsWhat,window:M.windowWhat,side_window:M.sideWindowWhat,kv_pool:M.kvPoolWhat,floor_gb:M.floorWhat};return s[e]?{ok:!0,kind:`explain`,target:{step:`parallelism`,field:e},result:s[e]}:Z(M.tool.noField(`any`,e))}if(n===`select_profile`){let e=String(i.id??i.profile??``),t=o.profiles.find(t=>t.id===e);return t?c(`weights`,`profile`)?Z(M.tool.stale,!0):s.profile===e?{ok:!0,kind:`action`,actions:[],undo:[],target:{step:`weights`,field:`profile`},summary:M.toolDone.profile(t.label),result:M.tool.same}:{ok:!0,kind:`action`,actions:[{type:`profile`,id:e}],undo:[{type:`profile`,id:s.profile}],target:{step:`weights`,field:`profile`},summary:M.toolDone.profile(t.label),result:M.tool.ok}:Z(M.tool.noProfile(e))}if(n===`toggle`){let e=String(i.component??i.id??``),t=Xe(`bool`,i.on);return typeof t==`boolean`?$e(o,s,e,t,c(`components`,e)):Z(M.tool.badArgs(`on must be true or false`))}let d=Ye(i.step);if(!d)return Z(M.tool.noStep(String(i.step)));let f=String(i.field??``),m=u(o).find(e=>e.id===d)?.fields.find(e=>e.field===f);if(!m)return Z(M.tool.noField(d,f));if(r.has(f))return Z(M.tool.secret(m.label));let h=Xe(m.type,i.value),_=g(m,h);if(_)return Z(M.tool.badArgs(`${f}: ${_}`));if(d===`components`)return $e(o,s,f,h,c(d,f));if(c(d,f))return Z(M.tool.stale,!0);let v={step:d,field:f},y=M.toolDone.field(m.label,Ze(h));if(d===`weights`)return Qe({name:`select_profile`,args:{id:h}},t);if(d===`parallelism`){let e=f,t=e===`side_window`?s.side_window??s.window:s[e];return t===h?{ok:!0,kind:`action`,actions:[],undo:[],target:v,summary:y,result:M.tool.same}:{ok:!0,kind:`action`,actions:[{type:`parallel`,key:e,value:h}],undo:[{type:`parallel`,key:e,value:t}],target:v,summary:y,result:M.tool.ok}}if(d===`methods`){let e=a(o,f);if(!e)return Z(M.tool.noField(d,f));let t=s[e.group][f]??e.field.recommended;return t===h?{ok:!0,kind:`action`,actions:[],undo:[],target:v,summary:y,result:M.tool.same}:{ok:!0,kind:`action`,actions:[{type:`field`,group:e.group,id:f,value:h}],undo:[{type:`field`,group:e.group,id:f,value:t}],target:v,summary:y,result:M.tool.ok}}return Z(M.tool.noField(d,f))}function $e(e,t,n,r,i){let a=e.components.find(e=>e.id===n);if(!a)return Z(M.tool.noComponent(n));if(a.required&&!r)return Z(M.tool.required(a.label));let o=s(e,t.components)[n];if(r&&o.disabled&&!o.required)return Z(o.reason??M.tool.noComponent(n));if(i)return Z(M.tool.stale,!0);let c={step:`components`,field:n},l=r?M.toolDone.toggleOn(a.label):M.toolDone.toggleOff(a.label);return(a.required||(t.components[n]??a.default))===r?{ok:!0,kind:`action`,actions:[],undo:[],target:c,summary:l,result:M.tool.same}:{ok:!0,kind:`action`,actions:[{type:`component`,id:n,on:r}],undo:[{type:`components`,set:{...t.components}}],target:c,summary:l,result:M.tool.ok}}var et=0;function tt(e){return e.charAt(0).toUpperCase()+e.slice(1)}var Q=class extends T{constructor(...e){super(...e),this.host=null,this.step=`models`,this.chat=[],this.finished=!1,this.phase=`checking`,this.status=null,this.figure=`sleeping`,this.progress=null,this.streaming=!1,this.partial=``,this.composer=``,this.error=null,this.notice=null,this.sources=[],this.open=!1,this.peek=!1,this.chips=new Map,this.ctrl=null,this.loadPoll=null,this.ping=null,this.loadStartedAt=0,this.lastUserText=``,this.opener=null,this.onKey=e=>{if(!this.open||!this.sheetMode())return;if(e.key===`Escape`){e.preventDefault(),this.close();return}if(e.key!==`Tab`)return;let t=this.querySelector(`.wren-panel`);if(!t)return;let n=[...t.querySelectorAll(`button:not([disabled]), textarea:not([disabled]), a[href], input:not([disabled])`)].filter(e=>e.offsetParent!==null);if(!n.length)return;let r=n[0],i=n[n.length-1];e.shiftKey&&document.activeElement===r?(e.preventDefault(),i.focus()):!e.shiftKey&&document.activeElement===i?(e.preventDefault(),r.focus()):t.contains(document.activeElement)||(e.preventDefault(),r.focus())}}connectedCallback(){super.connectedCallback(),this.checkStatus(),this.ping=setInterval(()=>{this.phase===`ready`&&document.visibilityState===`visible`&&this.checkStatus(!0)},3e4),document.addEventListener(`keydown`,this.onKey)}disconnectedCallback(){this.ctrl?.abort(),this.loadPoll&&clearTimeout(this.loadPoll),this.ping&&clearInterval(this.ping),document.removeEventListener(`keydown`,this.onKey),this.lockScroll(!1),super.disconnectedCallback()}updated(e){e.has(`finished`)&&this.finished&&this.phase!==`finished`&&this.sleepForInstall()}emit(e){this.figure=Te(this.figure,e)}async checkStatus(e=!1){if(this.finished){this.phase=`finished`,this.figure=`sleeping`;return}try{let e=await _.assistant.status();this.applyStatus(e)}catch(t){if(e)return;this.phase=`failed`,this.error=w(t).status===0?M.wrenOffline:w(t).message,this.emit(`error`)}}applyStatus(e){switch(this.status=e,e.state){case`loaded`:this.phase!==`ready`&&(this.phase=`ready`,this.progress=null,this.figure=`loading`,this.emit(`loaded`));break;case`loading`:this.phase=`loading`,this.progress=this.timeProgress(e.loading_s??null),this.emit(`load-progress`),this.scheduleLoadPoll();break;case`refused`:this.phase=`refused`,this.emit(`error`);break;case`failed`:this.phase=`failed`,this.error=e.reason??M.wrenFailed,this.emit(`error`);break;default:this.phase=e.model==null?`refused`:`asleep`,this.figure=`sleeping`}}timeProgress(e){let t=e??(Date.now()-this.loadStartedAt)/1e3;return Math.min(.95,1-Math.exp(-t/8))}scheduleLoadPoll(){this.loadPoll&&clearTimeout(this.loadPoll),this.loadPoll=setTimeout(()=>void this.checkStatus(!0),500)}async wake(){this.error=null,this.loadStartedAt=Date.now(),this.phase=`loading`,this.progress=0,this.emit(`load-start`);try{this.applyStatus(await _.assistant.load())}catch(e){let t=w(e),n=e instanceof C?e.body:void 0;n?.assistant&&(this.status=n.assistant),t.status===409?(this.phase=`refused`,n?.assistant?.reason&&(this.status={...this.status,reason:n.assistant.reason})):(this.phase=`failed`,this.error=t.status===0?M.wrenOffline:t.message),this.emit(`error`)}}async sleep(){this.stop();try{this.applyStatus(await _.assistant.unload())}catch{}this.phase=`asleep`,this.emit(`unload`)}async sleepForInstall(){this.ctrl?.abort(),this.phase=`finished`,this.emit(`unload`),this.close();try{await _.assistant.unload()}catch{}}stop(){this.ctrl&&this.ctrl.abort()}async send(e){let t=e.trim();if(!t||this.streaming||!this.host)return;this.lastUserText=t,this.composer=``,this.error=null,this.notice=null;let n=[...this.chat,{role:`user`,content:t}];this.setChat(n),await this.turn(n)}async retry(){if(!this.lastUserText)return;this.error=null;let e=this.chat[this.chat.length-1]?.role===`user`?this.chat:[...this.chat,{role:`user`,content:this.lastUserText}];this.setChat(e),await this.turn(e)}setChat(e){this.chat=e,this.host?.saveChat(e),this.updateComplete.then(()=>this.scrollLog())}async turn(e){let t=this.host,n=Date.now(),r=e;this.streaming=!0,this.emit(`send`),this.ctrl=new AbortController;let i=this.ctrl.signal;try{for(let e=1;;e++){let a=``,o=[],s=[],c=!0,l=null;if(this.partial=``,await v({messages:r,page:t.pageState()},e=>{e.type===`state`?this.emit(e.state===`thinking`?`state-thinking`:e.state===`loading`?`load-progress`:`first-token`):e.type===`text`?(a+=e.delta,this.partial=a,c?(c=!1,this.emit(`first-token`)):this.emit(`delta`),this.fig?.blip(),this.fabFig?.blip(),this.scrollLog()):e.type===`tool_call`?o.push({id:e.id||`call_${++et}`,name:e.name,args:e.args}):e.type===`done`?this.sources=e.sources??[]:e.type===`error`&&(l=e.message)},i),l&&/another tab/i.test(l)){this.notice=M.wrenBusy;break}if(l)throw new C(`/v1/setup/assistant/chat`,500,l,`stream`);for(let e of o){this.emit(`tool-call`),this.sheetPeek();let r=await t.execute(e,n);this.fig?.hop(),s.push({role:`tool`,tool_call_id:e.id,content:JSON.stringify(r.ok?{ok:!0,result:r.result,...r.summary?{summary:r.summary}:{}}:{ok:!1,error:r.result,...r.stale?{stale:!0}:{}})}),this.chips=new Map(this.chips).set(e.id,{summary:r.summary,ok:r.ok,stale:r.stale,undo:r.undo,undone:!1}),this.emit(`acted`)}let u={role:`assistant`,content:a,...o.length?{tool_calls:o}:{}};if(r=[...r,u,...s],this.partial=``,this.setChat(r),!o.length)break;if(e>=4){this.notice=M.wrenRoundLimit;break}}this.emit(`done`)}catch(e){if(e.name===`AbortError`)this.partial&&this.setChat([...r,{role:`assistant`,content:`${this.partial} ${M.wrenStopped}`}]),this.partial=``,this.emit(`stop`);else{let t=w(e);t.status===409?this.notice=M.wrenBusy:t.status===0?this.error=M.wrenOffline:t.status===503||/unload|asleep|not loaded/i.test(t.message)?(this.error=M.wrenWentToSleep,this.checkStatus(!0)):this.error=t.message||M.wrenError,this.partial&&this.setChat([...r,{role:`assistant`,content:this.partial}]),this.partial=``,this.emit(t.status===409?`done`:`error`)}}finally{this.streaming=!1,this.ctrl=null}}undoChip(e){let t=this.chips.get(e);t?.undo&&!t.undone&&(t.undo(),this.chips=new Map(this.chips).set(e,{...t,undone:!0}))}scrollLog(){let e=this.log;e&&(e.scrollTop=e.scrollHeight)}sheetMode(){return typeof matchMedia==`function`&&matchMedia(`(max-width: 1100px)`).matches}toggleOpen(){this.open?this.close():(this.opener=document.activeElement,this.open=!0,this.lockScroll(!0),this.updateComplete.then(()=>(this.querySelector(`.wren-panel .wren-input, .wren-panel .btn-primary, .wren-close`)??null)?.focus()))}close(){this.open&&(this.open=!1,this.peek=!1,this.lockScroll(!1),this.opener?.focus?.())}lockScroll(e){document.documentElement.classList.toggle(`wren-sheet-open`,e&&this.sheetMode())}sheetPeek(){this.open&&this.sheetMode()&&(this.peek=!0,setTimeout(()=>this.peek=!1,1800))}statusWord(){return this.phase===`finished`?M.wrenFinished:this.phase===`asleep`?M.wrenAsleep:this.phase===`refused`?M.wrenCannotLoad:this.phase===`failed`?M.wrenFailed:this.phase===`loading`?M.wrenWaking(this.progress):M.wrenStatus[this.figure]??``}render(){let e=this.figure;return y`
      <button class="wren-fab ${this.open?`is-open`:``}" aria-label=${this.open?M.wrenClose:M.wrenOpen} aria-expanded=${this.open} aria-controls="wren-panel" @click=${()=>this.toggleOpen()} data-figure=${e}>
        <qse-wren-figure class="wren-fab-fig" .state=${e} .size=${50} .progress=${this.progress}></qse-wren-figure>
      </button>
      <div class="wren-scrim ${this.open?`is-open`:``}" @click=${()=>this.close()}></div>
      <aside id="wren-panel" class="wren-panel ${this.open?`is-open`:``} ${this.peek?`is-peek`:``}" aria-label=${M.wrenName} data-phase=${this.phase} data-figure=${e} role="complementary">
        <div class="wren-handle" aria-hidden="true"></div>
        <header class="wren-head">
          <qse-wren-figure class="wren-main" .state=${e} .size=${76} .progress=${this.progress}></qse-wren-figure>
          <div class="wren-titles">
            <h2 class="wren-name">${M.wrenName}</h2>
            <p class="wren-status" role="status" aria-live="polite" data-wren-status>${this.statusWord()}</p>
            ${this.status?.model&&this.phase===`ready`?y`<p class="wren-model">${this.status.model}${this.status.fallback?y` <span title=${M.wrenFallback}>(${M.wrenFallback})</span>`:S}</p>`:S}
          </div>
          <div class="wren-head-tools">
            ${this.phase===`ready`&&!this.streaming?y`<button class="btn btn-ghost btn-icon" title=${M.wrenSleep} aria-label=${M.wrenSleep} @click=${()=>void this.sleep()}>${D(`moon`)}</button>`:S}
            <button class="btn btn-ghost btn-icon wren-close" aria-label=${M.wrenClose} title=${M.wrenClose} @click=${()=>this.close()}>${D(`x`)}</button>
          </div>
        </header>
        ${this.renderBody()}
      </aside>
    `}hero(){return y`<qse-wren-figure class="wren-hero" .state=${this.figure} .size=${148} .progress=${this.progress}></qse-wren-figure>`}renderBody(){switch(this.phase){case`checking`:return y`<div class="wren-body wren-center">${this.hero()}</div>`;case`finished`:return y`<div class="wren-body wren-center" data-wren-finished>${this.hero()}<p>${M.wrenFinished}</p></div>${this.renderComposer(!0)}`;case`asleep`:return y`<div class="wren-body wren-center">
            ${this.hero()}
            <p class="wren-lead">${M.wrenAsleep}</p>
            <p class="muted small">${M.wrenAsleepBody}</p>
            <button class="btn btn-primary wren-wake" @click=${()=>void this.wake()}>${M.wrenWake}</button>
          </div>`;case`loading`:return y`<div class="wren-body wren-center">
            ${this.hero()}
            <p class="wren-lead">${M.wrenWaking(this.progress)}</p>
            <div class="wren-loadbar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${this.progress==null?S:Math.round(this.progress*100)} aria-label=${M.wrenWaking(null)}><span style="width:${Math.round((this.progress??0)*100)}%"></span></div>
          </div>`;case`refused`:{let e=this.status,t=e?.model==null&&e!=null?M.wrenNoModel:M.wrenCannotLoadMemory(e?.claim_gib??null,this.host?.memAvailable()??null);return y`<div class="wren-body wren-center" data-wren-refused>
            ${this.hero()}
            <p class="wren-lead">${M.wrenCannotLoad}</p>
            <p class="muted small">${e?.reason?tt(e.reason):t}</p>
            ${e?.reason?y`<p class="muted small">${t}</p>`:S}
            ${e?.model==null?S:y`<button class="btn wren-wake" @click=${()=>void this.wake()}>${D(`refresh`)} ${M.wrenRetry}</button>`}
          </div>`}case`failed`:return y`<div class="wren-body wren-center" data-wren-failed>
            ${this.hero()}
            <p class="wren-lead">${M.wrenFailed}</p>
            <p class="muted small">${this.error??``}</p>
            <button class="btn wren-wake" @click=${()=>void this.wake()}>${D(`refresh`)} ${M.wrenRetry}</button>
          </div>`;case`ready`:return this.renderChat()}}renderChat(){let e=this.chat;return y`
      <div class="wren-body wren-log" aria-live="off">
        ${e.length===0?y`<p class="wren-msg is-assistant wren-hello">${M.wrenHello}</p>`:S}
        ${ae(e.map((e,t)=>({m:e,i:t})),({i:e})=>e,({m:t,i:n})=>this.renderMessage(t,n===e.length-1))}
        ${this.partial?y`<div class="wren-msg is-assistant is-streaming">${se(ce(this.partial))}</div>`:S}
        ${this.streaming&&!this.partial?y`<div class="wren-msg is-assistant wren-typing" aria-hidden="true"><span></span><span></span><span></span></div>`:S}
        ${this.notice?y`<p class="wren-note">${this.notice}</p>`:S}
        ${this.error?y`<div class="wren-error" role="alert" data-wren-error>
              <span>${this.error}</span>
              <button class="btn btn-sm" @click=${()=>void this.retry()}>${D(`refresh`)} ${M.wrenRetry}</button>
            </div>`:S}
      </div>
      ${this.renderChips()} ${this.renderComposer(!1)}
    `}renderMessage(e,t){if(e.role===`user`)return y`<div class="wren-msg is-user">${e.content}</div>`;if(e.role===`assistant`)return e.content.trim()?y`<div class="wren-msg is-assistant">${se(ce(e.content))}
        ${t&&this.sources.length?y`<p class="wren-sources">${M.wrenSources} ${this.sources.map((e,t)=>y`${t?`, `:``}<span title=${e.path}>${e.title||e.path}</span>`)}</p>`:S}
      </div>`:S;let n={};try{n=JSON.parse(e.content)}catch{}let r=e.tool_call_id?this.chips.get(e.tool_call_id):void 0;if(n.ok===!1||r?.ok===!1)return y`<p class="wren-note" data-tool-rejected>${n.stale||r?.stale?M.tool.stale:M.wrenCorrected}</p>`;let i=r?.summary??n.summary;return i?y`<div class="wren-chip ${r?.undone?`is-undone`:``}" data-tool-chip>
      <span class="wren-chip-dot" aria-hidden="true"></span>
      <span>${i}</span>
      ${r?.undo?y`<button class="btn btn-ghost btn-sm" ?disabled=${r.undone} @click=${()=>this.undoChip(e.tool_call_id)}>${r.undone?M.wrenUndone:M.wrenUndo}</button>`:S}
    </div>`:S}renderChips(){if(this.streaming)return S;let e=M.wrenChips[this.step]??[];return y`<div class="wren-suggest">${e.map(e=>y`<button class="wren-suggestion" @click=${()=>void this.send(e)}>${e}</button>`)}</div>`}renderComposer(e){return y`<form
      class="wren-composer"
      @submit=${e=>{e.preventDefault(),this.send(this.composer)}}
    >
      <label class="sr-only" for="wren-input">${M.wrenPlaceholder}</label>
      <textarea
        id="wren-input"
        class="wren-input"
        rows="1"
        .value=${this.composer}
        ?disabled=${e}
        placeholder=${e?M.wrenFinished:M.wrenPlaceholder}
        @focus=${()=>this.emit(`focus`)}
        @blur=${()=>!this.composer&&this.emit(`blur`)}
        @input=${e=>{this.composer=e.target.value,this.emit(`input`)}}
        @keydown=${e=>{e.key===`Enter`&&!e.shiftKey&&!e.isComposing&&(e.preventDefault(),this.send(this.composer))}}
      ></textarea>
      ${this.streaming?y`<button type="button" class="btn wren-stop" @click=${()=>this.stop()} aria-label=${M.wrenStop}>${D(`stop`)}<span>${M.wrenStop}</span></button>`:y`<button type="submit" class="btn btn-primary wren-send" ?disabled=${e||!this.composer.trim()} aria-label=${M.wrenSend}>${D(`send`)}</button>`}
    </form>`}};E([k({attribute:!1})],Q.prototype,`host`,void 0),E([k({attribute:!1})],Q.prototype,`step`,void 0),E([k({attribute:!1})],Q.prototype,`chat`,void 0),E([k({type:Boolean})],Q.prototype,`finished`,void 0),E([x()],Q.prototype,`phase`,void 0),E([x()],Q.prototype,`status`,void 0),E([x()],Q.prototype,`figure`,void 0),E([x()],Q.prototype,`progress`,void 0),E([x()],Q.prototype,`streaming`,void 0),E([x()],Q.prototype,`partial`,void 0),E([x()],Q.prototype,`composer`,void 0),E([x()],Q.prototype,`error`,void 0),E([x()],Q.prototype,`notice`,void 0),E([x()],Q.prototype,`sources`,void 0),E([x()],Q.prototype,`open`,void 0),E([x()],Q.prototype,`peek`,void 0),E([x()],Q.prototype,`chips`,void 0),E([n(`qse-wren-figure.wren-main`)],Q.prototype,`fig`,void 0),E([n(`qse-wren-figure.wren-fab-fig`)],Q.prototype,`fabFig`,void 0),E([n(`.wren-log`)],Q.prototype,`log`,void 0),E([n(`.wren-input`)],Q.prototype,`input`,void 0),Q=E([A(`qse-wren`)],Q);var nt=[8192,32768,65536,131072,262144],rt=e=>e>=1024?`${Math.round(e/1024)}k`:String(e),$=class extends T{constructor(...e){super(...e),this.path=[],this.wiz=m(),this.storageProblem=null,this.storageFailed=!1,this.list={state:`loading`},this.model=null,this.device=null,this.serverEst=null,this.estLocal=!1,this.plan=null,this.confirmOpen=!1,this.applying=!1,this.applyError=null,this.jobInitial=null,this.previousProfile=null,this.resetOpen=!1,this.unauthorized=!1,this.winNarrow=!1,this.store=null,this.devPoll=null,this.estTimer=null,this.estCtrl=null,this.planTimer=null,this.planCtrl=null,this.warnedDiff=!1,this.touchedAt=new Map,this.loadingModel=null,this.onUnauthorized=()=>{this.unauthorized=!0},this.onSignedIn=()=>{this.unauthorized=!1,this.list.state!==`ready`&&this.loadList();let e=this.path[0];e&&this.model?.state!==`ready`&&this.loadModel(e),this.loadDevice(),this.wiz.selection&&this.queueEstimate(),this.wiz.step===`review`&&this.queuePlan()},this.onReset=()=>{this.wiz=m(),this.plan=null,this.jobInitial=null,location.hash!==`#/setup`&&(location.hash=`#/setup`)},this.host={pageState:()=>we(this.m,this.wiz.step,this.wiz.selection,this.currentEstimate(),{errors:this.visibleErrors()}),execute:(e,t)=>this.executeTool(e,t),saveChat:e=>this.dispatch({type:`chat`,chat:e}),memAvailable:()=>this.device?.mem_available_gb??null}}connectedCallback(){super.connectedCallback(),this.store=t();let e=c(this.store);this.wiz=e.state,this.storageProblem=e.problem,(e.problem===`corrupt`||e.problem===`version`)&&d(this.store),this.loadList(),this.devPoll=ie(()=>this.loadDevice(),5e3),this.devPoll.start(),window.addEventListener(`qse-setup-reset`,this.onReset),window.addEventListener(`qse-setup-unauthorized`,this.onUnauthorized),window.addEventListener(`qse-signed-in`,this.onSignedIn)}disconnectedCallback(){this.devPoll?.stop(),this.estCtrl?.abort(),this.planCtrl?.abort(),window.removeEventListener(`qse-setup-reset`,this.onReset),window.removeEventListener(`qse-setup-unauthorized`,this.onUnauthorized),window.removeEventListener(`qse-signed-in`,this.onSignedIn),super.disconnectedCallback()}async loadList(){this.list={state:`loading`};try{let e=await _.models();this.list=e.models.length||e.errors.length?{state:`ready`,data:e}:{state:`empty`,note:M.modelsEmpty}}catch(e){let t=w(e);this.list={state:`error`,endpoint:t.endpoint,message:t.status===401?M.unauthorized:t.message}}}async loadModel(e){if(this.loadingModel!==e){this.loadingModel=e,this.model={state:`loading`};try{let t=await _.model(e);this.model={state:`ready`,data:t},await this.openModel(t,this.routeStep()??void 0)}catch(t){let n=w(t);this.model={state:`error`,endpoint:n.endpoint,message:n.status===404?`${M.modelsError} (${e})`:n.message}}finally{this.loadingModel=null}}}async loadDevice(){try{this.device=await _.device()}catch{}}async openModel(e,t){let n=null;if(!(this.wiz.model===e.id&&this.wiz.selection))try{n=(await _.current(e.id)).selection}catch{n=null}this.dispatch({type:`open`,model:e,step:t,seed:n},!1)}get m(){return this.model?.state===`ready`?this.model.data:null}routeStep(){let e=this.path[1];return i(e)?e:null}willUpdate(e){e.has(`path`)&&this.syncFromRoute()}syncFromRoute(){let[e,t]=this.path;if(!e){this.wiz.step!==`models`&&(this.wiz={...this.wiz,step:`models`});return}let n=this.m;if(!n||n.id!==e){this.loadModel(e);return}if(this.wiz.model!==e||!this.wiz.selection){this.openModel(n,i(t)?t:void 0);return}let r=i(t)?t:`overview`;r===`install`&&!this.wiz.job&&(r=`review`),r===`models`&&(r=`overview`),r!==this.wiz.step&&this.dispatch({type:`go`,step:r},!1)}hashFor(e){return e.step===`models`||!e.model?`#/setup`:`#/setup/${e.model}/${e.step}`}dispatch(e,t=!0){let n=f(this.wiz,e,this.m);if(n===this.wiz)return;let r=n.step!==this.wiz.step;this.wiz=n,this.store&&(this.storageFailed=!h(this.store,n));let i=this.hashFor(n);location.hash!==i&&(t?location.hash=i:history.replaceState(null,``,i)),r&&(t&&window.scrollTo({top:0,behavior:W()?`auto`:`smooth`}),n.step===`review`&&this.queuePlan()),e.type!==`go`&&e.type!==`next`&&e.type!==`previous`&&e.type!==`chat`&&e.type!==`dismiss`&&(this.queueEstimate(),n.step===`review`&&this.queuePlan())}user(e,t,n){this.touchedAt.set(qe(e,t),Date.now()),this.dispatch(n)}selKey(e){return JSON.stringify(e)}queueEstimate(){this.estTimer&&clearTimeout(this.estTimer),this.estTimer=setTimeout(()=>void this.confirmEstimate(),150)}async confirmEstimate(){let e=this.wiz.selection;if(!e)return;this.estCtrl?.abort();let t=this.estCtrl=new AbortController,n=this.selKey(e);try{let r=await _.estimate(e,t.signal);if(t.signal.aborted||n!==this.selKey(this.wiz.selection))return;this.serverEst={key:n,est:r},this.estLocal=!1;let i=this.m?H(this.m,e):null;i&&Math.abs(i.total_gb-r.total_gb)>.05&&!this.warnedDiff&&(this.warnedDiff=!0,console.warn(`setup: the server's estimate (${r.total_gb} GiB) and the page's (${i.total_gb} GiB) differ; showing the server's`))}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;this.estLocal=!0}}currentEstimate(){let e=this.m,t=this.wiz.selection;if(!e||!t)return null;let n=H(e,t,this.device?.mem_available_gb??null),r=this.serverEst;return r&&r.key===this.selKey(t)&&(!n||Math.abs(r.est.total_gb-n.total_gb)>.05)?{...r.est,available_gb:this.device?.mem_available_gb??r.est.available_gb}:n}queuePlan(){this.planTimer&&clearTimeout(this.planTimer),this.planTimer=setTimeout(()=>void this.loadPlan(),200)}async loadPlan(){let e=this.wiz.selection;if(!e)return;this.planCtrl?.abort();let t=this.planCtrl=new AbortController;(!this.plan||this.plan.state!==`ready`)&&(this.plan={state:`loading`});try{let n=await _.plan(e,t.signal);t.signal.aborted||(this.plan={state:`ready`,data:n})}catch(e){if(e.name===`AbortError`||t.signal.aborted)return;let n=w(e);this.plan={state:`error`,endpoint:n.endpoint,message:n.message}}}visibleErrors(){let e=this.issues()[this.wiz.step]??[];return this.wiz.step!==`review`||this.plan?.state!==`ready`?e:[...e,...this.plan.data.errors.filter(t=>!e.includes(t))]}async executeTool(e,t){let n=Qe(e,{model:this.m,selection:this.wiz.selection,step:this.wiz.step,touchedAt:this.touchedAt,turnStartedAt:t});if(!n.ok)return{ok:!1,result:n.error,summary:null,undo:null,stale:n.stale};if(n.kind===`read_device`)try{let e=await _.device();this.device=e;let t={mode:e.mode,mem_total_gb:e.mem_total_gb,mem_available_gb:e.mem_available_gb,gpu:e.gpu,disk:e.disk,engine:e.engine,active_profile:e.active_profile,profiles_on_disk:e.profiles_on_disk,password_set:e.password_set};return{ok:!0,result:JSON.stringify(t),summary:null,undo:null}}catch(e){return{ok:!1,result:w(e).message,summary:null,undo:null}}if(n.kind===`explain`)return n.target.step===this.wiz.step&&this.highlight(n.target.step,n.target.field),{ok:!0,result:n.result,summary:null,undo:null};let{target:r}=n;r.field&&r.step!==this.wiz.step&&this.dispatch({type:`go`,step:r.step});for(let e of n.actions)this.dispatch(e);await this.updateComplete,r.field&&this.highlight(r.step,r.field);let i=n.undo.length?()=>{for(let e of n.undo)this.dispatch(e);r.field&&this.highlight(r.step,r.field)}:null;return{ok:!0,result:n.result,summary:n.actions.length?n.summary:null,undo:i}}async highlight(e,t){if(!t)return;await this.updateComplete;let n=this.querySelector(`[data-field="${CSS.escape(qe(e,t))}"]`);if(!n)return;let r=n.getBoundingClientRect();(r.top<120||r.bottom>innerHeight-80)&&n.scrollIntoView({block:`center`,behavior:W()?`auto`:`smooth`}),n.setAttribute(`data-wren-acting`,``),n.querySelector(`:scope > .wren-mini`)?.remove();let i=document.createElement(`span`);i.className=`wren-mini`,i.setAttribute(`aria-hidden`,`true`),n.append(i),ne(ze,i),setTimeout(()=>{n.removeAttribute(`data-wren-acting`),i.remove()},1600)}issues(){let e=this.m,t=this.wiz.selection;return!e||!t?{}:ve(e,t,this.device,this.currentEstimate())}finishBlockers(){let e=this.m,t=this.wiz.selection;if(!e||!t)return[];let n=[],r=this.currentEstimate();r&&!r.fits&&n.push(M.reviewOver(V(r))),U(e,t,this.device)&&n.push(M.blockedToken);let i=Object.keys(this.issues()).filter(e=>e!==`review`);return i.length&&n.push(M.blockedInvalid(i.map(e=>j[e].toLowerCase()))),this.plan?.state===`ready`&&!this.plan.data.ok&&!n.length&&n.push(...this.plan.data.errors),n}async finish(){let e=this.wiz.selection;if(!e||this.applying||this.finishBlockers().length)return;this.applying=!0,this.applyError=null,this.previousProfile=this.device?.active_profile??null,this.querySelector(`qse-wren`)?.sleepForInstall();let t=e;try{let e=await _.apply({selection:t});this.jobInitial=e.job,this.confirmOpen=!1,this.dispatch({type:`job`,id:e.job.id})}catch(e){this.applyError=this.applyErrorText(e)}finally{this.applying=!1}}applyErrorText(e){let t=w(e);if(t.status===401)return null;let n=e instanceof C?e.body:void 0;return n?.plan?(this.plan={state:`ready`,data:n.plan},this.confirmOpen=!1,n.plan.errors.join(` `)||t.message):e instanceof C&&e.type===`job_running`?M.blockedBusy:e instanceof C&&e.type===`busy`?M.blockedRequests(n?.error?.running??1):`${M.installError} ${t.message}`}render(){let e=this.wiz.step,t=this.m,n=this.path.length>0;return y`<div class="view view-setup" data-step=${n?e:`models`}>
      <div class="setup">
        <div class="setup-main">
          ${this.renderBanners()}
          ${this.unauthorized?y`<qse-setup-signin></qse-setup-signin>`:n?this.renderModelArea():this.renderTiles()}
          <footer class="setup-foot">
            <button class="btn btn-ghost btn-sm" aria-expanded=${this.resetOpen} aria-controls="setup-reset" @click=${()=>this.resetOpen=!this.resetOpen}>${D(`trash`)} ${M.resetLink}</button>
          </footer>
          ${this.resetOpen?y`<section class="panel setup-reset" id="setup-reset"><header class="panel-head"><h2 class="panel-title">${M.resetHeading}</h2></header><qse-reset inSetup></qse-reset></section>`:S}
        </div>
        <qse-wren class="setup-wren" .host=${this.host} .step=${n?e:`models`} .chat=${this.wiz.chat} .finished=${e===`install`&&!!this.wiz.job}></qse-wren>
      </div>
      ${t?this.renderConfirm(t):S}
    </div>`}renderBanners(){let e=this.storageProblem,t=[];return(e===`unavailable`||this.storageFailed)&&t.push(M.storageOff),e===`corrupt`&&t.push(M.storageCorrupt),e===`version`&&t.push(M.storageOld),this.wiz.notice&&this.wiz.step!==`components`&&this.wiz.step!==`parallelism`&&t.push(this.wiz.notice),y`${t.map(e=>y`<div class="banner setup-banner" role="status" data-setup-notice>${D(`info`)} <span>${e}</span></div>`)}`}renderTiles(){let e=this.list,t=this.device,n=this.wiz.model&&this.wiz.selection?this.wiz:null;return y`<section class="setup-tiles">
      <header class="setup-hero">
        <h1 class="setup-h1">${j.models}</h1>
        <p class="setup-lead">${re.models}</p>
        ${t?.mem_total_gb==null?S:y`<p class="setup-device muted num" data-device-line>${M.deviceLine(t.mem_total_gb,t.mem_available_gb)}</p>`}
      </header>
      ${e.state===`loading`?O(4):S}
      ${e.state===`error`?te(e.endpoint,`${M.modelsError} ${e.message}`,()=>void this.loadList()):S}
      ${e.state===`empty`?ee(M.modelsEmpty):S}
      ${e.state===`ready`?y`<div class="mtiles">
              ${e.data.models.map(e=>{let t=n?.model===e.id,r=t?`#/setup/${e.id}/${n.step===`models`?`overview`:n.step}`:`#/setup/${e.id}/overview`;return y`<a class="mtile ${e.recommended?`is-rec`:``}" href=${r} data-model=${e.id}>
                  <span class="mtile-top"><span class="mtile-family">${e.family}</span>${e.recommended?y`<span class="sbadge sbadge-rec">${M.tileRecommended}</span>`:S}</span>
                  <span class="mtile-name">${e.name}</span>
                  <span class="mtile-params">${M.tileParams(e.params)}</span>
                  <span class="mtile-pitch">${e.pitch}</span>
                  ${e.best_speed?y`<span class="mtile-speed"><span class="mtile-speed-n num">${e.best_speed.tok_s.toFixed(1)}</span><span class="mtile-speed-u">tok/s</span><span class="mtile-speed-note">${e.best_speed.task}, ${e.best_speed.profile}</span></span>`:S}
                  <span class="mtile-go"><span class="btn btn-primary">${t?`${M.tileOpen}: ${N[n.step]}`:M.tileOpen}</span></span>
                </a>`})}
            </div>
            ${e.data.errors.length?y`<div class="setup-load-errors" role="status"><p>${M.modelLoadErrors(e.data.errors.length)}</p><ul>${e.data.errors.map(e=>y`<li><code>${e.path}</code> <span class="muted">${e.message}</span></li>`)}</ul></div>`:S}`:S}
      <section class="guide" aria-labelledby="guide-h">
        <h2 class="guide-h" id="guide-h">${M.guideSteps}</h2>
        <ol class="guide-list">${l.slice(1).map(e=>y`<li><span class="guide-t">${N[e]}</span><span class="guide-d">${oe[e]}</span></li>`)}</ol>
      </section>
    </section>`}renderModelArea(){let e=this.model;if(!e||e.state===`loading`)return O(8);if(e.state===`error`)return te(e.endpoint,e.message,()=>void this.loadModel(this.path[0]));if(e.state!==`ready`||!this.wiz.selection)return O(8);let t=e.data,n=this.wiz.selection,r=this.currentEstimate(),i=this.wiz.step,a=r&&!r.fits?xe(t,n,r):null;return y`
      <div class="setup-sticky">
        <qse-memory-bar .model=${t} .estimate=${r} .available=${this.device?.mem_available_gb??null} .local=${this.estLocal} .overText=${a}></qse-memory-bar>
        ${this.estLocal?y`<p class="setup-local" role="status" data-local-estimate>${M.memLocal}</p>`:S}
      </div>
      ${this.renderRail(t)}
      <section class="setup-step" data-step=${i} aria-labelledby="setup-step-title">
        <header class="step-head">
          <p class="step-kicker muted num">${M.stepOf(p(i)+1,l.length)}</p>
          <h1 class="step-title" id="setup-step-title" tabindex="-1">${j[i]}</h1>
          <p class="step-intro">${re[i]}</p>
        </header>
        ${this.renderStep(t,n,r)}
      </section>
      ${this.renderNav()}
    `}renderRail(e){let t=this.wiz.step,n=this.issues(),r=p(t);return y`<nav class="rail-steps" aria-label="Setup steps">
      <div class="rail-phone" aria-hidden="true">
        <span class="num">${M.stepOf(r+1,l.length)}</span><span class="rail-phone-label">${j[t]}</span>
        <span class="rail-phone-bar"><span style="width:${(r+1)/l.length*100}%"></span></span>
      </div>
      <ol class="steps">
        ${l.map((i,a)=>{let o=this.wiz.visited.includes(i),s=o&&(n[i]?.length??0)>0,c=i===t,l=o&&!s&&a<r,u=i!==`install`||!!this.wiz.job,d=i===`models`?`#/setup`:`#/setup/${e.id}/${i}`;return y`<li class="steps-item ${c?`is-current`:``} ${l?`is-done`:``} ${s?`is-bad`:``} ${o?`is-visited`:``}" data-rail=${i}>
            ${u?y`<a href=${d} aria-current=${c?`step`:S} title=${s?n[i].join(` `):j[i]}>
                  <span class="steps-node num" aria-hidden="true">${s?`!`:l?D(`check`):a+1}</span>
                  <span class="steps-label">${N[i]}</span>
                  ${s?y`<span class="sr-only">(needs a fix)</span>`:S}
                </a>`:y`<span class="steps-link is-locked" aria-disabled="true"><span class="steps-node num" aria-hidden="true">${a+1}</span><span class="steps-label">${N[i]}</span></span>`}
          </li>`})}
      </ol>
    </nav>`}renderNav(){let e=this.wiz.step;if(e===`install`)return S;let t=p(e);return y`<nav class="setup-nav" aria-label="Step navigation">
      <button class="btn setup-prev" @click=${()=>t<=1?this.dispatch({type:`tiles`}):this.dispatch({type:`previous`})}>${M.previous}</button>
      <span class="setup-nav-mid muted small num">${M.stepOf(t+1,l.length)}</span>
      ${e===`review`?y`<button class="btn btn-primary setup-finish" ?disabled=${this.finishBlockers().length>0||this.plan?.state!==`ready`} @click=${()=>this.confirmOpen=!0}>${M.finish}</button>`:y`<button class="btn btn-primary setup-next" @click=${()=>this.dispatch({type:`next`})}><span>${M.next}<span class="setup-next-to">: ${N[l[t+1]]}</span></span></button>`}
    </nav>`}renderStep(e,t,n){switch(this.wiz.step){case`overview`:return this.stepOverview(e,t);case`weights`:return this.stepWeights(e,t);case`components`:return this.stepComponents(e,t);case`parallelism`:return this.stepParallel(e,t,n);case`methods`:return this.stepMethods(e,t);case`review`:return this.stepReview(e,t,n);case`install`:return this.stepInstall();default:return y``}}stepOverview(e,t){let n=e.parallelism.window.max;return y`<div class="ov">
      <div class="ov-facts">
        <p class="ov-arch">${e.architecture}</p>
        <dl class="ov-dl">
          <div><dt>${M.colWeights}</dt><dd>${e.profiles.length} sizes, ${e.profiles.map(e=>e.label.replace(/ bpw$/,``)).join(`, `)}</dd></div>
          <div><dt>${M.window}</dt><dd class="num">${n.toLocaleString(`en-US`)} ${M.tokensUnit}</dd></div>
          <div><dt>${M.license}</dt><dd>${e.license}</dd></div>
        </dl>
        <ul class="ov-links">${e.links.map(e=>y`<li><a href=${e.url} target="_blank" rel="noopener noreferrer">${e.label} ${D(`external`)}</a></li>`)}</ul>
      </div>
      <section class="panel ov-table" aria-label="Measured speed">
        <qse-measure-table .model=${e} .selected=${t.profile}></qse-measure-table>
      </section>
    </div>`}stepWeights(e,t){let n=this.device,r=U(e,t,n),i=e.measurements.rows,a=Math.max(...i.filter(e=>e.comparable).map(e=>e.cells.greedy_512?.code?.tok_s??0));return y`
      ${r?y`<div class="banner banner-offline" role="alert" data-token-blocked>${D(`warn`)} ${M.privateNoToken}</div>`:S}
      ${!r&&n?.hf_token_present&&e.profiles.some(e=>e.hf_private&&!n.profiles_on_disk?.[e.id])?y`<p class="wnote muted small">${D(`info`)} ${M.privateRepo}</p>`:S}
      <div class="wcards" role="radiogroup" aria-label=${j.weights}>
        ${e.profiles.map(e=>{let r=t.profile===e.id,o=i.find(t=>t.profile===e.id),s=o?.cells.greedy_512,c=n?.profiles_on_disk?.[e.id],l=n?.active_profile===e.id,u=e.quality.top1_pct;return y`<label class="wcard ${r?`is-selected`:``} ${e.recommended?`is-rec`:``}" data-profile-card=${e.id} data-field=${r?`weights.profile`:S}>
            <input class="sr-only" type="radio" name="profile" .checked=${r} value=${e.id} @change=${()=>this.user(`weights`,`profile`,{type:`profile`,id:e.id})} />
            <span class="wcard-head">
              <span class="wcard-radio" aria-hidden="true"></span>
              <span class="wcard-name">${e.label}</span>
              ${l?y`<span class="sbadge sbadge-now">${M.activeNow}</span>`:S}
            </span>
            ${e.badges.length?y`<span class="wcard-badges">${e.badges.map(e=>y`<span class="sbadge ${e===`recommended`?`sbadge-rec`:``}">${e===`recommended`?M.recommended:e}</span>`)}</span>`:S}
            <span class="wcard-meters">
              <span class="wmeter" title=${e.quality.source}>
                <span class="wmeter-k">${M.weightsTop1}</span>
                <span class="wmeter-v num">${u==null?`–`:`${u.toFixed(1)} %`}</span>
                <span class="wmeter-bar q"><span style="width:${u==null?0:Math.max(4,(u-80)/20*100)}%"></span></span>
              </span>
              <span class="wmeter">
                <span class="wmeter-k">${M.weightsSpeed}</span>
                <span class="wmeter-v num">${s?.prose&&s?.code?y`${s.prose.tok_s.toFixed(1)} / ${s.code.tok_s.toFixed(1)}`:y`<span class="muted">${o&&!o.comparable?M.notComparable:M.notMeasured}</span>`}</span>
                <span class="wmeter-bar s"><span style="width:${s?.code&&o?.comparable?s.code.tok_s/a*100:0}%"></span></span>
              </span>
            </span>
            <span class="wcard-facts num">
              <span><span class="wcard-k">${M.weightsKl}</span> ${e.quality.kl_vs_bf16==null?`–`:e.quality.kl_vs_bf16.toFixed(4)}</span>
              <span><span class="wcard-k">${M.weightsMemory}</span> ${e.weights_gib.toFixed(1)} GiB</span>
              <span><span class="wcard-k">${M.weightsDisk}</span> ${c?M.onDisk:M.toDownload(e.disk_gb)}</span>
            </span>
            ${e.note?y`<span class="wcard-note">${e.note}</span>`:S}
            ${e.hf_private&&!c&&n?.hf_token_present===!1?y`<span class="wcard-private is-blocked" data-private>${D(`warn`)} ${M.privateNoTokenShort}</span>`:S}
          </label>`})}
      </div>`}stepComponents(e,t){let n=s(e,t.components),r=[...new Set(e.components.map(e=>e.group))],i=L(e,t.profile);return y`
      <div class="step-tools">
        <button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`components`})}>${D(`refresh`)} ${M.resetToRecommended}</button>
      </div>
      <p class="comp-notice" role="status" aria-live="polite" data-cascade>${this.wiz.notice??``}</p>
      ${r.map(r=>y`<section class="comp-group" aria-labelledby=${`cg-${r}`}>
          <h2 class="comp-group-title" id=${`cg-${r}`}>${M.groups[r]??r}</h2>
          <ul class="comp-list">
            ${e.components.filter(e=>e.group===r).map(e=>{let r=o(e,t.components),a=n[e.id],s=e.id===`drafters`&&i.drafters_gib!=null?i.drafters_gib:e.memory_gib;return y`<li>
                  <label class="comp ${r?`is-on`:``} ${a.disabled?`is-disabled`:``} ${a.required?`is-required`:``}" data-field=${`components.${e.id}`} data-component=${e.id}>
                    <input type="checkbox" class="comp-input" .checked=${r} ?disabled=${a.disabled} aria-describedby=${`cw-${e.id}${a.reason?` cr-${e.id}`:``}`} @change=${t=>this.user(`components`,e.id,{type:`component`,id:e.id,on:t.target.checked})} />
                    <span class="comp-box" aria-hidden="true">${D(`check`)}</span>
                    <span class="comp-main">
                      <span class="comp-label">${e.label}${e.default&&!e.required?y` <span class="comp-rec">${M.recommendedValue}</span>`:S}</span>
                      <span class="comp-what" id=${`cw-${e.id}`}>${e.what}</span>
                      ${a.reason?y`<span class="comp-reason" id=${`cr-${e.id}`} data-reason>${a.reason}</span>`:S}
                    </span>
                    <span class="comp-cost num">
                      <span class="comp-mem">${M.memoryCost(s)}</span>
                      ${e.memory_max_gib==null?S:y`<span class="muted">${M.memoryUpTo(e.memory_max_gib)}</span>`}
                      ${e.disk_gb>0?y`<span class="muted">${M.diskCost(e.disk_gb)}</span>`:S}
                    </span>
                  </label>
                </li>`})}
          </ul>
        </section>`)}`}stepParallel(e,t,n){let r=e.parallelism,i=t.side_window??t.window,a=n?.context_tokens_fit??0,o=Math.floor(a/t.slots),s=(e,t)=>this.user(`parallelism`,e,{type:`parallel`,key:e,value:t}),c=(e,t,n)=>y`<div class="win-ctl">
      <div class="seg seg-wrap" role="radiogroup" aria-label=${e===`window`?M.window:M.sideWindow}>
        ${nt.filter(t=>t>=r[e].min&&t<=n).map(n=>y`<button class="seg-btn ${t===n?`is-on`:``}" role="radio" aria-checked=${t===n} @click=${()=>s(e,n)}>${rt(n)}</button>`)}
      </div>
      <input class="num-input num" type="number" inputmode="numeric" min=${r[e].min} max=${n} step=${r[e].step} .value=${String(t)} aria-label=${`${e===`window`?M.window:M.sideWindow} (${M.tokensUnit})`} @change=${t=>s(e,Number(t.target.value))} />
      <span class="muted small">${M.tokensUnit}</span>
    </div>`;return y`<div class="spar">
      <div class="spar-fields">
        <div class="pfield" data-field="parallelism.slots">
          <div class="pfield-k"><span class="pfield-label" id="pf-slots">${M.slots}</span><span class="pfield-what">${M.slotsWhat}</span></div>
          <div class="seg" role="radiogroup" aria-labelledby="pf-slots">
            ${Array.from({length:r.slots.max-r.slots.min+1},(e,t)=>r.slots.min+t).map(e=>y`<button class="seg-btn ${t.slots===e?`is-on`:``}" role="radio" aria-checked=${t.slots===e} @click=${()=>s(`slots`,e)}>${e}${e===r.slots.default?y`<span class="seg-rec" title=${M.recommended}>•</span>`:S}</button>`)}
          </div>
        </div>
        <div class="pfield" data-field="parallelism.window">
          <div class="pfield-k"><span class="pfield-label">${M.window}</span><span class="pfield-what">${M.windowWhat}</span></div>
          ${c(`window`,t.window,r.window.max)}
        </div>
        ${t.slots>1?y`<div class="pfield" data-field="parallelism.side_window">
              <div class="pfield-k"><span class="pfield-label">${M.sideWindow}</span><span class="pfield-what">${M.sideWindowWhat}</span></div>
              ${c(`side_window`,i,t.window)}
            </div>`:S}
        <div class="pfield" data-field="parallelism.kv_pool">
          <div class="pfield-k"><span class="pfield-label" id="pf-pool">${M.kvPool}</span><span class="pfield-what">${M.kvPoolWhat}</span></div>
          <label class="switch"><input type="checkbox" role="switch" .checked=${t.kv_pool} aria-labelledby="pf-pool" @change=${e=>s(`kv_pool`,e.target.checked)} /><span>${t.kv_pool?`on`:`off`}</span></label>
        </div>
        <div class="pfield" data-field="parallelism.floor_gb">
          <div class="pfield-k"><span class="pfield-label" id="pf-floor">${M.floor}</span><span class="pfield-what">${M.floorWhat}</span></div>
          <div class="floor-ctl">
            <input type="range" min=${r.floor_gb.min} max=${r.floor_gb.max} step="1" .value=${String(t.floor_gb)} aria-labelledby="pf-floor" @input=${e=>s(`floor_gb`,Number(e.target.value))} />
            <output class="num">${t.floor_gb} GiB</output>
          </div>
        </div>
        <div class="step-tools"><button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`parallelism`})}>${D(`refresh`)} ${M.resetToRecommended}</button></div>
        ${this.wiz.notice?y`<p class="comp-notice" role="status">${this.wiz.notice}</p>`:S}
      </div>
      <aside class="spar-readout panel" aria-live="polite" data-fit-tokens=${a}>
        <p class="spar-big num">${a.toLocaleString(`en-US`)}</p>
        <p class="spar-big-u">${M.fitsUnit}</p>
        <div class="spar-slots" aria-hidden="true">
          ${Array.from({length:t.slots},(e,n)=>{let a=n===0?t.window:i;return y`<div class="spar-slot">
              <span class="spar-slot-n num">${n+1}</span>
              <span class="spar-slot-track"><span class="spar-slot-win" style="width:${a/r.window.max*100}%"></span><span class="spar-slot-fit" style="width:${Math.min(a,o)/r.window.max*100}%"></span></span>
              <span class="spar-slot-k num">${rt(a)}</span>
            </div>`})}
        </div>
        <p class="small">${M.fitsPerSlot(o,t.slots)}</p>
        ${n?y`<p class="small muted">${M.worstCase(n.kv_worst_gb,n.fits_worst??!1)}</p>`:S}
      </aside>
    </div>`}fieldControl(e,t,n){let r=n=>this.user(`methods`,t.id,{type:`field`,group:e,id:t.id,value:n}),i=`mf-${t.id}`;if(t.type===`enum`){let e=t.options??[];return e.length<=3?y`<div class="seg seg-wrap" role="radiogroup" aria-labelledby=${`${i}-l`}>${e.map(e=>y`<button class="seg-btn ${n===e.value?`is-on`:``}" role="radio" aria-checked=${n===e.value} @click=${()=>r(e.value)}>${e.label}</button>`)}</div>`:y`<select id=${i} class="sel" @change=${e=>r(e.target.value)}>${e.map(e=>y`<option value=${e.value} ?selected=${n===e.value}>${e.label}</option>`)}</select>`}return t.type===`bool`?y`<label class="switch"><input id=${i} type="checkbox" role="switch" .checked=${n===!0} @change=${e=>r(e.target.checked)} /><span>${n?`on`:`off`}</span></label>`:t.type===`int`||t.type===`number`?y`<input id=${i} class="num-input num" type="number" inputmode=${t.type===`int`?`numeric`:`decimal`} min=${t.min??S} max=${t.max??S} step=${t.type===`int`?1:.05} .value=${String(n)} @change=${e=>{let t=e.target.value;r(t===``?NaN:Number(t))}} />`:y`<input id=${i} class="text-input" type="text" .value=${String(n)} spellcheck="false" autocapitalize="off" @change=${e=>r(e.target.value)} />`}fieldRow(e,t,n){let r=n[e][t.id]??t.recommended,i=g(t,r),a=t.options?.find(e=>e.value===t.recommended)?.label??String(t.recommended);return y`<div class="mfield ${i?`is-bad`:``}" data-field=${`methods.${t.id}`}>
      <div class="pfield-k">
        <label class="pfield-label" id=${`mf-${t.id}-l`} for=${`mf-${t.id}`}>${t.label}</label>
        <span class="pfield-what">${t.what}</span>
        <span class="mfield-rec muted small">${M.recommendedValue}: <span class="num">${a}</span>${r===t.recommended?S:y` <button class="linkish" @click=${()=>this.user(`methods`,t.id,{type:`field`,group:e,id:t.id,value:t.recommended})}>${M.resetToRecommended.toLowerCase()}</button>`}</span>
      </div>
      <div class="mfield-ctl">${this.fieldControl(e,t,r)}${i?y`<p class="mfield-err" role="alert">${M.invalidValue(i)}</p>`:S}</div>
    </div>`}stepMethods(e,t){let n=e.advanced.some(e=>g(e,t.advanced[e.id]??e.recommended));return y`<div class="methods">
      ${e.methods.map(e=>this.fieldRow(`methods`,e,t))}
      <details class="adv" ?open=${n}>
        <summary><span class="adv-title">${M.advanced}</span> <span class="muted small">${M.advancedNote}</span></summary>
        ${e.advanced.map(e=>this.fieldRow(`advanced`,e,t))}
      </details>
      <div class="step-tools"><button class="btn btn-sm" @click=${()=>this.dispatch({type:`recommended`,scope:`methods`})}>${D(`refresh`)} ${M.resetToRecommended}</button></div>
    </div>`}stepReview(e,t,n){let r=this.plan,i=this.wiz.step,a=n&&!n.fits?be(e,t):[],o=this.finishBlockers(),s=this.device,c=L(e,t.profile),l=r?.state===`ready`?r.data:null;return y`<div class="review">
      <section class="rv-card rv-memory ${n&&!n.fits?`is-over`:`is-ok`}" data-review-memory>
        <h2 class="rv-h">${M.reviewMemory}</h2>
        ${n?y`<p class="rv-say">${n.fits?M.reviewFits(n.total_gb,n.budget_gb??n.device_gb-n.floor_gb):xe(e,t,n)}</p>`:S}
        ${a.length?y`<h3 class="rv-sub">${M.howToFix}</h3>
              <ul class="rv-fixes">${a.map(t=>y`<li><span>${t.text}</span><span class="muted small num">−${t.saves.toFixed(1)} GiB</span><a class="btn btn-sm" href=${`#/setup/${e.id}/${t.step}`}>${M.fixGo}: ${N[t.step]}</a></li>`)}</ul>`:S}
      </section>

      <section class="rv-card">
        <h2 class="rv-h">${M.reviewWeights}</h2>
        <p>${s?.active_profile===c.id?M.reviewWeightsSame(c.label):M.reviewWeightsSwitch(s?.active_profile?e.profiles.find(e=>e.id===s.active_profile)?.label??s.active_profile:null,c.label)}</p>
        ${l&&!l.profile.on_disk?y`<p class="muted">${M.reviewDownload(l.profile.download_gb)}</p>`:S}
        ${l?y`<p class="muted small">${l.restart?M.reviewRestart:M.reviewNoRestart}</p>`:S}
      </section>

      <section class="rv-card rv-changes">
        <h2 class="rv-h">${M.reviewChanges}</h2>
        ${r==null||r.state===`loading`?O(4):S}
        ${r?.state===`error`?y`<p class="rv-warn" role="alert">${D(`warn`)} ${M.planError} <span class="muted">${r.message}</span> <button class="btn btn-sm" @click=${()=>void this.loadPlan()}>${M.retry}</button></p>`:S}
        ${l?l.changes.length?y`<div class="table-scroll"><table class="table rv-table">
                <thead><tr><th scope="col">${M.reviewKey}</th><th scope="col">${M.reviewNow}</th><th scope="col">${M.reviewAfter}</th><th scope="col">${M.reviewFrom}</th></tr></thead>
                <tbody>${l.changes.map(e=>y`<tr data-env=${e.key}><th scope="row"><code>${e.key}</code></th><td class="rv-old">${e.current??y`<span class="muted">${M.reviewUnset}</span>`}</td><td class="rv-new">${e.next??y`<span class="muted">${M.reviewUnset}</span>`}</td><td class="muted small">${e.from}</td></tr>`)}</tbody>
              </table></div>`:y`<p class="muted">${M.reviewNoChanges}</p>`:S}
        ${l&&(l.errors.length||l.warnings.length)?y`<ul class="rv-notes">
              ${l.errors.map((t,n)=>{let r=l.error_codes?.[n],a=ge(r);return y`<li class="is-err" data-error-code=${r??S}>${D(`warn`)} <span>${t}</span>${a&&a!==i?y` <a class="btn btn-sm" href=${`#/setup/${e.id}/${a}`}>${M.planFix}: ${N[a]}</a>`:S}</li>`})}
              ${l.warnings.map(e=>y`<li>${D(`info`)} ${e}</li>`)}
            </ul>`:S}
      </section>

      ${l?.commands.length?y`<section class="rv-card">
            <h2 class="rv-h">${M.reviewCommands}</h2>
            <p class="muted small">${M.reviewCommandsNote}</p>
            <pre class="rv-cmds">${l.commands.join(`
`)}</pre>
          </section>`:S}

      ${s&&!s.password_set?this.securityCard():s?.password_set?y`<p class="rv-secure muted">${D(`check`)} ${M.securitySet}</p>`:S}

      ${o.length?y`<ul class="rv-blockers" role="status" data-finish-blocked>${o.map(e=>y`<li>${D(`warn`)} ${e}</li>`)}</ul>`:S}
      ${this.applyError?y`<p class="rv-warn" role="alert">${D(`warn`)} ${this.applyError}</p>`:S}
    </div>`}get sec(){return this.wiz.selection?.advanced.admin_password===`generate`?`generate`:`keep`}loginOff(){let e=this.m?.advanced.find(e=>e.id===`dashboard_login`);return!!e&&(this.wiz.selection?.advanced.dashboard_login??e.recommended)!==`on`}securityCard(){let e=(e,t,n)=>y`<label class="sec-opt ${this.sec===e?`is-on`:``}">
      <input type="radio" name="sec" .checked=${this.sec===e} @change=${()=>this.user(`methods`,`admin_password`,{type:`field`,group:`advanced`,id:`admin_password`,value:e})} />
      <span><span class="sec-opt-label">${t}</span>${n?y`<span class="muted small">${n}</span>`:S}</span>
    </label>`;return y`<section class="rv-card rv-security" data-security-warning role="note">
      <h2 class="rv-h">${D(`warn`)} ${M.securityTitle}</h2>
      <p>${M.securityBody}</p>
      <div class="sec-opts" role="radiogroup" aria-label=${M.securityChoice}>
        ${e(`keep`,M.securityKeep)} ${e(`generate`,M.securityGenerate,M.securityGenerateNote)}
      </div>
      ${this.sec===`generate`&&this.loginOff()?y`<p class="sec-login" data-login-off>${D(`info`)} <span>${M.securityLoginOff}</span>
            <button class="btn btn-sm" @click=${()=>this.user(`methods`,`dashboard_login`,{type:`field`,group:`advanced`,id:`dashboard_login`,value:`on`})}>${M.securityLoginOn}</button></p>`:S}
    </section>`}renderConfirm(e){let t=this.plan?.state!==`ready`||this.plan.data.restart;return y`<dialog
      class="setup-confirm"
      aria-labelledby="sc-t"
      aria-describedby="sc-b"
      @cancel=${e=>this.applying&&e.preventDefault()}
      @close=${()=>this.confirmOpen=!1}
    >
      ${this.confirmOpen?y`<h2 id="sc-t">${M.confirmTitle}</h2>
            <p id="sc-b">${M.confirmBody(t)}</p>
            <p class="muted small">${L(e,this.wiz.selection.profile).label}</p>
            ${this.applyError?y`<p class="rv-warn" role="alert">${this.applyError}</p>`:S}
            <div class="setup-confirm-foot">
              <button class="btn" ?disabled=${this.applying} @click=${()=>this.confirmOpen=!1}>${M.cancel}</button>
              <button class="btn btn-primary setup-confirm-go" autofocus ?disabled=${this.applying} @click=${()=>void this.finish()}>${this.applying?M.installStarting:M.confirmGo}</button>
            </div>`:S}
    </dialog>`}updated(){let e=this.querySelector(`dialog.setup-confirm`);e&&(this.confirmOpen&&!e.open?e.showModal():!this.confirmOpen&&e.open&&e.close())}stepInstall(){let e=this.wiz.job;return e?y`<qse-install-progress .jobId=${e} .initial=${this.jobInitial} .previousProfile=${this.previousProfile} .onBack=${()=>this.dispatch({type:`job`,id:null})}></qse-install-progress>`:y`<p class="muted">${M.installStarting}</p>`}};E([k({attribute:!1})],$.prototype,`path`,void 0),E([x()],$.prototype,`wiz`,void 0),E([x()],$.prototype,`storageProblem`,void 0),E([x()],$.prototype,`storageFailed`,void 0),E([x()],$.prototype,`list`,void 0),E([x()],$.prototype,`model`,void 0),E([x()],$.prototype,`device`,void 0),E([x()],$.prototype,`serverEst`,void 0),E([x()],$.prototype,`estLocal`,void 0),E([x()],$.prototype,`plan`,void 0),E([x()],$.prototype,`confirmOpen`,void 0),E([x()],$.prototype,`applying`,void 0),E([x()],$.prototype,`applyError`,void 0),E([x()],$.prototype,`jobInitial`,void 0),E([x()],$.prototype,`previousProfile`,void 0),E([x()],$.prototype,`resetOpen`,void 0),E([x()],$.prototype,`unauthorized`,void 0),E([x()],$.prototype,`winNarrow`,void 0),$=E([A(`qse-setup`)],$);export{$ as QseSetup};