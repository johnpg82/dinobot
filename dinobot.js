// Chrome Dino × Laya — self-improving bot (console drop-in)
// Paste into the DevTools console on chrome://dino (or Chrome's "No internet" page), click the game, press ` to toggle.
// Decisions: Laya (open-source Jev-compatible System-1 model, Apache-2.0) proposes a move for each obstacle — its answers are
// baked in because the dino page blocks web workers and storage (🧠 Live runs it in-browser via onnxruntime-web on normal pages) —
// times a physics check of what the game allows, refined by a contextual bandit that learns from every obstacle and crash,
// a self-tuning jump timer, and by watching you play (👀 Train). The brain persists in this tab (window.name) or localStorage.
{
    const WORKER_SRC = "const L={choice:0,score:1,noul:2},V=[\"choice\",\"score\",\"noul\"],X=.5,W=5,$=t=>Math.round(t*1e4)/1e4,D=t=>Number.isFinite(+t)?Math.min(W,Math.max(X,+t)):1;function j(t){return t==null?\"null\":typeof t==\"string\"?JSON.stringify(t):typeof t==\"number\"?Number.isInteger(t)?String(t):JSON.stringify(t):typeof t==\"boolean\"?t?\"true\":\"false\":Array.isArray(t)?\"[\"+t.map(j).join(\", \")+\"]\":\"{\"+Object.entries(t).map(([e,n])=>`${JSON.stringify(e)}: ${j(n)}`).join(\", \")+\"}\"}const A=t=>typeof t==\"string\"?t:j(t);function Z(t){let e=t.criteria;return t.type===\"choice\"&&Array.isArray(e)&&(e=Object.fromEntries(e.map(n=>[n,null]))),{t:t.type,ins:typeof t.instructions==\"string\"?t.instructions:j(t.instructions),crit:e}}function K(t){if(t.t===\"choice\")return Object.entries(t.crit).map(([n,o])=>o==null||o===\"\"?String(n):`${n}: ${A(o)}`);if(t.t===\"score\")return t.crit.map((n,o)=>`level ${o}: ${A(n)}`);const e=t.crit||{};return[\"false: \"+(e.false?A(e.false):\"no, the statement does not hold\"),\"true: \"+(e.true?A(e.true):\"yes, the statement holds\")]}const v=t=>t<=2?\"2\":t<=5?\"3-5\":t<=10?\"6-10\":\"11+\",tt=(t,e)=>`${V[t]}:${v(e)}`;function et(t){const e=Math.max(...t),n=t.map(s=>Math.exp(s-e)),o=n.reduce((s,r)=>s+r,0);return n.map(s=>s/o)}function nt(t){const e=t.length;if(e<2)return 1;let n=0;for(const o of t)n-=o*Math.log(Math.max(o,1e-12));return Math.min(1,Math.max(0,1-n/Math.log(e)))}function st(t,e,n,o,s,r){const h=c=>c.split(e.maskTok).join(\" \"),f=K(o);let g=t(`${o.t} question: ${h(o.ins)}`),a=f.map(c=>[e.mask,...t(\" \"+h(c)).slice(0,48)]);const l=c=>c.reduce((w,I)=>w+I.length,0);let u=r-l(a);if(u<16){const c=Math.max(4,Math.floor((r-16)/Math.max(1,a.length)));a=a.map(w=>w.slice(0,c)),u=r-l(a)}g=g.slice(0,Math.max(8,u));const i=[e.cls,...g,e.sep],k=[];for(const c of a)k.push(i.length),i.push(...c);i.push(e.sep);const y=Math.max(0,s-i.length-1);return i.push(...n.slice(0,y),e.sep),{ids:i.slice(0,s),markers:k.filter(c=>c<s)}}class N{constructor(e,n,o,s){this.ort=e,this.session=n,this.tok=o,this.config=s;const r=h=>{const f=o.token_to_id(h);if(f===void 0)throw new Error(\"missing token \"+h);return f};this.ids={cls:r(\"[CLS]\"),sep:r(\"[SEP]\"),mask:r(\"[MASK]\"),pad:r(\"[PAD]\"),maskTok:\"[MASK]\"},this.temperature=(s.temperature||[1,1,1]).map(D),this.tByOpt=Object.fromEntries(Object.entries(s.temperature_by_options||{}).map(([h,f])=>[h,D(f)])),this._qcache=new Map}static async create({ort:e,Tokenizer:n,model:o,tokenizerJson:s,tokenizerConfig:r,config:h,sessionOptions:f={}}){const g=await e.InferenceSession.create(o,{graphOptimizationLevel:\"all\",...f}),a=new n(s,r);return new N(e,g,a,h)}encode=e=>this.tok.encode(e,{add_special_tokens:!1}).ids;_enc(e){let n=this._qcache.get(e);return n||(n=this.encode(e),this._qcache.size>2e3&&this._qcache.clear(),this._qcache.set(e,n)),n}async systemOne(e,n,{padTo:o=this.padTo}={}){const s=Object.keys(n);if(!s.length)throw new Error(\"systemOne: at least one question is required\");const{max_len:r=512,head_max_len:h=192}=this.config,f=(typeof e==\"string\"?e:j(e)).split(this.ids.maskTok).join(\" \"),g=this.encode(f),a=s.map(p=>{const m=Z(n[p]),{ids:x,markers:d}=st(q=>this._enc(q),this.ids,g,m,r,h);if(d.length!==K(m).length)throw new Error(`question ${p}: options do not fit`);return{q:m,ids:x,markers:d,qtype:L[m.t]}}),l=a.length,u=Math.max(o||0,...a.map(p=>p.ids.length)),i=Math.max(this.padOptions||0,...a.map(p=>p.markers.length)),k=new BigInt64Array(l*u).fill(BigInt(this.ids.pad)),y=new BigInt64Array(l*u),c=new BigInt64Array(l*i),w=new Uint8Array(l*i),I=new BigInt64Array(l);let J=0;a.forEach((p,m)=>{p.ids.forEach((x,d)=>{k[m*u+d]=BigInt(x),y[m*u+d]=1n}),J+=p.ids.length,p.markers.forEach((x,d)=>{c[m*i+d]=BigInt(x),w[m*i+d]=1}),I[m]=BigInt(p.qtype)});const T=this.ort.Tensor,F=(globalThis.performance||Date).now(),P=await this.session.run({input_ids:new T(\"int64\",k,[l,u]),attention_mask:new T(\"int64\",y,[l,u]),marker_pos:new T(\"int64\",c,[l,i]),marker_mask:new T(\"bool\",w,[l,i]),qtype:new T(\"int64\",I,[l])}),H=(globalThis.performance||Date).now()-F,Q=P.logits.data,Y=P.act_probs.data,G=P.act_probs.dims[1]||1,S={};return a.forEach((p,m)=>{const x=p.markers.length,d=p.q,q=this.tByOpt[tt(p.qtype,x)]??this.temperature[p.qtype]??1,b=et(Array.from(Q.subarray(m*i,m*i+x),_=>_/q)),C={confidence:$(nt(b)),answer_confidence:$(Math.max(...b)),action:{act_probability:$(Y[m*G]??0)}};if(d.t===\"choice\"){const _=Object.keys(d.crit);S[s[m]]={type:\"choice\",choice:_[b.indexOf(Math.max(...b))],probabilities:Object.fromEntries(_.map((E,R)=>[E,$(b[R])])),...C}}else d.t===\"score\"?S[s[m]]={type:\"score\",score:$(b.reduce((_,E,R)=>_+R*E,0)),legend:Object.fromEntries(d.crit.map((_,E)=>[String(E),A(_)])),probabilities:Object.fromEntries(b.map((_,E)=>[String(E),$(_)])),...C}:S[s[m]]={type:\"noul\",noul:$(b[1]),...C,confidence:$(Math.max(b[1],1-b[1]))}}),{model:\"laya-web\",answers:S,usage:{input_tokens:J,output_tokens:0,ms:Math.round(H)}}}}const U=\"1.30.0\",ot=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${U}/dist/ort.webgpu.bundle.min.mjs`,rt=\"https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs\",it=\"laya-onnx-v1\";let M=null,z=null;const O=(t,e={})=>self.postMessage({type:t,...e});async function B(t,e,n=!1){const o=self.caches?await caches.open(it).catch(()=>null):null;let s=o?await o.match(t):void 0;const r=!!s;if(!s){if(s=await fetch(t),!s.ok)throw new Error(`${e}: HTTP ${s.status}`);const h=+s.headers.get(\"content-length\")||0,f=s.body.getReader(),g=[];let a=0,l=-1;for(;;){const{done:i,value:k}=await f.read();if(i)break;g.push(k),a+=k.length;const y=h?Math.floor(a/h*100):-1;y!==l&&y%5===0&&(l=y,O(\"progress\",{label:e,pct:y,mb:Math.round(a/1e6)}))}const u=new Blob(g);try{o&&await o.put(t,new Response(u))}catch(i){O(\"log\",{msg:`cache put failed for ${e}: ${i.message}`})}s=new Response(u)}return O(\"log\",{msg:`${e}: ${r?\"from browser cache\":\"downloaded\"}`}),n?s.json():new Uint8Array(await s.arrayBuffer())}async function at({repo:t,prefer:e,padTo:n=0,padOptions:o=0,sessionExtra:s={}}){const r=`https://huggingface.co/${t}/resolve/main/`,[h,f]=await Promise.all([import(ot),import(rt)]);h.env.wasm.wasmPaths=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${U}/dist/`,h.env.wasm.numThreads=Math.max(1,Math.min(4,(navigator.hardwareConcurrency||4)>>1));const[g,a,l,u,i]=await Promise.all([B(r+\"laya_config.json\",\"config\",!0),B(r+\"tokenizer/tokenizer.json\",\"tokenizer\",!0),B(r+\"tokenizer/tokenizer_config.json\",\"tokenizer config\",!0),B(r+\"laya.onnx\",\"model graph\"),B(r+\"laya.onnx.data\",\"model weights (~1.7 GB, one time)\")]),k=e===\"wasm\"?[\"wasm\"]:self.navigator.gpu?[\"webgpu\",\"wasm\"]:[\"wasm\"];let y;for(const c of k)try{const w=performance.now();M=await N.create({ort:h,Tokenizer:f.Tokenizer,model:u,tokenizerJson:a,tokenizerConfig:l,config:g,sessionOptions:{executionProviders:[c],externalData:[{path:\"laya.onnx.data\",data:i}],...s}}),M.padTo=n,M.padOptions=o,n&&(M.config={...M.config,max_len:Math.min(M.config.max_len||512,n)}),z=c,O(\"log\",{msg:`session ready on ${c} in ${Math.round(performance.now()-w)} ms`});break}catch(w){y=w,O(\"log\",{msg:`${c} failed: ${w.message}`})}if(!M)throw y;return{backend:z}}self.onmessage=async({data:t})=>{try{if(t.type===\"init\")O(\"ready\",await at(t));else if(t.type===\"bench\"){const e=[];for(let n=0;n<(t.n||5);n++){const o=await M.systemOne(t.state,t.questions);e.push(o.usage.ms)}O(\"bench\",{id:t.id,times:e,backend:z})}else if(t.type===\"decide\"){const e=await M.systemOne(t.state,t.questions);O(\"decision\",{id:t.id,result:e,backend:z})}}catch(e){O(\"error\",{id:t.id,msg:String(e&&e.message||e)})}};\n";
    const LAYA_BAKED = {"model":"receptron/laya-onnx","prior":{"small1":[0.3294,0.261,0.1297,0.28],"small2":[0.3101,0.2757,0.1498,0.2644],"small3":[0.3216,0.2713,0.1522,0.2549],"large1":[0.2978,0.2641,0.1495,0.2885],"large2":[0.2969,0.2803,0.1667,0.2561],"large3":[0.2531,0.3287,0.1778,0.2404],"birdLow":[0.3348,0.2173,0.1641,0.2838],"birdMid":[0.0219,0.049,0.8923,0.0368],"birdHigh":[0.3576,0.2217,0.1569,0.2638]}}; // Laya's answer per obstacle kind: [jump, hop, duck, run]
    // ---- Dino × Laya bot: main-thread part (WORKER_SRC and LAYA_BAKED are injected above at build time) ----
    (() => {
      'use strict';
      if (window.__dinoBot) { try { window.__dinoBot.uninstall(); } catch (e) {} }
    
      // ---- situations: shared by the bot and the offline bake script (keep in sync: build.mjs inlines this file) ----
      const SIT = (() => {
        const KINDS = ['small1', 'small2', 'small3', 'large1', 'large2', 'large3', 'birdLow', 'birdMid', 'birdHigh'];
        const SPEEDS = ['slow', 'medium', 'fast'];
        const FOLLOW = ['none', 'close'];
        const WHAT = {
          small1: 'A small cactus is coming. It is short and sits on the ground, so the T-Rex cannot go under it.',
          small2: 'Two small cacti side by side are coming. They are short and sit on the ground, so the T-Rex cannot go under them.',
          small3: 'Three small cacti side by side are coming. They are short but wide and sit on the ground, so the T-Rex cannot go under them.',
          large1: 'A tall cactus is coming. It is as tall as the T-Rex and sits on the ground, so the T-Rex cannot go under it.',
          large2: 'Two tall cacti side by side are coming. They are as tall as the T-Rex and sit on the ground, so the T-Rex cannot go under them.',
          large3: 'Three tall cacti side by side are coming. They are as tall as the T-Rex and very wide, and they sit on the ground, so the T-Rex cannot go under them.',
          birdLow: 'A pterodactyl is coming, flying low near the ground, so the T-Rex cannot go under it.',
          birdMid: 'A pterodactyl is coming at head height. A crouching T-Rex fits under it; a standing T-Rex does not.',
          birdHigh: 'A pterodactyl is coming, flying high. The standing T-Rex fits under it.',
        };
        const PACE = { slow: 'The game is still slow.', medium: 'The game is getting faster.', fast: 'The game is very fast.' };
        const NEXT = { none: 'Nothing else is close behind it.', close: 'Another obstacle comes right after it.' };
        const key = (kind, speed, follow) => `${kind}|${speed}|${follow}`;
        const parse = k => { const [kind, speed, follow] = k.split('|'); return { kind, speed, follow }; };
        const text = kind => WHAT[kind]; // what Laya is asked about (pace and follow-ups made its answers worse, so they're context for the learner only)
        const describe = (kind, speed, follow) => `${WHAT[kind]} ${PACE[speed]} ${NEXT[follow]}`;
        const allKeys = () => {
          const out = [];
          for (const kind of KINDS) for (const speed of SPEEDS) for (const follow of FOLLOW) {
            if (kind.startsWith('bird') && speed === 'slow') continue; // pterodactyls only appear from speed 8.5
            out.push(key(kind, speed, follow));
          }
          return out;
        };
        return { KINDS, SPEEDS, FOLLOW, key, parse, text, describe, allKeys };
      })();
      // ---- the one question Laya answers for every situation (Jev system_one "choice" question) ----
      const ACTIONS = {
        jump: 'jump high over it',
        hop: 'hop a little over it',
        duck: 'crouch under it',
        run: 'keep running, it does not touch the T-Rex',
      };
      const ANAMES = Object.keys(ACTIONS);
      const QUESTIONS = {
        move: {
          type: 'choice',
          instructions: 'You are playing the Chrome dinosaur game. The T-Rex runs forward by itself and the run ends if it touches an obstacle. Which move gets it past the next obstacle?',
          criteria: ACTIONS,
        },
      };
    
      // ================= config =================
      const CFG = {
        toggleKey: 'Backquote',     // ` toggles Play/Off, Shift+` toggles Train
        yieldToHumanMs: 1500,       // in Play mode, pressing Space/↑/↓ hands you control for this long (it learns from you meanwhile)
        autoStart: true,            // in Play mode it starts, resumes and restarts the game by itself
        restartDelayMs: 1500,       // pause after a crash before restarting (the game ignores restarts for 1.2 s)
        fastDrop: true,             // once it's past an obstacle, press ↓ to land sooner (the pro move for close obstacles)
        decideAtMs: 520,            // pick a move when the obstacle is this many ms away (late enough to see what follows it)
        epsilon: 0.03,              // chance of trying another plausible move
        priorHalfLife: 4,           // encounters of a situation until Laya's prior counts half as much as experience
        demoHalfLife: 12,           // own encounters until your demonstrations count half as much
        timingStep: 30,             // ms the trigger moves after a crash that was clearly too early / too late
        timingRate: 0.3,            // share of the measured timing error corrected after each clean jump
        closeFollowMs: 450,         // another obstacle arriving within this many ms after this one counts as "right after it"
        speedBuckets: [8, 10.5],    // slow < 8 ≤ medium < 10.5 ≤ fast (pterodactyls appear from speed 8.5)
        liveLaya: false,            // true = also run the 1.7 GB Laya model live in a worker (normal https pages only; chrome://dino blocks workers)
        repo: 'receptron/laya-onnx',
      };
      const STORE_KEY = 'dinoLaya.brain.v1';
      // Chrome's offline/dino page (chrome-error://) kills the tab if it starts a download, so there the brain goes to the clipboard.
      const NO_DOWNLOADS = location.protocol === 'chrome-error:' || self.origin === 'null';
      const TAU0 = { jump: 262, hop: 200, duck: 320 };          // ms before the obstacle's centre meets the T-Rex's centre (game physics, 60–240 Hz)
      const TAU_LIMITS = { jump: [150, 420], hop: [130, 300], duck: [130, 700] };
      // System 2: which moves the game's collision geometry allows at all (simulated offline with the game's jump physics
      // and collision boxes). Multiplied into Laya's System-1 prior so it never starts out ducking into a cactus.
      const FEASIBLE = {
        small1: [1, 1, 0.02, 0.02], small2: [1, 1, 0.02, 0.02], small3: [1, 0.9, 0.02, 0.02],
        large1: [1, 0.7, 0.02, 0.02], large2: [1, 0.5, 0.02, 0.02], large3: [1, 0.05, 0.02, 0.02],
        birdLow: [1, 1, 0.02, 0.02], birdMid: [0.8, 0.3, 1, 0.02], birdHigh: [0.02, 0.02, 1, 1],
      };
      const KIND_LABEL = { small1: 'small cactus', small2: '2 small cacti', small3: '3 small cacti', large1: 'tall cactus', large2: '2 tall cacti',
        large3: '3 tall cacti', birdLow: 'low bird', birdMid: 'head-height bird', birdHigh: 'high bird' };
    
      // ================= logging =================
      const t0 = performance.now();
      const LOG = { on: true, decisions: false, moves: true, timing: false, crashes: true, trace: false };
      const stamp = () => ((performance.now() - t0) / 1000).toFixed(1).padStart(6);
      const log = (msg, ...rest) => LOG.on && console.log(`%c[DinoBot ${stamp()}s]%c ${msg}`, 'color:#2b8a3e;font-weight:bold', 'color:inherit', ...rest);
      const warn = (msg, ...rest) => console.warn(`[DinoBot ${stamp()}s] ${msg}`, ...rest);
    
      // ================= helpers =================
      const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
      const sum = a => a.reduce((x, y) => x + y, 0);
      const num = (...v) => { for (const x of v) if (typeof x === 'number' && Number.isFinite(x)) return x; return 0; };
      const zeros = () => ANAMES.map(() => 0);
      const MI = Object.fromEntries(ANAMES.map((n, i) => [n, i]));
      const argmax = a => a.indexOf(Math.max(...a));
      const speedBucket = s => (s < CFG.speedBuckets[0] ? 'slow' : s < CFG.speedBuckets[1] ? 'medium' : 'fast');
    
      // ================= finding the game =================
      // Current Chrome builds the game as a TypeScript bundle: Runner isn't on window, but it is a top-level class
      // (Runner.getInstance()). Older builds and copies of the game use Runner.instance_. As a last resort the DevTools
      // console's getEventListeners() finds it (the game listens to keydown with itself as the handler), and failing
      // that we catch it the next time it schedules a frame (it calls this.update.bind(this)).
      let R = null;
      const isRunner = r => !!(r && typeof r === 'object' && r.tRex && r.horizon && 'crashed' in r);
      function findRunner() {
        const tries = [
          () => window.Runner && window.Runner.instance_,
          () => typeof Runner !== 'undefined' && (Runner.instance_ || (typeof Runner.getInstance === 'function' && Runner.getInstance())),
          () => typeof runnerInstance !== 'undefined' && runnerInstance,
          () => { const L = typeof getEventListeners === 'function' && getEventListeners(document); return L && (L.keydown || []).map(l => l.listener).find(isRunner); },
        ];
        for (const t of tries) { try { const r = t(); if (isRunner(r)) return r; } catch (e) {} }
        return null;
      }
      let origBind = null;
      function hookBind() {
        if (origBind) return;
        origBind = Function.prototype.bind;
        const orig = origBind;
        Function.prototype.bind = function (thisArg, ...args) {
          if (!R && isRunner(thisArg)) { R = thisArg; unhookBind(); setTimeout(onFound, 0); }
          return orig.call(this, thisArg, ...args);
        };
      }
      function unhookBind() { if (origBind) { Function.prototype.bind = origBind; origBind = null; } }
    
      // ================= keyboard (the bot's hands) =================
      const KEYS = { jump: [' ', 'Space', 32], down: ['ArrowDown', 'ArrowDown', 40], enter: ['Enter', 'Enter', 13] };
      function key(type, [k, code, keyCode]) {
        const ev = new KeyboardEvent(type, { key: k, code, keyCode, which: keyCode, bubbles: true, cancelable: true });
        if (ev.keyCode !== keyCode) for (const p of ['keyCode', 'which']) Object.defineProperty(ev, p, { get: () => keyCode });
        document.dispatchEvent(ev);
      }
      const held = { jump: false, down: false };
      function hold(name, on) {
        if (held[name] === on) return;
        held[name] = on;
        key(on ? 'keydown' : 'keyup', KEYS[name]);
      }
      const releaseAll = () => { hold('jump', false); hold('down', false); };
      const tap = k => { key('keydown', k); key('keyup', k); };
    
      // ================= reading the game =================
      const meta = new WeakMap(); // per obstacle: situation, chosen move, what happened
      let frameMs = 1000 / 60;
      function trexState() {
        const t = R.tRex, c = t.config || {};
        const W = num(c.width, c.WIDTH, 44), H = num(c.height, c.HEIGHT, 47);
        return { x: num(t.xPos), y: num(t.yPos), ground: num(t.groundYPos, 93), W, H, jumping: !!t.jumping, ducking: !!t.ducking,
          speedDrop: !!t.speedDrop, vy: num(t.jumpVelocity), reachedMin: !!t.reachedMinHeight, dropV: num(c.dropVelocity, c.DROP_VELOCITY, -5) };
      }
      function kindOf(o) {
        const type = String((o.typeConfig && o.typeConfig.type) || '').toLowerCase().replace(/_/g, '');
        const size = clamp(o.size || 1, 1, 3);
        if (type.includes('small')) return 'small' + size;
        if (type.includes('large')) return 'large' + size;
        if (type.includes('ptero') || type.includes('bird')) return o.yPos >= 90 ? 'birdLow' : o.yPos >= 62 ? 'birdMid' : 'birdHigh';
        return null; // collectables and anything unknown: harmless
      }
      function readScene(now) {
        const t = trexState(), speed = num(R.currentSpeed, 6);
        const list = (R.horizon && R.horizon.obstacles) || [];
        const obs = [];
        for (const o of list) {
          let m = meta.get(o);
          if (!m) { m = { kind: kindOf(o), lastX: o.xPos, lastT: now, v: null, obs: {} }; meta.set(o, m); }
          if (o.xPos !== m.lastX && now > m.lastT) { // measured speed in px/ms (frame rate and floor() rounding included)
            const inst = (m.lastX - o.xPos) / (now - m.lastT);
            if (inst > 0 && inst < 3) m.v = m.v == null ? inst : m.v * 0.8 + inst * 0.2;
            m.lastX = o.xPos; m.lastT = now;
          }
          const w = num(o.width, ((o.typeConfig && o.typeConfig.width) || 0) * (o.size || 1));
          const h = num(o.typeConfig && o.typeConfig.height);
          const v = m.v || Math.max(0.05, Math.floor((speed + num(o.speedOffset)) * 0.06 * frameMs) / frameMs);
          obs.push({ o, m, kind: m.kind, x: o.xPos, y: o.yPos, w, h, v,
            ttc: ((o.xPos + w / 2) - (t.x + t.W / 2)) / v,   // ms until its centre meets the T-Rex's centre
            tti: (o.xPos - (t.x + t.W)) / v });              // ms until its front touches the T-Rex's front
        }
        const ahead = obs.filter(b => b.kind && b.x + b.w > t.x + 2); // not yet passed (back edge still ahead of the T-Rex's back)
        return { t, speed, obs, focus: ahead[0] || null, next: ahead[1] || null };
      }
      const score = () => {
        try { const d = Math.ceil(R.distanceRan || 0); return R.distanceMeter && R.distanceMeter.getActualDistance ? R.distanceMeter.getActualDistance(d) : Math.round(d * 0.025); }
        catch (e) { return 0; }
      };
      function situation(f, next, S) {
        const gap = next ? (next.x - (f.x + f.w)) / f.v : Infinity;
        const follow = gap < CFG.closeFollowMs ? 'close' : 'none';
        const speed = speedBucket(S.speed);
        return { kind: f.kind, speed, follow, key: SIT.key(f.kind, speed, follow), gapMs: gap };
      }
    
      // ================= brain (saved between page loads in this tab) =================
      // chrome://dino denies localStorage/IndexedDB, so there the brain lives in window.name, which survives reloads of the tab.
      const store = (() => {
        try { localStorage.setItem('__dino_t', '1'); localStorage.removeItem('__dino_t');
          return { kind: 'localStorage', get: () => localStorage.getItem(STORE_KEY), set: v => localStorage.setItem(STORE_KEY, v), clear: () => localStorage.removeItem(STORE_KEY) };
        } catch (e) {}
        const TAG = 'dinoLayaBrain:';
        return { kind: 'this tab', get: () => (typeof window.name === 'string' && window.name.startsWith(TAG) ? window.name.slice(TAG.length) : null),
          set: v => { window.name = TAG + v; }, clear: () => { if (String(window.name).startsWith(TAG)) window.name = ''; } };
      })();
      function normalizeBrain(b) {
        if (!b || b.v !== 1) b = { v: 1 };
        b.prior ||= {};             // live Laya answers per obstacle kind (the baked answers are used otherwise)
        b.Q ||= {}; b.N ||= {};     // experience: average outcome of each move in each situation
        b.D ||= {}; b.D2 ||= {};    // what you did: per exact situation / per obstacle kind
        b.T ||= {};                 // learned trigger times: move|kind|speed -> { tau, n }
        b.crashes ||= [];           // recent crash diagnoses
        b.stats = Object.assign({ runs: 0, botRuns: 0, yourRuns: 0, best: 0, bestBot: 0, bestYou: 0, cleared: 0, crashes: 0,
          decisions: 0, demos: 0, scores: [] }, b.stats || {});
        return b;
      }
      function loadBrain() { let b = null; try { b = JSON.parse(store.get() || 'null'); } catch (e) {} return normalizeBrain(b); }
      const brain = loadBrain();
      let lastSave = 0;
      function saveBrain() {
        lastSave = performance.now();
        try { if (brain.crashes.length > 80) brain.crashes.splice(0, brain.crashes.length - 80); store.set(JSON.stringify(brain)); }
        catch (e) { warn('could not save brain: ' + e.message); }
      }
    
      // ================= System 1 (Laya) × System 2 (physics) prior =================
      const uniform = ANAMES.map(() => 1 / ANAMES.length);
      function priorFor(kind) {
        const laya = brain.prior[kind] || (LAYA_BAKED.prior && LAYA_BAKED.prior[kind]) || uniform;
        const f = FEASIBLE[kind] || [1, 1, 1, 1];
        const p = laya.map((x, i) => x * f[i]), z = sum(p) || 1;
        return p.map(x => x / z);
      }
    
      // ================= learner: contextual bandit over the prior + your demonstrations =================
      function demoDist(st) {
        const exact = brain.D[st.key], coarse = brain.D2[st.kind];
        const d = exact && sum(exact) >= 2 ? exact : coarse && sum(coarse) > 0 ? coarse : null;
        if (!d) return null;
        const n = sum(d);
        return { n, p: d.map(c => (c + 0.2) / (n + 0.2 * ANAMES.length)) };
      }
      function choose(st, explore = true) {
        const prior = priorFor(st.kind), demo = demoDist(st);
        const q = brain.Q[st.key] || zeros(), n = brain.N[st.key] || zeros();
        const visits = sum(n);
        const dTrust = demo ? demo.n / (demo.n + 2) : 0;
        const wDemo = dTrust * CFG.demoHalfLife / (CFG.demoHalfLife + visits);                      // your moves count most early on
        const wPrior = CFG.priorHalfLife / (CFG.priorHalfLife + visits) * (1 - 0.7 * dTrust);       // Laya counts until you or experience say otherwise
        const wExp = visits / (visits + CFG.priorHalfLife);
        const scores = ANAMES.map((_, i) => wPrior * 2 * Math.log(prior[i] + 1e-3) + (demo ? wDemo * 3 * Math.log(demo.p[i]) : 0)
          + wExp * 3 * q[i] + 0.2 * Math.sqrt(Math.log(visits + 2) / (n[i] + 1)));
        let i = argmax(scores), why;
        const plausible = ANAMES.map((_, j) => j).filter(j => j !== i && prior[j] >= 0.08 && q[j] > -0.4);
        if (explore && plausible.length && Math.random() < CFG.epsilon) { i = plausible[Math.floor(Math.random() * plausible.length)]; why = 'explore'; }
        else { // credit the part of the score that put this move ahead of the runner-up
          const j = argmax(scores.map((x, k) => (k === i ? -Infinity : x)));
          const part = { laya: wPrior * 2 * (Math.log(prior[i] + 1e-3) - Math.log(prior[j] + 1e-3)),
            you: demo ? wDemo * 3 * (Math.log(demo.p[i]) - Math.log(demo.p[j])) : -Infinity, learned: wExp * 3 * (q[i] - q[j]) };
          why = Object.entries(part).sort((a, b) => b[1] - a[1])[0][0];
        }
        return { move: ANAMES[i], i, why, prior: prior[i], q: q[i], visits };
      }
      function learn(key, i, r, who) {
        const q = brain.Q[key] || (brain.Q[key] = zeros()), n = brain.N[key] || (brain.N[key] = zeros());
        n[i]++; q[i] += Math.max(0.08, 1 / n[i]) * (r - q[i]);
        if (LOG.decisions) log(`  learn ${key} ${ANAMES[i]} r=${r.toFixed(2)} → Q=${q[i].toFixed(2)} (n=${n[i]}, ${who})`);
      }
      function recordDemo(st, move) {
        const i = MI[move];
        (brain.D[st.key] ||= zeros())[i]++;
        (brain.D2[st.kind] ||= zeros())[i]++;
        brain.stats.demos++;
      }
    
      // ================= timing (self-tuning trigger per move, obstacle and speed) =================
      const tauKey = (move, kind, speed) => `${move}|${kind}|${speed}`;
      function tauFor(move, kind, speed) {
        const e = brain.T[tauKey(move, kind, speed)];
        if (e) return clamp(e.tau, TAU_LIMITS[move][0], TAU_LIMITS[move][1]);
        let s = 0, w = 0; // borrow what it learned for the same move at other speeds, then for other obstacles
        for (const [k, v] of Object.entries(brain.T)) {
          const [m, kd] = k.split('|'); if (m !== move) continue;
          const wt = (kd === kind ? 3 : 0.5) * Math.min(v.n, 10); s += v.tau * wt; w += wt;
        }
        return w ? (s + TAU0[move] * 2) / (w + 2) : TAU0[move];
      }
      function nudgeTau(move, kind, speed, target, rate, why) {
        if (!(move in TAU0) || !Number.isFinite(target)) return;
        const k = tauKey(move, kind, speed);
        const e = brain.T[k] || (brain.T[k] = { tau: tauFor(move, kind, speed), n: 0 });
        const before = e.tau;
        e.tau = clamp(e.tau + rate * (target - e.tau), TAU_LIMITS[move][0], TAU_LIMITS[move][1]);
        e.n++;
        if (LOG.timing || (why && Math.abs(e.tau - before) >= 5)) log(`⏱ ${move} timing for ${KIND_LABEL[kind]} (${speed}): ${before.toFixed(0)} → ${e.tau.toFixed(0)} ms${why ? ' — ' + why : ''}`);
      }
      // After a clean jump: compare the clearance under the T-Rex's feet as the obstacle came in and as it left.
      // Lower at the front = took off late, lower at the back = took off early. Shift the trigger to even them out.
      function timingFromTrace(m) {
        const tr = m.trace;
        if (!tr || tr.cf == null || tr.cb == null || m.pressTTC == null) return null;
        const dFrames = clamp((tr.cb - tr.cf) / (Math.abs(tr.vf) + Math.abs(tr.vb) + 0.5), -4, 4);
        return m.pressTTC + dFrames * frameMs;
      }
    
      // ================= Laya live (optional) =================
      let worker = null, layaState = 'baked', layaBackend = null, layaPct = null, inflight = null, reqId = 0;
      const layaQueue = [];
      function startLaya() {
        if (worker) return true;
        try {
          worker = new Worker(URL.createObjectURL(new Blob([WORKER_SRC], { type: 'text/javascript' })), { type: 'module' });
        } catch (e) { worker = null; }
        if (!worker) { warn('this page does not allow web workers, so Laya cannot run live here — using its baked answers'); layaState = 'baked'; return false; }
        layaState = 'loading';
        let lastPct = -1;
        worker.onmessage = ({ data: m }) => {
          if (m.type === 'progress') { layaPct = m.pct; if (m.pct !== lastPct && m.pct % 10 === 0) { lastPct = m.pct; log(`downloading ${m.label}: ${m.pct}% (${m.mb} MB)`); } return; }
          if (m.type === 'log') return log('laya: ' + m.msg);
          if (m.type === 'ready') { layaState = 'live'; layaBackend = m.backend; log(`🧠 Laya live on ${String(m.backend).toUpperCase()} — re-asking it about every obstacle`); for (const k of SIT.KINDS) askLaya(k); return; }
          if (m.type === 'error') { warn('laya error: ' + m.msg); if (inflight && m.id === inflight.id) { inflight = null; pumpLaya(); } if (layaState === 'loading') layaState = 'baked'; return; }
          if (m.type === 'decision' && inflight && m.id === inflight.id) {
            const a = m.result.answers.move;
            brain.prior[inflight.kind] = ANAMES.map(n => a.probabilities[n] ?? 0);
            log(`🧠 laya → ${KIND_LABEL[inflight.kind]}: ${ANAMES.map(n => `${n} ${Math.round((a.probabilities[n] || 0) * 100)}%`).join(', ')} (${m.result.usage.ms} ms)`);
            inflight = null; pumpLaya();
          }
        };
        worker.onerror = e => { e.preventDefault && e.preventDefault(); warn('Laya worker failed (' + (e.message || e.type) + ') — this page probably blocks workers; using the baked answers'); try { worker.terminate(); } catch (err) {} worker = null; layaState = 'baked'; };
        worker.postMessage({ type: 'init', repo: CFG.repo, padTo: 224, padOptions: 8, sessionExtra: { freeDimensionOverrides: { batch: 1, seq: 224, options: 8 } } });
        log('starting Laya worker (first run downloads ~1.7 GB of weights, then the browser caches them)');
        return true;
      }
      function askLaya(kind) { if (!layaQueue.includes(kind)) layaQueue.push(kind); pumpLaya(); }
      function pumpLaya() {
        if (!worker || layaState !== 'live' || inflight || !layaQueue.length) return;
        const kind = layaQueue.shift();
        inflight = { id: 'q' + (++reqId), kind };
        worker.postMessage({ type: 'decide', id: inflight.id, state: SIT.text(kind), questions: QUESTIONS });
      }
    
      // ================= one obstacle at a time =================
      let mode = window.__DINOBOT_START_OFF ? 'off' : 'bot'; // 'bot' plays, 'train' learns from you, 'off' does nothing
      let yieldUntil = 0, rafId = 0, lastNow = 0, lastFind = 0, lastStartTry = 0, crashAt = 0, wasCrashed = false;
      let runInfo = null, prevFocus = null, jumpFor = null, prevT = null, lastDecision = null, lastCrash = null, lastSummary = performance.now();
      let errors = 0;
      const lastMoveByKey = {};
      const humanSeen = new Set();
      const traceBuf = [];
    
      function decide(f, S, who) {
        const st = situation(f, S.next, S);
        f.m.st = st; f.m.who = who; f.m.decidedAt = performance.now();
        if (who !== 'bot') return;
        const c = choose(st);
        f.m.move = c.move; f.m.why = c.why;
        brain.stats.decisions++;
        lastDecision = { kind: f.kind, move: c.move, why: c.why };
        const changed = lastMoveByKey[st.key] !== c.move; lastMoveByKey[st.key] = c.move;
        if (LOG.moves && (changed || c.why === 'explore' || LOG.decisions)) log(`▶ ${KIND_LABEL[f.kind]} (${st.speed}${st.follow === 'close' ? ', another right after' : ''}) → ${c.move} via ${c.why} (prior ${(c.prior * 100).toFixed(0)}%, Q ${c.q.toFixed(2)}, seen ${c.visits}×)`);
      }
    
      function motor(now, S) {
        const { t, focus: f, next } = S;
        let wantJump = false, wantDown = false;
        if (f && f.m.move) {
          const m = f.m, mv = m.move, tau = tauFor(mv === 'run' ? 'jump' : mv, f.kind, m.st.speed);
          if (mv === 'jump' || mv === 'hop') {
            if (!m.pressedAt && f.ttc <= tau) {
              if (!t.jumping && !t.ducking) { wantJump = true; m.pressedAt = now; m.pressTTC = f.ttc; m.pressTau = tau; jumpFor = m; }
              else if (!m.blocked) m.blocked = { ttc: f.ttc, air: t.jumping };
            }
            if (m.pressedAt && jumpFor === m && m.pressedAt !== now) {
              if (t.jumping) wantJump = mv === 'hop' ? !t.reachedMin : t.vy < t.dropV;   // hop: let go as soon as the game allows
              else if (now - m.pressedAt < 90) wantJump = true;                           // the game hasn't started the jump yet
              else if (!m.landedAt) { m.pressedAt = 0; jumpFor = null; }                  // the press didn't take: allow another one
            }
          } else if (mv === 'duck') {
            if (f.ttc <= tau || m.duckAt) { wantDown = true; if (!m.duckAt) { m.duckAt = now; m.duckTTC = f.ttc; } }
          }
        }
        // fast drop: past the obstacle we jumped for and still in the air — land now, unless this jump carries us over the next one too
        if (CFG.fastDrop && t.jumping && !wantJump && jumpFor && jumpFor.cleared && !jumpFor.dropDone) {
          if (!f || f.kind === 'birdHigh' || lookahead(t, f) !== 'over') wantDown = true;
        }
        // don't touch down with ↓ still held: the game turns a speed-drop landing into "ducking while still jumping", which
        // freezes the T-Rex on the ground (for a frame at 60 Hz, up to a second at 240 Hz) so it can't jump the next obstacle
        if (wantDown && t.jumping && (t.ducking || (t.speedDrop && t.y + Math.round((t.vy + 1) * 3 * frameMs / (1000 / 60)) >= t.ground - 24))) {
          wantDown = false;
          if (jumpFor) jumpFor.dropDone = true; // one drop per jump: pressing ↓ again would reset the fall speed
        }
        // ducking blocks jumping: let go of ↓ first, jump on the next frame
        if (wantJump && t.ducking) { wantJump = false; wantDown = false; if (f && f.m.pressedAt === now) f.m.pressedAt = 0; }
        // ↓ held from the air doesn't turn into a duck on landing: re-press it
        if (wantDown && held.down && !t.jumping && !t.ducking && f && f.m.move === 'duck') { hold('down', false); wantDown = false; }
        if (!wantDown) hold('down', false);
        hold('jump', wantJump);
        if (wantDown) hold('down', true);
        if (LOG.trace) {
          traceBuf.push({ t: Math.round(now), y: t.y, vy: +t.vy.toFixed(1), j: t.jumping, d: t.ducking, f: f && f.kind, fx: f && Math.round(f.x - t.x), ttc: f && Math.round(f.ttc),
            mv: f && f.m.move, p: f && !!f.m.pressedAt, b: f && !!f.m.blocked, wj: wantJump, wd: wantDown, jf: jumpFor ? (jumpFor.cleared ? 'cleared' : 'active') : null,
            la: t.jumping && f ? lookahead(t, f) : null, n: next && Math.round(next.x - t.x) });
          if (traceBuf.length > 90) traceBuf.shift();
        }
      }
    
      // System 2 lookahead with the game's own jump physics: from where the T-Rex is now (in the air), does it sail over
      // obstacle b ('over'), land before it arrives ('land'), or come down on it ('hit')?
      function lookahead(t, b, drop = false) {
        const c = R.tRex.config || {}, fe = frameMs / (1000 / 60);
        const g = num(c.gravity, c.GRAVITY, 0.6), coef = num(c.speedDropCoefficient, c.SPEED_DROP_COEFFICIENT, 3);
        let y = t.y, v = drop ? 1 : t.vy, x = b.x;
        const step = b.v * frameMs;
        for (let k = 0; k < 400; k++) {
          y += Math.round(v * (drop ? coef : 1) * fe); v += g * fe;
          x -= step;
          if (y > t.ground) return 'land';
          if (x + b.w < t.x + 2) return 'over';
          if (x + 1 < t.x + 25 && x + b.w - 1 > t.x + 11 && y + 44 >= b.y + 1) return 'hit';
        }
        return 'land';
      }
    
      // what the T-Rex did around each obstacle (used to read your moves and to tune timing)
      function observe(now, S) {
        const { t } = S;
        for (const b of S.obs) {
          if (!b.kind) continue;
          const overlap = t.x + 1 < b.x + b.w - 1 && t.x + t.W - 1 > b.x + 1;
          if (overlap) {
            const ob = b.m.obs;
            if (t.jumping) { ob.air = true; ob.rise = Math.max(ob.rise || 0, t.ground - t.y); }
            else if (t.ducking) ob.duck = true; else ob.ground = true;
          }
        }
        // clearance under the feet of the T-Rex for the obstacle the current jump is for
        if (jumpFor && t.jumping) {
          const b = S.obs.find(x => x.m === jumpFor);
          if (b) {
            const fx0 = t.x + 11, fx1 = t.x + 25, feet = t.y + 44;            // the lowest collision boxes of the running T-Rex
            const tr = jumpFor.trace || (jumpFor.trace = { rise: 0 });
            tr.rise = Math.max(tr.rise, t.ground - t.y);
            const vy = prevT && prevT.jumping ? t.y - prevT.y : 0;
            if (fx0 < b.x + b.w - 1 && fx1 > b.x + 1) {
              const c = (b.y + 1) - feet;
              if (tr.cf == null) { tr.cf = c; tr.vf = vy; }
              tr.cb = c; tr.vb = vy;
            }
          }
        }
      }
    
      function inferHumanMove(m) {
        const ob = m.obs;
        if (ob.air) return (ob.rise || 0) >= 68 ? 'jump' : 'hop';
        if (ob.duck) return 'duck';
        return 'run';
      }
    
      function onCleared(m, now) {
        m.cleared = true;
        const st = m.st;
        if (!st) return;
        brain.stats.cleared++;
        const human = m.who !== 'bot';
        const move = human ? inferHumanMove(m) : m.move;
        if (!move) return;
        learn(st.key, MI[move], 1, human ? 'you' : 'bot');
        if (human) {
          recordDemo(st, move);
          const seen = `${st.kind}:${move}`;
          if (LOG.moves && !humanSeen.has(seen)) { humanSeen.add(seen); log(`👀 you: ${KIND_LABEL[st.kind]} → ${move}`); }
          if (LOG.moves && brain.stats.demos % 25 === 0) log(`👀 ${brain.stats.demos} moments learned from you — ${SIT.KINDS.filter(k => brain.D2[k]).map(k => `${KIND_LABEL[k]} ${ANAMES[argmax(brain.D2[k])]}`).join(', ')}`);
        }
        if ((move === 'jump' || move === 'hop') && m === jumpFor) {
          const ideal = timingFromTrace(m);
          if (ideal != null) nudgeTau(move, st.kind, st.speed, ideal, human ? CFG.timingRate * 0.7 : CFG.timingRate);
        }
      }
    
      function diagnose(m, t) {
        // which move it was, why it failed, how much to blame the move vs the timing
        const st = m.st, kind = m.kind;
        const human = !m.move;
        const move = human ? (t.jumping ? ((m.obs.rise || 0) >= 68 || t.vy < -5 ? 'jump' : 'hop') : t.ducking ? 'duck' : 'run') : m.move;
        const ground = kind.startsWith('small') || kind.startsWith('large') || kind === 'birdLow';
        let why, r = -1, dTau = 0;
        if (move === 'run') why = 'ran into it';
        else if (move === 'duck') {
          if (ground) why = 'ducked into it';
          else if (!t.ducking) { why = 'ducked too late'; r = -0.3; dTau = +CFG.timingStep * 1.5; }
          else why = 'ducking did not clear it';
        } else { // jump / hop
          if (kind === 'birdHigh') why = 'jumped into a high bird';
          else if (!human && !m.pressedAt && m.blocked) { why = m.blocked.air ? 'still in the air from the last obstacle' : 'was ducking'; r = -0.2; }
          else if (!t.jumping) {
            if (m.blocked) { why = m.blocked.air ? 'still in the air from the last obstacle' : 'was ducking'; r = -0.2; }
            else if (!human && !m.pressedAt) { why = 'never jumped'; r = -0.3; dTau = Math.max(CFG.timingStep * 2, TAU0[move] - tauFor(move, kind, st ? st.speed : 'medium')); }
            else why = 'did not jump';
          } else if (t.vy < 0) { why = `${move === 'hop' ? 'hopped' : 'jumped'} too late`; r = -0.3; dTau = +CFG.timingStep; }
          else if (move === 'hop' && (m.trace ? m.trace.rise : 99) < 60) { why = 'hop too low for it'; r = -0.7; }
          else { why = `${move === 'hop' ? 'hopped' : 'jumped'} too early`; r = -0.3; dTau = -CFG.timingStep; }
          // move the trigger past the press that failed, not just a fixed step
          if (!human && m.pressTTC != null && st && dTau) {
            const cur = tauFor(move, kind, st.speed);
            dTau = dTau > 0 ? Math.max(dTau * 0.5, m.pressTTC + CFG.timingStep * 0.6 - cur) : Math.min(dTau * 0.5, m.pressTTC - CFG.timingStep * 0.6 - cur);
          }
        }
        return { move, why, r, dTau, human };
      }
    
      function onCrash(now, humanCtl) {
        crashAt = now;
        const t = trexState(), sc = score();
        const o = R.horizon && R.horizon.obstacles && R.horizon.obstacles[0];
        const m = o && meta.get(o);
        let line = `💥 crashed at ${sc}`;
        if (m && m.kind) {
          if (!m.st) { const S = readScene(now); const f = S.obs.find(b => b.m === m); if (f) decide(f, S, 'you'); }
          const d = diagnose(m, t), st = m.st;
          if (st) {
            learn(st.key, MI[d.move], d.r, d.human ? 'you' : 'bot');
            if (d.dTau && !d.human) nudgeTau(d.move, st.kind, st.speed, tauFor(d.move, st.kind, st.speed) + d.dTau, 1, d.why);
            if (/still in the air/.test(d.why) && m.prevMove === 'jump' && m.prevKey) learn(m.prevKey, MI.jump, -0.5, 'blame'); // the jump before this one landed too late
          }
          lastCrash = { kind: m.kind, move: d.move, why: d.why, who: d.human ? 'you' : 'bot', score: sc };
          brain.crashes.push({ k: st ? st.key : m.kind, m: d.move, why: d.why, who: lastCrash.who, s: sc,
            at: m.pressTTC != null ? Math.round(m.pressTTC) : undefined, dx: Math.round(o.xPos - t.x), y: t.y, vy: +t.vy.toFixed(1), v: m.v ? +m.v.toFixed(3) : undefined });
          line += ` on ${KIND_LABEL[m.kind]}: ${d.human ? 'you ' : ''}${d.move} — ${d.why}`;
        }
        brain.stats.crashes++;
        const who = humanCtl ? 'you' : 'bot';
        brain.stats.runs++; brain.stats[who === 'bot' ? 'botRuns' : 'yourRuns']++;
        brain.stats.best = Math.max(brain.stats.best, sc);
        if (who === 'bot') brain.stats.bestBot = Math.max(brain.stats.bestBot, sc); else brain.stats.bestYou = Math.max(brain.stats.bestYou, sc);
        brain.stats.scores.push([sc, who === 'bot' ? 1 : 0]); if (brain.stats.scores.length > 50) brain.stats.scores.shift();
        if (LOG.crashes) log(line);
        if (LOG.trace) api.lastCrashTrace = traceBuf.slice();
        prevFocus = null; jumpFor = null; runInfo = null;
        releaseAll(); saveBrain();
      }
    
      function tick(now) {
        if (!R) {
          if (now - lastFind > 1000) { lastFind = now; R = findRunner(); if (R) onFound(); }
          return;
        }
        const dt = now - lastNow; lastNow = now;
        if (mode === 'off') { releaseAll(); if (R.crashed && !wasCrashed) { wasCrashed = true; prevFocus = jumpFor = null; } return; }
        const humanCtl = mode === 'train' || now < yieldUntil;
        if (humanCtl) releaseAll();
    
        if (R.crashed) {
          if (!wasCrashed) { wasCrashed = true; onCrash(now, humanCtl); }
          if (!humanCtl && CFG.autoStart && now - crashAt > CFG.restartDelayMs) { crashAt = now; tap(KEYS.enter); }
          return;
        }
        if (wasCrashed) { wasCrashed = false; prevFocus = null; jumpFor = null; }
        if (!R.playing) {
          if (!humanCtl && CFG.autoStart && now - lastStartTry > 700) {
            lastStartTry = now;
            if (!R.activated && !R.playingIntro) tap(KEYS.jump); else if (R.paused) key('keyup', KEYS.jump);
          }
          prevT = null;
          return;
        }
        if (dt > 0 && dt < 100) frameMs = frameMs * 0.95 + dt * 0.05;
        if (!runInfo) runInfo = { start: now };
    
        const S = readScene(now), t = S.t, f = S.focus;
        // obstacle passed without a crash
        if (prevFocus && (!f || f.m !== prevFocus)) { onCleared(prevFocus, now); if (f && f.m !== prevFocus) { f.m.prevMove = prevFocus.move || inferHumanMove(prevFocus); f.m.prevKey = prevFocus.st && prevFocus.st.key; } }
        prevFocus = f ? f.m : null;
        // the jump has ended
        if (prevT && prevT.jumping && !t.jumping && jumpFor) { jumpFor.landedAt = now; if (jumpFor.cleared) jumpFor = null; }
        // a jump started that the bot didn't press: you did it
        if (prevT && !prevT.jumping && t.jumping && humanCtl && f) { jumpFor = f.m; if (f.m.pressTTC == null) f.m.pressTTC = f.m.humanPressTTC ?? f.ttc + frameMs; }
        if (f && f.kind && !f.m.st && f.ttc <= CFG.decideAtMs) decide(f, S, humanCtl ? 'you' : 'bot');
        if (f && f.m.st && !humanCtl && !f.m.move) { f.m.who = 'bot'; const c = choose(f.m.st); f.m.move = c.move; f.m.why = c.why; } // you handed control back mid-obstacle
        if (!humanCtl) motor(now, S);
        observe(now, S);
        prevT = t;
        drawDebug(S);
    
        if (now - lastSave > 15000) saveBrain();
        if (now - lastSummary > 30000) { lastSummary = now; log('📊 ' + summary()); }
      }
      function frame(now) {
        rafId = requestAnimationFrame(frame);
        try { tick(now); } catch (e) { if (++errors <= 5) warn('tick error: ' + (e && e.stack || e)); }
      }
    
      // ================= you (keyboard) =================
      const human = { jumpAt: 0, downAt: 0 };
      const onKey = e => {
        if (!e.isTrusted) return; // the bot's own key events
        if (e.code === CFG.toggleKey && e.type === 'keydown') { e.preventDefault(); e.stopImmediatePropagation(); e.shiftKey ? api.train() : api.toggle(); return; }
        const isJump = e.keyCode === 32 || e.keyCode === 38, isDown = e.keyCode === 40;
        if (!isJump && !isDown && e.keyCode !== 13) return;
        const now = performance.now();
        if (mode === 'bot' && R && R.playing && !R.crashed && (isJump || isDown)) {
          if (now >= yieldUntil) { releaseAll(); log('🎮 you took over — it watches and learns until you stop pressing keys'); }
          yieldUntil = now + CFG.yieldToHumanMs;
        }
        if (e.type === 'keydown' && isJump && !e.repeat && R && R.playing && !R.crashed && !R.tRex.jumping && !R.tRex.ducking) {
          try { const S = readScene(now); if (S.focus) S.focus.m.humanPressTTC = S.focus.ttc; } catch (err) {}
        }
      };
      window.addEventListener('keydown', onKey, true);
      window.addEventListener('keyup', onKey, true);
    
      // ================= summary / panel =================
      const avg = a => (a.length ? sum(a) / a.length : 0);
      function movesLearned() {
        const out = [];
        for (const kind of SIT.KINDS) {
          const keys = SIT.allKeys().filter(k => k.startsWith(kind + '|'));
          const tally = zeros();
          for (const k of keys) tally[choose({ ...SIT.parse(k), key: k }, false).i]++;
          out.push([kind, ANAMES[argmax(tally)]]);
        }
        return out;
      }
      function timingSummary() {
        const g = {};
        for (const [k, v] of Object.entries(brain.T)) { const m = k.split('|')[0]; (g[m] ||= []).push(v); }
        return Object.keys(TAU0).filter(m => m !== 'duck').map(m => {
          const e = g[m] || [], n = sum(e.map(v => v.n));
          return `${m} ${Math.round(n ? sum(e.map(v => v.tau * v.n)) / n : TAU0[m])} ms${n ? ` (${n} jumps)` : ''}`;
        }).join(' · ');
      }
      function summary() {
        const S = brain.stats, last = S.scores.slice(-10).map(x => x[0]);
        const cur = R && (R.playing || R.paused) && !R.crashed ? score() : 0;
        return `${cur ? `this run ${cur} · ` : ''}runs ${S.runs} · best ${Math.max(S.best, cur)} (bot ${S.bestBot}, you ${S.bestYou})${last.length ? ` · last ${last.length} avg ${Math.round(avg(last))}` : ''} · cleared ${S.cleared} · moments from you ${S.demos} · situations ${Object.keys(brain.N).length}`;
      }
    
      const panel = document.createElement('div');
      panel.setAttribute('data-dinobot', '');
      panel.style.cssText = 'all:initial;position:fixed;right:12px;bottom:12px;z-index:2147483646;width:276px;padding:8px 10px;box-sizing:border-box;' +
        'font:12px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;color:#e6f4ea;background:rgba(20,32,24,.95);border:1px solid #3d9a57;' +
        'border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.45);user-select:none;';
      const btn = (attrs, label, title, extra = '') => `<button ${attrs} title="${title}" style="all:unset;box-sizing:border-box;flex:1;text-align:center;padding:5px 0;` +
        `color:#e6f4ea;background:#1f3a29;border:1px solid #3d9a57;border-radius:5px;cursor:pointer;font:inherit;${extra}">${label}</button>`;
      panel.innerHTML = `
        <div data-k="head" style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;cursor:move">
          <b style="letter-spacing:.05em">DINO × LAYA</b><span><span data-k="brain" style="opacity:.75;font-size:11px"></span>
          ${btn('data-a="min"', '–', 'Minimise', 'flex:none;width:20px;padding:0 0 1px;margin-left:6px;')}</span></div>
        <div data-k="body">
          <div style="display:flex;gap:6px;margin-bottom:6px">
            ${btn('data-m="bot"', '🤖 Play', 'The bot plays (`)')}${btn('data-m="train"', '👀 Train', 'You play, it learns from you (Shift+`)')}${btn('data-m="off"', '⏹ Off', 'Do nothing')}</div>
          <div style="display:flex;gap:6px;margin-bottom:6px">
            ${btn('data-a="export"', '⬇ Brain', NO_DOWNLOADS ? 'Copy what it has learned to the clipboard (the dino page blocks downloads)' : 'Download what it has learned (.json)')}${btn('data-a="import"', '⬆ Import', 'Load a brain .json (merge or replace)')}
            ${btn('data-a="debug"', '🔍 See', 'Show what it sees on the game')}${btn('data-a="laya"', '🧠 Live', 'Run the 1.7 GB Laya model live (normal https pages only)')}</div>
          <input type="file" data-k="file" accept=".json,application/json" style="display:none">
          <div data-k="status" style="min-height:17px"></div>
          <div data-k="moves" style="font-size:11px;opacity:.9"></div>
          <div data-k="stats" style="font-size:11px;opacity:.75"></div>
          <div style="opacity:.55;font-size:10.5px;margin-top:4px">\` play/off · Shift+\` train · Space/↑ jump · ↓ duck</div></div>`;
      const q = k => panel.querySelector(`[data-k="${k}"]`);
      for (const t of ['pointerdown', 'pointerup', 'touchstart', 'touchend', 'click']) panel.addEventListener(t, e => e.stopPropagation());
      for (const b of panel.querySelectorAll('button')) {
        b.addEventListener('mousedown', e => e.stopPropagation(), true);
        b.addEventListener('click', e => {
          e.stopPropagation(); b.blur();
          if (b.dataset.m) return api.setMode(b.dataset.m);
          const a = b.dataset.a;
          if (a === 'export') return api.exportBrain();
          if (a === 'import') return q('file').click();
          if (a === 'debug') return api.debug(!dbg);
          if (a === 'laya') return api.laya();
          if (a === 'min') { const body = q('body'); body.style.display = body.style.display === 'none' ? '' : 'none'; }
        });
      }
      q('file').addEventListener('change', async e => {
        const file = e.target.files[0]; e.target.value = ''; if (!file) return;
        const how = confirm(`Import "${file.name}"\n\nOK = MERGE with what this bot already knows\nCancel = REPLACE it`) ? 'merge' : 'replace';
        try { api.importBrain(await file.text(), how); } catch (err) { warn('import failed: ' + err.message); }
      });
      let drag = null; // drag the panel by its header
      q('head').addEventListener('mousedown', e => { if (e.target.closest('button')) return; const r = panel.getBoundingClientRect(); drag = { dx: e.clientX - r.left, dy: e.clientY - r.top }; e.preventDefault(); });
      const onDragMove = e => { if (!drag) return; panel.style.left = clamp(e.clientX - drag.dx, 0, innerWidth - 60) + 'px'; panel.style.top = clamp(e.clientY - drag.dy, 0, innerHeight - 30) + 'px'; panel.style.right = panel.style.bottom = 'auto'; };
      const onDragUp = () => { drag = null; };
      window.addEventListener('mousemove', onDragMove); window.addEventListener('mouseup', onDragUp);
      document.body.appendChild(panel);
    
      let movesCache = '', movesAt = 0, flashMsg = null, flashUntil = 0;
      const flash = (msg, ms = 4000) => { flashMsg = msg; flashUntil = performance.now() + ms; };
      const panelTimer = setInterval(() => {
        const now = performance.now();
        for (const b of panel.querySelectorAll('button[data-m]')) {
          const on = b.dataset.m === mode;
          b.style.background = on ? (mode === 'train' ? '#6b4e12' : mode === 'bot' ? '#2b6e3c' : '#5a2323') : '#1f3a29';
          b.style.borderColor = on ? '#ffd27a' : '#3d9a57';
        }
        q('brain').textContent = layaState === 'live' ? `laya: live ${layaBackend || ''}` : layaState === 'loading' ? `laya: loading${layaPct != null ? ' ' + layaPct + '%' : '…'}` : 'laya: baked';
        let s;
        if (!R) s = 'Looking for the dino game… (open chrome://dino)';
        else if (mode === 'off') s = '⏹ Off — you have full control';
        else if (R.crashed) s = lastCrash ? `💥 ${KIND_LABEL[lastCrash.kind]}: ${lastCrash.who === 'you' ? 'you ' : ''}${lastCrash.move} — ${lastCrash.why}` : '💥 crashed';
        else if (!R.playing) s = mode === 'train' ? '👀 Train: click the game, press Space to start' : R.activated ? 'Paused — resuming…' : 'Starting…';
        else if (mode === 'train' || now < yieldUntil) s = `👀 ${mode === 'train' ? 'Training' : 'You have control'} — ${brain.stats.demos} moments learned from you`;
        else s = lastDecision ? `🤖 ${KIND_LABEL[lastDecision.kind]} → ${lastDecision.move} (${lastDecision.why}) · score ${score()}` : `🤖 Playing · score ${score()}`;
        q('status').textContent = now < flashUntil ? flashMsg : s;
        if (now - movesAt > 2000) {
          movesAt = now;
          const ml = Object.fromEntries(movesLearned());
          const group = ks => { const c = {}; for (const k of ks) c[ml[k]] = (c[ml[k]] || 0) + 1; return Object.entries(c).sort((a, b) => b[1] - a[1]).map(([m]) => m).join('/'); };
          movesCache = `Moves: small cacti ${group(['small1', 'small2', 'small3'])} · tall ${group(['large1', 'large2', 'large3'])} · birds: low ${ml.birdLow}, head ${ml.birdMid}, high ${ml.birdHigh}\nTiming: ` + timingSummary();
        }
        // the game inverts the whole page at night (filter on <html>): invert the panel back so it stays readable
        const inv = /invert\(1\)/.test(getComputedStyle(document.documentElement).filter) ? 'invert(1)' : '';
        if (panel.style.filter !== inv) { panel.style.filter = inv; if (dbg) dbg.style.filter = inv; }
        q('moves').textContent = movesCache; q('moves').style.whiteSpace = 'pre-line';
        q('stats').textContent = summary() + ` · saved in ${store.kind}`;
      }, 300);
    
      // ================= debug overlay: what it sees =================
      let dbg = null, dbgCtx = null, dbgPosAt = 0;
      const COLORS = { jump: '#e8590c', hop: '#f59f00', duck: '#1c7ed6', run: '#2b8a3e' };
      function drawDebug(S) {
        if (!dbg || !R.canvas) return;
        const now = performance.now();
        const d0 = R.dimensions || {}, dims = { width: num(d0.width, d0.WIDTH, 600), height: num(d0.height, d0.HEIGHT, 150) };
        if (now - dbgPosAt > 400) {
          dbgPosAt = now;
          const r = R.canvas.getBoundingClientRect();
          Object.assign(dbg.style, { left: r.left + 'px', top: r.top + 'px', width: r.width + 'px', height: r.height + 'px' });
          const W = Math.round(dims.width * 2), H = Math.round(dims.height * 2);
          if (dbg.width !== W || dbg.height !== H) { dbg.width = W; dbg.height = H; }
        }
        const g = dbgCtx, k = dbg.width / dims.width;
        g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, dbg.width, dbg.height); g.setTransform(k, 0, 0, k, 0, 0);
        const t = S.t;
        g.lineWidth = 1; g.font = '9px monospace';
        g.strokeStyle = '#7048e8'; g.strokeRect(t.x + 1, t.y + 1, (t.ducking ? 59 : t.W) - 2, t.H - 2);
        for (const b of S.obs) {
          if (!b.kind) continue;
          const mv = b.m.move || (b.m.st ? inferHumanMove(b.m) : null);
          g.strokeStyle = b === S.focus ? (COLORS[mv] || '#e03131') : '#adb5bd';
          g.strokeRect(b.x + 1, b.y + 1, b.w - 2, b.h - 2);
          if (b === S.focus) {
            const mvT = mv === 'hop' ? 'hop' : mv === 'duck' ? 'duck' : 'jump', st = b.m.st;
            const tau = tauFor(mvT, b.kind, st ? st.speed : speedBucket(S.speed));
            const xTrig = t.x + t.W / 2 + tau * b.v - b.w / 2; // where its left edge will be when the bot acts
            if (mv !== 'run') { g.setLineDash([3, 3]); g.strokeStyle = COLORS[mv] || '#e03131'; g.beginPath(); g.moveTo(xTrig, 20); g.lineTo(xTrig, 140); g.stroke(); g.setLineDash([]); }
            g.fillStyle = COLORS[mv] || '#e03131';
            g.fillText(`${KIND_LABEL[b.kind]}${mv ? ' → ' + mv : ''}`, Math.max(2, Math.min(b.x - 10, dims.width - 120)), Math.max(10, b.y - 14));
            g.fillText(`${Math.round(b.ttc)} ms (acts at ${Math.round(tau)})`, Math.max(2, Math.min(b.x - 10, dims.width - 120)), Math.max(19, b.y - 5));
          }
        }
      }
    
      // ================= export / import =================
      const stampName = () => new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
      const brainPayload = () => ({ format: 'dino-laya-brain', version: 1, exportedAt: new Date().toISOString(), actions: ACTIONS, questions: QUESTIONS, brain });
      function copyText(text) {
        const ta = document.createElement('textarea');
        ta.value = text; ta.setAttribute('readonly', ''); ta.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0';
        document.body.appendChild(ta); ta.select();
        let ok = false; try { ok = document.execCommand('copy'); } catch (e) {}
        ta.remove();
        return ok;
      }
      function download(text, name) {
        if (NO_DOWNLOADS) return false;
        try {
          const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' })); a.download = name;
          document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 60000);
          return true;
        } catch (e) { return false; }
      }
      function mergeBrain(dst, src) {
        const add = (a, b) => b.map((v, i) => (a?.[i] || 0) + v);
        for (const k of ['D', 'D2']) for (const [key, arr] of Object.entries(src[k])) dst[k][key] = add(dst[k][key], arr);
        for (const [key, n2] of Object.entries(src.N)) {
          const n1 = dst.N[key] || zeros(), q1 = dst.Q[key] || zeros(), q2 = src.Q[key] || zeros();
          dst.Q[key] = q1.map((x, i) => (n1[i] + n2[i] ? (x * n1[i] + q2[i] * n2[i]) / (n1[i] + n2[i]) : 0));
          dst.N[key] = add(n1, n2);
        }
        for (const [key, e] of Object.entries(src.T)) {
          const d = dst.T[key];
          dst.T[key] = d ? { tau: (d.tau * d.n + e.tau * e.n) / Math.max(1, d.n + e.n), n: d.n + e.n } : { ...e };
        }
        for (const [key, p] of Object.entries(src.prior)) dst.prior[key] ||= p;
        for (const [k, v] of Object.entries(src.stats)) if (typeof v === 'number') dst.stats[k] = /^best/.test(k) ? Math.max(dst.stats[k] || 0, v) : (dst.stats[k] || 0) + v;
        dst.crashes = dst.crashes.concat(src.crashes).slice(-80);
      }
    
      // ================= debug overlay toggle, hooks =================
      function onFound() {
        unhookBind();
        log(`🦖 found the game (speed ${num(R.currentSpeed).toFixed(1)}, ${R.crashed ? 'crashed' : R.playing ? 'running' : 'waiting'}). Brain: ${Object.keys(brain.N).length} situations, ${brain.stats.runs} runs, best ${brain.stats.best}.`);
      }
      const onResize = () => { dbgPosAt = 0; };
      window.addEventListener('resize', onResize);
    
      // ================= public API =================
      const api = window.__dinoBot = {
        CFG, LOG, brain,
        setMode(m) {
          if (!['bot', 'train', 'off'].includes(m) || m === mode) return;
          mode = m; releaseAll(); yieldUntil = 0;
          log(m === 'bot' ? '🤖 PLAY — the bot is driving' : m === 'train' ? '👀 TRAIN — click the game and play; it learns which move you use for each obstacle and how you time it' : '⏹ OFF — you have control');
          saveBrain();
        },
        toggle() { api.setMode(mode === 'bot' ? 'off' : 'bot'); },
        train() { api.setMode(mode === 'train' ? 'off' : 'train'); },
        get mode() { return mode; },
        get game() { return R; },
        state() { return R && R.playing ? readScene(performance.now()) : null; },
        stats: () => summary(),
        timing: () => Object.fromEntries(Object.entries(brain.T).map(([k, v]) => [k, `${Math.round(v.tau)} ms (n=${v.n})`])),
        policy: () => Object.fromEntries(SIT.allKeys().map(k => [k, choose({ ...SIT.parse(k), key: k }, false).move])),
        debug(on = true) {
          if (on && !dbg) {
            dbg = document.createElement('canvas'); dbgCtx = dbg.getContext('2d'); dbgPosAt = 0;
            dbg.style.cssText = 'position:fixed;pointer-events:none;z-index:2147483645;';
            document.body.appendChild(dbg);
          }
          if (!on && dbg) { dbg.remove(); dbg = null; dbgCtx = null; }
        },
        save: saveBrain,
        reset() { store.clear(); const fresh = normalizeBrain(null); for (const k of Object.keys(brain)) delete brain[k]; Object.assign(brain, fresh); saveBrain(); log('brain reset'); },
        forgetDemos() { brain.D = {}; brain.D2 = {}; brain.stats.demos = 0; saveBrain(); log('forgot what it learned from watching you'); },
        exportBrain() {
          saveBrain();
          const text = JSON.stringify(brainPayload());
          const what = `${Object.keys(brain.N).length} situations, ${brain.stats.demos} moments from you, ${Object.keys(brain.T).length} timings`;
          if (download(text, `dino-laya-brain-${stampName()}.json`)) log(`⬇ downloaded the brain: ${what}`);
          else if (copyText(text)) { log(`📋 brain copied to the clipboard (${what}, ${Math.round(text.length / 1024)} KB) — Chrome blocks downloads from the dino page, so paste it into a .json file to keep it`); flash('📋 Brain copied — paste it into a .json file'); }
          else log(`⬇ couldn't copy here — run copy(__dinoBot.exportBrain()) in the console, then paste it into a .json file`);
          return text;
        },
        importBrain(src, how = 'merge') {
          const j = typeof src === 'string' ? JSON.parse(src) : src;
          const b = normalizeBrain(JSON.parse(JSON.stringify(j.brain || j)));
          if (how === 'replace') { for (const k of Object.keys(brain)) delete brain[k]; Object.assign(brain, b); } else mergeBrain(brain, b);
          saveBrain(); movesAt = 0;
          log(`⬆ imported brain (${how}): ${Object.keys(brain.N).length} situations, ${brain.stats.demos} moments from you`);
        },
        laya() { return startLaya(); },
        uninstall() {
          cancelAnimationFrame(rafId); clearInterval(panelTimer); releaseAll(); saveBrain(); unhookBind();
          panel.remove(); api.debug(false);
          window.removeEventListener('keydown', onKey, true); window.removeEventListener('keyup', onKey, true);
          window.removeEventListener('mousemove', onDragMove); window.removeEventListener('mouseup', onDragUp);
          window.removeEventListener('resize', onResize);
          try { worker && worker.terminate(); } catch (e) {}
          delete window.__dinoBot; console.log('[DinoBot] uninstalled');
        },
      };
    
      R = findRunner();
      if (R) onFound(); else { hookBind(); log('waiting for the dino game — open chrome://dino (or go offline) and paste this again if nothing happens'); }
      log(`installed. Laya's answers are baked in (${LAYA_BAKED.model}); brain saved in ${store.kind}.`);
      log('Panel bottom-right: 🤖 Play · 👀 Train · ⏹ Off · ⬇ Brain · ⬆ Import · 🔍 See · 🧠 Live. Keys: ` play/off, Shift+` train. Click the game first so it gets your keys.');
      if (CFG.liveLaya) startLaya();
      rafId = requestAnimationFrame(frame);
    })();
    }