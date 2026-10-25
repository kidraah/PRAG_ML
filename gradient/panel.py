"""
PRAG — the Gradient's tabs in the control panel: Corpus, Training and Ask.

launch.py leaves placeholders in its page (/*GRADIENT_CSS*/, <!--GRADIENT_NAV-->,
<!--GRADIENT_SECTIONS-->, <!--GRADIENT_PROFILE-->, //GRADIENT_JS); inject()
fills them. Everything shown comes from gradient/api.py.
"""

ICONS = {
    "corpus": "M12 3C7.58 3 4 4.79 4 7v10c0 2.21 3.59 4 8 4s8-1.79 8-4V7c0-2.21-3.58-4-8-4zm0 2c3.87 0 6 1.5 6 2s-2.13 "
              "2-6 2-6-1.5-6-2 2.13-2 6-2zm6 12c0 .5-2.13 2-6 2s-6-1.5-6-2v-2.23c1.61.78 3.72 1.23 6 1.23s4.39-.45 "
              "6-1.23V17zm0-5c0 .5-2.13 2-6 2s-6-1.5-6-2V9.77C7.61 10.55 9.72 11 12 11s4.39-.45 6-1.23V12z",
    "training": "M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L2 16.99z",
    "ask": "M21 6h-2v9H6v2c0 .55.45 1 1 1h11l4 4V7c0-.55-.45-1-1-1zm-4 6V3c0-.55-.45-1-1-1H3c-.55 0-1 .45-1 1v14l4-4h10"
           "c.55 0 1-.45 1-1z",
}
NAV = "".join(
    f'<button class="nav-btn" data-tab="{tab}" title="{name}"><svg viewBox="0 0 24 24" aria-hidden="true">'
    f'<path d="{ICONS[tab]}"/></svg><span class="btn-text">{name}</span></button>\n        '
    for tab, name in (("corpus", "Corpus"), ("training", "Training"), ("ask", "Ask")))

CSS = r"""
/* The Gradient: corpus, training, ask */
.g-stats { display: flex; flex-wrap: wrap; gap: 14px 40px; margin: 18px 0 4px; }
.g-stat b { display: block; font: 600 30px/1.1 var(--display); letter-spacing: 1px; }
.g-stat span { color: var(--muted); font-size: 14.5px; }
.g-pair { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0 30px; align-items: start; }
.g-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 2px; }
.g-chips span { font: 600 14px var(--body); padding: 3px 10px; border: 1px solid var(--line); background: var(--glass); }
.g-bars { margin-top: 10px; }
.g-bar { display: grid; grid-template-columns: 150px minmax(0, 1fr) 54px 74px; gap: 10px; align-items: center; padding: 4px 0;
         font-size: 15px; }
.g-bar .track2 { height: 8px; background: var(--rule); }
.g-bar .track2 i { display: block; height: 100%; background: var(--viz-after); }
.g-bar.political .track2 i { background: var(--viz-bar); }
.g-bar .n { font: 500 12px var(--mono); text-align: right; }
.g-bar .w { font: 500 12px var(--mono); color: var(--muted); text-align: right; }
.g-bar .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.g-mini { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 15px; color: var(--muted); margin-top: 8px; }
.g-mini b { color: var(--text); font: 500 13px var(--mono); }
.g-item { padding: 12px 0; border-top: 1px solid var(--rule); }
.g-item .head { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; }
.g-item .who { font-weight: 700; }
.g-item .body { margin-top: 4px; overflow-wrap: anywhere; }
.g-item .take { margin-top: 6px; color: var(--muted); font-style: italic; }
.g-item .score { font: 500 12px var(--mono); color: var(--muted); }
.g-log { max-height: 300px; overflow: auto; background: var(--console-bg); color: var(--console-text); padding: 12px 16px;
         margin-top: 14px; }
.g-conf { border-collapse: collapse; font: 500 13px var(--mono); margin-top: 8px; }
.g-conf td, .g-conf th { border: 1px solid var(--rule); padding: 6px 12px; text-align: right; }
.g-conf th { font: 500 10px var(--display); letter-spacing: 1px; color: var(--muted); }
.g-chosen td { font-weight: 700; }
#g-prefs table.data, #g-feed-card table.data { font-size: 14.5px; }
#g-prefs table.data th, #g-prefs table.data td, #g-feed-card table.data th, #g-feed-card table.data td { padding: 8px 6px; }
.g-words { font-size: 15px; margin: 6px 0; }
.g-words b { font: 500 10px var(--display); letter-spacing: 1.2px; color: var(--muted); margin-right: 8px; }
.g-curve { margin-top: 10px; max-width: 520px; }
.g-curve svg { width: 100%; height: auto; display: block; }
.g-curve .grid { stroke: var(--viz-grid); }
.g-curve .axis { fill: var(--muted); font: 500 11px var(--mono); }
.g-curve .train { stroke: var(--viz-bar); fill: none; stroke-width: 2; }
.g-curve .val { stroke: var(--viz-after); fill: none; stroke-width: 2; stroke-dasharray: 5 4; }
.g-curve circle { fill: var(--viz-after); }
.g-ask-form { display: grid; gap: 12px; max-width: 900px; }
.g-try { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 15px; }
.g-try .hint { margin-right: 4px; }
.g-answer { margin-top: 22px; padding-left: 16px; border-left: 3px solid var(--line); }
.g-answer .q { color: var(--muted); font-size: 15px; margin-bottom: 6px; }
.g-answer .text { font-size: 17.5px; line-height: 1.5; max-width: 80ch; white-space: pre-wrap; }
.g-answer .meta { margin-top: 8px; font: 500 11.5px var(--mono); color: var(--muted); }
.g-cite { font: 600 11px var(--mono); vertical-align: super; padding: 0 3px; margin: 0 1px; cursor: pointer; color: var(--link);
          background: var(--glass); border: 1px solid var(--line); }
.g-evidence { margin-top: 10px; }
.g-ev { display: grid; grid-template-columns: 34px minmax(0, 1fr); gap: 2px 12px; padding: 10px 0; border-top: 1px solid var(--rule);
        font-size: 15px; }
.g-ev .n { font: 600 12px var(--mono); color: var(--muted); padding-top: 2px; }
.g-ev.cited .n { color: var(--text); }
.g-ev.flash { background: var(--hover); }
.g-ev .sub { font: 500 11.5px var(--mono); color: var(--muted); margin-top: 4px; }
.g-debate { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px 30px; margin-top: 20px; }
.g-turn { border: 2px solid var(--line); background: var(--panel); padding: 14px 16px; box-shadow: var(--shadow-sm); }
.g-turn h4 { font: 600 12px var(--display); letter-spacing: 1.4px; margin: 0 0 8px; }
.g-turn .text { white-space: pre-wrap; line-height: 1.5; }
.g-turn.right { grid-column: 2; }
.g-turn.left { grid-column: 1; }
.g-moderator { grid-column: 1 / 3; border-top: 4px solid var(--line); padding-top: 14px; }
.g-moderator .text { white-space: pre-wrap; line-height: 1.55; max-width: 90ch; }
.g-meter { display: grid; grid-template-columns: 200px minmax(0, 1fr) 70px; gap: 12px; align-items: center; margin: 10px 0; }
.g-meter .track2 { height: 10px; background: var(--rule); position: relative; }
.g-meter .track2 i { display: block; height: 100%; background: var(--viz-after); }
.g-meter .track2 em { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--warn); }
.g-meter b { font: 500 13px var(--mono); text-align: right; }
.g-notice-ok { border-left-color: var(--ok); }
/* Ask: chat and debate windows */
.g-askbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; margin: 0 0 18px; }
.g-models { font: 500 11.5px var(--mono); color: var(--muted); }
.g-models.bad { color: var(--warn); }
.g-win { display: grid; grid-template-rows: auto minmax(0, 1fr) auto; height: calc(100vh - 220px); min-height: 540px;
         background: var(--panel); border: 2px solid var(--line); box-shadow: var(--shadow-lg); margin-bottom: 30px; }
.g-win-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; padding: 12px 16px;
              border-bottom: 2px solid var(--line); background: var(--bar); color: var(--bar-text); }
.g-contact { display: flex; align-items: center; gap: 10px; padding: 6px 12px 6px 6px; cursor: pointer; text-align: left;
             background: transparent; color: var(--bar-text); border: 2px solid transparent; font: inherit; }
.g-contact:hover { border-color: var(--rule); }
.g-contact.on { border-color: var(--line); background: var(--hover); }
.g-contact b { display: block; font: 600 15px var(--body); }
.g-contact small { display: block; font-size: 13px; color: var(--bar-muted); }
.g-avatar { flex: none; width: 36px; height: 36px; display: grid; place-items: center; font: 700 12px var(--display);
            letter-spacing: 1px; border: 2px solid var(--line); }
.g-avatar.p1 { background: var(--accent); color: var(--on-accent); }
.g-avatar.p2 { background: var(--panel); color: var(--text); box-shadow: inset 0 0 0 2px var(--panel), inset 0 0 0 4px var(--line); }
.g-avatar.mod { background: var(--glass); color: var(--muted); font-size: 10px; }
.g-vs { font: 600 12px var(--display); letter-spacing: 2px; color: var(--bar-muted); }
.g-title { font: 600 14px var(--display); letter-spacing: 1px; }
.g-thread { overflow-y: auto; overscroll-behavior: contain; padding: 22px 22px 8px; display: flex; flex-direction: column; gap: 18px;
         scrollbar-width: thin; scrollbar-color: var(--line) transparent; }
.g-msg { display: flex; gap: 12px; align-items: flex-start; max-width: min(78ch, 86%); }
.g-msg.user { align-self: flex-end; }
.g-msg.right { align-self: flex-end; flex-direction: row-reverse; }
.g-msg .g-body { min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.g-msg.right .g-body { align-items: flex-end; }
.g-name { font: 500 11px var(--mono); color: var(--muted); }
.g-bubble { padding: 11px 15px; font-size: 17px; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere;
            background: var(--glass); border: 2px solid var(--line); }
.g-msg.user .g-bubble { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
[data-theme="brutal"] .g-msg.user .g-bubble { border-color: var(--line); }
.g-bubble .g-cite { font-size: 10.5px; vertical-align: 2px; }
.g-caret::after { content: "_"; animation: blink 1.1s steps(1) infinite; margin-left: 1px; }
.g-dots { display: inline-flex; gap: 5px; padding: 6px 0; }
.g-dots i { width: 7px; height: 7px; background: var(--muted); animation: g-dot 1.2s infinite; }
.g-dots i:nth-child(2) { animation-delay: .2s; } .g-dots i:nth-child(3) { animation-delay: .4s; }
@keyframes g-dot { 0%, 60%, 100% { opacity: .25; } 30% { opacity: 1; } }
.g-srcs { display: flex; flex-wrap: wrap; gap: 6px; }
.g-chip { font: 500 11.5px var(--mono); padding: 3px 8px; border: 1px solid var(--line); background: transparent; color: var(--muted);
          cursor: pointer; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.g-chip:hover, .g-chip.open { color: var(--text); background: var(--hover); }
.g-chip.cited { color: var(--text); border-width: 2px; }
.g-src { border: 1px solid var(--rule); border-left: 3px solid var(--line); padding: 10px 12px; font-size: 15px; max-width: 70ch;
         background: var(--panel); }
.g-src .sub { font: 500 11.5px var(--mono); color: var(--muted); margin-top: 6px; }
.g-round { align-self: center; font: 600 11px var(--display); letter-spacing: 2px; color: var(--muted); display: flex; align-items: center;
           gap: 12px; width: 100%; }
.g-round::before, .g-round::after { content: ""; flex: 1; height: 1px; background: var(--rule); }
.g-topic { align-self: center; text-align: center; max-width: 60ch; }
.g-topic small { display: block; font: 500 10.5px var(--display); letter-spacing: 2px; color: var(--muted); }
.g-topic b { font: 600 22px/1.3 var(--display); letter-spacing: .5px; }
.g-mod { align-self: stretch; border: 2px solid var(--line); background: var(--glass); padding: 14px 18px; margin: 4px 0 6px; }
.g-mod .g-name { margin-bottom: 8px; }
.g-mod .text { white-space: pre-wrap; line-height: 1.55; font-size: 16px; }
.g-mod .text b { font: 600 12px var(--display); letter-spacing: 1.4px; text-transform: uppercase; color: var(--text); }
.g-empty { margin: auto; text-align: center; max-width: 54ch; padding: 30px 10px; }
.g-empty .g-avatar { width: 64px; height: 64px; font-size: 18px; margin: 0 auto 14px; }
.g-empty h3 { font: 600 20px var(--display); letter-spacing: .5px; margin: 0 0 8px; }
.g-empty p { color: var(--muted); margin: 0 0 16px; }
.g-suggest { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.g-suggest button { font: 600 15px var(--body); padding: 8px 12px; border: 2px solid var(--line); background: var(--glass); color: var(--text);
                    cursor: pointer; }
.g-suggest button:hover { background: var(--hover); }
.g-composer { display: flex; gap: 10px; align-items: flex-end; padding: 12px 14px 8px; border-top: 2px solid var(--line); }
.g-composer textarea { flex: 1; resize: none; min-height: 48px; max-height: 180px; font-size: 17px; line-height: 1.4; padding: 12px 14px; }
.g-send { flex: none; height: 48px; min-width: 48px; display: inline-grid; place-items: center; padding: 0 16px; cursor: pointer;
          font: 700 15px var(--body); background: var(--accent); color: var(--on-accent); border: 2px solid var(--accent); }
.g-send svg { width: 20px; height: 20px; fill: currentColor; }
.g-send:disabled { opacity: .35; cursor: not-allowed; }
.g-send.stop { background: var(--panel); color: var(--text); border-color: var(--line); }
[data-theme="brutal"] .g-send { border-color: var(--line); }
.g-composer .seg { align-self: center; }
.g-foot { padding: 0 16px 10px; margin: 0; font-size: 13px; color: var(--muted); }
@media (max-width: 760px) { .g-msg { max-width: 94%; } .g-win { height: calc(100vh - 160px); } }

@media (max-width: 1100px) {
  .g-pair, .g-debate { grid-template-columns: minmax(0, 1fr); }
  .g-turn.right, .g-turn.left, .g-moderator { grid-column: 1; }
  .g-bar { grid-template-columns: 110px minmax(0, 1fr) 46px 64px; }
}
"""

SECTIONS = r"""
      <section id="tab-corpus" hidden>
        <div class="card">
          <h2 class="headline">The training corpus</h2>
          <p class="lede">Everything the Gradient learns from, one row per unique piece of content. Each persona's
            <em>feed</em> is the part of it that persona has seen: the reels its sessions recorded, the reels written
            for its feed, and the headlines the recommender served it. Chat and debate can only cite a persona's own
            feed.</p>
          <div class="g-stats" id="g-stats"></div>
          <div class="table-scroll"><table class="data" id="g-origins"></table></div>
        </div>
        <div class="g-pair" id="g-personas"></div>
        <div class="card">
          <h3>How the corpus grew</h3>
          <p class="hint">Every ingestion, in order. Retraining reads all of them; each model version records which.</p>
          <div class="table-scroll"><table class="data" id="g-batches"></table></div>
        </div>
        <div class="card">
          <div class="status"><h3>Browse a feed</h3><span class="spacer"></span><span class="hint" id="g-browse-count"></span></div>
          <div class="row"><div class="seg small" id="g-browse-account" role="radiogroup" aria-label="Account"></div>
            <select id="g-browse-origin" aria-label="Where it came from"><option value="">Every source</option></select>
            <input type="search" id="g-browse-q" placeholder="Search the text" aria-label="Search"></div>
          <div id="g-browse"></div>
          <div class="actions"><button class="btn" id="g-browse-more" hidden>Show more</button></div>
        </div>
      </section>

      <section id="tab-training" hidden>
        <div class="card">
          <h2 class="headline">Training</h2>
          <p class="lede">Two kinds of model per persona. The <strong>preference model</strong> is a classifier that
            learns what that persona watches to the end. The <strong>persona language model</strong> is a LoRA adapter
            on a small open LLM, trained on that persona's feed and its reactions, so it talks like them. Both retrain
            as the corpus grows; every version is kept.</p>
          <div class="actions">
            <button class="btn primary" id="g-train-update">Update the corpus and retrain</button>
            <button class="btn" id="g-train-prefs">Retrain the preference models</button>
          </div>
          <div class="row" style="margin-top:18px"><span class="hint">Persona language model:</span>
            <div class="seg small" id="g-lora-account" role="radiogroup" aria-label="Account"></div>
            <label class="hint" for="g-lora-iters">steps</label><input type="number" id="g-lora-iters" value="500" min="50" max="2000" step="50" style="width:100px">
            <button class="btn" id="g-train-lora">Train a new adapter</button>
            <button class="btn" id="g-train-lora-resume">Continue with what's new</button></div>
          <p class="hint" id="g-job-state"></p>
          <pre class="g-log" id="g-job-log" hidden></pre>
        </div>
        <div class="g-pair" id="g-prefs"></div>
        <div class="card" id="g-feed-card"></div>
        <div class="g-pair" id="g-loras"></div>
      </section>

      <section id="tab-ask" hidden>
        <div class="g-askbar">
          <div class="seg" id="g-mode" role="radiogroup" aria-label="What to do"></div>
          <span class="g-models" id="g-avail"></span>
        </div>
        <div id="g-stage"></div>
      </section>
"""

PROFILE = r"""
        <div class="card" id="g-profile-card" hidden>
          <h3>The structured profile</h3>
          <p class="hint">What the feed builder and the reel writer use: how much this persona leans towards each topic
            (0 to 1), which side of political stories they prefer, and which languages. The Corpus tab shows how the
            feed came out.</p>
          <div id="g-profile"></div>
        </div>
"""

JS = r"""
// ── The Gradient: the corpus, training, and asking the personas ──

const G = {overview: null, models: null, mode: store.get("g.mode", "chat"), browse: {account: null, offset: 0, items: []},
           job: null, poll: null, modelsTimer: null, history: [], loraAccount: null};
const gPct = (x) => (x == null ? "–" : `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`);
const gNum = (x) => (x == null ? "–" : Number(x).toLocaleString());
const gFix = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));
const VIA = {session: "Instagram sessions", generator: "Written for its feed", recommender: "Served by the recommender",
             demo_session: "Shown in DEMO simulated sessions"};
const STANCES = {supportive: "Supportive of the government", critical: "Critical of it", neutral: "Neutral reporting",
                 not_political: "Not political"};

function gradientShow(name) {
  clearTimeout(G.modelsTimer);
  if (name === "corpus") gLoadCorpus();
  if (name === "training") gLoadModels();
  if (name === "ask") gAskInit();
  if (name === "bias") gProfileCard();
}

function gSeg(id, options, value, onPick) {
  const el = $(id);
  el.replaceChildren(...options.map(([v, label]) => {
    const input = h("input", {type: "radio", name: id, value: v, checked: v === value});
    input.onchange = () => { el.querySelectorAll("label").forEach((l) => l.classList.toggle("on", l.contains(input))); onPick(v); };
    return h("label", {class: v === value ? "on" : null}, input, label);
  }));
}

// ── Corpus ──

async function gLoadCorpus() {
  try { G.overview = await api("/api/gradient/overview"); } catch (e) { toast(e.message, true); return; }
  const o = G.overview;
  $("g-stats").replaceChildren(
    h("div", {class: "g-stat"}, h("b", {text: gNum(o.items)}), h("span", {text: "unique items"})),
    h("div", {class: "g-stat"}, h("b", {text: gNum(o.groups)}), h("span", {text: "stories (split groups)"})),
    ...o.personas.map((p) => h("div", {class: "g-stat"}, h("b", {text: gNum(p.feed_total)}),
      h("span", {text: `in @${p.account}'s feed`}))),
    h("div", {class: "g-stat"}, h("b", {text: gNum(o.batches.length)}), h("span", {text: "ingestion batches"})));
  $("g-origins").replaceChildren(h("tr", {}, ...["Where it came from", "Items", "Share"].map((t, i) =>
      h("th", {class: i ? "num" : null, text: t}))),
    ...o.origins.map((r) => h("tr", {}, h("td", {text: r.name}), h("td", {class: "num", text: gNum(r.n)}),
      h("td", {class: "num", text: gPct(r.n / o.items)}))),
    h("tr", {}, h("td", {text: "Languages"}), h("td", {colspan: 2, class: "num", text: Object.entries(o.languages)
      .sort((a, b) => b[1] - a[1]).map(([k, v]) => `${{en: "English", hi: "Hindi", hinglish: "Hinglish"}[k] || k} ${gNum(v)}`).join(" · ")})),
    h("tr", {}, h("td", {text: "Train / validation / test"}), h("td", {colspan: 2, class: "num",
      text: ["train", "val", "test"].map((k) => gNum(o.splits[k] || 0)).join(" / ")})));
  $("g-personas").replaceChildren(...o.personas.map(gPersonaCard));
  $("g-batches").replaceChildren(h("tr", {}, ...["Step", "When", "What", "Added"].map((t, i) =>
      h("th", {class: i === 0 || i === 3 ? "num" : null, text: t}))),
    ...o.batches.map((b) => h("tr", {}, h("td", {class: "num", text: b.step}), h("td", {text: localTime(b.created_at)}),
      h("td", {text: b.note || b.kind}), h("td", {class: "num", text: gBatchCount(b.counts)}))));
  if (!G.browse.account) G.browse.account = o.personas[0].account;
  gSeg("g-browse-account", o.personas.map((p) => [p.account, "@" + p.account]), G.browse.account,
       (v) => { G.browse.account = v; gBrowse(true); });
  const sel = $("g-browse-origin");
  if (sel.options.length === 1) o.origins.forEach((r) => sel.append(h("option", {value: r.origin, text: r.name})));
  gBrowse(true);
}

function gBatchCount(counts) {
  let total = 0;
  const walk = (v) => { if (typeof v === "number") total += v; else if (v && typeof v === "object") Object.entries(v)
    .forEach(([k, x]) => { if (!["usd", "failed_briefs"].includes(k)) walk(x); }); };
  walk(counts);
  return gNum(total);
}

function gPersonaCard(p) {
  const top = p.topics.slice(0, 16), max = Math.max(...top.map((t) => t.share), 0.01);
  const stance = Object.entries(p.stances).sort((a, b) => b[1] - a[1]);
  return h("div", {class: "card"},
    h("div", {class: "status"}, h("h3", {text: "@" + p.account}), h("span", {class: "badge", text: p.name})),
    h("p", {class: "lede", text: p.summary || ""}),
    h("div", {class: "g-chips"}, ...p.secondary.map((s) => h("span", {text: s}))),
    h("div", {class: "g-mini"}, ...Object.entries(p.feed).map(([via, n]) => h("span", {}, h("b", {text: gNum(n)}), " " + (VIA[via] || via).toLowerCase())),
      h("span", {}, h("b", {text: gPct(p.feed_would_watch / Math.max(p.feed_labelled, 1))}), " of its feed it would watch (Jev)"),
      h("span", {}, h("b", {text: gPct(p.would_watch / Math.max(p.labelled, 1))}), ` of all ${gNum(p.labelled)} judged items`),
      h("span", {}, h("b", {text: gNum(p.takes)}), " reactions in its own words")),
    h("h4", {class: "sub", text: "What its feed is made of"}),
    h("p", {class: "hint", text: "Share of the feed by topic; the number on the right is how much the persona leans " +
      "towards that topic (its profile weight, 0 to 1). Darker bars are political topics."}),
    h("div", {class: "g-bars"}, ...top.map((t) => h("div", {class: "g-bar" + (t.political ? " political" : ""),
        title: `${t.name}: ${gNum(t.n)} items (${gPct(t.share)})`},
      h("span", {class: "name", text: t.name}), h("span", {class: "track2"}, h("i", {style: `width:${(t.share / max * 100).toFixed(1)}%`})),
      h("span", {class: "n", text: gPct(t.share)}), h("span", {class: "w", text: t.weight == null ? "" : `lean ${t.weight.toFixed(2)}`})))),
    h("h4", {class: "sub", text: "Political stance of its feed"}),
    h("div", {class: "g-mini"}, ...stance.map(([k, n]) => h("span", {}, h("b", {text: gPct(n / p.feed_total)}), " " + (STANCES[k] || k).toLowerCase()))),
    h("h4", {class: "sub", text: "Languages"}),
    h("div", {class: "g-mini"}, ...Object.entries(p.languages).sort((a, b) => b[1] - a[1]).map(([k, n]) =>
      h("span", {}, h("b", {text: gPct(n / p.feed_total)}), " " + ({en: "English", hi: "Hindi", hinglish: "Hinglish"}[k] || k)))));
}

async function gBrowse(reset) {
  if (reset) { G.browse.offset = 0; G.browse.items = []; }
  const q = new URLSearchParams({account: G.browse.account, offset: G.browse.offset, origin: $("g-browse-origin").value,
                                 q: $("g-browse-q").value.trim()});
  let d;
  try { d = await api("/api/gradient/items?" + q); } catch (e) { toast(e.message, true); return; }
  G.browse.items.push(...d.items);
  G.browse.offset += d.items.length;
  $("g-browse-count").textContent = `${gNum(d.total)} items`;
  $("g-browse").replaceChildren(...G.browse.items.map(gItemRow));
  $("g-browse-more").hidden = G.browse.offset >= d.total;
}

function gItemRow(it) {
  const verdict = it.jev_label == null ? "not judged yet" : it.jev_label ? "would watch" : "would scroll past";
  return h("div", {class: "g-item"},
    h("div", {class: "head"}, it.source_url ? link(it.source_url, it.creator || it.title || it.origin_name) :
        h("span", {class: "who", text: it.creator || it.title || ""}),
      h("span", {class: "tag-cat", text: it.origin_name}),
      it.topic ? h("span", {class: "badge", text: it.topic.replace(/_/g, " ")}) : null,
      it.stance && it.stance !== "not_political" ? h("span", {class: "badge", text: it.stance}) : null,
      h("span", {class: "score", text: `Jev ${gFix(it.jev_score, 2)}: ${verdict}`}),
      it.shown_at ? h("span", {class: "score", text: "shown " + localTime(it.shown_at)}) : null),
    h("div", {class: "body", text: it.text}),
    it.take ? h("div", {class: "take", text: `${it.action}: “${it.take}”`}) : null);
}

async function gProfileCard() {
  const card = $("g-profile-card");
  if (!card) return;
  if (!G.overview) { try { G.overview = await api("/api/gradient/overview"); } catch { return; } }
  const p = G.overview.personas.find((x) => x.account === account());
  card.hidden = !p;
  if (!p) return;
  const topics = [...p.topics].filter((t) => t.weight != null).sort((a, b) => b.weight - a.weight);
  $("g-profile").replaceChildren(
    h("div", {class: "g-bars"}, ...topics.map((t) => h("div", {class: "g-bar" + (t.political ? " political" : "")},
      h("span", {class: "name", text: t.name}), h("span", {class: "track2"}, h("i", {style: `width:${t.weight * 100}%`})),
      h("span", {class: "n", text: t.weight.toFixed(2)}), h("span", {class: "w", text: gPct(t.share) + " of feed"})))),
    h("div", {class: "g-mini"}, ...Object.entries(p.stance_prefs).map(([k, v]) => h("span", {}, h("b", {text: v.toFixed(2)}), " " + (STANCES[k] || k).toLowerCase()))),
    h("div", {class: "g-mini"}, ...Object.entries(p.languages_pref).map(([k, v]) => h("span", {}, h("b", {text: v.toFixed(2)}), " " + ({en: "English", hi: "Hindi", hinglish: "Hinglish"}[k] || k)))));
}

// ── Training ──

async function gLoadModels() {
  try { G.models = await api("/api/gradient/models"); } catch (e) { toast(e.message, true); return; }
  const m = G.models, runs = m.runs;
  const accounts = [...new Set(runs.map((r) => r.account).filter((a) => a !== "shared"))];
  const all = accounts.length ? accounts : ["mlproject01", "mlproject02"];
  if (!G.loraAccount) G.loraAccount = all[0];
  gSeg("g-lora-account", all.map((a) => [a, "@" + a]), G.loraAccount, (v) => { G.loraAccount = v; });
  const job = m.job, running = job && job.code == null;
  $("g-job-state").textContent = job ? `${job.label}: ${running ? "running…" : job.code === 0 ? "finished" : "stopped or failed (exit " + job.code + ")"}` : "";
  $("g-job-log").hidden = !job;
  if (job) { const log = $("g-job-log"), end = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.textContent = job.lines.join("\n"); if (end) log.scrollTop = log.scrollHeight; }
  ["g-train-update", "g-train-prefs", "g-train-lora", "g-train-lora-resume"].forEach((id) => { $(id).disabled = running; });
  $("g-prefs").replaceChildren(...all.map((a) => gPrefCard(a, runs.filter((r) => r.account === a && r.kind === "preference"))));
  const feed = runs.filter((r) => r.kind === "feed");
  $("g-feed-card").replaceChildren(...gPrefBody("Whose feed is it?", feed, "Which account's feed a reel came from: the " +
    "algorithm's side. Trained on items that are in exactly one feed."));
  $("g-loras").replaceChildren(...all.map((a) => gLoraCard(a, runs.filter((r) => r.account === a && r.kind === "lora"))));
  if (running && tab === "training") G.modelsTimer = setTimeout(gLoadModels, 1500);
}

function gPrefCard(account, runs) {
  return h("div", {class: "card"}, ...gPrefBody(`@${account}: preference model`, runs,
    "Will this persona watch it to the end and engage? The label is Jev's judgement for this persona; the model reads only the reel's text."));
}

function gPrefBody(title, runs, about) {
  if (!runs.length) return [h("h3", {text: title}), h("p", {class: "hint", text: "Not trained yet: press “Update the corpus and retrain”."})];
  const r = runs[runs.length - 1].report, d = r.data, c = r.candidates[r.chosen];
  const cols = ["Model", "Val PR-AUC", "PR-AUC", "ROC-AUC", "Prec.", "Recall", "F1", "P@50", "Brier"];
  const SHORT = {majority: "Majority", rules: "Quick rules", tfidf: "TF-IDF + LR", embedding: "Embeddings + LR"};
  const rowOf = (k, v) => h("tr", {class: k === r.chosen ? "g-chosen" : null}, h("td", {title: v.name, text: SHORT[k] + (k === r.chosen ? " ✓" : "")}),
    ...[v.val.pr_auc, v.test.pr_auc, v.test.roc_auc, v.test.precision, v.test.recall, v.test.f1, v.test.precision_at_50, v.test.brier]
      .map((x) => h("td", {class: "num", text: gFix(x)})));
  const conf = c.test.confusion;
  return [
    h("div", {class: "status"}, h("h3", {text: title}), h("span", {class: "badge ok", text: `v${r.version}`})),
    h("p", {class: "hint", text: about}),
    h("div", {class: "g-mini"}, h("span", {}, h("b", {text: gNum(d.train_n)}), " train"), h("span", {}, h("b", {text: gNum(d.val_n)}), " validation"),
      h("span", {}, h("b", {text: gNum(d.test_n)}), " test"), h("span", {}, h("b", {text: gPct(d.test_positives / Math.max(d.test_n, 1))}), " positive in test")),
    h("div", {class: "table-scroll"}, h("table", {class: "data"}, h("tr", {}, ...cols.map((t, i) => h("th", {class: i ? "num" : null, text: t}))),
      ...Object.entries(r.candidates).map(([k, v]) => rowOf(k, v)))),
    h("p", {class: "hint", text: `✓ chosen. Every column but the first is on the test split. LR: logistic regression; P@50: precision of the 50 it's surest about. Chosen on validation (${r.chosen_because}); decision threshold ${c.threshold}, also picked on validation. Test numbers are reported, never tuned on.`}),
    h("table", {class: "g-conf"}, h("tr", {}, h("th", {text: "test"}), h("th", {text: "predicted no"}), h("th", {text: "predicted yes"})),
      h("tr", {}, h("th", {text: "actually no"}), h("td", {text: gNum(conf.tn)}), h("td", {text: gNum(conf.fp)})),
      h("tr", {}, h("th", {text: "actually yes"}), h("td", {text: gNum(conf.fn)}), h("td", {text: gNum(conf.tp)}))),
    h("h4", {class: "sub", text: "What it leans on (TF-IDF words)"}),
    h("p", {class: "g-words"}, h("b", {text: "TOWARDS YES"}), r.top_features.towards_yes.join(", ")),
    h("p", {class: "g-words"}, h("b", {text: "TOWARDS NO"}), r.top_features.towards_no.join(", ")),
    h("h4", {class: "sub", text: "By source (test)"}),
    h("div", {class: "table-scroll"}, h("table", {class: "data"}, h("tr", {}, ...["Slice", "Items", "PR-AUC", "F1"].map((t, i) => h("th", {class: i ? "num" : null, text: t}))),
      ...[...r.slices.origin, ...r.slices.language].map((s) => h("tr", {}, h("td", {text: s.slice}), h("td", {class: "num", text: gNum(s.n)}),
        h("td", {class: "num", text: s.note ? "–" : gFix(s.pr_auc)}), h("td", {class: "num", text: s.note ? "–" : gFix(s.f1)}))))),
    h("h4", {class: "sub", text: "Versions (same frozen evaluation set)"}),
    h("div", {class: "table-scroll"}, h("table", {class: "data"}, h("tr", {}, ...["Version", "Trained", "Items", "Frozen-eval PR-AUC", "Frozen-eval F1"].map((t, i) => h("th", {class: i > 1 ? "num" : null, text: t}))),
      ...runs.map((x) => h("tr", {}, h("td", {text: "v" + x.version}), h("td", {text: localTime(x.created_at)}), h("td", {class: "num", text: gNum(x.items)}),
        h("td", {class: "num", text: gFix(x.report.frozen_eval && x.report.frozen_eval.pr_auc)}), h("td", {class: "num", text: gFix(x.report.frozen_eval && x.report.frozen_eval.f1)}))))),
    r.errors.length ? h("details", {}, h("summary", {class: "hint", text: "Where it's wrong (test examples)"}),
      ...r.errors.map((e) => h("p", {class: "hint", text: `${e.kind} (p=${e.p}, Jev ${gFix(e.teacher, 2)}, ${e.origin}): ${e.text}`}))) : null,
  ];
}

function gLoraCard(account, runs) {
  if (!runs.length) return h("div", {class: "card"}, h("h3", {text: `@${account}: persona language model`}),
    h("p", {class: "hint", text: "No adapter yet. Pick the account above and press “Train a new adapter” (about 15–25 minutes on this Mac)."}));
  const r = runs[runs.length - 1].report, e = r.eval;
  return h("div", {class: "card"},
    h("div", {class: "status"}, h("h3", {text: `@${account}: persona language model`}), h("span", {class: "badge ok", text: `lora-v${r.version}`})),
    h("p", {class: "hint", text: `${r.base_model} + a rank-${r.lora.rank} LoRA adapter, ${gNum(r.examples.train)} training examples from this feed, ${r.iters} steps` +
      (r.resumed_from ? `, continued from lora-v${r.resumed_from}` : "") + `, ${r.minutes} min.`}),
    h("div", {class: "g-mini"},
      h("span", {}, h("b", {text: gFix(e.base.ppl, 2)}), " held-out perplexity, base model"),
      h("span", {}, h("b", {text: gFix(e.adapter.ppl, 2)}), " with this adapter"),
      e.other ? h("span", {}, h("b", {text: gFix(e.other.ppl, 2)}), ` this adapter on @${e.other.account}'s examples`) : null),
    h("p", {class: "hint", text: "Lower is better: how surprised the model is by this persona's own held-out words. The adapter should beat the base model on its persona, and fit the other persona worse than its own."}),
    gCurve(r.train_loss, r.val_loss),
    runs.length > 1 ? h("div", {class: "table-scroll"}, h("table", {class: "data"}, h("tr", {}, ...["Version", "Trained", "Examples", "Base ppl", "Adapter ppl"].map((t, i) => h("th", {class: i > 1 ? "num" : null, text: t}))),
      ...runs.map((x) => h("tr", {}, h("td", {text: "lora-v" + x.version}), h("td", {text: localTime(x.created_at)}), h("td", {class: "num", text: gNum(x.report.examples.train)}),
        h("td", {class: "num", text: gFix(x.report.eval.base.ppl, 2)}), h("td", {class: "num", text: gFix(x.report.eval.adapter.ppl, 2)}))))) : null);
}

function gCurve(train, val) {
  if (!train || !train.length) return null;
  const W = 520, H = 180, P = {l: 40, r: 10, t: 10, b: 24};
  const all = [...train, ...(val || [])], maxIt = Math.max(...all.map((p) => p.iter)), hi = Math.max(...all.map((p) => p.loss)), lo = Math.min(...all.map((p) => p.loss));
  const x = (i) => P.l + (i / maxIt) * (W - P.l - P.r), y = (l) => P.t + (1 - (l - lo) / Math.max(hi - lo, 1e-6)) * (H - P.t - P.b);
  const line = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.iter).toFixed(1)},${y(p.loss).toFixed(1)}`).join("");
  const s = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Training and validation loss by step"});
  for (const f of [0, 0.5, 1]) { const l = lo + f * (hi - lo); s.append(svg("line", {class: "grid", x1: P.l, x2: W - P.r, y1: y(l), y2: y(l)}),
    svg("text", {class: "axis", x: 4, y: y(l) + 4}, l.toFixed(2))); }
  s.append(svg("text", {class: "axis", x: W - P.r - 60, y: H - 6}, `step ${maxIt}`));
  s.append(svg("path", {class: "train", d: line(train)}));
  if (val && val.length) { s.append(svg("path", {class: "val", d: line(val)})); val.forEach((p) => s.append(svg("circle", {cx: x(p.iter), cy: y(p.loss), r: 3}))); }
  return h("div", {class: "g-curve"}, s, h("div", {class: "legend"}, h("span", {text: "solid: training loss"}), h("span", {text: "dashed: validation loss"})));
}

async function gTrain(body) {
  try { G.models = await api("/api/gradient/train", body); toast("Started."); } catch (e) { toast(e.message, true); return; }
  gLoadModels();
}

// ── Ask ──

const G_MODES = [["chat", "Chat"], ["debate", "Debate"], ["compare", "One question, two worldviews"], ["predict", "Feed predictor"]];
const G_TRY = {
  chat: {mlproject01: ["Are the new expressways worth the tolls?", "Kya chal raha hai cricket mein?", "What's the latest at the border?",
                       "Any good phone launches lately?"],
         mlproject02: ["What's going on with the gig workers' strike?", "How bad is the air in Delhi right now?",
                       "Any good indie music lately?", "What do you think of the new labour codes?"]},
  debate: ["The Agniveer scheme", "Delhi's air pollution", "Vande Bharat trains and fares", "Internet shutdowns", "MGNREGA wages"],
  compare: ["Is the government handling unemployment well?", "What's happening at the border with China?",
            "Should India host the Olympics?", "Are government schools getting better?"],
  predict: ["Naya expressway khul gaya! Delhi se Jaipur ab sirf 3 ghante. Truck drivers bhi khush, toll thoda zyada hai par time bachta hai. #infrastructure",
            "Delivery riders in Bengaluru went offline today after the app cut per-order pay again. 'We work 12 hours and take home less than minimum wage,' one rider said. #gigworkers"],
};
const G_WHO = {mlproject01: {tile: "01", cls: "p1"}, mlproject02: {tile: "02", cls: "p2"}};
const SEND_ICON = "M2.01 21L23 12 2.01 3 2 10l15 2-15 2z";
G.chats = (() => { try { return JSON.parse(localStorage.getItem("feedlab.g.chats")) || {}; } catch { return {}; } })();
G.chatWith = store.get("g.chat", "mlproject01");
G.debate = null;
G.open = new Set();  // source cards the viewer opened

function gSaveChats() {
  try { localStorage.setItem("feedlab.g.chats", JSON.stringify(G.chats, (k, v) => (k === "typing" ? undefined : v))); } catch {}
}

function gPersonaName(account) {
  const p = G.overview && G.overview.personas.find((x) => x.account === account);
  return p ? p.name : "";
}

function gAskInit() {
  gSeg("g-mode", G_MODES, G.mode, (v) => { G.mode = v; store.set("g.mode", v); gStage(); });
  if (!G.overview) api("/api/gradient/overview").then((o) => { G.overview = o; gStage(); }).catch(() => {});
  api("/api/gradient/models").then((m) => { G.models = m; gAvail(); }).catch(() => {});
  gAvail();
  gStage();
}

function gAvail() {
  const t = G.models && G.models.talk, el = $("g-avail");
  if (!t) { el.textContent = ""; return; }
  el.classList.toggle("bad", !t.ok);
  el.textContent = !t.ok ? `The persona models can't run yet: ${t.why}` :
    Object.entries(t.adapters).map(([a, v]) => `@${a} ${v || "base model"}`).join(" · ") + ` · ${t.base.split("/").pop()}`;
}

function gBusy() { return G.job && (G.job.status === "queued" || G.job.status === "running"); }

function gStage() {
  const stage = $("g-stage");
  if (G.mode === "chat") return stage.replaceChildren(gChatWindow());
  if (G.mode === "debate") return stage.replaceChildren(gDebateWindow());
  const input = G.mode === "predict" ? h("textarea", {id: "g-text", rows: 5, placeholder: "Paste a reel's caption, the text on it, or what's said in it"})
    : h("input", {type: "text", id: "g-text", maxlength: 500, placeholder: "Ask both personas the same question"});
  if (G.mode !== "predict") input.onkeydown = (e) => { if (e.key === "Enter") gGo(); };
  const go = h("button", {class: "btn primary", id: "g-go", text: G.mode === "predict" ? "Predict" : "Ask both"});
  go.onclick = gGo;
  stage.replaceChildren(h("div", {class: "card"},
    h("p", {class: "lede", text: G.mode === "predict" ? "Paste any reel: each persona's preference model says whether it would watch it, and the whose-feed model says which feed it would land in."
      : "Both personas answer the same question, each from its own feed, with how far apart their answers and their evidence are."}),
    h("div", {class: "g-ask-form", style: "margin-top:14px"}, input,
      h("div", {class: "g-try"}, h("span", {class: "hint", text: "Try:"}), ...G_TRY[G.mode].map((t) => {
        const b = h("button", {class: "linkish", text: t.length > 60 ? t.slice(0, 58) + "…" : t});
        b.onclick = () => { $("g-text").value = t; gGo(); };
        return b;
      })),
      h("div", {class: "actions", style: "margin-top:4px"}, go, h("span", {class: "hint", id: "g-busy"})))),
    h("div", {id: "g-out"}));
}

// one window frame: head, the log, the composer
function gWindow(head, log, composer, foot) {
  return h("div", {class: "g-win"}, h("div", {class: "g-win-head"}, ...head),
    h("div", {class: "g-thread", id: "g-log", role: "log", "aria-live": "polite"}, ...log), composer,
    foot ? h("p", {class: "g-foot", text: foot}) : null);
}

function gComposer(placeholder, onSend, extra) {
  const ta = h("textarea", {id: "g-input", rows: 1, maxlength: 500, placeholder, "aria-label": placeholder});
  const fit = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight + 4, 180) + "px"; };
  ta.oninput = fit;
  ta.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!gBusy()) onSend(); } };
  const busy = gBusy();
  const send = h("button", {class: "g-send" + (busy ? " stop" : ""), id: "g-send", title: busy ? "Stop" : "Send", "aria-label": busy ? "Stop" : "Send"});
  if (busy) send.textContent = "Stop";
  else send.append(svg("svg", {viewBox: "0 0 24 24", "aria-hidden": "true"}), "");
  if (!busy) send.firstChild.append(svg("path", {d: SEND_ICON}));
  send.onclick = () => (gBusy() ? gStop() : onSend());
  return h("div", {class: "g-composer"}, ...(extra || []), ta, send);
}

function gAvatar(account, extra) {
  const w = G_WHO[account] || {tile: "?", cls: "mod"};
  return h("div", {class: `g-avatar ${w.cls}${extra ? " " + extra : ""}`, "aria-hidden": "true", text: w.tile});
}

function gScroll(force) {
  const log = $("g-log");
  if (!log) return;
  if (force || log.scrollHeight - log.scrollTop - log.clientHeight < 140) log.scrollTop = log.scrollHeight;
}

// a persona's reply: avatar, name, the text with its citations, then its sources
function gReply(m, key, right) {
  const body = h("div", {class: "g-bubble"});
  if (m.typing && !m.text) body.append(h("span", {class: "g-dots", "aria-label": "typing"}, h("i"), h("i"), h("i")));
  else {
    String(m.text || "").split(/\[(\d{1,2})\]/).forEach((part, i) => {
      if (i % 2 === 0) return body.append(part);
      const b = h("button", {class: "g-cite", text: part, title: "Show the reel"});
      b.onclick = () => { G.open.add(`${key}:${part}`); gRedraw(); };
      body.append(b);
    });
    if (m.typing) body.classList.add("g-caret");
  }
  const srcs = (m.evidence || []).length && !m.typing ? h("div", {class: "g-srcs"}, ...m.evidence.map((e, i) => {
    const n = i + 1, id = `${key}:${n}`, cited = (m.cited || []).includes(n);
    const chip = h("button", {class: "g-chip" + (cited ? " cited" : "") + (G.open.has(id) ? " open" : ""),
      text: `[${n}] ${e.creator || e.title || e.origin_name}`, title: e.text.slice(0, 200)});
    chip.onclick = () => { G.open.has(id) ? G.open.delete(id) : G.open.add(id); gRedraw(); };
    return chip;
  })) : null;
  const opened = (m.evidence || []).map((e, i) => G.open.has(`${key}:${i + 1}`) ? h("div", {class: "g-src"},
    h("div", {}, h("b", {text: `[${i + 1}] `}), e.source_url ? link(e.source_url, e.creator || e.title || "source") : h("b", {text: e.creator || e.title || ""}),
      " ", h("span", {class: "tag-cat", text: e.origin_name})),
    h("div", {text: e.text.length > 420 ? e.text.slice(0, 418) + "…" : e.text}),
    h("div", {class: "sub", text: `relevance ${gFix(e.similarity, 2)} · this persona's preference ${gFix(e.preference, 2)}` +
      (e.moved_up_by_model ? " · ranked up by its preference model" : "")})) : null).filter(Boolean);
  const name = `@${m.account}` + (m.adapter ? ` · ${m.adapter}` : "") + (m.typing ? " · typing…" : "") +
    (!m.typing && m.grounded === false ? " · nothing in its feed on this" : "") + (m.error ? " · " + m.error : "");
  return h("div", {class: "g-msg" + (right ? " right" : "")}, gAvatar(m.account),
    h("div", {class: "g-body"}, h("div", {class: "g-name", text: name}), body, srcs, ...opened));
}

function gRedraw() {  // draw again, following new messages (unless scrolled up) and keeping whatever is being typed
  const log = $("g-log"), input = $("g-input");
  const follow = !log || log.scrollHeight - log.scrollTop - log.clientHeight < 140, keep = log ? log.scrollTop : 0;
  const typed = input ? input.value : "", focused = input && document.activeElement === input;
  const caret = input ? [input.selectionStart, input.selectionEnd] : null;
  gStage();
  const again = $("g-log"), box = $("g-input");
  if (again) again.scrollTop = follow ? again.scrollHeight : keep;
  if (box && typed) { box.value = typed; box.oninput(); }
  if (box && focused) { box.focus(); box.setSelectionRange(...caret); }
}

// ── Chat ──

function gChatWindow() {
  const who = G.chatWith, msgs = G.chats[who] || [];
  const contacts = ["mlproject01", "mlproject02"].map((a) => {
    const b = h("button", {class: "g-contact" + (a === who ? " on" : ""), "aria-pressed": a === who ? "true" : "false"},
      gAvatar(a), h("span", {}, h("b", {text: "@" + a}), h("small", {text: gPersonaName(a)})));
    b.onclick = () => { if (gBusy()) return toast("Wait for the answer, or stop it.", true); G.chatWith = a; store.set("g.chat", a); gStage(); gScroll(true); };
    return b;
  });
  const clear = h("button", {class: "text-btn", text: "New chat"});
  clear.onclick = () => { if (gBusy()) return; G.chats[who] = []; G.open.clear(); gSaveChats(); gStage(); };
  const log = msgs.length ? msgs.map((m, i) => m.role === "user"
      ? h("div", {class: "g-msg user"}, h("div", {class: "g-body"}, h("div", {class: "g-bubble", text: m.text})))
      : gReply(m, `${who}-${i}`, false))
    : [h("div", {class: "g-empty"}, gAvatar(who), h("h3", {text: "@" + who}),
        h("p", {text: (G.overview && (G.overview.personas.find((x) => x.account === who) || {}).summary) ||
          "Ask anything. It answers from what its feed has shown it."}),
        h("div", {class: "g-suggest"}, ...G_TRY.chat[who].map((t) => {
          const b = h("button", {text: t});
          b.onclick = () => gSendChat(t);
          return b;
        })))];
  const win = gWindow([...contacts, h("span", {class: "spacer"}), clear], log,
    gComposer(`Message @${who}`, () => gSendChat()),
    `@${who} answers only from its own feed. Click a number or a source to see the reel.`);
  setTimeout(() => gScroll(true), 0);
  return win;
}

async function gSendChat(preset) {
  const who = G.chatWith, input = $("g-input"), text = (preset || (input && input.value) || "").trim();
  if (!text || gBusy()) return;
  const msgs = (G.chats[who] = G.chats[who] || []);
  const history = msgs.filter((m) => m.text && !m.error).slice(-6).map((m) => ({role: m.role, text: m.text}));
  msgs.push({role: "user", text}, {role: "assistant", account: who, text: "", typing: true, evidence: []});
  gStage();
  if ($("g-input")) $("g-input").focus();
  gScroll(true);
  try { G.job = await api("/api/gradient/talk/start", {kind: "chat", account: who, text, history}); }
  catch (e) { Object.assign(msgs[msgs.length - 1], {typing: false, error: e.message}); gStage(); return; }
  gStage();
  gPollJob();
}

// ── Debate ──

function gDebateWindow() {
  const d = G.debate, rounds = Number(store.get("g.rounds", 2));
  const head = [gAvatar("mlproject01"), h("b", {class: "g-title", text: "@mlproject01"}), h("span", {class: "g-vs", text: "VS"}),
    h("b", {class: "g-title", text: "@mlproject02"}), gAvatar("mlproject02"), h("span", {class: "spacer"})];
  const fresh = h("button", {class: "text-btn", text: "New debate"});
  fresh.onclick = () => { if (gBusy()) return; G.debate = null; G.open.clear(); gStage(); };
  if (d) head.push(fresh);
  const log = [];
  if (!d) {
    log.push(h("div", {class: "g-empty"}, h("div", {style: "display:flex;gap:10px;justify-content:center;margin-bottom:14px"},
        gAvatar("mlproject01", ""), gAvatar("mlproject02", "")),
      h("h3", {text: "Give them something to argue about"}),
      h("p", {text: "Each persona argues from its own feed, answers the other, and a neutral moderator sums up."}),
      h("div", {class: "g-suggest"}, ...G_TRY.debate.map((t) => { const b = h("button", {text: t}); b.onclick = () => gStartDebate(t); return b; }))));
  } else {
    log.push(h("div", {class: "g-topic"}, h("small", {text: `DEBATE · ${d.rounds} ROUND${d.rounds === 1 ? "" : "S"}`}), h("b", {text: d.topic})));
    let round = 0;
    (d.events || []).forEach((e, i) => {
      if (e.type === "turn") {
        if (e.round !== round) { round = e.round; log.push(h("div", {class: "g-round", text: `ROUND ${round}`})); }
        log.push(gReply(e, `d-${i}`, e.account === "mlproject02"));
      } else if (e.type === "moderator") {
        log.push(h("div", {class: "g-round", text: "MODERATOR"}),
          h("div", {class: "g-mod"}, h("div", {class: "g-name", text: "The base model, with no persona" + (e.typing ? " · writing…" : "")}),
            e.typing && !e.text ? h("span", {class: "g-dots"}, h("i"), h("i"), h("i")) : gModText(e.text, e.typing)));
      }
    });
    if (d.error) log.push(h("p", {class: "notice", text: d.error}));
    if (d.status === "stopped") log.push(h("div", {class: "g-round", text: "STOPPED"}));
  }
  const roundsSeg = h("div", {class: "seg small", id: "g-rounds", role: "radiogroup", "aria-label": "Rounds"});
  const win = gWindow(head, log, gComposer("Give them a topic to debate", () => gStartDebate(), [roundsSeg]),
    "Each side can only cite its own feed. The other side's words are an argument to answer, not evidence.");
  setTimeout(() => gSeg("g-rounds", [["1", "1 round"], ["2", "2 rounds"], ["3", "3 rounds"]], String(rounds), (v) => store.set("g.rounds", Number(v))), 0);
  return win;
}

function gModText(text, typing) {  // the moderator's summary: its headings bold, no markdown asterisks
  const box = h("div", {class: "text" + (typing ? " g-caret" : "")});
  String(text || "").split("\n").forEach((line, i) => {
    const heading = line.match(/^\s*(?:#+\s*)?\*{0,2}([^*]{3,60}?)\*{0,2}:?\*{0,2}\s*$/);
    if (i) box.append("\n");
    if (heading && /\*\*|^#/.test(line.trim())) box.append(h("b", {text: heading[1].replace(/:$/, "")}));
    else box.append(line.replace(/\*\*/g, ""));
  });
  return box;
}

async function gStartDebate(preset) {
  const input = $("g-input"), topic = (preset || (input && input.value) || "").trim();
  if (!topic || gBusy()) return;
  const rounds = Number(store.get("g.rounds", 2));
  G.debate = {topic, rounds, events: [], status: "queued"};
  G.open.clear();
  gStage();
  try { G.job = await api("/api/gradient/talk/start", {kind: "debate", text: topic, rounds}); }
  catch (e) { G.debate.error = e.message; G.debate.status = "failed"; gStage(); return; }
  gStage();
  gPollJob();
}

// ── Two worldviews and the predictor ──

async function gGo() {
  const text = ($("g-text").value || "").trim();
  if (!text) { toast("Type something first.", true); return; }
  if (G.mode === "predict") return gPredict(text);
  try { G.job = await api("/api/gradient/talk/start", {kind: "compare", text}); } catch (e) { toast(e.message, true); return; }
  $("g-go").disabled = true;
  gPollJob();
}

async function gStop() { if (G.job) { try { await api("/api/gradient/talk/stop", {id: G.job.id}); } catch {} } }

async function gPollJob() {
  clearTimeout(G.poll);
  try { G.job = await api("/api/gradient/talk?id=" + G.job.id); } catch (e) { toast(e.message, true); return; }
  const j = G.job, busy = gBusy();
  if (j.kind === "chat") {
    const msgs = G.chats[j.params.account] || [], last = msgs[msgs.length - 1], ev = j.events[0];
    if (last && last.role === "assistant") {
      if (ev) Object.assign(last, ev);
      if (!busy) { last.typing = false; if (j.status === "failed") last.error = j.error; if (j.status === "stopped") last.error = "stopped"; }
    }
    if (!busy) gSaveChats();
  } else if (j.kind === "debate" && G.debate) {
    Object.assign(G.debate, {events: j.events, status: j.status, error: j.status === "failed" ? j.error : null});
  }
  if (j.kind === "compare" && G.mode === "compare") {
    $("g-busy").textContent = busy ? "The personas are answering…" : j.status === "failed" ? "Failed: " + j.error : "";
    gDrawCompare(j);
    if (!busy) $("g-go").disabled = false;
  } else if (tab === "ask" && G.mode === j.kind) {
    gRedraw();
  }
  if (busy) G.poll = setTimeout(gPollJob, 350);
  else if (j.status === "failed") toast(j.error, true);
}

function gCited(text, evidence, key) {  // the answer with each [n] a button that points at its reel
  const parts = String(text).split(/\[(\d{1,2})\]/);
  return parts.map((p, i) => {
    if (i % 2 === 0) return document.createTextNode(p);
    const b = h("button", {class: "g-cite", text: p, title: "Show reel " + p});
    b.onclick = () => { const el = document.getElementById(`${key}-ev-${p}`); if (el) { el.scrollIntoView({block: "nearest", behavior: "smooth"});
      el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1200); } };
    return b;
  });
}

function gEvidence(a, key) {
  if (!a.evidence.length) return h("p", {class: "hint", text: "Nothing in this persona's feed was close enough to count as evidence."});
  return h("details", {class: "g-evidence", open: true}, h("summary", {class: "hint",
      text: `Evidence from @${a.account}'s feed: ${a.evidence.length} reels retrieved, ${a.cited.length} cited` +
        (a.invalid_citations.length ? `, ${a.invalid_citations.length} invalid citation(s) removed` : "")}),
    ...a.evidence.map((e, i) => h("div", {class: "g-ev" + (a.cited.includes(i + 1) ? " cited" : ""), id: `${key}-ev-${i + 1}`},
      h("span", {class: "n", text: `[${i + 1}]`}),
      h("div", {}, h("div", {class: "head"}, e.source_url ? link(e.source_url, e.creator || e.title || "source") : h("b", {text: e.creator || e.title || ""}),
          " ", h("span", {class: "tag-cat", text: e.origin_name})),
        h("div", {text: e.text.length > 320 ? e.text.slice(0, 318) + "…" : e.text}),
        h("div", {class: "sub", text: `relevance ${gFix(e.similarity, 2)} · persona preference ${gFix(e.preference, 2)}` +
          (e.moved_up_by_model ? " · ranked up by the preference model" : "")})))));
}

function gAnswerBlock(a, key, question) {
  return h("div", {class: "g-answer"},
    question ? h("div", {class: "q", text: `You asked @${a.account}: ${question}`}) : null,
    h("div", {class: "text"}, ...gCited(a.text, a.evidence, key)),
    h("div", {class: "meta", text: `@${a.account} · ${a.adapter || "base model, no adapter"} · preference model ${a.preference_model || "none"}` +
      (a.grounded ? "" : " · no evidence found")}),
    gEvidence(a, key));
}

function gDrawCompare(j) {
  const answers = j.events.filter((e) => e.type === "answer"), g = j.events.find((e) => e.type === "gap");
  const gap = g ? h("div", {class: "card"}, h("h3", {text: "How far apart are they?"}),
    h("div", {class: "g-mini"}, h("span", {}, h("b", {text: gFix(g.answer_similarity, 2)}), " similarity of the two answers (1 = same meaning)"),
      h("span", {}, h("b", {text: gFix(g.evidence_divergence, 2)}), " divergence of their evidence's political stance (0 = same mix, 1 = nothing shared)")),
    ...Object.entries(g.stance_mix).map(([a, mix]) => h("div", {class: "g-mini"}, h("span", {text: `@${a}'s evidence:`}),
      ...Object.entries(mix).filter(([, v]) => v).map(([k, v]) => h("span", {}, h("b", {text: gPct(v)}), " " + (STANCES[k] || k).toLowerCase())))),
    ...Object.entries(g.cross_preference).map(([a, c]) => h("p", {class: "hint", text:
      `@${a}'s preference model: ${gPct(c.own)} likely to engage with its own evidence, ${gPct(c.other)} with the other persona's.`}))) : null;
  $("g-out").replaceChildren(h("div", {class: "g-pair"}, ...answers.map((a, i) => h("div", {class: "card"}, gAnswerBlock(a, `w${i}`, null)))), gap);
}

async function gPredict(text) {
  let d;
  try { d = await api("/api/gradient/predict", {text}); } catch (e) { toast(e.message, true); return; }
  const meter = (label, p, threshold) => h("div", {class: "g-meter"}, h("span", {text: label}),
    h("span", {class: "track2"}, h("i", {style: `width:${(p || 0) * 100}%`}), threshold != null ? h("em", {style: `left:${threshold * 100}%`, title: "decision threshold"}) : null),
    h("b", {text: gPct(p)}));
  const blocks = d.personas.map((p) => h("div", {class: "card"},
    h("div", {class: "status"}, h("h3", {text: "@" + p.account}), h("span", {class: "badge" + (p.p >= p.threshold ? " ok" : ""),
      text: p.p == null ? "no model yet" : p.p >= p.threshold ? "would watch to the end" : "would scroll past"})),
    meter("Chance it engages", p.p, p.threshold),
    h("p", {class: "hint", text: `${p.kind || ""}, ${p.model || ""}. The orange tick is the decision threshold, picked on validation.`}),
    h("h4", {class: "sub", text: "Closest reels in its feed"}),
    ...p.nearest.map((e) => h("div", {class: "g-item"}, h("div", {class: "head"}, h("span", {class: "who", text: e.creator || e.title || ""}),
      h("span", {class: "tag-cat", text: e.origin_name}), h("span", {class: "score", text: `relevance ${gFix(e.similarity, 2)}`})),
      h("div", {class: "body", text: e.text.slice(0, 260)})))));
  const feed = d.feed ? h("div", {class: "card"}, h("h3", {text: "Whose feed would it land in?"}),
    meter("@" + d.feed.first, d.feed.p_first), meter("@" + d.feed.second, 1 - d.feed.p_first),
    h("p", {class: "hint", text: `${d.feed.kind}, ${d.feed.model}: trained on which account's feed each reel was in.`})) : null;
  $("g-out").replaceChildren(feed, h("div", {class: "g-pair"}, ...blocks));
}

document.addEventListener("DOMContentLoaded", () => {
  $("g-browse-origin").onchange = () => gBrowse(true);
  let t = null;
  $("g-browse-q").oninput = () => { clearTimeout(t); t = setTimeout(() => gBrowse(true), 300); };
  $("g-browse-more").onclick = () => gBrowse(false);
  $("g-train-update").onclick = () => gTrain({what: "update"});
  $("g-train-prefs").onclick = () => gTrain({what: "prefs"});
  $("g-train-lora").onclick = () => gTrain({what: "lora", account: G.loraAccount, iters: Number($("g-lora-iters").value)});
  $("g-train-lora-resume").onclick = () => gTrain({what: "lora", account: G.loraAccount, iters: Number($("g-lora-iters").value), resume: true});
  $("account").addEventListener("change", () => { if (tab === "bias") gProfileCard(); });
});
"""


def inject(page: str, _llm_name: str = "") -> str:
    return (page.replace("/*GRADIENT_CSS*/", CSS).replace("<!--GRADIENT_NAV-->", NAV)
            .replace("<!--GRADIENT_ISLAND-->", "").replace("<!--GRADIENT_PROFILE-->", PROFILE)
            .replace("<!--GRADIENT_SECTIONS-->", SECTIONS).replace("//GRADIENT_JS", JS))
