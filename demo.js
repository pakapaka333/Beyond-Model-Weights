/* demo.js — Beyond Model Weights interactive demo (vanilla JS, no dependencies) */
(function () {
  'use strict';

  /* ════════ Strings (EN / JA). Paper title, model/method names and BibTeX stay English. ════════ */
  var STR = {
    en: {
      back: '← Back to Top', kicker: 'Preprint · 2026',
      aff1: 'Institute of Science Tokyo', aff2: 'University of Liverpool', aff4: 'AIST', aff5: 'NII LLMC',
      pillArxiv: 'arXiv (coming soon)', pillCode: 'Code',
      hook: 'Fairness in diffusion language models depends on how you sample.',
      tldrH: 'TL;DR',
      f1q: 'Do sampling configurations change a DLM’s social bias?',
      f1a: 'Yes — both the score and the ranking among models.',
      f1d: 'For every model, the bias score moves by more than 40% across 24 configurations, and the least biased model changes too.',
      f2q: 'What counts as successful debiasing?',
      f2a: 'Lowering the bias magnitude and suppressing its variation.',
      f2d: 'Debiasing methods for autoregressive language models (ARLMs) also lower the bias of DLMs, but the variation across sampling configurations largely remains. The magnitude and the variation must be assessed together.',
      f3q: 'How comprehensive is this study?',
      f3a: '11 bias categories × 3 DLMs × 24 sampling configurations × 9 training states',
      f3d: 'Open-BBQ (10k items, 11 categories) on LLaDA-8B, LLaDA-1.5 and Dream-7B over remasking strategy × block length × TPS, for the untuned model and 8 SFT / dOPSD variants; utility on GSM8K, IFEval and MMLU.',
      prelimH: 'Background: Sampling configurations specific to DLMs',
      prelimLede: 'DLM generation is governed mainly by three DLM-specific settings that have no counterpart in ARLMs.',
      figAlt: 'One rollout of a masked diffusion language model with block length 4, low-confidence remasking and two tokens per step.',
      srcFig1: '(Source: Fig. 1)',
      kBt: 'Block length B', kBd: 'How many tokens one block holds; blocks are filled left to right.',
      kRt: 'Remasking strategy', kRd: 'Which masked positions are committed at each step: the most confident ones, or random ones.',
      kTt: 'Tokens per step (TPS)', kTd: 'How many tokens are committed per denoising step; higher is faster, but the text quality drops.',
      exploreH: 'Demo: What is the least biased model?',
      exploreLede: 'Changing the sampling configuration changes the bias scores and ranks. ', exploreQ: 'So, which model is the least biased?',
      sweep: '▶ Sweep all 24', stop: '■ Stop',
      cState: 'Model type', cMethod: 'Debiasing method', base: 'Untuned', debiased: 'After debiasing',
      cMetric: 'Bias score', mAmb: 'Ambiguous', mDis: 'Disambiguated',
      cRemask: 'Remasking strategy', rkConf: 'Confidence-based', rkRand: 'Random',
      cB: 'Block length B', cTPS: 'Tokens per step (TPS)',
      ord: ['1st', '2nd', '3rd'],
      podH: 'Fair Ranking', podHn: '(value = bias score; the closer to 0, the fairer)',
      fair: 'Fair', biased: 'Biased',
      podCapAmb: '', podCapDis: '1st = closest to 0',
      bsS: 'What is the bias score?',
      bs1: 'The score is 0 when the model picks the two options equally often for the context, 1 when it always picks the stereotype, and −1 when it always picks the opposite.',
      stripH: 'The 24 configurations and the one selected',
      stripBy: 'Colour cells by rank of',
      rowConf: 'Confidence', rowRand: 'Random', rowConfS: 'Conf.', rowRandS: 'Rand.',
      legClick: 'Click a cell to jump to it.',
      answer1: 'The least biased model cannot be pinned down to one.',
      live: 'Across all 24, {b} is least biased in {nb}, {w} most biased in {nw}; the ranking shown between the models holds in only {nc}.',
      srcExplore: '(Source: data of Figs. 2–3)',
      tipBox: 'Box failure {v}: most completions lack a \\boxed{} answer.',
      rangesH: 'Demo: What is successful debiasing?',
      rangesLede1: 'When the bias score moves with the sampling configuration, what is successful debiasing? ',
      rangesLede2: 'Fairness terms pull each method’s scores toward zero, but the variation stays at about the level of the untuned model.',
      cModel: 'Model', lgConf: 'confidence-based', lgRand: 'random', lgDef: 'default config', lgDefOut: 'default (B = 256, outside the grid)',
      lgBase: 'untuned at default', hMethod: 'method', hSpan: 'span',
      srcRange: '(Source: Fig. {n})',
      mBase: 'Untuned model (no debiasing training).',
      mSft: 'Supervised fine-tuning on the bias-reduction data.',
      mDopsd: 'On-policy self-distillation for DLMs (Dat et al., 2026).',
      tLr: 'Log-ratio (Qian et al., 2019)', tJs: 'JS divergence from uniform (Zhao et al., 2024)', tVar: 'Variance of log-probabilities (ours)',
      ellNote: 'ℓ(g): log-probability of group term g; G: group of terms to equalize.',
      utilH: 'Analysis: Does debiasing a DLM cost utility?',
      utilLeadB: 'No dramatic degradation. ',
      utilLede: 'Significant degradation appears in some cases, but none amounts to a model collapse. The debiasing effects shown above are therefore not an artefact of a collapsed model.',
      scX: 'Ambiguous bias', scXt: 'ambiguous bias (closer to 0 is fairer)', scYt: 'accuracy ↑ (higher is better)', scAcc: 'accuracy', noTerm: 'no fairness term',
      utilCap: '† / ‡: significant improvement / degradation vs. untuned (paired cluster bootstrap). (Source: Figs. 5, 11)',
      copy: 'Copy', copied: 'Copied',
      toLight: 'Switch to light mode', toDark: 'Switch to dark mode'
    },
    ja: {
      back: '← Topに戻る', kicker: 'プレプリント · 2026',
      aff1: '東京科学大学', aff2: 'リバプール大学', aff4: '産業技術総合研究所', aff5: '国立情報学研究所 LLMC',
      pillArxiv: 'arXiv（近日公開）', pillCode: 'コード',
      hook: 'Fairness in diffusion language models depends on how you sample.',
      tldrH: 'TL;DR',
      f1q: 'サンプリング設定は DLM の社会的バイアスを変える？',
      f1a: '変わる - スコアもモデル間の順位も。',
      f1d: 'どのモデルでも、24 通りのサンプリング設定でバイアススコアが 40% 以上動き、最もバイアスの小さいモデルも入れ替わる。',
      f2q: 'デバイアスが成功するとは？',
      f2a: 'バイアスの大きさを小さくし、かつその変動を抑えること。',
      f2d: '自己回帰言語モデル（ARLM）向けのデバイアス手法は DLM のバイアスも下げるが、サンプリング設定による変動はおおむね残ってしまう。バイアスの大きさと変動は共に見なければならない。',
      f3q: 'この研究ではどれほど網羅的に調査した？',
      f3a: '11 バイアス × 3 モデル × 24 生成設定 × 9 学習',
      f3d: 'Open-BBQ（10k 件、11 カテゴリ）で LLaDA-8B・LLaDA-1.5・Dream-7B を、リマスキング戦略 × ブロック長 × TPS について、未学習と 8 種の SFT / dOPSD 版で評価し、有用性は GSM8K・IFEval・MMLU で測りました。',
      prelimH: '背景知識: DLM に固有の生成設定',
      prelimLede: 'DLM の生成は、ARLM には対応する設定がない、3つの固有な設定から主に左右される。',
      figAlt: 'マスク拡散言語モデルの 1 回の生成。ブロック長 4、低確信度リマスキング、1 ステップ 2 トークン。',
      srcFig1: '（出典: Fig. 1）',
      kBt: 'ブロック長 B', kBd: '1 ブロックに含まれるトークン数。ブロックは左から順に埋まる。',
      kRt: 'リマスキング戦略', kRd: '各ステップでどのマスク位置を確定させるか。確信度の高い位置から確定するか、ランダムな位置から確定するか。',
      kTt: '1ステップの確定トークン数（TPS）', kTd: '1 回のデノイジングで確定するトークン数。大きいほど高速だが、文章の品質は下がる。',
      exploreH: 'デモ: 最もバイアスの小さいモデルは？',
      exploreLede: 'サンプリング設定を変えると、バイアススコアと順位が変わる。', exploreQ: 'では、最もバイアスの小さいモデルはどれでしょう？',
      sweep: '▶ 24 通りを順に再生', stop: '■ 停止',
      cState: 'モデルの種類', cMethod: 'バイアス除去手法', base: '未学習', debiased: 'デバイアス学習後',
      cMetric: 'バイアススコア', mAmb: '曖昧文脈', mDis: '非曖昧文脈',
      cRemask: 'リマスキング戦略', rkConf: '確信度ベース', rkRand: 'ランダム',
      cB: 'ブロック長 B', cTPS: 'ステップあたりの確定トークン数 TPS',
      ord: ['1 位', '2 位', '3 位'],
      podH: '公平性ランキング', podHn: '（値はバイアススコア。値が０に近いほど公平。）',
      fair: '公平', biased: '偏見',
      podCapAmb: '', podCapDis: '1 位 = 最も 0 に近い',
      bsS: 'バイアススコアとは？',
      bs1: '文脈に応じて選択肢を同じ頻度で答えると 0、常にステレオタイプを選ぶと 1、常に逆を選ぶと −1 になる。',
      stripH: '24通りの設定と現在選択している設定',
      stripBy: '順位で色分けするモデル',
      rowConf: '確信度', rowRand: 'ランダム', rowConfS: '確信度', rowRandS: 'ランダム',
      legClick: 'セルをクリックするとその設定に移動します。',
      answer1: '最もバイアスの小さいモデルを一つに決めることはできない。',
      live: '24 通り中、{b} が最小は {nb} 通り、{w} が最大は {nw} 通り。いま見ているモデル間の順位は {nc} 設定でしか成り立たない。',
      srcExplore: '（出典: Fig. 2–3 のデータ）',
      tipBox: '\\boxed{} 失敗率 {v}: ほとんどの出力に \\boxed{} 形式の回答がありません。',
      rangesH: 'デモ: デバイアスが成功するとは？',
      rangesLede1: 'バイアススコアが設定によって変動するとき、デバイアスの成功とは何か。',
      rangesLede2: '公平性項は各手法のスコアを 0 に近づけるが、変動は未学習モデルと同程度のまま残る。',
      cModel: 'モデル', lgConf: '確信度ベース', lgRand: 'ランダム', lgDef: '既定設定', lgDefOut: '既定設定（B = 256、グリッド外）',
      lgBase: '既定設定での未学習', hMethod: '手法', hSpan: '幅',
      srcRange: '（出典: Fig. {n}）',
      mBase: 'デバイアス学習をしていない未学習モデル。',
      mSft: 'バイアス低減データによる教師ありファインチューニング。',
      mDopsd: 'DLM 向けのオンポリシー自己蒸留（Dat et al., 2026）。',
      tLr: '対数比（Qian et al., 2019）', tJs: '一様分布からの JS 距離（Zhao et al., 2024）', tVar: '対数確率の分散（本研究）',
      ellNote: 'ℓ(g): グループ語 g の対数確率、G: そろえたい語のグループ。',
      utilH: '分析: DLM のバイアス除去は有用性を落とす？',
      utilLeadB: '劇的な劣化は起きない。',
      utilLede: 'いくつかの設定で有意な劣化はあるが、モデルが崩壊するような劣化ではない。したがって、上記で見たデバイアスの効果は、モデル崩壊による幻影ではない。',
      scX: '曖昧文脈のバイアス', scXt: '曖昧文脈のバイアス（0 に近いほど公平）', scYt: '精度 ↑（高いほど良い）', scAcc: '精度', noTerm: '公平性項なし',
      utilCap: '† / ‡: 未学習に対する有意な改善 / 低下（対応のあるクラスターブートストラップ）。（出典: Fig. 5, 11）',
      copy: 'コピー', copied: 'コピーしました',
      toLight: 'ライトモードに切り替え', toDark: 'ダークモードに切り替え'
    }
  };

  var LANG = document.documentElement.getAttribute('lang') === 'ja' ? 'ja' : 'en';
  function t(key, vars) {
    var s = (STR[LANG][key] !== undefined ? STR[LANG][key] : STR.en[key]);
    if (vars && typeof s === 'string') s = s.replace(/\{(\w+)\}/g, function (_, k) { return vars[k] !== undefined ? vars[k] : ''; });
    return s;
  }
  var hooks = [];  // re-render callbacks after a language change

  function applyStatic() {
    document.documentElement.setAttribute('lang', LANG);
    document.querySelectorAll('[data-i18n]').forEach(function (e) { e.textContent = t(e.getAttribute('data-i18n')); });
    document.querySelectorAll('[data-i18n-aria]').forEach(function (e) { e.setAttribute('aria-label', t(e.getAttribute('data-i18n-aria'))); });
    document.querySelectorAll('[data-i18n-alt]').forEach(function (e) { e.setAttribute('alt', t(e.getAttribute('data-i18n-alt'))); });
    var home = LANG === 'ja' ? '/' : '/en/';
    document.getElementById('backLink').href = home;
    document.getElementById('authorLink').href = home;
    document.querySelectorAll('.lang-btn').forEach(function (b) {
      var on = b.getAttribute('data-lang') === LANG;
      b.classList.toggle('active', on); b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    updateToggle(resolveTheme());
  }
  document.querySelectorAll('.lang-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      LANG = b.getAttribute('data-lang');
      try { localStorage.setItem('lang', LANG); } catch (e) {}
      applyStatic();
      hooks.forEach(function (f) { f(); });
    });
  });

  /* ── Theme (same localStorage key and data-theme attribute as the main site) ── */
  function resolveTheme() {
    var a = document.documentElement.getAttribute('data-theme');
    if (a === 'dark' || a === 'light') return a;
    var s = null;
    try { s = localStorage.getItem('theme'); } catch (e) {}
    if (s === 'dark' || s === 'light') return s;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function updateToggle(theme) {
    var btn = document.getElementById('themeToggle');
    var dark = theme === 'dark';
    btn.textContent = dark ? '☀️' : '🌙';
    btn.title = dark ? t('toLight') : t('toDark');
    btn.setAttribute('aria-label', btn.title);
  }
  document.getElementById('themeToggle').addEventListener('click', function () {
    var next = resolveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
    updateToggle(next);
  });

  /* ── BibTeX copy ── */
  var copyBtn = document.getElementById('copyBib');
  copyBtn.addEventListener('click', function () {
    var pre = document.getElementById('bibText');
    var done = function () { copyBtn.textContent = t('copied'); setTimeout(function () { copyBtn.textContent = t('copy'); }, 1500); };
    var fallback = function () {
      var r = document.createRange(); r.selectNodeContents(pre);
      var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
      try { document.execCommand('copy'); done(); } catch (e) {}
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done, fallback);
    else fallback();
  });

  applyStatic();

  /* ── Helpers ── */
  var METRIC_IDX = { amb_bias: 0, dis_bias: 1 };  // columns of results.json grid rows: [amb_bias, dis_bias, amb_acc, box_fail]
  var MODEL_VAR = { llada: '--m-llada', llada15: '--m-llada15', dream: '--m-dream' };
  var SVGNS = 'http://www.w3.org/2000/svg';

  function fmt(v) {
    if (v === null || v === undefined || isNaN(v)) return '–';
    var a = Math.round(Number((Math.abs(v) * 1000).toFixed(4))) / 1000;
    if (a === 0) return '.000';
    return (v < 0 ? '−' : '') + a.toFixed(3).replace(/^0/, '');
  }
  function fmt2(v, step) {
    var d = 2;
    if (step) { var e = Math.floor(Math.log10(step)), f = Math.round(step / Math.pow(10, e) * 10) / 10; d = Math.max(2, -e + (f === 2.5 ? 1 : 0)); }
    if (Math.abs(v) < 1e-9) return '.' + '0'.repeat(d);
    return (v < 0 ? '−' : '') + Math.abs(v).toFixed(d).replace(/^0/, '');
  }
  function stepOf(ticks) { return ticks.length > 1 ? ticks[1] - ticks[0] : 0; }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function sv(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function niceStep(raw) {
    var p = Math.pow(10, Math.floor(Math.log10(raw)));
    var f = raw / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  }
  function niceDomain(lo, hi, nTicks) {
    if (hi - lo < 1e-9) hi = lo + 0.01;
    var step = niceStep((hi - lo) / (nTicks || 4));
    var nlo = Math.floor(lo / step + 1e-9) * step, nhi = Math.ceil(hi / step - 1e-9) * step;
    var ticks = [];
    for (var x = nlo; x <= nhi + step * 1e-6; x += step) ticks.push(Math.abs(x) < 1e-12 ? 0 : x);
    return { lo: nlo, hi: nhi, ticks: ticks };
  }

  /* ── Tooltip ── */
  var tip = document.getElementById('tip');
  function showTip(build, x, y) {
    clear(tip); tip.className = 'tip'; build(tip); tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(x + 14, window.innerWidth - w - 8)) + 'px';
    tip.style.top = Math.max(8, y + 16 + h > window.innerHeight ? y - h - 12 : y + 16) + 'px';
  }
  function hideTip() { tip.hidden = true; }
  function bindTip(node, build) {
    node.addEventListener('pointermove', function (e) { showTip(build, e.clientX, e.clientY); });
    node.addEventListener('pointerleave', hideTip);
    node.addEventListener('focus', function () { var r = node.getBoundingClientRect(); showTip(build, r.left + r.width / 2, r.bottom - 6); });
    node.addEventListener('blur', hideTip);
  }
  window.addEventListener('scroll', hideTip, { passive: true });

  /* ── Fairness-term formulas (MathML, constant strings) ── */
  var ELL = function (g) { return '<mi>ℓ</mi><mo stretchy="false">(</mo>' + g + '<mo stretchy="false">)</mo>'; };
  var FORMULA = {
    log_ratio: '<math display="block"><msup><mi>V</mi><mtext>log-ratio</mtext></msup><mo stretchy="false">(</mo><mi>G</mi><mo stretchy="false">)</mo><mo>=</mo><mo>|</mo>' +
      ELL('<msub><mi>g</mi><mn>1</mn></msub>') + '<mo>−</mo>' + ELL('<msub><mi>g</mi><mn>2</mn></msub>') + '<mo>|</mo><mo>,</mo><mspace width="0.6em"/><mi>G</mi><mo>=</mo><mo stretchy="false">{</mo><msub><mi>g</mi><mn>1</mn></msub><mo>,</mo><msub><mi>g</mi><mn>2</mn></msub><mo stretchy="false">}</mo></math>',
    js_div: '<math display="block"><msup><mi>V</mi><mtext>JS</mtext></msup><mo stretchy="false">(</mo><mi>G</mi><mo stretchy="false">)</mo><mo>=</mo><mi>JS</mi><mo stretchy="false">(</mo><mi>r</mi><mo>∥</mo><mi>u</mi><mo stretchy="false">)</mo><mo>,</mo><mspace width="0.6em"/><mi>r</mi><mo stretchy="false">(</mo><mi>g</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><msup><mi>e</mi><mrow>' + ELL('<mi>g</mi>') + '</mrow></msup><mrow><munder><mo>∑</mo><mrow><msup><mi>g</mi><mo>′</mo></msup><mo>∈</mo><mi>G</mi></mrow></munder><msup><mi>e</mi><mrow>' + ELL('<msup><mi>g</mi><mo>′</mo></msup>') + '</mrow></msup></mrow></mfrac></math>',
    logprob_var: '<math display="block"><msup><mi>V</mi><mtext>var</mtext></msup><mo stretchy="false">(</mo><mi>G</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mo>|</mo><mi>G</mi><mo>|</mo></mrow></mfrac><munder><mo>∑</mo><mrow><mi>g</mi><mo>∈</mo><mi>G</mi></mrow></munder><msup><mrow><mo>(</mo>' + ELL('<mi>g</mi>') + '<mo>−</mo><mfrac><mn>1</mn><mrow><mo>|</mo><mi>G</mi><mo>|</mo></mrow></mfrac><munder><mo>∑</mo><mrow><msup><mi>g</mi><mo>′</mo></msup><mo>∈</mo><mi>G</mi></mrow></munder>' + ELL('<msup><mi>g</mi><mo>′</mo></msup>') + '<mo>)</mo></mrow><mn>2</mn></msup></math>'
  };
  function termOf(id) {
    if (id.indexOf('log_ratio') >= 0) return 'log_ratio';
    if (id.indexOf('js_div') >= 0) return 'js_div';
    if (id.indexOf('logprob_var') >= 0) return 'logprob_var';
    return 'none';
  }
  function famOf(id) { return id === 'base' ? 'base' : id.indexOf('sft') === 0 ? 'sft' : 'dopsd'; }
  function buildMethodCard(tt, id, label) {
    tt.classList.add('tip-wide');
    tt.appendChild(el('div', 'tv', label));
    var fam = famOf(id);
    tt.appendChild(el('div', 'tl', fam === 'base' ? t('mBase') : fam === 'sft' ? t('mSft') : t('mDopsd')));
    var term = termOf(id);
    if (term !== 'none') {
      tt.appendChild(el('div', 'tw', term === 'log_ratio' ? t('tLr') : term === 'js_div' ? t('tJs') : t('tVar')));
      var m = el('div', 'tmath'); m.innerHTML = FORMULA[term]; tt.appendChild(m);
      tt.appendChild(el('div', 'tl', t('ellNote')));
    }
  }

  /* ════════ Data-driven parts ════════ */
  fetch('data/results.json').then(function (r) {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(init).catch(function (err) {
    document.getElementById('bars').textContent = 'Could not load data/results.json (' + err.message + ').';
  });

  function init(D) {
    var meta = D.meta;
    var models = meta.models.map(function (m) { return m.id; });
    var short = {}, full = {};
    meta.models.forEach(function (m) { short[m.id] = m.short; full[m.id] = m.label; });
    var order = meta.grid.order;
    var UNREL = meta.unreliable_box_fail;
    var debMethods = meta.methods.filter(function (m) { return m.id !== 'base'; });
    var mLabel = {}; meta.methods.forEach(function (m) { mLabel[m.id] = m.label.replace(/\+/g, ' + '); });
    function lbl(id) { return id === 'base' ? t('base') : mLabel[id]; }

    var S = { state: 'base', method: meta.default_debiased, rk: 'confidence', B: 32, TPS: 1, metric: 'amb_bias', stripBy: 'llada', rangeModel: 'llada' };

    function curMethod() { return S.state === 'base' ? 'base' : S.method; }
    function idxOf(rk, B, TPS) {
      for (var i = 0; i < order.length; i++) { var c = order[i]; if (c.remask_kind === rk && c.B === B && c.TPS === TPS) return i; }
      return -1;
    }
    function curIdx() { return idxOf(S.rk, S.B, S.TPS); }
    function val(m, method, i, metric) { return D.grid[m][method][i][METRIC_IDX[metric]]; }
    function boxFail(m, method, i) { return D.grid[m][method][i][3]; }
    function score(metric, v) { return meta.metrics[metric].rank === 'abs' ? Math.abs(v) : v; }
    function ranking(method, i, metric) {
      return models.slice().sort(function (a, b) {
        var d = score(metric, val(a, method, i, metric)) - score(metric, val(b, method, i, metric));
        return d !== 0 ? d : models.indexOf(a) - models.indexOf(b);
      });
    }
    function mvar(m) { return 'var(' + MODEL_VAR[m] + ')'; }

    /* ---------- controls ---------- */
    var methodSel = document.getElementById('methodSel');
    debMethods.forEach(function (m) { var o = el('option', null, mLabel[m.id]); o.value = m.id; methodSel.appendChild(o); });
    methodSel.value = S.method;
    var stripSel = document.getElementById('stripSel');
    var rangeSel = document.getElementById('rangeModelSel');
    models.forEach(function (m) { var o = el('option', null, full[m]); o.value = m; rangeSel.appendChild(o); });
    rangeSel.value = S.rangeModel;
    function fillStripSel() {
      clear(stripSel);
      models.forEach(function (m) { var o = el('option', null, short[m]); o.value = m; stripSel.appendChild(o); });
      stripSel.value = S.stripBy;
    }
    fillStripSel();

    function syncSegs() {
      document.querySelectorAll('#controls .seg').forEach(function (seg) {
        var key = seg.getAttribute('data-key');
        seg.querySelectorAll('button').forEach(function (b) {
          var on = String(S[key]) === b.getAttribute('data-val');
          b.setAttribute('aria-checked', on ? 'true' : 'false'); b.tabIndex = on ? 0 : -1;
        });
      });
      methodSel.disabled = S.state === 'base';
      methodSel.value = S.method;
    }
    document.querySelectorAll('#controls .seg').forEach(function (seg) {
      var key = seg.getAttribute('data-key');
      seg.addEventListener('click', function (e) {
        var b = e.target.closest('button'); if (!b) return;
        var v = b.getAttribute('data-val');
        S[key] = (key === 'B' || key === 'TPS') ? Number(v) : v;
        if (key === 'rk' || key === 'B' || key === 'TPS') stopSweep();
        renderExplorer();
      });
      seg.addEventListener('keydown', function (e) {
        if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
        var bs = Array.prototype.slice.call(seg.querySelectorAll('button'));
        var cur = bs.findIndex(function (b) { return b.getAttribute('aria-checked') === 'true'; });
        var nx = bs[(cur + (e.key === 'ArrowRight' ? 1 : bs.length - 1)) % bs.length];
        nx.click(); nx.focus(); e.preventDefault();
      });
    });
    methodSel.addEventListener('change', function () { S.method = methodSel.value; S.state = 'debiased'; renderExplorer(); });
    stripSel.addEventListener('change', function () { S.stripBy = stripSel.value; renderStrip(); });
    rangeSel.addEventListener('change', function () { S.rangeModel = rangeSel.value; renderRanges(); });

    /* ---------- podium ---------- */
    var podium = document.getElementById('podium');
    var pods = {};
    models.forEach(function (m) {
      var p = el('div', 'pod'); p.style.setProperty('--mc', mvar(m));
      var lab = el('div', 'pod-label');
      var nm = el('div', 'pod-name'); var sw = el('i'); nm.appendChild(sw); nm.appendChild(el('span', null, short[m]));
      var v = el('div', 'pod-val');
      lab.appendChild(nm); lab.appendChild(v);
      var step = el('div', 'pod-step'); var rk = el('span', 'pod-rank'); step.appendChild(rk);
      p.appendChild(lab); p.appendChild(step);
      podium.appendChild(p);
      pods[m] = { el: p, val: v, rank: rk, step: step };
    });
    var SLOT = [1, 0, 2];             // rank 0 → centre, rank 1 → left, rank 2 → right
    var STEP_H = [86, 62, 44];

    /* ---------- bars ---------- */
    var barsEl = document.getElementById('bars');
    var axisEl = document.getElementById('axis');
    var rows = {};
    models.forEach(function (m) {
      var row = el('div', 'brow'); row.tabIndex = 0; row.style.setProperty('--mc', mvar(m));
      var name = el('div', 'bname'); var sw = el('i'); name.appendChild(sw); name.appendChild(el('span', null, short[m]));
      var track = el('div', 'track'); var grid = el('div', 'gridlayer'); var bar = el('div', 'bar');
      track.appendChild(grid); track.appendChild(bar);
      var bval = el('div', 'bval');
      row.appendChild(name); row.appendChild(track); row.appendChild(bval);
      barsEl.appendChild(row);
      rows[m] = { row: row, grid: grid, bar: bar, bval: bval, track: track };
      bindTip(row, function (tt) {
        var method = curMethod(), i = curIdx();
        tt.appendChild(el('div', 'tv', fmt(val(m, method, i, S.metric))));
        tt.appendChild(el('div', 'tl', full[m] + ' · ' + lbl(method)));
        var bf = boxFail(m, method, i);
        if (bf > UNREL) tt.appendChild(el('div', 'tw', t('tipBox', { v: fmt(bf) })));
      });
    });

    function explorerDomain(method, metric) {
      var lo = 0, hi = 0;
      models.forEach(function (m) {
        for (var i = 0; i < order.length; i++) {
          if (boxFail(m, method, i) > UNREL) continue;
          var v = val(m, method, i, metric); lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
      });
      var pad = (hi - lo) * 0.04;
      return niceDomain(lo - (lo < 0 ? pad : 0), hi + pad, 4);
    }

    function renderPodium(rk, method, i, metric) {
      var w = podium.clientWidth, slotW = w / 3;
      rk.forEach(function (m, r) {
        var P = pods[m];
        P.el.style.width = (slotW - 8) + 'px';
        P.el.style.transform = 'translateX(' + (SLOT[r] * slotW + 4) + 'px)';
        P.step.style.height = STEP_H[r] + 'px';
        P.el.classList.toggle('first', r === 0);
        P.val.textContent = fmt(val(m, method, i, metric));
        P.rank.textContent = t('ord')[r];
      });
      podium.setAttribute('aria-label', rk.map(function (m, r) { return t('ord')[r] + ' ' + short[m]; }).join(', '));
      document.getElementById('podCap').textContent = metric === 'dis_bias' ? t('podCapDis') : t('podCapAmb');
    }

    var bfair = document.getElementById('bfair');
    function renderBars() {
      var method = curMethod(), i = curIdx(), metric = S.metric;
      var dom = explorerDomain(method, metric), span = dom.hi - dom.lo;
      var x = function (v) { return ((Math.min(dom.hi, Math.max(dom.lo, v)) - dom.lo) / span) * 100; };
      var rk = ranking(method, i, metric);
      models.forEach(function (m) {
        var R = rows[m];
        clear(R.grid);
        dom.ticks.forEach(function (tk) { var g = el('div', 'gridline' + (tk === 0 ? ' zero' : '')); g.style.left = x(tk) + '%'; R.grid.appendChild(g); });
        var v = val(m, method, i, metric), x0 = x(0), xv = x(v);
        R.bar.style.left = Math.min(x0, xv) + '%';
        R.bar.style.width = Math.max(0.25, Math.abs(xv - x0)) + '%';
        R.bar.className = 'bar ' + (v < 0 ? 'neg' : 'pos') + (v > dom.hi || v < dom.lo ? ' clamp' : '');
        R.bval.textContent = fmt(v);
        R.row.setAttribute('aria-label', full[m] + ': ' + fmt(v));
      });
      clear(axisEl);
      var tb = rows[models[0]].track.getBoundingClientRect(), rb = barsEl.getBoundingClientRect();
      axisEl.style.marginLeft = (tb.left - rb.left) + 'px';
      axisEl.style.width = tb.width + 'px';
      var every = tb.width / dom.ticks.length < 46 ? 2 : 1, z0 = Math.max(0, dom.ticks.indexOf(0));
      dom.ticks.forEach(function (tk, k) {
        if ((k - z0) % every !== 0) return;
        var s = el('span', 'tick', fmt2(tk, stepOf(dom.ticks))); s.style.left = x(tk) + '%'; axisEl.appendChild(s);
      });
      // "Fair" above the zero line, "Biased" above the right end of the axis
      bfair.style.marginLeft = (tb.left - rb.left) + 'px'; bfair.style.width = tb.width + 'px';
      bfair.firstChild.style.left = x(0) + '%';
      renderPodium(rk, method, i, metric);
    }

    /* ---------- strip ---------- */
    var stripEl = document.getElementById('strip');
    var cells = [];
    function buildStrip() {
      clear(stripEl); cells = [];
      stripEl.appendChild(el('div', 'hdr', ''));
      meta.grid.TPS.forEach(function (tp) { stripEl.appendChild(el('div', 'hdr', 'TPS ' + tp)); });
      stripEl.appendChild(el('div', 'hdr', 'B'));
      meta.grid.TPS.forEach(function () {
        var g = el('div', 'cg hdr');
        meta.grid.B.forEach(function (b) { g.appendChild(el('span', null, String(b))); });
        stripEl.appendChild(g);
      });
      meta.grid.remask_kind.forEach(function (rk) {
        var rl = el('div', 'rlab');
        rl.appendChild(el('span', 'full', rk === 'confidence' ? t('rowConf') : t('rowRand')));
        rl.appendChild(el('span', 'short', rk === 'confidence' ? t('rowConfS') : t('rowRandS')));
        stripEl.appendChild(rl);
        meta.grid.TPS.forEach(function (tp) {
          var g = el('div', 'cg'); g.setAttribute('role', 'row');
          meta.grid.B.forEach(function (b) {
            var i = idxOf(rk, b, tp);
            var c = el('button', 'cell'); c.type = 'button'; c.setAttribute('role', 'gridcell');
            c.addEventListener('click', function () { stopSweep(); S.rk = rk; S.B = b; S.TPS = tp; renderExplorer(); });
            g.appendChild(c); cells[i] = c;
          });
          stripEl.appendChild(g);
        });
      });
    }
    function renderStrip() {
      var method = curMethod(), metric = S.metric, ci = curIdx();
      var PCT = [20, 55, 92];
      for (var i = 0; i < order.length; i++) {
        var c = cells[i]; clear(c);
        c.className = 'cell' + (i === ci ? ' cur' : '');
        c.style.background = '';
        var rk = ranking(method, i, metric);
        var inner = el('span', 'cin');
        var r = rk.indexOf(S.stripBy);
        c.style.background = 'color-mix(in oklab, ' + mvar(S.stripBy) + ' ' + PCT[r] + '%, var(--surface))';
        if (r === 2) c.classList.add('on-dark');
        inner.appendChild(el('span', 'cnum', String(r + 1)));
        c.appendChild(inner);
        var cfg = order[i];
        c.setAttribute('aria-label', (cfg.remask_kind === 'confidence' ? t('rkConf') : t('rkRand')) + ', B ' + cfg.B + ', TPS ' + cfg.TPS + ': ' +
          rk.map(function (m) { return short[m]; }).join(' < '));
      }
      var lg = document.getElementById('stripLegend'); clear(lg);
      PCT.forEach(function (p, r) {
        var s = el('span'); var sw = el('span', 'sw'); sw.style.background = 'color-mix(in oklab, ' + mvar(S.stripBy) + ' ' + p + '%, var(--surface))';
        s.appendChild(sw); s.appendChild(document.createTextNode(t('ord')[r])); lg.appendChild(s);
      });
      lg.appendChild(el('span', null, t('legClick')));
    }

    function renderSentence() {
      var method = curMethod(), metric = S.metric, ci = curIdx();
      var best = {}, worst = {}, orders = {};
      models.forEach(function (m) { best[m] = 0; worst[m] = 0; });
      for (var i = 0; i < order.length; i++) {
        var rk = ranking(method, i, metric);
        best[rk[0]]++; worst[rk[2]]++;
        var k = rk.join('<'); orders[k] = (orders[k] || 0) + 1;
      }
      var top = function (o) { return models.slice().sort(function (a, b) { return o[b] - o[a]; })[0]; };
      var bm = top(best), wm = top(worst);
      var live = document.getElementById('liveSentence'); clear(live);
      var vars = { b: short[bm], nb: best[bm], w: short[wm], nw: worst[wm], no: Object.keys(orders).length, nc: orders[ranking(method, ci, metric).join('<')] };
      live.appendChild(document.createTextNode(t('live', vars)));
    }

    /* ---------- sweep ---------- */
    var sweepBtn = document.getElementById('sweepBtn');
    var timer = null, sweepI = 0;
    function sweepLabel() { sweepBtn.textContent = timer ? t('stop') : t('sweep'); sweepBtn.setAttribute('aria-pressed', timer ? 'true' : 'false'); }
    function stopSweep() { if (timer) { clearInterval(timer); timer = null; } sweepLabel(); }
    function stepTo(i) { var c = order[i]; S.rk = c.remask_kind; S.B = c.B; S.TPS = c.TPS; renderExplorer(); }
    sweepBtn.addEventListener('click', function () {
      if (timer) { stopSweep(); return; }
      sweepI = 0; stepTo(0);
      timer = setInterval(function () { sweepI = (sweepI + 1) % order.length; stepTo(sweepI); }, 700);
      sweepLabel();
    });

    function renderExplorer() { syncSegs(); renderBars(); renderStrip(); renderSentence(); }

    /* ---------- ranges (Fig. 4 / 8 / 10) ---------- */
    var rangeChart = document.getElementById('rangeChart');
    var FIG = { llada: 4, llada15: 8, dream: 10 };
    function renderRanges() {
      var m = S.rangeModel, metric = 'amb_bias', dc = meta.default_config[m], mc = mvar(m);
      var data = meta.methods.map(function (me) {
        var out = { id: me.id, family: me.family };
        ['confidence', 'random'].forEach(function (rk) {
          var vs = [];
          for (var i = 0; i < order.length; i++) if (order[i].remask_kind === rk) vs.push(val(m, me.id, i, metric));
          out[rk] = { min: Math.min.apply(null, vs), max: Math.max.apply(null, vs) };
        });
        out.def = dc.in_grid ? val(m, me.id, dc.index, metric) : D.defaults[m][me.id][0];
        return out;
      });
      var baseDef = D.defaults[m].base[0], lo = 0, hi = baseDef;
      data.forEach(function (r) {
        ['confidence', 'random'].forEach(function (rk) { lo = Math.min(lo, r[rk].min); hi = Math.max(hi, r[rk].max); });
        lo = Math.min(lo, r.def); hi = Math.max(hi, r.def);
      });
      var dom = niceDomain(lo, hi, 7), span = dom.hi - dom.lo;
      var x = function (v) { return ((v - dom.lo) / span) * 100; };

      clear(rangeChart);
      var head = el('div', 'rhead');
      head.appendChild(el('span', null, t('hMethod'))); head.appendChild(el('span')); head.appendChild(el('span', null, t('hSpan')));
      rangeChart.appendChild(head);
      var tracks = [];
      data.forEach(function (r, k) {
        var row = el('div', 'rrow' + (k > 0 && data[k - 1].family !== r.family ? ' fam' : ''));
        row.style.setProperty('--mc', mc);
        var lab = lbl(r.id);
        var rn = el('div', 'rname' + (r.id === 'base' ? ' base' : '')); rn.tabIndex = 0;
        rn.appendChild(el('span', 'full', lab));
        rn.appendChild(el('span', 'short' + (lab.indexOf(' + ') > 0 ? ' sub' : ''), lab.indexOf(' + ') > 0 ? '+ ' + lab.split(' + ')[1] : lab));
        bindTip(rn, function (tt) { buildMethodCard(tt, r.id, lab); });
        row.appendChild(rn);
        var tr = el('div', 'rtrack');
        var g = el('div', 'rgrid');
        dom.ticks.forEach(function (tk) { var gl = el('div', 'gridline' + (tk === 0 ? ' zero' : '')); gl.style.left = x(tk) + '%'; g.appendChild(gl); });
        tr.appendChild(g);
        var bl = el('div', 'rbase'); bl.style.left = x(baseDef) + '%'; tr.appendChild(bl);
        [['confidence', 'c'], ['random', 'r']].forEach(function (p) {
          var d = r[p[0]]; var b = el('div', 'rbar ' + p[1]);
          b.style.left = x(d.min) + '%'; b.style.width = (x(d.max) - x(d.min)) + '%'; tr.appendChild(b);
        });
        var dm = el('div', 'rdef' + (dc.in_grid ? '' : ' out')); dm.style.left = x(r.def) + '%'; tr.appendChild(dm);
        row.appendChild(tr); tracks.push(tr);
        var sp = el('div', 'rspan');
        sp.appendChild(el('span', null, fmt(r.confidence.max - r.confidence.min)));
        sp.appendChild(el('span', null, fmt(r.random.max - r.random.min)));
        row.appendChild(sp);
        row.setAttribute('aria-label', lab + ': ' + fmt(r.confidence.min) + '–' + fmt(r.confidence.max) + ' / ' + fmt(r.random.min) + '–' + fmt(r.random.max));
        rangeChart.appendChild(row);
      });
      var ax = el('div', 'raxis'); ax.appendChild(el('span')); var axi = el('div', 'axis');
      var rEvery = tracks[0].clientWidth / dom.ticks.length < 46 ? 2 : 1, rz = Math.max(0, dom.ticks.indexOf(0));
      dom.ticks.forEach(function (tk, k) { if ((k - rz) % rEvery !== 0) return; var s = el('span', 'tick', fmt2(tk, stepOf(dom.ticks))); s.style.left = x(tk) + '%'; axi.appendChild(s); });
      ax.appendChild(axi); ax.appendChild(el('span')); rangeChart.appendChild(ax);

      var lg = document.getElementById('rangeLegend'); clear(lg);
      var key = function (cls, bg, txt) { var s = el('span', 'k'); var k2 = el('span', cls); if (bg) k2.style.background = bg; s.appendChild(k2); s.appendChild(document.createTextNode(txt)); lg.appendChild(s); };
      key('kb', mc, t('lgConf'));
      key('kb', 'color-mix(in srgb, ' + mc + ' 45%, var(--surface-2))', t('lgRand'));
      key(dc.in_grid ? 'kd' : 'kd out', null, dc.in_grid ? t('lgDef') : t('lgDefOut'));
      key('kl', null, t('lgBase') + ' (' + fmt(baseDef) + ')');
      document.getElementById('rangeSrc').textContent = t('srcRange', { n: FIG[m] });
    }

    /* ---------- utility scatter grid (Fig. 5 / 11) ---------- */
    var scGrid = document.getElementById('scGrid');
    var BENCH = [['gsm8k', 'GSM8K'], ['ifeval', 'IFEval'], ['mmlu', 'MMLU']];
    var TERM_VAR = { none: '--ink', log_ratio: '--t-lr', js_div: '--t-js', logprob_var: '--t-var' };
    function shapePath(fam, cx, cy, s) {
      if (fam === 'sft') return 'M' + (cx - s) + ' ' + (cy - s) + 'h' + (2 * s) + 'v' + (2 * s) + 'h' + (-2 * s) + 'z';
      if (fam === 'dopsd') { var d = s * 1.3; return 'M' + cx + ' ' + (cy - d) + 'L' + (cx + d) + ' ' + cy + 'L' + cx + ' ' + (cy + d) + 'L' + (cx - d) + ' ' + cy + 'z'; }
      return 'M' + (cx - s) + ' ' + cy + 'a' + s + ' ' + s + ' 0 1 0 ' + (2 * s) + ' 0a' + s + ' ' + s + ' 0 1 0 ' + (-2 * s) + ' 0';
    }
    function renderLegendSc() {
      var lg = document.getElementById('scLegend'); clear(lg);
      var item = function (fam, colorVar, txt) {
        var s = el('span', 'k');
        var svg = sv('svg', { width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true' });
        var p = sv('path', { d: shapePath(fam, 7, 7, fam === 'base' ? 5 : 4.2) });
        if (fam === 'base') { p.setAttribute('fill', 'var(--surface-2)'); p.setAttribute('stroke', 'var(--ink)'); p.setAttribute('stroke-width', '2'); }
        else p.setAttribute('fill', 'var(' + colorVar + ')');
        svg.appendChild(p); s.appendChild(svg); s.appendChild(document.createTextNode(txt)); lg.appendChild(s);
      };
      item('base', '--ink', t('base')); item('sft', '--muted', 'SFT'); item('dopsd', '--muted', 'dOPSD');
      lg.appendChild(el('span', 'sep', '|'));
      var cs = function (v, txt) { var s = el('span', 'k'); var c = el('span', 'kc'); c.style.background = 'var(' + v + ')'; s.appendChild(c); s.appendChild(document.createTextNode(txt)); lg.appendChild(s); };
      cs('--ink', t('noTerm')); cs('--t-lr', 'log-ratio'); cs('--t-js', 'JS div.'); cs('--t-var', 'log-prob. var.');
    }
    function renderScatter() {
      renderLegendSc();
      clear(scGrid);
      var yt = el('div', 'sc-ytitle'); yt.setAttribute('aria-hidden', 'true');
      yt.appendChild(el('span', null, t('scYt').replace('↑', '→')));  // rotated −90°, so → reads as ↑
      scGrid.appendChild(yt);
      scGrid.appendChild(el('div', 'sc-corner'));
      BENCH.forEach(function (b) { scGrid.appendChild(el('div', 'sc-col', b[1])); });
      models.forEach(function (m) {
        var rl = el('div', 'sc-row'); var sw = el('i'); sw.style.background = mvar(m); rl.appendChild(sw); rl.appendChild(el('span', null, short[m]));
        scGrid.appendChild(rl);
        var xs = meta.methods.map(function (me) { return D.defaults[m][me.id][0]; });
        var xd = niceDomain(Math.min(0, Math.min.apply(null, xs)), Math.max.apply(null, xs), 3);
        BENCH.forEach(function (b) {
          var cell = el('div', 'sc-cell'); scGrid.appendChild(cell);
          var W = Math.max(90, cell.clientWidth), H = Math.round(Math.min(184, Math.max(124, W * 0.68)));
          var ys = meta.methods.map(function (me) { return D.utility[m][me.id][b[0]]; });
          var ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys), pad = (yhi - ylo) * 0.12 + 0.002;
          var yd = niceDomain(ylo - pad, yhi + pad, 3);
          var narrow = W < 150;
          var L = narrow ? 30 : 36, R = 8, T = 8, Bm = narrow ? 18 : 20;
          var X = function (v) { return L + (v - xd.lo) / (xd.hi - xd.lo) * (W - L - R); };
          var Y = function (v) { return T + (1 - (v - yd.lo) / (yd.hi - yd.lo)) * (H - T - Bm); };
          var svg = sv('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'img' });
          svg.setAttribute('aria-label', full[m] + ', ' + b[1]);
          yd.ticks.forEach(function (tk) {
            svg.appendChild(sv('line', { x1: L, x2: W - R, y1: Y(tk), y2: Y(tk), class: 'sc-gl' }));
            var tx = sv('text', { x: L - 5, y: Y(tk) + 4, class: 'sc-tick', 'text-anchor': 'end' }); tx.textContent = fmt2(tk, stepOf(yd.ticks)); svg.appendChild(tx);
          });
          var xEvery = (W - L - R) / xd.ticks.length < 38 ? 2 : 1, xz = Math.max(0, xd.ticks.indexOf(0));
          xd.ticks.forEach(function (tk, k) {
            if ((k - xz) % xEvery !== 0) return;
            var tx = sv('text', { x: X(tk), y: H - Bm + 14, class: 'sc-tick', 'text-anchor': 'middle' }); tx.textContent = fmt2(tk, stepOf(xd.ticks)); svg.appendChild(tx);
          });
          svg.appendChild(sv('line', { x1: L, x2: W - R, y1: H - Bm, y2: H - Bm, class: 'sc-axis' }));
          svg.appendChild(sv('line', { x1: L, x2: L, y1: T, y2: H - Bm, class: 'sc-axis' }));
          var base = D.utility[m].base[b[0]], bx = D.defaults[m].base[0];
          svg.appendChild(sv('line', { x1: L, x2: W - R, y1: Y(base), y2: Y(base), class: 'sc-ref' }));
          svg.appendChild(sv('line', { x1: X(bx), x2: X(bx), y1: T, y2: H - Bm, class: 'sc-ref' }));
          var pts = meta.methods.filter(function (me) { return me.id !== 'base'; }).concat(meta.methods.filter(function (me) { return me.id === 'base'; }));
          pts.forEach(function (me) {
            var U = D.utility[m][me.id], xv = D.defaults[m][me.id][0], yv = U[b[0]], fam = famOf(me.id), term = termOf(me.id);
            var gx = X(xv), gy = Y(yv);
            var g = sv('g', { class: 'sc-pt', tabindex: '0' });
            var p = sv('path', { d: shapePath(fam, gx, gy, fam === 'base' ? 5.5 : 4.2) });
            if (fam === 'base') { p.setAttribute('fill', 'var(--surface-2)'); p.setAttribute('stroke', 'var(--ink)'); p.setAttribute('stroke-width', '2.2'); }
            else { p.setAttribute('fill', 'var(' + TERM_VAR[term] + ')'); p.setAttribute('stroke', 'var(--surface-2)'); p.setAttribute('stroke-width', '1.5'); }
            g.appendChild(sv('circle', { cx: gx, cy: gy, r: 11, fill: 'transparent' }));
            g.appendChild(p);
            var sig = U.sig[b[0]];
            if (sig) { var st = sv('text', { x: gx + 6, y: gy - 4, class: 'sc-sig' }); st.textContent = sig === 'up' ? '†' : '‡'; g.appendChild(st); }
            g.setAttribute('aria-label', lbl(me.id) + ': ' + t('scX') + ' ' + fmt(xv) + ', ' + b[1] + ' ' + fmt(yv) + (sig ? (sig === 'up' ? ' †' : ' ‡') : ''));
            bindTip(g, function (tt) {
              tt.appendChild(el('div', 'tv', lbl(me.id)));
              tt.appendChild(el('div', null, b[1] + ' ' + t('scAcc') + ': ' + fmt(yv) + (sig ? (sig === 'up' ? ' †' : ' ‡') : '')));
              tt.appendChild(el('div', 'tl', t('scX') + ': ' + fmt(xv)));
            });
            svg.appendChild(g);
          });
          cell.appendChild(svg);
        });
      });
      scGrid.appendChild(el('div', 'sc-xlab', t('scXt')));
    }

    buildStrip();
    renderExplorer();
    renderRanges();
    renderScatter();
    sweepLabel();
    hooks.push(function () { fillStripSel(); buildStrip(); renderExplorer(); renderRanges(); renderScatter(); sweepLabel(); });
    var rT = null, lastW = window.innerWidth;
    window.addEventListener('resize', function () {
      clearTimeout(rT);
      rT = setTimeout(function () { renderBars(); renderRanges(); if (window.innerWidth !== lastW) { lastW = window.innerWidth; renderScatter(); } }, 150);
    });
  }
})();
