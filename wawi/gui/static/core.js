/* WAWI GUI – core utilities: API access, DOM helpers, forms, plots, jobs, modals, file browser */
'use strict';

const qs = new URLSearchParams(location.search);
if (qs.get('token')) sessionStorage.setItem('wawi_token', qs.get('token'));
const TOKEN = qs.get('token') || sessionStorage.getItem('wawi_token') || '';

const C = { ink: '#12324a', ink2: '#3c5568', ink3: '#6f8291', sea: '#1d7a8c', wind: '#b8862a', alert: '#b23a3a',
  grid: '#e3e9ec', rule: '#cfd9de', ok: '#2f7d4f' };
const SERIES = ['#1d7a8c', '#b8862a', '#12324a', '#b23a3a', '#5f8f3a', '#7b5ea7', '#3c8dbc', '#c46b9c', '#8a6d3b', '#4a4a4a'];
const DOFS = ['x', 'y', 'z', 'θx', 'θy', 'θz'];

/* ------------------------------------------------------------------ event bus & state */
const bus = {
  h: {},
  on(ev, fn) { (this.h[ev] = this.h[ev] || []).push(fn); },
  emit(ev, d) { (this.h[ev] || []).forEach(fn => { try { fn(d); } catch (e) { console.error(e); } }); },
};
const S = { state: null, model: null, info: null };

/* ------------------------------------------------------------------ API */
class ApiError extends Error {
  constructor(msg, detail) { super(msg); this.detail = detail || {}; }
}
async function api(name, params = {}) {
  let r;
  try {
    r = await fetch('/api/' + name, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
      body: JSON.stringify(params),
    });
  } catch (e) {
    throw new ApiError('Lost connection to the WAWI GUI server. Is it still running in the terminal?');
  }
  let j;
  try { j = await r.json(); } catch (e) { throw new ApiError(`Server returned ${r.status}`); }
  if (!j.ok) throw new ApiError(j.error || 'Unknown error', j);
  return j.data;
}
/* Run an endpoint that may return a background job; resolves with the job's result */
async function run(name, params = {}) {
  const d = await api(name, params);
  if (d && typeof d === 'object' && 'job' in d && 'done' in d) {
    Jobs.touch();
    if (d.done) return d.result;
    Jobs.select(d.job);
    return Jobs.wait(d.job);
  }
  return d;
}
function downloadUrl(id) { return `/download/${id}?token=${encodeURIComponent(TOKEN)}`; }
function triggerDownload(id) {
  const a = document.createElement('a'); a.href = downloadUrl(id); a.download = ''; document.body.appendChild(a); a.click(); a.remove();
}
function downloadText(name, text, type = 'text/csv') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; document.body.appendChild(a); a.click(); a.remove();
}

/* ------------------------------------------------------------------ DOM helpers */
function h(tag, attrs = {}, ...kids) {
  const [t, ...cls] = tag.split('.');
  const el = document.createElement(t || 'div');
  if (cls.length) el.className = cls.join(' ');
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className += (el.className ? ' ' : '') + v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return el;
}
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
function fmt(v, d = 4) {
  if (v === null || v === undefined) return '–';
  if (typeof v !== 'number') return String(v);
  if (!isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(Math.max(1, d - 1));
  return Number(v.toPrecision(d)).toString();
}
function cplx(a) { return a && a.complex ? a : null; }
function re(a) { return a && a.complex ? a.re : a; }
function im(a) { return a && a.complex ? a.im : null; }
function card(title, lead, ...kids) {
  const opts = (typeof title === 'object' && title !== null && !(title instanceof Node)) ? title : { title };
  return h('section.card' + (opts.cls ? '.' + opts.cls : ''), {},
    opts.title ? h('h2', {}, opts.title) : null, lead ? h('p.lead', {}, lead) : null, ...kids);
}
function btn(label, onclick, cls = '') { return h('button.btn' + (cls ? '.' + cls.split(' ').join('.') : ''), { type: 'button', onclick }, label); }
function emptyBox(text) { return h('div.empty', {}, text); }

/* Wrap an async action: disable the button while running, toast errors */
function act(fn, { button, success } = {}) {
  return async (ev) => {
    const b = button || (ev && ev.currentTarget instanceof HTMLButtonElement ? ev.currentTarget : null);
    if (b) b.disabled = true;
    try {
      const r = await fn(ev);
      if (success) toast(typeof success === 'function' ? success(r) : success);
      return r;
    } catch (e) {
      showError(e);
    } finally {
      if (b) b.disabled = false;
      refreshState();
    }
  };
}

/* ------------------------------------------------------------------ toasts & modal */
function toast(msg, { error = false, detail = null, timeout } = {}) {
  const t = h('div.toast' + (error ? '.err' : ''), {}, h('div', {}, msg));
  if (detail) {
    t.append(h('div.acts', {}, h('a', { onclick: () => openModal('Error details', h('pre.out', { style: { maxHeight: '60vh' } }, detail)) }, 'Show details')));
  }
  t.append(h('div.acts', {}, h('a', { onclick: () => t.remove() }, 'Dismiss')));
  $('#toasts').append(t);
  setTimeout(() => t.remove(), timeout || (error ? 14000 : 4000));
}
function showError(e) {
  console.error(e);
  const d = e.detail || {};
  const detail = [d.traceback, d.log && !d.traceback ? d.log : null].filter(Boolean).join('\n') || null;
  toast(e.message || String(e), { error: true, detail });
}
function openModal(title, body, footer = []) {
  const m = $('#modal');
  clear(m).append(
    h('div.mh', {}, h('h3', {}, title), btn('Close', closeModal, 'ghost small')),
    h('div.mb', {}, body),
    footer.length ? h('div.mf', {}, footer) : null);
  $('#modalBg').classList.add('open');
}
function closeModal() { $('#modalBg').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
$('#modalBg').addEventListener('click', e => { if (e.target.id === 'modalBg') closeModal(); });

/* ------------------------------------------------------------------ file browser */
const FS = { last: localStorage.getItem('wawi.fs.last') || '' };
function browse({ kind = 'file', ext = null, title, start } = {}) {
  /* kind: 'file' | 'dir' | 'save' */
  return new Promise((resolve) => {
    let cur = null, selected = null;
    const pathIn = h('input', { type: 'text', class: 'expr' });
    const list = h('div.fslist');
    const nameIn = h('input', { type: 'text', placeholder: 'file name' });
    const info = h('div.small.muted', { style: { marginTop: '6px' } });
    async function load(p) {
      try {
        const d = await api('fs/list', { path: p });
        cur = d; pathIn.value = d.path; selected = null; clear(list);
        FS.last = d.path; localStorage.setItem('wawi.fs.last', d.path);
        info.textContent = d.is_model_folder ? 'This folder contains modal.json – it can be imported as a WAWI model folder.' : '';
        list.append(h('div.fsitem', { ondblclick: () => load(d.parent), onclick: () => load(d.parent) }, h('span.ic', {}, '↰'), '..'));
        for (const e of d.entries) {
          const okExt = !ext || e.dir || ext.some(x => e.name.toLowerCase().endsWith(x));
          const it = h('div.fsitem' + (okExt ? '' : '.dim'), {}, h('span.ic', {}, e.dir ? '▸' : '·'), e.name,
            h('span.sz', {}, e.dir ? '' : (e.size > 1e6 ? (e.size / 1e6).toFixed(1) + ' MB' : (e.size / 1e3).toFixed(1) + ' kB')));
          it.onclick = () => {
            $$('.fsitem', list).forEach(x => x.classList.remove('sel')); it.classList.add('sel');
            selected = e; if (!e.dir && kind === 'save') nameIn.value = e.name;
          };
          it.ondblclick = () => { if (e.dir) load(join(d.path, e.name, d.sep)); else if (kind !== 'dir') finish(join(d.path, e.name, d.sep)); };
          list.append(it);
        }
      } catch (e) { showError(e); }
    }
    function join(a, b, sep) { return a.endsWith(sep) ? a + b : a + sep + b; }
    function finish(p) { closeModal(); resolve(p); }
    const quick = h('div.row', { style: { marginTop: '6px' } },
      btn('Home', () => load(cur ? cur.home : ''), 'ghost small'),
      btn('Server working folder', () => load(cur ? cur.cwd : ''), 'ghost small'));
    pathIn.addEventListener('keydown', e => { if (e.key === 'Enter') load(pathIn.value); });
    const body = h('div', {}, h('div.pathrow', {}, pathIn, btn('Go', () => load(pathIn.value), 'small')), quick, list, info,
      kind === 'save' ? h('div.field', { style: { marginTop: '8px' } }, h('label', {}, 'File name'), nameIn) : null);
    const choose = btn(kind === 'dir' ? 'Use this folder' : kind === 'save' ? 'Save here' : 'Open file', () => {
      if (!cur) return;
      if (kind === 'dir') finish(selected && selected.dir ? join(cur.path, selected.name, cur.sep) : cur.path);
      else if (kind === 'save') { if (nameIn.value.trim()) finish(join(cur.path, nameIn.value.trim(), cur.sep)); }
      else if (selected && !selected.dir) finish(join(cur.path, selected.name, cur.sep));
      else if (selected && selected.dir) load(join(cur.path, selected.name, cur.sep));
    }, 'primary');
    openModal(title || (kind === 'dir' ? 'Choose folder' : kind === 'save' ? 'Save as' : 'Open file'), body,
      [h('span.small.muted', { style: { marginRight: 'auto' } }, 'Paths refer to the computer running the WAWI GUI server.'),
        btn('Cancel', () => { closeModal(); resolve(null); }), choose]);
    load(start || FS.last || '');
  });
}

/* Upload files (or a whole folder) to the server; resolves with server-side path */
function uploadPicker({ folder = false, accept = '' } = {}) {
  return new Promise((resolve) => {
    const inp = h('input', { type: 'file', accept: accept || null, multiple: folder || null });
    if (folder) { inp.webkitdirectory = true; }
    inp.onchange = async () => {
      const files = [...inp.files];
      if (!files.length) return resolve(null);
      let dir = '', base = '', last = '';
      try {
        for (const f of files) {
          const rel = folder ? (f.webkitRelativePath || f.name) : f.name;
          const r = await fetch(`/upload?dir=${dir}&path=${encodeURIComponent(rel)}`, { method: 'POST', headers: { 'X-Token': TOKEN }, body: f });
          const j = await r.json(); if (!j.ok) throw new ApiError(j.error);
          dir = j.data.dir; base = j.data.base; last = j.data.path;
        }
        if (folder) {
          const top = (files[0].webkitRelativePath || '').split('/')[0];
          resolve(top ? base + '/' + top : base);
        } else resolve(last);
        toast(`Uploaded ${files.length} file(s)`);
      } catch (e) { showError(e); resolve(null); }
    };
    inp.click();
  });
}

/* ------------------------------------------------------------------ forms */
/*  field spec: {k, label, def, hint, type: text|expr|num|check|select|multi|path|textarea, opts, span, ext, pathKind, code} */
function makeForm(id, fields, { cols = '', persist = true } = {}) {
  const store = persist ? JSON.parse(localStorage.getItem('wawi.form.' + id) || '{}') : {};
  const inputs = {};
  const root = h('div.form' + (cols ? '.' + cols : ''));
  for (const f of fields) {
    if (f.type === 'heading') { root.append(h('div.field.full', {}, h('label', { style: { fontWeight: 600, color: C.ink, marginTop: '4px' } }, f.label))); continue; }
    const init = (store[f.k] !== undefined) ? store[f.k] : f.def;
    let input, wrap;
    const lab = h('label', { for: `f_${id}_${f.k}` }, f.label || f.k, f.code ? h('code', {}, f.code) : null);
    if (f.type === 'check') {
      input = h('input', { type: 'checkbox', id: `f_${id}_${f.k}` }); input.checked = !!init;
      wrap = h('div.field.check', {}, input, lab);
    } else if (f.type === 'select') {
      input = h('select', { id: `f_${id}_${f.k}` }, (f.opts || []).map(o => Array.isArray(o) ? h('option', { value: o[0] }, o[1]) : h('option', { value: o }, o)));
      if (init !== undefined) input.value = init;
      wrap = h('div.field', {}, lab, input);
    } else if (f.type === 'multi') {
      const vals = new Set(init || []);
      input = h('div.multi', { id: `f_${id}_${f.k}` }, (f.opts || []).map(o => {
        const [v, l] = Array.isArray(o) ? o : [o, o];
        const c = h('input', { type: 'checkbox', value: v }); c.checked = vals.has(v);
        return h('label', {}, c, l);
      }));
      wrap = h('div.field', {}, lab, input);
    } else if (f.type === 'textarea') {
      input = h('textarea', { id: `f_${id}_${f.k}`, class: 'code', rows: f.rows || 4, spellcheck: 'false' });
      input.value = init ?? '';
      wrap = h('div.field', {}, lab, input);
    } else {
      input = h('input', { type: 'text', id: `f_${id}_${f.k}`, class: (f.type === 'expr' || f.type === 'path') ? 'expr' : '', placeholder: f.ph || '', spellcheck: 'false' });
      input.value = init ?? '';
      if (f.type === 'path') {
        const b = btn('Browse…', async () => {
          const p = await browse({ kind: f.pathKind || 'file', ext: f.ext, start: input.value || undefined });
          if (p) { input.value = p; input.dispatchEvent(new Event('change')); }
        }, 'small');
        wrap = h('div.field', {}, lab, h('div.pathrow', {}, input, b));
      } else wrap = h('div.field', {}, lab, input);
    }
    if (f.span === 2) wrap.classList.add('span2');
    if (f.span === 'full') wrap.classList.add('full');
    if (f.hint) wrap.append(h('div.hint', {}, f.hint));
    if (f.title) wrap.title = f.title;
    inputs[f.k] = { f, input, wrap };
    root.append(wrap);
  }
  function value(k) {
    const { f, input } = inputs[k];
    if (f.type === 'check') return input.checked;
    if (f.type === 'multi') return $$('input', input).filter(c => c.checked).map(c => c.value);
    return input.value;
  }
  const api_ = {
    el: root, inputs,
    get() { const o = {}; for (const k in inputs) o[k] = value(k); if (persist) localStorage.setItem('wawi.form.' + id, JSON.stringify(o)); return o; },
    set(obj) {
      for (const [k, v] of Object.entries(obj || {})) {
        if (!inputs[k]) continue;
        const { f, input } = inputs[k];
        if (f.type === 'check') input.checked = !!v;
        else if (f.type === 'multi') $$('input', input).forEach(c => { c.checked = (v || []).includes(c.value); });
        else input.value = (v === null || v === undefined) ? '' : (typeof v === 'object' ? JSON.stringify(v) : String(v));
      }
    },
    reset() { const o = {}; fields.forEach(f => { if (f.k) o[f.k] = f.def ?? (f.type === 'check' ? false : ''); }); api_.set(o); },
    setOptions(k, opts, keep = true) {
      const { f, input } = inputs[k]; const cur = value(k); f.opts = opts;
      if (f.type === 'select') {
        clear(input).append(...opts.map(o => Array.isArray(o) ? h('option', { value: o[0] }, o[1]) : h('option', { value: o }, o)));
        if (keep && opts.some(o => (Array.isArray(o) ? o[0] : o) === cur)) input.value = cur;
      } else if (f.type === 'multi') {
        const vals = new Set(keep ? cur : []);
        clear(input).append(...opts.map(o => { const [v, l] = Array.isArray(o) ? o : [o, o]; const c = h('input', { type: 'checkbox', value: v }); c.checked = vals.has(v); return h('label', {}, c, l); }));
      }
    },
  };
  return api_;
}

/* ------------------------------------------------------------------ subtabs */
function subtabs(container, tabs, { key } = {}) {
  /* tabs: [{id, label, build: (el) => void}] ; builds lazily on first show */
  const bar = h('div.subtabs'); const holder = h('div');
  const built = {}; const els = {};
  const show = (id) => {
    $$('button', bar).forEach(b => b.classList.toggle('active', b.dataset.id === id));
    for (const t of tabs) {
      if (!els[t.id]) { els[t.id] = h('div.sub'); holder.append(els[t.id]); }
      els[t.id].classList.toggle('active', t.id === id);
      if (t.id === id && !built[t.id]) { built[t.id] = true; t.build(els[t.id]); }
      if (t.id === id && t.onShow) t.onShow(els[t.id]);
    }
    if (key) localStorage.setItem('wawi.tab.' + key, id);
  };
  for (const t of tabs) bar.append(h('button', { type: 'button', 'data-id': t.id, onclick: () => show(t.id) }, t.label));
  container.append(bar, holder);
  const initial = (key && localStorage.getItem('wawi.tab.' + key)) || tabs[0].id;
  show(tabs.some(t => t.id === initial) ? initial : tabs[0].id);
  return { show };
}

/* ------------------------------------------------------------------ plots */
function baseLayout(extra = {}) {
  return Object.assign({
    margin: { l: 64, r: 18, t: 30, b: 46 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#ffffff',
    font: { family: getComputedStyle(document.body).fontFamily, size: 12, color: C.ink },
    xaxis: { gridcolor: C.grid, zerolinecolor: C.rule, linecolor: C.rule },
    yaxis: { gridcolor: C.grid, zerolinecolor: C.rule, linecolor: C.rule },
    legend: { orientation: 'h', y: -0.18, x: 0 },
    hovermode: 'closest', colorway: SERIES,
  }, extra);
}
/* Plotly.js >= 3 only accepts {text: …} titles; normalise string titles everywhere */
function fixTitles(o) {
  if (Array.isArray(o)) { o.forEach(fixTitles); return o; }
  if (o && typeof o === 'object') {
    for (const k of Object.keys(o)) {
      if (k === 'title' && typeof o[k] === 'string') o[k] = { text: o[k] };
      else if (o[k] && typeof o[k] === 'object' && !(o[k] instanceof Node) && !ArrayBuffer.isView(o[k]) && !(Array.isArray(o[k]) && typeof o[k][0] !== 'object')) fixTitles(o[k]);
    }
  }
  return o;
}
function plot(el, traces, layout = {}, config = {}) {
  if (!window.Plotly) {
    clear(el).append(emptyBox('Plotly could not be loaded (no local copy and no internet). Install the "plotly" Python package (a wawi dependency).'));
    return;
  }
  const L = baseLayout(layout);
  for (const ax of Object.keys(layout)) {
    if (/^[xy]axis\d*$/.test(ax)) L[ax] = Object.assign({ gridcolor: C.grid, zerolinecolor: C.rule, linecolor: C.rule }, layout[ax]);
  }
  if (traces.filter(t => t.showlegend !== false).length > 9 && !layout.legend) {
    L.legend = { orientation: 'v', x: 1.01, y: 1, font: { size: 11 } };
    L.margin = Object.assign({}, L.margin, { r: 110 });
  }
  fixTitles(L); traces.forEach(t => { if (t.colorbar) fixTitles(t.colorbar); });
  return Plotly.react(el, traces, L, Object.assign({ responsive: true, displaylogo: false,
    toImageButtonOptions: { format: 'png', scale: 2 }, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }, config));
}
function plotDiv(cls = '') { return h('div.plot' + (cls ? '.' + cls : '')); }
function images(el, imgs) { (imgs || []).forEach(src => el.append(h('img.fig', { src }))); }

/* simple table */
function table(cols, rows, { onRow, cls = '', selected } = {}) {
  const t = h('table.grid' + (onRow ? '.click' : '') + (cls ? '.' + cls : ''));
  t.append(h('thead', {}, h('tr', {}, cols.map(c => h('th' + (c.l ? '.l' : ''), {}, c.label)))));
  const tb = h('tbody');
  rows.forEach((r, i) => {
    const tr = h('tr', { onclick: onRow ? () => { $$('tr', tb).forEach(x => x.classList.remove('sel')); tr.classList.add('sel'); onRow(r, i); } : null },
      cols.map(c => h('td' + (c.l ? '.l' : ''), {}, c.f ? c.f(r, i) : fmt(r[c.k]))));
    if (selected === i) tr.classList.add('sel');
    tb.append(tr);
  });
  t.append(tb);
  return h('div.tablewrap', {}, t);
}
function toCSV(header, rows) {
  const esc = v => (v === null || v === undefined) ? '' : (typeof v === 'string' && /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  return [header.map(esc).join(','), ...rows.map(r => r.map(esc).join(','))].join('\n');
}

/* ------------------------------------------------------------------ jobs drawer */
const Jobs = {
  waiters: {}, sel: null, timer: null, list: [],
  touch() { this.schedule(150); },
  schedule(ms = 700) { clearTimeout(this.timer); this.timer = setTimeout(() => this.tick(), ms); },
  wait(id) { return new Promise((res, rej) => { this.waiters[id] = { res, rej }; this.schedule(200); }); },
  select(id) { this.sel = id; this.renderLog(); },
  async tick() {
    let busy = false;
    try {
      const st = await refreshState(true);
      busy = st && st.busy;
      for (const id of Object.keys(this.waiters)) {
        const j = await api('job/get', { id: +id });
        if (['done', 'error', 'cancelled'].includes(j.status)) {
          const w = this.waiters[id]; delete this.waiters[id];
          if (j.status === 'done') w.res(j.result);
          else w.rej(new ApiError(j.error || j.status, { traceback: j.traceback, log: j.log }));
        }
      }
      if (this.sel) await this.renderLog();
    } catch (e) { /* server gone: keep trying slowly */ }
    if (busy || Object.keys(this.waiters).length) this.schedule(700);
  },
  render(jobs) {
    this.list = jobs || [];
    const running = this.list.find(j => j.status === 'running');
    const latest = running || this.list[0];
    const dot = $('#drawerDot'); dot.className = 'dot ' + (latest ? latest.status : '');
    const prog = $('#drawerProg');
    if (running) {
      $('#drawerStatus').textContent = `${running.label} — ${running.last_line || 'running…'}`;
      prog.style.visibility = 'visible';
      if (running.progress !== null && running.progress !== undefined) { prog.classList.remove('indet'); prog.firstChild.style.width = running.progress + '%'; }
      else prog.classList.add('indet');
    } else {
      prog.style.visibility = 'hidden';
      $('#drawerStatus').textContent = latest ? `${latest.label} — ${latest.status}${latest.elapsed ? ` (${latest.elapsed.toFixed(1)} s)` : ''}` : 'No jobs yet';
    }
    const jl = clear($('#jobList'));
    for (const j of this.list) {
      jl.append(h('div.job' + (j.id === this.sel ? '.sel' : ''), { onclick: () => { this.select(j.id); this.render(this.list); } },
        h('div.t', {}, h('span', {}, h('span.dot.' + j.status), j.label), h('span.muted', {}, j.elapsed ? j.elapsed.toFixed(1) + ' s' : j.status)),
        j.error ? h('div.small', { style: { color: C.alert } }, j.error.slice(0, 140)) : null));
    }
  },
  async renderLog() {
    if (!this.sel) return;
    try {
      const j = await api('job/get', { id: this.sel });
      $('#jobTitle').textContent = `#${j.id} ${j.label} — ${j.status}`;
      const pre = $('#jobLog'); const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
      pre.textContent = j.log || (j.status === 'queued' ? 'Waiting for the previous job to finish…' : '');
      if (atBottom) pre.scrollTop = pre.scrollHeight;
      $('#jobCancel').disabled = !['running', 'queued'].includes(j.status);
    } catch (e) { /* ignore */ }
  },
};
$('#drawerHead').onclick = () => {
  const d = $('#drawer'); d.classList.toggle('open');
  $('#drawerToggle').textContent = d.classList.contains('open') ? 'Jobs ▾' : 'Jobs ▴';
  if (!Jobs.sel && Jobs.list[0]) Jobs.select(Jobs.list[0].id);
};
$('#drawerHead').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#drawerHead').click(); } };
$('#jobCancel').onclick = act(async () => { await api('job/cancel', { id: Jobs.sel }); toast('Cancellation requested'); Jobs.touch(); });
$('#jobClear').onclick = act(async () => { await api('job/clear'); Jobs.touch(); });
function openDrawer() { const d = $('#drawer'); if (!d.classList.contains('open')) $('#drawerHead').click(); }

/* ------------------------------------------------------------------ global state / chips */
let _lastModelSig = null;
async function refreshState(silent) {
  let st;
  try { st = await api('system/state'); } catch (e) { if (!silent) showError(e); return null; }
  S.state = st; S.model = st.model;
  const m = st.model;
  const chips = clear($('#chips'));
  // Colour = topic (sea / wind / results); status = ✓ and solid outline (on) vs. dashed outline (not yet)
  const chip = (txt, on, cls = '', title = '') => chips.append(h('span.chip' + (cls ? '.' + cls : '') + (on ? '.on' : ''),
    { title: (on ? '✓ ' + txt + ' available' : txt + ': not yet') + (title ? '\n' + title : '') }, (on ? '✓ ' : '') + txt));
  chip(m ? `Model · ${m.n_modes ?? '–'} modes` : 'No model', !!m, '', m && m.path ? m.path : '');
  chip('Hydro', m && m.has_hydro, 'sea'); chip('Aero', m && m.has_aero, 'wind');
  chip('Sea state', m && m.seastate, 'sea', m && m.seastate || ''); chip('Wind state', m && m.windstate, 'wind', m && m.windstate || '');
  chip('Modes', m && m.has_eig, 'res'); chip('Response', m && m.has_freqsim, 'res');
  const nav = (p, ok) => { const a = $(`nav.side a[data-page=${p}]`); if (a) a.classList.toggle('done', !!ok); };
  nav('model', !!m); nav('env', m && (m.seastate || m.windstate)); nav('analysis', m && (m.has_eig || m.has_freqsim)); nav('results', m && m.has_freqsim);
  if (st.imports && st.imports.wawi) $('#wawiVersion').textContent = 'v' + (st.imports.wawi.version || '');
  Jobs.render(st.jobs);
  const sig = m ? JSON.stringify([m.repr, m.n_modes, m.has_aero, m.has_hydro, m.seastate, m.windstate, m.has_eig, m.has_freqsim, m.has_static, m.aero_sections]) : null;
  if (sig !== _lastModelSig) { _lastModelSig = sig; bus.emit('model', m); }
  return st;
}
