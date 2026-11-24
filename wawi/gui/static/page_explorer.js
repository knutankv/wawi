/* WAWI GUI – API explorer, variables & console */
'use strict';

const BOUND_GUESS = {
  Model: 'model', Seastate: 'seastate', Windstate: 'windstate', Hydro: 'model.hydro', Aero: 'model.aero', ModalDry: 'model.modal_dry',
  Results: 'model.results', Pontoon: 'model.hydro.pontoons[0]', PontoonType: 'model.hydro.pontoon_types[0]', Environment: 'model.hydro.environment',
  AeroSection: 'list(model.aero.sections.values())[0]', ADs: 'list(model.aero.sections.values())[0].ADs', DragElement: 'model.drag_elements[0]',
  ScreeningCase: 'screening_case', Node: 'model.hydro.pontoons[0].node', LoadCoefficients: 'load_coefficients',
};

PAGES.explorer = (root) => {
  const search = h('input', { type: 'text', placeholder: 'Search functions, classes, methods…' });
  const treeEl = h('div.tree'); const pane = h('div');
  let tree = null;
  async function load(refresh) {
    tree = await api('explorer/tree', { refresh });
    renderTree();
    if (tree.compat && tree.compat.length) pane.prepend(h('div.warnbox', {}, 'NumPy 2 compatibility aliases installed for wawi: ' + tree.compat.join(', ')));
  }
  function renderTree() {
    const q = search.value.trim().toLowerCase();
    clear(treeEl);
    for (const m of tree.modules) {
      const match = (s) => !q || s.toLowerCase().includes(q);
      const items = [];
      for (const mem of m.members) {
        const subs = (mem.members || []).filter(s => match(mem.name + '.' + s.name));
        if (!match(mem.qual) && !subs.length) continue;
        const it = h('div.item', { onclick: () => select(m, mem, null, it) }, h('span.k', {}, mem.kind === 'class' ? 'class' : 'def'), h('span.nm', {}, mem.name));
        items.push(it);
        if (mem.kind === 'class') {
          for (const s of (q ? subs : mem.members)) {
            const si = h('div.item.member', { onclick: () => select(m, mem, s, si) }, h('span.k', {}, { property: 'prop', method: 'meth', staticmethod: 'stat', classmethod: 'cls' }[s.kind] || ''), h('span.nm', {}, s.name));
            items.push(si);
          }
        }
      }
      if (!items.length && q) continue;
      const d = h('details', { open: !!q || null }, h('summary', {}, h('b', {}, m.name.replace('wawi.model._', 'wawi.model › ')), m.error ? h('span.badge.alert', { style: { marginLeft: '6px' } }, 'unavailable') : h('span.muted.small', {}, ` ${m.members.length}`)));
      if (m.error) d.append(h('div.small.muted', { style: { padding: '2px 8px 6px 22px' } }, m.error));
      items.forEach(i => d.append(i));
      treeEl.append(d);
    }
  }
  search.oninput = renderTree;
  async function select(mod, mem, sub, itemEl) {
    $$('.item', treeEl).forEach(i => i.classList.remove('active')); itemEl.classList.add('active');
    const isMember = !!sub;
    const kind = isMember ? sub.kind : mem.kind;
    const qual = mem.qual + (isMember ? '.' + sub.name : '');
    const params = (isMember ? sub.params : mem.params) || [];
    const docEl = h('div.docbox', {}, 'Loading documentation…');
    api('explorer/doc', { target: isMember && kind === 'property' ? mem.qual : qual }).then(d => { docEl.textContent = d.doc; }).catch(e => { docEl.textContent = String(e.message); });
    const boundIn = h('input', { type: 'text', class: 'expr', value: BOUND_GUESS[mem.name] || mem.name.toLowerCase() });
    const assignIn = h('input', { type: 'text', value: '', placeholder: '_', style: { maxWidth: '160px' } });
    const out = h('div');
    const fields = params.filter(p => !['VAR_POSITIONAL', 'VAR_KEYWORD'].includes(p.kind)).map(p => ({ k: p.name, label: p.name, type: 'expr', ph: p.default !== undefined ? `default: ${p.default}` : 'required' }));
    const hasKw = params.some(p => p.kind === 'VAR_KEYWORD');
    if (hasKw) fields.push({ k: '__kwargs', label: '**kwargs', type: 'expr', span: 'full', ph: 'dict(key=value, …)' });
    if (params.some(p => p.kind === 'VAR_POSITIONAL')) fields.push({ k: '__args', label: '*args', type: 'expr', span: 'full', ph: '[a, b]' });
    const form = makeForm('x_' + qual, fields);
    const target = () => (isMember && (kind === 'method' || kind === 'property')) ? `${boundIn.value.trim()}.${sub.name}` : qual;
    const call = act(async () => {
      const v = form.get(); const kwargs = {}; const args = [];
      for (const p of params) {
        if (p.kind === 'VAR_POSITIONAL' || p.kind === 'VAR_KEYWORD') continue;
        const val = (v[p.name] || '').trim();
        if (!val) { if (p.default === undefined) throw new ApiError(`Parameter '${p.name}' is required`); continue; }
        if (p.kind === 'POSITIONAL_ONLY') args.push(val); else kwargs[p.name] = val;
      }
      const tgt = target(); const name = assignIn.value.trim();
      if ((v.__kwargs || '').trim() || (v.__args || '').trim()) {
        const parts = [...args, ...(v.__args ? [`*(${v.__args})`] : []), ...Object.entries(kwargs).map(([k, x]) => `${k}=${x}`), ...(v.__kwargs ? [`**(${v.__kwargs})`] : [])];
        const code = `${name || '_'} = ${tgt}(${parts.join(', ')})\n${name || '_'}`;
        const r = await run('console/exec', { code });
        showResult(r, name || '_');
        return;
      }
      const payload = { target: tgt, args, kwargs, assign_to: name };
      const r = await run('explorer/call', payload);
      showResult(r, r.stored_as);
    });
    function showResult(r, name) {
      clear(out);
      const s = r.summary || {};
      out.append(h('div.row', {}, h('b', {}, `Result stored as ${name}`), h('span.badge', {}, s.type || ''), s.shape ? h('span.badge', {}, s.shape.join('×')) : null),
        h('pre.out', {}, s.str || s.repr || JSON.stringify(s.value ?? '', null, 1)));
      images(out, r.images);
      const acts = h('div.row', { style: { marginTop: '8px' } }, btn('Inspect', () => { showPage('console'); PAGES._console.inspect(name); }, 'small'));
      if (s.kind === 'array') acts.append(btn('Plot', () => { showPage('console'); PAGES._console.inspect(name, true); }, 'small'));
      out.append(acts);
    }
    const evalProp = act(async () => {
      const r = await api('vars/inspect', { expr: target() });
      clear(out).append(h('pre.out', {}, r.str || r.repr || JSON.stringify(r.value)), h('div.row', {}, btn('Open in inspector', () => { showPage('console'); PAGES._console.inspect(target()); }, 'small')));
    });
    const sig = isMember ? sub.signature : mem.signature;
    clear(pane).append(card({ title: qual }, null,
      h('div.sig', {}, kind === 'property' ? '(property)' : `${isMember ? sub.name : mem.name}${sig || '(…)'}`),
      h('div', { style: { margin: '10px 0' } }, docEl),
      isMember && (kind === 'method' || kind === 'property') ? h('div.field', { style: { marginBottom: '10px' } }, h('label', {}, 'Object (instance expression)'), boundIn) : null,
      kind === 'property' ? h('div.row', {}, btn('Evaluate property', evalProp, 'primary')) : [
        fields.length ? form.el : h('p.muted.small', {}, 'No parameters.'),
        h('div.row', { style: { marginTop: '12px' } }, btn(kind === 'class' ? 'Create instance' : 'Call', call, 'primary'), 'Store result as', assignIn,
          h('span.small.muted', {}, 'Arguments are Python expressions evaluated in the session (np, wawi, model, …).'))],
      h('hr.thin'), out));
  }
  root.append(h('div.explorer', {}, card({}, null, search, h('div', { style: { height: '8px' } }), treeEl),
    h('div', {}, pane)));
  clear(pane).append(emptyBox('Choose a function, class or method on the left to read its documentation and call it with your own arguments.'));
  load().catch(showError);
};

/* ------------------------------------------------------------------ variables & console */
const SNIPPETS = {
  'Std. dev. of girder forces': "std = model.get_result_std(key='girder_forces')\nstd[0::6]  # axial force N",
  'Plot first mode (matplotlib)': "phi = model.get_phi('full')\nplt.plot(np.real(phi[2::6, 0]))\nplt.title('Vertical component, mode 1')",
  'Compare with/without drag': "m2 = model.copy()\nm2.run_freqsim(np.arange(0.001, 2, 0.005), include_selfexcited=['hydro', 'drag_pontoons'], drag_iterations=5, print_progress=False)\nmodel.get_result_std(key='hydro')[2::6], m2.get_result_std(key='hydro')[2::6]",
  'Inhomogeneous sea state': "seastate = Seastate(lambda x, y: 8.3 + 0.1*x/1500, lambda x: 2.1 - 0.5*x/1500, 8, lambda x: 90 - 10*x/1500, 12, centered_dirdist=False)\nmodel.assign_seastate(seastate)",
  'Wave excitation CPSD + simulation': "omega = np.arange(0.05, 2.0, 0.05)\nSqq = model.evaluate_waveaction(omega, transform_by_phi=False)\nfrom wawi.time_domain import simulate_mdof\nq, t = simulate_mdof(Sqq, omega, fs=10.0, tmax=300)\nplt.plot(t, q[2])",
  'ADs plot of a section': "list(model.aero.sections.values())[0].ADs.plot(v=np.arange(0.1, 5, 0.01))",
};

PAGES.console = (root) => {
  const list = h('div.varlist'); const insp = h('div'); const out = h('div.console-out'); const ta = h('textarea.code', { rows: 7, spellcheck: 'false', placeholder: 'Python code – the namespace contains np, plt, wawi, model, seastate, windstate and all classes. Ctrl+Enter runs.' });
  const hist = JSON.parse(localStorage.getItem('wawi.hist') || '[]'); let hi = hist.length;
  let selected = null;
  async function refresh() {
    let vs; try { vs = await api('vars/list'); } catch (e) { return; }
    clear(list);
    if (!vs.length) list.append(h('div.small.muted', { style: { padding: '8px' } }, 'No variables yet.'));
    vs.forEach(v => list.append(h('div.var' + (v.name === selected ? '.sel' : ''), { onclick: () => inspect(v.name) }, h('span.nm', {}, v.name),
      h('span.ty', {}, v.shape ? `${v.dtype} ${v.shape.join('×')}` : v.type))));
  }
  async function inspect(expr, autoplot = false) {
    selected = expr;
    let d; try { d = await api('vars/inspect', { expr }); } catch (e) { showError(e); return; }
    const plotEl = plotDiv();
    const xIn = h('input', { type: 'text', class: 'expr', placeholder: 'x values (optional), e.g. model.results.omega' });
    const doPlot = act(async () => {
      const r = await api('vars/data', { expr, x: xIn.value });
      const dat = r.data;
      const real = dat && dat.complex ? dat.re : dat, imag = dat && dat.complex ? dat.im : null;
      if (r.shape.length === 1) {
        const x = r.x || real.map((_, i) => i);
        plot(plotEl, [{ x, y: real, name: imag ? 'Re' : expr, mode: 'lines' }, ...(imag ? [{ x, y: imag, name: 'Im', mode: 'lines' }] : [])], { title: { text: expr, font: { size: 13 } } });
      } else if (r.shape.length === 2) {
        plot(plotEl, [{ type: 'heatmap', z: real, colorscale: 'Viridis' }], { title: { text: expr + (imag ? ' (real part)' : ''), font: { size: 13 } }, yaxis: { autorange: 'reversed' } });
      } else clear(plotEl).append(emptyBox(`${r.shape.length}-D array – plot a slice, e.g. ${expr}[0, 0, :]`));
    });
    const attrRows = (d.attrs || []).map(a => h('div.var', { onclick: () => inspect(`${expr}.${a.name}`) }, h('span.nm', {}, a.name), h('span.ty', {}, a.kind === 'property' ? 'property' : `${a.type || ''} ${a.repr || ''}`)));
    const methRows = (d.methods || []).map(m => h('div.var', { title: m.doc, onclick: () => { ta.value = `${expr}.${m.name}(${(m.params || []).filter(p => p.default === undefined && !p.kind.startsWith('VAR')).map(p => p.name + '=').join(', ')})`; ta.focus(); } }, h('span.nm', {}, m.name), h('span.ty', {}, m.signature || '')));
    const entryRows = (d.entries || []).map(e => h('div.var', { onclick: () => inspect(e.keyexpr) }, h('span.nm', {}, e.key), h('span.ty', {}, `${e.type} ${e.repr}`)));
    const parent = expr.includes('.') || expr.includes('[') ? expr.replace(/(\.[A-Za-z_]\w*|\[[^\]]*\])$/, '') : null;
    clear(insp).append(card({ title: expr }, null,
      h('div.row', {}, h('span.badge', {}, d.type), d.shape ? h('span.badge', {}, d.shape.join('×') + ' ' + d.dtype) : null,
        parent && parent !== expr ? btn('↑ ' + parent, () => inspect(parent), 'small ghost') : null, h('span.spacer'),
        d.kind === 'array' ? btn('Plot', doPlot, 'small') : null,
        btn('NPZ', act(async () => triggerDownload((await api('vars/download', { expr, format: 'npz' })).download)), 'small ghost'),
        d.kind === 'array' ? btn('CSV', act(async () => triggerDownload((await api('vars/download', { expr, format: 'csv' })).download)), 'small ghost') : null,
        btn('Pickle', act(async () => triggerDownload((await api('vars/download', { expr, format: 'pickle' })).download)), 'small ghost'),
        /^[A-Za-z_]\w*$/.test(expr) ? btn('Delete', act(async () => { await api('vars/delete', { name: expr }); clear(insp); refresh(); }), 'small ghost danger') : null),
      h('pre.out', {}, d.str || d.repr || JSON.stringify(d.value ?? d.head ?? '')),
      d.kind === 'array' ? h('div', {}, h('div.row', { style: { marginTop: '8px' } }, xIn), plotEl) : null,
      entryRows.length ? h('details.adv', { open: true }, h('summary', {}, `Items (${entryRows.length})`), h('div.varlist', {}, entryRows)) : null,
      attrRows.length ? h('details.adv', { open: true }, h('summary', {}, `Attributes and properties (${attrRows.length})`), h('div.varlist', { style: { maxHeight: '300px', overflow: 'auto' } }, attrRows)) : null,
      methRows.length ? h('details.adv', {}, h('summary', {}, `Methods (${methRows.length}) – click to insert a call`), h('div.varlist', { style: { maxHeight: '300px', overflow: 'auto' } }, methRows)) : null));
    refresh();
    if (autoplot && d.kind === 'array') doPlot();
  }
  const runCode = act(async () => {
    const code = ta.value; if (!code.trim()) return;
    hist.push(code); if (hist.length > 100) hist.shift(); hi = hist.length; localStorage.setItem('wawi.hist', JSON.stringify(hist));
    out.append(h('div.in', {}, code.split('\n').map((l, i) => (i ? '... ' : '>>> ') + l).join('\n')));
    try {
      const d = await api('console/exec', { code });
      let res = d;
      if (!d.done) res = { result: await Jobs.wait(d.job), log: (await api('job/get', { id: d.job })).log };
      if (res.log) out.append(h('div', {}, res.log.replace(/\n$/, '')));
      images(out, (res.result || {}).images);
      ta.value = '';
    } catch (e) {
      out.append(h('div.err', {}, (e.detail && (e.detail.log || e.detail.traceback)) || e.message));
    }
    out.scrollTop = out.scrollHeight;
    refresh();
  });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runCode(); }
    if (e.key === 'ArrowUp' && e.altKey && hi > 0) { hi--; ta.value = hist[hi]; }
    if (e.key === 'ArrowDown' && e.altKey && hi < hist.length) { hi++; ta.value = hist[hi] || ''; }
    if (e.key === 'Tab') { e.preventDefault(); const s = ta.selectionStart; ta.value = ta.value.slice(0, s) + '    ' + ta.value.slice(ta.selectionEnd); ta.selectionStart = ta.selectionEnd = s + 4; }
  });
  const snip = h('select', {}, h('option', { value: '' }, 'Insert example…'), Object.keys(SNIPPETS).map(k => h('option', { value: k }, k)));
  snip.onchange = () => { if (snip.value) ta.value = SNIPPETS[snip.value]; snip.value = ''; ta.focus(); };
  const exprIn = h('input', { type: 'text', class: 'expr', placeholder: 'any expression, e.g. model.hydro.pontoons[0]' });
  exprIn.addEventListener('keydown', e => { if (e.key === 'Enter') inspect(exprIn.value.trim()); });
  out.append(h('div', { style: { color: '#8ea4b3' } }, 'WAWI session console. Code runs in the background job queue, so long computations do not block the GUI.'));
  root.append(h('div.explorer', {},
    h('div', {}, card('Variables', null, h('div.row', { style: { marginBottom: '6px' } }, btn('Refresh', refresh, 'small ghost')), h('div', { style: { maxHeight: '45vh', overflow: 'auto' } }, list)),
      card('Inspect expression', null, h('div.pathrow', {}, exprIn, btn('Inspect', () => inspect(exprIn.value.trim()), 'small')))),
    h('div', {}, card('Python console', null, out, h('div', { style: { height: '10px' } }), ta,
      h('div.row', { style: { marginTop: '8px' } }, btn('Run', runCode, 'primary'), snip, h('span.small.muted', {}, 'Ctrl+Enter to run · Alt+↑/↓ history'), h('span.spacer'), btn('Clear output', () => clear(out), 'small ghost'))), insp)));
  PAGES._console = { inspect, refresh };
  refresh(); bus.on('model', refresh);
};
