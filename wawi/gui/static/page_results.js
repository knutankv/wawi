/* WAWI GUI – Results page */
'use strict';

PAGES.results = (root) => {
  const tabs = subtabs(root, [
    { id: 'modes', label: 'Mode shapes', build: buildResModes, onShow: el => el._refresh && el._refresh() },
    { id: 'stats', label: 'Response statistics', build: buildResStats, onShow: el => el._refresh && el._refresh() },
    { id: 'psd', label: 'Response spectra', build: buildResPsd, onShow: el => el._refresh && el._refresh() },
    { id: 'static', label: 'Static displacement', build: buildResStatic, onShow: el => el._refresh && el._refresh() },
    { id: 'system', label: 'System matrices & FRF', build: buildResSystem },
    { id: 'export', label: 'Export', build: buildResExport },
  ], { key: 'results' });
  PAGES._results = {
    tab: (id) => tabs.show(id),
    showMode: (source, i) => { tabs.show('modes'); setTimeout(() => PAGES._viewer && PAGES._viewer.showMode(source, i).catch(showError), 50); },
  };
};

let OVERVIEW = null;
async function overview() { OVERVIEW = await api('results/overview'); return OVERVIEW; }
function keyOptions(o, withModal = true) {
  return [...(withModal ? [['', 'Modal coordinates']] : []), ...Object.entries(o.keys).map(([k, s]) => [k, `${k} (${s[0]} rows)`])];
}

/* ------------------------------------------------------------------ modes */
function buildResModes(el) {
  const v = new Viewer3D({ modes: true }); PAGES._viewer = v;
  const tbl = h('div'); const srcSel = h('select', {}, h('option', { value: 'wet' }, 'Wet / in-wind modes'), h('option', { value: 'dry' }, 'Dry modes'));
  const props = h('div');
  const renderTable = () => {
    const o = OVERVIEW; clear(tbl);
    if (srcSel.value === 'wet') {
      const r = o.modal;
      if (!r) { tbl.append(emptyBox('No wet/in-wind modes yet – run the modal analysis.')); return; }
      tbl.append(table([{ label: 'Mode', f: i => i + 1 }, { label: 'fn [Hz]', f: i => fmt(r.fn[i]) }, { label: 'Tn [s]', f: i => fmt(r.Tn[i]) }, { label: 'ξ [%]', f: i => fmt(r.xi[i] * 100, 3) },
        ...(r.mpc ? [{ label: 'MPC', f: i => fmt(r.mpc[i], 3) }] : [])],
        r.fn.map((_, i) => i), { onRow: i => v.showMode('wet', i).catch(showError) }));
    } else {
      const d = o.dry; if (!d) { tbl.append(emptyBox('No dry modes.')); return; }
      tbl.append(table([{ label: 'Mode', f: i => i + 1 }, { label: 'fn [Hz]', f: i => fmt(d.fn[i]) }, { label: 'Tn [s]', f: i => fmt(d.Tn[i]) }, { label: 'ξ0 [%]', f: i => fmt((d.xi0[i] ?? d.xi0[0]) * 100, 3) }],
        d.fn.map((_, i) => i), { onRow: i => v.showMode('dry', i).catch(showError) }));
    }
  };
  srcSel.onchange = renderTable;
  el._refresh = async () => {
    if (!S.model) { clear(el).append(emptyBox('Load a model first.')); el._built = false; return; }
    if (!el._built) {
      clear(el).append(h('div.cols', {}, h('div', {}, card('Modes', 'Click a row to display the mode shape.', h('div.row', { style: { marginBottom: '8px' } }, srcSel), tbl),
        card('Modal mass, stiffness and damping', 'Diagonal terms of the combined system (Results.mdiag / kdiag / cdiag).', btn('Compute', act(async () => {
          const r = await run('results/modal_props');
          clear(props).append(table([{ label: 'Mode', f: i => i + 1 }, { label: 'm', f: i => fmt(r.mdiag[i]) }, { label: 'k', f: i => fmt(r.kdiag[i]) }, { label: 'c', f: i => fmt(r.cdiag[i]) }], r.mdiag.map((_, i) => i)));
        }), 'small'), props)), v.el));
      el._built = true;
      try { await v.load(); } catch (e) { showError(e); }
    }
    await overview(); if (!OVERVIEW.modal) srcSel.value = 'dry'; renderTable();
  };
  bus.on('model', () => { if (el._built) { v.load().catch(() => {}); overview().then(renderTable).catch(() => {}); } });
}

/* ------------------------------------------------------------------ statistics */
function buildResStats(el) {
  const f = makeForm('rstats', [
    { k: 'quantity', label: 'Quantity', type: 'select', opts: [['std', 'Standard deviation'], ['expmax', 'Expected maximum'], ['peakfactor', 'Peak factor']], def: 'std' },
    { k: 'key', label: 'Output (mode-shape key)', type: 'select', opts: [] },
    { k: 'subset', label: 'Restrict to aero section', type: 'select', opts: [['', '–']] },
    { k: 'h', label: 'Response type', type: 'select', opts: [['disp', 'Displacement / force'], ['vel', 'Velocity (×ω²)'], ['acc', 'Acceleration (×ω⁴)'], ['custom', 'Custom h(ω)']], def: 'disp' },
    { k: 'h_custom', label: 'Custom h(ω)', type: 'expr', ph: 'lambda om: om**2' },
    { k: 'T', label: 'Duration T [s]', type: 'expr', def: '3600' },
    { k: 'convert_to', label: 'Coordinate system (std)', type: 'select', opts: [['', 'As defined by φ'], ['global', 'Global'], ['local', 'Local']], def: '' },
    { k: 'modes', label: 'Modes to include', type: 'expr', ph: 'all, e.g. np.arange(10)' },
  ]);
  const xSel = h('select', {}, ['arclength', 'x', 'y', 'z', 'index'].map(x => h('option', { value: x }, x)));
  const compBox = h('div.multi');
  const scale = h('input', { type: 'text', value: '1', style: { width: '70px' } });
  const pl = plotDiv('tall'); const tbl = h('div');
  let res = null; const pins = [];
  const draw = () => {
    if (!res) return;
    const comps = $$('input', compBox).filter(c => c.checked).map(c => +c.value);
    const xk = res.pos[xSel.value] ? xSel.value : 'index';
    const X = res.pos[xk]; const sc = +scale.value || 1;
    const tr = [];
    comps.forEach((c, i) => {
      const ax = i ? String(i + 1) : '';
      pins.forEach((p, pi) => { if (p.values[0].length > c) tr.push({ x: p.pos[xk] || p.pos.index, y: p.values.map(r => r[c] === null ? null : r[c] * sc), name: p.name, mode: 'lines', line: { dash: 'dot', color: SERIES[(pi + 1) % SERIES.length] }, xaxis: 'x' + ax, yaxis: 'y' + ax, legendgroup: p.name, showlegend: i === 0 }); });
      tr.push({ x: X, y: res.values.map(r => r[c] === null ? null : r[c] * sc), name: 'Current', mode: res.labels.length > 60 ? 'lines' : 'lines+markers', marker: { size: 4 }, line: { color: C.ink, width: 2 }, text: res.labels, xaxis: 'x' + ax, yaxis: 'y' + ax, legendgroup: 'cur', showlegend: i === 0 });
    });
    const lay = { grid: { rows: Math.max(comps.length, 1), columns: 1, pattern: 'independent' }, height: Math.max(380, 200 * comps.length), legend: { orientation: 'h', y: 1.04, x: 0 }, margin: { l: 70, r: 18, t: 30, b: 46 } };
    comps.forEach((c, i) => { const ax = i ? String(i + 1) : ''; lay['yaxis' + ax] = { title: res.comps[c] || `c${c}` }; lay['xaxis' + ax] = { title: i === comps.length - 1 ? xk : '' }; });
    plot(pl, tr, lay);
  };
  const fillTable = () => {
    clear(tbl).append(table([{ label: 'Point', l: true, f: (r, i) => res.labels[i] }, ...res.comps.map((c, j) => ({ label: c, f: r => fmt(r[j]) }))], res.values),
      h('div.row', { style: { marginTop: '6px' } }, btn('Download CSV', () => downloadText(`${res.quantity}_${res.key || 'modal'}.csv`,
        toCSV(['point', ...Object.keys(res.pos), ...res.comps], res.values.map((r, i) => [res.labels[i], ...Object.keys(res.pos).map(k => res.pos[k][i]), ...r]))), 'small ghost')));
  };
  const compute = act(async () => {
    const a = f.get();
    res = await api('results/stats', { ...a, h: a.h === 'custom' ? a.h_custom : a.h });
    const prev = new Set($$('input', compBox).filter(c => c.checked).map(c => c.value));
    clear(compBox).append(...res.comps.map((c, j) => { const i = h('input', { type: 'checkbox', value: j }); i.checked = prev.size ? prev.has(String(j)) : j < 3; i.onchange = draw; return h('label', {}, i, c); }));
    draw(); fillTable();
  });
  xSel.onchange = draw; scale.onchange = draw;
  el._refresh = async () => {
    if (!S.model) { clear(el).append(emptyBox('Load a model first.')); el._built = false; return; }
    const o = await overview();
    if (!el._built) {
      clear(el).append(h('div.cols', {}, card('Response statistics along the structure', 'Converts the modal response spectra to physical quantities through the selected mode-shape matrix (get_result_std / get_result_expmax / get_result_peakfactor).',
        f.el, h('div.row', { style: { marginTop: '12px' } }, btn('Compute', compute, 'primary'))),
      card({}, null, h('div.row', {}, 'x-axis', xSel, 'Scale', scale, btn('Pin curve', () => { if (res) { pins.push({ ...res, name: prompt('Name for pinned curve', `${res.quantity} ${pins.length + 1}`) || `pin ${pins.length + 1}` }); draw(); } }, 'small ghost'),
        btn('Clear pins', () => { pins.length = 0; draw(); }, 'small ghost')), h('div.row', {}, 'Components', compBox), pl, tbl)));
      el._built = true;
    }
    f.setOptions('key', keyOptions(o)); f.setOptions('subset', [['', '–'], ...o.aero_groups.map(g => [g, g])]);
    if (!o.has_freqsim) clear(tbl).append(emptyBox('No frequency-domain results yet. Run Analysis → Response.'));
  };
}

/* ------------------------------------------------------------------ spectra */
function buildResPsd(el) {
  const f = makeForm('rpsd', [
    { k: 'key', label: 'Output (mode-shape key)', type: 'select', opts: [] },
    { k: 'sel', label: 'Select by', type: 'select', opts: [['node', 'FE node label'], ['pontoon', 'Pontoon'], ['index', 'Row index of φ']], def: 'index' },
    { k: 'node', label: 'Node label', type: 'expr' },
    { k: 'pontoon', label: 'Pontoon', type: 'select', opts: [] },
    { k: 'index', label: 'Row indices', type: 'expr', def: '[0, 1, 2]' },
    { k: 'dofs', label: 'DOFs (node / pontoon)', type: 'multi', opts: DOFS.map((d, i) => [String(i), d]), def: ['0', '1', '2'], span: 'full' },
    { k: 'convert_to', label: 'Coordinate system', type: 'select', opts: [['', 'As defined by φ'], ['global', 'Global'], ['local', 'Local']], def: '' },
    { k: 'h', label: 'Response type', type: 'select', opts: [['disp', 'Displacement / force'], ['vel', 'Velocity'], ['acc', 'Acceleration']], def: 'disp' },
    { k: 'modes', label: 'Modes to include', type: 'expr', ph: 'all' },
  ]);
  const logY = h('input', { type: 'checkbox' }); const logX = h('input', { type: 'checkbox' }); const cross = h('input', { type: 'checkbox' });
  const pl = plotDiv('tall'); const info = h('div.stats'); let res = null;
  const draw = () => {
    if (!res) return;
    clear(info).append(...res.names.map((n, i) => h('div.stat', {}, h('b', {}, fmt(res.std[i], 4)), h('span', {}, `σ ${n}`))));
    if (cross.checked && res.cross) {
      const n = res.names.length; const tr = []; const lay = { grid: { rows: n, columns: n, pattern: 'independent' }, height: Math.max(420, 170 * n), showlegend: false, margin: { l: 60, r: 10, t: 30, b: 40 } };
      const Sre = re(res.cross), Sim = im(res.cross);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const ax = i * n + j ? String(i * n + j + 1) : '';
        tr.push({ x: res.omega, y: Sre[i][j], line: { color: C.sea, width: 1.4 }, xaxis: 'x' + ax, yaxis: 'y' + ax, name: 'Re' });
        if (Sim) tr.push({ x: res.omega, y: Sim[i][j], line: { color: C.alert, width: 1 }, xaxis: 'x' + ax, yaxis: 'y' + ax, name: 'Im' });
        lay['xaxis' + ax] = { type: logX.checked ? 'log' : 'linear', title: i === n - 1 ? 'ω' : '' };
        lay['yaxis' + ax] = { title: j === 0 ? res.names[i] : '' };
      }
      lay.title = { text: 'Cross-spectral density matrix (blue: real, red: imaginary)', font: { size: 13 } };
      plot(pl, tr, lay);
    } else {
      plot(pl, res.names.map((n, i) => ({ x: res.omega, y: res.auto[i], name: n, mode: 'lines' })),
        { xaxis: { title: 'ω [rad/s]', type: logX.checked ? 'log' : 'linear' }, yaxis: { title: 'Auto-spectral density', type: logY.checked ? 'log' : 'linear' }, title: { text: 'Response spectra', font: { size: 13 } } });
    }
  };
  [logX, logY, cross].forEach(c => c.onchange = draw);
  const compute = act(async () => { res = await api('results/psd', f.get()); draw(); });
  el._refresh = async () => {
    if (!S.model) { clear(el).append(emptyBox('Load a model first.')); el._built = false; return; }
    const o = await overview();
    if (!el._built) {
      clear(el).append(h('div.cols', {}, card('Response spectra', 'Auto and cross spectral densities of selected physical responses (Model.get_result_psd), modal coordinates, or generalised coordinates (get_gen_S).',
        f.el, h('div.row', { style: { marginTop: '12px' } }, btn('Plot spectra', compute, 'primary'))),
      card({}, null, h('div.row', {}, h('label.small', {}, logX, ' log ω'), h('label.small', {}, logY, ' log S'), h('label.small', {}, cross, ' cross-spectra matrix (≤ 6)'),
        btn('CSV', () => res && downloadText('psd.csv', toCSV(['omega', ...res.names], res.omega.map((w, k) => [w, ...res.auto.map(a => a[k])]))), 'small ghost')), info, pl)));
      el._built = true;
    }
    f.setOptions('key', [...keyOptions(o), ['generalized', 'Generalised (combined-system) modes']]);
    f.setOptions('pontoon', o.pontoons);
  };
}

/* ------------------------------------------------------------------ static */
function buildResStatic(el) {
  const f = makeForm('rstatic', [{ k: 'key', label: 'Output (mode-shape key)', type: 'select', opts: [] }, { k: 'subset', label: 'Restrict to aero section', type: 'select', opts: [['', '–']] }]);
  const pl = plotDiv('tall'); const xSel = h('select', {}, ['arclength', 'x', 'y', 'z', 'index'].map(x => h('option', { value: x }, x))); let res = null;
  const draw = () => {
    if (!res) return; const xk = res.pos[xSel.value] ? xSel.value : 'index';
    const n = res.comps.length; const lay = { grid: { rows: n, columns: 1, pattern: 'independent' }, height: 140 * n + 80, showlegend: false, margin: { l: 70, r: 10, t: 20, b: 40 } };
    plot(pl, res.comps.map((c, j) => ({ x: res.pos[xk], y: res.values.map(r => r[j]), xaxis: 'x' + (j ? j + 1 : ''), yaxis: 'y' + (j ? j + 1 : ''), mode: 'lines', line: { color: C.wind } })),
      Object.assign(lay, ...res.comps.map((c, j) => ({ ['yaxis' + (j ? j + 1 : '')]: { title: c } }))));
  };
  xSel.onchange = draw;
  el._refresh = async () => {
    if (!S.model) { clear(el).append(emptyBox('Load a model first.')); el._built = false; return; }
    const o = await overview();
    if (!el._built) {
      clear(el).append(h('div.cols', {}, card({ title: 'Static displacement', cls: 'wind' }, 'Physical static response from results.y_static.', f.el,
        h('div.row', { style: { marginTop: '12px' } }, btn('Plot', act(async () => { res = await api('results/static', f.get()); draw(); }), 'primary'), btn('Show deformed shape', () => PAGES._results.showMode('static', 0), 'ghost'))),
      card({}, null, h('div.row', {}, 'x-axis', xSel), pl)));
      el._built = true;
    }
    f.setOptions('key', keyOptions(o, false)); f.setOptions('subset', [['', '–'], ...o.aero_groups.map(g => [g, g])]);
  };
}

/* ------------------------------------------------------------------ system matrices */
function buildResSystem(el) {
  const f = makeForm('rsys', [
    { k: 'include', label: 'Include', type: 'multi', opts: INCL_OPTS, def: ['hydro', 'aero'], span: 'full' },
    { k: 'omega_k', label: 'Frequency ω for matrices', type: 'expr', def: '0.5' },
    { k: 'frf_modes', label: 'FRF diagonal terms (modes)', type: 'expr', def: '[0, 1, 2, 3]' },
    { k: 'omega', label: 'ω for FRF', type: 'expr', def: 'np.linspace(0.01, 2, 400)', span: 2 },
  ]);
  const pK = plotDiv('short'), pC = plotDiv('short'), pM = plotDiv('short'), pF = plotDiv();
  el.append(h('div.cols', {}, card('System matrices and frequency response', 'Modal K(ω), C(ω), M(ω) of the combined system and the diagonal of the frequency response function (get_system_matrices / get_frf_fun).', f.el,
    h('div.row', { style: { marginTop: '12px' } }, btn('Evaluate', act(async () => {
      modelGuard();
      const r = await api('results/system', f.get());
      const hm = (el2, M, t) => plot(el2, [{ type: 'heatmap', z: M.map(row => row.map(v => v === null ? null : Math.sign(v) * Math.log10(1 + Math.abs(v)))), colorscale: 'RdBu', reversescale: true, zmid: 0, colorbar: { title: 'sgn·log₁₀(1+|x|)' } }],
        { title: { text: `${t} at ω = ${r.omega_k}`, font: { size: 13 } }, yaxis: { autorange: 'reversed', title: 'mode' }, xaxis: { title: 'mode' }, margin: { l: 50, r: 10, t: 30, b: 40 } });
      hm(pK, r.K, 'K'); hm(pC, r.C, 'C'); hm(pM, r.M, 'M');
      if (r.frf) plot(pF, r.frf.modes.map((m, i) => ({ x: r.frf.omega, y: r.frf.absH[i], name: `|H${m + 1},${m + 1}|`, mode: 'lines' })), { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: '|H|', type: 'log' }, title: { text: 'Frequency response (modal diagonal)', font: { size: 13 } } });
    }), 'primary'))),
  card({}, null, h('div.cols.even', {}, pK, pC), pM, pF)));
}

/* ------------------------------------------------------------------ export */
function buildResExport(el) {
  const f = makeForm('rexport', [{ k: 'path', label: 'Save NPZ to (optional)', type: 'path', pathKind: 'save', ext: ['.npz'], span: 'full' }]);
  el.append(h('div.cols.even', {}, card('Export results', 'Writes omega, S, psi, lambd, modal properties and y_static to a NumPy .npz archive.', f.el,
    h('div.row', { style: { marginTop: '12px' } }, btn('Export', act(async () => { const r = await api('results/export_npz', f.get()); if (r.download) triggerDownload(r.download); else toast('Saved ' + r.path); }), 'primary'))),
  card('Save everything', 'The .wwi model file (Model → Open & save) stores the model together with all results, so it can be reopened later in the GUI or in Python with Model.from_wwi.')));
}
