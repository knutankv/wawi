/* WAWI GUI – Analysis page */
'use strict';

PAGES.analysis = (root) => {
  subtabs(root, [
    { id: 'eig', label: 'Modal analysis', build: buildEig },
    { id: 'freq', label: 'Response (frequency domain)', build: buildFreq },
    { id: 'static', label: 'Static wind', build: buildStatic },
    { id: 'flutter', label: 'Flutter', build: buildFlutter },
    { id: 'screen', label: 'Sea-state screening', build: buildScreening },
    { id: 'loads', label: 'Load time histories', build: buildLoads },
    { id: 'prep', label: 'Preparation', build: buildPrep },
  ], { key: 'analysis' });
};

const INCL_OPTS = [['hydro', 'Hydrodynamic'], ['aero', 'Aerodynamic'], ['drag_elements', 'Drag elements'], ['drag_pontoons', 'Pontoon drag']];
function aeroSectionField(k = 'aero_sections') { return { k, label: 'Aero sections (none ticked = all)', type: 'multi', opts: [], span: 'full' }; }
function syncSections(form, k = 'aero_sections') {
  const upd = () => form.setOptions(k, (S.model && S.model.aero_sections) || []);
  upd(); bus.on('model', upd);
}
function modelGuard() { if (!S.model) throw new ApiError('Load or import a model first (Model page).'); }

/* ------------------------------------------------------------------ modal analysis */
function buildEig(el) {
  const f = makeForm('eig', [
    { k: 'include', label: 'Include in system matrices', type: 'multi', opts: INCL_OPTS, def: ['hydro', 'aero', 'drag_elements', 'drag_pontoons'], span: 'full' },
    { k: 'freq_kind', label: 'Solver', type: 'select', opts: [['true', 'iteig_freq (frequency iteration)'], ['false', 'iteig (state-space iteration)']], def: 'true', span: 2 },
    { k: 'normalize', label: 'Normalise mode shapes', type: 'check', def: false },
    { k: 'U0', label: 'Mean wind speed U0 [m/s]', type: 'expr', hint: 'Empty: U0 of the assigned wind state. Only self-excited aero forces enter, no static wind.' },
    { k: 'omega_aero', label: 'Prepare aero matrices on ω', type: 'expr', def: 'np.arange(0.001, 2, 0.05)', span: 2, hint: 'Leave empty to use existing or default preparation' },
    aeroSectionField(),
  ]);
  const fA = makeForm('eig_adv', [
    { k: 'itmax', label: 'Max iterations', type: 'expr', ph: '15' }, { k: 'tol', label: 'Tolerance', type: 'expr', ph: '1e-4' },
    { k: 'mac_min', label: 'Minimum MAC', type: 'expr', ph: '0.98' }, { k: 'w_initial', label: 'Initial ω guesses', type: 'expr', ph: '[0.3, 0.6, 1.5]' },
    { k: 'reference_omega', label: 'Reference ω (iteig_freq)', type: 'expr' }, { k: 'omega_ref', label: 'Reference ω (iteig)', type: 'expr' },
    { k: 'keep_full', label: 'Keep full eigen-solution', type: 'check', def: false }, { k: 'print_warnings', label: 'Print warnings', type: 'check', def: true },
    { k: 'divergence_protection', label: 'Divergence protection', type: 'check', def: true }, { k: 'track_by_psi', label: 'Track modes by shape (iteig)', type: 'check', def: true },
    { k: 'smooth_omega', label: 'Smooth matrices on ω', type: 'expr', span: 2, hint: 'Savitzky–Golay smoothing of K, C, M before solving' },
    { k: 'smooth_window', label: 'Window length', type: 'expr', def: '11' }, { k: 'smooth_polyorder', label: 'Polynomial order', type: 'expr', def: '3' },
    { k: 'kwargs', label: 'Other keyword arguments', type: 'expr', span: 'full', ph: 'dict(…)' },
  ]);
  syncSections(f);
  // Placeholder of the U0 field shows the value that is used when the field is left empty
  const updU0 = (m) => {
    f.inputs.U0.input.placeholder = !m || !m.has_aero ? 'no aero part'
      : m.wind_U0 != null ? `${fmt(m.wind_U0, 4)} (assigned wind state)` : 'no wind state assigned';
  };
  updU0(S.model); bus.on('model', updU0);
  const out = h('div');
  const showTable = (r) => {
    clear(out);
    if (!r) return;
    const rows = r.fn.map((_, i) => i);
    if (r.U0 != null) out.append(h('p.small.muted', {}, `In-wind modes at U0 = ${fmt(r.U0, 4)} m/s`));
    out.append(table([{ label: 'Mode', f: i => i + 1 }, { label: 'ωn [rad/s]', f: i => fmt(r.wn[i]) }, { label: 'fn [Hz]', f: i => fmt(r.fn[i]) },
      { label: 'Tn [s]', f: i => fmt(r.Tn[i]) }, { label: 'Td [s]', f: i => fmt(r.Td[i]) }, { label: 'ξ [%]', f: i => fmt(r.xi[i] * 100, 3) },
      ...(r.mpc ? [{ label: 'MPC', f: i => fmt(r.mpc[i], 3) }] : [])], rows,
    { onRow: (i) => gotoMode('wet', i) }), h('p.small.muted', {}, 'Click a mode to see its shape (Results → Mode shapes).'));
  };
  const runIt = act(async () => {
    modelGuard();
    const a = f.get(), b = fA.get();
    const r = await run('analysis/eig', { ...a, ...b, freq_kind: a.freq_kind === 'true' });
    toast(`Modal analysis finished: ${r.fn.length} modes`); showTable(r);
  });
  /* sweep */
  const fs = makeForm('eig_sweep', [
    { k: 'speeds', label: 'Mean wind speeds U0', type: 'expr', def: 'np.arange(10, 61, 10)', span: 2 },
    { k: 'n_track', label: 'Modes to track', type: 'expr', def: '10' },
    { k: 'include', label: 'Include', type: 'multi', opts: INCL_OPTS, def: ['aero', 'hydro'], span: 'full' },
    { k: 'omega_aero', label: 'Prepare aero matrices on ω', type: 'expr', def: 'np.arange(0.001, 2, 0.05)', span: 2 },
  ]);
  const pw = plotDiv('short'), px = plotDiv('short');
  const sweep = act(async () => {
    modelGuard();
    const r = await run('analysis/eig_sweep', fs.get());
    const n = r.wn[0].length;
    plot(pw, [...Array(n).keys()].map(k => ({ x: r.U, y: r.wn.map(row => row[k]), name: `mode ${k + 1}`, mode: 'lines+markers' })), { xaxis: { title: 'U0 [m/s]' }, yaxis: { title: 'ωn [rad/s]' }, title: { text: 'In-wind natural frequencies', font: { size: 13 } } });
    plot(px, [...Array(n).keys()].map(k => ({ x: r.U, y: r.xi.map(row => row[k] * 100), name: `mode ${k + 1}`, mode: 'lines+markers' })), { xaxis: { title: 'U0 [m/s]' }, yaxis: { title: 'ξ [%]' }, title: { text: 'Modal damping', font: { size: 13 } }, shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: C.alert, dash: 'dot' } }] });
  });
  el.append(h('div.cols', {},
    h('div', {}, card('Iterative modal analysis', 'Wet and in-wind complex modes of the combined system (Model.run_eig). Frequency-dependent added mass, damping and stiffness are iterated to convergence.',
      f.el, h('details.adv', {}, h('summary', {}, 'Solver settings'), fA.el),
      h('div.row', { style: { marginTop: '12px' } }, btn('Run modal analysis', runIt, 'primary'))),
    card({ title: 'In-wind modal sweep', cls: 'wind' }, 'Repeats run_eig for several mean wind speeds using the assigned wind state.', fs.el,
      h('div.row', { style: { marginTop: '12px' } }, btn('Run sweep', sweep, 'wind')))),
    h('div', {}, card('Modes of the combined system', null, out), card({}, null, pw, px))));
  bus.on('model', async (m) => { if (m && m.has_eig) { try { const o = await api('results/overview'); showTable(o.modal); } catch (e) { } } else clear(out).append(emptyBox('No modal results yet.')); });
}
function gotoMode(source, i) {
  showPage('results'); const r = PAGES._results; if (r) r.showMode(source, i);
}

/* ------------------------------------------------------------------ frequency domain */
function buildFreq(el) {
  const f = makeForm('freq', [
    { k: 'omega', label: 'Frequency axis ω [rad/s] for the FRF', type: 'expr', def: 'np.arange(0.001, 2, 0.005)', span: 'full' },
    { k: 'omega_hydro', label: 'Coarser ω for wave excitation', type: 'expr', def: 'np.arange(0.001, 2, 0.05)', span: 2 },
    { k: 'omega_aero', label: 'Coarser ω for wind excitation/aero matrices', type: 'expr', def: 'np.arange(0.001, 2, 0.05)', span: 2 },
    { k: 'prepare_aero', label: 'Prepare aero matrices on that ω first', type: 'check', def: true, span: 2 },
    { k: 'include_action', label: 'Excitation', type: 'multi', opts: [['hydro', 'Waves'], ['aero', 'Wind buffeting']], def: ['hydro', 'aero'], span: 'full' },
    { k: 'include_selfexcited', label: 'Motion-induced forces', type: 'multi', opts: INCL_OPTS, def: ['hydro', 'aero', 'drag_elements', 'drag_pontoons'], span: 'full' },
    { k: 'drag_iterations', label: 'Drag linearisation iterations', type: 'expr', def: '0', hint: '> 0 activates stochastic linearisation of quadratic drag' },
    { k: 'tol', label: 'Drag convergence tolerance', type: 'expr', def: '0.01' },
    aeroSectionField(),
  ]);
  const fA = makeForm('freq_adv', [
    { k: 'max_rel_error', label: 'Max relative error (θ integration)', type: 'expr', def: '0.01' },
    { k: 'interpolation_kind', label: 'Excitation interpolation', type: 'select', opts: ['linear', 'quadratic', 'cubic'], def: 'linear' },
    { k: 'theta_interpolation', label: 'Transfer function θ interpolation', type: 'select', opts: ['linear', 'quadratic'], def: 'linear' },
    { k: 'ensure_at_peaks', label: 'Add damped natural frequencies to ω', type: 'check', def: true },
    { k: 'reset_Cquad_lin', label: 'Reset linearised drag first', type: 'check', def: true },
    { k: 'kwargs', label: 'Other keyword arguments', type: 'expr', span: 'full', ph: 'dict(…)' },
  ]);
  syncSections(f);
  const out = h('div');
  el.append(h('div.cols', {}, card('Frequency-domain response', 'Multimodal response spectra from wave and wind excitation (Model.run_freqsim). Results are modal; convert them to physical quantities on the Results page.',
    f.el, h('details.adv', {}, h('summary', {}, 'Numerical settings'), fA.el),
    h('div.row', { style: { marginTop: '12px' } }, btn('Run response analysis', act(async () => {
      modelGuard();
      const r = await run('analysis/freqsim', { ...f.get(), ...fA.get() });
      clear(out).append(h('div.stats', {}, h('div.stat', {}, h('b', {}, r.n_freq), h('span', {}, 'frequencies')), h('div.stat', {}, h('b', {}, r.shape[0]), h('span', {}, 'modes')),
        h('div.stat', {}, h('b', {}, `${fmt(r.omega[0])}–${fmt(r.omega[1])}`), h('span', {}, 'ω range [rad/s]'))),
      btn('Show response statistics', () => { showPage('results'); PAGES._results && PAGES._results.tab('stats'); }, 'primary'));
      toast('Response analysis finished');
    }), 'primary'))),
  card('Status', null, out, h('p.small.muted', {}, 'Tip: running the modal analysis first adds the damped natural frequencies to the ω axis (ensure_at_peaks), which resolves sharp resonance peaks.'))));
  clear(out).append(emptyBox('No response computed in this session yet.'));
}

/* ------------------------------------------------------------------ static */
function buildStatic(el) {
  const f = makeForm('static', [
    { k: 'include_selfexcited', label: 'Include', type: 'multi', opts: [['aero', 'Static aerodynamic stiffness']], def: ['aero'], span: 'full' },
    { k: 'omega_static', label: 'ω for static aero stiffness', type: 'expr', def: '0.015', hint: 'Quasi-static limit of the prepared aero stiffness' },
    aeroSectionField(),
  ]);
  syncSections(f);
  const out = h('div');
  el.append(h('div.cols', {}, card({ title: 'Static wind response', cls: 'wind' }, 'Mean wind load from the static load coefficients and the resulting modal displacement (results.y_static). Implements Model.run_static robustly in the GUI.',
    f.el, h('div.row', { style: { marginTop: '12px' } }, btn('Run static analysis', act(async () => {
      modelGuard();
      const r = await run('analysis/static', f.get());
      clear(out).append(h('p', {}, `Static modal displacement computed for ${r.y_static.length} modes.`),
        btn('Show displacements', () => { showPage('results'); PAGES._results && PAGES._results.tab('static'); }, 'primary'),
        btn('Show deformed shape', () => gotoMode('static', 0), 'ghost'));
    }), 'wind'))), card('Result', null, out)));
}

/* ------------------------------------------------------------------ flutter */
function buildFlutter(el) {
  const f = makeForm('flutter', [
    { k: 'section', label: 'Aero section', type: 'select', opts: [] },
    { k: 'method', label: 'Routine', type: 'select', opts: [['cont', 'itflutter_cont'], ['naive', 'itflutter_cont_naive']], def: 'cont' },
    { k: 'ad_source', label: 'Aerodynamic derivatives', type: 'select', opts: [['section', 'From aero section'], ['flatplate', 'Flat plate (Theodorsen)'], ['expr', 'Expression']], def: 'section' },
    { k: 'ad_expr', label: 'AD expression (dict or ADs)', type: 'expr', span: 2 },
    { k: 'B', label: 'Deck width B (override)', type: 'expr' },
    { k: 'rho', label: 'Air density', type: 'expr', def: '1.225' },
    { k: 'V0', label: 'Start speed V0', type: 'expr', def: '10' }, { k: 'dV', label: 'Initial speed step', type: 'expr', def: '5' },
    { k: 'overshoot_factor', label: 'Overshoot factor', type: 'expr', def: '0.5' },
    { k: 'itmax_V', label: 'Max speed iterations', type: 'expr', def: '50' }, { k: 'itmax_f', label: 'Max frequency iterations', type: 'expr', def: '15' },
    { k: 'tol_V', label: 'Speed tolerance', type: 'expr', def: '1e-3' }, { k: 'tol_f', label: 'Frequency tolerance', type: 'expr', def: '1e-4' },
    { k: 'x_coord', label: 'Span coordinate', type: 'select', opts: [['arclength', 'Arc length'], ['x', 'x'], ['y', 'y'], ['z', 'z']], def: 'arclength' },
    { k: 'keep_all', label: 'Keep all steps', type: 'check', def: true }, { k: 'track_by_psi', label: 'Track modes by shape', type: 'check', def: true },
  ]);
  const upd = () => f.setOptions('section', (S.model && S.model.aero_sections) || []); upd(); bus.on('model', upd);
  const out = h('div'), pd = plotDiv('short'), pf = plotDiv('short');
  // Critical mode shape at a chosen speed step (default: the flutter speed), shown in the 3D viewer
  const shapeRow = (r) => {
    const steps = r.steps || [];
    if (!steps.length) return null;
    const sel = h('select', {}, ...steps.map((s, n) => h('option', { value: n },
      `V = ${fmt(s.V, 5)} m/s · ${s.mode != null ? 'mode ' + s.mode : 'real root'} · ${s.stable ? 'stable' : 'unstable'}`)));
    if (r.flutter_step != null) sel.value = r.flutter_step;
    return h('div.row', { style: { margin: '8px 0', gap: '8px', alignItems: 'center' } },
      h('label', {}, 'Critical mode shape at ', sel),
      btn('Show mode shape', () => gotoMode('flutter', +sel.value), 'small wind'));
  };
  el.append(h('div.cols', {}, card({ title: 'Multimodal flutter', cls: 'wind' }, 'Iterative flutter analysis on the dry modes with aerodynamic derivatives integrated along the selected section (wawi.wind.itflutter_cont).',
    f.el, h('div.row', { style: { marginTop: '12px' } }, btn('Find flutter speed', act(async () => {
      modelGuard();
      const r = await run('analysis/flutter', f.get());
      const found = r.V_flutter !== null && r.V_flutter !== undefined;
      const warn = !found || !r.converged || r.oscillating === false;
      clear(out).append(h('div.stats', {},
        h('div.stat', {}, h('b', {}, found ? fmt(r.V_flutter, 4) + ' m/s' : '–'), h('span', {}, found && r.V_unstable != null ? `Critical wind speed (unstable at ${fmt(r.V_unstable, 4)})` : 'Critical wind speed')),
        h('div.stat', {}, h('b', {}, r.oscillating === false ? '0 (real root)' : fmt(r.omega_flutter, 4) + ' rad/s'), h('span', {}, 'Flutter frequency')),
        h('div.stat', {}, h('b', {}, r.critical_mode != null ? r.critical_mode : '–'), h('span', {}, 'Critical mode')),
        h('div.stat', {}, h('b', {}, r.min_xi != null ? fmt(r.min_xi * 100, 3) + ' %' : '–'), h('span', {}, 'Min. damping just above'))),
      h('p.small', { style: { color: warn ? C.alert : C.ink2 } }, (r.converged ? '' : '⚠ ') + (r.message || '')),
      shapeRow(r),
      h('p.small.muted', {}, 'Full result stored as flutter_result.'));
      if (r.wd) {
        const n = r.wd[0].length; const ord = r.V.map((v, i) => i).sort((a, b) => r.V[a] - r.V[b]);
        const V = ord.map(i => r.V[i]);
        const val = (a, i, k) => (a[i] && a[i][k] !== null && a[i][k] !== undefined) ? a[i][k] : null;
        const rm = r.root_mode || [];
        const crit = r.critical_col;
        // One root per mode (conjugate partners dropped), ranked by lowest damping; the critical mode first
        const minXi = k => Math.min(...ord.map(i => { const x = val(r.xi, i, k); return x === null ? Infinity : x; }));
        const modes = [...Array(n).keys()].filter(k => rm[k] != null);
        let shown = modes.sort((a, b) => minXi(a) - minXi(b)).slice(0, 30);
        if (crit != null && rm[crit] != null) shown = [crit, ...shown.filter(k => k !== crit)].slice(0, 30);
        const tr = (k, y) => ({ x: V, y, name: `mode ${rm[k]}` + (k === crit ? ' (critical)' : ''), mode: 'lines+markers',
          marker: { size: k === crit ? 6 : 4 }, line: { width: k === crit ? 3.5 : 1.2, color: k === crit ? C.alert : undefined },
          opacity: k === crit ? 1 : 0.55 });
        const vline = found ? [{ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: r.V_flutter, x1: r.V_flutter, line: { color: C.alert, dash: 'dash', width: 1 } }] : [];
        plot(pd, shown.map(k => tr(k, ord.map(i => { const x = val(r.xi, i, k); return x === null ? null : x * 100; }))),
          { xaxis: { title: 'V [m/s]' }, yaxis: { title: 'ξ [%]' }, title: { text: 'Damping vs wind speed (lowest-damped modes)', font: { size: 13 } },
            shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: C.alert, dash: 'dot' } }, ...vline] });
        plot(pf, shown.map(k => tr(k, ord.map(i => val(r.wd, i, k)))),
          { xaxis: { title: 'V [m/s]' }, yaxis: { title: 'ωd [rad/s]' }, title: { text: 'Frequency vs wind speed', font: { size: 13 } }, shapes: vline });
      }
    }), 'wind'))), card('Result', null, out, pd, pf)));
}

/* ------------------------------------------------------------------ screening */
function buildScreening(el) {
  const f = makeForm('screen', [
    { k: 'seastate', label: 'Base sea state', type: 'select', opts: [['editor', 'Edited sea state (Environment)'], ['model', 'Assigned to model']], def: 'editor', span: 2 },
    { type: 'heading', label: 'Parameter variations (leave empty to keep the base value)' },
    { k: 'Hs', label: 'Hs', type: 'expr', def: 'np.arange(0.5, 2.6, 0.5)' }, { k: 'Tp', label: 'Tp', type: 'expr', def: 'np.arange(4, 12, 2)' },
    { k: 'gamma', label: 'γ', type: 'expr' }, { k: 'theta0', label: 'θ0 [°]', type: 'expr' }, { k: 's', label: 's', type: 'expr' },
    { k: 'U', label: 'Current U', type: 'expr' }, { k: 'thetaU', label: 'Current θU [°]', type: 'expr' },
    { k: 'independent', label: 'All combinations (grid)', type: 'check', def: true, hint: 'Untick to pair the arrays element-wise' },
    { k: 'json_path', label: 'Or a ScreeningCase JSON file', type: 'path', ext: ['.json'], span: 'full' },
    { type: 'heading', label: 'Response' },
    { k: 'omega', label: 'ω', type: 'expr', def: 'np.arange(0.001, 2, 0.01)', span: 2 },
    { k: 'key', label: 'Mode-shape key', type: 'select', opts: ['hydro', 'full'], def: 'hydro' },
    { k: 'quantity', label: 'Quantity', type: 'select', opts: [['std', 'Standard deviation'], ['expmax', 'Expected maximum']], def: 'std' },
    { k: 'T', label: 'Duration T [s] (exp. max)', type: 'expr', def: '3600' },
    { k: 'include_selfexcited', label: 'Motion-induced forces', type: 'multi', opts: INCL_OPTS, def: ['hydro'], span: 'full' },
  ]);
  const updKeys = async () => { if (S.model) { try { const o = await api('results/overview'); f.setOptions('key', Object.keys(o.keys)); } catch (e) { } } };
  bus.on('model', updKeys); updKeys();
  const out = h('div'), pl = plotDiv();
  const dofSel = h('select', {}, DOFS.map((d, i) => h('option', { value: i }, d)));
  let res = null;
  const draw = () => {
    if (!res) return;
    const d = +dofSel.value; const pars = res.parameters;
    const mx = res.values.map(row => { const v = row.filter((_, j) => j % 6 === d); return Math.max(...v.filter(x => x !== null)); });
    if (pars.length === 2 && f.get().independent) {
      const xs = [...new Set(res.combos.map(c => c[pars[0]]))], ys = [...new Set(res.combos.map(c => c[pars[1]]))];
      const z = ys.map(y => xs.map(x => { const i = res.combos.findIndex(c => c[pars[0]] === x && c[pars[1]] === y); return i >= 0 ? mx[i] : null; }));
      plot(pl, [{ type: 'heatmap', x: xs, y: ys, z, colorscale: [[0, '#f4f9fa'], [0.5, '#5aa9b8'], [1, '#0f3d4a']], colorbar: { title: res.quantity } }], { xaxis: { title: pars[0] }, yaxis: { title: pars[1] }, title: { text: `Max ${res.quantity} of DOF ${DOFS[d]} along the structure`, font: { size: 13 } } });
    } else {
      plot(pl, [{ x: res.combos.map((c, i) => i + 1), y: mx, mode: 'lines+markers', text: res.combos.map(c => JSON.stringify(c)), line: { color: C.sea } }], { xaxis: { title: 'Case' }, yaxis: { title: res.quantity }, title: { text: `Max ${res.quantity} of DOF ${DOFS[d]}`, font: { size: 13 } } });
    }
    clear(out).append(table([{ label: 'Case', f: (r, i) => i + 1 }, ...pars.map(p => ({ label: p, f: r => fmt(r[p]) })), { label: `max ${DOFS[d]}`, f: (r, i) => fmt(mx[i]) }], res.combos),
      btn('Download CSV', () => downloadText('screening.csv', toCSV(['case', ...pars, ...res.values[0].map((_, j) => `row${j}`)], res.combos.map((c, i) => [i + 1, ...pars.map(p => c[p]), ...res.values[i]]))), 'small ghost'));
  };
  dofSel.onchange = draw;
  el.append(h('div.cols', {}, card({ title: 'Sea-state screening', cls: 'sea' }, 'Runs the response analysis for every combination of sea-state parameters (wawi.model.ScreeningCase). Results are stored as screening_result.',
    f.el, h('div.row', { style: { marginTop: '12px' } }, btn('Run screening', act(async () => {
      modelGuard();
      const a = f.get(); const parvar = {};
      ['Hs', 'Tp', 'gamma', 'theta0', 's', 'U', 'thetaU'].forEach(k => { if (a[k].trim()) parvar[k] = a[k]; });
      res = await run('analysis/screening', { ...a, parvar });
      draw();
    }), 'sea'))), card('Result', null, h('div.row', {}, 'DOF', dofSel), pl, out)));
}

/* ------------------------------------------------------------------ load simulation */
function buildLoads(el) {
  const f = makeForm('loads', [
    { k: 'source', label: 'Excitation', type: 'select', opts: [['wave', 'Wave forces on pontoons'], ['wind', 'Wind buffeting (modal)']], def: 'wave', span: 2 },
    { k: 'omega', label: 'ω', type: 'expr', def: 'np.arange(0.05, 2.0, 0.05)', span: 2 },
    { k: 'fs', label: 'Sampling frequency [Hz]', type: 'expr', def: '10.0' }, { k: 'tmax', label: 'Duration [s]', type: 'expr', def: '300' },
    { k: 'transform_by_phi', label: 'Modal forces (transform by φ)', type: 'check', def: false },
    { k: 'to_global', label: 'Rotate pontoon forces to global axes', type: 'check', def: false },
    { k: 'seed', label: 'Random seed', type: 'expr' },
    { k: 'max_rel_error', label: 'Max relative error', type: 'expr', def: '0.01' },
    aeroSectionField(),
  ]);
  syncSections(f);
  const pt = plotDiv('tall'), ps = plotDiv('short'); const sel = h('select', { multiple: true, size: 8, style: { minWidth: '200px' } });
  let res = null;
  const draw = () => {
    if (!res) return;
    const idx = [...sel.selectedOptions].map(o => +o.value);
    plot(pt, idx.map(i => ({ x: res.t, y: res.q[i], name: res.labels[i], mode: 'lines', line: { width: 1 } })), { xaxis: { title: 't [s]' }, yaxis: { title: 'Force / moment' }, title: { text: 'Simulated time histories', font: { size: 13 } } });
    plot(ps, idx.map(i => ({ x: res.omega, y: res.Sqq_diag[i], name: res.labels[i], mode: 'lines' })), { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'Auto-spectral density' }, title: { text: 'Excitation spectra', font: { size: 13 } } });
  };
  sel.onchange = draw;
  el.append(h('div.cols', {}, card({ title: 'Load time histories', cls: 'sea' }, 'Evaluates the excitation cross-spectral density (evaluate_waveaction / evaluate_windaction) and simulates correlated time series with wawi.time_domain.simulate_mdof. Stored as q_sim, t_q, Sqq, omega_Sqq.',
    f.el, h('div.row', { style: { marginTop: '12px' } }, btn('Simulate loads', act(async () => {
      modelGuard();
      res = await run('analysis/load_simulation', f.get());
      clear(sel).append(...res.labels.map((l, i) => h('option', { value: i, selected: i < 3 }, `${l} (σ = ${fmt(res.std[i], 3)})`)));
      draw();
    }), 'sea'))),
  card('Result', 'Select components (Ctrl/Cmd-click for several).', h('div.row', { style: { alignItems: 'flex-start' } }, sel, h('div', { style: { flex: 1, minWidth: 0 } }, pt)), ps)));
}

/* ------------------------------------------------------------------ preparation */
function buildPrep(el) {
  const fa = makeForm('prep_aero', [{ k: 'omega', label: 'ω for aero matrices', type: 'expr', def: 'np.arange(0.001, 2, 0.05)', span: 2 }, aeroSectionField()]);
  const fw = makeForm('prep_wave', [
    { k: 'omega', label: 'ω for wave excitation', type: 'expr', def: 'np.arange(0.001, 2, 0.05)', span: 2 },
    { k: 'interpolation_kind', label: 'Interpolation', type: 'select', opts: ['linear', 'quadratic', 'cubic'], def: 'linear' },
    { k: 'theta_interpolation', label: 'θ interpolation', type: 'select', opts: ['quadratic', 'linear'], def: 'quadratic' },
    { k: 'max_rel_error', label: 'Max relative error', type: 'expr', def: '0.01' }]);
  const fd = makeForm('prep_wind', [
    { k: 'omega', label: 'ω for wind excitation', type: 'expr', def: 'np.arange(0.001, 2, 0.05)', span: 2 },
    { k: 'include', label: 'Include', type: 'multi', opts: [['dynamic', 'Dynamic (buffeting)'], ['static', 'Static load']], def: ['dynamic'], span: 'full' },
    { k: 'interpolation_kind', label: 'Interpolation', type: 'select', opts: ['linear', 'quadratic', 'cubic'], def: 'linear' }, aeroSectionField()]);
  syncSections(fa); syncSections(fd);
  el.append(h('div.cols.even', {},
    card({ title: 'Aerodynamic stiffness and damping', cls: 'wind' }, 'Builds interpolants of the motion-induced aerodynamic matrices (Aero.prepare_aero_matrices). Reused by modal and response analyses until the wind state changes.', fa.el,
      h('div.row', { style: { marginTop: '12px' } }, btn('Prepare aero matrices', act(async () => { modelGuard(); await run('analysis/prepare_aero', fa.get()); toast('Aero matrices prepared'); }), 'wind'))),
    h('div', {},
      card({ title: 'Precompute wave excitation', cls: 'sea' }, 'Stores the wave excitation spectrum so several response analyses (e.g. with/without drag) reuse it (Model.precompute_waveaction).', fw.el,
        h('div.row', { style: { marginTop: '12px' } }, btn('Precompute wave action', act(async () => { modelGuard(); await run('analysis/precompute', { what: 'wave', ...fw.get() }); toast('Wave action stored'); }), 'sea'))),
      card({ title: 'Precompute wind excitation', cls: 'wind' }, 'Model.precompute_windaction.', fd.el,
        h('div.row', { style: { marginTop: '12px' } }, btn('Precompute wind action', act(async () => { modelGuard(); await run('analysis/precompute', { what: 'wind', ...fd.get() }); toast('Wind action stored'); }), 'wind'))))));
}
