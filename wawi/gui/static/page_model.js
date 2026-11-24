/* WAWI GUI – Model page: open / import / build / overview / edit / pontoon types / aero / 3D */
'use strict';

const PAGES = {};

PAGES.model = (root) => {
  subtabs(root, [
    { id: 'open', label: 'Open & save', build: buildOpen },
    { id: 'build', label: 'Build from objects', build: buildBuilder },
    { id: 'overview', label: 'Overview', build: buildOverview, onShow: (el) => el._refresh && el._refresh() },
    { id: 'edit', label: 'Edit', build: buildEdit, onShow: (el) => el._refresh && el._refresh() },
    { id: 'ptypes', label: 'Pontoon types', build: buildPtypes, onShow: (el) => el._refresh && el._refresh() },
    { id: 'aero', label: 'Aerodynamics', build: buildAero, onShow: (el) => el._refresh && el._refresh() },
    { id: 'view', label: '3D view', build: buildView, onShow: (el) => el._refresh && el._refresh() },
  ], { key: 'model' });
};

function needModel(el, what = 'this') {
  if (S.model) return false;
  clear(el).append(emptyBox(`Load, import or build a model first to see ${what}.`));
  return true;
}

/* ------------------------------------------------------------------ open & save */
function buildOpen(el) {
  const fLoad = makeForm('load', [
    { k: 'path', label: 'Model file (.wwi) or model folder', type: 'path', ext: ['.wwi'], span: 'full', ph: '/path/to/model.wwi' },
    { k: 'method', label: 'Loader', type: 'select', opts: [['auto', 'Automatic'], ['wwi', 'Model.from_wwi'], ['load_model', 'wawi.io.load_model']], def: 'auto' },
    { k: 'sort_nodes_by_x', label: 'Sort nodes by x (folders)', type: 'check', def: true },
    { k: 'save_if_nonexistent', label: 'Save model.wwi in folder', type: 'check', def: false },
  ]);
  const loadCard = card({ title: 'Open a saved model' }, 'Load a pickled WAWI model (.wwi, made with Model.to_wwi) or a folder that wawi.io.load_model understands.',
    fLoad.el,
    h('div.row', { style: { marginTop: '12px' } },
      btn('Open model', act(async () => { const r = await run('model/load', fLoad.get()); afterLoad(r); }), 'primary'),
      btn('Upload .wwi…', act(async () => { const p = await uploadPicker({ accept: '.wwi' }); if (p) { fLoad.set({ path: p }); const r = await run('model/load', fLoad.get()); afterLoad(r); } }), 'ghost')));

  const fImp = makeForm('import', [
    { k: 'folder', label: 'Model folder', type: 'path', pathKind: 'dir', span: 'full', hint: 'Folder with modal.json, pontoon.json, element.json, … (see the wawi folder-import example).' },
    { k: 'n_modes', label: 'Number of dry modes', type: 'expr', ph: 'all' },
    { k: 'xi0', label: 'Structural damping ξ0', type: 'expr', ph: 'from modal.json' },
    { k: 'sort_nodes_by_x', label: 'Sort nodes by x', type: 'check', def: true },
    { k: 'interpolation_kind', label: 'Hydro interpolation', type: 'select', opts: ['linear', 'quadratic', 'cubic', 'nearest'], def: 'linear' },
  ]);
  const fFiles = makeForm('import_files', [
    { k: 'modal', label: 'Dry modes', def: 'modal.json' },
    { k: 'pontoon', label: 'Pontoons', def: 'pontoon.json' },
    { k: 'pontoon_types', label: 'Pontoon type settings', def: 'pontoon_types.json' },
    { k: 'pontoon_stl', label: 'Pontoon geometry (STL)', def: 'pontoon.stl' },
    { k: 'eldef', label: 'Elements', def: 'element.json' },
    { k: 'orientations', label: 'Element orientations', def: 'orientations.json' },
    { k: 'aero_sections', label: 'Aero sections', def: 'aero_sections.json' },
    { k: 'drag_elements', label: 'Drag elements', def: 'drag_elements.json' },
  ]);
  const impCard = card({ title: 'Import a model folder', cls: 'sea' }, 'Builds the model with wawi.io.import_folder from JSON/NPZ input files. Clear a file name to skip that part.',
    fImp.el,
    h('details.adv', {}, h('summary', {}, 'Input file names'), fFiles.el),
    h('div.row', { style: { marginTop: '12px' } },
      btn('Import folder', act(async () => { const r = await run('model/import_folder', { ...fImp.get(), ...fFiles.get() }); afterLoad(r); }), 'primary'),
      btn('Upload folder…', act(async () => { const p = await uploadPicker({ folder: true }); if (p) fImp.set({ folder: p }); }), 'ghost'),
      btn('Convert folder to .wwi', act(async () => {
        const f = fImp.get(); if (!f.folder) throw new ApiError('Choose a folder first');
        const r = await run('model/convert_to_wwi', { model_path: f.folder }); toast('Wrote ' + r.path);
      }), 'ghost')));

  const fSave = makeForm('save', [{ k: 'path', label: 'Save as', type: 'path', pathKind: 'save', ext: ['.wwi'], span: 'full', ph: 'model.wwi' }]);
  const saveCard = card('Save the active model', 'Writes the model including assigned states and results with Model.to_wwi (dill pickle).',
    fSave.el, h('div.row', { style: { marginTop: '12px' } },
      btn('Save', act(async () => { const r = await run('model/save', fSave.get()); toast('Saved to ' + r.path); })),
      btn('Download .wwi', act(async () => { const r = await run('model/download', { name: 'model.wwi' }); triggerDownload(r.download); }), 'ghost')));

  const varName = h('input', { type: 'text', value: 'model_copy', style: { maxWidth: '180px' } });
  const varSel = h('select', { style: { maxWidth: '200px' } });
  const fillVars = async () => {
    try {
      const vs = await api('vars/list');
      clear(varSel).append(...vs.filter(v => v.type.endsWith('Model')).map(v => h('option', { value: v.name }, v.name)));
    } catch (e) { /* ignore */ }
  };
  const varsCard = card('Model variables', 'Keep copies of a model to compare variants (for example with and without drag), and switch which one the GUI works on.',
    h('div.row', {}, 'Copy active model to', varName, btn('Copy', act(async () => { await run('model/copy_to_var', { name: varName.value }); toast(`Copied to ${varName.value}`); fillVars(); }), 'small')),
    h('div.row', { style: { marginTop: '8px' } }, 'Make active', varSel, btn('Activate', act(async () => { await run('model/activate_var', { name: varSel.value }); toast('Active model changed'); }), 'small'),
      btn('Refresh list', fillVars, 'small ghost')),
    h('hr.thin'),
    btn('Close active model', act(async () => { await run('model/close'); }), 'danger small'));
  fillVars(); bus.on('model', fillVars);

  const welcome = h('div.welcome', {}, h('h2', {}, 'Predict the response of a structure to wind and waves'),
    h('p.muted', {}, 'Work through the steps in the side bar. Every long computation runs in the background; follow it in the jobs bar at the bottom.'),
    h('ol', {}, h('li', {}, 'Model: open a .wwi file, import a model folder, or build one from pontoon types.'),
      h('li', {}, 'Environment: define and assign a sea state and/or a wind state.'),
      h('li', {}, 'Analysis: run the iterative modal analysis, the frequency-domain response, flutter or screening.'),
      h('li', {}, 'Results: plot modes, response spectra and statistics along the structure.')),
    h('p.small.muted', {}, 'The API explorer and the console give access to every function and class in wawi.'));
  el.append(h('div.cols.even', {}, h('div', {}, card({}, null, welcome), loadCard, saveCard), h('div', {}, impCard, varsCard)));
}

function afterLoad(r) {
  toast(`Model ready: ${r.n_pontoons || 0} pontoons, ${r.n_modes ?? '–'} modes${r.aero_sections && r.aero_sections.length ? ', aero: ' + r.aero_sections.join(', ') : ''}`);
  refreshState();
}

/* ------------------------------------------------------------------ builder */
function buildBuilder(el) {
  const types = JSON.parse(localStorage.getItem('wawi.build.types') || '[]');
  const pontoons = JSON.parse(localStorage.getItem('wawi.build.pontoons') || '[]');
  const typeBox = h('div'); const pontBox = h('div');
  const save = () => { localStorage.setItem('wawi.build.types', JSON.stringify(types)); localStorage.setItem('wawi.build.pontoons', JSON.stringify(pontoons)); };

  function renderTypes() {
    clear(typeBox);
    if (!types.length) typeBox.append(emptyBox('No pontoon types yet. Add one from hydrodynamic analysis results.'));
    types.forEach((t, i) => {
      const f = makeForm('bt' + i, [
        { k: 'label', label: 'Label', def: t.label },
        { k: 'source', label: 'Source', type: 'select', opts: [['npz', 'NPZ file'], ['wadam', 'WADAM .lis'], ['wamit', 'WAMIT (prefix)'], ['expr', 'Python expressions']], def: t.source },
        { k: 'path', label: 'File (or WAMIT path prefix)', type: 'path', span: 2, def: t.path },
        { k: 'Cd', label: 'Drag coefficients Cd', type: 'expr', def: t.Cd, ph: '[1.0, 1.0, 2.0]' },
        { k: 'area', label: 'Drag areas', type: 'expr', def: t.area, ph: '[100, 200, 200]' },
        { k: 'stl_path', label: 'STL geometry (optional)', type: 'path', ext: ['.stl'], span: 2, def: t.stl_path },
        { k: 'interpolation_kind', label: 'Interpolation', type: 'select', opts: ['linear', 'quadratic', 'cubic'], def: t.interpolation_kind || 'linear' },
      ], { persist: false });
      const adv = makeForm('bta' + i, [
        { type: 'heading', label: 'NPZ array names' },
        ...['A', 'B', 'omega', 'Q', 'theta', 'omegaQ', 'Kh', 'M0'].map(k => ({ k: 'key_' + k, label: k, def: t['key_' + k] || '', ph: { A: 'M', B: 'C', omega: 'omega', Q: 'Q', theta: 'theta_Q', omegaQ: 'omega_Q', Kh: 'K', M0: 'M0' }[k] })),
        { type: 'heading', label: 'WAMIT scaling' },
        { k: 'rho', label: 'rho', def: t.rho || '1020' }, { k: 'L', label: 'L', def: t.L || '1.0' }, { k: 'g', label: 'g', def: t.g || '9.80665' },
        { k: 'theta_deg', label: 'WAMIT headings in degrees', type: 'check', def: t.theta_deg !== false },
        { type: 'heading', label: 'Expressions (source = Python expressions)' },
        ...['A', 'M0', 'B', 'Kh', 'omega', 'Q', 'theta', 'omegaQ'].map(k => ({ k, label: k, type: 'expr', def: t[k] || '' })),
      ], { persist: false });
      const upd = () => { Object.assign(types[i], f.get(), adv.get()); save(); };
      f.el.addEventListener('change', upd); adv.el.addEventListener('change', upd);
      typeBox.append(h('div', { style: { border: '1px solid var(--rule-2)', borderRadius: '6px', padding: '10px', marginBottom: '10px' } },
        f.el, h('details.adv', {}, h('summary', {}, 'Source options'), adv.el),
        h('div.row.end', {}, btn('Remove type', () => { types.splice(i, 1); save(); renderTypes(); renderPontoons(); }, 'small danger'))));
    });
  }
  function renderPontoons() {
    clear(pontBox);
    if (!pontoons.length) { pontBox.append(emptyBox('No pontoons yet. Add rows, paste a table or load a pontoon.json file.')); return; }
    const typeLabels = types.map(t => t.label);
    const cols = ['label', 'node', 'x', 'y', 'z', 'rotation', 'type'];
    const tb = h('tbody');
    pontoons.forEach((p, i) => {
      tb.append(h('tr', {}, cols.map(c => h('td.l', {}, c === 'type'
        ? (() => { const s = h('select', {}, typeLabels.map(t => h('option', { value: t }, t))); s.value = p.type || typeLabels[0] || ''; s.onchange = () => { p.type = s.value; save(); }; return s; })()
        : (() => { const inp = h('input', { type: 'text', value: p[c] ?? '', style: { width: c === 'label' ? '110px' : '80px' } }); inp.onchange = () => { p[c] = inp.value; save(); }; return inp; })())),
        h('td', {}, btn('×', () => { pontoons.splice(i, 1); save(); renderPontoons(); }, 'small ghost'))));
    });
    pontBox.append(h('div.tablewrap', {}, h('table.grid', {}, h('thead', {}, h('tr', {}, [...cols.map(c => h('th.l', {}, c === 'rotation' ? 'rotation [deg]' : c)), h('th')])), tb)));
  }
  const pasteArea = h('textarea.code', { rows: 4, placeholder: 'label, node, x, y, z, rotation_deg, type   (one pontoon per line)' });
  const fModal = makeForm('build_modal', [
    { k: 'source', label: 'Dry modes', type: 'select', opts: [['none', 'None (wave excitation / load simulation only)'], ['json', 'modal.json file'], ['npz', 'NPZ file (phi_<key>, m, k / omega_n, xi0)'], ['expr', 'Python expressions']], def: 'none', span: 'full' },
    { k: 'path', label: 'File', type: 'path', span: 'full', ext: ['.json', '.npz'] },
    { k: 'phi', label: 'phi (dict of arrays)', type: 'expr', ph: "dict(hydro=phi_h)", span: 2 },
    { k: 'm', label: 'm', type: 'expr' }, { k: 'k', label: 'k', type: 'expr' }, { k: 'omega_n', label: 'omega_n', type: 'expr' }, { k: 'xi0', label: 'xi0', type: 'expr', def: '0.0' },
  ]);
  const fEnv = makeForm('build_env', [
    { k: 'n_modes', label: 'Number of modes', type: 'expr' },
    { k: 'g', label: 'g', type: 'expr', def: '9.80655' }, { k: 'rho', label: 'Water density', type: 'expr', def: '1025' },
    { k: 'depth', label: 'Depth', type: 'expr', def: 'np.inf' }, { k: 'waterlevel', label: 'Water level', type: 'expr', def: '0' },
  ]);
  renderTypes(); renderPontoons();
  el.append(h('div.cols.even', {},
    h('div', {},
      card({ title: 'Pontoon types', cls: 'sea' }, 'Frequency-dependent added mass, radiation damping, restoring stiffness and wave excitation transfer functions (PontoonType.from_numeric / from_wadam).',
        typeBox, btn('Add pontoon type', () => { types.push({ label: `type${types.length + 1}`, source: 'npz' }); save(); renderTypes(); renderPontoons(); })),
      card('Dry modes and environment', 'Optional dry modal basis. Without it the model still evaluates wave excitation and simulates load time histories.', fModal.el, h('hr.thin'), fEnv.el)),
    h('div', {},
      card('Pontoons', 'Node label and coordinates of each pontoon; rotation about z in degrees.',
        pontBox,
        h('div.row', { style: { marginTop: '10px' } },
          btn('Add row', () => { const n = pontoons.length + 1; pontoons.push({ label: `P${n}`, node: String(n), x: '0', y: '0', z: '0', rotation: '0', type: types[0] ? types[0].label : '' }); save(); renderPontoons(); }, 'small'),
          btn('Load pontoon.json…', act(async () => {
            const p = await browse({ ext: ['.json'] }); if (!p) return;
            const data = await api('fs/read_json', { path: p });
            pontoons.length = 0;
            for (const [lab, v] of Object.entries(data)) pontoons.push({ label: lab, node: String(v.node), x: String(v.coordinates[0]), y: String(v.coordinates[1]), z: String(v.coordinates[2]), rotation: String(v.rotation), type: v.pontoon_type });
            save(); renderPontoons();
          }), 'small ghost'),
          btn('Clear', () => { pontoons.length = 0; save(); renderPontoons(); }, 'small ghost danger')),
        h('details.adv', {}, h('summary', {}, 'Paste a table'), pasteArea, btn('Add pasted rows', () => {
          pasteArea.value.split('\n').map(l => l.trim()).filter(Boolean).forEach(l => {
            const c = l.split(/[,;\t]\s*|\s+/); pontoons.push({ label: c[0], node: c[1], x: c[2], y: c[3], z: c[4], rotation: c[5] || '0', type: c[6] || (types[0] && types[0].label) });
          });
          pasteArea.value = ''; save(); renderPontoons();
        }, 'small'))),
      card({}, null, h('div.row', {}, btn('Build model', act(async () => {
        const fe = fEnv.get();
        const r = await run('model/build', { ptypes: types, pontoons, modal: fModal.get(), n_modes: fe.n_modes, environment: fe });
        afterLoad(r);
      }), 'primary'), h('span.small.muted', {}, 'Creates Model.from_nodes_and_types (or Model(hydro=Hydro(…)) without dry modes).'))))));
}

/* ------------------------------------------------------------------ overview */
function buildOverview(el) {
  el._refresh = async () => {
    if (needModel(el, 'the model overview')) return;
    clear(el).append(h('div.muted', {}, 'Loading…'));
    let d;
    try { d = await api('model/info'); } catch (e) { showError(e); return; }
    S.info = d;
    const s = d.summary;
    const kv = (o) => h('dl.kv', {}, Object.entries(o).flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v === null || v === undefined ? '–' : (typeof v === 'object' ? JSON.stringify(v) : String(v)))]));
    const left = h('div'), right = h('div');
    left.append(card('Model', null, kv({
      'File': s.path, 'Dry modes used / total': `${s.n_modes ?? '–'} / ${s.n_modes_total ?? '–'}`, 'Mode shape keys (phi)': (s.phi_keys || []).join(', '),
      'Pontoons': s.n_pontoons, 'Aero sections': (s.aero_sections || []).join(', ') || 'none', 'Drag elements': s.n_drag_elements,
      'FE elements': d.eldef ? `${d.eldef.n_nodes} nodes, ${d.eldef.n_elements} elements` : 'none',
      'Mode shapes local to pontoons': d.flags.local, 'Phases lead': d.flags.phases_lead,
      'Aero matrices prepared': s.aero_prepared, 'Wave action precomputed': s.wave_precomputed, 'Wind action precomputed': s.wind_precomputed,
    })));
    if (d.modal && !d.modal.error) {
      const md = d.modal;
      left.append(card('Dry modes', `Natural frequencies of the dry (in-vacuum) modes used as basis. Mode shape matrices: ${Object.entries(md.phi).map(([k, v]) => `${k} ${v.join('×')}`).join(', ')}.`,
        table([{ label: 'Mode', f: (r, i) => md.mode_ix[i] + 1 }, { label: 'f [Hz]', f: (r, i) => fmt(md.fn[i]) }, { label: 'T [s]', f: (r, i) => fmt(md.Tn[i]) },
          { label: 'ξ0 [%]', f: (r, i) => fmt((md.xi0[i] ?? md.xi0[0]) * 100, 3) }, { label: 'm', f: (r, i) => fmt(md.m[i]) }, { label: 'k', f: (r, i) => fmt(md.k[i]) }],
          md.fn.map(() => ({})))));
    }
    if (d.hydro) {
      const hy = d.hydro;
      right.append(card({ title: 'Hydrodynamics', cls: 'sea' }, `Environment: g = ${fmt(hy.environment.g)}, ρ = ${fmt(hy.environment.rho)}, depth = ${hy.environment.depth ?? '∞'}, water level = ${fmt(hy.environment.waterlevel)}.`,
        table([{ label: 'Pontoon', k: 'label', l: true, f: r => r.label }, { label: 'Node', f: r => r.node }, { label: 'x', k: 'x' }, { label: 'y', k: 'y' }, { label: 'z', k: 'z' },
          { label: 'Rot. [°]', k: 'rotation' }, { label: 'Type', l: true, f: r => r.type }, { label: 'Cd', f: r => r.Cd.map(v => fmt(v, 3)).join(' ') }], hy.pontoons),
        h('div', { style: { marginTop: '10px' } }, hy.types.map(t => h('div.small', {}, h('b', {}, t.label), ` — ω ${t.omega ? `${fmt(t.omega[0])}…${fmt(t.omega[1])} rad/s (${t.omega[2]} values)` : 'n/a'}; `,
          ['M', 'C', 'K', 'Q'].filter(q => t['has_' + q]).join(', ') + ' defined', t.theta_deg ? `; ${t.theta_deg.length} headings` : '')))));
    }
    if (d.aero) {
      right.append(card({ title: 'Aerodynamics', cls: 'wind' }, d.aero.windstate ? 'Wind state assigned.' : 'No wind state assigned yet.',
        table([{ label: 'Section', l: true, f: r => r.name }, { label: 'B', k: 'B' }, { label: 'D', k: 'D' }, { label: 'Cd', k: 'Cd' }, { label: "Cd'", k: 'dCd' },
          { label: 'Cl', k: 'Cl' }, { label: "Cl'", k: 'dCl' }, { label: 'Cm', k: 'Cm' }, { label: "Cm'", k: 'dCm' }, { label: 'ADs', l: true, f: r => r.ad_type }, { label: 'Elements', k: 'n_elements' }], d.aero.sections)));
    }
    if (d.drag_elements && d.drag_elements.length) {
      right.append(card('Drag elements', null, table([{ label: 'Group', l: true, f: r => r.group }, { label: 'Count', k: 'n' }, { label: 'Cd', f: r => (r.Cd || []).map(v => fmt(v, 3)).join(' ') },
        { label: 'D', f: r => (r.D || []).map(v => fmt(v, 3)).join(' ') }, { label: 'ρ', k: 'rho' }, { label: 'Type', l: true, f: r => r.eltype }], d.drag_elements)));
    }
    clear(el).append(h('div.cols.even', {}, left, right));
  };
  bus.on('model', () => { if (el.classList.contains('active')) el._refresh(); });
}

/* ------------------------------------------------------------------ edit */
function buildEdit(el) {
  el._refresh = async () => {
    if (needModel(el, 'model settings')) return;
    let d; try { d = await api('model/info'); } catch (e) { showError(e); return; }
    clear(el);
    const left = h('div'), right = h('div');
    const fGen = makeForm('edit_gen', [
      { k: 'n_modes', label: 'Number of dry modes', type: 'expr', def: d.modal ? String(d.modal.n_modes) : '' , hint: d.modal ? `of ${d.modal.n_total} available` : '' },
      { k: 'xi0', label: 'Structural damping ξ0', type: 'expr', def: d.modal ? d.modal.xi0_raw : '', hint: 'scalar or array per mode' },
      { k: 'm_min', label: 'Minimum modal mass m_min', type: 'expr', def: d.modal ? String(d.modal.m_min ?? 0) : '' },
      { k: 'x0', label: 'Wave phase reference x0', type: 'expr', def: d.hydro && d.hydro.x0 ? JSON.stringify(d.hydro.x0) : '' },
      { k: 'phases_lead', label: 'Transfer-function phases lead', type: 'check', def: d.flags.phases_lead },
    ], { persist: false });
    left.append(card('General', 'Changing the number of modes clears prepared aerodynamic matrices and precomputed actions.', fGen.el,
      h('div.row', { style: { marginTop: '10px' } }, btn('Apply', act(async () => { const r = await run('model/set', fGen.get()); toast(r.changed.join(', ') || 'No change'); el._refresh(); }), 'primary'))));
    if (d.hydro) {
      const e = d.hydro.environment;
      const fEnv = makeForm('edit_env', [
        { k: 'g', label: 'g', type: 'expr', def: String(e.g) }, { k: 'rho', label: 'Water density ρ', type: 'expr', def: String(e.rho) },
        { k: 'depth', label: 'Depth', type: 'expr', def: e.depth === null ? 'np.inf' : String(e.depth) }, { k: 'waterlevel', label: 'Water level', type: 'expr', def: String(e.waterlevel) },
      ], { persist: false });
      left.append(card({ title: 'Environment', cls: 'sea' }, null, fEnv.el, h('div.row', { style: { marginTop: '10px' } }, btn('Apply', act(async () => { await run('model/set', { environment: fEnv.get() }); toast('Environment updated'); }), 'primary'))));
      const ptForms = d.hydro.types.map(t => ({ t, f: makeForm('edit_pt_' + t.label, [
        { k: 'Cd', label: `${t.label}: Cd`, type: 'expr', def: JSON.stringify(t.Cd) }, { k: 'area', label: 'Drag area', type: 'expr', def: JSON.stringify(t.area) }], { persist: false, cols: 'two' }) }));
      left.append(card({ title: 'Quadratic drag on pontoon types', cls: 'sea' }, 'Cd and projected area per DOF (3 or 6 values). Used by stochastic linearisation (drag_pontoons).',
        ptForms.map(x => x.f.el), h('div.row', { style: { marginTop: '10px' } }, btn('Apply', act(async () => {
          await run('model/set', { ptypes: ptForms.map(x => ({ label: x.t.label, ...x.f.get() })) }); toast('Pontoon types updated');
        }), 'primary'))));
      const pRows = d.hydro.pontoons;
      const tb = h('tbody');
      const edits = {};
      pRows.forEach(p => {
        const mk = (k, v) => { const i = h('input', { type: 'text', value: v, style: { width: '150px' } }); i.onchange = () => { (edits[p.label] = edits[p.label] || { label: p.label })[k] = i.value; }; return i; };
        const tri = (k) => { const s = h('select', {}, ['default', 'true', 'false'].map(o => h('option', { value: o }, o))); s.value = p[k] === null ? 'default' : String(p[k]); s.onchange = () => { (edits[p.label] = edits[p.label] || { label: p.label })[k] = s.value === 'default' ? null : s.value === 'true'; }; return s; };
        tb.append(h('tr', {}, h('td.l', {}, p.label), h('td.l', {}, mk('Cd', JSON.stringify(p.Cd))), h('td.l', {}, mk('Cquad', JSON.stringify(p.Cquad))), h('td', {}, tri('current_affects_Q')), h('td', {}, tri('current_affects_k'))));
      });
      right.append(card({ title: 'Pontoons', cls: 'sea' }, 'Per-pontoon drag coefficient override, direct quadratic damping and current effects (Doppler shift of transfer functions / wave numbers).',
        h('div.tablewrap', {}, h('table.grid', {}, h('thead', {}, h('tr', {}, ['Pontoon', 'Cd', 'Cquad', 'Current → Q', 'Current → k'].map(c => h('th.l', {}, c)))), tb)),
        h('div.row', { style: { marginTop: '10px' } }, btn('Apply', act(async () => { await run('model/set', { pontoons: Object.values(edits) }); toast('Pontoons updated'); }), 'primary'))));
    }
    if (d.aero) {
      const secForms = d.aero.sections.map(s => ({ s, f: makeForm('edit_as_' + s.name, [
        { type: 'heading', label: `Section "${s.name}" (${s.n_elements} elements, ADs: ${s.ad_type})` },
        ...['B', 'D', 'Cd', 'dCd', 'Cl', 'dCl', 'Cm', 'dCm'].map(k => ({ k, label: k.startsWith('d') ? k.slice(1) + "'" : k, type: 'expr', def: String(s[k] ?? '') })),
        { k: 'admittance', label: 'Admittance function', type: 'expr', def: s.admittance || '', span: 2, ph: 'lambda fred: …' },
        { k: 'ads', label: 'Aerodynamic derivatives', type: 'select', opts: [['keep', s.quasisteady ? 'Quasi-steady (regenerate)' : 'Keep current'], ['quasisteady', 'Quasi-steady from coefficients'], ['flatplate', 'Flat plate (Theodorsen)'], ['expr', 'Expression (ADs object or dict)']], def: 'keep', span: 2 },
        { k: 'ads_expr', label: 'AD expression', type: 'expr', span: 'full', ph: "dict(H1=lambda v: …, …) or wawi.wind.ADs(…)" },
      ], { persist: false }) }));
      right.append(card({ title: 'Aero sections', cls: 'wind' }, 'Load coefficients and their slopes; quasi-steady ADs are regenerated from them. Prepared aero matrices are reset on apply.',
        secForms.map(x => x.f.el), h('div.row', { style: { marginTop: '10px' } }, btn('Apply', act(async () => {
          await run('model/set', { aero_sections: secForms.map(x => ({ name: x.s.name, ...x.f.get() })) }); toast('Aero sections updated');
        }), 'primary'))));
    }
    const dragTa = h('textarea.code', { rows: 6, placeholder: '{"mooring": {"sections": "MooringWire", "Cd": [0.1, 1.2, 1.2], "D": 0.146, "rho": 1025.0}}' });
    left.append(card('Drag elements', 'Assign quadratic drag to FE elements by section name (Model.assign_drag_elements, same format as drag_elements.json).', dragTa,
      h('div.row', { style: { marginTop: '8px' } }, btn('Assign drag elements', act(async () => { const r = await run('model/set', { drag_elements_json: dragTa.value }); toast(r.changed.join(', ')); }), 'small'))));
    left.append(card('Remove or reset', null, h('div.row', {},
      btn('Remove aero part', act(async () => { if (confirm('Remove the aerodynamic part (model.aero = None)?')) await run('model/set', { remove: ['aero'] }); }), 'small danger'),
      btn('Remove drag elements', act(async () => { await run('model/set', { remove: ['drag_elements'] }); }), 'small'),
      btn('Unassign sea state', act(async () => { await run('model/set', { remove: ['seastate'] }); }), 'small'),
      btn('Unassign wind state', act(async () => { await run('model/set', { remove: ['windstate'] }); }), 'small'),
      btn('Clear precomputed actions', act(async () => { await run('model/set', { clear_precomputed: true }); toast('Cleared'); }), 'small'),
      btn('Reset results', act(async () => { await run('model/set', { reset_results: true }); toast('Results reset'); }), 'small'))));
    el.append(h('div.cols.even', {}, left, right));
  };
}

/* ------------------------------------------------------------------ pontoon types */
function buildPtypes(el) {
  const sel = h('select'); const qSel = h('select', {}, ['M', 'C', 'K'].map(q => h('option', { value: q }, { M: 'Added + structural mass M(ω)', C: 'Radiation damping C(ω)', K: 'Restoring stiffness K(ω)' }[q])));
  const offdiag = h('input', { type: 'checkbox' });
  const dofSel = h('select', {}, DOFS.map((d, i) => h('option', { value: i }, 'Q ' + d)));
  const thSel = h('select');
  const p1 = plotDiv(), p2 = plotDiv(), p3 = plotDiv();
  let data = null;
  const drawMat = () => {
    if (!data) return;
    const q = qSel.value; const M = data[q];
    if (!M) { plot(p1, [], { title: { text: `${q} not defined for this type`, font: { size: 13 } } }); return; }
    const tr = [];
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
      if (!offdiag.checked && i !== j) continue;
      if (offdiag.checked && j < i) continue;
      tr.push({ x: data.omega, y: M[i][j], name: `${q}${i + 1}${j + 1}`, mode: 'lines', line: { width: i === j ? 2 : 1, dash: i === j ? 'solid' : 'dot' } });
    }
    plot(p1, tr, { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: q }, legend: { orientation: 'v', x: 1.02, y: 1 }, margin: { r: 90, l: 70, t: 20, b: 46 } });
  };
  const drawQ = () => {
    if (!data || !data.Qabs) { plot(p2, [], { title: { text: 'No excitation transfer function Q', font: { size: 13 } } }); clear(p3); return; }
    const d = +dofSel.value;
    plot(p2, [{ type: 'heatmap', x: data.omegaQ, y: data.thetaQ_deg, z: data.Qabs[d], colorscale: [[0, '#f4f9fa'], [0.5, '#5aa9b8'], [1, '#0f3d4a']], colorbar: { title: '|Q|' } }],
      { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'Heading θ [°]' }, title: { text: `|Q| for ${DOFS[d]}`, font: { size: 13 } } });
    const t = +thSel.value || 0;
    plot(p3, DOFS.map((nm, k) => ({ x: data.omegaQ, y: data.Qabs[k][t], name: nm, mode: 'lines' })),
      { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: '|Q|', type: 'log' }, title: { text: `All DOFs at θ = ${fmt(data.thetaQ_deg[t])}°`, font: { size: 13 } } });
  };
  const load = act(async () => {
    if (!sel.value) return;
    data = await api('model/pontoon_type_data', { label: sel.value });
    if (data.thetaQ_deg) {
      clear(thSel).append(...data.thetaQ_deg.map((t, i) => h('option', { value: i }, `θ = ${fmt(t)}°`)));
      const z = data.thetaQ_deg.reduce((b, t, i) => Math.abs(t) < Math.abs(data.thetaQ_deg[b]) ? i : b, 0); thSel.value = z;
    }
    drawMat(); drawQ();
  });
  sel.onchange = load; qSel.onchange = drawMat; offdiag.onchange = drawMat; dofSel.onchange = drawQ; thSel.onchange = drawQ;
  el._refresh = async () => {
    if (!S.model || !S.model.has_hydro) { clear(el).append(emptyBox('The active model has no pontoons.')); el._built = false; return; }
    if (!el._built) {
      clear(el).append(
        card({ title: 'Hydrodynamic coefficients', cls: 'sea' }, null, h('div.row', {}, 'Pontoon type', sel, qSel, h('label.small', {}, offdiag, ' off-diagonal terms')), p1),
        h('div.cols.even', {}, card({ title: 'Wave excitation transfer function', cls: 'sea' }, null, h('div.row', {}, dofSel), p2),
          card({ title: 'Per heading', cls: 'sea' }, null, h('div.row', {}, thSel), p3)));
      el._built = true;
    }
    const info = await api('model/info');
    const cur = sel.value;
    clear(sel).append(...info.hydro.types.map(t => h('option', { value: t.label }, t.label)));
    if (cur) sel.value = cur;
    load();
  };
}

/* ------------------------------------------------------------------ aero */
function buildAero(el) {
  const sel = h('select'); const vmax = h('input', { type: 'text', value: '5', style: { width: '70px' } });
  const pl = plotDiv('tall');
  const draw = act(async () => {
    const d = await api('model/aero_ads', { section: sel.value, vmax: vmax.value });
    const traces = [];
    ['P', 'H', 'A'].forEach((g, gi) => {
      for (let i = 1; i <= 6; i++) {
        const k = g + i; if (!d.ads[k]) continue;
        traces.push({ x: d.v, y: d.ads[k], name: k + '*', xaxis: 'x' + (gi + 1), yaxis: 'y' + (gi + 1), mode: 'lines', line: { color: SERIES[i - 1], width: 1.6 }, legendgroup: 'n' + i, showlegend: gi === 0 });
      }
    });
    plot(pl, traces, {
      grid: { rows: 1, columns: 3, pattern: 'independent' },
      xaxis: { title: 'v = U/(Bω)' }, xaxis2: { title: 'v = U/(Bω)' }, xaxis3: { title: 'v = U/(Bω)' },
      yaxis: { title: 'P*' }, yaxis2: { title: 'H*' }, yaxis3: { title: 'A*' },
      title: { text: `Aerodynamic derivatives (${d.type})`, font: { size: 13 } }, legend: { orientation: 'h', y: -0.15 },
    });
  });
  sel.onchange = draw;
  el._refresh = async () => {
    if (!S.model || !S.model.has_aero) { clear(el).append(emptyBox('The active model has no aerodynamic part.')); el._built = false; return; }
    if (!el._built) {
      clear(el).append(card({ title: 'Aerodynamic derivatives', cls: 'wind' }, 'Self-excited force coefficients of each aero section as functions of reduced velocity. Change them in Model → Edit.',
        h('div.row', {}, 'Section', sel, 'v max', vmax, btn('Plot', draw, 'small')), pl));
      el._built = true;
    }
    const cur = sel.value;
    clear(sel).append(...(S.model.aero_sections || []).map(s => h('option', { value: s }, s)));
    if (cur) sel.value = cur;
    draw();
  };
}

/* ------------------------------------------------------------------ 3D view */
function buildView(el) {
  const v = new Viewer3D({ modes: true });
  el._viewer = v;
  const out = h('div');
  const fR = makeForm('render', [
    { k: 'kwargs', label: 'Extra keyword arguments for Model.plot / plot_mode', type: 'expr', span: 'full', ph: "dict(plot_water=True, annotate_pontoon_type=True)" },
    { k: 'folder', label: 'Export mode shapes to folder', type: 'path', pathKind: 'dir', span: 'full' },
    { k: 'n_modes', label: 'Number of modes to export', type: 'expr' }, { k: 'format', label: 'Format', type: 'select', opts: ['png', 'pdf', 'svg', 'eps'], def: 'png' },
  ]);
  const renderCard = card('Native WAWI rendering (pyvista)', 'Off-screen rendering with the model\'s own pyvista plots, and batch export of mode shapes (Model.export_modeshapes). Needs pyvista with off-screen support.',
    fR.el, h('div.row', { style: { marginTop: '10px' } },
      btn('Render model', act(async () => { const r = await run('model/render', { kind: 'model', kwargs: fR.get().kwargs }); clear(out).append(h('img.fig', { src: r.image })); })),
      btn('Render current mode', act(async () => { const r = await run('model/render', { kind: 'mode', mode: String((+v.modeIn.value || 1) - 1), use_dry: v.srcSel.value === 'dry', kwargs: fR.get().kwargs }); clear(out).append(h('img.fig', { src: r.image })); })),
      btn('Export mode shapes', act(async () => { const f = fR.get(); await run('model/export_modeshapes', f); toast('Exported to ' + f.folder); })),
      btn('Pontoon axes (2D)', act(async () => { const r = await api('model/plot_2d_tmats', {}); clear(out); images(out, r.images); }), 'ghost')),
    out);
  el._refresh = async () => {
    if (needModel(el, 'the 3D view')) { el._built = false; return; }
    if (!el._built) { clear(el).append(v.el, h('div', { style: { height: '18px' } }), renderCard); el._built = true; }
    try { await v.load(); } catch (e) { showError(e); }
  };
  bus.on('model', () => { if (el.classList.contains('active') && el._built) v.load().catch(() => {}); });
}
