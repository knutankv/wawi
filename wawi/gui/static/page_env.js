/* WAWI GUI – Environment page: sea state, wind state, sea surface simulation */
'use strict';

PAGES.env = (root) => {
  subtabs(root, [
    { id: 'sea', label: 'Sea state', build: buildSea },
    { id: 'wind', label: 'Wind state', build: buildWind },
    { id: 'surface', label: 'Sea surface simulation', build: buildSurface },
  ], { key: 'env' });
};

const FUN_HINT = 'number, or a function of x and/or y, e.g. lambda x: 2.1 - 0.5*x/1500';

function buildSea(el) {
  const f = makeForm('sea', [
    { k: 'Hs', label: 'Significant wave height Hs [m]', type: 'expr', def: '2.1', hint: FUN_HINT, span: 2 },
    { k: 'Tp', label: 'Peak period Tp [s]', type: 'expr', def: '8.3', span: 2 },
    { k: 'gamma', label: 'JONSWAP peak enhancement γ', type: 'expr', def: '8', span: 2 },
    { k: 'theta0', label: 'Mean wave direction θ0', type: 'expr', def: '90', span: 2, hint: 'direction of propagation, counter-clockwise from x' },
    { k: 's', label: 'Spreading exponent s (cos-2s)', type: 'expr', def: '12', span: 2, hint: 'np.inf gives long-crested waves' },
    { k: 'angle_unit', label: 'Angle unit', type: 'select', opts: [['deg', 'degrees'], ['rad', 'radians']], def: 'deg' },
    { k: 'name', label: 'Name', def: '' },
    { k: 'U', label: 'Current speed U [m/s]', type: 'expr', def: '', ph: '0' },
    { k: 'thetaU', label: 'Current direction θU', type: 'expr', def: '', ph: '0' },
    { k: 'depth', label: 'Water depth', type: 'expr', def: '', ph: 'infinite' },
    { k: 'centered_dirdist', label: 'Centred directional distribution', type: 'check', def: true },
    { k: 'use_robust_D', label: 'Robust D(θ) evaluation', type: 'check', def: true },
  ]);
  const fAdv = makeForm('sea_adv', [
    { k: 'options', label: 'Options', type: 'expr', span: 'full', ph: "{'keep_coherence': True, 'truncate_theta': [-np.pi/2, np.pi/2], 'dtheta': 0.05}", hint: 'keep_coherence=False ignores spatial coherence; truncate_theta + dtheta fix the integration grid over θ.' },
    { k: 'pontoon_options', label: 'Pontoon options', type: 'expr', span: 'full', ph: "{'current_affects_Q': True, 'current_affects_k': True}" },
    { k: 'origin', label: 'Origin of (x, y) for parameter functions', type: 'expr', ph: '[0, 0]', span: 2 },
    { k: 'ranges', label: 'Validity ranges', type: 'expr', ph: "{'x': [-1500, 1500]}", span: 2 },
    { k: 'plot_label', label: 'Plot label', def: '' },
  ]);
  const status = h('pre.out', { style: { display: 'none' } });
  const pS = plotDiv('short'), pD = plotDiv('short'), p2 = plotDiv(), pP = plotDiv('short'), mpl = h('div');
  const evx = h('input', { type: 'text', value: '0', style: { width: '80px' } }), evy = h('input', { type: 'text', value: '0', style: { width: '80px' } });
  const which = h('select', {}, h('option', { value: 'editor' }, 'Edited sea state'), h('option', { value: 'model' }, 'Assigned to model'));
  const statsBox = h('div.stats');
  const params = () => ({ ...f.get(), ...fAdv.get() });
  const showStatus = (txt) => { status.style.display = 'block'; status.textContent = txt; };

  async function drawPlots() {
    const d = await api('seastate/plot_data', { which: which.value, x: evx.value, y: evy.value });
    clear(statsBox).append(
      h('div.stat', {}, h('b', {}, fmt(d.params_at_xy.Hs, 3) + ' m'), h('span', {}, 'Hs at (x, y)')),
      h('div.stat', {}, h('b', {}, fmt(d.Hs_m0, 3) + ' m'), h('span', {}, '4√m0 from S(ω)')),
      h('div.stat', {}, h('b', {}, fmt(d.Tp_peak, 3) + ' s'), h('span', {}, 'Tp at spectral peak')),
      h('div.stat', {}, h('b', {}, fmt(d.theta0_abs * 180 / Math.PI, 4) + '°'), h('span', {}, 'θ0 at (x, y)')),
      h('div.stat', {}, h('b', {}, d.short_crested ? 'short-crested' : 'long-crested'), h('span', {}, d.homogeneous ? 'homogeneous' : 'inhomogeneous')));
    plot(pS, [{ x: d.omega, y: d.S, mode: 'lines', line: { color: C.sea, width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(29,122,140,0.12)', name: 'S(ω)' }],
      { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'Sη(ω) [m²s/rad]' }, title: { text: 'Wave spectrum (JONSWAP)', font: { size: 13 } }, showlegend: false });
    const thdeg = d.theta.map(t => t * 180 / Math.PI);
    plot(pD, [{ x: thdeg, y: d.D, mode: 'lines', line: { color: C.ink, width: 2 }, name: 'D(θ)' }],
      { xaxis: { title: d.theta_shift ? 'θ − θ0 [°] (centred)' : 'θ [°]' }, yaxis: { title: 'D(θ) [1/rad]' }, title: { text: 'Directional distribution (cos-2s)', font: { size: 13 } }, showlegend: false });
    if (d.S2d) plot(p2, [{ type: 'contour', x: d.S2d.omega, y: d.S2d.theta.map(t => t * 180 / Math.PI), z: d.S2d.z, colorscale: [[0, '#ffffff'], [0.2, '#cfe8ec'], [0.6, '#3f97a8'], [1, '#0d3440']], contours: { coloring: 'heatmap' }, colorbar: { title: 'Sη(ω,θ)' } }],
      { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'θ [°]' }, title: { text: 'Two-dimensional spectrum Sη(ω,θ) = S(ω)·D(θ)', font: { size: 13 } } });
    else clear(p2).append(emptyBox('Long-crested sea: no 2D spectrum.'));
    if (d.pontoons && d.pontoons.length) {
      const x = d.pontoons.map((p, i) => i + 1);
      plot(pP, ['Hs', 'Tp', 'theta0', 'U'].map((k, i) => ({ x, y: d.pontoons.map(p => p[k]), name: k === 'theta0' ? 'θ0 [°]' : k, mode: 'lines+markers', yaxis: i ? 'y' + (i + 1) : 'y', marker: { size: 5 } })),
        { grid: { rows: 4, columns: 1, pattern: 'coupled' }, yaxis: { title: 'Hs' }, yaxis2: { title: 'Tp' }, yaxis3: { title: 'θ0 [°]' }, yaxis4: { title: 'U' },
          xaxis: { title: 'Pontoon number' }, height: 420, showlegend: false, title: { text: 'Sea-state parameters at the pontoons', font: { size: 13 } },
          margin: { l: 60, r: 10, t: 30, b: 40 } });
    } else clear(pP).append(emptyBox('Load a model with pontoons to see parameters at each pontoon.'));
  }
  const build = (assign) => act(async () => {
    const r = await api('seastate/build', { ...params(), assign });
    showStatus(r.str + (r.homogeneous ? '' : `\nFunction-valued parameters: ${r.fun_pars.join(', ')}`));
    if (assign) toast('Sea state assigned to the model');
    which.value = 'editor'; await drawPlots();
  });
  const loadJson = act(async () => {
    const p = await browse({ ext: ['.json'], title: 'Open sea state JSON' }); if (!p) return;
    const r = await api('seastate/from_json', { path: p });
    f.reset(); fAdv.reset(); f.set(r.form); fAdv.set(r.form); showStatus(r.str); await drawPlots();
  });
  const saveJson = act(async () => {
    const p = await browse({ kind: 'save', ext: ['.json'], title: 'Save sea state JSON' });
    const r = await api('seastate/to_json', { form: params(), path: p || '' });
    if (r.download) triggerDownload(r.download); else toast('Saved ' + r.path);
  });
  const fromModel = act(async () => { const r = await api('seastate/from_model'); f.reset(); fAdv.reset(); f.set(r.form); fAdv.set(r.form); showStatus(r.str); which.value = 'model'; await drawPlots(); });

  el.append(h('div.cols.wide-left', {},
    h('div', {}, card({ title: 'Sea state', cls: 'sea' }, 'JONSWAP spectrum with cos-2s spreading (wawi.model.Seastate). Any parameter can vary over the wave field as a Python function of x and y.',
      f.el, h('details.adv', {}, h('summary', {}, 'More options'), fAdv.el),
      h('div.row', { style: { marginTop: '12px' } }, btn('Build and assign', build(true), 'sea'), btn('Build only', build(false)),
        h('span.spacer'), btn('Open JSON…', loadJson, 'ghost small'), btn('Save JSON…', saveJson, 'ghost small'), btn('Copy from model', fromModel, 'ghost small')),
      status)),
    h('div', {}, card({}, null, h('div.row', {}, 'Show', which, 'at x =', evx, 'y =', evy, btn('Update plots', act(drawPlots), 'small'),
      h('span.spacer'), btn('plot_S2d', act(async () => { const r = await api('seastate/mpl', { kind: 'S2d', which: which.value, x: evx.value, y: evy.value }); clear(mpl); images(mpl, r.images); }), 'small ghost'),
      btn('plot_dir_and_crests', act(async () => { const r = await api('seastate/mpl', { kind: 'crests', which: which.value, x: evx.value, y: evy.value }); clear(mpl); images(mpl, r.images); }), 'small ghost')),
      statsBox, h('div.cols.even', {}, pS, pD), p2, pP, mpl))));
  bus.on('model', (m) => { if (m && m.seastate && !status.textContent) { which.value = 'model'; drawPlots().catch(() => {}); } });
}

function buildWind(el) {
  const f = makeForm('wind', [
    { k: 'U0', label: 'Mean wind speed U0 [m/s]', type: 'expr', def: '20.0' },
    { k: 'direction', label: 'Direction [°] (coming from, clockwise)', type: 'expr', def: '90.0' },
    { k: 'spectrum_type', label: 'Turbulence spectrum', type: 'select', opts: [['kaimal', 'Kaimal'], ['vonkarman', 'von Kármán']], def: 'kaimal' },
    { k: 'rho', label: 'Air density ρ', type: 'expr', def: '1.225' },
    { type: 'heading', label: 'Turbulence intensities' },
    { k: 'Iu', label: 'Iu', type: 'expr', def: '0.136' }, { k: 'Iv', label: 'Iv', type: 'expr', def: '' }, { k: 'Iw', label: 'Iw', type: 'expr', def: '0.072' },
    { type: 'heading', label: 'Integral length scales [m]' },
    { k: 'Lux', label: 'Lux', type: 'expr', def: '115' }, { k: 'Lvx', label: 'Lvx', type: 'expr', def: '' }, { k: 'Lwx', label: 'Lwx', type: 'expr', def: '9.58' },
    { type: 'heading', label: 'Kaimal spectral shape parameters' },
    { k: 'Au', label: 'Au', type: 'expr', def: '6.8' }, { k: 'Av', label: 'Av', type: 'expr', def: '' }, { k: 'Aw', label: 'Aw', type: 'expr', def: '9.4' },
    { type: 'heading', label: 'Coherence decay coefficients' },
    { k: 'Cux', label: 'Cux', type: 'expr' }, { k: 'Cvx', label: 'Cvx', type: 'expr' }, { k: 'Cwx', label: 'Cwx', type: 'expr' },
    { k: 'Cuy', label: 'Cuy', type: 'expr', def: '10.0' }, { k: 'Cvy', label: 'Cvy', type: 'expr' }, { k: 'Cwy', label: 'Cwy', type: 'expr', def: '6.5' },
    { k: 'Cuz', label: 'Cuz', type: 'expr' }, { k: 'Cvz', label: 'Cvz', type: 'expr' }, { k: 'Cwz', label: 'Cwz', type: 'expr' },
  ]);
  const fAdv = makeForm('wind_adv', [
    { k: 'scaling', label: 'Mean-wind scaling U(x)/U0', type: 'expr', span: 'full', ph: 'lambda x: (x[2]/10)**0.12', hint: 'Function of the node coordinate vector x = [x, y, z]; makes the mean wind inhomogeneous.' },
    { k: 'x_ref', label: 'Reference point x_ref', type: 'expr', ph: '[0, 0, 0]' }, { k: 'name', label: 'Name' },
    { k: 'options', label: 'Options', type: 'expr', span: 'full' },
  ]);
  const status = h('pre.out', { style: { display: 'none' } });
  const which = h('select', {}, h('option', { value: 'editor' }, 'Edited wind state'), h('option', { value: 'model' }, 'Assigned to model'));
  const fcoh = h('input', { type: 'text', value: '0.1', style: { width: '70px' } });
  const pS = plotDiv(), pC = plotDiv('short'), pU = plotDiv('short'), statsBox = h('div.stats');
  const params = () => ({ ...f.get(), ...fAdv.get() });
  const showStatus = (t) => { status.style.display = 'block'; status.textContent = t; };
  async function drawPlots() {
    const d = await api('windstate/plot_data', { which: which.value, f_coh: fcoh.value });
    clear(statsBox).append(
      h('div.stat', {}, h('b', {}, d.beaufort ?? '–'), h('span', {}, 'Beaufort description')),
      ...['u', 'v', 'w'].map((c, i) => h('div.stat', {}, h('b', {}, fmt(d.sigma[i], 3) + ' m/s'), h('span', {}, `σ${c}`))));
    const tr = [];
    ['u', 'v', 'w'].forEach((c, i) => {
      tr.push({ x: d.omega, y: d.kaimal[i], name: `S${c} Kaimal`, mode: 'lines', line: { color: SERIES[i], width: d.spectrum_type === 'kaimal' ? 2.4 : 1.2, dash: d.spectrum_type === 'kaimal' ? 'solid' : 'dot' } });
      tr.push({ x: d.omega, y: d.vonkarman[i], name: `S${c} von Kármán`, mode: 'lines', line: { color: SERIES[i], width: d.spectrum_type !== 'kaimal' ? 2.4 : 1.2, dash: d.spectrum_type !== 'kaimal' ? 'solid' : 'dot' } });
    });
    plot(pS, tr, { xaxis: { title: 'ω [rad/s]', type: 'log' }, yaxis: { title: 'S(ω) [m²/s]', type: 'log' }, title: { text: 'Turbulence spectra (solid: selected type)', font: { size: 13 } }, legend: { orientation: 'v', x: 1.02, y: 1 }, margin: { r: 150, l: 70, t: 30, b: 46 } });
    plot(pC, Object.entries(d.coh).filter(([k, v]) => v && v.some(x => x < 0.9999)).map(([k, v]) => ({ x: d.coh_dist, y: v, name: k, mode: 'lines' })),
      { xaxis: { title: 'Separation Δ [m]' }, yaxis: { title: 'Coherence', range: [0, 1.02] }, title: { text: `Davenport coherence at f = ${d.f_coh} Hz`, font: { size: 13 } } });
    if (d.U_profile) plot(pU, [{ x: d.U_profile.x, y: d.U_profile.U, mode: 'markers', marker: { color: C.wind, size: 5 }, name: 'U' }],
      { xaxis: { title: 'x [m]' }, yaxis: { title: 'U [m/s]' }, title: { text: 'Mean wind at aero-section nodes', font: { size: 13 } }, showlegend: false });
    else clear(pU);
  }
  const build = (assign) => act(async () => {
    const r = await api('windstate/build', { ...params(), assign }); showStatus(r.str);
    if (assign) toast('Wind state assigned to the model');
    which.value = 'editor'; await drawPlots();
  });
  const loadJson = act(async () => {
    const p = await browse({ ext: ['.json'], title: 'Open wind state JSON' }); if (!p) return;
    const r = await api('windstate/from_json', { path: p }); f.reset(); fAdv.reset(); f.set(r.form); fAdv.set(r.form); showStatus(r.str); await drawPlots();
  });
  const saveJson = act(async () => {
    const p = await browse({ kind: 'save', ext: ['.json'], title: 'Save wind state JSON' });
    const r = await api('windstate/to_json', { form: params(), path: p || '' });
    if (r.download) triggerDownload(r.download); else toast('Saved ' + r.path);
  });
  const fromModel = act(async () => { const r = await api('windstate/from_model'); f.reset(); fAdv.reset(); f.set(r.form); fAdv.set(r.form); showStatus(r.str); which.value = 'model'; await drawPlots(); });
  el.append(h('div.cols.wide-left', {},
    h('div', {}, card({ title: 'Wind state', cls: 'wind' }, 'Mean wind, single-point turbulence spectra and coherence (wawi.model.Windstate).',
      f.el, h('details.adv', {}, h('summary', {}, 'Inhomogeneous mean wind and more'), fAdv.el),
      h('div.row', { style: { marginTop: '12px' } }, btn('Build and assign', build(true), 'wind'), btn('Build only', build(false)), h('span.spacer'),
        btn('Open JSON…', loadJson, 'ghost small'), btn('Save JSON…', saveJson, 'ghost small'), btn('Copy from model', fromModel, 'ghost small')),
      status)),
    h('div', {}, card({}, null, h('div.row', {}, 'Show', which, 'coherence at f =', fcoh, 'Hz', btn('Update plots', act(drawPlots), 'small')), statsBox, pS, h('div.cols.even', {}, pC, pU)))));
  bus.on('model', (m) => { if (m && m.windstate && !status.textContent) { which.value = 'model'; drawPlots().catch(() => {}); } });
}

function buildSurface(el) {
  const f = makeForm('surface', [
    { k: 'which', label: 'Sea state', type: 'select', opts: [['editor', 'Edited sea state'], ['model', 'Assigned to model']], def: 'editor' },
    { k: 'x', label: 'x grid', type: 'expr', def: 'np.arange(-1500, 1500, 10)', span: 2 },
    { k: 'y', label: 'y grid', type: 'expr', def: 'np.arange(-300, 50, 10)', span: 2 },
    { k: 'omega', label: 'Frequencies ω', type: 'expr', def: 'np.arange(0.01, 3.0, 0.05)', span: 2 },
    { k: 'fs', label: 'Sampling frequency fs [Hz]', type: 'expr', def: '2.0' },
    { k: 'theta', label: 'Directions θ (optional)', type: 'expr', ph: 'sea state default' },
    { k: 'time_history', label: 'Full time history', type: 'check', def: true },
    { k: 'max_frames', label: 'Frames to display', type: 'expr', def: '40' },
    { k: 'gif', label: 'Also make GIF (animate_surface)', type: 'check', def: false },
    { k: 'fps', label: 'GIF frames per second', type: 'expr', def: '10' },
  ]);
  const pl = plotDiv('tall'); const slider = h('input', { type: 'range', min: 0, max: 0, value: 0, style: { width: '320px' } });
  const tlab = h('span.small.muted'); const extra = h('div.row');
  let res = null, playing = null;
  const show = (i) => {
    if (!res) return;
    tlab.textContent = `t = ${fmt(res.t[i], 4)} s  (frame ${i + 1}/${res.frames.length})`;
    Plotly.restyle(pl, { z: [res.frames[i]] }, [0]);
  };
  slider.oninput = () => show(+slider.value);
  const play = btn('Play', () => {
    if (playing) { clearInterval(playing); playing = null; play.textContent = 'Play'; return; }
    play.textContent = 'Pause';
    playing = setInterval(() => { slider.value = (+slider.value + 1) % res.frames.length; show(+slider.value); }, 120);
  }, 'small');
  el.append(h('div.cols', {}, card({ title: 'Simulate the sea surface', cls: 'sea' }, 'Monte Carlo realisation of the wave field with Seastate.simulate. Results are stored as eta_sim, t_sim, x_sim, y_sim.',
    f.el, h('div.row', { style: { marginTop: '12px' } }, btn('Simulate', act(async () => {
      res = await run('seastate/simulate', f.get());
      let mx = 0; for (const fr of res.frames) for (const row of fr) for (const v of row) { if (v !== null && Math.abs(v) > mx) mx = Math.abs(v); }
      plot(pl, [{ type: 'heatmap', x: res.x, y: res.y, z: res.frames[0], zmin: -mx, zmax: mx, colorscale: 'RdBu', reversescale: true, colorbar: { title: 'η [m]' } }],
        { xaxis: { title: 'x [m]' }, yaxis: { title: 'y [m]', scaleanchor: 'x' }, title: { text: `Surface elevation (σ = ${fmt(res.std, 3)} m)`, font: { size: 13 } } });
      slider.max = res.frames.length - 1; slider.value = 0; show(0);
      clear(extra); if (res.gif_download) extra.append(btn('Download GIF', () => triggerDownload(res.gif_download), 'small'));
    }), 'sea'))),
  card({}, null, h('div.row', {}, play, slider, tlab, extra), pl)));
}
