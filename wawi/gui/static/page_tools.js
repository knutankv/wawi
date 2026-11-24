/* WAWI GUI – stand-alone calculators */
'use strict';

const TOOLDEFS = [
  { id: 'wave_spectrum', title: 'Wave spectra', desc: 'JONSWAP (wave.jonswap), DNV JONSWAP (jonswap_dnv) and Pierson–Moskowitz (pm2), with the JONSWAP peak-enhancement factor.', cls: 'sea',
    fields: [{ k: 'Hs', label: 'Hs [m]', type: 'expr', def: '2.0' }, { k: 'Tp', label: 'Tp [s]', type: 'expr', def: '8.0' }, { k: 'gamma', label: 'γ', type: 'expr', def: '3.3' },
      { k: 'sigma', label: 'σ (below, above peak)', type: 'expr', def: '[0.07, 0.09]' }, { k: 'omega', label: 'ω', type: 'expr', def: 'np.linspace(0.05, 3, 400)', span: 2 },
      { k: 'kinds', label: 'Spectra', type: 'multi', opts: [['jonswap', 'JONSWAP'], ['jonswap_dnv', 'JONSWAP (DNV)'], ['pm2', 'Pierson–Moskowitz']], def: ['jonswap', 'pm2'], span: 'full' },
      { k: 'alpha', label: 'Phillips α (optional, Hs from γ, α, Tp)', type: 'expr', span: 2 }],
    draw(r, out) {
      const p = plotDiv(); out.append(h('div.stats', {}, Object.entries(r.series).map(([k, s]) => h('div.stat', {}, h('b', {}, `${fmt(s.Hs_m0, 3)} m / ${fmt(s.Tp_peak, 3)} s`), h('span', {}, `${k}: 4√m0 / peak period`))),
        r.Hs_from_alpha !== undefined ? h('div.stat', {}, h('b', {}, fmt(r.Hs_from_alpha, 4) + ' m'), h('span', {}, 'Hs from γ, α, Tp')) : null), p);
      plot(p, Object.entries(r.series).map(([k, s]) => ({ x: r.omega, y: s.S, name: k, mode: 'lines' })), { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'S(ω) [m²s/rad]' } });
    } },
  { id: 'dirdist', title: 'Directional spreading', desc: 'cos-2s distributions (wave.dirdist, dirdist_decimal, dirdist_robust, dirdist_decimal_inv); the integral should be 1.', cls: 'sea',
    fields: [{ k: 's', label: 's', type: 'expr', def: '10' }, { k: 'theta0', label: 'θ0 [°]', type: 'expr', def: '0' },
      { k: 'kinds', label: 'Functions', type: 'multi', opts: ['dirdist_robust', 'dirdist_decimal', 'dirdist', 'dirdist_decimal_inv'], def: ['dirdist_robust', 'dirdist_decimal'], span: 'full' }],
    draw(r, out) {
      const p = plotDiv(); out.append(h('div.stats', {}, Object.entries(r.series).map(([k, s]) => h('div.stat', {}, h('b', {}, s.error ? 'error' : fmt(s.integral, 5)), h('span', {}, `∫D dθ, ${k}`)))), p);
      plot(p, Object.entries(r.series).filter(([k, s]) => s.D).map(([k, s]) => ({ x: r.theta.map(t => t * 180 / Math.PI), y: s.D, name: k, mode: 'lines' })), { xaxis: { title: 'θ [°]' }, yaxis: { title: 'D(θ) [1/rad]' } });
    } },
  { id: 'dispersion', title: 'Dispersion relation', desc: 'Wave number, wave length and celerity for finite depth and current (wave.dispersion_relation_scalar).', cls: 'sea',
    fields: [{ k: 'depth', label: 'Depths [m]', type: 'expr', def: '[np.inf, 50, 20]' }, { k: 'U', label: 'Current U [m/s]', type: 'expr', def: '0' },
      { k: 'theta_rel_U', label: 'Current angle rel. waves [°]', type: 'expr', def: '0' }, { k: 'g', label: 'g', type: 'expr', def: '9.81' }, { k: 'omega', label: 'ω', type: 'expr', def: 'np.linspace(0.05, 3, 200)', span: 2 }],
    draw(r, out) {
      const p1 = plotDiv('short'), p2 = plotDiv('short'); out.append(h('div.cols.even', {}, p1, p2));
      const nm = s => s.depth === null ? 'deep water' : `h = ${s.depth} m`;
      plot(p1, r.series.map(s => ({ x: r.omega, y: s.wavelength, name: nm(s), mode: 'lines' })), { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'Wave length [m]', type: 'log' } });
      plot(p2, r.series.map(s => ({ x: r.omega, y: s.celerity, name: nm(s), mode: 'lines' })), { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'Phase speed [m/s]' } });
    } },
  { id: 'wave_coherence', title: 'Wave coherence', desc: 'Coherence of wave elevation between two points for a cos-2s sea (wave.get_coh_cos2s).', cls: 'sea',
    fields: [{ k: 'dx', label: 'Δx [m]', type: 'expr', def: '100' }, { k: 'dy', label: 'Δy [m]', type: 'expr', def: '0' }, { k: 's', label: 's', type: 'expr', def: '10' },
      { k: 'theta0', label: 'θ0 [°]', type: 'expr', def: '0' }, { k: 'depth', label: 'Depth', type: 'expr', def: 'np.inf' }, { k: 'k_max', label: 'Series terms', type: 'expr', def: '10' },
      { k: 'omega', label: 'ω', type: 'expr', def: 'np.linspace(0.05, 2, 200)', span: 2 }],
    draw(r, out) { const p = plotDiv(); out.append(p); plot(p, [{ x: r.omega, y: r.abs, name: '|coh|' }, { x: r.omega, y: r.re, name: 'Re', line: { dash: 'dot' } }, { x: r.omega, y: r.im, name: 'Im', line: { dash: 'dot' } }], { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'Coherence' } }); } },
  { id: 'wind_spectrum', title: 'Turbulence spectra', desc: 'Kaimal (wind.kaimal_auto) and von Kármán (von_karman_auto) spectra and the Beaufort description (get_beaufort).', cls: 'wind',
    fields: [{ k: 'U', label: 'U [m/s]', type: 'expr', def: '20' }, { k: 'I', label: 'I = [Iu, Iv, Iw]', type: 'expr', def: '[0.136, 0.1, 0.072]' }, { k: 'Lx', label: 'Lx = [Lux, Lvx, Lwx]', type: 'expr', def: '[115, 30, 9.58]' },
      { k: 'A', label: 'A = [Au, Av, Aw]', type: 'expr', def: '[6.8, 9.4, 9.4]' }, { k: 'omega', label: 'ω', type: 'expr', def: 'np.logspace(-3, 1, 300)', span: 2 }],
    draw(r, out) {
      const p = plotDiv(); out.append(h('div.stats', {}, h('div.stat', {}, h('b', {}, r.beaufort ?? '–'), h('span', {}, 'Beaufort')),
        ...['u', 'v', 'w'].map((c, i) => h('div.stat', {}, h('b', {}, `${fmt(r.sigma[i] ** 2, 3)} / ${fmt(r.var_kaimal[i], 3)}`), h('span', {}, `σ²${c} target / ∫Kaimal`)))), p);
      plot(p, ['u', 'v', 'w'].flatMap((c, i) => [{ x: r.omega, y: r.kaimal[i], name: `S${c} Kaimal`, line: { color: SERIES[i] } }, { x: r.omega, y: r.vonkarman[i], name: `S${c} von Kármán`, line: { color: SERIES[i], dash: 'dot' } }]),
        { xaxis: { title: 'ω [rad/s]', type: 'log' }, yaxis: { title: 'S(ω)', type: 'log' } });
    } },
  { id: 'terrain', title: 'Terrain roughness', desc: 'Roughness factor profile c_r(z) for several roughness lengths (wind_code.terrain_roughness).', cls: 'wind',
    fields: [{ k: 'z0', label: 'z0 values [m]', type: 'expr', def: '[0.003, 0.01, 0.05, 0.3]', span: 2 }, { k: 'zmin', label: 'zmin [m]', type: 'expr', def: '1.0' }, { k: 'z', label: 'z', type: 'expr', def: 'np.linspace(0.5, 200, 200)', span: 2 }],
    draw(r, out) { const p = plotDiv(); out.append(p); plot(p, r.series.map(s => ({ x: s.cr, y: r.z, name: `z0 = ${s.z0}`, mode: 'lines' })), { xaxis: { title: 'c_r(z)' }, yaxis: { title: 'z [m]' } }); } },
  { id: 'ads', title: 'Aerodynamic derivatives', desc: 'Theodorsen flat plate (wind.flatplate_ads) or quasi-steady derivatives from load coefficients (wind.quasisteady_ads).', cls: 'wind',
    fields: [{ k: 'kind', label: 'Type', type: 'select', opts: [['flatplate', 'Flat plate'], ['quasisteady', 'Quasi-steady']], def: 'flatplate' },
      { k: 'B', label: 'B', type: 'expr', def: '20' }, { k: 'D', label: 'D', type: 'expr', def: '3' },
      { k: 'Cd', label: 'Cd', type: 'expr', def: '0.7' }, { k: 'dCd', label: "Cd'", type: 'expr', def: '0' }, { k: 'Cl', label: 'Cl', type: 'expr', def: '-0.4' },
      { k: 'dCl', label: "Cl'", type: 'expr', def: '4.0' }, { k: 'Cm', label: 'Cm', type: 'expr', def: '-0.07' }, { k: 'dCm', label: "Cm'", type: 'expr', def: '1.0' }, { k: 'vmax', label: 'v max', type: 'expr', def: '10' }],
    draw(r, out) {
      const p = plotDiv('tall'); out.append(p); const tr = [];
      ['P', 'H', 'A'].forEach((g, gi) => { for (let i = 1; i <= 6; i++) { const k = g + i; if (r.ads[k]) tr.push({ x: r.v, y: r.ads[k], name: k + '*', xaxis: 'x' + (gi + 1), yaxis: 'y' + (gi + 1), line: { color: SERIES[i - 1] }, legendgroup: 'n' + i, showlegend: gi === 0 }); } });
      plot(p, tr, { grid: { rows: 1, columns: 3, pattern: 'independent' }, yaxis: { title: 'P*' }, yaxis2: { title: 'H*' }, yaxis3: { title: 'A*' }, xaxis: { title: 'v' }, xaxis2: { title: 'v' }, xaxis3: { title: 'v' } });
    } },
  { id: 'response_stats', title: 'Spectral statistics', desc: 'Spectral moments, zero-crossing period, peak factor and expected maximum (wawi.random). Use a JONSWAP spectrum or any arrays from the session, e.g. last_psd[0,0] and model.results.omega.',
    fields: [{ k: 'source', label: 'Spectrum', type: 'select', opts: [['jonswap', 'JONSWAP from parameters'], ['expr', 'Arrays (expressions)']], def: 'jonswap' },
      { k: 'T', label: 'Duration T [s]', type: 'expr', def: '3600' },
      { k: 'Hs', label: 'Hs', type: 'expr', def: '2.0' }, { k: 'Tp', label: 'Tp', type: 'expr', def: '8.0' }, { k: 'gamma', label: 'γ', type: 'expr', def: '3.3' },
      { k: 'S', label: 'S (expression)', type: 'expr', ph: 'np.real(last_psd[0,0])', span: 2 }, { k: 'omega', label: 'ω (expression)', type: 'expr', ph: 'model.results.omega', span: 2 }],
    draw(r, out) {
      const p1 = plotDiv('short'), p2 = plotDiv('short');
      out.append(h('div.stats', {}, [['σ', r.std], ['m0', r.m0], ['m2', r.m2], ['m4', r.m4], ['ν0⁺ [Hz]', r.v0], ['Tz [s]', r.Tz], [`kp (T = ${r.T} s)`, r.peakfactor], ['Expected max', r.expmax]].map(([k, v]) => h('div.stat', {}, h('b', {}, fmt(v, 4)), h('span', {}, k)))), h('div.cols.even', {}, p1, p2));
      plot(p1, [{ x: r.omega, y: r.S, fill: 'tozeroy', line: { color: C.sea } }], { xaxis: { title: 'ω' }, yaxis: { title: 'S' }, showlegend: false });
      plot(p2, [{ x: r.curve.T, y: r.curve.kp, line: { color: C.ink } }], { xaxis: { title: 'T [s]', type: 'log' }, yaxis: { title: 'Peak factor' }, showlegend: false });
    } },
  { id: 'gumbel', title: 'Gumbel probability paper', desc: 'Plot sampled maxima on Gumbel paper (prob.gumbel_log).',
    fields: [{ k: 'umax', label: 'Sample of maxima (expression)', type: 'expr', def: 'np.random.gumbel(10, 2, 100)', span: 'full' }],
    draw(r, out) { const p = plotDiv(); out.append(p); plot(p, [{ x: r.umax, y: r.loglogF, mode: 'markers', marker: { color: C.ink } }], { xaxis: { title: 'Maximum' }, yaxis: { title: '−ln(−ln F)' }, showlegend: false }); } },
  { id: 'rayleigh', title: 'Rayleigh damping', desc: 'Mass- and stiffness-proportional coefficients giving damping ratio ξ at two frequencies (structural.rayleigh_damping_fit).',
    fields: [{ k: 'xi', label: 'ξ', type: 'expr', def: '0.005' }, { k: 'omega_1', label: 'ω1 [rad/s]', type: 'expr', def: '0.2' }, { k: 'omega_2', label: 'ω2 [rad/s]', type: 'expr', def: '3.0' }],
    draw(r, out) { const p = plotDiv(); out.append(h('div.stats', {}, h('div.stat', {}, h('b', {}, fmt(r.alpha, 5)), h('span', {}, 'α (mass)')), h('div.stat', {}, h('b', {}, fmt(r.beta, 5)), h('span', {}, 'β (stiffness)'))), p);
      plot(p, [{ x: r.omega, y: r.xi.map(v => v === null ? null : v * 100), line: { color: C.ink } }], { xaxis: { title: 'ω [rad/s]' }, yaxis: { title: 'ξ [%]' }, showlegend: false }); } },
  { id: 'fit_jonswap', title: 'Fit JONSWAP to a spectrum', desc: 'Least-squares fit of Hs, Tp and γ (signal.fit_jonswap) to spectral data from the session.', cls: 'sea',
    fields: [{ k: 'S', label: 'S (expression)', type: 'expr', span: 2, ph: 'S_measured' }, { k: 'omega', label: 'ω (expression)', type: 'expr', span: 2, ph: 'omega_measured' },
      { k: 'sigma', label: 'σ', type: 'expr', def: '[0.07, 0.09]' }, { k: 'initial_values', label: 'Initial values', type: 'expr', ph: "{'Tp': 8, 'gamma': 3}" }],
    draw(r, out) { const p = plotDiv(); out.append(h('div.stats', {}, [['Hs', r.Hs], ['Tp', r.Tp], ['γ', r.gamma]].map(([k, v]) => h('div.stat', {}, h('b', {}, fmt(v, 4)), h('span', {}, k)))), p);
      plot(p, [{ x: r.omega, y: r.S, name: 'data', mode: 'markers', marker: { size: 4, color: C.ink } }, { x: r.omega, y: r.fit, name: 'fit', line: { color: C.sea } }], { xaxis: { title: 'ω' }, yaxis: { title: 'S' } }); } },
  { id: 'time_sim', title: 'Simulate time series from a spectrum', desc: 'Gaussian multi-component time series from a cross-spectral density matrix S[n, n, nω] (time_domain.simulate_mdof).',
    fields: [{ k: 'S', label: 'S (expression)', type: 'expr', span: 2, ph: 'Sqq or a 1-D spectrum' }, { k: 'omega', label: 'ω (expression)', type: 'expr', span: 2, ph: 'omega_Sqq' },
      { k: 'fs', label: 'fs [Hz]', type: 'expr', def: '10' }, { k: 'tmax', label: 'Duration [s]', type: 'expr', def: '600' }, { k: 'seed', label: 'Seed', type: 'expr' }, { k: 'assign_to', label: 'Store as', def: 'q_tool' }],
    draw(r, out) { const p = plotDiv(); out.append(h('div.small.muted', {}, `Stored as ${r.vars.join(', ')}. Sample σ vs target σ: ` + r.std.slice(0, 6).map((s, i) => `${fmt(s, 3)}/${fmt(r.target_std[i], 3)}`).join(', ')), p);
      plot(p, r.q.slice(0, 6).map((q, i) => ({ x: r.t, y: q, name: `component ${i + 1}`, line: { width: 1 } })), { xaxis: { title: 't [s]' } }); } },
  { id: 'linear_drag', title: 'Linearised drag damping', desc: 'Equivalent linear damping ½ρ A Cd √(8/π) σ_u̇ (wave.linear_drag_damping).', cls: 'sea',
    fields: [{ k: 'Cd', label: 'Cd', type: 'expr', def: '[1.0, 1.0, 2.0]' }, { k: 'area', label: 'Area', type: 'expr', def: '[100, 200, 200]' }, { k: 'std_udot', label: 'σ of velocity', type: 'expr', def: '[0.1, 0.1, 0.05]' }, { k: 'rho', label: 'ρ', type: 'expr', def: '1025' }],
    draw(r, out) { out.append(h('div.stats', {}, r.c.map((v, i) => h('div.stat', {}, h('b', {}, fmt(v, 4)), h('span', {}, `c ${DOFS[i]} [Ns/m]`))))); } },
];

PAGES.tools = (root) => {
  const list = h('div'); const pane = h('div');
  const show = (t) => {
    $$('.item', list).forEach(i => i.classList.toggle('active', i.dataset.id === t.id));
    localStorage.setItem('wawi.tool', t.id);
    const f = makeForm('tool_' + t.id, t.fields); const out = h('div');
    clear(pane).append(card({ title: t.title, cls: t.cls }, t.desc, f.el,
      h('div.row', { style: { marginTop: '12px' } }, btn('Calculate', act(async () => { const r = await api('tools/run', { tool: t.id, params: f.get() }); clear(out); t.draw(r, out); }), 'primary')), h('hr.thin'), out));
  };
  TOOLDEFS.forEach(t => list.append(h('div.item', { 'data-id': t.id, onclick: () => show(t) }, h('span.nm', { style: { fontFamily: 'inherit' } }, t.title))));
  root.append(h('div.explorer', {}, card({}, null, h('div.tree', {}, list)), pane));
  show(TOOLDEFS.find(t => t.id === localStorage.getItem('wawi.tool')) || TOOLDEFS[0]);
};
