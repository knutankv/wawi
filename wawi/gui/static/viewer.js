/* WAWI GUI – interactive 3D viewer for model geometry, mode shapes and static deformation */
'use strict';

class Viewer3D {
  constructor({ modes = true } = {}) {
    this.g = null; this.shape = null; this.phase = 0; this.anim = null; this.withModes = modes;
    this.scaleIn = h('input', { type: 'range', min: 0, max: 100, value: 50 });
    this.srcSel = h('select', {}, h('option', { value: 'dry' }, 'Dry modes'), h('option', { value: 'wet' }, 'Wet / in-wind modes'), h('option', { value: 'static' }, 'Static displacement'), h('option', { value: 'flutter', title: 'Critical mode shape at each speed step of the last flutter analysis ("Mode" = speed step)' }, 'Flutter (critical mode)'));
    this.modeIn = h('input', { type: 'number', min: 1, value: 1, style: { width: '64px' } });
    this.chkStates = h('input', { type: 'checkbox', checked: true });
    this.chkNodes = h('input', { type: 'checkbox' });
    this.chkWater = h('input', { type: 'checkbox', checked: true });
    this.chkTrue = h('input', { type: 'checkbox' });
    this.chkMesh = h('input', { type: 'checkbox', checked: true });
    this.meshLabel = h('label', { title: 'Pontoon geometry from the pontoon types (STL meshes)', style: { display: 'none' } }, this.chkMesh, 'Pontoons 3D');
    this.animBtn = btn('Animate', () => this.toggleAnim(), 'small');
    this.plotEl = h('div.plot');
    this.period = 2.5;  // animation period [s]
    // Plotly's 3D redraw resets the camera controller, which cuts off a rotation/zoom in progress.
    // So animation frames pause while the mouse button is held on the plot and briefly after scrolling.
    this._hold = 0;
    this.plotEl.addEventListener('pointerdown', () => { this._hold = Infinity; }, true);
    window.addEventListener('pointerup', () => { if (this._hold === Infinity) this._hold = performance.now() + 150; }, true);
    this.plotEl.addEventListener('wheel', () => { if (this._hold !== Infinity) this._hold = performance.now() + 400; }, { passive: true, capture: true });
    this.info = h('div.modeinfo', { style: { display: 'none' } });
    this.legend = h('div.legend');
    const bar = h('div.bar', {},
      modes ? [h('label', {}, 'Show', this.srcSel), h('label', {}, 'Mode', this.modeIn),
        btn('◀', () => this.step(-1), 'small ghost'), btn('▶', () => this.step(1), 'small ghost'),
        h('label', {}, 'Scale', this.scaleIn), this.animBtn] : null,
      h('span.spacer'),
      h('label', {}, this.chkStates, 'Wind/wave'), h('label', {}, this.chkWater, 'Water'), this.meshLabel, h('label', {}, this.chkNodes, 'Nodes'), h('label', { title: 'Untick to exaggerate thin dimensions for readability' }, this.chkTrue, 'True proportions'),
      btn('Reload', act(() => this.load()), 'small ghost'));
    this.el = h('div.viewer', {}, bar, this.plotEl, this.info, this.legend);
    const redraw = () => this.draw();
    [this.chkStates, this.chkNodes, this.chkWater].forEach(c => c.onchange = redraw);
    this.chkMesh.onchange = redraw;
    this.chkTrue.onchange = () => this.draw(true);
    this.scaleIn.oninput = redraw;
    this.srcSel.onchange = act(() => this.loadShape());
    this.modeIn.onchange = act(() => this.loadShape());
  }
  async load() {
    this.g = await api('model/geometry');
    this.shape = null; this.info.style.display = 'none';
    this.draw(true);
    // Show the selected mode right away (mode 1 by default); keep the plain geometry if none is available
    if (this.withModes) await this.loadShape().catch(() => {});
  }
  pontoonSetup() {
    // Per geometry: node index of each pontoon, and its mesh rotated into global orientation (cached)
    const g = this.g;
    if (this._pset && this._pset.g === g) return this._pset;
    const xyz0 = g.nodes.xyz;
    const node = (g.pontoons || []).map(p => {
      let best = -1, bd = Infinity;
      xyz0.forEach((q, i) => { const d = (q[0] - p.xyz[0]) ** 2 + (q[1] - p.xyz[1]) ** 2 + (q[2] - p.xyz[2]) ** 2; if (d < bd) { bd = d; best = i; } });
      return bd < 1e-6 ? best : -1;
    });
    const meshes = [];
    (g.pontoon_place || []).forEach((pl, i) => {
      const M = pl && g.pontoon_meshes[pl.type];
      if (!M || !M.xyz) return;
      const R = pl.R;
      const rv = M.xyz.map(v => [0, 1, 2].map(r => R[r][0] * v[0] + R[r][1] * v[1] + R[r][2] * v[2]));
      const lo = [0, 1, 2].map(k => Math.min(...rv.map(v => v[k]))), hi = [0, 1, 2].map(k => Math.max(...rv.map(v => v[k])));
      meshes.push({ i, rv, ijk: M.ijk, pos: pl.xyz, lo, hi });
    });
    this.meshLabel.style.display = meshes.length ? '' : 'none';
    this._pset = { g, node, meshes };
    return this._pset;
  }
  nodeAmp(i) {
    // Displacement amplitude of node i at the current scale (used for fixed axis ranges)
    if (!this.shape || i < 0) return [0, 0, 0];
    const s = Math.abs(this.scale()); const d = this.shape.disp; const dr = re(d), di = im(d);
    return [0, 1, 2].map(k => s * Math.hypot(dr[i][k], di ? di[i][k] : 0));
  }
  step(d) { this.modeIn.value = Math.max(1, (+this.modeIn.value || 1) + d); this.loadShape().catch(showError); }
  async showMode(source, mode) { this.srcSel.value = source; this.modeIn.value = mode + 1; await this.loadShape(); }
  async loadShape() {
    if (!this.g) this.g = await api('model/geometry');
    this.shape = await api('model/shape', { source: this.srcSel.value, mode: String((+this.modeIn.value || 1) - 1) });
    if (this.shape.n_modes) this.modeIn.max = this.shape.n_modes;
    const i = this.shape.info;
    this.info.style.display = 'block';
    clear(this.info).append(h('b', {}, i.label), i.fn !== undefined ? h('div', {}, `f = ${fmt(i.fn)} Hz · T = ${fmt(i.Tn)} s`) : null,
      i.xi !== undefined && i.xi !== null ? h('div', {}, `ξ = ${fmt(i.xi * 100, 3)} %`) : null);
    this.phase = 0;
    this.draw();  // keep the current view (a new model resets it in load())
  }
  toggleAnim() {
    if (this.anim) { cancelAnimationFrame(this.anim); this.anim = null; this.animBtn.textContent = 'Animate'; this.phase = 0; this.draw(); return; }
    if (!this.shape) return;
    this.animBtn.textContent = 'Stop';
    // Time-based phase on requestAnimationFrame: smooth at any frame rate; a frame is skipped while
    // the previous one is still rendering, so slow models drop frames instead of lagging behind.
    let t0 = performance.now() - (this.phase / (2 * Math.PI)) * this.period * 1000;
    let busy = false, last = performance.now();
    const tick = (now) => {
      if (!this.anim) return;
      this.anim = requestAnimationFrame(tick);
      const held = this._hold === Infinity || now < this._hold;
      if (held) t0 += now - last;  // paused while the user rotates/zooms; resume where it stopped
      last = now;
      if (busy || held) return;
      this.phase = 2 * Math.PI * ((now - t0) / 1000) / this.period;
      busy = true;
      let p;
      try { p = this.draw(false, true); } catch (e) { console.error('WAWI viewer: animation frame failed', e); }
      Promise.resolve(p).catch(e => console.error('WAWI viewer: animation frame failed', e)).finally(() => { busy = false; });
    };
    this.anim = requestAnimationFrame(tick);
  }
  liveScene() {
    // Plotly's internal 3D scene object (null before the first plot)
    const fl = this.plotEl._fullLayout;
    return (fl && fl.scene && fl.scene._scene) || null;
  }
  liveCamera() {
    // Current camera of the 3D scene, including a rotation/zoom still in progress (not yet in layout)
    const sc = this.liveScene();
    try { return sc ? sc.getCamera() : null; } catch (e) { return null; }
  }
  liveAspect() {
    // Aspect ratio as the 3D scene currently has it. Passing it back unchanged keeps Plotly from
    // registering an aspect-ratio "GUI edit" every frame (console warnings, slower frames).
    const sc = this.liveScene();
    try {
      const a = sc && sc.glplot && sc.glplot.getAspectratio();
      return a ? { x: a.x, y: a.y, z: a.z } : null;
    } catch (e) { return null; }
  }
  scale() {
    if (!this.shape) return 0;
    const auto = this.shape.info.auto_scale || 1;
    return auto * Math.pow(10, (this.scaleIn.value - 50) / 25);  // 0.01x … 100x of auto
  }
  deformed(xyz) {
    const s = this.scale();
    const d = this.shape.disp; const dr = re(d), di = im(d);
    const c = Math.cos(this.phase), sn = Math.sin(this.phase);
    return xyz.map((p, i) => [0, 1, 2].map(k => p[k] + s * (dr[i][k] * c - (di ? di[i][k] * sn : 0))));
  }
  lines(xyz, conn, mask) {
    const x = [], y = [], z = [];
    conn.forEach((e, i) => {
      if (mask && !mask[i]) return;
      const a = xyz[e[0]], b = xyz[e[1]];
      x.push(a[0], b[0], null); y.push(a[1], b[1], null); z.push(a[2], b[2], null);
    });
    return { x, y, z };
  }
  envelope() {
    // Bounding box of the structure over a full oscillation cycle (independent of the animation phase)
    const xyz0 = this.g.nodes.xyz;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    const s = this.scale();
    const d = this.shape ? this.shape.disp : null; const dr = d ? re(d) : null, di = d ? im(d) : null;
    xyz0.forEach((p, i) => [0, 1, 2].forEach(k => {
      const amp = dr ? Math.abs(s) * Math.hypot(dr[i][k], di ? di[i][k] : 0) : 0;
      lo[k] = Math.min(lo[k], p[k] - amp); hi[k] = Math.max(hi[k], p[k] + amp);
    }));
    return { lo, hi };
  }
  ranges(extraPts) {
    // Fixed axis ranges: structure envelope + water plane + wind/wave arrows, with a small margin
    const { lo, hi } = this.envelope();
    extraPts.forEach(p => [0, 1, 2].forEach(k => { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }));
    const L = Math.max(...[0, 1, 2].map(k => hi[k] - lo[k])) || 1;
    return [0, 1, 2].map(k => {
      const m = 0.04 * L + 0.02 * (hi[k] - lo[k]);
      const a = lo[k] - m, b = hi[k] + m;
      return b - a > 1e-9 * L ? [a, b] : [a - 0.05 * L, b + 0.05 * L];
    });
  }
  aspect(rng) {
    const e = rng.map(r => r[1] - r[0]);
    const mx = Math.max(...e) || 1;
    const r = e.map(v => v / mx);
    if (this.chkTrue.checked) return { x: r[0], y: r[1], z: r[2] };
    return { x: Math.max(r[0], 0.3), y: Math.max(r[1], 0.3), z: Math.max(r[2], 0.18) };
  }
  draw(reset = false, quick = false) {
    const g = this.g; if (!g) return;
    const xyz0 = g.nodes.xyz;
    const conn = g.elements.conn;
    const aeroSet = new Set(Object.values(g.aero_groups || {}).flat());
    const isAero = g.elements.label.map(l => aeroSet.has(l));
    const hasShape = !!this.shape;
    const xyz = hasShape ? this.deformed(xyz0) : xyz0;
    const traces = [];
    const nodeIndex = {}; g.nodes.label.forEach((l, i) => { nodeIndex[String(l)] = i; });
    if (hasShape) {
      const u = this.lines(xyz0, conn);
      traces.push({ type: 'scatter3d', mode: 'lines', ...u, line: { color: '#c3cfd6', width: 2 }, name: 'Undeformed', hoverinfo: 'skip' });
    }
    const a = this.lines(xyz, conn, isAero), o = this.lines(xyz, conn, isAero.map(v => !v));
    traces.push({ type: 'scatter3d', mode: 'lines', ...o, line: { color: C.ink2, width: hasShape ? 5 : 4 }, name: 'Structure', hoverinfo: 'skip' });
    if (a.x.length) traces.push({ type: 'scatter3d', mode: 'lines', ...a, line: { color: C.wind, width: hasShape ? 6 : 5 }, name: 'Aero sections', hoverinfo: 'skip' });
    if (this.chkNodes.checked) {
      traces.push({ type: 'scatter3d', mode: 'markers', x: xyz.map(p => p[0]), y: xyz.map(p => p[1]), z: xyz.map(p => p[2]),
        marker: { size: 2.5, color: C.ink }, text: g.nodes.label.map(l => 'Node ' + l), hoverinfo: 'text', name: 'Nodes' });
    }
    const extraPts = [];
    if (g.pontoons) {
      const ps = this.pontoonSetup();
      // pontoons sit on nodes: displaced position = displaced node
      const pp = g.pontoons.map((p, i) => ps.node[i] >= 0 ? xyz[ps.node[i]] : p.xyz);
      const showMesh = this.chkMesh.checked && ps.meshes.length;
      if (showMesh) {
        const X = [], Y = [], Z = [], I = [], J = [], K = [];
        ps.meshes.forEach(M => {
          const o = X.length, t = pp[M.i];
          M.rv.forEach(v => { X.push(v[0] + t[0]); Y.push(v[1] + t[1]); Z.push(v[2] + t[2]); });
          M.ijk.forEach(f => { I.push(f[0] + o); J.push(f[1] + o); K.push(f[2] + o); });
          const a = this.nodeAmp(ps.node[M.i]);
          extraPts.push([0, 1, 2].map(k => M.pos[k] + M.lo[k] - a[k]), [0, 1, 2].map(k => M.pos[k] + M.hi[k] + a[k]));
        });
        traces.push({ type: 'mesh3d', x: X, y: Y, z: Z, i: I, j: J, k: K, color: '#7f97a4', flatshading: true,
          lighting: { ambient: 0.55, diffuse: 0.75, specular: 0.15, roughness: 0.6 }, lightposition: { x: 1e5, y: -2e5, z: 3e5 },
          hoverinfo: 'skip', name: 'Pontoon geometry', showscale: false });
      }
      traces.push({ type: 'scatter3d', mode: 'markers', x: pp.map(p => p[0]), y: pp.map(p => p[1]), z: pp.map(p => p[2]),
        marker: { size: showMesh ? 2 : 3.5, color: C.sea, symbol: 'diamond' },
        text: g.pontoons.map(p => `${p.label}<br>type ${p.type}<br>rotation ${(p.rotation * 180 / Math.PI).toFixed(1)}°`),
        hoverinfo: 'text', name: 'Pontoons' });
    }
    const ext = g.extent || [1, 1, 1];
    const L = Math.max(...ext) || 1;
    const mid = [0, 1, 2].map(k => { const v = xyz0.map(p => p[k]); return (Math.min(...v) + Math.max(...v)) / 2; });
    const zw = g.waterlevel ?? 0;
    if (this.chkWater.checked && g.pontoons) {
      const pad = 0.15 * L;
      const xs = xyz0.map(p => p[0]), ys = xyz0.map(p => p[1]);
      const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
      extraPts.push([x0, y0, zw], [x1, y1, zw]);
      traces.push({ type: 'mesh3d', x: [x0, x1, x1, x0], y: [y0, y0, y1, y1], z: [zw, zw, zw, zw], i: [0, 0], j: [1, 2], k: [2, 3],
        color: '#7cc3cf', opacity: 0.16, hoverinfo: 'skip', name: 'Water', showscale: false });
    }
    const legend = clear(this.legend);
    if (this.chkStates.checked) {
      const arrow = (origin, vec, len, color, name, text) => {
        const tip = [origin[0] + vec[0] * len, origin[1] + vec[1] * len, origin[2] + (vec[2] || 0) * len];
        const c = 0.4 * len;  // room for the cone head around the tip
        extraPts.push(origin, [tip[0] - c, tip[1] - c, tip[2] - c], [tip[0] + c, tip[1] + c, tip[2] + c]);
        traces.push({ type: 'scatter3d', mode: 'lines', x: [origin[0], tip[0]], y: [origin[1], tip[1]], z: [origin[2], tip[2]],
          line: { color, width: 7 }, name, hoverinfo: 'text', text: [text, text] });
        traces.push({ type: 'cone', x: [tip[0]], y: [tip[1]], z: [tip[2]], u: [vec[0]], v: [vec[1]], w: [vec[2] || 0], anchor: 'tail',
          sizemode: 'absolute', sizeref: len * 0.35, colorscale: [[0, color], [1, color]], showscale: false, hoverinfo: 'skip', name });
      };
      if (g.wave && g.wave.homogeneous) {
        const v = [Math.cos(g.wave.theta0), Math.sin(g.wave.theta0), 0];
        const len = 0.12 * L;
        const o = [mid[0] - v[0] * len * 1.15, mid[1] - v[1] * len * 1.15, zw];
        arrow(o, v, len, C.sea, 'Waves', `Waves: θ0 = ${(g.wave.theta0 * 180 / Math.PI).toFixed(1)}°, Hs = ${fmt(g.wave.Hs, 3)} m, Tp = ${fmt(g.wave.Tp, 3)} s`);
        legend.append(h('span', {}, h('i', { style: { background: C.sea } }), `Waves θ0 = ${(g.wave.theta0 * 180 / Math.PI).toFixed(0)}°, Hs ${fmt(g.wave.Hs, 3)} m`));
      } else if (g.wave && g.wave.per_pontoon && g.pontoons) {
        g.pontoons.forEach((p, i) => {
          const w = g.wave.per_pontoon[i]; if (!w || w.theta0 === null) return;
          const v = [Math.cos(w.theta0), Math.sin(w.theta0), 0];
          arrow([p.xyz[0] - v[0] * 0.05 * L, p.xyz[1] - v[1] * 0.05 * L, zw], v, 0.05 * L, C.sea, 'Waves', `${p.label}: θ0 ${(w.theta0 * 180 / Math.PI).toFixed(1)}°, Hs ${fmt(w.Hs, 3)} m, Tp ${fmt(w.Tp, 3)} s`);
        });
        legend.append(h('span', {}, h('i', { style: { background: C.sea } }), 'Inhomogeneous sea state (per pontoon)'));
      }
      if (g.wind) {
        const v = g.wind.vec; const len = 0.12 * L;
        const zt = Math.max(...xyz0.map(p => p[2])) + 0.05 * L;
        const o = [mid[0] - v[0] * len * 1.15, mid[1] - v[1] * len * 1.15, zt];
        arrow(o, v, len, C.wind, 'Wind', `Wind: U = ${fmt(g.wind.U0, 3)} m/s from ${fmt(g.wind.direction, 4)}°`);
        legend.append(h('span', {}, h('i', { style: { background: C.wind } }), `Wind U = ${fmt(g.wind.U0, 3)} m/s from ${fmt(g.wind.direction, 4)}°`));
      }
    }
    if (quick && this.plotEl.data) {
      // Animation frame: ranges depend only on the deformation envelope, so the layout stays as it is.
      // Every redraw re-applies layout.scene.camera, so pass the live camera (including a rotation/zoom
      // still in progress) in a new layout object: the user can rotate while it animates.
      const old = this.plotEl.layout;
      if (!old.scene) return Plotly.react(this.plotEl, traces, old);
      const scene = { ...old.scene };
      const cam = this.liveCamera(), asp = this.liveAspect();
      if (cam) scene.camera = cam;
      if (asp) scene.aspectratio = asp;
      return Plotly.react(this.plotEl, traces, { ...old, scene });
    }
    const rng = this.ranges(extraPts);
    const ax = (title, nticks, r) => ({ title, nticks, range: r, autorange: false, backgroundcolor: '#f7fafb', gridcolor: C.grid, showbackground: true });
    // Camera: keep the user's current view (live camera) unless the view is reset
    const layout = {
      margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)',
      scene: { aspectmode: 'manual', aspectratio: this.aspect(rng),
        camera: (!reset && this.liveCamera()) || { eye: { x: 0.12, y: -0.72, z: 0.42 } },
        xaxis: ax('x', 6, rng[0]), yaxis: ax('y', 5, rng[1]), zaxis: ax('z', 4, rng[2]) },
    };
    if (!window.Plotly) { clear(this.plotEl).append(emptyBox('Plotly not available')); return; }
    Plotly.react(this.plotEl, traces, fixTitles(layout), { responsive: true, displaylogo: false });
  }
}