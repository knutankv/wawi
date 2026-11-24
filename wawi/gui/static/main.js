/* WAWI GUI – bootstrapping and navigation */
'use strict';

const TITLES = { model: 'Model', env: 'Environment', analysis: 'Analysis', results: 'Results', tools: 'Calculators', explorer: 'API explorer', console: 'Variables & console' };
const built = {};
function showPage(id) {
  if (!TITLES[id]) id = 'model';
  $$('nav.side a').forEach(a => a.classList.toggle('active', a.dataset.page === id));
  $$('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + id));
  $('#pageTitle').textContent = TITLES[id];
  if (!built[id]) {
    built[id] = true;
    try { PAGES[id]($('#page-' + id)); } catch (e) { showError(e); }
  } else {
    const active = $('#page-' + id + ' .sub.active');
    if (active && active._refresh) active._refresh();
  }
  localStorage.setItem('wawi.page', id);
  if (location.hash !== '#' + id) history.replaceState(null, '', location.pathname + location.search + '#' + id);
  window.dispatchEvent(new Event('resize'));
}
(async function init() {
  for (const id of Object.keys(TITLES)) $('#pages').append(h('div.page', { id: 'page-' + id }));
  $$('nav.side a').forEach(a => a.addEventListener('click', () => showPage(a.dataset.page)));
  if (!TOKEN) {
    $('#pages').prepend(h('div.page.active', {}, card('Session token missing', 'Open the address printed in the terminal where you started the GUI (it ends with ?token=…).')));
    return;
  }
  await refreshState();
  showPage((location.hash || '').slice(1) || localStorage.getItem('wawi.page') || 'model');
  setTimeout(() => { if (!window.Plotly) toast('Plotly could not be loaded, so charts are unavailable. Install the "plotly" Python package or allow internet access.', { error: true }); }, 4000);
  setInterval(() => { if (!document.hidden) Jobs.schedule(50); }, 5000);
})();
