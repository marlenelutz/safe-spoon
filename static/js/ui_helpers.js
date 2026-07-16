// Pure, stateless helpers used by front.html's Component class.
//
// front.html's dc-runtime template engine (support.js) locates its reactive
// component with `document.querySelector('script[data-dc-script]')` — a
// single-match query, not querySelectorAll — so the `class Component extends
// DCLogic` body cannot itself be split across files. What CAN move out is
// logic that doesn't touch `this.state`/`this.setState`: style-string
// builders and rubric data-shape helpers. Component methods of the same
// name (e.g. `_tab`) are kept as one-line delegations to these, so the many
// existing call sites in renderVals() don't need to change.
(function (global) {
  'use strict';

  function tabStyle(active) {
    return 'border:none;cursor:pointer;font-family:inherit;font-size:12px;font-weight:' + (active ? '600' : '500') + ';padding:5px 13px;border-radius:7px;transition:all .15s;' +
      (active ? 'background:var(--panel,#f7f8f6);color:var(--accent,#0b758e);box-shadow:0 1px 3px rgba(0,0,0,.07);' : 'background:transparent;color:var(--ink3,#8a9389);');
  }

  function catTabStyle(active, color) {
    return 'display:flex;align-items:center;gap:7px;border:none;cursor:pointer;font-family:inherit;font-size:12.5px;font-weight:' + (active ? '600' : '400') + ';padding:10px 13px;background:transparent;transition:all .15s;border-bottom:2px solid ' +
      (active ? 'var(--accent,#0b758e)' : 'transparent') + ';color:' + (active ? 'var(--ink,#1a1e1b)' : 'var(--ink3,#8a9389)') + ';';
  }

  function statusChipStyle(status) {
    const map = {
      done: 'background:var(--soft,#d9eff3);color:var(--accent,#0b758e);',
      draft: 'background:#fdf3e3;color:#b9762f;',
      empty: 'background:var(--track,#eceef2);color:var(--ink3,#98a0ab);',
    };
    return 'font-size:10.5px;font-weight:700;padding:2px 9px;border-radius:99px;flex-shrink:0;' + map[status];
  }

  function statusLabel(status) {
    return { done: 'Guideline ready', draft: 'Draft', empty: 'Needs guideline' }[status];
  }

  function statusDotStyle(status) {
    const c = { done: 'var(--accent,#0b758e)', draft: '#c48a2a', empty: '#c4c8c0' }[status];
    return 'width:7px;height:7px;border-radius:50%;flex-shrink:0;background:' + c + ';';
  }

  function labelCheckStyle(on) {
    return 'display:flex;align-items:center;gap:5px;font-size:11px;font-weight:' + (on ? '600' : '400') + ';cursor:pointer;user-select:none;padding:3px 8px;border-radius:6px;transition:all .12s;border:1px solid ' +
      (on ? 'var(--accent,#0b758e)' : 'var(--border,#dde0da)') + ';background:' + (on ? 'var(--soft,#d9eff3)' : 'transparent') + ';color:' + (on ? 'var(--accent,#0b758e)' : 'var(--ink3,#8a9389)') + ';';
  }

  function softColor(hex) {
    try {
      const n = parseInt(hex.slice(1), 16);
      const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
      const mix = (c) => Math.round(c + (255 - c) * 0.87);
      return 'rgb(' + mix(r) + ',' + mix(g) + ',' + mix(b) + ')';
    } catch (e) { return '#d9eff3'; }
  }

  function rgbaColor(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }

  // ---------- rubric data-shape helpers ----------

  function emptyCell() {
    return { expected_behavior: '', risk_signals: '', is_override: false, inherited_from_cell_id: null };
  }

  // LLM-suggested cells occasionally come back with risk_signals (or even
  // expected_behavior) as a JSON array instead of the prose string the
  // prompt asks for — coerce to a display string so it can't reach the
  // textarea (or later, the sqlite bind) as a non-string value.
  function _cellText(v) {
    if (v == null) return '';
    if (Array.isArray(v)) return v.join(', ');
    return String(v);
  }

  // Ensures every criterion has one cell per current risk profile (filling
  // in blanks for profiles added after a rubric was first drafted).
  function normalizeCriteria(criteria, profiles) {
    return (criteria || []).map(c => {
      const cells = {};
      for (const p of profiles) {
        const raw = c.cells && c.cells[p.id];
        cells[p.id] = raw
          ? {
              ...emptyCell(), ...raw,
              expected_behavior: _cellText(raw.expected_behavior),
              risk_signals: _cellText(raw.risk_signals),
            }
          : emptyCell();
      }
      return { title: c.title || '', description: c.description || '', cells };
    });
  }

  // Clones another rubric's criteria (a parent or a similar-confirmed match)
  // into a fresh draft: every cell is marked NOT overridden and linked back
  // to the source cell's id, so the editor can show "inherited from X"
  // until the annotator actually edits a cell (updateCellField then flips
  // is_override to true — see front.html).
  function cloneCriteriaForInherit(sourceCriteria) {
    return (sourceCriteria || []).map(c => {
      const cells = {};
      for (const [profileId, cell] of Object.entries(c.cells || {})) {
        cells[profileId] = {
          expected_behavior: _cellText(cell.expected_behavior),
          risk_signals: _cellText(cell.risk_signals),
          is_override: false,
          inherited_from_cell_id: cell.id != null ? cell.id : (cell.inherited_from_cell_id || null),
        };
      }
      return { title: c.title || '', description: c.description || '', cells };
    });
  }

  global.SafeSpoonHelpers = {
    tabStyle, catTabStyle, statusChipStyle, statusLabel, statusDotStyle, labelCheckStyle,
    softColor, rgbaColor,
    emptyCell, normalizeCriteria, cloneCriteriaForInherit,
  };
})(window);
