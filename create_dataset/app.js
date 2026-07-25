/* ============================================================
   Medical NER Labeler – app.js
   ============================================================ */

'use strict';

/* =================== State =================== */
const state = {
  rawText: '',        // original file content
  fileName: '',
  labels: [],         // [{text, position:[s,e], type, assertions}]
  selection: null,    // {text, start, end}
};

/* =================== DOM refs =================== */
const fileInput     = document.getElementById('file-input');
const dropZone      = document.getElementById('drop-zone');
const docContent    = document.getElementById('doc-content');
const docArea       = document.getElementById('doc-area');

const selText       = document.getElementById('sel-text');
const posStart      = document.getElementById('pos-start');
const posEnd        = document.getElementById('pos-end');
const selType       = document.getElementById('sel-type');
const assertionPills = document.querySelectorAll('.pill-btn');
const assertionHint = document.getElementById('assertion-hint');
const btnAdd        = document.getElementById('btn-add');

const labelList     = document.getElementById('label-list');
const labelEmpty    = document.getElementById('label-empty');
const btnExport     = document.getElementById('btn-export');
const btnClearAll   = document.getElementById('btn-clear-all');

const statCount     = document.getElementById('stat-count');
const statFile      = document.getElementById('stat-file');

const toast         = document.getElementById('toast');

/* =================== Types that support assertions =================== */
const ASSERTION_TYPES = new Set(['TRIỆU_CHỨNG', 'CHẨN_ĐOÁN', 'THUỐC']);

/* =================== Toast =================== */
let toastTimer;
function showToast(msg, type = 'info') {
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.className = 'toast'; }, 2800);
}

/* =================== File loading =================== */
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) loadFile(file);
});

// Drag-and-drop on the doc area
docArea.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
docArea.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
docArea.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.txt')) loadFile(file);
  else showToast('Chỉ hỗ trợ file .txt', 'error');
});

function loadFile(file) {
  const reader = new FileReader();
  reader.onload = evt => {
    state.rawText  = evt.target.result;
    state.fileName = file.name;
    state.labels   = [];
    renderDoc();
    renderLabelList();
    updateStats();
    statFile.textContent = file.name;
    showToast(`Đã tải: ${file.name}`, 'success');
  };
  reader.readAsText(file, 'UTF-8');
}

/* =================== Render document =================== */
function renderDoc() {
  dropZone.hidden    = true;
  docContent.hidden  = false;

  // Build list of highlighted ranges sorted by start
  const sorted = [...state.labels].sort((a, b) => a.position[0] - b.position[0]);

  let html   = '';
  let cursor = 0;
  const text = state.rawText;

  for (const lbl of sorted) {
    const [s, e] = lbl.position;
    if (s < cursor) continue; // overlap – skip
    // plain text before
    html += escHtml(text.slice(cursor, s));
    // highlighted span
    const cls = lbl.type.toLowerCase().replace(/_/g, '_');
    html += `<mark class="hl ${cls}" data-id="${lbl.id}" title="${lbl.type}">${escHtml(text.slice(s, e))}</mark>`;
    cursor = e;
  }
  html += escHtml(text.slice(cursor));

  docContent.innerHTML = html;

  // Click on highlight → scroll to card
  docContent.querySelectorAll('.hl').forEach(el => {
    el.addEventListener('click', e => {
      e.stopPropagation();
      const id = el.dataset.id;
      const card = document.querySelector(`.label-card[data-id="${id}"]`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        card.classList.add('flash');
        setTimeout(() => card.classList.remove('flash'), 1000);
      }
    });
  });
}

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/* =================== Text selection =================== */
docContent.addEventListener('mouseup', handleSelection);
docContent.addEventListener('touchend', handleSelection);

function handleSelection() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;

  const selectedStr = sel.toString();
  if (!selectedStr.trim()) return;

  // Calculate absolute character offset in rawText
  const range    = sel.getRangeAt(0);
  const preRange = document.createRange();
  preRange.setStart(docContent, 0);
  preRange.setEnd(range.startContainer, range.startOffset);
  const startOffset = preRange.toString().length;
  const endOffset   = startOffset + selectedStr.length;

  state.selection = { text: selectedStr, start: startOffset, end: endOffset };
  selText.value   = selectedStr;
  posStart.textContent = startOffset;
  posEnd.textContent   = endOffset;

  validateAddBtn();
}

/* =================== Type selector =================== */
selType.addEventListener('change', () => {
  const type = selType.value;
  const enabled = ASSERTION_TYPES.has(type);

  assertionPills.forEach(p => {
    p.disabled = !enabled;
    if (!enabled) p.classList.remove('active');
  });

  if (enabled) {
    assertionHint.textContent = 'Chọn 0 hoặc nhiều assertions.';
  } else if (type === '') {
    assertionHint.textContent = 'Chọn loại thực thể trước để kích hoạt.';
  } else {
    assertionHint.textContent = 'Loại này không có assertions.';
  }

  validateAddBtn();
});

/* =================== Assertion pills =================== */
assertionPills.forEach(btn => {
  btn.addEventListener('click', () => {
    if (!btn.disabled) btn.classList.toggle('active');
  });
});

function getSelectedAssertions() {
  return [...document.querySelectorAll('.pill-btn.active')].map(b => b.dataset.value);
}

function resetAssertions() {
  assertionPills.forEach(p => {
    p.classList.remove('active');
    p.disabled = true;
  });
  assertionHint.textContent = 'Chọn loại thực thể trước để kích hoạt.';
}

/* =================== Validate / enable Add button =================== */
function validateAddBtn() {
  const ok = state.selection && state.selection.text.trim() && selType.value;
  btnAdd.disabled = !ok;
}

/* =================== Add label =================== */
btnAdd.addEventListener('click', addLabel);

function addLabel() {
  if (!state.selection || !selType.value) return;

  const assertions = ASSERTION_TYPES.has(selType.value) ? getSelectedAssertions() : [];

  const label = {
    id:         crypto.randomUUID(),
    text:       state.selection.text,
    position:   [state.selection.start, state.selection.end],
    type:       selType.value,
    assertions,
  };

  state.labels.push(label);
  state.labels.sort((a, b) => a.position[0] - b.position[0]);

  // Reset form
  state.selection = null;
  selText.value = '';
  posStart.textContent = '–';
  posEnd.textContent = '–';
  selType.value = '';
  resetAssertions();
  btnAdd.disabled = true;
  window.getSelection()?.removeAllRanges();

  renderDoc();
  renderLabelList();
  updateStats();
  showToast('Đã thêm nhãn!', 'success');
}

/* =================== Render label list =================== */
function renderLabelList() {
  labelList.innerHTML = '';

  if (state.labels.length === 0) {
    labelList.appendChild(labelEmpty);
    labelEmpty.hidden = false;
    return;
  }
  labelEmpty.hidden = true;

  state.labels.forEach(lbl => {
    const li = document.createElement('li');
    li.className = `label-card ${lbl.type.toLowerCase().replace(/_/g, '_')}`;
    li.dataset.id = lbl.id;

    const assertionBadges = lbl.assertions.map(a =>
      `<span class="label-meta-assertion">${a}</span>`
    ).join('');

    li.innerHTML = `
      <div class="label-card-dot"></div>
      <div class="label-card-body">
        <div class="label-card-text" title="${escHtml(lbl.text)}">${escHtml(lbl.text)}</div>
        <div class="label-card-meta">
          <span class="label-meta-type">${lbl.type}</span>
          <span class="label-meta-pos">[${lbl.position[0]}, ${lbl.position[1]}]</span>
          ${assertionBadges}
        </div>
      </div>
      <button class="label-card-del" data-id="${lbl.id}" title="Xóa nhãn">×</button>
    `;

    li.querySelector('.label-card-del').addEventListener('click', e => {
      e.stopPropagation();
      deleteLabel(lbl.id);
    });

    // Click card → scroll to highlight
    li.addEventListener('click', () => {
      const hl = docContent.querySelector(`.hl[data-id="${lbl.id}"]`);
      if (hl) {
        hl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        hl.style.outline = '2px solid var(--accent)';
        setTimeout(() => { hl.style.outline = ''; }, 1000);
      }
    });

    labelList.appendChild(li);
  });
}

function deleteLabel(id) {
  state.labels = state.labels.filter(l => l.id !== id);
  renderDoc();
  renderLabelList();
  updateStats();
  showToast('Đã xóa nhãn.', 'warn');
}

/* =================== Clear all =================== */
btnClearAll.addEventListener('click', () => {
  if (state.labels.length === 0) return;
  if (!confirm('Xóa tất cả nhãn? Thao tác này không thể hoàn tác.')) return;
  state.labels = [];
  renderDoc();
  renderLabelList();
  updateStats();
  showToast('Đã xóa tất cả nhãn.', 'warn');
});

/* =================== Export JSON =================== */
btnExport.addEventListener('click', exportJson);

function exportJson() {
  if (!state.rawText) {
    showToast('Chưa tải file nào!', 'error');
    return;
  }

  // Build output: omit internal `id` field, keep only spec fields
  const output = state.labels.map(({ text, position, type, assertions }) => ({
    text,
    type,
    assertions,
    position,
  }));

  const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');

  // Derive output filename from input filename
  const base = state.fileName.replace(/\.txt$/i, '');
  a.href     = url;
  a.download = `${base}_labels.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showToast(`Xuất ${output.length} nhãn → ${a.download}`, 'success');
}

/* =================== Stats =================== */
function updateStats() {
  statCount.textContent = state.labels.length;
}

/* =================== Resize handle =================== */
const resizeHandle = document.getElementById('resize-handle');
const panelDoc     = document.getElementById('panel-doc');
const panelAnno    = document.getElementById('panel-anno');
const appMain      = document.querySelector('.app-main');

let isResizing = false;

resizeHandle.addEventListener('mousedown', e => {
  isResizing = true;
  resizeHandle.classList.add('dragging');
  document.body.style.userSelect = 'none';
  document.body.style.cursor = 'col-resize';
});

document.addEventListener('mousemove', e => {
  if (!isResizing) return;
  const mainRect  = appMain.getBoundingClientRect();
  const totalW    = mainRect.width - 5; // minus handle width
  let   docW      = e.clientX - mainRect.left;
  docW            = Math.max(300, Math.min(docW, totalW - 300));
  const annoW     = totalW - docW;
  panelDoc.style.flex  = `0 0 ${docW}px`;
  panelAnno.style.flex = `0 0 ${annoW}px`;
});

document.addEventListener('mouseup', () => {
  if (!isResizing) return;
  isResizing = false;
  resizeHandle.classList.remove('dragging');
  document.body.style.userSelect = '';
  document.body.style.cursor     = '';
});

/* =================== Keyboard shortcut =================== */
// Enter → Add label (when button is active)
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    if (!btnAdd.disabled) addLabel();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    exportJson();
  }
});

/* =================== Init assertion pills disabled =================== */
assertionPills.forEach(p => { p.disabled = true; });

/* =================== Forward wheel → doc-content =================== */
// panel-doc has overflow:hidden which can swallow wheel events before they
// reach doc-content. Forward them explicitly so scrolling always works.
panelDoc.addEventListener('wheel', e => {
  if (!docContent.hidden && docContent.scrollHeight > docContent.clientHeight) {
    e.preventDefault();
    docContent.scrollTop += e.deltaY;
  }
}, { passive: false });
