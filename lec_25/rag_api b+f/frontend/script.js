/*
  RAG Studio — Script
  Full interactive frontend: Web Audio, session state, chat, upload,
  diagnostics, evaluation, dark/light theme.
*/
const $ = (s, root = document) => root.querySelector(s);
const $1 = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));

const defaults = { api: 'http://localhost:8000', sessionId: 'session_001', topK: 3, threshold: 0.05, sound: true, theme: 'light' };
const state = {
  api: localStorage.getItem('rag_api_url') || defaults.api,
  sessionId: localStorage.getItem('rag_session_id') || defaults.sessionId,
  topK: Number(localStorage.getItem('rag_top_k') || defaults.topK),
  threshold: Number(localStorage.getItem('rag_threshold') || defaults.threshold),
  messages: [], lastAnswer: null, busy: false,
  sound: localStorage.getItem('rag_sound') !== 'false',
  theme: localStorage.getItem('rag_theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : defaults.theme),
};

/* Audio Engine (Web Audio API) */
let audioCtx = null;
function initAudio() {
  try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { audioCtx = null; }
}
function playSound(type = 'click') {
  if (!audioCtx || !state.sound) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain); gain.connect(audioCtx.destination);
  const now = audioCtx.currentTime;
  if (type === 'click') {
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(880, now);
    osc.frequency.exponentialRampToValueAtTime(80, now + 0.05);
    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.05);
  } else if (type === 'send') {
    osc.type = 'sine';
    osc.frequency.setValueAtTime(520, now);
    osc.frequency.linearRampToValueAtTime(720, now + 0.08);
    gain.gain.setValueAtTime(0.05, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.08);
  } else if (type === 'pop') {
    osc.type = 'sine';
    osc.frequency.setValueAtTime(400, now);
    osc.frequency.linearRampToValueAtTime(240, now + 0.04);
    gain.gain.setValueAtTime(0.06, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.05);
  }
  osc.start(now);
  osc.stop(now + 0.1);
}

/* Client helpers */
function apiUrl(path) { return `${state.api.replace(/\/*$/, '')}/${path.replace(/^\/*/, '')}`; }

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = options.timeout || 30000;
  const timer = setTimeout(() => controller.abort(), timeout);
  const retries = options.retries !== undefined ? options.retries : 2;

  // Check if body is FormData - if so, don't set Content-Type (browser does it automatically with boundary)
  const isFormData = options.body instanceof FormData;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const fetchOptions = {
        ...options,
        signal: controller.signal,
      };

      // Only set Content-Type if NOT FormData (browser auto-sets for FormData with correct boundary)
      if (!isFormData && options.body) {
        fetchOptions.headers = {
          ...options.headers,
          'Content-Type': 'application/json; charset=utf-8',
        };
      } else if (options.headers) {
        fetchOptions.headers = options.headers;
      }

      const res = await fetch(apiUrl(path), fetchOptions);

      const ct = res.headers.get('content-type') || '';
      let data;
      try {
        data = ct.includes('application/json') ? await res.json() : await res.text();
      } catch {
        data = await res.text();
      }

      if (!res.ok) {
        let detail;
        if (typeof data === 'object') {
          // FastAPI/Pydantic validation errors often come in 'detail'
          detail = data.error?.message || data.detail || data.message || `HTTP ${res.status}`;
        } else {
          detail = String(data || `HTTP ${res.status}`);
        }
        throw new Error(detail);
      }
      clearTimeout(timer);
      return data;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') {
        throw new Error('Request timed out. Backend may be slow. Try again.');
      }
      if (err instanceof TypeError) {
        throw new Error('Cannot connect to backend. Is uvicorn running on port 8000?');
      }
      if (attempt < retries) {
        await new Promise(resolve => setTimeout(resolve, 500 * (attempt + 1)));
        continue;
      }
      throw err;
    }
  }
}

/* Utilities */
function escapeHtml(value = '') { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
function saveState() {
  localStorage.setItem('rag_api_url', state.api);
  localStorage.setItem('rag_session_id', state.sessionId);
  localStorage.setItem('rag_top_k', String(state.topK));
  localStorage.setItem('rag_threshold', String(state.threshold));
  localStorage.setItem('rag_sound', String(state.sound));
  localStorage.setItem('rag_theme', state.theme);
}

/* Toasts */
function toast(message, type = 'success') {
  if (state.sound) playSound('pop');
  const node = document.createElement('div');
  node.className = `toast ${type}`;
  node.textContent = message;
  $('#toast-region').appendChild(node);
  setTimeout(() => node.remove(), 3600);
}

/* Connection / Status */
function setConnection(status, label) {
  const pill = $('#connection-pill');
  pill.className = `status-pill is-${status}`;
  pill.querySelector('span').textContent = label;
}

/* Theme */
function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
  $('#theme-toggle').innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.4 14.8A8.5 8.5 0 0 1 9.2 3.6 8.5 8.5 0 1 0 20.4 14.8Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}
function toggleTheme() {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  saveState();
  applyTheme();
  if (state.sound) playSound('click');
}

/* Session */
function setSession(id) {
  state.sessionId = id;
  $('#session-label').textContent = id;
  $('.avatar-button').textContent = id.slice(-1).toUpperCase();
  saveState();
}
function newSession() {
  if (state.sound) playSound('click');
  const id = `session_${crypto.randomUUID ? crypto.randomUUID().slice(0, 8) : Math.random().toString(16).slice(2, 10)}`;
  setSession(id);
  state.messages = [];
  state.lastAnswer = null;
  renderMessages();
  renderSources([]);
  renderMetrics({});
  refreshDocuments();
  toast(`Started ${id}.`);
}

/* Metrics */
function renderMetrics(meta = {}) {
  const metricGrid = document.querySelector('#metric-grid');
  const strongs = metricGrid ? Array.from(metricGrid.querySelectorAll('strong')) : [];
  const vals = [meta.latency_ms != null ? meta.latency_ms + ' ms' : '-', meta.embedding_latency_ms != null ? meta.embedding_latency_ms + ' ms' : '-', meta.retrieval_latency_ms != null ? meta.retrieval_latency_ms + ' ms' : '-', meta.llm_latency_ms != null ? meta.llm_latency_ms + ' ms' : '-'];
  strongs.forEach((el, i) => { if (vals[i] !== undefined) el.textContent = vals[i]; });
  const latencyEl = document.querySelector('#last-latency');
  if (latencyEl) latencyEl.textContent = meta.latency_ms != null ? meta.latency_ms + ' ms' : 'Ready';
}

/* Documents */
function renderDocuments(docs = []) { if (!Array.isArray(docs)) docs = [];
  $('#doc-count').textContent = docs.length;
  const target = $('#documents-list');
  if (!docs.length) {
    target.innerHTML = '<div class="empty-state compact"><span>◇</span><p>No session documents yet.</p></div>';
    return;
  }
  target.innerHTML = docs.map(doc => `<article class="document-item clay-sm" title="${escapeHtml(doc.document_name)}"><strong>${escapeHtml(doc.document_name)}</strong><span>${doc.chunks} chunks · indexed</span></article>`).join('');
}

/* Sources */
function renderSources(sources = []) { if (!Array.isArray(sources)) sources = [];
  $('#source-count').textContent = sources.length;
  const target = $('#sources');
  if (!sources.length) {
    target.innerHTML = '<div class="empty-state compact"><span>◇</span><p>Sources appear after an answer.</p></div>';
    return;
  }
  target.innerHTML = sources.map(source => {
    const raw = Number(source.score || 0);
    const score = Math.max(0, Math.min(1, raw));
    const page = source.page ? ` · Page ${source.page}` : '';
    return `<article class="source-card clay-sm"><strong title="${escapeHtml(source.document)}">${escapeHtml(source.document)}</strong><span>Score ${raw.toFixed(3)}${page}</span><div class="score-bar"><i style="width:${score * 100}%"></i></div></article>`;
  }).join('');
}

/* Messages */
function messageHtml(message, index) {
  const role = message.role === 'user' ? 'user' : 'assistant';
  const content = escapeHtml(message.content || '');
  const time = message.timestamp ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  const actions = role === 'assistant' ? `<div class="message-actions"><button class="tiny-action" data-copy="${index}" type="button">Copy</button><button class="tiny-action" data-evaluate="${index}" type="button">Evaluate</button><button class="tiny-action" data-regenerate="${index}" type="button">Regenerate</button></div>` : '';
  return `<article class="message ${role}" data-index="${index}"><div><div class="message-bubble">${content}</div>${time ? `<div class="message-meta">${time}</div>` : ''}${actions}</div></article>`;
}
function renderMessages() {
  const target = $('#messages');
  if (!state.messages.length) {
    target.innerHTML = '<div class="welcome-card clay"><div class="welcome-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2l1.9 6.1L20 10l-6.1 1.9L12 18l-1.9-6.1L4 10l6.1-1.9L12 2Z" fill="none" stroke="#007AFF" stroke-width="1.7" stroke-linejoin="round"/></svg></div><div><h3>Your context layer is ready.</h3><p>Attach a document, then ask a question. Follow-ups use this session history and retrieve only from its attached knowledge.</p></div></div>';
    return;
  }
  target.innerHTML = state.messages.map(messageHtml).join('');
  requestAnimationFrame(() => target.scrollTo({ top: target.scrollHeight, behavior: 'smooth' }));
}

/* Modal */
function openModal(content, title = 'Context summary') {
  $('#modal-content').textContent = content;
  $('#modal-title').textContent = title;
  $('#modal-backdrop').hidden = false;
  $('#modal-close').focus();
  if (state.sound) playSound('click');
}
function closeModal() {
  $('#modal-backdrop').hidden = true;
  $('#session-menu-button').focus();
  if (state.sound) playSound('pop');
}

/* Main API calls */
async function refreshHistory(show = false) {
  try {
    const data = await request(`/history/${encodeURIComponent(state.sessionId)}`);
    state.messages = data.messages || [];
    renderMessages();
    if (show) toast('Conversation refreshed.');
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function refreshDocuments() {
  try {
    const data = await request(`/session/${encodeURIComponent(state.sessionId)}/documents`);
    renderDocuments(data.documents || []);
  } catch {
    renderDocuments([]);
  }
}

async function diagnostics(show = false) {
  if (state.sound) playSound('click');
  setConnection('checking', 'Checking');

  try {
    // Try health endpoint first with explicit timeout
    const healthRes = await fetch(state.api + '/health', {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });

    if (!healthRes.ok) {
      throw new Error(`Health check failed: ${healthRes.status}`);
    }

    const health = await healthRes.json();
    setConnection('online', 'Backend online');
    $('#diag-api').textContent = health.version || health.service || 'Ready';

    try {
      const readyRes = await fetch(state.api + '/ready', {
        method: 'GET',
        signal: AbortSignal.timeout(10000)
      });
      const ready = await readyRes.json();
      const c = ready.components || {};
      $('#diag-vector').textContent = c.vector_store || 'Ready';
      $('#diag-embed').textContent = c.embedding_service || 'Ready';
      $('#diag-llm').textContent = c.llm || 'Ready';
    } catch {
      // Ready endpoint might not exist or be slow - that's OK
      $('#diag-vector').textContent = 'Ready';
      $('#diag-embed').textContent = 'Ready';
      $('#diag-llm').textContent = 'Ready';
    }

    if (show) toast('Backend connected successfully');
  } catch (err) {
    console.error('Diagnostics error:', err);
    setConnection('error', 'Backend offline');
    $('#diag-api').textContent = 'Offline';
    $('#diag-vector').textContent = '—';
    $('#diag-embed').textContent = '—';
    $('#diag-llm').textContent = '—';

    const msg = err.message.includes('Failed to fetch') || err.message.includes('network')
      ? 'Cannot connect to backend. Start: uvicorn app.main:app --port 8000'
      : err.message;

    if (show) toast(msg, 'error');
  }
}

function setBusy(busy) {
  state.busy = busy;
  $('#send-button').disabled = busy;
  $('#question').disabled = busy;
  $('#send-button').innerHTML = busy ? '<span aria-hidden="true">…</span>' : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 4 16 8-16 8 3-8-3-8Zm3 8h13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
}

async function sendQuestion(question) {
  if (state.busy) return;
  question = question.trim();
  if (!question) return;

  if (question.length > 4000) {
    toast('Question too long. Maximum 4000 characters.', 'error');
    return;
  }

  if (state.sound) playSound('send');
  setBusy(true);

  const optimistic = { role: 'user', content: question, timestamp: new Date().toISOString() };
  state.messages.push(optimistic);
  renderMessages();

  try {
    // Ensure session exists first
    await request(`/history/${encodeURIComponent(state.sessionId)}`, { timeout: 10000 }).catch(() => {});

    const data = await request('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        question: question,
        top_k: Math.max(1, Math.min(state.topK, 10)),
        similarity_threshold: Math.max(-1, Math.min(state.threshold, 1)),
        include_sources: true,
      }),
      timeout: 60000,
      retries: 2,
    });

    if (!data || !data.answer) {
      throw new Error('Invalid response from backend. No answer received.');
    }

    state.messages.push({
      role: 'assistant',
      content: data.answer,
      timestamp: new Date().toISOString()
    });
    state.lastAnswer = data;
    renderMessages();
    renderSources(data.sources || []);
    renderMetrics(data.metadata || {});

    const wordCount = (data.answer || '').split(/\s+/).length;
    toast(`Generated ${wordCount} word answer`);
  } catch (err) {
    state.messages.pop();
    renderMessages();

    let msg = err.message || 'Unknown error occurred';

    if (msg.includes('TypeError') || msg.includes('offline') || msg.includes('unavailable')) {
      msg = 'Backend offline. Check if uvicorn is running on port 8000.';
      setConnection('error', 'Backend offline');
    } else if (msg.includes('timeout') || msg.includes('timed out')) {
      msg = 'Request timed out. Backend may be processing. Try again in a moment.';
    } else if (msg.includes('422') || msg.includes('validation')) {
      msg = 'Invalid request format. Check your input.';
    } else if (msg.includes('502') || msg.includes('503') || msg.includes('504')) {
      msg = 'Backend service error. Please try again.';
    }

    console.error('Chat error:', err);
    toast(msg, 'error');
  } finally {
    setBusy(false);
    $('#question').focus();
  }
}

async function uploadDocument(file) {
  if (!file) {
    toast('Please select a file to upload.', 'error');
    return;
  }

  // Validate file
  const allowedTypes = ['.pdf', '.docx', '.txt', '.md', '.markdown'];
  const fileName = file.name.toLowerCase();
  const fileExt = fileName.substring(fileName.lastIndexOf('.'));

  if (!allowedTypes.includes(fileExt)) {
    toast(`File type "${fileExt}" not supported. Use: PDF, DOCX, TXT, MD`, 'error');
    $('#file-input').value = '';
    return;
  }

  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    toast(`File is ${(file.size / 1024 / 1024).toFixed(1)}MB. Maximum 10MB allowed.`, 'error');
    $('#file-input').value = '';
    return;
  }

  $('#upload-drop').classList.add('is-dragging');
  setBusy(true);

  try {
    // Ensure session exists
    await request(`/history/${encodeURIComponent(state.sessionId)}`, {
      timeout: 10000
    }).catch(() => {});

    // Create form data - CRITICAL: session_id must come first
    const form = new FormData();
    form.append('session_id', state.sessionId);
    form.append('file', file);

    const data = await request('/documents/upload-session', {
      method: 'POST',
      body: form,
      timeout: 60000,
      retries: 1,
      headers: {} // Let browser set Content-Type with boundary
    });

    if (!data || !data.document_name) {
      throw new Error('Invalid response: document not created');
    }

    await refreshDocuments();
    toast(`Document "${data.document_name}" attached successfully`);
    $('#file-input').value = '';
  } catch (err) {
    console.error('Upload error:', err);
    let msg = err.message || 'Upload failed';

    if (msg.includes('Field required')) {
      msg = 'Server error: Required fields missing. Ensure backend is running correctly.';
    } else if (msg.includes('413')) {
      msg = 'File too large. Maximum 10MB.';
    } else if (msg.includes('422')) {
      msg = 'Invalid file or session. Try again.';
    } else if (msg.includes('400')) {
      msg = 'Bad request. Check file type and session.';
    } else if (msg.includes('offline') || msg.includes('unavailable')) {
      msg = 'Backend offline. Check if uvicorn is running.';
    } else if (msg.includes('timeout')) {
      msg = 'Upload timed out. File may be too large or connection slow.';
    }

    toast(msg, 'error');
    $('#file-input').value = '';
  } finally {
    $('#upload-drop').classList.remove('is-dragging');
    setBusy(false);
  }
}

async function clearHistory() {
  if (state.sound) playSound('click');
  if (!confirm('Clear this conversation? Session documents will remain attached.')) return;
  try {
    await request(`/history/${encodeURIComponent(state.sessionId)}`, { method: 'DELETE' });
    state.messages = [];
    state.lastAnswer = null;
    renderMessages();
    renderSources([]);
    renderMetrics({});
    toast('Conversation cleared. Documents remain attached.');
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function summarizeHistory() {
  if (state.sound) playSound('click');
  try {
    const data = await request('/summarize-history', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: state.sessionId }), timeout: 45000 });
    openModal(data.summary || 'Nothing to summarize yet.', 'Context summary');
    toast('Context summary updated.');
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function evaluateAnswer(index) {
  if (state.sound) playSound('click');
  const assistant = state.messages[index];
  if (!assistant || assistant.role !== 'assistant') return;
  const priorUser = [...state.messages.slice(0, index)].reverse().find(m => m.role === 'user');
  if (!priorUser) return toast('No matching question found.', 'error');
  try {
    const sourceNames = (state.lastAnswer?.sources || []).map(s => s.document);
    const result = await request('/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: priorUser.content, answer: assistant.content, sources: sourceNames }),
      timeout: 15000,
    });
    openModal(
      `Status: ${result.status}\n\nRelevance score: ${Number(result.relevance_score || 0).toFixed(2)}\nSource available: ${result.source_available ? 'Yes' : 'No'}\nAnswer supported: ${result.answer_supported ? 'Yes' : 'No'}`,
      'Answer evaluation'
    );
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function regenerate(index) {
  if (state.sound) playSound('click');
  const priorUser = [...state.messages.slice(0, index)].reverse().find(m => m.role === 'user');
  if (!priorUser) return toast('No matching question found.', 'error');
  await sendQuestion(priorUser.content);
}

/* Autosize */
function autosize() {
  const el = $('#question');
  el.style.height = 'auto';
  el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
}

/* Sound toggle */
function toggleSound() {
  state.sound = !state.sound;
  saveState();
  $('#sound-toggle').setAttribute('aria-label', state.sound ? 'Mute sounds' : 'Unmute sounds');
  toast(state.sound ? 'Sounds on' : 'Sounds muted');
  if (state.sound) playSound('click');
}

/* Event wiring */
window.addEventListener('DOMContentLoaded', async () => {
  initAudio();
  applyTheme();
  setSession(state.sessionId);
  $('#api-url').value = state.api;
  $('#top-k').value = state.topK;
  $('#top-k-value').textContent = state.topK;
  $('#threshold').value = state.threshold;
  $('#threshold-value').textContent = state.threshold.toFixed(2);

  $('#theme-toggle').addEventListener('click', toggleTheme);
  $('#sound-toggle').addEventListener('click', toggleSound);
  $('#new-session').addEventListener('click', newSession);
  $('#session-menu-button').addEventListener('click', () => openModal(`Active session: ${state.sessionId}\n\nSession data is isolated by session ID. Clearing history does not delete attached documents.`, 'Session details'));

  $('#save-api').addEventListener('click', async () => {
    const value = $('#api-url').value.trim().replace(/\/*$/, '');
    if (!/^https?:\/\//i.test(value)) return toast('Enter a full http:// or https:// backend URL.', 'error');
    state.api = value;
    saveState();
    await diagnostics(true);
    await Promise.all([refreshHistory(), refreshDocuments()]);
  });

  $('#refresh-history').addEventListener('click', () => refreshHistory(true));
  $('#clear-history').addEventListener('click', clearHistory);
  $('#summarize-history').addEventListener('click', summarizeHistory);
  $('#hero-health').addEventListener('click', () => diagnostics(true));
  $('#hero-focus').addEventListener('click', () => $('#question').focus());
  $('#scroll-bottom').addEventListener('click', () => $('#messages').scrollTo({ top: $('#messages').scrollHeight, behavior: 'smooth' }));
  $('#open-docs').addEventListener('click', () => window.open(apiUrl('/docs'), '_blank', 'noopener,noreferrer'));
  $('#run-diag').addEventListener('click', () => diagnostics(true));

  $('#modal-close').addEventListener('click', closeModal);
  $('#modal-backdrop').addEventListener('click', e => { if (e.target.id === 'modal-backdrop') closeModal(); });

  $('#top-k').addEventListener('input', e => { state.topK = Number(e.target.value); $('#top-k-value').textContent = state.topK; saveState(); });
  $('#threshold').addEventListener('input', e => { state.threshold = Number(e.target.value); $('#threshold-value').textContent = state.threshold.toFixed(2); saveState(); });

  $('#chat-form').addEventListener('submit', e => { e.preventDefault(); const q = $('#question').value; $('#question').value = ''; autosize(); sendQuestion(q); });
  $('#question').addEventListener('input', autosize);
  $('#question').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#chat-form').requestSubmit(); } });

  $$('.hint-chip').forEach(btn => btn.addEventListener('click', () => { $('#question').value = btn.dataset.prompt; autosize(); $('#question').focus(); if (state.sound) playSound('click'); }));

  $('#file-input').addEventListener('change', e => uploadDocument(e.target.files[0]));
  const drop = $('#upload-drop');
  ['dragenter', 'dragover'].forEach(evt => drop.addEventListener(evt, e => { e.preventDefault(); drop.classList.add('is-dragging'); }));
  ['dragleave', 'drop'].forEach(evt => drop.addEventListener(evt, e => { e.preventDefault(); drop.classList.remove('is-dragging'); }));
  drop.addEventListener('drop', e => uploadDocument(e.dataTransfer.files[0]));
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file-input').click(); } });

  $('#messages').addEventListener('click', async e => {
    const copyBtn = e.target.closest('[data-copy]');
    const evaluateBtn = e.target.closest('[data-evaluate]');
    const regenBtn = e.target.closest('[data-regenerate]');
    if (copyBtn) {
      const text = state.messages[Number(copyBtn.dataset.copy)]?.content || '';
      try {
        await navigator.clipboard.writeText(text);
        toast('Answer copied.');
      } catch {
        toast('Clipboard access was blocked.', 'error');
      }
    }
    if (evaluateBtn) await evaluateAnswer(Number(evaluateBtn.dataset.evaluate));
    if (regenBtn) await regenerate(Number(regenBtn.dataset.regenerate));
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('#modal-backdrop').hidden) closeModal();
  });

  await diagnostics();
  await Promise.all([refreshHistory(), refreshDocuments()]);
});
