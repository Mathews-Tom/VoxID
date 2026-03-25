/**
 * VoxID UI — Main application.
 *
 * Path-based routing via history.pushState:
 *   /              → Dashboard (identity list)
 *   /dashboard     → Dashboard
 *   /identity/:id  → Identity detail
 *   /enrollment    → Enrollment wizard (consent → record → summary)
 *   /generate      → Generation playground
 *
 * All API calls go to /api/* prefix.
 */

import { AudioCapture } from './audio-capture.js';
import { LiveWaveform } from './waveform.js';
import { Meters } from './meters.js';
import { Playback } from './playback.js';
import { EnrollmentController } from './enrollment-controller.js';
import { renderScriptPanel } from './annotation-renderer.js';

// ── State ──────────────────────────────────────────

const ctrl = new EnrollmentController();
const playback = new Playback();
let capture = null;
let waveform = null;
let meters = null;

let selectedIdentity = null;
let selectedStyles = [];
let promptsPerStyle = 5;
let consentBlob = null;
let currentRecordingBlob = null;
let meterAnimId = null;

// Enrollment sub-step (managed within /enrollment path)
let enrollStep = 'select'; // select | consent | record | summary

// ── API helpers ────────────────────────────────────

async function api(path, opts = {}) {
    const res = await fetch(`/api${path}`, opts);
    if (!res.ok) {
        let detail;
        try { detail = await res.json(); } catch { detail = { detail: res.statusText }; }
        const err = new Error(detail.detail?.message || detail.detail || res.statusText);
        err.status = res.status;
        err.detail = detail.detail;
        throw err;
    }
    if (res.status === 204) return null;
    return res.json();
}

// ── Navigation ─────────────────────────────────────

function navigate(path) {
    history.pushState(null, '', path);
    route();
}

// Intercept link clicks for SPA navigation
document.addEventListener('click', (e) => {
    const link = e.target.closest('a[href]');
    if (!link) return;
    const href = link.getAttribute('href');
    if (href.startsWith('/') && !href.startsWith('/api') && !href.startsWith('/assets') && !href.startsWith('/docs')) {
        e.preventDefault();
        navigate(href);
    }
});

// ── Boot ───────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    route();
    window.addEventListener('popstate', route);
});

function route() {
    const path = location.pathname;

    // Hide all screens
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));

    // Enrollment steps bar
    const enrollSteps = document.getElementById('enroll-steps');
    enrollSteps.style.display = path === '/enrollment' ? 'flex' : 'none';

    // Update top nav
    document.querySelectorAll('.nav-link').forEach(a => a.classList.remove('active'));
    if (path === '/' || path === '/dashboard' || path.startsWith('/identity/')) {
        document.querySelector('[data-nav="dashboard"]')?.classList.add('active');
    } else if (path === '/enrollment') {
        document.querySelector('[data-nav="enroll"]')?.classList.add('active');
    } else if (path === '/generate') {
        document.querySelector('[data-nav="generate"]')?.classList.add('active');
    }

    // Route to screen
    if (path.startsWith('/identity/')) {
        document.getElementById('identity-detail').classList.add('active');
        const id = decodeURIComponent(path.slice('/identity/'.length));
        initDetailScreen(id);
    } else if (path === '/generate') {
        document.getElementById('generate').classList.add('active');
        initGenerateScreen();
    } else if (path === '/enrollment') {
        showEnrollStep();
    } else {
        // Default: dashboard
        document.getElementById('dashboard').classList.add('active');
        initDashboard();
    }
}

// ── Enrollment sub-routing ─────────────────────────

function showEnrollStep() {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById('enroll-steps').style.display = 'flex';

    const stepMap = { select: 0, consent: 1, record: 2, summary: 3 };
    document.querySelectorAll('#enroll-steps .step').forEach((el, i) => {
        el.classList.remove('active', 'done');
        if (i === stepMap[enrollStep]) el.classList.add('active');
        else if (i < stepMap[enrollStep]) el.classList.add('done');
    });

    document.getElementById(enrollStep).classList.add('active');

    if (enrollStep === 'select') initSelectScreen();
    else if (enrollStep === 'consent') initConsentScreen();
    else if (enrollStep === 'record') initRecordScreen();
    else if (enrollStep === 'summary') initSummaryScreen();
}

function enrollNavigate(step) {
    enrollStep = step;
    showEnrollStep();
}

// ══════════════════════════════════════════════════
// Dashboard
// ══════════════════════════════════════════════════

async function initDashboard() {
    const list = document.getElementById('dashboard-list');
    const newBtn = document.getElementById('btn-dashboard-new');
    const createForm = document.getElementById('dashboard-create-form');

    list.innerHTML = '<div class="spinner" style="margin:24px auto;display:block"></div>';

    newBtn.onclick = () => {
        createForm.style.display = createForm.style.display === 'none' ? 'block' : 'none';
    };
    document.getElementById('btn-dash-cancel').onclick = () => { createForm.style.display = 'none'; };
    document.getElementById('btn-dash-create').onclick = async () => {
        const id = document.getElementById('dash-new-id').value.trim();
        const name = document.getElementById('dash-new-name').value.trim();
        if (!id || !name) return;
        try {
            await api('/identities', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id, name }),
            });
            createForm.style.display = 'none';
            document.getElementById('dash-new-id').value = '';
            document.getElementById('dash-new-name').value = '';
            initDashboard();
        } catch (e) { alert(e.message); }
    };

    try {
        const data = await api('/identities');
        if (data.identities.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🎤</div>
                    <div>No voice identities yet.</div>
                    <div style="margin-top:8px;font-size:13px">Create one to get started.</div>
                </div>`;
            return;
        }

        const details = await Promise.all(
            data.identities.map(async id => {
                const [identity, styles, consent] = await Promise.all([
                    api(`/identities/${encodeURIComponent(id)}`),
                    api(`/identities/${encodeURIComponent(id)}/styles`),
                    api(`/enroll/consent/${encodeURIComponent(id)}/status`).catch(() => ({ has_consent: false })),
                ]);
                return { ...identity, styles: styles.styles, has_consent: consent.has_consent };
            })
        );

        list.innerHTML = details.map(d => `
            <div class="identity-card" data-id="${esc(d.id)}">
                <div class="identity-avatar">${esc(d.name.charAt(0).toUpperCase())}</div>
                <div class="identity-info">
                    <div class="identity-name">${esc(d.name)}</div>
                    <div class="identity-meta">
                        <span>${esc(d.id)}</span>
                        <span>${d.styles.length} style${d.styles.length !== 1 ? 's' : ''}</span>
                        <span>${d.created_at?.slice(0, 10) || '—'}</span>
                        <span class="consent-badge ${d.has_consent ? 'yes' : 'no'}">
                            ${d.has_consent ? 'consent ✓' : 'no consent'}
                        </span>
                    </div>
                </div>
                <div class="identity-actions">
                    <a href="/enrollment" class="btn btn-ghost" style="padding:6px 12px;font-size:12px"
                       onclick="event.stopPropagation()">Enroll</a>
                </div>
            </div>
        `).join('');

        list.querySelectorAll('.identity-card').forEach(card => {
            card.onclick = () => navigate(`/identity/${encodeURIComponent(card.dataset.id)}`);
        });
    } catch (e) {
        list.innerHTML = `<div class="quality-result failed">Failed to load identities: ${esc(e.message)}</div>`;
    }
}

// ══════════════════════════════════════════════════
// Identity Detail
// ══════════════════════════════════════════════════

async function initDetailScreen(identityId) {
    const content = document.getElementById('identity-detail-content');
    content.innerHTML = '<div class="spinner" style="margin:24px auto;display:block"></div>';

    document.getElementById('btn-detail-back').onclick = () => navigate('/dashboard');
    document.getElementById('btn-detail-delete').onclick = async () => {
        if (!confirm(`Delete identity "${identityId}" and all its data? This cannot be undone.`)) return;
        try {
            await api(`/identities/${encodeURIComponent(identityId)}`, { method: 'DELETE' });
            navigate('/dashboard');
        } catch (e) { alert(e.message); }
    };

    try {
        const [identity, stylesList, consent] = await Promise.all([
            api(`/identities/${encodeURIComponent(identityId)}`),
            api(`/identities/${encodeURIComponent(identityId)}/styles`),
            api(`/enroll/consent/${encodeURIComponent(identityId)}/status`).catch(() => ({ has_consent: false })),
        ]);

        const styleDetails = await Promise.all(
            stylesList.styles.map(sid =>
                api(`/identities/${encodeURIComponent(identityId)}/styles/${encodeURIComponent(sid)}`).catch(() => null)
            )
        );

        let html = `
            <div class="detail-header">
                <div class="identity-avatar" style="width:56px;height:56px;font-size:24px">
                    ${esc(identity.name.charAt(0).toUpperCase())}
                </div>
                <div>
                    <div class="detail-title">${esc(identity.name)}</div>
                    <div class="detail-id">${esc(identity.id)}</div>
                </div>
                <span class="consent-badge ${consent.has_consent ? 'yes' : 'no'}" style="margin-left:auto">
                    ${consent.has_consent ? 'consent recorded' : 'no consent'}
                </span>
            </div>
            <div class="panel">
                <div class="panel-label">Details</div>
                <div class="detail-grid">
                    <span class="detail-label">Default style</span><span>${esc(identity.default_style)}</span>
                    <span class="detail-label">Created</span><span>${esc(identity.created_at)}</span>
                    <span class="detail-label">Description</span><span>${esc(identity.description || '—')}</span>
                </div>
            </div>
            <div class="panel">
                <div class="panel-label">Styles (${stylesList.styles.length})</div>`;

        if (styleDetails.length === 0) {
            html += `<div style="color:var(--text-2);font-size:13px;padding:12px 0">
                No styles enrolled. <a href="/enrollment" style="color:var(--primary)">Start enrollment</a>
            </div>`;
        } else {
            for (const s of styleDetails) {
                if (!s) continue;
                const audioUrl = `/api/identities/${encodeURIComponent(identityId)}/styles/${encodeURIComponent(s.id)}/audio`;
                html += `
                    <div class="style-item">
                        <span class="style-item-name">${esc(s.label || s.id)}</span>
                        <span class="style-item-engine">${esc(s.default_engine)}</span>
                        <span style="font-size:12px;color:var(--text-2)">${esc(s.language)}</span>
                        <div class="style-item-audio">
                            <audio controls class="inline-audio" preload="none">
                                <source src="${audioUrl}" type="audio/wav">
                            </audio>
                        </div>
                    </div>`;
            }
        }
        html += `</div>`;

        // Quick generate
        html += `
            <div class="panel">
                <div class="panel-label">Quick Generate</div>
                <div class="form-group">
                    <textarea id="detail-gen-text" rows="2"
                        style="width:100%;background:var(--bg);border:1px solid var(--surface-2);border-radius:var(--radius);color:var(--text);padding:10px;font-family:var(--font);font-size:14px;resize:vertical"
                        placeholder="Type text to generate speech..."></textarea>
                </div>
                <div class="controls" style="justify-content:flex-start">
                    <button class="btn btn-primary" id="btn-detail-generate">Generate</button>
                </div>
                <div id="detail-gen-result"></div>
            </div>`;

        content.innerHTML = html;

        document.getElementById('btn-detail-generate').onclick = async () => {
            const text = document.getElementById('detail-gen-text').value.trim();
            if (!text) return;
            const btn = document.getElementById('btn-detail-generate');
            const result = document.getElementById('detail-gen-result');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Generating...';
            result.innerHTML = '';
            try {
                const gen = await api('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, identity_id: identityId }),
                });
                const audioUrl = `/api/generate/audio?path=${encodeURIComponent(gen.audio_path)}`;
                result.innerHTML = `
                    <div style="margin-top:12px">
                        <audio controls style="width:100%" autoplay>
                            <source src="${audioUrl}" type="audio/wav">
                        </audio>
                        <div class="quality-metrics" style="margin-top:6px">
                            Style: ${esc(gen.style_used)} &nbsp; Sample rate: ${gen.sample_rate} Hz
                        </div>
                    </div>`;
            } catch (e) {
                result.innerHTML = `<div class="quality-result failed">${esc(e.message)}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate';
            }
        };
    } catch (e) {
        content.innerHTML = `<div class="quality-result failed">Failed to load identity: ${esc(e.message)}</div>`;
    }
}

// ══════════════════════════════════════════════════
// Generation Playground
// ══════════════════════════════════════════════════

async function initGenerateScreen() {
    const identitySel = document.getElementById('gen-identity');
    const styleSel = document.getElementById('gen-style');
    const textEl = document.getElementById('gen-text');
    const genBtn = document.getElementById('btn-generate');
    const routeBtn = document.getElementById('btn-gen-route');
    const resultEl = document.getElementById('gen-result');
    const audioPlayer = document.getElementById('gen-audio-player');
    const audioEl = document.getElementById('gen-audio');
    const metaEl = document.getElementById('gen-meta');
    const routeResult = document.getElementById('gen-route-result');

    try {
        const data = await api('/identities');
        identitySel.innerHTML = '<option value="">Select identity...</option>' +
            data.identities.map(id => `<option value="${id}">${id}</option>`).join('');
    } catch { identitySel.innerHTML = '<option value="">Failed to load</option>'; }

    identitySel.onchange = async () => {
        const id = identitySel.value;
        genBtn.disabled = !id || !textEl.value.trim();
        routeBtn.disabled = !id || !textEl.value.trim();
        styleSel.innerHTML = '<option value="">Auto-detect</option>';
        if (!id) return;
        try {
            const styles = await api(`/identities/${encodeURIComponent(id)}/styles`);
            styleSel.innerHTML = '<option value="">Auto-detect</option>' +
                styles.styles.map(s => `<option value="${s}">${s}</option>`).join('');
        } catch { /* ignore */ }
    };

    textEl.oninput = () => {
        genBtn.disabled = !identitySel.value || !textEl.value.trim();
        routeBtn.disabled = !identitySel.value || !textEl.value.trim();
    };

    genBtn.onclick = async () => {
        const text = textEl.value.trim();
        const identity = identitySel.value;
        const style = styleSel.value || undefined;
        if (!text || !identity) return;

        genBtn.disabled = true;
        genBtn.innerHTML = '<span class="spinner"></span> Generating...';
        resultEl.innerHTML = '';
        audioPlayer.style.display = 'none';

        try {
            const gen = await api('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, identity_id: identity, style }),
            });
            const audioUrl = `/api/generate/audio?path=${encodeURIComponent(gen.audio_path)}`;
            audioEl.src = audioUrl;
            audioPlayer.style.display = 'block';
            audioEl.play();
            metaEl.innerHTML = `Style: <strong>${esc(gen.style_used)}</strong> &nbsp; Sample rate: ${gen.sample_rate} Hz &nbsp; Identity: ${esc(gen.identity_id)}`;
        } catch (e) {
            resultEl.innerHTML = `<div class="quality-result failed">${esc(e.message)}</div>`;
        } finally {
            genBtn.disabled = false;
            genBtn.textContent = 'Generate';
        }
    };

    routeBtn.onclick = async () => {
        const text = textEl.value.trim();
        const identity = identitySel.value;
        if (!text || !identity) return;

        routeBtn.disabled = true;
        routeBtn.innerHTML = '<span class="spinner"></span> Routing...';
        routeResult.innerHTML = '';

        try {
            const result = await api('/route', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, identity_id: identity }),
            });
            const scores = Object.entries(result.scores || {})
                .sort(([, a], [, b]) => b - a)
                .map(([style, score]) =>
                    `<span class="route-score-chip ${style === result.style ? 'selected' : ''}">${esc(style)}: ${score.toFixed(2)}</span>`
                ).join('');
            routeResult.innerHTML = `
                <div class="route-result">
                    <div>Routed to: <strong>${esc(result.style)}</strong>
                        <span style="color:var(--text-2);font-size:12px;margin-left:8px">
                            confidence: ${result.confidence.toFixed(2)} (${esc(result.tier)})
                        </span>
                    </div>
                    <div class="route-scores">${scores}</div>
                </div>`;
        } catch (e) {
            routeResult.innerHTML = `<div class="quality-result failed">${esc(e.message)}</div>`;
        } finally {
            routeBtn.disabled = false;
            routeBtn.textContent = 'Route (dry run)';
        }
    };
}

// ══════════════════════════════════════════════════
// Enrollment: Identity Selection
// ══════════════════════════════════════════════════

async function initSelectScreen() {
    const sel = document.getElementById('identity-select');
    const newForm = document.getElementById('new-identity-form');
    const beginBtn = document.getElementById('btn-begin');

    try {
        await ctrl.fetchIdentities();
        sel.innerHTML = '<option value="">Select identity...</option>' +
            ctrl.identities.map(id => `<option value="${id}">${id}</option>`).join('');
    } catch { sel.innerHTML = '<option value="">Failed to load identities</option>'; }

    selectedStyles = [...document.querySelectorAll('.style-checkbox:checked')].map(c => c.value);

    sel.onchange = () => {
        selectedIdentity = sel.value;
        beginBtn.disabled = !selectedIdentity || selectedStyles.length === 0;
    };

    document.getElementById('btn-show-new').onclick = () => {
        newForm.style.display = newForm.style.display === 'none' ? 'block' : 'none';
    };

    document.getElementById('btn-create-identity').onclick = async () => {
        const id = document.getElementById('new-id').value.trim();
        const name = document.getElementById('new-name').value.trim();
        if (!id || !name) return;
        try {
            await ctrl.createIdentity(id, name);
            await ctrl.fetchIdentities();
            sel.innerHTML = '<option value="">Select identity...</option>' +
                ctrl.identities.map(i => `<option value="${i}">${i}</option>`).join('');
            sel.value = id;
            selectedIdentity = id;
            newForm.style.display = 'none';
        } catch (e) { alert(e.message); }
    };

    document.querySelectorAll('.style-checkbox').forEach(cb => {
        cb.onchange = () => {
            selectedStyles = [...document.querySelectorAll('.style-checkbox:checked')].map(c => c.value);
            beginBtn.disabled = !selectedIdentity || selectedStyles.length === 0;
        };
    });

    document.getElementById('prompts-per-style').onchange = (e) => {
        promptsPerStyle = parseInt(e.target.value) || 5;
    };

    beginBtn.onclick = async () => {
        beginBtn.disabled = true;
        beginBtn.innerHTML = '<span class="spinner"></span> Creating session...';
        try {
            await ctrl.createSession(selectedIdentity, selectedStyles, promptsPerStyle);
            document.getElementById('session-id-display').textContent = ctrl.session.session_id;
            enrollNavigate('consent');
        } catch (e) { alert(e.message); }
        finally {
            beginBtn.disabled = false;
            beginBtn.textContent = 'Begin Enrollment';
        }
    };
}

// ══════════════════════════════════════════════════
// Enrollment: Consent Recording
// ══════════════════════════════════════════════════

async function initConsentScreen() {
    const stmtEl = document.getElementById('consent-statement');
    const qualEl = document.getElementById('consent-quality');
    const waveCanvas = document.getElementById('consent-waveform');
    const timeEl = document.getElementById('consent-time');
    const recordBtn = document.getElementById('btn-consent-record');
    const playBtn = document.getElementById('btn-consent-play');
    const nextBtn = document.getElementById('btn-consent-next');

    try {
        const status = await ctrl.getConsentStatus(selectedIdentity);
        if (status.has_consent) {
            stmtEl.innerHTML = '<strong>Consent already recorded.</strong> You can proceed or re-record.';
            nextBtn.disabled = false;
        }
    } catch { /* ignore */ }

    try {
        const data = await ctrl.getConsentStatement(selectedIdentity);
        if (!stmtEl.querySelector('strong')) stmtEl.textContent = `"${data.statement}"`;
    } catch (e) { stmtEl.textContent = `Error: ${e.message}`; }

    if (!capture) { capture = new AudioCapture(); await capture.init(); }

    const liveWave = new LiveWaveform(waveCanvas);
    liveWave.attach(capture.analyser);
    let recording = false;

    recordBtn.onclick = async () => {
        if (!recording) {
            recording = true;
            recordBtn.textContent = '■ Stop';
            recordBtn.className = 'btn btn-danger';
            playBtn.disabled = true;
            qualEl.innerHTML = '';
            liveWave.start();
            capture.startRecording();
            const timerInterval = setInterval(() => {
                if (!recording) { clearInterval(timerInterval); return; }
                timeEl.textContent = formatTime(capture.elapsed);
            }, 100);
        } else {
            recording = false;
            recordBtn.textContent = '● Record';
            recordBtn.className = 'btn btn-primary';
            liveWave.stop();
            consentBlob = capture.stopRecording();
            playBtn.disabled = false;

            qualEl.innerHTML = '<span class="spinner"></span> Validating...';
            try {
                await ctrl.uploadConsent(selectedIdentity, consentBlob);
                qualEl.innerHTML = '<div class="quality-result passed">Consent accepted</div>';
                nextBtn.disabled = false;
            } catch (e) {
                const detail = e.detail || {};
                const reasons = detail.rejection_reasons || [e.message];
                qualEl.innerHTML = `
                    <div class="quality-result failed">
                        <strong>Rejected</strong>
                        <div class="rejection-reasons">${reasons.map(r => `<div>- ${r}</div>`).join('')}</div>
                        <div class="quality-metrics">
                            SNR: ${detail.snr_db?.toFixed(1) ?? '—'} dB &nbsp;
                            RMS: ${detail.rms_dbfs?.toFixed(1) ?? '—'} dBFS &nbsp;
                            Speech: ${detail.speech_ratio ? (detail.speech_ratio * 100).toFixed(0) + '%' : '—'} &nbsp;
                            Duration: ${detail.total_duration_s?.toFixed(1) ?? '—'}s
                        </div>
                    </div>`;
            }
        }
    };

    playBtn.onclick = () => { if (consentBlob) playback.playBlob(consentBlob); };
    nextBtn.onclick = () => enrollNavigate('record');
    document.getElementById('btn-consent-back').onclick = () => enrollNavigate('select');
}

// ══════════════════════════════════════════════════
// Enrollment: Prompted Recording
// ══════════════════════════════════════════════════

async function initRecordScreen() {
    const scriptEl = document.getElementById('record-script');
    const waveCanvas = document.getElementById('record-waveform');
    const timeEl = document.getElementById('record-time');
    const metersEl = document.getElementById('record-meters');
    const historyEl = document.getElementById('record-history');
    const qualEl = document.getElementById('record-quality');
    const counterEl = document.getElementById('prompt-counter');
    const styleBadgeEl = document.getElementById('style-badge');
    const recordBtn = document.getElementById('btn-record');
    const playBtn = document.getElementById('btn-record-play');
    const skipBtn = document.getElementById('btn-skip');

    if (!capture) { capture = new AudioCapture(); await capture.init(); }
    if (!waveform) { waveform = new LiveWaveform(waveCanvas); waveform.attach(capture.analyser); }
    if (!meters) { meters = new Meters(metersEl); }

    await ctrl.refreshSession();
    renderPrompt();

    let recording = false;

    function renderPrompt() {
        const prompt = ctrl.currentPrompt;
        const style = ctrl.currentStyle;
        if (!prompt || !style) { enrollNavigate('summary'); return; }

        styleBadgeEl.textContent = style;
        const progress = ctrl.progress;
        let totalDone = 0, totalAll = 0;
        for (const s of Object.keys(progress)) {
            const p = progress[s];
            totalDone += (p.accepted || 0) + (p.rejected || 0);
            totalAll += p.total_prompts || 0;
        }
        counterEl.textContent = `${totalDone + 1} / ${totalAll}`;
        scriptEl.innerHTML = renderScriptPanel(prompt);
        qualEl.innerHTML = '';
        playBtn.disabled = true;
        currentRecordingBlob = null;
        if (meters && progress[style]) meters.updateCoverage(progress[style]);
        renderHistory();
    }

    function renderHistory() {
        const styleHistory = ctrl.historyForStyle(ctrl.currentStyle);
        if (styleHistory.length === 0) {
            historyEl.innerHTML = '<div style="color:var(--text-2);font-size:13px">No recordings yet</div>';
            return;
        }
        historyEl.innerHTML = styleHistory.map((h, i) => {
            const icon = h.accepted ? '✓' : '✗';
            const cls = h.accepted ? 'color:var(--success)' : 'color:var(--error)';
            const meta = h.accepted
                ? `SNR: ${h.report.snr_db.toFixed(0)} dB &nbsp; ${h.report.total_duration_s.toFixed(1)}s`
                : h.report.rejection_reasons.join('; ');
            return `<div class="history-item">
                <span class="history-status" style="${cls}">${icon}</span>
                <span class="history-text">${esc(h.prompt)}</span>
                <span class="history-meta">${meta}</span>
                <button class="history-play" data-idx="${i}" data-accepted="${h.accepted}">▶</button>
            </div>`;
        }).join('');

        historyEl.querySelectorAll('.history-play').forEach(btn => {
            btn.onclick = () => {
                const idx = parseInt(btn.dataset.idx);
                const entry = styleHistory[idx];
                if (entry.accepted && entry.sampleIndex !== null) playback.playURL(ctrl.sampleAudioURL(entry.sampleIndex));
                else if (entry.blob) playback.playBlob(entry.blob);
            };
        });
    }

    function startMeterLoop() {
        if (meterAnimId) cancelAnimationFrame(meterAnimId);
        const loop = () => {
            if (!recording) return;
            meterAnimId = requestAnimationFrame(loop);
            const m = capture.getMetrics();
            if (meters && m) meters.updateLive(m);
            timeEl.textContent = formatTime(capture.elapsed);
        };
        meterAnimId = requestAnimationFrame(loop);
    }

    recordBtn.onclick = async () => {
        if (!recording) {
            recording = true;
            recordBtn.textContent = '■ Stop';
            recordBtn.className = 'btn btn-danger';
            playBtn.disabled = true;
            skipBtn.disabled = true;
            qualEl.innerHTML = '';
            waveform.start();
            capture.startRecording();
            meters.reset();
            startMeterLoop();
        } else {
            recording = false;
            recordBtn.textContent = '● Record';
            recordBtn.className = 'btn btn-primary';
            skipBtn.disabled = false;
            waveform.stop();
            currentRecordingBlob = capture.stopRecording();
            playBtn.disabled = false;

            qualEl.innerHTML = '<span class="spinner"></span> Validating...';
            try {
                const { result } = await ctrl.uploadSample(currentRecordingBlob);
                const r = result.quality_report;
                const metricsHtml = `SNR: ${r.snr_db.toFixed(1)} dB &nbsp; RMS: ${r.rms_dbfs.toFixed(1)} dBFS &nbsp; Speech: ${(r.speech_ratio * 100).toFixed(0)}% &nbsp; Duration: ${r.total_duration_s.toFixed(1)}s`;
                if (result.accepted) {
                    qualEl.innerHTML = `<div class="quality-result passed"><strong>Accepted</strong><div class="quality-metrics">${metricsHtml}</div></div>`;
                    meters.updateFromReport(r);
                    const style = ctrl.currentStyle || ctrl.session?.current_style;
                    if (ctrl.progress[style]) meters.updateCoverage(ctrl.progress[style]);
                } else {
                    qualEl.innerHTML = `<div class="quality-result failed"><strong>Rejected</strong><div class="rejection-reasons">${r.rejection_reasons.map(r => `<div>- ${r}</div>`).join('')}</div><div class="quality-metrics">${metricsHtml}</div></div>`;
                }
                renderPrompt();
            } catch (e) {
                qualEl.innerHTML = `<div class="quality-result failed">Upload error: ${esc(e.message)}</div>`;
            }
        }
    };

    playBtn.onclick = () => { if (currentRecordingBlob) playback.playBlob(currentRecordingBlob); };
    skipBtn.onclick = async () => { await ctrl.refreshSession(); renderPrompt(); };
    document.getElementById('btn-next-style').onclick = () => enrollNavigate('summary');
    document.getElementById('btn-record-back').onclick = () => enrollNavigate('consent');
}

// ══════════════════════════════════════════════════
// Enrollment: Summary
// ══════════════════════════════════════════════════

async function initSummaryScreen() {
    const container = document.getElementById('summary-content');
    const finalizeBtn = document.getElementById('btn-finalize');

    await ctrl.refreshSession();
    const progress = ctrl.progress;

    let html = `<div style="margin-bottom:16px;font-size:15px">Identity: <strong>${esc(selectedIdentity)}</strong></div>`;

    for (const style of selectedStyles) {
        const p = progress[style] || {};
        const styleHistory = ctrl.historyForStyle(style);
        const accepted = styleHistory.filter(h => h.accepted);
        const bestIdx = accepted.length > 0
            ? accepted.reduce((best, h, i) => h.report.snr_db > accepted[best].report.snr_db ? i : best, 0) : -1;

        html += `<div class="style-card"><div class="style-card-header">
            <span class="style-card-title">${esc(style)}</span>
            <span>Accepted: ${accepted.length} &nbsp; Coverage: ${(p.coverage_percent || 0).toFixed(0)}%</span>
        </div>
        ${accepted.map((h, i) => `<div class="history-item">
            <span class="history-status" style="color:var(--success)">✓</span>
            <span class="history-text">${esc(h.prompt)}</span>
            <span class="history-meta">SNR: ${h.report.snr_db.toFixed(0)} dB</span>
            ${i === bestIdx ? '<span class="best-badge">★ best</span>' : ''}
            <button class="history-play" onclick="window.__playAccepted(${h.sampleIndex})">▶</button>
        </div>`).join('')}
        </div>`;
    }

    container.innerHTML = html;
    window.__playAccepted = (idx) => { playback.playURL(ctrl.sampleAudioURL(idx)); };

    finalizeBtn.onclick = async () => {
        finalizeBtn.disabled = true;
        finalizeBtn.innerHTML = '<span class="spinner"></span> Finalizing...';
        try {
            const result = await ctrl.completeSession();
            container.innerHTML += `<div class="quality-result passed" style="margin-top:16px">
                <strong>Enrollment complete!</strong>
                <div>Styles registered: ${result.styles_registered.join(', ')}</div>
                <div style="margin-top:8px"><a href="/dashboard" style="color:var(--primary)">← Back to Dashboard</a></div>
            </div>`;
            finalizeBtn.style.display = 'none';
        } catch (e) {
            container.innerHTML += `<div class="quality-result failed" style="margin-top:16px">Error: ${esc(e.message)}</div>`;
            finalizeBtn.disabled = false;
            finalizeBtn.textContent = '✓ Finalize Enrollment';
        }
    };

    document.getElementById('btn-summary-back').onclick = () => enrollNavigate('record');
}

// ── Helpers ────────────────────────────────────────

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function esc(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}
