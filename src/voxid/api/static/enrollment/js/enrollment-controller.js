/**
 * Enrollment controller — API client and session state manager.
 *
 * Wraps all /enroll/*, /identities endpoints and maintains local
 * state mirroring the server session.
 */

async function api(path, opts = {}) {
    const url = `/api${path}`;
    const res = await fetch(url, opts);
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

export class EnrollmentController {
    constructor() {
        this.session = null;
        this.identities = [];
        this.history = [];           // { prompt, style, accepted, report, blob, sampleIndex }
        this._rejectedBlobs = [];    // keep blobs for rejected recordings
    }

    // ── Identity API ──────────────────────────────────

    async fetchIdentities() {
        const data = await api('/identities');
        this.identities = data.identities;
        return this.identities;
    }

    async createIdentity(id, name, description) {
        return api('/identities', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id, name, description }),
        });
    }

    async getIdentity(id) {
        return api(`/identities/${id}`);
    }

    // ── Consent API ───────────────────────────────────

    async getConsentStatement(identityId) {
        return api(`/enroll/consent/statement?identity_id=${encodeURIComponent(identityId)}`);
    }

    async getConsentStatus(identityId) {
        return api(`/enroll/consent/${encodeURIComponent(identityId)}/status`);
    }

    async uploadConsent(identityId, audioBlob) {
        const form = new FormData();
        form.append('file', audioBlob, 'consent.wav');
        return api(`/enroll/consent/${encodeURIComponent(identityId)}`, {
            method: 'POST',
            body: form,
        });
    }

    // ── Session API ───────────────────────────────────

    async createSession(identityId, styles, promptsPerStyle = 5) {
        this.session = await api('/enroll/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                identity_id: identityId,
                styles,
                prompts_per_style: promptsPerStyle,
            }),
        });
        this.history = [];
        this._rejectedBlobs = [];
        return this.session;
    }

    async refreshSession() {
        if (!this.session) return null;
        this.session = await api(`/enroll/sessions/${this.session.session_id}`);
        return this.session;
    }

    async uploadSample(audioBlob) {
        const form = new FormData();
        form.append('file', audioBlob, 'sample.wav');
        const result = await api(`/enroll/sessions/${this.session.session_id}/samples`, {
            method: 'POST',
            body: form,
        });

        // Track history
        const entry = {
            prompt: this.session.current_prompt?.text || '—',
            style: this.session.current_style,
            accepted: result.accepted,
            report: result.quality_report,
            blob: result.accepted ? null : audioBlob,
            sampleIndex: result.accepted ? this._acceptedCount() : null,
        };
        this.history.push(entry);

        if (!result.accepted) {
            this._rejectedBlobs.push(audioBlob);
        }

        // Update local session state
        await this.refreshSession();
        return { result, entry };
    }

    async completeSession() {
        return api(`/enroll/sessions/${this.session.session_id}/complete`, {
            method: 'POST',
        });
    }

    // ── Sample Audio ──────────────────────────────────

    sampleAudioURL(sampleIndex) {
        return `/api/enroll/sessions/${this.session.session_id}/samples/${sampleIndex}/audio`;
    }

    // ── Helpers ───────────────────────────────────────

    _acceptedCount() {
        return this.history.filter(h => h.accepted).length;
    }

    get currentPrompt() {
        return this.session?.current_prompt;
    }

    get currentStyle() {
        return this.session?.current_style;
    }

    get isComplete() {
        return this.session?.current_prompt === null && this.session?.current_style === null;
    }

    get progress() {
        return this.session?.progress || {};
    }

    /** Get history entries for a specific style */
    historyForStyle(style) {
        return this.history.filter(h => h.style === style);
    }
}
