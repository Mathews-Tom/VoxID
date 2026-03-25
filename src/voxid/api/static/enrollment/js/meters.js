/**
 * Level meters, SNR gauge, and speech ratio indicator.
 *
 * Renders into DOM elements using CSS classes from enrollment.css.
 * Update loop runs at ~15fps via requestAnimationFrame throttle.
 */

const DB_MIN = -60;
const DB_MAX = 0;
const HOT_DB = -9;
const CLIP_DB = -0.5;

/** Convert dBFS to a 0-1 fraction for meter bar width */
function dbToFraction(db) {
    return Math.max(0, Math.min(1, (db - DB_MIN) / (DB_MAX - DB_MIN)));
}

/** Get CSS class based on dB level */
function levelClass(db) {
    if (db > CLIP_DB) return 'clip';
    if (db > HOT_DB) return 'hot';
    return '';
}

export class Meters {
    constructor(container) {
        this._container = container;
        this._els = {};
        this._build();
    }

    _build() {
        this._container.innerHTML = `
            <div class="meters-row">
                <div class="panel" style="padding: 14px">
                    <div class="panel-label">Quality Meters</div>
                    <div class="meter">
                        <span class="meter-label">RMS</span>
                        <div class="meter-bar"><div class="meter-fill" id="m-rms-fill"></div></div>
                        <span class="meter-value" id="m-rms-val">— dBFS</span>
                    </div>
                    <div class="meter" style="margin-top:8px">
                        <span class="meter-label">Peak</span>
                        <div class="meter-bar"><div class="meter-fill" id="m-peak-fill"></div></div>
                        <span class="meter-value" id="m-peak-val">— dBFS</span>
                    </div>
                    <div class="meter" style="margin-top:8px">
                        <span class="meter-label">Speech</span>
                        <div class="meter-bar"><div class="meter-fill" id="m-speech-fill" style="background:var(--success)"></div></div>
                        <span class="meter-value" id="m-speech-val">—</span>
                        <span class="speech-dot" id="m-speech-dot"></span>
                    </div>
                    <div class="meter" style="margin-top:8px">
                        <span class="meter-label">SNR</span>
                        <div class="meter-bar"><div class="meter-fill" id="m-snr-fill" style="background:var(--success)"></div></div>
                        <span class="meter-value" id="m-snr-val">— dB</span>
                    </div>
                </div>
                <div class="panel" style="padding: 14px">
                    <div class="panel-label">Phoneme Coverage</div>
                    <div id="m-coverage-pct" style="font-size:24px;font-weight:600;margin-bottom:4px">—%</div>
                    <div class="coverage-bar"><div class="coverage-fill" id="m-coverage-fill"></div></div>
                    <div class="coverage-missing" id="m-coverage-missing" style="margin-top:8px">—</div>
                    <div style="margin-top:12px">
                        <div class="meter" style="margin-top:4px">
                            <span class="meter-label">Nasals</span>
                            <div class="meter-bar"><div class="meter-fill" id="m-nasals-fill" style="background:var(--warning)"></div></div>
                            <span class="meter-value" id="m-nasals-val">—</span>
                        </div>
                        <div class="meter" style="margin-top:4px">
                            <span class="meter-label">Affric</span>
                            <div class="meter-bar"><div class="meter-fill" id="m-affric-fill" style="background:var(--warning)"></div></div>
                            <span class="meter-value" id="m-affric-val">—</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this._els = {
            rmsFill:   this._container.querySelector('#m-rms-fill'),
            rmsVal:    this._container.querySelector('#m-rms-val'),
            peakFill:  this._container.querySelector('#m-peak-fill'),
            peakVal:   this._container.querySelector('#m-peak-val'),
            speechFill:this._container.querySelector('#m-speech-fill'),
            speechVal: this._container.querySelector('#m-speech-val'),
            speechDot: this._container.querySelector('#m-speech-dot'),
            snrFill:   this._container.querySelector('#m-snr-fill'),
            snrVal:    this._container.querySelector('#m-snr-val'),
            covPct:    this._container.querySelector('#m-coverage-pct'),
            covFill:   this._container.querySelector('#m-coverage-fill'),
            covMissing:this._container.querySelector('#m-coverage-missing'),
            nasalsFill:this._container.querySelector('#m-nasals-fill'),
            nasalsVal: this._container.querySelector('#m-nasals-val'),
            affricFill:this._container.querySelector('#m-affric-fill'),
            affricVal: this._container.querySelector('#m-affric-val'),
        };
    }

    /** Update live meters from AudioCapture metrics */
    updateLive(metrics) {
        if (!metrics) return;

        const { rmsDb, peakDb, isSpeech } = metrics;

        this._els.rmsFill.style.width = `${dbToFraction(rmsDb) * 100}%`;
        this._els.rmsFill.className = `meter-fill ${levelClass(rmsDb)}`;
        this._els.rmsVal.textContent = `${rmsDb.toFixed(1)} dBFS`;

        this._els.peakFill.style.width = `${dbToFraction(peakDb) * 100}%`;
        this._els.peakFill.className = `meter-fill ${levelClass(peakDb)}`;
        this._els.peakVal.textContent = `${peakDb.toFixed(1)} dBFS`;

        this._els.speechDot.className = `speech-dot ${isSpeech ? 'active' : ''}`;
    }

    /** Update speech ratio (0-1) */
    updateSpeechRatio(ratio) {
        this._els.speechFill.style.width = `${ratio * 100}%`;
        this._els.speechVal.textContent = `${(ratio * 100).toFixed(0)}%`;
    }

    /** Update SNR display */
    updateSNR(snrDb) {
        if (snrDb === null) return;
        const frac = Math.min(1, Math.max(0, snrDb / 60));
        this._els.snrFill.style.width = `${frac * 100}%`;
        this._els.snrVal.textContent = `${snrDb.toFixed(0)} dB`;

        if (snrDb < 20) this._els.snrFill.style.background = 'var(--error)';
        else if (snrDb < 30) this._els.snrFill.style.background = 'var(--warning)';
        else this._els.snrFill.style.background = 'var(--success)';
    }

    /** Update phoneme coverage from server progress data */
    updateCoverage(progress) {
        if (!progress) return;
        // progress is the session.progress_summary() for current style
        const pct = progress.coverage_percent ?? 0;
        this._els.covPct.textContent = `${pct.toFixed(0)}%`;
        this._els.covFill.style.width = `${pct}%`;

        const missing = progress.missing_phonemes ?? [];
        this._els.covMissing.textContent = missing.length > 0
            ? `Missing: ${missing.join(', ')}`
            : 'All phonemes covered';

        // Nasals and affricates
        const nc = progress.nasals_covered ?? 0;
        const nt = progress.nasals_target ?? 6;
        this._els.nasalsFill.style.width = `${Math.min(100, (nc / nt) * 100)}%`;
        this._els.nasalsVal.textContent = `${nc}/${nt}`;

        const ac = progress.affricates_covered ?? 0;
        const at = progress.affricates_target ?? 4;
        this._els.affricFill.style.width = `${Math.min(100, (ac / at) * 100)}%`;
        this._els.affricVal.textContent = `${ac}/${at}`;
    }

    /** Update from server quality report */
    updateFromReport(report) {
        this.updateLive({ rmsDb: report.rms_dbfs, peakDb: report.peak_dbfs, isSpeech: true });
        this.updateSpeechRatio(report.speech_ratio);
        this.updateSNR(report.snr_db);
    }

    reset() {
        this._els.rmsFill.style.width = '0%';
        this._els.rmsVal.textContent = '— dBFS';
        this._els.peakFill.style.width = '0%';
        this._els.peakVal.textContent = '— dBFS';
        this._els.speechDot.className = 'speech-dot';
        this._els.speechVal.textContent = '—';
        this._els.speechFill.style.width = '0%';
        this._els.snrFill.style.width = '0%';
        this._els.snrVal.textContent = '— dB';
    }
}
