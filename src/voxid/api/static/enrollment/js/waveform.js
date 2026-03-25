/**
 * Canvas-based waveform renderer.
 *
 * Two modes:
 *   1. Live scrolling — fed by AnalyserNode during recording
 *   2. Static — renders a full audio buffer for playback review
 */

const COLORS = {
    safe: '#1ABC9C',
    hot:  '#F39C12',
    clip: '#E74C3C',
    bg:   '#1A1A2E',
    line: 'rgba(231, 76, 60, 0.4)',
    cursor: '#E8634A',
};

const CLIP_THRESHOLD_DB = -0.5;
const HOT_THRESHOLD_DB = -9;

export class LiveWaveform {
    constructor(canvas) {
        this._canvas = canvas;
        this._ctx = canvas.getContext('2d');
        this._animId = null;
        this._analyser = null;
        this._dataBuffer = null;
        this._history = [];
        this._maxHistory = 0;
    }

    attach(analyser) {
        this._analyser = analyser;
        this._dataBuffer = new Uint8Array(analyser.fftSize);
        this._resize();
        // Store enough columns to fill the canvas width
        this._maxHistory = this._canvas.width;
        this._history = [];
    }

    start() {
        if (this._animId) return;
        this._history = [];
        this._draw();
    }

    stop() {
        if (this._animId) {
            cancelAnimationFrame(this._animId);
            this._animId = null;
        }
    }

    clear() {
        this._history = [];
        const ctx = this._ctx;
        const w = this._canvas.width;
        const h = this._canvas.height;
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, w, h);
    }

    _resize() {
        const rect = this._canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this._canvas.width = rect.width * dpr;
        this._canvas.height = rect.height * dpr;
        this._ctx.scale(dpr, dpr);
        this._maxHistory = Math.floor(rect.width);
    }

    _draw() {
        this._animId = requestAnimationFrame(() => this._draw());
        if (!this._analyser) return;

        this._analyser.getByteTimeDomainData(this._dataBuffer);

        // Compute column min/max from current frame
        let min = 1, max = -1;
        for (let i = 0; i < this._dataBuffer.length; i++) {
            const v = (this._dataBuffer[i] / 128) - 1;
            if (v < min) min = v;
            if (v > max) max = v;
        }

        // Peak in dBFS for color selection
        const peak = Math.max(Math.abs(min), Math.abs(max));
        const peakDb = 20 * Math.log10(peak + 1e-9);

        let color = COLORS.safe;
        if (peakDb > CLIP_THRESHOLD_DB) color = COLORS.clip;
        else if (peakDb > HOT_THRESHOLD_DB) color = COLORS.hot;

        this._history.push({ min, max, color });
        if (this._history.length > this._maxHistory) {
            this._history.shift();
        }

        this._renderHistory();
    }

    _renderHistory() {
        const ctx = this._ctx;
        const rect = this._canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        const midY = h / 2;

        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, w, h);

        // Clipping threshold line
        const clipLinear = Math.pow(10, CLIP_THRESHOLD_DB / 20);
        const clipY1 = midY - clipLinear * midY;
        const clipY2 = midY + clipLinear * midY;
        ctx.strokeStyle = COLORS.line;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, clipY1);
        ctx.lineTo(w, clipY1);
        ctx.moveTo(0, clipY2);
        ctx.lineTo(w, clipY2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Waveform bars — draw from right to left (newest on right)
        const startX = w - this._history.length;
        for (let i = 0; i < this._history.length; i++) {
            const col = this._history[i];
            const x = startX + i;
            const y1 = midY + col.min * midY;
            const y2 = midY + col.max * midY;
            ctx.strokeStyle = col.color;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, y1);
            ctx.lineTo(x, y2);
            ctx.stroke();
        }
    }
}

/**
 * Static waveform for playback — renders a full buffer once.
 */
export class StaticWaveform {
    constructor(canvas) {
        this._canvas = canvas;
        this._ctx = canvas.getContext('2d');
        this._duration = 0;
        this._cursorPos = 0;
    }

    render(audioBuffer, sampleRate) {
        this._duration = audioBuffer.length / sampleRate;
        const rect = this._canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this._canvas.width = rect.width * dpr;
        this._canvas.height = rect.height * dpr;
        this._ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const midY = h / 2;
        const samplesPerPixel = Math.ceil(audioBuffer.length / w);

        this._ctx.fillStyle = COLORS.bg;
        this._ctx.fillRect(0, 0, w, h);

        for (let x = 0; x < w; x++) {
            const start = x * samplesPerPixel;
            const end = Math.min(start + samplesPerPixel, audioBuffer.length);
            let min = 1, max = -1;
            for (let i = start; i < end; i++) {
                const v = audioBuffer[i];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            const peak = Math.max(Math.abs(min), Math.abs(max));
            const peakDb = 20 * Math.log10(peak + 1e-9);
            let color = COLORS.safe;
            if (peakDb > CLIP_THRESHOLD_DB) color = COLORS.clip;
            else if (peakDb > HOT_THRESHOLD_DB) color = COLORS.hot;

            this._ctx.strokeStyle = color;
            this._ctx.lineWidth = 1;
            this._ctx.beginPath();
            this._ctx.moveTo(x, midY + min * midY);
            this._ctx.lineTo(x, midY + max * midY);
            this._ctx.stroke();
        }
    }
}
