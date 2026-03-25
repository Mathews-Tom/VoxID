/**
 * Audio capture via Web Audio API + AudioWorklet.
 *
 * Captures raw PCM at 48 kHz mono with all browser audio processing
 * disabled (echoCancellation, noiseSuppression, autoGainControl = false).
 * Per Wildspoof 2026 findings, browser processing degrades speaker
 * similarity in voice cloning enrollment.
 */

const SAMPLE_RATE = 48000;
const SPEECH_THRESHOLD_DB = -35;
const SPEECH_THRESHOLD_LINEAR = Math.pow(10, SPEECH_THRESHOLD_DB / 20);

export class AudioCapture {
    constructor() {
        this._ctx = null;
        this._stream = null;
        this._source = null;
        this._analyser = null;
        this._workletNode = null;
        this._chunks = [];
        this._recording = false;
        this._startTime = 0;
        this._noiseFloor = null; // captured during noise phase
    }

    get analyser() { return this._analyser; }
    get sampleRate() { return SAMPLE_RATE; }
    get isRecording() { return this._recording; }

    get elapsed() {
        if (!this._recording) return 0;
        return (performance.now() - this._startTime) / 1000;
    }

    async init() {
        this._ctx = new AudioContext({ sampleRate: SAMPLE_RATE });

        this._stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            }
        });

        this._source = this._ctx.createMediaStreamSource(this._stream);

        this._analyser = this._ctx.createAnalyser();
        this._analyser.fftSize = 2048;
        this._analyser.smoothingTimeConstant = 0.3;
        this._source.connect(this._analyser);

        const workletUrl = new URL('../worklets/recorder-worklet.js', import.meta.url);
        await this._ctx.audioWorklet.addModule(workletUrl);
        this._workletNode = new AudioWorkletNode(this._ctx, 'recorder-processor');
        this._workletNode.port.onmessage = (e) => {
            if (this._recording) {
                this._chunks.push(new Float32Array(e.data));
            }
        };
    }

    startRecording() {
        this._chunks = [];
        this._recording = true;
        this._startTime = performance.now();
        this._source.connect(this._workletNode);
        this._workletNode.connect(this._ctx.destination); // needed for worklet to process
    }

    stopRecording() {
        this._recording = false;
        try {
            this._workletNode.disconnect();
        } catch { /* may already be disconnected */ }
        return this._getAudioBlob();
    }

    /** Get real-time metrics from the AnalyserNode */
    getMetrics() {
        if (!this._analyser) return null;

        const buf = new Float32Array(this._analyser.fftSize);
        this._analyser.getFloatTimeDomainData(buf);

        let sumSq = 0;
        let peak = 0;
        for (let i = 0; i < buf.length; i++) {
            const v = buf[i];
            sumSq += v * v;
            const abs = Math.abs(v);
            if (abs > peak) peak = abs;
        }

        const rms = Math.sqrt(sumSq / buf.length);
        const rmsDb = 20 * Math.log10(rms + 1e-9);
        const peakDb = 20 * Math.log10(peak + 1e-9);
        const isSpeech = rms > SPEECH_THRESHOLD_LINEAR;

        return { rmsDb, peakDb, isSpeech, elapsed: this.elapsed };
    }

    /** Get time-domain data for waveform rendering */
    getTimeDomainData(buffer) {
        if (!this._analyser) return;
        this._analyser.getByteTimeDomainData(buffer);
    }

    /** Capture noise floor from first 500ms of a recording */
    captureNoiseFloor() {
        const metrics = this.getMetrics();
        if (metrics) {
            this._noiseFloor = metrics.rmsDb;
        }
        return this._noiseFloor;
    }

    /** Approximate SNR using captured noise floor */
    getApproxSNR() {
        if (this._noiseFloor === null) return null;
        const metrics = this.getMetrics();
        if (!metrics) return null;
        return metrics.rmsDb - this._noiseFloor;
    }

    destroy() {
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
        }
        if (this._ctx && this._ctx.state !== 'closed') {
            this._ctx.close();
        }
    }

    /** Encode accumulated Float32 chunks as 16-bit PCM WAV Blob */
    _getAudioBlob() {
        const totalLength = this._chunks.reduce((s, c) => s + c.length, 0);
        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of this._chunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
        }
        return encodeWAV(merged, SAMPLE_RATE);
    }
}

/** Encode Float32Array as 16-bit PCM WAV Blob */
function encodeWAV(samples, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const dataLength = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);                    // chunk size
    view.setUint16(20, 1, true);                     // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
    view.setUint16(32, numChannels * bytesPerSample, true);
    view.setUint16(34, bitsPerSample, true);

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataLength, true);

    // PCM samples — Float32 → Int16 with clamping
    let writeOffset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(writeOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        writeOffset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}
