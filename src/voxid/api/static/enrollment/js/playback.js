/**
 * Audio playback from Blob (rejected recordings kept in memory)
 * or from server URL (accepted recordings).
 */

export class Playback {
    constructor() {
        this._audio = new Audio();
        this._currentUrl = null;
    }

    /** Play audio from a Blob (client-side, for rejected recordings) */
    playBlob(blob) {
        this._cleanup();
        this._currentUrl = URL.createObjectURL(blob);
        this._audio.src = this._currentUrl;
        this._audio.play();
    }

    /** Play audio from a server URL (for accepted recordings) */
    playURL(url) {
        this._cleanup();
        this._audio.src = url;
        this._audio.play();
    }

    stop() {
        this._audio.pause();
        this._audio.currentTime = 0;
    }

    get isPlaying() {
        return !this._audio.paused;
    }

    onEnded(callback) {
        this._audio.onended = callback;
    }

    _cleanup() {
        this.stop();
        if (this._currentUrl) {
            URL.revokeObjectURL(this._currentUrl);
            this._currentUrl = null;
        }
    }

    destroy() {
        this._cleanup();
    }
}
