/**
 * AudioWorkletProcessor that buffers raw PCM frames from the mic.
 * Runs on the audio rendering thread — does minimal work (copy + post).
 */
class RecorderProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input[0] && input[0].length > 0) {
            this.port.postMessage(input[0].slice());
        }
        return true;
    }
}

registerProcessor('recorder-processor', RecorderProcessor);
