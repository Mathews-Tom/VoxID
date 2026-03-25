/**
 * Annotation parser and renderer.
 *
 * Parses bracket-syntax annotations in prompt text and renders
 * them as styled HTML spans with a dynamic legend.
 *
 * Syntax:
 *   [*word*]          → strong emphasis
 *   [~word~]          → moderate emphasis
 *   [.]               → short pause (0.3s)
 *   [..]              → long pause (0.8s)
 *   [breath]          → breath marker
 *   [>slow]...[/slow] → slow rate
 *   [>fast]...[/fast] → fast rate
 *   [_word_]          → drop (lower volume/pitch)
 *   [^word^]          → rise (upward intonation)
 *   [%word%]          → whisper
 */

const ANNOTATION_TYPES = {
    'emphasis-strong':   { label: 'strong emphasis',   swatch: 'emphasis-s' },
    'emphasis-moderate': { label: 'moderate emphasis',  swatch: 'emphasis-m' },
    'pause-short':       { label: 'short pause',       swatch: 'pause' },
    'pause-long':        { label: 'long pause',        swatch: 'pause' },
    'breath':            { label: 'breath',             swatch: 'pause' },
    'speed-slow':        { label: 'slow',               swatch: 'slow' },
    'speed-fast':        { label: 'fast',               swatch: 'fast' },
    'drop':              { label: 'drop',                swatch: 'drop' },
    'rise':              { label: 'rise',                swatch: 'rise' },
    'whisper':           { label: 'whisper',             swatch: 'whisper' },
};

/**
 * Parse annotated text into an array of segments.
 * Each segment is { type: 'text'|annotation-type, content: string }.
 */
export function parseAnnotations(text) {
    if (!text) return [{ type: 'text', content: text || '' }];

    const segments = [];
    let i = 0;
    let buf = '';

    while (i < text.length) {
        if (text[i] === '[') {
            // Flush buffered text
            if (buf) { segments.push({ type: 'text', content: buf }); buf = ''; }

            // Try to match annotation patterns
            const rest = text.slice(i);

            // [..] long pause (must check before [.])
            if (rest.startsWith('[..]')) {
                segments.push({ type: 'pause-long', content: '' });
                i += 4; continue;
            }
            // [.] short pause
            if (rest.startsWith('[.]')) {
                segments.push({ type: 'pause-short', content: '' });
                i += 3; continue;
            }
            // [breath]
            if (rest.startsWith('[breath]')) {
                segments.push({ type: 'breath', content: '' });
                i += 8; continue;
            }
            // [>slow]...[/slow]
            const slowMatch = rest.match(/^\[>slow\](.*?)\[\/slow\]/s);
            if (slowMatch) {
                segments.push({ type: 'speed-slow', content: slowMatch[1] });
                i += slowMatch[0].length; continue;
            }
            // [>fast]...[/fast]
            const fastMatch = rest.match(/^\[>fast\](.*?)\[\/fast\]/s);
            if (fastMatch) {
                segments.push({ type: 'speed-fast', content: fastMatch[1] });
                i += fastMatch[0].length; continue;
            }
            // [*word*] strong emphasis
            const strongMatch = rest.match(/^\[\*(.*?)\*\]/);
            if (strongMatch) {
                segments.push({ type: 'emphasis-strong', content: strongMatch[1] });
                i += strongMatch[0].length; continue;
            }
            // [~word~] moderate emphasis
            const modMatch = rest.match(/^\[~(.*?)~\]/);
            if (modMatch) {
                segments.push({ type: 'emphasis-moderate', content: modMatch[1] });
                i += modMatch[0].length; continue;
            }
            // [_word_] drop
            const dropMatch = rest.match(/^\[_(.*?)_\]/);
            if (dropMatch) {
                segments.push({ type: 'drop', content: dropMatch[1] });
                i += dropMatch[0].length; continue;
            }
            // [^word^] rise
            const riseMatch = rest.match(/^\[\^(.*?)\^\]/);
            if (riseMatch) {
                segments.push({ type: 'rise', content: riseMatch[1] });
                i += riseMatch[0].length; continue;
            }
            // [%word%] whisper
            const whisperMatch = rest.match(/^\[%(.*?)%\]/);
            if (whisperMatch) {
                segments.push({ type: 'whisper', content: whisperMatch[1] });
                i += whisperMatch[0].length; continue;
            }

            // No match — treat [ as literal
            buf += text[i];
            i++;
        } else {
            buf += text[i];
            i++;
        }
    }

    if (buf) segments.push({ type: 'text', content: buf });
    return segments;
}

/**
 * Render parsed segments into HTML string.
 */
export function renderAnnotatedHTML(segments) {
    return segments.map(seg => {
        if (seg.type === 'text') {
            return escapeHTML(seg.content);
        }
        const cls = `ann-${seg.type}`;
        if (seg.type === 'pause-short' || seg.type === 'pause-long') {
            return `<span class="${cls}"></span>`;
        }
        if (seg.type === 'breath') {
            return `<span class="${cls}"></span>`;
        }
        return `<span class="${cls}">${escapeHTML(seg.content)}</span>`;
    }).join('');
}

/**
 * Build legend HTML showing only annotation types present in the segments.
 */
export function renderLegend(segments) {
    const types = new Set(segments.filter(s => s.type !== 'text').map(s => s.type));
    if (types.size === 0) return '';

    const items = [...types].map(t => {
        const info = ANNOTATION_TYPES[t];
        if (!info) return '';
        return `<span class="legend-item">
            <span class="legend-swatch ${info.swatch}"></span>
            ${info.label}
        </span>`;
    }).join('');

    return `<div class="annotation-legend">${items}</div>`;
}

/**
 * Render a full script panel: annotated text + legend + direction notes.
 */
export function renderScriptPanel(prompt) {
    const annotatedText = prompt.annotated_text || prompt.text;
    const segments = parseAnnotations(annotatedText);
    const html = renderAnnotatedHTML(segments);
    const legend = renderLegend(segments);
    const notes = prompt.direction_notes
        ? `<div class="direction-notes">${escapeHTML(prompt.direction_notes)}</div>`
        : '';

    return `
        <div class="panel">
            <div class="panel-label">Script</div>
            <div class="script-text">${html}</div>
            ${legend}
            ${notes}
        </div>
    `;
}

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
