// A small Markdown renderer for the transcript (VIS-19): paragraphs, headings, lists, quotes,
// rules, tables, fenced code with a language tag, inline code / bold / italic / links. Every
// character from the model is escaped first; the only HTML that comes out is what this file
// writes. An unclosed fence at the end (a stream in progress) renders as a code block anyway.

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const SAFE_URL = /^(https?:|mailto:|#|\/)/i;

export function inline(src: string): string {
  let s = escapeHtml(src);
  // code spans first so nothing inside them is styled
  const codes: string[] = [];
  s = s.replace(/`([^`\n]+)`/g, (_m, c: string) => {
    codes.push(`<code>${c}</code>`);
    return `\u0000${codes.length - 1}\u0000`;
  });
  s = s.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (_m, text: string, url: string) => (SAFE_URL.test(url) ? `<a href="${url}" rel="noopener noreferrer" target="_blank">${text}</a>` : `${text} (${url})`));
  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>').replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>').replace(/(^|[^_\w])_([^_\n]+)_(?!\w)/g, '$1<em>$2</em>');
  s = s.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');
  s = s.replace(/ {2,}\n/g, '<br>\n');
  s = s.replace(/\u0000(\d+)\u0000/g, (_m, i: string) => codes[Number(i)]);
  return s;
}

const FENCE = /^(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;

export function renderMarkdown(src: string): string {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const out: string[] = [];
  let i = 0;
  let para: string[] = [];
  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${inline(para.join('\n'))}</p>`);
      para = [];
    }
  };
  while (i < lines.length) {
    const line = lines[i];
    const fence = FENCE.exec(line);
    if (fence) {
      flushPara();
      const mark = fence[1];
      const lang = fence[2];
      const body: string[] = [];
      i++;
      while (i < lines.length && !(lines[i].startsWith(mark) && lines[i].trim() === mark)) body.push(lines[i++]);
      const closed = i < lines.length;
      if (closed) i++;
      out.push(`<div class="md-code${closed ? '' : ' is-open'}"${lang ? ` data-lang="${escapeHtml(lang)}"` : ''}><pre><code${lang ? ` class="lang-${escapeHtml(lang)}"` : ''}>${escapeHtml(body.join('\n'))}\n</code></pre></div>`);
      continue;
    }
    if (/^\s*$/.test(line)) {
      flushPara();
      i++;
      continue;
    }
    const h = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (h) {
      flushPara();
      const n = Math.min(6, h[1].length + 2); // the transcript's headings start two sizes down
      out.push(`<h${n}>${inline(h[2])}</h${n}>`);
      i++;
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      flushPara();
      out.push('<hr>');
      i++;
      continue;
    }
    if (/^\s*>/.test(line)) {
      flushPara();
      const q: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ''));
      out.push(`<blockquote>${renderMarkdown(q.join('\n'))}</blockquote>`);
      continue;
    }
    if (/^\s*[-*+]\s+/.test(line) || /^\s*\d+[.)]\s+/.test(line)) {
      flushPara();
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const re = ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*+]\s+/;
      const items: string[] = [];
      while (i < lines.length && re.test(lines[i])) {
        let item = lines[i++].replace(re, '');
        // continuation lines indented under the item
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !re.test(lines[i]) && !FENCE.test(lines[i])) item += '\n' + lines[i++].trim();
        items.push(`<li>${inline(item)}</li>`);
      }
      out.push(`<${ordered ? 'ol' : 'ul'}>${items.join('')}</${ordered ? 'ol' : 'ul'}>`);
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
      flushPara();
      const cells = (l: string) =>
        l
          .trim()
          .replace(/^\|/, '')
          .replace(/\|$/, '')
          .split('|')
          .map((c) => c.trim());
      const head = cells(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(
        `<div class="md-table"><table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${head.map((_c, k) => `<td>${inline(r[k] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`,
      );
      continue;
    }
    para.push(line);
    i++;
  }
  flushPara();
  return out.join('\n');
}
