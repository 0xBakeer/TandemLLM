// the transcript's Markdown renderer: escaping, fences, inline, lists, links, tables.
import { describe, expect, it } from 'vitest';
import { inline, renderMarkdown } from '../src/playground/markdown';

describe('escaping', () => {
  it('nothing from the model is inserted as HTML', () => {
    const h = renderMarkdown('<script>alert(1)</script> & <b>x</b>');
    expect(h).not.toContain('<script');
    expect(h).toContain('&lt;script&gt;');
    expect(h).toContain('&amp;');
  });
  it('a javascript: link is not a link', () => {
    expect(inline('[x](javascript:alert(1))')).not.toContain('<a');
    expect(inline('[x](https://a.b/c)')).toBe('<a href="https://a.b/c" rel="noopener noreferrer" target="_blank">x</a>');
  });
});

describe('blocks', () => {
  it('a closed fence with a language tag', () => {
    const h = renderMarkdown('before\n\n```python\nprint("<hi>")\n```\n\nafter');
    expect(h).toContain('<div class="md-code" data-lang="python"><pre><code class="lang-python">print(&quot;&lt;hi&gt;&quot;)\n</code></pre></div>');
    expect(h).toContain('<p>before</p>');
    expect(h).toContain('<p>after</p>');
  });
  it('an unclosed fence (streaming) still renders as a code block, marked open', () => {
    const h = renderMarkdown('```js\nlet a = 1');
    expect(h).toContain('md-code is-open');
    expect(h).toContain('let a = 1');
  });
  it('headings start two sizes down, lists, quotes, rules, tables', () => {
    const h = renderMarkdown('# Title\n\n- one\n- two\n\n1. a\n2. b\n\n> quoted\n\n---\n\n| k | v |\n|---|---|\n| a | 1 |\n');
    expect(h).toContain('<h3>Title</h3>');
    expect(h).toContain('<ul><li>one</li><li>two</li></ul>');
    expect(h).toContain('<ol><li>a</li><li>b</li></ol>');
    expect(h).toContain('<blockquote><p>quoted</p></blockquote>');
    expect(h).toContain('<hr>');
    expect(h).toContain('<th>k</th><th>v</th>');
    expect(h).toContain('<td>a</td><td>1</td>');
  });
  it('paragraph inline: code, bold, italic, strike, hard break', () => {
    expect(inline('use `a<b` and **bold** and *it* and ~~no~~')).toBe('use <code>a&lt;b</code> and <strong>bold</strong> and <em>it</em> and <del>no</del>');
    expect(inline('a  \nb')).toBe('a<br>\nb');
    expect(inline('snake_case_word stays')).toBe('snake_case_word stays');
  });
});
