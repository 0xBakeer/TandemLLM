// Theme: follows prefers-color-scheme until a choice is made; the choice is remembered.

export type Theme = 'dark' | 'light';
const KEY = 'qse.theme';

export function currentTheme(): Theme {
  const t = document.documentElement.getAttribute('data-theme');
  return t === 'light' ? 'light' : 'dark';
}

export function setTheme(t: Theme): void {
  document.documentElement.setAttribute('data-theme', t);
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* private mode */
  }
  window.dispatchEvent(new CustomEvent('qse-theme', { detail: t }));
}

export function toggleTheme(): Theme {
  const next: Theme = currentTheme() === 'dark' ? 'light' : 'dark';
  setTheme(next);
  return next;
}

/** Re-follow the OS when nothing was chosen (the inline script in index.html does the first paint). */
export function watchSystemTheme(): void {
  const mq = matchMedia('(prefers-color-scheme: light)');
  mq.addEventListener('change', (e) => {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem(KEY);
    } catch {
      /* ignore */
    }
    if (stored !== 'dark' && stored !== 'light') {
      document.documentElement.setAttribute('data-theme', e.matches ? 'light' : 'dark');
      window.dispatchEvent(new CustomEvent('qse-theme', { detail: e.matches ? 'light' : 'dark' }));
    }
  });
}
