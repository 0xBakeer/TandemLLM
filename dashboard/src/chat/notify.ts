// VIS-109 (plan note Chat 8, decision D14): local notifications only. The Notification API shows
// "<chat title>" with "finished", "waits for permission" or "error" when a run ends or asks while
// this tab is hidden or on another chat, after the user allowed notifications once; never message
// text, because a lock screen is public. No Web Push: nothing leaves the box. The app badge and
// the tab title count the chats waiting for permission and those with an unread answer.

export type NotifyKind = 'finished' | 'permission' | 'error';
export type NotifyPrefs = Record<NotifyKind, boolean>;

const KEY = 'qse.chat.notify';
export const NOTIFY_WORDS: Record<NotifyKind, string> = { finished: 'finished', permission: 'waits for permission', error: 'error' };

export function notifyPrefs(): NotifyPrefs {
  try {
    return { finished: true, permission: true, error: true, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') };
  } catch {
    return { finished: true, permission: true, error: true };
  }
}

export function setNotifyPrefs(p: NotifyPrefs): void {
  localStorage.setItem(KEY, JSON.stringify(p));
}

/** 'granted', 'denied', 'default', or 'unsupported' (iOS outside the installed app, plain http) */
export function notifyState(): NotificationPermission | 'unsupported' {
  return typeof Notification === 'undefined' || !window.isSecureContext ? 'unsupported' : Notification.permission;
}

/** what a global event means for this device, or null */
export function notifyKind(ev: { type: string; status?: string }): NotifyKind | null {
  if (ev.type === 'tool.asking') return 'permission';
  if (ev.type !== 'run.done') return null;
  return ev.status === 'error' ? 'error' : ev.status === 'done' ? 'finished' : null;
}

export async function notify(kind: NotifyKind, chat: { id: string; title: string }, here: boolean): Promise<void> {
  if (notifyState() !== 'granted' || !notifyPrefs()[kind] || (here && !document.hidden)) return;
  const opts: NotificationOptions = { body: NOTIFY_WORDS[kind], tag: `${chat.id}:${kind}`, icon: './favicon.svg', data: { url: `./#/chat/${encodeURIComponent(chat.id)}` } };
  // Android Chrome only shows them through the service worker; elsewhere the page can
  const reg = 'serviceWorker' in navigator ? await navigator.serviceWorker.getRegistration().catch(() => undefined) : undefined;
  if (reg) await reg.showNotification(chat.title, opts);
  else new Notification(chat.title, opts);
}

let baseTitle: string | null = null;

/** the app badge and the tab title: chats waiting for permission plus unread answers */
export function badge(n: number): void {
  baseTitle ??= document.title.replace(/^\(\d+\) /, '');
  document.title = n ? `(${n}) ${baseTitle}` : baseTitle;
  const nav = navigator as Navigator & { setAppBadge?: (n: number) => Promise<void>; clearAppBadge?: () => Promise<void> };
  void (n ? nav.setAppBadge?.(n) : nav.clearAppBadge?.())?.catch(() => undefined);
}
