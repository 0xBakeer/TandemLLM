// Community weights (CONTRACT.md 2026-10-13, "API (as built)"): the adapter from the API's entries
// to profiles, the card model, the search and the sort behind the "Community weights" section of
// the weights step. Pure, no lit: community-view.ts renders it.
//
// The API lists a model's packs in `community` as entries, not profiles. The page makes one profile
// per entry (withCommunityProfiles): id = entry.profile ("community/<id>"), label = the name, the
// memory figures the server computed from our own profile of the same format and nearest bpw
// (`memory.like`), and the rest of that profile (its largest window, its tools' figures). So the
// reducer, the estimate, Review, the plan and Wren's select_profile treat a pack like one of ours.
//
// Every entry is fetched from GitHub, so every string is untrusted: the card model keeps text as
// text (lit renders interpolations escaped) and makes a link only from an https URL; the avatar
// only from github.com.

import type { CommunityEntry, SetupCommunityDrafters, SetupDevice, SetupModality, SetupModel, SetupProfile, SetupSelection, SetupSocialNetwork } from '../api/types';

export const COMMUNITY_PREFIX = 'community/';
export const DOCS_URL = 'https://github.com/0xBakeer/TandemLLM/blob/main/docs/community-weights.md';

export function isCommunity(p: Pick<SetupProfile, 'id' | 'community'>): boolean {
  return !!p.community || p.id.startsWith(COMMUNITY_PREFIX);
}
/** our own weight profiles: the radio cards */
export function ownProfiles(m: Pick<SetupModel, 'profiles'>): SetupProfile[] {
  return m.profiles.filter((p) => !isCommunity(p));
}
/** the packs' profiles, as withCommunityProfiles made them */
export function communityProfiles(m: Pick<SetupModel, 'profiles'>): SetupProfile[] {
  return m.profiles.filter((p) => !!p.community);
}

/** the profile a pack stands for: our profile the server's memory figures came from, with the
 *  pack's own identity, size and figures over it */
export function profileOfEntry(m: Pick<SetupModel, 'profiles'>, e: CommunityEntry): SetupProfile {
  const own = ownProfiles(m);
  const like = own.find((p) => p.id === e.memory.like) ?? own.find((p) => p.format === e.format) ?? own[0];
  return {
    ...like,
    id: e.profile,
    label: e.name,
    format: e.format,
    bpw: e.bpw,
    disk_gb: e.size_gb,
    weights_gib: e.memory.weights_gib,
    drafters_gib: e.memory.drafters_gib,
    hf_repo: e.hf.repo,
    hf_revision: e.hf.revision,
    hf_private: false,
    recommended: false,
    badges: [],
    quality: { kl_vs_bf16: null, top1_pct: null, ppl: null, date: e.added, source: 'not measured against BF16' },
    note: null,
    // the drafters' files beside the pack; none when the pack runs without drafters
    assets_required: e.drafters === 'none' ? [] : (like?.assets_required ?? []),
    community: e,
  };
}

/** the model with one profile per community entry after our own (the API's `community` list, or
 *  `entries` from a sync); a model without the list is returned as is */
export function withCommunityProfiles(m: SetupModel, entries: CommunityEntry[] | undefined = m.community): SetupModel {
  if (!entries) return m;
  const own = ownProfiles(m);
  return { ...m, community: entries, profiles: [...own, ...entries.filter((e) => e.profile.startsWith(COMMUNITY_PREFIX)).map((e) => profileOfEntry({ profiles: own }, e))] };
}

/** true when the selection's pack cannot run images and the Vision tool is still on: the server
 *  turns it off in the checked selection, so the page does the same */
export function packVisionOff(m: Pick<SetupModel, 'profiles' | 'components'>, sel: Pick<SetupSelection, 'profile' | 'components'>): boolean {
  const p = m.profiles.find((x) => x.id === sel.profile);
  return !!p?.community && !p.community.modalities.image && m.components.some((c) => c.id === 'vision') && sel.components.vision !== false;
}
/** the pack the selection names cannot run images (the Vision tool stays off) */
export function packNoImage(m: Pick<SetupModel, 'profiles'>, profile: string): boolean {
  const p = m.profiles.find((x) => x.id === profile);
  return !!p?.community && !p.community.modalities.image;
}

// ---- links (untrusted input) --------------------------------------------------------------------

/** an https URL or null; anything else (javascript:, data:, http:) is dropped */
export function httpsUrl(u: unknown): string | null {
  if (typeof u !== 'string') return null;
  try {
    const url = new URL(u);
    return url.protocol === 'https:' ? url.href : null;
  } catch {
    return null;
  }
}
/** the avatar comes from github.com alone */
export function avatarUrl(u: unknown): string | null {
  const url = httpsUrl(u);
  return url && url.startsWith('https://github.com/') ? url : null;
}

/** two letters for the avatar's fallback: "thewhitemage." → "TH", "0xWhiteMage" → "0X" */
export function initials(name: string | null, login: string): string {
  const src = (name || login).trim();
  const words = src.split(/[\s._-]+/).filter(Boolean);
  const two = words.length >= 2 ? words[0][0] + words[1][0] : src.replace(/[^A-Za-z0-9]/g, '').slice(0, 2);
  return (two || '?').toUpperCase();
}

export interface SocialLink {
  kind: SetupSocialNetwork;
  /** the link's name for assistive tech and the tooltip */
  label: string;
  href: string;
  /** what the search matches: the handle or the host */
  handle: string;
}

const NETS: SetupSocialNetwork[] = ['x', 'huggingface', 'bluesky', 'mastodon', 'website'];
const NET_NAME: Record<SetupSocialNetwork, string> = { x: 'X', huggingface: 'Hugging Face', bluesky: 'Bluesky', mastodon: 'Mastodon', website: 'website' };

/** the server's ready-made links, https only, in the fixed order, with a label when it gave none */
export function socialLinks(e: Pick<CommunityEntry, 'author' | 'social_links'>): SocialLink[] {
  const out: SocialLink[] = [];
  for (const kind of NETS) {
    const l = (e.social_links ?? []).find((x) => x.network === kind);
    const href = l && httpsUrl(l.url);
    if (!href) continue;
    const own = kind === 'mastodon' || kind === 'website' ? null : e.author.social?.[kind];
    const handle = (own ?? href.slice(8)).replace(/^@/, '').replace(/\/$/, '');
    const who = e.author.name || e.author.github;
    const label = l.label || (kind === 'mastodon' ? `${who} on Mastodon` : kind === 'website' ? `${who}'s website` : `@${handle} on ${NET_NAME[kind]}`);
    out.push({ kind, label, href, handle });
  }
  return out;
}

// ---- the card -----------------------------------------------------------------------------------

export const MODALITIES: SetupModality[] = ['text', 'image', 'audio', 'video'];
export const MODALITY_LABEL: Record<SetupModality, string> = { text: 'Text', image: 'Image', audio: 'Voice', video: 'Video' };

export interface ModalityState {
  id: SetupModality;
  label: string;
  /** false: in the pack, but TandemLLM cannot run it yet (greyed, with `why`) */
  runs: boolean;
  why: string;
}

/** the modalities the pack contains, in the fixed order, each with whether it runs here */
export function modalityStates(e: Pick<CommunityEntry, 'modalities' | 'modalities_in_pack' | 'load'>): ModalityState[] {
  const inPack = new Set(e.modalities_in_pack ?? []);
  for (const [k, v] of Object.entries(e.modalities)) if (v) inPack.add(k as SetupModality);
  return MODALITIES.filter((id) => inPack.has(id)).map((id) => {
    const runs = !!e.modalities[id];
    const label = MODALITY_LABEL[id];
    return { id, label, runs, why: runs ? `${label}: runs in TandemLLM` : `${label}: in the pack, not supported for ${e.load} packs yet` };
  });
}

export function draftersText(d: SetupCommunityDrafters): string {
  if (d === 'tandem-exl3') return 'Uses TandemLLM drafters (K3 + K4)';
  if (d === 'dflash2') return 'Uses DFlash2 drafters';
  return 'No drafters: one token a pass';
}

export interface SpeedCell {
  task: 'prose' | 'code' | 'chat';
  tok_s: number;
}

export interface CommunityCard {
  /** the profile id, community/<id> */
  id: string;
  slug: string;
  name: string;
  description: string;
  finetuneOf: string | null;
  /** the fine-tune source on Hugging Face, when the id looks like one */
  finetuneUrl: string | null;
  author: { login: string; name: string | null; url: string | null; avatar: string | null; initials: string; social: SocialLink[] };
  format: string;
  bpw: number | null;
  sizeGb: number;
  load: 'standalone' | 'overlay';
  modalities: ModalityState[];
  drafters: string;
  license: string;
  licenseUrl: string | null;
  hfUrl: string | null;
  hfRepo: string;
  prUrl: string | null;
  status: CommunityEntry['status'];
  verified: { build: string; device: string; date: string; notes: string | null; speeds: SpeedCell[] } | null;
  tags: string[];
  added: string;
  /** the profile the Use button selects (its memory figures feed the bar) */
  profile: SetupProfile;
  /** the searched text, lower case */
  haystack: string;
}

const HF_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+$/;

export function cardOf(p: SetupProfile): CommunityCard | null {
  const e = p.community;
  if (!e) return null;
  const social = socialLinks(e);
  const speeds: SpeedCell[] = [];
  if (e.verified) for (const task of ['prose', 'code', 'chat'] as const) if (typeof e.verified.tok_s[task] === 'number') speeds.push({ task, tok_s: e.verified.tok_s[task]! });
  const parts = [e.name, e.author.github, `@${e.author.github}`, e.author.name ?? '', ...social.map((s) => s.handle), ...(e.tags ?? []), e.finetune_of ?? '', e.format, e.description ?? ''];
  return {
    id: p.id,
    slug: e.id,
    name: e.name,
    description: e.description ?? '',
    finetuneOf: e.finetune_of,
    finetuneUrl: e.finetune_of && HF_ID.test(e.finetune_of) ? `https://huggingface.co/${e.finetune_of}` : null,
    author: { login: e.author.github, name: e.author.name, url: httpsUrl(e.author_url), avatar: avatarUrl(e.avatar_url), initials: initials(e.author.name, e.author.github), social },
    format: e.format,
    bpw: e.bpw,
    sizeGb: e.size_gb,
    load: e.load,
    modalities: modalityStates(e),
    drafters: draftersText(e.drafters),
    license: e.license,
    licenseUrl: httpsUrl(e.license_url),
    hfUrl: httpsUrl(e.hf_url) ?? (HF_ID.test(e.hf.repo) ? `https://huggingface.co/${e.hf.repo}` : null),
    hfRepo: e.hf.repo,
    prUrl: httpsUrl(e.links?.pr),
    status: e.status,
    verified: e.verified ? { build: e.verified.build, device: e.verified.device, date: e.verified.date, notes: e.verified.notes ?? null, speeds } : null,
    tags: e.tags ?? [],
    added: e.added,
    profile: p,
    haystack: parts.join('\n').toLowerCase(),
  };
}

export function cardsOf(m: Pick<SetupModel, 'profiles'>): CommunityCard[] {
  return communityProfiles(m)
    .map(cardOf)
    .filter((c): c is CommunityCard => !!c);
}

// ---- search and sort ----------------------------------------------------------------------------

/** every word of the query must appear somewhere in the card's name, author, handles, tags,
 *  fine-tune source, format or description; a leading @ is ignored */
export function searchCards(cards: CommunityCard[], query: string): CommunityCard[] {
  const words = query
    .toLowerCase()
    .split(/\s+/)
    .map((w) => w.replace(/^@/, ''))
    .filter(Boolean);
  if (!words.length) return cards;
  return cards.filter((c) => words.every((w) => c.haystack.includes(w)));
}

export type SortKey = 'newest' | 'name' | 'size' | 'bpw' | 'speed';
export const SORT_KEYS: SortKey[] = ['newest', 'name', 'size', 'bpw', 'speed'];
export const SORT_LABEL: Record<SortKey, string> = { newest: 'Newest', name: 'Name', size: 'Smallest download', bpw: 'Most bits a weight', speed: 'Fastest prose' };

/** the speed a card sorts by: prose, else the best measured task; null when not verified */
export function speedOf(c: CommunityCard): number | null {
  if (!c.verified || !c.verified.speeds.length) return null;
  return c.verified.speeds.find((s) => s.task === 'prose')?.tok_s ?? Math.max(...c.verified.speeds.map((s) => s.tok_s));
}

export function sortCards(cards: CommunityCard[], key: SortKey): CommunityCard[] {
  const byName = (a: CommunityCard, b: CommunityCard) => a.name.localeCompare(b.name, 'en', { sensitivity: 'base' });
  const out = [...cards];
  switch (key) {
    case 'name':
      return out.sort(byName);
    case 'size':
      return out.sort((a, b) => a.sizeGb - b.sizeGb || byName(a, b));
    case 'bpw':
      return out.sort((a, b) => (b.bpw ?? -1) - (a.bpw ?? -1) || byName(a, b));
    case 'speed':
      // measured first, fastest first; the unmeasured keep their order by name
      return out.sort((a, b) => {
        const sa = speedOf(a);
        const sb = speedOf(b);
        if (sa == null && sb == null) return byName(a, b);
        if (sa == null) return 1;
        if (sb == null) return -1;
        return sb - sa || byName(a, b);
      });
    default:
      return out.sort((a, b) => b.added.localeCompare(a.added) || byName(a, b));
  }
}

export function isSortKey(v: unknown): v is SortKey {
  return typeof v === 'string' && (SORT_KEYS as string[]).includes(v);
}

/** "1 pack", "2 of 3 packs" */
export function countText(shown: number, total: number): string {
  const packs = (n: number) => `${n} ${n === 1 ? 'pack' : 'packs'}`;
  return shown === total ? packs(total) : `${shown} of ${packs(total)}`;
}

/** what the card's foot says about the box: running now, on disk, or the download */
export function boxState(p: SetupProfile, dev: SetupDevice | null): 'running' | 'on-disk' | 'download' {
  if (dev?.active_profile === p.id) return 'running';
  if (dev?.profiles_on_disk?.[p.id]) return 'on-disk';
  return 'download';
}
