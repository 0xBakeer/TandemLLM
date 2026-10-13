// Community weights (CONTRACT.md 2026-10-13, "API (as built)"): the adapter from entries to
// profiles, the card model, the untrusted-text rules, the search and the sort, and the Vision rule,
// against the mock's three entries (ThinkingCap and two marked mock entries).
import { describe, expect, it } from 'vitest';
import community from '../mock/setup/community-qwen3.8-27b.json';
import model from '../mock/setup/qwen3.8-27b.json';
import type { CommunityEntry, CommunitySync, SetupDevice, SetupModel } from '../src/api/types';
import { howToFix } from '../src/setup/checks';
import { avatarUrl, boxState, cardOf, cardsOf, communityProfiles, countText, httpsUrl, initials, isCommunity, modalityStates, ownProfiles, packVisionOff, profileOfEntry, searchCards, socialLinks, SORT_KEYS, sortCards, speedOf, withCommunityProfiles } from '../src/setup/community';
import { checkSelection, estimate } from '../src/setup/estimate';
import { defaults, emptyState, restoreSelection, setupReducer } from '../src/setup/state';

const OWN = model as unknown as SetupModel;
const C = community as unknown as { community: CommunityEntry[]; community_sync: CommunitySync };
const ENTRIES = C.community;
const M = withCommunityProfiles({ ...OWN, community: ENTRIES, community_sync: C.community_sync });
const thinking = ENTRIES[0];
const cards = cardsOf(M);
const byId = (id: string) => cards.find((c) => c.id === id)!;
const TC = 'community/thinkingcap-27b-kearuga-exl3-4.00';

describe('entries become profiles', () => {
  it('one profile per entry after our own, from the profile the server took the memory figures from', () => {
    expect(ownProfiles(M).map((p) => p.id)).toEqual(OWN.profiles.map((p) => p.id));
    expect(communityProfiles(M).map((p) => p.id)).toEqual(ENTRIES.map((e) => e.profile));
    const p = profileOfEntry(OWN, thinking);
    const like = OWN.profiles.find((x) => x.id === 'exl3-4.0')!;
    expect(p).toMatchObject({ id: TC, label: thinking.name, format: 'exl3', bpw: 4, disk_gb: 16.4, weights_gib: thinking.memory.weights_gib, drafters_gib: thinking.memory.drafters_gib, hf_repo: thinking.hf.repo, hf_revision: thinking.hf.revision, hf_private: false, recommended: false, badges: [] });
    expect(p.assets_required).toEqual(like.assets_required);
    expect(p.quality.kl_vs_bf16).toBeNull();
    expect(isCommunity(p)).toBe(true);
    expect(isCommunity(OWN.profiles[0])).toBe(false);
  });
  it('a pack without drafters needs no drafter files', () => {
    expect(profileOfEntry(OWN, ENTRIES[1]).assets_required).toEqual([]);
    expect(profileOfEntry(OWN, ENTRIES[1]).drafters_gib).toBe(0);
  });
  it('a model without the list is returned as is; a sync replaces the list', () => {
    expect(withCommunityProfiles(OWN)).toBe(OWN);
    const one = withCommunityProfiles(M, [ENTRIES[2]]);
    expect(communityProfiles(one).map((p) => p.id)).toEqual([ENTRIES[2].profile]);
    expect(ownProfiles(one).length).toBe(OWN.profiles.length);
  });
  it('the reducer selects a pack and the estimate counts its weights', () => {
    const st = setupReducer(setupReducer(emptyState(), { type: 'open', model: M }, M), { type: 'profile', id: TC }, M);
    expect(st.selection!.profile).toBe(TC);
    const e = estimate(M, st.selection!);
    expect(e.parts.find((p) => p.id === 'weights')!.gb).toBeCloseTo(thinking.memory.weights_gib, 2);
  });
  it('a memory fix never proposes a community pack', () => {
    const sel = { ...defaults(M), profile: 'exl3-6.0', slots: 4 };
    for (const f of howToFix(M, sel)) expect(f.text).not.toMatch(/ThinkingCap|Mock/);
  });
});

describe('the Vision rule: a pack that cannot run images turns the tool off', () => {
  it('in the reducer, with the note, and the tool comes back with our own weights by hand', () => {
    const open = setupReducer(emptyState(), { type: 'open', model: M }, M);
    expect(open.selection!.components.vision).toBe(true);
    expect(packVisionOff(M, { profile: TC, components: open.selection!.components })).toBe(true);
    const st = setupReducer(open, { type: 'profile', id: TC }, M);
    expect(st.selection!.components.vision).toBe(false);
    expect(st.notice).toMatch(/Vision is off/);
    const back = setupReducer(st, { type: 'profile', id: 'exl3-4.5' }, M);
    expect(back.selection!.components.vision).toBe(false);
    expect(packVisionOff(M, back.selection!)).toBe(false);
  });
  it('in a restored selection and in the estimate copy of the check', () => {
    const r = restoreSelection(M, { ...defaults(M), profile: TC });
    expect(r.selection.components.vision).toBe(false);
    const checked = checkSelection(M, { ...defaults(M), profile: TC });
    expect(checked.components.vision).toBe(false);
    expect(checkSelection(M, { ...defaults(M), profile: 'exl3-4.5' }).components.vision).toBe(true);
  });
});

describe('the card', () => {
  it('ThinkingCap: the maker, the fine-tune line, the readout, the modalities and the measured speeds', () => {
    const c = byId(TC);
    expect(c.slug).toBe('thinkingcap-27b-kearuga-exl3-4.00');
    expect(c.author).toMatchObject({ login: '0xWhiteMage', name: 'thewhitemage.', url: 'https://github.com/0xWhiteMage', avatar: 'https://github.com/0xWhiteMage.png?size=96', initials: 'TH', social: [] });
    expect(c.finetuneOf).toBe('bottlecapai/ThinkingCap-Qwen3.8-27B');
    expect(c.finetuneUrl).toBe('https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B');
    expect([c.format, c.bpw, c.sizeGb]).toEqual(['exl3', 4, 16.4]);
    expect(c.modalities.map((m) => [m.id, m.runs])).toEqual([
      ['text', true],
      ['image', false],
    ]);
    expect(c.modalities[1].why).toBe('Image: in the pack, not supported for standalone packs yet');
    expect(c.drafters).toBe('Uses TandemLLM drafters (K3 + K4)');
    expect(c.verified!.speeds).toEqual([
      { task: 'prose', tok_s: 39.6 },
      { task: 'code', tok_s: 95.9 },
      { task: 'chat', tok_s: 51.3 },
    ]);
    expect(c.hfUrl).toBe(`https://huggingface.co/${thinking.hf.repo}/tree/${thinking.hf.revision}`);
    expect(c.licenseUrl).toMatch(/^https:\/\/huggingface\.co\//);
    expect(c.prUrl).toBe('https://github.com/0xBakeer/TandemLLM-early-access/pull/1');
    expect(c.status).toBe('verified');
  });
  it('a stock quant says so, a submitted one has no speeds, no drafters is said plainly', () => {
    const c = byId('community/mock-stock-27b-exl3-3.00');
    expect(c.finetuneOf).toBeNull();
    expect(c.verified).toBeNull();
    expect(speedOf(c)).toBeNull();
    expect(c.drafters).toBe('No drafters: one token a pass');
    expect(c.modalities.map((m) => m.id)).toEqual(['text', 'audio']);
  });
  it("the server's social links come in the fixed order, only those given", () => {
    const c = byId('community/mock-sparktuner-27b-exl3-5.00');
    expect(c.author.social.map((s) => [s.kind, s.href])).toEqual([
      ['x', 'https://x.com/sparktuner_mock'],
      ['huggingface', 'https://huggingface.co/mock-sparktuner'],
      ['bluesky', 'https://bsky.app/profile/sparktuner.bsky.social'],
      ['website', 'https://example.com/sparktuner'],
    ]);
    expect(c.author.social[0].label).toBe('@sparktuner_mock on X');
    expect(c.author.social.map((s) => s.handle)).toEqual(['sparktuner_mock', 'mock-sparktuner', 'sparktuner.bsky.social', 'example.com/sparktuner']);
  });
  it('a modality that is on in `modalities` counts as in the pack even when the list forgets it', () => {
    expect(modalityStates({ modalities: { text: true, image: true, audio: false, video: false }, modalities_in_pack: ['text'], load: 'overlay' }).map((m) => m.id)).toEqual(['text', 'image']);
  });
  it('a profile without the community field is not a card', () => {
    expect(cardOf(OWN.profiles[0])).toBeNull();
  });
  it('the box state: running, on disk, or the download', () => {
    const packs = communityProfiles(M);
    const dev = { active_profile: TC, profiles_on_disk: { [packs[1].id]: true } } as unknown as SetupDevice;
    expect(boxState(packs[0], dev)).toBe('running');
    expect(boxState(packs[1], dev)).toBe('on-disk');
    expect(boxState(packs[2], dev)).toBe('download');
    expect(boxState(packs[0], null)).toBe('download');
  });
});

describe('untrusted text and links', () => {
  it('only https URLs become links', () => {
    expect(httpsUrl('https://example.com/a')).toBe('https://example.com/a');
    expect(httpsUrl('http://example.com')).toBeNull();
    expect(httpsUrl('javascript:alert(1)')).toBeNull();
    expect(httpsUrl('data:text/html,hi')).toBeNull();
    expect(httpsUrl(42)).toBeNull();
  });
  it('the avatar comes from github.com alone', () => {
    expect(avatarUrl('https://github.com/0xWhiteMage.png?size=96')).toBe('https://github.com/0xWhiteMage.png?size=96');
    expect(avatarUrl('https://evil.example/0xWhiteMage.png')).toBeNull();
    expect(avatarUrl('http://github.com/x.png')).toBeNull();
  });
  it('a social link that is not https is dropped; a missing label is made', () => {
    const links = socialLinks({
      author: { github: 'x', name: 'X', social: { x: 'ok_handle', website: 'https://ok.example' } },
      social_links: [
        { network: 'x', label: '', url: 'https://x.com/ok_handle' },
        { network: 'mastodon', label: 'm', url: 'http://insecure/@me' },
        { network: 'website', label: '', url: 'https://ok.example/' },
      ],
    });
    expect(links.map((l) => [l.kind, l.label, l.handle])).toEqual([
      ['x', '@ok_handle on X', 'ok_handle'],
      ['website', "X's website", 'ok.example'],
    ]);
  });
  it('a fine-tune source that is not a Hugging Face id stays text; a bad hf_url falls back to the repo', () => {
    const p = profileOfEntry(OWN, { ...thinking, finetune_of: 'https://evil.example/x', hf_url: 'javascript:alert(1)' });
    expect(cardOf(p)!.finetuneUrl).toBeNull();
    expect(cardOf(p)!.finetuneOf).toBe('https://evil.example/x');
    expect(cardOf(p)!.hfUrl).toBe(`https://huggingface.co/${thinking.hf.repo}`);
  });
  it('initials: two words, else the first two letters, never empty', () => {
    expect(initials('Mock Quanter', 'x')).toBe('MQ');
    expect(initials('thewhitemage.', '0xWhiteMage')).toBe('TH');
    expect(initials(null, '0xWhiteMage')).toBe('0X');
    expect(initials('', '')).toBe('?');
  });
});

describe('search', () => {
  it('matches the name, the author, the handles, the tags and the fine-tune source, case blind', () => {
    const ids = (q: string) => searchCards(cards, q).map((c) => c.slug);
    expect(ids('thinkingcap')).toEqual(['thinkingcap-27b-kearuga-exl3-4.00']);
    expect(ids('@0xwhitemage')).toEqual(['thinkingcap-27b-kearuga-exl3-4.00']);
    expect(ids('thewhitemage')).toEqual(['thinkingcap-27b-kearuga-exl3-4.00']);
    expect(ids('bottlecapai')).toEqual(['thinkingcap-27b-kearuga-exl3-4.00']);
    expect(ids('sparktuner_mock')).toEqual(['mock-sparktuner-27b-exl3-5.00']);
    expect(ids('reasoning')).toEqual(['thinkingcap-27b-kearuga-exl3-4.00']);
    expect(ids('fine-tune').length).toBe(2);
  });
  it('every word must match; blank matches all; nothing matches gives an empty list', () => {
    expect(searchCards(cards, 'mock code').map((c) => c.slug)).toEqual(['mock-sparktuner-27b-exl3-5.00']);
    expect(searchCards(cards, '  ').length).toBe(3);
    expect(searchCards(cards, 'nothing-like-this')).toEqual([]);
  });
});

describe('sort', () => {
  const slugs = (k: (typeof SORT_KEYS)[number]) => sortCards(cards, k).map((c) => c.slug);
  it('newest first by the added date', () => {
    expect(slugs('newest')).toEqual(['thinkingcap-27b-kearuga-exl3-4.00', 'mock-stock-27b-exl3-3.00', 'mock-sparktuner-27b-exl3-5.00']);
  });
  it('by name, case blind', () => {
    expect(slugs('name')).toEqual(['mock-sparktuner-27b-exl3-5.00', 'mock-stock-27b-exl3-3.00', 'thinkingcap-27b-kearuga-exl3-4.00']);
  });
  it('smallest download first, most bits first', () => {
    expect(slugs('size')).toEqual(['mock-stock-27b-exl3-3.00', 'thinkingcap-27b-kearuga-exl3-4.00', 'mock-sparktuner-27b-exl3-5.00']);
    expect(slugs('bpw')).toEqual(['mock-sparktuner-27b-exl3-5.00', 'thinkingcap-27b-kearuga-exl3-4.00', 'mock-stock-27b-exl3-3.00']);
  });
  it('fastest prose first; packs we have not measured come last', () => {
    expect(slugs('speed')).toEqual(['thinkingcap-27b-kearuga-exl3-4.00', 'mock-sparktuner-27b-exl3-5.00', 'mock-stock-27b-exl3-3.00']);
  });
  it('does not change the input', () => {
    const before = cards.map((c) => c.slug);
    sortCards(cards, 'name');
    expect(cards.map((c) => c.slug)).toEqual(before);
  });
  it('the count line', () => {
    expect(countText(1, 1)).toBe('1 pack');
    expect(countText(3, 3)).toBe('3 packs');
    expect(countText(1, 3)).toBe('1 of 3 packs');
  });
});
