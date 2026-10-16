// The model popup's selection (issue #16): a model with profiles offers one pick per profile;
// a model without any — every relayed backend model, and the fake engine — must still be
// pickable, as the bare model with no profile. Before the fix the profile row was the only
// clickable thing, so such a model rendered as a title with nothing to tap: visible, unselectable.
import { describe, expect, it } from 'vitest';
import { modelPickTargets } from '../src/chat/composer';

describe('modelPickTargets', () => {
  it('a model with profiles picks model + profile, one per profile', () => {
    const m = { id: 'engine-model', profiles: [{ id: 'fast', name: 'Fast' }, { id: 'deep', name: 'Deep' }] };
    expect(modelPickTargets(m)).toEqual([
      { model: 'engine-model', profile: 'fast' },
      { model: 'engine-model', profile: 'deep' },
    ]);
  });
  it('a model without profiles — a relayed backend — picks itself with no profile', () => {
    expect(modelPickTargets({ id: 'Qwen/Qwen3.8-Flash-Next', profiles: [] })).toEqual([
      { model: 'Qwen/Qwen3.8-Flash-Next', profile: null },
    ]);
  });
});
