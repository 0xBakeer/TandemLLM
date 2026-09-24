import { describe, expect, it } from 'vitest';
import { parsePrometheus, value, histogram, histogramDelta, histogramQuantile, rate, delta, liveNumbers } from '../src/lib/prom';

const TEXT = `# HELP qse_requests_total generations that finished, by the reason they finished
# TYPE qse_requests_total counter
qse_requests_total{finish_reason="abandoned"} 1
qse_requests_total{finish_reason="length"} 50
qse_requests_total{finish_reason="stop"} 812
# TYPE qse_requests_running gauge
qse_requests_running 1
# TYPE qse_requests_waiting gauge
qse_requests_waiting 0
# TYPE qse_generation_tokens_total counter
qse_generation_tokens_total 12800
# HELP qse_time_to_first_token_seconds arrival of the request to its first token
# TYPE qse_time_to_first_token_seconds histogram
qse_time_to_first_token_seconds_bucket{le="0.5"} 10
qse_time_to_first_token_seconds_bucket{le="0.75"} 12
qse_time_to_first_token_seconds_bucket{le="1.0"} 50
qse_time_to_first_token_seconds_bucket{le="+Inf"} 50
qse_time_to_first_token_seconds_sum 38.07
qse_time_to_first_token_seconds_count 50
# TYPE qse_spec_accept_per_block histogram
qse_spec_accept_per_block_bucket{le="1.0"} 174
qse_spec_accept_per_block_bucket{le="2.0"} 400
qse_spec_accept_per_block_bucket{le="+Inf"} 1600
qse_spec_accept_per_block_sum 6338
qse_spec_accept_per_block_count 1600
# TYPE qse_spec_decode_num_draft_tokens_total counter
qse_spec_decode_num_draft_tokens_total 24000
# TYPE qse_spec_decode_num_accepted_tokens_total counter
qse_spec_decode_num_accepted_tokens_total 6338
# TYPE qse_engine_info gauge
qse_engine_info{version="0.1.0-rc4",model="qwen38-spark-engine",label="with \\"quotes\\" and \\\\ backslash",nl="a\\nb"} 1
qse_cache_bytes{cache="state"} 6.1e9
`;

describe('Prometheus text parser', () => {
  const s = parsePrometheus(TEXT, 1000);

  it('reads counters with labels and sums across a label filter', () => {
    expect(value(s, 'qse_requests_total')).toBe(863);
    expect(value(s, 'qse_requests_total', { finish_reason: 'stop' })).toBe(812);
    expect(value(s, 'qse_requests_total', { finish_reason: 'nope' })).toBeNull();
  });

  it('reads gauges, exponents and untyped samples', () => {
    expect(value(s, 'qse_requests_running')).toBe(1);
    expect(value(s, 'qse_cache_bytes', { cache: 'state' })).toBe(6.1e9);
  });

  it('a missing family is null, never zero (feature off)', () => {
    expect(value(s, 'qse_does_not_exist')).toBeNull();
    expect(histogram(s, 'qse_does_not_exist')).toBeNull();
  });

  it('parses histograms with +Inf, sum and count', () => {
    const h = histogram(s, 'qse_time_to_first_token_seconds');
    expect(h).not.toBeNull();
    expect(h!.count).toBe(50);
    expect(h!.sum).toBeCloseTo(38.07);
    expect(h!.buckets.map((b) => b.le)).toEqual([0.5, 0.75, 1.0, Infinity]);
    expect(h!.buckets[3].count).toBe(50);
  });

  it('unescapes label values', () => {
    const f = s.families.get('qse_engine_info')!;
    expect(f.samples[0].labels.label).toBe('with "quotes" and \\ backslash');
    expect(f.samples[0].labels.nl).toBe('a\nb');
    expect(f.help).toBe('');
  });

  it('records HELP and TYPE', () => {
    expect(s.families.get('qse_requests_total')!.type).toBe('counter');
    expect(s.families.get('qse_requests_total')!.help).toMatch(/generations that finished/);
    expect(s.families.get('qse_time_to_first_token_seconds')!.type).toBe('histogram');
  });

  it('histogram_quantile interpolates inside the bucket', () => {
    const h = histogram(s, 'qse_time_to_first_token_seconds')!;
    // rank 25 of 50 lies in the (0.75, 1.0] bucket: 12 below, 38 in it
    const p50 = histogramQuantile(h, 0.5)!;
    expect(p50).toBeCloseTo(0.75 + (0.25 * (25 - 12)) / 38, 5);
    expect(histogramQuantile({ buckets: [], sum: 0, count: 0 }, 0.5)).toBeNull();
  });

  it('bucket deltas and rates between two scrapes', () => {
    const later = parsePrometheus(TEXT.replace('qse_generation_tokens_total 12800', 'qse_generation_tokens_total 13140').replace('le="+Inf"} 50', 'le="+Inf"} 52').replace('_count 50', '_count 52'), 6000);
    expect(rate(s, later, 'qse_generation_tokens_total')).toBeCloseTo(68); // 340 tokens / 5 s
    expect(delta(s, later, 'qse_generation_tokens_total')).toBe(340);
    const d = histogramDelta(histogram(s, 'qse_time_to_first_token_seconds'), histogram(later, 'qse_time_to_first_token_seconds'))!;
    expect(d.count).toBe(2);
    expect(d.buckets.find((b) => b.le === Infinity)!.count).toBe(2);
    expect(rate(s, s, 'qse_generation_tokens_total')).toBeNull(); // no time moved
  });
});

describe('live strip numbers', () => {
  it('decode tok/s from 340 tokens over 5 s is 68', () => {
    const a = parsePrometheus(TEXT, 0);
    const b = parsePrometheus(TEXT.replace('qse_generation_tokens_total 12800', 'qse_generation_tokens_total 13140'), 5000);
    const l = liveNumbers(a, b);
    expect(l.decodeTps).toBeCloseTo(68);
    expect(l.generating).toBe(true);
    expect(l.running).toBe(1);
    expect(l.specReported).toBe(true);
  });

  it('tokens per block 4.57 and acceptance 23.9 % from the deltas', () => {
    const a = parsePrometheus(TEXT, 0);
    const bText = TEXT.replace('qse_spec_accept_per_block_sum 6338', 'qse_spec_accept_per_block_sum 6795')
      .replace('qse_spec_accept_per_block_count 1600', 'qse_spec_accept_per_block_count 1700')
      .replace('qse_spec_decode_num_draft_tokens_total 24000', 'qse_spec_decode_num_draft_tokens_total 25350')
      .replace('qse_spec_decode_num_accepted_tokens_total 6338', 'qse_spec_decode_num_accepted_tokens_total 6660');
    const l = liveNumbers(a, parsePrometheus(bText, 5000));
    expect(l.tokensPerBlock).toBeCloseTo(4.57, 2);
    expect(l.acceptance! * 100).toBeCloseTo(23.85, 1);
  });

  it('without speculation families the tiles are "not reported"', () => {
    const noSpec = TEXT.split('\n')
      .filter((l) => !l.includes('qse_spec_'))
      .join('\n');
    const l = liveNumbers(parsePrometheus(noSpec, 0), parsePrometheus(noSpec, 5000));
    expect(l.specReported).toBe(false);
    expect(l.tokensPerBlock).toBeNull();
    expect(l.acceptance).toBeNull();
  });

  it('idle engine: no generation rate, falls back to whole-histogram TTFT', () => {
    const a = parsePrometheus(TEXT.replace('qse_requests_running 1', 'qse_requests_running 0'), 0);
    const b = parsePrometheus(TEXT.replace('qse_requests_running 1', 'qse_requests_running 0'), 5000);
    const l = liveNumbers(a, b);
    expect(l.generating).toBe(false);
    expect(l.decodeTps).toBe(0);
    expect(l.ttftP50).not.toBeNull();
  });
});
