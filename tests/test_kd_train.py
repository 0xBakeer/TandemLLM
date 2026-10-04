"""The Kolibri block-drafter trainer (tools/kd_train.py) on a tiny random Kolibri, CPU only.

  * the taps pass gives the engine's own argmax at every position;
  * packed blocks are isolated: a block's draft is the same alone or beside others (so the batched
    gate equals a lone draft call);
  * a few steps on one sequence lower the loss;
  * the real speculative loop (draft, verify eight rows, truncate) writes exactly the plain greedy
    text, and the teacher-forced replay of the gate sees the same accepted lengths on it;
  * the CLI trains, exports, and the export loads into the engine's DSparkModule.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine.kolibri.model import KolibriEngine  # noqa: E402
from tools import kd_train as kd  # noqa: E402
from tools import kolibri_tiny as tiny  # noqa: E402

TAPS = [0, 2]
IDS = [5, 17, 3, 88, 41, 41, 2, 60, 7, 19, 33, 71, 12, 9, 22, 8, 40, 41, 2, 60]


def _eng(d):
    st, _ = tiny.write(d)
    return st, KolibriEngine.load(st, None, device="cpu", max_len=128, graphs=False, log=lambda s: None)


def _drafter(eng, seed=0, markov=8):
    c = eng.cfg
    raw = kd.drafter_config(2, 64, 4, 2, TAPS, 6, 4, markov, mask_id=c.vocab - 2, hidden=c.hidden,
                            vocab=c.vocab, head_dim=32)
    from engine.drafters.dspark import DSparkConfig
    params = kd.init_params(DSparkConfig(raw), "cpu", seed)
    with torch.no_grad():
        params["markov_head.markov_w2.weight"].normal_(0, 0.02)
    cfg, m = kd.build_module(raw, params)
    head = (eng.head.w.float() * eng.head.s[:, None])
    return raw, params, cfg, m, head


def test_target_pass_is_the_engine_argmax():
    with tempfile.TemporaryDirectory() as d:
        _, eng = _eng(d)
        ids = torch.tensor(IDS)
        eng.reset()
        taps, label, top_ids, top_lp = kd.target_pass(eng, ids, TAPS, topk=4)
        assert taps.shape == (len(IDS), len(TAPS) * eng.cfg.hidden)
        eng.reset()
        ref = eng.forward(IDS)
        assert torch.equal(label, ref.argmax(-1))
        assert torch.equal(top_ids[:, 0].long(), label)
        lp = torch.log_softmax(ref, -1).gather(-1, top_ids.long())
        assert (lp - top_lp.float()).abs().max() < 1e-2


def test_blocks_are_isolated():
    with tempfile.TemporaryDirectory() as d:
        _, eng = _eng(d)
        _, _, cfg, m, head = _drafter(eng)
        ids = torch.tensor(IDS)
        eng.reset()
        taps, *_ = kd.target_pass(eng, ids, TAPS)
        anchors = torch.tensor([6, 9, 13, 17])
        both = kd.run_blocks(m, cfg, eng.emb, ids, taps, anchors)
        for i, a in enumerate(anchors.tolist()):
            one = kd.run_blocks(m, cfg, eng.emb, ids, taps, torch.tensor([a]))
            assert torch.allclose(both[i], one[0], atol=1e-5), a


def test_a_few_steps_lower_the_loss():
    with tempfile.TemporaryDirectory() as d:
        _, eng = _eng(d)
        _, params, cfg, m, head = _drafter(eng)
        ids = torch.tensor(IDS)
        eng.reset()
        taps, label, top_ids, top_lp = kd.target_pass(eng, ids, TAPS)
        anchors = torch.arange(4, len(IDS) - 1)
        opt = torch.optim.AdamW(list(params.values()), lr=3e-3)
        losses = []
        for _ in range(25):
            opt.zero_grad()
            pred = kd.run_blocks(m, cfg, eng.emb, ids, taps, anchors)
            loss, *_ = kd.loss_of(m, cfg, pred, head, ids, label, top_ids, top_lp, anchors, 0.5, 4.0)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        assert losses[-1] < 0.7 * losses[0], losses


def test_real_loop_is_lossless_and_matches_the_replay():
    with tempfile.TemporaryDirectory() as d:
        _, eng = _eng(d)
        _, params, cfg, m, head = _drafter(eng)
        # train it a little on its own greedy text so some drafts get accepted
        prompt = IDS[:6]
        plain = kd.greedy_plain(eng, prompt, TAPS, "cpu", max_new=40)
        seq = torch.tensor(prompt + plain)
        eng.reset()
        taps, label, top_ids, top_lp = kd.target_pass(eng, seq, TAPS)
        anchors = torch.arange(len(prompt), len(seq) - 1)
        opt = torch.optim.AdamW(list(params.values()), lr=3e-3)
        for _ in range(60):
            opt.zero_grad()
            pred = kd.run_blocks(m, cfg, eng.emb, seq, taps, anchors)
            loss, *_ = kd.loss_of(m, cfg, pred, head, seq, label, top_ids, top_lp, anchors, 0.5, 4.0)
            loss.backward()
            opt.step()
        from engine.kolibri.verify import ReplayVerifier
        r = kd.real_loop(m, cfg, eng, eng.emb, head, ReplayVerifier(eng), prompt, TAPS, "cpu", max_new=40)
        n = min(len(plain), len(r["gen"]))
        assert r["gen"][:n] == plain[:n]
        assert sum(r["accepted"]) > 0
        s = kd.Seq({"id": "x", "src": "t", "kind": "chat", "lang": "en", "effort": "none", "mode": "greedy",
                    "prompt_ids": prompt, "gen_ids": r["gen"][:n]}, 1000)
        res = kd.acceptance(m, cfg, eng, eng.emb, head, [s], TAPS, "cpu")
        rounds = res["ALL"]["rounds"]
        assert abs(res["ALL"]["chains"][str(cfg.block_size - 1)] - res["ALL"]["tokens_per_round"]) < 1e-9
        # the loop's rounds over the same text: the replay walks the same anchors
        assert abs(rounds - len(r["accepted"])) <= 1, (rounds, r["accepted"])


def test_cli_trains_exports_and_loads():
    with tempfile.TemporaryDirectory() as d:
        st, eng = _eng(d)
        recs = []
        for i in range(6):
            prompt = IDS[i:i + 6]
            g = kd.greedy_plain(eng, prompt, TAPS, "cpu", max_new=24)
            recs.append({"id": f"s{i}", "src": "t", "kind": "chat", "lang": "de" if i % 2 else "en",
                         "split": "train" if i < 4 else "heldout", "effort": "none", "mode": "greedy", "prompt_ids": prompt,
                         "gen_ids": g, "finish": "length"})
        os.makedirs(os.path.join(d, "gen"))
        torch.save(recs[:4], os.path.join(d, "gen", "gen-train-000000.pt"))
        torch.save(recs[4:], os.path.join(d, "gen", "gen-heldout-000000.pt"))
        out = os.path.join(d, "out")
        argv = ["kd_train.py", "--set", st, "--gen", os.path.join(d, "gen", "gen-train-*.pt"),
                "--held", os.path.join(d, "gen", "gen-heldout-*.pt"), "--out", out, "--device", "cpu",
                "--layers", "2", "--inter", "64", "--heads", "4", "--kv-heads", "2", "--head-dim", "32",
                "--taps", "0,2", "--window", "6", "--block", "4", "--markov", "8",
                "--mask-id", str(eng.cfg.vocab - 2), "--max-len", "64", "--chunk", "16",
                "--anchors", "8", "--accum", "1", "--steps", "8", "--stop-step", "6", "--warmup", "2", "--eval-every", "3",
                "--real-loop", "1", "--tap-norm"]
        old = sys.argv
        sys.argv = argv
        try:
            kd.main()
        finally:
            sys.argv = old
        fin = os.path.join(out, "final")
        raw = json.load(open(os.path.join(fin, "config.json")))
        from safetensors.torch import load_file
        from engine.drafters.dspark import DSparkConfig, DSparkModule
        w = load_file(os.path.join(fin, "model.safetensors"))
        cfg = DSparkConfig(raw)
        assert set(cfg.expected_tensors()) == set(w)
        DSparkModule(cfg, w)
        assert os.path.exists(os.path.join(out, "step-000003", "model.safetensors"))
        assert w["norm.weight"].float().std() > 0       # started at the target's final norm, not ones
        assert os.path.exists(os.path.join(out, "resume.pt"))  # no --ckpt-dir: the state stays in --out
        loop = json.load(open(os.path.join(out, "real_loop.json")))
        assert loop["verify"] == "replay" and loop["all_same"]
        assert os.path.exists(os.path.join(out, "eval_log.jsonl"))


def test_tap_norm_folds_into_fc_at_export():
    with tempfile.TemporaryDirectory() as d:
        _, eng = _eng(d)
        raw, params, cfg, m, head = _drafter(eng)
        ids = torch.tensor(IDS)
        eng.reset()
        taps, *_ = kd.target_pass(eng, ids, TAPS)
        m.tap_scale = taps.float().pow(2).mean(0).sqrt().clamp_min(1e-3) * 3.0
        anchors = torch.tensor([6, 9, 13])
        a = kd.run_blocks(m, cfg, eng.emb, ids, taps, anchors)
        out = kd.export(params, raw, os.path.join(d, "x"), {}, m.tap_scale)
        from safetensors.torch import load_file
        from engine.drafters.dspark import DSparkConfig, DSparkModule
        w = {k: v.float() for k, v in load_file(os.path.join(out, "model.safetensors")).items()}
        m2 = DSparkModule(DSparkConfig(raw), w)
        b = kd.run_blocks(m2, cfg, eng.emb, ids, taps, anchors)
        assert (a - b).abs().max() <= 0.05 * a.abs().max(), (a - b).abs().max()
