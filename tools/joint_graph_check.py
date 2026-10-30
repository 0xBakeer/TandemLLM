#!/usr/bin/env python3
"""SPD-163 on the real model: does the joint pass from a CUDA graph (engine/models/qwen3_5/joint_graphs.py)
give each slot what its own verify gives it?

For each slot count E: 2E engines over ONE copy of the weights, E that verify each round in the eager
joint pass (`engine.model.forward_tree_multi`, which gives each block its lone tree verify bit for bit:
tools/parallel_verify_check.py) and E that verify the same rounds through the joint graphs. Each pair starts from the same prompt (random ids, its own length, so the engines sit at
different contexts), gets the same tree each round (random shapes from a chain to a bush, 2 to T_pad
nodes, so most blocks are padded) and commits the same path: the lone run's greedy walk. Per round:
the logits bit for bit, and that the round was a replay; at the end the recurrent state and KV rows.
The rounds alternate the fold's parity, cross a graph's capture and its replays, and reuse graphs
across rounds with different row counts.

    python tools/joint_graph_check.py --slots 2,4,8 --rounds 10
    python tools/joint_graph_check.py --slots 2,3,4,5,6,7,8 --tpads 4,6,8,12,16 --solo --rounds 6

SPD-187: `--tpads` forces each row bucket the round planner can take (the graphs' FORCE_T; every (E, T) with
E x T within the row cap), and `--solo` adds E engines that verify each block alone (`forward_tree`): the
eager joint pass and the graphed one are then both held to the lone verify, logits, state and KV.

Run with the served environment (ops/serve.env sourced) and QWEN38_JOINT_GRAPHS=1 QWEN38_JOINT_GDN=1
QWEN38_GDN_AB=1; the server stopped.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tree(rng, anchor: int, n: int):
    from engine.tree import DraftTree
    parents, path = [-1], [0]
    for i in range(1, n):
        path = path[:rng.randint(1, len(path))] if rng.random() < 0.6 else path
        parents.append(path[-1])
        path.append(i)
    tokens = [anchor] + [rng.randint(1000, 100000) for _ in range(n - 1)]
    return DraftTree(tokens, parents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="2,4,8")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--tpads", default="", help="row buckets to force (default: the family's t_pad)")
    ap.add_argument("--solo", action="store_true", help="also hold both passes to each block's lone verify")
    a = ap.parse_args()
    from engine.config import load_config
    from engine.loader import Weights
    from engine.models.qwen3_5 import joint_graphs as JG
    from engine.models.qwen3_5.adapter import Qwen38Engine
    import engine.model as M
    if not JG.ON:
        sys.exit("QWEN38_JOINT_GRAPHS=1 is not set")
    cfg = load_config(None)
    w = Weights(cfg.path, skip_mtp=True)
    rng = random.Random(23)
    g = torch.Generator().manual_seed(29)
    ok_all = True
    cases = []
    for E in (int(x) for x in a.slots.split(",")):
        for t in ([int(x) for x in a.tpads.split(",")] if a.tpads else [None]):
            cases.append((E, t))
    for E, t_force in cases:
        alone = [Qwen38Engine(cfg, w, max_len=a.max_len) for _ in range(E)]
        joint = [Qwen38Engine(cfg, w, max_len=a.max_len) for _ in range(E)]
        solo = [Qwen38Engine(cfg, w, max_len=a.max_len) for _ in range(E)] if a.solo else []
        cap_rows = Qwen38Engine.joint_policy(alone[0])["max_rows"]
        if t_force is not None and (E * t_force > cap_rows or t_force > JG.MAX_T):
            continue
        JG.FORCE_T = t_force
        T = t_force if t_force is not None else JG.t_pad(E, cap_rows)
        pos, anchors = [], []
        with torch.no_grad():
            for i in range(E):
                p = torch.randint(1000, 100000, (rng.randint(150, 900),), generator=g).cuda()
                la = alone[i].forward(p, start=0, last_only=True)
                lj = joint[i].forward(p, start=0, last_only=True)
                assert torch.equal(la, lj), "the prefills differ"
                if solo:
                    assert torch.equal(la, solo[i].forward(p, start=0, last_only=True)), "the prefills differ"
                pos.append(p.numel())
                anchors.append(int(p[-1]))
            jg = JG.graphs_for(joint[0])
            bit, replays, solo_bit = 0, 0, 0
            for r in range(a.rounds):
                trees = [_tree(rng, anchors[i], rng.randint(2, T)) for i in range(E)]
                blocks = [torch.tensor(t.tokens).cuda() for t in trees]
                lone = M.forward_tree_multi([(alone[i], blocks[i], trees[i].parents, pos[i]) for i in range(E)])
                r0 = jg.stats["replayed"]
                multi = JG.forward_tree_multi([(joint[i], blocks[i], trees[i].parents, pos[i])
                                               for i in range(E)])
                replayed = jg.stats["replayed"] > r0
                replays += replayed
                same = [torch.equal(x, y) for x, y in zip(lone, multi)]
                d = max(float((x.float() - y.float()).abs().max()) for x, y in zip(lone, multi))
                bit += all(same)
                if solo:
                    ref = [solo[i].forward_tree(blocks[i], trees[i].parents, start=pos[i]) for i in range(E)]
                    s_ok = all(torch.equal(x, y) for x, y in zip(ref, lone)) and \
                        all(torch.equal(x, y) for x, y in zip(ref, multi))
                    solo_bit += s_ok
                    lone = ref
                for i in range(E):
                    picks = lone[i].argmax(-1).tolist()
                    path, new = alone[i].accept_tree(trees[i], picks)
                    alone[i].commit_tree(path)
                    joint[i].commit_tree(path)
                    if solo:
                        solo[i].commit_tree(path)
                    pos[i] += len(path)
                    anchors[i] = new[-1]
                print(f"E={E} T_pad={T} round {r:2d} rows {[len(t.tokens) for t in trees]}: "
                      f"{'replay' if replayed else 'EAGER'} logits "
                      f"{'IDENTICAL' if all(same) else 'differ ' + str([i for i, s in enumerate(same) if not s])}"
                      f" (max |d| {d:.3g})", flush=True)
            st = []
            for i in range(E):
                Sa, Sj = alone[i].committed_state(), joint[i].committed_state()
                n = alone[i].kv.length
                st.append(torch.equal(Sa, Sj) and n == joint[i].kv.length and torch.equal(
                    alone[i].kv.k[..., :n, :], joint[i].kv.k[..., :n, :]) and torch.equal(
                    alone[i].kv.v[..., :n, :], joint[i].kv.v[..., :n, :]))
                if solo:
                    So = solo[i].committed_state()
                    st[-1] = st[-1] and torch.equal(Sa, So) and n == solo[i].kv.length and torch.equal(
                        alone[i].kv.k[..., :n, :], solo[i].kv.k[..., :n, :]) and torch.equal(
                        alone[i].kv.v[..., :n, :], solo[i].kv.v[..., :n, :])
            ok = bit == a.rounds and replays == a.rounds and all(st) and (not solo or solo_bit == a.rounds)
            ok_all &= ok
            print(f"JOINT_GRAPH E={E} T_pad={T} rounds={a.rounds} replays={replays} bit_identical={bit}/{a.rounds} "
                  + (f"lone_identical={solo_bit}/{a.rounds} " if solo else "")
                  + f"state_and_kv_identical={all(st)} {'OK' if ok else 'FAIL'}", flush=True)
        JG.FORCE_T = None
        del alone, joint, solo
        torch.cuda.empty_cache()
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
