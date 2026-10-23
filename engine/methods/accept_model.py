"""The round planner's acceptance model (SPD-187), learned online: P(a draft node is on the accepted path).

Two predictors, both learned from every verified block, prequentially (each block is predicted with the state
before it, its errors recorded, and only then learned from):

  * the table: decayed (hits, predicted) sums by (depth, rank, class) -- the node's depth in its tree, its
    rank by drafter score, the slot's curve class -- times a factor of the slot's current request (its own
    hits over its predictions, reset when the slot's request changes): p = q x ratio x factor;
  * the net (the q-head of idea 2): 11 features (log q, depth, rank, source, siblings, best sibling, parent
    q, the slot's acceptance EMA, log2 context, E, q) -> 16 tanh -> 1, logistic, Adam on the log loss over
    a window of recent nodes, trained in a background thread (never on a round's thread).

`predict` gives the model in use: the net while its rolling log loss (decayed, over the same nodes) is under
the table's by `margin`, else the table. The planner's gain curves read these probabilities; they choose
which rows are verified, never their arithmetic (greedy and seeded outputs do not depend on them).

State (`state()` / `load()`): JSON numbers only (sums, factors' prior, net weights, the error history).
"""

from __future__ import annotations

import math
import threading
from collections import deque

import numpy as np

SRC = {"df2-greedy": 0, "df2-tree": 1, "mtp": 2, "look": 3, "sib": 4}
CLASSES = ("copy", "strong", "fair", "weak")


def node_features(parents, scores, q, depth, rank, source, slot_ema, ctx, E):
    """Per non-root node the net's 11 features and the table's key parts (depth, rank)."""
    n = len(parents)
    kids: dict = {}
    for c in range(1, n):
        kids.setdefault(parents[c], []).append(c)
    rows = []
    for c in range(1, n):
        s = float(q[c] if q is not None and q[c] is not None else scores[c])
        sib = kids.get(parents[c], [])
        best_other = max([float(scores[k]) for k in sib if k != c] or [0.0])
        ps = float(q[parents[c]] if q is not None and parents[c] and q[parents[c]] is not None else
                   (scores[parents[c]] if parents[c] else 1.0))
        rows.append([math.log(max(s, 1e-6)), depth[c], rank[c], SRC.get(source[c] if source else "", 5),
                     len(sib), math.log(max(best_other, 1e-6)), math.log(max(ps, 1e-6)), slot_ema,
                     math.log2(max(ctx, 1)), E, s])
    return rows


class _Net:
    def __init__(self, d=11, h=16, seed=0):
        r = np.random.default_rng(seed)
        self.W1, self.b1 = r.normal(0, 0.3, (d, h)), np.zeros(h)
        self.W2, self.b2 = r.normal(0, 0.3, (h,)), 0.0
        self.mu, self.sd = np.zeros(d), np.ones(d)
        self.t = 0
        self.m = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), 0.0]
        self.v = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), 0.0]

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        return 1 / (1 + np.exp(-(np.tanh(Z @ self.W1 + self.b1) @ self.W2 + self.b2)))

    def fit(self, X, y, steps=200, bs=256, lr=3e-3, seed=0):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        r = np.random.default_rng(seed)
        for _ in range(steps):
            b = r.integers(0, len(Z), bs)
            zb, yb = Z[b], y[b]
            hid = np.tanh(zb @ self.W1 + self.b1)
            p = 1 / (1 + np.exp(-(hid @ self.W2 + self.b2)))
            g = (p - yb) / bs
            grads = [zb.T @ (np.outer(g, self.W2) * (1 - hid ** 2)), (np.outer(g, self.W2) * (1 - hid ** 2)).sum(0),
                     hid.T @ g, float(g.sum())]
            self.t += 1
            params = [self.W1, self.b1, self.W2]
            for i in range(3):
                self.m[i] = 0.9 * self.m[i] + 0.1 * grads[i]
                self.v[i] = 0.999 * self.v[i] + 0.001 * grads[i] ** 2
                params[i] -= lr * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)
            self.m[3] = 0.9 * self.m[3] + 0.1 * grads[3]
            self.v[3] = 0.999 * self.v[3] + 0.001 * grads[3] ** 2
            self.b2 -= lr * (self.m[3] / (1 - 0.9 ** self.t)) / (math.sqrt(self.v[3] / (1 - 0.999 ** self.t)) + 1e-8)

    def state(self):
        return {"W1": self.W1.tolist(), "b1": self.b1.tolist(), "W2": self.W2.tolist(), "b2": self.b2,
                "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    def load(self, d):
        self.W1, self.b1 = np.array(d["W1"]), np.array(d["b1"])
        self.W2, self.b2 = np.array(d["W2"]), float(d["b2"])
        self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])


class AcceptModel:
    def __init__(self, decay: float = 0.999, margin: float = 0.02, train_every: int = 64, window: int = 40000,
                 background: bool = True):
        self.decay, self.margin, self.train_every = decay, margin, train_every
        self.tab: dict = {}                    # (depth, rank, class) -> [hits, predicted]
        self.slot: dict = {}                   # slot -> [hits, predicted, request key]
        self.ema: dict = {}                    # slot -> accepted EMA (a net feature)
        self.net = _Net()
        self.net_ready = False
        self.buf_X: deque = deque(maxlen=window)
        self.buf_y: deque = deque(maxlen=window)
        self.blocks = 0
        self.ll = {"table": [0.0, 0.0], "net": [0.0, 0.0]}     # decayed (sum of log loss, count)
        self.acc_err = [0.0, 0.0, 0.0]                           # decayed (sum |err|, sum err, count)
        self.history: list = []                                  # every 256 blocks: the rolling errors
        self.lock = threading.Lock()
        self.background = background
        self._training = False

    # ---- prediction ---------------------------------------------------------------------------
    @staticmethod
    def _key(depth, rank, cls):
        return (min(int(depth), 6), min(int(rank), 15), cls)

    def _table(self, qs, depth, rank, cls, slot):
        f = self.slot.get(slot)
        fac = (f[0] + 2.0) / (f[1] + 2.0) if f is not None else 1.0
        out = []
        for s, d, r in zip(qs, depth, rank):
            h = self.tab.get(self._key(d, r, cls))
            ratio = (h[0] + 1.0) / (h[1] + 1.0) if h is not None else 1.0
            out.append(min(0.999, max(1e-4, s * ratio * fac)))
        return out

    def use_net(self) -> bool:
        t, n = self.ll["table"], self.ll["net"]
        return (self.net_ready and t[1] > 2000 and n[1] > 2000
                and n[0] / n[1] < (1.0 - self.margin) * t[0] / t[1])

    def predict(self, parents, scores, q, depth, rank, source, cls, slot, ctx, E) -> list:
        """Per node (the root first, as 1.0) the probability it is on the accepted path."""
        n = len(parents)
        if n <= 1:
            return [1.0]
        qs = [float(q[c] if q is not None and q[c] is not None else scores[c]) for c in range(1, n)]
        if self.use_net():
            X = np.array(node_features(parents, scores, q, depth, rank, source, self.ema.get(slot, 2.0), ctx, E))
            with self.lock:
                p = self.net.predict(X)
            return [1.0] + [float(min(0.999, max(1e-4, v))) for v in p]
        return [1.0] + self._table(qs, depth[1:], rank[1:], cls, slot)

    # ---- learning (prequential) ---------------------------------------------------------------
    def observe(self, parents, scores, q, depth, rank, source, cls, slot, ctx, E, path, request=None) -> None:
        """A verified block: score both predictors on it (before learning), then learn it."""
        n = len(parents)
        if n <= 1:
            return
        f = self.slot.get(slot)
        if f is None or f[2] != request:
            self.slot[slot] = f = [0.0, 0.0, request]           # a new request on this slot
            self.ema[slot] = 2.0
        on = set(path)
        y = [1.0 if c in on else 0.0 for c in range(1, n)]
        qs = [float(q[c] if q is not None and q[c] is not None else scores[c]) for c in range(1, n)]
        pt = self._table(qs, depth[1:], rank[1:], cls, slot)
        X = node_features(parents, scores, q, depth, rank, source, self.ema.get(slot, 2.0), ctx, E)
        d = self.decay
        # the errors, before anything learns from this block
        ll_t = sum(-math.log(p if t else 1 - p) for p, t in zip(pt, y))
        self.ll["table"] = [d * self.ll["table"][0] + ll_t, d * self.ll["table"][1] + len(y)]
        if self.net_ready:
            with self.lock:
                pn = self.net.predict(np.array(X))
            ll_n = sum(-math.log(min(0.999, max(1e-4, p)) if t else 1 - min(0.999, max(1e-4, p)))
                       for p, t in zip(pn, y))
            self.ll["net"] = [d * self.ll["net"][0] + ll_n, d * self.ll["net"][1] + len(y)]
        use = self.predict(parents, scores, q, depth, rank, source, cls, slot, ctx, E)[1:]
        err = sum(use) - sum(y)                               # expected accepted nodes - accepted
        self.acc_err = [d * self.acc_err[0] + abs(err), d * self.acc_err[1] + err, d * self.acc_err[2] + 1]
        # learn
        for s, dd, r, t in zip(qs, depth[1:], rank[1:], y):
            k = self._key(dd, r, cls)
            h = self.tab.setdefault(k, [0.0, 0.0])
            h[0] = d * h[0] + t
            h[1] = d * h[1] + s
        f[0] += sum(y)
        f[1] += sum(pt)
        self.ema[slot] = 0.9 * self.ema.get(slot, 2.0) + 0.1 * (len(path) - 1)
        self.buf_X.extend(X)
        self.buf_y.extend(y)
        self.blocks += 1
        if self.blocks % 256 == 0:
            self.history.append(self.errors())
            del self.history[:-64]
        if self.blocks % self.train_every == 0 and len(self.buf_y) >= 2000 and not self._training:
            self._train()

    def _train(self) -> None:
        X, y = np.array(self.buf_X), np.array(self.buf_y)

        def run():
            net = _Net()
            if self.net_ready:
                with self.lock:
                    net.load(self.net.state())
            net.fit(X, y, seed=self.blocks)
            with self.lock:
                self.net = net
                self.net_ready = True
            self._training = False
        self._training = True
        if self.background:
            threading.Thread(target=run, name="accept-model-train", daemon=True).start()
        else:
            run()

    # ---- report / state -----------------------------------------------------------------------
    def errors(self) -> dict:
        t, n, a = self.ll["table"], self.ll["net"], self.acc_err
        return {"blocks": self.blocks, "ll_table": round(t[0] / t[1], 4) if t[1] else None,
                "ll_net": round(n[0] / n[1], 4) if n[1] else None, "use": "net" if self.use_net() else "table",
                "acc_mae": round(a[0] / a[2], 3) if a[2] else None, "acc_bias": round(a[1] / a[2], 3) if a[2] else None}

    def state(self) -> dict:
        with self.lock:
            net = self.net.state() if self.net_ready else None
        return {"tab": [[list(k), v] for k, v in self.tab.items()], "net": net, "ll": self.ll,
                "acc_err": self.acc_err, "blocks": self.blocks, "history": self.history[-16:]}

    def load(self, st: dict, keep: float = 0.5) -> None:
        """Warm start from a saved state, its sums scaled by `keep` so a new workload re-learns fast."""
        self.tab = {tuple(k): [v[0] * keep, v[1] * keep] for k, v in st.get("tab", [])}
        if st.get("net"):
            self.net.load(st["net"])
            self.net_ready = True
        ll = st.get("ll") or {}
        self.ll = {k: [v[0] * keep, v[1] * keep] for k, v in ll.items()} or self.ll
        self.history = list(st.get("history") or [])
