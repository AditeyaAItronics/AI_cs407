"""
CS F407 - Laboratory: Bayesian Networks and Autoregressive Language Models

First-order (bigram) and second-order (trigram) autoregressive language models
built from counts, viewed as the Bayesian networks
    X1 -> X2 -> X3 -> ...                      (first order)
    X_{t-2} -> X_t <- X_{t-1}                  (second order)
No machine-learning library or pretrained model is used.

Usage:  python bn_language_model.py
"""

import random
from collections import Counter, defaultdict

START, END = "<START>", "<END>"

CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

MAX_LEN = 20   # safety limit on generated sentence length (tokens, excluding markers)


def tokenise(sentence):
    return sentence.lower().split()


class NGramModel:
    """Autoregressive model P(X_t | previous `order` tokens), estimated by counting.

    order = 1 -> P(X_t | X_{t-1})           context is a 1-tuple
    order = 2 -> P(X_t | X_{t-2}, X_{t-1})  context is a 2-tuple
    Sentences are padded with `order` START tokens and one END token.
    """

    def __init__(self, sentences, order=1):
        self.order = order
        self.counts = defaultdict(Counter)        # transition counts C(context, next)
        for sentence in sentences:
            tokens = [START] * order + tokenise(sentence) + [END]
            for i in range(order, len(tokens)):
                context = tuple(tokens[i - order:i])
                self.counts[context][tokens[i]] += 1
        # conditional probability tables  P(next | context) = C(context, next) / sum_k C(context, k)
        self.cpt = {
            context: {w: c / sum(nexts.values()) for w, c in nexts.items()}
            for context, nexts in self.counts.items()
        }
        self.vocab = sorted({w for nexts in self.counts.values() for w in nexts} - {END})

    def distribution(self, *context):
        """P(X_t | context). Unseen contexts raise a KeyError with a clear message."""
        context = tuple(context)
        if context not in self.cpt:
            raise KeyError(f"context {context} never observed in training data; "
                           f"P(X_t | {context}) is undefined")
        return self.cpt[context]

    def most_probable(self, *context):
        dist = self.distribution(*context)
        best = max(dist.values())
        # ties broken alphabetically so greedy decoding is deterministic
        return min(w for w, p in dist.items() if p == best)

    def sample_next(self, context, rng):
        dist = self.distribution(*context)
        words = list(dist)
        return rng.choices(words, weights=[dist[w] for w in words])[0]

    def generate(self, mode="sample", rng=None):
        context = [START] * self.order
        out = []
        while len(out) < MAX_LEN:
            if mode == "greedy":
                word = self.most_probable(*context)
            else:
                word = self.sample_next(context, rng)
            if word == END:
                return " ".join(out)
            out.append(word)
            context = context[1:] + [word]
        return " ".join(out) + " ...[truncated at MAX_LEN]"

    def sentence_probability(self, sentence):
        tokens = [START] * self.order + tokenise(sentence) + [END]
        p = 1.0
        for i in range(self.order, len(tokens)):
            p *= self.cpt.get(tuple(tokens[i - self.order:i]), {}).get(tokens[i], 0.0)
        return p

    # ---- statistics for Part XIII ---------------------------------------------
    def stats(self):
        next_symbols = len(self.vocab) + 1                         # words + END
        observed_contexts = len(self.cpt)
        # contexts that could occur: any sequence of `order` symbols from START + vocab,
        # where START may only appear as a prefix of padding
        symbols = [START] + self.vocab
        if self.order == 1:
            possible = len(symbols)
        else:
            possible = 1 + len(self.vocab) + len(self.vocab) ** 2  # (S,S), (S,w), (w,w')
        return {
            "possible contexts": possible,
            "observed contexts": observed_contexts,
            "zero-probability (unseen) contexts": possible - observed_contexts,
            "full CPT free parameters": possible * (next_symbols - 1),
            "non-zero probabilities stored": sum(len(d) for d in self.cpt.values()),
            "observed free parameters": sum(len(d) - 1 for d in self.cpt.values()),
        }


def fmt(dist):
    return ", ".join(f"{w}: {p:.3f}" for w, p in sorted(dist.items(), key=lambda x: (-x[1], x[0])))


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


if __name__ == "__main__":
    rng = random.Random(407)
    m1 = NGramModel(CORPUS, order=1)
    m2 = NGramModel(CORPUS, order=2)

    header("Part III: training data with markers")
    for s in CORPUS:
        print(" ".join([START] + tokenise(s) + [END]))
    print("vocabulary:", m1.vocab)

    header("Part IV / Q3: first-order CPT  P(next | current)")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
        counts = dict(m1.counts[(w,)])
        print(f"{w:>8} | counts {counts}")
        print(f"{'':>8} | P      {fmt(m1.distribution(w))}")
    print("\nzero-probability transitions from the five required words:")
    for w in ["the", "cat", "dog", "sat", "ran"]:
        zeros = [v for v in m1.vocab + [END] if v not in m1.cpt[(w,)]]
        print(f"  P(. | {w}) = 0 for: {zeros}")

    header("Part VII / Q8: normalisation test  sum_v P(v | context) = 1")
    for name, model in [("first-order", m1), ("second-order", m2)]:
        totals = {c: sum(d.values()) for c, d in model.cpt.items()}
        bad = {c: t for c, t in totals.items() if abs(t - 1.0) > 1e-12}
        print(f"{name}: {len(totals)} contexts checked, min total {min(totals.values()):.15f}, "
              f"max total {max(totals.values()):.15f}, failures: {len(bad)}")
    for w in [START, "the", "cat", "dog", "sat", "ran"]:
        print(f"  {w:>8}: {sum(m1.distribution(w).values())}")

    header("Q7: behaviour on an unseen word")
    try:
        m1.distribution("bird")
    except KeyError as err:
        print("KeyError:", err)

    header("Part VIII / Q9: next-word prediction (argmax)")
    for w in ["the", "cat", "dog", "sat", "ran", "on", "to"]:
        print(f"P(X_t+1 | X_t = {w:<4}) = {{{fmt(m1.distribution(w))}}}  -> argmax: {m1.most_probable(w)}")

    header("Part IX: 20 sentences sampled from the first-order model")
    samples1 = [m1.generate("sample", rng) for _ in range(20)]
    training = set(CORPUS)
    for i, s in enumerate(samples1, 1):
        print(f"{i:2d}. {s:<45} {'(training sentence)' if s in training else '(new)'}"
              f"  P = {m1.sentence_probability(s):.3g}")
    with open("generated_first_order.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(samples1) + "\n")

    header("Part X / Q10: greedy vs sampling (first-order, 5 each)")
    print("Mode A - greedy:")
    for _ in range(5):
        print("  ", m1.generate("greedy"))
    print("Mode B - sampling:")
    for _ in range(5):
        print("  ", m1.generate("sample", rng))

    header("Part XI/XII: second-order CPT  P(next | previous two)")
    for ctx in [(START, START), (START, "the"), ("the", "cat"), ("the", "dog"),
                ("cat", "sat"), ("dog", "ran"), ("on", "the"), ("to", "the")]:
        print(f"P(X_t | {ctx[0]}, {ctx[1]}) = {{{fmt(m2.distribution(*ctx))}}}")
    try:
        m2.distribution("cat", "on")
    except KeyError as err:
        print("unseen context example -> KeyError:", err)

    header("Part XIII: comparison of the two models")
    s1, s2 = m1.stats(), m2.stats()
    print(f"{'measure':<38}{'first-order':>13}{'second-order':>14}")
    for key in s1:
        print(f"{key:<38}{s1[key]:>13}{s2[key]:>14}")

    samples2 = [m2.generate("sample", rng) for _ in range(20)]
    with open("generated_second_order.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(samples2) + "\n")
    for name, samples in [("first-order", samples1), ("second-order", samples2)]:
        uniq = set(samples)
        new = [s for s in uniq if s not in training]
        print(f"\n{name}: 20 samples, {len(uniq)} distinct, {len(new)} distinct sentences not in training data")
        for s in sorted(uniq):
            print(f"   {s:<45} {'(training)' if s in training else '(new)'}"
                  f"  P1 = {m1.sentence_probability(s):.3g}  P2 = {m2.sentence_probability(s):.3g}")
