"""
CS F407 - Laboratory: Neural Models (Learning, Depth, Activations, Output Layers)

Runs every experiment in the lab and prints the evidence needed for the report:
  Task 1  - a single affine layer + sigmoid cannot learn XOR
  Task 4A - 2-2-1 network learns XOR (loss, probabilities, labels)
  Task 4B - backpropagation check (W1.grad, mean-loss averaging, finite differences)
  Task 4C - symmetry problem with identical (zero / constant) initialisation
  Task 4D - activation experiment (sigmoid / tanh / relu)
  Task 5  - three-class extension with softmax + cross-entropy, p - y check

Usage:  python neural_models_lab.py
"""

import torch
import torch.nn as nn

torch.set_printoptions(precision=4, sci_mode=False)

# ---------------------------------------------------------------------------
# Data: redundant safety sensors, disagreement warning = XOR
# ---------------------------------------------------------------------------
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y_XOR = torch.tensor([[0.], [1.], [1.], [0.]])
Y_3CLASS = torch.tensor([0, 1, 1, 2])  # 0: both off, 1: disagree, 2: both on

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def make_net(activation="tanh", hidden=2, out=1):
    """2 -> hidden -> out. The output layer returns logits (no final sigmoid);
    the sigmoid / softmax lives inside the loss for numerical stability."""
    return nn.Sequential(
        nn.Linear(2, hidden),
        ACTIVATIONS[activation](),
        nn.Linear(hidden, out),
    )


def train(model, loss_fn, targets, steps=3000, lr=0.05, trace=None):
    """Full-batch training with Adam. `trace(step, model)` is an optional hook
    called after each backward() and before the optimiser step."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial_loss = None
    for step in range(steps):
        opt.zero_grad()
        logits = model(X)                    # forward pass
        loss = loss_fn(logits, targets)      # scalar loss
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()                      # reverse-mode AD
        if trace is not None:
            trace(step, model)
        opt.step()                           # parameter update
    with torch.no_grad():
        final_loss = loss_fn(model(X), targets).item()
    return initial_loss, final_loss


# ---------------------------------------------------------------------------
# Task 1: a linear model cannot represent XOR
# ---------------------------------------------------------------------------
def task1_linear_model():
    header("Task 1: single affine layer + sigmoid on XOR")
    torch.manual_seed(0)
    linear = nn.Linear(2, 1)
    init, final = train(linear, nn.BCEWithLogitsLoss(), Y_XOR, steps=5000)
    with torch.no_grad():
        p = torch.sigmoid(linear(X))
    print(f"initial loss {init:.4f}  final loss {final:.4f}  (ln 2 = {torch.log(torch.tensor(2.)).item():.4f})")
    print("probabilities:", p.squeeze().tolist())
    print("weights:", linear.weight.data.tolist(), "bias:", linear.bias.data.tolist())
    correct = int(((p > 0.5).float() == Y_XOR).sum())
    print(f"correct: {correct}/4  (no line separates XOR, so at most 3/4 is possible; the "
          "BCE optimum is w = 0, p = 0.5 for every input, i.e. the model learns nothing)")


# ---------------------------------------------------------------------------
# Task 4A: basic learning check for the 2-2-1 network
# ---------------------------------------------------------------------------
def task4a_learning(seed=0, activation="tanh"):
    header(f"Task 4A: 2-2-1 network ({activation}), seed {seed}")
    torch.manual_seed(seed)
    model = make_net(activation)
    init, final = train(model, nn.BCEWithLogitsLoss(), Y_XOR)
    with torch.no_grad():
        p = torch.sigmoid(model(X)).squeeze()
    labels = (p > 0.5).long()
    print(f"initial loss {init:.4f}  ->  final loss {final:.6f}")
    for x, prob, lab, y in zip(X.tolist(), p.tolist(), labels.tolist(), Y_XOR.squeeze().long().tolist()):
        print(f"  x={x}  p={prob:.4f}  label={lab}  target={y}  {'OK' if lab == y else 'WRONG'}")
    print(f"all four correct: {bool((labels == Y_XOR.squeeze().long()).all())}")
    with torch.no_grad():
        h = model[1](model[0](X))
    print("learned hidden representation h(x) (rows = inputs):\n", h)
    return model


# ---------------------------------------------------------------------------
# Task 4B: backpropagation check
# ---------------------------------------------------------------------------
def task4b_backprop(seed=0):
    header("Task 4B: backpropagation check on W1 (first-layer weights)")
    torch.manual_seed(seed)
    model = make_net("tanh")
    W1 = model[0].weight

    # Mean loss over the batch -> gradient is the average of per-example gradients.
    loss = nn.BCEWithLogitsLoss(reduction="mean")(model(X), Y_XOR)
    model.zero_grad()
    loss.backward()
    batch_grad = W1.grad.clone()
    print("W1.grad = dL/dW1 (mean loss over 4 examples):\n", batch_grad)

    per_example = []
    for i in range(4):
        model.zero_grad()
        nn.BCEWithLogitsLoss()(model(X[i:i + 1]), Y_XOR[i:i + 1]).backward()
        per_example.append(W1.grad.clone())
    avg = torch.stack(per_example).mean(0)
    print("average of the 4 per-example gradients:\n", avg)
    print("max |difference|:", (avg - batch_grad).abs().max().item())

    # Central finite difference on every entry of W1 (float64 for accuracy).
    model64 = make_net("tanh").double()
    model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    X64, Y64 = X.double(), Y_XOR.double()
    loss_fn = nn.BCEWithLogitsLoss()
    model64.zero_grad()
    loss_fn(model64(X64), Y64).backward()
    W = model64[0].weight
    fd = torch.zeros_like(W)
    eps = 1e-6
    with torch.no_grad():
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] += eps
                lp = loss_fn(model64(X64), Y64).item()
                W[i, j] -= 2 * eps
                lm = loss_fn(model64(X64), Y64).item()
                W[i, j] += eps
                fd[i, j] = (lp - lm) / (2 * eps)
    print("finite-difference estimate of dL/dW1:\n", fd)
    print("max |autograd - finite difference|:", (fd - W.grad).abs().max().item())


# ---------------------------------------------------------------------------
# Task 4C: symmetry experiment
# ---------------------------------------------------------------------------
def task4c_symmetry():
    for name, value in [("all weights = 0", 0.0), ("all weights = 0.5 (identical, nonzero)", 0.5)]:
        header(f"Task 4C: symmetry experiment, {name}")
        model = make_net("tanh")
        with torch.no_grad():
            for prm in model.parameters():
                prm.fill_(value)

        def trace(step, m):
            if step in (0, 1, 10, 100, 1000, 2999):
                W1 = m[0].weight
                print(f"  step {step:4d}  W1 row0 {W1[0].tolist()}  row1 {W1[1].tolist()}"
                      f"  grad rows equal: {torch.equal(W1.grad[0], W1.grad[1])}")

        _, final = train(model, nn.BCEWithLogitsLoss(), Y_XOR, trace=trace)
        with torch.no_grad():
            p = torch.sigmoid(model(X)).squeeze()
        W1 = model[0].weight
        print(f"final loss {final:.4f}  probabilities {p.tolist()}")
        print(f"hidden rows identical after training: {torch.equal(W1[0], W1[1])}")


# ---------------------------------------------------------------------------
# Task 4D: activation experiment
# ---------------------------------------------------------------------------
def task4d_activations(seed=0, early_step=0, n_seeds=20):
    header(f"Task 4D: activation experiment (seed {seed}, grad norm at step {early_step})")
    print(f"{'activation':<10} {'final loss':>11} {'4/4 correct?':>13} {'early ||dL/dW1||_2':>20}")
    for act in ACTIVATIONS:
        torch.manual_seed(seed)
        model = make_net(act)
        norms = {}

        def trace(step, m):
            if step == early_step:
                norms["g"] = m[0].weight.grad.norm().item()

        _, final = train(model, nn.BCEWithLogitsLoss(), Y_XOR, trace=trace)
        with torch.no_grad():
            ok = bool(((torch.sigmoid(model(X)) > 0.5).float() == Y_XOR).all())
        print(f"{act:<10} {final:>11.6f} {str(ok):>13} {norms['g']:>20.6f}")

    # Four points and one seed are not enough to rank activations, so also report
    # how often each activation succeeds over several random initialisations.
    print(f"\nrobustness over {n_seeds} seeds (same data, same optimiser):")
    print(f"{'activation':<10} {'runs 4/4 correct':>17} {'mean early grad norm':>22}")
    for act in ACTIVATIONS:
        wins, gsum = 0, 0.0
        for s in range(n_seeds):
            torch.manual_seed(s)
            model = make_net(act)
            norms = {}

            def trace(step, m):
                if step == early_step:
                    norms["g"] = m[0].weight.grad.norm().item()

            train(model, nn.BCEWithLogitsLoss(), Y_XOR, trace=trace)
            with torch.no_grad():
                wins += bool(((torch.sigmoid(model(X)) > 0.5).float() == Y_XOR).all())
            gsum += norms["g"]
        print(f"{act:<10} {wins:>11}/{n_seeds:<5} {gsum / n_seeds:>22.6f}")

    # Diagnostic for the "Think About It" box: saturation vs dead ReLU.
    print("\ndiagnostic: pre-activations a1 and hidden derivatives at initialisation (seed 0)")
    for act in ("sigmoid", "relu"):
        torch.manual_seed(seed)
        model = make_net(act)
        with torch.no_grad():
            a1 = model[0](X)
            h = model[1](a1)
            d = h * (1 - h) if act == "sigmoid" else (a1 > 0).float()
        print(f"  {act}: a1 =\n{a1}\n  f'(a1) =\n{d}")


# ---------------------------------------------------------------------------
# Task 5: three-class extension
# ---------------------------------------------------------------------------
def task5_multiclass(seed=0):
    header("Task 5: three-class output with softmax + cross-entropy")
    torch.manual_seed(seed)
    model = make_net("tanh", hidden=2, out=3)
    print("final weight matrix shape (out x hidden):", tuple(model[2].weight.shape))
    print("logits per example:", model(X).shape[1])

    # dL/dz = (p - y) / N for mean cross-entropy over N examples. Checked at
    # initialisation, where the gradient is still large enough to read.
    z = model(X).detach().requires_grad_(True)
    nn.CrossEntropyLoss()(z, Y_3CLASS).backward()
    p = torch.softmax(z.detach(), dim=1)
    y = nn.functional.one_hot(Y_3CLASS, 3).float()
    print("\nautograd dL/dz at initialisation:\n", z.grad)
    print("(p - y) / N:\n", (p - y) / len(X))
    print("max |difference|:", (z.grad - (p - y) / len(X)).abs().max().item(), "\n")

    init, final = train(model, nn.CrossEntropyLoss(), Y_3CLASS)
    with torch.no_grad():
        logits = model(X)
        p = torch.softmax(logits, dim=1)
    print(f"initial loss {init:.4f}  ->  final loss {final:.6f}")
    for x, probs, y in zip(X.tolist(), p, Y_3CLASS.tolist()):
        pred = int(probs.argmax())
        print(f"  x={x}  p={probs.tolist()}  pred={pred}  target={y}  {'OK' if pred == y else 'WRONG'}")

    print("\nsoftmax vector for x=[0,1]:", p[1].tolist(), " sum =", p[1].sum().item())

    shifted = torch.softmax(logits + 100.0, dim=1)
    print("same logits + 100 ->", shifted[1].tolist(),
          " max |diff| =", (shifted - p).abs().max().item())

    naive = torch.exp(logits.double() + 1000.0)
    print("naive exp(logits + 1000):", naive[1].tolist(), "-> overflow, inf/inf = nan:",
          (naive / naive.sum(1, keepdim=True))[1].tolist())
    z = logits + 1000.0
    stable = torch.exp(z - z.max(1, keepdim=True).values)
    print("max-subtracted softmax  :", (stable / stable.sum(1, keepdim=True))[1].tolist())


if __name__ == "__main__":
    task1_linear_model()
    task4a_learning()
    task4b_backprop()
    task4c_symmetry()
    task4d_activations()
    task5_multiclass()
