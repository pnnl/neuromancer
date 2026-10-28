"""
Seal a trained NeuroMANCER solution map's output into a re-runnable, tamper-evident bundle.

This example extends the parametric QP in Part_2_pQP.py: it trains the same pQP solution map, then seals the
solution x*(p) it produces for a chosen parameter point into a proof-core evidence bundle. The bundle pins the
trained weights (by content hash) and the parameters as the re-run recipe, and records the solution by content
hash. Anyone can then, OFFLINE and with no account:

  1. VERIFY the bundle (every hash recomputes; the signature checks out),
  2. REPRODUCE the result (re-load the same weights, re-run the map on the same parameters, and confirm the
     solution re-derives), and
  3. see TAMPERING caught (changing any sealed field fails verification).

Why this might be useful for NeuroMANCER: a trained solution map is a deterministic function of its weights and
inputs, so its output is independently re-derivable. This turns "here is x*(p)" into "here is x*(p), and here
is a record a third party can re-run to confirm it."

proof-core is MIT and pure standard library (no deps, not on PyPI). Install it alongside neuromancer:
    pip install neuromancer
    pip install "git+https://gitlab.com/proof-core/proof-core.git"

Run:
    python proof_core_sealed_pQP.py

HONEST SCOPE: re-derivation holds to the extent the forward pass is deterministic (fixed trained weights,
CPU/float determinism; GPU atomics widen the band). The seal attests the solution re-derives from this map on
these parameters and is unaltered; it does not attest the learned map is optimal or physically correct. Under
proof-core's public demo key the bundle verifies as "integrity-only"; signed with a production key it also
carries origin.
"""
import json

import torch
import torch.nn as nn

import neuromancer.slim as slim
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.problem import Problem
from neuromancer.system import Node
from neuromancer.trainer import Trainer

from proof_core import Case, Fidelity, Result, Verdict, build_bundle, canon, sha
from proof_verify.verify import classify_bundle

CAPABILITY = "sci-compute:neuromancer"
_CAVEAT = (
    "the seal attests the solution re-derives from this NeuroMANCER map identity and these parameters on "
    "re-run; it re-derives only to the extent the forward pass is deterministic (fixed weights, CPU/float "
    "determinism), and it does not attest that the learned map is optimal or physically correct"
)


def build_pqp_problem():
    """The Part_2_pQP solution map + objective + constraints, as a NeuroMANCER Problem."""
    func = blocks.MLP(insize=2, outsize=2, bias=True, linear_map=slim.maps["linear"],
                      nonlin=nn.ReLU, hsizes=[80] * 4)
    sol_map = Node(func, ["p1", "p2"], ["x"], name="map")
    x = variable("x")[:, [0]]
    y = variable("x")[:, [1]]
    p1, p2 = variable("p1"), variable("p2")
    obj = (x ** 2 + y ** 2).minimize(weight=1.0, name="obj")
    q = 100.0
    c1 = q * (-x - y + p1 <= 0)
    c1.name = "c1"
    c2 = q * (x + y - p1 - 5 <= 0)
    c2.name = "c2"
    c3 = q * (x - y + p2 - 5 <= 0)
    c3.name = "c3"
    c4 = q * (-x + y - p2 <= 0)
    c4.name = "c4"
    return Problem([sol_map], PenaltyLoss([obj], [c1, c2, c3, c4]))


def loaders(seed=408, nsim=3000):
    torch.manual_seed(seed)
    def sample():
        return {"p1": torch.FloatTensor(nsim, 1).uniform_(1.0, 11.0),
                "p2": torch.FloatTensor(nsim, 1).uniform_(1.0, 11.0)}
    out = []
    for name in ("train", "dev", "test"):
        ds = DictDataset(sample(), name=name)
        out.append(torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0,
                                               collate_fn=ds.collate_fn, shuffle=(name == "train")))
    return out


def weights_digest(state_dict):
    """Content hash of the trained weights -> the map's identity, so the sealed solution is pinned to the exact
    trained map."""
    canon_sd = {k: v.detach().cpu().tolist() for k, v in sorted(state_dict.items())}
    return sha(canon(canon_sd))


def solve(problem, p1, p2):
    """Run the trained solution map at one parameter point; return the JSON-safe solution x* = (x, y)."""
    batch = {"p1": torch.tensor([[float(p1)]]), "p2": torch.tensor([[float(p2)]]), "name": "eval"}
    with torch.no_grad():
        out = problem.step(batch)
    x = out["x"].detach().cpu().numpy().flatten().tolist()
    return {"x1": round(float(x[0]), 6), "x2": round(float(x[1]), 6)}


def seal(map_id, parameters, solution):
    """Seal the solution into a proof-core bundle using the public MIT kernel (demo key -> integrity-only)."""
    case = Case(capability=CAPABILITY, inputs={"map": map_id, "parameters": parameters})
    result = Result(capability=CAPABILITY, fidelity=Fidelity.SCREENING, verdict=Verdict.PASS,
                    outputs={"solution_sha256": sha(canon(solution)), "x1": solution["x1"], "x2": solution["x2"]},
                    caveats=(_CAVEAT,), provenance={})
    bundle = build_bundle(case, result, engine="neuromancer-proof-core-example", engine_version="0.1.0",
                          data_library=("neuromancer-trained-weights", map_id))
    return bundle


def main():
    train_loader, dev_loader, test_loader = loaders()
    problem = build_pqp_problem()
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    trainer = Trainer(problem, train_loader, dev_loader, test_loader, optimizer,
                      epochs=400, patience=100, warmup=100,
                      train_metric="train_loss", dev_metric="dev_loss",
                      test_metric="test_loss", eval_metric="dev_loss")
    best_model = trainer.train()
    problem.load_state_dict(best_model)

    # The solution map's output for one parameter point, and the identity of the map that produced it.
    params = {"p1": 5.0, "p2": 5.0}
    map_id = weights_digest(problem.state_dict())
    solution = solve(problem, params["p1"], params["p2"])
    print(f"\nsolution map x*(p1=5, p2=5) = {solution}")

    # 1) SEAL
    bundle = seal(map_id, params, solution)
    print(f"sealed. bundle_sha256 = {bundle['bundle_sha256']}")

    # 2) VERIFY (offline, no account)
    v = classify_bundle(bundle)
    print(f"verify -> state {v['state']!r}  integrity_ok {v['integrity_ok']}")

    # 3) REPRODUCE: re-load the SAME weights into a fresh Problem, re-run the map on the SAME parameters.
    fresh = build_pqp_problem()
    fresh.load_state_dict(best_model)
    re_solution = solve(fresh, params["p1"], params["p2"])
    reproduced = sha(canon(re_solution)) == sha(canon(solution))
    print(f"reproduce -> re-derived {re_solution}; matches sealed solution: {reproduced}")

    # 4) TAMPER: change a sealed parameter; verification must fail.
    tampered = json.loads(json.dumps(bundle))
    tampered["manifest"]["inputs"]["value"]["inputs"]["parameters"]["p1"] = 6.0
    print(f"tamper (p1 -> 6.0) -> integrity_ok {classify_bundle(tampered)['integrity_ok']} (caught)")


if __name__ == "__main__":
    main()
