# aclnnSwiGLU Probe Verdict

Agent SwiGLU-probe, ac01, CANN 8.3.RC1, 2026-04-21.

## Header analysis

Header `aclnn_swi_glu.h` exists. Two-phase API:

```cpp
aclnnStatus aclnnSwiGluGetWorkspaceSize(
    const aclTensor *x, int64_t dim, const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnSwiGlu(void *workspace, uint64_t ws, aclOpExecutor *exec,
                        aclrtStream stream);
```

**Structure:** takes a SINGLE concatenated tensor `x` plus `int64_t dim`, and
splits `x` in half along `dim` internally — first half = A (Swish branch),
second half = B (gate branch). Not separate gate/up inputs. Formula per
CANN-SKILLS `ops/activation/swi_glu/SKILL.md`:

  `out_i = Swish(A_i) * B_i`   (Swish ≡ SiLU here; β = 1)

**Dtype matrix:** the header is auto-generated and carries no explicit dtype
list in its doc-comment, but the SKILL hardware table lists Atlas A2/A3 and
Ascend 950 as supported. F16 × F16 → F16 empirically works on ac01 (this
probe). No `expertTokens` / `groupIndex` parameter on this (non-Quant)
variant, so FFNV3-style MoE-gating rejection does not apply. The MoE-gated
form is the separate `aclnnSwiGluQuant` (has `groupIndexOptional`), which
we do not use here.

**W8 sibling noted:** `aclnnSwiGluQuant`, `aclnnDequantSwigluQuant`,
`aclnnGroupedMatmulSwigluQuant`. Out of scope for this probe — our
prod chain's SiLU+Mul is F16, only the matmuls are W8.

## Standalone correctness

Harness: `/tmp/swiglu_probe/test_swiglu.cpp`. Inputs: `gate[1,3072]` F16 and
`up[1,3072]` F16, random ~U(-1,1), seed 0xC0FFEE. Reference path =
`aclnnSilu(gate) → aclnnInplaceMul(up)`. Candidate path = pre-concat
`[gate || up]` into `[1, 6144]`, `aclnnSwiGlu(x, dim=-1, out)`. Results
deterministic across 3 runs:

| Path                         | max_abs_diff vs ref | max_rel_diff | mismatches | Wall median (μs) |
|------------------------------|---------------------|--------------|------------|------------------|
| ref (Silu + InplaceMul)      | 0.0                 | 0.0          | 0 / 3072   | 35.09            |
| aclnnSwiGlu                  | 4.883e-4 (= 1 ulp)  | 5.13e-3      | 817 / 3072 | 34.31            |

Speedup: 1.02x (essentially flat — the concat cost plus single-kernel
launch roughly cancels the saved dispatch).

The max_abs_diff is **exactly** one F16 ulp in [-1,+1] (2^-11 = 4.8828e-4);
the mismatches come from the fused kernel rounding silu(A)*B once,
whereas the reference rounds silu(A) first, then multiplies by B. Drift is
rounding-order only — no algorithmic divergence.

## Verdict

- [ ] GREEN
- [x] YELLOW
- [ ] RED

**Op accepts F16×F16→F16 and runs correctly, but is NOT byte-identical
to the Silu+InplaceMul chain** (1-ulp drift on 27% of elements).
Frame-count-identity gate still applies if wired; byte-identity gate
cannot be met unless we accept re-baselining.

Additional cost to integrate, not measured by this probe: the candidate
requires a concatenated `[gate || up]` input. In the current engine, gate
and up are produced by two separate W8 matmul calls into two separate
device buffers. To feed `aclnnSwiGlu` without an extra concat dispatch,
we'd need either (a) re-layout the matmul outputs so gate/up land in
adjacent halves of a single buffer, or (b) add an `aclnnCat`/copy step
that itself costs ~1 kernel and eats the 0.8-μs saving. Wall-time win is
therefore best-case 1.02x and realistically break-even.

## Patch

- `/tmp/swiglu_probe/test_swiglu.cpp` on ac01 (standalone harness).
- `/tmp/swiglu_probe.patch` on ac01, scp'd to Mac `/tmp/swiglu_probe.patch`.
- No engine files touched.

Build: `g++ -O2 -std=c++17 test_swiglu.cpp -I$ASCEND/include -L$ASCEND/lib64 -lascendcl -lopapi -lnnopbase -o test_swiglu`.

## Recommendation for PM

Do not dispatch a wiring agent. `aclnnSwiGlu` is functionally correct for
our dtype/shape (unlike FFNV3 A16W8 which was flat-rejected), so the deck
can honestly say "the op works" — but the two practical blockers are:
(1) the 1-ulp drift breaks byte-identity, forcing a frame-count-parity
re-baseline for any PR, and (2) the single-tensor concat requirement
converts the claimed "save 1 dispatch" into either a buffer-layout
refactor or a wash-trade concat dispatch, with measured wall win of
1.02x (~0.8 μs out of 35 μs; below noise floor at CP-frame scope: 75
dispatches/frame × 0.8 μs ≈ 60 μs/frame, <0.2% at our current 31.6 fps).
Recommend parking this in the deck alongside FFNV3 under "fused ops
evaluated, not shipped — byte-identity + layout cost > benefit."
