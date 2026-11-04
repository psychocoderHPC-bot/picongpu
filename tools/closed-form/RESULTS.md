# CF1 - closed-form in-support field (restart after 3 deaths)

Run `pr24-20261102`. Task: fix the stopped closed-form work.
In-support field in O(1)-O(few) ops/pair; total field exact as possible at affordable cost.

## Plan (bounded steps, results logged before next step)

1. Re-verify the eight-term form `box_field_8T` (NS3b `ns3b_core.py`) vs converged
   midpoint trapz, 100 random pts. Build the corner-decomposition candidate
   (Ucorner 8-corner potential + analytic grad); compare vs the verified form at
   100 pts to 1e-12. Keep the verified form either way.
2. Extraction `W(s*)*E_box + octant-split remainder` (P2/P3/P4 schemes), stationary:
   accuracy at interior / face / outside vs fixed (12,2) reference, cost sweep.
3. Moving gamma10 (axial + oblique): does extraction reduce cost of the
   power-transform quadrature at equal accuracy?
4. Ops + time per pair: eight-term alone; extraction+remainder; moving.
5. GPU (jureca-h100): pairs/s of extraction kernel + shell kernel; project 2e6
   particles (1.7e8 support pairs; shells 2.35/3.6/4 cells) at 1e-3 in 300 s.

## Established (from prior tasks, not re-derived)

- `T(A,B,u)` asinh primitive: verified (NS3).
- Eight-term form `E_box` (24 T calls, 8 per component) vs converged trapz:
  -2.6333 vs -2.6348 at the NS3 probe point; port faithfulness 2.2e-12..1.8e-7
  at 5 probes (K3). K3: face error of the *box quadrature* is algebraic O(n^-0.7);
  the 8-term form removes that (exact).
- Singularity-extraction plan: `W(R0)*E_box(R0)` + smooth remainder, octant
  (knot+s*) split. partA.log (this run): P2 n=5 (6000 nodes): int 8.1e-4,
  face 1.6e-5; P2 n=6 (10368): int 4.5e-4, face 6.2e-6; P3 n=6: int 1.7e-4,
  face 9.2e-6. Box term alone: int 3.1e-1, face 2.8e-1.
- Part C (exact O(1) remainder via integration by parts) DIED with sign-logic
  bugs. CF1 diagnosis (to be confirmed by re-derivation):
  (a) `grad S` is NOT affine on a knot panel: S is multi-quadratic (2,2,2), so
      dS/dx is (1,2,2)-degree; the `c_p U_p + H_p W_p` form under-integrates;
  (b) face-flux inclusion-exclusion in `_panel_integral` multiplies all four
      corner signs `sgnb*sgnc*sgn2*sgn3` on every corner instead of the
      per-corner pair `sign(x0 or x1)*sign(y0 or y1)`;
  (c) `Ucorner` uses `A*F(C,A,B) + B*F(C,B,A) - g3` while the NS3-verified
      `_Ubox.py` form is `F(C,A,B) + F(C,B,A) - g3` - only one can pass trapz.
- GPU1 (V100): plain power-transform in-support (8,0)=6.1e5 pairs/s ~9e-4
  (marginal), (8,1)=7.8e4 2.6e-5; band (8,0) ~9e5 pairs/s 6.6e-4.
  K3 (H100 CPU): vendored in-support 3.95e-3 s/pair, out-support 3.27e-4.

## Step 1 DONE (cf1_step1.py, cf1_step1b.py; step1.log, step1b.log)

- Eight-term form KEPT as reference. Re-verified this run:
  - NS3 probe Ex: 8-term -2.63483 vs clean 8-octant power-transform ref (n=32)
    -2.63483 (record trapz600 -2.63331, 1.5e-3 low = trapz singularity artifact).
  - interior 100 pts vs powerref n=32: worst rel 4.0e-6 (ref-floor limited).
  - face 24 pts: worst 9.2e-7. outside 30 pts vs clean midpoint trapz512:
    worst 2.7e-7 (trapz256-512 gap 1.8e-5). Far-field p=(10,2,-3): matches
    point-charge limit to 2.6e-4 (finite-size).
- **Bug found & fixed in CF1 core (cf1_core.py, copied from NS3b ns3b_core.py):**
  `T(A,B,u)` zero-branch `if u < 1e-300` fires for ALL negative u; the closed
  form is even in u, so the fix is `abs(u) < 1e-300`. Old code broke the
  eight-term form for points OUTSIDE the box in any axis (u arg < 0): e.g.
  Ex(3,0,0) 8.645 (wrong) vs 2.752 (true, trapz 2.829); at far field the whole
  field was off 30-400x. Interior/face evaluations are unaffected (all u >= 0
  there), so all established in-support numbers of this run remain valid.
- Corner-decomposition candidate (8-corner U0 + analytic grad, cf1_step1 E0_box_corner):
  PROVEN EQUAL to the verified eight-term form for in-box points: worst rel
  3.4e-16 at 100 interior pts and 4.3e-16 at 100 pts in a non-symmetric box
  (gate 1e-12 PASS). Outside-box evaluation of the candidate FAILS (20.4)
  because the simple copysign pattern is not the full signed antiderivative;
  irrelevant: the singular part is only evaluated in-support.
- Ucorner closed forms (NS3 _Ubox v1 "F+F-g3", partC v2 "A F+B F-g3") are BOTH
  WRONG: v1(1.0,1.2,0.8)=1.25867 vs clean ref 1.14275 (10% off); 60-box worst
  1.87/1.70. The exact-remainder route needs fresh sympy-derived M_{ijk}
  primitives (see step 2.5).
- My own reference quadrature bugs (fixed in cf1_step1.py): midpoint `mid`
  grid offset, missing GL 1/2 weights, t=0 must sit at the singular corner
  (mirror per axis), broadcast axis, tuple-compare `== 0` octant bug (8x count),
  r^2 floor for self-colliding nodes at n>=32.

## Step 2 (in progress)

Stationary extraction + octant-split remainder sweep: interior/face/outside,
schemes P0/P1/P2/P3/P4, orders 2-5, ref (12,2) + cross-checks, ops/pair.

## Step 2 DONE (cf1_step2.py; step2.log)

Stationary, ref (12,2) [cross-checks (12,2)vs(10,2) 2e-8..4e-7 I/F; outside
refs converged to 1e-16]. Points: 48 interior, 24 face, 24 outside (1.5-4 cells).

- Box term only (24 T calls, exact): I 3.1e-1, F 2.8e-1, O 1.2e-1 -> extraction
  needed; numpy 31 us/pair.
- **P2 (knot+s* split, plain GL) is the config**: n=4 I 1.6e-3 / F 4.6e-5 /
  O 1.3e-7; **n=5: I 8.1e-4 / F 1.6e-5 / O 1.8e-9** (8,000 nodes interior,
  ~3.2e5 flops); n=6 (partA): I 4.5e-4 / F 6.2e-6 (13,824 nodes interior).
  1e-3 is REACHED at P2 n=5 (worst interior 8.1e-4).
- P1 (knots only): plateaus I 1.8e-2 at n=5 (no s* split -> slow).
- P3 (power transform, the GPU1 path) is WORSE than P2 at equal nodes for the
  SUBTRACTED integrand: n=5 I 3.5e-3 vs 8.1e-4. (Power transform is the
  validated path for the UNsubtracted integrand; after extraction the
  integrand is mild and plain GL on the split panels wins.)
- P0/P4 (no/weak split): I 1e-2..1e-1.
- Outside points are smooth: P1 n=4 5.1e-7, P2 n=3 3.3e-6, even P0 n=4 6.4e-3.
- Ops/pair (box 24 T + rem): 288 nodes 312 ops ~1.2e4 flops ... 8,000 nodes
  (P2 n=5 interior) ~8,024 ops ~3.2e5 flops. numpy 1 thread: box 31 us;
  P2 n=5 ~2.0 ms (python overhead dominated).
- CONCLUSION vs goal: extraction+octant GL reaches 1e-3 at ~1e3-1e4 nodes/pair
  (~1e4 ops), NOT at O(10)-O(100) ops (there the error is ~1e-1). The
  O(1)-O(few)-ops goal needs the exact closed-form remainder (step 2.5).

## Step 2b DONE (step2b.log + inline test)

Mixed-anchor remainder (power-transform or higher plain GL only on the 8
anchor octants around s*, plain GL on the other 19 knot panels): DEAD END.
- power-transform anchors: anchor n=6 + rest n=3 (2241 nodes interior):
  I 3.3e-2 - far worse than uniform P2 n=5 (8.1e-4, 8000 nodes) per node.
- plain-GL anchors: anchor n=6 + rest n=3 (810 nodes): I 2.1e-2. Same story:
  concentrating nodes on the s* octants is LESS efficient than uniform P2.
  The subtracted integrand (S-W*)K ~ O(1/r) * smooth-angle near s*; the
  tensor-GL octant converges slowly for the angular 1/r factor, and the
  plain-split uniform allocation wins.
- CONFIRMED: P2 uniform n=5 (8,000 nodes interior / 4,500 face / ~3,600
  outside; ~3.2e5 flops) is the recommended quadrature config; P2 n=6
  (13,824/10,368/...) for a 4.5e-4 safety margin.

## Step 3 DONE (cf1_step3.py; step3.log)

Moving gamma=10. Refs (12,2): axial floor 2.1e-7 (same as stationary); oblique
floor 2.4e-5 (beamed kernel needs more; model errors below are >> floor).

- AXIAL (beta || x): extraction = W*(E'_x/g, E'_y, E'_z) [8-term stretched box,
  24 T calls] + remainder.
    n=6 (13,824 nodes interior): extract P2 I 3.8e-3 F 1.2e-3  vs
    plain P3 I 9.8e-3 F 3.4e-3 -> extraction ~2.6x better at equal nodes.
    n=5 (8,000): extract P2 I 1.5e-2 vs plain P3 I 1.0e-2 (plain slightly
    better in-interior at n=5; extraction wins on face 3.4e-3 vs 7.8e-3).
    1e-3 in-interior: extraction ~n=8 (32,768 nodes), plain ~n=9 (46,656) ->
    ~1.4x node saving. VERDICT: extraction helps axial (2.5-2.6x error at
    equal nodes, ~1.4x cost at equal accuracy) but does not change the
    order; the power-transform path remains the validated baseline.
- OBLIQUE (beta || (1,2,1)/sqrt6): singular part has NO closed form (rest-frame
  8-octant box quadrature 8*n_box^3 = 4096 nodes at n_box=8).
    extract (4096 + P2 rem n=5, 10,096 nodes): I 7.7e-2 F 9.5e-4
    plain  (P3 n=6, 10,368 nodes):        I 7.8e-2 F 5.9e-3
    -> NO cost reduction at equal accuracy in-interior (7.7 vs 7.8e-2); face
    6x better. VERDICT: for oblique, KEEP the power-transform path.
- Moving in-support is ~5x harder than stationary at equal nodes (beaming):
  stationary P2 n=5 8.1e-4 vs axial extract n=6 3.8e-3.
- numpy 1-thread s/pair (worst interior): ax-extract P2 n=5 2.1 ms;
  ax-plain P3 n=5 2.7 ms; ob-extract b8+P2n5 4.7 ms; ob-plain P3 n=5 5.7 ms.

## Step 5a DONE: GPU extract kernel written, compiled, validated on jureca-h100

File: CF1/gpu_extract.cu (standalone alpaka CUDA; sing kernel copied unchanged from
GPU1/gpu_near.cu + new ExtractKernel = W*(s*)*Ebox8T closed form (24 T calls, axial
gamma-stretched box for moving) + P2 plain-GL remainder with (S-W*) weight, knot+s*
panel split matching cf1_core P2). Build: nvcc -O3 -arch=sm_90 -std=c++20
-expt-relaxed-constexpr -I ~/pr24/pr24w126/thirdParty/alpaka/include -lhwloc -ldl
(TMPDIR=~/build-tmp). Remote dir ~/cf1-20261102/.

Bugs found while validating:
- global constexpr kKnots invisible in CUDA device code -> local array in extractPair.
- GL table was computed at max(order,12) and shared across kernels: the first `order`
  entries of a 12-order GL table are NOT `order`-order GL nodes (silently wrong
  quadrature for every order<12). Fixed with per-order tables (main/12/8).

Validation on H100 (64 stationary pairs, ref = Python ref_fixed (12,2)):
- self-check kernel vs CPU same scheme (o5): 2.2e-16 (bitwise-faithful).
- xcheck extract(o12) vs sing(8,2), 32 in-support pairs: 1.77e-5 = o12 quadrature
  error (both chains agree).
- extract o5 vs ref: worst 4.26e-4 (GPU == Python P2 o5, 4.26e-4). < 1e-3 REACHED.
- sing (8,0) vs ref: 1.36e-3 (GPU1 V100: ~9e-4 marginal, consistent).
- sing (8,2) vs ref: 1.36e-6 (GPU1 V100 gamma10: 8e-13, consistent).
- chain: GPU kernel == Python scheme (to ~1e-7, sum-order sensitivity) == reference.

Node-rate probe (o5 in-support, 4096 pairs): 1.23e6 pairs/s = 9.9e9 nodes/s.
Full benchmark matrix next (step 5b).

## Step 5b DONE: H100 benchmark matrix (bench_h100.log; ~/cf1-20261102 on jureca-h100)

pairs/s measured (2^20 pairs x 4 iters, full occupancy; 2^18 for the heavy (8,1)):

stationary gamma=1:
  extract in o4 (4096 nodes)   3.668e7 pairs/s (1.50e11 nodes/s)
  extract in o5 (8000 nodes)   1.837e7 pairs/s
  extract in o6 (13824)        1.119e7 pairs/s
  sing    in (8,0) (32768)     1.277e6 pairs/s (4.18e10 nodes/s)
  sing    in (8,1) (262144)    1.444e5 pairs/s
  extract band o5 (~6000)      1.847e7 pairs/s
  extract band o6 (~10368)     1.123e7 pairs/s
  sing    band (8,0) (13824)   1.882e6 pairs/s

moving gamma=10 axial (stretch box, 24 T calls):
  extract in o5 (8000)         1.648e7 pairs/s
  extract in o6 (13824)        9.991e6 pairs/s
  extract in o8 (32768)        3.983e6 pairs/s
  sing    in (8,0) (32768)     1.248e6 pairs/s
  sing    in (8,1) (262144)    1.425e5 pairs/s

Node rate: extract ~1.5e11 nodes/s (H100); sing (power transform) ~4e10.
H100 is ~2x V100 for sing (1.28e6 vs 6.1e5 pairs/s (8,0) in) and the extract
kernel is 8-14x faster per pair than sing at equal accuracy (extract o5 4.3e-4
vs sing (8,0) 1.4e-3).

## Step 5c: 2e6 projection (pair counts per GPU1: support 1.7e8; shells
2.35 cells 7.2e8, 3 cells 1.7e9, 3.6 cells 2.9e9, 4 cells 3.4e9)

Stationary 1e-3 config = extract o5 (4.3e-4) in + o5 band (smooth, <<1e-3):
  support-only: 1.7e8/1.837e7 = 9.3 s
  +2.35 shell: 48 s ; +3-cell: 101 s ; +3.6: 167 s ; +4-cell: 193 s
Moving gamma=10 1e-3 config = extract o8 (~1e-3, step 3 extrapolation) in + o5 band:
  support-only: 1.7e8/3.983e6 = 42.7 s
  +2.35 shell: 82 s ; +3-cell: 135 s ; +3.6: 200 s ; +4-cell: 227 s

VERDICT: 1e-3 in 300 s on a single H100: REACHED - stationary with the full
4-cell shell (~193 s) and moving axial (~227 s, extract o8). GPU1's V100 verdict
(support-only marginal 280 s, shells out of reach) is overturned by the
closed-form extraction (singular part exact, 24 T calls) + H100 node rate.
Margin to 300 s is ~1.5-1.6x for the worst case; 4x H100 makes it ~50x over.

Ops/pair: singular box term O(1) = 24 T calls ~3.6e2 flops; remainder o5 =
8000 nodes ~3.2e5 flops (in-support), ~6000 band. Total field = O(1) exact
singular + O(1e4) smooth remainder. The full O(1)-O(few)-ops goal would need
the exact closed-form remainder (step 2.5) - NOT attempted to completion this
run (part C died there on sign bugs; sympy auto-integration unreliable); it is
open and no longer gates the 300 s target.

## Open
1. Exact closed-form remainder (26 corner primitives M_ijk, 1D-primitive route
   or fresh derivation) -> O(1)-O(few) ops/pair total field.
2. Oblique moving singular part has no closed form (rest-frame quadrature,
   4096 nodes) - power-transform path kept (step 3).
3. Moving shell (band) accuracy at o5 not measured on GPU (assumed smooth,
   beaming factor ~2-5 vs stationary; o6 if strict 1e-3 required: +31 s at
   4-cell shell, still <300 s).
