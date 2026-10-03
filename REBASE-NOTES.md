# TensorFold v0.6.4 rebase (branch tensorfold-v0.6.4)

The 70 patches of recipe v1.5 were diffs against TensorFold **v0.6.0**'s site-packages. This
branch rebases them onto **v0.6.4** (676de9f, 2026-10-03; 170 commits between the tags). To try it:

    ./start.sh          # from this folder; scripts/local.sh already sets TF_VERSION=v0.6.4

Revert to production: check out main (its patches target v0.6.0 and local.sh there has no
TF_VERSION override), or just run start.sh from the production folder. Nothing here touches the
production checkout, image, kernel cache (this branch uses ~/.cache/tensorfold-glm53-v064) or
container until start.sh runs.

## What changed

- **69 patches** rebased; every one applies cleanly, in order, on a pristine v0.6.4 tree
  (verified: fresh tree + `patch -p0` per patch, plus `py_compile` of every touched .py).
- **0058-server-client-gone-poll is deleted**: TensorFold v0.6.2 #218 implements it.
- The patches hash changes, so prepare.sh builds a new image locally (no prebuilt
  `v0.6.4-<hash>` exists on ghcr yet) and kernels recompile at first start (~minutes a Spark).
- Upstream v0.6.4 does **not** add concurrency to GLM-5.3-Flash (its `--parallel` is Flash
  Next only; the glm5_next doc still says "serves one request at a time"), so the recipe's
  whole multi-stream stack stays.

## Decisions worth knowing

- **EXL3 route**: upstream added a generic EXL3 path (`cuda/exl3/experts`, `exl3_generic`)
  alongside the old `glm5_next/cuda/exl3_mm`. The recipe keeps **its** `exl3_mm` route (all the
  recipe's performance work: prompt experts, prompt kernels, decode kernels, index regs);
  the generic route stays in the tree for other shapes. The Buffers/moe_block code is the
  recipe's (plan for EXL3, unconditional `grouped.route`, `Scratch(..., prompt=...)`).
- **Tool calls**: the merged server runs both streamers — the recipe's `GlmCallStreamer`
  (whole-call-sent-once, cut-call, open-call handling) for GLM, upstream's
  `ToolCallStreamer` for non-GLM. `finish` is cut-aware; `call_deltas`/`calls_streamed` and
  `stop_sequence` all ride in the reply.
- **The unified communicator (#219)** is used: `open_comm(rank, world, master, port)` in the
  engine; the RoCE wrapper (0006/0052) wraps whatever it returns.
- **Upstreamed-and-dropped hunks** (v0.6.1-v0.6.3 already contain them, sometimes in nicer
  form): 0023 effort max (#225-era effort ladder), 0037 tokenize routes (#237), 0057 thinking
  alias (#236), 0059 refused bodies (#181/#244), 0060 keep earlier reasoning (#236), 0058
  (#218). Their files resolve to v0.6.4's text; some still carry recipe extras (0057's
  once-per-kind log of ignored thinking values; 0037 keeps v0.6.4's CapacityError handling).
- **qmm kernel name bumped to `tensorfold_qmm_v6`** (recipe changes + upstream's
  qmm_group.cu both in the source set); later recipe bumps chain from there.
- **Vision**: v0.6.4's `ImageLimits` plumbing (#239), `_IMAGE_ROLES` (#235) and `read_body`
  (#244) are kept; the recipe's GLM frontend (`vision/glm.py`, `load_images`/`load_videos`
  hooks, `tool_media` marks) rides on top. Body limit raised to 96 MiB at the read
  (50 pictures / 4 clips as data URLs), as the recipe did.
- **/metrics** carries both: v0.6.4's decode histogram, footprint and vLLM-name mirrors, and
  the recipe's `_health` tables; reads go through the recipe's `_read` retry helper.
- **/v1/decisions** (#281) integrated: `follow()` reads a decision header (len 2) before the
  chat header, whose `images` and `*rest`/`SHARED_MOST` (0015) split is preserved.

## First-start fix (2026-10-03)

The first attempt failed compiling `tensorfold_qmm_v6`: patches 0016 and 0019 had both left a
copy of the cfg decode-kernel set (`qmm_cfg`, `dispatch_cfg`, `qmm_cfg_cuda`) in
`cuda/kernels/qmm.{cpp,cu}`. The final tree now carries production's single 0-15 config set
(verified byte-equal to the v0.6.0 recipe's final kernel text); the dedupe rides in the last
patch of the set. Everything before that point had worked: the image built with all 69 patches,
`tensorfold_roce_v1`, `tensorfold_glm_l2pf_v2` and `tensorfold_glm_hc_v2` compiled, and both
ranks loaded ~93 GiB of weights before the qmm build failed.

## Second-start fix (2026-10-03)

All seven CUDA extensions then compiled (qmm_v6 included) and both ranks loaded ~92 GiB,
but graph capture hit `Exl3RoutedExperts has no attribute shared_suh`: upstream 8613488
loads EXL3 experts as its universal `Exl3RoutedExperts`, while the recipe's `exl3_mm.routed`
needs its own `Exl3Experts`. `weights.py` now builds the recipe's object exactly as the
v0.6.0 recipe did (`exl3_words` + the 12-arg constructor); the only remaining weights.py
delta vs production is v0.6.4's mixed-bit refusal (#226), which our uniform-4bpw checkpoint
passes. `exl3_mm.py` is byte-identical to production's.

## Third-start fix (2026-10-03)

Graph capture, drafter calibration, vision and the parallel pool all came up; the server
died constructing the app: the merged ThinkingOffTemplate kept v0.6.4's `clear=` parameter
name while its caller (0060, as in production) passes `clear_thinking=`. The class is now
production's verbatim. A tree-wide construction check (every class instantiation's
keywords against its __init__ / dataclass fields, import-resolved) now reports only two
pre-existing upstream notes in deepseek_v4 — nothing in the GLM path.

## Not verified here (no GPU on this box)

- CUDA kernel compilation and the actual serve (start.sh does both). py_compile passed for
  every touched .py; the .cpp/.cu hunks were merged by hand and checked for duplicate
  definitions, but only a real build compiles them.
- Reply exactness ("drafted == serial") and the throughput numbers: re-measure on v0.6.4
  (upstream notes tree-attention fp32 partial sums #268 can shift last bits vs 0.6.3).

## Method

Three-way merges per patch against a reference tree (v0.6.0 + all original patches), with
upstream's per-file diffs as context; conflicts resolved by hand; the final patch set
regenerated from the merge tree's git history (one commit per patch). 40 files were
hand-resolved; 72 merged clean by diff3; the rest apply untouched.
