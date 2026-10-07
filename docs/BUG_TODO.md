# BUG TODO — Mark Six codebase review (2026-10-05)

All items verified by running code, not just reading. Status: `[ ]` open, `[x]` fixed.

## P0 — data pipeline (silently corrupts data)

- [x] **F1 · `src/hkjc_fetch.py` — fetch failure is misread as "no draws in window" → permanent silent holes** — FIXED: tri-state `ok`/`empty`/`error` from `_post_graphql` up through `_fetch_window`/`fetch_since_last`/`fetch_last_n`; error windows never write a seg file or `last_all.json`; CLI exit 1 if any window errored (regression tests: `tests/test_pipeline.py::test_cli_since_last_*`, `test_fetch_history_*`).

- [x] **F2 · `src/build_history.py` — `_load_seg` kept unopened (non-`Result`) draws** — FIXED: `_load_seg` filters `is_result()` and non-list JSON; `build`/`update` also drop rows without numbers. Regression: `test_build_history_merges_sorts_filters_and_preserves` (Pending row stays out of the master).

- [x] **F3 · `src/build_history.py` — `--mode update` appended rows newest-first** — FIXED: `build` sorts by `_sort_key` (real date) before writing; `update` also sorts the merged master chronologically. Regression: same build_history test asserts date-monotonic output.

- [x] **F4 · `src/build_history.py` — raw cache overwrote richer master rows** — FIXED: `_merge_rec` merges field-wise; empty/newer-empty never overwrites a populated field, but a genuinely richer raw record does fill missing fields. Regression: `test_build_history_merges_*` (turnover 1000 preserved, 9999 picked up).

- [x] **F5 · `src/hkjc_fetch.py` — relative `--out` / `data/raw` resolved against CWD** — FIXED: `main()` resolves relative paths against `BASE_DIR`; `fetch_since_last`/`fetch_latest` default `out_dir` to the repo's `RAW_DIR`.

## P1 — model layer (the "cannot predict" experiment is not valid as written)

- [x] **M1 · `model/ml_models.py` — `predicted_numbers` predicted the LAST KNOWN draw, not the next** — FIXED: `_predict_top6` uses `window_features(nums, len(nums), lookback)` (t = next, not-yet-happened draw). Regression: `tests/test_models.py::test_window_for_next_draw_uses_latest_known_draw`.

- [x] **M2 · `model/stat_models.py` — Markov evaluation used `nums[-1]` as the state for every `t`** — FIXED: evaluation loop is walk-forward with `state=nums[t-1]`; display prediction still uses the final draw. Regression: `test_markov_causal_evaluation_and_sticky_prediction` (the old state-leak scores 7.46 on the same fixture; fixed causal scores 1.91).

- [x] **M11 (new, found while testing M2) · Markov transition-matrix row convention inverted between `_add_transition` and `_proba_from`** — FIXED: `_add_transition` used `int(i not in prev_set)` (row 1 = was absent) while `_proba_from` reads `int(i in present)` (row 1 = was present) — the two rows of the matrix were silently swapped. On real random data the two conventions are indistinguishable (both ≈ ln 49), which is why it hid; on any trending data the model predicts the **complement** of the true trend (sticky [44..49] → predicted {34..43}-ish). Unified convention: row 1 = was present, row 0 = was absent. Regression: same sticky-fixture test asserts `predicted_numbers == {44..49}`.

- [x] **M3 · sample-level train/test split cut through a draw** — FIXED: `base.draw_folds()` splits on whole draws (1-fold = chronological 70/30 of draws; multi-fold = contiguous draw blocks); `pick_draws` caps by whole draws; ML models and `predict.py` both use it. Regression: `test_draw_folds_disjoint_and_no_vector_leak`, `test_max_draws_limits_whole_draws`.

- [x] **M4 · baseline/metric semantics** — FIXED: every distribution the metric sees is normalised to sum 1 (Markov normalises its 49 per-number probs; `_proba_matrix` renormalises after padding) → all log-losses are directly comparable to the `ln 49` baseline; `uniform_baseline_log_loss` is emitted per result and the UI reads it (no more hardcoded 3.8918).

- [x] **M5 · zero-probability columns** — FIXED: `_proba_matrix` fills never-seen class columns with 1e-6 and renormalises rows; `class_coverage` reported per result. Regression: `test_proba_matrix_*`.

- [x] **M6 · MLP never converged (ll 9.15)** — FIXED: `StandardScaler` + early stopping + `max_iter=200`; on real data ll ≈ 3.96–4.0 (≈ baseline, honest "no signal"). LR also re-tuned (L2 `alpha=1.0`) and GNB decorrelated (`PCA(20)`) — the old unregularised LR (ll 33) and un-PCA'd GNB (ll 25) were calibration artifacts, documented in `ml_models.py`.

- [x] **M7 · `python model/registry.py` failed (relative imports)** — FIXED: registry inserts the repo root into `sys.path` and uses absolute `model.*` imports; `python model/registry.py --lookback 10` works.

- [x] **M8 · shared singleton `MODELS` with `_last_clf`** — FIXED: `make_models()` builds fresh instances per `run_all` call; `MODELS` kept only for back-compat.

- [x] **M9 · `predict.py` called `predict_proba` 49× per fold** — FIXED: `predict.py` now shares the registry skeleton (`_proba_matrix` once per fold).

- [x] **M10 · `sorted(top)` destroyed probability order** — FIXED: top-6 kept in `argsort` probability order in all models (the UI now labels chips by probability rank, not number).

## P2 — web app (`app/main.py`)

- [x] **A1 · `python app/main.py` → `NameError: uvicorn`** — FIXED: `import uvicorn` moved inside the `__main__` guard.
- [x] **A2 · CPU-heavy model run inside `async def` blocked the event loop** — FIXED: heavy endpoints are sync `def` (threadpool) + results cached keyed by (lookback, csv mtime+size) + `heavy_lock` serialises computation. Measured: health 0.14 s during a model run (was 13.7 s); warm /api/models ≈ 0.00 s (was 14.4 s).
- [x] **A3 · `/api/fetch`: no lock, no cap, no timeout** — FIXED: single-flight `_fetch_lock`, task dict TTL-pruned (1 h / max 20), subprocess timeouts (fetch 900 s / build 300 s).
- [x] **A4 · fetch "success" decided by row count** — FIXED: success requires both subprocess exit codes 0; a failed fetch sets `status: "error"` with a message, and the build step is skipped. Regression: `tests/test_api.py::test_fetch_failure_propagates_as_error`.
- [x] **A5 · `/api/fetch` unauthenticated** — FIXED: optional `FETCH_TOKEN` env → `X-Fetch-Token` header required (403 otherwise); single-flight returns `busy` instead of stacking fetches.
- [x] **A6 · `/api/model(s)` 500 when `lookback >= n_draws`** — FIXED: endpoints return 400 with a clear message; model `error` fields surfaced (registry no longer blanket-swallows).
- [x] **A7 · GET endpoints rewrote result files (write races)** — FIXED: API calls `run(df, write_results=False)`; files are written only by the CLI paths.
- [x] **A8 · `/api/health` hardcoded `data_source: "real"`** — FIXED: derived from the actually-loaded `df.attrs["data_source"]`; missing/non-real data → `unavailable` and 503s (the app never serves the synthetic sample).
- [x] **A9 · dead code / bad docstring** — FIXED: unused imports removed, docstring fixed.
- [x] **A10 (new) · `_get_models` and `_get_model` shared one cache key `(lookback, stamp)`** but stored a `list` vs a `dict` → `/api/models` could return the dict (or vice versa) after interleaved calls. FIXED: namespaced keys `("models", …)` / `("model", …)` + type guards. Regression: `test_models_and_single_model_cache_coexist`.

## P3 — analysis (`analyze/frequency.py`)

- [x] **S1 · `main_count`/`main_expected` were sums of ball *values*** — FIXED: `main_count` = total balls drawn, `main_expected` = count/49.
- [x] **S2 · `chi_square_uniform` dof/max_dev used the unmasked pool** — FIXED: `dof = participating categories − 1`; `max_dev` over the masked array; empty input guarded. (The caller reindexes each era to its own pool, so the mask is a defensive second layer.)
- [x] **S3 · `load()` crashed on bad rows / hard-coded date format** — FIXED: `_parse_numbers_row` skips malformed rows (counted in `dropped_rows`); `_parse_dates` accepts `DD/MM/YYYY` and ISO (bad dates counted in `bad_dates`, kept as NaT for the era filters to ignore). Regressions: `tests/test_pipeline.py::test_load_*`.
- [x] **S4 · unguarded `sys.stdout.reconfigure`** — FIXED in `analyze/frequency.py` (and `run_all.py`).
- [x] **S5 · `odd_even_mean` mislabelled** — FIXED: renamed to `even_count_mean` / `big_count_mean` (the UI does not consume the old names).

## P4 — frontend / deploy

- [x] **W1 · `index.html` unbounded poll loop** — FIXED: bounded `MAX_POLLS=200`, numeric `interval` guard (fallback 3 s), error state shows the task status, and the model sweep re-run was removed.
- [x] **W2 · unguarded `analysis.extra_chi2.p_value`** — FIXED: `|| {}` fallback.
- [x] **W3 · hardcoded baseline / `Math.min(...[])`** — FIXED: baseline read from `uniform_baseline_log_loss` (fallback `Math.log(49)`) in chart + table + delta column; empty-model guard.
- [x] **W4 (new) · prediction chips labelled by number order** — FIXED: tooltip/header now state chips are ordered by **probability rank** (matching the backend's argsort order).
- [x] **D1 · `docker-compose.yml` port mapping `8000:8000`-ish with `PORT` mismatch** — FIXED: `${PORT:-8000}:${PORT:-8000}` (Dokploy injects PORT; both sides must match or the proxy 502s).
- [x] **D2 · `.dockerignore` bloated image** — FIXED: `data/raw/**`, result JSONs, `tests/`, `docs/` excluded; the seed master CSV is kept (entrypoint copies it into the volume).
- [x] **D3 · `requirements.txt` unused deps** — FIXED: trimmed to the 7 actually imported (`requests pandas numpy scikit-learn scipy fastapi uvicorn`).
- [x] **R1 · `run_all.py --real/--latest` passed `--allow-sample`; `check=True` traceback** — FIXED: real/latest path never passes `--allow-sample`; fetch failure → non-zero exit and abort (no analysis on stale data); a new explicit `--allow-sample` flag exists for the offline path. Verified: offline `--latest` exits 1, master CSV untouched.
- [x] **T1 · no tests** — DONE: `tests/` added (33 tests, all offline): model folds/leakage/calibration, Markov causality + row convention, proba padding, insufficient-data, `load()` row handling, chi-square dof, `build_history` merge/sort/filter/preserve, `hkjc_fetch` tri-state (monkeypatched `_post_graphql`, no network), and the full API contract (`tests/test_api.py` via TestClient: 503/400/422/403/failure-propagation/single-flight/cache-coexistence).

## Notes from the review run
- All P0–P4 items fixed and covered by `tests/` (33 tests, `python -m pytest tests/` — fully offline; HKJC is unreachable from this machine by network policy, so fetch paths are tested via monkeypatched `_post_graphql`).
- `data/raw/last_all.json` stale `[]` from the pre-fix probing runs was deleted; the raw cache is gitignored and rebuildable. `data/mark6_history.csv` unchanged (4390 draws).
- Tracked artifacts `analyze/results/analysis.json` and `model/results/model_result.json` were regenerated from real data with the final code (chi-square p ≈ 0.27–0.83 across eras; predict ll 3.98 vs baseline 3.89).
- `/api/models` full run took 14.4 s pre-fix (event loop blocked); post-fix warm cache ≈ 0.00 s and `/api/health` stays at ~0.1 s even mid-training.

---

# Round 2 — full re-review (2026-10-06)

Pre-commit/pre-push pass. Full re-read of every tracked source file at review time (~4200 lines:
`app/`, `src/`, `analyze/`, `model/`, `tests/`, `run_all.py`, Docker/deploy, `index.html`)
plus the Simplified→Traditional conversion pass. Status: `[ ]` open, `[x]` fixed.

> **Re-verified 2026-10-06 (evening) against the current tree.** The working tree was
> refactored after the initial review (several files slimmed: `analyze/frequency.py` 1196→425,
> `model/base.py` 585→172, `app/main.py` 588→370, `src/build_history.py` 510→262, `run_all.py`
> 363→114; `src/hkjc_fetch.py` gained a retry loop). Items re-checked: R2-5 reclassified
> (current design is sound), R2-6 extended, R2-14/R2-15/R2-19 reworded to current code;
> all others still valid as written. Note: the refactored files themselves (the slimmed
> docstrings/logic) have **not** had a full review pass — treat a round-3 skim as follow-up.

## Conversion pass (done, verified)

- [x] **C1 · All tracked source/doc/data files converted to Traditional Chinese** — OpenCC `s2hk` in-place (formatting, string delimiters, `newline=''` preserved), then post-fixed to the codebase's *committed* conventions (measured from `git show HEAD`: `說`/`為`/`啟`/`裡`/`儀表`; the codebase uses standard traditional + Cantonese 嘅/咗/唔/係, **not** HK variants).
- [x] **C2 · OpenCC multi-target `STCharacters` drift fixed** — shared simplified chars (表|錶, 里|裏, 干, 只, 回, 周|週, 征|徵, 后|後, 系|係, 於|于, …; *the simplified chars in this list are intentional examples, not leftover drift*) cannot be auto-disambiguated by the pure-Python OpenCC reimplementation; fixed by context: 儀錶→儀表 (17), 裏→裡 (7), 説→說 (5), 啓→啟 (1), 回傳系「→回傳係「 (Cantonese, 1), 儀表盤→儀表板 (5, char-level 盘→盤 artifact; word convention is 板). Every remaining shared-char occurrence audited against context (週×40 徵×32 後×60 於×21 裡×8 只×43 迴×11 — all correct; 下游 keeps 游, which is correct traditional for 下游).
- [x] **C3 · Final verification** — zero Simplified-only characters remain in any tracked file (sweep: `\u4e00-\u9fff` minus the committed traditional set); `main.js` (gitignored 3.5 MB bundle) and `data/raw/*.json` contain 0 CJK and were untouched; `q.json` has 0 CJK; `python -m pytest -q` → 33 passed.

## To-do list — final status (all R2 items implemented 2026-10-06)

The round-2 re-review (post-refactor) found 25 issues (R2-1…R2-25). **All are now
implemented** in the same session; 10 new regression tests were added to `tests/`
(43 total, all offline, all passing). Implementation notes per item:

### Done (round 2)

- [x] **R2-1 · `fetch_history` reversed range** — `return 0` → `return 0, []` (the CLI unpacks a tuple; `[]` = no gaps, same as the 0-requests short-circuit).
- [x] **R2-2 · `/api/fetch` open when token unset** — unset `FETCH_TOKEN` → endpoint uniformly returns **503** (disabled) for all callers, with a `detail` the Web UI now surfaces. Required token → constant-time compare. README + compose document it; deploy must set `FETCH_TOKEN`.
- [x] **R2-3 · unpinned requirements** — exact pins (requests 2.34.2 / pandas 3.0.3 / numpy 2.4.6 / scikit-learn 1.9.0 / scipy 1.17.1 / fastapi 0.136.3 / uvicorn 0.49.0 + starlette 1.2.1 / httpx 0.28.1), matching the tested 3.14 env.
- [x] **R2-4 · `multiclass_samples` OOM risk** — **byte budget** (256 MB) instead of a fixed draw count, so default-lookback behavior is bit-identical; only oversized calls (e.g. lookback 60) get capped to the most recent N draws (absolute indices preserved) with a printed notice.
- [x] **R2-5 · `last_all.json` overwrite risk** — re-verified: range-based superset property, **no fix needed** (closed).
- [x] **R2-6 · `update_latest` append-only** — rewritten to full read → dedup-merge → **whole-file rewrite in date order** (temp file + `replace`); verifies header == `COLUMNS` first (legacy schema → abort, file untouched); fetch failure → no write. Backfilled (Pending→Result) draws now land in their correct chronological position.
- [x] **R2-7 · root + `chmod 777`** — `Dockerfile` runs as non-root `app` user (`useradd --system`, `chown` the data dir, `USER app`); entrypoint does a writability check with a clear root-volume diagnostic instead of `chmod 777`.
- [x] **R2-8 · `eligibility_stats` bottleneck** — vectorized: one presence matrix `(49, n)` + masked per-number `sum`, dates as one boolean array; ragged-row fallback to the old loop. ~49 DataFrame passes → 1.
- [x] **R2-9 · `.dockerignore`** — added `main.js` (3.5 MB), `tests/`, `q.json`, deploy files (`env`/compose; `docker-entrypoint.sh` kept — the build `COPY`s it).
- [x] **R2-10 · `analysis.json` tracked & rewritten** — both: `run()` now writes only when content changed (`_write_analysis`, mtime no longer churns → in-process caches stop thrashing), **and** `analyze/results/*.json` gitignored (`.gitkeep` kept) + `git rm --cached` so it no longer tracks.
- [x] **R2-11 · constant-time compare** — `hmac.compare_digest`.
- [x] **R2-12 · compose** — dropped fixed `container_name`; added `FETCH_TOKEN=${FETCH_TOKEN:-}` placeholder; added `healthcheck` on `/api/health`.
- [x] **R2-13 · `.env` tracked** — **kept tracked** (Dokploy compose `${PORT}` substitution depends on the file) but hardened: prominent no-secrets header, `PORT` only, plus a new tracked `.env.example`; `.gitignore` documents the contract. (Original "untrack" idea dropped — untracking would break the existing Dokploy deploy.)
- [x] **R2-14 · analysis cache unbounded + stale stamp** — cap at 8 keys (LRU half-evict) under the dedicated lock; stamp re-checked **after** compute — a CSV rewrite mid-compute is not cached.
- [x] **R2-15 · 429 terminal + no session reuse** — 429 now retries with backoff (respects `Retry-After`); one lazily-built `requests.Session` reused across requests.
- [x] **R2-16 · `run_all --real`** — `--real` now defaults to incremental (`--since-last`); `--full` for 1993→today; `--full`/`--latest` mutually exclusive. README documents both.
- [x] **R2-17 · `_load_seg` re-normalize** — shape guard: records already normalized (have `draw_id`, no GraphQL `id`) are accepted as-is, not re-normalized (which dropped their `numbers`).
- [x] **R2-18 · `gen_sample` `sys` import** — normal `import sys` in the import block (reconfigure kept in `try/except`).
- [x] **R2-19 · dead branch in `fetch_latest`** — removed the unreachable `if not draws` conditional in the success print.
- [x] **R2-20 · frontend** — SRI `integrity` + `crossorigin` on the Chart.js CDN tag; `Chart`-undefined notice instead of a silent broken dashboard; `esc()` HTML-escaping on all interpolated backend strings; Traditional-first font stack (PingFang TC / JhengHei / Noto Sans TC); UI surfaces `response.detail` (incl. the 503 token notice) on fetch.
- [x] **R2-21 · `q.json`** — `git rm --cached` + gitignored (file kept locally as the GraphQL scratch).
- [x] **R2-22 · misleading `.gitignore` comment** — comment now says `mark6_sample.csv` is intentionally tracked (deterministic, small).
- [x] **R2-23 · `_last_clf` reuse hazard** — `_fit_predict` stamps `clf._owner = model name`; `_predict_top6` refuses a clf owned by another model (returns None instead of borrowing weights).
- [x] **R2-24 · `src/__init__.py`** — added (docstring only), so `import src.xxx` is explicit for tests/CI.
- [x] **R2-25 · version parity** — `Dockerfile` base is now **`python:3.14-slim`** (was 3.12) → container interpreter matches the tested dev env, combined with the exact pins (R2-3). *Note: the container build/run itself still needs one CI or first-deploy run (no Docker CLI in this environment); the entrypoint writability check + non-root user make any volume/permission problem fail loudly with a readable message instead of corrupting state.*

## Pre-commit checklist (round 2 — final)

- [x] All R2 items implemented + 10 new regression tests; `python -m pytest -q` → **43 passed** (all offline; ~29 s).
- [x] `git rm --cached q.json` (R2-21) and `analyze/results/analysis.json` (R2-10) — both gitignored, `.gitkeep` keeps the dirs.
- [x] Zero Simplified-only characters in tracked files (sweep clean; the remaining candidates are the annotated C2 examples in this doc, the shared char 下游, and the shared 台 in 後台).
- [x] README updated for the changed contracts (`--real`/`--full`, `FETCH_TOKEN` 503 behavior, non-root container, Dokploy volume note).
- [ ] **`git add` the untracked `tests/`, `docs/BUG_TODO.md`, `src/__init__.py`, `.env.example`, `analyze/results/.gitkeep`** — part of the deliverable.
- [ ] **Commit message must state the mixed diff** — round-1 fixes + mid-session refactor/slimming + Traditional-Chinese conversion (C1–C3) + round-2 fixes (R2-1…R2-25), all in one tree.
- [ ] **Deploy (after push)** — set `FETCH_TOKEN` (random long string) in the Dokploy Environment Variables **before** exposing the domain, else `/api/fetch` returns 503 (by design). If a previous root-owned `mark6_app_data` volume exists, recreate it (back up `data/mark6_history.csv` first) for the non-root user.
- [ ] **First build** — confirm the 3.14-slim image builds and `/api/health` passes the compose healthcheck (the one thing not verifiable locally).
