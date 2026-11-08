"""Final cell: merge decoded results into a full-coverage submission.json."""
import json
import os
import traceback

import numpy as np

import arc_v6_config as cfg
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

OUTPUTS = os.environ.get("ARC_V6_OUTPUTS", "/kaggle/inference_outputs")


def load_decoder(dataset, stores):
    decoder = ArcDecoder(dataset, n_guesses=2)
    for store, run_name in stores:
        if os.path.isdir(store):
            decoder.load_decoded_results(store, run_name=run_name)
    known = set(dataset.keys)
    stale = [k for k in decoder.decoded_results if k not in known]
    for k in stale:  # never let an unknown key break the merge
        del decoder.decoded_results[k]
    return decoder


def keep_a_top1(sel_a, sel_p):
    """v8: pass A's top-1 is always one of the two attempts; the other is the best distinct pooled grid.

    With no pass-B candidates for an output, the pooled ranking equals pass A's, so this is v7's choice.
    """
    selected, changed = {}, 0
    for bk in set(sel_a) | set(sel_p):
        ranked_a, ranked_p = list(sel_a.get(bk, [])), list(sel_p.get(bk, []))
        if not ranked_a:
            selected[bk] = ranked_p[:2]
        else:
            first = ranked_a[0]
            other = next((g for g in ranked_p + ranked_a[1:] if not np.array_equal(g, first)), None)
            selected[bk] = [first] + ([other] if other is not None else [])
        before = ranked_a[:2]
        changed += len(before) != len(selected[bk]) or any(
            not np.array_equal(x, y) for x, y in zip(before, selected[bk]))
    return selected, changed


def model_submission(challenges_path, outputs_dir, solutions_path=None, use_b=True):
    data = ArcDataset.from_file(challenges_path)
    if solutions_path:
        data = data.load_replies(solutions_path)
    multi = data.split_multi_replies()
    outputs_b = cfg.outputs_dir_b()
    decoder_a = load_decoder(multi, [(outputs_dir, "")])
    pooled = load_decoder(multi, [(outputs_dir, ""), (outputs_b, ".b")]) if use_b else decoder_a
    selected, changed = keep_a_top1(decoder_a.run_selection_algo(), pooled.run_selection_algo())
    stats = {"outputs_a": len(decoder_a.decoded_results), "outputs_pooled": len(pooled.decoded_results),
             "pass_b_files": len(os.listdir(outputs_b)) if use_b and os.path.isdir(outputs_b) else 0,
             "changed_vs_pass_a": int(changed)}
    return data, decoder_a, pooled, data.get_submission(selected), stats


def merge(test, fallback, candidate):
    """Per attempt: take the model grid if valid, else keep the fallback grid."""
    merged, used = {}, 0
    for key, task in test.items():
        entries = []
        for i in range(len(task["test"])):
            entry = dict(fallback[key][i])
            for attempt in ("attempt_1", "attempt_2"):
                try:
                    grid = np.asarray(candidate[key][i][attempt]).tolist()
                except Exception:
                    continue
                if cfg.valid_grid(grid):
                    entry[attempt] = grid
                    used += grid != [[0]]
            entries.append(entry)
        merged[key] = entries
    return merged, used


def main(work_dir=None, outputs_dir=OUTPUTS):
    work_dir = work_dir or cfg.WORK_DIR
    test_path = cfg.test_path()
    with open(test_path) as f:
        test = json.load(f)
    fallback = cfg.fallback_submission(test)
    final, used, error, pool_stats = fallback, 0, None, {}
    if cfg.is_rerun():
        try:
            try:
                _, _, _, candidate, pool_stats = model_submission(test_path, outputs_dir)
            except Exception:  # v8: a pass-B problem must never cost pass A's answers
                pool_stats = {"pooling_error": traceback.format_exc()[-1500:]}
                cfg.log("pooling_failed_using_pass_a")
                _, _, _, candidate, stats_a = model_submission(test_path, outputs_dir, use_b=False)
                pool_stats.update(stats_a)
            final, used = merge(test, fallback, candidate)
            cfg.validate_submission(test, final)
        except Exception:
            error = traceback.format_exc()
            final, used = fallback, 0
    else:
        # Commit smoke: model ran on evaluation tasks; report them separately, keep the
        # test-file submission full-coverage fallback so the version stays submittable.
        comp = cfg.competition_dir()
        sol = os.path.join(comp, "arc-agi_evaluation_solutions.json")
        try:
            data, decoder, pooled, candidate, pool_stats = model_submission(
                cfg.work_challenges_path(), outputs_dir, sol if os.path.exists(sol) else None)
            cfg.atomic_json(os.path.join(work_dir, "eval-smoke-submission.json"), candidate)
            if pooled.decoded_results and os.path.exists(sol):
                for label, dec in (("pass_a", decoder), ("pooled", pooled)):
                    try:
                        print(f"*** {label}")
                        dec.benchmark_selection_algos()
                    except Exception as exc:  # author benchmark fails when nothing is solved
                        cfg.log("benchmark_skipped", label=label, error=type(exc).__name__)
                ran = sorted({b.rsplit("_", 1)[0] for b in pooled.decoded_results})
                sel_a = decoder.run_selection_algo()
                solved_a = sum(np.mean([any(np.array_equal(r, g) for g in sel_a.get(f"{k}_{i}", [])[:2])
                                        for i, r in enumerate(data.replies[k])]) for k in ran)
                solved = sum(np.mean([any(np.array_equal(r, e[a]) for a in ("attempt_1", "attempt_2"))
                                      for r, e in zip(data.replies[k], candidate[k])]) for k in ran)
                cfg.log("eval_smoke_score", solved=float(solved), tasks=len(ran), solved_pass_a=float(solved_a),
                        contaminated_by_pretraining=True)
        except Exception:
            error = traceback.format_exc()
    cfg.validate_submission(test, final)
    cfg.atomic_json(os.path.join(work_dir, "submission.json"), final)
    record = {"rerun": cfg.is_rerun(), "test_tasks": len(test),
              "test_inputs": sum(len(t["test"]) for t in test.values()),
              "model_attempts_used": used, "error": error,
              "decoded_files": len(os.listdir(outputs_dir)) if os.path.isdir(outputs_dir) else 0, **pool_stats}
    cfg.atomic_json(os.path.join(work_dir, "v6-assembly.json"), record)
    cfg.log("assembled", **{k: v for k, v in record.items() if k != "error"}, error=bool(error))
    if error:
        print(error[-3000:])
    return record


if __name__ == "__main__":
    main()
