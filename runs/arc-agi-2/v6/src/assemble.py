"""Final cell: merge decoded results into a full-coverage submission.json."""
import json
import os
import traceback

import numpy as np

import arc_v6_config as cfg
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

OUTPUTS = os.environ.get("ARC_V6_OUTPUTS", "/kaggle/inference_outputs")


def model_submission(challenges_path, outputs_dir, solutions_path=None):
    data = ArcDataset.from_file(challenges_path)
    if solutions_path:
        data = data.load_replies(solutions_path)
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    if os.path.isdir(outputs_dir):
        decoder.load_decoded_results(outputs_dir)
    known = set(decoder.dataset.keys)
    stale = [k for k in decoder.decoded_results if k not in known]
    for k in stale:  # never let an unknown key break the merge
        del decoder.decoded_results[k]
    return data, decoder, data.get_submission(decoder.run_selection_algo())


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
    final, used, error = fallback, 0, None
    if cfg.is_rerun():
        try:
            _, _, candidate = model_submission(test_path, outputs_dir)
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
            data, decoder, candidate = model_submission(cfg.work_challenges_path(), outputs_dir,
                                                        sol if os.path.exists(sol) else None)
            cfg.atomic_json(os.path.join(work_dir, "eval-smoke-submission.json"), candidate)
            if decoder.decoded_results and os.path.exists(sol):
                try:
                    decoder.benchmark_selection_algos()
                except Exception as exc:  # author benchmark fails when nothing is solved
                    cfg.log("benchmark_skipped", error=type(exc).__name__)
                ran = sorted({b.rsplit("_", 1)[0] for b in decoder.decoded_results})
                solved = sum(np.mean([any(np.array_equal(r, e[a]) for a in ("attempt_1", "attempt_2"))
                                      for r, e in zip(data.replies[k], candidate[k])]) for k in ran)
                cfg.log("eval_smoke_score", solved=float(solved), tasks=len(ran), contaminated_by_pretraining=True)
        except Exception:
            error = traceback.format_exc()
    cfg.validate_submission(test, final)
    cfg.atomic_json(os.path.join(work_dir, "submission.json"), final)
    record = {"rerun": cfg.is_rerun(), "test_tasks": len(test),
              "test_inputs": sum(len(t["test"]) for t in test.values()),
              "model_attempts_used": used, "error": error,
              "decoded_files": len(os.listdir(outputs_dir)) if os.path.isdir(outputs_dir) else 0}
    cfg.atomic_json(os.path.join(work_dir, "v6-assembly.json"), record)
    cfg.log("assembled", **{k: v for k, v in record.items() if k != "error"}, error=bool(error))
    if error:
        print(error[-3000:])
    return record


if __name__ == "__main__":
    main()
