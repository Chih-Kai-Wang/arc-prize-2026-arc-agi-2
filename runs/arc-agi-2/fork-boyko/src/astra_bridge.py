"""Conservative ID-agnostic Astra expert promotion for ARC-AGI-2.

Loads only task train demonstrations and query inputs. Never reads evaluation solutions.
Baseline submission is deep copied and untouched; returns candidate overlay and diagnostics.
"""
from copy import deepcopy
import time
from candidate_solver import fit, predict_candidates, reconstruct, rectangles, Abstain
from arc2_d4_tools import validate_grid


def structural_colors(train_pairs, test_input):
    """Cheap necessary conditions, run before the expensive D4 search."""
    if len(train_pairs) < 3 or not test_input:
        return []
    common = set(v for row in train_pairs[0]['input'] for v in row)
    for p in train_pairs[1:]:
        common &= set(v for row in p['input'] for v in row)
    common &= set(v for row in test_input for v in row)
    colors = []
    for color in sorted(common):
        box = rectangles(test_input, color)
        if box is None:
            continue
        mH, mW = box[1]-box[0]+1, box[3]-box[2]+1
        if mH*mW >= len(test_input)*len(test_input[0]):
            continue
        good = True
        for pair in train_pairs:
            box2=rectangles(pair['input'],color)
            o=pair['output']
            if box2 is None or any(color in row for row in o):
                good = False;break
            h2,w2=box2[1]-box2[0]+1,box2[3]-box2[2]+1
            if (h2,w2) not in ((len(o),len(o[0])), (len(o[0]),len(o))):
                good=False;break
        if good:
            colors.append(color)
    return colors


def expert_guess(train_pairs,test_input):
    """Return (prediction, meta), or (None, meta) on any uncertainty."""
    t0=time.monotonic()
    colors=structural_colors(train_pairs,test_input)
    if not colors:
        return None,{'status':'abstain','reason':'shape/mask prefilter','sec':round(time.monotonic()-t0,3)}
    try:
        for i,p in enumerate(train_pairs):
            others=train_pairs[:i]+train_pairs[i+1:]
            outs=predict_candidates(others,p['input'])
            if p['output'] not in outs:
                return None,{'status':'abstain','reason':f'LOO fold {i} mismatch','sec':round(time.monotonic()-t0,3)}
        models=fit(train_pairs)
        if len(models)!=1:
            return None,{'status':'abstain','reason':'multiple fitted models','models':len(models),'sec':round(time.monotonic()-t0,3)}
        color,label=models[0]
        if color not in colors:
            return None,{'status':'abstain','reason':'mask prefilter contradicts fit','sec':round(time.monotonic()-t0,3)}
        restored, info=reconstruct(test_input,color)
        predicted=predict_candidates(train_pairs,test_input)
        if len(predicted)!=1:
            return None,{'status':'abstain','reason':'multiple outputs','sec':round(time.monotonic()-t0,3)}
        grid=predicted[0]
        validate_grid(grid)
        best=info['best_context_known_cells']
        next_best=info['runner_up_known_cells']
        mask_area=info['mask_area']
        # Conservative confidence gate before taking one of the two Kaggle attempts.
        if best < 50 or best < 3*mask_area or best < 4*max(1,next_best):
            return None,{'status':'abstain','reason':'context confidence gate','best':best,'runner_up':next_best,
                         'mask_area':mask_area,'sec':round(time.monotonic()-t0,3)}
        return grid,{'status':'confident','mask_area':mask_area,'best':best,'runner_up':next_best,
                     'mask_color':color,'orientation':label,'sec':round(time.monotonic()-t0,3)}
    except Exception as exc:
        return None,{'status':'abstain','reason':type(exc).__name__+': '+str(exc)[:160],
                     'sec':round(time.monotonic()-t0,3)}


def run_expert(queries, baseline_submission, promote=True, time_budget_seconds=300):
    """ID-blind; never changes attempt_1. Does not need or use ground-truth labels."""
    updated=deepcopy(baseline_submission)
    diagnostics={}
    total_start=time.monotonic()
    for task_id,task in queries.items():
        if time.monotonic()-total_start >= time_budget_seconds:
            diagnostics[task_id]={'status':'time_budget_exhausted'}
            continue
        try:
            train=task['train']
            for ti,q in enumerate(task['test']):
                if time.monotonic()-total_start >= time_budget_seconds:
                    diagnostics[f'{task_id}:{ti}']={'status':'time_budget_exhausted'}
                    continue
                grid,info=expert_guess(train,q['input'])
                info['test_index']=ti
                if grid is None:
                    if info.get('reason')!='shape/mask prefilter':
                        diagnostics[f'{task_id}:{ti}']=info
                    continue
                baseline=baseline_submission[task_id][ti]
                if grid in (baseline['attempt_1'],baseline['attempt_2']):
                    info['status']='already_in_qwen_attempts'
                elif promote:
                    updated[task_id][ti]['attempt_2']=grid
                    info['status']='promoted_to_attempt_2'
                    info['replaced_attempt_2_shape']=[len(baseline['attempt_2']),len(baseline['attempt_2'][0])]
                else:
                    info['status']='shadow_only'
                diagnostics[f'{task_id}:{ti}']=info
        except Exception as exc:
            diagnostics[task_id]={'status':'error','reason':str(exc)[:160]}
    return updated,diagnostics

