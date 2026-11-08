import os

# # For Debugging
# # Disable torch.compile autocompilation to isolate pure functions
# os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
# # Enable overwriting the compiled cache if it was created (without manually editing the cache)
# os.environ["UNSLOTH_COMPILE_OVERWRITE"] = "1"
# # # Enable compiler logging to see where exactly the code crashes
# os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

# os.environ["UNSLOTH_MOE_BACKEND"] = "unsloth_triton"  # Forced activation of the Triton MoE backend
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["WANDB_DISABLED"] = "true"
os.environ['WANDB_MODE'] = 'disabled'

from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter
import gc
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict
from typing import Any, Union
from transformers import DataCollatorForLanguageModeling
from contextlib import redirect_stdout, redirect_stderr
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import bz2
import pickle
import logging; logging.disable(logging.WARNING)

is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
is_rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', "") != "" 
is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Interactive'  # Interactive Mode
is_batch = os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Batch'
is_local = not is_kaggle
is_background = is_rerun or is_batch
is_dialog = not is_background

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15

class UnslothFixedTrainer(UnslothTrainer):
    # Issue https://github.com/unslothai/unsloth/issues/2435
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss
    #pass

class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch

def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)
    suffixes = defaultdict(list)
    candidates = dict()
    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))
    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]
    while time.time() - start_time < 540 and time.time() < end_time:
        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0
        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1
        if num_alive_beams == 0:
            break
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )
        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))
    return suffixes

@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result

@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length-1:query_length-1+len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result

def worker(rank, queue, end_time):
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # import torch
    # torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    # torch.backends.cuda.enable_math_sdp(False)

    # #import torch.cuda.amp as amp
    # #amp.GradScaler = lambda **kwargs: torch.cuda.amp.GradScaler(enabled=False, **kwargs)
    # temporary_location=f"/kaggle/tmp-{rank}"
    # os.makedirs(temporary_location, exist_ok=True)

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    train_n = int(os.getenv("ARC_N_TRAIN_AUG", "8"))
    eval_n = int(os.getenv("ARC_N_EVAL_AUG", "1"))
    lora_seed = int(os.getenv("ARC_LORA_SEED", "42"))
    train_seed = int(os.getenv("ARC_TRAIN_AUG_SEED", "1"))
    eval_seed = int(os.getenv("ARC_EVAL_AUG_SEED", "2"))
    task_cap = float(os.getenv("ARC_TASK_CAP", "950"))
    dir_outputs = os.getenv("ARC_OUT_DIR", "/kaggle/inference_outputs")
    print(f"[Rank {rank}] Boyko model; TTT={train_n}x8; DFS={eval_n}x8; lora_seed={lora_seed}; output={dir_outputs}")
    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,  # Important: Forces memory optimization
        random_state=lora_seed,
        use_rslora=True,
        loftq_config=None,
        #ensure_weight_tying=True,
        #modules_to_save=["embed_tokens", "lm_head"],
        #temporary_location=temporary_location,
    )
    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",  # "adamw_torch_fused",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=lora_seed,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",  # "steps" / "no",
        #logging_steps=1,
        fp16=False,
        bf16=True,
        # Disable FSDP (use standard DDP)
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )
    max_seq_length = 8192
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/kaggle/input/models/konstantinboyko/qwen3-4b-bfloat16-v02-01-01/transformers/default/1" if is_kaggle \
            else "/kaggle/input/qwen3_4b_grids15_sft139-transformers-bfloat16-v1",
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
        #dtype=torch.bfloat16,
        #attn_implementation="sdpa",
    )
    model = FastLanguageModel.get_peft_model(model, **peft_params)
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}
    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )
    formatter = QwenFormatter(tokenizer=tokenizer)
    max_new_tokens = formatter.max_new_tokens()
    max_score = -np.log(0.2)
    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    arc_test_set = ArcDataset.from_file(test_path)
    # dir_outputs is configured per pass above.
    os.makedirs(dir_outputs, exist_ok=True)
    while True:
        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break
        key = queue.get()
        if key is None:
            break
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        load_result = set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )
        model = FastLanguageModel.for_training(model)
        puzzle_ds = arc_test_set.change_keys([key])
        train_ds = puzzle_ds.augment(n=train_n, shfl_keys=True, seed=train_seed)
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            trainer = UnslothFixedTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**train_args),
            )
            stats = trainer.train()
            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
            del trainer
        model = FastLanguageModel.for_inference(model)
        gc.collect()
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training")
        torch.cuda.reset_peak_memory_stats()
        print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")
        puzzle_ds_multi = puzzle_ds.split_multi_replies()
        eval_ds = puzzle_ds_multi.augment(n=eval_n, seed=eval_seed)
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)
        # Geometry order in ArcDataset.augment: 0=id,1=rot90,2=rot180,3=rot270,
        # 4=transpose,5=transpose+rot90,6=transpose+rot180,7=transpose+rot270.
        # Pair geometries of identical HxW/WxH shapes for model-batched DFS.
        # This reduces exactly to Boyko's 4-per-batch grouping when eval_n=2.
        batches = []
        for side in (0, 4):
            for test_id, subkeys in test_id_to_subkeys.items():
                expected = 8 * eval_n
                if len(subkeys) != expected:
                    raise RuntimeError(f"Unexpected number of DFS views for {test_id}: {len(subkeys)} != {expected}")
                for g1, g2 in ((side, side + 2), (side + 1, side + 3)):
                    batch = subkeys[g1 * eval_n:(g1 + 1) * eval_n] + subkeys[g2 * eval_n:(g2 + 1) * eval_n]
                    if batch:
                        batches.append(batch)
        with torch.inference_mode():
            known_scores = {}
            for subkeys in batches:
                spend_time = time.time() - start_time
                if spend_time > task_cap or time.time() > end_time:
                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                    break
                print(f"[Rank {rank}] decoding {subkeys}")
                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))
                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)
                for subkey_id, scored_beams in dfs_result:
                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []
                    for beam_score, tokens in scored_beams:
                        array = formatter.convert_tokens_to_array(tokens)
                        if array is None:
                            continue
                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                        grid_id = (bk, tuple(map(tuple, solution)))
                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                            aug_dataset = ArcDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)
                            aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(formatter):
                                aug_queries.append(augmented_sample["input"])
                                aug_answers.append(augmented_sample["reply"])
                            augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                            augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                            augmented_scores = augmented_scores1 + augmented_scores2
                            known_scores[grid_id] = augmented_scores
                        decoded_result.append({
                            "beam_score": beam_score,
                            "score_aug": augmented_scores,
                            "solution": solution,
                        })
                    if len(decoded_result):
                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                            pickle.dump(decoded_result, f)
        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")