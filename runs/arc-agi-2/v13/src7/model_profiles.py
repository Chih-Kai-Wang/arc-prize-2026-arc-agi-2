"""Model profiles for the v13 LLM phase.

Default profile `qwen38_27b_fp8` holds the exact constants previously hard-coded in `v13_llm.py` and `build.py`.
Template profile `qwen4_27b_template` has all fields populated with `TODO_FILL` sentinels for the swap-in fields and
`launchable=False`; `v13_llm.py` refuses to run it (`raise Skip`) until every `TODO_FILL` has been replaced.
"""
import copy
import json

DEFAULT_PROFILE = "qwen38_27b_fp8"
TODO_FILL = "TODO_FILL"

_ALIASES = {
    "kaggle_model_source": "model_source",
    "model_slug": "model_source",
    "sha256_manifest": "sha_manifest",
    "manifest": "sha_manifest",
    "hash_manifest": "sha_manifest",
    "vllm_req": "vllm_version",
    "vllm_version_req": "vllm_version",
    "tp": "tensor_parallel_size",
    "tp_size": "tensor_parallel_size",
    "num_speculative_tokens": "mtp_tokens",
    "think_tag": "think_tags",
    "smoke": "smoke_body",
    "smoke_request": "smoke_body",
    "smoke_payload": "smoke_body",
}


class ProfileDict(dict):
    """Dict with key aliases and think_start_tag / think_end_tag views."""

    def _resolve(self, key):
        return _ALIASES.get(key, key)

    def __getitem__(self, key):
        if key in ("think_start_tag", "think_end_tag"):
            tags = dict.__getitem__(self, "think_tags")
            if isinstance(tags, (tuple, list)) and len(tags) == 2:
                return tags[0] if key == "think_start_tag" else tags[1]
            if isinstance(tags, dict):
                return tags.get("start" if key == "think_start_tag" else "end")
            return tags
        return dict.__getitem__(self, self._resolve(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, self._resolve(key), value)

    def __contains__(self, key):
        if key in ("think_start_tag", "think_end_tag"):
            return dict.__contains__(self, "think_tags")
        return dict.__contains__(self, self._resolve(key))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


PROFILES = {
    "qwen38_27b_fp8": ProfileDict({
        "name": "qwen38_27b_fp8",
        "launchable": True,
        "model_source": "mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1",
        "sha_manifest": "qwen38_fp8_sha256.json",
        "weight_glob": "layers-0.safetensors",
        "vllm_version": "0.19.0",
        "vllm_wheel_glob": "vllm-0.19.0-*.whl",
        "served_model_name": "m",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.93,
        "max_model_len": 131072,
        "min_num_seqs": 48,
        "kv_cache_dtype": "fp8",
        "attention_backend": "TRITON_ATTN",
        "reasoning_parser": "qwen3",
        "trust_remote_code": True,
        "quantization": None,
        "dtype": None,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "speculative_method": "mtp",
        "mtp_tokens": 2,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
        "server_attempts": ((2, False), (0, False), (0, True)),
        "chat_template_kwargs": {
            "reasoning_effort_key": "reasoning_effort",
            "enable_thinking_key": "enable_thinking",
        },
        "reasoning_effort_key": "reasoning_effort",
        "enable_thinking_key": "enable_thinking",
        "think_tags": ("<think>", "</think>"),
        "reasoning_fields": ("reasoning_content", "reasoning"),
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "smoke_body": {
            "model": "m",
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 32,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }),
    "qwen4_27b_template": ProfileDict({
        "name": "qwen4_27b_template",
        "launchable": False,
        "model_source": TODO_FILL,
        "sha_manifest": TODO_FILL,
        "weight_glob": "layers-0.safetensors",
        "vllm_version": TODO_FILL,
        "vllm_wheel_glob": "vllm-0.19.0-*.whl",
        "served_model_name": "m",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.93,
        "max_model_len": 131072,
        "min_num_seqs": 48,
        "kv_cache_dtype": "fp8",
        "attention_backend": "TRITON_ATTN",
        "reasoning_parser": TODO_FILL,
        "trust_remote_code": True,
        "quantization": TODO_FILL,
        "dtype": None,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "speculative_method": "mtp",
        "mtp_tokens": 2,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
        "server_attempts": ((2, False), (0, False), (0, True)),
        "chat_template_kwargs": {
            "reasoning_effort_key": "reasoning_effort",
            "enable_thinking_key": "enable_thinking",
        },
        "reasoning_effort_key": "reasoning_effort",
        "enable_thinking_key": "enable_thinking",
        "think_tags": TODO_FILL,
        "reasoning_fields": ("reasoning_content", "reasoning"),
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "smoke_body": {
            "model": "m",
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 32,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }),
}


def _has_todo(val):
    if isinstance(val, str):
        return TODO_FILL in val
    if isinstance(val, dict):
        return any(_has_todo(k) or _has_todo(v) for k, v in val.items())
    if isinstance(val, (list, tuple, set)):
        return any(_has_todo(x) for x in val)
    return False


def todo_fields(profile):
    return [k for k, v in profile.items() if _has_todo(v)]


def is_launchable(profile):
    todos = todo_fields(profile)
    if todos:
        return False
    if profile.get("name") == "qwen4_27b_template":
        return True
    return bool(profile.get("launchable", True))


def vllm_cmd(profile, vpy, model, port, max_len, conc, mtp, eager):
    cmd = [
        vpy, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model),
        "--served-model-name", str(profile["served_model_name"]),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tensor-parallel-size", str(profile["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(profile["gpu_memory_utilization"]),
        "--max-model-len", str(max_len),
        "--max-num-seqs", str(max(profile["min_num_seqs"], conc)),
    ]
    if profile.get("kv_cache_dtype"):
        cmd += ["--kv-cache-dtype", str(profile["kv_cache_dtype"])]
    if profile.get("attention_backend"):
        cmd += ["--attention-backend", str(profile["attention_backend"])]
    if profile.get("reasoning_parser"):
        cmd += ["--reasoning-parser", str(profile["reasoning_parser"])]
    if profile.get("quantization"):
        cmd += ["--quantization", str(profile["quantization"])]
    if profile.get("dtype"):
        cmd += ["--dtype", str(profile["dtype"])]
    if profile.get("trust_remote_code"):
        cmd += ["--trust-remote-code"]
    if profile.get("limit_mm_per_prompt") is not None:
        cmd += ["--limit-mm-per-prompt", json.dumps(profile["limit_mm_per_prompt"])]
    if mtp:
        spec = dict(profile["speculative_config"], num_speculative_tokens=mtp)
        cmd += ["--speculative-config", json.dumps(spec)]
    if eager:
        cmd += ["--enforce-eager"]
    return cmd


def build_chat_template_kwargs(profile, effort=None, enable_thinking=True):
    ctk = profile.get("chat_template_kwargs") or {}
    effort_key = ctk.get("reasoning_effort_key") or profile.get("reasoning_effort_key")
    think_key = ctk.get("enable_thinking_key") or profile.get("enable_thinking_key")
    kw = {}
    if effort is not None and effort_key:
        kw[effort_key] = effort
    if think_key:
        kw[think_key] = enable_thinking
    return kw


def smoke_payload(profile):
    return copy.deepcopy(profile["smoke_body"])


def tokenize_payload(profile, messages, effort):
    return {
        "model": profile["served_model_name"],
        "messages": messages,
        "chat_template_kwargs": build_chat_template_kwargs(profile, effort=effort, enable_thinking=True),
    }


def chat_payload(profile, messages, cap, seed, effort):
    samp = profile["sampling"]
    return {
        "model": profile["served_model_name"],
        "messages": messages,
        "max_tokens": cap,
        "temperature": samp["temperature"],
        "top_p": samp["top_p"],
        "top_k": samp["top_k"],
        "min_p": samp["min_p"],
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "chat_template_kwargs": build_chat_template_kwargs(profile, effort=effort, enable_thinking=True),
    }


def extract_delta_reasoning(delta, profile):
    for field in profile.get("reasoning_fields", ("reasoning_content", "reasoning")):
        val = delta.get(field)
        if val:
            return val
    return None


def split_thinking(content, reasoning, profile=None):
    """If the server did not split reasoning from content, split on the profile's think tags."""
    if reasoning or not content:
        return content, reasoning
    prof = profile or PROFILES[DEFAULT_PROFILE]
    start_tag = prof.get("think_start_tag")
    end_tag = prof.get("think_end_tag")
    if (isinstance(start_tag, str) and isinstance(end_tag, str)
            and start_tag and end_tag and not _has_todo(start_tag) and not _has_todo(end_tag)
            and start_tag in content):
        _, rest = content.split(start_tag, 1)
        if end_tag in rest:
            think_part, after = rest.split(end_tag, 1)
            return after.lstrip("\r\n"), think_part
        return "", rest
    return content, reasoning
