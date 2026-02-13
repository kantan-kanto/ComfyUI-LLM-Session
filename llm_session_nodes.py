# ComfyUI-LLM-Session
# Copyright (C) 2026 kantan-kanto (https://github.com/kantan-kanto)
#
# Local LLM session nodes for ComfyUI (GGUF / llama.cpp via llama-cpp-python).
# - LLM Session Chat: persistent multi-turn chat with file-based history and optional summarization
# - LLM Dialogue Cycle: two-model turn-based dialogue runner without graph cycles
import os
import io
import sys
import contextlib
import base64
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
import folder_paths
import hashlib


# ============================================================================
# Simple Defaults (config-driven)
# ============================================================================
# LLM Session Chat (Simple) loads optional defaults from:
#   <this_repo>/config/simple_defaults.json
#
# - If the file is missing or invalid, built-in safe defaults are used.
# - Unknown keys are ignored.
#
# This keeps the Simple node easy to use while still allowing customization.

_SIMPLE_DEFAULTS_BUILTIN: Dict[str, Any] = {
    "schema_version": 1,
    "system_prompt": "You are a helpful assistant.",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "n_gpu_layers": 0,
    "n_ctx": 4096,
    "max_turns": 6,
    "summarize_old_history": True,
    "summary_chunk_turns": 6,
    "max_tokens_summary": 128,
    "summary_max_chars": 1500,
    "dynamic_max_tokens": True,
    "min_generation_tokens": 96,
    "safety_margin_tokens": 64,
    "prompt_cache_mode": "disk",
    "kv_state_mode": "memory",
    "log_level": "timing",
    "suppress_backend_logs": True,
    "repeat_penalty": 1.12,
    "repeat_last_n": 256,
    "rewrite_continue": True,
    "reset_session": False,
    "stream_to_console": False,
}

_SIMPLE_ALLOWED_KEYS = set(_SIMPLE_DEFAULTS_BUILTIN.keys()) - {"schema_version"}

def _simple_config_path() -> str:
    try:
        base = Path(__file__).parent
        return str(base / "config" / "simple_defaults.json")
    except Exception:
        return ""

def _load_simple_defaults(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load Simple defaults from JSON, falling back to built-in safe defaults."""
    cfg_path = (config_path or "").strip() or _simple_config_path()
    defaults = dict(_SIMPLE_DEFAULTS_BUILTIN)

    if not cfg_path:
        return defaults
    if not os.path.exists(cfg_path):
        return defaults

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return defaults
        if int(raw.get("schema_version", 1)) != 1:
            # Future-proof: unknown schema -> ignore
            return defaults

        for k, v in raw.items():
            if k not in _SIMPLE_ALLOWED_KEYS:
                continue
            defaults[k] = v
    except Exception:
        return defaults

    # Best-effort type coercion + clamping (never raise)
    def _as_int(x, d):
        try:
            return int(x)
        except Exception:
            return d

    def _as_float(x, d):
        try:
            return float(x)
        except Exception:
            return d

    def _as_bool(x, d):
        try:
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return bool(x)
            s = str(x).strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
        except Exception:
            pass
        return d

    # Numeric clamps (match UI ranges loosely)
    defaults["max_tokens"] = max(1, _as_int(defaults.get("max_tokens"),  512))
    defaults["n_ctx"] = max(512, _as_int(defaults.get("n_ctx"), 4096))
    defaults["n_gpu_layers"] = _as_int(defaults.get("n_gpu_layers"), 0)
    defaults["max_turns"] = max(0, _as_int(defaults.get("max_turns"), 6))
    defaults["summary_chunk_turns"] = max(1, _as_int(defaults.get("summary_chunk_turns"), 6))
    defaults["max_tokens_summary"] = max(16, _as_int(defaults.get("max_tokens_summary"), 128))
    defaults["summary_max_chars"] = max(200, _as_int(defaults.get("summary_max_chars"), 1500))
    defaults["min_generation_tokens"] = max(1, _as_int(defaults.get("min_generation_tokens"), 96))
    defaults["safety_margin_tokens"] = max(0, _as_int(defaults.get("safety_margin_tokens"), 64))
    defaults["repeat_last_n"] = max(0, _as_int(defaults.get("repeat_last_n"), 256))

    defaults["temperature"] = min(2.0, max(0.0, _as_float(defaults.get("temperature"), 0.7)))
    defaults["top_p"] = min(1.0, max(0.05, _as_float(defaults.get("top_p"), 0.9)))
    defaults["repeat_penalty"] = min(2.0, max(1.0, _as_float(defaults.get("repeat_penalty"), 1.12)))

    # Enums / strings
    defaults["prompt_cache_mode"] = str(defaults.get("prompt_cache_mode", "disk")).lower()
    if defaults["prompt_cache_mode"] not in ("disk", "memory", "off"):
        defaults["prompt_cache_mode"] = _SIMPLE_DEFAULTS_BUILTIN["prompt_cache_mode"]

    defaults["kv_state_mode"] = str(defaults.get("kv_state_mode", "memory")).lower()
    if defaults["kv_state_mode"] not in ("memory", "off"):
        defaults["kv_state_mode"] = _SIMPLE_DEFAULTS_BUILTIN["kv_state_mode"]

    defaults["log_level"] = str(defaults.get("log_level", "timing")).lower()
    if defaults["log_level"] not in ("minimal", "timing", "debug"):
        defaults["log_level"] = _SIMPLE_DEFAULTS_BUILTIN["log_level"]

    # Booleans
    defaults["summarize_old_history"] = _as_bool(defaults.get("summarize_old_history"), True)
    defaults["dynamic_max_tokens"] = _as_bool(defaults.get("dynamic_max_tokens"), True)
    defaults["suppress_backend_logs"] = _as_bool(defaults.get("suppress_backend_logs"), True)
    defaults["rewrite_continue"] = _as_bool(defaults.get("rewrite_continue"), True)
    defaults["reset_session"] = _as_bool(defaults.get("reset_session"), False)
    defaults["stream_to_console"] = _as_bool(defaults.get("stream_to_console"), False)

    # System prompt
    sp = defaults.get("system_prompt")
    defaults["system_prompt"] = str(sp) if sp is not None else _SIMPLE_DEFAULTS_BUILTIN["system_prompt"]

    return defaults

# llama-cpp-python imports
try:
    # Qwen2.5-VL and Qwen3-VL vision support via Qwen2VLChatHandler / Qwen3VLChatHandler
    from llama_cpp import Llama

    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        QWEN2_AVAILABLE = True
    except ImportError:
        QWEN2_AVAILABLE = False   

    try:
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        QWEN3_AVAILABLE = True
    except ImportError:
        QWEN3_AVAILABLE = False
    
    # LLaVA vision support (llava-v1.5/1.6 etc.) via Llava* chat handler
    # (class names vary across forks/versions; keep it best-effort)
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        LLAVA_AVAILABLE = True
    except ImportError:
        LLAVA_AVAILABLE = False
        Llava15ChatHandler = None  # type: ignore 

    # Llama vision support (llama-3.2 etc.) via Llama* chat handler
    # (class names vary across forks/versions; keep it best-effort)
    try:
        from llama_cpp.llama_chat_format import Llama3VisionAlphaChatHandler
        LLAMA_AVAILABLE = True
    except ImportError:
        LLAMA_AVAILABLE = False
        Llama3VisionAlphaChatHandler = None  # type: ignore 

    # MiniCPM-V 2.6 vision support via MiniCPMv26ChatHandler
    try:
        from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
        MINICPM26_AVAILABLE = True
    except ImportError:
        MINICPM26_AVAILABLE = False
        MiniCPMv26ChatHandler = None  # type: ignore

    # Gemma 3 vision support via Gemma3ChatHandler (in upstream llama-cpp-python it subclasses Llava15ChatHandler)
    # Ref: upstream PR shows Gemma3ChatHandler(Llava15ChatHandler). :contentReference[oaicite:0]{index=0}
    try:
        from llama_cpp.llama_chat_format import Gemma3ChatHandler
        GEMMA3_AVAILABLE = True
    except ImportError:
        GEMMA3_AVAILABLE = False
        Gemma3ChatHandler = None  # type: ignore

    # GLM-4.6V vision support via GLM46VChatHandler
    try:
        from llama_cpp.llama_chat_format import GLM46VChatHandler
        GLM46V_AVAILABLE = True
    except ImportError:
        GLM46V_AVAILABLE = False
        GLM46VChatHandler = None  # type: ignore

    LLAMA_CPP_AVAILABLE = True
except Exception as e:
    LLAMA_CPP_AVAILABLE = False
    _LLAMA_CPP_IMPORT_ERROR = repr(e)
    QWEN2_AVAILABLE = False
    QWEN3_AVAILABLE = False
    LLAVA_AVAILABLE = False
    LLAMA_AVAILABLE = False
    MINICPM26_AVAILABLE = False
    GEMMA3_AVAILABLE = False
    GLM46V_AVAILABLE = False
    print("[LLM Session] Warning: llama-cpp-python not available")

# ============================================================================
# Utility Functions
# ============================================================================

def _list_gguf_recursive(models_dir: str) -> tuple[list[str], list[str]]:
    """
    models_dir 配下を再帰的に探索し、.gguf を相対パスで返す。
    - model: mmproj 以外
    - mmproj: ファイル名が mmproj で始まるもの
    """
    base = Path(models_dir)
    if not base.exists():
        return [], []

    models: list[str] = []
    mmprojs: list[str] = []

    for p in base.rglob("*.gguf"):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()  # サブフォルダを含む相対パス
        if p.name.startswith("mmproj"):
            mmprojs.append(rel)
        else:
            models.append(rel)

    models.sort(key=str.lower)
    mmprojs.sort(key=str.lower)
    return models, mmprojs

def _get_llm_model_roots() -> list[str]:
    """
    Collect LLM model roots from:
    - default ComfyUI models/LLM
    - extra_model_paths.yaml entries registered as folder key "LLM" / "llm"
    """
    roots: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        if not path:
            return
        norm = os.path.normpath(os.path.abspath(path))
        key = norm.lower()
        if key in seen:
            return
        if not os.path.isdir(norm):
            return
        seen.add(key)
        roots.append(norm)

    # Primary ComfyUI models/LLM
    _add(os.path.join(folder_paths.models_dir, "LLM"))

    # Extra paths loaded via extra_model_paths.yaml
    if hasattr(folder_paths, "get_folder_paths"):
        for folder_key in ("LLM", "llm"):
            try:
                for p in folder_paths.get_folder_paths(folder_key):
                    _add(p)
            except Exception:
                pass

    return roots

def _list_gguf_recursive_multi(roots: list[str]) -> tuple[list[str], list[str]]:
    """
    Search multiple roots and return merged relative paths.
    Relative paths are deduplicated (first root wins if names collide).
    """
    models: list[str] = []
    mmprojs: list[str] = []
    seen_models: set[str] = set()
    seen_mmprojs: set[str] = set()

    for root in roots or []:
        m_list, mp_list = _list_gguf_recursive(root)
        for rel in m_list:
            k = rel.lower()
            if k in seen_models:
                continue
            seen_models.add(k)
            models.append(rel)
        for rel in mp_list:
            k = rel.lower()
            if k in seen_mmprojs:
                continue
            seen_mmprojs.add(k)
            mmprojs.append(rel)

    models.sort(key=str.lower)
    mmprojs.sort(key=str.lower)
    return models, mmprojs

def _safe_join_under(base_dir: str, rel_path: str) -> str:
    base = Path(base_dir).resolve()
    p = (base / rel_path).resolve()
    # パストラバーサル対策：base 配下に解決されることを保証
    if base != p and base not in p.parents:
        raise ValueError(f"Invalid path (outside models_dir): {rel_path}")
    return str(p)

def _resolve_llm_relpath(rel_path: str, roots: Optional[list[str]] = None) -> str:
    """
    Resolve a user-selected relative model path under known LLM roots.
    Returns the first existing match; falls back to the first root for validation.
    """
    roots = roots or _get_llm_model_roots()
    if not roots:
        default_root = os.path.join(folder_paths.models_dir, "LLM")
        return _safe_join_under(default_root, rel_path)

    for root in roots:
        try:
            candidate = _safe_join_under(root, rel_path)
        except Exception:
            continue
        if os.path.exists(candidate):
            return candidate

    return _safe_join_under(roots[0], rel_path)

def encode_image_base64(pil_image: Image.Image, max_pixels: int = 262144) -> str:
    """
    Convert PIL image to base64 encoded string
    
    Args:
        pil_image: PIL Image object
        max_pixels: Maximum pixels for token saving
    
    Returns:
        Base64 encoded image string
    """
    # Resize image
    width, height = pil_image.size
    total_pixels = width * height
    
    if total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # JPEG compress and base64 encode
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", optimize=True, quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def tensor2pil(image_tensor) -> List[Image.Image]:
    """
    Convert ComfyUI tensor to PIL Image list
    
    Args:
        image_tensor: ComfyUI image tensor (B, H, W, C)
    
    Returns:
        List of PIL Image objects
    """
    batch_count = image_tensor.size(0) if len(image_tensor.shape) > 3 else 1
    
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image_tensor[i]))
        return out
    
    # Convert to numpy array (0-255)
    np_image = np.clip(255.0 * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return [Image.fromarray(np_image)]

def detect_language(text: str) -> str:
    """
    Detect language from text (simple)
    
    Args:
        text: Input text
    
    Returns:
        'zh', 'ja', or 'en'
    """
    # Check for CJK character ranges
    has_chinese = False
    has_japanese = False
    
    for char in text:
        # Japanese-specific characters (Hiragana/Katakana)
        if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':
            has_japanese = True
        # Chinese characters (Hanzi/Kanji - shared but check for Japanese context)
        elif '\u4e00' <= char <= '\u9fff':
            has_chinese = True
    
    # Japanese has priority if Hiragana/Katakana detected
    if has_japanese:
        return 'ja'
    if has_chinese:
        return 'zh'
    
    return 'en'

def _detect_history_language(history: Dict[str, Any]) -> str:
    """
    Detect primary language from conversation history
    
    Args:
        history: Conversation history dictionary
    
    Returns:
        'zh', 'ja', or 'en'
    """
    turns = history.get("turns", [])
    if not turns:
        return "en"
    
    # Check last few turns (up to 3)
    recent_text = ""
    for turn in turns[-3:]:
        recent_text += turn.get("user", {}).get("text", "") + " "
        recent_text += turn.get("assistant", {}).get("text", "") + " "
    
    if not recent_text.strip():
        return "en"
    
    return detect_language(recent_text)

# ============================================================================
# Model Manager
# ============================================================================


# =============================================================================
# Chat session history (file-based, ComfyUI hidden)
# =============================================================================

def _now_iso_jst() -> str:
    """Return current time in ISO8601 with JST offset."""
    # Avoid depending on system tz; format explicitly as +09:00
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).isoformat(timespec="seconds")

def _make_suppress_backend_logs(suppress: bool):
    """Return a context manager that suppresses stdout/stderr if suppress is True."""
    if not suppress:
        return contextlib.nullcontext()
    class _Suppress:
        def __enter__(self):
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        def __exit__(self, exc_type, exc, tb):
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            return False
    return _Suppress()

def _safe_output_dir() -> str:
    """Best-effort output dir resolution across ComfyUI versions."""
    if hasattr(folder_paths, "get_output_directory"):
        try:
            return folder_paths.get_output_directory()
        except Exception:
            pass
    for attr in ("output_directory", "output_dir"):
        if hasattr(folder_paths, attr):
            d = getattr(folder_paths, attr)
            if isinstance(d, str) and d:
                return d
    # Fallback: cwd/output
    return os.path.join(os.getcwd(), "output")

def default_sessions_dir() -> str:
    return os.path.join(_safe_output_dir(), "llm_session_sessions")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _history_path(session_id: str, history_dir: Optional[str] = None) -> str:
    session_id = (session_id or "default").strip()
    safe = "".join(c for c in session_id if c.isalnum() or c in ("-", "_", "."))
    if not safe:
        safe = "default"
    base = (history_dir or "").strip() or default_sessions_dir()
    _ensure_dir(base)
    return os.path.join(base, f"{safe}.json")

def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write JSON atomically-ish with .tmp + .bak."""
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    bak = path + ".bak"
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        try:
            os.replace(path, bak)
        except Exception:
            # If replace fails, keep going; tmp->path is the important part
            pass
    os.replace(tmp, path)

def _new_history(session_id: str, system_prompt: str, model_sig: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "meta": {
            "session_id": session_id,
            "created_at": _now_iso_jst(),
            "updated_at": _now_iso_jst(),
            "model_signature": model_sig or {}
        },
        "system_prompt": system_prompt or "",
        "summary": {
            "enabled": False,
            "text": "",
            "updated_at": ""
        },
        "turns": []
    }

def load_history(session_id: str, history_dir: Optional[str], system_prompt: str,
                 model_sig: Optional[Dict[str, Any]] = None,
                 reset_session: bool = False) -> tuple[Dict[str, Any], str]:
    """Load session history JSON, or create new. Returns (history, path)."""
    path = _history_path(session_id, history_dir)
    if reset_session:
        hist = _new_history(session_id, system_prompt, model_sig=model_sig)
        _atomic_write_json(path, hist)
        return hist, path

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if not isinstance(hist, dict) or hist.get("schema_version") != 1:
                raise ValueError("Unsupported history schema")
            # Keep last known system prompt as fallback, but prefer current input when provided
            if system_prompt:
                hist["system_prompt"] = system_prompt
            # Update model signature reference info (best effort)
            if model_sig:
                hist.setdefault("meta", {}).setdefault("model_signature", {}).update(model_sig)
            return hist, path
        except Exception as e:
            # Try backup
            bak = path + ".bak"
            if os.path.exists(bak):
                try:
                    with open(bak, "r", encoding="utf-8") as f:
                        hist = json.load(f)
                    if isinstance(hist, dict) and hist.get("schema_version") == 1:
                        if system_prompt:
                            hist["system_prompt"] = system_prompt
                        if model_sig:
                            hist.setdefault("meta", {}).setdefault("model_signature", {}).update(model_sig)
                        return hist, path
                except Exception:
                    pass
            # If both fail, start fresh
            hist = _new_history(session_id, system_prompt, model_sig=model_sig)
            _atomic_write_json(path, hist)
            return hist, path

    hist = _new_history(session_id, system_prompt, model_sig=model_sig)
    _atomic_write_json(path, hist)
    return hist, path

def _coerce_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

def build_chat_messages(history: Dict[str, Any],
                        user_text: str,
                        image_tensor=None,
                        max_turns: Optional[int] = None,
                        summarize_old_history: bool = True,
                        system_prompt: str = "") -> List[Dict[str, Any]]:
    """Build chat-completion messages. Image is included only for this turn."""
    sys = (system_prompt or "").strip() or (history.get("system_prompt") or "").strip()
    summary = ""
    if summarize_old_history:
        summary = (history.get("summary") or {}).get("text") or ""

    turns = history.get("turns") or []
    if max_turns is not None:
        mt = max(0, _coerce_int(max_turns, 0))
        if mt > 0:
            turns = turns[-mt:]
        else:
            turns = []

    messages: List[Dict[str, Any]] = []
    if sys:
        messages.append({"role": "system", "content": sys})
    if summary.strip():
        # Keep summary short and isolated; system is fine
        messages.append({"role": "system", "content": f"Conversation summary:\n{summary.strip()}"})

    for t in turns:
        u = ((t.get("user") or {}).get("text")) or ""
        a = ((t.get("assistant") or {}).get("text")) or ""
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})

    # Current turn
    if image_tensor is not None:
        try:
            pil_list = tensor2pil(image_tensor)
            if pil_list:
                # Use first image in batch for this turn (keeps it simple in phase 1)
                img_b64 = encode_image_base64(pil_list[0])
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": user_text or ""}
                    ]
                })
                return messages
        except Exception:
            # fall through to text-only
            pass

    messages.append({"role": "user", "content": user_text or ""})
    return messages


# ============================================================================
# Chat completion compatibility helpers
# ============================================================================
# Some GGUF models / chat templates do not support the "system" role.
# For those models, we fall back by folding system messages into the first user turn.
# This keeps the node compatible with a broader set of instruction-tuned GGUF models
# without requiring model-specific templates.

def _fold_system_into_user_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a new messages list with all system-role contents folded into the first user message."""
    sys_parts: List[str] = []
    out: List[Dict[str, Any]] = []

    for msg in (messages or []):
        if (msg or {}).get("role") == "system":
            c = (msg or {}).get("content")
            if isinstance(c, str) and c.strip():
                sys_parts.append(c.strip())
            continue
        out.append(msg)

    sys_text = "\n\n".join(sys_parts).strip()
    if not sys_text:
        return out

    # Find first user giving it the system prefix
    for i, msg in enumerate(out):
        if (msg or {}).get("role") != "user":
            continue
        content = (msg or {}).get("content")

        # Text-only user content
        if isinstance(content, str):
            out[i]["content"] = sys_text + "\n\n" + content
            return out

        # Multi-part content (e.g., image_url + text)
        if isinstance(content, list):
            # Try to prepend to the first text part; if none, insert one.
            new_list = []
            injected = False
            for part in content:
                if (not injected) and isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text") or ""
                    part = dict(part)
                    part["text"] = sys_text + "\n\n" + t
                    injected = True
                new_list.append(part)
            if not injected:
                new_list.append({"type": "text", "text": sys_text})
            out[i]["content"] = new_list
            return out

        # Unknown content type; replace with string
        out[i]["content"] = sys_text
        return out

    # No user message exists; prepend one
    return [{"role": "user", "content": sys_text}] + out

def _create_chat_completion_robust(llm: "Llama", messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """create_chat_completion with a fallback for models that do not support system role."""
    try:
        return llm.create_chat_completion(messages=messages, **kwargs)
    except Exception as e:
        s = str(e)
        if "System role not supported" not in s and "system role not supported" not in s:
            raise
        # Retry with system folded into user
        folded = _fold_system_into_user_messages(messages)
        return llm.create_chat_completion(messages=folded, **kwargs)

def _iter_chat_completion_robust(llm: "Llama", messages: List[Dict[str, Any]], **kwargs):
    """Streaming create_chat_completion with system-role fallback."""
    try:
        return llm.create_chat_completion(messages=messages, stream=True, **kwargs)
    except Exception as e:
        s = str(e)
        if "System role not supported" not in s and "system role not supported" not in s:
            raise
        folded = _fold_system_into_user_messages(messages)
        return llm.create_chat_completion(messages=folded, stream=True, **kwargs)

def _extract_stream_content(chunk: Any) -> str:
    """Best-effort extraction of streamed text from llama-cpp-python chunks."""
    try:
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta")
        if isinstance(delta, dict):
            return delta.get("content") or ""
        msg = choice.get("message")
        if isinstance(msg, dict):
            return msg.get("content") or ""
        txt = choice.get("text")
        return txt or ""
    except Exception:
        return ""

def _make_summary_prompt(existing_summary: str, turns_chunk: list) -> list:
    """
    Build messages for summarizing a chunk of turns.
    We keep the summary factual and compact to control n_ctx and speed.
    """
    chunk_lines = []
    for t in turns_chunk:
        u = (t.get("user") or {}).get("text", "")
        a = (t.get("assistant") or {}).get("text", "")
        if u:
            chunk_lines.append(f"USER: {u}")
        if a:
            chunk_lines.append(f"ASSISTANT: {a}")
    chunk_text = "\n".join(chunk_lines).strip()

    user_payload = []
    if existing_summary.strip():
        user_payload.append("Existing summary:\n" + existing_summary.strip())
    if chunk_text:
        user_payload.append("New conversation chunk:\n" + chunk_text)
    user_text = "\n\n".join(user_payload).strip() or "(empty)"

    return [
        {
            "role": "system",
            "content": (
                "Summarize the conversation into a compact rolling memory. "
                "Keep ONLY: key facts, decisions, constraints, user preferences, and open TODOs. "
                "Do NOT add new information. Be concise. Use short sentences."
            )
        },
        {"role": "user", "content": user_text}
    ]

def _summarize_with_model(model: "Llama", existing_summary: str, turns_chunk: list,
                         temperature: float, max_tokens: int, suppress_logs: bool = False) -> str:
    msgs = _make_summary_prompt(existing_summary, turns_chunk)
    with _make_suppress_backend_logs(suppress_logs):
        resp = _create_chat_completion_robust(model, msgs, temperature=float(temperature), max_tokens=int(max_tokens))
    return (resp["choices"][0]["message"]["content"] or "").strip()

def maybe_compact_summary(model: "Llama",
                          history: Dict[str, Any],
                          summary_max_chars: int = 1500,
                          temperature: float = 0.2,
                          max_tokens_summary: int = 128,
                          suppress_logs: bool = False) -> Dict[str, Any]:
    """
    If summary grows too large, re-summarize it to keep it compact.
    This is a safety valve to keep prompts small and fast.
    """
    sm = history.get("summary") or {}
    text = (sm.get("text") or "")
    limit = max(200, _coerce_int(summary_max_chars, 1500))
    if len(text) <= limit:
        return history

    msgs = [
        {
            "role": "system",
            "content": (
                "Compress the given summary while preserving key facts, constraints, decisions, "
                "user preferences, and open TODOs. Remove redundancy. Keep it short."
            )
        },
        {"role": "user", "content": text}
    ]
    with _make_suppress_backend_logs(suppress_logs):
        resp = _create_chat_completion_robust(model, msgs, temperature=float(temperature), max_tokens=int(max_tokens_summary))
    compact = (resp["choices"][0]["message"]["content"] or "").strip()
    history.setdefault("summary", {})
    history["summary"]["enabled"] = True
    history["summary"]["text"] = compact
    history["summary"]["updated_at"] = _now_iso_jst()
    return history

def maybe_summarize_history(model: "Llama",
                           history: Dict[str, Any],
                           max_turns: int,
                           summarize_old_history: bool = True,
                           summary_chunk_turns: int = 3,
                           temperature: float = 0.2,
                           max_tokens_summary: int = 128,
                           summary_max_chars: int = 1500,
                           suppress_logs: bool = False) -> Dict[str, Any]:
    """
    Phase 1.5:
    - Do NOT summarize every turn. Summarize only when overflow reaches summary_chunk_turns.
    - Summarize a fixed-size chunk from the oldest overflow turns, then trim that chunk.
    - Keep a rolling summary in history['summary'] and keep it compact.
    """
    if not summarize_old_history:
        return history

    turns = history.get("turns") or []
    mt = max(0, _coerce_int(max_turns, 12))
    if mt == 0:
        # No prior turns kept; optionally keep summary only if user wants it (we leave it unchanged)
        history["turns"] = []
        return history

    if len(turns) <= mt:
        return history

    chunk = max(1, _coerce_int(summary_chunk_turns, 3))
    overflow_n = len(turns) - mt
    if overflow_n < chunk:
        # Not enough overflow to justify a summary generation (saves time)
        return history

    # Take the oldest chunk from overflow
    to_summarize = turns[:chunk]
    remaining = turns[chunk:]

    history.setdefault("summary", {})
    history["summary"]["enabled"] = True
    existing = (history["summary"].get("text") or "").strip()

    new_sum = _summarize_with_model(
        model=model,
        existing_summary=existing,
        turns_chunk=to_summarize,
        temperature=temperature,
        max_tokens=max_tokens_summary,
        suppress_logs=suppress_logs,
    )

    history["summary"]["text"] = new_sum
    history["summary"]["updated_at"] = _now_iso_jst()
    history["turns"] = remaining

    # Compact summary if it grows too large
    history = maybe_compact_summary(
        model=model,
        history=history,
        summary_max_chars=summary_max_chars,
        temperature=temperature,
        max_tokens_summary=max_tokens_summary,
        suppress_logs=suppress_logs,
    )

    return history


class GGUFModelManager:
    """GGUF model manager class

    Notes on robustness:
    - Qwen3-VL uses an mmproj (vision projector). In practice, llama-cpp / chat handlers
      may keep mmproj-related state alive longer than expected. To avoid "stale mmproj"
      issues when switching models (e.g., 8B -> 4B), we:
        * treat (model_path, mmproj_path, n_ctx, n_gpu_layers, vision_mode) as a signature
        * explicitly unload + gc before loading a different signature
        * make mmproj auto-detection choose the best match for the selected model
    """

    def __init__(self):
        self.model: Optional[Llama] = None
        self.chat_handler = None

        # Keep a full signature of what's currently loaded
        self._signature: Optional[tuple] = None
        self.current_model_path: Optional[str] = None
        self.current_mmproj_path: Optional[str] = None

        # Optional override for where prompt cache should be stored
        self.prompt_cache_dir_override: Optional[str] = None

    def _normalize_path(self, p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        return os.path.normpath(p)

    def _infer_is_qwen2(self, model_path: str) -> bool:
        model_name_lower = os.path.basename(model_path).lower()
        return "qwen2" in model_name_lower

    def _infer_is_qwen3(self, model_path: str) -> bool:
        model_name_lower = os.path.basename(model_path).lower()
        return "qwen3" in model_name_lower

    def _infer_is_llava(self, model_path: str) -> bool:
        model_name_lower = os.path.basename(model_path).lower()
        # Accept common naming patterns: llava-v1.6-*, llava-*, llava1.6, etc.
        return "llava" in model_name_lower

    def _infer_is_llama(self, model_path: str) -> bool:
        model_name_lower = os.path.basename(model_path).lower()
        # Accept common naming patterns: llama-3.2, etc.
        return "llama" in model_name_lower

    def _infer_is_minicpm2(self, model_path: str) -> bool:
        model_name_lower = os.path.basename(model_path).lower()
        # Accept common naming patterns: llama-3.2, etc.
        return "minicpm-v-2_6" in model_name_lower

    def _infer_is_gemma3(self, model_path: str) -> bool:
        model_name_lower = os.path.basename(model_path).lower()
        # Common: gemma-3-*, gemma3-*, gemma_3_*
        return ("gemma-3" in model_name_lower) or ("gemma3" in model_name_lower) or ("gemma_3" in model_name_lower)

    def _infer_is_glm46v(self, model_path: str) -> bool:
        model_name_lower = os.path.basename(model_path).lower()
        # Common: glm-4.6v, glm-4.6v-flash, glm_4.6v, glm4.6v
        return "glm-4" in model_name_lower

    def _auto_detect_mmproj(self, model_path: str) -> Optional[str]:
        """
        Auto-detect mmproj by "family prefix" match.

        Rule:
        - Determine model family by basename(model).startswith one of:
            qwen2, qwen3, llava, llama, minicpm-v-2_6, gemma-3, glm-4
        - In the same directory, scan mmproj-*.gguf
        - Keep only those whose mmproj_name (after 'mmproj-') startswith the same family keyword
        - If exactly one match -> return it
        - Else (0 or >1) -> raise ValueError
        """
        model_dir = os.path.dirname(model_path)
        base = os.path.basename(model_path)
        name = base[:-5] if base.lower().endswith(".gguf") else base
        name_l = name.lower()

        # Model family keywords (startswith)
        families = ["qwen2", "qwen3", "llava", "llama", "minicpm-v-2_6", "gemma-3", "glm-4"]
        family = next((k for k in families if name_l.startswith(k)), None)

        if family is None:
            raise ValueError(
                "mmproj auto-detect failed: model name does not start with any supported family prefix.\n"
                f"model: {base}\n"
                f"supported prefixes: {', '.join(families)}"
            )

        if not os.path.exists(model_dir):
            raise ValueError(
                "mmproj auto-detect failed: model directory does not exist.\n"
                f"dir: {model_dir}"
            )

        # Collect mmproj-*.gguf in the same dir
        mmproj_files = [
            f for f in os.listdir(model_dir)
            if f.startswith("mmproj-") and f.endswith(".gguf")
        ]

        # Filter by family prefix on mmproj_name
        matches = []
        for f in mmproj_files:
            mmname = f[len("mmproj-"):-len(".gguf")]
            if mmname.lower().startswith(family):
                matches.append(f)

        matches.sort(key=str.lower)

        if len(matches) == 1:
            fname = matches[0]
            cand = os.path.join(model_dir, fname)
            print(f"[GGUFModelManager] Auto-detected mmproj (family={family}): {fname}")
            return self._normalize_path(cand)

        if len(matches) == 0:
            raise ValueError(
                "mmproj auto-detect failed: no mmproj matched the model family prefix.\n"
                f"model: {base}\n"
                f"family: {family}\n"
                f"dir: {model_dir}\n"
                f"mmproj candidates: {', '.join(sorted(mmproj_files, key=str.lower)) or '(none)'}"
            )

        # len(matches) > 1
        raise ValueError(
            "mmproj auto-detect failed: multiple mmproj files matched the model family prefix.\n"
            f"model: {base}\n"
            f"family: {family}\n"
            f"dir: {model_dir}\n"
            f"matched: {', '.join(matches)}\n"
            "Please select mmproj manually."
        )

    def _make_signature(
        self,
        model_path: str,
        mmproj_path: Optional[str],
        n_ctx: int,
        n_gpu_layers: int,
        use_vision: bool,
    ) -> tuple:
        return (
            self._normalize_path(model_path),
            self._normalize_path(mmproj_path),
            int(n_ctx),
            int(n_gpu_layers),
            bool(use_vision),
        )

    def load_model(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> Llama:
        """Load GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            msg = "llama_cpp is not available"
            if "_LLAMA_CPP_IMPORT_ERROR" in globals():
                msg += f" ({_LLAMA_CPP_IMPORT_ERROR})"
            raise RuntimeError(msg)

        model_path = self._normalize_path(model_path)

        # Infer Qwen version from model name
        is_qwen2 = self._infer_is_qwen2(model_path)
        is_qwen3 = self._infer_is_qwen3(model_path)
        # Infer LLaVA from model name
        is_llava = self._infer_is_llava(model_path)
        # Infer Llama from model name
        is_llama = self._infer_is_llama(model_path)
        # Infer MiniCPM-V 2.6
        is_minicpm26 = self._infer_is_gemma3(model_path)
        # Infer Gemma 3
        is_gemma3 = self._infer_is_gemma3(model_path)
        # Infer GLM-4.6V / GLM-4V family (heuristic)
        is_glm46v = self._infer_is_glm46v(model_path)

        # If user explicitly selected "(Not required)", force text-only.
        force_no_mmproj = (mmproj_path == "(Not required)")
        if force_no_mmproj:
            mmproj_path = None

        # Qwen25-VL requires mmproj (unless explicitly disabled)
        if is_qwen2 and not force_no_mmproj:
            if mmproj_path is None:
                mmproj_path = self._auto_detect_mmproj(model_path)
                if mmproj_path is None:
                    model_dir = os.path.dirname(model_path)
                    raise ValueError(
                        "Qwen25-VL requires mmproj file!\n"
                        "Please download mmproj file from the model's GGUF repo.\n"
                        f"Expected location: {model_dir}{os.sep}mmproj-*.gguf"
                    )
            else:
                mmproj_path = self._normalize_path(mmproj_path)

            print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")

        # Qwen3-VL requires mmproj (unless explicitly disabled)
        if is_qwen3 and not force_no_mmproj:
            if mmproj_path is None:
                mmproj_path = self._auto_detect_mmproj(model_path)
                if mmproj_path is None:
                    model_dir = os.path.dirname(model_path)
                    raise ValueError(
                        "Qwen3-VL requires mmproj file!\n"
                        "Please download mmproj file from the model's GGUF repo.\n"
                        f"Expected location: {model_dir}{os.sep}mmproj-*.gguf"
                    )
            else:
                mmproj_path = self._normalize_path(mmproj_path)

            print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")

        # LLaVA requires mmproj to actually use vision (unless explicitly disabled)
        if is_llava and not force_no_mmproj:
            if mmproj_path is None:
                mmproj_path = self._auto_detect_mmproj(model_path)
                if mmproj_path is None:
                    model_dir = os.path.dirname(model_path)
                    raise ValueError(
                        "LLaVA requires mmproj file for vision!\n"
                        "Please download the matching mmproj file from the model's GGUF repo.\n"
                        f"Expected location: {model_dir}{os.sep}mmproj-*.gguf"
                    )
            else:
                mmproj_path = self._normalize_path(mmproj_path)

            print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")

        # Llama requires mmproj to actually use vision (unless explicitly disabled)
        if is_llama and not force_no_mmproj:
            if mmproj_path is None:
                mmproj_path = self._auto_detect_mmproj(model_path)
                if mmproj_path is None:
                    model_dir = os.path.dirname(model_path)
                    raise ValueError(
                        "Llama requires mmproj file for vision!\n"
                        "Please download the matching mmproj file from the model's GGUF repo.\n"
                        f"Expected location: {model_dir}{os.sep}mmproj-*.gguf"
                    )
            else:
                mmproj_path = self._normalize_path(mmproj_path)

            print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")

        # MiniCPM-V 2.6 vision: usually uses mmproj as well (unless explicitly disabled)
        if is_minicpm26 and not force_no_mmproj:
            if mmproj_path is None:
                mmproj_path = self._auto_detect_mmproj(model_path)
                if mmproj_path is None:
                    model_dir = os.path.dirname(model_path)
                    raise ValueError(
                        "MiniCPM-V 2.6 (vision) requires mmproj file for vision!\n"
                        "Please download the matching mmproj file from the model's GGUF repo.\n"
                        f"Expected location: {model_dir}{os.sep}mmproj-*.gguf"
                    )
            else:
                mmproj_path = self._normalize_path(mmproj_path)

            print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")

        # Gemma 3 vision: usually uses mmproj as well (unless explicitly disabled)
        if is_gemma3 and not force_no_mmproj:
            if mmproj_path is None:
                mmproj_path = self._auto_detect_mmproj(model_path)
                if mmproj_path is None:
                    model_dir = os.path.dirname(model_path)
                    raise ValueError(
                        "Gemma-3 (vision) requires mmproj file for vision!\n"
                        "Please download the matching mmproj file from the model's GGUF repo.\n"
                        f"Expected location: {model_dir}{os.sep}mmproj-*.gguf"
                    )
            else:
                mmproj_path = self._normalize_path(mmproj_path)

            print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")

        # GLM-4.6V vision: many GGUF repos still ship a separate mmproj; require if using a vision handler.
        if is_glm46v and not force_no_mmproj and GLM46V_AVAILABLE:
            if mmproj_path is None:
                mmproj_path = self._auto_detect_mmproj(model_path)
                if mmproj_path is None:
                    model_dir = os.path.dirname(model_path)
                    raise ValueError(
                        "GLM-4.6V (vision) requires mmproj file for vision (handler present)!\n"
                        "Please download the matching mmproj file from the model's GGUF repo.\n"
                        f"Expected location: {model_dir}{os.sep}mmproj-*.gguf"
                    )
            else:
                mmproj_path = self._normalize_path(mmproj_path)

            print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")

        # Decide vision mode + initialize handler
        use_vision = False
        chat_handler = None

        if is_qwen2 and (not force_no_mmproj) and QWEN2_AVAILABLE:
            if mmproj_path is not None and os.path.exists(mmproj_path):
                try:
                    print(f"[GGUFModelManager] Qwen25-VL with mmproj: {mmproj_path}")
                    chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                    use_vision = True
                except Exception as e:
                    print(f"[GGUFModelManager] Warning: Failed to initialize Qwen25-VL chat handler: {e}")
                    chat_handler = None
                    use_vision = False
            else:
                print("[GGUFModelManager] Error: Qwen25-VL requires an existing mmproj file")
                chat_handler = None
                use_vision = False

        elif is_qwen3 and (not force_no_mmproj) and QWEN3_AVAILABLE:
            if mmproj_path is not None and os.path.exists(mmproj_path):
                try:
                    print(f"[GGUFModelManager] Qwen3-VL with mmproj: {mmproj_path}")
                    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path)
                    use_vision = True
                except Exception as e:
                    print(f"[GGUFModelManager] Warning: Failed to initialize Qwen3-VL chat handler: {e}")
                    chat_handler = None
                    use_vision = False
            else:
                print("[GGUFModelManager] Error: Qwen3-VL requires an existing mmproj file")
                chat_handler = None
                use_vision = False
        elif is_minicpm26 and (not force_no_mmproj) and MINICPM26_AVAILABLE:
            if mmproj_path is not None and os.path.exists(mmproj_path):
                try:
                    print(f"[GGUFModelManager] Gemma-3 with mmproj: {mmproj_path}")
                    chat_handler = Gemma3ChatHandler(clip_model_path=mmproj_path)
                    use_vision = True
                except Exception as e:
                    print(f"[GGUFModelManager] Warning: Failed to initialize Gemma-3 chat handler: {e}")
                    chat_handler = None
                    use_vision = False
        elif is_gemma3 and (not force_no_mmproj) and GEMMA3_AVAILABLE:
            if mmproj_path is not None and os.path.exists(mmproj_path):
                try:
                    print(f"[GGUFModelManager] Gemma-3 with mmproj: {mmproj_path}")
                    chat_handler = Gemma3ChatHandler(clip_model_path=mmproj_path)
                    use_vision = True
                except Exception as e:
                    print(f"[GGUFModelManager] Warning: Failed to initialize Gemma-3 chat handler: {e}")
                    chat_handler = None
                    use_vision = False
            else:
                print("[GGUFModelManager] Error: Gemma-3 requires an existing mmproj file for vision")
                chat_handler = None
                use_vision = False

        elif is_llava and (not force_no_mmproj) and LLAVA_AVAILABLE:
            if mmproj_path is not None and os.path.exists(mmproj_path):
                try:
                    print(f"[GGUFModelManager] LLaVA with mmproj: {mmproj_path}")
                    # Llava15ChatHandler is used for llava-v1.5/v1.6 style models in many builds
                    chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                    use_vision = True
                except Exception as e:
                    print(f"[GGUFModelManager] Warning: Failed to initialize LLaVA chat handler: {e}")
                    chat_handler = None
                    use_vision = False
            else:
                print("[GGUFModelManager] Error: LLaVA requires an existing mmproj file for vision")
                chat_handler = None
                use_vision = False

        elif is_llama and (not force_no_mmproj) and LLAMA_AVAILABLE:
            if mmproj_path is not None and os.path.exists(mmproj_path):
                try:
                    print(f"[GGUFModelManager] Llama with mmproj: {mmproj_path}")
                    # Llama3VisionAlphaChatHandler is used for llava-3.2 style models in many builds
                    chat_handler = Llama3VisionAlphaChatHandler(clip_model_path=mmproj_path)
                    use_vision = True
                except Exception as e:
                    print(f"[GGUFModelManager] Warning: Failed to initialize Llama chat handler: {e}")
                    chat_handler = None
                    use_vision = False
            else:
                print("[GGUFModelManager] Error: Llama requires an existing mmproj file for vision")
                chat_handler = None
                use_vision = False

        elif is_glm46v and (not force_no_mmproj) and GLM46V_AVAILABLE:
            if mmproj_path is not None and os.path.exists(mmproj_path):
                try:
                    print(f"[GGUFModelManager] GLM-4.6V with mmproj: {mmproj_path}")
                    chat_handler = GLM46VChatHandler(clip_model_path=mmproj_path)  # type: ignore[misc]
                    use_vision = True
                except Exception as e:
                    print(f"[GGUFModelManager] Warning: Failed to initialize GLM-4.6V chat handler: {e}")
                    chat_handler = None
                    use_vision = False
            else:
                print("[GGUFModelManager] Error: GLM-4.6V requires an existing mmproj file for vision (handler present)")
                chat_handler = None
                use_vision = False

        else:
            print("[GGUFModelManager] Using text-only mode")
            chat_handler = None
            use_vision = False

        new_sig = self._make_signature(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            use_vision=use_vision,
        )

        # If signature matches, reuse
        if self.model is not None and self._signature == new_sig:
            print(f"[GGUFModelManager] Using cached model: {model_path}")
            return self.model

        # Otherwise, explicitly unload to avoid stale mmproj/handler state
        if self.model is not None:
            print("[GGUFModelManager] Signature changed -> unloading previous model to avoid stale state")
            self.unload_model()

        print(f"[GGUFModelManager] Loading model: {model_path}")
        print(f"[GGUFModelManager] n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")

        # Store handler on manager (used later to decide if images are supported)
        self.chat_handler = chat_handler

        # Model loading
        if use_vision and self.chat_handler is not None:
            print("[GGUFModelManager] Loading with vision support")
            self.model = Llama(
                model_path=model_path,
                chat_handler=self.chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                # Vision models often need this; safe default for vision path.
                logits_all=True,
            )
        else:
            print("[GGUFModelManager] Loading in text-only mode")
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )

        self.current_model_path = model_path
        self.current_mmproj_path = self._normalize_path(mmproj_path)
        self._signature = new_sig

        print("[GGUFModelManager] Model loaded successfully")
        return self.model

    def _default_prompt_cache_dir(self, model_path: str, mmproj_path: str, n_ctx: int) -> str:
        """
        Compute a stable cache directory for prompt/KV cache.
        We key by model+mmproj+n_ctx so caches are not mixed across incompatible settings.
        """
        base = (self.prompt_cache_dir_override or os.path.join(_safe_output_dir(), "llm_session_sessions", "prompt_cache"))
        os.makedirs(base, exist_ok=True)
        key_src = f"{os.path.abspath(model_path)}|{os.path.abspath(mmproj_path or '')}|n_ctx={int(n_ctx)}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join(base, key)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def configure_prompt_cache(self, llm, model_path: str, mmproj_path: str, n_ctx: int, cache_mode: str = "disk") -> None:
        """
        Enable llama-cpp-python prompt/KV caching when supported.

        - "disk": persistent cache under output/.../prompt_cache/<hash>/
        - "memory": in-memory cache (if supported)
        - "off": disable (best effort)
        """
        cache_mode = (cache_mode or "disk").lower()
        try:
            # llama-cpp-python provides cache helpers in many versions (names vary).
            import llama_cpp
        except Exception as e:
            if cache_mode != "off":
                print(f"[GGUFModelManager] Prompt cache requested but llama_cpp import failed: {e}")
            return

        # Determine desired cache object
        cache_obj = None
        cache_desc = None
        try:
            if cache_mode == "off":
                # Best-effort disable: set cache to None if setter exists
                if hasattr(llm, "set_cache"):
                    llm.set_cache(None)
                self._current_cache_info = None
                print("[GGUFModelManager] Prompt cache: OFF")
                return

            if cache_mode == "disk":
                cache_dir = self._default_prompt_cache_dir(model_path, mmproj_path, n_ctx)
                # Common class names across versions
                if hasattr(llama_cpp, "LlamaDiskCache"):
                    cache_obj = llama_cpp.LlamaDiskCache(cache_dir)
                    cache_desc = f"disk:{cache_dir}"
                elif hasattr(llama_cpp, "LlamaCache"):
                    # Some versions use LlamaCache(directory=...)
                    try:
                        cache_obj = llama_cpp.LlamaCache(cache_dir)
                        cache_desc = f"disk:{cache_dir}"
                    except Exception:
                        cache_obj = None
                elif hasattr(llama_cpp, "DiskCache"):
                    cache_obj = llama_cpp.DiskCache(cache_dir)
                    cache_desc = f"disk:{cache_dir}"
                else:
                    # No known disk cache class
                    cache_obj = None

            if cache_mode == "memory" and cache_obj is None:
                if hasattr(llama_cpp, "LlamaRAMCache"):
                    cache_obj = llama_cpp.LlamaRAMCache()
                    cache_desc = "memory"
                elif hasattr(llama_cpp, "LlamaCache"):
                    try:
                        cache_obj = llama_cpp.LlamaCache()
                        cache_desc = "memory"
                    except Exception:
                        cache_obj = None
                elif hasattr(llama_cpp, "RAMCache"):
                    cache_obj = llama_cpp.RAMCache()
                    cache_desc = "memory"

            if cache_obj is None:
                print(f"[GGUFModelManager] Prompt cache: not supported by this llama-cpp-python build (mode={cache_mode})")
                return

            # Apply cache to model (different versions expose different APIs)
            applied = False
            if hasattr(llm, "set_cache"):
                llm.set_cache(cache_obj)
                applied = True
            # Some versions accept .cache attribute
            if not applied and hasattr(llm, "cache"):
                try:
                    llm.cache = cache_obj
                    applied = True
                except Exception:
                    pass

            if applied:
                self._current_cache_info = {
                    "mode": cache_mode,
                    "desc": cache_desc,
                }
                print(f"[GGUFModelManager] Prompt cache enabled: {cache_desc}")
            else:
                print(f"[GGUFModelManager] Prompt cache object created but could not be applied (mode={cache_mode})")
        except Exception as e:
            print(f"[GGUFModelManager] Prompt cache setup failed (mode={cache_mode}): {e}")

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            print(f"[GGUFModelManager] Unloading model: {self.current_model_path}")
        try:
            if self.model is not None:
                del self.model
        finally:
            self.model = None

        try:
            if self.chat_handler is not None:
                del self.chat_handler
        finally:
            self.chat_handler = None

        self.current_model_path = None
        self.current_mmproj_path = None
        self._signature = None

        # Encourage timely cleanup (important for llama-cpp backends and mmproj state)
        import gc as _gc
        _gc.collect()

# Global model manager
_model_manager = GGUFModelManager()

# In-memory KV/state cache (session_id -> {signature:str, state:any})
_MEM_KV_STATE = {}


# ============================================================================
# ComfyUI Node Registration
# ============================================================================


# =============================================================================
# LLM Session Chat (Simple)
# =============================================================================

class LLMSessionChatSimpleNode:
    """LLM Session Chat (Simple)

    Minimal UI inputs with config-driven defaults.
    Defaults are loaded from config/simple_defaults.json when present.
    """

    @classmethod
    def INPUT_TYPES(cls):
        roots = _get_llm_model_roots()
        available_models, available_mmprojs = _list_gguf_recursive_multi(roots)

        mmproj_options = available_mmprojs + ["(Auto-detect)", "(Not required)"]
        if not available_mmprojs:
            mmproj_options = ["(Auto-detect)", "(Not required)"]

        if not available_models:
            available_models = ["(No GGUF models found in models/LLM/)"]

        return {
            "required": {
                "user_text": ("STRING", {"multiline": True, "default": "", "tooltip": "User message for this turn"}),
                "session_id": ("STRING", {"default": "default", "tooltip": "Session ID (maps to a history file). Same ID continues the chat."}),
                "model": (available_models, {"default": available_models[0], "tooltip": "GGUF model file in models/LLM/"}),
                "mmproj": (mmproj_options, {"default": "(Auto-detect)", "tooltip": "Manual selection is recommended."}),
                "history_dir": ("STRING", {"default": "", "tooltip": "Optional directory for history/caches. Empty uses output/llm_session_sessions/."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image input for this turn only (never saved to history)"}),
                "config_path": ("STRING", {"default": "", "tooltip": "Optional override path to simple_defaults.json (advanced)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("assistant_text",)
    FUNCTION = "chat_stream"
    CATEGORY = "LLM/Session"
    DESCRIPTION = "Minimal entry node for persistent local GGUF chat sessions (config-driven defaults)."

    def chat_stream(
        self,
        user_text: str,
        session_id: str,
        model: str,
        mmproj: str,
        history_dir: str,
        image=None,
        config_path: str = "",
        stream_to_console: bool = True,
    ) -> tuple:
        defaults = _load_simple_defaults(config_path=config_path)

        # Delegate to the full node implementation
        node = LLMSessionChatNode()
        return node.chat_stream(
            user_text=user_text,
            session_id=session_id,
            model=model,
            mmproj=mmproj,
            system_prompt=defaults["system_prompt"],
            max_tokens=int(defaults["max_tokens"]),
            temperature=float(defaults["temperature"]),
            top_p=float(defaults["top_p"]),
            n_gpu_layers=int(defaults["n_gpu_layers"]),
            n_ctx=int(defaults["n_ctx"]),
            image=image,
            max_turns=int(defaults["max_turns"]),
            summarize_old_history=bool(defaults["summarize_old_history"]),
            summary_chunk_turns=int(defaults["summary_chunk_turns"]),
            max_tokens_summary=int(defaults["max_tokens_summary"]),
            summary_max_chars=int(defaults["summary_max_chars"]),
            dynamic_max_tokens=bool(defaults["dynamic_max_tokens"]),
            min_generation_tokens=int(defaults["min_generation_tokens"]),
            safety_margin_tokens=int(defaults["safety_margin_tokens"]),
            prompt_cache_mode=str(defaults["prompt_cache_mode"]),
            repeat_penalty=float(defaults["repeat_penalty"]),
            repeat_last_n=int(defaults["repeat_last_n"]),
            rewrite_continue=bool(defaults["rewrite_continue"]),
            kv_state_mode=str(defaults["kv_state_mode"]),
            log_level=str(defaults["log_level"]),
            suppress_backend_logs=bool(defaults["suppress_backend_logs"]),
            history_dir=history_dir or "",
            reset_session=bool(defaults["reset_session"]),
            stream_to_console=bool(defaults["stream_to_console"]),
        )


# =============================================================================
# LLM Session Chat - file-based hidden history (Phase 1)
# =============================================================================

class LLMDialogueCycleSimpleNode:
    """
    LLM Dialogue Cycle (Simple)

    Minimal-input wrapper around LLM Dialogue Cycle using a JSON config file for defaults.
    - Keeps UI simple (few parameters)
    - Users can override defaults via config/simple_defaults.json
    - Safe fallback to built-in defaults if the config is missing or invalid
    """

    @classmethod
    def INPUT_TYPES(cls):
        roots = _get_llm_model_roots()
        available_models, available_mmprojs = _list_gguf_recursive_multi(roots)

        mmproj_options = available_mmprojs + ["(Auto-detect)", "(Not required)"]
        if not available_mmprojs:
            mmproj_options = ["(Auto-detect)", "(Not required)"]

        if not available_models:
            available_models = ["(No GGUF models found in models/LLM/)"]

        return {
            "required": {
                "initial_user_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Initial user message (sent to Model A only)."
                }),
                
                "system": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Shared system prompt. Leave empty to use config/default."
                }),
                "systemA": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System prompt override for Model A. Leave empty to use config/default."
                }),
                "systemB": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System prompt override for Model B. Leave empty to use config/default."
                }),
"session_id": ("STRING", {
                    "default": "default",
                    "tooltip": "Session id. A uses {id}_A, B uses {id}_B."
                }),
                "cycles": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of turns. 1 = A then B."
                }),
                "modelA": (available_models, {
                    "default": available_models[0],
                    "tooltip": "GGUF model for role A"
                }),
                "modelB": (available_models, {
                    "default": available_models[0],
                    "tooltip": "GGUF model for role B"
                }),
                "history_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to store histories, summaries, transcript, and caches. Empty uses ComfyUI output."
                }),
            },
            "optional": {
                "config_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional path to a JSON config file. If empty, uses <node_dir>/config/simple_defaults.json."
                }),
                "force_text_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force text-only mode for both models (disables mmproj auto-detect)."
                }),
                "reset_session": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, overwrite existing session history with a fresh session."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcript_text",)
    FUNCTION = "chat_cycle_simple"
    CATEGORY = "LLM/Session"
    DESCRIPTION = "Run a two-model dialogue cycle with minimal UI; defaults are configurable via JSON."

    def chat_cycle_simple(
        self,
        initial_user_text: str,
        system: str,
        systemA: str,
        systemB: str,
        session_id: str,
        cycles: int,
        modelA: str,
        modelB: str,
        history_dir: str,
        config_path: str = "",
        force_text_only: bool = False,
        reset_session: bool = False,
    ) -> tuple:
        # Resolve models list placeholder
        if "(No GGUF models found" in (modelA or "") or "(No GGUF models found" in (modelB or ""):
            return ("",)

        # Load defaults from config (or fallback)
        defaults = _load_simple_defaults(config_path or "")

        # System prompts (shared + optional overrides)
        # UI fields take priority when non-empty; otherwise fall back to config/defaults.
        system_prompt = (system or "").strip() or str(defaults.get("system_prompt") or "You are a helpful assistant.")
        system_prompt_A = (systemA or "").strip() or str(defaults.get("system_prompt_A") or "")
        system_prompt_B = (systemB or "").strip() or str(defaults.get("system_prompt_B") or "")

        # Sampling / runtime parameters
        max_tokens = int(defaults.get("max_tokens") or 512)
        temperature = float(defaults.get("temperature") or 0.7)
        top_p = float(defaults.get("top_p") or 0.9)
        n_gpu_layers = int(defaults.get("n_gpu_layers") or 0)
        n_ctx = int(defaults.get("n_ctx") or 4096)

        # History + summary parameters
        max_turns = int(defaults.get("max_turns") or 6)
        summarize_old_history = bool(defaults.get("summarize_old_history") if "summarize_old_history" in defaults else True)
        summary_chunk_turns = int(defaults.get("summary_chunk_turns") or 6)
        max_tokens_summary = int(defaults.get("max_tokens_summary") or 128)
        summary_max_chars = int(defaults.get("summary_max_chars") or 1500)

        # Dynamic context protection
        dynamic_max_tokens = bool(defaults.get("dynamic_max_tokens") if "dynamic_max_tokens" in defaults else True)
        min_generation_tokens = int(defaults.get("min_generation_tokens") or 96)
        safety_margin_tokens = int(defaults.get("safety_margin_tokens") or 64)

        # Cache + repetition controls
        prompt_cache_mode = str(defaults.get("prompt_cache_mode") or "disk")
        kv_state_mode = str(defaults.get("kv_state_mode") or "memory")
        repeat_penalty = float(defaults.get("repeat_penalty") or 1.12)
        repeat_last_n = int(defaults.get("repeat_last_n") or 256)
        rewrite_continue = bool(defaults.get("rewrite_continue") if "rewrite_continue" in defaults else True)

        # Logging
        log_level = str(defaults.get("log_level") or "timing")
        suppress_backend_logs = bool(defaults.get("suppress_backend_logs") if "suppress_backend_logs" in defaults else True)

        # mmproj handling:
        # - Default is auto-detect for both roles.
        # - Force text-only disables mmproj auto-detect.
        mmprojA = "(Not required)" if force_text_only else "(Auto-detect)"
        mmprojB = "(Not required)" if force_text_only else "(Auto-detect)"

        node = LLMDialogueCycleNode()
        return node.chat_cycle(
            initial_user_text=initial_user_text,
            session_id=session_id,
            cycles=int(cycles),
            modelA=modelA,
            mmprojA=mmprojA,
            modelB=modelB,
            mmprojB=mmprojB,
            system_prompt=system_prompt,
            system_prompt_A=system_prompt_A,
            system_prompt_B=system_prompt_B,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            n_gpu_layers=int(n_gpu_layers),
            n_ctx=int(n_ctx),
            max_turns=int(max_turns),
            summarize_old_history=bool(summarize_old_history),
            summary_chunk_turns=int(summary_chunk_turns),
            max_tokens_summary=int(max_tokens_summary),
            summary_max_chars=int(summary_max_chars),
            dynamic_max_tokens=bool(dynamic_max_tokens),
            min_generation_tokens=int(min_generation_tokens),
            safety_margin_tokens=int(safety_margin_tokens),
            prompt_cache_mode=prompt_cache_mode,
            kv_state_mode=kv_state_mode,
            repeat_penalty=float(repeat_penalty),
            repeat_last_n=int(repeat_last_n),
            rewrite_continue=bool(rewrite_continue),
            log_level=log_level,
            suppress_backend_logs=bool(suppress_backend_logs),
            history_dir=history_dir or "",
            reset_session=bool(reset_session),
            stream_to_console=bool(defaults.get("stream_to_console") or False),
        )


# =============================================================================
# LLM Dialogue Cycle (Simple)
# =============================================================================

class LLMSessionChatNode:
    """
    LLM Session Chat - Local GGUF vision language models with file-based chat history.

    Phase 1 design:
    - History is stored/updated as a JSON file under output/llm_session_sessions/
    - History is NOT shown in ComfyUI UI (only assistant response is returned)
    - Images are used ONLY for the current turn and never persisted
    - Supports max_turns / summarize_old_history / system_prompt
    """

    @classmethod
    def INPUT_TYPES(cls):
        roots = _get_llm_model_roots()
        available_models, available_mmprojs = _list_gguf_recursive_multi(roots)

        mmproj_options = available_mmprojs + ["(Auto-detect)", "(Not required)"]
        if not available_mmprojs:
            mmproj_options = ["(Auto-detect)", "(Not required)"]

        if not available_models:
            available_models = ["(No GGUF models found in models/LLM/)"]

        return {
            "required": {
                "user_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "User message for this turn"
                }),
                "session_id": ("STRING", {
                    "default": "default",
                    "tooltip": "Session ID (maps to a history file). Same ID continues the chat."
                }),
                "model": (available_models, {
                    "default": available_models[0],
                    "tooltip": "GGUF model file in models/LLM/"
                }),
                "mmproj": (mmproj_options, {
                    "default": "(Auto-detect)",
                    "tooltip": "Manual selection is recommended."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "System prompt (conversation policy). Saved into the history file."
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Maximum tokens to generate for this turn"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling (top_p). Lower = safer/more conservative."
                }),
                "n_gpu_layers": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Number of layers to offload to GPU. 0=CPU. -1=all."
                }),
                "n_ctx": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 131072,
                    "step": 256,
                    "tooltip": "Context length (must be supported by the model)"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image input for this turn only (never saved to history)"
                }),
                "prompt_cache_mode": (["disk", "memory", "off"], {
                    "default": "disk",
                    "tooltip": "Enable llama.cpp prompt/KV caching to speed up repeated prefixes (best effort). 'disk' persists under output/llm_session_sessions/prompt_cache/."
                }),
                "kv_state_mode": (["memory", "off"], {
                    "default": "memory",
                    "tooltip": "Enable in-memory llama-cpp-python save_state/load_state cache per session."
                }),
                "log_level": (["minimal", "timing", "debug"], {
                    "default": "timing",
                    "tooltip": "Console logging verbosity for LLM Session Chat."
                }),
                "suppress_backend_logs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Suppress backend stdout/stderr during generation."
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.12,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Repetition penalty to reduce looping outputs (especially on continue)."
                }),
                "repeat_last_n": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Apply repeat_penalty over the last N tokens. 0 disables."
                }),
                "rewrite_continue": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Rewrite inputs starting with 'continue' into an explicit continuation instruction to reduce repetition."
                }),
                "max_turns": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 200,
                    "tooltip": "Keep only the last N turns in live context. 0 means no prior turns."
                }),
                "summarize_old_history": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Summarize overflow turns into a rolling summary when turns exceed max_turns."
                }),
                
                "summary_chunk_turns": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Summarize overflow in chunks of this many turns (reduces summary frequency)."
                }),
                "max_tokens_summary": ("INT", {
                    "default": 128,
                    "min": 16,
                    "max": 2048,
                    "tooltip": "Max tokens for summary generation (kept small for speed)."
                }),
                "summary_max_chars": ("INT", {
                    "default": 1500,
                    "min": 200,
                    "max": 20000,
                    "tooltip": "If the rolling summary exceeds this size, it will be re-summarized to stay compact."
                }),
                "dynamic_max_tokens": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Dynamically shrink max_tokens (and/or turns) when prompt would exceed n_ctx."
                }),
                "min_generation_tokens": ("INT", {
                    "default": 96,
                    "min": 1,
                    "max": 4096,
                    "tooltip": "Minimum tokens to allow for generation when dynamic_max_tokens is enabled."
                }),
                "safety_margin_tokens": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 2048,
                    "tooltip": "Token margin reserved to reduce the chance of exceeding n_ctx."
                }),
                "history_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Optional directory for history files. Empty uses output/llm_session_sessions/"
                }),
                "reset_session": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, overwrite existing session history file with a fresh session."
                }),
                "stream_to_console": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stream tokens to console while generating."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("assistant_text",)
    FUNCTION = "chat_stream"
    CATEGORY = "LLM/Session"
    DESCRIPTION = "Persistent multi-turn chat with local GGUF models using file-based history."

    def chat_stream(self,
             user_text: str,
             session_id: str,
             model: str,
             mmproj: str,
             system_prompt: str,
             max_tokens: int,
             temperature: float,
             top_p: float,
             n_gpu_layers: int,
             n_ctx: int,
             image=None,
             max_turns: int = 12,
             summarize_old_history: bool = True,
             summary_chunk_turns: int = 3,
             max_tokens_summary: int = 128,
             summary_max_chars: int = 1500,
             dynamic_max_tokens: bool = True,
             min_generation_tokens: int = 96,
             safety_margin_tokens: int = 64,
             prompt_cache_mode: str = "disk",
             repeat_penalty: float = 1.12,
             repeat_last_n: int = 256,
             rewrite_continue: bool = True,
             kv_state_mode: str = "memory",
             log_level: str = "timing",
             suppress_backend_logs: bool = True,
             history_dir: str = "",
             reset_session: bool = False,
             stream_to_console: bool = False) -> tuple:
        global _model_manager
        _t_total = time.perf_counter()
        def _log_total(status: str):
            dt = time.perf_counter() - _t_total
            print(f"[LLM Session Chat] {status} in {dt:.2f} seconds")

        if not LLAMA_CPP_AVAILABLE:
            msg = "llama_cpp is not available"
            if "_LLAMA_CPP_IMPORT_ERROR" in globals():
                msg += f" ({_LLAMA_CPP_IMPORT_ERROR})"
            raise RuntimeError(msg)
            _log_total("Finished (error)")
            return ("",)

        if "(No GGUF models found" in model:
            print("[LLM Session Chat] Error: No GGUF models found in models/LLM/")
            _log_total("Finished (error)")
            return ("",)

        # Resolve paths
        roots = _get_llm_model_roots()
        model_path = _resolve_llm_relpath(model, roots=roots)
        if not os.path.exists(model_path):
            print(f"[LLM Session Chat] Error: Model not found: {model_path}")
            _log_total("Finished (error)")
            return ("",)

        mmproj_path = None
        if mmproj == "(Not required)":
            # Sentinel to force text-only (prevents mmproj auto-detect / VL handler)
            mmproj_path = "(Not required)"
        elif mmproj != "(Auto-detect)":
            mmproj_path = os.path.normpath(_resolve_llm_relpath(mmproj, roots=roots))
            if not os.path.exists(mmproj_path):
                print(f"[LLM Session Chat] Warning: mmproj not found: {mmproj_path}")
                mmproj_path = None  # fall back to auto-detect
        # Best-effort model signature for the history file (for user visibility / future KV work)
        model_sig = {
            "model_file": os.path.basename(model_path),
            "mmproj_file": os.path.basename(mmproj_path) if mmproj_path else "",
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
        }

        # Load / initialize history (file-based)
        history, hist_path = load_history(
            session_id=session_id,
            history_dir=(history_dir or None),
            system_prompt=system_prompt,
            model_sig=model_sig,
            reset_session=bool(reset_session),
        )

        # Continue rewrite with language detection (improved version)
        _is_continue = bool(re.match(r"^\s*continue\b", (user_text or ""), flags=re.IGNORECASE))
        user_text_for_model = user_text or ""
        if rewrite_continue and _is_continue:
            # Detect language from conversation history
            detected_lang = _detect_history_language(history)
            
            # Language-specific continue prompts
            CONTINUE_PROMPTS = {
                "zh": (
                    "从最后一个完整的句子继续。"
                    "不要重复之前的短语、列表、标题或段落。"
                    "如果你开始重复词语或地名，立即停止并写一个一句话摘要。"
                ),
                "ja": (
                    "最後の完全な文から続けてください。"
                    "以前のフレーズ、リスト、見出し、セクションを繰り返さないでください。"
                    "単語や地名を繰り返し始めたら、すぐに停止して1文の要約を書いてください。"
                ),
                "en": (
                    "Continue from the last complete sentence. "
                    "Do not repeat previous phrases, lists, headings, or sections. "
                    "If you start repeating words or place names, stop immediately and write a 1-sentence summary instead."
                ),
            }
            
            # Use detected language prompt, fallback to English with simplified instruction
            user_text_for_model = CONTINUE_PROMPTS.get(
                detected_lang,
                "Continue from the last complete sentence. Do not repeat previous content."
            )
            
            if log_level == "debug":
                print(f"[LLM Session Chat] Continue detected, language: {detected_lang}")


        # Store prompt-cache under the same directory as the history file
        try:
            _model_manager.prompt_cache_dir_override = os.path.join(os.path.dirname(hist_path), "prompt_cache")
            os.makedirs(_model_manager.prompt_cache_dir_override, exist_ok=True)
        except Exception:
            _model_manager.prompt_cache_dir_override = None

        # Prepare current-turn image tensor (optional)
        img_tensor = image if image is not None else None

        # Load model (reuses global manager; unloads safely when signature changes)
        if _model_manager is None:
            _model_manager = GGUFModelManager()

        try:
            llm = _model_manager.load_model(
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_ctx=int(n_ctx),
                n_gpu_layers=int(n_gpu_layers),
                verbose=False,
            )
        except Exception as e:
            print(f"[LLM Session Chat] Error loading model: {e}")
            _log_total("Finished (error)")
            return ("",)

        # Best-effort prompt/KV cache configuration (llama-cpp-python feature; may be unsupported)
        try:
            _model_manager.configure_prompt_cache(
                llm,
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_ctx=int(n_ctx),
                cache_mode=prompt_cache_mode,
            )
        except Exception as e:
            print(f"[LLM Session Chat] Prompt cache setup warning: {e}")


        # Build messages (history summary + last turns + current turn w/ optional image)
        messages = build_chat_messages(
            history=history,
            user_text=user_text_for_model or "",
            image_tensor=img_tensor,
            max_turns=int(max_turns) if max_turns is not None else None,
            summarize_old_history=bool(summarize_old_history),
            system_prompt=system_prompt or "",
        )

        # In-memory KV/state cache (best-effort)
        if (kv_state_mode or "off").lower() == "memory" and image is None:
            try:
                _turns_ctx = (history.get("turns") or [])
                _mt = int(max_turns) if (max_turns is not None) else None
                if _mt is not None and _mt >= 0:
                    _turns_ctx = _turns_ctx[-_mt:] if _mt > 0 else []
                _summary_txt = ""
                if bool(summarize_old_history) and history.get("summary", {}).get("enabled", False):
                    _summary_txt = history.get("summary", {}).get("text", "") or ""
                _effective_system = (system_prompt or history.get("system_prompt", "") or "").strip()
                _kv_prefix_material = json.dumps({
                    "model_path": os.path.abspath(model_path) if model_path else "",
                    "mmproj_path": os.path.abspath(mmproj_path) if mmproj_path else "",
                    "n_ctx": int(n_ctx) if n_ctx is not None else None,
                    "n_gpu_layers": int(n_gpu_layers) if n_gpu_layers is not None else None,
                    "system": _effective_system,
                    "summary": _summary_txt,
                    "turns": _turns_ctx,
                }, ensure_ascii=False, sort_keys=True)
                _kv_sig = hashlib.sha256(_kv_prefix_material.encode("utf-8")).hexdigest()
                entry = _MEM_KV_STATE.get(session_id)
                if entry and entry.get("signature") == _kv_sig and hasattr(llm, "load_state"):
                    llm.load_state(entry.get("state"))
                    if log_level != "minimal":
                        print("[LLM Session Chat] KV state: HIT (memory)")
                else:
                    if log_level != "minimal":
                        reason = "no state" if not entry else "prefix changed"
                        print(f"[LLM Session Chat] KV state: MISS ({reason})")
            except Exception as e:
                if log_level == "debug":
                    print(f"[LLM Session Chat] KV state: DISABLED ({e})")

        
        # Generate (with optional dynamic fallback when n_ctx is exceeded)
        assistant_text = ""
        gen_tokens = int(max_tokens)
        turns_limit = int(max_turns) if max_turns is not None else None

        def _is_ctx_error(err: Exception) -> bool:
            s = str(err)
            return ("exceeds n_ctx" in s) or ("Prompt exceeds n_ctx" in s) or ("n_ctx" in s and "exceed" in s)

        attempts = 0
        last_err = None
        while attempts < 6:
            try:
                _t_attempt = time.perf_counter()
                resp = None
                _kwargs = {
                    "messages": messages,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_tokens": int(gen_tokens),
                }
                # Repetition controls (supported by many llama-cpp-python builds)
                if repeat_last_n and int(repeat_last_n) > 0:
                    _kwargs["repeat_last_n"] = int(repeat_last_n)
                if repeat_penalty and float(repeat_penalty) != 1.0:
                    _kwargs["repeat_penalty"] = float(repeat_penalty)

                with _make_suppress_backend_logs(bool(suppress_backend_logs) and (log_level != "debug")):
                    if stream_to_console:
                        pieces: List[str] = []
                        out = sys.__stdout__
                        try:
                            stream_iter = _iter_chat_completion_robust(llm, messages, **{k:v for k,v in _kwargs.items() if k!='messages'})
                        except TypeError:
                            _kwargs.pop("repeat_last_n", None)
                            _kwargs.pop("repeat_penalty", None)
                            _kwargs.pop("top_p", None)
                            stream_iter = _iter_chat_completion_robust(llm, messages, **{k:v for k,v in _kwargs.items() if k!='messages'})
                        for chunk in stream_iter:
                            token = _extract_stream_content(chunk)
                            if not token:
                                continue
                            pieces.append(token)
                            try:
                                out.write(token)
                                out.flush()
                            except Exception:
                                pass
                        assistant_text = "".join(pieces)
                    else:
                        try:
                            resp = _create_chat_completion_robust(llm, messages, **{k:v for k,v in _kwargs.items() if k!='messages'})
                        except TypeError:
                            # Older builds may not accept repetition kwargs
                            _kwargs.pop("repeat_last_n", None)
                            _kwargs.pop("repeat_penalty", None)
                            _kwargs.pop("top_p", None)
                            resp = _create_chat_completion_robust(llm, messages, **{k:v for k,v in _kwargs.items() if k!='messages'})
                        assistant_text = resp["choices"][0]["message"]["content"]
                _dt = time.perf_counter() - _t_attempt
                print(f"[LLM Session Chat] Generation attempt {attempts+1} succeeded in {_dt:.2f} seconds (max_tokens={int(gen_tokens)}, turns_limit={turns_limit})")
                break
            except Exception as e:
                _dt = time.perf_counter() - _t_attempt
                print(f"[LLM Session Chat] Generation attempt {attempts+1} failed in {_dt:.2f} seconds (max_tokens={int(gen_tokens)}, turns_limit={turns_limit}): {e}")
                last_err = e
                if dynamic_max_tokens and _is_ctx_error(e):
                    # 1) Shrink generation budget first
                    if gen_tokens > int(min_generation_tokens):
                        gen_tokens = max(int(min_generation_tokens), gen_tokens // 2)
                    else:
                        # 2) Reduce live turns as a fallback
                        if turns_limit is not None and turns_limit > 0:
                            turns_limit = max(0, turns_limit - 1)
                        else:
                            # 3) As last resort, compact rolling summary (if any)
                            if summarize_old_history:
                                try:
                                    history = maybe_compact_summary(
                                        model=llm,
                                        history=history,
                                        summary_max_chars=int(summary_max_chars),
                                        temperature=0.2,
                                        max_tokens_summary=int(max_tokens_summary),
                                        suppress_logs=(log_level != "debug"),
                                    )
                                except Exception:
                                    pass

                    # Rebuild messages with updated limits (and possibly updated history summary)
                    messages = build_chat_messages(
                        history=history,
                        user_text=user_text_for_model or "",
                        image_tensor=img_tensor,
                        max_turns=turns_limit,
                        summarize_old_history=bool(summarize_old_history),
                        system_prompt=system_prompt or "",
                    )
                    attempts += 1
                    continue

                # Non-context error: fail fast
                print(f"[LLM Session Chat] Error during generation: {e}")
                _log_total("Finished (error)")
                return ("",)

        if not assistant_text:
            print(f"[LLM Session Chat] Error during generation: {last_err}")
            _log_total("Finished (error)")
            return ("",)

        # Update history (do not persist image; keep image_note empty in phase 1)
        history.setdefault("turns", []).append({
            "t": _now_iso_jst(),
            "user": {
                "text": user_text or "",
                "image_note": ""
            },
            "assistant": {
                "text": assistant_text or ""
            },
            # Added: Save parameters used in this execution
            "params": {
                # parameters (required values)
                "max_tokens_req": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repeat_penalty": float(repeat_penalty) if repeat_penalty is not None else None,
                "repeat_last_n": int(repeat_last_n) if repeat_last_n is not None else None,

                # Retained actually used values , if dynamic degeneracy is effective
                "dynamic_max_tokens": bool(dynamic_max_tokens),
                "max_tokens_used": int(gen_tokens),
                "turns_limit_used": int(turns_limit) if turns_limit is not None else None,

                # Record the use of images
                "image_used": (image is not None),
                "streamed": bool(stream_to_console),
            }
        })
        history.setdefault("meta", {})["updated_at"] = _now_iso_jst()
        # Save only the latest execution parameters (overwrite each time)
        history.setdefault("meta", {})["last_params"] = {
            # Runtime context/load system
            "prompt_cache_mode": (prompt_cache_mode or "off"),
            "kv_state_mode": (kv_state_mode or "off"),

            # History control
            "max_turns": int(max_turns) if max_turns is not None else None,
            "summarize_old_history": bool(summarize_old_history),
            "summary_chunk_turns": int(summary_chunk_turns),
            "max_tokens_summary": int(max_tokens_summary),
            "summary_max_chars": int(summary_max_chars),

            "saved_at": _now_iso_jst(),
        }
        history["system_prompt"] = system_prompt or history.get("system_prompt", "")

        # Summarize overflow + trim turns (if enabled)
        if summarize_old_history and max_turns is not None:
            try:
                _t_sum = time.perf_counter()
                history = maybe_summarize_history(
                    model=llm,
                    history=history,
                    max_turns=int(max_turns),
                    summarize_old_history=bool(summarize_old_history),
                    summary_chunk_turns=int(summary_chunk_turns),
                    temperature=0.2,
                    max_tokens_summary=int(max_tokens_summary),
                    summary_max_chars=int(summary_max_chars),
                    suppress_logs=(log_level != "debug"),
                )
                _dt_sum = time.perf_counter() - _t_sum
                if log_level != "minimal":
                    print(f"[LLM Session Chat] Summarization step finished in {_dt_sum:.2f} seconds")
            except Exception as e:
                _dt_sum = time.perf_counter() - _t_sum
                if log_level != "minimal":
                    print(f"[LLM Session Chat] Summarization step failed in {_dt_sum:.2f} seconds: {e}")

                # Never fail the turn if summarization breaks
                pass

        # Save history
        try:
            _atomic_write_json(hist_path, history)
        except Exception as e:
            print(f"[LLM Session Chat] Warning: failed to save history: {e}")

        
        # Save in-memory KV/state for next turn (best-effort)
        if (kv_state_mode or "off").lower() == "memory" and image is None:
            try:
                if hasattr(llm, "save_state"):
                    _turns_ctx2 = (history.get("turns") or [])
                    _mt2 = int(max_turns) if (max_turns is not None) else None
                    if _mt2 is not None and _mt2 >= 0:
                        _turns_ctx2 = _turns_ctx2[-_mt2:] if _mt2 > 0 else []
                    _summary_txt2 = ""
                    if bool(summarize_old_history) and history.get("summary", {}).get("enabled", False):
                        _summary_txt2 = history.get("summary", {}).get("text", "") or ""
                    _effective_system2 = (system_prompt or history.get("system_prompt", "") or "").strip()
                    _kv_prefix_material2 = json.dumps({
                        "model_path": os.path.abspath(model_path) if model_path else "",
                        "mmproj_path": os.path.abspath(mmproj_path) if mmproj_path else "",
                        "n_ctx": int(n_ctx) if n_ctx is not None else None,
                        "n_gpu_layers": int(n_gpu_layers) if n_gpu_layers is not None else None,
                        "system": _effective_system2,
                        "summary": _summary_txt2,
                        "turns": _turns_ctx2,
                    }, ensure_ascii=False, sort_keys=True)
                    _sig2 = hashlib.sha256(_kv_prefix_material2.encode("utf-8")).hexdigest()
                    _MEM_KV_STATE[session_id] = {"signature": _sig2, "state": llm.save_state()}
                    if log_level == "debug":
                        print("[LLM Session Chat] KV state: SAVED (memory)")
            except Exception as e:
                if log_level == "debug":
                    print(f"[LLM Session Chat] KV state save skipped: {e}")
        _log_total("Finished")

        return (assistant_text,)


# =============================================================================
# LLM Dialogue Cycle - single node A<->B loop (no graph cycle)
# =============================================================================

def _transcript_path(session_id: str, history_dir: Optional[str] = None) -> str:
    """Return transcript file path: {session_id}.txt under the same base as history files."""
    session_id = (session_id or "default").strip()
    safe = "".join(c for c in session_id if c.isalnum() or c in ("-", "_", "."))
    if not safe:
        safe = "default"
    base = (history_dir or "").strip() or default_sessions_dir()
    _ensure_dir(base)
    return os.path.join(base, f"{safe}.txt")

def _append_transcript_lines(path: str, lines: List[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")

def _resolve_model_and_mmproj(roots: list[str], model: str, mmproj: str) -> tuple[str, Optional[str]]:
    model_path = _resolve_llm_relpath(model, roots=roots)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    mmproj_path = None
    if mmproj == "(Not required)":
        mmproj_path = "(Not required)"  # sentinel
    elif mmproj != "(Auto-detect)":
        mmproj_path = _resolve_llm_relpath(mmproj, roots=roots)
        if not os.path.exists(mmproj_path):
            mmproj_path = None  # fall back to auto-detect

    return model_path, mmproj_path

def _chat_one_turn(
    *,
    user_text: str,
    session_id: str,
    model: str,
    mmproj: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    n_ctx: int,
    max_turns: int = 12,
    summarize_old_history: bool = True,
    summary_chunk_turns: int = 3,
    max_tokens_summary: int = 128,
    summary_max_chars: int = 1500,
    dynamic_max_tokens: bool = True,
    min_generation_tokens: int = 96,
    safety_margin_tokens: int = 64,
    prompt_cache_mode: str = "disk",
    repeat_penalty: float = 1.12,
    repeat_last_n: int = 256,
    rewrite_continue: bool = True,
    kv_state_mode: str = "memory",
    log_level: str = "timing",
    suppress_backend_logs: bool = True,
    history_dir: str = "",
    reset_session: bool = False,
    stream_to_console: bool = False,
) -> str:
    """
    One chat turn using the same history/summary logic as LLMSessionChatNode,
    but returns assistant_text only.
    """
    global _model_manager

    if not LLAMA_CPP_AVAILABLE:
        msg = "llama_cpp is not available"
        if "_LLAMA_CPP_IMPORT_ERROR" in globals():
            msg += f" ({_LLAMA_CPP_IMPORT_ERROR})"
        raise RuntimeError(msg)

    if "(No GGUF models found" in model:
        return ""

    roots = _get_llm_model_roots()
    try:
        model_path, mmproj_path = _resolve_model_and_mmproj(roots, model, mmproj)
    except Exception:
        return ""

    model_sig = {
        "model_file": os.path.basename(model_path),
        "mmproj_file": os.path.basename(mmproj_path) if (mmproj_path and mmproj_path != "(Not required)") else "",
        "n_ctx": int(n_ctx),
        "n_gpu_layers": int(n_gpu_layers),
    }

    history, hist_path = load_history(
        session_id=session_id,
        history_dir=(history_dir or None),
        system_prompt=system_prompt,
        model_sig=model_sig,
        reset_session=bool(reset_session),
    )

    # Continue rewrite (same behavior as LLMSessionChatNode)
    _is_continue = bool(re.match(r"^\s*continue\b", (user_text or ""), flags=re.IGNORECASE))
    user_text_for_model = user_text or ""
    if rewrite_continue and _is_continue:
        detected_lang = _detect_history_language(history)
        CONTINUE_PROMPTS = {
            "zh": (
                "从最后一个完整的句子继续。"
                "不要重复之前的短语、列表、标题或段落。"
                "如果你开始重复词语或地名，立即停止并写一个一句话摘要。"
            ),
            "ja": (
                "最後の完全な文から続けてください。"
                "以前のフレーズ、リスト、見出し、セクションを繰り返さないでください。"
                "単語や地名を繰り返し始めたら、すぐに停止して1文の要約を書いてください。"
            ),
            "en": (
                "Continue from the last complete sentence. "
                "Do not repeat previous phrases, lists, headings, or sections. "
                "If you start repeating words or place names, stop immediately and write a 1-sentence summary instead."
            ),
        }
        user_text_for_model = CONTINUE_PROMPTS.get(
            detected_lang,
            "Continue from the last complete sentence. Do not repeat previous content."
        )
        if log_level == "debug":
            print(f"[LLM Dialogue Cycle] Continue detected, language: {detected_lang}")

    # Store prompt-cache under the same directory as the history file
    try:
        _model_manager.prompt_cache_dir_override = os.path.join(os.path.dirname(hist_path), "prompt_cache")
        os.makedirs(_model_manager.prompt_cache_dir_override, exist_ok=True)
    except Exception:
        _model_manager.prompt_cache_dir_override = None

    # Load model
    if _model_manager is None:
        _model_manager = GGUFModelManager()

    try:
        llm = _model_manager.load_model(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=int(n_ctx),
            n_gpu_layers=int(n_gpu_layers),
            verbose=False,
        )
    except Exception:
        return ""

    # Best-effort prompt/KV cache
    try:
        _model_manager.configure_prompt_cache(
            llm,
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=int(n_ctx),
            cache_mode=prompt_cache_mode,
        )
    except Exception:
        pass

    # Build messages (no image support in cycle node by design)
    messages = build_chat_messages(
        history=history,
        user_text=user_text_for_model or "",
        image_tensor=None,
        max_turns=int(max_turns) if max_turns is not None else None,
        summarize_old_history=bool(summarize_old_history),
        system_prompt=system_prompt or "",
    )

    # In-memory KV/state cache (best-effort). This can speed up long-running sessions.
    if (kv_state_mode or "off").lower() == "memory":
        try:
            _turns_ctx = (history.get("turns") or [])
            _mt = int(max_turns) if (max_turns is not None) else None
            if _mt is not None and _mt >= 0:
                _turns_ctx = _turns_ctx[-_mt:] if _mt > 0 else []
            _summary_txt = ""
            if bool(summarize_old_history) and history.get("summary", {}).get("enabled", False):
                _summary_txt = history.get("summary", {}).get("text", "") or ""
            _effective_system = (system_prompt or history.get("system_prompt", "") or "").strip()
            _kv_prefix_material = json.dumps({
                "model_path": os.path.abspath(model_path) if model_path else "",
                "mmproj_path": os.path.abspath(mmproj_path) if mmproj_path else "",
                "n_ctx": int(n_ctx) if n_ctx is not None else None,
                "n_gpu_layers": int(n_gpu_layers) if n_gpu_layers is not None else None,
                "system": _effective_system,
                "summary": _summary_txt,
                "turns": _turns_ctx,
            }, ensure_ascii=False, sort_keys=True)
            _kv_sig = hashlib.sha256(_kv_prefix_material.encode("utf-8")).hexdigest()
            entry = _MEM_KV_STATE.get(session_id)
            if entry and entry.get("signature") == _kv_sig and hasattr(llm, "load_state"):
                llm.load_state(entry.get("state"))
                if log_level != "minimal":
                    print("[LLM Dialogue Cycle] KV state: HIT (memory)")
            else:
                if log_level != "minimal":
                    reason = "no state" if not entry else "prefix changed"
                    print(f"[LLM Dialogue Cycle] KV state: MISS ({reason})")
        except Exception as e:
            if log_level == "debug":
                print(f"[LLM Dialogue Cycle] KV state: DISABLED ({e})")

    # Generate with dynamic fallback on n_ctx overflow (same pattern)
    assistant_text = ""
    gen_tokens = int(max_tokens)
    turns_limit = int(max_turns) if max_turns is not None else None

    def _is_ctx_error(err: Exception) -> bool:
        s = str(err)
        return ("exceeds n_ctx" in s) or ("Prompt exceeds n_ctx" in s) or ("n_ctx" in s and "exceed" in s)

    attempts = 0
    last_err = None
    while attempts < 6:
        try:
            _kwargs = {
                "messages": messages,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(gen_tokens),
            }
            if repeat_last_n and int(repeat_last_n) > 0:
                _kwargs["repeat_last_n"] = int(repeat_last_n)
            if repeat_penalty and float(repeat_penalty) != 1.0:
                _kwargs["repeat_penalty"] = float(repeat_penalty)

            with _make_suppress_backend_logs(bool(suppress_backend_logs) and (log_level != "debug")):
                if stream_to_console:
                    pieces: List[str] = []
                    out = sys.__stdout__
                    try:
                        stream_iter = _iter_chat_completion_robust(llm, messages, **{k:v for k,v in _kwargs.items() if k!='messages'})
                    except TypeError:
                        _kwargs.pop("repeat_last_n", None)
                        _kwargs.pop("repeat_penalty", None)
                        _kwargs.pop("top_p", None)
                        stream_iter = _iter_chat_completion_robust(llm, messages, **{k:v for k,v in _kwargs.items() if k!='messages'})
                    for chunk in stream_iter:
                        token = _extract_stream_content(chunk)
                        if not token:
                            continue
                        pieces.append(token)
                        try:
                            out.write(token)
                            out.flush()
                        except Exception:
                            pass
                    assistant_text = "".join(pieces).strip()
                else:
                    try:
                        resp = _create_chat_completion_robust(llm, messages, **{k:v for k,v in _kwargs.items() if k!='messages'})
                    except TypeError:
                        _kwargs.pop("repeat_last_n", None)
                        _kwargs.pop("repeat_penalty", None)
                        _kwargs.pop("top_p", None)
                        resp = _create_chat_completion_robust(llm, messages, **{k:v for k,v in _kwargs.items() if k!='messages'})

                    assistant_text = (resp["choices"][0]["message"]["content"] or "").strip()
            break
        except Exception as e:
            last_err = e
            if dynamic_max_tokens and _is_ctx_error(e):
                # 1) shrink max_tokens
                if gen_tokens > int(min_generation_tokens):
                    gen_tokens = max(int(min_generation_tokens), gen_tokens // 2)
                else:
                    # 2) reduce live turns
                    if turns_limit is not None and turns_limit > 0:
                        turns_limit = max(0, turns_limit - 1)
                    else:
                        # 3) compact rolling summary
                        if summarize_old_history:
                            try:
                                history = maybe_compact_summary(
                                    model=llm,
                                    history=history,
                                    summary_max_chars=int(summary_max_chars),
                                    temperature=0.2,
                                    max_tokens_summary=int(max_tokens_summary),
                                    suppress_logs=(log_level != "debug"),
                                )
                            except Exception:
                                pass

                messages = build_chat_messages(
                    history=history,
                    user_text=user_text_for_model or "",
                    image_tensor=None,
                    max_turns=turns_limit,
                    summarize_old_history=bool(summarize_old_history),
                    system_prompt=system_prompt or "",
                )
                attempts += 1
                continue
            return ""

    if not assistant_text:
        if log_level == "debug":
            print(f"[LLM Dialogue Cycle] generation failed: {last_err}")
        return ""

    # Update history
    history.setdefault("turns", []).append({
        "t": _now_iso_jst(),
        "user": {"text": user_text or "", "image_note": ""},
        "assistant": {"text": assistant_text or ""},   
    })
    history.setdefault("meta", {})["updated_at"] = _now_iso_jst()
    # Save only the latest execution parameters (overwrite each time)
    history.setdefault("meta", {})["last_params"] = {
        # parameters (required values)
        "max_tokens_req": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repeat_penalty": float(repeat_penalty) if repeat_penalty is not None else None,
        "repeat_last_n": int(repeat_last_n) if repeat_last_n is not None else None,

        # Runtime context/load system
        "prompt_cache_mode": (prompt_cache_mode or "off"),
        "kv_state_mode": (kv_state_mode or "off"),

        # History control
        "max_turns": int(max_turns) if max_turns is not None else None,
        "summarize_old_history": bool(summarize_old_history),
        "summary_chunk_turns": int(summary_chunk_turns),
        "max_tokens_summary": int(max_tokens_summary),
        "summary_max_chars": int(summary_max_chars),

        # Retained actually used values , if dynamic degeneracy is effective
        "dynamic_max_tokens": bool(dynamic_max_tokens),
        "max_tokens_used": int(gen_tokens),
        "turns_limit_used": int(turns_limit) if turns_limit is not None else None,

        "saved_at": _now_iso_jst(),
    }
    history["system_prompt"] = system_prompt or history.get("system_prompt", "")

    # Summarize overflow
    if summarize_old_history and max_turns is not None:
        try:
            history = maybe_summarize_history(
                model=llm,
                history=history,
                max_turns=int(max_turns),
                summarize_old_history=bool(summarize_old_history),
                summary_chunk_turns=int(summary_chunk_turns),
                temperature=0.2,
                max_tokens_summary=int(max_tokens_summary),
                summary_max_chars=int(summary_max_chars),
                suppress_logs=(log_level != "debug"),
            )
        except Exception:
            pass

    try:
        _atomic_write_json(hist_path, history)
    except Exception:
        pass

    # Save in-memory KV/state for next turn (best-effort)
    if (kv_state_mode or "off").lower() == "memory":
        try:
            if hasattr(llm, "save_state"):
                _turns_ctx2 = (history.get("turns") or [])
                _mt2 = int(max_turns) if (max_turns is not None) else None
                if _mt2 is not None and _mt2 >= 0:
                    _turns_ctx2 = _turns_ctx2[-_mt2:] if _mt2 > 0 else []
                _summary_txt2 = ""
                if bool(summarize_old_history) and history.get("summary", {}).get("enabled", False):
                    _summary_txt2 = history.get("summary", {}).get("text", "") or ""
                _effective_system2 = (system_prompt or history.get("system_prompt", "") or "").strip()
                _kv_prefix_material2 = json.dumps({
                    "model_path": os.path.abspath(model_path) if model_path else "",
                    "mmproj_path": os.path.abspath(mmproj_path) if mmproj_path else "",
                    "n_ctx": int(n_ctx) if n_ctx is not None else None,
                    "n_gpu_layers": int(n_gpu_layers) if n_gpu_layers is not None else None,
                    "system": _effective_system2,
                    "summary": _summary_txt2,
                    "turns": _turns_ctx2,
                }, ensure_ascii=False, sort_keys=True)
                _sig2 = hashlib.sha256(_kv_prefix_material2.encode("utf-8")).hexdigest()
                _MEM_KV_STATE[session_id] = {"signature": _sig2, "state": llm.save_state()}
                if log_level != "minimal":
                    print("[LLM Dialogue Cycle] KV state: SAVED (memory)")
            else:
                if log_level != "minimal":
                    print("[LLM Dialogue Cycle] KV state: UNSUPPORTED (no save_state)")
        except Exception as e:
            if log_level == "debug":
                print(f"[LLM Dialogue Cycle] KV state save skipped: {e}")

    return assistant_text


class LLMDialogueCycleNode:
    """
    LLM Dialogue Cycle
    - Run A->B->A->B... inside one node.
    - Separate histories: {session_id}_A and {session_id}_B
    - Save full transcript to {session_id}.txt (append)
    """

    @classmethod
    def INPUT_TYPES(cls):
        roots = _get_llm_model_roots()
        available_models, available_mmprojs = _list_gguf_recursive_multi(roots)

        mmproj_options = available_mmprojs + ["(Auto-detect)", "(Not required)"]
        if not available_mmprojs:
            mmproj_options = ["(Auto-detect)", "(Not required)"]

        if not available_models:
            available_models = ["(No GGUF models found in models/LLM/)"]

        return {
            "required": {
                "initial_user_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Initial user message (sent to Model A only)"}),
                "session_id": ("STRING", {"default": "default", "tooltip": "Base session id. A uses {id}_A, B uses {id}_B, transcript uses {id}.txt"}),
                "cycles": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1, "tooltip": "Number of round trips. 1 = A then B."}),

                "modelA": (available_models, {"default": available_models[0], "tooltip": "GGUF model for role A"}),
                "mmprojA": (mmproj_options, {"default": "(Auto-detect)", "tooltip": "mmproj for modelA"}),

                "modelB": (available_models, {"default": available_models[0], "tooltip": "GGUF model for role B"}),
                "mmprojB": (mmproj_options, {"default": "(Auto-detect)", "tooltip": "mmproj for modelB"}),

                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant.", "tooltip": "Shared system prompt for both roles"}),
                "system_prompt_A": ("STRING", {"multiline": True, "default": "", "tooltip": "Role-specific system prompt for model A (overrides shared prompt if set)"}),
                "system_prompt_B": ("STRING", {"multiline": True, "default": "", "tooltip": "Role-specific system prompt for model B (overrides shared prompt if set)"}),

                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 1.0, "step": 0.01}),
                "n_gpu_layers": ("INT", {"default": 0, "min": -1, "max": 200, "step": 1}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 131072, "step": 256}),
            },
            "optional": {
                "max_turns": ("INT", {"default": 12, "min": 0, "max": 200}),
                "summarize_old_history": ("BOOLEAN", {"default": True}),
                "summary_chunk_turns": ("INT", {"default": 3, "min": 1, "max": 50}),
                "max_tokens_summary": ("INT", {"default": 128, "min": 16, "max": 2048}),
                "summary_max_chars": ("INT", {"default": 1500, "min": 200, "max": 20000}),

                "dynamic_max_tokens": ("BOOLEAN", {"default": True}),
                "min_generation_tokens": ("INT", {"default": 96, "min": 1, "max": 4096}),
                "safety_margin_tokens": ("INT", {"default": 64, "min": 0, "max": 2048}),

                "prompt_cache_mode": (["disk", "memory", "off"], {"default": "disk"}),
                "kv_state_mode": (["memory", "off"], {"default": "memory"}),

                "repeat_penalty": ("FLOAT", {"default": 1.12, "min": 1.0, "max": 2.0, "step": 0.01}),
                "repeat_last_n": ("INT", {"default": 256, "min": 0, "max": 4096}),
                "rewrite_continue": ("BOOLEAN", {"default": True}),
                "log_level": (["minimal", "timing", "debug"], {"default": "timing"}),
                "suppress_backend_logs": ("BOOLEAN", {"default": True}),

                "history_dir": ("STRING", {"default": "", "tooltip": "Optional history directory. Empty => output/llm_session_sessions/"}),
                "reset_session": ("BOOLEAN", {"default": False, "tooltip": "If true, resets both {id}_A and {id}_B histories (transcript file is not deleted)."}),
                "stream_to_console": ("BOOLEAN", {"default": False, "tooltip": "Stream tokens to console while generating."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcript_text",)
    FUNCTION = "chat_cycle"
    CATEGORY = "LLM/Session"
    DESCRIPTION = "Run a chat cycle between two local GGUF models inside one node (A<->B), saving transcript."

    def chat_cycle(
        self,
        initial_user_text: str,
        session_id: str,
        cycles: int,
        modelA: str,
        mmprojA: str,
        modelB: str,
        mmprojB: str,
        system_prompt: str,
        system_prompt_A: str,
        system_prompt_B: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        n_gpu_layers: int,
        n_ctx: int,
        max_turns: int = 12,
        summarize_old_history: bool = True,
        summary_chunk_turns: int = 3,
        max_tokens_summary: int = 128,
        summary_max_chars: int = 1500,
        dynamic_max_tokens: bool = True,
        min_generation_tokens: int = 96,
        safety_margin_tokens: int = 64,
        prompt_cache_mode: str = "disk",
        kv_state_mode: str = "memory",
        repeat_penalty: float = 1.12,
        repeat_last_n: int = 256,
        rewrite_continue: bool = True,
        log_level: str = "timing",
        suppress_backend_logs: bool = True,
        history_dir: str = "",
        reset_session: bool = False,
        stream_to_console: bool = False,
    ) -> tuple:
        base_id = (session_id or "default").strip() or "default"
        sidA = f"{base_id}_A"
        sidB = f"{base_id}_B"

        # transcript file
        tpath = _transcript_path(base_id, history_dir or None)

        reset = bool(reset_session)

        lastA = ""
        lastB = ""
        transcript_lines: List[str] = []

        first = initial_user_text or ""
        line0 = f"[{_now_iso_jst()}] USER → A: {first}"
        _append_transcript_lines(tpath, [line0])
        transcript_lines.append(line0)

        msg = first
        for _ in range(int(max(1, cycles))):
            # A turn
            sysA = (system_prompt_A or "").strip() or (system_prompt or "")
            lastA = _chat_one_turn(
                user_text=msg,
                session_id=sidA,
                model=modelA,
                mmproj=mmprojA,
                system_prompt=sysA,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                max_turns=max_turns,
                summarize_old_history=summarize_old_history,
                summary_chunk_turns=summary_chunk_turns,
                max_tokens_summary=max_tokens_summary,
                summary_max_chars=summary_max_chars,
                dynamic_max_tokens=dynamic_max_tokens,
                min_generation_tokens=min_generation_tokens,
                safety_margin_tokens=safety_margin_tokens,
                prompt_cache_mode=prompt_cache_mode,
                repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n,
                rewrite_continue=rewrite_continue,
                kv_state_mode=kv_state_mode,
                log_level=log_level,
                suppress_backend_logs=suppress_backend_logs,
                history_dir=history_dir,
                reset_session=reset,
                stream_to_console=bool(stream_to_console),
            )
            lineA = f"[{_now_iso_jst()}] A: {lastA}"
            _append_transcript_lines(tpath, [lineA])
            transcript_lines.append(lineA)
            msg = lastA or ""
            reset = False  # reset only on first use

            # B turn
            sysB = (system_prompt_B or "").strip() or (system_prompt or "")
            lastB = _chat_one_turn(
                user_text=msg,
                session_id=sidB,
                model=modelB,
                mmproj=mmprojB,
                system_prompt=sysB,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                max_turns=max_turns,
                summarize_old_history=summarize_old_history,
                summary_chunk_turns=summary_chunk_turns,
                max_tokens_summary=max_tokens_summary,
                summary_max_chars=summary_max_chars,
                dynamic_max_tokens=dynamic_max_tokens,
                min_generation_tokens=min_generation_tokens,
                safety_margin_tokens=safety_margin_tokens,
                prompt_cache_mode=prompt_cache_mode,
                repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n,
                rewrite_continue=rewrite_continue,
                kv_state_mode=kv_state_mode,
                log_level=log_level,
                suppress_backend_logs=suppress_backend_logs,
                history_dir=history_dir,
                reset_session=False,
                stream_to_console=bool(stream_to_console),
            )
            lineB = f"[{_now_iso_jst()}] B: {lastB}"
            _append_transcript_lines(tpath, [lineB])
            transcript_lines.append(lineB)
            msg = lastB or ""

        return ("\n".join(transcript_lines),)


# ============================================================================
# ComfyUI Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LLMSessionChatSimpleNode": LLMSessionChatSimpleNode,
    "LLMDialogueCycleSimpleNode": LLMDialogueCycleSimpleNode,
    "LLMSessionChatNode": LLMSessionChatNode,
    "LLMDialogueCycleNode": LLMDialogueCycleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMSessionChatSimpleNode": "LLM Session Chat (Simple)",
    "LLMDialogueCycleSimpleNode": "LLM Dialogue Cycle (Simple)",
    "LLMSessionChatNode": "LLM Session Chat",
    "LLMDialogueCycleNode": "LLM Dialogue Cycle",
}


# ============================================================================
# Cleanup on module unload
# ============================================================================

def cleanup():
    """Cleanup on module unload"""
    global _model_manager
    if _model_manager is not None:
        _model_manager.unload_model()

import atexit
atexit.register(cleanup)
