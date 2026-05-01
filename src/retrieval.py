"""
retrieval.py — Structured safety report generation using Llama-3.2-3B-Instruct.

No external knowledge base. No retrieval. No RAG.

The model receives ONLY the numerical facts the pipeline already computed:
  - detected event class
  - trajectory feature values (speed, heading change, lateral deviation, etc.)
  - safety label probabilities across all 5 classes

RTX 4060 (8 GB VRAM) notes
---------------------------
Llama-3.2-3B in FP16 requires ~6-7 GB VRAM.  To stay safely within 8 GB:
  - Default: load in 4-bit (BitsAndBytes NF4) → ~1.7 GB VRAM.
  - Fallback: if bitsandbytes is not installed, load on CPU in FP32 (~12 GB RAM).
  - Install bitsandbytes: pip install bitsandbytes  (Windows CUDA supported >=0.41.3)

A strict system prompt acts as the guardrail:
  - Output is forced into a fixed JSON schema.
  - The model is explicitly forbidden from adding information not in the input.
  - Temperature=0.0 makes output deterministic and consistent.

Fallback: if transformers/GPU not available, a deterministic template formatter
is used so the pipeline never crashes during CPU-only testing.
"""

import json
import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

try:
    import bitsandbytes  # noqa: F401
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


# ── model config ──────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = """You are an autonomous vehicle safety analyst generating structured diagnostic reports.

STRICT RULES:
1. Use ONLY the numerical facts provided in the user message.
2. Do NOT invent sensor readings, road conditions, weather, or any detail not given.
3. Do NOT reference other vehicles, pedestrians, or infrastructure unless explicitly stated.
4. Output MUST be valid JSON matching this exact schema — no extra text before or after:

{
  "event_type": "<string: the detected event name>",
  "severity": "<string: Low | Medium | High | Critical>",
  "primary_indicator": "<string: one sentence describing the most significant metric>",
  "secondary_indicators": ["<string>", "<string>"],
  "recommended_action": "<string: one concrete recommended action>",
  "confidence": "<string: Low | Medium | High based on how clearly metrics indicate the event>"
}

Severity mapping (use ONLY the provided metric values to decide):
- Critical : max_speed > 4.0 OR (max_speed > 3.0 AND max_lateral_dev > 2.0)
- High     : max_heading_change > 1.2 OR max_lateral_dev > 1.5
- Medium   : max_heading_change > 0.8 OR oscillation_score > 0.25
- Low      : everything else that is not Safe
"""


# ── LLM report generator ──────────────────────────────────────────────────────

class LLMReportGenerator:
    """
    Loads Llama-3.2-3B-Instruct once and generates structured JSON safety reports.
    Uses 4-bit quantization by default to fit within 8 GB VRAM (RTX 4060).
    Falls back to CPU (FP32) if bitsandbytes is not installed.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = None, use_4bit: bool = True):
        self._use_llm = _LLM_AVAILABLE

        if not self._use_llm:
            print("[LLMReportGenerator] transformers not available — using template fallback.")
            return

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[LLMReportGenerator] Loading {model_name} ...")

        load_kwargs = {}

        if device == "cuda" and use_4bit and _BNB_AVAILABLE:
            # 4-bit NF4 quantization — ~1.7 GB VRAM, safe for RTX 4060
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["quantization_config"] = bnb_cfg
            load_kwargs["device_map"] = "auto"
            print("  Mode: 4-bit NF4 quantization (bitsandbytes) — ~1.7 GB VRAM")

        elif device == "cuda" and not _BNB_AVAILABLE:
            # bitsandbytes not installed — load in FP16 on CPU to avoid OOM
            # RTX 4060 has 8 GB; FP16 Llama-3.2-3B needs ~6-7 GB which is tight.
            # Running on CPU with FP32 (~12 GB RAM) is safer.
            self.device = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs["device_map"] = "cpu"
            print("  Mode: CPU FP32 (bitsandbytes not installed — 'pip install bitsandbytes' for GPU)")

        else:
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs["device_map"] = "cpu"
            print("  Mode: CPU FP32")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        print("[LLMReportGenerator] Model ready.")

    # ── prompt construction ───────────────────────────────────────────────────

    def _build_user_message(self, event_name: str, features: dict, class_probs: list) -> str:
        class_names = ["Safe", "SharpTurn", "OscillatoryMotion", "HighSpeedRisk", "NearCollisionRisk"]
        prob_str = ", ".join(f"{n}={p:.3f}" for n, p in zip(class_names, class_probs))
        return (
            f"Generate a safety diagnosis report for the following trajectory prediction:\n\n"
            f"Detected event type   : {event_name}\n"
            f"Max speed             : {features.get('max_speed', 0):.4f} m/s\n"
            f"Mean speed            : {features.get('mean_speed', 0):.4f} m/s\n"
            f"Max heading change    : {features.get('max_heading_change', 0):.4f} rad\n"
            f"Mean heading change   : {features.get('mean_heading_change', 0):.4f} rad\n"
            f"Heading variance      : {features.get('heading_variance', 0):.4f}\n"
            f"Max lateral deviation : {features.get('max_lateral_dev', 0):.4f} m\n"
            f"Mean lateral deviation: {features.get('mean_lateral_dev', 0):.4f} m\n"
            f"Oscillation score     : {features.get('oscillation_score', 0):.4f}\n"
            f"Total distance        : {features.get('total_distance', 0):.4f} m\n"
            f"Final displacement    : {features.get('final_displacement', 0):.4f} m\n"
            f"Class probabilities   : {prob_str}\n\n"
            f"Output the JSON report now:"
        )

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, event_name: str, features: dict, class_probs: list = None) -> dict:
        if class_probs is None:
            class_probs = [0.0] * 5

        if event_name == "Safe":
            return {
                "event_type": "Safe",
                "severity": "None",
                "primary_indicator": "Predicted trajectory shows no safety-critical motion pattern.",
                "secondary_indicators": [],
                "recommended_action": "No intervention required. Continue monitoring.",
                "confidence": "High",
            }

        if not self._use_llm:
            return self._template_fallback(event_name, features)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": self._build_user_message(event_name, features, class_probs)},
        ]

        # transformers 5.x returns BatchEncoding; older versions return a plain tensor
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized["input_ids"].to(self.model.device)
        else:
            input_ids = tokenized.to(self.model.device)

        prompt_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=1.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][prompt_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            # repair truncated output: LLM sometimes stops before the closing }
            if clean.startswith("{") and not clean.endswith("}"):
                clean = clean.rstrip(",\n ") + "\n}"
            return json.loads(clean)
        except json.JSONDecodeError:
            print(f"[LLMReportGenerator] JSON parse failed — template fallback.")
            print(f"  Raw output: {raw[:300]}")
            return self._template_fallback(event_name, features)

    # ── template fallback ─────────────────────────────────────────────────────

    def _template_fallback(self, event_name: str, features: dict) -> dict:
        ms  = features.get("max_speed", 0)
        mhc = features.get("max_heading_change", 0)
        mld = features.get("max_lateral_dev", 0)
        osc = features.get("oscillation_score", 0)

        if ms > 4.0 or (ms > 3.0 and mld > 2.0):
            severity = "Critical"
        elif mhc > 1.2 or mld > 1.5:
            severity = "High"
        elif mhc > 0.8 or osc > 0.25:
            severity = "Medium"
        else:
            severity = "Low"

        return {
            "event_type": event_name,
            "severity": severity,
            "primary_indicator": (
                f"Max speed of {ms:.2f} m/s with heading change of {mhc:.2f} rad "
                f"is consistent with {event_name.lower()} behavior."
            ),
            "secondary_indicators": [
                f"Lateral deviation of {mld:.2f} m observed from expected straight-line path.",
                f"Oscillation score of {osc:.2f} indicates "
                f"{'unstable' if osc > 0.2 else 'moderate'} motion pattern.",
            ],
            "recommended_action": (
                "Apply conservative fallback trajectory and reduce speed to safe threshold."
                if severity in ("High", "Critical")
                else "Monitor trajectory and prepare intervention if pattern persists."
            ),
            "confidence": "High" if ms > 2.0 or mhc > 0.8 else "Medium",
        }


# ── top-level pipeline ────────────────────────────────────────────────────────

class DiagnosisPipeline:
    """Drop-in replacement for the previous RAG-based DiagnosisPipeline."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = None, use_4bit: bool = True):
        self.generator = LLMReportGenerator(model_name=model_name, device=device, use_4bit=use_4bit)

    def run(self, event_name: str, features: dict, class_probs: list = None) -> dict:
        report = self.generator.generate(event_name, features, class_probs or [0.0] * 5)
        return {"event": event_name, "features": features, "report": report}


# ── legacy stubs ──────────────────────────────────────────────────────────────

class SimpleRetriever:
    def __init__(self, documents):
        pass
    def retrieve(self, query, top_k=3):
        return []


def build_diagnosis(pred_label: int, retrieved_docs: list) -> str:
    if pred_label == 0:
        return "Safe — no safety-critical pattern detected."
    return "Unsafe — run DiagnosisPipeline for a full structured report."
