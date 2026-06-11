# -*- coding: utf-8 -*-
"""
Qwen3-VL reader (component)

Reads a layer region crop and returns its transcription together with a
token-logprob confidence, and describes noise crops in prose. The model is loaded
once with device_map="auto" so accelerate places/shards the model across
whatever GPUs CUDA_VISIBLE_DEVICES exposes (one GPU, or several). Attention
defaults to "sdpa" (flash-attn is optional). The detection/indexing workflow
drives this; this module is import-only (no executable)

NOTE: the Qwen3-VL weights (~16 GB for the 7B variant in bf16) are downloaded on
first use from Hugging Face. Actually constructing QwenVLM requires:
  pip install -e ".[detection]"  (torch>=2.6, transformers>=4.57, qwen-vl-utils>=0.0.14)
and a GPU with enough VRAM (7B in bf16 needs ~16 GB - a cluster GPU)

Confidence: greedy decoding (do_sample=False) means each step's chosen-token
probability is the model's confidence; we average them (geometric mean via
logprobs_to_confidence) into a per-region transcription confidence in [0, 1]

API changes from Qwen2.5-VL:
  - Class: Qwen2_5_VLForConditionalGeneration -> Qwen3VLForConditionalGeneration
  - Input: processor.apply_chat_template(tokenize=True, return_dict=True)
    then inputs.pop("token_type_ids", None) before generate
  - No process_vision_info needed for PIL image inputs
"""

import re
import torch
import os

from src.core.confidence import logprobs_to_confidence, group_tokens_to_words
from src.detection.vlm.prompts import PROMPT_TRANSCRIBE, PROMPT_NOISE

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _strip_prompt_leak(text, instruction):
    """
    Remove any fragment of the instruction that leaked into the model output

    Qwen3-VL occasionally echoes part of the prompt when a crop contains little
    or no text. We detect leaks by looking for distinctive substrings of the
    instruction (case-insensitive) and stripping lines that contain them
    """
    # Build fingerprint phrases from the instruction (skip short words)
    words = instruction.lower().split()
    # Use consecutive 4-word windows as fingerprints
    fingerprints = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]

    lines = text.split("\n")
    clean = []
    for line in lines:
        line_low = line.lower()
        if any(fp in line_low for fp in fingerprints):
            continue
        clean.append(line)

    return "\n".join(clean).strip()


class QwenVLM:
    """
    Lazy-loaded Qwen3-VL reader for region transcription and noise description
    """

    def __init__(self, model_id=DEFAULT_MODEL_ID, device_map="auto",
                 dtype=torch.bfloat16, attn_implementation="sdpa"):
        """
        Load the VLM once in full bfloat16

        device_map='auto' lets accelerate span visible GPUs - set CUDA_VISIBLE_DEVICES
        before calling to control which GPUs are exposed. Qwen3-VL-8B in bf16 needs
        ~16 GB and fits comfortably on the L40S (48 GB), so no quantization is used

        Args:
            model_id (str): Hugging Face id of the Qwen3-VL model
            device_map (str): 'auto' or a torch device string
            dtype (dtype): bf16 by default
            attn_implementation (str): 'sdpa' or 'flash_attention_2'
        """
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    def _run(self, image, instruction, max_new_tokens=512):
        """
        Run one image and instruction and return transcription with confidence

        Args:
            image (PIL.Image): the region crop to analyse
            instruction (str): the task prompt
            max_new_tokens (int): generation budget
        Returns:
            Tuple[str, float, list]: text, confidence, and word_confidences
        """
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": instruction},
        ]}]

        # apply_chat_template with tokenize=True handles image encoding internally
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )

        in_len   = inputs.input_ids.shape[1]
        out_ids  = gen.sequences[:, in_len:]
        out_text = self.processor.batch_decode(
            out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        out_text = _strip_prompt_leak(out_text, instruction)

        transition = self.model.compute_transition_scores(
            gen.sequences, gen.scores, normalize_logits=True)
        token_logprobs = [float(s) for s in transition[0].tolist()]
        words = self._word_confidences(out_ids[0].tolist(), token_logprobs)
        return out_text, logprobs_to_confidence(token_logprobs), words

    def _word_confidences(self, out_ids, token_logprobs):
        """
        Aggregate per-token logprobs into per-word confidences
        """
        tok = self.processor.tokenizer
        special = set(getattr(tok, "all_special_ids", []) or [])
        kept = [(tid, lp) for tid, lp in zip(out_ids, token_logprobs) if tid not in special]
        if not kept:
            return []
        ids  = [t for t, _ in kept]
        lps  = [lp for _, lp in kept]
        toks = tok.convert_ids_to_tokens(ids)
        words = []
        for group in group_tokens_to_words(toks):
            text = tok.convert_tokens_to_string([toks[i] for i in group]).strip()
            if not text:
                continue
            words.append({"text": text,
                          "confidence": logprobs_to_confidence([lps[i] for i in group])})
        return words

    def transcribe_region(self, crop_image, max_new_tokens=512):
        """
        Transcribe one printed or handwritten region crop
        """
        return self._run(crop_image, PROMPT_TRANSCRIBE, max_new_tokens)

    def describe_noise(self, crop_image, max_new_tokens=256):
        """
        Describe one noise crop or layer in prose
        """
        text, conf, _ = self._run(crop_image, PROMPT_NOISE, max_new_tokens)
        return text, conf