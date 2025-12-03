# mydatasets/llamamedvqa_dataset.py
import os
from typing import Optional, Any, Dict, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from paths import IMAGES_MIMIC_PATH

# Include [Q] in the prompt now
VQA_PROMPT_TEMPLATE = "[Q] Question: {q} Answer:"

VIEW_KEYS = ("AP", "PA", "LAT")
VIEW_ALIASES = {
    "AP": {"ap"},
    "PA": {"pa"},
    "LAT": {"lat", "lateral", "ll", "l"},
}

def _canon_view_name(s: str) -> Optional[str]:
    s = (s or "").strip().lower()
    for k, al in VIEW_ALIASES.items():
        if s in al:
            return k
    return None


class LlamaMedVQADataset(Dataset):
    """
    Multi-view VQA dataset:
      - csv columns: image_path (possibly comma/semicolon-separated list), question, answer,
                     optional view_position (aligned list).
      - __getitem__ returns grouped PIL images per view, plus texts.
      - build_collate_fn(processor, num_query_tokens, image_token_id, view_token_ids) creates:
          * multi_pixel_values: list[dict(view -> list[tensor(3,H,W)])]
          * input_ids, attention_mask with per-view special tokens + image tokens:
                [AP] Q*<image> [PA] Q*<image> [LAT] Q*<image>  + tokenized prompt (incl. [Q])
          * labels from answers (pad -> -100)
    """
    def __init__(self, csv_path: str, image_root: Optional[str] = None, prompt_template: str = VQA_PROMPT_TEMPLATE):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root or IMAGES_MIMIC_PATH
        self.prompt_template = prompt_template
        self.df = self.df.dropna(subset=["image_path", "question", "answer"]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _split_csv_field(val: Optional[str]) -> List[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        s = str(val).strip().strip('"').strip("'")
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip() != ""]
        return parts if parts else []

    def _resolve_image_path(self, rel_or_abs: str) -> str:
        return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(self.image_root, rel_or_abs)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_cells = self._split_csv_field(row["image_path"])
        views = self._split_csv_field(row.get("view_position", None))

        per_view_images: Dict[str, List[Image.Image]] = {k: [] for k in VIEW_KEYS}
        per_view_paths:  Dict[str, List[str]] = {k: [] for k in VIEW_KEYS}

        for i, p in enumerate(img_cells):
            vp = views[i] if i < len(views) else ""
            v = _canon_view_name(vp)
            if v is None:
                continue
            abspath = self._resolve_image_path(p)
            try:
                img = Image.open(abspath).convert("RGB")
                per_view_images[v].append(img)
                per_view_paths[v].append(p)
            except Exception:
                continue

        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        prompt = self.prompt_template.format(q=question)

        return {
            "per_view_images": per_view_images,
            "per_view_paths": per_view_paths,
            "prompt": prompt,
            "answer": answer,
            "question": question,
        }

    # ---------- collate ----------
    def build_collate_fn(
        self,
        processor: Any,
        num_query_tokens: int,
        image_token_id: int,
        view_token_ids: Dict[str, int],
    ):
        """
        Returns a callable that:
          - builds multi_pixel_values as a list[dict(view -> list[tensor(3,H,W)])]
          - prepends per-view prefixes:
                [AP] Q*<image> [PA] Q*<image> [LAT] Q*<image>
          - tokenizes prompts (which already contain [Q])
          - tokenizes answers as labels (pad->-100)
        """
        tok = processor.tokenizer
        Q = int(num_query_tokens)
        # For each view: 1 special token + Q image tokens
        VIEW_BLOCK_LEN = 1 + Q
        PREFIX_LEN = 3 * VIEW_BLOCK_LEN  # AP + PA + LAT

        def _images_to_pixel_tensors(imgs: List[Image.Image]) -> List[torch.FloatTensor]:
            px = []
            for im in imgs:
                enc = processor(images=im, return_tensors="pt")
                px.append(enc["pixel_values"][0])  # (3, H, W)
            return px

        def _collate(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            B = len(examples)

            multi_pixel_values: List[Dict[str, List[torch.FloatTensor]]] = []
            all_questions, all_prompts, all_answers = [], [], []
            all_paths: List[List[str]] = []

            for ex in examples:
                per_view_imgs = ex["per_view_images"]
                sample_dict = {}
                sample_paths = []
                for v in VIEW_KEYS:
                    px_list = _images_to_pixel_tensors(per_view_imgs.get(v, []))
                    sample_dict[v] = px_list
                    sample_paths.extend(ex["per_view_paths"].get(v, []))
                multi_pixel_values.append(sample_dict)
                all_questions.append(ex["question"])
                all_prompts.append(ex["prompt"])
                all_answers.append(ex["answer"])
                all_paths.append(sample_paths if sample_paths else [""])

            # 1) Tokenize text prompts (which include [Q])
            txt = tok(
                all_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = txt.input_ids       # (B, L_text)
            text_attention = txt.attention_mask  # (B, L_text)

            # 2) Build fixed per-view prefix:
            #    [AP] Q*<image> [PA] Q*<image> [LAT] Q*<image>
            prefix_ids_list = []
            for _ in range(B):
                prefix_ids = []
                for v in VIEW_KEYS:
                    prefix_ids.append(view_token_ids[v])  # special view token
                    prefix_ids += [image_token_id] * Q    # Q image placeholders
                prefix_ids_list.append(prefix_ids)

            prefix_ids = torch.tensor(prefix_ids_list, dtype=text_input_ids.dtype)  # (B, PREFIX_LEN)
            prefix_attention = torch.ones_like(prefix_ids)                          # all 1s

            # 3) Concatenate prefix + text
            input_ids = torch.cat([prefix_ids, text_input_ids], dim=1)             # (B, PREFIX_LEN + L_text)
            attention_mask = torch.cat([prefix_attention, text_attention], dim=1)  # same shape

            # 4) Decoder labels from answers (no change)
            tgt = tok(all_answers, padding=True, truncation=True, return_tensors="pt")
            labels = tgt.input_ids
            pad_id = tok.pad_token_id
            if pad_id is None:
                tok.pad_token = tok.eos_token
                pad_id = tok.pad_token_id
            labels[labels == pad_id] = -100

            batch = {
                "multi_pixel_values": multi_pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "references": all_answers,
                "prompts": all_prompts,
                "questions": all_questions,
                "image_paths": [";".join(p) for p in all_paths],
            }
            return batch

        return _collate
