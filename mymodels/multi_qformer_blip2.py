# mymodels/multi_qformer_blip2.py
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, List, Tuple
from transformers import Blip2ForConditionalGeneration, Blip2Config
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2QFormerModel,
    Blip2ForConditionalGenerationModelOutput,
)

#https://huggingface.co/docs/transformers/model_doc/blip-2#transformers.Blip2Model.forward
#https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L1590

# Canonical view order & aliases
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


class MultiQformerBlip2ForConditionalGeneration(Blip2ForConditionalGeneration):
    """
    BLIP-2 variant with:
      - Shared vision encoder
      - Per-view QFormers + per-view query tokens
      - Per-view language projections
      - Fixed slots: AP, PA, LAT (always 3 * num_query_tokens placeholders)
      - Repeated images per view: QFormer run per image, token-wise mean-pool
      - Missing views: visual block zeroed out by a view mask
      - Special tokens are handled at the tokenizer / collate level, not here.
    """
    config: Blip2Config

    def __init__(self, config: Blip2Config):
        # Builds standard BLIP-2, including original qformer/query_tokens/language_projection
        super().__init__(config)

        # Per-view branches
        self.qformer_by_view = nn.ModuleDict()
        self.language_projection_by_view = nn.ModuleDict()
        self.query_tokens_by_view = nn.ParameterDict()

        q_hidden = config.qformer_config.hidden_size
        num_q = config.num_query_tokens

        for v in VIEW_KEYS:
            self.qformer_by_view[v] = Blip2QFormerModel._from_config(config.qformer_config)
            self.language_projection_by_view[v] = nn.Linear(q_hidden, config.text_config.hidden_size)
            self.query_tokens_by_view[v] = nn.Parameter(torch.zeros(1, num_q, q_hidden))

        # Original single-branch modules (self.qformer, self.query_tokens, self.language_projection)
        # remain present so HF can load pretrained weights into them. We then copy those weights
        # into the per-view branches in load_state_dict.

    # ---------------- weight loading hook (accepts HF kwargs like assign) ----------------
    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """
        Accepts extra kwargs like `assign` used by HF during sharded loading,
        forwards them to super(), then copies pretrained weights into AP/PA/LAT branches.
        """
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)

        with torch.no_grad():
            # Copy QFormer to each view
            if hasattr(self, "qformer"):
                src_qformer_sd = self.qformer.state_dict()
                for v in VIEW_KEYS:
                    self.qformer_by_view[v].load_state_dict(src_qformer_sd)

            # Copy query tokens to each view
            if hasattr(self, "query_tokens"):
                for v in VIEW_KEYS:
                    self.query_tokens_by_view[v].copy_(self.query_tokens)

            # Copy language projection to each view
            if hasattr(self, "language_projection"):
                src_proj_sd = self.language_projection.state_dict()
                for v in VIEW_KEYS:
                    self.language_projection_by_view[v].load_state_dict(src_proj_sd)

        return result
    # ----------------------------------------------------------------

    def _encode_images(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool,
    ) -> torch.FloatTensor:
        """
        pixel_values: (B, 3, H, W)
        returns: image_embeds (B, seq_len, vis_hidden)
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )
        return vision_outputs[0]

    def _qformer_per_view(
        self,
        view: str,
        image_embeds_list: List[torch.FloatTensor],
        dtype_out: torch.dtype,
    ) -> torch.FloatTensor:
        """
        Run QFormer for one view across possible repeated images. Mean-pool across repeats.
        image_embeds_list: list of tensors, each (1, seq, vis_hidden)
        returns: (1, num_query_tokens, q_hidden) in dtype_out
        """
        qformer = self.qformer_by_view[view]
        query_tokens = self.query_tokens_by_view[view]

        outputs_per_repeat = []
        for img_embeds in image_embeds_list:
            img_mask = torch.ones(img_embeds.size()[:-1], dtype=torch.long, device=img_embeds.device)
            q_tokens = query_tokens.expand(img_embeds.shape[0], -1, -1)  # (1, Q, q_hidden)

            q_out = qformer(
                query_embeds=q_tokens,
                encoder_hidden_states=img_embeds,
                encoder_attention_mask=img_mask,
                return_dict=True,
            ).last_hidden_state  # (1, Q, q_hidden)
            outputs_per_repeat.append(q_out)

        if len(outputs_per_repeat) == 0:
            # Return zeros if this view is absent (will also be masked later)
            zeros = torch.zeros(
                (1, self.config.num_query_tokens, self.config.qformer_config.hidden_size),
                device=query_tokens.device,
                dtype=query_tokens.dtype,
            )
            return zeros.to(dtype_out)

        if len(outputs_per_repeat) == 1:
            pooled = outputs_per_repeat[0]
        else:
            pooled = torch.stack(outputs_per_repeat, dim=0).mean(dim=0)  # (1, Q, q_hidden)

        if pooled.dtype != dtype_out:
            pooled = pooled.to(dtype_out)

        return pooled

    def _prepare_inputs_embeds_and_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.LongTensor]:
        """
        Ensure we have inputs_embeds, attention_mask and the mask over <image> tokens (expanded to embed dim).
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        special_image_mask = (input_ids == self.config.image_token_id)
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return inputs_embeds, special_image_mask, attention_mask

    # ---------- forward ----------
    def forward(
        self,
        multi_pixel_values: List[Dict[str, List[torch.FloatTensor]]],
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs,
    ) -> Union[tuple, Blip2ForConditionalGenerationModelOutput]:

        dtype_out = self.get_input_embeddings().weight.dtype
        per_sample_proj = []

        # Build visual prefix per sample
        for sample_dict in multi_pixel_values:
            per_view_q = {}
            view_presence: Dict[str, float] = {}

            for v in VIEW_KEYS:
                img_list = sample_dict.get(v, [])
                img_embeds_list = []
                for img_tensor in img_list:
                    img_tensor_b = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
                    vis = self._encode_images(
                        pixel_values=img_tensor_b,
                        interpolate_pos_encoding=interpolate_pos_encoding,
                    )  # (1, seq, vis_hidden)
                    img_embeds_list.append(vis)

                present = len(img_embeds_list) > 0
                view_presence[v] = 1.0 if present else 0.0

                per_view_q[v] = self._qformer_per_view(v, img_embeds_list, dtype_out)

            # Project per-view and apply view mask (zero out blocks for missing views)
            proj_outs = []
            for v in VIEW_KEYS:
                q = per_view_q[v]  # (1, Q, q_hidden)
                proj = self.language_projection_by_view[v](q)  # (1, Q, lm_hidden)
                if proj.dtype != dtype_out:
                    proj = proj.to(dtype_out)

                mask_val = view_presence[v]
                if mask_val == 0.0:
                    proj = proj * 0.0
                proj_outs.append(proj)

            proj_cat = torch.cat(proj_outs, dim=1)  # (1, 3Q, lm_hidden)
            per_sample_proj.append(proj_cat)

        language_model_inputs = torch.cat(per_sample_proj, dim=0)  # (B, 3Q, lm_hidden)

        # Prepare text embeddings and masks
        inputs_embeds, special_image_mask, attention_mask = self._prepare_inputs_embeds_and_mask(
            input_ids, inputs_embeds, attention_mask
        )

        # Replace <image> token positions by visual prefix
        language_model_inputs = language_model_inputs.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.to(language_model_inputs.device).masked_scatter(
            special_image_mask, language_model_inputs
        )

        # LM forward (same as parent)
        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )
            logits = outputs[0]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                loss_fct = nn.CrossEntropyLoss(reduction="mean")
                loss = loss_fct(
                    shift_logits.view(-1, self.config.text_config.vocab_size),
                    shift_labels.view(-1),
                )
        else:
            kwargs["return_dict"] = True
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                **kwargs,
            )
            loss = outputs.loss
            logits = outputs.logits

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=None,
            qformer_outputs=None,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        multi_pixel_values: List[Dict[str, List[torch.FloatTensor]]],
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if hasattr(self, "hf_device_map"):
            self._preprocess_accelerate()

        dtype_out = self.get_input_embeddings().weight.dtype
        per_sample_proj = []

        for sample_dict in multi_pixel_values:
            per_view_q = {}
            view_presence: Dict[str, float] = {}

            for v in VIEW_KEYS:
                img_list = sample_dict.get(v, [])
                img_embeds_list = []
                for img_tensor in img_list:
                    img_tensor_b = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
                    vis = self._encode_images(
                        pixel_values=img_tensor_b,
                        interpolate_pos_encoding=interpolate_pos_encoding,
                    )
                    img_embeds_list.append(vis)

                present = len(img_embeds_list) > 0
                view_presence[v] = 1.0 if present else 0.0

                per_view_q[v] = self._qformer_per_view(v, img_embeds_list, dtype_out)

            proj_outs = []
            for v in VIEW_KEYS:
                q = per_view_q[v]
                proj = self.language_projection_by_view[v](q)
                if proj.dtype != dtype_out:
                    proj = proj.to(dtype_out)

                mask_val = view_presence[v]
                if mask_val == 0.0:
                    proj = proj * 0.0
                proj_outs.append(proj)

            proj_cat = torch.cat(proj_outs, dim=1)  # (1, 3Q, lm_hidden)
            per_sample_proj.append(proj_cat)

        language_model_inputs = torch.cat(per_sample_proj, dim=0)  # (B, 3Q, lm_hidden)

        # Prepare text side
        if inputs_embeds is None:
            if input_ids is None:
                # If no text prompt is given, use 3Q <image> tokens + BOS as in original BLIP-2
                Q = self.config.num_query_tokens
                image_tokens = [self.config.image_token_id] * (3 * Q)
                start_tokens = image_tokens + [self.config.text_config.bos_token_id]
                input_ids = torch.tensor([start_tokens], dtype=torch.long, device=language_model_inputs.device)
                input_ids = input_ids.repeat(language_model_inputs.shape[0], 1)
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Image mask
        special_image_mask = (input_ids == self.config.image_token_id)
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        language_model_inputs = language_model_inputs.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, language_model_inputs)

        inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        if not self.language_model.config.is_encoder_decoder:
            inputs["input_ids"] = input_ids

        return self.language_model.generate(**inputs, **generate_kwargs)
