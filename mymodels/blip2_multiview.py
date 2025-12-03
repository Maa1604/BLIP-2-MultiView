# mymodels/blip2_padchestgr.py
import torch
from transformers import Blip2Processor
from .multi_qformer_blip2 import MultiQformerBlip2ForConditionalGeneration, VIEW_KEYS
from .model_utility import count_parameters, save_parameter_info

def build_model_and_processor():
    """
    Loads Multi-View BLIP-2 (per-view QFormers) + processor.
    Adds special tokens [AP], [PA], [LAT], [Q] to tokenizer and resizes LM embeddings.
    Returns: (model, processor, view_token_ids)
    """
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

    # Load model (this will load pretrained BLIP-2 weights into base modules)
    model = MultiQformerBlip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # ---- Add special tokens to tokenizer vocabulary ----
    special_tokens = ["[AP]", "[PA]", "[LAT]", "[Q]"]
    added = processor.tokenizer.add_tokens(special_tokens)
    if added > 0:
        # Resize LM embeddings to accommodate added tokens
        model.resize_token_embeddings(len(processor.tokenizer))

    # Map tokens to IDs for use in collate
    token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)
    view_token_ids = {
        "AP": token_ids[0],
        "PA": token_ids[1],
        "LAT": token_ids[2],
        "Q": token_ids[3],
    }

    # ---- Freeze everything by default ----
    model.requires_grad_(False)

    # Explicitly freeze original single-branch modules
    if hasattr(model, "qformer"):
        for p in model.qformer.parameters():
            p.requires_grad = False
    if hasattr(model, "language_projection"):
        for p in model.language_projection.parameters():
            p.requires_grad = False
    if hasattr(model, "query_tokens"):
        model.query_tokens.requires_grad_(False)

    # ---- Unfreeze exactly what you trained before, now x3 (each view) ----
    for v in VIEW_KEYS:
        # 1) Entire QFormer branch
        for _, p in model.qformer_by_view[v].named_parameters():
            p.requires_grad = True

        # 2) Per-view language projection
        for p in model.language_projection_by_view[v].parameters():
            p.requires_grad = True

        # 3) Per-view query tokens
        model.query_tokens_by_view[v].requires_grad_(True)

    count_parameters(model)
    save_parameter_info(model, output_file="blip2_parameters.txt")

    # Return view_token_ids so dataset collate can use [AP], [PA], [LAT], [Q]
    return model, processor, view_token_ids
