# =====  0-shot inference  =====
python scripts/infer_on_idefics.py \
    --idefics-config-path configs/9b_idefics_lora/9b_idefics_lora-model.yaml \
    --inference-config-path configs/9b_idefics_lora/9b_idefics_lora-inference.yaml \
    --prompts "https://datasets-server.huggingface.co/assets/jmhessel/newyorker_caption_contest/--/explanation/train/0/image/image.jpg" \
    --prompts "Question: How is this picture uncanny? Answer: "

# Result:
# ```
# ```


# =====  1-shot inference (from https://huggingface.co/datasets/jmhessel/newyorker_caption_contest/viewer/explanation/train)  =====
python scripts/infer_on_idefics.py \
    --idefics-config-path configs/9b_idefics_lora/9b_idefics_lora-model.yaml \
    --inference-config-path configs/9b_idefics_lora/9b_idefics_lora-inference.yaml \
    --prompts "User:" \
    --prompts "https://datasets-server.huggingface.co/cached-assets/jmhessel/newyorker_caption_contest/--/explanation/train/1/image/image.jpg" \
    --prompts "How is this picture uncanny?\nAssistant: It is unusual to see such a giant book." \
    --prompts "User:" \
    --prompts "https://datasets-server.huggingface.co/assets/jmhessel/newyorker_caption_contest/--/explanation/train/1/image/image.jpg" \
    --prompts "How is this picture uncanny?\n"

# Result:
# [WIP]
