# =====  0-shot inference  =====
python scripts/infer_on_idefics.py \
    --idefics-config-path configs/idefics_config-9b_vanilla.yaml \
    --inference-config-path configs/9b_idefics_lora/9b_idefics_lora-inference.yaml \
    --prompts "https://datasets-server.huggingface.co/assets/jmhessel/newyorker_caption_contest/--/explanation/train/0/image/image.jpg" \
    --prompts "Question: How is this picture uncanny? Answer: "

# Result:
# ```
# Question: How is this picture uncanny? Answer: It’s a snowman with a face.
# 
# I’m not sure if this is a joke or not, but I’m going to go with it being a joke.
# 
# I’m not sure if this is
# ```


# =====  1-shot inference (from https://huggingface.co/datasets/jmhessel/newyorker_caption_contest/viewer/explanation/train)  =====
python scripts/infer_on_idefics.py \
    --idefics-config-path configs/idefics_config-9b_vanilla.yaml \
    --inference-config-path configs/9b_idefics_lora/9b_idefics_lora-inference.yaml \
    --prompts "User:" \
    --prompts "https://datasets-server.huggingface.co/cached-assets/jmhessel/newyorker_caption_contest/--/explanation/train/1/image/image.jpg" \
    --prompts "How is this picture uncanny?\nAssistant: It is unusual to see such a giant book." \
    --prompts "User:" \
    --prompts "https://datasets-server.huggingface.co/assets/jmhessel/newyorker_caption_contest/--/explanation/train/1/image/image.jpg" \
    --prompts "How is this picture uncanny?\n"

# Result:
# ```
# User: How is this picture uncanny?\nAssistant: It is unusual to see such a giant book.User: How is this picture uncanny?\nAssistant: It is unusual to see such a giant book.User:
# 
# The Uncanny Valley is a concept in the field of robotics and human-computer interaction. It refers to the unsettling feeling some people experience when
# ```