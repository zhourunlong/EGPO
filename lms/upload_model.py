import os

from utils import load_model_tokenizer

model, tokenizer = load_model_tokenizer("/local1/vectorzh/experiments/sft_models/Qwen2.5-1.5B-Instruct-SFT-OpenHermes-2.5-Standard-SFT-20250602-002609/checkpoint-9780")

hf_username = os.environ.get("HF_USR_NAME", "")
if hf_username != "":
    hf_token = os.environ.get("HF_TOKEN", "")
    hub_path = hf_username + "/" + "Qwen2.5-1.5B-Instruct-SFT-OpenHermes-2.5-Standard-SFT"
    print("Pushing model to hub at " + hub_path)
    model.push_to_hub(hub_path, use_auth_token=hf_token)
    tokenizer.push_to_hub(hub_path, use_auth_token=hf_token)
    print("Model pushed to hub.")
else:
    print("HF_USR_NAME not set, skipping model upload to Hugging Face Hub.")