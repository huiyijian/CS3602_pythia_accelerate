import torch
import torch.profiler
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
from custom_pythia import CustomGPTNeoXModel, GPTNeoXConfig
import gc

def profile_2_8b():
    print("Preparing Profiling for Pythia-2.8B...")
    model_dir = snapshot_download("EleutherAI/pythia-2.8b", revision='master')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use a fixed long input to stress Attention
    # 512 tokens
    seq_len = 512
    input_ids = torch.randint(0, 1000, (1, seq_len)).to(device)
    
    # --- Profile Original ---
    print("\n=== Profiling Original Model ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device).eval()
    
    # Warmup
    print("Warming up (Prefill)...")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        
    print("Running Profiler (Original - Decoding Step)...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False 
    ) as prof:
        with torch.no_grad():
            # Run 5 decoding steps
            for _ in range(5):
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    return # Stop here to see Original results clearly

    # Save weights for transfer before deleting
    original_state_dict = model.state_dict()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # --- Profile Optimized ---
    print("\n=== Profiling Optimized Model ===")
    config = GPTNeoXConfig.from_pretrained(model_dir)
    custom_model = CustomGPTNeoXModel(config).to(dtype=torch.float16).to(device).eval()
    
    # Transfer weights
    print("Transfer weights...")
    new_state_dict = {}
    custom_model_state = custom_model.state_dict()
    for key, value in original_state_dict.items():
        new_key = key.replace("gpt_neox.", "")
        if new_key in custom_model_state:
            new_state_dict[new_key] = value
    custom_model.load_state_dict(new_state_dict, strict=False)
    del original_state_dict
    torch.cuda.empty_cache()
    
    # Warmup
    print("Warming up (Prefill)...")
    with torch.no_grad():
        outputs = custom_model(input_ids, use_cache=True)
        past_key_values = outputs["past_key_values"]
        next_token = outputs["logits"][:, -1, :].argmax(dim=-1).unsqueeze(-1)
        
    print("Running Profiler (Optimized - Decoding Step)...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof_opt:
        with torch.no_grad():
            for _ in range(5):
                outputs = custom_model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs["past_key_values"]
                next_token = outputs["logits"][:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
    print(prof_opt.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    profile_2_8b()
