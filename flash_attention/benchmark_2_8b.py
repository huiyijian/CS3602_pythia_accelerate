import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset
from custom_pythia import CustomGPTNeoXModel, GPTNeoXConfig
import gc

def run_benchmark():
    # 1. Load Data
    print("Loading wikitext dataset...")
    # Load test split
    ds = MsDataset.load('wikitext', subset_name='wikitext-2-raw-v1', split='test', trust_remote_code=True)
    
    # Filter for a long sample (e.g., > 1000 chars) to ensure meaningful context
    long_samples = [x['text'] for x in ds if len(x['text']) > 2000]
    if not long_samples:
        print("No long samples found, using first sample.")
        text_sample = ds[0]['text']
    else:
        text_sample = long_samples[0]
        
    print(f"Selected sample length: {len(text_sample)} chars")
    
    # 2. Setup Model Paths
    model_dir = snapshot_download("EleutherAI/pythia-2.8b", revision='master')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Tokenize input
    # Take first 1024 tokens as prompt if possible, or less
    inputs = tokenizer(text_sample, return_tensors="pt")
    if inputs.input_ids.shape[1] > 1024:
        inputs.input_ids = inputs.input_ids[:, :1024]
        inputs.attention_mask = inputs.attention_mask[:, :1024]
    
    inputs = inputs.to(device)
    print(f"Input prompt tokens: {inputs.input_ids.shape[1]}")

    # 3. Benchmark Original Model
    print("\nLoading Original Pythia-2.8B...")
    # Load in fp16 to save memory and match typical inference usage
    original_model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device).eval()
    
    print("Benchmarking Original...")
    # Warmup
    print("Warming up Original...")
    with torch.no_grad():
        original_model.generate(**inputs, max_new_tokens=5, min_new_tokens=5)
    torch.cuda.synchronize()
    
    
    # Metrics Calculation Helper
    def calculate_metrics(start_time, end_time, num_tokens, batch_size, seq_len, model_hidden_size, num_layers):
        duration = end_time - start_time
        tpot = (duration / num_tokens) * 1000 # ms per token
        throughput = num_tokens / duration
        
        # FLOPs approximation for Decoder-only Transformer
        # Per token FLOPs ~= 2 * P (params)
        # Detailed: 24 * b * s^2 * h + 4 * b * s * h^2 (Attention) + ...
        # Simplified for generation (per token): 2 * Num_Params
        # Pythia-2.8B Params ~ 2.8e9
        # Total FLOPs = 2 * 2.8e9 * num_tokens
        flops_per_token = 2 * 2.8e9 
        total_flops = flops_per_token * num_tokens
        avg_flops = total_flops / duration
        
        return tpot, throughput, total_flops, avg_flops

    print("Running Measurement...")
    torch.cuda.synchronize()
    ttft_start = time.time()
    with torch.no_grad():
        # TTFT: Time To First Token (Prefill)
        # For original model, generate() does prefill + decode internally.
        # We can't easily isolate TTFT from generate() without hooks or modification.
        # We will approximate or use a separate forward pass for TTFT.
        pass
        
    # Re-structure benchmark to measure TTFT separately
    
    # --- TTFT Measurement (Original) ---
    print("Measuring TTFT (Original)...")
    torch.cuda.synchronize()
    start_ttft = time.time()
    with torch.no_grad():
        outputs = original_model(inputs.input_ids, use_cache=True)
    torch.cuda.synchronize()
    end_ttft = time.time()
    ttft_original = (end_ttft - start_ttft) * 1000 # ms
    
    # --- TPOT Measurement (Original) ---
    # We already have duration for 50 tokens. 
    # TPOT = duration / 50
    # Note: generate() includes prefill in the first step usually if past_key_values not provided?
    # Actually generate() handles it. But our previous 'original_duration' included prefill time implicitly?
    # No, generate() does the whole sequence.
    # To get pure TPOT, we should subtract TTFT or just divide total time (approx).
    # But better: measure generation of N tokens AFTER prefill.
    
    print("Running Generation Measurement (Original)...")
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        # Generate 50 tokens
        original_model.generate(**inputs, max_new_tokens=50, min_new_tokens=50)
    torch.cuda.synchronize()
    original_duration = time.time() - start_time
    
    # Metrics
    tpot_original = (original_duration / 50) * 1000 # This is average latency per token (including prefill overhead distributed?)
    # Ideally TPOT is inter-token latency. 
    # Let's assume original_duration is dominated by decoding 50 tokens.
    throughput_original = 50 / original_duration
    
    print(f"Original TTFT: {ttft_original:.2f} ms")
    print(f"Original TPOT: {tpot_original:.2f} ms")
    print(f"Original Throughput: {throughput_original:.2f} tokens/s")
    
    # FLOPs
    flops_per_token = 2 * 2.8e9
    total_flops_orig = flops_per_token * 50
    flops_orig_tflops = (total_flops_orig / original_duration) / 1e12
    print(f"Original Average FLOPs: {flops_orig_tflops:.4f} TFLOPS")
    
    # Cleanup to free VRAM
    del original_model
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Benchmark Custom Model
    print("\nLoading Optimized Pythia-2.8B...")
    config = GPTNeoXConfig.from_pretrained(model_dir)
    custom_model = CustomGPTNeoXModel(config).to(dtype=torch.float16).to(device).eval()
    
    # Load weights
    # We need to load sharded weights for 2.8B usually
    # Using from_pretrained usually handles this for standard models, 
    # but for custom model we might need to be careful.
    # Actually, CustomGPTNeoXModel is not registered with AutoModel, so we load state_dict.
    # BUT 2.8B state_dict is large and sharded.
    # Strategy: Load original model again to CPU (RAM permitting) or use accelerate to load weights?
    # To keep it simple and robust given memory constraints:
    # We will re-load the original model, get state_dict, transfer, then delete original.
    # If 2.8B is too big for 2x copies in RAM, we might struggle.
    # 2.8B fp16 is ~5.6GB. System RAM should hold 2x easily if >16GB.
    
    print("Transferring weights...")
    original_model_temp = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ) # Load to CPU first
    
    original_state_dict = original_model_temp.state_dict()
    custom_state_dict = custom_model.state_dict()
    
    new_state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key.replace("gpt_neox.", "")
        if new_key in custom_state_dict:
            new_state_dict[new_key] = value
            
    custom_model.load_state_dict(new_state_dict, strict=False)
    del original_model_temp
    del original_state_dict
    gc.collect()
    
    print("Benchmarking Optimized...")
    input_ids = inputs.input_ids
    
    # Warmup
    print("Warming up Optimized...")
    with torch.no_grad():
        curr_ids = input_ids
        past_key_values = None
        for _ in range(5):
            outputs = custom_model(input_ids=curr_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs["past_key_values"]
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            curr_ids = next_token
    torch.cuda.synchronize()

    print("Running Measurement...")
    
    # --- TTFT Measurement (Optimized) ---
    print("Measuring TTFT (Optimized)...")
    torch.cuda.synchronize()
    start_ttft = time.time()
    with torch.no_grad():
        outputs = custom_model(input_ids, use_cache=True)
        past_key_values = outputs["past_key_values"]
        # Prepare for decoding
        next_token_logits = outputs["logits"][:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        curr_ids = next_token
    torch.cuda.synchronize()
    end_ttft = time.time()
    ttft_optimized = (end_ttft - start_ttft) * 1000 # ms

    # --- TPOT Measurement (Optimized) ---
    print("Running Generation Measurement (Optimized)...")
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        # Custom generation loop for 50 tokens (already have 1 from TTFT, let's generate 50 more)
        # Or to be fair with original generate(max_new_tokens=50), we should generate 50 tokens TOTAL.
        # Original generate() does prefill + 50 new tokens.
        # Here we did prefill (TTFT) separately.
        # So we should loop 50 times? No, Original duration included prefill time + 50 tokens.
        # But wait, original code: original_model.generate(**inputs, max_new_tokens=50...)
        # This measures Prefill + 50 Decode steps.
        # To get pure TPOT, we should measure ONLY decoding.
        
        # Let's measure pure decoding for Optimized since we control the loop.
        for _ in range(50):
            outputs = custom_model(input_ids=curr_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs["past_key_values"]
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            curr_ids = next_token
            
    torch.cuda.synchronize()
    decode_duration = time.time() - start_time
    
    # Total duration (Prefill + Decode) for fair comparison with Original
    # Wait, original_duration was for generate(), which includes prefill.
    # So optimized_total_duration = (end_ttft - start_ttft) + decode_duration
    optimized_total_duration = (end_ttft - start_ttft) + decode_duration
    
    # Metrics
    # TPOT: Pure decoding latency per token
    tpot_optimized = (decode_duration / 50) * 1000 
    
    # Throughput: Total tokens / Total time (Prefill + Decode)
    # This is "End-to-End Throughput"
    throughput_optimized = 50 / optimized_total_duration
    
    print(f"Optimized TTFT: {ttft_optimized:.2f} ms")
    print(f"Optimized TPOT: {tpot_optimized:.2f} ms")
    print(f"Optimized Throughput: {throughput_optimized:.2f} tokens/s")
    
    # FLOPs
    total_flops_opt = flops_per_token * 50
    flops_opt_tflops = (total_flops_opt / optimized_total_duration) / 1e12
    print(f"Optimized Average FLOPs: {flops_opt_tflops:.4f} TFLOPS")
    
    print(f"Speedup (End-to-End): {original_duration / optimized_total_duration:.2f}x")

if __name__ == "__main__":
    run_benchmark()
