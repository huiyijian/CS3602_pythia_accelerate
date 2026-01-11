import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from datasets import load_dataset
from tqdm import tqdm
import gc
import sys

# 屏蔽烦人日志
from transformers import logging
logging.set_verbosity_error()

# ================= 配置区域 =================
DEVICE = "cuda:0"
TARGET_MODEL_ID = "EleutherAI/pythia-2.8b"

DTYPES = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16
}

NUM_SAMPLES = 15      
PPL_SAMPLES = 20      
MAX_NEW_TOKENS = 200  
TARGET_PARAMS = 2.8e9 

# ================= 工具函数 =================
def setup_env():
    torch.set_default_device(DEVICE)
    try:
        import seaborn as sns
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

def get_data(n):
    print("📚 Loading Wikitext...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return [x['text'] for x in ds if len(x['text']) > 200][:n]

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ================= PPL 计算 =================
def calculate_ppl(model, tokenizer, texts):
    nlls = []
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
            input_ids = encodings.input_ids
            target_ids = input_ids.clone()
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# ================= 核心测试逻辑 =================
def run_benchmark(dtype_name, torch_dtype, prompts_speed, prompts_ppl):
    print(f"\n🔄 Benchmarking {dtype_name}...")
    
    # 1. 初始化
    cleanup()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_ID, torch_dtype=torch_dtype, device_map=DEVICE
        )
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return None

    stats = {"ttft": [], "tpot": [], "throughput": [], "vram": 0.0, "ppl": 0.0}

    # 2. 测 PPL
    print(f"   📉 Calculating PPL...", end="\r")
    stats["ppl"] = calculate_ppl(model, tokenizer, prompts_ppl)
    print(f"   ✅ PPL: {stats['ppl']:.4f}")
    
    torch.cuda.reset_peak_memory_stats()

    # 3. 测速度
    print(f"   🚀 Measuring Speed & VRAM...", end="\r")
    
    dummy = tokenizer("Warmup", return_tensors="pt").to(DEVICE)
    model.generate(**dummy, max_new_tokens=5)
    
    for text in tqdm(prompts_speed, desc="Gen", leave=False, ncols=80):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, 
            pad_token_id=tokenizer.eos_token_id, streamer=streamer
        )
        
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        t_start = time.perf_counter()
        thread.start()
        
        try:
            first_token = next(iter(streamer))
            t_first = time.perf_counter()
        except StopIteration:
            t_first = time.perf_counter()
            first_token = ""
            
        res = first_token
        for t in streamer: res += t
        thread.join()
        t_end = time.perf_counter()
        
        n_tokens = len(tokenizer.encode(res, add_special_tokens=False))
        stats["ttft"].append(t_first - t_start)
        total_time = t_end - t_start
        if total_time > 0:
            stats["throughput"].append(n_tokens / total_time)
        if n_tokens > 1:
            stats["tpot"].append(((t_end - t_start) - (t_first - t_start)) / (n_tokens - 1))
        else:
            stats["tpot"].append(total_time)

    # 4. 获取显存峰值
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    stats["vram"] = peak_memory
    print(f"   💾 Peak VRAM: {peak_memory:.2f} GB")

    del model
    cleanup()
    
    return {
        "ttft": np.mean(stats["ttft"]),
        "tpot": np.mean(stats["tpot"]),
        "throughput": np.mean(stats["throughput"]),
        "vram": stats["vram"],
        "ppl": stats["ppl"]
    }

# ================= 修改后的绘图逻辑 (2行布局) =================
def plot_results(results):
    labels = list(results.keys())
    
    data = {
        "throughput": [results[l]['throughput'] for l in labels],
        "tpot": [results[l]['tpot'] * 1000 for l in labels],
        "ttft": [results[l]['ttft'] * 1000 for l in labels],
        "ppl": [results[l]['ppl'] for l in labels],
        "vram": [results[l]['vram'] for l in labels]
    }

    # === 创建 2行3列 的布局 ===
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    colors = ['#ff9999', '#66b3ff', '#99ff99'] # FP32, FP16, BF16
    
    # 定义每张图的位置和内容
    # (行索引, 列索引, 标题, Y轴标签, 数据)
    plot_configs = [
        # 第一行：速度指标
        (0, 0, "Throughput\n(Higher is Better)", "Tokens / Sec", data["throughput"]),
        (0, 1, "TPOT / Latency\n(Lower is Better)", "Milliseconds", data["tpot"]),
        (0, 2, "TTFT / Prefill\n(Lower is Better)", "Milliseconds", data["ttft"]),
        # 第二行：质量和资源
        (1, 0, "Perplexity / Quality\n(Lower is Better)", "PPL Score", data["ppl"]),
        (1, 1, "Peak VRAM Usage\n(Lower is Better)", "Memory (GB)", data["vram"])
    ]

    # 循环绘制前5张图
    for (row, col, title, ylabel, vals) in plot_configs:
        ax = axes[row, col]
        bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel)
        
        # 动态调整 Y 轴
        if "Perplexity" in title:
            min_v, max_v = min(vals), max(vals)
            ax.set_ylim(min_v * 0.95, max_v * 1.05)
        elif "VRAM" in title:
            ax.set_ylim(0, max(vals) * 1.3)
            
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            label_text = f"{height:.2f}"
            if "GB" in title: label_text += " GB"
            elif "Milli" in ylabel: label_text += " ms"
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label_text, ha='center', va='bottom', fontweight='bold')

    # === 隐藏第2行第3列的空图 (因为我们只有5个指标) ===
    axes[1, 2].axis('off')

    plt.suptitle(f"Full Precision Benchmark: Pythia 2.8B (Speed vs Quality vs Memory)", fontsize=20, y=1.02)
    plt.tight_layout()
    filename = "precision_benchmark_2rows.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ 绘图完成！(两行布局): {filename}")

# ================= 主程序 =================
if __name__ == "__main__":
    setup_env()
    cleanup()
    
    all_prompts = get_data(max(NUM_SAMPLES, PPL_SAMPLES))
    prompts_speed = all_prompts[:NUM_SAMPLES]
    prompts_ppl = all_prompts[:PPL_SAMPLES]
    
    results = {}
    
    for name, dtype in DTYPES.items():
        res = run_benchmark(name, dtype, prompts_speed, prompts_ppl)
        if res:
            results[name] = res
            
    if results:
        plot_results(results)