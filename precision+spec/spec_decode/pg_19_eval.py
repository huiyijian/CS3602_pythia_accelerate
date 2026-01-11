import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from datasets import load_dataset
from tqdm import tqdm
# === 新增：屏蔽烦人的 transformers 警告 ===
from transformers import logging
logging.set_verbosity_error()

# ================= 配置区域 =================
DEVICE = "cuda:0"
TARGET_MODEL_ID = "EleutherAI/pythia-2.8b"

MODELS_CONFIG = {
    "70M": {
        "id": "EleutherAI/pythia-70m",
        "color": "#d62728",  # 红色
        "marker": "o"
    },
    "160M": {
        "id": "EleutherAI/pythia-160m",
        "color": "#1f77b4",  # 蓝色
        "marker": "s"
    }
}

K_VALUES = list(range(3, 11))
NUM_SAMPLES = 15
MAX_NEW_TOKENS = 200

# ================= 工具类 =================
class ForwardCounter:
    def __init__(self): self.count = 0
    def __call__(self, m, i, o): self.count += 1

def setup_env():
    torch.set_default_device(DEVICE)
    plt.style.use('seaborn-v0_8-whitegrid')

def get_data(n):
    print(f"📚 Loading PG-19 (via emozilla/pg19)...")
    try:
        ds = load_dataset("emozilla/pg19", split="test", streaming=True)
        prompts = []
        iterator = iter(ds)
        
        # 使用 tqdm 显示数据加载进度
        pbar = tqdm(total=n, desc="Fetching Samples", unit="sample")
        while len(prompts) < n:
            try:
                item = next(iterator)
                text = item['text']
                if len(text) > 2000:
                    clean_text = text[:2000]
                    prompts.append(clean_text)
                    pbar.update(1)
            except StopIteration:
                break
        pbar.close()
        return prompts
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        return []

# ================= 核心测试逻辑 =================
def run_benchmark(target_model, draft_model, tokenizer, prompts, k_val, desc=""):
    total_tokens = 0
    total_time = 0
    total_steps = 0

    # 预热 (不显示进度条)
    dummy = tokenizer("Warmup", return_tensors="pt").to(DEVICE)
    if draft_model and k_val:
        target_model.generate(**dummy, assistant_model=draft_model, max_new_tokens=5, num_assistant_tokens=k_val)
    else:
        target_model.generate(**dummy, max_new_tokens=5)
    torch.cuda.synchronize()

    # 正式测试 (使用单一进度条)
    for text in tqdm(prompts, desc=desc, leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1500).to(DEVICE)
        
        counter = ForwardCounter()
        hook = target_model.register_forward_hook(counter)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, 
            pad_token_id=tokenizer.eos_token_id, streamer=streamer
        )
        
        if draft_model and k_val:
            gen_kwargs["assistant_model"] = draft_model
            gen_kwargs["num_assistant_tokens"] = k_val
            
        thread = Thread(target=target_model.generate, kwargs=gen_kwargs)
        
        t0 = time.perf_counter()
        thread.start()
        res = ""
        for t in streamer: res += t
        thread.join()
        t1 = time.perf_counter()
        
        hook.remove()
        
        n_tok = len(tokenizer.encode(res, add_special_tokens=False))
        total_tokens += n_tok
        total_time += (t1 - t0)
        total_steps += max(1, counter.count)

    avg_tp = total_tokens / total_time
    
    avg_acc = 0.0
    if k_val:
        tps = total_tokens / total_steps
        avg_acc = (tps - 1) / k_val
        avg_acc = min(1.0, max(0.0, avg_acc))

    return avg_tp, avg_acc

# ================= 绘图函数 =================
def plot_comparison(results, k_vals, baseline):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 图1：吞吐量
    ax1.set_title("Throughput on PG-19 (Target: 2.8B)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Num Assistant Tokens (K)", fontsize=12)
    ax1.set_ylabel("Throughput (tokens/s)", fontsize=12)
    
    ax1.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline ({baseline:.1f} t/s)')
    
    all_tp_values = [baseline]
    for label, data in results.items():
        cfg = MODELS_CONFIG[label]
        tp_data = data['throughput']
        all_tp_values.extend(tp_data)
        ax1.plot(k_vals, tp_data, color=cfg['color'], marker=cfg['marker'], linewidth=2.5, markersize=8, label=f'{label} Draft')
        ax1.fill_between(k_vals, baseline, tp_data, color=cfg['color'], alpha=0.1)

    min_y = min(all_tp_values)
    max_y = max(all_tp_values)
    margin = (max_y - min_y) * 0.1
    ax1.set_ylim(min(baseline, min_y) - margin, max_y + margin)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 图2：接收率
    ax2.set_title("Acceptance Rate on PG-19", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Num Assistant Tokens (K)", fontsize=12)
    ax2.set_ylabel("Acceptance Rate", fontsize=12)
    
    for label, data in results.items():
        cfg = MODELS_CONFIG[label]
        acc_data = data['acc_rate']
        ax2.plot(k_vals, acc_data, color=cfg['color'], marker=cfg['marker'], linewidth=2.5, markersize=8, label=f'{label} Draft')

    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    filename = "speculative_comparison_pg19_clean.png"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ 绘图完成！图片已保存为: {filename}")

# ================= 主程序 =================
if __name__ == "__main__":
    setup_env()
    
    print(f"🔄 Loading Target: {TARGET_MODEL_ID}...")
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    prompts = get_data(NUM_SAMPLES)
    if not prompts: exit()
    
    results = {
        "70M": {"throughput": [], "acc_rate": []},
        "160M": {"throughput": [], "acc_rate": []}
    }
    
    print("\n🏁 Running Baseline...")
    baseline_speed, _ = run_benchmark(target_model, None, tokenizer, prompts, None, desc="Baseline")
    print(f"   -> Baseline Speed: {baseline_speed:.2f} t/s")

    for label in ["70M", "160M"]:
        cfg = MODELS_CONFIG[label]
        print(f"\n🧪 Testing {label} Draft ({cfg['id']})...")
        
        draft_model = AutoModelForCausalLM.from_pretrained(cfg['id'], torch_dtype=torch.float16, device_map=DEVICE)
        
        # 使用单个进度条显示不同 K 值的测试
        pbar = tqdm(K_VALUES, desc=f"Scanning K (3-10)")
        for k in pbar:
            tp, acc = run_benchmark(target_model, draft_model, tokenizer, prompts, k, desc=f"K={k}")
            results[label]["throughput"].append(tp)
            results[label]["acc_rate"].append(acc)
            # 在进度条右侧实时显示结果
            pbar.set_postfix({"Speed": f"{tp:.1f} t/s", "Acc": f"{acc:.2f}"})
        
        del draft_model
        torch.cuda.empty_cache()

    plot_comparison(results, K_VALUES, baseline_speed)