import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from datasets import load_dataset
from tqdm import tqdm
import sys

# === 关键修正：强力屏蔽 Transformers 的啰嗦日志 ===
from transformers import logging
logging.set_verbosity_error()

# ================= 配置区域 =================
DEVICE = "cuda:0"
TARGET_MODEL_ID = "EleutherAI/pythia-2.8b"

# 对比组配置
MODELS_CONFIG = {
    "70M": {
        "id": "EleutherAI/pythia-70m",
        "color": "#d62728",  # 红色
        "marker": "o"
    },
    "160M": {
        "id": "EleutherAI/pythia-160m",
        "color": "#1f77b4",  # 蓝色
        "marker": "s"        # 方块
    }
}

# 1. K值取 3 到 10
K_VALUES = list(range(3, 11))

NUM_SAMPLES = 15  # 样本数
MAX_NEW_TOKENS = 200

# ================= 工具类 =================
class ForwardCounter:
    def __init__(self): self.count = 0
    def __call__(self, m, i, o): self.count += 1

def setup_env():
    torch.set_default_device(DEVICE)
    try:
        import seaborn as sns
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

def get_data(n):
    print(f"📚 Loading Wikitext-2...")
    try:
        # Wikitext 加载通常很快，简单的加载提示即可
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        candidates = [x['text'] for x in ds if len(x['text']) > 200]
        print(f"✅ Loaded {len(candidates)} candidates, selecting first {n}.")
        return candidates[:n]
    except Exception as e:
        print(f"❌ Data load failed: {e}")
        return []

# ================= 核心测试逻辑 =================
def run_benchmark(target_model, draft_model, tokenizer, prompts, k_val, desc_prefix=""):
    """
    运行测试并返回 (throughput, acceptance_rate)
    """
    total_tokens = 0
    total_time = 0
    total_steps = 0

    # 预热 (静默)
    dummy = tokenizer("Warmup", return_tensors="pt").to(DEVICE)
    if draft_model and k_val:
        target_model.generate(**dummy, assistant_model=draft_model, max_new_tokens=5, num_assistant_tokens=k_val)
    else:
        target_model.generate(**dummy, max_new_tokens=5)
    torch.cuda.synchronize()

    # === 优化点：使用 tqdm 显示进度，并输出到 stdout ===
    pbar = tqdm(prompts, desc=desc_prefix, leave=True, file=sys.stdout, ncols=100)

    for text in pbar:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        
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

        # === 优化点：实时显示当前速度 ===
        cur_speed = n_tok / (t1 - t0)
        pbar.set_postfix({"CurSpeed": f"{cur_speed:.1f}t/s"})

    avg_tp = total_tokens / total_time
    
    # 健壮的接收率计算公式
    avg_acc = 0.0
    if k_val:
        tps = total_tokens / total_steps
        avg_acc = (tps - 1) / k_val
        avg_acc = min(1.0, max(0.0, avg_acc))

    return avg_tp, avg_acc

# ================= 专门优化的绘图函数 =================
def plot_comparison(results, k_vals, baseline):
    # 创建 1 行 2 列的布局，宽屏显示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- 图1：吞吐量 (Throughput) ---
    ax1.set_title("Throughput Comparison (Wikitext)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Num Assistant Tokens (K)", fontsize=12)
    ax1.set_ylabel("Throughput (tokens/s)", fontsize=12)
    
    # 绘制 Baseline (虚线)
    ax1.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline ({baseline:.1f} t/s)')
    
    # 收集数据以确定 Y 轴范围
    all_tp_values = [baseline]
    
    for label, data in results.items():
        cfg = MODELS_CONFIG[label]
        tp_data = data['throughput']
        all_tp_values.extend(tp_data)
        
        # 绘制曲线
        ax1.plot(k_vals, tp_data, 
                 color=cfg['color'], marker=cfg['marker'], linewidth=2.5, markersize=8, 
                 label=f'{label} Draft')
        
        # 添加填充效果
        ax1.fill_between(k_vals, baseline, tp_data, color=cfg['color'], alpha=0.1)

    # === 关键修改：动态缩放 Y 轴，让差距更明显 ===
    min_y = min(all_tp_values)
    max_y = max(all_tp_values)
    margin = (max_y - min_y) * 0.1 # 上下留 10% 的余量
    y_bottom = min(baseline, min_y) - margin
    y_top = max_y + margin
    ax1.set_ylim(y_bottom, y_top)
    
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 图2：接收率 (Acceptance Rate) ---
    ax2.set_title("Acceptance Rate Comparison (Wikitext)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Num Assistant Tokens (K)", fontsize=12)
    ax2.set_ylabel("Acceptance Rate", fontsize=12)
    
    for label, data in results.items():
        cfg = MODELS_CONFIG[label]
        acc_data = data['acc_rate']
        
        ax2.plot(k_vals, acc_data, 
                 color=cfg['color'], marker=cfg['marker'], linewidth=2.5, markersize=8,
                 label=f'{label} Draft')

    ax2.set_ylim(0, 1.05) # 接收率固定 0-1
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    # === 关键修改：文件名设置为 wikitext fixed ===
    filename = "speculative_comparison_wikitext.png"
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
    
    # 存储结果容器
    results = {
        "70M": {"throughput": [], "acc_rate": []},
        "160M": {"throughput": [], "acc_rate": []}
    }
    
    # 1. 先测 Baseline (只测一次)
    print("\n🏁 Running Baseline...")
    baseline_speed, _ = run_benchmark(target_model, None, tokenizer, prompts, None, desc_prefix="Baseline")
    print(f"   -> Baseline Speed: {baseline_speed:.2f} t/s")

    # 2. 循环测试两个模型
    for label in ["70M", "160M"]:
        cfg = MODELS_CONFIG[label]
        print(f"\n🧪 Testing {label} Draft ({cfg['id']})...")
        
        draft_model = AutoModelForCausalLM.from_pretrained(cfg['id'], torch_dtype=torch.float16, device_map=DEVICE)
        
        # === 优化点：扁平化循环，不再使用嵌套 tqdm ===
        for k in K_VALUES:
            # 传递 desc_prefix 以获得独立的进度条
            tp, acc = run_benchmark(target_model, draft_model, tokenizer, prompts, k, desc_prefix=f"Step K={k}")
            
            results[label]["throughput"].append(tp)
            results[label]["acc_rate"].append(acc)
            
            # 打印当前 K 的汇总，防止进度条滚走后信息丢失
            print(f"      K={k}: Speed={tp:.1f} t/s | Acc={acc:.2f}")
        
        del draft_model
        torch.cuda.empty_cache()
        print(f"   Done {label}!")

    # 3. 绘图
    plot_comparison(results, K_VALUES, baseline_speed)