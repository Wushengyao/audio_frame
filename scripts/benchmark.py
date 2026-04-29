#!/usr/bin/env python3
"""Benchmark VoxCPM2 generation speed: CPU vs GPU."""
from pathlib import Path
import time, sys, os

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from voxcpm import VoxCPM

TEST_TEXTS = [
    # Short - single segment
    "肚子里传来一阵咕噜声，她这才想起自己已经一天没吃东西了。",
    # Medium - paragraph
    "窗外雨声淅淅沥沥，她坐在书桌前，手指轻轻敲击着键盘。屏幕上的光标一闪一闪，"
    "像极了此刻她忐忑不安的心。她深吸一口气，开始写下那封酝酿已久的信。",
    # Long - multi-sentence
    "夜幕降临，城市的灯火依次亮起。他站在天台上，俯瞰着这片他生活了十年的土地。"
    "风拂过他的发梢，带着初秋的凉意。手机在口袋里震动，是她的消息，只有简单的三个字：我想你。"
    "他笑了笑，转身下楼，决定今晚就飞过去给她一个惊喜。",
]


def benchmark(device: str, label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Loading VoxCPM2 on {label} ({device}) ...")
    t0 = time.time()

    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        optimize=(device != "cpu"),
        device=device,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    results = []
    for i, text in enumerate(TEST_TEXTS):
        t0 = time.time()
        wav = model.generate(text=text, inference_timesteps=10)
        gen_time = time.time() - t0
        duration = len(wav) / model.tts_model.sample_rate
        rtf = gen_time / duration if duration > 0 else float("inf")
        print(f"  [{i+1}/{len(TEST_TEXTS)}] {len(text)} chars -> {duration:.1f}s audio in {gen_time:.1f}s (RTF={rtf:.1f}x)")
        results.append({"text_len": len(text), "audio_sec": duration, "gen_sec": gen_time, "rtf": rtf})

    del model
    if device != "cpu":
        import torch
        torch.cuda.empty_cache()

    return {"label": label, "load_s": load_time, "segments": results}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-only", action="store_true")
    args = parser.parse_args()

    all_results = []

    if not args.cpu_only:
        all_results.append(benchmark("cuda", "GPU"))

    if not args.gpu_only:
        all_results.append(benchmark("cpu", "CPU"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n--- {r['label']} ---")
        print(f"  Model load: {r['load_s']:.1f}s")
        for seg in r["segments"]:
            print(f"  {seg['text_len']} chars -> {seg['audio_sec']:.1f}s audio, {seg['gen_sec']:.1f}s gen, RTF={seg['rtf']:.1f}x")

    if len(all_results) == 2:
        gpu, cpu = all_results
        print(f"\n--- Speedup ---")
        print(f"  Load time:  CPU={cpu['load_s']:.0f}s / GPU={gpu['load_s']:.0f}s = {cpu['load_s']/gpu['load_s']:.1f}x")
        for i in range(len(gpu["segments"])):
            cpu_rtf = cpu["segments"][i]["rtf"]
            gpu_rtf = gpu["segments"][i]["rtf"]
            print(f"  Segment {i+1}: CPU RTF={cpu_rtf:.1f}x / GPU RTF={gpu_rtf:.1f}x = {cpu_rtf/gpu_rtf:.1f}x faster on GPU")


if __name__ == "__main__":
    main()
