#!/usr/bin/env python3
"""CPU parallel VoxCPM2 benchmark."""
from pathlib import Path
import argparse
import multiprocessing
import os
import queue
import sys
import time
import traceback

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

TEXT = "夜幕降临，城市的灯火依次亮起。他站在天台上，俯瞰着这片他生活了十年的土地。风拂过他的发梢，带着初秋的凉意。"
MODEL_ID = "openbmb/VoxCPM2"


def worker(worker_id: int, text: str, ready_queue, start_event, result_queue):
    """One VoxCPM instance per process, 1 torch thread each."""
    try:
        import torch
        torch.set_num_threads(1)

        from voxcpm import VoxCPM

        load_t0 = time.time()
        model = VoxCPM.from_pretrained(MODEL_ID, optimize=False, device="cpu")
        load_s = time.time() - load_t0
        ready_queue.put({
            "worker": worker_id,
            "pid": os.getpid(),
            "load_s": load_s,
            "status": "ready",
        })

        start_event.wait()

        t0 = time.time()
        wav = model.generate(text=text, inference_timesteps=10)
        elapsed = time.time() - t0
        duration = len(wav) / model.tts_model.sample_rate
        result_queue.put({
            "worker": worker_id,
            "pid": os.getpid(),
            "load_s": load_s,
            "gen_s": elapsed,
            "audio_s": duration,
            "rtf": elapsed / duration,
            "status": "ok",
        })
        del model
    except Exception:
        error = {
            "worker": worker_id,
            "pid": os.getpid(),
            "status": "error",
            "traceback": traceback.format_exc(),
        }
        ready_queue.put(error)
        result_queue.put(error)


def run_parallel(n: int):
    ctx = multiprocessing.get_context("spawn")
    ready_q = ctx.Queue()
    result_q = ctx.Queue()
    start_event = ctx.Event()
    procs = []

    for i in range(n):
        p = ctx.Process(target=worker, args=(i, TEXT, ready_q, start_event, result_q))
        p.start()
        procs.append(p)

    ready = []
    for _ in range(n):
        ready.append(ready_q.get(timeout=900))

    wall_t0 = time.time()
    start_event.set()

    results = []
    for _ in range(n):
        try:
            results.append(result_q.get(timeout=900))
        except queue.Empty:
            results.append({"status": "timeout"})

    wall = time.time() - wall_t0

    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    # Sort by worker id
    ready.sort(key=lambda r: r.get("worker", 9999))
    results.sort(key=lambda r: r.get("worker", 9999))
    return wall, ready, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()

    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"Text: {len(TEXT)} chars")
    print()

    baselines = {}
    for n in args.workers:
        print(f"{'='*60}")
        print(f"Parallel workers: {n}")
        print(f"{'='*60}")
        wall, ready, results = run_parallel(n)
        ok_results = [r for r in results if r.get("status") == "ok"]
        total_audio = sum(r["audio_s"] for r in ok_results)
        avg_rtf = sum(r["rtf"] for r in ok_results) / len(ok_results) if ok_results else float("inf")
        throughput = total_audio / wall if wall > 0 else 0.0

        for r in ready:
            if r.get("status") == "ready":
                print(f"  Worker {r['worker']} ready: load={r['load_s']:.1f}s, pid={r['pid']}")
            else:
                print(f"  Worker {r.get('worker')} failed while loading")
        print(f"  Wall time:        {wall:.1f}s")
        print(f"  Total audio:      {total_audio:.1f}s")
        print(f"  Throughput:       {throughput:.2f}x realtime ({total_audio:.0f}s audio / {wall:.0f}s wall)")
        for r in results:
            if r.get("status") != "ok":
                print(f"    Worker {r.get('worker')}: {r.get('status')}")
                if r.get("traceback"):
                    print(r["traceback"])
                continue
            print(f"    Worker {r['worker']}: {r['gen_s']:.1f}s gen, {r['audio_s']:.1f}s audio, RTF={r['rtf']:.1f}x")
        baselines[n] = {"wall": wall, "total_audio": total_audio, "avg_rtf": avg_rtf, "throughput": throughput}

    # Scaling analysis
    if 1 in baselines and len(baselines) > 1:
        single = baselines[1]
        print(f"\n{'='*60}")
        print("SCALING ANALYSIS")
        print(f"{'='*60}")
        print(f"{'Workers':<10} {'Wall(s)':<10} {'Throughput':<14} {'Speedup':<10} {'Efficiency':<12}")
        for n in args.workers:
            d = baselines[n]
            speedup = single["wall"] / d["wall"]
            efficiency = speedup / n * 100
            print(f"{n:<10} {d['wall']:<10.1f} {d['throughput']:<14.2f}x {speedup:<10.2f}x {efficiency:<11.1f}%")

        print()
        if baselines[max(args.workers)]["throughput"] >= baselines[1]["throughput"] * max(args.workers) * 0.7:
            print("Memory bandwidth is NOT a severe bottleneck at this concurrency level.")
        else:
            print("Memory bandwidth IS a bottleneck - throughput does not scale with workers.")


if __name__ == "__main__":
    main()
