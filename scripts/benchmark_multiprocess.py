#!/usr/bin/env python3
"""Benchmark steady-state VoxCPM2 inference with multiple processes."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

MODEL_ID = "openbmb/VoxCPM2"
TEXT = "肚子里传来一阵咕噜声，她这才想起自己已经一天没吃东西了。"


def _set_common_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _worker(
    worker_id: int,
    runtime: str,
    physical_gpu: int | None,
    text: str,
    model_id: str,
    timesteps: int,
    optimize: bool,
    load_denoiser: bool,
    offline: bool,
    torch_threads: int,
    ready_queue: mp.Queue,
    start_event: mp.Event,
    result_queue: mp.Queue,
) -> None:
    try:
        _set_common_env()

        if runtime == "cuda" and physical_gpu is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

        if runtime == "cpu":
            os.environ["OMP_NUM_THREADS"] = str(torch_threads)
            os.environ["MKL_NUM_THREADS"] = str(torch_threads)

        import torch

        if runtime == "cpu":
            torch.set_num_threads(torch_threads)

        from voxcpm import VoxCPM

        load_t0 = time.perf_counter()
        model = VoxCPM.from_pretrained(
            model_id,
            load_denoiser=load_denoiser,
            local_files_only=offline,
            optimize=optimize,
            device=runtime,
        )
        load_s = time.perf_counter() - load_t0

        if runtime == "cuda":
            torch.cuda.synchronize()

        ready_queue.put(
            {
                "worker": worker_id,
                "pid": os.getpid(),
                "runtime": runtime,
                "physical_gpu": physical_gpu,
                "load_s": load_s,
                "status": "ready",
            }
        )

        start_event.wait()

        if runtime == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        gen_t0 = time.perf_counter()
        wav = model.generate(
            text=text,
            inference_timesteps=timesteps,
            retry_badcase=False,
        )
        if runtime == "cuda":
            torch.cuda.synchronize()
        gen_s = time.perf_counter() - gen_t0

        audio_s = len(wav) / model.tts_model.sample_rate
        result: dict[str, Any] = {
            "worker": worker_id,
            "pid": os.getpid(),
            "runtime": runtime,
            "physical_gpu": physical_gpu,
            "load_s": load_s,
            "gen_s": gen_s,
            "audio_s": audio_s,
            "rtf": gen_s / audio_s if audio_s > 0 else float("inf"),
            "status": "ok",
        }
        if runtime == "cuda":
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024 / 1024
        result_queue.put(result)
    except Exception:
        error = {
            "worker": worker_id,
            "pid": os.getpid(),
            "runtime": runtime,
            "physical_gpu": physical_gpu,
            "status": "error",
            "traceback": traceback.format_exc(),
        }
        ready_queue.put(error)
        result_queue.put(error)


def _assignments(runtime: str, workers: int, gpus: list[int]) -> list[int | None]:
    if runtime == "cpu":
        return [None] * workers
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id for cuda benchmarks")
    return [gpus[i % len(gpus)] for i in range(workers)]


def run_once(args: argparse.Namespace, workers: int) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    start_event = ctx.Event()
    procs: list[mp.Process] = []

    for worker_id, physical_gpu in enumerate(_assignments(args.runtime, workers, args.gpus)):
        proc = ctx.Process(
            target=_worker,
            args=(
                worker_id,
                args.runtime,
                physical_gpu,
                args.text,
                args.model_id,
                args.timesteps,
                args.optimize,
                args.load_denoiser,
                args.offline,
                args.torch_threads,
                ready_queue,
                start_event,
                result_queue,
            ),
        )
        proc.start()
        procs.append(proc)

    ready: list[dict[str, Any]] = []
    for _ in range(workers):
        ready.append(ready_queue.get(timeout=args.load_timeout_s))

    wall_t0 = time.perf_counter()
    start_event.set()

    results: list[dict[str, Any]] = []
    for _ in range(workers):
        try:
            results.append(result_queue.get(timeout=args.generate_timeout_s))
        except queue.Empty:
            results.append({"status": "timeout"})

    wall_s = time.perf_counter() - wall_t0

    for proc in procs:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)

    ready.sort(key=lambda item: item.get("worker", 9999))
    results.sort(key=lambda item: item.get("worker", 9999))
    return wall_s, ready, results


def print_result(workers: int, wall_s: float, ready: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 72}")
    print(f"Workers: {workers}")
    print(f"{'=' * 72}")
    for item in ready:
        if item["status"] == "ready":
            gpu = "" if item["physical_gpu"] is None else f", gpu={item['physical_gpu']}"
            print(f"  ready worker {item['worker']}: load={item['load_s']:.1f}s{gpu}, pid={item['pid']}")
        else:
            print(f"  worker {item.get('worker')} failed during load")

    ok_results = [item for item in results if item.get("status") == "ok"]
    total_audio = sum(item["audio_s"] for item in ok_results)
    throughput = total_audio / wall_s if wall_s > 0 else 0.0
    print(f"  wall generate time: {wall_s:.1f}s")
    print(f"  total audio:        {total_audio:.1f}s")
    print(f"  throughput:         {throughput:.2f}x realtime")
    for item in results:
        if item.get("status") != "ok":
            print(f"  worker {item.get('worker')}: {item.get('status')}")
            if item.get("traceback"):
                print(item["traceback"])
            continue
        gpu = "" if item["physical_gpu"] is None else f", gpu={item['physical_gpu']}"
        mem = ""
        if "peak_reserved_mib" in item:
            mem = f", peak_reserved={item['peak_reserved_mib']:.0f}MiB"
        print(
            f"  worker {item['worker']}: gen={item['gen_s']:.1f}s, "
            f"audio={item['audio_s']:.1f}s, RTF={item['rtf']:.2f}x{gpu}{mem}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--workers", type=int, nargs="+", default=[1])
    parser.add_argument("--gpus", type=int, nargs="*", default=[0])
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--load-denoiser", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use cached model files only")
    parser.add_argument("--load-timeout-s", type=int, default=900)
    parser.add_argument("--generate-timeout-s", type=int, default=900)
    optimize = parser.add_mutually_exclusive_group()
    optimize.add_argument("--optimize", dest="optimize", action="store_true")
    optimize.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.set_defaults(optimize=None)
    args = parser.parse_args()
    if args.optimize is None:
        args.optimize = args.runtime == "cuda"
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return args


def main() -> None:
    args = parse_args()
    _set_common_env()
    print(f"Runtime: {args.runtime}")
    print(f"Workers: {args.workers}")
    print(f"Text chars: {len(args.text)}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Optimize: {args.optimize}")
    if args.runtime == "cuda":
        print(f"GPU assignment pool: {args.gpus}")

    baseline_throughput: float | None = None
    for workers in args.workers:
        wall_s, ready, results = run_once(args, workers)
        print_result(workers, wall_s, ready, results)
        ok_results = [item for item in results if item.get("status") == "ok"]
        throughput = sum(item["audio_s"] for item in ok_results) / wall_s if wall_s > 0 else 0.0
        if baseline_throughput is None:
            baseline_throughput = throughput
        elif baseline_throughput > 0:
            print(f"  throughput speedup vs first run: {throughput / baseline_throughput:.2f}x")


if __name__ == "__main__":
    main()
