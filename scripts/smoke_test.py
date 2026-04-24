from pathlib import Path

import soundfile as sf
import torch

from voxcpm import VoxCPM


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "smoke_test.wav"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        device=device,
        optimize=False,
    )
    wav = model.generate(
        text="你好，这是一段用于验证 VoxCPM2 部署是否正常的测试语音。",
        cfg_value=2.0,
        inference_timesteps=4,
    )
    sf.write(output_path, wav, model.tts_model.sample_rate)
    print(f"saved: {output_path}")
    print(f"device: {device}")
    print(f"sample_rate: {model.tts_model.sample_rate}")
    print(f"samples: {len(wav)}")


if __name__ == "__main__":
    main()
