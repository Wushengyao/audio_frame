from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    control_instruction: str = ""
    reference_audio_base64: str | None = None
    reference_audio_filename: str = "reference.wav"
    prompt_text: str = ""
    cfg_value: float = 2.0
    normalize: bool = True
    denoise: bool = False
    inference_timesteps: int = 10


class TTSResponse(BaseModel):
    filename: str
    mime_type: str
    sample_rate: int
    audio_base64: str


class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.model_id = "openbmb/VoxCPM2"
        self.lock = Lock()

    def load(self):
        with self.lock:
            if self.model is not None:
                return self.model
            try:
                from .core import VoxCPM
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "VoxCPM runtime dependencies are not installed. "
                    "Run AUDIO_FRAME_FULL_INSTALL=1 ./start_wsl_services.sh or install audio_frame with `uv pip install -e .`."
                ) from exc

            logger.info("Loading VoxCPM model: %s", self.model_id)
            self.model = VoxCPM.from_pretrained(self.model_id, optimize=True)
            logger.info("VoxCPM model loaded.")
            return self.model


state = ModelState()
app = FastAPI(title="VoxCPM Audio Frame API")


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_id": state.model_id,
        "model_loaded": state.model is not None,
    }


@app.post("/api/tts", response_model=TTSResponse)
def synthesize(payload: TTSRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    try:
        with tempfile.TemporaryDirectory(prefix="voxcpm_api_") as tmp_dir:
            reference_path = _write_reference_audio(payload, Path(tmp_dir))
            sample_rate, wav = _generate_audio(payload, reference_path)
            audio_bytes = _encode_wav(sample_rate, wav)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS generation failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TTSResponse(
        filename="speech.wav",
        mime_type="audio/wav",
        sample_rate=sample_rate,
        audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
    )


def _write_reference_audio(payload: TTSRequest, tmp_dir: Path) -> Optional[str]:
    if not payload.reference_audio_base64:
        return None
    suffix = Path(payload.reference_audio_filename or "reference.wav").suffix or ".wav"
    reference_path = tmp_dir / f"reference{suffix}"
    try:
        reference_path.write_bytes(base64.b64decode(payload.reference_audio_base64))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid reference_audio_base64") from exc
    return str(reference_path)


def _generate_audio(payload: TTSRequest, reference_path: Optional[str]) -> tuple[int, np.ndarray]:
    model = state.load()
    control = payload.control_instruction.strip()
    prompt_text = payload.prompt_text.strip()
    final_text = f"({control}){payload.text.strip()}" if control and not prompt_text else payload.text.strip()
    kwargs = {
        "text": final_text,
        "reference_wav_path": reference_path,
        "cfg_value": float(payload.cfg_value),
        "inference_timesteps": int(payload.inference_timesteps),
        "normalize": bool(payload.normalize),
        "denoise": bool(payload.denoise) and bool(reference_path),
    }
    if reference_path and prompt_text:
        kwargs["prompt_wav_path"] = reference_path
        kwargs["prompt_text"] = prompt_text

    wav = model.generate(**kwargs)
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return int(model.tts_model.sample_rate), np.asarray(wav, dtype=np.float32).reshape(-1)


def _encode_wav(sample_rate: int, wav: np.ndarray) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    return buffer.getvalue()


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the VoxCPM Audio Frame HTTP API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--model-id", default="openbmb/VoxCPM2")
    args = parser.parse_args()
    state.model_id = args.model_id
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
