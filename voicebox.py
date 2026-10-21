"""Voicebox — el puente HTTP entre el agente y el servidor de voz (Kokoro para hablar, Whisper para oír).

Endpoints que usamos (Voicebox v0.5.0):
    GET  /health                     {"status", "gpu_type", ...}
    GET  /models/status              {"models": [{"model_name", "downloaded", "downloading", ...}]}
    POST /models/download            JSON {"model_name": "whisper-turbo"}
    GET  /profiles                   [{"id", "preset_engine", "preset_voice_id", ...}]
    POST /profiles                   crea un perfil de voz preset
    POST /transcribe                 multipart "file" + "model" -> {"text": ...}   (202 = Whisper descargándose)
    POST /generate/stream            JSON {"profile_id", "text", "language", "engine"} -> bytes de un WAV

Ejecutar (con Voicebox corriendo en otra terminal):
    uv run python voicebox.py
"""

# ---------- Imports ----------
import time
from typing import Any, Self

import httpx

from config import Language, get_settings


# ---------- Errores ----------
class VoiceboxError(RuntimeError):
    pass


class VoiceboxUnavailableError(VoiceboxError):
    pass


class ModelDownloadingError(VoiceboxError):
    pass


# ---------- Cliente ----------
class VoiceboxClient:
    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s)
        self._profiles: dict[str, str] = {}  # voice_id -> profile_id

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TransportError as e:
            raise VoiceboxUnavailableError(
                f"Voicebox no responde en {self._http.base_url}. Arráncalo con .\\scripts\\start_voicebox.ps1"
            ) from e
        if response.status_code == 202:
            raise ModelDownloadingError("Voicebox está descargando el modelo; prueba en un rato.")
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise VoiceboxError(f"{method} {path} -> {response.status_code}: {detail}")
        return response

    # ---------- Estado y modelos ----------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def _model(self, name: str) -> dict[str, Any]:
        for model in self._request("GET", "/models/status").json()["models"]:
            if model["model_name"] == name:
                return model
        raise VoiceboxError(f"Voicebox no conoce el modelo '{name}'")

    def model_downloaded(self, name: str) -> bool:
        return self._model(name)["downloaded"]

    def ensure_model(self, name: str, timeout_s: float = 1800.0) -> None:
        model = self._model(name)
        if model["downloaded"]:
            return
        if not model.get("downloading"):
            self._request("POST", "/models/download", json={"model_name": name})
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(3)
            if self.model_downloaded(name):
                return
        raise VoiceboxError(f"Se agotó el tiempo descargando '{name}'")

    # ---------- Voz ----------
    def profile_id(self, engine: str, voice_id: str, language: Language) -> str:
        if voice_id not in self._profiles:
            self._profiles[voice_id] = self._find_or_create_profile(engine, voice_id, language)
        return self._profiles[voice_id]

    def _find_or_create_profile(self, engine: str, voice_id: str, language: Language) -> str:
        for profile in self._request("GET", "/profiles").json():
            if profile.get("preset_engine") == engine and profile.get("preset_voice_id") == voice_id:
                return str(profile["id"])
        payload = {
            "name": f"Aeonether {voice_id}",
            "language": language,
            "voice_type": "preset",
            "preset_engine": engine,
            "preset_voice_id": voice_id,
            "default_engine": engine,
        }
        return str(self._request("POST", "/profiles", json=payload).json()["id"])

    def transcribe(self, wav: bytes, model: str) -> str:
        files = {"file": ("audio.wav", wav, "audio/wav")}
        response = self._request("POST", "/transcribe", files=files, data={"model": model})
        return str(response.json()["text"]).strip()

    def synthesize(self, text: str, profile_id: str, language: Language, engine: str) -> bytes:
        payload = {"profile_id": profile_id, "text": text, "language": language, "engine": engine}
        return self._request("POST", "/generate/stream", json=payload).content


# ---------- Main ----------
if __name__ == "__main__":
    settings = get_settings()
    whisper = f"whisper-{settings.whisper_model}"
    try:
        with VoiceboxClient(settings.voicebox_url, settings.voicebox_timeout_s) as vb:
            health = vb.health()
            print(f"Voicebox: {health.get('status')} | GPU: {health.get('gpu_type') or 'no (CPU)'}")

            for model in (settings.tts_engine, whisper):
                print(f"Preparando {model}...")
                vb.ensure_model(model)

            profiles = {}
            for language in ("es", "en"):
                voice = settings.voice_for(language)
                profiles[language] = vb.profile_id(settings.tts_engine, voice, language)
                print(f"Perfil {language} ({voice}): {profiles[language]}")

            wav = vb.synthesize(
                "Hola, esto es una prueba de ida y vuelta.", profiles["es"], "es", settings.tts_engine
            )
            print(f"WAV: {len(wav)} bytes, empieza con {wav[:4]!r}")
            print(f"Whisper oyó: {vb.transcribe(wav, settings.whisper_model)!r}")
    except VoiceboxError as e:
        print(f"Error: {e}")
