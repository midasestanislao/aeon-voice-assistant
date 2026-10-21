"""Configuración — todos los ajustes del agente en un solo sitio, con tipos y validados.

Cada campo se puede cambiar desde `.env` sin tocar el código (`llm_model` <- `LLM_MODEL`).
Un valor sin sentido (`LLM_TEMPERATURE=5`) falla al arrancar con un mensaje claro.

Ejecutar:
    uv run python config.py
"""

# ---------- Imports ----------
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------- Rutas y tipos ----------
ROOT = Path(__file__).resolve().parent  # la carpeta del proyecto: aquí viven .env y outputs/

Language = Literal["es", "en"]


# ---------- Ajustes ----------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openrouter_api_key: SecretStr = SecretStr("")
    llm_model: str = "openai/gpt-4o"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_temperature: float = Field(default=0.5, ge=0, le=2)
    agent_name: str = "Aeon"

    # Voicebox
    voicebox_url: str = "http://127.0.0.1:17493"
    voicebox_timeout_s: float = 120.0
    tts_engine: str = "kokoro"
    voice_es: str = "ef_dora"
    voice_en: str = "af_heart"
    whisper_model: Literal["base", "small", "medium", "large", "turbo"] = "turbo"

    # Memoria
    memory_path: Path = ROOT / "outputs" / "memoria.sqlite"
    memory_max_messages: int = Field(default=40, ge=4)

    # Micrófono
    sample_rate: int = 16_000
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    vad_start_ms: int = 150  # voz seguida para empezar a grabar
    vad_silence_ms: int = 800  # silencio seguido para terminar
    vad_preroll_ms: int = 300  # audio previo que se conserva
    max_turn_s: float = 30.0
    barge_in_ms: int = 300  # voz seguida que interrumpe al agente

    # Naturalidad
    first_chunk_chars: int = 25
    later_chunk_chars: int = 120
    pause_ms: int = Field(default=200, ge=0)

    def voice_for(self, language: Language) -> str:
        return self.voice_es if language == "es" else self.voice_en


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---------- Main ----------
if __name__ == "__main__":
    settings = get_settings()
    print(settings)
    print("es:", settings.voice_for("es"), "| en:", settings.voice_for("en"))
