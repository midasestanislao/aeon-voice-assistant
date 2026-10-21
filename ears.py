"""El oído — escucha el micrófono, detecta cuándo hablas (VAD) y lo convierte en texto con Whisper.

El VAD (webrtcvad) mira tramos de 30 ms de audio int16 y dice "hay voz" o "no hay voz":
    - empieza a grabar tras `vad_start_ms` de voz seguida (un golpe o un clic no cuenta)
    - guarda `vad_preroll_ms` de audio previo para no perder la primera sílaba
    - termina tras `vad_silence_ms` de silencio seguido

Ejecutar (con Voicebox corriendo en otra terminal):
    uv run python ears.py
"""

# ---------- Imports ----------
import collections
import io
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

from config import Settings, get_settings
from voicebox import VoiceboxClient

# ---------- Constantes ----------
FRAME_MS = 30  # webrtcvad solo acepta tramos de 10, 20 o 30 ms

# Lo que Whisper "oye" a veces en el ruido o en una respiración: no lo dijo nadie
PHANTOM_PHRASES = {
    "gracias.",
    "thank you.",
    "thanks for watching!",
    "subtítulos realizados por la comunidad de amara.org",
}


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ---------- Oído ----------
class Ears:
    def __init__(self, settings: Settings, voicebox: VoiceboxClient) -> None:
        self._settings = settings
        self._voicebox = voicebox
        self._rate = settings.sample_rate
        self._frame = self._rate * FRAME_MS // 1000  # muestras por tramo

    def _frames(self, ms: float) -> int:
        return max(1, int(ms // FRAME_MS))

    def _microphone(self) -> sd.InputStream:
        return sd.InputStream(samplerate=self._rate, channels=1, dtype="int16", blocksize=self._frame)

    def record(self, on_start: Callable[[], None] | None = None) -> np.ndarray:
        """Espera a que hables y graba hasta que te callas. Devuelve audio int16 mono."""
        s = self._settings
        vad = webrtcvad.Vad(s.vad_aggressiveness)
        preroll = collections.deque(maxlen=self._frames(s.vad_preroll_ms))
        frames: list[np.ndarray] = []
        voiced = silent = 0

        with self._microphone() as mic:
            while True:
                frame = mic.read(self._frame)[0][:, 0]
                speech = vad.is_speech(frame.tobytes(), self._rate)
                if not frames:  # todavía esperando a que empiece a hablar
                    preroll.append(frame)
                    voiced = voiced + 1 if speech else 0
                    if voiced >= self._frames(s.vad_start_ms):
                        frames = list(preroll)
                        if on_start:
                            on_start()
                else:
                    frames.append(frame)
                    silent = 0 if speech else silent + 1
                    if silent >= self._frames(s.vad_silence_ms) or len(frames) >= self._frames(
                        s.max_turn_s * 1000
                    ):
                        return np.concatenate(frames)

    def listen(self, on_start: Callable[[], None] | None = None) -> str | None:
        """Graba un turno y lo transcribe. Devuelve None si no se entendió nada."""
        wav = to_wav_bytes(self.record(on_start), self._rate)
        text = self._voicebox.transcribe(wav, self._settings.whisper_model)
        if not text or text.lower() in PHANTOM_PHRASES:
            return None
        return text

    def watch_for_speech(self, stop: threading.Event, on_speech: Callable[[], None]) -> None:
        """Vigila el micrófono hasta `stop`. Si alguien habla `barge_in_ms` seguidos, llama a `on_speech`."""
        vad = webrtcvad.Vad(self._settings.vad_aggressiveness)
        needed = self._frames(self._settings.barge_in_ms)
        voiced = 0
        with self._microphone() as mic:
            while not stop.is_set():
                frame = mic.read(self._frame)[0]
                voiced = voiced + 1 if vad.is_speech(frame.tobytes(), self._rate) else 0
                if voiced >= needed:
                    on_speech()
                    return


# ---------- Main ----------
if __name__ == "__main__":
    settings = get_settings()
    with VoiceboxClient(settings.voicebox_url, settings.voicebox_timeout_s) as vb:
        ears = Ears(settings, vb)
        print("Dictado: habla cuando quieras (Ctrl + C para salir).")
        try:
            while True:
                print("👂 ...")
                print("Oí:", ears.listen(on_start=lambda: print("🎙️  te escucho")) or "(nada)")
        except KeyboardInterrupt:
            pass
