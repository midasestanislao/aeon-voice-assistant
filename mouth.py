"""La boca — convierte el texto del LLM en voz, frase a frase.

Lección 2 (texto para voz):
    clean_for_speech   quita markdown y URLs que la voz leería en voz alta
    detect_language    ES o EN contando palabras típicas, sin otro modelo
    SentenceSplitter   corta el stream de tokens en frases para empezar a hablar pronto

Lección 3 (el locutor):
    Speaker            dos hilos: uno genera el audio de la siguiente frase mientras el otro suena

Ejecutar (con Voicebox corriendo en otra terminal):
    uv run python mouth.py
"""

# ---------- Imports ----------
import io
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator

import sounddevice as sd
import soundfile as sf

from config import Language

# ---------- Texto para voz ----------
_MARKDOWN = [
    (re.compile(r"```.*?```", re.DOTALL), " "),  # bloques de código
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # [texto](url) -> texto
    (re.compile(r"https?://\S+"), ""),  # URLs sueltas
    (re.compile(r"[*_`#>]+"), ""),  # negritas, cursivas, títulos, citas
    (re.compile(r"^\s*[-•]\s+", re.MULTILINE), ""),  # viñetas
]


def clean_for_speech(text: str) -> str:
    for pattern, replacement in _MARKDOWN:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


_WORD = re.compile(r"[a-záéíóúüñ']+", re.IGNORECASE)
_ES_WORDS = set(
    "el la los las un una de del que y en es por para con se lo mi tu su como pero más muy qué "
    "cómo dónde cuándo hola gracias sí está estoy eres soy hay puedes quiero hoy ahora bien bueno".split()
)
_EN_WORDS = set(
    "the an of and to in is are was for with on at it you your i my we what how where when who "
    "hello hi thanks yes please can could would will do does have has not but today now good".split()
)


def detect_language(text: str, fallback: Language) -> Language:
    """Cuenta palabras típicas de cada idioma (y tildes/ñ/¿¡ cuentan doble para español). Empate -> fallback."""
    words = [w.lower() for w in _WORD.findall(text)]
    es = sum(w in _ES_WORDS for w in words) + 2 * len(re.findall(r"[ñáéíóú¿¡]", text, re.IGNORECASE))
    en = sum(w in _EN_WORDS for w in words)
    if es == en:
        return fallback
    return "es" if es > en else "en"


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


class SentenceSplitter:
    """Junta tokens y devuelve frases completas. La 1.ª sale en cuanto es corta; las demás se agrupan."""

    def __init__(self, first_chars: int = 25, later_chars: int = 120) -> None:
        self._buffer = ""
        self._min_chars = first_chars
        self._later_chars = later_chars

    def feed(self, token: str) -> Iterator[str]:
        self._buffer += token
        while (cut := self._find_cut()) is not None:
            sentence, self._buffer = self._buffer[:cut], self._buffer[cut:]
            self._min_chars = self._later_chars
            yield from self._speakable(sentence)

    def flush(self) -> Iterator[str]:
        rest, self._buffer = self._buffer, ""
        yield from self._speakable(rest)

    def _find_cut(self) -> int | None:
        for match in _SENTENCE_END.finditer(self._buffer):
            if match.end() >= self._min_chars:
                return match.end()
        return None

    @staticmethod
    def _speakable(text: str) -> Iterator[str]:
        text = clean_for_speech(text)
        if any(ch.isalnum() for ch in text):
            yield text


# ---------- Audio ----------
def play_wav(wav: bytes) -> None:
    samples, sample_rate = sf.read(io.BytesIO(wav), dtype="float32")
    sd.play(samples, sample_rate)
    sd.wait()


# ---------- Locutor ----------
_END = object()  # señal de "no hay más trabajo"


class Speaker:
    """Habla frases en orden. `synthesize` y `play` se inyectan: en un test se pueden pasar funciones falsas."""

    def __init__(
        self,
        synthesize: Callable[[str, Language], bytes],
        language: Language = "es",
        *,
        play: Callable[[bytes], None] = play_wav,
        stop: Callable[[], None] = sd.stop,
        pause_s: float = 0.2,
    ) -> None:
        self.language: Language = language
        self.first_audio_s: float | None = None
        self._started = time.perf_counter()
        self._synthesize = synthesize
        self._play = play
        self._stop = stop
        self._pause_s = pause_s
        self._interrupted = threading.Event()
        self._sentences: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._synth_loop, daemon=True),
            threading.Thread(target=self._play_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def say(self, sentence: str, language: Language | None = None) -> None:
        """Encola una frase y vuelve al instante. Sin `language`, se detecta de la frase."""
        self.language = language or detect_language(sentence, self.language)
        self._sentences.put((sentence, self.language))

    def finish(self) -> None:
        """Espera a que suene todo lo encolado."""
        self._sentences.put(_END)
        for thread in self._threads:
            thread.join()

    def interrupt(self) -> None:
        self._interrupted.set()
        self._stop()

    def _synth_loop(self) -> None:
        while (item := self._sentences.get()) is not _END:
            if self.interrupted:
                continue
            try:
                self._audio.put(self._synthesize(*item))
            except Exception as e:  # una frase que falla no debe callar al resto
                print(f"\n[no se pudo sintetizar {item[0]!r}: {e}]")
        self._audio.put(_END)

    def _play_loop(self) -> None:
        while (wav := self._audio.get()) is not _END:
            if self.interrupted:
                continue
            if self.first_audio_s is None:
                self.first_audio_s = time.perf_counter() - self._started
            self._play(wav)
            self._interrupted.wait(self._pause_s)  # como sleep, pero termina al instante si interrumpen


# ---------- Main ----------
if __name__ == "__main__":
    from config import get_settings
    from voicebox import VoiceboxClient

    settings = get_settings()
    reply = "**Hola!** Soy Aeon. Hoy hace un día [precioso](https://x.com).\n- Hello, how are you today? Bien, gracias."

    splitter = SentenceSplitter(settings.first_chunk_chars, settings.later_chunk_chars)
    sentences = [s for token in reply.split(" ") for s in splitter.feed(token + " ")] + list(splitter.flush())
    for sentence in sentences:
        print(f"[{detect_language(sentence, 'es')}] {sentence}")

    with VoiceboxClient(settings.voicebox_url, settings.voicebox_timeout_s) as vb:

        def synthesize(text: str, language: Language) -> bytes:
            profile = vb.profile_id(settings.tts_engine, settings.voice_for(language), language)
            return vb.synthesize(text, profile, language, settings.tts_engine)

        speaker = Speaker(synthesize, pause_s=settings.pause_ms / 1000)
        for sentence in sentences:
            speaker.say(sentence)
        speaker.finish()
        print(f"Espera hasta oír: {speaker.first_audio_s:.2f} s")
