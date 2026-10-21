"""Mi agente de voz — une el oído, el cerebro y la boca en una conversación.

    ears.listen() -> brain.stream() -> SentenceSplitter -> Speaker -> (otra vez a escuchar)

Ejecutar (con Voicebox corriendo en otra terminal):
    uv run python main.py                 # manos libres
    uv run python main.py --texto         # escribiendo (sin micrófono)
    uv run python main.py --interrumpir   # puedes cortarle hablando (solo con audífonos)
    uv run python main.py --olvidar       # empieza sin memoria
"""

# ---------- Imports ----------
import argparse
import random
import threading

from brain import FILLERS, Brain, MissingApiKeyError
from config import Language, Settings, get_settings
from ears import Ears
from mouth import SentenceSplitter, Speaker, detect_language
from voicebox import VoiceboxClient, VoiceboxError

EXIT_WORDS = {"adiós", "adios", "salir", "goodbye", "bye", "exit"}


# ---------- Conversación ----------
class Conversation:
    def __init__(self, settings: Settings, voicebox: VoiceboxClient, brain: Brain, *, barge_in: bool) -> None:
        self.settings = settings
        self.voicebox = voicebox
        self.brain = brain
        self.ears = Ears(settings, voicebox)
        self.barge_in = barge_in
        self.language: Language = "es"

    def synthesize(self, text: str, language: Language) -> bytes:
        s = self.settings
        profile = self.voicebox.profile_id(s.tts_engine, s.voice_for(language), language)
        return self.voicebox.synthesize(text, profile, language, s.tts_engine)

    def prepare(self, *, with_whisper: bool) -> None:
        """Descarga lo que falte y precalienta Kokoro: si no, el primer turno tarda decenas de segundos."""
        models = [self.settings.tts_engine] + (
            [f"whisper-{self.settings.whisper_model}"] if with_whisper else []
        )
        for model in models:
            self.voicebox.ensure_model(model)
        for language in ("es", "en"):
            self.synthesize("Hola.", language)

    def respond(self, text: str) -> None:
        self.language = detect_language(text, self.language)
        speaker = Speaker(self.synthesize, self.language, pause_s=self.settings.pause_ms / 1000)
        splitter = SentenceSplitter(self.settings.first_chunk_chars, self.settings.later_chunk_chars)
        stop_watching = threading.Event()
        if self.barge_in:
            threading.Thread(
                target=self.ears.watch_for_speech, args=(stop_watching, speaker.interrupt), daemon=True
            ).start()

        filler_said = False

        def on_tool(name: str) -> None:
            nonlocal filler_said
            print(f"🔧 {name} ", end="", flush=True)
            if not filler_said:  # una sola frase de espera por turno
                filler_said = True
                speaker.say(random.choice(FILLERS[self.language]), self.language)

        print(f"{self.settings.agent_name}: ", end="", flush=True)
        try:
            for piece in self.brain.stream(text, on_tool):
                if speaker.interrupted:
                    print(" ✋", end="")
                    break
                print(piece, end="", flush=True)
                for sentence in splitter.feed(piece):
                    speaker.say(sentence)
            for sentence in splitter.flush():
                speaker.say(sentence)
        finally:
            # Pase lo que pase: dejamos de vigilar el micrófono y esperamos a que la boca termine
            print()
            stop_watching.set()
            speaker.finish()
        if speaker.first_audio_s is not None:
            print(f"(espera hasta oír: {speaker.first_audio_s:.2f} s)\n")

    def next_text(self, typing: bool) -> str | None:
        if typing:
            return input("Tú: ").strip() or None
        print("👂 ...")
        text = self.ears.listen(on_start=lambda: print("🎙️  te escucho"))
        if text:
            print(f"Tú: {text}")
        return text

    def run(self, *, typing: bool) -> None:
        while True:
            try:
                text = self.next_text(typing)
                if text is None:
                    continue
                if text.lower().strip(" .,!¡¿?") in EXIT_WORDS:
                    return
                self.respond(text)
            except (KeyboardInterrupt, EOFError):  # Ctrl+C, o se acabó la entrada (Ctrl+Z en Windows)
                return
            except Exception as e:  # un turno que falla (red, API, Voicebox) no termina la conversación
                print(f"\n[algo falló en este turno: {e}]\n")


# ---------- Main ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Agente de voz Aeon")
    parser.add_argument("--texto", action="store_true", help="escribir en vez de hablar")
    parser.add_argument(
        "--interrumpir", action="store_true", help="cortar al agente hablando (con audífonos)"
    )
    parser.add_argument("--olvidar", action="store_true", help="borrar la memoria antes de empezar")
    args = parser.parse_args()

    settings = get_settings()
    try:
        with (
            VoiceboxClient(settings.voicebox_url, settings.voicebox_timeout_s) as voicebox,
            Brain(settings) as brain,
        ):
            if args.olvidar:
                brain.forget()
            conversation = Conversation(settings, voicebox, brain, barge_in=args.interrumpir)
            print("Cargando voces...")
            conversation.prepare(with_whisper=not args.texto)
            print(f"{settings.agent_name} listo. Di o escribe 'adiós' para salir.\n")
            conversation.run(typing=args.texto)
    except (MissingApiKeyError, VoiceboxError) as e:
        print(f"Error: {e}")
    print("Adiós.")


if __name__ == "__main__":
    main()
