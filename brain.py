"""El cerebro — LLM (OpenRouter) + herramientas + memoria que sobrevive entre sesiones.

Un turno del agente es un bucle: model -> (¿pide herramientas?) -> tools -> model -> ... -> respuesta.
La conversación se guarda en SQLite (outputs/memoria.sqlite) con un checkpointer: al volver a
arrancar, el agente recuerda lo que hablamos. Para que la memoria no crezca sin límite (más lento,
más caro y al final no cabe en el modelo), antes de cada llamada solo le pasamos los últimos mensajes.

Ejecutar:
    uv run python brain.py
"""

# ---------- Imports ----------
import sqlite3
from collections.abc import Callable, Iterator
from typing import Self

from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from config import Settings, get_settings
from tools import TOOLS

# ---------- Prompt ----------
SYSTEM_PROMPT = """Eres {name}, un asistente de voz. Tus respuestas NO se leen: se convierten en voz y la persona \
las ESCUCHA. Escribe exactamente lo que dirías en voz alta.

- Responde SIEMPRE en el idioma del último mensaje del usuario (español o inglés). \
If the user writes in English, reply entirely in English.
- Entre 1 y 3 frases cortas. Si el tema es amplio, da lo esencial y ofrece contar más.
- NUNCA uses markdown, listas, emojis ni URLs. Escribe los símbolos como se dicen ("por ciento", "grados").
- Habla de forma cálida, cercana y natural, como en una conversación.
- Para fechas, noticias o datos actuales usa una herramienta; para cualquier cálculo, la calculadora. Nunca inventes.
- Si no entiendes lo que te dijeron (puede ser un error de transcripción), pide que lo repitan.
- Recuerdas conversaciones anteriores con esta persona: usa lo que sabes de ella cuando venga al caso.
"""

# Lo que dice mientras usa una herramienta, en vez de quedarse callado
FILLERS = {
    "es": ["Déjame ver…", "Un segundo…", "A ver, lo reviso…"],
    "en": ["Let me check…", "One sec…", "Let me look that up…"],
}

THREAD_ID = "principal"  # siempre la misma conversación: así se retoma al arrancar


class MissingApiKeyError(RuntimeError):
    pass


# ---------- Cerebro ----------
class Brain:
    def __init__(self, settings: Settings) -> None:
        api_key = settings.openrouter_api_key.get_secret_value()
        if not api_key:
            raise MissingApiKeyError("Falta OPENROUTER_API_KEY en .env (ver .env.example)")
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )

        settings.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(settings.memory_path, check_same_thread=False)
        self._memory = SqliteSaver(self._db)
        self._config = {"configurable": {"thread_id": THREAD_ID}}
        self._agent = create_agent(
            llm,
            TOOLS,
            system_prompt=SYSTEM_PROMPT.format(name=settings.agent_name),
            middleware=[_keep_recent(settings.memory_max_messages)],
            checkpointer=self._memory,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self._db.close()

    def stream(self, text: str, on_tool: Callable[[str], None] | None = None) -> Iterator[str]:
        """Devuelve los pedazos de la respuesta que hay que DECIR. Avisa con `on_tool` al usar una herramienta."""
        inputs = {"messages": [{"role": "user", "content": text}]}
        for chunk, metadata in self._agent.stream(inputs, self._config, stream_mode="messages"):
            # Solo los tokens del modelo; los resultados de las herramientas no se dicen en voz alta
            if metadata.get("langgraph_node") != "model" or not isinstance(chunk, AIMessageChunk):
                continue
            for call in chunk.tool_call_chunks:
                if call.get("name") and on_tool:
                    on_tool(call["name"])
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

    def forget(self) -> None:
        self._memory.delete_thread(THREAD_ID)


def _keep_recent(max_messages: int):
    """Middleware: antes de llamar al modelo, borra lo más viejo y deja ~`max_messages` mensajes.

    El corte empieza siempre en un mensaje del usuario, para no separar una petición de
    herramienta de su resultado (el modelo lo rechazaría).
    """

    @before_model
    def keep_recent(state, runtime):
        messages = state["messages"]
        if len(messages) <= max_messages:
            return None
        human = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        start = next((i for i in human if i >= len(messages) - max_messages), human[-1])
        if start == 0:
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[start:]]}

    return keep_recent


# ---------- Main ----------
if __name__ == "__main__":
    with Brain(get_settings()) as brain:
        print("Chat con memoria. 'olvidar' borra la memoria, 'exit' sale.")
        while (text := input("Tú: ").strip()) != "exit":
            if text == "olvidar":
                brain.forget()
                print("(memoria borrada)")
                continue
            if not text:
                continue
            print("Aeon: ", end="", flush=True)
            for piece in brain.stream(text, on_tool=lambda name: print(f"🔧 {name} ", end="", flush=True)):
                print(piece, end="", flush=True)
            print()
