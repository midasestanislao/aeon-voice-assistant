"""Las manos — herramientas que el agente puede usar: fecha y hora, calculadora y búsqueda web.

El modelo solo ve el nombre, la descripción (el docstring) y los parámetros de cada herramienta.
Decide cuándo usarla; quien la ejecuta de verdad es nuestro código.

Ejecutar:
    uv run python tools.py
"""

# ---------- Imports ----------
import ast
import operator
from datetime import datetime

from ddgs import DDGS
from langchain_core.tools import tool

# ---------- Calculadora segura ----------
# En vez de eval() (que ejecutaría cualquier código), analizamos la expresión con `ast`
# y solo calculamos números y estas operaciones.
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("solo se permiten números y + - * / % ( )")


def safe_calculate(expression: str) -> str:
    """Devuelve el resultado o el error como texto: así el modelo puede explicarlo en vez de romperse."""
    try:
        result = _evaluate(ast.parse(expression, mode="eval").body)
    except ZeroDivisionError:
        return "No se puede dividir entre cero."
    except (SyntaxError, ValueError) as e:
        return f"Expresión no válida: {e}"
    return f"{result:g}" if isinstance(result, float) else str(result)


# ---------- Herramientas ----------
@tool
def get_current_datetime() -> str:
    """Devuelve la fecha y la hora actuales. Úsala cuando pregunten qué hora, día o fecha es."""
    return datetime.now().strftime("%A %d/%m/%Y, %H:%M")


@tool
def calculate(expression: str) -> str:
    """Resuelve una operación matemática. Úsala SIEMPRE que haya que calcular algo.
    Recibe números y + - * / % ( ), por ejemplo "23 * 47" o "340 * 15 / 100"."""
    return safe_calculate(expression)


@tool
def web_search(query: str) -> str:
    """Busca información actual en internet (noticias, clima, datos recientes). Devuelve hasta 3 resultados."""
    try:
        results = DDGS().text(query, max_results=3)
    except Exception as e:
        return f"La búsqueda falló: {e}"
    if not results:
        return "No encontré resultados."
    return "\n\n".join(f"{r['title']}\n{r['body']}" for r in results)


TOOLS = [get_current_datetime, calculate, web_search]


# ---------- Main ----------
if __name__ == "__main__":
    for t in TOOLS:
        print(f"🔧 {t.name}: {t.description.splitlines()[0]}")
    print()
    print("Fecha:", get_current_datetime.invoke({}))
    for expression in ("23 * 47", "340 * 15 / 100", "-(2 + 3) % 4", "1 / 0", "__import__('os')", "2 ** 9999"):
        print(f"{expression!r:22} -> {calculate.invoke({'expression': expression})}")
    print()
    print(web_search.invoke({"query": "Voicebox jamiepine github"})[:300])
