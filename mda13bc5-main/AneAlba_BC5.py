# ============================================================
# CABECERA
# ============================================================
"""
============================================================
Nombre: Ane Alba
Proyecto: Spotify Analytics Assistant

URL Streamlit: https://TU-APP.streamlit.app
URL GitHub: https://github.com/TU-REPO

Descripción:
Aplicación interactiva que permite analizar hábitos de escucha
de Spotify mediante lenguaje natural, generando código dinámico
en Python para responder a las consultas del usuario.
============================================================
"""
# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json
from pathlib import Path

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un analista de datos que genera código Python para responder preguntas sobre hábitos de escucha de Spotify.

Trabajas únicamente con un DataFrame de pandas llamado df.
No cargues archivos. No inventes datos. No inventes columnas.

Cada fila representa una reproducción de una canción.

Columnas disponibles en df:
- ts: timestamp
- track_name: nombre de la canción
- artist_name: nombre del artista
- album_name: nombre del álbum
- ms_played: duración en milisegundos
- minutes_played: duración en minutos
- hours_played: duración en horas
- platform: dispositivo o plataforma
- shuffle: booleano
- skipped_clean: booleano
- skip_label: "Saltadas" o "No saltadas"
- reason_start: motivo de inicio
- reason_end: motivo de finalización
- year: año
- month: número de mes
- month_name_es: nombre del mes en español
- day: día del mes
- hour: hora del día
- weekday: número de día de la semana
- weekday_name_es: nombre del día en español
- is_weekend: True si es fin de semana
- week_part: "Entre semana" o "Fin de semana"
- semester: "H1" o "H2"
- season: "Invierno", "Primavera", "Verano" u "Otoño"

Rango temporal del dataset: desde {fecha_min} hasta {fecha_max}.
Plataformas disponibles: {plataformas}.
Valores de reason_start: {reason_start_values}.
Valores de reason_end: {reason_end_values}.

Debes responder SIEMPRE con un único objeto JSON válido y nada más.
No incluyas texto antes ni después.
No uses markdown.
No uses bloques de código.
Usa siempre comillas dobles en el JSON.
La interpretacion debe ser texto plano final en español.
Nunca uses Python dentro de interpretacion.
Nunca uses format, f-strings, variables ni expresiones dentro de interpretacion.

El formato debe ser exactamente uno de estos dos:

{{
  "tipo": "grafico",
  "codigo": "...",
  "interpretacion": "..."
}}

o

{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "..."
}}

Reglas:
- El código debe usar solo pandas y plotly, px o go.
- El código debe crear una variable final llamada fig.
- No uses matplotlib.
- No uses print.
- No uses subplots.
- Prioriza código simple y robusto.
- Usa títulos, ejes y etiquetas en español.
- Ignora nulos cuando sea necesario.
- Si la pregunta no se puede responder con este dataset, devuelve "fuera_de_alcance".

Interpretación de preguntas:
- más escuchado sin especificar unidades significa por número de reproducciones.
- en horas significa usar hours_played.
- tiempo de escucha significa sumar minutes_played o hours_played.
- top significa ordenar de mayor a menor.
- Para canciones, usa track_name.
- Para artistas, usa artist_name.
- Para meses, usa month para ordenar y month_name_es para mostrar.
- Para días de la semana, usa weekday para ordenar y weekday_name_es para mostrar.
- Para entre semana vs fin de semana, usa week_part.
- Para H1 vs H2, usa semester.
- Para estaciones, usa season.

Reglas de visualización:
- Rankings: usar gráfico de barras.
- Evolución temporal: usar gráfico de líneas o barras.
- Comparativas simples: usar barras.

- Ordena siempre de mayor a menor en rankings.
- Si se muestran horas del día, usar hour en orden ascendente.
- Si se muestran meses, usar month para ordenar y month_name_es para etiquetar.

- Mantén un diseño simple, limpio y consistente entre gráficos.
- No cambies colores por defecto si no es necesario.
- Evita templates adicionales salvo que sea necesario.

- Usa títulos claros y en español.
- Los ejes deben estar correctamente etiquetados en español.

- Evita annotations complejas, subplots y elementos innecesarios.
- Prioriza claridad visual frente a decoración.

- Para gráficos con muchas categorías (por ejemplo meses):
  - evita mostrar etiquetas de datos si afectan a la legibilidad
  - redondea valores si decides mostrarlos
  - prioriza un gráfico limpio sin saturación visual

- Para destacar valores máximos:
  - usa una columna auxiliar (por ejemplo "destacado") con valores "Máximo" y "Resto"
  - utiliza color solo si aporta claridad

Caso especial 1:
Si la pregunta es sobre a qué hora escucho más música, agrupa por hour, usa un gráfico de barras y destaca la hora máxima con una columna auxiliar llamada destacado.

Caso especial 2:
Si la pregunta es sobre qué porcentaje de canciones salto, cuenta cuántas reproducciones hay en cada valor de skip_label, crea un gráfico de barras simple con dos categorías, Saltadas y No saltadas, y escribe una interpretación descriptiva sin fórmulas ni código.

La interpretación:
- es obligatoria
- debe estar en español
- debe tener 1 o 2 frases
- debe responder directamente a la pregunta
- debe incluir el hallazgo principal
- debe sonar natural y clara, no robótica
- si es posible, debe mencionar el valor o tendencia más relevante observado
- no debe incluir cálculos, código ni expresiones dinámicas
"""

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "streaming_history.json"
    df = pd.read_json(json_path)

    # Eliminar episodios/podcasts y registros sin canción
    df = df[df["master_metadata_track_name"].notna()].copy()

    # Timestamp
    df["ts"] = pd.to_datetime(df["ts"])

    # Variables temporales
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month

    meses_es = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    df["month_name_es"] = df["month"].map(meses_es)

    df["day"] = df["ts"].dt.day
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.weekday

    dias_es = {
        0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
        4: "Viernes", 5: "Sábado", 6: "Domingo"
    }
    df["weekday_name_es"] = df["weekday"].map(dias_es)

    df["is_weekend"] = df["weekday"] >= 5
    df["week_part"] = df["is_weekend"].map({True: "Fin de semana", False: "Entre semana"})

    # Semestre
    df["semester"] = df["month"].apply(lambda x: "H1" if x <= 6 else "H2")

    # Estación
    def get_season(month):
        if month in [12, 1, 2]:
            return "Invierno"
        elif month in [3, 4, 5]:
            return "Primavera"
        elif month in [6, 7, 8]:
            return "Verano"
        else:
            return "Otoño"

    df["season"] = df["month"].apply(get_season)

    # Duración
    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"] = df["ms_played"] / 3600000

    # Booleans limpios
    df["skipped_clean"] = df["skipped"].fillna(False)
    df["skip_label"] = df["skipped_clean"].map({True: "Saltadas", False: "No saltadas"})

    # Nombres simplificados para facilitar el código del LLM
    df["track_name"] = df["master_metadata_track_name"]
    df["artist_name"] = df["master_metadata_album_artist_name"]
    df["album_name"] = df["master_metadata_album_album_name"]

    orden_meses = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]

    df["month_name_es"] = pd.Categorical(df["month_name_es"], categories=orden_meses, ordered=True)

    orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    df["weekday_name_es"] = pd.Categorical(df["weekday_name_es"], categories=orden_dias, ordered=True)

    return df

def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La aplicación sigue una arquitectura text-to-code en la que el usuario introduce una pregunta en lenguaje natural.
#    El LLM recibe únicamente la pregunta y un system prompt con instrucciones sobre cómo generar código Python válido.
#    Como salida, el LLM devuelve un JSON con código que utiliza pandas y plotly para construir un gráfico (fig).
#    Este código se ejecuta mediante exec() dentro de un entorno controlado que contiene df, pd, px y go.
#    El LLM no recibe los datos directamente para evitar problemas de privacidad, reducir el tamaño del input
#    y garantizar que el análisis se realice sobre el dataset real cargado en la aplicación.

#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    El system prompt define la estructura de salida (JSON), las librerías permitidas y las reglas de visualización.
#    También incluye casos especiales, como el cálculo del porcentaje de canciones saltadas.
#    Por ejemplo, la pregunta “¿Qué porcentaje de canciones salto?” funciona correctamente porque el prompt
#    especifica que debe usar value_counts(normalize=True) para calcular proporciones.
#    Sin esta instrucción, el modelo tendería a devolver conteos absolutos o código incorrecto.
#    Además, se restringe que no incluya cálculos dinámicos en el JSON, lo que evita errores de parsing.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    El usuario introduce una pregunta en la interfaz de Streamlit.
#    La aplicación envía la pregunta junto con el system prompt al LLM.
#    El LLM devuelve un JSON con el código y una interpretación en texto.
#    El código se parsea y se ejecuta mediante exec() en un entorno con acceso al dataframe.
#    Se obtiene una figura de Plotly (fig) que se muestra en la app.
#    Finalmente, se presenta el gráfico junto con una interpretación breve en lenguaje natural.