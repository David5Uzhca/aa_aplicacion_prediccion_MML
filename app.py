# ====================================================================
# app.py - Sistema Completo con LOGS EXHAUSTIVOS
# ====================================================================

import json
import logging
import os
import re
import sys
import time
import requests
import json
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from io import StringIO

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from logging.handlers import RotatingFileHandler


# ====================================================================
# 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ====================================================================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# ====================================================================
# 2. CONFIGURACIÓN DE LOGGING (PRIMERO, ANTES DE USAR logger EN NINGÚN LADO)
# ====================================================================
LOG_FILE = BASE_DIR / "app_logs.log"

logger = logging.getLogger('my_api_logger')
logger.setLevel(logging.INFO)
logger.propagate = False

# Handler para archivo (con rotación)
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=1024*1024, backupCount=3, encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Consola solo en desarrollo
if os.environ.get("ENV") != "production":
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Sistema de Logging Centralizado inicializado correctamente.")


# ====================================================================
# 3. CREACIÓN DE CARPETAS (AHORA SÍ PUEDES USAR logger)
# ====================================================================
for d in [MODEL_DIR, TEMPLATES_DIR, STATIC_DIR]:
    d.mkdir(exist_ok=True)
    logger.info(f"Directorio asegurado: {d}")


# ====================================================================
# 4. CREACIÓN DE LA APP FASTAPI
# ====================================================================
app = FastAPI(title="Supermercado El Despensa - IA para Inventario")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger.info("FastAPI app creada y archivos estáticos montados en /static.")


# ====================================================================
# 3. CONFIGURACIÓN DE LOGGING CENTRALIZADO
# ====================================================================
LOG_FILE = BASE_DIR / "app_logs.log"
logger = logging.getLogger('my_api_logger')
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=1024*1024, backupCount=3, encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if os.environ.get("ENV") != "production":
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Sistema de Logging Centralizado inicializado.")


# ====================================================================
# 4. CONFIGURACIÓN GLOBAL Y CONSTANTES
# ====================================================================

VERTEX_PROJECT_ID = "prediccion-478120"
VERTEX_REGION = "us-central1"
VERTEX_MODEL = "gemini-2.5-flash"

EMPRESA_INFO = {
    "nombre": "Supermercado El Despensa",
    "nicho": "Cadena de Supermercados con enfoque en productos frescos, calidad y precios competitivos.",
    "fundacion": 1998,
    "mision": "Ofrecer la mejor calidad en alimentos frescos y abarrotes, facilitando la vida de nuestros clientes con precios justos y variedad.",
    "horario_tienda": "Lunes a Sábado: 8:00 AM - 9:00 PM. Domingos: 9:00 AM - 7:00 PM.",
    "contacto": "atencionalcliente@eldespensa.com o 1800-Despensa (1800-337736)"
}
logger.info("Información de la empresa cargada.")

FAQS = [
    "¿Cuántos productos manejan en promedio? Manejamos más de 18,000 SKU (productos) en nuestras tiendas principales.",
    "¿Cuál es el horario de atención? Nuestras tiendas están abiertas de Lunes a Sábado de 8:00 AM a 9:00 PM, y los Domingos de 9:00 AM a 7:00 PM.",
    "¿Tienen servicio a domicilio? Sí, ofrecemos servicio a domicilio gratuito en compras mayores a $50 en zonas de cobertura.",
    "¿Cómo puedo aplicar a un puesto de trabajo? Visita nuestra sección 'Trabaja con Nosotros' en la web o envía tu CV a rrhh@eldespensa.com.",
    "¿Qué tipo de modelo de IA usan para predecir el stock? Utilizamos una Red Neuronal Recurrente (RNN) con arquitectura LSTM (Long Short-Term Memory) para optimizar el inventario de productos frescos.",
    "¿Cuál es el umbral de reabastecimiento que utiliza el sistema? El umbral es de 20 unidades, si el stock es menor o igual a 20, se marca la alerta."
]
logger.info(f"FAQs cargadas: {len(FAQS)} entradas.")

FEATURE_COLS = [
    "quantity_on_hand", "quantity_reserved", "quantity_available",
    "average_daily_usage", "reorder_point", "optimal_stock_level",
    "unit_cost", "total_value", "days_since_last_order", 
    "days_since_last_count", "days_to_expiration", "month", "day_of_week"
]
logger.info(f"Columnas de features definidas: {len(FEATURE_COLS)} columnas.")

MODEL_PATH = MODEL_DIR / "best_model.keras"
SCALER_X_PATH = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"
DATASET_PKL = MODEL_DIR / "dataset_limpio_quantity_on_hand.csv"
PRODUCTS_CSV = MODEL_DIR / "products.csv"

EXPECTED_INPUT_SHAPE = (7, 13)
logger.info("Configuración de rutas de modelos y datos completada.")

# Variables globales
MODEL = None
SCALER_X = None
SCALER_Y = None
DF_CLEAN = None
DF_PRODUCTS = None
ALL_PRODUCT_IDS = []
DISPLAY_PRODUCT_IDS = []
VERTEX_CLIENT_READY = False
llm_rag = None
llm_tool = None


# ====================================================================
# 5. DEFINICIÓN DE ESQUEMAS PYDANTIC (para Function Calling)
# ====================================================================

class NuevaFuncion1Input(BaseModel):
    detalle: str = Field(..., description="Detalle del reporte a generar")
    logger.info("Esquema NuevaFuncion1Input definido.")

class NuevaFuncion2Input(BaseModel):
    departamento: str = Field(..., description="Nombre del departamento")
    logger.info("Esquema NuevaFuncion2Input definido.")

class PrediccionProductoInput(BaseModel):
    product_id: str = Field(..., description="ID exacto del producto (ej: pdct0015).")
    target_date: str = Field(..., description="Fecha futura en formato YYYY-MM-DD.")
    logger.info("Esquema PrediccionProductoInput definido.")

class PrediccionGeneralInput(BaseModel):
    target_date: str = Field(..., description="Fecha futura en formato YYYY-MM-DD.")
    logger.info("Esquema PrediccionGeneralInput definido.")

class AgregarRegistrosInput(BaseModel):
    datos_externos: str = Field(..., description="Indica si los datos vienen de tabla o CSV.")
    logger.info("Esquema AgregarRegistrosInput definido.")

class AjustarModeloInput(BaseModel):
    fuente_datos: str = Field(..., description="Fuente de datos para reentrenar.")
    logger.info("Esquema AjustarModeloInput definido.")


# ====================================================================
# 6. HERRAMIENTAS (FUNCTIONS) PARA EL CHATBOT
# ====================================================================

def nueva_funcion_1(detalle: str) -> str:
    logger.info(f"Iniciando nueva_funcion_1 con detalle: {detalle}")
    result = f"Iniciando la generación del reporte de ventas detallado para {detalle}. Por favor, espere 5 minutos."
    logger.info(f"Resultado de nueva_funcion_1: {result}")
    return result

def nueva_funcion_2(departamento: str) -> str:
    logger.info(f"Iniciando nueva_funcion_2 con departamento: {departamento}")
    result = f"Consultando el historial de inventario del departamento de '{departamento}'. Resumen: El historial muestra alta rotación en las últimas 4 semanas."
    logger.info(f"Resultado de nueva_funcion_2: {result}")
    return result

LOCAL_API_BASE_URL = "http://10.128.0.2:8000"
def predecir_stock_producto(product_id: str, target_date: str) -> str:    
    url = f"{LOCAL_API_BASE_URL}/api/predict"
    payload = {
        "product_id": product_id,
        "date": target_date
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        stock_predicho = data.get('quantity_on_hand', 0)
        nombre_producto = data.get('product_name', product_id)
        fecha_predicha = data.get('fecha_predicha', target_date)
        
        umbral = 20 # El umbral de reabastecimiento es 20 unidades
        
        if stock_predicho <= umbral:
            conclusion = f"El producto {nombre_producto} ({product_id}) cuenta con un stock predicho de {stock_predicho:.2f} para la fecha {fecha_predicha}, por lo que REQUIERE REABASTECIMIENTO URGENTE."
        else:
            conclusion = f"El producto {nombre_producto} ({product_id}) cuenta con un stock predicho de {stock_predicho:.2f} para la fecha {fecha_predicha}, por lo que no requiere reabastecimiento en este momento."
            
        return conclusion
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: No se encontraron datos históricos o el producto {product_id} no existe. [Código 404]"
        return f"Error HTTP al llamar al servicio de predicción: {e}"
    
    except requests.exceptions.RequestException as e:
        return f"Error de conexión: No se pudo conectar al endpoint de predicción ({LOCAL_API_BASE_URL}). Asegúrese de que Uvicorn esté escuchando."
    except Exception as e:
        return f"Error inesperado al procesar la predicción: {e}"

def predecir_stock_general(target_date: str) -> str:
    logger.info(f"Iniciando predecir_stock_general con target_date: {target_date}")
    result = f"Activando la predicción general para todos los productos principales para la fecha {target_date}. Revisa la tabla de reabastecimiento."
    logger.info(f"Resultado de predecir_stock_general: {result}")
    return result

def agregar_nuevos_registros(datos_externos: str) -> str:
    logger.info(f"Iniciando agregar_nuevos_registros con datos_externos: {datos_externos}")
    result = "Preparado para añadir nuevos registros. Usa la interfaz web para subir CSV o confirmar datos de tabla."
    logger.info(f"Resultado de agregar_nuevos_registros: {result}")
    return result

def actualizar_modelo(fuente_datos: str) -> str:
    logger.info(f"Iniciando actualizar_modelo con fuente_datos: {fuente_datos}")
    result = f"El modelo se actualizará usando datos de {fuente_datos}. Revisa el log de reentrenamiento."
    logger.info(f"Resultado de actualizar_modelo: {result}")
    return result

TOOLS = [
    predecir_stock_producto,
    predecir_stock_general,
    agregar_nuevos_registros,
    actualizar_modelo,
    nueva_funcion_1,
    nueva_funcion_2,
]
logger.info(f"Herramientas para chatbot definidas: {len(TOOLS)} herramientas.")


# ====================================================================
# 7. LÓGICA DEL CHATBOT (Completa con Logging Detallado)
# ====================================================================

def responder_basico(query: str, log_entries: List[str]) -> str:
    logger.info(f"Iniciando responder_basico con query: {query}")
    log_entries.append("  -> LÓGICA BÁSICA: Ejecutando verificación de Regex (saludos/despedidas).")
    query_lower = query.lower().strip()

    patrones_saludo = r"^(hola|buen(o?s)? d(i|í)as|buenas tardes|qu(e|é) tal|saludos)"
    patrones_agradecimiento = r"^(gracias|muchas gracias|te lo agradezco|genial|perfecto)"
    patrones_despedida = r"^(adi(o|ó)s|chao|hasta luego|me despido|bye|nos vemos)"

    if re.search(patrones_saludo, query_lower):
        log_entries.append("  -> DECISIÓN: Coincidencia con 'Saludo'.")
        logger.info("Coincidencia con saludo detectada.")
        hora = datetime.now().hour
        if 5 <= hora < 12:
            momento = "Buenos días"
        elif 12 <= hora < 19:
            momento = "Buenas tardes"
        else:
            momento = "Buenas noches"
        result = f"{momento}, soy {EMPRESA_INFO['nombre']}. ¿En qué puedo ayudarte hoy con tu inventario y gestión?"
        logger.info(f"Respuesta básica generada: {result}")
        return result
    
    if re.search(patrones_agradecimiento, query_lower):
        log_entries.append("  -> DECISIÓN: Coincidencia con 'Agradecimiento'.")
        logger.info("Coincidencia con agradecimiento detectada.")
        result = "¡Para eso estamos! Me da gusto ayudarte. ¿Necesitas algo más?"
        logger.info(f"Respuesta básica generada: {result}")
        return result

    if re.search(patrones_despedida, query_lower):
        log_entries.append("  -> DECISIÓN: Coincidencia con 'Despedida'.")
        logger.info("Coincidencia con despedida detectada.")
        result = f"¡Adiós! Que tengas un excelente día. ¡Vuelve pronto!"
        logger.info(f"Respuesta básica generada: {result}")
        return result
    
    log_entries.append("  -> DECISIÓN: No se encontró coincidencia básica.")
    logger.info("No se encontró coincidencia básica en responder_basico.")
    return ""


def responder_faqs(query: str, log_entries: List[str]) -> str:
    logger.info(f"Iniciando responder_faqs con query: {query}")
    log_entries.append("  -> LÓGICA RAG/FAQ: Ejecutando modelo Gemini para consulta de FAQs.")
    global llm_rag
    
    if llm_rag is None:
        result = "Disculpe, el servicio de IA (Gemini) no está disponible en este momento."
        logger.warning("LLM RAG no disponible.")
        return result

    contexto = "\n".join(FAQS)
    
    # Simulación de ranking de FAQs por coincidencia simple (palabras clave)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    logger.info("Calculando ranking de FAQs por similitud TF-IDF.")
    vectorizer = TfidfVectorizer()
    faq_vectors = vectorizer.fit_transform(FAQS + [query])
    similarities = cosine_similarity(faq_vectors[-1], faq_vectors[:-1]).flatten()
    ranked_indices = similarities.argsort()[::-1]
    ranked_faqs = [(FAQS[i], similarities[i]) for i in ranked_indices]
    log_entries.append("  -> RANKING DE FAQs POR COINCIDENCIA:")
    for idx, (faq, score) in enumerate(ranked_faqs):
        log_entries.append(f"    - FAQ {idx+1} (score: {score:.4f}): {faq}")
    logger.info(f"Ranking de FAQs calculado: Top score {ranked_faqs[0][1]:.4f}")

    # Selección: Usar el top 3 para el contexto del prompt
    selected_faqs = "\n".join([faq for faq, _ in ranked_faqs[:3]])
    log_entries.append(f"  -> SELECCIÓN: Usando top 3 FAQs para el prompt (scores: {[s for _, s in ranked_faqs[:3]]}).")

    prompt_texto = f"""
    Eres un asistente de soporte de {EMPRESA_INFO['nombre']}. Utiliza la siguiente información de la empresa y las FAQs para responder a la pregunta del usuario. 
    Si la respuesta a la pregunta no está en el contexto, indica amablemente que no tienes la información.
    
    --- Información de la Empresa ---
    Nombre: {EMPRESA_INFO['nombre']} (Fundada en {EMPRESA_INFO['fundacion']}). Misión: {EMPRESA_INFO['mision']}
    Horario de atención: {EMPRESA_INFO['horario_tienda']}
    
    --- Base de Conocimiento (FAQs seleccionadas por relevancia) ---
    {selected_faqs}
    
    --- Pregunta del Usuario ---
    {query}
    """
    
    log_entries.append("  -> PROMPT ENVIADO AL LLM:")
    log_entries.append(prompt_texto)
    logger.info("Prompt para RAG preparado y enviado.")
    
    prompt_template = ChatPromptTemplate.from_template("{prompt_texto}")
    chain_rag = prompt_template | llm_rag | StrOutputParser()
    
    try:
        response = chain_rag.invoke({"prompt_texto": prompt_texto})
        log_entries.append("  -> DECISIÓN: Respuesta generada por RAG/LLM.")
        logger.info(f"Respuesta RAG generada: {response[:100]}...")
        return response
    except Exception as e:
        log_entries.append(f"  -> ERROR RAG: Fallo en la invocación de LangChain/Gemini. Detalle: {e}")
        logger.error(f"Error en RAG: {e}")
        return "Disculpe, ocurrió un error al consultar la base de conocimiento."


def responder_tool_calling(query: str, log_entries: List[str]) -> str:
    logger.info(f"Iniciando responder_tool_calling con query: {query}")
    log_entries.append("  -> LÓGICA FUNCTION CALLING: Ejecutando modelo Gemini para detección de herramienta.")

    global llm_tool
    
    if llm_tool is None:
        result = "Disculpe, el servicio de Function Calling no está disponible."
        logger.warning("LLM Tool no disponible.")
        return result
    
    try:
        response = llm_tool.invoke([HumanMessage(content=query)])
        logger.info("Respuesta de LLM Tool obtenida.")
    except Exception as e:
        log_entries.append(f"  -> ERROR TOOL CALLING: Fallo en la invocación de LangChain/Gemini. Detalle: {e}")
        logger.error(f"Error en tool calling: {e}")
        return "" # Para que pase al RAG

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        log_entries.append(f"  -> DECISIÓN: Se detectó llamada a función.")
        log_entries.append(f"    - FUNCIÓN DETECTADA: {tool_name}")
        log_entries.append(f"    - ARGUMENTOS EXTRAÍDOS: {tool_args}")
        logger.info(f"Tool call detectada: {tool_name} con args {tool_args}")
        
        # Ejecutar la función
        for tool_function in TOOLS:
            if tool_function.__name__ == tool_name:
                try:
                    result = tool_function(**tool_args)
                    log_entries.append(f"  -> RESULTADO FUNCIÓN: Éxito. {result}")
                    logger.info(f"Función {tool_name} ejecutada con éxito: {result}")
                    return f"✅ FUNCIÓN LLAMADA: {tool_name}. Mensaje de éxito: {result}"
                except Exception as e:
                    log_entries.append(f"  -> RESULTADO FUNCIÓN: Error al ejecutar la función de Python. Detalle: {e}")
                    logger.error(f"Error ejecutando {tool_name}: {e}")
                    return f"Error: La función {tool_name} falló al ejecutarse. Detalle: {e}"
        
        log_entries.append(f"  -> RESULTADO FUNCIÓN: La función {tool_name} no fue encontrada en el código.")
        logger.warning(f"Función {tool_name} no encontrada.")
        return f"Error: La función {tool_name} fue identificada pero no se pudo ejecutar."

    log_entries.append("  -> DECISIÓN: No se detectó ninguna llamada a función. Pasando a RAG/FAQ.")
    logger.info("No se detectó tool call, pasando a RAG.")
    return "" # No se identificó llamada a herramienta


def main_chatbot(query: str, log_entries: List[str]) -> str:
    logger.info(f"Iniciando main_chatbot con query: {query}")
    log_entries.append(f"-> ENTRADA DE TEXTO: {query}")

    # 1. Respuestas Básicas
    log_entries.append("-> PASO 1: Verificación de Respuestas Básicas (Saludo/Despedida).")
    logger.info("Iniciando verificación de respuestas básicas.")
    respuesta_basica = responder_basico(query, log_entries)
    
    if respuesta_basica:
        log_entries.append("-> FLUJO FINAL: Retornando Respuesta Básica.")
        logger.info(f"Respuesta básica seleccionada: {respuesta_basica}")
        return respuesta_basica

    # 2. Función Tool Calling
    log_entries.append("-> PASO 2: Verificación de Function Calling.")
    logger.info("Iniciando verificación de function calling.")
    respuesta_tool = responder_tool_calling(query, log_entries)
    
    if respuesta_tool:
        log_entries.append("-> FLUJO FINAL: Retornando Resultado de Function Calling.")
        logger.info(f"Respuesta tool calling seleccionada: {respuesta_tool}")
        return respuesta_tool
    
    # 3. Respuesta RAG/FAQ
    log_entries.append("-> PASO 3: Ejecutando Lógica RAG/FAQ.")
    logger.info("Iniciando lógica RAG/FAQ.")
    respuesta_rag = responder_faqs(query, log_entries)
    log_entries.append("-> FLUJO FINAL: Retornando Respuesta RAG/FAQ.")
    logger.info(f"Respuesta RAG/FAQ seleccionada: {respuesta_rag}")
    
    return respuesta_rag


# ====================================================================
# 8. CARGA DE RECURSOS AL INICIO (startup)
# ====================================================================

@app.on_event("startup")
async def startup_event():
    global MODEL, SCALER_X, SCALER_Y, DF_CLEAN, DF_PRODUCTS
    global ALL_PRODUCT_IDS, DISPLAY_PRODUCT_IDS, VERTEX_CLIENT_READY, llm_rag, llm_tool

    logger.info("Iniciando carga de recursos en startup_event.")
    try:
        logger.info(f"Cargando modelo desde {MODEL_PATH}")
        MODEL = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info(f"Cargando scaler_X desde {SCALER_X_PATH}")
        SCALER_X = pickle.load(open(SCALER_X_PATH, "rb"))
        logger.info(f"Cargando scaler_Y desde {SCALER_Y_PATH}")
        SCALER_Y = pickle.load(open(SCALER_Y_PATH, "rb"))
        logger.info("Modelo y scalers cargados.")
        
        logger.info(f"Cargando dataset histórico desde {DATASET_PKL}")
        DF_CLEAN = pd.read_csv(DATASET_PKL, parse_dates=["created_at"])
        DF_CLEAN = DF_CLEAN.sort_values(["product_id", "created_at"]).reset_index(drop=True)
        logger.info(f"Dataset histórico cargado: {len(DF_CLEAN)} filas.")
        
        logger.info(f"Cargando productos desde {PRODUCTS_CSV}")
        DF_PRODUCTS = pd.read_csv(PRODUCTS_CSV)
        logger.info(f"Productos cargados: {len(DF_PRODUCTS)} entradas.")
        
        set1 = set(DF_CLEAN['product_id'].unique())
        logger.info(f"Productos únicos en DF_CLEAN: {len(set1)}")
        set2 = set(DF_PRODUCTS['id'].unique())
        logger.info(f"Productos únicos en DF_PRODUCTS: {len(set2)}")
        ALL_PRODUCT_IDS = sorted(list(set1 & set2))
        DISPLAY_PRODUCT_IDS = ALL_PRODUCT_IDS[:30]
        logger.info(f"Productos comunes disponibles: {len(ALL_PRODUCT_IDS)}, display: {len(DISPLAY_PRODUCT_IDS)}")
        
        logger.info("Inicializando Vertex AI.")
        aiplatform.init(project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
        VERTEX_CLIENT_READY = True
        logger.info("Vertex AI inicializado.")
        
        logger.info("Inicializando LLM RAG.")
        llm_rag = ChatVertexAI(model_name=VERTEX_MODEL, temperature=0.0, project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
        logger.info("Inicializando LLM Tool con bind_tools.")
        llm_tool = ChatVertexAI(model_name=VERTEX_MODEL, project=VERTEX_PROJECT_ID, location=VERTEX_REGION).bind_tools(TOOLS)
        logger.info("LLMs inicializados.")

        logger.info("Todos los recursos cargados correctamente al iniciar.")
    except Exception as e:
        logger.critical(f"Error al cargar recursos: {e}")


# ====================================================================
# 9. FUNCIONES DE ML (Predicción y Reentrenamiento)
# ====================================================================

def get_last_7_rows(product_id: str):
    logger.info(f"Iniciando get_last_7_rows para product_id: {product_id}")
    if DF_CLEAN is None: 
        logger.warning("DF_CLEAN no cargado.")
        return None, None
    
    prod = DF_CLEAN[DF_CLEAN["product_id"].str.strip() == product_id.strip()].tail(7)
    logger.info(f"Filas obtenidas para {product_id}: {len(prod)}")
    
    X = prod[FEATURE_COLS].values.astype(np.float32)
    
    if len(prod) == 0:
        logger.warning(f"No hay datos para {product_id}.")
        return None, None

    if X.shape[0] < 7:
        t_exp = 7
        f_exp = X.shape[1] 
        pad = np.zeros((t_exp - X.shape[0], f_exp))
        X = np.vstack([pad, X])
        logger.info(f"Rellenado de {product_id} con {t_exp - X.shape[0]} filas de cero.")
    
    last_date = prod["created_at"].iloc[-1]
    logger.info(f"Última fecha para {product_id}: {last_date}")
    return X, last_date


def predict_stock_by_date(product_id: str, target_date_str: str):
    logger.info(f"Iniciando predict_stock_by_date para {product_id} en {target_date_str}")
    if MODEL is None or SCALER_X is None or SCALER_Y is None:
        logger.warning("Modelo o scalers no cargados.")
        return None
        
    window, last_date = get_last_7_rows(product_id)
    if window is None:
        logger.warning(f"Ventana no disponible para {product_id}.")
        return None
        
    target = pd.to_datetime(target_date_str)
    logger.info(f"Fecha objetivo: {target}, última fecha: {last_date}")
    current = window.copy()
    t_exp, f_exp = EXPECTED_INPUT_SHAPE
    
    if target <= last_date:
        result = max(0, round(float(window[-1, 0]), 2))
        logger.info(f"Predicción para fecha pasada: {result}")
        return result
        
    days = (target - last_date).days
    logger.info(f"Días a predecir: {days}")
    
    if window.shape[0] != t_exp or window.shape[1] != f_exp:
        logger.error(f"Forma de ventana incorrecta: {window.shape}")
        return None
    
    last_pred = window[-1, 0] 
    trend = current[-1] - current[-t_exp] 
    logger.info(f"Tendencia inicial calculada: {trend}")
    
    for day in range(days):
        logger.info(f"Prediciendo día {day+1}/{days}")
        X_in = SCALER_X.transform(current).reshape((1, t_exp, f_exp)).astype(np.float32)
        pred_scaled = MODEL.predict(X_in, verbose=0)[0][0]
        pred = SCALER_Y.inverse_transform([[pred_scaled]])[0][0]
        logger.info(f"Predicción escalada: {pred_scaled}, desescalada: {pred}")

        new_row = current[-1].copy()
        new_row[0] = pred 

        new_row[1:] = current[-1][1:] + trend[1:] * 0.08 
        logger.info(f"Nueva fila generada: {new_row[:5]}...")
        
        current = np.vstack([current[1:], new_row])
        last_pred = pred

    result = max(0, round(float(last_pred), 2))
    logger.info(f"Predicción final: {result}")
    return result


def retrain_model(new_data_df: pd.DataFrame) -> dict:
    logger.info("Iniciando retrain_model.")
    global MODEL, SCALER_X, SCALER_Y, DF_CLEAN, EXPECTED_INPUT_SHAPE, FEATURE_COLS
    
    if MODEL is None or DF_CLEAN is None:
        logger.error("Modelo o DF_CLEAN no cargados.")
        return {"success": False, "log": "ERROR: El modelo o DF_CLEAN no están cargados."}

    # Manejar renombramiento de fecha
    logger.info("Manejando renombramiento de columnas en new_data_df.")
    if 'fecha_predicha' in new_data_df.columns:
        new_data_df = new_data_df.copy() 
        new_data_df.rename(columns={'fecha_predicha': 'created_at'}, inplace=True)
        logger.info("Renombrada 'fecha_predicha' a 'created_at'.")
    
    if 'created_at' not in new_data_df.columns:
        logger.error("Columna 'created_at' no presente en new_data_df.")
        return {"success": False, "log": "ERROR: El DataFrame de entrada no contiene la columna 'created_at'."}
    
    new_rows = []
    logger.info(f"Procesando {len(new_data_df)} filas nuevas.")
    
    for idx, row in new_data_df.iterrows():
        logger.info(f"Procesando fila {idx+1}/{len(new_data_df)}")
        pid = row['product_id']
        pred_date = pd.to_datetime(row['created_at'])
        pred_stock = row['quantity_on_hand']
        logger.info(f"Producto: {pid}, Fecha: {pred_date}, Stock: {pred_stock}")

        last_row_data = DF_CLEAN[DF_CLEAN['product_id'] == pid].tail(1)
        logger.info(f"Última fila histórica para {pid}: {len(last_row_data)} entradas.")

        if not last_row_data.empty:
            last_row = last_row_data[FEATURE_COLS].values[0]
            last_date = last_row_data['created_at'].iloc[0]
            logger.info(f"Última fecha histórica: {last_date}")

            days_to_add = (pred_date - last_date).days
            logger.info(f"Días a añadir: {days_to_add}")
            
            if days_to_add <= 0:
                logger.warning(f"Fecha no futura, saltando fila {idx+1}.")
                continue

            simulated_row = last_row.copy()
            simulated_row[0] = pred_stock 
            new_data = {
                'product_id': pid,
                'created_at': pred_date,
                **{col: val for col, val in zip(FEATURE_COLS, simulated_row)}
            }
            new_rows.append(new_data)
            logger.info(f"Nueva fila simulada añadida para {pid}.")
            
    if not new_rows:
        logger.warning("No se generaron nuevas filas.")
        return {"success": True, "log": "Advertencia: No se generaron nuevas filas para añadir (fechas pasadas/actuales)."}

    new_df = pd.DataFrame(new_rows)
    logger.info(f"Nuevo DataFrame creado con {len(new_df)} filas.")
    cols_to_keep = DF_CLEAN.columns.tolist() 
    new_df = new_df[[col for col in new_df.columns if col in cols_to_keep]] 
    logger.info("Columnas filtradas para compatibilidad.")
    DF_CLEAN = pd.concat([DF_CLEAN, new_df], ignore_index=True)
    logger.info(f"DF_CLEAN actualizado: {len(DF_CLEAN)} filas totales.")
    DF_CLEAN.to_csv(str(DATASET_PKL), index=False)
    logger.info(f"Dataset guardado en {DATASET_PKL}.")
    
    log = f"Datos añadidos: {len(new_rows)} nuevas filas. DF_CLEAN total: {len(DF_CLEAN):,}.\n"
    
    X_retrain, y_retrain = [], []
    t_exp, f_exp = EXPECTED_INPUT_SHAPE
    logger.info("Generando secuencias para reentrenamiento.")
    
    for pid in DISPLAY_PRODUCT_IDS:
        logger.info(f"Procesando producto {pid} para reentrenamiento.")
        prod = DF_CLEAN[DF_CLEAN['product_id'] == pid]
        
        if len(prod) >= t_exp + 1:
            X_input_raw = prod[FEATURE_COLS].iloc[-t_exp-1:-1].values.astype(np.float32) 
            y_target_raw = prod[FEATURE_COLS].iloc[-1, 0] 
            logger.info(f"Secuencia cruda generada para {pid}: {X_input_raw.shape}")

            if X_input_raw.shape == EXPECTED_INPUT_SHAPE:
                X_retrain.append(X_input_raw)
                y_retrain.append(y_target_raw)
                logger.info(f"Secuencia añadida para {pid}.")
                
    if not X_retrain:
        log += "Advertencia: No hay secuencias completas (7+1) para reentrenar. No se realizó el reentrenamiento.\n"
        logger.warning("No hay secuencias para reentrenar.")
        return {"success": True, "log": log}

    X_retrain = np.array(X_retrain)
    y_retrain = np.array(y_retrain).reshape(-1, 1)
    logger.info(f"Arrays de reentrenamiento: X {X_retrain.shape}, y {y_retrain.shape}")
    X_flat = X_retrain.reshape(-1, f_exp)
    X_scaled = SCALER_X.transform(X_flat).reshape(X_retrain.shape)
    logger.info("X escalado.")
    
    y_scaled = SCALER_Y.transform(y_retrain).ravel()
    logger.info("y escalado.")
    
    log += f"Secuencias de reentrenamiento generadas: {X_scaled.shape[0]}. Iniciando reentrenamiento (1 epoch)...\n"
    logger.info("Iniciando fit del modelo.")
    
    history = MODEL.fit(
        X_scaled, y_scaled,
        epochs=1,
        batch_size=32,
        verbose=0
    )
    logger.info("Fit completado.")
    
    loss = history.history['loss'][0]
    log += f"Reentrenamiento completado (1 epoch). Nueva Loss: {loss:.4f}\n"
    logger.info(f"Nueva loss: {loss:.4f}")

    MODEL.save(str(MODEL_PATH))
    log += f"Modelo guardado en: {MODEL_PATH}\n"
    logger.info(f"Modelo guardado en {MODEL_PATH}.")
    
    return {"success": True, "log": log}


def process_external_data(uploaded_df: pd.DataFrame, log: str) -> dict:
    logger.info("Iniciando process_external_data.")
    global DF_CLEAN, FEATURE_COLS, DISPLAY_PRODUCT_IDS
    
    # 1. Validación de Columnas Mínimas
    required_cols = ["product_id", "created_at", "quantity_on_hand"]
    if not all(col in uploaded_df.columns for col in required_cols):
        log += "ERROR: El CSV debe contener las columnas 'product_id', 'created_at' y 'quantity_on_hand'.\n"
        logger.error("Columnas requeridas no presentes.")
        return {"success": False, "log": log}

    log += f"Datos externos recibidos: {len(uploaded_df):,} filas.\n"
    logger.info(f"Datos recibidos: {len(uploaded_df)} filas.")

    # 2. Convertir y Limpiar
    logger.info("Limpiando y convirtiendo datos externos.")
    uploaded_df['created_at'] = pd.to_datetime(uploaded_df['created_at'], errors='coerce')
    uploaded_df = uploaded_df.dropna(subset=['created_at'])
    logger.info(f"Filas después de dropna created_at: {len(uploaded_df)}")
    uploaded_df['quantity_on_hand'] = pd.to_numeric(uploaded_df['quantity_on_hand'], errors='coerce')
    uploaded_df = uploaded_df.dropna(subset=['quantity_on_hand'])
    logger.info(f"Filas después de dropna quantity_on_hand: {len(uploaded_df)}")
    
    # 3. LLENADO DE COLUMNAS FALTANTES
    all_df_clean_cols = DF_CLEAN.columns.tolist()
    base_upload_cols = uploaded_df.columns.tolist()
    logger.info(f"Columnas en DF_CLEAN: {len(all_df_clean_cols)}, en uploaded: {len(base_upload_cols)}")
    
    cols_to_copy = [col for col in all_df_clean_cols if col not in base_upload_cols]
    logger.info(f"Columnas a copiar: {len(cols_to_copy)}")

    final_new_data = []
    
    for pid in uploaded_df['product_id'].unique():
        logger.info(f"Procesando producto {pid} en datos externos.")
        last_known_row = DF_CLEAN[DF_CLEAN['product_id'] == pid].sort_values('created_at').tail(1)
        
        if last_known_row.empty:
            log += f"Advertencia: El producto {pid} no existe en el histórico. Saltando.\n"
            logger.warning(f"Producto {pid} no en histórico, saltando.")
            continue

        new_prod_data = uploaded_df[uploaded_df['product_id'] == pid].copy()
        logger.info(f"Filas para {pid}: {len(new_prod_data)}")
        
        for col in cols_to_copy:
            try:
                new_prod_data[col] = last_known_row[col].iloc[0]
                logger.info(f"Copiada columna {col} para {pid}.")
            except IndexError as e:
                log += f"Error interno al copiar columna {col} para {pid}: {e}\n"
                logger.error(f"Error copiando {col} para {pid}: {e}")
                continue

        new_prod_data = new_prod_data[all_df_clean_cols]
        logger.info(f"Estructura ajustada para {pid}.")
        final_new_data.append(new_prod_data)
        
    if not final_new_data:
        log += "ERROR: Ningún producto en el CSV subido se pudo mapear al histórico o pasó la validación.\n"
        logger.error("Ningún producto mapeado.")
        return {"success": False, "log": log}
    
    new_df_to_add = pd.concat(final_new_data, ignore_index=True)
    logger.info(f"Nuevo DF a añadir: {len(new_df_to_add)} filas.")
    
    # Filtrar fechas duplicadas
    current_ids = set(DF_CLEAN[['product_id', 'created_at']].apply(tuple, axis=1))
    logger.info(f"IDs actuales en histórico: {len(current_ids)}")
    new_df_to_add = new_df_to_add[~new_df_to_add[['product_id', 'created_at']].apply(tuple, axis=1).isin(current_ids)]
    logger.info(f"Después de filtrar duplicados: {len(new_df_to_add)} filas.")
    
    if new_df_to_add.empty:
         log += "Advertencia: Todos los datos subidos ya existen en el histórico o son duplicados. Reentrenamiento abortado.\n"
         logger.warning("Todos los datos son duplicados, abortando reentrenamiento.")
         return {"success": True, "log": log}
         
    # Ejecutar la lógica de reentrenamiento existente
    logger.info("Ejecutando retrain_model con datos nuevos.")
    retrain_result = retrain_model(new_df_to_add)
    
    log += retrain_result['log']
    logger.info(f"Resultado de reentrenamiento: success={retrain_result['success']}")
    
    return {"success": retrain_result['success'], "log": log}


# ====================================================================
# 10. ESQUEMAS PYDANTIC PARA LOS ENDPOINTS
# ====================================================================

class PredictRequest(BaseModel):
    product_id: str
    date: Optional[str] = None
    logger.info("Esquema PredictRequest definido.")

class RestockRequest(BaseModel):
    date: str
    threshold: Optional[float] = 20.0
    logger.info("Esquema RestockRequest definido.")

class ConclusionRequest(BaseModel):
    results: List[dict]
    logger.info("Esquema ConclusionRequest definido.")

class ChatQuery(BaseModel):
    query: str
    logger.info("Esquema ChatQuery definido.")


# ====================================================================
# 11. ENDPOINTS DE LA API
# ====================================================================

@app.post("/api/retrain")
async def api_retrain(req: Request):
    logger.info("Iniciando /api/retrain.")
    try:
        new_data_list = await req.json()
        logger.info(f"Datos recibidos en JSON: {len(new_data_list)} items.")
        
        if not new_data_list or not isinstance(new_data_list, list):
            logger.error("Datos no válidos o vacíos.")
            raise HTTPException(status_code=400, detail="Datos no válidos o vacíos para el reentrenamiento.")

        new_data_df = pd.DataFrame(new_data_list)
        logger.info(f"DataFrame creado: {len(new_data_df)} filas.")
        new_data_df = new_data_df.dropna(subset=['quantity_on_hand'])
        logger.info(f"Después de dropna quantity_on_hand: {len(new_data_df)}")
        new_data_df['quantity_on_hand'] = new_data_df['quantity_on_hand'].apply(lambda x: max(0, float(x)))
        logger.info("Quantity_on_hand ajustada a valores positivos.")
        
        result = retrain_model(new_data_df)
        logger.info(f"Resultado de retrain: success={result['success']}")
        
        if not result['success']:
             logger.error(f"Reentrenamiento fallido: {result['log']}")
             raise HTTPException(status_code=500, detail=result['log'])
             
        return result

    except Exception as e:
        logger.critical(f"Error en /api/retrain: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno durante el reentrenamiento: {e}")


@app.post("/api/conclusion")
async def api_conclusion(req: ConclusionRequest):
    logger.info("Iniciando /api/conclusion.")
    global VERTEX_CLIENT_READY, VERTEX_MODEL

    if not VERTEX_CLIENT_READY:
        logger.error("Vertex AI no configurado.")
        raise HTTPException(status_code=503, detail="El cliente de Vertex AI no está configurado.")
        
    if not req.results:
        logger.warning("No hay resultados para analizar.")
        return {"conclusion": "No hay resultados para analizar."}

    data_str = "Resultados de Predicción de Stock:\n"
    data_str += "---------------------------------------------------------\n"
    data_str += "ID | Stock Predicho | Necesita Reabastecer\n"
    data_str += "---|----------------|-----------------------\n"
    
    for item in req.results:
        logger.info(f"Procesando item: {item.get('product_id')}")
        needs_restock = "Sí" if item.get('needs_restock', False) or (item.get('quantity_on_hand', 0) <= 20 and 'needs_restock' not in item) else "No"
        stock = f"{item.get('quantity_on_hand', 0):.2f}"
        data_str += f"{item.get('product_id')} | {stock} | {needs_restock}\n"

    logger.info("Data_str para prompt preparada.")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un experto analista de inventario y gestión de almacén. Tu tarea es analizar los datos de predicción "
                "de stock y generar una conclusión breve y orientada a la acción para la gerencia."
            ),
            (
                "human",
                "Analiza los siguientes datos. El umbral de reabastecimiento es 20 unidades. Genera una conclusión en españo "
		"sin frases en negrita (**frase**)"
                "cubriendo: (1) Productos críticos (stock <= 5), (2) Resumen del porcentaje de productos que necesitan reabastecimiento (stock <= 20), "
                "y (3) Una recomendación de acción breve. \n\n--- DATOS ---\n{data}"
            )
        ]
    )
    logger.info("Prompt template definido.")
    
    llm = ChatVertexAI(
        model_name=VERTEX_MODEL,
        temperature=0.2,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_REGION
    )
    logger.info("LLM para conclusión inicializado.")
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({"data": data_str})
        logger.info(f"Respuesta de cadena: {response.content[:100]}...")
        return {"conclusion": response.content}

    except Exception as e:
        logger.error(f"Error al generar conclusión: {e}")
        return {"conclusion": f"Error al generar la conclusión con LangChain. Revise el log de Uvicorn: {e}"}


@app.post("/api/upload_and_retrain")
async def api_upload_and_retrain(file: UploadFile = File(...)):
    logger.info(f"Iniciando /api/upload_and_retrain con archivo: {file.filename}")
    log = f"Procesando archivo: {file.filename}\n"
    
    if not file.filename.endswith('.csv'):
        log += "ERROR: Formato de archivo no soportado. Debe ser un archivo CSV (.csv).\n"
        logger.error("Archivo no es CSV.")
        raise HTTPException(status_code=400, detail={"log": log})
    
    try:
        # 1. Leer el contenido del archivo subido
        content = await file.read()
        logger.info("Contenido del archivo leído.")
        s = str(content, 'utf-8')
        uploaded_df = pd.read_csv(StringIO(s))
        logger.info(f"DataFrame de upload creado: {len(uploaded_df)} filas.")
        
        # 2. Lógica de Estandarización y Renombramiento
        column_map = {}
        
        # Primero, verificamos si la columna de fecha es 'fecha_predicha' y la renombramos.
        # Si ya es 'created_at', no hacemos nada.
        if 'fecha_predicha' in uploaded_df.columns:
            column_map['fecha_predicha'] = 'created_at'
            log += "Columna 'fecha_predicha' renombrada a 'created_at'.\n"
            logger.info("Renombrada 'fecha_predicha' a 'created_at'.")
        
        # 3. Aplicar el renombramiento (si existe un mapeo)
        if column_map:
            uploaded_df.rename(columns=column_map, inplace=True)
            logger.info("Renombramiento aplicado.")
        
        # 4. Verificar las columnas obligatorias después del renombramiento
        required_cols = ["product_id", "created_at", "quantity_on_hand"]
        if not all(col in uploaded_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in uploaded_df.columns]
             log += f"ERROR: Faltan columnas obligatorias después de la estandarización: {missing}.\n"
             logger.error(f"Faltan columnas: {missing}")
             raise HTTPException(status_code=400, detail={"log": log})

        # 5. Ejecutar la lógica de procesamiento y reentrenamiento (Una sola llamada)
        # La función process_external_data debe recibir el DataFrame estandarizado
        result = process_external_data(uploaded_df, log)
        logger.info(f"Resultado de process_external_data: success={result['success']}")
        
        # 6. Manejo de la Respuesta
        if not result['success']:
             logger.error(f"Procesamiento fallido: {result['log']}")
             raise HTTPException(status_code=500, detail={"log": result['log']})
             
        return result

    except HTTPException:
        raise
    except Exception as e:
        log += f"Error inesperado al procesar el archivo: {e.__class__.__name__}: {e}\n"
        logger.critical(f"Error en /api/upload_and_retrain: {e}")
        raise HTTPException(status_code=500, detail={"log": log})


@app.get("/final-predict", response_class=HTMLResponse)
async def final_predict(request: Request):
    logger.info("Iniciando /final-predict.")
    if not DISPLAY_PRODUCT_IDS:
        logger.error("No hay productos cargados.")
        raise HTTPException(status_code=500, detail="No se pudo cargar la lista de productos válidos. Verifique 'products.csv' y 'dataset_limpio_quantity_on_hand.csv'.")
        
    try:
        logger.info("Cargando plantilla final_predict.html.")
        html = open(TEMPLATES_DIR / "final_predict.html", encoding="utf-8").read() 
        products_json_str = json.dumps(DISPLAY_PRODUCT_IDS) 
        products_list_raw = products_json_str.strip('[]').replace('"', '').replace(' ', '')
        html = html.replace("{{ products_json }}", products_list_raw) 
        logger.info("Plantilla procesada y lista.")
        return HTMLResponse(html)
    except FileNotFoundError:
         logger.error("Plantilla HTML no encontrada.")
         raise HTTPException(status_code=500, detail="Plantilla HTML no encontrada. Verifique que 'final_predict.html' esté en la carpeta templates.")


@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    logger.info("Iniciando /chatbot.")
    """Sirve la interfaz del Chatbot."""
    try:
        # Usamos la misma estructura de lectura de HTML
        logger.info("Cargando plantilla chatbot.html.")
        html = open(TEMPLATES_DIR / "chatbot.html", encoding="utf-8").read()
        logger.info("Plantilla cargada.")
        return HTMLResponse(html)
    except FileNotFoundError:
         logger.error("Plantilla chatbot.html no encontrada.")
         raise HTTPException(status_code=500, detail="Plantilla chatbot.html no encontrada.")


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    logger.info(f"Iniciando /api/predict con product_id: {req.product_id}, date: {req.date}")
    target_date = req.date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Fecha objetivo ajustada: {target_date}")
    
    pred = predict_stock_by_date(req.product_id, target_date)
    
    if pred is None:
        logger.error(f"No hay datos suficientes para {req.product_id}")
        raise HTTPException(status_code=404, detail=f"No hay datos suficientes o recursos no cargados para {req.product_id}")
    
    product_info = DF_PRODUCTS[DF_PRODUCTS['id'] == req.product_id]
    product_name = product_info['product_full_name'].iloc[0] if not product_info.empty else req.product_id
    logger.info(f"Nombre del producto: {product_name}")

    result = {
        "product_id": req.product_id, 
        "product_name": product_name,
        "fecha_predicha": target_date, 
        "quantity_on_hand": pred
    }
    logger.info(f"Respuesta generada: {result}")
    return result


@app.post("/api/restock")
async def api_restock(req: RestockRequest):
    logger.info(f"Iniciando /api/restock con date: {req.date}, threshold: {req.threshold}")
    if not DISPLAY_PRODUCT_IDS:
        logger.error("No hay IDs de producto para predecir.")
        raise HTTPException(status_code=500, detail="No hay IDs de producto para predecir.")

    products_to_predict = DISPLAY_PRODUCT_IDS
    logger.info(f"Productos a predecir: {len(products_to_predict)}")
    
    resultados = []
    target_date = req.date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Fecha objetivo ajustada: {target_date}")

    for pid in products_to_predict:
        logger.info(f"Prediciendo para {pid}")
        pred = predict_stock_by_date(pid, target_date)
        
        if pred is None:
            logger.warning(f"Predicción None para {pid}, saltando.")
            continue
            
        product_info = DF_PRODUCTS[DF_PRODUCTS['id'] == pid]
        product_name = product_info['product_full_name'].iloc[0] if not product_info.empty else pid
        logger.info(f"Nombre para {pid}: {product_name}")

        item = {
            "product_id": pid, 
            "product_name": product_name,
            "fecha_predicha": target_date, 
            "quantity_on_hand": pred, 
            "needs_restock": pred <= req.threshold
        }
        resultados.append(item)
        logger.info(f"Item añadido: {item}")
    
    logger.info(f"Resultados totales: {len(resultados)}")
    return resultados


@app.get("/health")
async def health():
    logger.info("Iniciando /health.")
    result = {
        "model_loaded": MODEL is not None, 
        "scalers_loaded": SCALER_X is not None and SCALER_Y is not None,
        "df_clean_loaded": DF_CLEAN is not None,
        "df_products_loaded": DF_PRODUCTS is not None,
        "vertex_ai_ready": VERTEX_CLIENT_READY, # Nuevo
        "total_products_available": len(ALL_PRODUCT_IDS),
        "products_in_interface": len(DISPLAY_PRODUCT_IDS),
        "expected_input_shape": EXPECTED_INPUT_SHAPE
    }
    logger.info(f"Estado de salud: {result}")
    return result


@app.post("/api/chatbot")
async def api_chatbot(req: ChatQuery):
    logger.info(f"Iniciando /api/chatbot con query: {req.query}")
    # Lista para almacenar el log paso a paso de esta solicitud
    log_entries: List[str] = []
    
    try:
        # --- 1. INICIO DEL PROCESO Y LOG DE ENTRADA ---
        start_time = time.time()
        
        log_entries.append(f"[{datetime.now().isoformat()}] INICIO DE PROCESO: Consulta recibida.")
        log_entries.append(f"-> QUERY: {req.query}")
        logger.info(f"Consulta recibida: {req.query}")

        if not VERTEX_CLIENT_READY:
             log_entries.append(f"-> ERROR: Cliente de Vertex AI no está configurado (HTTP 503)")
             logger.error("Vertex AI no configurado.")
             raise HTTPException(status_code=503, detail={"response": "El cliente de Vertex AI no está configurado.", "log": "\n".join(log_entries)})

        # --- 2. LLAMADA A LA FUNCIÓN PRINCIPAL DEL CHATBOT ---
        # La función main_chatbot modifica la lista log_entries in-situ
        response_text = main_chatbot(req.query, log_entries)
        log_entries.append(f"-> RESULTADO FINAL: {response_text}")
        logger.info(f"Respuesta final: {response_text[:100]}...")
        
        # --- 3. FINALIZACIÓN Y LOG DE SALIDA ---
        end_time = time.time()
        duration = end_time - start_time
        log_entries.append(f"[{datetime.now().isoformat()}] FIN DE PROCESO. Duración: {duration:.4f}s")
        logger.info(f"Proceso completado en {duration:.4f}s")
        
        # Registrar el log completo en el archivo centralizado (si el logger está configurado)
        logger.info(f"--- NUEVA CONSULTA DE CHATBOT FINALIZADA --- (Duración: {duration:.4f}s)")
        for entry in log_entries:
            logger.info(f"  [DETALLE_FLUJO] {entry.strip()}")
        logger.info(f"-------------------------------------------------")
        
        return {"response": response_text, "log": "\n".join(log_entries)}
        
    except HTTPException as e:
        # Manejo de HTTP Exceptions generadas por el propio código (ej. 503)
        detail_dict = e.detail if isinstance(e.detail, dict) else {"response": e.detail}
        
        # Aseguramos que el error retorne la respuesta y el log si están disponibles
        log_entries.append(f"[{datetime.now().isoformat()}] ERROR HTTPException (STATUS {e.status_code}): {detail_dict.get('response', 'Fallo de servicio')}")
        logger.error(f"Error HTTPException (Status {e.status_code}). Consulta: {req.query}")

        raise HTTPException(status_code=e.status_code, detail={"response": detail_dict.get('response', 'Error de servicio'), "log": "\n".join(log_entries)})
        
    except Exception as e:
        # Manejo de cualquier otro error fatal (ej. KeyError, fallo de TensorFlow)
        error_msg = f"Error interno no capturado: {type(e).__name__}: {e}"
        logger.critical(f"Error fatal en /api/chatbot: {e}")
        
        log_entries.append(f"[{datetime.now().isoformat()}] ERROR FATAL: {error_msg}")
        
        raise HTTPException(status_code=500, detail={"response": "Error interno del servidor al procesar la solicitud. Consulte el log para más detalles.", "log": "\n".join(log_entries)})


@app.get("/logs", response_class=HTMLResponse)
async def get_logs():
    logger.info("Iniciando /logs.")
    """Sirve los logs completos del servidor como HTML."""
    global LOG_FILE
    
    if not os.path.exists(LOG_FILE):
        logger.warning("Archivo de logs no encontrado.")
        return HTMLResponse("<h1>Archivo de Logs no encontrado.</h1>", status_code=404)
        
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
            logger.info("Logs leídos del archivo.")
            
        # Formato básico para HTML: envuelve el texto en una etiqueta <pre>
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Logs del Servidor</title>
            <style>
                body {{ font-family: monospace; background-color: #2e3436; color: #eeeeec; padding: 20px; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; }}
            </style>
        </head>
        <body>
            <h1>Logs Centralizados de FastAPI</h1>
            <pre>{log_content}</pre>
        </body>
        </html>
        """
        logger.info("HTML de logs generado.")
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error al leer logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo de logs: {e}")


# ====================================================================
# FIN DEL ARCHIVO - Listo para uvicorn app:app --reload
# ====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
