# ====================================================================
# 1. IMPORTACIONE
# ====================================================================

import json
import logging
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import httpx 

from datetime import datetime
from pathlib import Path
from typing import Optional, List
from io import StringIO
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# ====================================================================
# 2. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ====================================================================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
LOG_FILE = BASE_DIR / "app_logs.log"

# ====================================================================
# 3. CONFIGURACIÓN DE LOGGING CENTRALIZADO
# ====================================================================

logger = logging.getLogger('my_api_logger')
logger.setLevel(logging.INFO)
logger.propagate = False 

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
# 4. CREACIÓN DE LA APP FASTAPI Y MONTAJE
# ====================================================================
app = FastAPI(title="Supermercado El Despensa - IA para Inventario")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger.info("Archivos estáticos montados en /static.")

# ====================================================================
# 5. CONFIGURACIÓN GLOBAL Y CONSTANTES (Contexto y Autenticación)
# ====================================================================

# --- AUTENTICACIÓN: Confianza en el Service Account de la VM ---
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

MODEL_PATH = MODEL_DIR / "best_model.keras"
SCALER_X_PATH = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_Y.pkl"
DATASET_PKL = MODEL_DIR / "dataset_limpio_quantity_on_hand.csv"
PRODUCTS_CSV = MODEL_DIR / "products.csv"

EXPECTED_INPUT_SHAPE = (7, 13)

# Variables globales de estado (Inicializadas en None)
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
# 6. DEFINICIÓN DE ESQUEMAS PYDANTIC Y STUBS (CHATBOT TOOLS)
# ====================================================================

# --- Schemas Pydantic ---
class PredictRequest(BaseModel):
    product_id: str
    date: Optional[str] = None

class RestockRequest(BaseModel):
    date: str
    threshold: Optional[float] = 20.0

class ConclusionRequest(BaseModel):
    results: List[dict]

class ChatQuery(BaseModel):
    query: str

class NuevaFuncion1Input(BaseModel):
    """Información para la función 1: generar reporte de ventas."""
    detalle: str = Field(..., description="Detalle del reporte a generar (ej: 'productos de alta rotación' o 'ventas de la semana').")

class NuevaFuncion2Input(BaseModel):
    """Información para la función 2: consultar historial de departamento."""
    departamento: str = Field(..., description="Nombre del departamento a consultar (ej: 'lácteos' o 'bebidas').")

class PrediccionProductoInput(BaseModel):
    """Información necesaria para la predicción de stock de un solo producto."""
    product_id: str = Field(..., description="ID exacto del producto (ej: pdct0015).")
    target_date: str = Field(..., description="Fecha futura para la predicción en formato YYYY-MM-DD.")

class PrediccionGeneralInput(BaseModel):
    """Información necesaria para la predicción de stock de todos los productos principales."""
    target_date: str = Field(..., description="Fecha futura para la predicción en formato YYYY-MM-DD.")

class AgregarRegistrosInput(BaseModel):
    """Información para añadir nuevos registros de productos."""
    datos_externos: str = Field(..., description="Detalles sobre si los datos vienen de la tabla o de un archivo CSV.")

class AjustarModeloInput(BaseModel):
    """Información para reentrenar/ajustar el modelo con datos recientes."""
    fuente_datos: str = Field(..., description="Indica la fuente de los datos a usar (ej: 'tabla de prediccion' o 'datos externos').")


# --- Implementación de Stubs ---

def nueva_funcion_1(detalle: str) -> str:
    """Función de ejemplo 1: Simula la creación de un reporte de ventas detallado."""
    return f"Iniciando la generación del reporte de ventas detallado para {detalle}. Por favor, espere 5 minutos."

def nueva_funcion_2(departamento: str) -> str:
    """Función de ejemplo 2: Simula la consulta del inventario histórico de un departamento específico."""
    return f"Consultando el historial de inventario del departamento de '{departamento}'. Resumen: El historial muestra alta rotación en las últimas 4 semanas."

LOCAL_API_BASE_URL = "http://localhost:8000" # Usamos localhost para httpx

async def predecir_stock_producto(product_id: str, target_date: str) -> str:
    """Llama al endpoint /api/predict (asíncrono) y devuelve el stock predicho."""
    
    url = f"{LOCAL_API_BASE_URL}/api/predict"
    payload = {
        "product_id": product_id,
        "date": target_date
    }
    
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            stock_predicho = data.get('quantity_on_hand', 0)
            nombre_producto = data.get('product_name', product_id)
            fecha_predicha = data.get('fecha_predicha', target_date)
            
            umbral = 20
            
            if stock_predicho <= umbral:
                conclusion = f"El producto {nombre_producto} ({product_id}) cuenta con un stock predicho de {stock_predicho:.2f} para la fecha {fecha_predicha}, por lo que REQUIERE REABASTECIMIENTO URGENTE."
            else:
                conclusion = f"El producto {nombre_producto} ({product_id}) cuenta con un stock predicho de {stock_predicho:.2f} para la fecha {fecha_predicha}, por lo que no requiere reabastecimiento en este momento."
                
            return conclusion
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"Error: No se encontraron datos históricos o el producto {product_id} no existe. [Código 404]"
            return f"Error HTTP al llamar al servicio de predicción: {e}"
        
        except httpx.RequestError as e:
            return f"Error de conexión: Fallo de red interno al intentar conectar con el servicio: {e}."
        except Exception as e:
            return f"Error inesperado al procesar la predicción: {e}"

def predecir_stock_general(target_date: str) -> str:
    """Llama al endpoint /api/restock para obtener la predicción de stock de todos los productos principales."""
    return f"Activando la predicción general para todos los productos principales para la fecha {target_date}. Revisa la interfaz web para la tabla de reabastecimiento."

def agregar_nuevos_registros(datos_externos: str) -> str:
    """Función para registrar la intención de agregar nuevos datos (CSV o tabla)."""
    return f"Preparado para añadir nuevos registros de productos. Por favor, utiliza la interfaz web para subir el archivo CSV o haz clic en 'Añadir Datos y Actualizar Modelo' después de una predicción general."

def actualizar_modelo(fuente_datos: str) -> str:
    """Activa el endpoint /api/retrain o /api/upload_and_retrain para ajustar el modelo."""
    return f"El modelo se actualizará (ajustará) usando datos de {fuente_datos}. Por favor, revisa el log de reentrenamiento en la interfaz web para ver el progreso."

TOOLS = [
    predecir_stock_producto,
    predecir_stock_general,
    agregar_nuevos_registros,
    actualizar_modelo,
    nueva_funcion_1,
    nueva_funcion_2,
]

# ====================================================================
# 7. LÓGICA UNIFICADA DEL CHATBOT
# ====================================================================
# 7.1 Respuestas Básicas
def responder_basico(query: str, log_entries: List[str]) -> str:
    """Responde a saludos, agradecimientos y despedidas."""
    log_entries.append("  -> LÓGICA BÁSICA: Ejecutando verificación de Regex.")
    query_lower = query.lower().strip()

    patrones_saludo = r"^(hola|buen(o?s)? d(i|í)as|buenas tardes|qu(e|é) tal|saludos)"
    patrones_agradecimiento = r"^(gracias|muchas gracias|te lo agradezco|genial|perfecto)"
    patrones_despedida = r"^(adi(o|ó)s|chao|hasta luego|me despido|bye|nos vemos)"

    if re.search(patrones_saludo, query_lower):
        log_entries.append("  -> DECISIÓN: Coincidencia con 'Saludo'.")
        hora = datetime.now().hour
        if 5 <= hora < 12: momento = "Buenos días"
        elif 12 <= hora < 19: momento = "Buenas tardes"
        else: momento = "Buenas noches"
        return f"{momento}, soy **{EMPRESA_INFO['nombre']}**. ¿En qué puedo ayudarte hoy con tu inventario y gestión?"
    
    if re.search(patrones_agradecimiento, query_lower):
        log_entries.append("  -> DECISIÓN: Coincidencia con 'Agradecimiento'.")
        return "¡Para eso estamos! Me da gusto ayudarte. ¿Necesitas algo más?"

    if re.search(patrones_despedida, query_lower):
        log_entries.append("  -> DECISIÓN: Coincidencia con 'Despedida'.")
        return f"¡Adiós! Que tengas un excelente día. ¡Vuelve pronto!"
    
    log_entries.append("  -> DECISIÓN: No se encontró coincidencia básica.")
    return ""

# 7.2 Lógica RAG/FAQ (Síncrona)
def responder_faqs(query: str, log_entries: List[str]) -> str:
    """Responde a las FAQs utilizando RAG sobre la base de conocimiento."""
    global llm_rag
    log_entries.append("  -> LÓGICA RAG/FAQ: Ejecutando modelo Gemini para consulta de FAQs.")
    if llm_rag is None: return "Disculpe, el servicio de IA (Gemini) no está disponible en este momento."

    contexto = "\n".join(FAQS)
    log_entries.append(f"  -> CONTEXTO RAG (FAQs y Empresa) ENVIADO AL LLM: {EMPRESA_INFO['nombre']}, {len(FAQS)} FAQs.")

    prompt_texto = f"""
    Eres un asistente de soporte de {EMPRESA_INFO['nombre']}. Utiliza la siguiente información de la empresa y las FAQs para responder a la pregunta del usuario. 
    Si la respuesta a la pregunta no está en el contexto, indica amablemente que no tienes la información.
    
    --- Información de la Empresa ---
    Nombre: {EMPRESA_INFO['nombre']} (Fundada en {EMPRESA_INFO['fundacion']}). Misión: {EMPRESA_INFO['mision']}
    Horario de atención: {EMPRESA_INFO['horario_tienda']}
    
    --- Base de Conocimiento (FAQs) ---
    {contexto}
    
    --- Pregunta del Usuario ---
    {query}
    """
    
    prompt_template = ChatPromptTemplate.from_template("{prompt_texto}")
    chain_rag = prompt_template | llm_rag | StrOutputParser()
    
    try:
        response = chain_rag.invoke({"prompt_texto": prompt_texto})
        log_entries.append("  -> DECISIÓN: Respuesta generada por RAG/LLM.")
        return response
    except Exception as e:
        log_entries.append(f"  -> ERROR RAG: Fallo en la invocación de LangChain/Gemini. Detalle: {e}")
        return "Disculpe, ocurrió un error al consultar la base de conocimiento."

# 7.3 Lógica de Function Calling (Asíncrona)
async def responder_tool_calling(query: str, log_entries: List[str]) -> str: 
    global llm_tool
    log_entries.append("  -> LÓGICA FUNCTION CALLING: Ejecutando modelo Gemini para detección de herramienta.")
    if llm_tool is None: return "Disculpe, el servicio de Function Calling no está disponible."
    
    try:
        response = llm_tool.invoke([HumanMessage(content=query)])
    except Exception as e:
        log_entries.append(f"  -> ERROR TOOL CALLING: Fallo en la invocación de LangChain/Gemini. Detalle: {e}")
        return "" 

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        log_entries.append(f"  -> DECISIÓN: Se detectó llamada a función.")
        log_entries.append(f"    - FUNCIÓN DETECTADA: {tool_name}")
        log_entries.append(f"    - ARGUMENTOS EXTRAÍDOS: {tool_args}")
        
        for tool_function in TOOLS:
            if tool_function.__name__ == tool_name:
                try:
                    result = await tool_function(**tool_args)
                    log_entries.append(f"  -> RESULTADO FUNCIÓN: Éxito. {result}")
                    return f" FUNCIÓN LLAMADA: {tool_name}. Mensaje de éxito: {result}"
                except Exception as e:
                    log_entries.append(f"  -> RESULTADO FUNCIÓN: Error al ejecutar la función de Python. Detalle: {e}")
                    return f"Error: La función {tool_name} falló al ejecutarse. Detalle: {e}"
        
        log_entries.append(f"  -> RESULTADO FUNCIÓN: La función {tool_name} no fue encontrada en el código.")
        return f"Error: La función {tool_name} fue identificada pero no se pudo ejecutar."

    log_entries.append("  -> DECISIÓN: No se detectó ninguna llamada a función. Pasando a RAG/FAQ.")
    return ""

# 7.4 Función Principal del Chatbot (Asíncrona)
async def main_chatbot(query: str, log_entries: List[str]) -> str:
    log_entries.append("-> PASO 1: Verificación de Respuestas Básicas.")
    respuesta_basica = responder_basico(query, log_entries)
    if respuesta_basica:
        log_entries.append("-> FLUJO FINAL: Retornando Respuesta Básica.")
        return respuesta_basica

    log_entries.append("-> PASO 2: Verificación de Function Calling.")
    respuesta_tool = await responder_tool_calling(query, log_entries) # AWAIT AQUÍ
    if respuesta_tool:
        log_entries.append("-> FLUJO FINAL: Retornando Resultado de Function Calling.")
        return respuesta_tool
    
    log_entries.append("-> PASO 3: Ejecutando Lógica RAG/FAQ.")
    respuesta_rag = responder_faqs(query, log_entries)
    log_entries.append("-> FLUJO FINAL: Retornando Respuesta RAG/FAQ.")
    
    return respuesta_rag


# ====================================================================
# 8. CÓDIGO DE CARGA Y ML CORE (Funciones)
# ====================================================================
# Nota: predict_stock_by_date y retrain_model son síncronas

def get_last_7_rows(product_id: str):
    """Obtiene las últimas 7 filas NO ESCALADAS de un producto, rellenando si es necesario."""
    if DF_CLEAN is None: return None, None
    
    prod = DF_CLEAN[DF_CLEAN["product_id"].str.strip() == product_id.strip()].tail(7)
    
    X = prod[FEATURE_COLS].values.astype(np.float32)
    
    if len(prod) == 0:
        return None, None

    if X.shape[0] < 7:
        t_exp = 7
        f_exp = X.shape[1] 
        pad = np.zeros((t_exp - X.shape[0], f_exp))
        X = np.vstack([pad, X])
        print(f"DEBUG: Rellenado de {product_id} con {t_exp - X.shape[0]} filas de cero.")
    
    last_date = prod["created_at"].iloc[-1]
    return X, last_date

def predict_stock_by_date(product_id: str, target_date_str: str):
    """Predice el stock futuro simulando la evolución de las 13 features."""
    if MODEL is None or SCALER_X is None or SCALER_Y is None:
        return None
        
    window, last_date = get_last_7_rows(product_id)
    if window is None:
        return None
        
    target = pd.to_datetime(target_date_str)
    current = window.copy()
    t_exp, f_exp = EXPECTED_INPUT_SHAPE
    
    if target <= last_date:
        return max(0, round(float(window[-1, 0]), 2))
        
    days = (target - last_date).days
    
    if window.shape[0] != t_exp or window.shape[1] != f_exp:
        print(f"ERROR: Forma de ventana incorrecta: {window.shape}")
        return None
    
    last_pred = window[-1, 0] 
    trend = current[-1] - current[-t_exp] 
    
    for day in range(days):
        X_in = SCALER_X.transform(current).reshape((1, t_exp, f_exp)).astype(np.float32)
        pred_scaled = MODEL.predict(X_in, verbose=0)[0][0]
        pred = SCALER_Y.inverse_transform([[pred_scaled]])[0][0]

        new_row = current[-1].copy()
        new_row[0] = pred 

        new_row[1:] = current[-1][1:] + trend[1:] * 0.08 
        
        current = np.vstack([current[1:], new_row])
        last_pred = pred

    return max(0, round(float(last_pred), 2))

def retrain_model(new_data_df: pd.DataFrame) -> dict:
    global MODEL, SCALER_X, SCALER_Y, DF_CLEAN, EXPECTED_INPUT_SHAPE, FEATURE_COLS
    
    if MODEL is None or DF_CLEAN is None:
        return {"success": False, "log": "ERROR: El modelo o DF_CLEAN no están cargados."}

    if 'fecha_predicha' in new_data_df.columns:
        new_data_df = new_data_df.copy() 
        new_data_df.rename(columns={'fecha_predicha': 'created_at'}, inplace=True)
    
    if 'created_at' not in new_data_df.columns:
        return {"success": False, "log": "ERROR: El DataFrame de entrada no contiene la columna 'created_at'."}
    
    new_rows = []
    
    for _, row in new_data_df.iterrows():
        pid = row['product_id']
        pred_date = pd.to_datetime(row['created_at'])
        pred_stock = row['quantity_on_hand']

        last_row_data = DF_CLEAN[DF_CLEAN['product_id'] == pid].tail(1)

        if not last_row_data.empty:
            last_row = last_row_data[FEATURE_COLS].values[0]
            last_date = last_row_data['created_at'].iloc[0]

            days_to_add = (pred_date - last_date).days
            
            if days_to_add <= 0:
                continue

            simulated_row = last_row.copy()
            simulated_row[0] = pred_stock 
            new_data = {
                'product_id': pid,
                'created_at': pred_date,
                **{col: val for col, val in zip(FEATURE_COLS, simulated_row)}
            }
            new_rows.append(new_data)
            
    if not new_rows:
        return {"success": True, "log": "Advertencia: No se generaron nuevas filas para añadir (fechas pasadas/actuales)."}

    new_df = pd.DataFrame(new_rows)
    cols_to_keep = DF_CLEAN.columns.tolist() 
    new_df = new_df[[col for col in new_df.columns if col in cols_to_keep]] 
    DF_CLEAN = pd.concat([DF_CLEAN, new_df], ignore_index=True)
    DF_CLEAN.to_csv(str(DATASET_PKL), index=False)
    
    log = f"Datos añadidos: {len(new_rows)} nuevas filas. DF_CLEAN total: {len(DF_CLEAN):,}.\n"
    
    X_retrain, y_retrain = [], []
    t_exp, f_exp = EXPECTED_INPUT_SHAPE
    
    for pid in DISPLAY_PRODUCT_IDS:
        prod = DF_CLEAN[DF_CLEAN['product_id'] == pid]
        
        if len(prod) >= t_exp + 1:
            X_input_raw = prod[FEATURE_COLS].iloc[-t_exp-1:-1].values.astype(np.float32) 
            y_target_raw = prod[FEATURE_COLS].iloc[-1, 0] 

            if X_input_raw.shape == EXPECTED_INPUT_SHAPE:
                X_retrain.append(X_input_raw)
                y_retrain.append(y_target_raw)
                
    if not X_retrain:
        log += "Advertencia: No hay secuencias completas (7+1) para reentrenar. No se realizó el reentrenamiento.\n"
        return {"success": True, "log": log}

    X_retrain = np.array(X_retrain)
    y_retrain = np.array(y_retrain).reshape(-1, 1)
    X_flat = X_retrain.reshape(-1, f_exp)
    X_scaled = SCALER_X.transform(X_flat).reshape(X_retrain.shape)
    
    y_scaled = SCALER_Y.transform(y_retrain).ravel()
    
    log += f"Secuencias de reentrenamiento generadas: {X_scaled.shape[0]}. Iniciando reentrenamiento (1 epoch)...\n"
    
    history = MODEL.fit(
        X_scaled, y_scaled,
        epochs=1,
        batch_size=32,
        verbose=0
    )
    
    loss = history.history['loss'][0]
    log += f"Reentrenamiento completado (1 epoch). Nueva Loss: {loss:.4f}\n"

    MODEL.save(str(MODEL_PATH))
    log += f"Modelo guardado en: {MODEL_PATH}\n"
    
    return {"success": True, "log": log}


def process_external_data(uploaded_df: pd.DataFrame, log: str) -> dict:
    """
    Procesa un nuevo DataFrame de datos externos, lo fusiona con DF_CLEAN, 
    y ejecuta el reentrenamiento del modelo, asegurando que se llenen todas las 
    columnas descriptivas y auxiliares faltantes.
    """
    global DF_CLEAN, FEATURE_COLS, DISPLAY_PRODUCT_IDS
    
    required_cols = ["product_id", "created_at", "quantity_on_hand"]
    if not all(col in uploaded_df.columns for col in required_cols):
        log += "ERROR: El CSV debe contener las columnas 'product_id', 'created_at' y 'quantity_on_hand'.\n"
        return {"success": False, "log": log}

    log += f"Datos externos recibidos: {len(uploaded_df):,} filas.\n"

    uploaded_df['created_at'] = pd.to_datetime(uploaded_df['created_at'], errors='coerce')
    uploaded_df = uploaded_df.dropna(subset=['created_at'])
    uploaded_df['quantity_on_hand'] = pd.to_numeric(uploaded_df['quantity_on_hand'], errors='coerce')
    uploaded_df = uploaded_df.dropna(subset=['quantity_on_hand'])
    
    all_df_clean_cols = DF_CLEAN.columns.tolist()
    base_upload_cols = uploaded_df.columns.tolist()
    
    cols_to_copy = [col for col in all_df_clean_cols if col not in base_upload_cols]

    final_new_data = []
    
    for pid in uploaded_df['product_id'].unique():
        last_known_row = DF_CLEAN[DF_CLEAN['product_id'] == pid].sort_values('created_at').tail(1)
        
        if last_known_row.empty:
            log += f"Advertencia: El producto {pid} no existe en el histórico. Saltando.\n"
            continue

        new_prod_data = uploaded_df[uploaded_df['product_id'] == pid].copy()
        
        for col in cols_to_copy:
            try:
                new_prod_data[col] = last_known_row[col].iloc[0]
            except IndexError as e:
                log += f"Error interno al copiar columna {col} para {pid}: {e}\n"
                continue

        new_prod_data = new_prod_data[all_df_clean_cols]
        final_new_data.append(new_prod_data)
        
    if not final_new_data:
        log += "ERROR: Ningún producto en el CSV subido se pudo mapear al histórico o pasó la validación.\n"
        return {"success": False, "log": log}
    
    new_df_to_add = pd.concat(final_new_data, ignore_index=True)
    
    current_ids = set(DF_CLEAN[['product_id', 'created_at']].apply(tuple, axis=1))
    new_df_to_add = new_df_to_add[~new_df_to_add[['product_id', 'created_at']].apply(tuple, axis=1).isin(current_ids)]
    
    if new_df_to_add.empty:
         log += "Advertencia: Todos los datos subidos ya existen en el histórico o son duplicados. Reentrenamiento abortado.\n"
         return {"success": True, "log": log}
         
    retrain_result = retrain_model(new_df_to_add)
    
    log += retrain_result['log']
    
    return {"success": retrain_result['success'], "log": log}


# ====================================================================
# 9. Lógica de Carga al Inicio de la App
# ====================================================================

try:
    # 9.1 Carga de recursos de ML
    MODEL = tf.keras.models.load_model(str(MODEL_PATH))
    SCALER_X = pickle.load(open(SCALER_X_PATH, "rb"))
    SCALER_Y = pickle.load(open(SCALER_Y_PATH, "rb"))
    
    DF_CLEAN = pd.read_csv(DATASET_PKL, parse_dates=["created_at"])
    DF_CLEAN = DF_CLEAN.sort_values(["product_id", "created_at"]).reset_index(drop=True)
    DF_PRODUCTS = pd.read_csv(PRODUCTS_CSV) 
    
    products_in_clean_df = set(DF_CLEAN['product_id'].unique())
    products_in_info_df = set(DF_PRODUCTS['id'].unique())
    
    ALL_PRODUCT_IDS = sorted(list(products_in_clean_df & products_in_info_df))
    DISPLAY_PRODUCT_IDS = ALL_PRODUCT_IDS[:30]
    
    EXPECTED_INPUT_SHAPE = MODEL.input_shape[1:] 

    # 9.2 Inicialización de Vertex AI (Service Account de la VM)
    aiplatform.init(project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
    VERTEX_CLIENT_READY = True
    logger.info("Cliente de Vertex AI inicializado.")
    
    # --- INICIALIZACIÓN DE OBJETOS LLM ---
    llm_rag = ChatVertexAI(
        model_name=VERTEX_MODEL, temperature=0.0, project=VERTEX_PROJECT_ID, location=VERTEX_REGION
    )

    llm_tool = ChatVertexAI(
        model_name=VERTEX_MODEL, temperature=0.2, project=VERTEX_PROJECT_ID, location=VERTEX_REGION
    ).bind_tools(TOOLS)
    logger.info("Modelos LLM (Rag y Tools) de LangChain inicializados.")
    # --- FIN DE LA INICIALIZACIÓN DE OBJETOS LLM ---


except Exception as e:
    logger.error(f"Error fatal al cargar recursos o inicializar Vertex AI: {e}", exc_info=True)
    print(f"Error al cargar recursos o inicializar Vertex AI: {e}")


# ====================================================================
# 10. ENDPOINTS DE LA API
# ====================================================================

@app.post("/api/chatbot")
async def api_chatbot(req: ChatQuery):
    # ¡Función Asíncrona!
    log_entries: List[str] = []
    
    try:
        start_time = time.time()
        log_entries.append(f"[{datetime.now().isoformat()}] INICIO DE PROCESO: Consulta recibida.")
        log_entries.append(f"-> QUERY: {req.query}")

        if not VERTEX_CLIENT_READY:
             log_entries.append(f"-> ERROR: Cliente de Vertex AI no está configurado (HTTP 503)")
             raise HTTPException(status_code=503, detail={"response": "El cliente de Vertex AI no está configurado.", "log": "\n".join(log_entries)})

        # AWAIT AQUÍ para la función principal
        response_text = await main_chatbot(req.query, log_entries)
        
        end_time = time.time()
        duration = end_time - start_time
        
        log_entries.append(f"-> RESPUESTA FINAL: {response_text.splitlines()[0]}...")
        log_entries.append(f"[{datetime.now().isoformat()}] FIN DE PROCESO. Duración: {duration:.4f}s")
        
        logger.info(f"--- NUEVA CONSULTA DE CHATBOT FINALIZADA --- (Duración: {duration:.4f}s)")
        for entry in log_entries:
            logger.info(f"  [DETALLE_FLUJO] {entry.strip()}")
        logger.info(f"-------------------------------------------------")
        
        return {"response": response_text, "log": "\n".join(log_entries)}
        
    except HTTPException as e:
        detail_dict = e.detail if isinstance(e.detail, dict) else {"response": e.detail}
        log_entries.append(f"[{datetime.now().isoformat()}] ERROR HTTPException (STATUS {e.status_code}): {detail_dict.get('response', 'Fallo de servicio')}")
        logger.error(f"Error HTTPException en /api/chatbot (Status {e.status_code}). Consulta: {req.query}")

        raise HTTPException(status_code=e.status_code, detail={"response": detail_dict.get('response', 'Error de servicio'), "log": "\n".join(log_entries)})
        
    except Exception as e:
        error_msg = f"Error interno no capturado: {type(e).__name__}: {e}"
        print(f"Error fatal en /api/chatbot: {e}")
        log_entries.append(f"[{datetime.now().isoformat()}] ERROR FATAL: {error_msg}")
        logger.critical(f"ERROR FATAL en /api/chatbot. Detalle: {error_msg}", exc_info=True)
        
        raise HTTPException(status_code=500, detail={"response": "Error interno del servidor al procesar la solicitud. Consulte el log para más detalles.", "log": "\n".join(log_entries)})


@app.get("/final-predict", response_class=HTMLResponse)
async def final_predict(request: Request):
    if not DISPLAY_PRODUCT_IDS:
        raise HTTPException(status_code=500, detail="No se pudo cargar la lista de productos válidos.")
        
    try:
        products = DISPLAY_PRODUCT_IDS
        html = open(TEMPLATES_DIR / "final_predict.html", encoding="utf-8").read() 
        products_json_str = json.dumps(products) 
        products_list_raw = products_json_str.strip('[]').replace('"', '').replace(' ', '')
        html = html.replace("{{ products_json }}", products_list_raw) 
        
        return HTMLResponse(html)
    except FileNotFoundError:
         raise HTTPException(status_code=500, detail="Plantilla HTML no encontrada. Verifique que 'final_predict.html' esté en la carpeta templates.")


@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    """Sirve la interfaz del Chatbot."""
    try:
        html = open(TEMPLATES_DIR / "chatbot.html", encoding="utf-8").read()
        return HTMLResponse(html)
    except FileNotFoundError:
         raise HTTPException(status_code=500, detail="Plantilla chatbot.html no encontrada.")


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    # Nota: predict_stock_by_date es SÍNCRONA, pero FastAPI la maneja en un thread pool.
    target_date = req.date or datetime.now().strftime("%Y-%m-%d")
    
    pred = predict_stock_by_date(req.product_id, target_date)
    
    if pred is None:
        raise HTTPException(status_code=404, detail=f"No hay datos suficientes o recursos no cargados para {req.product_id}")
    
    product_info = DF_PRODUCTS[DF_PRODUCTS['id'] == req.product_id]
    product_name = product_info['product_full_name'].iloc[0] if not product_info.empty else req.product_id

    return {
        "product_id": req.product_id, 
        "product_name": product_name,
        "fecha_predicha": target_date, 
        "quantity_on_hand": pred
    }

@app.post("/api/restock")
async def api_restock(req: RestockRequest):
    if not DISPLAY_PRODUCT_IDS:
        raise HTTPException(status_code=500, detail="No hay IDs de producto para predecir.")

    products_to_predict = DISPLAY_PRODUCT_IDS
    
    resultados = []
    target_date = req.date or datetime.now().strftime("%Y-%m-%d")

    for pid in products_to_predict:
        pred = predict_stock_by_date(pid, target_date)
        
        if pred is None:
            continue
            
        product_info = DF_PRODUCTS[DF_PRODUCTS['id'] == pid]
        product_name = product_info['product_full_name'].iloc[0] if not product_info.empty else pid

        resultados.append({
            "product_id": pid, 
            "product_name": product_name,
            "fecha_predicha": target_date, 
            "quantity_on_hand": pred, 
            "needs_restock": pred <= req.threshold
        })
    
    return resultados

@app.post("/api/retrain")
async def api_retrain(req: Request):
    try:
        new_data_list = await req.json()
        
        if not new_data_list or not isinstance(new_data_list, list):
            raise HTTPException(status_code=400, detail="Datos no válidos o vacíos para el reentrenamiento.")

        new_data_df = pd.DataFrame(new_data_list)
        new_data_df = new_data_df.dropna(subset=['quantity_on_hand'])
        new_data_df['quantity_on_hand'] = new_data_df['quantity_on_hand'].apply(lambda x: max(0, float(x)))
        
        result = retrain_model(new_data_df)
        
        if not result['success']:
             raise HTTPException(status_code=500, detail=result['log'])
             
        return result

    except Exception as e:
        print(f"Error en api_retrain: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno durante el reentrenamiento: {e}")


@app.post("/api/upload_and_retrain")
async def api_upload_and_retrain(file: UploadFile = File(...)):
    """
    Recibe un archivo CSV, lo procesa, fusiona con el histórico y reentrena el modelo.
    """
    log = f"Procesando archivo: {file.filename}\n"
    
    if not file.filename.endswith('.csv'):
        log += "ERROR: Formato de archivo no soportado. Debe ser un archivo CSV (.csv).\n"
        raise HTTPException(status_code=400, detail={"log": log})
    
    try:
        content = await file.read()
        s = str(content, 'utf-8')
        uploaded_df = pd.read_csv(StringIO(s))
        
        column_map = {}
        if 'fecha_predicha' in uploaded_df.columns:
            column_map['fecha_predicha'] = 'created_at'
            log += "Columna 'fecha_predicha' renombrada a 'created_at'.\n"
        
        if column_map:
            uploaded_df.rename(columns=column_map, inplace=True)
        
        required_cols = ["product_id", "created_at", "quantity_on_hand"]
        if not all(col in uploaded_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in uploaded_df.columns]
             log += f"ERROR: Faltan columnas obligatorias después de la estandarización: {missing}.\n"
             raise HTTPException(status_code=400, detail={"log": log})

        result = process_external_data(uploaded_df, log)
        
        if not result['success']:
             raise HTTPException(status_code=500, detail={"log": result['log']})
             
        return result

    except HTTPException:
        raise
    except Exception as e:
        log += f"Error inesperado al procesar el archivo: {e.__class__.__name__}: {e}\n"
        print(f"Error en api_upload_and_retrain: {e}")
        raise HTTPException(status_code=500, detail={"log": log})

@app.post("/api/conclusion")
async def api_conclusion(req: ConclusionRequest):
    global VERTEX_CLIENT_READY, VERTEX_MODEL

    if not VERTEX_CLIENT_READY:
        raise HTTPException(status_code=503, detail="El cliente de Vertex AI no está configurado.")
        
    if not req.results:
        return {"conclusion": "No hay resultados para analizar."}

    data_str = "Resultados de Predicción de Stock:\n"
    data_str += "---------------------------------------------------------\n"
    data_str += "ID | Stock Predicho | Necesita Reabastecer\n"
    data_str += "---|----------------|-----------------------\n"
    
    for item in req.results:
        needs_restock = "Sí" if item.get('needs_restock', False) or (item.get('quantity_on_hand', 0) <= 20 and 'needs_restock' not in item) else "No"
        stock = f"{item.get('quantity_on_hand', 0):.2f}"
        data_str += f"{item.get('product_id')} | {stock} | {needs_restock}\n"

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un experto analista de inventario y gestión de almacén. Tu tarea es analizar los datos de predicción "
                "de stock y generar una conclusión breve y orientada a la acción para la gerencia."
            ),
            (
                "human",
                "Analiza los siguientes datos. El umbral de reabastecimiento es 20 unidades. Genera una conclusión en español "
                "cubriendo: (1) Productos críticos (stock <= 5), (2) Resumen del porcentaje de productos que necesitan reabastecimiento (stock <= 20), "
                "y (3) Una recomendación de acción breve. \n\n--- DATOS ---\n{data}"
            )
        ]
    )
    
    llm = ChatVertexAI(
        model_name=VERTEX_MODEL,
        temperature=0.2,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_REGION
    )

    chain = prompt_template | llm
    
    try:
        response = chain.invoke({"data": data_str})

        return {"conclusion": response.content}

    except Exception as e:
        print(f"Error al llamar a la API de Vertex AI (LangChain): {e}")
        return {"conclusion": f"Error al generar la conclusión con LangChain. Revise el log de Uvicorn: {e}"}

@app.get("/health")
async def health():
    return {
        "model_loaded": MODEL is not None, 
        "scalers_loaded": SCALER_X is not None and SCALER_Y is not None,
        "df_clean_loaded": DF_CLEAN is not None,
        "df_products_loaded": DF_PRODUCTS is not None,
        "vertex_ai_ready": VERTEX_CLIENT_READY, 
        "total_products_available": len(ALL_PRODUCT_IDS),
        "products_in_interface": len(DISPLAY_PRODUCT_IDS),
        "expected_input_shape": EXPECTED_INPUT_SHAPE
    }

@app.get("/logs", response_class=HTMLResponse)
async def get_logs():
    """Sirve los logs completos del servidor como HTML."""
    global LOG_FILE
    
    if not os.path.exists(LOG_FILE):
        return HTMLResponse("<h1>Archivo de Logs no encontrado.</h1>", status_code=404)
        
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
            
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
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo de logs: {e}")