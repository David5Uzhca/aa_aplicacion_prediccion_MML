from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from google.cloud import aiplatform
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from io import StringIO
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

import re
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import json
import os
import sys
import importlib.util

# ====================================================================
# CONFIGURACIÓN GLOBAL
# ====================================================================

# --- CONFIGURACIÓN VERTEX AI (¡REEMPLAZA ESTOS VALORES!) ---
VERTEX_PROJECT_ID = "prediccion-478120"  # <--- TU ID DE PROYECTO GCP
VERTEX_REGION = "us-central1"               # <--- TU REGIÓN GCP (Ej: us-central1)
VERTEX_MODEL = "gemini-2.5-flash"           # Modelo rápido para RAG/Function Calling
VERTEX_CLIENT_READY = False

# --- INFORMACIÓN DE LA EMPRESA (CONTEXTO CHATBOT) ---
EMPRESA_INFO = {
    "nombre": "StockWise Market Intelligence",
    "nicho": "Gestión de Inventario y Cadena de Suministro para Retail/Supermercados",
    "fundacion": 2023,
    "mision": "Asegurar la disponibilidad óptima de productos frescos y de alta rotación en supermercados mediante predicciones de inventario impulsadas por IA (LSTM) y análisis de riesgo en tiempo real.",
    "horario": "Soporte Técnico 24/7. Horario de Análisis y Reporte: Lunes a Sábado, 6:00 AM a 6:00 PM (GMT-5).",
    "contacto": "soporte@stockwisemarket.com o +593 99 123 4567"
}

# --- BASE DE CONOCIMIENTO (FAQS GLOBAL) ---
FAQS = [
    "¿Qué tipo de modelo de IA usan para predecir el stock? Utilizamos una Red Neuronal Recurrente (RNN) con arquitectura LSTM (Long Short-Term Memory).",
    "¿Cuál es el umbral de reabastecimiento que utiliza el sistema? El umbral es de 20 unidades.",
    "¿Puedo agregar datos? Sí, puedes usar la función 'Importar Datos Externos' para subir un CSV con stock real y fecha para reentrenar.",
    "¿Qué información incluye la ventana de 7 días? Incluye datos de stock físico, stock reservado, stock disponible, costos, valor total, y variables de tiempo como el día de la semana.",
    "¿Qué modelo de Google Cloud potencia el asistente? El análisis gerencial (conclusiones y reportes) es generado por el modelo fundacional Gemini 2.5 Flash, accesible a través de la API de Vertex AI.",
]


# ====================================================================
# DEFINICIONES DE FUNCIONES DEL CHATBOT (Function Calling)
# ====================================================================

# --- STUBS DE FUNCIONES DE NEGOCIO ---

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


# --- IMPLEMENTACIÓN DE LOS STUBS ---

def nueva_funcion_1(detalle: str) -> str:
    """
    Función de ejemplo 1: Simula la creación de un reporte de ventas detallado.
    Útil para queries sobre reportes que no sean de stock.
    """
    return f"Iniciando la generación del reporte de ventas detallado para {detalle}. Por favor, espere 5 minutos."

def nueva_funcion_2(departamento: str) -> str:
    """
    Función de ejemplo 2: Simula la consulta del inventario histórico de un departamento específico.
    Útil para queries sobre el rendimiento pasado de una categoría.
    """
    return f"Consultando el historial de inventario del departamento de '{departamento}'. Resumen: El historial muestra alta rotación en las últimas 4 semanas."

def predecir_stock_producto(product_id: str, target_date: str) -> str:
    """Llama al endpoint /api/predict para obtener la predicción de un producto específico."""
    return f"Activando la predicción de stock para el producto {product_id} para la fecha {target_date}. Revisa la interfaz web para el resultado."

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
# LÓGICA UNIFICADA DEL CHATBOT
# ====================================================================

# 5.2 Respuestas Básicas
def responder_basico(query: str) -> str:
    """Responde a saludos, agradecimientos y despedidas."""
    query = query.lower().strip()

    # Patrones Regulares (Regex)
    patrones_saludo = r"^(hola|buen(o?s)? d(i|í)as|buenas tardes|qu(e|é) tal|saludos)"
    patrones_agradecimiento = r"^(gracias|muchas gracias|te lo agradezco|genial|perfecto)"
    patrones_despedida = r"^(adi(o|ó)s|chao|hasta luego|me despido|bye|nos vemos)"

    if re.search(patrones_saludo, query):
        hora = datetime.now().hour
        if 5 <= hora < 12:
            momento = "Buenos días"
        elif 12 <= hora < 19:
            momento = "Buenas tardes"
        else:
            momento = "Buenas noches"
        return f"{momento}, soy **{EMPRESA_INFO['nombre']}**. ¿En qué puedo ayudarte hoy con tu inventario?"
    
    if re.search(patrones_agradecimiento, query):
        return "¡Para eso estamos! Me da gusto ayudarte. ¿Necesitas algo más?"

    if re.search(patrones_despedida, query):
        return f"¡Adiós! Que tengas un excelente día. Si necesitas más predicciones de stock, ¡vuelve pronto!"
    
    return ""

# 5.3 Lógica RAG/FAQ
def responder_faqs(query: str) -> str:
    global llm_rag
    
    if llm_rag is None:
        return "Disculpe, el servicio de IA (Gemini) no está disponible en este momento."

    contexto = "\n".join(FAQS)
    
    prompt_texto = f"""
    Eres un asistente de soporte de {EMPRESA_INFO['nombre']}. Utiliza la siguiente información de la empresa y las FAQs para responder a la pregunta del usuario. 
    Si la respuesta a la pregunta no está en el contexto, indica amablemente que no tienes la información.
    
    --- Información de la Empresa ---
    Nombre: {EMPRESA_INFO['nombre']} (Fundada en {EMPRESA_INFO['fundacion']}). Misión: {EMPRESA_INFO['mision']}
    Horario de atención: {EMPRESA_INFO['horario']}
    
    --- Base de Conocimiento (FAQs) ---
    {contexto}
    
    --- Pregunta del Usuario ---
    {query}
    """
    
    # 3. CREAR Y EJECUTAR LA CADENA RAG
    prompt_template = ChatPromptTemplate.from_template("{prompt_texto}")
    chain_rag = prompt_template | llm_rag | StrOutputParser()
    response = chain_rag.invoke({"prompt_texto": prompt_texto})
    
    return response

# 5.4 Lógica de Function Calling
def responder_tool_calling(query: str) -> str:
    global llm_tool
    
    response = llm_tool.invoke([HumanMessage(content=query)])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        for tool_function in TOOLS:
            if tool_function.__name__ == tool_name:
                try:
                    result = tool_function(**tool_args)
                    return f"FUNCIÓN LLAMADA: {tool_name}. Mensaje de éxito: {result}"
                except Exception as e:
                    return f"Error: La función {tool_name} falló al ejecutarse con argumentos {tool_args}. Detalle: {e}"
        
        return f"Error: La función {tool_name} fue identificada pero no se pudo ejecutar."
    return ""


# 5.5 Función Principal del Chatbot
def main_chatbot(query: str) -> str:
    # 1. Respuestas Básicas (Máxima prioridad)
    respuesta_basica = responder_basico(query)
    if respuesta_basica:
        return respuesta_basica

    # 2. Función Tool Calling (Alta prioridad)
    respuesta_tool = responder_tool_calling(query)
    
    if respuesta_tool.startswith("FUNCIÓN LLAMADA:") or respuesta_tool.startswith("Error:"):
        return respuesta_tool
    
    # 3. Respuesta RAG/FAQ (Si no se identificó Tool Calling o respuesta básica)
    return responder_faqs(query)


# ====================================================================
# INICIALIZACIÓN DE FASTAPI Y CARGA DE MODELOS
# ====================================================================

# --- 1. Definición de Schemas (Pydantic) ---

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

# --- 2. Inicialización y Configuración ---
app = FastAPI(title="Re-stock Predictor API Dev")

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Montaje de archivos estáticos
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL_PATH = MODEL_DIR / "best_model.keras"
SCALER_X_PATH = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"
DATASET_PKL = MODEL_DIR / "dataset_limpio_quantity_on_hand.csv" 
PRODUCTS_CSV = MODEL_DIR / "products.csv"

FEATURE_COLS = [
    "quantity_on_hand", "quantity_reserved", "quantity_available",
    "average_daily_usage", "reorder_point", "optimal_stock_level",
    "unit_cost", "total_value", "days_since_last_order", 
    "days_since_last_count", "days_to_expiration", "month", "day_of_week"
]

MODEL = None
SCALER_X = None
SCALER_Y = None
DF_CLEAN = None
DF_PRODUCTS = None
ALL_PRODUCT_IDS = []
DISPLAY_PRODUCT_IDS = []
EXPECTED_INPUT_SHAPE = (7, 13)

# Objetos LLM globales que se inicializarán en el bloque try
llm_rag = None
llm_tool = None


# --- 3. Carga de modelo y recursos ---
try:
    # 3.1 Carga de recursos de ML
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

    # 3.2 Inicialización de Vertex AI
    aiplatform.init(project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
    VERTEX_CLIENT_READY = True
    print("Cliente de Vertex AI inicializado.")
    
    # --- INICIALIZACIÓN DE OBJETOS LLM ---
    llm_rag = ChatVertexAI(
        model_name=VERTEX_MODEL,
        temperature=0.0,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_REGION
    )

    llm_tool = ChatVertexAI(
        model_name=VERTEX_MODEL,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_REGION
    ).bind_tools(TOOLS)
    # --- FIN DE LA INICIALIZACIÓN DE OBJETOS LLM ---


except Exception as e:
    print(f"Error al cargar recursos o inicializar Vertex AI: {e}")


# ====================================================================
# FUNCIONES DE PREDICCIÓN Y REENTRENAMIENTO (ML CORE)
# ====================================================================

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

    # --- CORRECCIÓN: Manejar el Renombramiento y la Estandarización de la Fecha ---
    if 'fecha_predicha' in new_data_df.columns:
        new_data_df = new_data_df.copy() 
        new_data_df.rename(columns={'fecha_predicha': 'created_at'}, inplace=True)
    
    if 'created_at' not in new_data_df.columns:
        return {"success": False, "log": "ERROR: El DataFrame de entrada no contiene la columna 'created_at'."}
    # -----------------------------------------------------------------------
    
    new_rows = []
    
    for _, row in new_data_df.iterrows():
        pid = row['product_id']
        pred_date = pd.to_datetime(row['created_at']) # Usar columna estandarizada
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
    
    # 1. Validación de Columnas Mínimas
    required_cols = ["product_id", "created_at", "quantity_on_hand"]
    if not all(col in uploaded_df.columns for col in required_cols):
        log += "ERROR: El CSV debe contener las columnas 'product_id', 'created_at' y 'quantity_on_hand'.\n"
        return {"success": False, "log": log}

    log += f"Datos externos recibidos: {len(uploaded_df):,} filas.\n"

    # 2. Convertir y Limpiar (igual que antes)
    uploaded_df['created_at'] = pd.to_datetime(uploaded_df['created_at'], errors='coerce')
    uploaded_df = uploaded_df.dropna(subset=['created_at'])
    uploaded_df['quantity_on_hand'] = pd.to_numeric(uploaded_df['quantity_on_hand'], errors='coerce')
    uploaded_df = uploaded_df.dropna(subset=['quantity_on_hand'])
    
    # --- 3. PROCESAMIENTO Y LLENADO DE COLUMNAS FALTANTES ---
    
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
        
        # Copiar todos los valores faltantes 
        for col in cols_to_copy:
            try:
                new_prod_data[col] = last_known_row[col].iloc[0]
            except IndexError as e:
                log += f"Error interno al copiar columna {col} para {pid}: {e}\n"
                continue

        # 4. Asegurar la estructura
        new_prod_data = new_prod_data[all_df_clean_cols]
        final_new_data.append(new_prod_data)
        
    if not final_new_data:
        log += "ERROR: Ningún producto en el CSV subido se pudo mapear al histórico o pasó la validación.\n"
        return {"success": False, "log": log}
    
    new_df_to_add = pd.concat(final_new_data, ignore_index=True)
    
    # Filtrar fechas duplicadas
    current_ids = set(DF_CLEAN[['product_id', 'created_at']].apply(tuple, axis=1))
    new_df_to_add = new_df_to_add[~new_df_to_add[['product_id', 'created_at']].apply(tuple, axis=1).isin(current_ids)]
    
    if new_df_to_add.empty:
         log += "Advertencia: Todos los datos subidos ya existen en el histórico o son duplicados. Reentrenamiento abortado.\n"
         return {"success": True, "log": log}
         
    # Ejecutar la lógica de reentrenamiento existente
    retrain_result = retrain_model(new_df_to_add)
    
    log += retrain_result['log']
    
    return {"success": retrain_result['success'], "log": log}


# ====================================================================
# RUTAS DE LA API (Endpoints)
# ====================================================================

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


@app.post("/api/conclusion")
async def api_conclusion(req: ConclusionRequest):
    """
    Utiliza LangChain como intermediario para generar una conclusión con Vertex AI (Gemini).
    """
    global VERTEX_CLIENT_READY, VERTEX_MODEL

    if not VERTEX_CLIENT_READY:
        raise HTTPException(status_code=503, detail="El cliente de Vertex AI no está configurado.")
        
    if not req.results:
        return {"conclusion": "No hay resultados para analizar."}

    # 1. Formatear los datos para el LLM (Lógica sin cambios)
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
    
    # 3. Inicializar el LLM de LangChain (ChatVertexAI)
    # LangChain se autentica usando la sesión de aiplatform.init()
    llm = ChatVertexAI(
        model_name=VERTEX_MODEL,
        temperature=0.2,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_REGION
    )

    # 4. Crear y Ejecutar la Cadena (Chain)
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({"data": data_str})

        return {"conclusion": response.content}

    except Exception as e:
        print(f"Error al llamar a la API de Vertex AI (LangChain): {e}")
        return {"conclusion": f"Error al generar la conclusión con LangChain. Revise el log de Uvicorn: {e}"}


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
        # 1. Leer el contenido del archivo subido
        content = await file.read()
        s = str(content, 'utf-8')
        uploaded_df = pd.read_csv(StringIO(s))
        
        # 2. Lógica de Estandarización y Renombramiento
        column_map = {}
        
        # Primero, verificamos si la columna de fecha es 'fecha_predicha' y la renombramos.
        # Si ya es 'created_at', no hacemos nada.
        if 'fecha_predicha' in uploaded_df.columns:
            column_map['fecha_predicha'] = 'created_at'
            log += "Columna 'fecha_predicha' renombrada a 'created_at'.\n"
        
        # 3. Aplicar el renombramiento (si existe un mapeo)
        if column_map:
            uploaded_df.rename(columns=column_map, inplace=True)
        
        # 4. Verificar las columnas obligatorias después del renombramiento
        required_cols = ["product_id", "created_at", "quantity_on_hand"]
        if not all(col in uploaded_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in uploaded_df.columns]
             log += f"ERROR: Faltan columnas obligatorias después de la estandarización: {missing}.\n"
             raise HTTPException(status_code=400, detail={"log": log})

        # 5. Ejecutar la lógica de procesamiento y reentrenamiento (Una sola llamada)
        # La función process_external_data debe recibir el DataFrame estandarizado
        result = process_external_data(uploaded_df, log)
        
        # 6. Manejo de la Respuesta
        if not result['success']:
             raise HTTPException(status_code=500, detail={"log": result['log']})
             
        return result

    except HTTPException:
        # Re-raise HTTPException si ya se lanzó con un código específico (400 o 500)
        raise
    except Exception as e:
        log += f"Error inesperado al procesar el archivo: {e.__class__.__name__}: {e}\n"
        print(f"Error en api_upload_and_retrain: {e}")
        raise HTTPException(status_code=500, detail={"log": log})

@app.get("/final-predict", response_class=HTMLResponse)
async def final_predict(request: Request):
    if not DISPLAY_PRODUCT_IDS:
        raise HTTPException(status_code=500, detail="No se pudo cargar la lista de productos válidos. Verifique 'products.csv' y 'dataset_limpio_quantity_on_hand.csv'.")
        
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
        # Usamos la misma estructura de lectura de HTML
        html = open(TEMPLATES_DIR / "chatbot.html", encoding="utf-8").read()
        return HTMLResponse(html)
    except FileNotFoundError:
         raise HTTPException(status_code=500, detail="Plantilla chatbot.html no encontrada.")


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
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

@app.get("/health")
async def health():
    return {
        "model_loaded": MODEL is not None, 
        "scalers_loaded": SCALER_X is not None and SCALER_Y is not None,
        "df_clean_loaded": DF_CLEAN is not None,
        "df_products_loaded": DF_PRODUCTS is not None,
        "vertex_ai_ready": VERTEX_CLIENT_READY, # Nuevo
        "total_products_available": len(ALL_PRODUCT_IDS),
        "products_in_interface": len(DISPLAY_PRODUCT_IDS),
        "expected_input_shape": EXPECTED_INPUT_SHAPE
    }

@app.post("/api/chatbot")
async def api_chatbot(req: ChatQuery):
    """
    Endpoint principal para recibir consultas del chatbot y dirigirlas a la lógica unificada.
    """
    try:
        if not VERTEX_CLIENT_READY:
             raise HTTPException(status_code=503, detail="El cliente de Vertex AI no está configurado.")

        response_text = main_chatbot(req.query)
        
        return {"response": response_text}
        
    except Exception as e:
        print(f"Error en el procesamiento del Chatbot: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta del chatbot.")