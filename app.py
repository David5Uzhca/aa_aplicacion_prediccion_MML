from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime

# --- INTEGRACIÓN VERTEX AI ---
import os
# La biblioteca google-cloud-aiplatform se autentica automáticamente
from google.cloud import aiplatform
# --- CORRECCIÓN DE IMPORTACIONES DE VERTEX AI ---
from vertexai.preview.generative_models import GenerativeModel
# --- CONFIGURACIÓN DE VERTEX AI (REEMPLAZA ESTOS VALORES) ---
# Necesitas tu Project ID y la región donde está la VM
VERTEX_PROJECT_ID = "prediccion-478120"  # <--- ¡IMPORTANTE! Reemplazar con tu ID
VERTEX_REGION = "us-central1"               # <--- Reemplazar con la región de tu VM
VERTEX_MODEL = "gemini-2.5-flash"           # Usamos el modelo más rápido de Gemini
VERTEX_CLIENT_READY = False

# --- 1. Definición de Schemas (Pydantic) ---

class PredictRequest(BaseModel):
    product_id: str
    date: Optional[str] = None

class RestockRequest(BaseModel):
    date: str
    threshold: Optional[float] = 20.0

class ConclusionRequest(BaseModel):
    results: List[dict] # Schema para recibir los datos de la tabla

# --- 2. Inicialización y Configuración ---
app = FastAPI(title="Re-stock Predictor API Dev")

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

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

    # 3.2 Inicialización de Vertex AI (Debe estar dentro del try si falla)
    aiplatform.init(project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
    VERTEX_CLIENT_READY = True
    print("✅ Cliente de Vertex AI inicializado.")

except Exception as e:
    print(f"Error al cargar recursos o inicializar Vertex AI: {e}")

# --- 4. Funciones de Predicción y Utilidad (Sin Cambios) ---

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
        X_in = SCALER_X.transform(current).reshape(1, t_exp, f_exp)
        pred_scaled = MODEL.predict(X_in, verbose=0)[0][0]
        pred = SCALER_Y.inverse_transform([[pred_scaled]])[0][0]

        new_row = current[-1].copy()
        new_row[0] = pred 

        new_row[1:] = current[-1][1:] + trend[1:] * 0.08 
        
        current = np.vstack([current[1:], new_row])
        last_pred = pred

    return max(0, round(float(last_pred), 2))

# --- 5. Funciones de Reentrenamiento (Sin Cambios) ---

def retrain_model(new_data_df: pd.DataFrame) -> dict:
    global MODEL, SCALER_X, SCALER_Y, DF_CLEAN, EXPECTED_INPUT_SHAPE, FEATURE_COLS
    
    if MODEL is None or DF_CLEAN is None:
        return {"success": False, "log": "ERROR: El modelo o DF_CLEAN no están cargados."}

    new_rows = []
    
    for _, row in new_data_df.iterrows():
        pid = row['product_id']
        pred_date = pd.to_datetime(row['fecha_predicha'])
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

# --- 6. Nuevo Endpoint de Conclusión de IA ---

@app.post("/api/conclusion")
async def api_conclusion(req: ConclusionRequest):
    """
    Recibe los resultados de la tabla y utiliza Gemini (via Vertex AI) para generar una conclusión.
    """
    global VERTEX_CLIENT_READY, VERTEX_MODEL

    if not VERTEX_CLIENT_READY:
        raise HTTPException(status_code=503, detail="El cliente de Vertex AI no está configurado o no se pudo autenticar. Verifique Project ID y permisos de VM.")
        
    if not req.results:
        return {"conclusion": "No hay resultados para analizar."}

    # 1. Formatear los datos para el LLM
    data_str = "Resultados de Predicción de Stock:\n"
    data_str += "---------------------------------------------------------\n"
    data_str += "ID | Stock Predicho | Necesita Reabastecer\n"
    data_str += "---|----------------|-----------------------\n"
    
    for item in req.results:
        # Intenta obtener 'needs_restock' (de la pred. general) o calcúlalo (de la pred. unitaria)
        needs_restock = "Sí" if item.get('needs_restock', False) or (item.get('quantity_on_hand', 0) <= 20 and 'needs_restock' not in item) else "No"
        stock = f"{item.get('quantity_on_hand', 0):.2f}"
        data_str += f"{item.get('product_id')} | {stock} | {needs_restock}\n"

    # 2. Instrucción (Prompt) para el LLM
    prompt = (
        f"Analiza los siguientes resultados de predicción de inventario para un supermercado. "
        f"El umbral de reabastecimiento es 20 unidades. Genera una conclusión en español que sea útil para la gerencia, "
        f"cubriendo los siguientes puntos:\n"
        f"1. Productos clave que requieren atención inmediata (stock <= 5).\n"
        f"2. Un resumen del porcentaje de productos que necesitan reabastecimiento (stock <= 20).\n"
        f"3. Una recomendación de acción breve.\n\n"
        f"--- DATOS ---\n{data_str}"
    )

    try:
        # 3. LLAMADA A VERTEX AI DENTRO DEL BLOQUE TRY
        
        # 3.1 Inicializar el modelo Fundacional
        model = GenerativeModel(VERTEX_MODEL)

        # 3.2 Generar el contenido
        response = model.generate_content(
            contents=prompt
        )
        
        return {"conclusion": response.text}

    except Exception as e:
        print(f"Error al llamar a la API de Vertex AI: {e}")
        return {"conclusion": f"Error al generar la conclusión. Verifique el log de Uvicorn: {e}"}

# --- 7. Rutas de la API (Endpoints) ---

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
