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

class PredictRequest(BaseModel):
    product_id: str
    date: Optional[str] = None

class RestockRequest(BaseModel):
    date: str
    threshold: Optional[float] = 20.0

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


try:
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

except Exception as e:
    print(f"Error al cargar recursos del modelo: {e}")


# ... (código anterior) ...

def get_last_7_rows(product_id: str):
    """Obtiene las últimas 7 filas NO ESCALADAS de un producto, rellenando si es necesario."""
    if DF_CLEAN is None: return None, None
    
    # Usamos .str.strip() para evitar errores de espacios en los IDs (mantenemos la corrección)
    prod = DF_CLEAN[DF_CLEAN["product_id"].str.strip() == product_id.strip()].tail(7)
    
    # Extraemos solo las features
    X = prod[FEATURE_COLS].values.astype(np.float32)
    
    if len(prod) == 0:
        return None, None # Si no hay datos, sigue fallando (404)

    # Si faltan filas (menos de 7), rellenamos la ventana con ceros al inicio (pre-padding)
    if X.shape[0] < 7:
        t_exp = 7
        f_exp = X.shape[1] 
        # Creamos una matriz de ceros del tamaño faltante
        pad = np.zeros((t_exp - X.shape[0], f_exp))
        # Apilamos los ceros encima de los datos existentes
        X = np.vstack([pad, X])
        print(f"DEBUG: Rellenado de {product_id} con {t_exp - X.shape[0]} filas de cero.")
    
    last_date = prod["created_at"].iloc[-1]
    return X, last_date # X ahora siempre tiene la forma (7, 13)

def predict_stock_by_date(product_id: str, target_date_str: str):
    """
    Predice el stock futuro simulando la evolución de las 13 features.
    """
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



# ... (después de la función predict_stock_by_date) ...

# --- 6. Función de Reentrenamiento y Actualización de Datos ---

def retrain_model(new_data_df: pd.DataFrame) -> dict:
    """
    Añade los nuevos datos al DF_CLEAN, recalcula escaladores, 
    crea nuevas secuencias y reentrena el modelo por 1 epoch.
    Retorna métricas y logs.
    """
    global MODEL, SCALER_X, SCALER_Y, DF_CLEAN, EXPECTED_INPUT_SHAPE, FEATURE_COLS
    
    # Asegurarse de que los recursos clave existan
    if MODEL is None or DF_CLEAN is None:
        return {"success": False, "log": "ERROR: El modelo o DF_CLEAN no están cargados."}

    # --- 6.1 INGESTA DE NUEVOS DATOS (Simulación de 13 features) ---
    
    # Usamos la última fecha de DF_CLEAN para los nuevos datos
    last_real_date = DF_CLEAN['created_at'].max()
    
    # Crear un DataFrame con los nuevos datos, asumiendo que new_data_df tiene:
    # product_id, fecha_predicha, quantity_on_hand
    
    # Esto es complejo, ya que solo tenemos 'quantity_on_hand' (feature 0) y 'fecha_predicha'.
    # Para la ingesta, vamos a crear filas simuladas con las 12 features auxiliares 
    # basándonos en la última fila conocida de cada producto (similar a la predicción).

    new_rows = []
    
    for _, row in new_data_df.iterrows():
        pid = row['product_id']
        pred_date = pd.to_datetime(row['fecha_predicha'])
        pred_stock = row['quantity_on_hand']
        
        # Obtener la última fila real para simular la inercia de las 12 features
        last_row_data = DF_CLEAN[DF_CLEAN['product_id'] == pid].tail(1)

        if not last_row_data.empty:
            last_row = last_row_data[FEATURE_COLS].values[0]
            last_date = last_row_data['created_at'].iloc[0]

            # Calcular el número de días a predecir
            days_to_add = (pred_date - last_date).days
            
            # Si la predicción es hoy o en el pasado, no la añadimos como nueva fila (ya está cubierta)
            if days_to_add <= 0:
                continue

            # La simulación más simple: Usar la última fila real como base para los auxiliares
            simulated_row = last_row.copy()
            simulated_row[0] = pred_stock # Actualizar con el stock predicho
            
            # Aquí podríamos simular la evolución de los 12 features auxiliares (índice 1 en adelante)
            # Para simplificar el reentrenamiento, solo añadiremos 1 fila: la predicción.
            
            new_data = {
                'product_id': pid,
                'created_at': pred_date,
                **{col: val for col, val in zip(FEATURE_COLS, simulated_row)}
            }
            new_rows.append(new_data)
            
    if not new_rows:
         return {"success": True, "log": "Advertencia: No se generaron nuevas filas para añadir (fechas pasadas/actuales)."}

    # Convertir a DataFrame y estandarizar columnas
    new_df = pd.DataFrame(new_rows)
    
    # Seleccionar solo las columnas necesarias para DF_CLEAN
    cols_to_keep = DF_CLEAN.columns.tolist() 
    new_df = new_df[[col for col in new_df.columns if col in cols_to_keep]] # Asegurarse de que las columnas coincidan
    
    # Añadir los nuevos datos al DataFrame limpio global
    DF_CLEAN = pd.concat([DF_CLEAN, new_df], ignore_index=True)
    
    # Guardar el DataFrame actualizado en el CSV (aprenderá de ellos en el futuro)
    DF_CLEAN.to_csv(str(DATASET_PKL), index=False)
    
    log = f"Datos añadidos: {len(new_rows)} nuevas filas. DF_CLEAN total: {len(DF_CLEAN):,}.\n"
    
    # --- 6.2 PREPARACIÓN DE SECUENCIAS PARA REENTRENAMIENTO ---
    
    # **NOTA DE IMPLEMENTACIÓN:** La creación de secuencias (fase 1) es compleja y pesada. 
    # Para un reentrenamiento eficiente, se reentrenará usando SOLO la última secuencia conocida 
    # para cada producto, incluyendo la nueva fila.
    
    X_retrain, y_retrain = [], []
    t_exp, f_exp = EXPECTED_INPUT_SHAPE
    
    for pid in DISPLAY_PRODUCT_IDS:
        # Obtener las últimas 7 filas (incluyendo la nueva fila si está en DF_CLEAN)
        prod = DF_CLEAN[DF_CLEAN['product_id'] == pid]
        
        # Requiere al menos t_exp + 1 para formar una secuencia (Input X y Target Y)
        if len(prod) >= t_exp + 1:
            # Seleccionar la última ventana de entrada (X) y el siguiente target (Y)
            
            # X: últimos t_exp (7) días
            X_seq = prod[FEATURE_COLS].tail(t_exp).values.astype(np.float32)
            
            # Y: el valor del día siguiente (la nueva predicción real)
            # Como estamos prediciendo el mismo día, necesitamos una lógica de Y diferente.
            
            # **Asunción simplificada:** La predicción que acabamos de añadir es el nuevo Y.
            # Volvemos a generar la secuencia de 7 días ANTES de la nueva fila.
            X_input_raw = prod[FEATURE_COLS].iloc[-t_exp-1:-1].values.astype(np.float32) 
            y_target_raw = prod[FEATURE_COLS].iloc[-1, 0] # El stock del último día añadido

            # Asegurar que X_input_raw tiene la forma correcta (t_exp, f_exp)
            if X_input_raw.shape == EXPECTED_INPUT_SHAPE:
                X_retrain.append(X_input_raw)
                y_retrain.append(y_target_raw)
                
    if not X_retrain:
        log += "Advertencia: No hay secuencias completas (7+1) para reentrenar. No se realizó el reentrenamiento.\n"
        return {"success": True, "log": log}

    X_retrain = np.array(X_retrain)
    y_retrain = np.array(y_retrain).reshape(-1, 1)

    # 6.3 ESCALADO y REENTRENAMIENTO
    
    # Volvemos a escalar los datos de reentrenamiento con los escaladores existentes
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


# --- 7. Nuevo Endpoint de la API ---
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
        "total_products_available": len(ALL_PRODUCT_IDS),
        "products_in_interface": len(DISPLAY_PRODUCT_IDS),
        "expected_input_shape": EXPECTED_INPUT_SHAPE
    }