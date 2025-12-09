===========================================
Documentación Técnica - Predictor de Stock (LSTM + Vertex AI)
===========================================

ÍNDICE
------
1. Descripción general
  1.1 Funcionalidades Clave
2. Dependencias y requisitos
  2.1 Requisitos de Software
  2.2 Requisitos de Hardware (Instancia de VM)
  2.3 Requisitos de Autenticación (Google Cloud)
3. Estructura del proyecto
4. Configuración del Entorno y Despliegue
  4.1 Clonación e Instalación
  4.2 Configuración de Vertex AI
  4.3 Ejecución del Servidor
5. Endpoints de la API
6. Lógica de Reentrenamiento y Límite de Datos

----------------------
1. DESCRIPCIÓN GENERAL
----------------------
Este proyecto implementa un sistema de predicción de inventario y reabastecimiento para un supermercado utilizando técnicas de Machine Learning (Redes Neuronales Recurrentes) y servicios de Inteligencia Artificial Generativa de Google Cloud.
El objetivo principal es predecir el nivel de stock (`quantity_on_hand`) de productos para una fecha futura y generar análisis ejecutivos sobre las necesidades de reabastecimiento.

	1.1 Funcionalidades Clave
  * Forecasting de Stock: Predicción de series de tiempo basada en una ventana de 7 días, utilizando un modelo LSTM que maneja 13 features por paso de tiempo.
  * Aprendizaje Continuo (Retrain): Permite añadir datos de predicción a los históricos y reentrenar el modelo con un *epoch* para mantener la precisión.
  * Análisis Gerencial (LLM): Utiliza la API de Vertex AI (Gemini) para analizar los resultados de la tabla y generar conclusiones sobre riesgos y acciones de reabastecimiento.
  * API Web: Interfaz web simple construida con FastAPI y Uvicorn para interactuar con el modelo y los servicios.


----------------------
2. DEPENDENCIAS Y REQUISITOS
----------------------
Para ejecutar el proyecto, se requiere un entorno Python 3.9+ y las siguientes bibliotecas, detalladas en `requirements.txt`.

	2.1 Requisitos de Software
  * Python: Versión 3.9 o superior.
  * Framework Web: FastAPI
  * Servidor ASGI: Uvicorn
  * Machine Learning: TensorFlow/Keras, NumPy, Pandas, Scikit-learn
  * Google Cloud: google-cloud-aiplatform y vertexai (para la funcionalidad de conclusión).

	2.2 Requisitos de Hardware (Instancia de VM)
	┌───────────┬────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐
	│   RECURSO │      REQUISITO MINIMO      │         RAZON 										       │
	├───────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
	│Tipo de VM │ `e2-micro` (GCP Free Tier) │ Suficiente para servir la API y realizar inferencia (la carga pesada fue en el entrenamiento).      │
	│    RAM    │            1 GB            │ Mínimo para cargar TensorFlow/Keras. 							       │
	│   Disco   │    10 GB (SSD/Standard)    │ Suficiente para el sistema operativo y el almacenamiento del modelo (620 MB de TensorFlow + datos). │
	└───────────┴────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┘

	2.3 Requisitos de Autenticación (Google Cloud)
	Para que el endpoint `/api/conclusion` funcione, la VM de Google Cloud debe tener asignado un Service Account con los siguientes roles (o el rol genérico de Vertex AI User):
	  * Vertex AI User
	  * Service Usage Consumer (Para que la API de Vertex AI pueda ser consumida).

----------------------
3. ESTRUCTURA DEL PROYECTO
----------------------
.
└── aa_aplicacion_prediccion_MML/
    ├── .venv/                         # Entorno virtual de Python (Ignorado por Git)
    ├── .gitignore                     # Define archivos grandes y binarios a excluir
    ├── app.py                         # **Núcleo de la API, carga de modelos, lógica de negocio y endpoints.**
    ├── requirements.txt               # Lista de dependencias de Python
    ├── models/
    │   ├── best_model.keras           # Modelo LSTM entrenado
    │   ├── scaler_X.pkl               # Escalador para las variables de entrada (X)
    │   ├── scaler_y.pkl               # Escalador para la variable objetivo (Y)
    │   ├── products.csv               # Información estática de los productos
    │   └── dataset_limpio_quantity_on_hand.csv # Histórico de datos de series de tiempo
    ├── static/
    │   └── style.css                  # Hoja de estilos para la interfaz web
    └── templates/
        └── final_predict.html         # Interfaz de usuario (Frontend)

----------------------
4. CONFIGURACION DEL ENTORNO Y DESPLIEGUE
----------------------
	4.1 Clonación e Instalación
		Clonar el repositorio
			git clone https://github.com/David5Uzhca/aa_aplicacion_prediccion_MML.git

		Crear e instalar dependencias
			python3 -m venv .venv
			source .venv/bin/activate
			pip install -r requirements.txt

	4.2 Configuración de Vertex AI
	Establecer la región y el ID del proyecto de GCP.
		VERTEX_PROJECT_ID = "id-de-proyecto-gcp"
		VERTEX_REGION = "región-de-tu-VM"

	4.3 Ejecución del Servidor
			uvicorn app:app --host 0.0.0.0 --port 8000

----------------------
5. ENDPOINTS DE LA API
----------------------
	┌───────────────┬──────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
	│      RUTA     │      METODO      │            DESCRIPCION 		               									      │
	├───────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
	│/final-predict │       GET        │                                 Interfaz Web: Retorna el final_predict.html (Frontend).                                  │
	│    /health    │       GET        │                       Estado: Verifica la carga de modelos, scalers y la conectividad a Vertex AI.                       │
	│ /api/predict  │       POST       │                    Predicción Unitaria: Calcula el stock futuro para un product_id y date específico.                    │
	│ /api/restock  │       POST       │                      Predicción General: Calcula el stock futuro para los 30 productos principales.                      │
	│ /api/retrain  │       POST       │  Reentrenamiento: Recibe la tabla de resultados, añade los datos y ejecuta un epoch de entrenamiento en el modelo LSTM.  │
	│/api/conclusion│       POST       │         Análisis LLM: Recibe los resultados y retorna una conclusión generada por el modelo Gemini de Vertex AI.         │
	└───────────────┴──────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

----------------------
6. LOGICA DE REENTRENAMIENTO Y LIMITE DE DATOS
----------------------
La función retrain_model opera bajo los siguientes principios:

  * Ingesta: Los resultados predichos (quantity_on_hand) se simulan con las 12 features auxiliares (usando la inercia de la última fila real) y se añaden al dataset_limpio_quantity_on_hand.csv.
  * Secuencia: Se crea una nueva secuencia de entrada (7 días antes del último dato) y el objetivo (el último stock añadido) para el reentrenamiento.
  * Proceso: El modelo global (MODEL`) se actualiza mediante un único epoch (epochs=1) sobre el pequeño dataset de nuevas secuencias, garantizando un aprendizaje rápido sin sobrecargar la VM.


