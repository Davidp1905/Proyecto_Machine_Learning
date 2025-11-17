# app.py -- run: uvicorn app:app --reload --port 8001
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# ============================
# Inicializar FastAPI
# ============================

app = FastAPI()

# Permitir llamadas desde Angular en localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Cargar artefactos del modelo
# ============================

model = joblib.load("modelo_random_forest_final.pkl")
selector = joblib.load("selector_kbest.pkl")
columnas_originales = joblib.load("columnas_originales.pkl")

PROMEDIO_LINEAS_MEDIA = 0.0   
PROMEDIO_AREAS_MEDIA = 0.0    


# ============================
# Modelo de entrada (lo que manda Angular)
# ============================

class EntradaModelo(BaseModel):
    genero: str                          # "Masculino" / "Femenino"
    campesino: bool
    estrato: int                         # 0 a 6
    nivel_educacion: str                 # texto: "Educación Primaria", etc.
    discapacidad: bool
    tipo_formacion: str                  # "Virtual" / "Híbrida"
    victima_conflicto: bool
    tiempo_segundos: float
    puntaje_eje: float                   # Puntaje_eje_tematico_selecionado
    autoidentificacion_etnica: str       # "Indígena", "Afro...", "Ningún grupo étnico", "Rrom o gitano"
    eje_final: str                       # "Análisis de Datos", ...
    nivel: str                           # "Básico" / "Avanzado"

    # OPCIONAL: el usuario puede escribirlos o no
    promedio_lineas: float | None = None
    promedio_areas: float | None = None


# ============================
# Construir la fila para el modelo
# ============================

def construir_fila(data: EntradaModelo) -> pd.DataFrame:
    """
    Construye un DataFrame con UNA fila y todas las columnas
    que espera el modelo, a partir de la entrada del usuario.
    """

    # Inicializar todas las columnas en 0
    row = {col: 0 for col in columnas_originales}

    # ---------- Variables numéricas / binarias ----------

    # Genero: mujer=0, hombre=1
    row["Genero"] = 1 if data.genero.lower() == "masculino" else 0

    row["Campesino"] = 1 if data.campesino else 0
    row["Estrato"] = data.estrato

    # Mapeo nivel_educacion (1 a 7)
    nivel_map = {
        "Educación Primaria": 1,
        "Educación Media": 2,
        "Educación Secundaria Básica": 3,
        "Educación Técnica Profesional": 4,
        "Educación Tecnológica": 5,
        "Educación Universitaria Pregrado": 6,
        "Especialización": 7,
    }
    row["NIvel_educacion"] = nivel_map.get(data.nivel_educacion, 3)  # default 3

    row["Discapacidad"] = 1 if data.discapacidad else 0
    row["Victima_del_conflicto"] = 1 if data.victima_conflicto else 0

    # Tipo_formacion: Virtual=0, Híbrida=1
    row["Tipo_formacion"] = 1 if data.tipo_formacion == "Híbrida" else 0

    # Tipo_de_formacion_Virtual: 1 si Virtual
    if "Tipo_de_formacion_Virtual" in row:
        row["Tipo_de_formacion_Virtual"] = 1 if data.tipo_formacion == "Virtual" else 0

    # tiempo y puntaje
    row["tiempo_segundos"] = data.tiempo_segundos
    row["Puntaje_eje_tematico_selecionado"] = data.puntaje_eje

    # ---------- Etnia → etnica_* ----------
    etnia = data.autoidentificacion_etnica.lower()
    if "indígena" in etnia:
        row["etnica_Indígena"] = 1
    elif "negro" in etnia or "afro" in etnia:
        row["etnica_Negro, Mulato, Afrodescendiente, Afrocolombiano"] = 1
    elif "rrom" in etnia or "gitano" in etnia:
        row["etnica_Rrom o gitano"] = 1
    else:
        row["etnica_Ningún grupo étnico"] = 1

    # ---------- eje_final_* ----------
    eje = data.eje_final.lower()
    if "análisis" in eje:
        row["eje_final_Análisis de Datos"] = 1
    elif "arquitectura" in eje:
        row["eje_final_Arquitectura en la nube"] = 1
    elif "inteligencia" in eje:
        row["eje_final_Inteligencia artificial"] = 1
    elif "programación" in eje:
        row["eje_final_Programación"] = 1

    # ---------- Nivel_* ----------
    if data.nivel.lower() == "avanzado":
        if "Nivel_Avanzado" in row:
            row["Nivel_Avanzado"] = 1
        if "Nivel_Básico" in row:
            row["Nivel_Básico"] = 0
    else:
        if "Nivel_Básico" in row:
            row["Nivel_Básico"] = 1
        if "Nivel_Avanzado" in row:
            row["Nivel_Avanzado"] = 0

    # ---------- lineas_* ----------
    # El usuario normal no conoce estas columnas.
    # Se quedan en 0 (ya están así desde el inicio).

    # ---------- promedio_lineas / promedio_areas ----------

    # OPCIÓN 1 (POR DEFECTO): usar el promedio del dataset
    if "promedio_lineas" in row:
        row["promedio_lineas"] = PROMEDIO_LINEAS_MEDIA
    if "promedio_areas" in row:
        row["promedio_areas"] = PROMEDIO_AREAS_MEDIA

    # OPCIÓN 2: usar los valores digitados por el usuario
    # (DESCOMENTAR SI QUIERES USARLOS)
    #
    if "promedio_lineas" in row and data.promedio_lineas is not None:
        row["promedio_lineas"] = data.promedio_lineas
    if "promedio_areas" in row and data.promedio_areas is not None:
        row["promedio_areas"] = data.promedio_areas

    # Construir DataFrame
    df_row = pd.DataFrame([row])

    # Asegurar orden de columnas original
    df_row = df_row[columnas_originales]

    return df_row


# ============================
# Endpoint de predicción
# ============================

@app.post("/predict")
def predict(data: EntradaModelo):
    # 1. Construir DataFrame completo
    X_full = construir_fila(data)

    # 2. Aplicar SelectKBest (el mismo que usaste al entrenar)
    X_kbest = selector.transform(X_full)

    # 3. Predecir con el modelo XGBoost final
    prob = model.predict_proba(X_kbest)[0, 1]
    pred = model.predict(X_kbest)[0]

    return {
        "probabilidad_exito": float(prob),
        "prediccion": int(pred)
    }
