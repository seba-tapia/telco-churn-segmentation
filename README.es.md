# Telco Churn — Segmentación y Predicción de Fuga de Clientes

Proyecto end-to-end que identifica **qué clientes de una empresa de telecomunicaciones tienen mayor riesgo de irse (churn)**, los agrupa en **segmentos accionables**, y expone esa información a través de una **API** y un **dashboard interactivo**.

---

## 📌 Resumen para negocio (sin conocimientos técnicos)

Esta sección está pensada para marketing, retención de clientes, o cualquier persona del negocio que quiera entender **qué hace el proyecto y qué decisiones puede tomar con él**, sin entrar en el detalle técnico.

### ¿Qué problema resuelve?

De cada 10 clientes de la empresa, **cerca de 3 terminan yéndose (churn)**. Este proyecto analiza el historial de clientes (cuánto pagan, hace cuánto son clientes, qué servicios usan, cómo se comunican con soporte, etc.) para responder dos preguntas:

1. **¿Qué tipos de clientes tenemos?** (segmentación)
2. **¿Cuáles tienen más probabilidad de irse, y por qué?** (predicción + explicación)

Con esto, el equipo de retención puede **priorizar a quién contactar primero** en vez de tratar a todos los clientes por igual.

### Los 4 segmentos de clientes que encontramos

| Segmento | Quiénes son | Tasa de fuga | Qué hacer |
|---|---|---|---|
| 🟢 **Leales de bajo costo** | Clientes antiguos (más de 4 años), pagan poco por mes, casi no usan servicios extra | **3%** — casi nadie se va | No requieren atención urgente. Son la base estable del negocio. |
| 🔵 **Premium leales** | Clientes antiguos, pagan mucho por mes, usan varios servicios adicionales (seguridad, soporte técnico, streaming) | **13%** | Cuidarlos con beneficios exclusivos — son los de mayor valor y ya están comprometidos. |
| 🟠 **En riesgo con alto gasto** | Clientes de antigüedad media, pagan bastante, pero usan pocos servicios adicionales | **33%** | Foco de retención: son clientes valiosos que se están yendo. Ofrecerles servicios que aumenten su compromiso (soporte, seguridad) puede evitar la fuga. |
| 🔴 **Nuevos de bajo compromiso** | Clientes muy recientes (menos de 1 año), pagan un monto medio, casi no usan servicios adicionales | **37%** — el grupo más riesgoso | Foco de onboarding: reforzar la experiencia en los primeros meses (los primeros 90 días parecen ser críticos). |

**El hallazgo más importante para el negocio**: no es cuánto paga el cliente lo que más influye en la fuga, sino **la combinación de antigüedad baja + poco uso de servicios adicionales**. Un cliente nuevo que no está usando los servicios extra (seguridad online, soporte técnico premium, protección de dispositivo) es la señal de alerta más temprana que tenemos, incluso antes de que se acerque la fecha de renovación de su contrato.

### ¿Qué tan confiable es el modelo que predice la fuga?

En lenguaje simple: si tomamos 10 clientes que **realmente** se van a ir, el modelo detecta correctamente a **entre 4 y 7 de ellos**, dependiendo de qué tan "sensible" configuremos la alerta:

- **Configuración conservadora** (menos alertas, pero más precisas): detecta a 4 de cada 10 que se van, y casi todas las alertas que genera son acertadas.
- **Configuración sensible** (más alertas, aunque algunas sean falsas alarmas): detecta a 6-7 de cada 10 que se van, a cambio de generar más alertas sobre clientes que en realidad no se iban a ir.

**La decisión de qué configuración usar es una decisión de negocio, no técnica**: depende de cuánto cuesta contactar a un cliente que no se iba a ir (una llamada, un descuento) versus cuánto cuesta perder a un cliente que sí se iba (todo su valor futuro). Esa decisión debe tomarla el equipo de retención junto con el equipo de datos — está documentada en detalle en `06_Churn_Model.ipynb` para cuando se quiera avanzar en eso.

### Limitaciones honestas de este análisis

- El modelo está entrenado con datos históricos de un período específico; si el negocio cambia (nueva competencia, nuevos planes, etc.), el modelo debería reentrenarse.
- El modelo predice probabilidad de fuga, no garantiza nada — es una herramienta de priorización, no una bola de cristal.
- Existía una variable en los datos originales ("puntaje de satisfacción") que parecía predecir el churn casi perfectamente, pero al investigarla se determinó que era una consecuencia del churn, no una causa (se mide después de que el cliente ya decidió irse). Por eso se excluyó del modelo — de lo contrario, el modelo hubiera parecido mucho más preciso de lo que realmente es en la práctica.

---

## 🛠️ Documentación técnica

### Arquitectura del proyecto

```
Datos crudos (CSV)
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   Limpieza   │ ──▶ │   Feature    │ ──▶ │  Segmentación  │ ──▶ │   Modelado    │
│  (clean.py)  │     │ Engineering  │     │ (clustering.py)│     │  (train.py)   │
└─────────────┘     └──────────────┘     └───────────────┘     └──────┬───────┘
                                                                        │
                                                              Selección automática
                                                              del mejor modelo por
                                                              ROC-AUC (select.py)
                                                                        │
                                                                        ▼
                                                          models/churn_model.joblib
                                                                        │
                                              ┌─────────────────────────┴───────────────────────┐
                                              ▼                                                   ▼
                                    API REST (api/main.py)                          Dashboard (dashboard.py)
                                    /predict, /predict_batch                        Segmentos, PCA, funnel,
                                                                                     drivers, predicción individual
```

Todo el pipeline es **config-driven**: rutas, columnas, hiperparámetros y umbrales viven en `config/config.yaml`, no hardcodeados en el código.

### Estructura del repositorio

```
telco-churn-segmentation/
├── config/
│   └── config.yaml              # Configuración central del proyecto
├── data/
│   ├── raw/                     # Dataset original (Telco-Customer-Churn.csv)
│   └── processed/               # Datos limpios, con features, y segmentados
├── models/
│   ├── churn_model.joblib       # Artefacto del modelo ganador (+ scaler si aplica)
│   └── metrics_report.json      # Métricas comparativas de los 3 modelos evaluados
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Segmentation.ipynb
│   ├── 05_Funnel_Analysis.ipynb
│   └── 06_Churn_Model.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load.py
│   │   └── clean.py
│   ├── features/
│   │   └── engineering.py
│   ├── segmentation/
│   │   └── clustering.py
│   └── modeling/
│       ├── train.py
│       ├── evaluate.py
│       └── select.py
├── api/
│   ├── main.py                  # API FastAPI (predicción + drivers)
│   └── model_loader.py          # Carga el artefacto entrenado (no reentrena)
├── dashboard.py                 # Dashboard Streamlit
├── run_pipeline.py              # Orquestador: corre el pipeline completo o por pasos
└── requirements.txt
```

### Cómo correr el proyecto

**1. Instalar dependencias**

```bash
pip install -r requirements.txt
```

**2. Correr el pipeline completo** (limpieza → features → segmentación → entrenamiento → selección del mejor modelo → export)

```bash
python run_pipeline.py --step all
```

También se puede correr paso a paso para debugging:

```bash
python run_pipeline.py --step load           # solo carga
python run_pipeline.py --step clean          # carga + limpieza
python run_pipeline.py --step features       # + feature engineering
python run_pipeline.py --step segmentation   # + clustering
python run_pipeline.py --step modeling       # + entrenamiento, evaluación y export del modelo
```

Al finalizar, vas a tener `models/churn_model.joblib` (el modelo ganador, seleccionado automáticamente por ROC-AUC) y `models/metrics_report.json` (métricas de los 3 modelos evaluados).

**3. Levantar la API**

```bash
uvicorn api.main:app --reload
```

Documentación interactiva en `http://127.0.0.1:8000/docs`. Endpoints disponibles:

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Health check — confirma que la API está viva y qué modelo tiene cargado |
| `/predict` | POST | Predicción de churn + drivers para un cliente |
| `/predict_batch` | POST | Predicción para una lista de clientes |

**4. Levantar el dashboard**

```bash
streamlit run dashboard.py
```

Incluye: distribución de segmentos, visualización PCA, funnel de estado de clientes, drivers del modelo (SHAP o coeficientes según el modelo ganador), y predicción individual interactiva.

### Resultados técnicos actuales

| Modelo | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** (ganador) | 0.789 | 0.435 | 0.522 | **0.831** |
| Random Forest | 0.786 | 0.437 | 0.520 | 0.827 |
| XGBoost | 0.781 | 0.463 | 0.529 | 0.820 |

El modelo ganador se selecciona automáticamente por ROC-AUC en `src/modeling/select.py`. Los tres modelos quedan muy cerca entre sí porque, tras eliminar variables con data leakage, la relación entre las features y el churn es mayormente lineal.

**Decisiones metodológicas documentadas:**
- Se excluyó `SatisfactionScore` del modelo y del clustering por ser una variable con data leakage severo (ver conclusiones de `06_Churn_Model.ipynb` y `04_Segmentation.ipynb`).
- `k=4` en el clustering no maximiza el silhouette score (que es máximo en k=2), pero se eligió por generar segmentos con tasas de churn más diferenciadas y accionables para negocio (ver conclusiones de `04_Segmentation.ipynb`).
- El threshold de decisión recomendado (0.35 en vez del 0.5 por defecto) está documentado en `06_Churn_Model.ipynb` y se guarda en el artefacto (`churn_model.joblib`) al momento de exportar el modelo, pero todavía no se lee ni se aplica en las respuestas de la API/dashboard — queda como mejora futura sujeta a validación con el equipo de retención.

### Próximas mejoras posibles

- Leer `decision_threshold` desde el artefacto en `model_loader.py` y aplicarlo en `main.py`/`dashboard.py` (por ejemplo, agregando un campo `at_risk: true/false` a la respuesta de la API calculado con ese umbral en vez de asumir el 0.5 implícito de `predict_proba`). El valor ya se persiste en `churn_model.joblib`; falta el último paso de leerlo y usarlo.
- Fijar versiones exactas de las librerías en `requirements.txt` (`pip freeze`) para reproducibilidad garantizada.
- Agregar tests automatizados para el pipeline (carga, limpieza, features) y para los endpoints de la API.
- Reentrenar periódicamente con datos nuevos y trackear si el modelo ganador cambia con el tiempo (model drift).
- Los notebooks (`01`–`06`) duplican parte de la lógica de `src/*.py` en vez de importarla, para mantener la exploración separada de la producción. Esto ya generó una vez una inconsistencia real (un fix aplicado en `clean.py` no se propagó a `02_Cleaning.ipynb`), así que vale la pena revisar periódicamente que ambos lados coincidan, o migrar los notebooks a importar directamente desde `src/`.
