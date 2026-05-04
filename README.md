# Evolución del sentimiento hacia la inteligencia artificial en X: un estudio longitudinal en español (2022-2025)

**Trabajo de Fin de Grado — Grado en Análisis de Datos en la Empresa (Business Analytics)**  
Universidad Autónoma de Madrid · Curso 2025/2026

**Autora:** Isabela Bulgaru Merla  
**Tutor:** Ramón Mahía Casado

---

## Descripción

Este repositorio contiene el código y los datos del TFG, que analiza la evolución del sentimiento de la comunidad hispanohablante sobre la inteligencia artificial en X (anteriormente Twitter) entre junio de 2022 y diciembre de 2025.

El análisis combina:
- **Análisis de sentimiento** mediante RoBERTuito, modelo preentrenado en 500 millones de tuits en español.
- **Modelado de tópicos** mediante BERTopic, con embeddings multilingües, reducción dimensional UMAP y clustering HDBSCAN.
- **Análisis estadístico** de series temporales con el test de Pettitt y el test de Mann-Whitney U.

El corpus está compuesto por **58.496 tuits en español** extraídos de X entre junio de 2022 y diciembre de 2025.

---

## Estructura del repositorio

```
TFG/
├── data/
│   ├── raw/                  # Datos brutos extraídos de X
│   └── processed/            # Corpus limpio con etiquetas de sentimiento y tópicos
├── scripts/
│   ├── 01_limpieza_datos.py          # Filtrado, normalización geográfica y feature engineering
│   ├── 02_preprocesamiento_texto.py  # Limpieza de texto para sentimiento y tópicos
│   ├── 03_sentimiento_robertuito.py  # Etiquetado automático con RoBERTuito
│   ├── 04_modelado_topicos.py        # Pipeline BERTopic: embeddings, UMAP, HDBSCAN
│   └── 05_analisis_resultados.py     # Generación de gráficos y tests estadísticos
├── outputs/
│   └── figures/              # Figuras generadas
├── requirements.txt
├── CITATION.cff
└── README.md
```

---

## Pipeline

El análisis sigue un pipeline secuencial de cinco fases. Cada script lee un CSV de entrada y genera un CSV de salida:

| Fase | Script | Entrada | Salida |
|------|--------|---------|--------|
| 1. Limpieza | `01_limpieza_datos.py` | Datos brutos | Corpus limpio |
| 2. Preprocesamiento | `02_preprocesamiento_texto.py` | Corpus limpio | Corpus preprocesado |
| 3. Sentimiento | `03_sentimiento_robertuito.py` | Corpus preprocesado | Corpus con etiquetas |
| 4. Tópicos | `04_modelado_topicos.py` | Corpus preprocesado | Corpus con tópicos |
| 5. Resultados | `05_analisis_resultados.py` | Corpus completo | Figuras y estadísticos |

---

## Instalación

```bash
git clone https://github.com/kradshiny/TFG.git
cd TFG
pip install -r requirements.txt
```

El análisis de sentimiento requiere GPU para tiempos razonables. Se ha desarrollado con Google Colab (T4 GPU). El tiempo de inferencia sobre el corpus completo es de aproximadamente 55 minutos.

---

## Parámetros principales

**BERTopic:**
- Modelo de embeddings: `paraphrase-multilingual-MiniLM-L12-v2`
- UMAP: 15 vecinos, 5 componentes
- HDBSCAN: tamaño mínimo de clúster 200, 10 muestras mínimas

**RoBERTuito:**
- Modelo: `pysentimiento/robertuito-sentiment-analysis`
- Inferencia en batches de 1.000 tuits

---

## Datos

Los datos brutos no se incluyen en el repositorio por restricciones de tamaño y de los términos de uso de X. El corpus procesado (sin texto original) está disponible en la carpeta `data/processed/`.

La extracción se realizó mediante [twitterapi.io](https://twitterapi.io) con las palabras clave `"Inteligencia Artificial"`, `"ChatGPT"` e `"IA"`, filtrando por idioma español y excluyendo retuits y respuestas.

---

## Cómo citar

Si usas este código o los datos en tu investigación, por favor cita el trabajo usando el archivo `CITATION.cff` o el siguiente formato:

> Bulgaru Merla, I. (2026). *Evolución del sentimiento hacia la inteligencia artificial en X: un estudio longitudinal en español (2022-2025)*. Trabajo de Fin de Grado, Universidad Autónoma de Madrid. https://github.com/kradshiny/TFG

---

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
