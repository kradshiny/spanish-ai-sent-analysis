# =============================================================================
# 04_modelado_topicos.py
# Modelado de tópicos con LDA (selección de k) y BERTopic
# Input:  data_sentiment.csv
# Output: data_final.csv
# =============================================================================
#%%
import pandas as pd
import ast
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------------------

data_wo_na = pd.read_csv('data_sentiment.csv', low_memory=False)

#%%
topic_model = BERTopic.load("modelo_bertopic")
docs = data_wo_na['text_bertopic'].tolist()
topics, probs = topic_model.transform(docs)

#%%
# -----------------------------------------------------------------------------
# 2. BERTOPIC
# -----------------------------------------------------------------------------

stopwords_es = [
    # Lista actual
    "la", "de", "en", "el", "que", "para", "con", 
    "una", "los", "las", "su", "por", "más", "se", 
    "no", "es", "al", "del", "un", "le", "lo", "a",
    
    # Pronombres y determinantes
    "me", "mi", "yo", "te", "tu", "si", "nos", "este",
    "esta", "esto", "ese", "esa", "sus", "nuestro", "nuestra",
    "nuestros", "nuestras", "ellos", "ellas", "él", "ella",
    
    # Verbos auxiliares y copulativos
    "ha", "han", "hay", "ser", "estar", "tiene", "tienen",
    "puede", "pueden", "hacer", "hace", "son", "fue", "era",
    "sido", "siendo", "está", "están",
    
    # Preposiciones y conjunciones
    "sobre", "como", "pero", "también", "ya", "así", "e",
    "o", "u", "ni", "ante", "bajo", "desde", "hasta",
    "hacia", "entre", "sin", "tras",
    
    # Adverbios genéricos
    "cómo", "qué", "cuál", "cuáles", "dónde", "cuando",
    "muy", "bien", "aquí", "ahora", "aún", "solo",
    
    # Palabras que aparecen en casi todos los tópicos
    "inteligencia", "artificial", "ia", "inteligenciaartificial",
]

vectorizer = CountVectorizer(stop_words=stopwords_es)

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=200,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    language='multilingual',
    calculate_probabilities=True,
    verbose=True
)

docs = data_wo_na['text_bertopic'].tolist()

topics, probs = topic_model.fit_transform(docs)

#%%
data_wo_na["bertopic_topic"] = topics
data_wo_na["bertopic_prob"]  = probs.max(axis=1)

#%%
new_topics = topic_model.reduce_outliers(
    data_wo_na["text_bertopic"], 
    topics, 
    strategy="distributions",
    threshold=0.1
)
topic_model.update_topics(data_wo_na["text_bertopic"], topics=new_topics)

# Ver cómo ha cambiado la distribución
topic_info_new = topic_model.get_topic_info()
print(topic_info_new)

# -----------------------------------------------------------------------------
# 4. INSPECCIÓN DE TÓPICOS
# -----------------------------------------------------------------------------
#%%
topic_info = topic_model.get_topic_info()

with open('topic_info.txt', 'w', encoding='utf-8') as f:
    f.write(topic_info.to_string())
    f.write('\n\n')
    for topic_id in topic_info["Topic"].values:
        if topic_id != -1:
            f.write(f"\nTópico {topic_id}: {topic_model.get_topic(topic_id)}\n")

print("Guardado: topic_info.txt")

topic_info["palabras_clave"] = topic_info["Topic"].apply(
    lambda t: ", ".join([word for word, _ in topic_model.get_topic(t)[:8]])
)

topic_info["pct_documentos"] = (topic_info["Count"] / topic_info["Count"].sum() * 100).round(1)

tabla_topicos = topic_info[["Topic", "palabras_clave", "pct_documentos"]].copy()
tabla_topicos.columns = ["Tópico", "Palabras clave", "% documentos"]

print(tabla_topicos)
tabla_topicos.to_excel('tabla_topicos.xlsx')

# %%
# -----------------------------------------------------------------------------
# 5. TÍTULOS DE TÓPICOS
# -----------------------------------------------------------------------------
titulos = pd.read_excel('tabla_topicos.xlsx', sheet_name=0)
titulos_dict = dict(zip(titulos['Tópico'], titulos['Nombre asignado']))

data_wo_na['bertopic_name'] = data_wo_na['bertopic_topic'].replace(titulos_dict)

# %%
# -----------------------------------------------------------------------------
# 6. CLASIFICACIÓN EN TEMAS 
# -----------------------------------------------------------------------------
topicos = pd.read_excel('tabla_topicos.xlsx', sheet_name=1)

categorias = dict(zip(topicos['Número'], topicos['Nombre']))

data_wo_na['bertopic_category'] = data_wo_na['bertopic_topic'].replace(categorias)
print(data_wo_na['bertopic_category'])
#%%
#print(data_wo_na['bertopic_topic'].value_counts())
print(data_wo_na['bertopic_name'].value_counts())

# -----------------------------------------------------------------------------
# 7. GUARDADO
# -----------------------------------------------------------------------------
#%%
data_wo_na.to_csv('data_final.csv', index=False)
print("Guardado: data_final.csv")

topic_model.save("modelo_bertopic")
print("Guardado: modelo_bertopic")


# %%
data_wo_na.to_excel('data_final.xlsx')
# %%
