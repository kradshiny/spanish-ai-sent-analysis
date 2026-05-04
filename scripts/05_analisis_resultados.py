# =============================================================================
# 05_analisis_resultados.py
# Análisis de resultados: distribución del corpus, sentimiento y tópicos
# Input:  data_final.csv, data_clean_with_2022.csv
# Output: gráficos en carpeta /graficos/
# =============================================================================

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURACIÓN DE CARPETAS DE SALIDA
# =============================================================================
# %%
CARPETAS = {
    '4_1': 'graficos/4_1_descripcion_corpus',
    '4_2': 'graficos/4_2_analisis_sentimiento',
    '4_3': 'graficos/4_3_modelado_topicos',
    '4_4': 'graficos/4_4_eventos_disruptivos',
}

for carpeta in CARPETAS.values():
    os.makedirs(carpeta, exist_ok=True)

def guardar(fig, apartado, nombre):
    """Guarda la figura en la carpeta del apartado correspondiente."""
    ruta = os.path.join(CARPETAS[apartado], f"{nombre}.png")
    fig.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Guardado: {ruta}")

# =============================================================================
# CONFIGURACIÓN VISUAL
# =============================================================================
# %%
plt.rcParams["font.family"] = "Arial"

colores = {
    'POS':        '#2171b5',
    'NEU':        '#9ecae1',
    'NEG':        '#e34a33',
    'dark_blue':  '#08306b',
    'mid_blue':   '#2171b5',
    'blue':       '#6baed6',
    'light_blue': '#9ecae1',
    'pale_blue':  '#c6dbef',
    'red':        '#e34a33',
    'dark_red':   '#cb181d',
}

etiquetas = {'POS': 'Positivo', 'NEU': 'Neutral', 'NEG': 'Negativo'}

colores_pie = [
    "#08306b", "#2171b5", "#6baed6", "#9ecae1", "#c6dbef",
    "#fcbba1", "#fc7050", "#e34a33", "#cb181d",
]

eventos = {
    "ChatGPT\n(nov. 2022)":       "2022-11",
    "GPT-4/\nveto Italia\n(mar. 2023)": "2023-03",
    "Crisis\nOpenAI\n(nov. 2023)": "2023-11",
    "EU AI Act\n(mar. 2024)":      "2024-03",
    "GPT-4o\n(may. 2024)":         "2024-05",
    "DeepSeek R1\n(ene. 2025)":    "2025-01",
}

eventos_mes = {
    'Lanzamiento ChatGPT':      '2022-11',
    'GPT-4 + veto italiano':    '2023-03',
    'Crisis gobernanza OpenAI': '2023-11',
    'EU AI Act':                '2024-03',
    'GPT-4o':                   '2024-05',
    'DeepSeek R1':              '2025-01',
}

rename_paises = {
    'Venezuela, Bolivarian Republic of': 'Venezuela',
    'Bolivia, Plurinational State of':   'Bolivia',
    'United States': 'Estados Unidos',
    'Spain':         'España',
    'Mexico':        'México',
    'Argentina':     'Argentina',
    'Colombia':      'Colombia',
    'Chile':         'Chile',
    'Ecuador':       'Ecuador',
    'Peru':          'Perú',
    'Uruguay':       'Uruguay',
    'Panama':        'Panamá',
    'El Salvador':   'El Salvador',
    'Paraguay':      'Paraguay',
    'Costa Rica':    'Costa Rica',
}

# =============================================================================
# A. CARGA DE DATOS
# =============================================================================
# %%
data_wo_na = pd.read_csv('data_final.csv', low_memory=False)
data       = pd.read_csv('data_clean_with_2022.csv', low_memory=False)

# =============================================================================
# B. PREPARACIÓN TEMPORAL
# =============================================================================
# %%
data_wo_na = data_wo_na[data_wo_na['createdAt'].astype(str).str.len() > 4]
data_wo_na['createdAt'] = pd.to_datetime(data_wo_na['createdAt'], format='mixed', errors='coerce')
data_wo_na.loc[:, 'month_year'] = data_wo_na['createdAt'].dt.tz_localize(None).dt.to_period('M')
data_wo_na['month_str'] = data_wo_na['month_year'].astype(str)

# =============================================================================
# 4.1. DESCRIPCIÓN DEL CORPUS
# =============================================================================

# %%
# Tabla descriptiva
print("=" * 50)
print("ESTADÍSTICOS DEL CORPUS")
print("=" * 50)
print(f"Observaciones originales:  {data.shape[0]:,}")
print(f"Observaciones eliminadas:  {data.shape[0] - data_wo_na.shape[0]:,}")
print(f"Corpus final:              {data_wo_na.shape[0]:,}")
print(f"Periodo:                   {data_wo_na['month_str'].min()} – {data_wo_na['month_str'].max()}")
print(f"Meses:                     {data_wo_na['month_year'].nunique()}")
print(f"Media obs./mes:            {data_wo_na.groupby('month_str').size().mean():.0f}")
print(f"Media obs./año:            {data_wo_na.groupby(data_wo_na['createdAt'].dt.year).size().mean():.0f}")
print(f"Tamaño mínimo de muestra:  384")

# %%
# Gráfico 4.1-1: Distribución temporal del corpus
data_month = data_wo_na.groupby('month_year').size()

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(data_month.index.astype(str), data_month.values, color=colores['mid_blue'])
ax.axhline(y=384, color=colores['dark_red'], linestyle='--', linewidth=1.2,
           label='Tamaño mínimo de muestra (n = 384)')
ax.set_xticks(range(0, len(data_month), 3))
ax.set_xticklabels(data_month.index.astype(str)[::3], rotation=90, fontsize=9)
ax.set_xlabel("Mes", fontsize=11)
ax.set_ylabel("Número de tuits", fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9)
plt.tight_layout()
guardar(fig, '4_1', 'grafico_4-1-1_distribucion_temporal')

# %%
# Gráfico 4.1-2: Distribución geográfica
paises_raw = data_wo_na['pais_normalizado'].value_counts()
paises     = paises_raw[paises_raw.index != 'Unknown']
top12      = paises.head(12).copy()
top12.index = [rename_paises.get(i, i) for i in top12.index]
total_geo  = paises.sum()
pcts       = (top12 / total_geo * 100).round(1)

bar_colors = ([colores['dark_blue']] + [colores['mid_blue']] * 2 +
              [colores['blue']] * 3 + [colores['light_blue']] * 3 +
              [colores['pale_blue']] * 3)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(range(len(top12)), top12.values, color=bar_colors[:len(top12)],
        edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top12)))
ax.set_yticklabels(top12.index, fontsize=10.5)
ax.invert_yaxis()
ax.set_xlabel('Número de tuits', fontsize=10.5)
ax.set_xlim(0, top12.values[0] * 1.22)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=9.5)
for i, (val, pct) in enumerate(zip(top12.values, pcts.values)):
    ax.text(val + 120, i, f'{val:,}  ({pct}%)', va='center', fontsize=9.5, color='#333333')
fig.text(0.01, -0.02,
         'Nota: Se excluyen los 22.561 tuits (38,6%) sin localización identificable. '
         'Porcentajes calculados sobre el total geolocalizados.',
         fontsize=8.5, color='#555555', style='italic')
plt.tight_layout()
guardar(fig, '4_1', 'grafico_4-1-2_distribucion_geografica')

# %%
# Gráfico 4.1-3: Características descriptivas del corpus (longitud, hashtags, URLs)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# Panel A: Longitud del tuit
length_cap = data_wo_na['length'].dropna()
length_cap = length_cap[length_cap <= 1500]

ax = axes[0]
ax.hist(length_cap, bins=50, color=colores['mid_blue'], edgecolor='white', linewidth=0.4)
ax.axvline(data_wo_na['length'].median(), color=colores['dark_red'], linewidth=1.5,
           linestyle='--', label=f"Mediana: {int(data_wo_na['length'].median())} car.")
ax.axvline(data_wo_na['length'].mean(), color=colores['red'], linewidth=1.5,
           linestyle=':', label=f"Media: {int(data_wo_na['length'].mean())} car.")
ax.set_xlabel('Caracteres', fontsize=10)
ax.set_ylabel('Frecuencia', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=8.5)
ax.tick_params(labelsize=9)

# Panel B: Número de hashtags
len_h = data_wo_na['len_hashtags'].dropna().clip(upper=6)
counts_h = len_h.value_counts().sort_index()
labels_h = [str(int(i)) if i < 6 else '6+' for i in counts_h.index]

ax = axes[1]
ax.bar(range(len(counts_h)), counts_h.values, color=colores['mid_blue'],
       edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(counts_h)))
ax.set_xticklabels(labels_h, fontsize=9.5)
ax.set_xlabel('Hashtags', fontsize=10)
ax.set_ylabel('Frecuencia', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)
for i, v in enumerate(counts_h.values):
    ax.text(i, v + 200, f'{v/len(data_wo_na)*100:.1f}%', ha='center', fontsize=8, color='#333333')

# Panel C: Número de URLs
len_u = data_wo_na['len_url'].dropna().clip(upper=5)
counts_u = len_u.value_counts().sort_index()
labels_u = [str(int(i)) if i < 5 else '5+' for i in counts_u.index]

ax = axes[2]
ax.bar(range(len(counts_u)), counts_u.values, color=colores['blue'],
       edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(counts_u)))
ax.set_xticklabels(labels_u, fontsize=9.5)
ax.set_xlabel('URLs', fontsize=10)
ax.set_ylabel('Frecuencia', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)
for i, v in enumerate(counts_u.values):
    ax.text(i, v + 200, f'{v/len(data_wo_na)*100:.1f}%', ha='center', fontsize=8, color='#333333')

plt.tight_layout()
guardar(fig, '4_1', 'grafico_4-1-3_caracteristicas_descriptivas')

# %%
# Gráfico 4.1-4: Hashtags más frecuentes
hashtags_df = data_wo_na.explode('hashtags')
top_hashtags = hashtags_df['hashtags'].value_counts().head(15).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top_hashtags.index, top_hashtags.values, color=colores['mid_blue'])
ax.set_xlabel('Frecuencia', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
guardar(fig, '4_1', 'grafico_4-1-4_hashtags_frecuentes')

# =============================================================================
# 4.2. ANÁLISIS DE SENTIMIENTO
# =============================================================================

# %%
# Gráfico 4.2-1: Distribución global del sentimiento
dist_global = data_wo_na['sentiment_robertuito'].value_counts().reset_index()
dist_global.columns = ['sentimiento', 'n']
dist_global['pct'] = (dist_global['n'] / len(data_wo_na) * 100).round(1)
dist_global['sentimiento'] = dist_global['sentimiento'].map(etiquetas)
dist_global = dist_global.set_index('sentimiento').reindex(
    ['Negativo', 'Neutral', 'Positivo']).reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.pie(
    dist_global['n'],
    labels=dist_global['sentimiento'],
    colors=['#e34a33', '#9ecae1', '#2171b5'],
    autopct="%1.1f%%"
)
# título eliminado — se añade en Word", fontsize=13)
ax.set_position([0.38, 0.05, 0.24, 0.9])
guardar(fig, '4_2', 'grafico_4-2-1_distribucion_global_sentimiento')

# %%
# Gráfico 4.2-2: Evolución temporal del sentimiento (líneas)
sent_month     = data_wo_na.groupby(['month_year', 'sentiment_robertuito']).size().unstack()
sent_month_pct = sent_month.div(sent_month.sum(axis=1), axis=0) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

for sent in ['POS', 'NEG']:
    col = sent_month_pct[sent]
    ma3 = col.rolling(window=3, center=True).mean()
    ax1.plot(col.index.astype(str), ma3.values,
             label=etiquetas[sent], color=colores[sent], linewidth=1.8)

idx = sent_month_pct.index.astype(str).tolist()
for etiqueta, fecha in eventos.items():
    if fecha in idx:
        x_pos = idx.index(fecha)
        y_max = ax1.get_ylim()[1] - 4
        y_pos = min(max(sent_month_pct['POS'].iloc[x_pos],
                        sent_month_pct['NEG'].iloc[x_pos]) + 2, y_max)
        ax1.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax1.text(x_pos - 0.3, y_pos, etiqueta, fontsize=6.5,
                 ha='right', va='bottom', color='gray')
        ax2.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

ax1.set_xlabel('Mes', fontsize=11)
ax1.set_ylabel('Porcentaje de tuits (%)', fontsize=11)
ax1.legend(fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.plot(sent_month_pct.index.astype(str), sent_month_pct['NEU'].values,
         color=colores['NEU'], linewidth=1.8)
ax2.set_xlabel('Mes', fontsize=11)
ax2.set_ylabel('Porcentaje de tuits (%)', fontsize=11)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.setp(ax1.get_xticklabels(), rotation=90, fontsize=9)
plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=9)
plt.subplots_adjust(hspace=0.5)
guardar(fig, '4_2', 'grafico_4-2-2_evolucion_temporal_sentimiento')

# =============================================================================
# 4.3. MODELADO DE TÓPICOS
# =============================================================================

# %%
data_wo_na = data_wo_na.loc[data_wo_na['bertopic_name'] != 'Sin clasificar', :]

# Gráfico 4.3-1: Distribución de los 50 tópicos
topic_group = data_wo_na.groupby('bertopic_name').size().reset_index(name='n')
topic_group['pct'] = (topic_group['n'] / len(data_wo_na) * 100).round(1)
topic_group = topic_group.sort_values('pct', ascending=True)

fig, ax = plt.subplots(figsize=(14, 14))
ax.barh(topic_group['bertopic_name'], topic_group['pct'], color=colores['mid_blue'])
# título eliminado — se añade en Word
ax.set_xlabel('% del corpus', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
guardar(fig, '4_3', 'grafico_4-3-1_distribucion_50_topicos')

# %%
# Gráfico 4.3-2: Distribución por categorías temáticas
category_group = data_wo_na.groupby('bertopic_category').size().reset_index(name='n')
category_group = category_group.loc[category_group['bertopic_category'] != 'Sin clasificar', :]

fig, ax = plt.subplots(figsize=(14, 5))
ax.pie(category_group['n'], labels=category_group['bertopic_category'],
       colors=colores_pie, autopct="%1.1f%%")
# título eliminado — se añade en Word
guardar(fig, '4_3', 'grafico_4-3-2_distribucion_categorias')

# %%
# Gráfico 4.3-3: Mapa de calor del peso mensual por categoría
data_plot = data_wo_na[data_wo_na['bertopic_category'] != 'Sin clasificar'].copy()

heatmap_data = data_plot.groupby(['month_str', 'bertopic_category']).size()\
    .unstack(fill_value=0)
heatmap_pct = heatmap_data.div(data_wo_na.groupby('month_str').size(), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_pct.T, ax=ax, cmap='Blues', linewidths=0.3,
            fmt='.1f', annot=True, annot_kws={'size': 7})
ax.set_xlabel('Mes', fontsize=11)
ax.set_xticklabels(heatmap_pct.index, rotation=90, fontsize=7.5)
plt.tight_layout()
guardar(fig, '4_3', 'grafico_4-3-3_heatmap_categorias')

# %%
# Gráfico 4.3-4: Evolución de los 6 tópicos más frecuentes
top6 = [
    'IA en educación y formación',
    'ChatGPT y modelos OpenAI',
    'IA en salud y medicina',
    'IA y mercado laboral',
    'DeepSeek',
    'IA en empresas y negocios',
]

colores_top6 = [
    "#08306b", "#2171b5", "#6baed6",
    "#cb181d", "#e34a33", "#737373",
]

top_time = data_wo_na[data_wo_na['bertopic_name'].isin(top6)]\
    .groupby(['month_str', 'bertopic_name']).size().unstack(fill_value=0)
top_time_pct = top_time.div(data_wo_na.groupby('month_str').size(), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 5))
top_time_pct.plot(ax=ax, linewidth=2, color=colores_top6)

idx_list = top_time_pct.index.tolist()
ax.set_ylim(0, top_time_pct.values.max() * 1.4)
for etiqueta, fecha in eventos.items():
    if fecha in idx_list:
        x_pos = idx_list.index(fecha)
        ax.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(x_pos - 0.3, ax.get_ylim()[1] * 0.95, etiqueta, fontsize=6.5,
                ha='right', va='top', color='gray')

ax.set_xlabel('Mes', fontsize=11)
ax.set_ylabel('% del corpus', fontsize=11)
ax.set_xticks(range(0, len(top_time_pct), 3))
ax.set_xticklabels(top_time_pct.index[::3], rotation=90, fontsize=7.5)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
guardar(fig, '4_3', 'grafico_4-3-4_evolucion_topicos_principales')

# %%
# Gráfico 4.3-5: Distribución del sentimiento por categoría
data_sent   = data_wo_na[data_wo_na['bertopic_category'] != 'Sin clasificar']
cat_sent    = data_sent.groupby(['bertopic_category', 'sentiment_robertuito']).size().unstack(fill_value=0)
cat_sent_pct = cat_sent.div(cat_sent.sum(axis=1), axis=0) * 100
cat_sent_pct = cat_sent_pct.sort_values('POS', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
cat_sent_pct[['NEG', 'NEU', 'POS']].plot(
    kind='barh', stacked=True, ax=ax,
    color=["#e34a33", "#9ecae1", "#2171b5"],
    edgecolor='none'
)
# título eliminado — se añade en Word
ax.set_xlabel('% documentos', fontsize=11)
ax.set_ylabel('')
ax.legend(title='Sentimiento', labels=['Negativo', 'Neutral', 'Positivo'], fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
guardar(fig, '4_3', 'grafico_4-3-5_sentimiento_por_categoria')

# %%
# Gráfico 4.3-6: Distribución del sentimiento por tópico (Anexo)
topic_sent    = data_sent.groupby(['bertopic_name', 'sentiment_robertuito']).size().unstack(fill_value=0)
topic_sent_pct = topic_sent.div(topic_sent.sum(axis=1), axis=0) * 100
topic_sent_pct = topic_sent_pct.sort_values('POS', ascending=True)

fig, ax = plt.subplots(figsize=(10, 16))
topic_sent_pct[['NEG', 'NEU', 'POS']].plot(
    kind='barh', stacked=True, ax=ax,
    color=["#e34a33", "#9ecae1", "#2171b5"],
    edgecolor='none'
)
# título eliminado — se añade en Word
ax.set_xlabel('% documentos', fontsize=11)
ax.set_ylabel('')
ax.legend(title='Sentimiento', labels=['Negativo', 'Neutral', 'Positivo'], fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
guardar(fig, '4_3', 'grafico_4-3-6_sentimiento_por_topico_anexo')

# %%
# Gráfico 4.3-7: Dispersión % positivo vs % negativo por tópico
topic_sent_pct['categoria'] = data_sent.groupby('bertopic_name')['bertopic_category'].first()

fig, ax = plt.subplots(figsize=(10, 8))
for cat, color in zip(cat_sent_pct.index, colores_pie):
    mask = topic_sent_pct['categoria'] == cat
    ax.scatter(
        topic_sent_pct[mask]['POS'],
        topic_sent_pct[mask]['NEG'],
        label=cat, color=color, s=60, alpha=0.8
    )

extremos = topic_sent_pct[
    (topic_sent_pct['POS'] > topic_sent_pct['POS'].quantile(0.85)) |
    (topic_sent_pct['NEG'] > topic_sent_pct['NEG'].quantile(0.85))
]
for idx_name, row in extremos.iterrows():
    ax.annotate(idx_name, (row['POS'], row['NEG']),
                fontsize=8, xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('% sentimiento positivo', fontsize=11)
ax.set_ylabel('% sentimiento negativo', fontsize=11)
# título eliminado — se añade en Word
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
guardar(fig, '4_3', 'grafico_4-3-7_dispersion_sentimiento_topico')

# =============================================================================
# 4.4. IMPACTO DE EVENTOS DISRUPTIVOS EN EL SENTIMIENTO
# =============================================================================

# %%
sent_monthly     = data_wo_na.groupby(['month_str', 'sentiment_robertuito']).size().unstack(fill_value=0)
sent_monthly_pct = sent_monthly.div(sent_monthly.sum(axis=1), axis=0) * 100
pct_pos = sent_monthly_pct['POS']
pct_neg = sent_monthly_pct['NEG']

# %%
# Gráfico 4.4-1: Test de Pettitt
def pettitt_test(x):
    n = len(x)
    U = np.zeros(n)
    for t in range(n):
        for i in range(t):
            for j in range(t, n):
                U[t] += np.sign(x[j] - x[i])
    K = np.max(np.abs(U))
    t_change = np.argmax(np.abs(U))
    p_value = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    return t_change, K, p_value

t_change, K, p_value = pettitt_test(pct_pos.values)
change_month = pct_pos.index[t_change]
print(f"Punto de cambio detectado: {change_month}")
print(f"Estadístico K:             {K:.2f}")
print(f"p-valor:                   {p_value:.4f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(pct_pos)), pct_pos.values, color=colores['mid_blue'], linewidth=1.5)
ax.axvline(x=t_change, color=colores['dark_red'], linestyle='--', linewidth=1.5,
           label=f'Punto de cambio: {change_month}')
ax.axvline(x=list(pct_pos.index).index('2022-11'), color='black',
           linestyle='--', linewidth=1.2, label='Lanzamiento ChatGPT (nov. 2022)')
ax.set_xticks(range(0, len(pct_pos), 3))
ax.set_xticklabels(pct_pos.index[::3], rotation=90, fontsize=7.5)
ax.set_xlabel('Mes', fontsize=11)
ax.set_ylabel('% sentimiento positivo', fontsize=11)
# título eliminado — se añade en Word', fontsize=13, pad=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
guardar(fig, '4_4', 'grafico_4-4-1_test_pettitt')

# %%
# Gráfico 4.4-2: Comparación por periodos (Mann-Whitney)
from scipy import stats

sentiment_map = {'POS': 1, 'NEU': 0, 'NEG': -1}
data_wo_na['sent_num'] = data_wo_na['sentiment_robertuito'].map(sentiment_map)

pre           = data_wo_na[data_wo_na['month_str'].between('2022-07', '2022-11')]['sent_num']
adopcion      = data_wo_na[data_wo_na['month_str'].between('2022-12', '2023-12')]['sent_num']
consolidacion = data_wo_na[data_wo_na['month_str'].between('2024-01', '2025-12')]['sent_num']

stat1, p1 = stats.mannwhitneyu(pre, adopcion,       alternative='two-sided')
stat2, p2 = stats.mannwhitneyu(pre, consolidacion,  alternative='two-sided')
stat3, p3 = stats.mannwhitneyu(adopcion, consolidacion, alternative='two-sided')

print(f"Pre vs Adopción:           U = {stat1:.0f}, p = {p1:.6f}")
print(f"Pre vs Consolidación:      U = {stat2:.0f}, p = {p2:.6f}")
print(f"Adopción vs Consolidación: U = {stat3:.0f}, p = {p3:.6f}")

periodos   = ['Pre\n(jul–nov 2022)', 'Adopción\n(dic 2022–dic 2023)', 'Consolidación\n(ene 2024–dic 2025)']
datos      = [pre, adopcion, consolidacion]
dist_list  = []
for p_data, nombre in zip(datos, periodos):
    dist = p_data.map({1: 'Positivo', 0: 'Neutral', -1: 'Negativo'})\
                 .value_counts(normalize=True) * 100
    dist_list.append(dist)

dist_df = pd.DataFrame(dist_list, index=periodos)[['Negativo', 'Neutral', 'Positivo']]

fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(periodos))
width = 0.25
ax.bar(x - width, dist_df['Negativo'], width, label='Negativo', color='#e34a33', edgecolor='none')
ax.bar(x,         dist_df['Neutral'],  width, label='Neutral',  color='#9ecae1', edgecolor='none')
ax.bar(x + width, dist_df['Positivo'], width, label='Positivo', color='#2171b5', edgecolor='none')
ax.set_xticks(x)
ax.set_xticklabels(periodos, fontsize=10)
ax.set_ylabel('% documentos', fontsize=11)
# título eliminado — se añade en Word
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
guardar(fig, '4_4', 'grafico_4-4-2_comparacion_periodos_mann_whitney')

# %%
# Gráfico 4.4-3: Estudio de eventos (event study)
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
axes = axes.flatten()
media = pct_pos.mean()
std   = pct_pos.std()

for i, (nombre, mes) in enumerate(eventos_mes.items()):
    if mes not in list(pct_pos.index):
        continue
    idx_pos = list(pct_pos.index).index(mes)
    ventana = pct_pos.iloc[max(0, idx_pos - 3): idx_pos + 4]
    x       = range(len(ventana))

    axes[i].plot(x, ventana.values, color=colores['mid_blue'],
                 linewidth=2, marker='o', markersize=4)
    axes[i].axhline(y=media, color='#737373', linestyle='--', linewidth=1, label='Media global')
    axes[i].fill_between(x, media - 1.96 * std, media + 1.96 * std,
                         alpha=0.1, color=colores['mid_blue'], label='IC 95%')
    axes[i].axvline(x=min(3, len(ventana) - 1), color=colores['red'],
                    linestyle='--', linewidth=1.5)
    axes[i].set_title(nombre, fontsize=10)
    axes[i].set_xticks(range(len(ventana)))
    axes[i].set_xticklabels(ventana.index, rotation=45, fontsize=7)
    axes[i].set_ylabel('')
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

axes[0].legend(fontsize=8)
plt.tight_layout()
guardar(fig, '4_4', 'grafico_4-4-3_event_study')

print("\n✓ Todos los gráficos guardados correctamente.")