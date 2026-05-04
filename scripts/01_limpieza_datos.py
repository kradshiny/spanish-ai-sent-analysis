# =============================================================================
# 01_limpieza_datos.py
# Limpieza y preparación del corpus de tuits
# Input:  data_clean_with_2022.csv
# Output: data_wo_na.csv
# =============================================================================

# %%
import pandas as pd
import unidecode
import re
import pycountry
import plotly.express as px
from countrystatecity_countries import get_countries, get_states_of_country

# %%
# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------------------

data = pd.read_csv('data_clean_with_2022.csv', low_memory=False)

# -----------------------------------------------------------------------------
# 2. EXPLORACIÓN INICIAL
# -----------------------------------------------------------------------------
# %%
print(data.shape)
print(list(data.columns))
data.info()

# %%
pd.set_option('display.max_rows', None)
print(data.describe().T)

# %%
# Distribución temporal
data['createdAt'] = pd.to_datetime(data['createdAt'], format="mixed")
data['year']  = data['createdAt'].dt.year
data['month'] = data['createdAt'].dt.month
data['day']   = data['createdAt'].dt.day

# %%
print(data['year'].value_counts())
print(data['month'].value_counts())
print(data.groupby(['year', 'month']).size())

# -----------------------------------------------------------------------------
# 3. FILTRADO DE ERRORES DE EXTRACCIÓN
# -----------------------------------------------------------------------------
# %%
print(data['lang'].value_counts())

# %%
# Quedarse únicamente con los tweets en español y eliminar los valores en el año 2026
data_clean = data.loc[data['lang'] == 'es', :]       # 179 observaciones
data_clean = data_clean.loc[data_clean['year'] != 2026, :]  # 57 observaciones
print(data_clean.shape)

# %%
print(data_clean['year'].value_counts())
print(data_clean['lang'].value_counts())

# -----------------------------------------------------------------------------
# 4. TRATAMIENTO DE VALORES FALTANTES
# -----------------------------------------------------------------------------
# %%
# Filter columns that are >80% missing values
na_columns = data_clean.isna().sum() / data_clean.shape[0]
cols_na_names = na_columns[na_columns > 0.8].index
print(cols_na_names)

# %%
data_wo_na = data_clean.loc[:, ~data_clean.columns.isin(cols_na_names)]

# %%
pd.set_option('display.max_columns', None)
print(data_wo_na.head())

# -----------------------------------------------------------------------------
# 5. NORMALIZACIÓN DE LOCALIZACIÓN GEOGRÁFICA
# -----------------------------------------------------------------------------
# %%
def clean_location(text):
    if pd.isna(text):
        return pd.NA
    else:
        text = text.lower()
        text = " ".join(text.split())
        text = unidecode.unidecode(text)
        text = re.sub(r'[.-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

    if text == '':
        return pd.NA
    else:
        return text


def split_country_city(text):
    if pd.isna(text):
        return pd.NA
    parts = [p.strip() for p in text.split(',')]
    country = parts[-1]
    return country

# %%
data_wo_na['location_clean']   = data_wo_na['author.location'].apply(clean_location)
data_wo_na['location_country'] = data_wo_na['location_clean'].apply(split_country_city)
print(data_wo_na['location_country'].value_counts())

# %%
# Construir diccionarios de países, capitales, estados de México y España
country      = get_countries()
country_list = [(city.native, city.capital, city.region) for city in country]
countries    = pd.DataFrame(country_list, columns=['country', 'capital', 'region'])
countries_df = countries.loc[countries['region'].isin(['Americas', 'Europe'])]
countries_df = countries_df.loc[countries_df['country'] != 'United States Minor Outlying Islands']

mexico        = get_states_of_country('MX')
mexico_states = [state.native for state in mexico]
mexico_df     = pd.DataFrame(mexico_states, columns=['states'])
mexico_df['mexico'] = 'México'

spain        = get_states_of_country('ES')
spain_states = [state.native for state in spain]
spain_df     = pd.DataFrame(spain_states, columns=['states'])
spain_df['spain'] = 'spain'

# %%
# Aplicar clean_location a los diccionarios
for col in list(countries_df.columns):
    countries_df[col] = countries_df[col].apply(clean_location)
for col in list(spain_df.columns):
    spain_df[col] = spain_df[col].apply(clean_location)
for col in list(mexico_df.columns):
    mexico_df[col] = mexico_df[col].apply(clean_location)

countries_dict = dict(zip(countries_df['country'], countries_df['country']))
capitals_dict  = dict(zip(countries_df['capital'],  countries_df['country']))
mexico_dict    = dict(zip(mexico_df['states'],       mexico_df['mexico']))
spain_dict     = dict(zip(spain_df['states'],        spain_df['spain']))

values_countries = countries_df['country'].to_list()
values_capitals  = countries_df['capital'].to_list()
values_mexico    = mexico_df['states'].to_list()
values_spain     = spain_df['states'].to_list()

list_tuples = [
    (countries_dict, values_countries),
    (capitals_dict,  values_capitals),
    (mexico_dict,    values_mexico),
    (spain_dict,     values_spain)
]

col_names = ['country_general', 'country_capitals', 'country_mexico', 'country_spain']
for i, (countries_d, values) in enumerate(list_tuples):
    patron = r"\b(" + "|".join([v for v in values if pd.notna(v)]) + r")\b"
    data_wo_na[col_names[i]] = data_wo_na['location_country'].str.extract(patron, expand=False).map(countries_d)

data_wo_na['country_general'].fillna(data_wo_na['country_capitals'], inplace=True)
data_wo_na['country_general'].fillna(data_wo_na['country_mexico'],   inplace=True)
data_wo_na['country_general'].fillna(data_wo_na['country_spain'],    inplace=True)
data_wo_na['country_general'].fillna(data_wo_na['location_country'], inplace=True)

data_wo_na['location_short'] = data_wo_na['country_general'].apply(lambda x: len(x) < 4 if pd.notna(x) else False)
print(data_wo_na.loc[data_wo_na['location_short'], 'country_general'].value_counts())

# %%
# Correcciones manuales de abreviaciones
country_dict = {
    'cdmx': 'mexico', 'usa': 'united states', 'us': 'united states',
    'mx': 'mexico', 'mex': 'mexico', 'df': 'mexico', 'ny': 'united states',
    'nyc': 'united states', 'la': 'united states', 'fl': 'united states',
    'dc': 'colombia', 'gdl': 'mexico', 'bcn': 'spain', 'sp': 'spain',
    'vzla': 'venezuela', 'ca': 'venezuela', 'ec': 'ecuador',
    'rd': 'republica dominicana', 'uy': 'uruguay', 'arg': 'argentina',
    'rep dom': 'republica dominicana', 'tx': 'united states',
    'wa': 'united states', 'il': 'united states', 'edomex': 'mexico',
    'eeuu': 'united states', 'ma': 'spain', 'ver': 'mexico',
    'oax': 'mexico', 'co': 'colombia', 'cl': 'united states',
}
data_wo_na['location_clean'] = data_wo_na['country_general'].replace(country_dict)

pd.set_option('display.max_rows', None)
print(data_wo_na['location_clean'].value_counts())

# %%
# Correcciones manuales de ciudades y regiones
dict_errors = {
    'ciudad autonoma de buenos aire': 'argentina',
    'dominican republic': 'republica dominicana',
    'espana': 'spain', 'medellin': 'colombia', 'distrito federal': 'mexico',
    'terrassa': 'spain', 'bilbao': 'spain', 'miami': 'united states',
    'cartagena de indias': 'colombia', 'america latina': 'latinoamerica',
    'america': 'latinoamerica', 'hispanoamerica': 'latinoamerica',
    'latam': 'latinoamerica', 'ciudad imagen': 'spain',
    'iberoamerica': 'latinoamerica', 'mar del plata': 'argentina',
    'veracruz': 'mexico', 'la pampa': 'argentina',
    'brclecespe latam': 'latinoamerica', 'santa cruz de la sierra': 'bolivia',
    'catalunya': 'spain', 'new york': 'united stated',
    'centroamerica': 'latinoamerica', 'tf +34 67873941': 'spain',
    'monterrey': 'mexico', 'california': 'united states', 'caba': 'argentina',
    'cartagena': 'spain', 'santander': 'spain', 'guayaquil': 'ecuador',
    'coahuila': 'mexico', 'espanya': 'spain', 'argentinabuenos aires': 'argentina',
    'spain & latam': 'spain', 'basque country': 'spain',
    'central america': 'latinoamerica', 'rep dominicana': 'republica dominicana',
    'caracasvenezuela': 'venezuela', 'mexicocentroamericavenezuela': 'latinoamerica',
    'florida': 'united states', 'extremadura': 'spain', 'palma de mallorca': 'spain',
    'mendoza': 'argentina', 'sao paulo': 'brazil', 'benito juarez': 'mexico',
    'general arenales': 'argentina', 'cali': 'colombia', 'vigo': 'spain',
    'latin america': 'latinoamerica', 'vitoriagasteiz': 'spain',
    'valparaiso': 'chile', 'xalapaveracruz': 'mexico', 'elche': 'spain',
    'cuernavaca': 'mexico', 'santa fe': 'spain', 'logrono (spain)': 'spain',
    'samborondon': 'ecuador', 'new york city': 'united states',
    'quitoecuador': 'ecuador', 'antofagasta': 'chile', 'euskadi': 'spain',
    'limaperu': 'peru', 'somewhere in spain': 'spain', 'la habana': 'cuba',
    'playa del carmen': 'mexico', 'barranquilla': 'colombia',
    'san sebastian de los reyes': 'spain', 'los angeles': 'united states',
    'cantabria': 'spain', 'comunitat valenciana': 'spain',
    'cc alianza mall guacara': 'venezuela', 'bucaramanga': 'colombia',
    'mallorca': 'spain', 'gava': 'spain', 'maiquetia': 'venezuela',
    'tenerife': 'spain', 'hermosillo': 'mexico', 'capital federal': 'argentina',
    'pamplona': 'spain', 'ciudad ojeda': 'venezuela', 'cancun': 'mexico',
    'tucuman': 'argentina', 'houston texas': 'united states', 'logrono': 'spain',
    'ontario': 'canada', 'texas': 'united states', 'san sebastian spain': 'spain',
    'colmenar viejo': 'spain', 'mostoles': 'spain', 'argentine': 'argentina',
    'america latina y el caribe': 'latinoamerica', 'antioquia': 'colombia',
    'integracion latinoamericana': 'latinoamerica', 'saltillo': 'mexico',
    'bahia blanca': 'argentina', 'medellin antioquia': 'colombia',
    'distrito feder': 'mexico', 'toluca': 'mexico', 'los condes': 'chile',
    'chubut': 'argentina', 'canary islands': 'spain', 'marbella': 'spain',
    'madrid espana': 'spain', 'fuengirola': 'spain', 'vaticano': 'italy',
    'san sebastian': 'spain', 'alcala de henares': 'spain', 'ponferrada': 'spain',
    'americas': 'latinoamerica', 'pinotepa nacional': 'mexico', 'zulia': 'colombia',
    'palma': 'spain', 'euskal herria': 'spain', 'donostia': 'spain',
    'mazatlan': 'mexico', 'centroamerica y dominicana': 'latinoamerica',
    'vitoria': 'spain', 'huila': 'colombia', 'miraflores': 'colombia',
    'la guaira': 'venezuela', 'tijuana': 'mexico', 'valle del cauca': 'colombia',
    'vina del mar': 'chile', 'entre rios': 'argentina',
    'continente americano': 'latinoamerica', 'dominican r': 'republica dominicana',
    'cadizfornia': 'spain', 'cucuta': 'colombia', 'dc eeuu': 'united states',
    'calvia/astorga': 'spain', 'tunja boyaca': 'colombia',
    'jerez de la frontera': 'spain', 'pamplonairuna': 'spain', 'ourense': 'spain',
    'espagne': 'spain', 'gijon': 'spain', 'sabadell': 'spain',
    'latino america': 'latinoamerica', 'catalonia': 'spain',
}
data_wo_na['location_final'] = data_wo_na['location_clean'].replace(dict_errors)
print(data_wo_na['location_final'].value_counts())

# %%
# Normalización con pycountry
def identificar_pais(texto):
    if not isinstance(texto, str):
        return "Unknown"
    texto = texto.strip()
    try:
        pais = pycountry.countries.lookup(texto)
        return pais.name
    except LookupError:
        return "Unknown"

data_wo_na["pais_normalizado"] = data_wo_na["location_final"].apply(identificar_pais)
print(data_wo_na['pais_normalizado'].value_counts())

# %%
# Mapa choropleth
conteo = data_wo_na[data_wo_na["pais_normalizado"] != "Unknown"]["pais_normalizado"].value_counts().reset_index()
conteo.columns = ["pais", "cantidad"]

fig = px.choropleth(
    conteo,
    locations="pais",
    locationmode="country names",
    color="cantidad",
    color_continuous_scale="Blues",
    title="Distribución de tweets por país"
)
fig.show()

# %%
# -----------------------------------------------------------------------------
# 6. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

data_wo_na['hashtags']     = data_wo_na['text'].str.findall(r'#\w+')
data_wo_na['len_hashtags'] = data_wo_na['hashtags'].apply(lambda x: len(x))

data_wo_na_hashtags = data_wo_na.explode('hashtags')
print(data_wo_na_hashtags['hashtags'].value_counts().head(10))

data_wo_na['url']     = data_wo_na['text'].str.findall(r'http[s]?://\S+')
data_wo_na['len_url'] = data_wo_na['url'].apply(lambda x: len(x))


data_wo_na['length'] = data_wo_na['text'].apply(len)
data_wo_na[['length']].boxplot()

# %%
# -----------------------------------------------------------------------------
# 7. GUARDADO
# -----------------------------------------------------------------------------

data_wo_na.to_csv('data_wo_na.csv', index=False)
print("Guardado: data_wo_na.csv")
