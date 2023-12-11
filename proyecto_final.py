#!/usr/bin/env python
# coding: utf-8

# <img src="proyecto_final_portada.png" alt="Portada Proyecto Final" style="height: 600px; width:600px;"/>
# 
# 

# El sector de las telecomunicaciones (telecom) en India está cambiando rápidamente, con la creación de más y más empresas de telecomunicaciones y muchos clientes decidiendo cambiar entre proveedores. "Churn" se refiere al proceso en el que los clientes o suscriptores dejan de utilizar los servicios o productos de una empresa. Comprender los factores que influyen en la retención de clientes como también lograr predecir la posibilidad de "churn" es crucial para que las empresas de telecomunicaciones mejoren la calidad de sus servicios y la satisfacción del cliente. Como científicos de datos en este proyecto, nuestro objetivo es explorar la compleja dinámica del comportamiento y la demografía de los clientes en el sector de las telecomunicaciones en India para predecir la "salida" o "churn" de los clientes, utilizando dos conjuntos de datos completos de cuatro importantes socios de telecomunicaciones: Airtel, Reliance Jio, Vodafone y BSNL:
# 
# - `telecom_demographics.csv` contiene información relacionada con la demografía de los clientes indios:
# 
# | Variable             | Descripción                                      |
# |----------------------|--------------------------------------------------|
# | `customer_id `         | Identificador único de cada cliente.             |
# | `telecom_partner `     | El socio de telecomunicaciones asociado con el cliente.|
# | `gender `              | El género del cliente.                      |
# | `age `                 | La edad del cliente.                         |
# | `state`                | El estado indio en el que se encuentra el cliente.|
# | `city`                 | La ciudad en la que se encuentra el cliente.       |
# | `pincode`              | El código PIN de la ubicación del cliente.          |
# | `registration_event` | Cuando el cliente se registró con el socio de telecomunicaciones.|
# | `num_dependents`      | El número de dependientes (por ejemplo, niños) que tiene el cliente.|
# | `estimated_salary`     | El salario estimado del cliente.                 |
# 
# - `telecom_usage` contiene información sobre los patrones de uso de los clientes indios:
# 
# | Variable   | Descripción                                                  |
# |------------|--------------------------------------------------------------|
# | `customer_id` | Identificador único de cada cliente.                         |
# | `calls_made` | El número de llamadas realizadas por el cliente.                    |
# | `sms_sent`   | El número de mensajes de SMS enviados por el cliente.             |
# | `data_used`  | La cantidad de datos utilizada por el cliente.                     |
# | `churn`    | Variable binaria que indica si el cliente ha cancelado o no (1 = cancelado, 0 = no cancelado).|
# 
# ---

# #### Estructura del notebook:
# ##### 1. Importación de librerías y dependencias 
# 

# In[1]:


#Importamos las librerías princiaples
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from machine_learning_utils import make_prediction, verify_results #Funciones que creamos para predecir y visualizar resultados en "machine_learning_utils.py"
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import RandomOverSampler



# ##### 2. Análisis exploratorio de los datos y variables relevantes
# 

# Leemos los datasets y procesamos los datos desde archivos locales .csv

# In[2]:


#Lectura de los archivos .csv con pandas
df1 = pd.read_csv('telecom_demographics.csv')
df2 = pd.read_csv('telecom_usage.csv')

print(df1.shape, df2.shape)


# In[3]:


#Unimos los dos dataframes con la variable customer id
df_full = pd.merge(df1,df2,on='customer_id')
df_full.head(5)


# In[4]:


df_full.shape


# El dataset tiene 6500 filas y 15 columnas luego del join

# In[5]:


df_full.isna().sum()


# In[6]:


df_full.duplicated().sum()


# No hay valores vacíos ni tampoco repetidos o duplicados.

# In[7]:


df_full= df_full.drop(axis=1, columns=['customer_id', 'pincode']) #Eliminamos customer_id que ya no nos sirve


# Podemos comenzar entonces, con el Análisis estadístico de las variables númericas del dataset

# In[8]:


df_full.describe(exclude='O') 


# Algunos datos relevantes del análisis estadísitico de la distribución de los datos:
# - La edad promedio de los clientes de 46 años. La edad mínima es 18 y la edad máxima es de 74
# 
# - Estimated salary como genralmente sucede con los salarios tiene una desviación estandar enorme, implicando un desbalance entre la distribución de los salarios. 
# 
# - Aparecen números negativos en el mínimo de llamadas hechas, sms enviados y en la cantidad de datos utilizados. Deberemos decidir como se interpretarán y usarán estos datos.
# 
# 

# Análisis estadístico de las variables categóricas del dataset

# In[9]:


df_full.describe(include='O').T


# El análisis de variables categóricas nos permite revisar la frecuencia y la cantidad de valores nulos:
# 
#  - No hay valores nulos, tenemos 4 empresas, 2 géneros y con mayor frecuencia masculinos.
#  
#  - La ciudad con más frecuencia es Delhi, relacionada con el estado Karnataka. 
# 

# In[10]:


sns.set_palette('deep') #Seteamos la paleta de colores


# In[11]:


def vars_categoricas_graf(dataframe, x, y): #Función que permite graficar variables categóricas y ver la cantidad de renuncias por categoría
        if x == y:
                churn_counts = dataframe[x].value_counts().reset_index()
                churn_counts = churn_counts.replace({0:'No', 1: 'Sí'})
                sns.barplot(data=churn_counts, x='churn', y='count', hue='churn', legend=False)
                plt.ylabel('Cantidad de bajas de clientes')
                plt.xlabel('¿El cliente se dió la baja?')
                plt.show()
                print(churn_counts)

                
        else:   
                grouped_df = df_full.groupby(x)[y].sum().reset_index() #agrupamos y sumamos para obtener la cantidad total de bajas por variable categórica

                plt.subplots(figsize=(8,6))
                sns.barplot(data=grouped_df, x=x, y=y, hue=x, legend=False) #graficamos
                plt.xticks(rotation=90)
                plt.xlabel('Tipos de '+ x + ' diferentes')
                plt.ylabel('Cantidad de bajas de clientes') 
                plt.show()
                print('Top 5 variables con mas churn', '\n', '\n', grouped_df.sort_values(by=y, ascending=False).head(5))


# In[12]:


vars_categoricas_graf(df_full, 'churn', 'churn')


# In[13]:


df_full.columns


# Podemos ver que nuestro tipo de predicción es hacia una clase desbalanceada, que representa aproximadamente el 20% del total de posibilidades.

# In[14]:


vista_group = df_full.groupby(['state', 'num_dependents'])['churn'].sum().reset_index().sort_values(by='churn', ascending=False)

plt.subplots(figsize=(12,6))

sns.barplot(data= vista_group, x='state', y='churn', hue='num_dependents',palette='deep')

plt.xticks(rotation=90)

plt.show()


# In[15]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calcular el porcentaje de churn 0 y churn 1 por estado
percent_churn_by_state = df_full.groupby('state')['churn'].value_counts(normalize=True).unstack().reset_index()

# Configurar el estilo de Seaborn (opcional)
sns.set(style="whitegrid")

# Crear el gráfico de barras apiladas
plt.figure(figsize=(10, 6))
sns.barplot(x='state', y=0, data=percent_churn_by_state, label='No')
sns.barplot(x='state', y=1, data=percent_churn_by_state, label='Sí')

# Configuraciones adicionales
plt.title('Porcentaje de Churn por Estado')
plt.xlabel('Estado')
plt.ylabel('Porcentaje de Churn')
plt.xticks(rotation=90)
plt.legend(title='¿Churn?', loc='upper left', bbox_to_anchor=(1, 1))

# Mostrar el gráfico
plt.show()


# In[16]:


vars_categoricas_graf(df_full, 'state', 'churn')


# Parecer ser que el estado de Mizoram es el que mayor cantidad de bajas presentas, podría ser un dato clave para preguntar qué sucedió en las ciudades del estado.

# In[17]:


vars_categoricas_graf(df_full, 'age', 'churn')


# Es interesante ver que hay un alto índice de renuncias entre los 35,36,37 años y los 56,60 y 63 años. 

# In[18]:


vars_categoricas_graf(df_full, 'telecom_partner', 'churn')


# In[19]:


df_full['telecom_partner'].value_counts()


# Si bien todas las empresas tienen una cantidad de renuncias similares, podemos ver que Vodafone lidera en las que más bajas tiene t podríamos decir que no es de las que más clientes tienen, esto es una llamada de atención para Vodafone.

# In[20]:


vars_categoricas_graf(df_full, 'gender', 'churn')


# In[21]:


vars_categoricas_graf(df_full, 'num_dependents', 'churn')


# Las personas con 1 hijo tienen mayor probabilidad de renunciar, sin embargo, no es una gran probabilidad, vemos una distribución similar entre los otras opciones.

# In[22]:


vars_categoricas_graf(df_full, 'calls_made', 'churn')


# Podemos ver que no hay una estricta relación entre la cantidad de llamadas y las bajas. 
# 
# Podemos notar que hay mayor bajas en niveles bajos y altos de llamadas realizadas. 
# 
# Esto puede que genere en las personas que llaman poco querer cambiar de plan (porque pueden estar pagando un precio más elevado del uso que le dan al servicio o bien, personas que realizan muchas llamadas y reciben facturas con precios muy elevados). 

# In[23]:


vars_categoricas_graf(df_full, 'sms_sent', 'churn')


# En el caso de sms enviados, sucede algo muy similar con llamadas realizadas.

# ##### Veamos que sucede con la distribución de las variables númericas

# In[24]:


def distribution_plot(df, column):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()


# In[25]:


columnas_numericas = df_full.select_dtypes(include=['int', 'float']).columns

for columna in columnas_numericas:
    if columna not in ['churn', 'customer_id', 'num_dependents']:
        distribution_plot(df_full, columna)


# In[26]:


corr = df_full.corr(numeric_only=True)
plt.figure(figsize=(8, 6))  # Tamaño de la figura
sns.set(font_scale=1)  # Escala del tamaño de la fuente
sns.heatmap(corr, cmap='Blues', annot=True, fmt='.2f', linewidths=.2)

# Configuraciones adicionales (opcional)
plt.title('Gráfico de correlación de variables númericas', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotación de etiquetas en el eje x
plt.yticks(rotation=0)  # Rotación de etiquetas en el eje y
plt.tight_layout()

# Mostrar el mapa de calor
plt.show()


# No vemos ninguna relación directa o lineal con las variables númericas. Probemos con la variable de tiempo. Primero vamos a procesarla

# In[27]:


df_full['registration_event'] = pd.to_datetime(df_full['registration_event']) #Convertimos a variable de tipo tiempo

df_full['year'] = df_full['registration_event'].dt.year #dividimos en año, mes y día
df_full['month'] = df_full['registration_event'].dt.month
df_full['day'] = df_full['registration_event'].dt.day
df_full.head(5)


# In[28]:


df_churn_by_year = df_full.groupby('year')['churn'].sum().reset_index()

# Gráfico de línea para visualizar la frecuencia de churn por año
plt.figure(figsize=(8, 6))
sns.lineplot(x='year', y='churn', data=df_churn_by_year, marker='o')
plt.title('Frecuencia de Churn por Año de Registro')
plt.xlabel('Año de Registro')
plt.ylabel('Número de Churns')
plt.show()


# In[29]:


df_churn_by_year = df_full.groupby('year')['churn'].sum().reset_index()

years =df_churn_by_year['year'][0:3]

churn_rates = df_churn_by_year['churn'][0:3]

plt.pie(churn_rates, labels=years, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Tasa de Churn por Año')
plt.show()


# Es muy interesante ver que el año en el que llega la pandemia (2020) se produce la mayor cantidad de bajas y esto puede ser debido a que mucha gente tuvo que dejar de trabajar debido al confinamiento y decidió darse de baja del servicio para reducir costos.

# In[30]:


df_pre_processed = df_full.copy()


# In[31]:


le = LabelEncoder() # Instanciamos LaberEncoder

enc = OneHotEncoder(handle_unknown='ignore') # Instanciamos One Hot Encoder

df_pre_processed['telecom_partner'] = le.fit_transform(df_pre_processed['telecom_partner'])

df_pre_processed = pd.get_dummies(df_pre_processed, columns=['gender'], drop_first=True) # Eliminamos una columna para evitar la dummy_trap o tramp de las variables dummys

df_pre_processed['state'] = le.fit_transform(df_pre_processed['state'])

df_pre_processed['city'] = le.fit_transform(df_pre_processed['city'])

df_pre_processed = df_pre_processed.drop(axis=1, columns='registration_event')


# In[32]:


df_pre_processed['gender_M'] = df_pre_processed.gender_M.replace({False: 0, True: 1})


# In[33]:


# columns = ['calls_made', 'sms_sent','data_used']

# for column in columns: # Bucle for para recorrer las columnas con negativos
#     df_pre_processed[column] = df_pre_processed[column].clip(lower=0)


# In[34]:


numerical_columns = ['age',
       'estimated_salary', 'calls_made', 'sms_sent', 'data_used']

scaler = MinMaxScaler()

for column in numerical_columns:
    df_pre_processed[column] = scaler.fit_transform(df_pre_processed[[column]])


# In[35]:


df_pre_processed.head(5)


# In[36]:


corr = df_pre_processed.corr(method='spearman')
plt.figure(figsize=(8, 6))  # Tamaño de la figura
sns.set(font_scale=1)  # Escala del tamaño de la fuente
sns.heatmap(corr, cmap='Blues', annot=True, fmt='.2f', linewidths=.2)

# Configuraciones adicionales (opcional)
plt.title('Gráfico de correlación de variables númericas', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotación de etiquetas en el eje x
plt.yticks(rotation=0)  # Rotación de etiquetas en el eje y
plt.tight_layout()

# Mostrar el mapa de calor
plt.show()


# In[37]:


X= df_pre_processed.drop(axis=1, columns='churn')
y= df_pre_processed['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=42)


# In[38]:


from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

undersampler = RandomUnderSampler(sampling_strategy=0.28, random_state=42)
X_train, y_train = undersampler.fit_resample(X, y)

print ("Distribution before resampling {}".format(Counter(y)))

print ("Distribution labels after resampling {}".format(Counter(y_train)))


# In[39]:


model = RandomForestClassifier()
y_pred = make_prediction(model, X_train, y_train, X_test)
verify_results(y_test, y_pred)


# In[40]:


xgbc = XGBClassifier()

y_pred = make_prediction(xgbc, X_train, y_train, X_test)
verify_results(y_test, y_pred)


# In[48]:


gb_classifier = GradientBoostingClassifier(loss='exponential', learning_rate=0.1)

y_pred = make_prediction(gb_classifier, X_train, y_train, X_test)
verify_results(y_test, y_pred)


# In[42]:


adaboost_classifier = AdaBoostClassifier()

y_pred = make_prediction(adaboost_classifier, X_train, y_train, X_test)
verify_results(y_test, y_pred)


# In[43]:


import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt



# Plot feature importances
plot_importance(xgbc)
plt.show()


# In[44]:


# Get feature importances
feature_importances = model.feature_importances_
feature_names = X_train.columns  # Assuming you have named columns in your DataFrame

# Sort feature importances in descending order
indices = feature_importances.argsort()[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[indices])
plt.xticks(range(len(feature_importances)), feature_names[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.show()


# In[45]:


# Get feature importances
feature_importances = gb_classifier.feature_importances_
feature_names = X_train.columns  # Assuming you have named columns in your DataFrame

# Sort feature importances in descending order
indices = feature_importances.argsort()[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[indices])
plt.xticks(range(len(feature_importances)), feature_names[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Gradient Boosting Feature Importances")
plt.show()

