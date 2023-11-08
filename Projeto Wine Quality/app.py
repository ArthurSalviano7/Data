
# Sistema de Classificação de Vinhos - Modelo Preditivo

import streamlit as st

import pandas as pd
import numpy as np

import pickle 
from PIL import Image

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


def converte_clasficacao_XGBoost_para_classe(classificacao_vinho_XGBoost):
    
    if classificacao_vinho_XGBoost == 0:
        classificacao_vinho = 3
    elif classificacao_vinho_XGBoost == 1:
        classificacao_vinho = 4
    elif classificacao_vinho_XGBoost == 2:
        classificacao_vinho = 5
    elif classificacao_vinho_XGBoost == 3:
        classificacao_vinho = 6
    elif classificacao_vinho_XGBoost == 4:
        classificacao_vinho = 7
    elif classificacao_vinho_XGBoost == 5:
        classificacao_vinho = 8

    return classificacao_vinho 

st.title('Sistema de Classificação de Vinhos')


data_file = 'vinhos_classificacao.csv'
df = pd.read_csv(data_file)

#st.write(df)

atributos = list(df.columns.values)
atributos.remove('quality')

st.sidebar.title("Informe os dados do vinho")

# guardar os valores
atributos_valores = {}
for atributo in atributos:
    minimo, media, maximo = df[atributo].min(), df[atributo].mean(), df[atributo].max()
    atributos_valores[atributo] = {"min": minimo, "media": media, "max": maximo }

with st.sidebar:
    with st.form(key='my_form'):

        atributo = 'fixed acidity'
        fixed_acidity = st.number_input(atributo, min_value=atributos_valores[atributo]['min'], max_value=atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f") # step=atributos_valores[atributo]['media']
        
        atributo = 'volatile acidity'
        volatile_acidity = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")
        
        atributo = 'citric acid'
        citric_acid = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")
        
        atributo = 'residual sugar'
        residual_sugar = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")
        
        atributo = 'chlorides'
        chlorides = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")
        
        atributo = 'free sulfur dioxide'
        free_sulfur_dioxide  = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")
        
        atributo = 'total sulfur dioxide'
        total_sulfur_dioxide = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")
        
        atributo = 'density'
        density = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.16f")
        
        atributo = 'pH'
        pH = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")
        
        atributo = 'sulphates'
        sulphates = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")
        
        atributo = 'alcohol'
        alcohol = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],format="%.6f")

        predict_button = st.form_submit_button(label='Prever')


# Pagina pricipal
arquivo_modelo = 'ModeloXGBoost.pkl'
with open(arquivo_modelo, 'rb') as f:
    modelo = pickle.load(f)

def previsao_classificao_vinho(modelo, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                    free_sulfur_dioxide, chlorides, total_sulfur_dioxide, density, pH, sulphates, alcohol):

    new_X = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                    free_sulfur_dioxide, chlorides, total_sulfur_dioxide, density, pH, sulphates, alcohol])
    classificacao_vinho_XGBoost = modelo.predict(new_X.reshape(1, -1) )[0]

    classificacao_vinho = converte_clasficacao_XGBoost_para_classe(classificacao_vinho_XGBoost)

    return classificacao_vinho

imagem = 'vinho.jpeg'
image = Image.open(imagem)
st.image(image, width=500)

if predict_button:
    classificaco_vinho  = previsao_classificao_vinho(modelo, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                    free_sulfur_dioxide, chlorides, total_sulfur_dioxide, density, pH, sulphates, alcohol)

    st.markdown('## Classificação do vinho (3-8): ' + \
                str(classificaco_vinho ) )

