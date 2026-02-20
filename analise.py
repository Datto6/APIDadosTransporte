import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

st.markdown(
    """
    Agora, cabe a você selecionar o que você quer ver
"""
)

options = st.multiselect(
    "Quantos modos de transporte selecionar",
    ["Onibus", "Barca", "Metrô", "Van"],
)

tipo_grafico = st.radio(
    "Escolher o tipo de gráfico",
    ["Histograma", "Linha", "Área","Tabela de Torta"],
)

# st.dataframe(st.session_state.dataframes[0])
st.session_state.dataframes[0]['hora'] = pd.to_datetime(
    st.session_state.dataframes[0]['Data da Transação'],
    dayfirst=True,
    errors='coerce'
).dt.strftime('%Hh')  #extraindo horario

tabela = st.session_state.dataframes[0].groupby(['Nº Carro', 'hora']).size().unstack(fill_value=0) #agrupando por numero de carro, e hora 
#size retorna o numero de linhas, que no nosso caso eh quantas vezes o carro aparece
#unstack para des-esculhambar o novo dataframe que sai do groupby

horas = [f'{i:02d}h' for i in range(24)] #cria string com colunas de hora
tabela = tabela.reindex(columns=horas, fill_value=0)#reseta os indices de acordo com o string formatado de  horarios
# traz o Nº Carro de volta como coluna visível
tabela = tabela.reset_index()

por_hora = tabela.drop(columns='Nº Carro').sum(axis=0)
tabela=por_hora.plot(kind='bar', figsize=(10,5))
st.pyplot(tabela.figure)
st.write("Você Selecionou:", options)