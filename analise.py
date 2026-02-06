import streamlit as st
import numpy as np
import pandas as pd
from Insercao_Arquivos import dataframes


st.markdown(
    """
    Agora, cabe a você selecionar o que você quer ver
    **Insira seus arquivos .csv para iniciar a análise!**
    ### Tipos de Processos Disponíveis
    - Visualização(gráficos)
    - Separação em Arquivos Diferentes
    - Análise separada por modo de transporte, ou modo de pagamento, por hora, dia da semana
    -Conjugada ao calendário
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



st.write("Você Selecionou:", options)