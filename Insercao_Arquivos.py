import streamlit as st
import numpy as np
import pandas as pd

arquivos=st.file_uploader("Insira aqui a pasta com os arquivos que voce quer analisar", type="csv", accept_multiple_files="directory")
for arq in arquivos:
    df=pd.read_csv(arq,sep=';')
    st.session_state.dataframes.append(df)