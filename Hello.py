import streamlit as st

st.set_page_config(
    page_title="Transporte Inteligente"
)
analise = st.Page("analise.py", title="Analise de Dados")
insercao_arquivos = st.Page("Insercao_Arquivos.py", title="Insercao de Arquivos")
menu = st.Page("menu.py", title="Menu")
pg = st.navigation([menu,insercao_arquivos,analise])

if 'dataframes' not in st.session_state:
    st.session_state.dataframes = []
pg.run()