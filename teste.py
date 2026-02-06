import streamlit as st
import numpy as np
import pandas as pd

st.file_uploader("Insira aqui a pasta com os arquivos que voce quer analisar", type="csv", accept_multiple_files="directory")