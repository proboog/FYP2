import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Association rule mining app', layout='centered')

st.title("Association rule mining application")

st.sidebar.success("Select Auto scan or Manual scan")
