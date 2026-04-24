import streamlit as st
import polars as pl
import lets_plot
from lets_plot import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
@st.cache_data
def load_data():
    df = pl.read_csv("EPD_data.csv")
    return df
