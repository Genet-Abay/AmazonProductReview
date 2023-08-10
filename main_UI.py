import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from controller import *

sns.set_theme(style="darkgrid")

sns.set()
st.title("Review sentiment classification")
sentiment=''
input_data = st.file_uploader("upload csv file containing ASIN and review columns", type=["csv"])
if input_data is not None:
    df = pd.read_csv(input_data)
    
    if st.checkbox("show uploaded data"):
        st.dataframe(df)    

    fig = plot_wrdCloud(df, 0)
    st.pyplot(fig)

    if st.button('Classify review'):
        df['sentiment'] = get_sentiment(df)
        fig2= plot_barplot(df)
        st.pyplot(fig2)
        st.dataframe(df)

else:
    st.write("No data uploaded")
