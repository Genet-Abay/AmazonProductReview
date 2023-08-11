import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from controller import *

sns.set_theme(style="darkgrid")

sns.set()
st.title("Review sentiment classification")


input_data = ''
df=pd.DataFrame()
with st.sidebar:
    input_data = st.file_uploader("upload csv file containing ASIN and review columns", type=["csv"])
    if input_data is not None:
        df = pd.read_csv(input_data)    
        if st.checkbox("show uploaded data"):
            st.dataframe(df)    
if input_data is not None and not df.empty:
    list_asin= df['asins'].unique().tolist()
    asin_option = st.selectbox('Select ASIN to display wordcloud', list_asin)   
    fig = plot_wrdCloud(df, asin_option)
    st.pyplot(fig)

    if st.button('Classify review'):
        df['sentiment'] = get_sentiment(df)
        
        fig2= plot_barplot(df)
        st.pyplot(fig2)
        st.subheader('Review content with classified sentiments')
        st.dataframe(df)

else:
    st.write("No data uploaded")
