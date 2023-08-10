from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle as pkl
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def download_given_data(given_data):
    try:
        with open(os.path.join("temp/", given_data.name), 'wb') as f:
            f.write(given_data)
        return 1    

    except:
        return 0


def get_sentiment(df):
    with open("vectorizer.pkl", 'rb') as fv:
        tfidf_vectorizer = pkl.load(fv)
    vectorized_txt = tfidf_vectorizer.transform(df['reviews'])

    with open('model.pkl', 'rb') as fm:
        model = pkl.load(fm)
    sentmt = model.predict(vectorized_txt)
    return sentmt

def plot_wrdCloud(df, pos):
    list_asins = df['asins'].unique()

    curr_asin = list_asins[pos]
    filtered_product = df[df['asins']==curr_asin]
    concat_words = filtered_product['reviews'].str.cat(sep=' ')

    tokens = concat_words.split()
    comment_words=''
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        
    comment_words += " ".join(tokens)+" "

    # text(x=0.5, y=0.5, "Frequent word from review text for product with ASIN")

    fig, ax = plt.subplots(figsize = (6, 6))
    wordcloud = WordCloud(max_words=40, width = 400, height = 400, background_color ='white', stopwords = stopwords, min_font_size = 7).generate(comment_words)

            
    
    ax.imshow(wordcloud)
    ax.axis("off")
    plt.tight_layout(pad = 0)    
    plt.show()
    return fig

def plot_barplot(df, ):
     fig, ax = plt.subplots()
     pd.crosstab(df['asins'],df['sentiment']).plot(kind="bar",stacked=True, figsize=(10, 4), ax=ax)
     ax.set_title("Sentment rank for each ASIN")
     ax.set_xlabel('product ASIN')
     ax.set_ylabel('review sentiment')
     return fig