import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv("consolidated.csv")


def read_csv():
    df = pd.read_csv("consolidated.csv")


def sort_by_value(group):
    return group.sort_values(by='cluster')


def preprocess_articles(selected_k):
    documents = df['Content'].values.astype("U")
    vectorizer = TfidfVectorizer(stop_words='english')
    features = vectorizer.fit_transform(documents)
    k = int(selected_k)
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    model.fit(features)
    df['cluster'] = model.labels_
    # Apply the sorting function to each group
    sorted_df = df.groupby('cluster').apply(sort_by_value)
    sorted_df.to_csv(f'data/clustered_and_sorted.csv', index=False)
    read_csv()
    st.success('Clustering completed, please refresh page!', icon="✅")


def display_clusters():
    df = pd.read_csv("data/clustered_and_sorted.csv")
    st.title("News Clustering")
    # st.table(df)
    st.write(df)


def main():
    st.sidebar.title('News Clustering')
    st.write("TONDERAI MUTOMBWA (R204739S)")
    selected_k = st.sidebar.text_input('Enter the number of clusters', '10')
    display_clusters()
    if st.sidebar.button('Cluster Articles'):
        preprocess_articles(selected_k)


if __name__ == "__main__":
    main()
