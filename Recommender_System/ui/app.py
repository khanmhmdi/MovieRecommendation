import streamlit as st 
import pickle

movies = pickle.load(open('C:/Users/hasan/OneDrive/Desktop/ml_final_project/jupyter/recommendation/movies_list.pkl', 'rb'))
similarity = pickle.load(open('C:/Users/hasan/OneDrive/Desktop/ml_final_project/jupyter/recommendation/similarity.pkl', 'rb'))
movies_list = movies['title'].values

st.header("Movie Recommender System")
selectvalue = st.selectbox("Select movie from dropdown", movies_list)


def recommand(movie):
    index=movies[movies['title']==movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    recommand_movies = []
    for i in distance[1:6]:
        recommand_movies.append(movies.iloc[i[0]].title)
    return recommand_movies


if st.button("Show Recommend"):
    movies_name = recommand(selectvalue)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: 
        st.text(movies_name[0])
    with col2: 
        st.text(movies_name[1])
    with col3: 
        st.text(movies_name[2])
    with col4: 
        st.text(movies_name[3])
    with col5: 
        st.text(movies_name[4])
               
                    
        
