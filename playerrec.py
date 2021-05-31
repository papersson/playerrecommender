import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import pdist, squareform

def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()

df = pd.read_pickle('./data/df.pkl')
FEATURES = ['distribution', 'shot_stopping', 'dominance', 'tenacity', 'awareness', 'power', 'mobility', 'composure', 'passing', 'dribbling', 'shooting',
           'height_cm', 'preferred_foot_Left', 'preferred_foot_Right', 'overall']

class PlayerRecommender():

    @st.cache
    def _build_similarity_matrix(self, df, metric='cosine'):
        return squareform(pdist(df.values, metric=metric))

    def train(self, df, metric='cosine'):
        self.df = df
        self.df_norm = self.df[FEATURES].copy()
        self.df_norm = (self.df_norm - self.df_norm.min()) / (self.df_norm.max() - self.df_norm.min())
        self.matrix = self._build_similarity_matrix(self.df_norm)#squareform(pdist(self.df.values, metric=metric))
        return self

    def predict(self, p_name, n_results=20, lwr=0, upr=99):
        # Locate row in similarity matrix and display most similar entries in that row
        idx = self.df.loc[self.df.short_name == p_name].index[0]
        most_similar_idxs = np.argsort(self.matrix[idx, :]) 
        res = self.df.loc[most_similar_idxs, ['short_name', 'age', 'preferred_foot', 'Position', 'club', 'overall']][1:n_results]
        res.columns = ['Name', 'Age', 'Preferred Foot', 'Position', 'Club', 'FIFA Rating']
        res.index = [""] * len(res) # Hide dataframe indices
        return res


rec = PlayerRecommender()
rec.train(df)

st.title('Player Recommender')

# user_input = st.text_input("Recommend me a player similar to:", 'L. Messi')
choice = st.selectbox("Recommend me a player similar to: ", df.short_name)
#data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
recommendations = rec.predict(choice, n_results=50)
# Notify the reader that the data was successfully loaded.
#data_load_state.text(f'Players similar to: {choice}')

st.write(recommendations)


