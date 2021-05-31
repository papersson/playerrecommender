import numpy as np
import pandas as pd
import streamlit as st

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


#st.set_page_config(layout='wide')

df = pd.read_pickle('./data/df.pkl')
df_std = pd.read_pickle('./data/df_std.pkl')
features = ['distribution', 'shot_stopping', 'dominance', 'tenacity', 'awareness', 'power', 'mobility', 'composure', 'passing', 'dribbling', 'shooting',
           'height_cm', 'preferred_foot_Left', 'preferred_foot_Right', 'overall']
print(df.head())

class PlayerRecommender():

    @st.cache
    def _get_matrix(self):
        return np.load('./data/matrix.npy')

    def train(self, df, metric='cosine'):
        self.df = df
        #self.df_norm = self.df[features].copy()
        #self.df_norm = (self.df_norm - self.df_norm.min()) / (self.df_norm.max() - self.df_norm.min())
        self.matrix = self._get_matrix()#squareform(pdist(self.df.values, metric=metric))
        return self

    def predict(self, p_name, n_results=20, lwr=0, upr=99):
        # Locate row in similarity matrix and display most similar entries in that row
        try:
            idx = self.df.short_name[df.short_name.str.contains(p_name, case=False)].index[0]
        except:
            idx = 0
            st.write('Name error! Defaulting to L. Messi.')
        most_similar_idxs = np.argsort(self.matrix[idx, :]) 
        res = self.df.loc[most_similar_idxs, ['short_name', 'age', 'preferred_foot', 'Position', 'club', 'overall']][1:n_results]
        res.columns = ['Name', 'Age', 'Preferred Foot', 'Position', 'Club', 'FIFA Rating']
        res.index = [""] * len(res) # Hide dataframe indices
        return res


rec = PlayerRecommender()
rec.train(df, df_std)

st.title('Player Recommender')

# user_input = st.text_input("Recommend me a player similar to:", 'L. Messi')
choice = st.selectbox("Recommend me a player similar to: ", df.short_name)
#data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
recommendations = rec.predict(choice, n_results=50)
# Notify the reader that the data was successfully loaded.
#data_load_state.text(f'Players similar to: {choice}')

st.write(recommendations)


