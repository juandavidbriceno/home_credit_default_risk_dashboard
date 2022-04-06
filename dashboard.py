import dill
import streamlit as st
import numpy as np
import pandas as pd
import time
from urllib.request import urlopen
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Methods.
x = [1,2,3,4,5,6,7,8]
y = [1,4,9,16,25,36,49,64]
z = [-1,-4,-9,-16,-25,-36,-49,-64]
dict_val = {'x': x,
        'y': y,
        'z': z}
df_graph = pd.DataFrame.from_dict(dict_val)

#Normally on : 'https://home-credit-default-risk-api.herokuapp.com/'
#@st.cache(suppress_st_warning=True)
#def depict_explanations(df, var):
#    f, ax = plt.subplots(2, 3, figsize=(10,10), sharex=False)
#    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
#    for i in range(0,2):
#        for j in range(0,3):
#            #print(str(i)+','+str(j))
#            p = sns.barplot(y = df['y'], x = df[var],ax = ax[i, j])
#            p.set_title(str(i)+','+str(j))
#            p.set_xlabel("X-Axis", fontsize = 20)
#            p.set_ylabel("Y-Axis", fontsize = 20)
#    st.pyplot(f)

#    
def depict_explanations(df, var):
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Scatter(x=df_graph['x'].tolist() , y=df_graph[var].tolist()),row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df_graph['x'].tolist() , y=df_graph[var].tolist()),row=1, col=2
    )
    fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    return fig
    
#Recalculates explanation.
def recalculate(df_explanation, var, coefficient_slicer):
    c = coefficient_slicer
    df_explanation[var] = df_explanation[var]*c
    return df_explanation

#1.Loading data.
@st.cache #mise en cache de la fonction pour exécution unique
def load_data():
    df = pd.DataFrame({
        'SK_ID_CURR': ['1001a', '1001a', '1001a', 1001, 1005, 1006,  1007,  1009],
        'LABELS': [0, 0, 0, 1, 1, 1, 1, 1],
        'Type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        'Coefficient': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    })
    return df

#Obtain data.
dataframe = load_data()
liste_id = dataframe['SK_ID_CURR'].tolist()


#2.affichage formulaire
st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")
id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )

sample_en_regle = str(list(dataframe[dataframe['LABELS'] == 1].sample(2)[['SK_ID_CURR', 'LABELS']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples d\'id de clients en règle : ' +sample_en_regle
sample_en_defaut = str(list(dataframe[dataframe['LABELS'] == 0].sample(2)[['SK_ID_CURR', 'LABELS']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut

#The sebsibility checkbox is available since the beginning.
sensibility = st.sidebar.checkbox('Do sensibility')

if id_input == '': #lorsque rien n'a été saisi
    st.write(chaine_en_defaut)
    st.write(chaine_en_regle)

elif(int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API
    
    API_url = "http://127.0.0.1:5000/credit/"+id_input
    
    #Developing for client's score prediction. 
    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        proba = API_data[id_input]
        chaine = 'The probability of the client '+str(id_input)+' is: '+str(proba) 
    st.markdown(chaine)
    
    #Developing for characteristics that explain the score (graphs).
    st.subheader("Caractéristiques influençant le score")
    with st.spinner('Chargement des détails de la prédiction...'):
        #Load data about images.
        SK_ID_CURR = [1001, 1005, 1006,  1007]
        graph_explanations = [df_graph, df_graph, df_graph, df_graph]
        dictionary_graphs = dict(zip(SK_ID_CURR,graph_explanations))
        
        #Take explanation for user.
        df_explanation = dictionary_graphs[int(id_input)]
        
        #explanation = chargement_explanation(str(id_input), 
        #dataframe, 
        #StackedClassifier(), 
        #sample=False)
  
    #Variable choice for interpretation.
    variable = list(df_explanation.columns)
    variable.remove('x')
    make_choice = st.multiselect('Select meaningful variables to depict explanation:', variable)
    if make_choice : 
        pl = st.empty()
        with st.spinner('Chargement des détails de la prédiction...'):
            fig = depict_explanations(df_explanation, make_choice[0])
            pl.write(fig)
        if sensibility==True:
            coefficient = st.sidebar.slider('Change variable value', 0, 5)
            with st.spinner('Chargement des détails de la prédiction...'):
                df_explanation = recalculate(df_explanation,make_choice[0],coefficient)
                fig = depict_explanations(df_explanation, make_choice[0])
                pl.write(fig)
            #df_explanation[make_choice] = df_explanation[make_choice]*coefficient
            #fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")   

else: 
    st.write('Identifiant non reconnu')