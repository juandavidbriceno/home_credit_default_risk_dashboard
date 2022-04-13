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
from toolbox.p7_dashboard_fonctions_aide import*
from toolbox.p7_traitement_données_et_modélisation_fonctions_aide import*

@st.cache #mise en cache de la fonction pour exécution unique
def load_data():
    #Obtain complete dataframe.
    final_dict_df =load_object('static/data/final_dict_df.pkl')
    X_total, X_total_original, X_train, X_test, y_train, X_train_original, X_test_original= final_dict_df['X_total'], final_dict_df['X_total_original'], final_dict_df['X_train'],final_dict_df['X_test'],final_dict_df['y_train'],final_dict_df['X_train_original'],final_dict_df['X_test_original']
    
    #Get values.
    clients = X_train.index
    labels = y_train['TARGET']
    
    #Create new dataframe.
    df_clients = pd.DataFrame(clients)
    df_clients.index = X_train.index
    df_labels = pd.DataFrame(labels)
    df_labels.index = X_train.index
    df = pd.concat([df_clients,df_labels], axis=1)
    df = df.reset_index(drop=True)
    df.columns = ['SK_ID_CURR','LABELS']
    return df, X_total, X_total_original, X_train, X_train_original, y_train 

#1.Obtain data to start.
dataframe, X_total, X_total_original, X_train, X_train_original, y_train = load_data()
liste_id = dataframe['SK_ID_CURR'].tolist()
classifier = load_object('static/classifiers/classifier_gd_sr.pkl').best_estimator_
scaler = load_object('static/scalers_and_encoders/scaler.pkl')
encoder = load_object('static/scalers_and_encoders/enc.pkl')
colors = obtain_list_n_different_colors(100)

#2.affichage formulaire
st.title('Dashboard Credit Score')
st.subheader("Prediction of client's credit score and comparison with the ensemble of clients")
id_input = st.text_input('Please select the identification number of a client:', )

sample_en_regle = str(list(dataframe[dataframe['LABELS'] == 0].sample(2)[['SK_ID_CURR', 'LABELS']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples of clients without default risk : ' +sample_en_regle
sample_en_defaut = str(list(dataframe[dataframe['LABELS'] == 1].sample(2)[['SK_ID_CURR', 'LABELS']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples of clients with default risk : ' + sample_en_defaut

#The sebsibility checkbox is available since the beginning.
sensibility = st.sidebar.checkbox('Do sensibility')

if id_input == '': #lorsque rien n'a été saisi
    st.write(chaine_en_defaut)
    st.write(chaine_en_regle)

elif(int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API
    
    API_url = "https://home-credit-default-risk-api.herokuapp.com/credit/"+str(id_input)
    
    #Developing for client's score prediction. 
    with st.spinner('Loading the client score...'):
        json_url = urlopen(API_url)
        API_data =json.loads(json_url.read())
        
        #Obtain requested values.
        predicted_class = API_data['risk']
        proba = API_data['def_prob']
        
        if predicted_class==1:
            status = "has risk"
        else:
            status = "hasn't risk"
        
        chaine = "The client identified with ID "+str(id_input)+" "+status+" with a default estimated probability of "+str(proba) 
    st.markdown(chaine)
    
    #Update state of database in regards the client selection.
    X_train, y_train, X_train_original = update_client_in_X_train_after_prediction(int(id_input),X_total, X_total_original, X_train, y_train, X_train_original, classifier)
    
    #Develop for explanations that explain the score (graphs).
    agree_1 = st.checkbox('Show first phase of interpretability')
    if agree_1:
        st.subheader("Characteristics that influence the score")
        with st.spinner('Loading details...'):
        #Take explanation for user.
            fig_1 = interpretation_credits_history_client(int(id_input), X_train_original, title ="Client Credit History")
            st.write(fig_1)
    
    agree_2 = st.checkbox('Show second phase of interpretability')
    if agree_2:
        with st.spinner('Loading details...'):
            fig_2 = interpretation_actual_credit_demand_client(int(id_input), X_train_original, title ="Client's Actual Demand in Regards to its Credit History")
            st.write(fig_2)
    
    agree_2 = st.checkbox('Show third phase of interpretability')
    if agree_2:
        st.subheader("Comparison with an ensemble of similar clients (k=100)")
        #Variable choice for interpretation.
        real_columns_1 = X_train_original.select_dtypes(include=[np.float,'int64']).columns
        variables_1 = list(real_columns_1)
        make_choice_1 = st.multiselect('Select meaningful variables to depict explanation:', variables_1)
        agree_3 = st.checkbox('Start analysis')
        if make_choice_1:
            if agree_3:
            #pl = st.empty()
                with st.spinner('Loading details of predicton...'):
                    fig_3 = interpretation_from_neighbors_client(int(id_input),X_train_original, y_train,make_choice_1,100, colors, '100 neighbors comparison')
                    st.write(fig_3)

    if sensibility==True:
        st.sidebar.markdown("Annotation: variables 'EXT_SOURCE_1', 'EXT_SOURCE_2', and 'EXT_SOURCE_3' could be varied from an scale between 1 to 100 cause their sliders are in percent terms.")
    #The client will select maximun two variables.
        real_columns_2 = X_train_original.select_dtypes(include=[np.float,'int64']).columns
        variables_2 = list(real_columns_2)
        make_choice_2 = st.sidebar.multiselect('Select meaningful variables to perform sensibility', variables_2)
        agree_4 = st.sidebar.checkbox('Done')
        if agree_4:
            if make_choice_2:
                VARS = []
                first_time = True
                #Create variables following user specifications.
                for i in range(0,len(make_choice_2)):
                    #Conditions.
                    if make_choice_2[i] == 'AMT_INCOME_TOTAL':
                        VAR_i = st.sidebar.slider(make_choice_2[i],25000,10000000, value=int(obtain_specific_value_column(int(id_input),X_train_original,make_choice_2[i])), step=1000)
                    elif make_choice_2[i] == 'EXT_SOURCE_1':
                        VAR_i = st.sidebar.slider(make_choice_2[i],0,100, value=int(float(obtain_specific_value_column(int(id_input),X_train_original,make_choice_2[i]))*100), step=1)
                    elif make_choice_2[i] == 'EXT_SOURCE_2':
                        VAR_i = st.sidebar.slider(make_choice_2[i],0,100, value=int(float(obtain_specific_value_column(int(id_input),X_train_original,make_choice_2[i]))*100), step=1)
                    elif make_choice_2[i] == 'EXT_SOURCE_3':
                        VAR_i = st.sidebar.slider(make_choice_2[i],0,100, value=int(float(obtain_specific_value_column(int(id_input),X_train_original,make_choice_2[i]))*100), step=1)
                    elif make_choice_2[i] == 'PAYMENT_RATE':
                        VAR_i = st.sidebar.slider(make_choice_2[i],0.00,0.30, value=float(obtain_specific_value_column(int(id_input),X_train_original,make_choice_2[i])), step=0.001)
                    elif make_choice_2[i] == 'RATIO_CREDIT_INCOME':
                        VAR_i = st.sidebar.slider(make_choice_2[i],float(X_total_original[make_choice_2[i]].min()),float(X_total_original[make_choice_2[i]].max()), value=float(obtain_specific_value_column(int(id_input),X_train_original,make_choice_2[i])), step=0.001)
                    elif make_choice_2[i] == 'RATIO_CREDIT_ANNUITY':
                        VAR_i = st.sidebar.slider(make_choice_2[i],float(X_total_original[make_choice_2[i]].min()),float(X_total_original[make_choice_2[i]].max()), value=float(obtain_specific_value_column(int(id_input),X_train_original,make_choice_2[i])), step=0.001)
                    else:
                        VAR_i = st.sidebar.slider(make_choice_2[i], max(0,int(X_total_original[make_choice_2[i]].min())),min(5000000,int(X_total_original[make_choice_2[i]].max())),int(obtain_specific_value_column(int(id_input),X_train_original,make_choice_2[i])))
                    VARS.append(VAR_i)
                
                first_time= False
                #Update the values of client information based on user dynamics.
                if(len(make_choice_2)==1):
                    if VARS[0] is not None and first_time ==False:
                        if make_choice_2[0]=='EXT_SOURCE_1' or make_choice_2[0]=='EXT_SOURCE_2' or make_choice_2[0]=='EXT_SOURCE_3':
                            new_client_row, X_train_original_1 = recalculate_row_and_X(int(id_input), X_train_original,make_choice_2[0],float(VARS[0]/100))
                        else:
                            new_client_row, X_train_original_1 = recalculate_row_and_X(int(id_input), X_train_original,make_choice_2[0],float(VARS[0]))     
                else:
                    if VARS[0] is not None and first_time ==False:
                        if make_choice_2[0]=='EXT_SOURCE_1' or make_choice_2[0]=='EXT_SOURCE_2' or make_choice_2[0]=='EXT_SOURCE_3':
                            new_client_row, X_train_original_1 = recalculate_row_and_X(int(id_input), X_train_original,make_choice_2[0],float(VARS[0]/100))
                        else:
                            new_client_row, X_train_original_1 = recalculate_row_and_X(int(id_input), X_train_original,make_choice_2[0],float(VARS[0]))
                            
                    for i in range(1,len(make_choice_2)):
                        if VARS[i] is not None and first_time ==False:
                            if make_choice_2[i]=='EXT_SOURCE_1' or make_choice_2[i]=='EXT_SOURCE_2' or make_choice_2[i]=='EXT_SOURCE_3':
                                new_client_row, X_train_original_1 = recalculate_row_and_X(int(id_input), X_train_original_1,make_choice_2[i],float(VARS[i]/100))
                            else:
                                new_client_row, X_train_original_1 = recalculate_row_and_X(int(id_input), X_train_original_1,make_choice_2[i],float(VARS[i]))
                        
                #Load new predictions.    
                with st.spinner('Chargement des news détails de la prédiction...'):
                    new_client_row_1 = get_client_row_from_df(int(id_input), X_train_original_1) 
                    prob_1 = get_default_probability_client_from_new_row(new_client_row_1, scaler, encoder, classifier)
                            
                    if prob_1>0.5:
                        status_1 = "has risk"
                    else:
                        status_1 = "hasn't risk"
        
                    chaine_1 = "Client "+status_1+" with probability of "+str(prob_1) 
                    st.sidebar.markdown(chaine_1)
                    
else: 
    st.write('Identifiant non reconnu')
