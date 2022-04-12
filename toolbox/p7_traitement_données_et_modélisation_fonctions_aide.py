from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import numpy as np
import pandas as pd
import math
import seaborn as sns
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client as client
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from random import randint
from scipy.spatial import distance
#from p7_consolidation_données_fonctions_aide import*
from toolbox.p7_consolidation_données_fonctions_aide import*

#Define a function that updates the state of a test set, in regards to change in a train set.
def df_test_update_in_regards_df_train(df_test,df_train):
    df_train_mod = df_train.copy()
    cols = list(df_train_mod.columns)
    #Update df_test.
    df_test_mod = df_test.copy()
    df_test_mod = df_test_mod[cols]
    return df_test_mod

#Defines a function that returns a list of columns having more than a certain percentage of null values.
def df_give_almost_null_columns_given_threshold(df,threshold):
    #Obtain description.
    df_mod = df.copy()
    df_description = give_df_info(df_mod)
    cols = df_description[df_description['Nan_percent']>threshold]
    cols = sorted(list(cols['Variable_names'].unique()))
    return cols

#Defines a function that removes from a dataframe columns having more null values than a specified certain percentage.
def df_remove_almost_null_columns_given_threshold(df,threshold):
    df_mod = df.copy()
    cols = df_give_almost_null_columns_given_threshold(df_mod,threshold)
    if(cols):
        for i in cols:
            df_mod = df_mod.drop([i], axis = 1)
    return df_mod

#Defines a function that removes from train and test sets columns having more null values than a specified certain percentage.
def X_remove_almost_null_columns_given_threshold(X_train, X_test,threshold):
    X_train_mod = X_train.copy()
    X_test_mod = X_test.copy()
    X_train_mod = df_remove_almost_null_columns_given_threshold(X_train_mod, threshold)
    X_test_mod = df_test_update_in_regards_df_train(X_test_mod, X_train_mod)
    return X_train_mod, X_test_mod

#Define functions to deal with missing values in independent real variables. 
def X_treat_missing_values_real(X_real):
    X_real_mod = X_real.copy()
    columns = list(X_real_mod.columns)
    columns_without_EXIT = columns.copy()
    columns_without_EXIT.remove('EXT_SOURCE_1')
    columns_without_EXIT.remove('EXT_SOURCE_2')
    columns_without_EXIT.remove('EXT_SOURCE_3')
    
    #Store EXT variables.
    EXT_df = X_real.copy()
    EXT_df = EXT_df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]
   
    #For real variables when values are 'nan' we replace by zero.
    X_real_mod = X_real_mod[columns_without_EXIT]
    X_real_mod = X_real_mod.fillna(0)
    
    #Append EXT variables.
    X_real_mod = pd.concat([X_real_mod,EXT_df],axis=1)
    X_real_mod = X_real_mod[columns]
    
    return X_real_mod

#Define functions to deal with missing values in independent categorical variables. 
def X_treat_missing_values_categorical(X_categorical):
    X_categorical_mod = X_categorical.copy()
    
    try:
    #For the variables'NAME_FAMILY_STATUS' When it is unknown, it is filled with 'Cash loans'.
        X_categorical_mod['NAME_CONTRACT_TYPE'] = X_categorical_mod['NAME_CONTRACT_TYPE'].fillna('Cash loans')
    except:
        a =1
            
    try:
        #For the variables'CODE_GENDER' When it is unknown, it is filled with 'M'.
        X_categorical_mod['CODE_GENDER'] = X_categorical_mod['CODE_GENDER'].fillna('M')
    except:
        a =1
    
    try:
        #For the variables'FLAG_OWN_CAR' When it is unknown, it is filled with 'N'.
        X_categorical_mod['FLAG_OWN_CAR'] = X_categorical_mod['FLAG_OWN_CAR'].fillna('N')
    except:
        a =1
    
    try:
        #For the variables'CNT_CHILDREN' When it is unknown, it is filled with 0.
        X_categorical_mod['CNT_CHILDREN'] = X_categorical_mod['CNT_CHILDREN'].fillna(0)
    except:
        a =1

    try:   
        #For the variables'CNT_CHILDREN' When it is unknown, it is filled with 0.
        X_categorical_mod['NAME_EDUCATION_TYPE'] = X_categorical_mod['NAME_EDUCATION_TYPE'].fillna('Secondary / secondary special')
    except:
        a =1
            
    try: 
        #For the variables'NAME_FAMILY_STATUS' When it is unknown, it is filled with 'Single / not married'.
        X_categorical_mod['NAME_FAMILY_STATUS'] = X_categorical_mod['NAME_FAMILY_STATUS'].fillna('Single / not married')
    except:
        a =1    
    
    try:
        #For the variables'NAME_FAMILY_STATUS' When it is unknown, it is filled with 'Single / not married'.
        X_categorical_mod['NAME_TYPE_SUITE'] = X_categorical_mod['NAME_TYPE_SUITE'].fillna('Other_A')
    except:
        a =1
    
    try:
        #For the variables'NAME_INCOME_TYPE' When it is unknown, it is filled with 'Other'.
        X_categorical_mod['NAME_INCOME_TYPE'] = X_categorical_mod['NAME_INCOME_TYPE'].fillna('Other')
    except:
        a =1
            
    try:
        #For the variables'OCCUPATION_TYPE' When it is unknown, it is filled with 'Other'.
        X_categorical_mod['OCCUPATION_TYPE'] = X_categorical_mod['OCCUPATION_TYPE'].fillna('Other')
    except:
        a =1
    
    try:
        #For the variables'ORGANIZATION_TYPE' When it is unknown, it is filled with 'Other'.
        X_categorical_mod['ORGANIZATION_TYPE'] = X_categorical_mod['ORGANIZATION_TYPE'].fillna('Other')
    except:
        a =1
    
    try:
        #For the variables'ORGANIZATION_TYPE' When it is unknown, it is filled with 'Other'.
        X_categorical_mod['HOUSETYPE_MODE'] = X_categorical_mod['HOUSETYPE_MODE'].fillna('Other')
    except:
        a =1
    
    try:
        #For the variables'WALLSMATERIAL_MODE' When it is unknown, it is filled with 'Others'.
        X_categorical_mod['WALLSMATERIAL_MODE'] = X_categorical_mod['WALLSMATERIAL_MODE'].fillna('Others')
    except:
        a =1
            
    try:
        #For the variables'WALLSMATERIAL_MODE' When it is unknown, it is filled with 'No'.
        X_categorical_mod['EMERGENCYSTATE_MODE'] = X_categorical_mod['EMERGENCYSTATE_MODE'].fillna('No')
    except:
        a =1
            
    return X_categorical_mod

#Define functions to deal with missing values in independent variables.
def X_treat_missing_values(X):
    X_mod = X.copy()
    cols = X_mod.columns
    #Work over real variables.
    X_mod_real = X_mod.select_dtypes(include=[np.float])
    X_mod_real = X_treat_missing_values_real(X_mod_real)
    #Work over categorical variables.
    X_mod_categorical = X_mod.select_dtypes(include=['object','bool','int64'])
    X_mod_categorical = X_treat_missing_values_categorical(X_mod_categorical)
    
    X_mod_return = pd.concat([X_mod_real,X_mod_categorical],axis =1)
    X_mod_return = X_mod_return[cols]
    return X_mod_return

#Defines a function that inputs variables 'EXT' in a X dataset.
def X_inputing_EXT_columns_first_time(X_train, X_test, n_trees =150):
    X_train = X_train.copy()
    X_test = X_test.copy()
    xgb_reg = []
    #using only numeric columns for predicting the EXT_SOURCES
    columns_for_modelling = list(set(X_test.dtypes[X_test.dtypes != 'object'].index.tolist())- set(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','SK_ID_CURR']))
    #we'll train an XGB Regression model for predicting missing EXT_SOURCE values
    #we will predict in the order of least number of missing value columns to max.
    for ext_col in ['EXT_SOURCE_2','EXT_SOURCE_3','EXT_SOURCE_1']:
    #X_model - datapoints which do not have missing values of given column
    #Y_train - values of column trying to predict with non missing values
    #X_train_missing - datapoints in application_train with missing values
    #X_test_missing - datapoints in application_test with missing values
        print(ext_col)
        X_model, X_train_missing, X_test_missing, Y_train = X_train[~X_train[ext_col].isna()][columns_for_modelling], X_train[X_train[ext_col].isna()][columns_for_modelling], X_test[X_test[ext_col].isna()][columns_for_modelling], X_train[ext_col][~X_train[ext_col].isna()]
        xg = XGBRegressor(n_estimators = n_trees, max_depth = 3, learning_rate = 0.1, n_jobs = -1, random_state = 59)
        xg.fit(X_model, Y_train)
        X_train[ext_col][X_train[ext_col].isna()] = xg.predict(X_train_missing)
        X_test[ext_col][X_test[ext_col].isna()] = xg.predict(X_test_missing)
        #adding the predicted column to columns for modelling for next column's prediction
        columns_for_modelling = columns_for_modelling + [ext_col]
        xgb_reg.append(xg)
        print('finish'+ext_col)
    return X_train, X_test, xgb_reg

#Defines a function that inputs variables 'EXT' in a X dataset given already trained inputters.
def X_inputing_EXT_columns(X_train, X_test,xgb_reg):
    X_train = X_train.copy()
    X_test = X_test.copy()
    #using only numeric columns for predicting the EXT_SOURCES
    columns_for_modelling = list(set(X_test.dtypes[X_test.dtypes != 'object'].index.tolist())- set(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','SK_ID_CURR']))

    #Use already trained inputters.
    xgb_reg = xgb_reg.copy()
    cont = 0
    
    for ext_col in ['EXT_SOURCE_2','EXT_SOURCE_3','EXT_SOURCE_1']:
        X_model, X_train_missing, X_test_missing, Y_train = X_train[~X_train[ext_col].isna()][columns_for_modelling], X_train[X_train[ext_col].isna()][columns_for_modelling], X_test[X_test[ext_col].isna()][columns_for_modelling], X_train[ext_col][~X_train[ext_col].isna()]
        X_train[ext_col][X_train[ext_col].isna()] = xgb_reg[cont].predict(X_train_missing)
        X_test[ext_col][X_test[ext_col].isna()] = xgb_reg[cont].predict(X_test_missing)
        #adding the predicted column to columns for modelling for next column's prediction
        columns_for_modelling = columns_for_modelling + [ext_col]
        cont = cont+1
    return X_train, X_test

#Defines a function that normalize real variables.
def X_normalize_numerical_variables(X,return_scaler=False):
    #Work over real part.
    X_mod = X.copy()
    
    #Work over real.
    X_mod_real = X_mod.select_dtypes(include=[np.float])
    cols = X_mod_real.columns
    scaler = MinMaxScaler()
    scaler.fit(X_mod_real)
    X_mod_real = pd.DataFrame(scaler.transform(X_mod_real),columns =cols)
    X_mod_real.index = X_mod.index
    
    #Work over int part.
    X_mod_int = X_mod.select_dtypes(include=['int64'])
    X_mod = pd.concat([X_mod_real, X_mod_int],axis =1)
        
    if(return_scaler==True):
        return X_mod, scaler
    else:
        return X_mod

#Defines a function that normalize real variables on a test dataframe.
def X_test_normalize_numerical_variables(X_test, scaler):
    #Work over real part.
    X_mod = X_test.copy()
    
    #Work over real.
    X_mod_real = X_mod.select_dtypes(include=[np.float])
    cols = X_mod_real.columns
    X_mod_real = pd.DataFrame(scaler.transform(X_mod_real),columns =cols)
    X_mod_real.index = X_mod.index
    
    #Work over int part.
    X_mod_int = X_mod.select_dtypes(include=['int64'])
    X_mod = pd.concat([X_mod_real, X_mod_int],axis =1)
    return X_mod

#Defines a function that encodes categorical columns on a dataframe,
def X_encode_categorical_variables(X_categorical, return_enc=False):
    #Encode variables.
    X_categorical_mod = X_categorical.copy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X_categorical_mod)
    X_categorical_mod = enc.transform(X_categorical_mod).toarray() 
    X_encode_categorical_variables = pd.DataFrame(X_categorical_mod, columns = enc.get_feature_names(X_categorical.columns))
    X_encode_categorical_variables.index = X_categorical.index 
    if(return_enc==True):
        return X_encode_categorical_variables, enc
    else:
        return X_encode_categorical_variables
    
#Defines a function that encodes categorical columns on a test dataframe,
def X_test_encode_categorical_variables(X_categorical, encoder):
    X_categorical_mod = X_categorical.copy()
    X_categorical_mod = encoder.transform(X_categorical_mod).toarray() 
    X_categorical_mod = pd.DataFrame(X_categorical_mod, columns = encoder.get_feature_names(X_categorical.columns))
    X_categorical_mod.index = X_categorical.index
    return X_categorical_mod

#Define a function that normalize and encode variavles on a dataframe.
def X_normalize_and_encode_variables(X):
    X_mod = X.copy()
    #Work over numerical variables.
    try:
        X_mod_numerical, scaler = X_normalize_numerical_variables(X_mod, True)
    except:
        X_mod_numerical = None 
        scaler = None
    #Work over categorical variables.
    try:
        X_mod_categorical = X_mod.select_dtypes(include=['object','bool'])
        X_mod_categorical, encoder = X_encode_categorical_variables(X_mod_categorical, True)
    except:
        X_mod_categorical = None
        encoder = None
    
    #Return stage.
    if X_mod_numerical is None:
        X_mod_return = X_mod_categorical
    
    if X_mod_categorical is None:
        X_mod_return = X_mod_numerical
    
    if X_mod_numerical is not None:
        if X_mod_categorical is not None:
            X_mod_return = pd.concat([X_mod_numerical,X_mod_categorical],axis=1)
    
    return X_mod_return, scaler, encoder

#Define a function that normalize and encode variavles on a test dataframe.
def X_test_normalize_and_encode_variables(X_test, scaler, encoder):
    X_mod = X_test.copy()
    #Work over numerical variables.
    if(scaler is not None):
        X_mod_numerical = X_test_normalize_numerical_variables(X_mod, scaler)
    #Work over categorical variables.
    if(encoder is not None):
        X_mod_categorical = X_mod.select_dtypes(include=['object','bool'])
        X_mod_categorical = X_test_encode_categorical_variables(X_mod_categorical,encoder)
    
    #Return.
    if scaler is None:
        if encoder is None:
            X_mod_return = None
        else:
            X_mod_return = X_mod_categorical
    else:
        if encoder is None:
            X_mod_return = X_mod_numerical
        else:
            X_mod_return = pd.concat([X_mod_numerical,X_mod_categorical],axis =1)
  
    return X_mod_return

#Define a function that randomly selects values of X, and y in regards to a class in the target variable.
def X_y_random_sample(X,y,n,value_y=0):
    X_mod = X.copy()
    y_mod = y.copy()
    y_mod = y_mod[y_mod['TARGET']==value_y]
    #Obtain random sample for y.
    y_mod = y_mod.sample(n=n)
    #Obtain random sample for X.
    X_mod = X_mod.loc[list(y_mod.index),:]
    return X_mod, y_mod

#Define a function that randomly selects values of X balancing classes in the target variable.
def X_y_random_sample_balanced_classes(X, y,n):
    X_1, y_1 = X_y_random_sample(X, y, int(n/2), value_y=0)
    X_2, y_2 = X_y_random_sample(X, y, int(n/2), value_y=1)
    X_mod = pd.concat([X_1,X_2],axis=0)
    y_mod = pd.concat([y_1,y_2],axis=0)
    
    #Shufflying.
    X_mod = X_mod.sample(frac = 1)
    y_mod = y_mod.loc[list(X_mod.index),:]
    return X_mod,y_mod

#Define a function that obtains for a client, the distances to other clients in regards the data.
def client_distances_with_others_df(client, X):
    #Create copies.
    X_mod = X.copy()
    
    #Obtain list of clients.
    clients = list(X_mod.index)
    clients.remove(client)
    
    #Obtain data of client.
    arr_client = X_mod.loc[[client],:].values
    
    #Obtain data of other clients.
    arr_other_clients = X_mod.loc[clients,:].values
    
    #Calcualte distances.
    distances = distance.cdist(arr_client, arr_other_clients, 'euclidean')
    distances = distances.T
    distances = distances.reshape(distances.shape[0],)
    distances = list(distances)
    
    #Create dataframe.
    distances = pd.DataFrame(list(zip(clients,distances)))
    distances.columns = ['SK_ID_CURR','distance']
    distances = distances.set_index('SK_ID_CURR')
    df_distances = distances.sort_values(by='distance', ascending=True)
    return df_distances

#Define a function that allows to obtain for a client, its neighbors given distances with other clients.
def obtain_neighbors_from_client_distances_with_others_df(df_distances, n_neighbors):
    df_distances_mod = df_distances.copy()
    df_distances_mod = df_distances_mod.iloc[0:n_neighbors,:]
    return list(df_distances_mod.index)

#Define a function that allows to find for a client its n closest neighbors information as a dataframe.
def client_neighbors_given_attributes(client_id, X_original, cols_attributes, n_neighbors = 50, return_normalized = False):
    
    #Obtain client data.
    X_client = X_original.copy()
    X_client = X_client[cols_attributes]
    X_client = X_client[X_client.index==client_id]
    
    #Copies.
    X_mod = X_original.copy()
    X_mod = X_mod[cols_attributes]
    X_mod, scaler, enc = X_normalize_and_encode_variables(X_mod)
    X_client_norm = X_mod[X_mod.index==client_id]
    
    #Obtain neighbors. 
    distances = client_distances_with_others_df(client_id,X_mod)
    list_neighbors = obtain_neighbors_from_client_distances_with_others_df(distances,n_neighbors)
    
    #Subset original dataframe by neighbors.
    X_to_return = X_original.copy()
    X_to_return = X_to_return[cols_attributes] 
    X_to_return = X_to_return.loc[list_neighbors,:]
    
    #Calcualte normalized dataframe.
    X_nor = X_mod.loc[list_neighbors,:]
    
    if(return_normalized==True):
        return X_to_return, X_nor, X_client, X_client_norm
    
    return X_to_return, X_client
    

#Define a function that allows to find for a client its n closest neighbors information as a dataframe.
def client_neighbors_given_attributes_and_target(client_id, X_original, y, cols_attributes, n_neighbors = 50, value_y=1, return_normalized=False):
    #Obtain client.
    X_client = X_original.copy()
    X_client = X_client[X_client.index==client_id]
    y_client = y.copy()
    y_client = y_client[y_client.index==client_id]
    
    #Make copy of X and y.
    X_mod = X_original.copy()
    y_mod = y.copy()
    
    #Filter y.
    y_mod = y_mod[y_mod['TARGET']==value_y]
    
    #Calculate commons between y and X.
    commons = lists_common_values(list(X_mod.index),list(y_mod.index))
    try:
        commons.remove(client_id)
    except:
        commons = commons
        
    #Filter X by commons.
    X_mod = X_mod[X_mod.index.isin(commons)]
    
    #Add client row to X_mod.
    X_mod = pd.concat([X_client,X_mod], axis=0)
    
    #Return
    if(return_normalized==True):
        #Obtain neightbors.
        X_to_return, X_nor, X_client, X_client_norm = client_neighbors_given_attributes(client_id, X_mod, cols_attributes, n_neighbors, True)
        y_mod = y_mod.loc[list(X_to_return.index),:]
        
        return X_to_return, X_nor, X_client, X_client_norm,  y_mod, y_client
    
    X_to_return, X_client  = client_neighbors_given_attributes(client_id, X_mod, cols_attributes, n_neighbors, False)
    y_mod = y_mod.loc[list(X_to_return.index),:]

    return X_to_return, X_client, y_mod, y_client

#1.Defining functons that help the plot procedure
#Defining a function that depicts client credit in pie.
def client_plotty_pie_plot(client_id, X, variables, plot_name = 'Accepted credits in Bureau'):
    data = X.copy()
    data = data[data.index==client_id]
    
    #Selecting data.
    np_vals = data[variables].values.T
    np_vals = np_vals.reshape(np_vals.shape[0],).tolist()
    x_vals = np_vals
    
    variables_ = variables
    variables_  = list(map(lambda x: x.replace('total_credits_active_b', 'Active_credits_bureauu'), variables_))
    variables_  = list(map(lambda x: x.replace('total_credits_closed_b', 'Closed_credits_bureau'), variables_))
    variables_  = list(map(lambda x: x.replace('is_active__sum_pa', 'Active_credits_others'), variables_))
    variables_  = list(map(lambda x: x.replace('total_credits_closed_pa', 'Closed_credits_others'), variables_))
    variables_  = list(map(lambda x: x.replace('_active_type_Consumer loans_pa', 'Active_credits_Consumer_loans'), variables_))
    variables_  = list(map(lambda x: x.replace('_active_type_Cash loans_pa', 'Active_credits_cash_loans'), variables_))
    variables_  = list(map(lambda x: x.replace('_active_type_Revolving loans_pa', 'Active_credits_Consumer_revolving_loans'), variables_))
    
    
    #Create plot.
    if (np.sum(np_vals) !=0):
        fig = px.pie(values=x_vals, names=variables_, color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textinfo='value')
    else:
        fig = px.pie(values=[1], names=['No_credits_accepted'], color_discrete_sequence = ['Grey'])
        fig.update_traces(textinfo='none')
    
    fig.update_layout(height=300, width=400, title_text=plot_name, title_x = 0.5,
                     font=dict(
                        size=13,
                        color="Black"
    ))
    return fig

#Defining a function that depicts client credit in pie.
def client_plotty_go_pie_plot(client_id, X, variables, colors = px.colors.sequential.RdBu):
    data = X.copy()
    data = data[data.index==client_id]
    
    #Selecting data.
    np_vals = data[variables].values.T
    np_vals = np_vals.reshape(np_vals.shape[0],).tolist()
    x_vals = np_vals
    
    variables_ = variables
    variables_  = list(map(lambda x: x.replace('total_credits_active_b', 'Active_credits_bureau'), variables_))
    variables_  = list(map(lambda x: x.replace('total_credits_closed_b', 'Closed_credits_bureau'), variables_))
    variables_  = list(map(lambda x: x.replace('is_active__sum_pa', 'Active_credits_others'), variables_))
    variables_  = list(map(lambda x: x.replace('total_credits_closed_pa', 'Closed_credits_others'), variables_))
    variables_  = list(map(lambda x: x.replace('_active_type_Consumer loans_pa', 'Active_credits_consumer_loans'), variables_))
    variables_  = list(map(lambda x: x.replace('_active_type_Cash loans_pa', 'Active_credits_cash_loans'), variables_))
    variables_  = list(map(lambda x: x.replace('_active_type_Revolving loans_pa', 'Active_credits_consumer_revolving_loans'), variables_))
    
    #Create plot.
    if (np.sum(np_vals) !=0):
        fig = go.Pie(labels=variables_,
                values=x_vals,
                showlegend=True,
                textinfo='value',
                marker_colors=colors)
    else:
        fig = go.Pie(labels=['No_credits_accepted'],
                values=[1],
                showlegend=True,
                textinfo='none',
                marker_colors=['Grey'])
    return fig

#Defines a function that for a client depicts a bar chart of its credit information.
def client_plotty_go_bar_plot(client_id, X, variables):
    #variables_  = list(map(lambda x: x.replace('total_credits_active_b', 'Active_credits_bureauu'), variables_))
    data = X.copy()
    data = data[data.index==client_id]

    #Selecting data.
    np_vals = data[variables].values.T
    np_vals = np_vals.reshape(np_vals.shape[0],).tolist()
    x_vals = np_vals

    #Changing variables names pa.
    variables_ = variables
    variables_  = list(map(lambda x: x.replace('AMT_CREDIT__mean_pa', 'AVG_AMT_CREDIT_others', ), variables_))
    variables_  = list(map(lambda x: x.replace('AMT_ANNUITY__mean_pa', 'AVG_AMT_ANNUITY_others', ), variables_))
    variables_  = list(map(lambda x: x.replace('active_AMT_CREDIT__mean_pa', 'AVG_ACT_AMT_CREDIT_others', ), variables_))
    variables_  = list(map(lambda x: x.replace('active_AMT_ANNUITY__mean_pa', 'AVG_ACT_ANNUITY_others', ), variables_))
    variables_  = list(map(lambda x: x.replace('active_AMT_CREDIT__sum_pa', 'SUM_ACT_AMT_CREDIT_others', ), variables_))
    variables_  = list(map(lambda x: x.replace('active_AMT_ANNUITY__sum_pa', 'SUM_ACT_AMT_ANNUITY_others', ), variables_))

    #'bureau_AMT_CREDIT_SUM_mean_b','bureau_AMT_CREDIT_SUM_sum_b',
    #'bureau_AMT_CREDIT_SUM_DEBT_mean_b','bureau_AMT_CREDIT_SUM_DEBT_sum_b']
    #Changing variables names b.
    variables_  = list(map(lambda x: x.replace('bureau_AMT_CREDIT_SUM_mean_b', 'AVG_AMT_CREDIT_bureau', ), variables_))
    variables_  = list(map(lambda x: x.replace('bureau_AMT_CREDIT_SUM_sum_b', 'SUM_AMT_CREDIT_bureau', ), variables_))
    variables_  = list(map(lambda x: x.replace('bureau_AMT_CREDIT_SUM_DEBT_mean_b', 'AVG_ACT_AMT_CREDIT_bureau', ), variables_))
    variables_  = list(map(lambda x: x.replace('bureau_AMT_CREDIT_SUM_DEBT_sum_b', 'SUM_AMT_ACT_CREDIT_bureau', ), variables_))
    
    #vars_bar_reported_days = ['total_reported_days_ccb','SK_DPD__mean_ccb',
    #'reported_days_sum_pi','total_reported_days_pcb','reported_days_ss'],avg_dpd_b
    #Changing variables names reported days..
    variables_  = list(map(lambda x: x.replace('total_reported_days_ccb', 'reported_days_credit_card_balance', ), variables_))
    variables_  = list(map(lambda x: x.replace('reported_days_sum_pi', 'reported_days_payment_installements', ), variables_))
    variables_  = list(map(lambda x: x.replace('total_reported_days_pcb', 'reported_days_pos_cash_balance', ), variables_))
    variables_  = list(map(lambda x: x.replace('reported_days_ss', 'reported_days_social_surroundings', ), variables_))
    variables_  = list(map(lambda x: x.replace('avg_dpd_b', 'reported_days_b', ), variables_))

    #'times_reported_delay_dues_ccb','times_reported_delay_dues_pi',
    #'times_reported_delay_dues_pcb'
    #Changing variables names reported delays.
    variables_  = list(map(lambda x: x.replace('times_reported_delay_dues_ccb', 'reported_delays_credit_card_balance', ), variables_))
    variables_  = list(map(lambda x: x.replace('times_reported_delay_dues_pi', 'reported_delays_payment_installements', ), variables_))
    variables_  = list(map(lambda x: x.replace('times_reported_delay_dues_pcb', 'reported_delays_pos_cash_balance', ), variables_))
    
    #Changing variables names general info.
    variables_  = list(map(lambda x: x.replace('bureau_years_CREDIT_max_b', 'max_years_credits_bureau', ), variables_))
    variables_  = list(map(lambda x: x.replace('max_years_credit_pa', 'max_years_credits_previous_application', ), variables_))
    
    #Create plot.
    fig = go.Bar(x=variables_, y=x_vals, legendgroup="group")
    return fig

#1.
#Define a function that plots pie chart credits with bureau for a client(pie).
def interpretation_credits_history_client(client_id, X, title ="Client Credit History"):
    #Build client figure.
    fig_go_1 = client_plotty_go_pie_plot(client_id, X, ['total_credits_active_b','total_credits_closed_b'])
    fig_go_2 = client_plotty_go_pie_plot(client_id, X, ['is_active__sum_pa','total_credits_closed_pa'])
    fig_go_3 = client_plotty_go_pie_plot(client_id, X, ['_active_type_Consumer loans_pa','_active_type_Cash loans_pa','_active_type_Revolving loans_pa'],colors =px.colors.sequential.Cividis_r)
    
    #Build credits general pie. 
    specs = [[{"type": "pie"},{"type": "pie"},{"type": "pie"} ]]
    subplot_titles=("Accepted bureau", "Accepted others","Types others")
    fig = make_subplots(rows=1, 
                          cols=3,
                          specs = specs,
                          subplot_titles =subplot_titles)
    
    #Draw figure.
    fig.add_trace(
        fig_go_1,row=1, col=1
    )
    fig.add_trace(
        fig_go_2,row=1, col=2
    )

    fig.add_trace(
        fig_go_3,row=1, col=3
    )

    fig.update_layout(height=375, width=850, title_text=title, title_x = 0.5,
                       font=dict(
                            size=13,
                            color="Black"
    ))
    
    return fig

#Defines a function that gives for a client, its general information interpretation (bars).
def interpretation_actual_credit_demand_client(id_client, X, title ="Client Credit History"):
    
    #Obtain plots.
    #general info.
    vars_bar_general_info = ['age','Expected_age_credit_payment','bureau_years_CREDIT_max_b', 'max_years_credit_pa']
    fig_1 = client_plotty_go_bar_plot(id_client, X,vars_bar_general_info)
    
    #Credits with bureau.
    vars_bar_b = ['AMT_CREDIT','bureau_AMT_CREDIT_SUM_mean_b','bureau_AMT_CREDIT_SUM_sum_b','bureau_AMT_CREDIT_SUM_DEBT_mean_b','bureau_AMT_CREDIT_SUM_DEBT_sum_b']
    fig_2 = client_plotty_go_bar_plot(id_client, X,vars_bar_b)
    
    #Credits with others.
    vars_bar_pp = ['AMT_CREDIT','AMT_CREDIT__mean_pa','AMT_ANNUITY__mean_pa','active_AMT_CREDIT__mean_pa','active_AMT_ANNUITY__mean_pa','active_AMT_CREDIT__sum_pa','active_AMT_ANNUITY__sum_pa']
    fig_3 = client_plotty_go_bar_plot(id_client, X,vars_bar_pp)
    
    #Credits reported days.
    vars_bar_reported_days = ['avg_dpd_b','total_reported_days_ccb','reported_days_sum_pi','total_reported_days_pcb','reported_days_ss']
    fig_4 = client_plotty_go_bar_plot(id_client, X,vars_bar_reported_days)
    
    #Credits reported delays.
    vars_bar_reported_delays = ['times_reported_delay_dues_ccb','times_reported_delay_dues_pi','times_reported_delay_dues_pcb']
    fig_5 = client_plotty_go_bar_plot(id_client, X,vars_bar_reported_delays)
    
    
    #Build figure.
    specs = [[{"type": "bar"},{"type": "bar"},{"type": "bar"}],[{"type": "bar"},{"type": "bar"},{"type": "bar"}]]
    subplot_titles=("General information", "Bureau","Others","Reported days", "Reported delays","")
    fig = make_subplots(rows=2, 
                          cols=3,
                          specs = specs,
                          subplot_titles =subplot_titles )

    fig.add_trace(
        fig_1,row=1, col=1
    )

    fig.add_trace(
        fig_2,row=1, col=2
    )
    fig.add_trace(
        fig_3,row=1, col=3
    )

    fig.add_trace(
        fig_4,row=2, col=1
    )
    fig.add_trace(
        fig_5,row=2, col=2
    )

    fig.update_layout(height=1200, width=800, title_text=title, title_x = 0.5,
                       font=dict(
                            size=13,
                            color="Black"
                            ),
                     showlegend=False)
    
    #To modify the axes ranges.
    #fig_1.update_yaxes(range=[0, 10], row=1, col=1)
    fig.update_yaxes(range=[0, 5000], row=2, col=1)
    fig.update_yaxes(range=[0, 20], row=2, col=2)
    fig.update_xaxes(tickangle=-45)
    return fig

#3.
#Obtain relevant dataframes for the clients interpretation based on neighbors.
def interpretation_obtain_dfs_neighbors_client(client_id, X, y, cols, n_neighbors):
    X_mod = X.copy()
    y_mod = y.copy()
    
    #Obtain df related to clients nearest neighbors.
    X_1, X_client = client_neighbors_given_attributes(client_id, X_mod, cols, n_neighbors = n_neighbors, return_normalized = False)
    y_1 = y_mod.loc[list(X_1.index),:]
    
    #Obtain df related to clients nearest neighbors for which credits was accepted.
    X_2, X_client, y_2, y_client = client_neighbors_given_attributes_and_target(client_id, X_mod, y, cols, n_neighbors = n_neighbors, value_y=1, return_normalized=False)
    
    #Obtain df related to clients nearest neighbors.
    X_3, X_client, y_3, y_client = client_neighbors_given_attributes_and_target(client_id, X_mod, y, cols, n_neighbors = n_neighbors, value_y=0, return_normalized=False)

    #Concatenate daframes with responses.
    df_1 = pd.concat([X_1,y_1],axis=1)
    df_2 = pd.concat([X_2,y_2],axis=1)
    df_3 = pd.concat([X_3,y_3],axis=1)
    df_4 = pd.concat([X_client,y_client],axis=1)
    
    #Return dictionary of dataframes.
    keys = ['neighbors','neigbors_y_accepted','neigbors_y_refused','client']
    list_dfs = [df_1, df_2,df_3,df_4] 
    dict_df = dict(zip(keys,list_dfs))
    return dict_df


#Define a method that allows to generate n different colors.
def obtain_list_n_different_colors(n_colors):
    colors = []
    n = n_colors
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors 


#Obtain from a dataframe of neighbors of a client the interpretation box plot of a variable.
def interpretation_variable_plotty_go_box_plot(df,df_key,variable, color ='steelblue'):
    fig = go.Box(y=df[variable],name=variable+'_'+df_key,marker_color=color)
    return fig

#Obtain from a dictionary of dataframes of interpreation the related traces.
def interpretation_obtain_variable_traces_from_dfs_neighbors_for_client(dict_df_client, var, color='steelblue'):
    traces = []
    #Obtain traces.
    for i in list(dict_df_client.keys()):
        trace = interpretation_variable_plotty_go_box_plot(dict_df_client[i],i,var,color)
        traces.append(trace)
    return traces[0], traces[1], traces[2], traces[3]

#Define a function that exhibits the interpretation of a client credit decision given some neighbors.
def interpretation_obtain_variable_traces_client(client_dict_dfs,cols,colors, title):
    colors_traces = colors[0:len(cols)]
    num_vars = len(cols)
    num_cols = 4
    num_rows = math.ceil(num_vars/num_cols)

    #Traces client,
    traces = {}
    
    #Update cols.
    #cols.append('TARGET')
    
    #Generate traces per variable.
    colors_cont = 0
    for i in cols:
        t1 , t2 ,t3 ,t4 = interpretation_obtain_variable_traces_from_dfs_neighbors_for_client(client_dict_dfs,i,colors_traces[colors_cont])
        traces[i] = [t1 , t2 ,t3 ,t4]
        colors_cont = colors_cont+1

    #Generate global figure.
    subplot_titles = tuple(cols)
    fig = make_subplots(rows=num_rows, cols=4,subplot_titles = subplot_titles)
    v=0
    for r in range(1,num_rows+1):
        for c in range(1, num_cols+1):
            cont = 0
            if(v<=(len(cols)-1)):
                while cont<4:
                    #print(cols[v]+'_'+str(r)+'_'+str(c))
                    fig.append_trace(traces[cols[v]][cont], row = r, col = c) 
                    cont=cont+1
            v=v+1   
    
    #Especify layout properties.
    fig.update_layout(height=390*num_rows, width=1050, title_text=title, title_x = 0.5,
                       font=dict(
                            size=9,
                            color="Black"
                            ),
                       showlegend=False)
    #fig.for_each_xaxis(lambda axis: axis.title.update(font=dict(color = 'black', size=10)))
    #fig.update_xaxes(tickangle=-90)
    return fig


#Define a function that exhibits the interpretation of a client credit decision given a number of neighbors.
def interpretation_from_neighbors_client(client_id, X_train, y_train, variables, n_clients, colors, title):
    client_dict_dfs = interpretation_obtain_dfs_neighbors_client(client_id, X_train, y_train, variables,n_clients)
    variables_ = variables.copy()
    #variables_.append('TARGET')
    fig = interpretation_obtain_variable_traces_client(client_dict_dfs,variables_,colors, title)
    return fig

#Defining methods helping the estimation of credit acceptance.
def get_client_row_from_df(id_client, df):
    #Copy.
    df_mod = df.copy()
    #Subsetting.
    df_mod = df_mod[df_mod.index==id_client]
    return df_mod


#Defining a method helping the estimation of credit acceptance.
def get_risk_classification_client(X_norm_client, classifier):
    #Get probability of acceptancce.
    desicion = classifier.predict(X_norm_client)[0]
    return desicion

#Defining methods helping the estimation of credit acceptance.
def get_default_probability_client(X_norm_client, classifier):
    #Get probability of acceptancce.
    prob_c = classifier.predict_proba(X_norm_client)[0,1]
    return prob_c

#Define a method to check if the client is in a training set.
def is_client_in_training_set(id_client, X_train):
    X_mod = X_train.copy()
    c = X_mod[X_mod.index==id_client]
    if c.empty:
        return(False)
    else:
        return(True)
    
#When the client doesn't belongs to the training set, he is added to the database for graphical proposals.
#Its y is the estimation through the classifier.
def update_client_in_X_train_after_prediction(id_client, X_total, X_total_original, X_train, y_train, X_train_original, classificator):
    X_mod = X_train.copy()
    X_total_mod = X_total.copy()
    y_mod = y_train.copy()
    condition = is_client_in_training_set(id_client, X_mod)
    X_mod_original = X_train_original.copy()
    X_total_original_mod = X_total_original.copy()
    
    #Validate.
    if (condition==False):
        
        #Add client to X_train.
        X_c = get_client_row_from_df(id_client, X_total_mod)
        X_mod = pd.concat([X_mod,X_c],axis=0)
        
        #Add client to X_train_original
        X_c_original = get_client_row_from_df(id_client, X_total_original_mod)
        X_mod_original = pd.concat([X_mod_original,X_c_original],axis=0)
        
        #get desicion.
        des_c = get_acceptance_client(get_client_row_from_df(id_client,X_total_mod),classificator)
        result = [des_c]
        y_new = pd.DataFrame(zip(result))
        y_new.columns = ['TARGET']
        y_new.index = [id_client]
        
        #Add client to y_train.
        y_mod = pd.concat([y_mod,y_new],axis=0)
    
    return X_mod, y_mod, X_mod_original

#Define functions that helps the update of the client information in the application.
#Define a function that recalculates a row given a pameter passed by parameter for a variable.
def recalculate_row(id_client, X, variable, val):
    #Obtain copy.
    X_mod = X.copy()
    
    #Store column value type.
    col_type = X_mod[variable].dtype
    
    #Obtain client row.
    c_row = get_client_row_from_df(id_client, X_mod)
        
    #Update value.
    if col_type == 'float':
        c_row[variable] = float(val)
    
    if col_type == 'int64':
        c_row[variable] = int(val)
    
    return c_row

#Define a function that replace a row in a dataframe by one passed by parameter. 
def replace_row_in_X(id_client, new_row, X):
    X_mod = X.copy()
    
    #Delete ancien row.
    X_mod = X_mod[X_mod.index!=id_client]
    
    #Append new row.
    X_mod = pd.concat([X_mod,new_row],axis=0)
    
    return X_mod

#Define a function that recalculates a row and a dataframe by parameters passed argument.
def recalculate_row_and_X(id_client, X, variable, val):
    #Create copy.
    X_mod = X.copy()
    
    #Calculate new row.
    new_row = recalculate_row(id_client, X, variable, val)
    
    #Calculate new X/
    X_mod = replace_row_in_X(id_client, new_row, X_mod)
    
    return new_row, X_mod

#Function that for a new row, calcualtes the probability of default.
def get_default_probability_client_from_new_row(c_new_row, scaler, enc, classifier):
    c_new_row_proccessed = X_test_normalize_and_encode_variables(c_new_row, scaler, enc)
    prob = get_default_probability_client(c_new_row_proccessed,classifier)
    return prob


#Defines a function that checks if an elements in a list are none or not.
def object_is_not_none_list(ls_items):
    b = []
    for i in ls_items:
        if (i is None):
            b.append(0)
        else:
            b.append(1)
    return b