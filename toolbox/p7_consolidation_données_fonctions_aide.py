import pandas as pd
#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))
#Defines a function that chunks a list.
def chunks(lst, size):
    new_list = [lst[i:i + size] for i in range(0, len(lst), size)]
    return new_list

#Defines a function that lists the name of files in a repository.
def get_files_names_in_repo(path):
    files_names = []
    for n_dir, _, file in os.walk(path_data):
        files_names = file
    files_names = sorted(files_names)    
    return files_names

#Check the missing values in columns within a pandas dataframe.
def cal_nan_percents(df):
    percent_missing = df.isnull().sum() / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})
    percent_missing = percent_missing.to_frame()
    percent_missing.columns = ["Nan_percent"]
    percent_missing =  percent_missing.reset_index(drop=True)
    columns = pd.DataFrame(df.columns,columns=['columns_names'])
    df_nan_percents = pd.concat([columns,percent_missing],axis=1)
    return df_nan_percents

#Define a function that gives the most important information about a df.
def give_df_info(df):
    types = pd.Series(df.dtypes).to_frame()
    types.columns = ["Data_types"] 
     
    percent_missing = df.isnull().sum() / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})
    percent_missing = pd.Series(percent_missing).to_frame()
    percent_missing.columns = ["Nan_percent"]
    
    uniqueValues = pd.Series(df.nunique()).to_frame()
    uniqueValues.columns = ["Unique_values"]
    
    #Extracts the variables names to add at the final dataframe.
    variables_names = list(types.index)
    variables_names = pd.DataFrame(variables_names)
    variables_names.columns = ["Variable_names"]
    
    #Aggregate dataframe with information values.
    df_info = pd.concat([types, percent_missing, uniqueValues], axis=1)

    #Creating the dataframe containing all the information of the edStatsCountry dataframe.
    Unique_values = []

    for index, row in df_info.iterrows():
        data_type = row['Data_types']
        Unique_value = row['Unique_values']
    
        if(data_type=="float64"):
            Unique_value = "real_variable"
        Unique_values.append(Unique_value)
    
    Unique_values = pd.DataFrame({'Unique_values':Unique_values})
    Unique_values = Unique_values.reset_index()
    Unique_values = Unique_values.drop(['index'], axis=1)

    df_info = df_info.drop(['Unique_values'], axis=1)
    df_info.reset_index(inplace=True)
    df_info = df_info.drop(['index'], axis=1)
    df_info = pd.concat([variables_names, df_info, Unique_values], axis=1)
    
    return df_info

#Defines a function that for a dataframe, return its dataypes counts.
def give_df_datatypes_counts(df):
    description_datatypes = df.dtypes.value_counts()
    return description_datatypes

#Defines a function that returns a dataframe of missing values (total and percent).
def df_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()*100)\
        .sort_values(ascending=False)
    return pd.concat([total, percent],
                     axis=1,
                     keys=['Total', 'Percent'])

#Define a function that depicts for every dataset, the percent of missing values in its columns.
def plot_dfs_percent_missings_in_columns(dict_df):
    #First stage.
    df_missing = pd.DataFrame()
    for n_df, df in dict_df.items():
        df_missing = pd.concat([df_missing,
                            df_missing_data(df)['Percent'].to_frame(name=n_df)],
                           axis=1)
    #Second stage.
    #Rearrange the ratio of missing values in each columns
    df_missing_pct = df_missing.T.stack().reset_index()
    df_missing_pct.columns = ['data_table', 'column', 'missing_pct']
    
    #Third stage.
    sns.stripplot(x='data_table', y='missing_pct', data=df_missing_pct,
                  linewidth=2, alpha = 0.5, size=7)
    plt.gcf().set_size_inches(7,4.5)
    plt.xticks(rotation=90, ha='right')
    ax = plt.gca()
    ax.set_ylabel('Missing value percent (%)')
    ax.set_xlabel('Files (datasets)')
    ax.set_title('Missing values in columns of each file',
                 fontweight='bold', pad=15)
    
#Defining functions to check the homogeneity between two dataframes.
#Define a function that subsets a dataframe based on a list of columns.
def df_subset_by_list_columns(df, list_cols):
    df = df.copy()
    df = df[list_cols]
    return df

#Define a function that gives the columns names of a dataframe.
def df_give_cols_names(df):
    column_names = sorted(df.columns.tolist())   
    return column_names

#Define a function that gives columns' names for a dataframe.
def df_give_object_cols_names(df):
    df = df.copy()
    df = df.select_dtypes(include=['object','bool']) #Not included int64
    column_names = sorted(df.columns.tolist())   
    return column_names

#Define a function that checks whether two dataframes have the same columns or not.
#If there are differents columns between them, it could shows the names of them.
def df1_has_same_cols_df2(df1,df2, return_dif_cols=False):
    col1_names = df_give_cols_names(df1)
    col2_names = df_give_cols_names(df2)
    cond = False
    #Comparison.
    if(col1_names==col2_names):
        cond=True
    else:
        diff1 = set(col1_names) - set(col2_names)
        diff2 = set(col2_names) - set(col1_names)
        diff = list(diff1) + list(diff2)
        diff = sorted(diff)
    #Return.
    if(return_dif_cols==True):
        if(cond==True):
            _ = ""
            return(cond, _)
        else:
            return(cond,diff)
    else:
            return cond
        
# Defines a function that for a client give common credits between two databases.     
def give_for_common_client_in_dfs_there_are_common_credits(client, df1, df2, col_credits, cols_clients, give_values = True):
    #Filter dfs.
    df1_mod = df1.copy()
    df2_mod = df2.copy()
    
    client_credits_1 = give_from_owner_its_credits_in_df(client, df1_mod, col_credits, cols_clients)
    client_credits_2 = give_from_owner_its_credits_in_df(client, df2_mod, col_credits, cols_clients)
    client_credits_1_and_2 = lists_common_values(client_credits_1,client_credits_2)
    there_are_commons = False
    if client_credits_1_and_2:
        there_are_commons = True
    if(give_values==True):
        return there_are_commons, client_credits_1_and_2
    else:
        return there_are_commons

#Defines a function removes nan values from two pair of lists.
def filter_common_credits_per_client(common_clients,common_credits_per_client):
    a = common_clients
    b = common_credits_per_client
    df = pd.DataFrame(list(zip(a,b)))
    df.columns = ["0","1"]
    df = df[df['1'].notna()]
    a = df["0"].tolist()
    b = df["1"].tolist()
    return a, b

#Defines a function that for common clients between two dataframes, returns clients having same credits in both dataframes.
def give_for_common_clients_in_dfs_common_credits(list_common,df1,df2,col_credits,cols_clients):
    #Filter dataframes.
    df1_mod = df1.copy()
    df1_mod = df1[df1[cols_clients].isin(list_common)]

    df2_mod = df2.copy()
    df2_mod = df2[df2[cols_clients].isin(list_common)]
    
    common_credits_per_client = []
    #Calculate for each client
    #print(len(list_common))
    #d = 0
    for i in list_common:
        #print(d)
        cond, vals = give_for_common_client_in_dfs_there_are_common_credits(i, df1_mod, df2_mod, col_credits, cols_clients, True)
        if(cond==True):
            common_credits_per_client.append(vals)
        else:
            common_credits_per_client.append(np.nan)
            
        #uddate sate of dfs.
        df1_mod = df1_mod[~df1_mod[cols_clients].isin([i])]
        df2_mod = df2_mod[~df2_mod[cols_clients].isin([i])]
    
    cmon_clients, cmon_cre_clients = filter_common_credits_per_client(list_common,common_credits_per_client)
            
    return cmon_clients, cmon_cre_clients

#Defines a function that for common clients between two dataframes, returns clients having same credits in both dataframes.
def give_for_common_clients_in_dfs_common_credits_in_steps(list_common,df1,df2,col_credits,cols_clients, n_vals):
    #Creating objects.
    lst = list_common.copy()
    full_a = []
    full_b = []
    #Chuncking.
    lst_chunks = chunks(lst,n_vals)
    
    for i in lst_chunks:
        a, b = give_for_common_clients_in_dfs_common_credits(i,df1,df2,col_credits,cols_clients)
        full_a = full_a+a
        full_b = full_b+b
    
    return full_a, full_b

#Defines a function that checks if there are common credits between two dataframes.
def there_are_common_credits_in_dfs_with_common_clients(list_common,df1,df2,col_credits,cols_clients, return_vals=False):
    cmon_clients, cmon_cre_clients = give_for_common_clients_in_dfs_common_credits_in_steps(list_common,df1,df2,col_credits,cols_clients,300)
    cond = False
    if(cmon_clients):
            cond = True
    
    if(return_vals==False):
        return cond
    else:
        dict_clients_credits = dict(zip(cmon_clients, cmon_cre_clients))
        if(cond==True):
            return cond, dict_clients_credits
        else:
            _ = ""
            return cond, _

#Defines a function that for a client subsets a dataframe.      
def df_subset_by_client(df, col_client, client):
    df = df.copy()
    #Subset by client.
    df = df[df[col_client].isin([client])]
    return df

#Defines a function that for a list of clients subsets a dataframe.      
def df_subset_by_list_clients(df, col_client, list_clients):
    df = df.copy()
    #Subset by client.
    df = df[df[col_client].isin(list_clients)]
    return df
        
#Defines a function that for a client and credit, subsets a dataframe.      
def df_subset_by_client_and_credit(df, col_credit, col_client, credit, client):
    df = df.copy()
    #Subset by client.
    df = df[df[col_client].isin([client])]
    #Subset by credit.
    df = df[df[col_credit].isin([credit])]
    return df
        
#Define a function that gives unique values for a column,
def col_give_unique_values(col):
    col_mod = col.copy()
    unique_vals = col_mod.unique()
    try:
        unique_vals = sorted(unique_vals)
    except :
        unique_vals = list(unique_vals)
        
    return(unique_vals)

#Defines a function that from two lists, returns a new list with different objects.
def calcualte_diff_vals_lists(l1, l2):
    diff1 = set(l1) - set(l2)
    diff2 = set(l2) - set(l1)
    diff = list(diff1) + list(diff2)
    diff = sorted(diff)
    return diff

#Defines a function that from two lists, returns a new list with common values between them.
def lists_common_values(l1,l2):
    a = sorted(l1)
    b = sorted(l2)
    #find common values in lists.
    common_vals = sorted(list(set(a).intersection(b)))
    return common_vals  

#Defines a function that gives for two columns, their common values.
def dfs_common_values_cols(col1, col2):
    a = sorted(col_give_unique_values(col1))
    b = sorted(col_give_unique_values(col2))
    #find common cols
    common_vals = sorted(list(set(a).intersection(b)))
    return common_vals

#Define a function that checks if all unique values in one categorical column are presented in another one.
def col1_unique_values_contain_col2_unique_values(col1, col2, return_not_common_values=False):
    #capture values.
    uni_1 = col_give_unique_values(col1)
    uni_2 = col_give_unique_values(col2)
    #Condition.
    check =  all(item in uni_1 for item in uni_2)
    #return
    if(check==False):
        diff = sorted(list(set(uni_2) - set(uni_1)))
        if(return_not_common_values==True):
            return check, diff
        else:
            return check
    else:
        if(return_not_common_values==True): 
            _ =""
            return check, _
        else:
            return check
    
#Define a function that checks similar columns between two dataframes.
#If there are common columns, it could shows them.
def dfs_have_common_cols(df1, df2, return_common_cols=False):
    cols1 = sorted(df1.columns.tolist()) 
    cols2 = sorted(df2.columns.tolist()) 
    cond = False
    #find common cols
    common_cols = sorted(list(set(cols1).intersection(cols2))) 
    #condition
    if(return_common_cols ==False):
        if not common_cols:
            return cond
        else:
            cond=True
            return cond
    else:
        if not common_cols:
            cond=False
            return cond, _
        else:
            cond=True
            return cond, common_cols
        
#Define a function that deletes rows in a dataframe in which a column matches certain values passed by list.
def df_removing_rows_based_on_list_values(df,col,list_not_common_values):
    df_mod = df
    #Filtering by list of values. 
    df_mod = df_mod[~df_mod[col].isin(list_not_common_values)]
    df_mod = df_mod.reset_index(drop=True)
    #return.
    return(df_mod)

#Define a function that checks if a dataframe column values contains the columns names of another dataframe.
def df1_col_vals_contains_df2_cols_names(df, col, df2, return_not_common_values=False):
    #capture values.
    df_col_vals = sorted(list(df[col].unique()))
    cols_2 = df_give_cols_names(df2)
    #Condition.
    check =  all(item in df_col_vals for item in cols_2)
    #return
    if(check==False):
        diff = sorted(list(set(cols_2) - set(df_col_vals)))
        if(return_not_common_values==True):
            return check, diff
        else:
            return check
    else:
        if(return_not_common_values==True): 
            _ =""
            return check, _
        else:
            return check
        
#Defines a function that check wheter a variable is contained in a dataframe.
def df_contains_variable(df, var_name):
    names = df_give_cols_names(df)
    if(var_name in names):
        return True
    else:
        return False

#Define a function that from common columns between two dataframes get common objective columns.
def dfs_obtain_common_categorical_columns(df1,df2, common_columns):
    df1 = df1.copy()
    df2 = df2.copy()
    #Subset by common columns.
    a = df_subset_by_list_columns(df1,common_columns)
    b = df_subset_by_list_columns(df2,common_columns)

    #Subset by object columns.
    cat_cols_a = df_give_object_cols_names(a)
    cat_cols_b = df_give_object_cols_names(b)
    a = df_subset_by_list_columns(a, cat_cols_a)
    b = df_subset_by_list_columns(b, cat_cols_b)
    cat_cols_a = df_give_object_cols_names(a)
    cat_cols_b = df_give_object_cols_names(b)
    #Return.
    if(cat_cols_a==cat_cols_b):
        comment = "Common object columns where found correctly"
        return comment, cat_cols_a
    else:
        _ = ""
        comment = "Common object columns where found correctly"
        return comment, _(df1,df2, common_columns)
    
#Defines a function that gives for a credit, its owner.
def give_from_credit_its_owner_in_df(credit, df, col_credits, col_clients):
    df_mod = df.copy()
    df_mod = df_mod[df_mod[col_credits].isin([credit])]
    df_mod = df_mod[col_clients]
    val = df_mod.values[0]
    return val

#Defines a function that gives for a person, the credits its has/had openned.
def give_from_owner_its_credits_in_df(owner, df, col_credits, col_clients):
    df_mod = df.copy()
    df_mod = df_mod[df_mod[col_clients].isin([owner])]
    df_mod = df_mod[col_credits]
    vals = sorted(df_mod.unique())
    return vals

#Defines a function that gives by argument a list of credit, and returns a list of owners.
def give_from_list_of_credits_list_of_owners(list_credits, df,col_credits, col_clients ):
    df_mod = df.copy()
    owners = []
    for i in list_credits:
        owner_i = give_from_credit_its_owner_in_df(i,df_mod,col_credits,col_clients)
        owners.append(owner_i)
    return owners

#Defines a function that gives for a person its credit history as a dictionary.
def give_from_owner_its_credits_information(owner, df_office, df_office_balance, col_credits, col_clients, print_results=False):
    #Create df copy.
    df_mod_1 = df_office.copy()
    #Obtain credits from user.
    credits = give_from_owner_its_credits_in_df(owner,df_mod_1,col_credits,col_clients)
    
    #Capture for each credit its info.
    df_mod_2 = df_office_balance.copy()
    l = []
    summary_statuses = []
    for i in  credits:
        df_mod_i = df_mod_2[df_mod_2[col_credits].isin([str(i)])]
        l_i = df_mod_i['STATUS'].value_counts().sum()
        summary_statuses_i = df_mod_i["STATUS"].value_counts()
        l.append(l_i)
        summary_statuses.append(summary_statuses_i)
    #Return results a dictionary.
    dict_owner = dict(zip(credits, zip(l, summary_statuses)))
    
    if(print_results == True):
        print("Client "+"'"+str(owner)+"' credit history information")
        for i in credits:
            print("")
            print("Credit number : "+str(i)+', lenght(months) : '+str(dict_owner[i][0]))
            print("Credit statuses information : ")
            print(str(dict_owner[i][1]))
    return dict_owner


#Defines functions for dataframes sharing common columns for clients and credits.
def give_for_common_client_in_dfs_there_are_common_credits(client, df1, df2, col_credits, cols_clients, give_values = True):
    client_credits_1 = give_from_owner_its_credits_in_df(client, df1, col_credits, cols_clients)
    client_credits_2 = give_from_owner_its_credits_in_df(client, df2, col_credits, cols_clients)
    client_credits_1_and_2 = lists_common_values(client_credits_1,client_credits_2)
    there_are_commons = False
    if client_credits_1_and_2:
        there_are_commons = True
        
    if(give_values==True):
        return there_are_commons, client_credits_1_and_2
    else:
        return there_are_commons

#Definition of functions found within the bibliography of credit risk assignment.
#Funcions taken from web exploration.
def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical

#Defines a function that works over categorical columns.
def count_categorical(df_office, group_var, df_name):
    df = df_office.copy()
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical

#Deploying modifications to 'office', and 'office_balance columns'.
#Returns a dataframe having one row for each client with statistics calculated for all his loans along monthly balance information.
def obtain_df_office_balance_by_client(df_office_balance, df_office, col_credits, col_clients, office_balance_name):
    office_balance_counts = count_categorical(df_office_balance, group_var = col_credits, df_name = office_balance_name)
    office_balance_agg = agg_numeric(df_office_balance, group_var = col_credits, df_name = office_balance_name)
    office_by_loan = office_balance_agg.merge(office_balance_counts, right_index = True, left_on = col_credits, how = 'outer')
    office_by_loan = office_by_loan.merge(df_office[[col_credits, col_clients]], on = col_credits, how = 'left')
    office_balance_by_client = agg_numeric(office_by_loan.drop(columns = [col_credits]), group_var = col_clients, df_name = 'client')
    #client_bureau_balance_MONTHS_BALANCE_count_max, client_bureau_balance_MONTHS_BALANCE_count_min
    office_balance_by_client = office_balance_by_client[['SK_ID_CURR','client_office_balance_MONTHS_BALANCE_count_max','client_office_balance_MONTHS_BALANCE_count_min','client_office_balance_MONTHS_BALANCE_count_sum','client_office_balance_STATUS_0_count_sum','client_office_balance_STATUS_1_count_sum','client_office_balance_STATUS_2_count_sum','client_office_balance_STATUS_3_count_sum','client_office_balance_STATUS_4_count_sum','client_office_balance_STATUS_5_count_sum','client_office_balance_STATUS_X_count_sum','client_office_balance_STATUS_C_count_sum']]
    office_balance_by_client.columns = ['SK_ID_CURR','max_months_credit_b','min_months_credit_b','total_months_credit_b','times_status_0','times_status_1','times_status_2','times_status_3','times_status_4','times_status_5', 'times_status_X', "times_status_C"]
    
    return office_balance_by_client


#Define functions to perform feature engineering from train and test datasets.
#Calculate ratio column.
def application_train_test_calculate_ratio_column(df, col1 ='DAYS_EMPLOYED', col2 ='DAYS_BIRTH', col_result ='started_working_early'):
    df = df.copy()
    df[col_result] = df[col1] / df[col2]
    return df

#Calculate count of family members.
def application_train_test_calculate_count_family_menbers(df, col1 ='CNT_FAM_MEMBERS', col2 ='CNT_CHILDREN'):
    df = df.copy()
    df = df.assign(CNT_FAM_MEMBERS_NOT_CHILD=lambda x: x[col1]-x[col2]).drop(columns=col1)
    return df

#Define a function that updates the "application_train", and "application_test" dataframes given extracted variables from other dataframes.
def merge_train_or_test_values_with_new_extracted_features(df_train_or_test, df_selected_variables):
    #Create copy.
    df_mod = df_train_or_test.copy()
    df_selected_variables_mod = df_selected_variables.copy()
    #Merging variables with 'application_train', and 'application_test'.
    df_merged = df_mod.merge(df_selected_variables_mod, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
    #Filling NaN values.
    cols = list(df_selected_variables_mod.columns)
    cols.remove('SK_ID_CURR')
    df_part_1 = df_merged[list(df_mod.columns)]
    df_part_2 = df_merged[cols].fillna(0)
    #concatenate.
    df_merged = pd.concat([df_part_1,df_part_2],axis=1)                      
    return df_merged

#Define a function that calcualtes features based on 'train' and 'test' dataframes.
def df_train_test_calculate_new_columns(df):
    df_mod = df.copy()
    df_mod = application_train_test_calculate_ratio_column(df_mod)
    df_mod = application_train_test_calculate_ratio_column(df_mod,'AMT_CREDIT','AMT_INCOME_TOTAL','RATIO_CREDIT_INCOME')
    df_mod = application_train_test_calculate_ratio_column(df_mod,'AMT_INCOME_TOTAL','CNT_FAM_MEMBERS','INCOME_PER_PERSON')
    df_mod = application_train_test_calculate_ratio_column(df_mod,'AMT_ANNUITY','AMT_INCOME_TOTAL','ANNUITY_INCOME_PERC')
    df_mod = application_train_test_calculate_ratio_column(df_mod,'AMT_ANNUITY','AMT_CREDIT','PAYMENT_RATE')
    df_mod = application_train_test_calculate_count_family_menbers(df_mod)
    return df_mod

#Define a function that merges variables from a dictionary of varibles, with a 'training' or 'test' set.
def df_train_test_calculate_new_columns_based_on_selected_variables(df, dict_selected_variables):
    df_mod = df.copy()
    for i in list(dict_selected_variables.keys()):
        df_mod = merge_train_or_test_values_with_new_extracted_features(df_mod,dict_selected_variables[i])
    return df_mod

#Defining function to prepare data before the prediction part.
def X_calculate_and_adjust_some_variables(X):
    try:
        #reported_days_ss.
        reported_days_dpd_ss = X['DEF_30_CNT_SOCIAL_CIRCLE']*15 +X['DEF_30_CNT_SOCIAL_CIRCLE']*30
        reported_days_dpd_ss = pd.DataFrame(reported_days_dpd_ss)
        reported_days_dpd_ss.columns = ['reported_days_ss']    
        X['reported_days_ss'] = reported_days_dpd_ss['reported_days_ss']

        #bureau_DAYS_CREDIT_sum_b.
        bureau_years_CREDIT_sum_b = (X['bureau_DAYS_CREDIT_sum_b']*-1)/365
        bureau_years_CREDIT_sum_b = pd.DataFrame(bureau_years_CREDIT_sum_b)
        bureau_years_CREDIT_sum_b.columns = ['bureau_years_CREDIT_sum_b']
        X['bureau_years_CREDIT_sum_b'] = bureau_years_CREDIT_sum_b['bureau_years_CREDIT_sum_b']
        X = X.drop(['bureau_DAYS_CREDIT_sum_b'], axis=1)

        #bureau_DAYS_CREDIT_min_b.
        bureau_years_CREDIT_max_b = (X['bureau_DAYS_CREDIT_min_b']*-1)/365
        bureau_years_CREDIT_max_b = pd.DataFrame(bureau_years_CREDIT_max_b)
        bureau_years_CREDIT_max_b.columns = ['bureau_years_CREDIT_max_b']
        X['bureau_years_CREDIT_max_b'] = bureau_years_CREDIT_max_b['bureau_years_CREDIT_max_b']
        X = X.drop(['bureau_DAYS_CREDIT_min_b'], axis=1)

        #DAYS_BIRTH.
        age = (X['DAYS_BIRTH']*-1)/365
        age = pd.DataFrame(age)
        age.columns = ['age']
        X['age'] = age['age']
        X = X.drop(['DAYS_BIRTH'], axis=1)

        #DAYS_EMPLOYED
        years_since_last_job = (-X['DAYS_EMPLOYED'])/365
        years_since_last_job = pd.DataFrame(years_since_last_job)
        years_since_last_job.columns = ['years_since_last_job']    
        X['years_since_last_job'] = years_since_last_job['years_since_last_job']
        X['years_since_last_job']= X['years_since_last_job'].map(lambda x: 0 if x<=0 else x)
        X = X.drop(['DAYS_EMPLOYED'], axis=1)

        #Calculate client's last job lenght ratio.
        lenght_last_job_time_over_lenght_life = (X['years_since_last_job'])/(X['age'])
        lenght_last_job_time_over_lenght_life = pd.DataFrame(lenght_last_job_time_over_lenght_life)
        lenght_last_job_time_over_lenght_life.columns = ['lenght_last_job_time_over_lenght_life']    
        X['lenght_last_job_time_over_lenght_life'] = lenght_last_job_time_over_lenght_life['lenght_last_job_time_over_lenght_life']

        #max_months_credit_b
        max_years_credit_b = (X['max_months_credit_b'])/12
        max_years_credit_b = pd.DataFrame(max_years_credit_b)
        max_years_credit_b.columns = ['max_years_credit_b']
        X['max_years_credit_b'] = max_years_credit_b['max_years_credit_b']
        X = X.drop(['max_months_credit_b'], axis=1)

        #total_months_credit_b
        total_years_credit_b = (X['total_months_credit_b'])/12
        total_years_credit_b = pd.DataFrame(total_years_credit_b)
        total_years_credit_b.columns = ['total_years_credit_b']
        X['total_years_credit_b'] = total_years_credit_b['total_years_credit_b']
        X = X.drop(['total_months_credit_b'], axis=1)

        #'total_lenght_days__max_pa'
        max_years_credit_pa = (X['total_lenght_days__max_pa'])/365
        max_years_credit_pa = pd.DataFrame(max_years_credit_pa)
        max_years_credit_pa.columns = ['max_years_credit_pa']
        X['max_years_credit_pa'] = max_years_credit_pa['max_years_credit_pa']
        X = X.drop(['total_lenght_days__max_pa'], axis=1)

        #'total_lenght_days__sum_pa'
        total_years_credit_pa = (X['total_lenght_days__sum_pa'])/365
        total_years_credit_pa = pd.DataFrame(total_years_credit_pa)
        total_years_credit_pa.columns = ['total_years_credit_pa']
        X['total_years_credit_pa'] = total_years_credit_pa['total_years_credit_pa']
        X = X.drop(['total_lenght_days__sum_pa'], axis=1)

        #'total_lenght_days__sum_pa'
        mean_years_credit_pa = (X['total_lenght_days__mean_pa'])/365
        mean_years_credit_pa = pd.DataFrame(mean_years_credit_pa)
        mean_years_credit_pa.columns = ['mean_years_credit_pa']
        X['mean_years_credit_pa'] = mean_years_credit_pa['mean_years_credit_pa']
        X = X.drop(['total_lenght_days__mean_pa'], axis=1)

        #Expected credit payment.
        Expected_age_credit_payment = ((X['AMT_CREDIT']/X['AMT_ANNUITY']) + X['age'])
        Expected_age_credit_payment = pd.DataFrame(Expected_age_credit_payment)
        Expected_age_credit_payment.columns = ['Expected_age_credit_payment']    
        X['Expected_age_credit_payment'] = Expected_age_credit_payment['Expected_age_credit_payment']

        #Payment overpass life expectancy.
        mask = X['Expected_age_credit_payment']>82.58
        mask = pd.DataFrame(mask)
        mask.columns = ['condition']
        X['Expected_age_credit_completion_overpass_life_expectancy'] = mask['condition'].map(lambda x: 1 if x==True else 0)
        
        #Ratio credit-annuity.
        RATIO_CREDIT_ANNUITY = (X['AMT_CREDIT']/X['AMT_ANNUITY'])
        RATIO_CREDIT_ANNUITY = pd.DataFrame(RATIO_CREDIT_ANNUITY)
        RATIO_CREDIT_ANNUITY.columns = ['RATIO_CREDIT_ANNUITY']    
        X['RATIO_CREDIT_ANNUITY'] = RATIO_CREDIT_ANNUITY['RATIO_CREDIT_ANNUITY']
        
        #Total credits closed in previous application.
        total_credits_closed_pa = X['total_credits_accepted_pa']- X['is_active__sum_pa']
        total_credits_closed_pa = pd.DataFrame(total_credits_closed_pa)
        total_credits_closed_pa.columns = ['total_credits_closed_pa']    
        X['total_credits_closed_pa'] = total_credits_closed_pa['total_credits_closed_pa']
    
    except:
        print("Error trying to modify variabless")

    return X
