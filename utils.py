import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extractRows(df,appear_times,columns,larger=True):
    # given the output that
    #     for the given columns, the num of certain pair appears more/less than "appear_times". Delete the others.
    # e.g. a = extractRows(name_out,5,columns = ['Name','ChargeOff'],larger=True)
    #            will select the ['Name','ChargeOff'] pair that appears more than 5 times.
    a = df[columns].value_counts()
    if larger:
        a_t = a[a>appear_times].index
    else:
        a_t = a[a<appear_times].index
    a_t = pd.DataFrame(a_t,columns=None)[0]
    name_list = []

    for item in a_t:
        name_list.append([item[i] for i in range(len(item))])
    df_cp = df.copy()
    drop_list = []
    for idx,item in enumerate(df_cp[columns].values):
        if list(item) in name_list:
            continue
        else:
            drop_list.append(idx)
    df_cp.drop(drop_list,inplace=True)

    return df_cp

def find_missing_zero(df_column):
    # find the missing/zero value
    nan_values_row_idx = []
    zero_values_row_idx = []
    for i,item in enumerate(df_column):
        if not item:
            zero_values_row_idx.append(i)
        elif isinstance(item,str):
            continue
        elif math.isnan(item):
            nan_values_row_idx.append(i)
    nan_values_row_idx = list(set(nan_values_row_idx))
    zero_values_row_idx = list(set(zero_values_row_idx))
    # BankStates = BankStates.drop(index=nan_values_row_idx)
    print(nan_values_row_idx)
    print(zero_values_row_idx)
    return nan_values_row_idx,zero_values_row_idx

def plotPassRate(out_df,num_values,show_nodes=200):
    y1 = np.ones(num_values)
    y2 = np.ones(num_values)
    for item in out_df.values:
        y1[item[1]]+=item[2]
        y2[item[1]]+=1
    passRate = []
    for i,j in zip(y1,y2):
        passRate.append(i/j)
    # PassRate of different term
    x = range(0,num_values) # I do norm, so it will larger than 1
    from scipy.interpolate import make_interp_spline
    xnew = np.linspace(min(x),max(x),show_nodes) #300 represents number of points to make between T.min and T.max
    power_smooth = make_interp_spline(x,passRate)(xnew)
    plt.plot(xnew,power_smooth)
    plt.show()
    #     # number of different term
    #     plt.plot(x,y2)
    #     plt.show()
    return y1,y2

def blacklistModify(input_data,input_out,thresold,column_name,npy_name,larger=True):
    a_t = extractRows(input_out,thresold,columns = [column_name],larger=larger)
    blanklist = list(set(list(a_t[column_name])))
    np.save(npy_name,np.array(blanklist))
    def foo(x):
        if x in blanklist:
            return 0
        else:
            return 1
    input_data[column_name] = input_data[column_name].map(lambda x: foo(x))

def whitelistModify(input_data,input_out,thresold,column_name,npy_name,larger=True):
    a_t = extractRows(input_out,thresold,columns = [column_name],larger=larger)
    whitelist = list(set(list(a_t[column_name])))
    np.save(npy_name,np.array(whitelist))
    def foo(x):
        if x in whitelist:
            return 1
        else:
            return 0
    input_data[column_name] = input_data[column_name].map(lambda x: foo(x))

def preprocess_str_extend(Name_df,familiar_names,column):
    length = len(familiar_names)
    def foo(x):
        for idx,familiar_name in enumerate(familiar_names): # trick reverse order to speed up
            if x in familiar_name:
                return length-idx-1
        return 0
    Name_df[column] = Name_df[column].map(lambda x: foo(x))
    return Name_df


def fooNAICS(x):
    if x:
        return 1
    else:
        return 0

blanklist = np.load("term_blackList.npy")
def fooTerm(x):
    if x in blanklist:
        return 0
    else:
        return 1

def fooApprovalFYs(x):
    if x[-1] == 'A':
        x = x[:-1]
    x = int(x)
    # I want it to be 0~n, not resize it 0-1 although it is continus. It will be considered later.
    return x-1969

def fooNoEmps(x):
    if x >100:
        return 2
    elif x>10:
        return 1
    else:
        return 0

def fooNewExists(x):
    if x==1.0:
        return 1
    else:
        return 0

def foo_createJobs(x):
    if x>80:
        return 2
    elif x>10:
        return 1
    else:
        return 0

def fooRetainedJobs(x):
    if x>=80:
        return 1.0
    else:
        return x/80

FranchiseCodesIdx = np.load("FranchiseCodesIdx.npy")
def fooFranchiseCodesIdx(x):
    if x in FranchiseCodesIdx:
        return list(FranchiseCodesIdx).index(x)+1
    else:
        return 0


def fooRevLineCr(x):
    if x=='T' or x=='Y':
        return 1
    else:
        return 0


def fooLowDoc(x):
    if x=='Y':
        return 1
    else:
        return 0

def fooDisbursementDate(x,np_index,values):
    avg = np.average(values)
    if not x == x:
        return avg
    return values[list(np_index).index(x)]