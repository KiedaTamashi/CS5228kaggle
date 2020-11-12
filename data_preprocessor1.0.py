import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb
import math
from copy import deepcopy
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode

from utils import *

####### modify this. #########
Ytrain_path = "./Ytrain.csv"



def preprocess_all(X_path , out_csv_name):
    Xtrain = pd.read_csv(X_path)

    # each column
    Names = Xtrain[['Id','Name']].copy()
    Citys = Xtrain[['Id','City']].copy()
    States= Xtrain[['Id','State']].copy()
    Zips = Xtrain[['Id','Zip']].copy()
    Banks = Xtrain[['Id','Bank']].copy()
    BankStates = Xtrain[['Id','BankState']].copy()
    NAICSs = Xtrain[['Id','NAICS']].copy()
    ApprovalDates = Xtrain[['Id','ApprovalDate']].copy()
    ApprovalFYs = Xtrain[['Id','ApprovalFY']].copy()
    Terms = Xtrain[['Id','Term']].copy()
    NoEmps = Xtrain[['Id','NoEmp']].copy()
    NewExists = Xtrain[['Id','NewExist']].copy()
    CreateJobs = Xtrain[['Id','CreateJob']].copy()
    RetainedJobs = Xtrain[['Id','RetainedJob']].copy()
    FranchiseCodes = Xtrain[['Id','FranchiseCode']].copy()
    UrbanRurals = Xtrain[['Id','UrbanRural']].copy()
    RevLineCrs = Xtrain[['Id','RevLineCr']].copy()
    LowDocs = Xtrain[['Id','LowDoc']].copy()
    DisbursementDates = Xtrain[['Id','DisbursementDate']].copy()
    DisbursementGrosss = Xtrain[['Id','DisbursementGross']].copy()
    BalanceGrosss = Xtrain[['Id','BalanceGross']].copy()
    GrAppvs = Xtrain[['Id','GrAppv']].copy()
    SBA_Appvs = Xtrain[['Id','SBA_Appv']].copy()


    # Names
    familiar_name = np.load("familiar_name.npy")
    def preprocess_str(Name_df,familiar_name,column):
        def foo(x):
            if x in familiar_name:
                return 1
            else:
                return 0
        Name_df[column] = Name_df[column].map(lambda x: foo(x))
        return Name_df
    Names = preprocess_str(Names,familiar_name,'Name')


    # Citys,States,Zips
    # We will not have address info according to the analysis in notebook.

    # Bank,Bank State
    familiar_banks = list(np.load("familiar_bank.npy",allow_pickle=True))
    Banks = preprocess_str_extend(Banks,familiar_banks,'Bank')

    # here we consider nan also as a bant state type.(different from the notebook)
    bankstate_bins = list(set(list(BankStates["BankState"])))
    BankStates["BankState"] = BankStates["BankState"].map(lambda x: bankstate_bins.index(x))

    #NAICS
    NAICSs['NAICS'] = NAICSs['NAICS'].map(lambda x: fooNAICS(x))

    # ApprovalDates
    familiar_date = np.load("familiar_date.npy")
    ApprovalDates = preprocess_str(ApprovalDates,familiar_date,'ApprovalDate')

    # ApprovalFYs
    ApprovalFYs['ApprovalFY'] = ApprovalFYs['ApprovalFY'].map(lambda x: fooApprovalFYs(x))

    # Term
    Terms["Term"] = Terms["Term"].map(lambda x: fooTerm(x))

    # NoEmps
    NoEmps['NoEmp'] = NoEmps['NoEmp'].map(lambda x: fooNoEmps(x))

    # NewExists
    NewExists['NewExist'] = NewExists['NewExist'].map(lambda x: fooNewExists(x))

    # CreateJobs
    CreateJobs['CreateJob'] = CreateJobs['CreateJob'].map(lambda x: foo_createJobs(x))

    #RetainedJobs
    RetainedJobs['RetainedJob'] = RetainedJobs['RetainedJob'].map(lambda x: fooRetainedJobs(x))

    # FranchiseCodes
    FranchiseCodes['FranchiseCode'] = FranchiseCodes['FranchiseCode'].map(lambda x:fooFranchiseCodesIdx(x))

    # UrbanRural
    # Keep it.

    # RevLineCr
    RevLineCrs['RevLineCr'] = RevLineCrs['RevLineCr'].map(lambda x:fooRevLineCr(x))

    # LowDoc
    LowDocs['LowDoc'] = LowDocs['LowDoc'].map(lambda x:fooLowDoc(x))

    # DisbursementDates
    DisbursementDatesIdx = list(np.load("DisbursementDatesIndex.npy",allow_pickle=True))
    DisbursementDatesFreq = list(np.load("DisbursementDatesFreq.npy",allow_pickle=True))
    #DisbursementDates['DisbursementDate'] = DisbursementDates['DisbursementDate'].map(lambda x:fooDisbursementDate(x,DisbursementDatesIdx,DisbursementDatesFreq))

    def foogross(x):
        tem = x[1:].split(",")
        item = ''.join(tem)
        item = item[:-4]
        num = int(item)
        return num


    DisbursementGrosss['DisbursementGross'] = DisbursementGrosss['DisbursementGross'].map(lambda x:foogross(x))
    SBA_Appvs['SBA_Appv'] = SBA_Appvs['SBA_Appv'].map(lambda x:foogross(x))
    GrAppvs['GrAppv'] = GrAppvs['GrAppv'].map(lambda  x:foogross(x))

    # Gross TODO (?) I just load pre1.0.csv from Wang.
    #d = pd.read_csv("Xtrain_pre1.0.csv")
    #Grosses = d[["Id","DisbursementGross","SBA_Appv","GrAppv"]]
    Grosses = [DisbursementGrosss,SBA_Appvs,GrAppvs]
    from functools import reduce

    dfs = [Names, Banks, BankStates, NAICSs,ApprovalDates,ApprovalFYs,Terms,NoEmps,NewExists,CreateJobs,RetainedJobs,
           FranchiseCodes,UrbanRurals,RevLineCrs,LowDocs,DisbursementGrosss,SBA_Appvs,GrAppvs]
    df_final = reduce(lambda left,right: pd.merge(left,right), dfs)

    df_final.to_csv(out_csv_name,index=False,header=['Id','Name','Bank','BankState','NAICS','ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp',
                                          'NewExist','CreateJob', 'RetainedJob', 'FranchiseCode', 'UrbanRural','RevLineCr','LowDoc',
                                          "DisbursementGross","SBA_Appv","GrAppv"])

# a= pd.read_csv("NewExists_to_DisbursementDates.csv")
# b= pd.read_csv("NAICS_to_NoEmp.csv")
# c= pd.read_csv("To_BankStates.csv")
# d = pd.read_csv("Xtrain_pre1.0.csv")
# d = d[["Id","DisbursementGross","SBA_Appv","GrAppv"]]
# df1=pd.merge(a,b)
# df2=pd.merge(c,d)
# df =pd.merge(df1,df2)
# df.to_csv("Xtrain_pre2.0.csv",index=False,header=['Id','Name','Bank','BankState','NAICS','ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp',
#                                       'NewExist','CreateJob', 'RetainedJob', 'FranchiseCode', 'UrbanRural','RevLineCr','LowDoc','DisbursementDate',
#                                       "DisbursementGross","SBA_Appv","GrAppv"])

if __name__=="__main__":
    preprocess_all(X_path="./Xtest.csv", out_csv_name="Xtest_pre.csv")
