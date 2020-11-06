import pandas as pd

a= pd.read_csv("NewExists_to_DisbursementDates.csv")
b= pd.read_csv("NAICS_to_NoEmp.csv")
c= pd.read_csv("To_BankStates.csv")
d = pd.read_csv("Xtrain_pre1.0.csv")

d = d[["Id","DisbursementGross","SBA_Appv","GrAppv"]]
df1=pd.merge(a,b)
df2=pd.merge(c,d)
df = pd.merge(df1,df2)
df.to_csv("Xtrain_pre2.0.csv",index=False,header=['Id','Name','Bank','BankState','NAICS','ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp',
                                      'NewExist','CreateJob', 'RetainedJob', 'FranchiseCode', 'UrbanRural','RevLineCr','LowDoc','DisbursementDate',
                                      "DisbursementGross","SBA_Appv","GrAppv"])
