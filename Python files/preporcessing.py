#import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read file
df=pd.read_csv("../glioma+grading+clinical+and+mutation/dataset/TCGA_GBM_LGG_Mutations_all.csv")

#drop irrelevant features
relevant_df=df.drop(['Case_ID','Project','Primary_Diagnosis','Age_at_diagnosis','Race'],axis=1)

#removing rows with '--' or unreported features
relevant_df=relevant_df.drop(relevant_df[relevant_df['Gender']=="--"].index)

#mapping others
relevant_df['Grade']=relevant_df['Grade'].map({'LGG':0,'GBM':1})
relevant_df['Gender']=relevant_df['Gender'].map({'Female':0,'Male':1})
#Mapping markers
relevant_df['IDH1']=relevant_df['IDH1'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['TP53']=relevant_df['TP53'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['ATRX']=relevant_df['ATRX'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['PTEN']=relevant_df['PTEN'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['EGFR']=relevant_df['EGFR'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['CIC']=relevant_df['CIC'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['MUC16']=relevant_df['MUC16'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['PIK3CA']=relevant_df['PIK3CA'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['NF1']=relevant_df['NF1'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['PIK3R1']=relevant_df['PIK3R1'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['FUBP1']=relevant_df['FUBP1'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['RB1']=relevant_df['RB1'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['NOTCH1']=relevant_df['NOTCH1'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['BCOR']=relevant_df['BCOR'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['CSMD3']=relevant_df['CSMD3'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['SMARCA4']=relevant_df['SMARCA4'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['GRIN2A']=relevant_df['GRIN2A'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['IDH2']=relevant_df['IDH2'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['FAT4']=relevant_df['FAT4'].map({'NOT_MUTATED':0,'MUTATED':1})
relevant_df['PDGFRA']=relevant_df['PDGFRA'].map({'NOT_MUTATED':0,'MUTATED':1})

#exporting data
relevant_df.to_csv('../glioma+grading+clinical+and+mutation/dataset/relevant_df')

