# Test the preprocessor

import csv
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge # Primal (covariance)
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge # Dual (kernel)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, json

from sklearn import datasets

if __name__ == "__main__":


    #data = datasets.load_boston()
    #df = pd.DataFrame(data.data, columns=data.feature_names)
    #target = pd.DataFrame(data.target, columns=["MEDV"])
    #x = df
    #y = target["MEDV"]

    #df = pd.read_csv( "sample_dsets/pdsk.csv", index_col=0 , delimiter=',' )
    #x = df[ ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10'] ]
    #y = df[ ['C11']]

    ## KNOWN TO BE RELATIONAL TO TOTAL COSTS
    # Emergency Department Indicator (confirmed) y - high, n - low
    # Hospital County (confirmed)
    # Age Group 
    # Ethnicity
    # Total Charges (?) Length of Stay (?)

    # ##### slight relation ########
    # Gender
    selected = ['Length of Stay','Total Charges']
    target = ['Total Costs']
    ohe = ['Hospital County','Age Group']
    tlabel = selected + ohe + target

    df = pd.read_csv( "sample_dsets/Combined10k.csv", index_col=0 , delimiter=',' )

    df = df[ tlabel ] 

    m5dummies = pd.get_dummies( df.filter( ohe, axis =1 ))
    df = df.drop( ohe, axis=1) 
    df = df.join( m5dummies )

    train, test = train_test_split( df, test_size=0.1)

    x_train = train.drop(target,axis=1)
    x_test = test.drop(target,axis=1)
    y_train = train.filter( target, axis=1) 
    y_test = test.filter( target, axis=1)

    #reg = Ridge(alpha=1.0)
    reg = Ridge()
    reg.fit( x_train, y_train)
    sample = reg.predict( x_test )
    print("W:",reg.coef_)
    print("MSE:")
    print( mean_squared_error( sample, y_test))
    print("R2S:")
    print( r2_score( sample, y_test) )

    #target = ["Total Costs"]

    #ignores = ["Operating Certificate Number","Facility Name","Discharge Year",\
    #        "CCS Diagnosis Description","CCS Procedure Description","APR DRG Description",\
    #        "APR MDC Description","APR Severity of Illness Description","APR Risk of Mortality",\
    #        "Attending Provider License Number","Operating Provider License Number",\
    #        "Other Provider License Number","Birth Weight","Abortion Edit Indicator",
    #        "Total Charges"]

    #selected = ['Age Group','Gender','Race','Ethnicity','Insurance Type','Hospital County',\
    #        'Facility Name','Length of Stay','Admit Day of Week','Type of Admission',\
    #        'Patient Disposition','Discharge Day of Week','CCS Diagnosis Code','CCS Procedure Code',\
    #        'APR DRG Code','APR MDC Code','APR Severity of Illness Code','APR Risk of Mortality',\
    #        'APR Medical Surgical Description','Emergency Department Indicator']

    #selected2 = ['Age Group','Insurance Type','Length of Stay','Type of Admission','CCS Diagnosis Code',\
    #        'CCS Procedure Code','Emergency Department Indicator']

    #labelencode = [
    #    "Age Group","Gender","Race","Ethnicity","Insurance Type","Hospital County",\
    #    "Facility Name","Admit Day of Week","Type of Admission",\
    #    "Patient Disposition","Discharge Day of Week","CCS Diagnosis Code","CCS Procedure Code",\
    #    "APR DRG Code","APR MDC Code","APR Severity of Illness Code","APR Risk of Mortality",\
    #    'APR Medical Surgical Description','Emergency Department Indicator']

    #labelencode2 = ['Age Group','Insurance Type','Type of Admission','CCS Diagnosis Code',\
    #        'CCS Procedure Code','Emergency Department Indicator']

    #sns.set(color_codes=True)
    #sns.regplot( x='Total Charges', y='Total Costs', data = df)
    #plt.show()


    
    #chose =['Gender','Emergency Department Indicator']
    #ohe = ['Ethnicity','Hospital County','Age Group']
    #lbe = ['Gender','Emergency Department Indicator']
    #sel = df[chose]
    #m5dummies = pd.get_dummies( df.filter( ohe, axis =1 ))
    #sel = sel.join(m5dummies)

    #lec = LabelEncoder()
    #for c in lbe:
    #    lec.fit( sel[c] )
    #    sel[c] = lec.transform( sel[c] )

    #print(sel.head(5))



    ## model 3
    #selected3 = ['Length of Stay','Total Charges','Gender','Emergency Department Indicator']
    #ohe = ['Age Group','Insurance Type','Type of Admission','Hospital County','Patient Disposition']
    #lbe = ['Gender','Emergency Department Indicator']
    #sle = ['Length of Stay','Total Charges']

    #sns.set()
    #sns.catplot( x='Emergency Department Indicator', y = 'Total Costs' , kind="point", data=df)
    #plt.show()

    ###replace all missing with mode
    #for c in df.columns.values.tolist():
    #    df[c].fillna( df[c].mode()[0], inplace=True)

    #xsel = df[selected3]
    #ytar = df[target]

    #m3dummies = pd.get_dummies(df.filter( ohe, axis = 1))
    #xsel = xsel.join( m3dummies )

    #lec = LabelEncoder()
    #for c in lbe:
    #    lec.fit( xsel[c] )
    #    xsel[c] = lec.transform( xsel[c] )

    # model 4
    #selected4 = ['Length of Stay','Total Charges']
    #ohe = ['Age Group','Insurance Type']

    #xsel = df[selected4]
    #ytar = df[target]

    #
    #scaler = MinMaxScaler()
    #xsel = pd.DataFrame(scaler.fit_transform( xsel.values ), index = xsel.index, columns=xsel.columns )

    #m4dummies = pd.get_dummies(df.filter( ohe, axis = 1 ))
    #xsel = xsel.join(m4dummies)

    ##scaler = MinMaxScaler()
    ##scaler = StandardScaler()
    ##xsel[ sle ] = scaler.fit_transform( xsel[ sle ] )
    ##ytar[ ytar.columns ] = scaler.fit_transform( ytar[ytar.columns] ) 

    #print( xsel.columns.values.tolist() )
    #print( xsel.head(5))

    #sns.catplot( x='Ethnicity', y='Total Costs', kind='bar', data=df)
    #plt.show()
    #sns.relplot( x='Age Group' , hue='Insurance Type', y=target, data=df)
    #plt.show()
    #sns.countplot( 'Type of Admission' , hue='Insurance Type', data=df)
    #plt.show()
    #sns.countplot( 'Type of Admission', hue='Patient Disposition', data=df)
    #plt.show()
    #sns.set(style="whitegrid", color_codes=True)
    #sns.set(rc={'figure.figsize':(11.7,8.27)})
    #sns.countplot('Age Group', data= df, hue=target)
    #sns.relplot( x='Length of Stay', y=target,col = 'Hospital County',
    #        hue = 'Insurance Type', data = df )
    #sns.despine(offset=10,trim=True)

    #xsel = df.filter(selected, axis=1)
    #ytar = df.filter( ["Total Costs"], axis=1)
    
    #dt = df[target]
    #del df[target]
