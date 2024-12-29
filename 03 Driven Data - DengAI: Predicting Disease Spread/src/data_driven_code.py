# This is code recomended by Driven data as a starting point

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# load the provided data
train_features = pd.read_csv('../data/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv',
                             index_col=[0,1,2])

train_labels = pd.read_csv('../data/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv',
                           index_col=[0,1,2])


# Seperate data for San Juan
sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']

print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)

# Remove `week_start_date` string.
sj_train_features = sj_train_features.drop('week_start_date', axis=1).copy()
iq_train_features = iq_train_features.drop('week_start_date', axis=1).copy()

def plot_vegetation_index():

    fig, ax = plt.subplots(figsize=(15,15))
    sj_train_features.ndvi_ne.plot.line(ax=ax, lw=0.8)

    plt.title('Vegetation Index over Time')
    plt.xlabel('Time')
    plt.close()
    fig.savefig('../fig/data_driven_001_ndvi')


# plot_vegetation_index()

# Fill NAs
sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)

# Let's see how our labels are distributed!

print('San Juan')
print('mean: ', sj_train_labels.mean()[0])
print('var :', sj_train_labels.var()[0])

print('\nIquitos')
print('mean: ', iq_train_labels.mean()[0])
print('var :', iq_train_labels.var()[0])

# Add total cases to features
sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases

# compute the correlations
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()

def plot_correlations(df, city):
    fig, ax = plt.subplots(figsize=(15,15))
    df.total_cases.drop('total_cases').sort_values(ascending=False).plot.barh(ax=ax)
    fig.savefig(f'../fig/corr_data_drive_{city}.png')

plot_correlations(sj_correlations, 'sj')
plot_correlations(sj_correlations, 'iq')


