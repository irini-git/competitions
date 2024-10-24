import pandas as pd
import os

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

class FluShotData:
    """
    Class responsible for Flu Shot Data
    """
    def __init__(self):
        self.load_data()

    def load_data(self):
        """
        Load raw data (features and labels) from csv files.
        :return:
        """
        df_labels = pd.read_csv(FILENAME_INPUT_DATA_LABELS)
        self.explore_data(df_labels)

    def explore_data(self, df):
        """
        Explore dataset, calculate stats and visualize
        :param df: dataframe to explore
        :return: output to screen, charts as files
        """
        print('Exploring dataset', '-'*10)
        print(f'Dataset has columns : {df.columns.tolist()}.')
        print(f'Dataset has {df.shape[0]} entries.')

        columns = ['h1n1_vaccine', 'seasonal_vaccine']
        # TODO : make this a dataframe
        for c in columns:
            print(df[c].value_counts(), df[c].value_counts()/df.shape[0])

    def get_current_location(self):
        print(os.getcwd())