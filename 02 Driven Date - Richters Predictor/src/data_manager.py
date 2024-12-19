import pandas as pd


# Constants
FILENAME_TEST_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv'
FILENAME_TRAIN_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Labels.csv'
FILENAME_TRAIN_LABELS = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Values.csv'


class EarthquakeData:
    def __init__(self):
        self.df_train_labels, self.df_train_values, self.df_test_values = self.load_data()
        self.explore_data()

    def load_data(self):
        """
        Load train and test data
        :return:
        """

        df_train_labels = pd.read_csv(FILENAME_TRAIN_LABELS)
        df_train_values = pd.read_csv(FILENAME_TRAIN_VALUES)
        df_test_values = pd.read_csv(FILENAME_TEST_VALUES)

        return df_train_labels, df_train_values, df_test_values

    def explore_data(self):
        """
        Explore data
        """

        # Explore columns, shape
        for df in [self.df_test_values]:
            print(df.columns)
            for c in df.columns:
                print(df[c].value_counts())
