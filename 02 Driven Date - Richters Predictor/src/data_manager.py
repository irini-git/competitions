import pandas as pd


# Constants
FILENAME_TEST_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv'
FILENAME_TRAIN_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Labels.csv'
FILENAME_TRAIN_LABELS = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Values.csv'


class EarthquakeData:
    def __init__(self):
        self.df_train_labels, self.df_train_values, self.df_test_values, self.df_train = self.load_data()
        self.explore_data()

    def load_data(self):
        """
        Load train and test data
        :return:
        """

        df_train_labels = pd.read_csv(FILENAME_TRAIN_LABELS)
        df_train_values = pd.read_csv(FILENAME_TRAIN_VALUES)
        df_test_values = pd.read_csv(FILENAME_TEST_VALUES)

        df_train = df_train_labels.merge(df_train_values)

        return df_train_labels, df_train_values, df_test_values, df_train

    def explore_data(self):
        """
        Explore data
        """

        # Explore columns, shape
        print('-'*10)
        for df in [self.df_train]:
            # for c in df.columns:
                # print(df[c].value_counts())
            print(f'Dataframe shape : {df.shape}')
            print(f'Null values in database : {df.isnull().sum().sum()}')
            print(df.info())
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df.describe(include=object))

