from data_manager import DengueData

# Initiate class
data = DengueData()

# Data exploration
# data.explore_data()

# Model
data.create_model_HistGradientBoostingRegressor(data.train_data_cleaned)

# Load predictions
data.load_predictions()


# END ------------