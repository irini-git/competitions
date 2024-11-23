from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

# Prepare data
x, y = make_multilabel_classification(n_samples=10000, n_features=20,
                                      n_classes=5, random_state=88)

for i in range(5):
    print(x[i]," =====> ", y[i])

# split the data into the train and test parts.¶
xtrain, xtest, ytrain, ytest=train_test_split(x, y, train_size=0.8, random_state=88)
print(len(xtest))

# Defining the model
kf = KFold(n_splits=5)
for fn, (trn_idx, val_idx) in enumerate(kf.split(xtrain, ytrain)):
    print (fn, (trn_idx, val_idx))

classifier = MultiOutputClassifier(XGBClassifier())

clf = Pipeline([('classify', classifier)])