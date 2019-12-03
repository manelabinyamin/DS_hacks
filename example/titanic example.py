import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import DataCleaning, FeatureEngineering, EDA
import pandas as pd
import numpy as np
# classification models
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# Load datasets ------------------------

train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
train_size, test_size = len(train), len(test)
all_data = pd.concat((train, test), axis=0, ignore_index=True, sort=False).drop('Survived', axis=1)

# clean data
clean_data = DataCleaning.clean_nans(all_data, method='best_predictor', feature_type={'Age':'numeric','SibSp':'numeric','Parch':'numeric'})


# process features ------------------------

# bin continuous features
features_to_bin = ['Age','Fare','Parch', 'SibSp']
rules = {}
for f in features_to_bin:
    rules[f] = FeatureEngineering.get_best_binning_rules(clean_data.head(train_size)[f], train['Survived'], target_type='categorical')
for f,bins in rules.items():
    clean_data['binned_'+f] = pd.cut(clean_data[f], bins=bins, labels=False)

# set Sex to float
clean_data['Sex'] = clean_data['Sex'] == 'male'
clean_data['Sex'] = clean_data['Sex'].astype(float)
# set Embarked to numbers
clean_data['Embarked'] = clean_data['Embarked'].map({'C': 0, 'S': 1, 'Q':2})

# plot results
# binned_train = clean_data[['Pclass','Sex','binned_Age','binned_Fare','binned_Parch','binned_SibSp','Embarked']]
# binned_train = pd.concat((binned_train.head(train_size), train['Survived']), axis=1)
# EDA.plot_numeric_features(binned_train,hue='Survived',fig_size=15)


# build binary features
features_to_1hot = ['Pclass','binned_Age','binned_Fare','binned_Parch', 'binned_SibSp', 'Embarked']
clean_data = FeatureEngineering.build_binary_features(clean_data, features_to_1hot)

# split to train and test
features_to_drop = ['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked',
                    'binned_Age', 'binned_Fare', 'binned_Parch', 'binned_SibSp']
clean_data.drop(features_to_drop, axis=1, inplace=True)

train_x, train_y = clean_data.head(train_size), train['Survived']
test_x = clean_data.tail(test_size)

print('---------------------------------------------------------------------')
print('Final features:')
train_x.info()

# train model
print('---------------------------------------------------------------------')
print('Cross-validation scores:')
models = {
    'LogisticRegression': LogisticRegression(solver='lbfgs'),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=3),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}
scores = pd.DataFrame(np.zeros([len(models), 2]), columns=['median', 'std', '25 percentage'], index=models.keys())
for m,clf in models.items():
    cv_score = cross_val_score(clf, train_x, train_y, cv=5)
    scores.loc[m]['median'] = np.median(cv_score)
    scores.loc[m]['std'] = np.std(cv_score)
    scores.loc[m]['25 percentage'] = np.quantile(cv_score,)

print(scores.head())
best_model = scores['median'].idxmax()
print('Best model: ',best_model)

# make prediction
DS_hacks_submission = test['PassengerId'].to_frame()

clf = models[best_model]
clf.fit(train_x, train_y)
DS_hacks_submission['Survived'] = clf.predict(test_x)
# DS_hacks_submission.to_csv('DS_hacks_submission.csv', index=False)