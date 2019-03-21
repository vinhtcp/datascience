# Workshop: Predicting House Prices

# =============================================================================
# STEP 1: Prepare data
# =============================================================================
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

# Importing the dataset
dataset = pd.read_csv('22_workshop_data.csv')


### Number ###
#	4	LotFrontage     #	5	LotArea         #	18	OverallQual     #	19	OverallCond
#	20	YearBuilt       #	21	YearRemodAdd    #	47	GrLivArea       #	48	BsmtFullBath
#	49	BsmtHalfBath    #	50	FullBath        #	51	HalfBath        #	52	BedroomAbvGr
#	53	KitchenAbvGr    #	55	TotRmsAbvGrd    #	57	Fireplaces      #	62	GarageCars
#	63	GarageArea      #	72	PoolArea        #	76	MiscVal         #	78	YrSold
### Text ####
#	6	Street          #	17	HouseStyle      #	42	CentralAir      #	66	PavedDrive
#	28	ExterQual       #	29	ExterCond       #	31	BsmtQual        #	32	BsmtCond
#	41	HeatingQC       #	54	KitchenQual     #	58	FireplaceQu     #	64	GarageQual
#	65	GarageCond      
dataset = dataset.iloc[:, [3, 4, 5, 16, 17, 18, 19, 20, 27, 28, 30, 31, 40, 41,
                           46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 61, 62,
                           63, 64, 65, 71, 75, 77, 80]].replace('NA', np.NaN)

# Missing values
dataset = dataset.dropna()
y = np.log(dataset.iloc[:, -1].values)

# Encoding categorical data & Avoiding the Dummy Variable Trap by using drop_first parameter
backupdataset = pd.get_dummies(dataset.iloc[:, :-1], drop_first=True)
backupdataset = backupdataset[['OverallQual', 'GrLivArea', 'LotArea', 'FullBath', 'GarageArea',
                   'LotFrontage', 'YearBuilt', 'GarageCars', 'YearRemodAdd']]
""", 'PoolArea',
                   'TotRmsAbvGrd', 'OverallCond', 'BsmtFullBath', 'Fireplaces', 'BedroomAbvGr',
                   'YrSold', 'HalfBath', 'KitchenQual_Gd', 'BsmtQual_Gd', 'ExterQual_Gd',
                   'HouseStyle_1Story'"""
X = backupdataset.values

# Scale numerical data
sc_X = StandardScaler()
# Number of numerical data is 17
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1, 1))
y = y.ravel()

# Splitting data into Train set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# STEP 4: RANDOM FOREST
# =============================================================================
# Fitting Random Forest to dataset
rfr = RandomForestRegressor(n_estimators = 298, max_depth = 14, criterion = 'mse', max_features='sqrt', random_state = 0)
rfr.fit(X_train, y_train)

# Cross validation (10-fold validation)
rfr_score = cross_val_score(estimator = rfr, X = X_train, y = y_train, cv = 10)
print("----------------------------------------------")
print("Step 4:")
print("Random Forest Model score:", rfr_score)
print("Mean score:", rfr_score.mean())
print("Standard Deviation:", rfr_score.std())
print("----------------------------------------------")

# Cross validation for test dataset (10-fold validation)
rfr_test_score = cross_val_score(estimator = rfr, X = X_test, y = y_test, cv = 10)
print("Test Mean score:", rfr_test_score.mean())
print("Test Standard Deviation:", rfr_test_score.std())

y_pred = np.exp(sc_y.inverse_transform(rfr.predict(X_test)))
y_test = np.exp(sc_y.inverse_transform(y_test))

# Show features' score by descending order
ftscr = sorted(zip(map(lambda x: round(x, 4), 
    rfr.feature_importances_), backupdataset.columns.values), reverse = True)
ftscr = np.array(ftscr).reshape(-1, 2)
print("Features sorted by their score:")
print(ftscr)

# Feature selection
# Fitting model using each importance as a threshold
thresholds = np.unique(np.sort(rfr.feature_importances_))
score_array = []
no_of_feature = []
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(rfr, threshold=thresh, prefit=True)
    select_X = selection.transform(X_train)
    # model
    selection_model = RandomForestRegressor(n_estimators = 100, random_state = 0)
    selection_model.fit(select_X, y_train)
    selection_model_score=cross_val_score(estimator = selection_model, X = select_X, y = y_train, cv = 10).mean() * 100
    score_array.append(selection_model_score)
    no_of_feature.append(select_X.shape[1])
    print("Thresh=%.6f, n=%d, Score: %.3f%%" % (thresh, select_X.shape[1], selection_model_score))
plt.plot(no_of_feature, score_array)
plt.show()

# Applying Grid Search to find the best parameter
parameters = [{'max_depth':[13, 14, 15, 16],
               'criterion':['mse'],
               'n_estimators':[ 298, 299],
               'max_features':['sqrt'],}]
grid_search = GridSearchCV(estimator = rfr, param_grid = parameters, cv = 10)
grid_search = grid_search.fit(X, y)
print("Best score:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)
