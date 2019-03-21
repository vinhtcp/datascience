# Workshop 22 Predicting House Prices

# =============================================================================
# STEP 1: Prepare data
# =============================================================================
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

# Importing the dataset
dataset = pd.read_csv('22_workshop_data.csv')

# Remove unneccessary items manually
# ???? QA: any other way?
#	1	Id	            #	21	YearRemsodAdd	#	41	HeatingQC	    #	61	GarageFinish
#	2	MSSubClass	    #	22	RoofStyle	    #	42	CentralAir	    #	62	GarageCars
#	3	MSZoning	    #	23	RoofMatl	    #	43	Electrical	    #	63	GarageArea
#	4	LotFrontage	    #	24	Exterior1st	    #	44	1stFlrSF	    #	64	GarageQual
#	5	LotArea 	    #	25	Exterior2nd	    #	45	2ndFlrSF	    #	65	GarageCond
#	6	Street	        #	26	MasVnrType	    #	46	LowQualFinSF	#	66	PavedDrive
#	7	Alley	        #	27	MasVnrArea	    #	47	GrLivArea	    #	67	WoodDeckSF
#	8	LotShape	    #	28	ExterQual	    #	48	BsmtFullBath	#	68	OpenPorchSF
#	9	LandContour	    #	29	ExterCond	    #	49	BsmtHalfBath	#	69	EnclosedPorch
#	10	Utilities	    #	30	Foundation	    #	50	FullBath	    #	70	3SsnPorch
#	11	LotConfig	    #	31	BsmtQual	    #	51	HalfBath	    #	71	ScreenPorch
#	12	LandSlope	    #	32	BsmtCond	    #	52	BedroomAbvGr	#	72	PoolArea
#	13	Neighborhood	#	33	BsmtExposure	#	53	KitchenAbvGr	#	73	PoolQC
#	14	Condition1	    #	34	BsmtFinType1	#	54	KitchenQual	    #	74	Fence
#	15	Condition2	    #	35	BsmtFinSF1	    #	55	TotRmsAbvGrd	#	75	MiscFeature
#	16	BldgType	    #	36	BsmtFinType2	#	56	Functional	    #	76	MiscVal
#	17	HouseStyle	    #	37	BsmtFinSF2	    #	57	Fireplaces	    #	77	MoSold
#	18	OverallQual	    #	38	BsmtUnfSF	    #	58	FireplaceQu	    #	78	YrSold
#	19	OverallCond	    #	39	TotalBsmtSF	    #	59	GarageType	    #	79	SaleType
#	20	YearBuilt	    #	40	Heating	        #	60	GarageYrBlt	    #	80	SaleCondition
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
### ??? WHY NOT??? HOW TO AUTO FILTER feature which have too much NA ###
#	58	FireplaceQu: too much NA
#	73	PoolQC: too much NA
dataset = dataset.iloc[:, [3, 4, 5, 16, 17, 18, 19, 20, 27, 28, 30, 31, 40, 41,
                           46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 61, 62,
                           63, 64, 65, 71, 75, 77, 80]].replace('NA', np.NaN)

# Missing values
# Option 1: Use tool like https://machinelearningmastery.com/how-to-handle-missing-values-in-machine-learning-data-with-weka/
# Option 2: Use python
# Option 2.1: Remove rows with missing values
# Current remove 1460 - 1103
dataset = dataset.dropna()
y = np.log(dataset.iloc[:, -1].values)

# Encoding categorical data & Avoiding the Dummy Variable Trap by using drop_first parameter
backupdataset = pd.get_dummies(dataset.iloc[:, :-1], drop_first=True)
X = backupdataset.values

# Scale numerical data
sc_X = StandardScaler()
# Number of numerical data is 20
X[:, :20] = sc_X.fit_transform(X[:, :20])
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1, 1))
y = y.ravel()

# Splitting data into Train set and Test set
# ??? Why test_size = 0.1 make XGBoost score lower than test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# STEP 2: LINEAR REGRESSION
# =============================================================================
# Fitting Linear Regression to dataset
lr = LinearRegression()
lr.fit(X_train, y_train)

# Cross validation (10-fold cross validation)
# ??? Why there is really low score in k cross validation
lr_score = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)
print("----------------------------------------------")
print("Step 2:")
print("Linear Regression Model score:", lr_score)
print("Mean score:", lr_score.mean())
print("Standard Deviation:", lr_score.std())
print("----------------------------------------------")

# =============================================================================
# STEP 3: Imporve the prediction with Regularization
# =============================================================================

# Ridge Regression with 10-fold cross validation
rr = RidgeCV(cv = 10)
rr_cv = rr.fit(X_train, y_train)
print("----------------------------------------------")
print("Step 3:")
print("Ridge Regression Model score: ", rr_cv.score(X_train, y_train))
print("Best alpha: ", rr_cv.alpha_)

# Lassor Regression with 10-fold cross validation
lssr = LassoCV(cv = 10, random_state = 0)
lssr_cv = lssr.fit(X_train, y_train)
print("Lasso Regression Model score: ", lssr_cv.score(X_train, y_train))
print("Best alpha: ", lssr_cv.alpha_)
print("----------------------------------------------")

# =============================================================================
# STEP 4: RANDOM FOREST
# =============================================================================
# =============================================================================
# STEP 5: Show which features are the most helpful to predict house prices using random forest
# =============================================================================
# Fitting Random Forest to dataset
rfr = RandomForestRegressor(n_estimators = 298,
                            max_depth = 14,
                            criterion = 'mse',
                            max_features='sqrt',
                            random_state = 0)
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

# =============================================================================
# STEP 7: BONUS Use XGBoost
# =============================================================================
# Fitting XGBoost to the dataset
xgbr = XGBRegressor()
xgbr.fit(X_train, y_train)

# Cross validation (10-fold validation)
xgbr_score = cross_val_score(estimator = xgbr, X = X_train, y = y_train, cv = 10)
print("----------------------------------------------")
print("Step 7:")
print("XGBoost score:", xgbr_score)
print("Mean score:", xgbr_score.mean())
print("Standard Deviation:", xgbr_score.std())
print("----------------------------------------------")

# Show features' score by descending order
ftscrxgboost = sorted(zip(map(lambda x: round(x, 4), 
    xgbr.feature_importances_), backupdataset.columns.values), reverse = True)
ftscrxgboost = np.array(ftscrxgboost).reshape(-1, 2)
print("Features sorted by their score:")
print(ftscrxgboost)

# Feature selection
# Fitting model using each importance as a threshold
thresholds = np.unique(np.sort(xgbr.feature_importances_))
score_array = []
no_of_feature = []
for thresh in thresholds:
	# select features using threshold
    selection = SelectFromModel(xgbr, threshold=thresh, prefit=True)
    select_X = selection.transform(X_train)
	# model
    selection_model = XGBRegressor()
    selection_model.fit(select_X, y_train)
    selection_model_score = cross_val_score(estimator = selection_model, X = select_X, y = y_train, cv = 10).mean() * 100
    score_array.append(selection_model_score)
    no_of_feature.append(select_X.shape[1])
    print("Thresh=%.6f, n=%d, Score: %.3f%%" % (thresh, select_X.shape[1], selection_model_score))
plt.plot(no_of_feature, score_array)
plt.show()

# Applying Grid Search to find the best parameter
parameters = [{'max_depth':[1], 'n_estimators':[10, 100]}]
grid_search = GridSearchCV(estimator = xgbr, param_grid = parameters, cv = 10)
grid_search = grid_search.fit(X, y)
print("Best score:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)

# =============================================================================
# TODO: use all feature
# =============================================================================
# =============================================================================
# TODO: another way to fill missing value
# =============================================================================
# =============================================================================
# TODO: LEARN MORE ABOUT
# - Regularization
#     + Ridge
#     + Lasso
#     + Other
# - XGBoost: why XGBoost, ensemble
# - Grid Search
# - Cross Validation
#     + Internal Cross Validation
#     + External Cross Validation
# - Train set + Validation set + Test set
# =============================================================================
# =============================================================================
# NOTE
# - Random Forest, XGBoost: feature selection using SelectFromModel,
# Linear Regression, Ridge, Lasso: feature selection using PValue?
# =============================================================================
"""#Backup code
X = dataset.values
# Encoding categorical data & Scale numerical data
# QA: What is TA condition? Ex: Excellent, Gd: Good, Fa: Fair, Po: Poor
categorical_column = [2, 3, 8, 9, 10, 11, 12, 13, 21, 26, 27, 28]
numerical_column = [0, 1, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 29, 30, 31]
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
transformer = make_column_transformer((StandardScaler(), numerical_column),(OneHotEncoder(), categorical_column))
X = transformer.fit_transform(X)
transformer.get_feature_names
# Feature selection
# https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
# Removing features with low variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
selector.fit_transform(X)"""