
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import mean_absolute_error


def MLR_cv(feature_transformed, targets):
    """
    This function is to attain the statisitical
    indicatos of Mulivariate linear regression model

    Parameters
    ----------
    feature_transformed: pandas.DataFrame
                        The features
    targets            : pandas.DataFrame
                        The targets

    Returns
    -------
    MLR_R2_average : The average R2 value of the validation groups
    MLR_MSE_average: The average MSE value of the validation groups
    MLR_MAE_average: The average MAE value of the validation groups
    MLR_R2_list    : The list of R2 values of the validation groups
    MLR_MSE_list   : The list of MSE values of the validation groups
    MLR_MAE_list   : The list of MAE values of the validation groups

    """

    MLR_MSE_list = []
    MLR_R2_list = []
    MLR_MAE_list = []

    loo = KFold(n_splits=10)
    for train, test in loo.split(feature_transformed):
        X_train = np.array(feature_transformed[train])
        y_train = np.array(targets)[train]
        X_test = np.array(feature_transformed[test])
        y_test = np.array(targets)[test]

        LR = LinearRegression()
        LR.fit(X_train, y_train)
        MLR_predict_results = LR.predict(X_test)
        MLR_MSE_list.append(mean_squared_error(y_test, MLR_predict_results))
        MLR_R2_list.append(metrics.r2_score(y_test, MLR_predict_results))
        MLR_MAE_list.append(mean_absolute_error(y_test, MLR_predict_results))

    MLR_MSE_average = np.mean(MLR_MSE_list)
    MLR_R2_average = np.mean(MLR_R2_list)
    MLR_MAE_average = np.mean(MLR_MAE_list)

    return MLR_R2_average, MLR_MSE_average, \
        MLR_MAE_average, MLR_R2_list, MLR_MSE_list, MLR_MAE_list
