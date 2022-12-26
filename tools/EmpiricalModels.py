import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import mean_absolute_error


def EMs_cv(feature_transformed, df_blast_data):
    """
    This function is to attain the statisitical
    indicatos of empirical models

    Parameters
    ----------
    feature_transformed: pandas.DataFrame
                        The features
    targets            : pandas.DataFrame
                        The targets

    Returns
    -------
    EMs_results: 2d array
                The indicators (R2, MSE, MAE) of three models
    EMs_array  : 2d array
                The arraies of three models recorded during
                cross validation
    """

    em_blast_data = df_blast_data
    # distance
    D = np.array(em_blast_data.iloc[1:, 8]).ravel()
    # maximum charge per delay
    Q = np.array(em_blast_data.iloc[1:, 7]).ravel()
    # measured value
    real_PPV = np.array(em_blast_data.iloc[1:, 10]).ravel()

    EM1_MSE_list = []
    EM1_R2_list = []
    EM1_MAE_list = []

    EM2_MSE_list = []
    EM2_R2_list = []
    EM2_MAE_list = []

    EM3_MSE_list = []
    EM3_R2_list = []
    EM3_MAE_list = []

    loo = KFold(n_splits=10)
    for train, test in loo.split(feature_transformed):
        D_cv = D[test]
        Q_cv = Q[test]
        real_PPV_cv = real_PPV[test]

        EM1_predict_results = 248.370 * np.power(
            D_cv / np.power(Q_cv, 0.5), -1.207)
        EM2_predict_results = 676.242 * np.power(
            D_cv / np.power(Q_cv, 0.33), -1.215)
        EM3_predict_results = 1.911 * np.power(
            Q_cv / np.power(D_cv, 2 / 3), 1.219)

        EM1_MSE_list.append(mean_squared_error(
            real_PPV_cv, EM1_predict_results))
        EM1_R2_list.append(metrics.r2_score(
            real_PPV_cv, EM1_predict_results))
        EM1_MAE_list.append(mean_absolute_error(
            real_PPV_cv, EM1_predict_results))

        EM2_MSE_list.append(mean_squared_error(
            real_PPV_cv, EM2_predict_results))
        EM2_R2_list.append(metrics.r2_score(
            real_PPV_cv, EM2_predict_results))
        EM2_MAE_list.append(mean_absolute_error(
            real_PPV_cv, EM2_predict_results))

        EM3_MSE_list.append(mean_squared_error(
            real_PPV_cv, EM3_predict_results))
        EM3_R2_list.append(metrics.r2_score(
            real_PPV_cv, EM3_predict_results))
        EM3_MAE_list.append(mean_absolute_error(
            real_PPV_cv, EM3_predict_results))

    EM1_MSE_average = np.mean(EM1_MSE_list)
    EM1_R2_average = np.mean(EM1_R2_list)
    EM1_MAE_average = np.mean(EM1_MAE_list)

    EM2_MSE_average = np.mean(EM2_MSE_list)
    EM2_R2_average = np.mean(EM2_R2_list)
    EM2_MAE_average = np.mean(EM2_MAE_list)

    EM3_MSE_average = np.mean(EM3_MSE_list)
    EM3_R2_average = np.mean(EM3_R2_list)
    EM3_MAE_average = np.mean(EM3_MAE_list)

    print("The R2 for Empirical model one is:", EM1_R2_average)
    print("The MSE for Empirical model one is:", EM1_MSE_average)
    print("The MAE for Empirical model one is:", EM1_MAE_average)

    print("The R2 for Empirical model two is:", EM2_R2_average)
    print("The MSE for Empirical model two is:", EM2_MSE_average)
    print("The MAE for Empirical model two is:", EM2_MAE_average)

    print("The R2 for Empirical model three is:", EM3_R2_average)
    print("The MSE for Empirical model three is:", EM3_MSE_average)
    print("The MAE for Empirical model three is:", EM3_MAE_average)

    EMs_results = \
        np.array([[EM1_R2_average, EM1_MSE_average, EM1_MAE_average],
                  [EM2_R2_average, EM2_MSE_average, EM2_MAE_average],
                  [EM3_R2_average, EM3_MSE_average, EM3_MAE_average]])

    EMs_arrays = \
        np.array([[EM1_R2_list, EM1_MSE_list, EM1_MAE_list],
                  [EM2_R2_list, EM2_MSE_list, EM2_MAE_list],
                  [EM3_R2_list, EM3_MSE_list, EM3_MAE_list]])

    return EMs_results, EMs_arrays
