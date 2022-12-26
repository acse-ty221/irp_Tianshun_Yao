from tools.PSOSVM import NonLinearModel
from tools.DataPre import FeatureSelect
from tools.PSOSVM import Loocv
from tools.MLRModels import MLR_cv
from tools.EmpiricalModels import EMs_cv
from sklearn.model_selection import GridSearchCV
from tools import gol
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR


def Grid_search(feature_transformed, targets):
    """
    Conduct the grid search process for SVM

    Parameters
    ----------
    feature_transformed: pandas.DataFrame
                        The features
    targets            : pandas.DataFrame
                        The targets

    Returns
    -------
    No returns as it directly save a set of figures

    """

    C_list = np.arange(0.001, 1000, 10)
    epsilon_list = np.arange(0.001, 1000, 10)
    gamma_list = np.arange(0.001, 8, 1)

    param = {'C': C_list, 'epsilon': epsilon_list, 'gamma': gamma_list}

    modelsvr = SVR(kernel='rbf')

    grids = GridSearchCV(modelsvr, param, cv=10, scoring='r2')
    grids.fit(feature_transformed, targets)

    print(grids.best_params_)
    R2_average, MSE_average, MAE_average, \
        R2_list, MSE_list, MAE_list = Loocv(
            feature_transformed, targets, C=71.6475784,
            epsilon=0.001, gamma=0.2590461599, fit=False)
    print("R2:", R2_average)
    print("MSE = ", MSE_average)
    print("MAE = ", MAE_average)


def variance_different_inputs(df_blast_data):
    """
    This function is to record and plot
    the variance of the PSO-SVM model with different
    input dimensions.

    Parameters
    ----------
    df_blast_data : The whole raw data from the dataset

    Returns
    -------
    No returns as it directly save the fig

    """

    R2_feature = []
    num_feature = []

    for i in range(9, 1, -1):
        data_fea = gol.get_value("data_fea")
        data_tar = gol.get_value("data_tar")
        targets = gol.get_value("targets")
        top_features_indices = gol.get_value("top_features_indices")
        FeatureSelect(data_fea, data_tar, n=i)
        new_feature = df_blast_data.iloc[1:, top_features_indices]
        feature = new_feature
        scaler = MinMaxScaler()
        # train standardized object
        scaler.fit(feature)
        # transform dataset
        t_feature_transformed = scaler.transform(feature)

        NLM_predict_results = NonLinearModel(
            t_feature_transformed, targets, PSO_opt=True)
        R2_feature.append(NLM_predict_results)
        num_feature.append(i)

    fig = plt.figure(figsize=(9, 10.5))
    num_feature = np.array(num_feature)
    num_feature = num_feature - 1
    plt.plot(num_feature, R2_feature, c='royalblue', marker='o')
    plt.xlabel("Input dimensions ", size=14)
    plt.ylabel("Fitness value (R2)", size=14)
    plt.grid()
    fig.savefig("../Pics/Different_inputs")
    plt.show()


def validation_compare_plot(feature_transformed, targets, df_blast_data):
    """
    This function is to compare the results (R2, MSE and MAE)
    of the three models and make a plot

    Parameters
    ----------
    feature_transformed: pandas.DataFrame
                        The features
    targets            : pandas.DataFrame
                        The targets

    Returns
    -------
    No returns as it directly save a set of figures
    """

    val_group_id = list([i for i in range(1, 11)])

    SVM_R2_average, SVM_MSE_average, SVM_MAE_average, \
        SVM_R2_list, SVM_MSE_list, \
        SVM_MAE_list = Loocv(
            feature_transformed, targets, C=71.4819934,
            epsilon=0.001, gamma=0.259461599, fit=False)
    MLR_R2_average, MLR_MSE_average, MLR_MAE_average, \
        MLR_R2_list, MLR_MSE_list, \
        MLR_MAE_list = MLR_cv(
            feature_transformed, targets)
    EMs_results, EMs_arrays = EMs_cv(feature_transformed, df_blast_data)

    EM1_R2_list = EMs_arrays[0][0]
    EM1_MSE_list = EMs_arrays[0][1]
    EM1_MAE_list = EMs_arrays[0][2]

    EM2_R2_list = EMs_arrays[1][0]
    EM2_MSE_list = EMs_arrays[1][1]
    EM2_MAE_list = EMs_arrays[1][2]

    EM3_R2_list = EMs_arrays[2][0]
    EM3_MSE_list = EMs_arrays[2][1]
    EM3_MAE_list = EMs_arrays[2][2]

    fig = plt.figure(figsize=(15, 30))
    plt.subplot(311)
    plt.plot(
        val_group_id,
        SVM_R2_list,
        marker='o',
        c='royalblue',
        label="PSO-SVM",
        linewidth=1.5)
    plt.plot(
        val_group_id,
        MLR_R2_list,
        marker='o',
        c='orange',
        label="MLR",
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM1_R2_list,
        marker='^',
        c='limegreen',
        label='USBM',
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM2_R2_list,
        marker='s',
        c='limegreen',
        label='Ambraseys-Hendron',
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM3_R2_list,
        marker='d',
        c='limegreen',
        label='Indian standard',
        linewidth=1.5)
    plt.xlabel("Validation group no.", fontsize=18)
    plt.ylabel("R2", fontsize=18)
    plt.xticks(np.arange(1, 10, 1), fontsize=14)
    plt.yticks(np.arange(0, 1, 0.2), fontsize=14)
    plt.legend(loc=2, prop={'size': 12})
    plt.grid()

    plt.subplot(312)
    plt.plot(
        val_group_id,
        SVM_MSE_list,
        marker='o',
        c='royalblue',
        label="PSO-SVM",
        linewidth=1.5)
    plt.plot(
        val_group_id,
        MLR_MSE_list,
        marker='o',
        c='orange',
        label="MLR",
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM1_MSE_list,
        marker='^',
        c='limegreen',
        label='USBM',
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM2_MSE_list,
        marker='s',
        c='limegreen',
        label='Ambraseys-Hendron',
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM3_MSE_list,
        marker='d',
        c='limegreen',
        label='Indian standard',
        linewidth=1.5)
    plt.xlabel("Validation group no.", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.xticks(np.arange(1, 10, 1), fontsize=14)
    plt.yticks(np.arange(5, 70, 5), fontsize=14)
    plt.legend(loc=2, prop={'size': 12})
    plt.grid()

    plt.subplot(313)
    plt.plot(
        val_group_id,
        SVM_MAE_list,
        marker='o',
        c='royalblue',
        label="PSO-SVM",
        linewidth=1.5)
    plt.plot(
        val_group_id,
        MLR_MAE_list,
        marker='o',
        c='orange',
        label="MLR",
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM1_MAE_list,
        marker='^',
        c='limegreen',
        label='USBM',
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM2_MAE_list,
        marker='s',
        c='limegreen',
        label='Ambraseys-Hendron',
        linewidth=1.5)
    plt.plot(
        val_group_id,
        EM3_MAE_list,
        marker='d',
        c='limegreen',
        label='Indian standard',
        linewidth=1.5)
    plt.xlabel("Validation group no.", fontsize=18)
    plt.ylabel("MAE", fontsize=18)
    plt.xticks(np.arange(1, 10, 1), fontsize=14)
    plt.yticks(np.arange(1, 10, 1), fontsize=14)
    plt.legend(loc=2, prop={'size': 12})
    plt.grid()
    fig.savefig("../Pics/Difference_CV")
