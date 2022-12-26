from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import metrics
from tools import gol
from sklearn.linear_model import LinearRegression


def Loocv_scatter_plot(feature_transformed, new_targets_tmp,
                       C=0.1, epsilon=0.1, gamma=0.1,
                       SVM=False, MLR=False):
    """
    This function is to plot the scatter plots of MLR or PSO-SVM model.

    Parameters
    ----------
    feature_transformed: pandas.DataFrame
                        The features
    new_targets_tmp    : pandas.DataFrame
                        The targets
    C                  : float (default 0.1)
                        The C hyperparameter of SVM
    epsilon            : float (default 0.1)
                        The epsilon hyperparameter of SVM
    gamma              : float (default 0.1)
                        The gamma hyperparameter of SVM
    SVM                : bool (default False)
                        The option of plotting the scatter of SVM

    MLR                : bool (default False)
                        The option of plotting the scatter of MLR

    Returns
    -------
    No returns as it directly save a set of figures
    """

    R2_list = []
    total_targets = np.array([])
    total_predicted_targets = np.array([])
    loo = KFold(n_splits=10)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    i = 0

    if (SVM):
        model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
        R2_tmp = 0.7716
        pic_path = '../Pics/PSO-SVM_True_vs_Pred'
    elif (MLR):
        model = LinearRegression()
        R2_tmp = 0.5788
        pic_path = '../Pics/MLR_True_vs_Pred'

    for train, test in loo.split(feature_transformed):
        X_train = np.array(feature_transformed[train])
        y_train = np.array(new_targets_tmp)[train]
        X_test = np.array(feature_transformed[test])
        y_test = np.array(new_targets_tmp)[test]
        total_targets = np.append(total_targets, y_test)

        model.fit(X_train, y_train)
        predict_results = model.predict(X_test)
        total_predicted_targets = np.append(
            total_predicted_targets, predict_results)
        R2_list.append(metrics.r2_score(predict_results, y_test))

        ax.scatter(y_test, predict_results,
                   label=str(i + 1) + 'th validatoin group', s=500)
        ax.set_xlabel("Measured value", fontsize=28)
        ax.set_ylabel("Predicted value", fontsize=28)
        i = i + 1

    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, "black", label='y = x')
    ax.text(0.4, 0.8, "R2 = " + str(R2_tmp),
            transform=ax.transAxes, size=40)
    ax.legend(prop={'size': 18})
    ax.grid()
    plt.show()
    fig.savefig(pic_path)

    return total_targets, total_predicted_targets


def Loocv_scatter_EM(feature_transformed, targets):
    """
    This function is to plot the scatter plots of empirical models.

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
    em_blast_data = gol.get_value("df_blast_data")

    loo = KFold(n_splits=10)
    i = 0

    fig = plt.figure(figsize=(40, 13))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    for train, test in loo.split(feature_transformed):
        D = np.array(em_blast_data.iloc[1:, 8]).ravel()
        Q = np.array(em_blast_data.iloc[1:, 7]).ravel()
        D = D[test]
        Q = Q[test]
        y_test = np.array(targets)[test]

        y_EM1 = 248.370 * np.power(D / np.power(Q, 0.5), -1.207)
        y_EM2 = 676.242 * np.power(D / np.power(Q, 0.33), -1.215)
        y_EM3 = 1.911 * np.power(Q / np.power(D, 2 / 3), 1.219)

        ax1.scatter(y_test, y_EM1, label=str(i + 1) +
                    'th validatoin group', s=500)
        ax1.set_xlabel("Measured value", fontsize=28)
        ax1.set_ylabel("Predicted value", fontsize=28)

        ax2.scatter(y_test, y_EM2, label=str(i + 1) +
                    'th validatoin group', s=500)
        ax2.set_xlabel("Measured value", fontsize=28)
        ax2.set_ylabel("Predicted value", fontsize=28)

        ax3.scatter(y_test, y_EM3, label=str(i + 1) +
                    'th validatoin group', s=500)
        ax3.set_xlabel("Measured value", fontsize=28)
        ax3.set_ylabel("Predicted value", fontsize=28)
        i = i + 1

    x = np.linspace(*ax1.get_xlim())
    ax1.plot(x, x, "black", label='y = x')
    ax1.text(0.5, 0.8, "R2 = " + str(0.7111),
             transform=ax1.transAxes, fontsize=28)
    # ax1.legend()
    ax1.grid()

    ax2.plot(x, x, "black", label='y = x')
    ax2.text(0.5, 0.8, "R2 = " + str(0.7052),
             transform=ax2.transAxes, fontsize=28)

    ax2.grid()

    ax3.plot(x, x, "black", label='y = x')
    ax3.text(0.5, 0.8, "R2 = " + str(0.4634),
             transform=ax3.transAxes, fontsize=28)

    ax3.grid()

    fig.savefig("../Pics/Empirical_Measure_vs_Pred")
    plt.show()
