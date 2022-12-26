
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools import gol


def LoadDataset(sheet_opt=2012):
    """
    Load the specific dataset into pandas dataframe

    Parameters
    ----------
    sheet_opt : int (2012 or 2018)
              The option of sheet name,
              2012 for hudaverdi_2012,
              2018 for lopes_2018

    Returns
    -------
    df_blast_data: pandas.DataFrame
                  The raw data of the datasheet
    feature      : pandas.DataFrame
                  The feature rows of the dataset
    targets      : pandas.DataFrame
                  The target rows of the dataset

    """
    if (sheet_opt == 2012):
        df_blast_data = pd.read_excel(
            './resources/ground_vibration.xlsx',
            sheet_name='hudaverdi_2012')
        feature = df_blast_data.iloc[1:, 1: 9]
        targets = df_blast_data.iloc[1:, -2]
        print("sheet named hudaverdi_2012 has been extracted successfully !")
    elif (sheet_opt == 2018):
        df_blast_data = pd.read_excel(
            'ground_vibration.xlsx',
            sheet_name='lopes_2018')
        feature = df_blast_data.iloc[:, [14, -2]]
        targets = df_blast_data.iloc[:, [-4]]
        print("sheet named lopes_2018 has been extracted successfully !")
    else:
        print("no such sheet in execl ground_vibration.xlsx")

    return df_blast_data, feature, targets


def DetectOutliers(feature, targets):
    """
    Detect the outliers in the dataset and
    replace them with the mean of the dataset
    without outliers

    Parameters
    ----------
    feature: ndarray or pd.DataFrame
        The feature of the dataset
    targets: ndarray or pd.DataFrame
        The targets of the dataset

    Returns
    -------
    new_targets: nd.array
        The new targets which outliers has been
        repleced

    """
    global new_targets

    print("\nOutlier detection process start!\n")
    data_y = np.array(targets).ravel()
    data_x = np.array(feature)

    n = 3  # 3 sigma rule
    ymean = np.mean(data_y)  # mean of the targets
    ystd = np.std(data_y)  # standard deviation
    threshold1 = ymean - n * ystd
    threshold2 = ymean + n * ystd

    outlier = []  # Save the outliers
    outlier_x = []  # Save the position of the outliers
    outlier_index = []  # Save the index

    # Detection start
    for i in range(0, len(data_y)):
        if (data_y[i] < threshold1) | (data_y[i] > threshold2):
            outlier.append(data_y[i])
            outlier_x.append(data_x[i])
            outlier_index.append(i)
        else:
            continue
    # if there are outliers
    if (outlier):
        print("\nOutliers have been detected!\n")
        print('\nThe outliers are\n')
        print(outlier)
        print('\nThe position of outliers are\n')
        for i in range(len(outlier_index)):
            print(outlier_index[0])

        # delete the outliers to calculate the mean
        np_target_without_outlier = np.delete(data_y, outlier_index)
        mean_without_outlier = np_target_without_outlier.mean()

        np_new_targets = np.array(targets).ravel()
        new_targets = pd.DataFrame(np_new_targets, columns=targets.columns)
        # replace the outlier value
        for i in range(len(outlier_index)):
            new_targets.iloc[outlier_index[i]] = mean_without_outlier

        gol.set_value("new_targets", new_targets)
        return 1

    # if there is no outlier
    else:
        new_targets = targets
        gol.set_value("new_targets", new_targets)
        print("\nIn this dataset, there is no outlier!\n")
        return 0


def DataScatter(feature, targets):
    """
    Plot the scatters of the feature and target

    Parameters
    ----------
    feature: ndarray or pd.DataFrame
        The feature dataset
    targets: ndarray or pd.DataFrame
        The targets dataset

    Returns
    -------
    No return as this function directly plot
    the scatter diagram

    """
    x = feature.iloc[:, 0]
    y = feature.iloc[:, 1]
    z = np.array(targets).ravel()

    fig3D = plt.figure()
    ax = Axes3D(fig3D)
    ax.scatter(x, y, z)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()

    plt.show()


def FeatureSelect(data_fea, data_tar, top_n=9, n=9):
    """
    Select the most related features based on the
    importance calculated by random forest algorithm and
    return the corresponding index of top related features.
    Also, a histogram will be plotted

    Parameters
    ----------
    data_fea: ndarray or pd.DataFrame
        The total feature rows
    data_tar: ndarray or pd.DataFrame
        The total target rows
    top_n: int
        The number of the most related features
        that are shown in the plot. Default value is 5
    n    : int
        The number of the most related features
        that are chosen as the feature to return

    Returns
    -------
    top_features_indices: nd.array
        The array of indices of the top features
        chosen to be returned

    """

    print("Random forest feature selection start!")

    global top_features_indices
    df_blast_data = gol.get_value("df_blast_data")
    if (data_fea.shape[1] >= 2):
        # set the random forset model
        model = RandomForestRegressor(random_state=1, max_depth=10)
        # RF does not accept nan, bool, str,
        # etc as input. Only number can be the input
        data_fea = data_fea.fillna(0)
        data_fea = pd.get_dummies(data_fea)

        # fit the model
        model.fit(data_fea, data_tar)

        # plot the histogram plot according to the importance
        features = data_fea.columns
        importances = model.feature_importances_
        indices = np.argsort(importances[0:top_n])
        # choose top most realted feature
        fig = plt.figure(figsize=(8, 8))
        plt.title('Index selection')
        plt.barh(range(len(indices)), importances[indices],
                 color='royalblue', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative importance of indicators')
        fig.savefig("../Pics/Importance_hist")
        plt.show()

        # attain the top features that will be returned
        top_features = [features[i] for i in indices[::-1][0:n]]
        print("\nThe top", n, "features are :", top_features)
        top_features_indices = []
        # get the indices
        for i in range(len(top_features)):
            top_features_indices.append(
                df_blast_data.columns.get_loc(top_features[i]))
        print("\nTheir indices are:", top_features_indices)
        gol.set_value("top_features_indices", top_features_indices)
        return 1

    else:
        print("There is no need to select!")
        return 0


def DataDistribution(df_blast_data):
    """
    Plot the scatter diagram of the blast data

    Parameters
    ----------
    df_blast_data : pd.DataFrame
        The whole raw data of the dataset

    Returns
    -------
    No returns as it directly save the figure

    """
    blast_data = df_blast_data.iloc[1:, 1:11]
    convert_blast_data = blast_data.convert_dtypes('float')
    hist_kwds_set = {'bins': 50, 'edgecolor': 'limegreen'},
    scatter_matrix = pd.plotting.scatter_matrix(convert_blast_data,
                                                figsize=(15, 15),
                                                c='royalblue',
                                                marker='o',
                                                diagonal='hist',
                                                hist_kwds=
                                                {'bins': 50, \
                                                           'edgecolor': 'limegreen'},
                                                alpha=0.8,
                                                range_padding=0.1,
                                                )

    for subaxis in scatter_matrix:
        for ax in subaxis:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_xlabel(ax.get_xlabel(), fontsize=20, rotation=0)
            ax.set_ylabel(ax.get_ylabel(), fontsize=20, rotation=90)

    pic = scatter_matrix[0][0].get_figure()
    pic.savefig("../Pics/Data_Distribution")
