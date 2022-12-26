from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import random


def Loocv(feature_transformed, new_targets_tmp, C, epsilon, gamma,
          fit=True, n_splits=10):
    """
    This function is to calculate the fitness value(R2) and other
    statisticla indices (MSE and MAE) based on k-fold
    cross validation

    Parameters
    ----------
    feature_transformed: ndarray or pd.DataFrame
        The features
    new_targets_tmp.   : ndarray or pd.DataFrame
        The targets
    C                  : float
        The trade off hyperparameter of SVM
    epsilon            : float
        The epsilon hyperparameter of SVM
    gamma              : float
        The gamma hyperparameter of SVM
    fit                : bool (default by True)
        A bool value deciding return
        R2 or a set of R2, MSE and MAE
    n_splits           : int (default by 10)
        The number of folds in cross validation
    Returns
    -------
    if (fit == True):
      R2_average: float
          The average R2 value of the validation groups
    else :
      R2_average : The average R2 value of the validation groups
      MSE_average: The average MSE value of the validation groups
      MAE_average: The average MAE value of the validation groups
      R2_list    : The list of R2 values of the validation groups
      MSE_list   : The list of MSE values of the validation groups
      MAE_list   : The list of MAE values of the validation groups

    """

    if (fit):
        R2_list = []

        loo = KFold(n_splits)
        for train, test in loo.split(feature_transformed):
            X_train = np.array(feature_transformed[train])
            y_train = np.array(new_targets_tmp)[train]
            X_test = np.array(feature_transformed[test])
            y_test = np.array(new_targets_tmp)[test]

            model_svr = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
            model_svr.fit(X_train, y_train)
            predict_results = model_svr.predict(X_test)

            R2_list.append(metrics.r2_score(y_test, predict_results))

        R2_average = np.mean(R2_list)

        return R2_average

    else:
        MSE_list = []
        R2_list = []
        MAE_list = []

        loo = KFold(n_splits)
        for train, test in loo.split(feature_transformed):
            X_train = np.array(feature_transformed[train])
            y_train = np.array(new_targets_tmp)[train]
            X_test = np.array(feature_transformed[test])
            y_test = np.array(new_targets_tmp)[test]

            model_svr = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
            model_svr.fit(X_train, y_train)
            predict_results = model_svr.predict(X_test)
            MSE_list.append(mean_squared_error(y_test, predict_results))
            R2_list.append(metrics.r2_score(y_test, predict_results))
            MAE_list.append(mean_absolute_error(y_test, predict_results))

        MSE_average = np.mean(MSE_list)
        R2_average = np.mean(R2_list)
        MAE_average = np.mean(MAE_list)
        return R2_average, \
            MSE_average, MAE_average, R2_list, MSE_list, MAE_list


class PSO:

    def __init__(self, parameters, feature_transformed, targets):
        """
        This is the initialization function of PSO class
        particle swarm optimization

        Reference: https://gist.github.com/dwiuzila/79903ecd98fca39474180ba273ded787

        Parameters
        ----------
        parameters: list or list like
                  The parameters of PSO,
                   including number of generations,
                   population size,
                   number of variables,
                   and variables bound

        Return
        ----------
        No return
        """
        # initialization of parameters
        self.num_gen = parameters[0]  # number of generations
        self.pop_size = parameters[1]  # population size of each generation
        # number of variables to be optimized
        self.num_var = len(parameters[2])
        self.bound = []  # limitation of variables to be optimized
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
        # the positions of all particles
        self.pop_x = np.zeros((self.pop_size, self.num_var))
        # the speeds of all particles
        self.pop_v = np.zeros((self.pop_size, self.num_var))
        # the best position of every particle
        self.p_best = np.zeros((self.pop_size, self.num_var))
        self.g_best = np.zeros((1, self.num_var))  # global optimized position

        self.feature_transformed = feature_transformed
        self.targets = targets
        # initialize the first generation randomly
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.num_var):
                self.pop_x[i][j] = random.uniform(
                    self.bound[0][j], self.bound[1][j])
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]  # store the best individual
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, ind_var):
        """
        This is a method to calculate the fitness value of each particle

        Parameters
        ----------
        ind_var: list or list like
                Storing the variables to be optimized, in this case,
                they are: C, epsilon anf gamma

        Return
        ----------
        fitness_value: float
                      The calulated fitness value, in this case,
                      it can represent R2, MSE or MAE
        """
        # X = feature_train
        # y = target_train
        x1 = ind_var[0]
        x2 = ind_var[1]
        x3 = ind_var[2]

        if x1 == 0:
            x1 = 0.001
        if x2 == 0:
            x2 = 0.001
        if x3 == 0:
            x3 = 0.001

        fitness_value = Loocv(
            self.feature_transformed,
            self.targets,
            C=x1,
            epsilon=x2,
            gamma=x3)
        return fitness_value

    def update_operator(self, pop_size, gen):
        """
        Update operator to update the speed and position of next moment

        Parameters
        ----------
        pop_size: int
                 The size of the population
        gen     : int
                 The currrent number of generation

        Return
        ----------
        No return as it will directly update the attributes of the object
        """

        t = gen
        n = self.num_gen
        w = (0.4 / n ** 2) * (t - n) ** 2 + 0.4
        c1 = -3 * t / n + 3.5
        c2 = 3 * t / n + 0.5

        for i in range(pop_size):
            # Update velocity vector
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (
                self.g_best - self.pop_x[i])
            # Update locations
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # out-of-bounds protection
            for j in range(self.num_var):
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]

            # Update p_best (local best) and g_best (global best)
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]

    def main(self):
        """
        Optimization process calling the methods to conduct PSO

        Parameters
        ----------
        No parameters

        Return
        ----------
        ng_best : list
                The list of the optimal position,
                which is the optimized hyperparameters of the model
        """
        print("\n PSO optimization process start! \n")

        popobj = []  # store the fitness value evolution
        self.ng_best = np.zeros((1, self.num_var))[
            0]  # initialize the global best
        for gen in range(self.num_gen):

            # draw 4D plot
            if (gen == 0 or gen == 10 or gen == 20 or gen == (self.num_gen - 1)):
                x_list = []
                y_list = []
                z_list = []
                c_list = []

                for p in range(self.pop_size):
                    # record the locations of particles in the population
                    x_list.append(self.pop_x[p][0])
                    y_list.append(self.pop_x[p][1])
                    z_list.append(self.pop_x[p][2])

                    c_list.append(self.fitness(self.pop_x[p]))

                fig = plt.figure(figsize=(9, 10.5))
                ax1 = fig.add_subplot(211, projection='3d')
                ax2 = fig.add_subplot(212)

                img1 = ax1.scatter(
                    x_list, y_list, z_list, c=c_list, cmap='winter')
                ax1.set_xlabel("C", fontsize=15)
                ax1.set_ylabel("epsilon: \u03B5", fontsize=15)
                ax1.set_zlabel("gamma: \u03B3", fontsize=15)
                ax1.tick_params(axis='x', labelsize=8)
                ax1.tick_params(axis='y', labelsize=8)
                ax1.tick_params(axis='z', labelsize=8)
                fig.colorbar(img1, ax=ax1)

                img2 = ax2.scatter(x_list, y_list, c=c_list, cmap='winter')
                ax2.yaxis.set_ticks_position('right')
                ax2.tick_params(axis='x', labelsize=8)
                ax2.tick_params(axis='y', labelsize=8)
                ax2.set_xlabel("C", fontsize=15)
                ax2.set_ylabel("epsilon: \u03B5", fontsize=15)
                ax2.grid()
                fig.colorbar(img2, ax=ax2)

                fig.savefig(
                    '../Pics/Particle_Distribution_' +
                    str(gen) +
                    'gen.png')
                plt.show()
                # draw end

            self.update_operator(self.pop_size, gen)  # Update
            popobj.append(self.fitness(self.g_best))
            print('---------- Generation {} ----------'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()  # Attain the global best
            print('Best position：{}'.format(self.ng_best))
        print('Highest fitness value：{}'.format(self.fitness(self.ng_best)))
        print("---- End of (successful) Searching ----")

        # Plot the convergence plot
        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(36, 20)
        plt.xlabel("Iterations", size=30)
        plt.ylabel("Fitness", size=30)
        plt.yticks(size=23)
        plt.xticks(size=23)
        t = [t for t in range(self.num_gen)]
        plt.plot(
            t,
            popobj,
            c='royalblue',
            linewidth=6,
            label='fitness value (R2)')
        plt.grid()
        fig.savefig('../Pics/Convergence_Plot')
        plt.show()
        ng_best = self.ng_best

        return ng_best


def NonLinearModel(feature_transformed, targets, PSO_opt=True):
    """
    NonLinearModel function to firstly randomly set a SVR model then apply
    PSO to optimize the hyperparameters. It will generate plots for the
    comparison of performance of the SVR model before and after optimization

    Parameters
    ----------
    feature_transformed: ndarray
                        The normalized feature of the dataset

    targets            : ndarray
                        The targets without detected outliers
    PSO_opt            : bool
                        Determine whether go with pso or not
    Return
    ----------
    No return but will plot diagrams to compare the performance
    """
    print("\nTrain SVR momdel\n")

    print("The R2 of a randomly initialized SVM: ",
          Loocv(feature_transformed, targets, C=1, epsilon=0.2, gamma=5))

    if (PSO_opt):
        NGEN = 60
        popsize = 30
        low = [0.001, 0.001, 0.01]
        up = [1000, 1000, 8]
        parameters = [NGEN, popsize, low, up]

        pso = PSO(parameters, feature_transformed, targets)
        best_hyper_para = pso.main()
        print(best_hyper_para)
        print("---------------------")
        print("The performance of the optimized SVR model is:")
        R2_optimal = Loocv(
            feature_transformed,
            targets,
            C=best_hyper_para[0],
            epsilon=best_hyper_para[1],
            gamma=best_hyper_para[2])
        print(
            "R2:",
            Loocv(
                feature_transformed,
                targets,
                C=best_hyper_para[0],
                epsilon=best_hyper_para[1],
                gamma=best_hyper_para[2]))

    return R2_optimal
