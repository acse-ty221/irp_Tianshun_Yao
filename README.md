## Individual research project: Applied machine learnging for vibration control

## Project Description

This repository is for the individual research project, of which the topic is applying machince learning for blast-induced ground vibration control. This project proposes a novel non-linear model based on support vector machine (SVM), applying particle swarm optimization (PSO) to optimise the hyperparameters of SVM and randoom forest (RF) to select most dominant features. The result indicates that the proposed model is caplble to provide a accurate and efficient prediction with a small sample dataset compared with multivariate linear regression (MLR) and three empirical models which are implemented as well. With favor of the accurate and efficient prediction of ground vibration provided by the porposed model, the blast design parameters can be modifed to control the vibration under a certain level. 

## Motivation

Blast is an important operation in mining industry that can break rocks into appreciate sizes. However, the usage of blasting operations can lead to undesirable outcomes. Ground vibration is one of the most harmful outcomes, it can lead to human discomfort, damage to surronding civil buildings and even can influence groundwater (Hudaverdi, 2012; Monjezi et al., 2016; Navarro Torres et al., 2018). Thus, to predict and control the ground vibration is important and vital for an community-friendly and environment-friendly mining operation. Traditionally, the predictoin is achieved by empirical models, which are established by filed mearsurment. But the ability of generlization of the empirical models is limited as the condition of mines varies a lot from location to location (Dumakor-Dupey et al., 2021). In this context, this study proposed a novel machine learning based model to predict the ground virbation. 

[1] Dumakor-Dupey, N. K., Arya, S., & Jha, A. (2021). Advances in blast-induced impact prediction—a review of machine learning applications. *Minerals*, *11*(6). https://doi.org/10.3390/min11060601

[2] Hudaverdi, T. (2012). Application of multivariate analysis for prediction of blast-induced ground vibrations. *Soil Dynamics and Earthquake Engineering*, *43*, 300–308.

[3] Monjezi, M., Singh, T. N., Khandelwal, M., Sinha, S., Singh, V., & Hosseini, I. (2016). Prediction and Analysis of Blast Parameters Using Artificial Neural Network: https://doi.org/10.1260/095745606777630323

[4] Navarro Torres, V. F., Silveira, L. G. C., Lopes, P. F. T., & de Lima, H. M. (2018). Assessing and controlling of bench blasting-induced vibrations to minimize impacts to a neighboring community. *Journal of Cleaner Production*, *187*, 514–524. https://doi.org/10.1016/j.jclepro.2018.03.210

## Dateset

The data set used in this study is collected at Akdaglar Quarry of Cendere basin, Istanbul. There are 88 observation points recorded. The data file is included in folder named 'resources'. The link is: https://github.com/ese-msc-2021/irp-ty221/blob/main/resources/ground_vibration.xlsx

## Get started 

To install the environment, please run the following command in the terminal

```
conda env create -f environment.yml
```

To activate the virtual enviornment, please run:

```
conda activate IRP_ty221
```

To check the environment list, please run:

```
conda env list
```

To quit the environment, please run:

```
deactivate
```



## What is included in this repository

In this repository, the following items are included: 

1. Environment creation

   Please follow the installatoin guidance described above.

2. 'RUNMODEL' folder

   This folder includes the proposed model and relevent analysis process.

3. 'tools' folder 

   The tools implmented in this study is included here.

4. 'resources' folder

   Ths folder includes the dataset.

5. '.github/workflows' folder

   Workfolw to run pytest and pep8 are included here.

6. Colab notebook with all outputs run in the cells for demenstrating

7. 'Pics' folder 

   The pics and plots used in this study are included here



## The proposed RF-PSO-SVM model

To run the proposed model, there are two pathes: 

1. Run the ‘RUN MODEL' section in the colab notebook in Google Colab.

2. Run the 'PSO_SVM.py' python file in the 'RUNMODEL' folder, remember to install and activate the environment firstly.

   In these two ways, the following functions will be called to build the model:

   ```
   tools.DataPre.LoadDataset
   tools.DataPre.DetectOutliers
   tools.DataPre.FeatureSelect
   tools.DataPre.DataDistribution
   tools.DataPre.NonLinearModel
   ```

## Other models for comparision

To investigate the superioity of the proposed RF-PSO-SVM model, four popular exsisting blast-induced ground vibration predictor were implemented as well.  They are multivariate linear regression and three empirical models. They are implemented in 

```
tools.MLRModels.MLR_cv
tools.EmpiricalModels.EMs_cv
```

## Results analysis tool

In this study, the models are evaluated and discussed in the following aspects:

1. Accuracy analysis

   In this study, prediction accuracy is evaluated with three indicators: R-squared, Mean squared error and Mean Absolute error.

   Besides, a tool that can make the scatter plot representing the predicted value and measured value of each model can be called for usage:

   'RUNMODELS/ScatterPlot.py'.

2. Variance of prediction with diferent input dimensions

   In this study, the feature selection is based on random forest. To access the incluence of different input dimensions, a tool can be used by running: 

   'RUNMODELS/VarianceInputDimensions.py'

   to make a variance plot. 

3. Prediction stability analysis

   To evaluate the ability of generlization of the models, k-fold cross-validation is employed to every model. In purpose of comparison, a tool is built to compare the variance of the prediction of the comparing models over the validation groups. It can be used by running:

   'RUNMODELS/StabilityAnalysis.py'

To see the plots of these results and analysis, you can check the notbook in Google Colab or run directly the relevent python file.

## Workflows

- **Pep8/**: Runs flake8 on all python scripts to check the format of the code 
- **pytest/**: Runs pytest to test whether the package can be imported correctly and certain functions can run successfully or not.

## Contact

For any questions, suggestions and issue reports, please contact me via:

- ty221@ic.ac.uk

## Acknowledgements

I would like to acknowledge the support of my supervisors:

- Dr. Paulo Lopes
- Dr. Pablo Brito-Parada

They have provided very kindful, patient, and helpful support during the project. 



