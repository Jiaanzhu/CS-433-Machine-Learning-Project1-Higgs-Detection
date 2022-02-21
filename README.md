# Higgs-Detection

GROUP: Number_1(Jiaan Zhu, Lei Wang, Qinyue Zheng)
  
Steps to get our best prediction:
1. open the terminal and cd to this folder.
2. add "train.csv" and "test.csv" as training dataset and test dataset accordingly to this folder.
3. 'python run.py'. !USE YOUR OWN PATH HERE
4. a csv file named as ".csv" will appear in this folder.


Sidenote:
1. The other functions including we used to cross-validate on regression model and get the best-performed model is "helper.py". 
2. Our exploratory data analysis, trained weights and results analysis can be checked in "Project_1.ipynb" (We left it under this folder in case you want to take a look) and also our PDF report.

Useful feedback from TA: 
As for the hyperparameter tuning, using a sequential pipeline to estimate each parameter is suboptimal and can yield bad results. Alternative approaches like doing grid search, or random search, to optimize the combination of the parameters would have been a better option. 


