GROUP: Number_1(Jiaan Zhu, Lei Wang, Qinyue Zheng)
  
Steps to get our best prediction:
        1. open the terminal and cd to this folder.
        2. add "train.csv" and "test.csv" as training dataset and test dataset accordingly to this folder.
        3. 'python run.py'. !USE YOUR OWN PATH HERE
        4. a csv file named as ".csv" will appear in this folder.


Sidenote:
1. The other functions including we used to cross-validate on regression model and get the best-performed model is "helper.py". 
2. Our exploratory data analysis, trained weights and results analysis can be checked in "Project_1.ipynb" (We left it under this folder in case you want to take a look) and also our PDF report.

# EPFL-Machine-Learning-Projects
Project1 in Machine Learning course

Project1:
Results:
(with GD)
1. remove all column with -999, accuracy: 0.7215
2. replace all -999 with mean, use test mean in test, accuracy: 0.7286
3. replace all -999 with mean, use train mean in test, accuracy: 0.7287
4. remove columns with number of -999 > 50%, then replace all -999 with mean, accuracy: 0.701
5. remove all columns with -999 except the first column: 0.7200
6. replace all -999 with median, use train median in test, accuracy: 0.7046 (wrongly implemented)
7. replace all -999 with median, use train median in test, accuracy: 0.7048 (correctly implemented)
8. replace all -999 with mean, use train mean in test, standardize, accuracy: 0.6598 (wrongly implemented)
9. replace all -999 with mean, use train mean in test, standardize, accuracy: 0.6860 (correctly implemented)

(with LS)
10-17 try normalize and standardize with removing outliers(3 * IQR or 1.5 * IQR): < 0.700

18 replace all -999 with mean, use train mean in test, augment, degree = 3, accuracy: 0.7845
19 replace all -999 with mean, use train mean in test, augment, degree = 4, accuracy: 0.7917
20 replace all -999 with mean, use train mean in test, standized, augment, degree = 4, accuracy: 0.6839
21 replace all -999 with mean, use train mean in test, augment, degree = 5, accuracy: 0.7957
22 Remove 10 outliers(IQR * 24), replace all -999 with mean, use train mean in test, augment, degree = 5, accuracy: 0.7968(slightly better than previous one)
23 Remove 10 outliers(IQR * 24), replace all -999 with mean, use train mean in test, standardize, augment, degree = 5, accuracy: 0.7780
24 4 group method, accuracy: 0.8097
25 4 group method, standarlized, accuracy: 0.7043
26 4 group method, Remove outliers(IQR * 24) accuracy: 0.8092
27 4 group method, Remove outliers(IQR * 30) accuracy: 0.8097
28 3 group method, accuracy: 0.7223
29 4 group method, augment 2i - 1, degree = 5, accuracy: 0.5384

