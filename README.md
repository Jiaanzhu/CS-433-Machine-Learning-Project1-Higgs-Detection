# EPFL-Machine-Learning-Projects
Two projects in Machine Learning course

Project1:
Results:(with GD)
1. remove all column with -999, accuracy: 0.7215
2. replace all -999 with mean, use test mean in test, accuracy: 0.7286
3. replace all -999 with mean, use train mean in test, accuracy: 0.7287
4. remove columns with number of -999 > 50%, then replace all -999 with mean, accuracy: 0.701
5. remove all columns with -999 except the first column: 0.7200
6. replace all -999 with median, use train median in test, accuracy: 0.7046 (wrongly implemented)
7. replace all -999 with median, use train median in test, accuracy: 0.7048 (correctly implemented)
8. replace all -999 with mean, use train mean in test, standardize, accuracy: 0.6598 (wrongly implemented)
9. replace all -999 with mean, use train mean in test, standardize, accuracy: 0.6860 (correctly implemented)
