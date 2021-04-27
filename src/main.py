from algorithms import MLAlgorithms
from algorithms import MLResults as mr
import pandas as pd


data_path = r"..\data\cellular_data.csv"
data = pd.read_csv(data_path, delimiter=";", decimal=",")
print("Data successfully uploaded")

print("Algorithms initialized started")
algorithm = MLAlgorithms(data)
print("Success")

print("Algorithms training started (cca. 5min)")
knn_result = algorithm.k_nearest_neighbors(n_neighbors=4)
decision_tree_result = algorithm.decision_tree()
logistic_regression_result = algorithm.logistic_regression()
naive_bayes_result = algorithm.naive_bayes()
random_forest_result = algorithm.random_forest(n_estimators=100)
print("Success")

print("#####################################################")
print("KNN results")
print("#####################################################")
mr.get_accuracy(algorithm.y_test, knn_result, verbose=True)
mr.get_class_report(algorithm.y_test, knn_result, verbose=True)
mr.get_confusion_matrix(algorithm.y_test, knn_result, verbose=True)

print("#####################################################")
print("decision_tree results")
print("#####################################################")
mr.get_accuracy(algorithm.y_test, decision_tree_result, verbose=True)
mr.get_class_report(algorithm.y_test, decision_tree_result, verbose=True)
mr.get_confusion_matrix(algorithm.y_test, decision_tree_result, verbose=True)

print("#####################################################")
print("logistic_regression results")
print("#####################################################")
mr.get_accuracy(algorithm.y_test, logistic_regression_result, verbose=True)
mr.get_class_report(algorithm.y_test, logistic_regression_result, verbose=True)
mr.get_confusion_matrix(algorithm.y_test, logistic_regression_result, verbose=True)

print("#####################################################")
print("naive_bayes results")
print("#####################################################")
mr.get_accuracy(algorithm.y_test, naive_bayes_result, verbose=True)
mr.get_class_report(algorithm.y_test, naive_bayes_result, verbose=True)
mr.get_confusion_matrix(algorithm.y_test, naive_bayes_result, verbose=True)

print("#####################################################")
print("random_forest results")
print("#####################################################")
mr.get_accuracy(algorithm.y_test, random_forest_result, verbose=True)
mr.get_class_report(algorithm.y_test, random_forest_result, verbose=True)
mr.get_confusion_matrix(algorithm.y_test, random_forest_result, verbose=True)
