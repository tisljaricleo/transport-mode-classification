import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class MLAlgorithms(object):
    """Contains ML algorithms

    Class that contains all ML algorithms necessary for this research implemented in sklearn

    ***Methods***
    - decision_tree
    - k_nearest_neighbors
    - logistic_regression
    - naive_bayes
    - random_forest
    - support_vector_machines
    """

    def __init__(self, data_path: str):
        """Initialization
        Initializes input dataset and splits it to train and test sets

        :param data_path: Path to input dataset
        :type data_path: str
        """

        self.data = pd.read_csv(data_path, delimiter=";", decimal=",")
        self.X = self.data.iloc[:, 1:3].values
        self.y = self.data.iloc[:, 8].values
        self.classifier = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=0.3,
                                                                                random_state=1)

    def decision_tree(self):
        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier.predict(self.X_test)

    def k_nearest_neighbors(self, n_neighbors: int):
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier.predict(self.X_test)

    def logistic_regression(self):
        self.classifier = LogisticRegression()
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier.predict(self.X_test)

    def naive_bayes(self):
        self.classifier = GaussianNB()
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier.predict(self.X_test)

    def random_forest(self, n_estimators: int):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators)
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier.predict(self.X_test)

    def support_vector_machines(self, kernel: str):
        self.classifier = SVC(kernel=kernel)
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier.predict(self.X_test)


class MLResults(object):
    """Algorithm results and metrics

    ***Methods***
    - get_accuracy: Gets overal accuracy of the algorithm
    - get_confusion_matrix
    - get_class_report: Gets classification report
    """
    @staticmethod
    def get_accuracy(y_test, y_pred, verbose=False):
        acc = metrics.accuracy_score(y_test, y_pred)
        if verbose:
            print("Accuracy:{0}".format(round(acc, 5)))
        return acc

    @staticmethod
    def get_confusion_matrix(y_test, y_pred, verbose=False):
        cm = confusion_matrix(y_test, y_pred)
        if verbose:
            print(cm)
        return cm

    @staticmethod
    def get_class_report(y_test, y_pred, verbose=False):
        cr = classification_report(y_test, y_pred)
        if verbose:
            print(cr)
        return cr
