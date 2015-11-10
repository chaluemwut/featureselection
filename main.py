from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR

def test_main():
    iris = load_iris()
    x, y = iris.data, iris.target
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, 2 , step=1)
    selector = selector.fit(x, y)
    print selector.support_
    

if __name__ == '__main__':
    test_main()