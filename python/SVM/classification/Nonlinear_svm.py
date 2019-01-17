import numpy as np
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC

polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree = 3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C = 10, loss = "hinge")),
    ))

polynomial_svm_clf.fit(X,y)