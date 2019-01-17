from sklearn.svm import SVR

svm_poly_reg = SVR(kernel = "poly", degree = 2, C =100, epsilon=0.1)
svm_poly_reg.fit(X, y)