from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon = 1.5, tol = 1e-4, C =1)
svm_reg.fit(X, y)