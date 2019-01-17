from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth = 2)
tree_clf.fit(X, y)

export_graphviz(
    tree_clf,
    out_file = "E://机器学习经典模型//决策树//iris_tree_clf.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    filled=True
)

print(tree_clf.predict_proba([[5, 1.5]]))
print(iris.target_names[tree_clf.predict([[5, 1.5]])])