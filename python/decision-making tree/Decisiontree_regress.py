from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz

iris = load_iris()
#注意输入的格式问题，否则会报错
X = iris.data[0:10, 0:1]
y = iris.data[0:10, 1:2]

tree_reg = DecisionTreeRegressor(max_depth = 2)
tree_reg.fit(X, y)

export_graphviz(
    tree_reg,
    out_file = "E://机器学习经典模型//决策树//iris_tree_reg.dot",
    filled=True
)

print(tree_reg.predict([[1]]))