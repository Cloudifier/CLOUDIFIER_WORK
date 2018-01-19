from sklearn.datasets import fetch_mldata
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
import numpy as np

mnist = fetch_mldata('mnist-original')
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.3)

params = [(depth, leaf) for depth in range(2, 20) for leaf in range(2, 20)]
best_test_acc = 0
best_max_depth = 0
best_max_leafs = 0
for depth, leaf in params:
    clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
    clf = clf.fit(X_train, y_train)
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)
    pred_train = clf.predict(X_train).reshape(X_train.shape[0], 1)
    pred_test = clf.predict(X_test).reshape(X_test.shape[0], 1)
    train_acc = np.sum(pred_train == y_train) / y_train.shape[0] * 100
    test_acc = np.sum(pred_test == y_test) / y_test.shape[0] * 100
    if best_test_acc < test_acc:
        best_test_acc = test_acc
        best_max_depth = depth
        best_max_leafs = leaf

    print('Max depth: {:}, Max leafs: {:}, Train acc: {:}%, Test acc: {:}%'.format(depth, leaf, train_acc, test_acc))

print('Best train acc: {:}, max depth: {:}, max leaf: {:}'.format(best_test_acc, best_max_depth, best_max_leafs))
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("mnist")