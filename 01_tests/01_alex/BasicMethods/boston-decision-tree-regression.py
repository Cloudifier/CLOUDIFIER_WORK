from sklearn.datasets import load_boston
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

boston = load_boston()
X = boston.data
y = boston.target.reshape(boston.target.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

params = [(depth, leaf) for depth in range(2, 20) for leaf in range(2, 20)]
best_test_acc = 0
best_max_depth = 0
best_max_leafs = 0
for depth, leaf in params:
    clf = tree.DecisionTreeRegressor(max_depth=depth, max_leaf_nodes=leaf)
    clf = clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train).reshape(X_train.shape[0], 1)
    pred_test = clf.predict(X_test).reshape(X_test.shape[0], 1)
    train_acc = r2_score(y_train, pred_train) * 100
    test_acc = r2_score(y_test, pred_test) * 100
    if best_test_acc < test_acc:
        best_test_acc = test_acc
        best_max_depth = depth
        best_max_leafs = leaf

    print('Max depth: {:}, Max leafs: {:}, Train acc: {:}%, Test acc: {:}%'.format(depth, leaf, train_acc, test_acc))

print('Best train acc: {:}, max depth: {:}, max leaf: {:}'.format(best_test_acc, best_max_depth, best_max_leafs))