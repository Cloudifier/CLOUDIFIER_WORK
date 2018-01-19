from sklearn import svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

mnist = fetch_mldata('mnist-original')
data = scale(mnist.data)
X_train, X_test, y_train, y_test = train_test_split(data, mnist.target, test_size=0.1)
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
decision_functions = ['ovo', 'ovr']
for kernel in kernels:
    for decision in decision_functions:
        print('Kernel: {:}, decision: {:}'.format(kernel, decision))
        classifier = svm.SVC(gamma=0.001, cache_size=500, kernel=kernel, decision_function_shape=decision)
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)

        print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))
        print(metrics.classification_report(y_test, predicted))
        print('\n')