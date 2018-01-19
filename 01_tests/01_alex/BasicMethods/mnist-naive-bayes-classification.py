from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

mnist = fetch_mldata('mnist-original')
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.9, random_state=1234)

clf = MultinomialNB()
clf.fit(X_train, y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
print('Train acc: {:}, Test acc: {:}'.format(accuracy_score(y_train, train_pred) * 100, accuracy_score(y_test, test_pred) * 100))
print('Train Classification Report \n {:}'.format(classification_report(y_train, train_pred)))
print('Test Classification Report \n {:}'.format(classification_report(y_test, test_pred)))
