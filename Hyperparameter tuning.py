import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
norm = Normalizer()
nsamples, nx, ny = x_train.shape
X_train, X_test, Y_train, Y_test = train_test_split(norm.transform(x_train.reshape((nsamples, nx * ny)))[:6000],
                                                    y_train[:6000], test_size=0.3, random_state=40)
x_test_norm = norm.transform(X_test)
x_train_norm = norm.transform(X_train)

params = {'algorithm': ('auto', 'brute'), 'n_neighbors': [3, 4], 'weights': ('uniform', 'distance')}
params2 = {'n_estimators': [800], 'max_features': ('sqrt', 'auto', 'log2'),
           'class_weight': ('balanced', 'balanced_subsample'), 'random_state': [40]}

search = GridSearchCV(KNeighborsClassifier(), params, scoring='accuracy', n_jobs=-1).fit(x_train_norm, Y_train)
search2 = GridSearchCV(RandomForestClassifier(), params2, scoring='accuracy', n_jobs=-1).fit(
    x_train_norm, Y_train)


@ignore_warnings(category=ConvergenceWarning)
def fit_predict_eval(model, X_train, X_test, target_train, target_test):
    mdl = model.fit(X_train, target_train)
    scr = mdl.score(X_test, target_test)
    if type(mdl).__name__ == 'KNeighborsClassifier':
        print('K-nearest neighbours algorithm')
    else:
        print('Random forest algorithm')
    print(f'best estimator: {mdl}\naccuracy: {scr}\n')


fit_predict_eval(KNeighborsClassifier(**search.best_params_), X_train, x_test_norm, Y_train, Y_test)
fit_predict_eval(RandomForestClassifier(**search2.best_params_), X_train, x_test_norm, Y_train, Y_test)
