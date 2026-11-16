import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

iris = datasets.load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['especie'] = y
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

param_grid = [
    {'kernel': ['linear'], 'C': [0.1,1,10]},
    {'kernel': ['rbf'], 'C': [0.1,1,10], 'gamma': [0.1,1,'scale']}
]

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_s, y_train)

res = pd.DataFrame(grid.cv_results_)
melhores = res.loc[res.groupby('param_kernel')['mean_test_score'].idxmax()
                   ('param_kernel', 'param', 'mean_test_score')]

print(melhores.to_string(index=False))
print(grid.best_params_, round(grid.best_score_, 4))

best = grid.best_estimator_
y_pred = best.predict(X_test_s)
print(accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=iris.target_names).plot(cmap='Greens')