import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data_classification = pd.read_csv('resources/preprocessed_student_depression_data.csv')
data_classification = data_classification.drop(columns=['Family History of Mental Illness'])

X_class = data_classification.drop(columns=['Depression'])
y_class = data_classification['Depression']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
model_class = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10, min_samples_split=5)
model_class.fit(X_train_class, y_train_class)
y_pred_class = model_class.predict(X_test_class)
print(f'Classification Report:\n{model_class.score(X_test_class, y_test_class)}')
print(f'Confusion Matrix:\n{pd.crosstab(y_test_class, y_pred_class, rownames=["Actual"], colnames=["Predicted"])}')
print(f'Accuracy: {model_class.score(X_test_class, y_test_class)}')
print(f'F1 Score: {model_class.score(X_test_class, y_test_class)}')
print(f'Precision: {model_class.score(X_test_class, y_test_class)}')
print(f'Recall: {model_class.score(X_test_class, y_test_class)}')
print(f'ROC AUC: {model_class.score(X_test_class, y_test_class)}')

model_tree = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
model_tree.fit(X_train_class, y_train_class)
y_pred_tree = model_tree.predict(X_test_class)
print(f'Decision Tree Classification Report:\n{model_tree.score(X_test_class, y_test_class)}')
print(f'Decision Tree Confusion Matrix:\n{pd.crosstab(y_test_class, y_pred_tree, rownames=["Actual"], colnames=["Predicted"])}')
print(f'Decision Tree Accuracy: {model_tree.score(X_test_class, y_test_class)}')
print(f'Decision Tree F1 Score: {model_tree.score(X_test_class, y_test_class)}')
print(f'Decision Tree Precision: {model_tree.score(X_test_class, y_test_class)}')
print(f'Decision Tree Recall: {model_tree.score(X_test_class, y_test_class)}')
print(f'Decision Tree ROC AUC: {model_tree.score(X_test_class, y_test_class)}')

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_class, y_train_class)
y_pred_knn = knn_model.predict(X_test_class)
print(f'KNN Classification Report:\n{knn_model.score(X_test_class, y_test_class)}')
print(f'KNN Confusion Matrix:\n{pd.crosstab(y_test_class, y_pred_knn, rownames=["Actual"], colnames=["Predicted"])}')
print(f'KNN Accuracy: {knn_model.score(X_test_class, y_test_class)}')
print(f'KNN F1 Score: {knn_model.score(X_test_class, y_test_class)}')
print(f'KNN Precision: {knn_model.score(X_test_class, y_test_class)}')
print(f'KNN Recall: {knn_model.score(X_test_class, y_test_class)}')
print(f'KNN ROC AUC: {knn_model.score(X_test_class, y_test_class)}')


