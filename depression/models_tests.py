import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib


data_classification = pd.read_csv('resources/preprocessed_student_depression_data.csv')
data_classification = data_classification.drop(columns=['Family History of Mental Illness'])

X_class = data_classification.drop(columns=['Depression'])
y_class = data_classification['Depression']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10, min_samples_split=5)
random_forest_model.fit(X_train_class, y_train_class)
y_pred_class = random_forest_model.predict(X_test_class)
print(f'RF: Classification Report:\n{random_forest_model.score(X_test_class, y_test_class)}')
print(f'RF: Confusion Matrix:\n{pd.crosstab(y_test_class, y_pred_class, rownames=["Actual"], colnames=["Predicted"])}')
print(f'RF: Accuracy: {accuracy_score(y_test_class, y_pred_class)}')
print(f'RF: F1 Score: {f1_score(y_test_class, y_pred_class)}')
print(f'RF: Precision: {precision_score(y_test_class, y_pred_class)}')
print(f'RF: Recall: {recall_score(y_test_class, y_pred_class)}')

model_tree = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
model_tree.fit(X_train_class, y_train_class)
y_pred_tree = model_tree.predict(X_test_class)
print(f'Decision Tree Classification Report:\n{model_tree.score(X_test_class, y_test_class)}')
print(f'Decision Tree Confusion Matrix:\n{pd.crosstab(y_test_class, y_pred_tree, rownames=["Actual"], colnames=["Predicted"])}')
print(f'Decision Tree Accuracy: {accuracy_score(y_test_class, y_pred_tree)}')
print(f'Decision Tree F1 Score: {f1_score(y_test_class, y_pred_tree)}')
print(f'Decision Tree Precision: {precision_score(y_test_class, y_pred_tree)}')
print(f'Decision Tree Recall: {recall_score(y_test_class, y_pred_tree)}')

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_class, y_train_class)
y_pred_knn = knn_model.predict(X_test_class)
print(f'KNN Classification Report:\n{knn_model.score(X_test_class, y_test_class)}')
print(f'KNN Confusion Matrix:\n{pd.crosstab(y_test_class, y_pred_knn, rownames=["Actual"], colnames=["Predicted"])}')
print(f'KNN Accuracy: {accuracy_score(y_test_class, y_pred_knn)}')
print(f'KNN F1 Score: {f1_score(y_test_class, y_pred_knn)}')
print(f'KNN Precision: {precision_score(y_test_class, y_pred_knn)}')
print(f'KNN Recall: {recall_score(y_test_class, y_pred_knn)}')

joblib.dump(random_forest_model, 'resources/rf_model.pkl')


