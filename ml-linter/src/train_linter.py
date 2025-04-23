import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
from feature_extraction import build_vectorizer, transform_features

# 1. Загрузка разделённых данных
df_train = pd.read_csv('ml-linter/data/train.csv')
df_test  = pd.read_csv('ml-linter/data/test.csv')

# 2. Векторизация и AST‑признаки
vect = build_vectorizer(df_train['code'])
X_train = transform_features(df_train['code'], vect)
X_test  = transform_features(df_test['code'],  vect)

# 3. Обучение дерева решений
clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, df_train['label'])

# 4. Оценка
y_pred = clf.predict(X_test)
print(classification_report(df_test['label'], y_pred))

# 5. Сохранение моделей
joblib.dump(vect, 'ml-linter/models/vectorizer.joblib')
joblib.dump(clf, 'ml-linter/models/clf.joblib')
print('Векторизатор и классификатор сохранены.')


#python ml-linter/src/train_linter.py