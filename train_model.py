import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib, os

# 1) Загрузка
df_train = pd.read_csv('data/train.csv')
df_test  = pd.read_csv('data/test.csv')

# 2) TF-IDF векторизация кода
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=5000)
X_train = vectorizer.fit_transform(df_train['code'])
X_test  = vectorizer.transform(df_test['code'])

# 3) Обучение классификатора
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, df_train['label'])

# 4) Оценка на тесте
y_pred = clf.predict(X_test)
print(classification_report(df_test['label'], y_pred))

# 5) Сохранение модели и векторизатора
os.makedirs('models', exist_ok=True)
joblib.dump(vectorizer, 'models/vectorizer.joblib')
joblib.dump(clf, 'models/classifier.joblib')
print('Модель и векторизатор сохранены в папке models/')
