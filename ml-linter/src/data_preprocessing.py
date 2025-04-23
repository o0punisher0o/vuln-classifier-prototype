import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path='ml-linter/data/code_snippets.csv'):
    df = pd.read_csv(path)
    return df['code'].tolist(), df['label'].tolist()

def split_data(X, y, test_size=0.2, random_state=42):
    # Простейшее разбиение без стратификации
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == '__main__':
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Сохраняем для обучения
    pd.DataFrame({'code': X_train, 'label': y_train}) \
      .to_csv('ml-linter/data/train.csv', index=False)
    pd.DataFrame({'code': X_test,  'label': y_test }) \
      .to_csv('ml-linter/data/test.csv',  index=False)
    print('Данные разделены: train =', len(X_train), ' test =', len(X_test))


#python ml-linter/src/data_preprocessing.py