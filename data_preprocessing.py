import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split

def load_juliet_java(dataset_dir):
    data, labels = [], []
    pattern = os.path.join(dataset_dir, 'src', 'testcases', '**', '*.java')
    for fp in glob.iglob(pattern, recursive=True):
        name = os.path.basename(fp).lower()
        if '_bad' in name:
            label = 1
        elif '_good' in name:
            label = 0
        else:
            # пропускаем файлы без явного good/bad в имени
            continue
        code = open(fp, encoding='utf-8', errors='ignore').read()
        data.append(code)
        labels.append(label)
    return pd.DataFrame({'code': data, 'label': labels})

if __name__ == '__main__':
    df = load_juliet_java('dataset/juliet-test-suite')
    print('Распределение меток:\n', df['label'].value_counts())  # проверяем, что есть оба класса
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    os.makedirs('data', exist_ok=True)
    df_train.to_csv('data/train.csv', index=False)
    df_test.to_csv('data/test.csv', index=False)
    print(f'Готово: train={df_train.shape}, test={df_test.shape}')


#python data_preprocessing.py