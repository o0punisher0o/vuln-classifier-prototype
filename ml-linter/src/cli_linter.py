import argparse, joblib
from feature_extraction import extract_ast_features
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_models():
    vect = joblib.load('ml-linter/models/vectorizer.joblib')
    clf  = joblib.load('ml-linter/models/clf.joblib')
    feat_names = joblib.load('ml-linter/models/feature_names.pkl')
    return vect, clf, feat_names

def analyze_file(path, vect, clf, feat_names):
    code = open(path, encoding='utf-8', errors='ignore').read()
    tfidf = vect.transform([code]).toarray()
    astf  = np.array([extract_ast_features(code)])
    X     = np.hstack([tfidf, astf])
    prob  = clf.predict_proba(X)[0][1]
    label= 'UNSAFE' if prob>0.5 else 'SAFE'
    print(f'Вероятность небезопасности: {prob:.2%} → {label}')
    # Бонус: подсветка строк с опасными вызовами
    print('\nСтроки с потенциальным риском:')
    for num, line in enumerate(code.splitlines(), 1):
        if any(func in line for func in ['eval(','exec(','os.system','subprocess','pickle.load']):
            print(f'  {num}: {line.strip()}')

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('file', help='Путь к файлу .py для анализа')
    args = p.parse_args()
    vect, clf, feat_names = load_models()
    analyze_file(args.file, vect, clf, feat_names)
