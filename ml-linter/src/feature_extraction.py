import ast, asttokens
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

DANGEROUS_FUNCS = {'eval','exec','os.system','subprocess','pickle.load'}

def extract_ast_features(code):
    """Считает, сколько раз встречаются опасные функции в AST."""
    try:
        atok = asttokens.ASTTokens(code, parse=True)
        counter = {f:0 for f in DANGEROUS_FUNCS}
        for node in ast.walk(atok.tree):
            if isinstance(node, ast.Call):
                func_name = ''
                if isinstance(node.func, ast.Attribute):
                    func_name = f"{node.func.value.id}.{node.func.attr}"
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name in counter:
                    counter[func_name] += 1
        return [counter[f] for f in DANGEROUS_FUNCS]
    except Exception:
        return [0]*len(DANGEROUS_FUNCS)

def build_vectorizer(corpus, max_features=500):
    vect = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=max_features)
    vect.fit(corpus)
    return vect

def transform_features(codes, vect):
    tfidf = vect.transform(codes).toarray()
    ast_feats = np.array([extract_ast_features(c) for c in codes])
    return np.hstack([tfidf, ast_feats])

if __name__ == '__main__':
    # Пример
    from data_preprocessing import load_dataset
    X, y = load_dataset()
    vect = build_vectorizer(X)
    X_vect = transform_features(X, vect)
    print('Признаков на образец:', X_vect.shape[1])
    # Сохраняем векторизатор и имена признаков
    joblib.dump(vect, 'ml-linter/models/vectorizer.joblib')
    joblib.dump(list(vect.get_feature_names_out()) + list(DANGEROUS_FUNCS), 'ml-linter/models/feature_names.pkl')
