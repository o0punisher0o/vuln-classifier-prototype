import argparse, joblib

def main():
    parser = argparse.ArgumentParser(description='Vulnerability Classifier CLI')
    parser.add_argument('file', help='Путь к файлу с кодом (.java)')
    args = parser.parse_args()

    code = open(args.file, encoding='utf-8', errors='ignore').read()
    vectorizer = joblib.load('models/vectorizer.joblib')
    clf         = joblib.load('models/classifier.joblib')

    X    = vectorizer.transform([code])
    prob = clf.predict_proba(X)[0][1]
    label = 'Уязвимый' if prob > 0.5 else 'Безопасный'

    print(f'Вероятность уязвимости: {prob:.2%}')
    print(f'Классификация: {label}')

if __name__ == '__main__':
    main()


#python cli.py C:\Users\PUNISHER\PycharmProjects\vuln-classifier-prototype\dataset\juliet-test-suite\src\testcases\CWE190_Integer_Overflow\s01\CWE190_Integer_Overflow__int_console_readLine_square_81_bad.java