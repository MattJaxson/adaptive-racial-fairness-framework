try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
except ImportError:
    print("Fairlearn is not installed. Please run: pip install fairlearn")
    exit(1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def adversarial_fairness_pipeline(data, features, outcome, sensitive_attr):
    """
    Run adversarial fairness pipeline on provided data.
    """
    X = data[features]
    y = data[outcome].apply(lambda x: 1 if x == 'Yes' else 0)
    s = data[sensitive_attr]

    if s.isnull().any():
        print("Warning: Missing sensitive attributes.")

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=0.3, random_state=42
    )

    estimator = LogisticRegression(solver='liblinear')
    mitigator = ExponentiatedGradient(estimator, constraints=DemographicParity())
    mitigator.fit(X_train, y_train, sensitive_features=s_train)

    y_pred = mitigator.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)
