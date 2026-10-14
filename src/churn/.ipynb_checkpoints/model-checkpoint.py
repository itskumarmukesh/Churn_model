from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocess(X):
    """Number: fill gaps with the median and scale. Text columns: one-hot encode; categories rarer than 1% are grouped together. Everything here is learned from the training part only, because it sits inside a pipeline"""

    num_cols = X.select_dtypes(include = "number").columns.tolist()
    cat_cols = X.select_dtypes(exclude = "number").columns.tolist()

    numeric_steps = make_pipeline(SimpleImputer(strategy = "median", add_indicator = True), StandardScaler())
    categorical_step = OneHotEncoder(handle_unknown = "infrequent_if_exist", min_frequency = 0.01, sparse_output = False)

    return ColumnTransformer([("num", numeric_steps, num_cols), ("cat", categorical_step, cat_cols)])

def build_models(X, random_state = 42):
    """Candidate models, listed from the simplest to the most complex. The best settings differ between datasets, so we offer several and let cross - validation choose"""

    return {
        "Logistic Regression": make_pipeline(
            build_preprocess(X),
            LogisticRegression(max_iter = 2000)),
        "Boosting (shallow)": make_pipeline(
            build_preprocess(X),
            HistGradientBoostingClassifier(learning_rate = 0.05, max_iter = 200, max_depth = 2, max_leaf_nodes = 31, min_samples_leaf = 20, l2_regularization = 5, random_state = random_state)),
        "Boosting (deep)": make_pipeline(
            build_preprocess(X),
            HistGradientBoostingClassifier(learning_rate = 0.05, max_iter = 200, random_state = random_state)),
    }