import numpy as np
import pandas as pd

from churn.evaluate import choose_threshold
from churn.evaluate import pick_primary
from churn.evaluate import profit_per_customer

BUSINESS = {
    "value_saved": 100,
    "cost_contact": 10,
    "success_rate": 0.5
}

def make_results(scores):
    """Build a small results table: models listed from simplest to most comples"""
    return pd.DataFrame({"model": list(scores.keys()),
                         "roc_auc": (scores.values())})

def test_pick_primary_keeps_simplest_model_when_gain_is_small():
    results = make_results({"Logistic Regression": 0.846,
                            "Boosting": 0.849})
    assert pick_primary(results) == "Logistic Regression"

def test_pick_primary_switches_when_gain_is_large():
    results = make_results({"Logistic Regression": 0.616,
                            "Boosting": 0.676})
    assert pick_primary(results) == "Boosting"

def test_pick_primary_compares_against_the_current_best():
    results = make_results({"Simple": 0.800, "Shallow": 0.806, "Deep": 0.809})
    assert pick_primary(results) == "Shallow"

def test_profit_per_customer_matches_hand_calculation():
    y_true = np.array([1, 1, 0,  0, 0, 0, 0, 0, 0, 0])
    flagged = np.array([True, True, True, False, False, False, False, False, False, False])
    assert profit_per_customer(y_true, flagged, BUSINESS) == 7.0

def test_choose_threshold_finds_a_profitable_threshold():
    y_true = np.array([1] * 50 + [0] * 50)
    prob = np.where(y_true == 1, 0.9, 0.1)
    threshold, table = choose_threshold(y_true, prob, BUSINESS)
    assert 0.1 < threshold <= 0.9

def test_choose_threshold_says_contact_nobody_when_nothing_pays():
    y_true = np.zeros(100, dtype = int)
    prob = np.full(100, 0.5)
    threshold, table = choose_threshold(y_true, prob, BUSINESS)
    assert threshold == 1.01