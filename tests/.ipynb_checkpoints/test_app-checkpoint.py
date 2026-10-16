from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


@pytest.mark.skipif(not (ROOT / "models" / "telco_model.joblib").exists(),
                    reason="run python -m churn.export first")
def test_app_runs_and_ranks_the_examples():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception

    options = list(at.selectbox[0].options)
    at.selectbox[0].select(options[1]).run()          # high-risk example
    high = float(at.metric[0].value.strip("%"))
    at.selectbox[0].select(options[2]).run()          # low-risk example
    low = float(at.metric[0].value.strip("%"))
    assert not at.exception
    assert high > low