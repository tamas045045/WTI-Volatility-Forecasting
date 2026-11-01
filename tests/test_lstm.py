"""Az LSTM-család tesztjei (gyors, kicsi beállítással).

1. A lineáris rész bitre azonos a vele azonos HAR-X előrejelzésével.
2. Reprodukálható: ugyanaz a seed -> ugyanaz az előrejelzés.
3. Nincs jövőbeli adat: a későbbi sorok elrontása nem változtatja meg a korábbi előrejelzéseket.
4. Azonos input: az LSTM és a HAR ugyanazokat az oszlopokat kapja.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from config import LSTM_PARAMS
from forecasting import load_model_data
from har import rolling_ols
from lstm import walk_forward

FAST = {**LSTM_PARAMS, "seeds": 2, "epochs_cold": 20, "epochs_warm": 5}
START, STOP = 500, 560          # csak ~12 heti újrabecslés, hogy a teszt gyors legyen


@pytest.fixture(scope="module")
def data():
    return load_model_data("L4", "Log", "RV_CJ")


def test_linear_part_equals_har(data):
    X, y, _, dates, names = data
    _, _, mu_lin = walk_forward(X, y, dates, 500, "Weekly", FAST, START, STOP)
    mu_har, _, _ = rolling_ols(X, y, dates, 500, "Weekly", names)
    np.testing.assert_allclose(mu_lin[START:STOP], mu_har[START:STOP], rtol=0, atol=1e-12)


def test_reproducible(data):
    X, y, _, dates, _ = data
    a, _, _ = walk_forward(X, y, dates, 500, "Weekly", FAST, START, STOP)
    b, _, _ = walk_forward(X, y, dates, 500, "Weekly", FAST, START, STOP)
    np.testing.assert_array_equal(a, b)


def test_no_future_data(data):
    X, y, _, dates, _ = data
    cut = 530
    clean, _, _ = walk_forward(X, y, dates, 500, "Weekly", FAST, START, cut)
    Xp, yp = X.copy(), y.copy()
    Xp[cut:] *= 7.0                 # a cut utáni jellemzők elrontása
    yp[cut - 1:] += 5.0             # a cut−1. sor célja a cut napon még nem ismert
    dirty, _, _ = walk_forward(Xp, yp, dates, 500, "Weekly", FAST, START, cut)
    # Gépi pontosságon belül egyeznek: a mátrixműveletek a memóriaelrendezéstől függően ~1e-15
    # kerekítési zajt adnak (egy puszta másolat is), a valódi szivárgás nagyságrendekkel nagyobb lenne.
    np.testing.assert_allclose(clean[START:cut], dirty[START:cut], rtol=0, atol=1e-10)


def test_same_columns_as_har():
    from config import level_columns
    for level in ["L1", "L2", "L3", "L4"]:
        _, _, _, _, cols = load_model_data(level, "Log", "RV")
        assert cols == level_columns(level, "RV")
