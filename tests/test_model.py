"""Tests for model training and evaluation."""

import numpy as np
import pytest

from src.models.train import quantile_loss


class TestQuantileLoss:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert quantile_loss(y, y, tau=0.2) == pytest.approx(0.0)

    def test_underestimation_penalty(self):
        """Underestimation (actual > pred) is penalized at rate tau=0.2."""
        y_true = np.array([10.0])
        y_pred = np.array([0.0])
        loss = quantile_loss(y_true, y_pred, tau=0.2)
        # error = 10, positive → tau * 10 = 2.0
        assert loss == pytest.approx(2.0)

    def test_overestimation_penalty(self):
        """Overestimation (pred > actual) is penalized at rate (1-tau)=0.8."""
        y_true = np.array([0.0])
        y_pred = np.array([10.0])
        loss = quantile_loss(y_true, y_pred, tau=0.2)
        # error = -10, negative → 0.8 * 10 = 8.0
        assert loss == pytest.approx(8.0)

    def test_asymmetry(self):
        """Overestimation should be penalized 4x more than underestimation at tau=0.2."""
        y = np.array([5.0])
        over_loss = quantile_loss(y, y + 1, tau=0.2)  # overestimate by 1
        under_loss = quantile_loss(y, y - 1, tau=0.2)  # underestimate by 1
        assert over_loss == pytest.approx(4 * under_loss)

    def test_tau_05_symmetric(self):
        """At tau=0.5, loss should be symmetric (= 0.5 * MAE)."""
        y_true = np.array([0.0])
        over = quantile_loss(y_true, np.array([1.0]), tau=0.5)
        under = quantile_loss(y_true, np.array([-1.0]), tau=0.5)
        assert over == pytest.approx(under)
