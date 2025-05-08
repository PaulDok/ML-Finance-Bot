import pandas as pd
from backtesting import Strategy


class MLBasedStrategy(Strategy):
    """
    A Strategy which utilizes pre-trained ML model
    """

    model = None  # ML model saved here
    features = []  # list of features
    threshold = 0.5  # Threshold for prediction of bullish growth in next tick
    y_pred_prob = None  # Model predicted probabilities

    def init(self) -> None:
        # Make a prediction by model and cutoff by threshold
        self.y_pred = (self.y_pred_prob >= self.threshold).astype(int)

        # Revealing object
        self.signal = self.I(pd.Series(self.y_pred).to_numpy)

    def next(self) -> None:
        if self.signal[-1] == 1:  # bullish
            if not self.position.is_long:
                self.buy()
        elif self.signal[-1] == 0:  # bearish
            if not self.position.is_short:
                self.sell()
