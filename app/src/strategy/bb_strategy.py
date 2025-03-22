import pandas as pd
import talib
from backtesting import Strategy


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands-based strategy
    """

    period = 55

    def init(self) -> None:
        self.close = self.data["Close"]
        self.ma = self.I(talib.SMA, self.close, timeperiod=self.period)
        self.std = self.I(pd.Series(self.close).rolling(window=self.period).std)

        # Upper and lower Bollinger Bands
        self.upperbb_1sd = self.I(pd.Series(self.ma + (1 * self.std)).to_numpy)
        self.upperbb_2sd = self.I(pd.Series(self.ma + (2 * self.std)).to_numpy)
        self.lowerbb_1sd = self.I(pd.Series(self.ma - (1 * self.std)).to_numpy)
        self.lowerbb_2sd = self.I(pd.Series(self.ma - (2 * self.std)).to_numpy)

    def next(self) -> None:
        # Position enter logic
        if self.close[-1] > self.upperbb_1sd[-1]:
            if not self.position.is_long:
                self.buy()
        elif self.close[-1] < self.lowerbb_1sd[-1]:
            if not self.position.is_short:
                self.sell()

        # Position exit logic
        if self.position.is_long:
            if self.close[-1] < self.upperbb_1sd[-1]:
                self.position.close()
            if self.close[-1] > self.lowerbb_1sd[-1]:
                self.position.close()
