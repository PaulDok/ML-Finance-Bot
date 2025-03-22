import pandas as pd
import talib
from backtesting import Strategy


class TEMAStrategy(Strategy):
    """
    Triple EMA-based strategy
    """

    period = 55

    def init(self) -> None:
        self.close = self.data["Close"]
        ema1 = talib.EMA(self.close, timeperiod=self.period)
        ema2 = talib.EMA(ema1, timeperiod=self.period)
        ema3 = talib.EMA(ema2, timeperiod=self.period)

        # Revealing object
        self.signal = self.I(pd.Series(((3 * ema1) - (3 * ema2) + ema3)).to_numpy)

    def next(self) -> None:
        if self.close[-1] > self.signal[-1]:  # self.signal[-1] == 1:
            if not self.position.is_long:
                self.buy()
        elif self.close[-1] < self.signal[-1]:  # self.signal[-1] == -1:
            if not self.position.is_short:
                self.sell()
