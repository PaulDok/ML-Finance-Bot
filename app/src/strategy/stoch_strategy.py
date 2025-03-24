import talib
from backtesting import Strategy


class STOCHStrategy(Strategy):
    """
    Stochastic Oscillator-based strategy
    """

    fastk_period = 14
    slowk_period = 7
    slowk_matype = 0
    slowd_period = 7
    slowd_matype = 0

    def init(self) -> None:
        high = self.data["High"]
        low = self.data["Low"]
        close = self.data["Close"]

        # Revealing object
        self.signal = self.I(
            talib.STOCH,
            high,
            low,
            close,
            fastk_period=self.fastk_period,
            slowk_period=self.slowk_period,
            slowk_matype=self.slowk_matype,
            slowd_period=self.slowd_period,
            slowd_matype=self.slowd_matype,
        )

    def next(self) -> None:
        if self.signal[0][-1] > self.signal[1][-1]:  # self.signal[-1] == 1:
            if not self.position.is_long:
                self.buy()
        elif self.signal[0][-1] < self.signal[1][-1]:  # self.signal[-1] == -1:
            if not self.position.is_short:
                self.sell()
