import talib
from backtesting import Strategy


class MACDStrategy(Strategy):
    """
    Moving Average Convergence-Divergence strategy
    """

    fastperiod = 12
    slowperiod = 26
    signalperiod = 9

    def init(self) -> None:
        price = self.data["Close"]

        # Revealing object
        self.signal = self.I(
            talib.MACD,
            price,
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            signalperiod=self.signalperiod,
        )

    def next(self) -> None:
        if self.signal[0][-1] > self.signal[1][-1]:  # self.signal[-1] == 1:
            if not self.position.is_long:
                self.buy()
        elif self.signal[0][-1] < self.signal[1][-1]:  # self.signal[-1] == -1:
            if not self.position.is_short:
                self.sell()
