import talib
from backtesting import Strategy
from backtesting.lib import crossover


class SmaCross(Strategy):
    """
    Basis Simple Moving Average crossover strategy
    """

    ma_fast_periods = 10
    ma_slow_periods = 20

    def init(self) -> None:
        price = self.data["Close"]
        self.ma1 = self.I(talib.SMA, price, self.ma_fast_periods)
        self.ma2 = self.I(talib.SMA, price, self.ma_slow_periods)

    def next(self) -> None:
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma2):
            self.sell()
