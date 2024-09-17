import dai
import numpy as np
import pandas as pd

class RegressionAnalysis:
    """回归分析
    # 最小二乘法：OLS
    # 
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.rebalance = 1     # T期

        self.data = self.preprocess(df)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 因为，在每个 t 期都要进行截面回归，因此不能用日频数据进行回归
        # 所以，要把原数据处理成 t 期数据，包括收益率
        df["return"] = df.groupby("instrument")["close"].apply(
            lambda x: x.shift((-1)*self.rebalance)/x-1)
        df["days_from_sd"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["date"].min())).dt.days
        df["rebalance"] = 0
        df.loc[df["days_from_sd"] % (rebalance) == 0, "rebalance"] = 1
        return df[df["rebalance"]==1][["date", "instrument", "factor", "return"]]
        

    def ols(self) -> None:
        """最小二乘法"""
        def _ols(self, data: pd.DataFrame):
            x
            pass
        df.groupby("date").apply(lambda x: _ols(x))
