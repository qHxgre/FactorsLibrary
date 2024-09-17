import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
import statsmodels.api as sm

def filter_pool(df: pd.DataFrame, index_comp: pd.DataFrame) -> pd.DataFrame:
    # 股票池筛选
    df = pd.merge(
        df, index_comp[['date', 'instrument']], how='inner', on=['date', 'instrument']
    )
    return df

def filter_bj(df: pd.DataFrame) -> pd.DataFrame:
    # 剔除北交所
    df = df[~df['instrument'].str.endswith('.BJ')]
    return df

def filter_new(df: pd.DataFrame, basic_info: pd.DataFrame) -> pd.DataFrame:
    # 剔除新股（即上市不满1年即252天）
    df = pd.merge(df, basic_info, how='left', on=['instrument'])
    df['list_days'] = (df['date'] - df['list_date']).dt.days
    df = df[df['list_days']>252]
    df.drop(['list_date', 'list_days'], axis=1, inplace=True)
    return df

def filter_st(df: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    # 剔除ST股票
    st = st[st['st']=='正常']
    df = pd.merge(df, st, how='inner', on=['date', 'instrument'])
    return df

def filter_suspend(df: pd.DataFrame) -> pd.DataFrame:
    # 剔除停牌股票：用成交量为0近似替代停牌
    if 'amount' not in df.columns:
        raise KeyError('Missing amount data in dataframe!')
    df = df[df['amount'] > 0 ]
    return df



def winsorize(df: pd.DataFrame, method: str, factor: str='factor') -> pd.DataFrame:
    """去极值：缩尾法/截断法/变化法

    Learning Notes
    Pandas中 transform 和 apply 的区别：
    1.输出形状：transform 会返回一个与输入数据相同大小（shape）的 DataFrame 或 Series，
        即使你的操作是聚合函数（如 mean、sum 等），transform 也会将计算结果广播到原始数
        据的大小。而 apply 则会返回一个根据操作改变的大小的 DataFrame 或 Series，如果
        你的操作是聚合函数，那么 apply 将会返回一个比原始数据小的结果。
    2. 操作类型：transform 只能接受返回 Series 或单个值的函数，而 apply 则可以接受任何函数。
    3. 应用场景：transform 常用于需要保持原始数据大小，对数据进行某种转换的场景，如数据标准化、
        填充缺失值等。而 apply 则更多用于需要改变数据大小，如聚合操作、自定义操作等。

    Args:
        df (pd.DataFrame): factor data
        method (str): winsorize method, e.g. std, percent, mad
        factor (str): factor name
    Returns:
        df (pd.DataFrame): data after winsorize
    """
    def _std(x):
        """3倍标准差"""
        n_stdevs = 3
        mean, stdev = x.mean(), x.std()
        return x.clip(
            lower = mean - n_stdevs * stdev,
            upper = mean + n_stdevs * stdev,
        )
    
    def _percent(x):
        """百分位法"""
        return x.clip(
            lower = x.quantile(0.01),
            upper = x.quantile(0.99),
        )
    
    def _mad(x):
        """ MAD - median absolute deviation
        1. 计算与中位数的绝对偏差
        2. 再计算这些偏差的中位数 - mad
        3. 阈值 = median +- n * mad
        """
        threshold = 3.5
        median = x.median()
        mad = (x - median).abs().median()
        return x.clip(
            lower = median - threshold * mad,
            upper = median + threshold * mad,
        )

    if method == 'std':
        df[factor] = df.groupby('date')[factor].transform(_std)
    elif method == 'percent':
        df[factor] = df.groupby('date')[factor].transform(_percent)
    elif method == 'mad':
        df[factor] = df.groupby('date')[factor].transform(_mad)
    else:
        raise ValueError(f'Unknown method {method}')
    return df

def standardize(df: pd.DataFrame, method: str, factor: str='factor') -> pd.DataFrame:
    """标准化
    
    Args:
        df (pd.DataFrame): factor data
        method (str): zscore, minmax
        factor (str): factor name
    Returns:
        df (pd.DataFrame): data after standardize
    """
    def _zscore(x):
        return (x - x.mean()) / x.std()
    
    def _minmax():
        return (x - x.min()) / (x.max() - x.min())
    
    if method == 'zscore':
        df[factor] = df.groupby('date')[factor].transform(_zscore)
    elif method == 'minmax':
        df[factor] = df.groupby('date')[factor].transform(_minmax)
    else:
        raise ValueError(f'Unknown method {method}')
    return df

def neutralize(
    df: pd.DataFrame,
    method: str,
    factor: str='str',
    size_data: Optional[pd.DataFrame]=None,
    industry_data: Optional[pd.DataFrame]=None
) -> pd.DataFrame:
    """中性化
    Args:
        df (pd.DataFrame): factor data
        method (str): size, industry, both
        factor (str): factor name
    """
    size_cols = []
    industry_cols = []
    if method == 'industry' or method == 'both':
        if industry_data is None:
            raise ValueError('No industry data!')
        industry_dummies = pd.get_dummies(industry_data['industry'], prefix='industry')
        industry_dummies = pd.concat([industry_data, industry_dummies], axis=1)
        industry_dummies = industry_dummies.drop(columns='industry')
        df = pd.merge(df, industry_dummies, how='left', on=['date', 'instrument'])
        industry_cols = [i for i in industry_dummies.columns.tolist() if i not in ['date','instrument']]
    if method == 'size' or method == 'both':
        if size_data is None:
            raise ValueError('No industry data!')
        df = pd.merge(df, size_data, how='left', on=['date', 'instrument'])
        size_cols = ['size']

    # TODO: 检查三个dataframe数据是否相等，或者merge起来是否有缺失值
    def check_data(df: pd.DataFrame) -> bool:
        return False
    if check_data:
        df = df.dropna(subset=['date', 'instrument'])

    independents = size_cols + industry_cols
    if len(independents) == 0:
        raise ValueError(f'the independent variables for neutralization is error. {independents}')

    def _resid(group, independents: list) -> pd.DataFrame:
        X = group[independents]
        X = sm.add_constant(X)
        model = sm.OLS(group[factor], X)
        results = model.fit()
        group['factor_neutralized'] = results.resid
        return group
    df = df.groupby('date').apply(lambda x: _resid(x, independents))
    return df
