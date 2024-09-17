import dai
import empyrical
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


def groupAnalyze(factor_data: pd.DataFrame, group_nums: int, benchmark_ret: pd.DataFrame) -> Tuple[Union[pd.Series, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gtype_map = {'SHORT': str(0), 'LONG': str(group_nums-1), 'LONGSHORT': 'ls'}

    """第一步：分组"""
    def cut(group, group_num=10):
        """分组"""
        group['group'] = pd.qcut(group['factor'], q=group_num, labels=False, duplicates='drop')
        group = group.dropna(subset=['group'], how='any')
        group['group'] =  group['group'].apply(int).apply(str)
        return group
    group_data = factor_data.groupby(
        'date', group_keys=False
    ).apply(cut, group_num=group_nums)


    """第二步：计算分组收益率"""
    # 计算每个分组每天的收益率
    groupret_data = group_data[['date','group','stk_ret']].groupby(
        ['date','group'], group_keys=False
    ).apply(lambda x: np.nanmean(x)).reset_index()
    groupret_data.rename(columns={0:'group_ret'}, inplace=True)

    # 利用pivot计算每个分组的收益率序列
    groupret_pivot = groupret_data.pivot(index='date', values='group_ret', columns='group')
    groupret_pivot[gtype_map['LONGSHORT']] = groupret_pivot[gtype_map['LONG']] - groupret_pivot[gtype_map['SHORT']]

    # 基准收益率
    bm_ret = benchmark_ret.set_index('date')
    groupret_pivot['bm'] = bm_ret['index_ret']
    groupret_pivot = groupret_pivot

    # 计算单利下的累积收益率
    gcumret_pivot = groupret_pivot.cumsum().round(4)


    """第三步：分组回测"""
    # 计算整个数据的 IC 情况
    df = group_data[['date', 'instrument', 'factor', 'stk_ret']]
    def _cal_ic(group: pd.DataFrame) -> float:
        return group['stk_ret'].corr(group['factor'], method="spearman")
    ic_stats = df.groupby('date', group_keys=False).apply(lambda x: _cal_ic(x)).reset_index()
    ic_stats.rename(columns={0:'ic'}, inplace=True)
    ic_stats['ic_cumsum'] = ic_stats['ic'].cumsum()
    ic_stats['ic_roll_ma'] = ic_stats['ic'].rolling(22).mean()

    summary_df = []
    for _type in ['LONG', 'SHORT', 'LONGSHORT']:   
        _dict = {}
        _dict.update(get_group_performance(groupret_pivot, gtype_map[_type]))
        _dict.update(get_ic_performance(group_data, gtype_map, _type))
        _dict.update(get_turnover_performance(group_data, gtype_map, _type))
        df = pd.DataFrame.from_dict(_dict, orient='index', columns=['value']).T
        df['portfolio'] = _type 
        summary_df.append(df)
    summary_df = pd.concat(summary_df, axis=0)
    summary_df.index = range(len(summary_df))
    performance_summary = summary_df.round(4)

    # 计算年度收益
    data = groupret_pivot.copy()
    annual_returns = data.groupby(data.index.year).sum()
    for i in [i for i in annual_returns.columns if i != "bm"]:
        annual_returns[i] -= annual_returns["bm"]
    annual_rets = annual_returns.round(4)
    return group_data, gcumret_pivot, ic_stats, performance_summary, annual_rets

def get_group_performance(groupret_pivot: pd.DataFrame, group_type: str='LS') -> dict:
    """获取分组收益率的绩效，绩效如下：
    * return_ratio: 总收益率,
    * annual_return: 年收益率,
    * excess_annual_return: 超额收益率,
    * sharpe: 夏普比率,
    * volatility: 波动率,
    * max_drawdown: 最大回撤率,
    * information: 信息比率,
    * win_percent: 胜率,
    * trading_days: 交易天数,
    """
    bm = groupret_pivot['bm']               # 基准收益率
    group = groupret_pivot[group_type]      # 分组收益率
    return {
        'return_ratio': group.sum(),
        'annual_return': group.sum() * 242 / len(group),
        'excess_return': (group-bm).sum(),
        'excess_annual_return': (group-bm).sum() * 242 / len((group-bm)),
        'sharpe': empyrical.sharpe_ratio(group, 0.035/242),
        'volatility': empyrical.annual_volatility(group),
        'max_drawdown': empyrical.max_drawdown(group),
        'information': group.mean() / group.std(),
        'win_percent': len(group[group>0]) / len(group),
        'trading_days': len(group),
    }

def get_ic_performance(group_data: Union[pd.Series, pd.DataFrame], gtype_map: dict, group_type: str) -> dict:
    """计算IC值"""
    def _cal_ic(group: pd.DataFrame) -> float:
        return group['stk_ret'].corr(group['factor'], method="spearman")

    def _get_ic_stats(ic_data: pd.DataFrame) -> dict:
        return {
            'ic_mean': np.nanmean(ic_data['group_ic']),                                         # IC均值
            'ic_above': ic_data[ic_data["group_ic"] > 0.02].shape[0] / ic_data.shape[0],        # IC均值大于0.02的比例
            'ic_abs_mean': np.nanmean(ic_data['abs_ic']),                                       # IC绝对值均值
            'ic_abs_above': ic_data[ic_data["abs_ic"] > 0.02].shape[0] / ic_data.shape[0],      # IC绝对值大于0.02的比例
            'ir': np.nanmean(ic_data['group_ic']) / np.nanstd(ic_data['group_ic']),             # IR
            'ic_5': ic_data['group_ic'].tail(3).mean(),                                         # 过去5个交易日的ic均值
            'ic_22': ic_data['group_ic'].tail(21).mean(),                                       # 过去22个交易日的ic均值
            'ic_252': ic_data['group_ic'].tail(252).mean(),                                     # 过去252个交易日的ic均值
        }

    # 计算分组的 IC 情况
    if group_type == 'LONG':
        df = group_data[group_data['group'] == gtype_map['LONG']][['date', 'instrument', 'factor', 'stk_ret']]
    elif group_type == 'SHORT':
        df = group_data[group_data['group'] == gtype_map['SHORT']][['date', 'instrument', 'factor', 'stk_ret']]
    elif group_type == 'LONGSHORT':
        df = group_data[group_data['group'].isin([gtype_map['LONG'], gtype_map['SHORT']])][['date', 'instrument', 'factor', 'stk_ret']]
    else:
        df = pd.DataFrame()
    ic_data = df.groupby('date', group_keys=False).apply(lambda x: _cal_ic(x)).reset_index()
    ic_data = ic_data.rename(columns={0:'group_ic'}).dropna()
    ic_data["abs_ic"] = ic_data["group_ic"].abs()
    ic_perf = _get_ic_stats(ic_data)
    return ic_perf

def get_turnover_performance(group_data: pd.DataFrame, gtype_map: dict, group_type: str) -> dict:
    """换手率分析"""
    def _get_turnover(df: pd.DataFrame) -> float:
        grouped = df.groupby('date')['instrument'].apply(set).to_dict()
        dates = df["date"].unique().tolist()
        dates.sort()
        overlap = []
        for i in range(1, len(dates)):
            previous_date = dates[i-1]
            current_date = dates[i]
            overlap_count = len(grouped[current_date].intersection(
                grouped[previous_date]))
            overlap.append((current_date, overlap_count))
        overlap_df = pd.DataFrame(overlap, columns=['date', 'turnover'])
        return np.nanmean(overlap_df['turnover'])

    df = group_data[["date", "instrument", "factor", "group"]]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    if group_type == 'LONG':
        df = df[df['group'] == gtype_map['LONG']]
        return {'turnover': _get_turnover(df)}
    elif group_type == 'SHORT':
        df = df[df['group'] == gtype_map['SHORT']]
        return {'turnover': _get_turnover(df)}
    elif group_type == 'LONGSHORT':
        long_df = df[df['group'] == gtype_map['LONG']]
        short_df = df[df['group'] == gtype_map['SHORT']]
        return {'turnover': _get_turnover(long_df) + _get_turnover(short_df)}

