import dai
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime, timedelta

def download_stock_price(instruments: list=[], start_date: str='2023-01-01', end_date: str='2024-01-01', fields: List[str]=['close', 'amount']) -> pd.DataFrame:
    """获取股票价格和收益率"""
    fields = ','.join([i for i in fields])
    ten_days_before_sd = (pd.Timestamp(start_date) - timedelta(days=10)).strftime('%Y-%m-%d')
    filter_ins = f"instrument IN {tuple(instruments)}" if instruments else "1=1"
    df = dai.query(f"""
        SELECT date, instrument, {fields}
        FROM cn_stock_bar1d
        WHERE {filter_ins}
        AND date between '{ten_days_before_sd}' and '{end_date}'
    ;""").df()
    return df

def download_stock_size(instruments: list=[], start_date: str='2023-01-01', end_date: str='2024-01-01') -> pd.DataFrame:
    """获取股票市值"""
    filter_ins = f"instrument IN {tuple(instruments)}" if instruments else "1=1"
    df = dai.query(f"""
        SELECT date, instrument, total_market_cap as size
        FROM cn_stock_valuation
        WHERE {filter_ins}
        AND date between '{start_date}' and '{end_date}' 
    ;""").df()
    return df

def download_industry_comp(start_date: str='2023-01-01', end_date: str='2024-01-01') -> pd.DataFrame:
    """下载行业成分数据"""
    df = dai.query(f"""
        SELECT date, instrument, industry_level1_name as industry
        FROM cn_stock_industry_component
        WHERE industry = 'sw2021'
        AND date between '{start_date}' and '{end_date}'
    """).df()
    return df

def download_index_price(start_date: str='2023-01-01', end_date: str='2024-01-01', benchmark: str='000300.SH') -> pd.DataFrame:
    """获取指数价格和收益率"""
    ten_days_before_sd = (pd.Timestamp(start_date) - timedelta(days=10)).strftime('%Y-%m-%d')
    df = dai.query(f"""
        SELECT date, instrument, close
        FROM cn_stock_index_bar1d
        WHERE date between '{ten_days_before_sd}' and '{end_date}'
        AND instrument == '{benchmark}'
    ;""").df()
    return df

def download_index_component(start_date: str='2023-01-01', end_date: str='2024-01-01', benchmark: str='000300.SH') -> pd.DataFrame:
    """下载指数成分数据"""
    df = dai.query(f"""
        SELECT
            date,
            member_code as instrument,
            instrument as index_code,
        FROM cn_stock_index_component
        WHERE date between '{start_date}' and '{end_date}' 
        AND instrument == '{benchmark}'
    ;""").df()
    return df

def download_basic_info() -> pd.DataFrame:
    """下载上市时间等基本信息数据"""
    df = dai.query("SELECT instrument, list_date FROM cn_stock_basic_info").df()
    return df

def download_st(instruments: List=[], start_date: str='2023-01-01', end_date: str='2024-01-01', quick: bool=False) -> pd.DataFrame:
    """ST状态"""
    if quick is True:
        df = dai.query(f"""
            SELECT date, instrument, is_risk_warning as st
            FROM cn_stock_status
            WHERE date between '{start_date}' and '{end_date}' 
        """).df()
        df['st'] = df['st'].map({0: '正常', 1: '风险'})
        return df 

    # 获取ST状态数据，并把相关数据展开
    st = dai.query("select * from cn_stock_st").df()
    st = st[st['special_treatment']!='']
    st['special_treatment'] = st['special_treatment'].astype(str)
    st['special_treatment'] = st['special_treatment'].str.split(";")
    st = st.explode("special_treatment")
    st[['st', 'date']] = st['special_treatment'].str.split(':', expand=True)
    st['date'] = pd.to_datetime(st['date'], format='%Y%m%d')
    st['st'] = st['st'].replace({'摘帽': '正常', '摘*': 'ST'})

    # 拼接到daily_instrument中
    daily_ins = dai.query("SELECT date, instrument FROM cn_stock_instruments", full_db_scan=True).df()
    df = pd.merge(daily_ins, st[['date', 'instrument', 'st']], how='left', on=['date', 'instrument'])
    def sort_fill(group: pd.DataFrame) -> pd.DataFrame:
        group.sort_values('date', inplace=True)
        group.fillna(method='ffill', inplace=True)
        return group
    df = df.groupby('instrument').apply(sort_fill)
    df.reset_index(drop=True, inplace=True)
    df['st'] = df['st'].fillna('正常')
    return df
