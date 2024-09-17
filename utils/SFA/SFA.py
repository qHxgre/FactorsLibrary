import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Tuple, Optional
from ..SFA.DataDownload import *
from ..SFA.DataProcess import *
from ..SFA.GroupAnalysis import groupAnalyze
from ..SFA.Plotting import (plot_group_nv, plot_ic)
from ..SFA.SaveReports import (insert_text_to_md, )


class SFA:
    """
    单因子分析 (Single Factor Analysis, SFA)

    Parameters:
    params: 单因子分析的参数, dict类型
        group_num: 分组回测数量
        factor_field: 因子名称
        stock_pool: 股票池
        direction: 因子方向
        benchmark: 基准代码
        preprocess: 是否进行数据处理, dict类型
            filter_pool: 默认False, True 则根据benchmark筛选
            filter_bj: 默认False, True 表示剔除北交所的股票
            filter_new: 默认False, True 表示剔除新股
            filter_st: 默认False, True 表示剔除ST/*ST股票
            filter_suspend: 默认False, True 表示剔除停牌股票
            winsorize: 极值处理
                False 表示不做极值处理
                其他选项: std, size, percent, mad
            standardize: 标准化处理
                False 表示不做极值处理
                其他选项: zscore, minmax
            neutralize: 中性化处理
                False 表示不做极值处理
                其他选项: size, industry, both

    Example:
    factor_data: 因子数据, pandas.DataFrame格式, 形如:
            instrument	    date	        factor
        0	000001.SZ	    2017-01-03	    1.564644
        1	000001.SZ	    2017-01-04	    1.521567
    params: 单因子分析参数，形如：
    params = {
        group_num: 5,
        factor_field: 'return_5',
        stock_pool: '沪深300',
        direction: 1,
        benchmark: '000300.SH',
        preprocess: {
            'filter_pool': True,
            'filter_bj': True,
            'filter_new': True,
            'filter_st': True,
            'filter_suspend': True,
            'winsorize': 'mad',
            'standardize': 'zscore',
            'neutralize': 'both',
        }
    }
    """
    def __init__(self, params: dict, factor_data: pd.DataFrame,):
        self.print_flag = True      # 是否打印相关日志
        self.params = params        # 单因子分析参数
        self.factor_data = factor_data.rename(columns={self.params['factor_field']:'factor'})
        # 数据检查
        self.check_params(self.params)
        self.check_data_format(self.factor_data)

        self.start_date = factor_data['date'].min()
        self.end_date = factor_data['date'].max()
        self.factor_name = 'factor'
        self.factor_data['factor'] *= self.params['direction']      # 调整因子方向
        self.benchmark = self.params['benchmark']

        # SFA 中间数据
        self.factor_processed = pd.DataFrame()      # 处理后的因子数据
        self.merge_data = pd.DataFrame()            # 因子数据、收益数据等合并后的数据
        self.stk_price = pd.DataFrame()             # 股票价格
        self.stk_ret = pd.DataFrame()               # 个股收益率
        self.stk_size = pd.DataFrame()              # 市值数据
        self.industry_comp = pd.DataFrame()         # 行业成分
        self.idx_price = pd.DataFrame()             # 指数价格
        self.idx_ret = pd.DataFrame()               # 指数收益率
        self.idx_comp = pd.DataFrame()              # 指数成分
        self.basic_info = pd.DataFrame()            # 上市时间 & 退市时间
        self.st = pd.DataFrame()                    # ST记录

        self.file_path = '/home/aiuser/work/FACTORS/REPORTS/'

    def run(self):
        """主函数"""
        t0 = datetime.now()
        self.print_log(f"=====>>>>> Start factor analyze")
        self.download_data()
        t1 = datetime.now()
        self.print_log(f"=====>>>>> STEP 1: Download Data, time spent: {t1-t0}")

        self.merging_data()
        t2 = datetime.now()
        self.print_log(f"=====>>>>> STEP 2: Merge Data, time spent: {t2-t1}")

        self.preprocess()
        t3 = datetime.now()
        self.print_log(f"=====>>>>> STEP 3: Preprocess Data, time spent: {t3-t2}")

        self.group_backtest()
        t4 = datetime.now()
        self.print_log(f"=====>>>>> STEP 4: Calculate group return, time spent: {t4-t3}")

    def print_log(self, msg: str):
        if self.print_flag is True:
            print(msg)

    def check_params(self, params: dict):
        """检查传入参数的合理"""
        benchmark_list = [
            '000001.SH', '000016.SH', '000300.SH', '000688.SH', '000852.SH',
            '000903.SH', '000905.SH', '399001.SZ', '399006.SZ', '399330.SZ',
            '899050.BJ'
        ]
        if not isinstance(params['group_num'], int):
            raise ValueError("参数-分组数量 (group_num) 应为 int 类型")
        if params['direction'] not in (-1, 1):
            raise ValueError(f"参数-因子方向 (direction: {params['direction']}) 的输入值应该为 0 或者 1")
        if params['benchmark'] not in benchmark_list:
            raise ValueError(f"参数-基准指数代码 {params['benchmark']} 有问题! 请检查是否在以下列表中：{benchmark_list}")
        if not isinstance(params['preprocess']['filter_pool'], bool):
            raise ValueError(f"参数-股票池筛选 (filter_pool) 应为 bool 类型")
        if not isinstance(params['preprocess']['filter_bj'], bool):
            raise ValueError(f"参数-剔除北交所 (filter_bj) 应为 bool 类型")
        if not isinstance(params['preprocess']['filter_new'], bool):
            raise ValueError(f"参数-剔除新股 (filter_new) 应为 bool 类型")
        if not isinstance(params['preprocess']['filter_st'], bool):
            raise ValueError(f"参数-剔除ST股票 (filter_st) 应为 bool 类型")
        if not isinstance(params['preprocess']['filter_suspend'], bool):
            raise ValueError(f"参数-剔除停牌股票 (filter_suspend) 应为 bool 类型")
        if params['preprocess']['winsorize'] not in (None, 'std', 'percent', 'mad'):
            raise ValueError(
                f"参数-极值处理 (winsorize: {params['preprocess']['winsorize']}) 有问题! 请检查是否在以下列表中：('std', 'percent', 'mad')"
            )
        if params['preprocess']['standardize'] not in (None, 'zscore', 'minmax'):
            raise ValueError(
                f"参数-标准化处理 (standardize: {params['preprocess']['standardize']}) 有问题! 请检查是否在以下列表中：('zscore', 'minmax')"
            )
        if params['preprocess']['neutralize'] not in (None, 'industry', 'size', 'both'):
            raise ValueError(
                f"参数-中性化处理 (neutralize: {params['preprocess']['neutralize']}) 有问题! 请检查是否在以下列表中：('industry', 'size', 'both')"
            )

    def check_data_format(self, df: pd.DataFrame):
        """检查传入因子数据的格式"""
        # 检查date列是否是日期型类型
        if df['date'].dtype != 'datetime64[ns]':
            raise ValueError("date列的数据格式应为datetime格式")
        # 检查factor列是否是浮点型数值
        if df['factor'].dtype != 'float64':
            raise ValueError("factor列的数据格式应为浮点型")

    def input_data(
        self, stk_price: pd.DataFrame, stk_size: pd.DataFrame, industry_comp: pd.DataFrame,
        idx_price: pd.DataFrame, idx_comp: pd.DataFrame, basic_info: pd.DataFrame, st: pd.DataFrame
    ) -> None:
        """外部输入数据"""
        self.stk_price = stk_price
        self.stk_size = stk_size
        self.industry_comp = industry_comp
        self.idx_price = idx_price
        self.idx_comp = idx_comp
        self.basic_info = basic_info
        self.st = st
        self.stk_ret, self.idx_ret = self.calc_ret(self.stk_price, self.idx_price)

    def download_data(self):
        """下载数据
        # 股票价格和收益率
        # 股票市值
        # 指数价格和收益率
        # 指数成分
        # ST数据
        """
        # 股票价格
        now = datetime.now()
        self.stk_price = download_stock_price(start_date=self.start_date, end_date=self.end_date, fields=['close', 'amount'])
        self.print_log(f"download stock price data, time spent: {datetime.now() - now}")

        # 股票市值
        if self.params['preprocess']['neutralize'] in ('size', 'both'):
            now = datetime.now()
            self.stk_size = download_stock_size(start_date=self.start_date, end_date=self.end_date)
            self.print_log(f"download stock size data, time spent: {datetime.now() - now}")

        # 行业分类
        if self.params['preprocess']['neutralize'] in ('industry', 'both'):
            now = datetime.now()
            self.industry_comp = download_industry_comp(start_date=self.start_date, end_date=self.end_date)
            self.print_log(f"download industry component data, time spent: {datetime.now() - now}")
        
        # 指数价格
        now = datetime.now()
        self.idx_price = download_index_price(start_date=self.start_date, end_date=self.end_date, benchmark=self.benchmark)
        self.print_log(f"download index price data, time spent: {datetime.now() - now}")

        # 指数成分
        if self.params['preprocess']['filter_pool'] is not False:
            now = datetime.now()
            self.idx_comp = download_index_component(start_date=self.start_date, end_date=self.end_date, benchmark=self.benchmark)
            self.print_log(f"download index component, time spent: {datetime.now() - now}")

        # 上市日期
        if self.params['preprocess']['filter_new'] is True:
            now = datetime.now()
            self.basic_info = download_basic_info()
            self.print_log(f"download basic info data to calculate days after list, time spent: {datetime.now() - now}")

        # ST信息
        if self.params['preprocess']['filter_st'] is True:
            now = datetime.now()
            self.st = download_st(start_date=self.start_date, end_date=self.end_date, quick=True)
            self.print_log(f"download st, time spent: {datetime.now() - now}")

        # 计算收益率
        self.stk_ret, self.idx_ret = self.calc_ret(self.stk_price, self.idx_price)

    def calc_ret(self, stk_price: pd.DataFrame, index_price: pd.DataFrame) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
        """计算相关收益率"""
        def get_daily_ret(df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
            """计算收益率. T0的因子对应的收益率是T+1日开盘买入,T+2开盘卖出"""
            df.sort_values(['instrument', 'date'], inplace=True)
            df['stk_ret'] = df.groupby('instrument')['close'].transform(lambda x: x.pct_change(1).shift(-1))
            return df[(df['date']>=self.start_date) & (df['date']<=self.end_date)]

        def get_bm_ret(df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
            df.sort_values(['instrument', 'date'], inplace=True)
            df['index_ret'] = df.groupby('instrument')['close'].transform(lambda x: x.pct_change(1).shift(-1))
            return df[(df['date']>=self.start_date) & (df['date']<=self.end_date)]

        return (get_daily_ret(stk_price), get_bm_ret(index_price))

    def merging_data(self) -> None:
        """把各类数据 merge 起来"""
        self.merge_data = pd.merge(self.factor_data, self.stk_ret, how='left', on=['date', 'instrument'])
        self.merge_data = pd.merge(
            self.merge_data, self.idx_ret, how='left', on=['date'],
            suffixes=['', '_idx']
        )

    def preprocess(self) -> None:
        """数据预处理
        # 股票池筛选
        # 剔除北交所
        # 剔除新股
        # 剔除ST股
        # 剔除停牌股
        # 极值处理
        # 标准化处理
        # 中性化处理
        """
        df = self.merge_data.copy()
        if self.params['preprocess']['filter_pool'] is True:
            now = datetime.now()
            df = filter_pool(df=df, index_comp=self.idx_comp)
            self.print_log(f"filter stock pool, time spent: {datetime.now() - now}")
        if self.params['preprocess']['filter_bj'] is True:
            now = datetime.now()
            df = filter_bj(df=df)
            self.print_log(f"exclude stocks with BJ, time spent: {datetime.now() - now}")
        if self.params['preprocess']['filter_new'] is True:
            now = datetime.now()
            df = filter_new(df=df, basic_info=self.basic_info)
            self.print_log(f"exclude new list stocks, time spent: {datetime.now() - now}")
        if self.params['preprocess']['filter_st'] is True:
            now = datetime.now()
            df = filter_st(df=df, st=self.st)
            self.print_log(f"exclude risky stocks, time spent: {datetime.now() - now}")
        if self.params['preprocess']['filter_suspend'] is True:
            now = datetime.now()
            df = filter_suspend(df=df)
            self.print_log(f"exclude suspended stocks, time spent: {datetime.now() - now}")
        if self.params['preprocess']['winsorize'] in ('std', 'percent', 'mad'):
            now = datetime.now()
            df = winsorize(df=df, method=self.params['preprocess']['winsorize'])
            self.print_log(f"winsorize, time spent: {datetime.now() - now}")
        if self.params['preprocess']['standardize'] in ('zscore', 'minmax'):
            now = datetime.now()
            df = standardize(df=df, method=self.params['preprocess']['standardize'])
            self.print_log(f"standardize, time spent: {datetime.now() - now}")
        if self.params['preprocess']['neutralize'] in ('industry', 'size', 'both'):
            now = datetime.now()
            df = neutralize(
                df=df,
                method=self.params['preprocess']['neutralize'],
                factor=self.factor_name,
                size_data=self.stk_size,
                industry_data=self.industry_comp,
            )
            self.print_log(f"neutralize, time spent: {datetime.now() - now}")
        elif self.params['preprocess']['neutralize'] is None:
            self.print_log(f"skip neutralize")
        self.factor_processed = df

    def group_backtest(self) -> None:
        """分组回测"""
        self.group_data, gcumret_pivot, ic_stats, self.performance_summary, self.annual_rets = groupAnalyze(
            factor_data = self.factor_processed,
            group_nums = self.params['group_num'],
            benchmark_ret = self.idx_ret,
        )

        # 打印相关绩效：
        # print("分组回测完成，打印相关绩效：")
        # print(f"### 1. 分组回测绩效: \n {performance_summary}")
        # print(f"### 2. 年度绩效：\n {annual_rets}")
        # print("### 3. 分组收益曲线: ")
        # plot_group_nv(gcumret_pivot)
        # print(f"### 4. IC 绩效汇总：\n ?????")
        # print("### 5. IC 曲线图:")
        # plot_ic(ic_stats)

        # 存储文件
        self.save_result('超大单主动买入量', 'active_buy_volume_large', 'FactorPool.md')

    def save_result(self, cn_name: str, en_name: str, file_name: str) -> None:
        """自动化存储因子分析结果"""
        text = f"## {cn_name} - {en_name}"
        text += f'\n整体绩效指标\n{self.performance_summary.to_markdown(index=False)}'
        text += f'\n年度绩效指标(多头组合)\n{self.annual_rets.to_markdown(index=False)}'
        text += rf"\n分组回测曲线图\n![分组回测曲线图_{cn_name}](.\images\groupbt_{en_name}.png)"
        text += rf"\nIC曲线图\n![IC曲线图_{cn_name}](.\images\ic_{en_name}.png)"

        plot_group_nv(gcumret_pivot)
        plot_ic(ic_stats)

        insert_text_to_md(self.file_path+file_name, text, "# Financial")


