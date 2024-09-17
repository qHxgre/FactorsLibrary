import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt


def plot_group_nv(df: pd.DataFrame, filepath: str) -> None:
    """画 分组收益率"""
    # 设置图表大小
    plt.figure(figsize=(14,7))

    cmap = plt.cm.Blues
    colors = cmap([0.1 + 0.6/(len(df.columns)-1)*i for i in range(len(df.columns)-2)])

    for i, group in enumerate(df.columns):
        if group == "ls":
            plt.plot(df.index, df[group], label=group, color="red")
        elif group == "bm":
            plt.plot(df.index, df[group], label=group, color="green")
        else:
            plt.plot(df.index, df[group], label=group, color=colors[i])

    plt.title("分组回测收益率")
    plt.xlabel("日期")
    plt.ylabel("收益率")
    plt.legend()
    plt.show()

    fig.savefig(filepath)

def plot_ic(df: pd.DataFrame) -> None:
    """画IC图"""
    # 设置图表大小和标题
    plt.figure(figsize=(14, 7))
    plt.title("IC分析")

    plt.bar(df['date'], df['ic'], color='blue', label='IC')
    plt.plot(df['date'], df['ic_cumsum'], color='red', label='IC Cumulative Sum')
    plt.plot(df['date'], df['ic_roll_ma'], color='green', alpha=0.5, label='IC Rolling Mean')
    plt.fill_between(df['date'], df['ic_roll_ma'], color='green', alpha=0.65)

    plt.legend(loc='upper left')
    plt.xlabel('日期')
    plt.ylabel('IC值')
    plt.xticks(rotation=45)
    plt.show()
