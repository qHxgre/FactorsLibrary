import numpy as np
import pandas as pd


def insert_text_to_md(file_name, text, position=None):
    """
    将指定字符串存储到指定md文件的指定位置。

    :param file_path: Markdown文件路径
    :param text: 要插入的字符串
    :param position: 要插入的位置，默认为文件末尾。可以是整数索引或特定字符串的位置。
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if position is None:       # 默认在文件末尾插入
            lines.append(text + '\n')
        elif isinstance(position, int):     # 在指定行索引位置插入
            lines.insert(position, text + '\n')
        elif isinstance(position, str):     # 在特定字符串位置插入
            for i, line in enumerate(lines):
                if position in line:
                    lines.insert(i + 1, text + '\n')
                    break
        else:
            raise ValueError("Position must be an integer, a string, or None.")
        with open(file_name, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        print("文本已成功插入。")
    except FileNotFoundError:
        print(f"文件 {file_name} 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")


def save_plot_and_generate_md(fig, file_path, md_file_path):
    """存储图片"""
    fig.savefig(file_path)
    insert_text_to_md(file_path, text, position)
