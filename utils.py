import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class DatabaseFetchError(Exception):
    """异常类，表示从数据库获取数据时发生的错误"""

    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception
        self.message = message

    def __str__(self):
        return f"DatabaseFetchError: {self.message}" + \
            (f" (Original exception: {self.original_exception})" if self.original_exception else "")


class DataValidationError(Exception):
    """异常类，表示数据校验失败"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"DataValidationError: {self.message}"


class OptimizationError(Exception):
    """异常类，表示优化过程中的错误"""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self):
        return f"OptimizationError: {self.message}" + \
            (f" (Details: {self.details})" if self.details else "")


def summarize_production_plans(df_produce, df_produce_mapping, day_range=1):
    """
    清洗生产计划数据

    :param df_produce: DataFrame 生产计划预入库
    :param df_produce_mapping: DataFrame 生产区域-仓库编码 映射关系
    :param day_range: 生产计划生效日期范围 (例如day_range=2，表示结束时间在今天和明天有效).
    :return: DataFrame 清洗整合的数据，包括每个item_code的qty以及汇总的expected_incoming_qty
    """
    # 数据整合
    df_combined = pd.merge(df_produce, df_produce_mapping, left_on='production_area', right_on='name')

    # 转换时间格式
    df_combined['end_time'] = pd.to_datetime(df_combined['end_time'])

    # 获取当前日期
    current_date = datetime.now().date()

    # 计算未来几天的日期范围
    future_dates = [current_date + timedelta(days=i) for i in range(day_range)]

    # 找出结束时间在指定日期范围内的生产计划
    plans_in_range = df_combined[df_combined['end_time'].dt.date.isin(future_dates)]

    # 汇总预期入库量，并聚合 item_code 及其对应的 qty
    expected_incoming = plans_in_range.groupby(['end_time', 'dict_code', 'item_code']).agg({
        'plan_qty': 'sum'
    }).reset_index()

    # 重命名列以反映数据含义
    expected_incoming.rename(
        columns={'end_time': 'date', 'dict_code': 'warehouse_code', 'plan_qty': 'item_qty'}, inplace=True)

    return expected_incoming


def adjust_max_stock(expected_incoming, warehouse_to_index, max_stock_per_warehouse):
    """
    将仓库库容减去对应的预入库量，以预留出仓库空间。
    """
    for index, row in expected_incoming.iterrows():
        warehouse_code = row['warehouse_code']
        item_qty = float(row['item_qty'])

        if warehouse_code in warehouse_to_index:
            # 找到仓库对应的索引
            warehouse_index = warehouse_to_index[warehouse_code]

            # 减去预期的入库量
            max_stock_per_warehouse[warehouse_index] -= item_qty

    return max_stock_per_warehouse


def update_safety_stock(expected_incoming, warehouse_to_index, item_to_index, max_safety_stock, min_safety_stock):
    """
    根据预入库量 更新对应货物的最大安全库存，但是不低于其最小安全库存
    """
    # 遍历 expected_incoming 来更新 max_safety_stock
    for index, row in expected_incoming.iterrows():
        warehouse_code = row['warehouse_code']
        item_code = row['item_code']
        item_qty = float(row['item_qty'])

        if warehouse_code in warehouse_to_index and item_code in item_to_index:
            # 找到仓库和商品对应的索引
            warehouse_index = warehouse_to_index[warehouse_code]
            item_index = item_to_index[item_code]

            # 减去预期的入库量，但不低于 min_safety_stock
            updated_stock = max(max_safety_stock[warehouse_index, item_index] - item_qty,
                                min_safety_stock[warehouse_index, item_index])
            max_safety_stock[warehouse_index, item_index] = updated_stock

    return max_safety_stock


def check_for_errors(current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock, df_warehouse_info,
                     df_item_info, item_to_index, warehouse_to_index):
    """
    数据校验
    """
    n_warehouses, m_goods = current_stock.shape
    errors = []

    for i in range(n_warehouses):
        for k in range(m_goods):
            if min_safety_stock[i, k] > max_stock_per_warehouse[i]:
                warehouse_i = find_name_by_index(df_warehouse_info, warehouse_to_index, i)
                good_k = find_name_by_index(df_item_info, item_to_index, k)
                errors.append(f"仓库{warehouse_i}中牌号{good_k} 的最小安全库存超出最大库容！")

    if any(stock < 0 for stock in max_stock_per_warehouse):
        errors.append("请检查最大安全库存设置不为负数")

    if any(stock < 0 for stock in min_safety_stock.flatten()):
        errors.append("请检查最小安全库存设置不为负数")

    if n_warehouses == 0 or m_goods == 0:
        errors.append("请检查仓库数量或牌号数量不能为零")

    for k in range(m_goods):
        total_stock = np.sum(current_stock[:, k])
        total_min_safety = np.sum(min_safety_stock[:, k])
        total_max_stock = np.sum(max_safety_stock[:, k])
        good_k = find_name_by_index(df_item_info, item_to_index, k)

        if total_stock < total_min_safety:
            errors.append(f"牌号{good_k} 的当前总库存低于最小安全库存总和！")
        if total_stock > total_max_stock:
            errors.append(f"牌号{good_k} 的当前总库存超出最大安全库存总和！")

    # 检查所有货物的总库存是否超过所有仓库的最大库容之和
    if np.sum(current_stock) > np.sum(max_stock_per_warehouse):
        errors.append("所有货物的总库存超出所有仓库的最大库容总和！")

    if errors:
        # Join all errors into a single message
        error_message = "error" + "; ".join(errors)
        raise DataValidationError(error_message)
    return None


def find_name_by_index(df, code_to_index, index):
    """
    一个通用函数，用于根据给定索引在 DataFrame 中查找名称。
    假设 DataFrame 的第一列为编码，第二列为名称。

    参数:
    df (pd.DataFrame): 包含编码和名称的 DataFrame。
    code_to_index (dict): 将编码映射到索引的字典。
    index (int): 要查找名称的索引。

    返回:
    str: 根据给定索引找到的名称，如果未找到则返回错误信息。
    """
    # 将 code_to_index 字典反转，映射索引到编码
    index_to_code = {v: k for k, v in code_to_index.items()}

    # 根据给定的索引找到对应的编码
    code = index_to_code.get(index)

    if code:
        # 根据编码找到对应的名称
        name = df.loc[df.iloc[:, 0] == code, df.columns[1]].values
        if name.size > 0:
            return name[0]
        else:
            return "在给定索引的编码对应的名称未找到。"
    else:
        return "在编码到索引的映射中未找到对应的索引。"
