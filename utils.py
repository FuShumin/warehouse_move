import pandas as pd
from datetime import datetime, timedelta


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
