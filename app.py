from database_operations import *
import pandas as pd
import numpy as np
from collections import defaultdict
from linear_programming import optimize_stock_distribution_percentage
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False
# Flask==2.3.0及以上
app.json.ensure_ascii = False


def fetch_and_calculate(client_id):
    client_id = client_id
    # client_id = 1691005891402997762, 1665993159813017601

    # %%
    # 执行SQL查询
    (df_special_rules, df_current_stock, df_max_stock, df_onload_stock, df_min_safe, df_max_safe, df_item_attributes,
     df_item_info, df_warehouse_info) = fetch_data(client_id)

    # SECTION 当前库存处理
    df_max_stock.rename(columns={'code': 'warehouse_code'}, inplace=True)
    # 去除包含 None 值的行
    df_current_stock = df_current_stock.dropna(subset=['warehouse_code', 'item_code'])
    # 确定唯一的仓库和商品代码
    unique_warehouses = df_max_stock['warehouse_code'].unique()
    unique_items = df_current_stock['item_code'].unique()
    n_warehouses = len(unique_warehouses)
    m_goods = len(unique_items)
    # 创建一个映射，将仓库和商品代码映射到数组的索引
    warehouse_to_index = {code: index for index, code in enumerate(unique_warehouses)}
    item_to_index = {code: index for index, code in enumerate(unique_items)}
    # 创建一个 NumPy 数组
    current_stock = np.zeros((n_warehouses, m_goods), dtype=int)
    # 填充数组
    for _, row in df_current_stock.iterrows():
        warehouse_code = row['warehouse_code']
        item_code = row['item_code']
        if warehouse_code in warehouse_to_index and item_code in item_to_index:
            i = warehouse_to_index[warehouse_code]
            j = item_to_index[item_code]
            current_stock[i, j] = row['available_stock']
    # 创建一个 NumPy 数组
    current_stock_report = np.zeros((n_warehouses, m_goods), dtype=float)
    # 填充数组
    for _, row in df_current_stock.iterrows():
        warehouse_code = row['warehouse_code']
        item_code = row['item_code']
        if warehouse_code in warehouse_to_index and item_code in item_to_index:
            i = warehouse_to_index[warehouse_code]
            j = item_to_index[item_code]
            current_stock_report[i, j] = row['available_stock']
    # SECTION 最大库容处理
    # 将 'safe_stock' 列转换为数字
    df_max_stock['safe_stock'] = pd.to_numeric(df_max_stock['safe_stock'], errors='coerce')

    # df_onload_stock['onload_stock'] = pd.to_numeric(df_onload_stock['onload_stock'], errors='coerce')
    # # 将在途库存转换为字典形式，以便更容易地访问
    # onload_stock_dict = df_onload_stock.set_index('warehouse_code').to_dict()['onload_stock']

    # 确保所有的 'safe_stock' 值都是非负的
    df_max_stock['safe_stock'] = df_max_stock['safe_stock'].apply(lambda x: max(0, x))

    # 按照 warehouse_to_index 的顺序对 DataFrame 进行排序
    df_max_stock['index'] = df_max_stock['warehouse_code'].map(warehouse_to_index)
    df_max_stock.sort_values('index', inplace=True)

    # 找出所有仓库和对应的最大库存
    max_stock_per_warehouse = df_max_stock[df_max_stock['warehouse_code'].isin(unique_warehouses)]['safe_stock'].values
    small_value = 1e-9
    max_stock_per_warehouse = np.where(max_stock_per_warehouse == 0, small_value, max_stock_per_warehouse)

    # SECTION 最小安全库存处理
    # 确保 'min_stock' 列是数字格式
    df_min_safe['min_stock'] = pd.to_numeric(df_min_safe['min_stock'], errors='coerce')

    # 创建一个 NumPy 数组存放最小安全库存，维度与 current_stock 相同
    min_safety_stock = np.zeros((n_warehouses, m_goods), dtype=int)

    # 填充数组
    for _, row in df_min_safe.iterrows():
        warehouse_code = row['warehouse_code']
        item_code = row['item_code']

        # 检查 None 值并跳过
        if warehouse_code is None or item_code is None:
            continue

        # 使用之前创建的映射找到数组的索引
        i = warehouse_to_index.get(warehouse_code, -1)
        k = item_to_index.get(item_code, -1)

        # 如果找不到对应的仓库或商品代码，跳过这一行
        if i == -1 or k == -1:
            continue

        min_safety_stock[i, k] = row['min_stock']

    # SECTION： 牌号最大安全库存处理
    # 确保 'max_stock' 列是数字格式
    df_max_safe['max_stock'] = pd.to_numeric(df_max_safe['max_stock'], errors='coerce')

    # 创建一个 NumPy 数组存放最大安全库存，维度与 current_stock 相同
    max_safety_stock = np.full((n_warehouses, m_goods), max_stock_per_warehouse[:, np.newaxis], dtype=int)

    # 填充数组
    for _, row in df_max_safe.iterrows():
        warehouse_code = row['warehouse_code']
        item_code = row['item_code']

        # 检查 None 值并跳过
        if warehouse_code is None or item_code is None:
            continue

        # 使用之前创建的映射找到数组的索引
        i = warehouse_to_index.get(warehouse_code, -1)
        k = item_to_index.get(item_code, -1)

        # 如果找不到对应的仓库或商品代码，跳过这一行
        if i == -1 or k == -1:
            continue

        max_safety_stock[i, k] = row['max_stock']

    # SECTION: 特殊规则
    # 初始化一个空列表用于存储转换后的规则
    special_rules_index_list = []

    # 遍历df_special_rules的每一行
    for _, row in df_special_rules.iterrows():
        item_code = row['item_code']
        start_code = row['start_code']
        end_code = row['end_code']

        # 使用映射找到对应的索引
        item_index = item_to_index.get(item_code, None)
        start_index = warehouse_to_index.get(start_code, None)
        end_index = warehouse_to_index.get(end_code, None)

        # 如果找到了对应的索引，则添加到新的DataFrame中
        if item_index is not None and start_index is not None and end_index is not None:
            special_rules_index_list.append({
                'item_index': item_index,
                'start_index': start_index,
                'end_index': end_index
            })

    # 将列表转换为DataFrame
    df_special_rules_index = pd.DataFrame(special_rules_index_list)
    # %%
    # SECTION 调用线性规划算法 获取actions
    try:
        actions = optimize_stock_distribution_percentage(current_stock, max_stock_per_warehouse,
                                                         min_safety_stock, max_safety_stock,
                                                         df_special_rules_index)
    except Exception as e:
        print(f"Error: {e}")
        actions = None
        # 创建一部字典来存储整合后的移库方案
    consolidated_actions = defaultdict(float)
    # 移库方案整合
    for i, j, k, qty in actions:
        key = (i, j, k)
        consolidated_actions[key] += qty

    # 将 item_code 映射到 item_name
    item_code_to_name = {row['item_code']: row['item_name'] for _, row in df_item_info.iterrows()}

    # 将 warehouse_code 映射到 name
    warehouse_code_to_name = {row['code']: row['name'] for _, row in df_warehouse_info.iterrows()}

    # SECTION 成品属性处理
    # 将 item_code 映射到成品属性和属性 ID
    item_code_to_attribute = {
        row['code']: {'attr_value': row['attr_value'], 'attr_id': row['attr_id']}
        for _, row in df_item_attributes.iterrows()
    }

    # 找出所有在 item_code_to_name 中的 item_code 对应的成品属性的值和 ID
    item_code_attribute_map = {
        code: item_code_to_attribute.get(code, {'attr_value': '未知成品属性', 'attr_id': '未知ID'})
        for code in item_code_to_name.keys()
    }
    final_report = []

    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 遍历actions表
    for start_warehouse_idx, target_warehouse_idx, item_code_idx, transfer_qty in actions:
        # Fetch item_code and warehouse names directly using item_code_idx and warehouse indexes
        item_code = [k for k, v in item_to_index.items() if v == item_code_idx][0]
        item_name = item_code_to_name.get(item_code, "Unknown Item")
        start_warehouse_code = [k for k, v in warehouse_to_index.items() if v == start_warehouse_idx][0]
        target_warehouse_code = [k for k, v in warehouse_to_index.items() if v == target_warehouse_idx][0]
        start_warehouse_name = warehouse_code_to_name.get(start_warehouse_code, "Unknown Warehouse")
        target_warehouse_name = warehouse_code_to_name.get(target_warehouse_code, "Unknown Warehouse")

        # 如果找不到对应的仓库或商品代码，跳过这一行
        if start_warehouse_code is None or item_code is None:
            continue

        # 使用 warehouse_to_index 和 item_to_index 映射找到 current_stock 中的索引
        start_warehouse_idx = warehouse_to_index.get(start_warehouse_code, -1)
        item_code_idx = item_to_index.get(item_code, -1)

        # 从 current_stock 中获取起始仓库的当前库存
        current_stock_qty = current_stock_report[start_warehouse_idx, item_code_idx]

        # 从 item_code_attribute_map 中获取成品属性，type
        item_attribute_info = item_code_attribute_map.get(item_code,
                                                          {'attr_value': '未知成品属性', 'attr_id': '未知ID'})
        item_attribute = item_attribute_info['attr_value']
        if item_attribute == '1':
            item_attribute = '在制品'
        elif item_attribute == '0':
            item_attribute = '成品烟'

        item_attribute_id = item_attribute_info['attr_value']

        row_dict = {
            "send_name": start_warehouse_name,
            "send_code": int(start_warehouse_code) if isinstance(start_warehouse_code,
                                                                 np.int64) else start_warehouse_code,
            "unload_name": target_warehouse_name,
            "unload_code": int(target_warehouse_code) if isinstance(target_warehouse_code,
                                                                    np.int64) else target_warehouse_code,
            "regulation_name": item_name,
            "regulation_code": item_code,
            "cigarette_type": int(item_attribute_id) if isinstance(item_attribute_id, np.int64) else item_attribute_id,
            "cigarette_type_name": item_attribute,
            "created_time": current_time,
            "recommend_qty": int(transfer_qty) if isinstance(transfer_qty, np.int64) else transfer_qty,
            "store_qty": float(current_stock_qty) if isinstance(current_stock_qty,
                                                                (np.int64, np.float64)) else current_stock_qty,
        }

        final_report.append(row_dict)
    # %%
    return final_report


@app.route('/get_final_report', methods=['POST'])
def get_final_report():
    params = request.json['params']  # 获取传入的 JSON 中的 'params'
    client_id = params.get('client_id', '0')
    regulation_name = params.get('regulation_name', None)
    cigarette_type = params.get('cigarette_type', None)
    send_code = params.get('send_code', None)
    order_by_items = request.json.get('orderByItem', [])  # 从 JSON 获取 'orderByItem'
    try:
        final_report = fetch_and_calculate(client_id)
        # 模糊搜索
        if regulation_name:
            final_report = [x for x in final_report if regulation_name.lower() in x.get('regulation_name', '').lower()]

        if cigarette_type:
            final_report = [x for x in final_report if x.get('cigarette_type') == cigarette_type]
        if send_code:
            final_report = [x for x in final_report if x.get('send_code') == send_code]

        # 使用排序函数
        final_report = sort_by_fields(final_report, order_by_items)

        return jsonify({"code": 0, "data": final_report, "message": "操作成功。"})

    except Exception as e:
        return jsonify({"code": -1, "message": str("生成方案失败")}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
