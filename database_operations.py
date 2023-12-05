import os
import pymysql
import pandas as pd


def get_database_connection():
    """Establish and return a database connection."""
    host = os.environ.get('DB_HOST', '10.1.20.196')
    port = int(os.environ.get('DB_PORT', 3306))
    user = os.environ.get('DB_USER', 'root')
    password = os.environ.get('DB_PASSWORD', 'nti56.com')
    database = os.environ.get('DB_NAME', 'lcs-sit')

    return pymysql.connect(host=host, port=port, user=user, password=password, database=database)


def execute_query(cursor, query, params=None):
    """Execute a SQL query and return the fetched data as a DataFrame."""
    cursor.execute(query, params)
    rows = cursor.fetchall()
    fields = [i[0] for i in cursor.description]
    return pd.DataFrame(rows, columns=fields)


# 用于处理排序参数的函数
def sort_by_fields(data, order_by_items):
    for order_item in reversed(order_by_items):
        field = order_item.get('field')
        order = order_item.get('order', 'asc')

        # Make sure that the field exists in all dictionaries
        if all(field in d for d in data):
            data.sort(key=lambda x: x[field], reverse=(order == 'desc'))
        else:
            print(f"Field '{field}' not found in all dictionaries.")
    return data


def fetch_data(client_id):
    with get_database_connection() as conn:
        with conn.cursor() as cursor:
            query = (
                "SELECT material_code AS item_code, start_code, end_code FROM lcs_dispatch_cp_algorithmic_rule WHERE "
                "deleted = 0 "
                "AND client_id = %s;")
            df_special_rules = execute_query(cursor, query, (client_id,))  # 特殊规则

            query = "SELECT warehouse_code, item_code, available_stock FROM stock WHERE client_id = %s;"
            df_current_stock = execute_query(cursor, query, (client_id,))  # 当前库存

            query = ("SELECT code, safe_stock FROM stock_warehouse_info WHERE client_id = %s AND is_delete=0 AND "
                     "safe_stock IS NOT NULL;")
            df_max_stock = execute_query(cursor, query, (client_id,))  # 最大库存

            query = ("SELECT warehouse_code, warehouse_name, item_code, item_name, onload_stock FROM stock "
                     "WHERE onload_stock > 0 AND client_id = %s;")
            df_onload_stock = execute_query(cursor, query, (client_id,))  # 在途库存

            query = """
                SELECT t.warehouse_code, t.item_code, t.min_stock
                FROM (
                    SELECT c.*, i.name AS warehouse_name
                    FROM stock_safe_config c
                    LEFT JOIN stock_warehouse_info i ON i.code = c.warehouse_code
                    WHERE c.deleted = 0
                ) t
                WHERE t.deleted = 0 AND client_id = %s;"""
            df_min_safe = execute_query(cursor, query, (client_id,))  # 牌号最小安全库存

            query = """
                SELECT t.warehouse_code, t.item_code, t.max_stock
                FROM (
                    SELECT c.*, i.name AS warehouse_name
                    FROM stock_safe_config c
                    LEFT JOIN stock_warehouse_info i ON i.code = c.warehouse_code
                    WHERE c.deleted = 0
                ) t
                WHERE t.deleted = 0 AND client_id = %s;"""
            df_max_safe = execute_query(cursor, query, (client_id,))  # 牌号最大安全库存

            query = """
                SELECT
                    material_info.code, material_info_attr.attr_value, material_info_attr.attr_id
                FROM
                    material_info
                JOIN material_info_attr ON material_info.id = material_info_attr.material_id
                JOIN material_attr ON material_info_attr.attr_id = material_attr.id
                WHERE
                    material_attr.name = '卷烟类型' AND material_info.client_id = %s;
            """
            df_item_attributes = execute_query(cursor, query, (client_id,))  # 成品属性

            query = ("SELECT code AS item_code, name AS item_name FROM material_info WHERE client_id = %s AND "
                     "is_delete = 0;")
            df_item_info = execute_query(cursor, query, (client_id,))  # 物品信息

            # 从 warehouse_info 表中找到 code（即 warehouse_code）对应的 name
            query = "SELECT code, name FROM stock_warehouse_info WHERE client_id = %s AND is_delete = 0;"
            df_warehouse_info = execute_query(cursor, query, (client_id,))  # 仓库信息
    return (df_special_rules, df_current_stock, df_max_stock, df_onload_stock, df_min_safe, df_max_safe,
            df_item_attributes, df_item_info, df_warehouse_info)
