import numpy as np
from gym import spaces


def reset_warehouse(seed=None, options=None):
    # 随机生成仓库数量 n 和牌号数量 m
    n_warehouses = np.random.randint(2, 10)
    m_goods = np.random.randint(2, 25)

    # 为每个仓库随机生成最大库存上限，不足10的最大库存补1
    max_stock_array = np.ones(10, dtype=int)
    max_stock_array[:n_warehouses] = np.random.randint(3_000, 10_000, size=n_warehouses)
    max_stock_per_warehouse = max_stock_array

    # 初始化每个仓库每个牌号的当前库存为 0
    current_stock = np.zeros((10, 25))

    # 计算所有仓库最大库存上限的总和，并预留一定数量的空间
    total_max_capacity = np.sum(max_stock_per_warehouse)
    reserved_space = 0.3 * total_max_capacity
    adjusted_max_capacity = total_max_capacity - reserved_space

    # 计算每个仓库的新的最大库存上限
    new_max_stock_per_warehouse = (max_stock_per_warehouse / total_max_capacity) * adjusted_max_capacity
    new_max_stock_per_warehouse = new_max_stock_per_warehouse.astype(int)

    # 为每个仓库随机生成当前库存
    for i in range(n_warehouses):
        alpha = 0.5  # 你可以调整这个值
        proportions = np.random.dirichlet(alpha * np.ones(m_goods), size=1)[0]

        # 根据比例和仓库的新的最大库存上限来计算每个牌号的库存
        current_stock[i, :m_goods] = np.round(proportions * new_max_stock_per_warehouse[i]).astype(int)

        # 检查并确保总库存不超过新的上限
        total_stock = np.sum(current_stock[i, :])
        if total_stock > new_max_stock_per_warehouse[i]:
            excess = total_stock - new_max_stock_per_warehouse[i]
            while excess > 0:
                for j in range(m_goods):
                    if current_stock[i, j] > 0 and excess > 0:
                        current_stock[i, j] -= 1
                        excess -= 1

    # 初始化预入库量
    zero_column = np.zeros((current_stock.shape[0], 1), dtype=int)
    current_stock = np.hstack((current_stock, zero_column))

    # 随机选择几个源仓库
    if n_warehouses // 2 <= 1:
        num_source_warehouses = 1  # 如果上限为1，直接设置为1
    else:
        num_source_warehouses = np.random.randint(1, n_warehouses // 2)
    source_warehouses = np.random.choice(n_warehouses, num_source_warehouses, replace=False)

    # 确定目标仓库：所有非源仓库
    all_warehouses = set(range(n_warehouses))
    source_warehouses_set = set(source_warehouses)
    target_warehouses = list(all_warehouses - source_warehouses_set)

    # 更新动作空间
    action_space = spaces.MultiDiscrete([
        len(source_warehouses),
        len(target_warehouses),
        m_goods,  # k: 货物类型索引，范围 0 到 m_goods-1
        1000  # q: 移动货物量
    ])

    # 计算目标仓库的总剩余容量
    total_remaining_target_capacity = 0
    for i in target_warehouses:
        remaining_capacity = max_stock_per_warehouse[i] - np.sum(current_stock[i, :-2])
        total_remaining_target_capacity += remaining_capacity

    # 初始化用于存储所有源仓库 excess stock 的变量
    total_excess_stock = 0

    # 随机设置每一个源仓库的 excess stock
    for i in source_warehouses:
        remaining_capacity = max_stock_per_warehouse[i] - np.sum(current_stock[i, :-1])

        # 确保这个 excess_stock 不会使得 total_excess_stock 超过目标仓库的剩余容量
        upper_bound = min(2000, total_remaining_target_capacity - total_excess_stock)
        if upper_bound > 0:
            excess_stock = np.random.randint(50, upper_bound)
            total_excess_stock += excess_stock
            current_stock[i, -1] = remaining_capacity + excess_stock  # 设置预入库值
        else:
            current_stock[i, -1] = remaining_capacity  # 如果无剩余空间，则 excess stock 为 0

    # 将最大库存加入obs
    max_stock_per_warehouse_reshaped = max_stock_per_warehouse[:, np.newaxis]
    current_stock = np.hstack((current_stock, max_stock_per_warehouse_reshaped))

    return current_stock, max_stock_per_warehouse, source_warehouses, target_warehouses, action_space


# 调用该函数以查看返回值
current_stock, max_stock_per_warehouse, source_warehouses, target_warehouses, action_space = reset_warehouse()
print("current_stock:", current_stock)
print("max_stock_per_warehouse:", max_stock_per_warehouse)
print("source_warehouses:", source_warehouses)
print("target_warehouses:", target_warehouses)
print("action_space:", action_space)

# %%
import numpy as np

n_warehouses = np.random.randint(2, 10)
m_goods = np.random.randint(2, 25)

max_stock_array = np.ones(10, dtype=int)
max_stock_array[:n_warehouses] = np.random.randint(2000, 100000, size=n_warehouses)
max_stock_per_warehouse = max_stock_array

current_stock = np.zeros((10, 25))

total_max_capacity = np.sum(max_stock_per_warehouse)
reserved_space = 0.3 * total_max_capacity
adjusted_max_capacity = total_max_capacity - reserved_space

new_max_stock_per_warehouse = (max_stock_per_warehouse / total_max_capacity) * adjusted_max_capacity
new_max_stock_per_warehouse = new_max_stock_per_warehouse.astype(int)

for i in range(n_warehouses):
    alpha = 0.5
    proportions = np.random.dirichlet(alpha * np.ones(m_goods), size=1)[0]

    current_stock[i, :m_goods] = np.round(proportions * new_max_stock_per_warehouse[i]).astype(int)

    total_stock = np.sum(current_stock[i, :])
    if total_stock > new_max_stock_per_warehouse[i]:
        excess = total_stock - new_max_stock_per_warehouse[i]
        while excess > 0:
            for j in range(m_goods):
                if current_stock[i, j] > 0 and excess > 0:
                    current_stock[i, j] -= 1
                    excess -= 1

zero_column = np.zeros((current_stock.shape[0], 1), dtype=int)
current_stock = np.hstack((current_stock, zero_column))

if n_warehouses // 2 <= 1:
    num_source_warehouses = 1
else:
    num_source_warehouses = np.random.randint(1, n_warehouses // 2)
source_warehouses = np.random.choice(n_warehouses, num_source_warehouses, replace=False)

all_warehouses = set(range(n_warehouses))
source_warehouses_set = set(source_warehouses)
target_warehouses = list(all_warehouses - source_warehouses_set)

total_remaining_target_capacity = 0
for i in target_warehouses:
    remaining_capacity = max_stock_per_warehouse[i] - np.sum(current_stock[i, :-2])
    total_remaining_target_capacity += remaining_capacity

total_excess_stock = 0
for i in source_warehouses:
    remaining_capacity = max_stock_per_warehouse[i] - np.sum(current_stock[i, :-1])
    upper_bound = min(2000, total_remaining_target_capacity - total_excess_stock)
    if upper_bound > 0:
        excess_stock = np.random.randint(50, upper_bound)
        total_excess_stock += excess_stock
        current_stock[i, -1] = remaining_capacity + excess_stock
    else:
        current_stock[i, -1] = remaining_capacity

max_stock_per_warehouse_reshaped = max_stock_per_warehouse[:, np.newaxis]
current_stock = np.hstack((current_stock, max_stock_per_warehouse_reshaped))

# normalized_front = current_stock[:, :-2] / max_stock_per_warehouse[:, None]
# normalized_last_two = current_stock[:, -2:] / np.array([1000, 2000])
# normalized_obs = np.hstack([normalized_front, normalized_last_two])
# flattened_obs = normalized_obs.flatten()

# %% 线性规划算法测试
# path: /mnt/data/generate_test_data.py

import numpy as np
import pandas as pd


def generate_test_data(n_warehouses=3, m_goods=2, seed=42):
    """
    Generate synthetic data for testing the stock optimization algorithm.
    """
    np.random.seed(seed)

    # Generate current stock levels for each warehouse and good
    current_stock = np.random.randint(100, 500, size=(n_warehouses, m_goods))

    # Generate maximum stock capacity for each warehouse
    max_stock_per_warehouse = np.random.randint(500, 1000, size=n_warehouses)

    # Generate minimum safety stock levels for each warehouse and good
    min_safety_stock = np.random.randint(50, 100, size=(n_warehouses, m_goods))

    # Create a DataFrame for special routing rules
    df_special_rules_index = pd.DataFrame({'item_index': [0],
                                           'start_index': [2],
                                           'end_index': [0]})

    return current_stock, max_stock_per_warehouse, min_safety_stock, df_special_rules_index


current_stock, max_stock_per_warehouse, min_safety_stock, df_special_rules_index = generate_test_data()
