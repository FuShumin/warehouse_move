import time

import numpy as np
import pandas as pd

np.random.seed(0)  # 设定随机种子以保证结果的可重现性

# 初始化变量
num_warehouses = 5  # 仓库数量
num_goods = 4  # 货物种类
current_stock = np.random.randint(10, 150, size=(num_warehouses, num_goods))  # 当前库存
print(current_stock)
# 随机定义每个仓库的最大库容量，介于200到300单位之间
max_warehouse_capacity = np.random.randint(200, 300, size=num_warehouses)  # 最大仓库容量

# 为每种货物在每个仓库生成随机的最小安全库存，介于5到30单位之间
min_safety_stock = np.random.randint(5, 30, size=(num_warehouses, num_goods))  # 最小安全库存

# 生成比最小安全库存大但在仓库容量范围内的随机最大安全库存
max_safety_stock = min_safety_stock + np.random.randint(20, 70, size=(num_warehouses, num_goods))  # 最大安全库存

# 定义具有唯一权重的优先级权重，为每个仓库设置不同权重
priority_weights = np.random.choice(range(1, num_warehouses + 1), size=num_warehouses, replace=False)  # 优先级权重

# 以DataFrame形式定义特殊规则
special_rules = pd.DataFrame({'item_index': [0], 'start_index': [2], 'end_index': [0]})  # 特殊规则


# 应用特殊规则并返回可行的转移数量的函数
def apply_special_rules(from_warehouse, to_warehouse, good, amount, df_special_rules_index):
    # 筛选适用于特定货物和源仓库的规则
    df_rules_for_good = df_special_rules_index[(df_special_rules_index['item_index'] == good) &
                                               (df_special_rules_index['start_index'] == from_warehouse)]
    # 如果源仓库的货物存在规则，检查转移是否被允许
    if not df_rules_for_good.empty:
        # 检查目标仓库是否在规则的end_index列中
        if to_warehouse in df_rules_for_good['end_index'].values:
            return amount  # 允许转移
        else:
            return 0  # 不允许向该仓库转移
    return amount  # 无特定规则，允许转移


# 贪婪转移策略函数
def greedy_transfer_strategy(current_stock, max_total_stock_per_warehouse, min_safety_stock, max_safety_stock,
                             priority_weights, df_special_rules_index):
    n_warehouses, m_goods = current_stock.shape  # 仓库和货物的数量
    transfer_actions = []  # 转移行动记录

    priority_order = np.argsort(priority_weights)  # 根据权重确定优先级顺序

    # 对每个仓库执行检查和转移操作
    for from_warehouse in range(n_warehouses):
        total_stock_from = np.sum(current_stock[from_warehouse, :])  # 计算起始仓库的总库存
        # 检查并执行可能的转移
        for good in range(m_goods):
            # 检查源仓库的当前库存是否超过最大安全库存
            if current_stock[from_warehouse, good] > max_safety_stock[from_warehouse, good]:
                # 计算需要减少的库存量
                amount_to_reduce = current_stock[from_warehouse, good] - max_safety_stock[from_warehouse, good]
                # 遍历优先级仓库以转移库存
                for to_warehouse in priority_order:
                    if to_warehouse != from_warehouse:
                        total_stock_to = np.sum(current_stock[to_warehouse, :])  # 目标仓库的总库存
                        available_space = max_total_stock_per_warehouse[to_warehouse] - total_stock_to  # 可用空间
                        # 如果目标仓库有可用空间，尝试转移
                        if available_space > 0:
                            # 检查源仓库的最小安全库存，确定可转移的最大数量
                            if current_stock[from_warehouse, good] - amount_to_reduce >= min_safety_stock[
                                from_warehouse, good]:
                                potential_transfer_amount = min(amount_to_reduce, available_space,
                                                                max_safety_stock[to_warehouse, good] - current_stock[
                                                                    to_warehouse, good])
                                # 应用特殊规则
                                transfer_amount = apply_special_rules(from_warehouse, to_warehouse, good,
                                                                      potential_transfer_amount, df_special_rules_index)
                                # 执行转移
                                current_stock[from_warehouse, good] -= transfer_amount
                                current_stock[to_warehouse, good] += transfer_amount
                                # 记录转移行为
                                if transfer_amount > 0:
                                    transfer_actions.append({
                                        'from_warehouse': from_warehouse,
                                        'to_warehouse': to_warehouse,
                                        'good': good,
                                        'amount': transfer_amount
                                    })
                                # 更新转移后的数量
                                amount_to_reduce -= transfer_amount
                                available_space -= transfer_amount
                                # 如果没有库存或空间需要转移，退出循环
                                if amount_to_reduce <= 0 or available_space <= 0:
                                    break
                        if amount_to_reduce <= 0:
                            break
    return transfer_actions, current_stock  # 返回转移行为和更新后的库存


# 测试贪婪转移策略函数
t0 = time.time()
transfer_actions, updated_stock = greedy_transfer_strategy(
    current_stock, max_warehouse_capacity, min_safety_stock, max_safety_stock, priority_weights, special_rules
)
t1 = time.time()
print(transfer_actions, f'\n', updated_stock)
print('time cost: ', t1 - t0)
