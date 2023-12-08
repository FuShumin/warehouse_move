from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, LpStatusOptimal, value, PULP_CBC_CMD
import numpy as np
import pandas as pd
import pulp

from utils import OptimizationError


def add_objective_with_priority(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars,
                                priority_weights, slack):
    """
    添加目标函数，其中包含了对爆仓量，货物平均，以及权重的考虑。
    """
    # 绝对偏差
    deviations = LpVariable.dicts("Deviation",
                                  ((i, k) for i in range(n_warehouses) for k in range(m_goods)),
                                  lowBound=0)

    slack_penalty = 500 * lpSum(slack[i, k] for i in range(n_warehouses) for k in range(m_goods))

    # 平均偏差优化目标
    for k in range(m_goods):
        stock_percentage_levels = [
            (current_stock[i, k] + lpSum(
                [transfer_vars[(j, i, k)] - transfer_vars[(i, j, k)] for j in range(n_warehouses)]))
            for i in range(n_warehouses)
        ]

        mean_percentage = lpSum(stock_percentage_levels) / n_warehouses

        for i in range(n_warehouses):
            prob += stock_percentage_levels[i] - mean_percentage <= deviations[
                (i, k)], f"Deviation_upper_warehouse_{i}_good_{k}"
            prob += mean_percentage - stock_percentage_levels[i] <= deviations[
                (i, k)], f"Deviation_lower_warehouse_{i}_good_{k}"

    total_deviation = lpSum(deviations[(i, k)] for i in range(n_warehouses) for k in range(m_goods))
    priority_cost = lpSum([transfer_vars[(i, j, k)] * priority_weights[j]
                           for i in range(n_warehouses) for j in range(n_warehouses)
                           for k in range(m_goods) if i != j])
    # 计算每个仓库的库存超额量
    excess_stock = {
        i: lpSum(current_stock[i, k] + lpSum(
            [transfer_vars[(j, i, k)] - transfer_vars[(i, j, k)] for j in range(n_warehouses)])
                 for k in range(m_goods)) - max_stock_per_warehouse[i]
        for i in range(n_warehouses)
        if sum(current_stock[i, k]
               for k in range(m_goods)) > max_stock_per_warehouse[i]
    }

    if len(excess_stock) > 0:
        total_excess_stock = lpSum(excess_stock[i] for i in excess_stock.keys())
    else:
        total_excess_stock = 0
    prob += (total_deviation + priority_cost + total_excess_stock + slack_penalty,
             "Minimize_Total_Deviation_and_Priority_Cost_and_ExcessStock")


def add_constraints(prob, current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock,
                    n_warehouses, m_goods, transfer_vars, slack):
    """
    添加常规约束条件：
    1. 最大最小安全库存；
    2. 每次移库量不小于0；
    3. 不在同仓库之间移库；
    """
    for i in range(n_warehouses):
        # 计算仓库 i 中所有货物的总库存量（包括转移）
        total_stock_in_warehouse = lpSum(current_stock[i, k] + lpSum(
            [transfer_vars[(j, i, k)] - transfer_vars[(i, j, k)] for j in range(n_warehouses)])
                                         for k in range(m_goods))

        # 添加约束以确保仓库 i 的总库存量不超过其最大库容量
        prob += total_stock_in_warehouse <= max_stock_per_warehouse[i], f"Max_stock_limit_warehouse_{i}"

        for k in range(m_goods):
            prob += slack[i, k] <= 10, f"slack_a_little_{i}_{k}"
            stock_after_transfers = current_stock[i, k] + lpSum(
                [transfer_vars[(j, i, k)] - transfer_vars[(i, j, k)] for j in range(n_warehouses)])
            prob += stock_after_transfers + slack[i, k] >= min_safety_stock[
                i, k], f"Min_safety_stock_warehouse_{i}_good_{k}"
            prob += stock_after_transfers - slack[i, k] <= max_safety_stock[
                i, k], f"Max_safety_stock_warehouse_{i}_good_{k}"
            prob += transfer_vars[(i, i, k)] == 0, f"No_transfer_within_same_warehouse_{i}_good_{k}"

            for j in range(n_warehouses):
                if i != j:
                    if current_stock[i, k] <= 0:
                        prob += transfer_vars[
                                    (i, j, k)] == 0, f"Zero_transfer_warehouse_{i}_to_{j}_good_{k}_negative_stock"
                    else:
                        prob += transfer_vars[(i, j, k)] >= 0, f"Non_negative_transfer_warehouse_{i}_to_{j}_good_{k}"
                        prob += transfer_vars[(i, j, k)] <= current_stock[
                            i, k], f"Max_transfer_warehouse_{i}_to_{j}_good_{k}"
                        if current_stock[i, k] <= min_safety_stock[i, k]:
                            prob += transfer_vars[(i, j, k)] == 0, f"Safety_min_direction_{i}_to_{j}_good_{k}"
                        elif current_stock[i, k] >= max_safety_stock[i, k]:
                            prob += transfer_vars[(i, j, k)] >= 0, f"Safety_max_direction_{i}_to_{j}_good_{k}"

                    prob += transfer_vars[(i, j, k)] <= max(0, max_stock_per_warehouse[j] - current_stock[j, k]), \
                        f"Single_transfer_no_overstock_warehouse_{i}_to_{j}_good_{k}"


def add_special_rules(prob, df_special_rules_index, n_warehouses, transfer_vars):
    """
    配置特殊规则约束
    """
    if df_special_rules_index.empty:
        print("没有特殊规则需要应用。")
        return
    # 记录每个起始仓库和商品的特殊规则
    special_rules = {}
    for _, row in df_special_rules_index.iterrows():
        item_index = row['item_index']
        start_index = row['start_index']
        end_index = row['end_index']
        special_rules.setdefault((start_index, item_index), set()).add(end_index)

    # 对于有特殊规则的起始仓库和商品添加约束
    for (start_index, item_index), end_indexes in special_rules.items():
        for j in range(n_warehouses):
            if j != start_index and j not in end_indexes:
                prob += transfer_vars[
                            (start_index, j, item_index)] == 0, f"Special_rule_{start_index}_to_{j}_good_{item_index}"


def extract_solution(n_warehouses, m_goods, transfer_vars, current_stock, max_stock_per_warehouse):
    """
    从求解器结果中解析移库量
    """
    actions = []
    for i in range(n_warehouses):
        for j in range(n_warehouses):
            if i != j:
                for k in range(m_goods):
                    transfer_amount = transfer_vars[(i, j, k)].varValue
                    transfer_amount = round(transfer_amount, 2)
                    if transfer_amount > 1e-5:
                        new_stock_source = current_stock[i, k] - transfer_amount
                        new_stock_dest = current_stock[j, k] + transfer_amount
                        if new_stock_source >= 0 and new_stock_dest <= max_stock_per_warehouse[j]:
                            actions.append((i, j, k, transfer_amount))
                        else:
                            raise ValueError(f"Transfer from warehouse {i} to {j} of good {k} is not feasible.")
    return actions


def optimize_stock_distribution_percentage(current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock,
                                           df_special_rules_index, priority_weights):
    # %%
    try:
        prob = LpProblem("Stock_Distribution_With_Priority_And_Rules", LpMinimize)
        n_warehouses, m_goods = current_stock.shape
        # 移库动作变量
        transfer_vars = LpVariable.dicts("Transfer",
                                         ((i, j, k) for i in range(n_warehouses) for j in range(n_warehouses) for k in
                                          range(m_goods)),
                                         lowBound=0, cat='Continuous')
        # 松弛变量
        slack = LpVariable.dicts("slack", ((i, k) for i in range(n_warehouses) for k in range(m_goods)), lowBound=0)

        add_objective_with_priority(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars,
                                    priority_weights, slack)
        add_constraints(prob, current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock,
                        n_warehouses, m_goods, transfer_vars, slack)
        add_special_rules(prob, df_special_rules_index, n_warehouses, transfer_vars)

        prob.solve(solver=PULP_CBC_CMD(msg=False))
        print("-" * 8, "pulp result", "-" * 8, )
        print("Status:", LpStatus[prob.status])
        print("Objective =", value(prob.objective))
        print("-" * 6, "result end here", "-" * 6, )

        actions = extract_solution(n_warehouses, m_goods, transfer_vars, current_stock,
                                   max_stock_per_warehouse)
        status = prob.status
        return actions, status
    except Exception as e:
        # 捕获优化过程中的任何异常，并抛出自定义的 OptimizationError
        raise OptimizationError(f"Optimization process encountered an error: {e}")





# %%
def main():
    current_stock = np.array([[0, 0], [0, 0], [0, 0], [100, 0], [7149, 0]])
    max_safety_stock = np.array([[10, 10], [10, 100], [10, 10], [5, 10000], [8000, 10000]])
    max_stock_per_warehouse = np.array([10., 10., 10., 10000., 10000.])
    min_safety_stock = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0]])
    df_special_rules_index = pd.DataFrame({})
    priority_weights = np.array([4, 4, 4, 4, 4])
    # %%

    prob = LpProblem("Stock_Distribution_With_Priority_And_Rules", LpMinimize)
    n_warehouses, m_goods = current_stock.shape
    transfer_vars = LpVariable.dicts("Transfer",
                                     ((i, j, k) for i in range(n_warehouses) for j in range(n_warehouses) for k in
                                      range(m_goods)),
                                     lowBound=0, cat='Continuous')
    slack = LpVariable.dicts("slack", ((i, k) for i in range(n_warehouses) for k in range(m_goods)), lowBound=0)

    add_objective_with_priority(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars,
                                priority_weights, slack)
    add_constraints(prob, current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock,
                    n_warehouses, m_goods, transfer_vars, slack)
    add_special_rules(prob, df_special_rules_index, n_warehouses, transfer_vars)

    prob.solve()
    # var_dicts = prob.variablesDict()
    actions = extract_solution(n_warehouses, m_goods, transfer_vars, current_stock,
                               max_stock_per_warehouse)

    print(actions)
    print("current_stock", current_stock)
    print("min_safety_stock", min_safety_stock)
    print("max_safety_stock", max_safety_stock)
    print("max_stock_per_warehouse", max_stock_per_warehouse)
    print('special rules', df_special_rules_index)
    print('done')


if __name__ == '__main__':
    main()
