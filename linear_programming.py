from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus,LpStatusOptimal
import numpy as np


def optimize_stock_distribution(current_stock, max_stock_per_warehouse, min_safety_stock):
    n_warehouses, m_goods = current_stock.shape

    # Initialize the optimization problem
    prob = LpProblem("Stock_Distribution", LpMinimize)

    # Create variables
    transfer_vars = LpVariable.dicts("Transfer",
                                     ((i, j, k) for i in range(n_warehouses) for j in range(n_warehouses) for k in
                                      range(m_goods)),
                                     lowBound=0, cat='Integer')

    # Objective function: Minimize the standard deviation of stock levels for each good across all warehouses
    for k in range(m_goods):
        stock_levels = [current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) +
                        lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) for i in range(n_warehouses)]

        mean_stock = lpSum(stock_levels) / n_warehouses
        std_dev = lpSum([(stock_level - mean_stock) for stock_level in stock_levels]) / n_warehouses
        prob += std_dev  # Add to objective function

    # Constraint 1: Cannot exceed maximum warehouse capacity
    for i in range(n_warehouses):
        prob += lpSum([current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) +
                       lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) for k in range(m_goods)]) <= \
                max_stock_per_warehouse[i]

    # Constraint 2: Cannot transfer more than current stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) <= current_stock[i, k]

    # Constraint 3: Stock must be above safety stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) + \
                    lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) >= min_safety_stock[i, k]

    # Solve the problem
    prob.solve()

    # Extract the actions
    actions = []
    for i in range(n_warehouses):
        for j in range(n_warehouses):
            for k in range(m_goods):
                if transfer_vars[(i, j, k)].varValue > 0:
                    actions.append((i, j, k, transfer_vars[(i, j, k)].varValue))

    return actions


def optimize_stock_distribution_percentage(current_stock, max_stock_per_warehouse, min_safety_stock,
                                           df_special_rules_index):
    n_warehouses, m_goods = current_stock.shape

    # Initialize the optimization problem
    prob = LpProblem("Stock_Distribution_Modified", LpMinimize)

    # Create variables
    transfer_vars = LpVariable.dicts("Transfer",
                                     ((i, j, k) for i in range(n_warehouses) for j in range(n_warehouses) for k in
                                      range(m_goods)),
                                     lowBound=0, cat='Integer')

    # Objective function: Minimize the standard deviation of stock percentage for each good across all warehouses
    for k in range(m_goods):
        stock_percentage_levels = [
            (current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) +
             lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)])) / max_stock_per_warehouse[i]
            for i in range(n_warehouses)]
        mean_percentage = lpSum(stock_percentage_levels) / n_warehouses
        std_dev_percentage = lpSum(
            [(stock_percentage - mean_percentage) for stock_percentage in stock_percentage_levels]) / n_warehouses
        prob += std_dev_percentage  # Add to objective function

    # Constraints remain the same as before

    # Constraint 1: Cannot exceed maximum warehouse capacity
    for i in range(n_warehouses):
        prob += lpSum([current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) +
                       lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) for k in range(m_goods)]) <= \
                max_stock_per_warehouse[i]

    # Constraint 2: Cannot transfer more than current stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) <= current_stock[i, k]

    # Constraint 3: Stock must be above safety stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) + \
                    lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) >= min_safety_stock[i, k]

    # 添加特殊规则约束
    for index, row in df_special_rules_index.iterrows():
        item_index = row['item_index']
        start_index = row['start_index']
        end_index = row['end_index']

        for j in range(n_warehouses):
            if j != end_index:
                prob += transfer_vars[(start_index, j, item_index)] == 0

    # 尝试求解问题
    prob.solve()
    # 检查解的状态
    if LpStatus[prob.status] == "Infeasible":
        # 问题无解，去掉特殊规则并重新求解
        for index, row in df_special_rules_index.iterrows():
            item_index = row['item_index']
            start_index = row['start_index']
            end_index = row['end_index']

            for j in range(n_warehouses):
                if j != end_index:
                    # 检查并删除特殊规则约束
                    constraint_name = f"{transfer_vars[(start_index, j, item_index)].name}"
                    if constraint_name in prob.constraints:
                        del prob.constraints[constraint_name]

            # 重新求解问题
            prob.solve()
            # if LpStatus[prob.status] == "Infeasible":
            #     print("The problem is infeasible.")
            #     return None
    # Extract the actions
    actions = []
    for i in range(n_warehouses):
        for j in range(n_warehouses):
            for k in range(m_goods):
                if transfer_vars[(i, j, k)].varValue > 0:
                    actions.append((i, j, k, transfer_vars[(i, j, k)].varValue))

    return actions


def calculate_stock_percentage_change(actions, current_stock, max_stock_per_warehouse):
    # 复制一份 current_stock 以避免直接修改
    current_stock_copy = np.copy(current_stock)

    # 计算初始库存百分比
    initial_percentage = current_stock_copy / max_stock_per_warehouse[:, None] * 100

    # 应用移库行动
    for i, j, k, qty in actions:
        current_stock_copy[i, k] -= qty
        current_stock_copy[j, k] += qty

    # 计算移库后的库存百分比
    final_percentage = current_stock_copy / max_stock_per_warehouse[:, None] * 100

    # 找出涉及的货物
    involved_goods = set(k for _, _, k, _ in actions)

    # 计算和显示标准差
    result = {}
    for k in involved_goods:
        initial_std_dev = np.std(initial_percentage[:, k])
        final_std_dev = np.std(final_percentage[:, k])
        result[k] = {
            'initial_percentage': initial_percentage[:, k],
            'final_percentage': final_percentage[:, k],
            'initial_std_dev': initial_std_dev,
            'final_std_dev': final_std_dev
        }

    return result


def generate_report(actions, current_stock, max_stock_per_warehouse):
    result = calculate_stock_percentage_change(actions, current_stock, max_stock_per_warehouse)
    report = []

    for k, data in result.items():
        section = f"### 货物 {k}\n"
        section += "- **初始库存百分比**：" + ", ".join([f"{x:.2f}%" for x in data['initial_percentage']]) + "\n"
        section += "- **移库后库存百分比**：" + ", ".join([f"{x:.2f}%" for x in data['final_percentage']]) + "\n"
        section += f"- **初始标准差**：{data['initial_std_dev']:.2f}\n"
        section += f"- **移库后标准差**：{data['final_std_dev']:.2f}\n"
        report.append(section)

    return "\n".join(report)


def map_index_to_readable_result(result, warehouse_to_index, item_to_index, warehouse_code_to_name, item_code_to_name):
    readable_result = {}
    for k, data in result.items():
        item = item_code_to_name.get(list(item_to_index.keys())[k], f"未知物品 {k}")
        initial_percentage = [f"{x:.2f}%" for x in data['initial_percentage']]
        final_percentage = [f"{x:.2f}%" for x in data['final_percentage']]
        initial_std_dev = f"{data['initial_std_dev']:.2f}"
        final_std_dev = f"{data['final_std_dev']:.2f}"

        readable_result[item] = {
            '该物品在各仓库的移库前占比': initial_percentage,
            '该物品在各仓库的移库后占比': final_percentage,
            '该物品在各仓库的移库前标准差': initial_std_dev,
            '该物品在各仓库的移库后标准差': final_std_dev
        }
    return readable_result


if __name__ == '__main__':
    # Test the function
    n_warehouses = 10  # Use a smaller number for demonstration
    m_goods = 50  # Use a smaller number for demonstration
    current_stock = np.random.randint(1000, 5000, size=(n_warehouses, m_goods))
    max_stock_per_warehouse = np.random.randint(10000, 20000, size=n_warehouses)
    min_safety_stock = np.full((10, 25), 100)

    actions = optimize_stock_distribution(current_stock, max_stock_per_warehouse, min_safety_stock)
    print(actions)
    # %%
    # Additional test cases
    current_stock_test = current_stock[:, :-2]
    max_stock_per_warehouse_test = current_stock_test[:, -1] - current_stock_test[:, -2]
    result = calculate_stock_percentage_change(actions, current_stock, max_stock_per_warehouse)
    report = generate_report(actions, current_stock, max_stock_per_warehouse)