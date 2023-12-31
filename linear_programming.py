from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, LpStatusOptimal
import numpy as np


def add_objective_function(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars):
    """
    Add the objective function to the linear programming problem.
    Objective is to minimize the standard deviation of stock percentage levels across different warehouses for each good.
    """
    for k in range(m_goods):
        stock_percentage_levels = [
            (current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) +
             lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)])) / max_stock_per_warehouse[i]
            for i in range(n_warehouses)]
        mean_percentage = lpSum(stock_percentage_levels) / n_warehouses
        std_dev_percentage = lpSum(
            [(stock_percentage - mean_percentage) for stock_percentage in stock_percentage_levels]) / n_warehouses
        prob += std_dev_percentage  # Add to objective function


def add_constraints(prob, current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock, n_warehouses, m_goods,
                    transfer_vars):
    """
    Add constraints to the linear programming problem.
    1. Max stock per warehouse
    2. Transferred quantity should be less than current stock
    3. Current stock should be more than minimum safety stock
    4. Current stock should be less than maximum safety stock
    """
    # Constraint 1: Max stock per warehouse
    for i in range(n_warehouses):
        prob += lpSum([current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) +
                       lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) for k in range(m_goods)]) <= \
                max_stock_per_warehouse[i]
    # Constraint 2: Transferred quantity should be less than current stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            max_transfer = max(0, current_stock[i, k])
            prob += lpSum([transfer_vars.get((i, j, k), 0) for j in range(n_warehouses)]) <= max_transfer

    # Constraint 3: Current stock should be more than minimum safety stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) + \
                    lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) >= min_safety_stock[i, k]

    # Constraint 4: for Maximum Safety Stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) + \
                    lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) <= max_safety_stock[i, k]


def add_special_rules(prob, df_special_rules_index, n_warehouses, transfer_vars):
    """
    Add special rules as constraints to the linear programming problem.
    """
    for index, row in df_special_rules_index.iterrows():

        item_index = row['item_index']
        start_index = row['start_index']
        end_index = row['end_index']
        for j in range(n_warehouses):
            if j != end_index:
                # List comprehension to collect all valid (start_index, j, item_index) tuples
                valid_keys = [(start_index, j, item_index) for j in range(n_warehouses) if
                              j != end_index and transfer_vars.get((start_index, j, item_index)) is not None]

                # Update the problem constraints for all valid keys
                for key in valid_keys:
                    prob += transfer_vars[key] == 0


def solve_problem(prob, df_special_rules_index, n_warehouses, transfer_vars):
    """
    Solve the linear programming problem and handle infeasibility by removing special rule constraints if needed.
    """
    prob.solve()
    # Handle infeasible cases
    # if LpStatus[prob.status] == "Infeasible":
    #     for index, row in df_special_rules_index.iterrows():
    #         item_index = row['item_index']
    #         start_index = row['start_index']
    #         end_index = row['end_index']
    #         for j in range(n_warehouses):
    #             if j != end_index:
    #                 # Check and remove the special rule constraint
    #                 if transfer_vars.get((start_index, j, item_index)) is not None:
    #                     constraint_name = f"{transfer_vars[(start_index, j, item_index)].name}"
    #                     if constraint_name in prob.constraints:
    #                         del prob.constraints[constraint_name]
        # Solve the problem again
        # prob.solve()
    if LpStatus[prob.status] == "Infeasible":
        return f"Errors found: {LpStatus[prob.status]}"


def extract_solution(prob, n_warehouses, m_goods, transfer_vars, current_stock):
    """
    Extract the optimal solution from the solved linear programming problem.
    """
    actions = []
    for i in range(n_warehouses):
        for j in range(n_warehouses):
            for k in range(m_goods):
                if transfer_vars[(i, j, k)].varValue > 1e-8 and current_stock[i, k] != 0:
                    actions.append((i, j, k, transfer_vars[(i, j, k)].varValue))
    return actions


def check_for_errors(current_stock, max_stock_per_warehouse, min_safety_stock, n_warehouses, m_goods):
    """
    Check for potential issues that might make the problem infeasible or cause errors.
    """
    errors = []

    # Check if minimum safety stock exceeds maximum warehouse capacity
    for i in range(n_warehouses):
        for k in range(m_goods):
            if min_safety_stock[i, k] > max_stock_per_warehouse[i]:
                errors.append(
                    f"Minimum safety stock for warehouse {i} and good {k} exceeds maximum warehouse capacity.")

    # Check if any maximum stock capacity is negative
    if any(stock < 0 for stock in max_stock_per_warehouse):
        errors.append("Negative values found in maximum stock capacities.")

    # # Check if any current stock is negative
    # if any(stock < 0 for stock in current_stock.flatten()):
    #     errors.append("Negative values found in current stock levels.")

    # Check if any minimum safety stock is negative
    if any(stock < 0 for stock in min_safety_stock.flatten()):
        errors.append("Negative values found in minimum safety stock levels.")

    # Check if the number of warehouses or goods is zero
    if n_warehouses == 0 or m_goods == 0:
        errors.append("Number of warehouses or goods cannot be zero.")
    # TODO: Add more error checks as needed

    return errors


def optimize_stock_distribution_percentage(current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock,
                                           df_special_rules_index):
    """
    Main function to optimize the stock distribution among different warehouses.
    """
    n_warehouses, m_goods = current_stock.shape

    # Check for potential errors
    errors = check_for_errors(current_stock, max_stock_per_warehouse, min_safety_stock, n_warehouses, m_goods)
    if errors:
        return f"Errors found: {errors}"

    # Initialize the linear programming problem
    prob = LpProblem("Stock_Distribution_Modified", LpMinimize)

    transfer_vars = LpVariable.dicts("Transfer",
                                     ((i, j, k) for i in range(n_warehouses) for j in range(n_warehouses) for k in
                                      range(m_goods)),
                                     lowBound=0, cat='Integer')

    add_objective_function(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars)
    # debug_info = debug_objective_function(prob)
    # print(debug_info)
    add_constraints(prob, current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock, n_warehouses, m_goods,
                    transfer_vars)
    # debug_info = debug_constraints(prob, current_stock, n_warehouses, m_goods, transfer_vars)
    # print(debug_info)
    add_special_rules(prob, df_special_rules_index, n_warehouses, transfer_vars)
    solve_problem(prob, df_special_rules_index, n_warehouses, transfer_vars)

    actions = extract_solution(prob, n_warehouses, m_goods, transfer_vars, current_stock)
    return actions


def calculate_stock_percentage_change(actions, current_stock, max_stock_per_warehouse):
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

def debug_constraints(prob, current_stock, n_warehouses, m_goods, transfer_vars):
    """
    Debug function to print and check constraints related to stock levels.
    """
    debug_info = []
    for i in range(n_warehouses):
        for k in range(m_goods):
            constraint_expr = lpSum([transfer_vars.get((i, j, k), 0) for j in range(n_warehouses)])
            debug_info.append(f"For warehouse {i}, good {k}: Constraint should be Transfer Sum <= {current_stock[i, k]}")
            debug_info.append(f"Constraint expression: {constraint_expr}")
    return debug_info

def debug_objective_function(prob):
    """
    Debug function to print the current objective function.
    """
    return f"Current Objective Function: {prob.objective}"

#
# def add_objective_function_v1(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars,
#                            C=1e6):
#     """
#     Add the objective function to the linear programming problem.
#     Objective is to minimize the chance of any individual warehouse getting full ("bursting").
#     Parameters:
#     - C: A large constant for penalizing warehouses that are already over capacity.
#     """
#     # Create a list to hold penalty terms for each warehouse
#     penalty_terms = []
#     for i in range(n_warehouses):
#         # Calculate the remaining storage capacity for each warehouse after transfers, considering the current stock
#         remaining_capacity = (
#                 max_stock_per_warehouse[i] -
#                 lpSum([current_stock[i, k] for k in range(m_goods)]) -
#                 lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses) for k in range(m_goods)])
#         )
#         # Create a penalty term based on remaining capacity, avoiding division
#         if remaining_capacity >= 0:
#             penalty_term = remaining_capacity + 1e-8
#         else:
#             penalty_term = C * (-remaining_capacity)
#         penalty_terms.append(penalty_term)
#     # Add the objective function with the penalty terms
#     prob += lpSum(penalty_terms)
#
#     # for k in range(m_goods):
#     #     stock_percentage_levels = [
#     #         (current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) +
#     #          lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)])) / max_stock_per_warehouse[i]
#     #         for i in range(n_warehouses)]
#     #     mean_percentage = lpSum(stock_percentage_levels) / n_warehouses
#     #     std_dev_percentage = lpSum(
#     #         [(stock_percentage - mean_percentage) for stock_percentage in stock_percentage_levels]) / n_warehouses
#     #     prob += std_dev_percentage  # Add to objective function
#
#
# def add_constraints_v1(prob, current_stock, max_stock_per_warehouse, min_safety_stock, n_warehouses, m_goods,
#                     transfer_vars):
#     # Constraint 2: Transferred quantity should be less than current stock
#     for i in range(n_warehouses):
#         for k in range(m_goods):
#             max_transfer = max(0, current_stock[i, k])
#             prob += lpSum([transfer_vars.get((i, j, k), 0) for j in range(n_warehouses)]) <= max_transfer
#
#     # Constraint 3: Current stock should be more than minimum safety stock
#     for i in range(n_warehouses):
#         for k in range(m_goods):
#             prob += current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) + \
#                     lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) >= min_safety_stock[i, k]
#
#     # Constraint 4: Prevent same-warehouse transfers
#     for i in range(n_warehouses):
#         for k in range(m_goods):
#             prob += transfer_vars[i, i, k] == 0