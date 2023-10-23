from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus


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


def add_constraints(prob, current_stock, max_stock_per_warehouse, min_safety_stock, n_warehouses, m_goods,
                    transfer_vars):
    """
    Add constraints to the linear programming problem.
    1. Max stock per warehouse
    2. Transferred quantity should be less than current stock
    3. Current stock should be more than minimum safety stock
    """
    # Constraint 1: Max stock per warehouse
    for i in range(n_warehouses):
        prob += lpSum([current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) +
                       lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) for k in range(m_goods)]) <= \
                max_stock_per_warehouse[i]
    # Constraint 2: Transferred quantity should be less than current stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) <= current_stock[i, k]
    # Constraint 3: Current stock should be more than minimum safety stock
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += current_stock[i, k] - lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses)]) + \
                    lpSum([transfer_vars[(j, i, k)] for j in range(n_warehouses)]) >= min_safety_stock[i, k]


def add_special_rules(prob, df_special_rules_index, n_warehouses, transfer_vars):
    """
    Add special rules as constraints to the linear programming problem.
    """
    # Create a dictionary to hold allowed target warehouses for each source warehouse
    allowed_targets = {}
    for index, row in df_special_rules_index.iterrows():
        item_index = row['item_index']
        start_index = row['start_index']
        end_index = row['end_index']

        if start_index not in allowed_targets:
            allowed_targets[start_index] = set()
        allowed_targets[start_index].add(end_index)

    # Add constraints based on allowed targets
    for start_index, targets in allowed_targets.items():
        for item_index in range(len(transfer_vars[0][0])):
            for j in range(n_warehouses):
                if j not in targets:
                    prob += transfer_vars[(start_index, j, item_index)] == 0


def solve_problem(prob, df_special_rules_index, n_warehouses, transfer_vars):
    """
    Solve the linear programming problem and handle infeasibility by removing special rule constraints if needed.
    """
    prob.solve()
    # Handle infeasible cases
    if LpStatus[prob.status] == "Infeasible":
        for index, row in df_special_rules_index.iterrows():
            item_index = row['item_index']
            start_index = row['start_index']
            end_index = row['end_index']
            for j in range(n_warehouses):
                if j != end_index:
                    # Check and remove the special rule constraint
                    constraint_name = f"{transfer_vars[(start_index, j, item_index)].name}"
                    if constraint_name in prob.constraints:
                        del prob.constraints[constraint_name]
        # Solve the problem again
        prob.solve()


def extract_solution(prob, n_warehouses, m_goods, transfer_vars):
    """
    Extract the optimal solution from the solved linear programming problem.
    """
    actions = []
    for i in range(n_warehouses):
        for j in range(n_warehouses):
            for k in range(m_goods):
                if transfer_vars[(i, j, k)].varValue > 0:
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


def optimize_stock_distribution_percentage(current_stock, max_stock_per_warehouse, min_safety_stock,
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
    add_constraints(prob, current_stock, max_stock_per_warehouse, min_safety_stock, n_warehouses, m_goods,
                    transfer_vars)
    add_special_rules(prob, df_special_rules_index, n_warehouses, transfer_vars)
    solve_problem(prob, df_special_rules_index, n_warehouses, transfer_vars)

    actions = extract_solution(prob, n_warehouses, m_goods, transfer_vars)
    return actions
