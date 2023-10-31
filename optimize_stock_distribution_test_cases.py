from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, LpStatusOptimal, LpAffineExpression


def add_objective_function(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars,
                           C=1e6):
    """
    Add the objective function to the linear programming problem.
    Objective is to minimize the chance of any individual warehouse getting full ("bursting").
    Parameters:
    - C: A large constant for penalizing warehouses that are already over capacity.
    """
    # Create a list to hold penalty terms for each warehouse
    penalty_terms = []
    for i in range(n_warehouses):
        # Calculate the remaining storage capacity for each warehouse after transfers, considering the current stock
        remaining_capacity = (
                max_stock_per_warehouse[i] -
                lpSum([current_stock[i, k] for k in range(m_goods)]) -
                lpSum([transfer_vars[(i, j, k)] for j in range(n_warehouses) for k in range(m_goods)])
        )
        # Create a penalty term based on remaining capacity, avoiding division
        if remaining_capacity >= 0:
            penalty_term = remaining_capacity + 1e-8
        else:
            penalty_term = C * (-remaining_capacity)
        penalty_terms.append(penalty_term)
    # Add the objective function with the penalty terms
    prob += lpSum(penalty_terms)

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

    # Constraint 4: Prevent same-warehouse transfers
    for i in range(n_warehouses):
        for k in range(m_goods):
            prob += transfer_vars[i, i, k] == 0
