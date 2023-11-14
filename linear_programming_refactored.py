from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, LpStatusOptimal
import numpy as np
import pandas as pd


def add_objective_with_priority(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars,
                                priority_weights):
    """
    Adds a linearized objective function to the problem to minimize the sum of absolute deviations
    from the mean stock percentage level for each good across warehouses,
    incorporating a priority strategy for warehouses.
    """
    # Create a dictionary to hold the deviation variables for the sum of absolute deviations objective
    deviations = LpVariable.dicts("Deviation",
                                  ((i, k) for i in range(n_warehouses) for k in range(m_goods)),
                                  lowBound=0)  # Deviations are non-negative

    # Add the objective to minimize the sum of deviations
    for k in range(m_goods):  # For each good
        stock_percentage_levels = [
            (current_stock[i, k] + lpSum(
                [transfer_vars[(j, i, k)] - transfer_vars[(i, j, k)] for j in range(n_warehouses)]))
            / max_stock_per_warehouse[i] for i in range(n_warehouses)
        ]

        # Calculate the mean stock percentage level for good k
        mean_percentage = lpSum(stock_percentage_levels) / n_warehouses

        # Add constraints for deviations and adjust the objective function
        for i in range(n_warehouses):
            # Constraint that links the deviation variables with the stock percentage levels
            prob += stock_percentage_levels[i] - mean_percentage <= deviations[
                (i, k)], f"Deviation_upper_warehouse_{i}_good_{k}"
            prob += mean_percentage - stock_percentage_levels[i] <= deviations[
                (i, k)], f"Deviation_lower_warehouse_{i}_good_{k}"

    # Add the deviations to the problem's objective
    total_deviation = lpSum(deviations[(i, k)] for i in range(n_warehouses) for k in range(m_goods))
    priority_cost = lpSum([transfer_vars[(i, j, k)] * priority_weights[j]
                           for i in range(n_warehouses) for j in range(n_warehouses)
                           for k in range(m_goods) if i != j])

    # Add the priority cost to the existing objective function
    prob += total_deviation + priority_cost, "Minimize_Total_Deviation_and_Priority_Cost"


def add_constraints_refactored(prob, current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock,
                               n_warehouses, m_goods, transfer_vars):
    """
    Adds constraints to the linear programming problem.
    Ensures post-transfer stock levels are within safety stock and capacity limits.
    """
    for i in range(n_warehouses):
        for k in range(m_goods):
            stock_after_transfers = current_stock[i, k] + lpSum(
                [transfer_vars[(j, i, k)] - transfer_vars[(i, j, k)] for j in range(n_warehouses)])
            prob += stock_after_transfers >= min_safety_stock[i, k], f"Min_safety_stock_warehouse_{i}_good_{k}"
            prob += stock_after_transfers <= max_safety_stock[i, k], f"Max_safety_stock_warehouse_{i}_good_{k}"

            for j in range(n_warehouses):  # For each potential transfer
                if i != j:
                    prob += transfer_vars[(i, j, k)] <= current_stock[
                        i, k], f"Max_transfer_warehouse_{i}_to_{j}_good_{k}"
                    prob += transfer_vars[(i, j, k)] >= 0, f"Non_negative_transfer_warehouse_{i}_to_{j}_good_{k}"
                    prob += transfer_vars[(i, j, k)] <= max_stock_per_warehouse[j] - current_stock[
                        j, k], f"No_overstock_warehouse_{j}_good_{k}"


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
                # Check if the transfer variable exists and if so, add a constraint that it must be zero
                if (start_index, j, item_index) in transfer_vars:
                    prob += transfer_vars[(
                        start_index, j, item_index)] == 0, f"Special_rule_{start_index}_to_{j}_good_{item_index}"


def extract_solution_refactored(prob, n_warehouses, m_goods, transfer_vars, current_stock, max_stock_per_warehouse):
    """
    Extracts the solution from the solved problem, checking for feasibility of individual actions.
    """
    actions = []  # to collect feasible actions
    for i in range(n_warehouses):  # Source warehouse
        for j in range(n_warehouses):  # Destination warehouse
            if i != j:
                for k in range(m_goods):  # For each good
                    transfer_amount = transfer_vars[(i, j, k)].varValue
                    if transfer_amount > 1e-5:
                        new_stock_source = current_stock[i, k] - transfer_amount
                        new_stock_dest = current_stock[j, k] + transfer_amount
                        if (new_stock_source >= 0 and new_stock_dest <= max_stock_per_warehouse[j]):
                            actions.append((i, j, k, transfer_amount))
                        else:
                            raise ValueError(f"Transfer from warehouse {i} to {j} of good {k} is not feasible.")
    return actions


# Initialization and problem solving
n_warehouses = 2
m_goods = 2
prob = LpProblem("Stock_Distribution_With_Priority_And_Rules", LpMinimize)

# Define current stock, minimum and maximum safety stock, and max stock per warehouse
current_stock = np.array([[40, 5], [10, 40]])  # Current stock for two goods in two warehouses
max_stock_per_warehouse = [50, 50]  # Max stock per warehouse for two warehouses
min_safety_stock = np.array([[10, 15], [5, 10]])  # Minimum safety stock for two goods in two warehouses
max_safety_stock = np.array([[40, 45], [30, 40]])  # Maximum safety stock for two goods in two warehouses
df_special_rules_index = pd.DataFrame({
    'item_index': [1],  # Good '1'
    'start_index': [0],  # Warehouse '0'
    'end_index': [1]  # Warehouse '1' where transfers are not allowed for Good '1' from Warehouse '0'
})
# Define the priority weights
priority_weights = [1, 2]  # Lower number indicates higher priority

# Define transfer variables for the corrected number of warehouses and goods
transfer_vars = LpVariable.dicts("Transfer",
                                 ((i, j, k) for i in range(2) for j in range(2) for k in range(2)),
                                 lowBound=0, cat='Continuous')

add_objective_with_priority(prob, current_stock, max_stock_per_warehouse, n_warehouses, m_goods, transfer_vars,
                            priority_weights)
add_constraints_refactored(prob, current_stock, max_stock_per_warehouse, min_safety_stock, max_safety_stock,
                           n_warehouses, m_goods, transfer_vars)
add_special_rules(prob, df_special_rules_index, n_warehouses, transfer_vars)

prob.solve()
actions = extract_solution_refactored(prob, n_warehouses, m_goods, transfer_vars, current_stock,
                                      max_stock_per_warehouse)
print(actions)
