import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MoveGoodsEnv(gym.Env):
    def __init__(self, n_warehouses=10, m_goods=25):
        super(MoveGoodsEnv, self).__init__()
        self.max_steps = 2048
        self.target_warehouses = [0]
        self.source_warehouses = [0]
        self.max_stock_per_warehouse = None
        self.n_warehouses = n_warehouses  # 仓库数量
        self.m_goods = m_goods  # 货物类型数量
        self.current_step = 0  # 步数初始化

        # 定义动作空间：四元组 (i, j, k, q)
        self.action_space = spaces.MultiDiscrete([
            10,
            10,
            25,  # k: 货物类型索引，范围 0 到 m_goods-1
            20  # q: 移动货物量  # todo 连续BOX scale到（0,1），再在环境里还原，取整。混合动作空间gym
        ])
        # 参考 IEOR 车间调度 库存调度 线性规划
        # 定义观察空间：每个仓库中每种货物的库存量
        self.observation_space = spaces.MultiDiscrete(
            [1000] * (self.n_warehouses * (self.m_goods + 2)))  # 包括incoming_stock
        # 其他初始设置，比如仓库的当前库存
        self.current_stock = np.zeros((self.n_warehouses, self.m_goods), dtype=int)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        # 随机生成仓库数量 n 和牌号数量 m
        self.n_warehouses = np.random.randint(2, 10)
        self.m_goods = np.random.randint(2, 25)
        # 为每个仓库随机生成最大库存上限，不足10的最大库存补1
        max_stock_array = np.ones(10, dtype=int)
        max_stock_array[:self.n_warehouses] = np.random.randint(2_00, 1_000, size=self.n_warehouses)
        self.max_stock_per_warehouse = max_stock_array

        # 初始化每个仓库每个牌号的当前库存为 0
        self.current_stock = np.zeros((10, 25))

        # 计算所有仓库最大库存上限的总和，并预留一定数量的空间
        total_max_capacity = np.sum(self.max_stock_per_warehouse)
        reserved_space = 0.3 * total_max_capacity
        adjusted_max_capacity = total_max_capacity - reserved_space

        # 计算每个仓库的新的最大库存上限
        new_max_stock_per_warehouse = (self.max_stock_per_warehouse / total_max_capacity) * adjusted_max_capacity
        new_max_stock_per_warehouse = new_max_stock_per_warehouse.astype(int)

        # 为每个仓库随机生成当前库存
        for i in range(self.n_warehouses):
            alpha = 0.5  # 你可以调整这个值
            proportions = np.random.dirichlet(alpha * np.ones(self.m_goods), size=1)[0]

            # 根据比例和仓库的新的最大库存上限来计算每个牌号的库存
            self.current_stock[i, :self.m_goods] = np.round(proportions * new_max_stock_per_warehouse[i]).astype(int)

            # 检查并确保总库存不超过新的上限
            total_stock = np.sum(self.current_stock[i, :])
            if total_stock > new_max_stock_per_warehouse[i]:
                excess = total_stock - new_max_stock_per_warehouse[i]
                while excess > 0:
                    for j in range(self.m_goods):
                        if self.current_stock[i, j] > 0 and excess > 0:
                            self.current_stock[i, j] -= 1
                            excess -= 1
        # 初始化预入库量
        zero_column = np.zeros((self.current_stock.shape[0], 1), dtype=int)
        self.current_stock = np.hstack((self.current_stock, zero_column))

        # 随机选择几个源仓库
        if self.n_warehouses // 2 <= 1:
            num_source_warehouses = 1  # 如果上限为1，直接设置为1
        else:
            num_source_warehouses = np.random.randint(1, self.n_warehouses // 2)
        self.source_warehouses = np.random.choice(self.n_warehouses, num_source_warehouses, replace=False)

        # 确定目标仓库：所有非源仓库
        all_warehouses = set(range(self.n_warehouses))
        source_warehouses_set = set(self.source_warehouses)
        self.target_warehouses = list(all_warehouses - source_warehouses_set)

        # 计算目标仓库的总剩余容量
        total_remaining_target_capacity = 0
        for i in self.target_warehouses:
            remaining_capacity = self.max_stock_per_warehouse[i] - np.sum(self.current_stock[i, :-2])
            total_remaining_target_capacity += remaining_capacity

        # 初始化用于存储所有源仓库 excess stock 的变量
        total_excess_stock = 0

        # 随机设置每一个源仓库的 excess stock
        for i in self.source_warehouses:
            remaining_capacity = self.max_stock_per_warehouse[i] - np.sum(self.current_stock[i, :-1])

            # 确保这个 excess_stock 不会使得 total_excess_stock 超过目标仓库的剩余容量
            upper_bound = min(100, total_remaining_target_capacity - total_excess_stock)
            if upper_bound > 0:
                excess_stock = np.random.randint(50, upper_bound)
                total_excess_stock += excess_stock
                self.current_stock[i, -1] = remaining_capacity + excess_stock  # 设置预入库值
            else:
                self.current_stock[i, -1] = remaining_capacity  # 如果无剩余空间，则 excess stock 为 0

        # # 将最大库存加入obs
        max_stock_per_warehouse_reshaped = self.max_stock_per_warehouse[:, np.newaxis]
        self.current_stock = np.hstack((self.current_stock, max_stock_per_warehouse_reshaped))
        normalized_observation = self.current_stock
        normalized_observation[:, -1] /= 4
        return normalized_observation.flatten(), {}  # 返回观察（当前库存量）

    def step(self, action):
        self.current_step += 1
        truncated = False
        # 首先检查是否达到最大步数，如果是，则提前终止 episode
        if self.current_step >= self.max_steps:
            truncated = True
        # 计算奖励
        reward = 0
        # 解析动作
        i, j, k, q = action
        q += 1
        # 确认动作是否有效
        # if i == j:  # 源仓库和目标仓库不能相同
        #     # 对前面的列（除最后两列以外）进行归一化
        #     normalized_front = self.current_stock[:, :-2] / self.max_stock_per_warehouse[:, None]
        #
        #     # 对最后两列用固定值500和1000进行归一化
        #     normalized_last_two = self.current_stock[:, -2:] / np.array([1000, 2000])
        #
        #     # 将这两部分合并到一起
        #     normalized_observation = np.hstack([normalized_front, normalized_last_two])
        #
        #     return normalized_observation.flatten(), -1, False, truncated, {
        #         'msg': 'Invalid action. Source and target warehouse cannot be the same.'}

        if i not in self.source_warehouses:  # 源仓库必须是预先定义的源仓库
            normalized_observation = self.current_stock
            normalized_observation[:, -1] /= 4
            return normalized_observation.flatten(), -1, False, truncated, {
                'msg': 'Invalid action. Source warehouse is not in the list of source warehouses.'}
        else:
            reward += 0.1

        if j not in self.target_warehouses:  # 目标仓库不能是源仓库
            normalized_observation = self.current_stock
            normalized_observation[:, -1] /= 4
            return normalized_observation.flatten(), -1, False, truncated, {
                'msg': 'Invalid action. Target warehouse is not in the list of target warehouses.'}
        else:
            reward += 0.1

        # 确保选取的货物不是最后一列（预入库量）
        # if k == self.current_stock.shape[1] - 2:
        #     normalized_observation = self.current_stock / self.max_stock_per_warehouse[:, None]
        #     return normalized_observation.flatten(), -1, False, False, {
        #         'msg': 'Invalid action. Cannot select the last column (pre-arrival stock).'}

        # 获取源仓库和目标仓库的当前库存
        source_current_stock = self.current_stock[i, k]
        target_current_stock = self.current_stock[j, k]

        # 检查源仓库是否有足够的库存
        if source_current_stock < q:
            normalized_observation = self.current_stock
            normalized_observation[:, -1] /= 4
            return normalized_observation.flatten(), -1, False, truncated, {
                'msg': 'Invalid action. Not enough stock in source warehouse.'}
        else:
            reward += 0.1
        # 检查目标仓库是否有足够的空间
        target_max_stock = self.max_stock_per_warehouse[j]
        if np.sum(self.current_stock[j, :-1]) + q > target_max_stock:
            normalized_observation = self.current_stock
            normalized_observation[:, -1] /= 4
            return normalized_observation.flatten(), -1, False, truncated, {
                'msg': 'Invalid action. Not enough space in target warehouse.'}
        else:
            reward += 0.1
        # 计算移库前的库存百分比标准差
        stock_percentage_before = self.current_stock[:, k] / self.max_stock_per_warehouse * 100
        std_before = np.std(stock_percentage_before)
        source_remaining_before = np.abs(self.max_stock_per_warehouse[i] - np.sum(self.current_stock[i, :-1]))

        # 执行移库
        self.current_stock[i, k] -= q  # 减少源仓库的库存
        self.current_stock[j, k] += q  # 增加目标仓库的库存
        print('从源仓库:',
              i, ' 移至仓库:', j, ' 选择牌号k:', k, ' 移动数量q:', q, '----源仓库:', self.source_warehouses, ' 目标仓库:', self.target_warehouses)
        # 计算移库后的库存百分比标准差
        stock_percentage_after = self.current_stock[:, k] / self.max_stock_per_warehouse * 100
        std_after = np.std(stock_percentage_after)
        source_remaining_after = np.abs(self.max_stock_per_warehouse[i] - np.sum(self.current_stock[i, :-1]))

        if std_after <= std_before:  # 如果标准差减小，说明库存更平衡了
            reward += (std_before - std_after)  # 额外奖励：减小的标准差量
            # print('std diff:', std_before - std_after)
        if source_remaining_after < source_remaining_before:  # 如果源仓库爆仓缓解，则奖励
            reward += (source_remaining_before - source_remaining_after) * 0.1
        else:
            reward -= (source_remaining_after - source_remaining_before)*0.1 + 0.05

            # 判断是否达到终止条件
        done = self.check_termination_condition()
        if done:
            reward += 500
            print('移库完成!')

        normalized_observation = self.current_stock
        normalized_observation[:, -1] /= 4
        new_obs = normalized_observation.flatten()
        # 返回新地观察、奖励、是否终止和其他信息
        return new_obs, reward, done, truncated, {}

    def check_termination_condition(self):
        # 检查所有仓库的库存量是否都低于或等于最大库存容量
        for i in range(self.n_warehouses):
            total_stock_in_warehouse = np.sum(self.current_stock[i, :-1])
            if total_stock_in_warehouse > self.max_stock_per_warehouse[i]:
                return False  # 有一个或多个仓库的当前库存超过了其最大容量，因此移库还不能结束
        # print('episode done.')
        return True  # 所有仓库的库存量都在安全范围内，仿真可以结束 # todo 不采用终止条件但是用固定步数，reward越大越好。
