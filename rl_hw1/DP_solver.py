import numpy as np
import copy

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, policy: np.ndarray,discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        # self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)
        self.policy = policy  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        # def step(self, state: int, action: int) -> tuple: Returns: tuple: (next_state, reward, done)
        # print("state:",state)
        next_state, reward, done = self.grid_world.step(state,action)
        next_action = self.policy[next_state]
        # print(self.policy)
        
        # state_coord = self.grid_world.get_state_list()[state]
        # print("state coord:",state_coord)
        # print("Type of self.grid_world:",type(self.grid_world))
        
        # if self.grid_world.__is_goal_state(state_coord): #it's a goal
        #     return reward
        # elif self.grid_world.__is_trap_state(state_coord): #it's a trap
        #     return reward
        if state == next_state: # it hits the wall and stuck, or it's terminal
            return reward

        q_value = reward + self.discount_factor * self.values[next_state]
        # for move in range(4): #move=0,1,2,3
        #     q_value += self.discount_factor * (self.policy[next_state,move]*self.values[next_state])
        return q_value
        
        raise NotImplementedError


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        v = 0
        for move in range(4):#move=0,1,2,3
            v += self.policy[state,move] * self.get_q_value(state,move)
        return v
        raise NotImplementedError

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        #synchronize: a copy is needed
        v_copy = copy.deepcopy(self.values)

        #start update
        for i in range(self.grid_world.get_state_space()): #for i in V(s)
            v_new = 0
            for move in range(4): #move=0,1,2,3
                next_state, reward, done = self.grid_world.step(i,move)
                if not i==next_state:
                    v_new += self.policy[i,move]*(reward+self.discount_factor*v_copy[next_state]*(1-done))
                else:
                    v_new += self.policy[i,move]*(reward)

            self.values[i] = v_new
        return
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while True:
            v_old = copy.deepcopy(self.values)
            print("v_old:",v_old)
            self.evaluate()
            print("self.values:",self.values)
            delta = np.max(np.abs(self.values - v_old))
            print("delta:",delta)
            if delta < self.threshold:
                return
        raise NotImplementedError


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action = self.policy[state]
        v = self.get_q_value(state,action)
        return v
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        #synchronize: a copy is needed
        def evaluate_per_iter(self):
            v_copy = self.values

            #start update
            for i in range(self.grid_world.get_state_space()): #for v un V(s)
                v_new = v_copy[i]
                self.values[i] = v_new

        while True:
            v_old = self.values
            evaluate_per_iter(self)
            delta = np.max(np.abs(self.values - v_old))
            if delta < self.threshold:
                break
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step

        p_old = self.policy
        for i in range(self.grid_world.get_state_space()): #for v un V(s)
            #first assume 0 is the best policy and have biggest values
            next_state, reward, done = self.grid_world.step(i,0)
            highest_value = self.values(next_state)
            self.policy[i] = 0
            for move in range(1,4): #move=1,2,3
                next_state, reward, done = self.grid_world.step(i,move)
                if self.values(next_state) > highest_value: #then update
                    self.policy[i] = move
                    highest_value = self.values(next_state)
        if_stable = p_old == self.policy
        return if_stable
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        while True:
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                break
        raise NotImplementedError


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        raise NotImplementedError


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        raise NotImplementedError
