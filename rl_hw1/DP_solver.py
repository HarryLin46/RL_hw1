import numpy as np
import copy
import heapq

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld,discount_factor: float ):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        # print("gamma:",discount_factor)
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        # self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)
        # self.policy = policy  # pi(s)
        self.policy = np.ones((grid_world.get_state_space(), 4)) / 4

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
        # next_action = self.policy[next_state]
        # print(self.policy)
        
        # state_coord = self.grid_world.get_state_list()[state]
        # print("state coord:",state_coord)
        # print("Type of self.grid_world:",type(self.grid_world))
        
        # if self.grid_world.__is_goal_state(state_coord): #it's a goal
        #     return reward
        # elif self.grid_world.__is_trap_state(state_coord): #it's a trap
        #     return reward
        # if state == next_state: # it hits the wall and stuck, or it's terminal
        #     return reward

        q_value = reward + self.discount_factor * self.values[next_state](1-done)
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
        # print("gamma in class iter:",discount_factor)
        super().__init__(grid_world,discount_factor)
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
            # print(i)
            v_new = 0
            for move in range(4): #move=0,1,2,3
                next_state, reward, done = self.grid_world.step(i,move)
                # if not i==next_state:
                v_new += self.policy[i,move]*(reward+self.discount_factor*v_copy[next_state]*(1-done))
                # else:
                #     v_new += self.policy[i,move]*(reward)
                # if i==0:
                    # print(f"after action {move} of v_new:",v_new)

            self.values[i] = v_new
        return
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        count=0
        while True:
            v_old = copy.deepcopy(self.values)
            # print("v_old:",v_old)
            self.evaluate()
            # print("self.values:",self.values)
            delta = np.max(np.abs(self.values - v_old))
            # print("delta:",delta)
            if delta < self.threshold:
                return
            count+=1
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
        v = 0
        for move in range(4):#move=0,1,2,3
            v += self.policy[state,move] * self.get_q_value(state,move)
        return v
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        #synchronize: a copy is needed
        def evaluate_per_iter():
            #synchronize: a copy is needed
            v_copy = copy.deepcopy(self.values)

            #start update
            for i in range(self.grid_world.get_state_space()): #for i in V(s)
                # print(i)
                v_new = 0
                for move in range(4): #move=0,1,2,3
                    next_state, reward, done = self.grid_world.step(i,move)
                    # if not i==next_state:
                    v_new += self.policy[i,move]*(reward+self.discount_factor*v_copy[next_state]*(1-done))
                    # else:
                    #     v_new += self.policy[i,move]*(reward)
                    # if i==0:
                        # print(f"after action {move} of v_new:",v_new)

                self.values[i] = v_new
            return

        while True:
            v_old = copy.deepcopy(self.values)
            # print("v_old:",v_old)
            evaluate_per_iter()
            # print("self.values:",self.values)
            delta = np.max(np.abs(self.values - v_old))
            # print("delta:",delta)
            if delta < self.threshold:
                print("finish evaluation:",self.values)
                return
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step

        p_old = copy.deepcopy(self.policy)
        for i in range(self.grid_world.get_state_space()): #for v un V(s)
            #first assume 0 is the best policy and have biggest values
            # next_state, reward, done = self.grid_world.step(i,0)
            # # highest_value = self.values[next_state]
            # self.policy[i,0] = 1
            #find the movements with maximum value
            values_nearby = np.zeros(4) 
            for move in range(4): #move=0,1,2,3
                next_state, reward, done = self.grid_world.step(i,move)
                values_nearby[move] = self.values[next_state]
            
            highest_value = np.max(values_nearby)
            highest_value_indices = np.where(values_nearby == highest_value)[0] #ex. [2], [1,3]

            update_policy = np.zeros(4)
            update_policy[highest_value_indices] = 1.0 / len(highest_value_indices) #ex. [0,1.0,0,0], [0,0.5,0,0.5]

            self.policy[i] = update_policy

        if_stable = (p_old == self.policy).all()
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
        
        #self policy is stochastic, with shape [#_of_state,4], saving the probability
        #change it to the deterministic one, with shape [#_of_state], , saving the moving direction
        self.policy = np.argmax(self.policy, axis=1)
        
        return
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
        v = 0
        for move in range(4):#move=0,1,2,3
            v += self.policy[state,move] * self.get_q_value(state,move)
        return v
        raise NotImplementedError

    # def policy_evaluation(self):
    #     """Evaluate the policy and update the values"""
    #     # TODO: Implement the policy evaluation step
    #     raise NotImplementedError

    # def policy_improvement(self):
    #     """Improve the policy based on the evaluated values"""
    #     # TODO: Implement the policy improvement step
    #     raise NotImplementedError
    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        #synchronize: a copy is needed
        v_copy = copy.deepcopy(self.values)

        #start update
        for i in range(self.grid_world.get_state_space()): #for i in V(s)
            # print(i)
            next_state, reward, done = self.grid_world.step(i,0)
            v_new = reward+self.discount_factor*v_copy[next_state]*(1-done)
            for move in range(1,4): #move=0,1,2,3
                next_state, reward, done = self.grid_world.step(i,move)
                # if not i==next_state:
                v_new = max(v_new,reward+self.discount_factor*v_copy[next_state]*(1-done))
                # else:
                #     v_new += self.policy[i,move]*(reward)
                # if i==0:
                    # print(f"after action {move} of v_new:",v_new)

            self.values[i] = v_new
        return

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while True:
            v_old = copy.deepcopy(self.values)
            # print("v_old:",v_old)
            self.evaluate()
            # print("self.values:",self.values)
            delta = np.max(np.abs(self.values - v_old))
            # print("delta:",delta)
            if delta < self.threshold:
                break
        
        #when V(s) is optimal, we can use it to find optimal policy
        for i in range(self.grid_world.get_state_space()): #for v un V(s)
            #first assume 0 is the best policy and have biggest values
            # next_state, reward, done = self.grid_world.step(i,0)
            # # highest_value = self.values[next_state]
            # self.policy[i,0] = 1
            #find the movements with maximum value
            values_nearby = np.zeros(4) 
            for move in range(4): #move=0,1,2,3
                next_state, reward, done = self.grid_world.step(i,move)
                values_nearby[move] = self.values[next_state]
            
            highest_value = np.max(values_nearby)
            highest_value_indices = np.where(values_nearby == highest_value)[0] #ex. [2], [1,3]

            update_policy = np.zeros(4)
            update_policy[highest_value_indices] = 1.0 / len(highest_value_indices) #ex. [0,1.0,0,0], [0,0.5,0,0.5]

            self.policy[i] = update_policy

        #self policy is stochastic, with shape [#_of_state,4], saving the probability
        #change it to the deterministic one, with shape [#_of_state], , saving the moving direction
        self.policy = np.argmax(self.policy, axis=1)
        return
        raise NotImplementedError


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    #in-place DP
    def evaluate_in_place(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        #asynchronize: a copy is not needed anymore
        # v_copy = copy.deepcopy(self.values)

        #start update
        for i in range(self.grid_world.get_state_space()): #for i in V(s)
            # print(i)
            next_state, reward, done = self.grid_world.step(i,0)
            v_new = reward+self.discount_factor*self.values[next_state]*(1-done)
            for move in range(1,4): #move=0,1,2,3
                next_state, reward, done = self.grid_world.step(i,move)
                # if not i==next_state:
                v_new = max(v_new,reward+self.discount_factor*self.values[next_state]*(1-done))
                # else:
                #     v_new += self.policy[i,move]*(reward)
                # if i==0:
                    # print(f"after action {move} of v_new:",v_new)

            self.values[i] = v_new
        return
    
    #Prioritize sweeping

    #define priority queue
    class Priority_queue:
        def __init__(self):
            self.queue = []

        def insert(self, error, state):
            heapq.heappush(self.queue, (-1*error, state)) #heapq will pop the state with "smallest(minus big)" error

        def pop(self):
            if len(self.queue) > 0:
                inverse_error, state = heapq.heappop(self.priority_queue)
                return -1*inverse_error, state
            else:
                return None, None


    def evaluate_prior_sweep(self,state):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        #asynchronize: a copy is not needed anymore
        # v_copy = copy.deepcopy(self.values)

        #determine the priority queue

        #start update
        next_state, reward, done = self.grid_world.step(state,0)
        v_new = reward+self.discount_factor*self.values[next_state]*(1-done)
        for move in range(1,4): #move=0,1,2,3
            next_state, reward, done = self.grid_world.step(state,move)
            # if not i==next_state:
            v_new = max(v_new,reward+self.discount_factor*self.values[next_state]*(1-done))
            # else:
            #     v_new += self.policy[i,move]*(reward)
            # if i==0:
                # print(f"after action {move} of v_new:",v_new)

        self.values[state] = v_new
        return
    
    def calculate_B_error(self,state):
        next_state, reward, done = self.grid_world.step(state,0)
        v_new = reward+self.discount_factor*self.values[next_state]*(1-done)
        for move in range(1,4): #move=0,1,2,3
            next_state, reward, done = self.grid_world.step(state,move)
            # if not i==next_state:
            v_new = max(v_new,reward+self.discount_factor*self.values[next_state]*(1-done))
            # else:
            #     v_new += self.policy[i,move]*(reward)
            # if i==0:
                # print(f"after action {move} of v_new:",v_new)
        return abs(self.values[state]-v_new)


    def run(self) -> None:
        """Run the algorithm until convergence"""
        is_inplace = False
        is_prior_sweep = True
        # TODO: Implement the async dynamic programming algorithm until convergence
        if is_inplace:
            while True:
                v_old = copy.deepcopy(self.values)
                # print("v_old:",v_old)
                self.evaluate_in_place()
                # print("self.values:",self.values)
                delta = np.max(np.abs(self.values - v_old))
                # print("delta:",delta)
                if delta < self.threshold:
                    break
            
            #when V(s) is optimal, we can use it to find optimal policy
            for i in range(self.grid_world.get_state_space()): #for v un V(s)
                #first assume 0 is the best policy and have biggest values
                # next_state, reward, done = self.grid_world.step(i,0)
                # # highest_value = self.values[next_state]
                # self.policy[i,0] = 1
                #find the movements with maximum value
                values_nearby = np.zeros(4) 
                for move in range(4): #move=0,1,2,3
                    next_state, reward, done = self.grid_world.step(i,move)
                    values_nearby[move] = self.values[next_state]
                
                highest_value = np.max(values_nearby)
                highest_value_indices = np.where(values_nearby == highest_value)[0] #ex. [2], [1,3]

                update_policy = np.zeros(4)
                update_policy[highest_value_indices] = 1.0 / len(highest_value_indices) #ex. [0,1.0,0,0], [0,0.5,0,0.5]

                self.policy[i] = update_policy

            #self policy is stochastic, with shape [#_of_state,4], saving the probability
            #change it to the deterministic one, with shape [#_of_state], , saving the moving direction
            self.policy = np.argmax(self.policy, axis=1)
            return
        
        elif is_prior_sweep:
            #Method2: use a heap to represent a queue (more complicated)
            #initially, create queue and add every state in queue
            # pq = self.Priority_queue()

            #use a dict to remember the update times of each state, outdated ine will be discarded

            #use an array, B_error to remember the Bellman error, delta=np.max(B_error)

            #Method1: use an array,B_error to remember the Bellman error, candidate = the index with max error, delta=np.max(B_error)
            B_error = np.zeros(self.grid_world.get_state_space())
            for i in range(self.grid_world.get_state_space()): #for v un V(s)
                 B_error[i] = self.calculate_B_error(i)

            # print(B_error)

            count=0
            while True:
                urgent_state = np.argmax(B_error)
                print(urgent_state)
                self.evaluate_prior_sweep(urgent_state)

                print("before:",B_error)
                print("values:",self.values)

                #update B_error, there are 5 states to update
                B_error[urgent_state] = self.calculate_B_error(urgent_state)
                
                for move in range(4): #move=0,1,2,3
                    next_state, reward, done = self.grid_world.step(urgent_state,move)
                    update_error = self.calculate_B_error(next_state)
                    B_error[next_state] = update_error

                print("after:",B_error)
                delta = np.max(B_error)
                # print(delta)
                count+=1
                if delta < self.threshold:# or count>12: #then self.values is optimal
                    break
        
            #when V(s) is optimal, we can use it to find optimal policy
            for i in range(self.grid_world.get_state_space()): #for v un V(s)
                #first assume 0 is the best policy and have biggest values
                # next_state, reward, done = self.grid_world.step(i,0)
                # # highest_value = self.values[next_state]
                # self.policy[i,0] = 1
                #find the movements with maximum value
                values_nearby = np.zeros(4) 
                for move in range(4): #move=0,1,2,3
                    next_state, reward, done = self.grid_world.step(i,move)
                    values_nearby[move] = self.values[next_state]
                
                highest_value = np.max(values_nearby)
                highest_value_indices = np.where(values_nearby == highest_value)[0] #ex. [2], [1,3]

                update_policy = np.zeros(4)
                update_policy[highest_value_indices] = 1.0 / len(highest_value_indices) #ex. [0,1.0,0,0], [0,0.5,0,0.5]

                self.policy[i] = update_policy

            #self policy is stochastic, with shape [#_of_state,4], saving the probability
            #change it to the deterministic one, with shape [#_of_state], , saving the moving direction
            self.policy = np.argmax(self.policy, axis=1)
            return

            
        raise NotImplementedError
