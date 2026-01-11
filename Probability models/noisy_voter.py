import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

class NoisyVoterModel:
    def __init__(self, n, r, a=1.0, b=1.0, initial_state=None):
        """
        Noisy Voter Model on complete graph
        
        Args:
            n: number of vertices
            r: strength of noise (can be > 1)
            a: bias for 0 -> 1 spontaneous switching
            b: bias for 1 -> 0 spontaneous switching
            initial_state: initial configuration (None for random)
        """
        self.n = n
        self.r = r
        self.a = a
        self.b = b
        if initial_state is None:
            self.state = np.random.randint(0, 2, n)
        else:
            self.state = np.array(initial_state)
    
    def simulate(self, T):
        """
        Simulate noisy voter model using Gillespie algorithm up to time T
        """
        t = 0
        times = [0]
        states = [self.state.copy()]
        
        while t < T:
            ones = np.sum(self.state)
            zeros = self.n - ones
            
            # Voting rate: pairs of different states (interaction) scaled by n^2
            voting_rate = 2 * ones * zeros / (self.n ** 2)
            # State-dependent spontaneous flip rates scaled by n^2
            noise_rate_0to1 = self.r * self.a * zeros / (self.n ** 2)
            noise_rate_1to0 = self.r * self.b * ones  / (self.n ** 2)
            total_rate = voting_rate + noise_rate_0to1 + noise_rate_1to0
            
            if total_rate == 0:
                break
            
            dt = np.random.exponential(1 / total_rate)
            t += dt
            
            if t > T:
                break
            
            u = np.random.rand()
            if u < voting_rate / total_rate:
                # Voting event: pick a voter and a target of opposite type
                voter = np.random.randint(0, self.n)
                # Find all vertices with opposite state
                opposite_state = 1 - self.state[voter]
                targets = np.where(self.state == opposite_state)[0]
                if len(targets) > 0:
                    target = np.random.choice(targets)
                    self.state[voter] = self.state[target]
            elif u < (voting_rate + noise_rate_0to1) / total_rate:
                # Spontaneous 0 -> 1 flip
                if zeros > 0:
                    zero_indices = np.where(self.state == 0)[0]
                    vertex = np.random.choice(zero_indices)
                    self.state[vertex] = 1
            else:
                # Spontaneous 1 -> 0 flip
                if ones > 0:
                    one_indices = np.where(self.state == 1)[0]
                    vertex = np.random.choice(one_indices)
                    self.state[vertex] = 0
            
            times.append(t)
            states.append(self.state.copy())
        
        return np.array(times), np.array(states)
    
    def get_proportion(self):
        """
        Returns the proportion of vertices with state 1
        """
        return np.sum(self.state) / self.n

    def get_state(self):
        return self.state.copy()

    def get_n(self):
        return self.n

    def get_r(self):
        return self.r

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b
    
    def show(self):
        """
        Display the current state as a graph with nodes colored by state
        """
        G = nx.empty_graph(self.n)
        colors = ['red' if self.state[i] == 1 else 'blue' for i in range(self.n)]
        
        plt.figure(figsize=(8, 8))
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300)
        nx.draw_networkx_labels(G, pos)
        plt.axis('off')
        plt.title(f"Noisy Voter Model (Proportion of 1s: {self.get_proportion():.2f})")
        plt.show()
    
    def simulate_and_show(self, T, interval=None):
        """
        Simulate up to time T and show plot changing over time
        """
        t = 0
        times = [0]
        states = [self.state.copy()]
        
        while t < T:
            ones = np.sum(self.state)
            zeros = self.n - ones
            
            # Voting rate: pairs of different states (interaction) scaled by n^2
            voting_rate = 2 * ones * zeros / (self.n ** 2)
            # State-dependent spontaneous flip rates scaled by n^2
            noise_rate_0to1 = self.r * self.a * zeros / (self.n ** 2)
            noise_rate_1to0 = self.r * self.b * ones  / (self.n ** 2)
            total_rate = voting_rate + noise_rate_0to1 + noise_rate_1to0
            
            if total_rate == 0:
                break
            
            dt = np.random.exponential(1 / total_rate)
            t += dt
            
            if t > T:
                break
            
            u = np.random.rand()
            if u < voting_rate / total_rate:
                # Voting event: pick a voter and a target of opposite type
                voter = np.random.randint(0, self.n)
                # Find all vertices with opposite state
                opposite_state = 1 - self.state[voter]
                targets = np.where(self.state == opposite_state)[0]
                if len(targets) > 0:
                    target = np.random.choice(targets)
                    self.state[voter] = self.state[target]
            elif u < (voting_rate + noise_rate_0to1) / total_rate:
                # Spontaneous 0 -> 1 flip
                if zeros > 0:
                    zero_indices = np.where(self.state == 0)[0]
                    vertex = np.random.choice(zero_indices)
                    self.state[vertex] = 1
            else:
                # Spontaneous 1 -> 0 flip
                if ones > 0:
                    one_indices = np.where(self.state == 1)[0]
                    vertex = np.random.choice(one_indices)
                    self.state[vertex] = 0
            
            times.append(t)
            states.append(self.state.copy())
        
        fractions = np.mean(states, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, fractions)
        plt.xlabel("Time")
        plt.ylabel("Fraction of state 1")
        plt.title(f"Noisy Voter Model (n={self.n}, r={self.r}, a={self.a}, b={self.b})")
        plt.grid()
        plt.show()
# Example usage
if __name__ == "__main__":
    n = 200
    r = 15.0
    T = 3000.0
    a = 1.0
    b = 2.0

    # Initial condition: first half ones, second half zeros
    initial_state = np.zeros(n, dtype=int)
    initial_state[: n // 2] = 1
    model = NoisyVoterModel(n, r, a=a, b=b, initial_state=initial_state)
    model.simulate_and_show(T)
    # # To estimate probability that first half dominates after time T
    # repeats=100
    # first_half_dominant=0
    # for i in range(repeats):
    #      model = NoisyVoterModel(n, r, a=a, b=b, initial_state=initial_state)
    #      model.simulate(T)
    #      if model.get_state()[: n // 2].sum() > n / 4:
    #          first_half_dominant += 1
    # print(f"Proportion of runs where first half dominates: {first_half_dominant / repeats:.2f}")
    