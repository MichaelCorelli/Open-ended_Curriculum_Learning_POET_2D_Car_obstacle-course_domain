import numpy as np
import random
import pygame
from main import CarEnvironment

RED = (255, 0, 0)

class POET:
    def __init__(self, car, agent, E_init, theta_init, alpha, noise_std, T, N_mutate, N_transfer, env_input_dim, hidden_dim, action_dim):
        self.car = car
        self.agent = agent
        
        self.E_init = E_init
        self.theta_init = theta_init
        self.alpha = alpha
        self.noise_std = noise_std
        self.T = T
        self.N_mutate = N_mutate
        self.N_transfer = N_transfer
        
        self.env_input_dim = env_input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        #number of samples
        self.n = 20
        #max number of children per reproduction
        self.max_children = 3
        #max number of children admitted per reproduction
        self.max_admitted = 3
        #max number of active environments
        self.capacity = 5
        #eligibility threshold
        self.threshold_el = 0.3
        #minimal criterion threshold
        self.threshold_c_min = 0.2
        self.threshold_c_max = 0.9
        #environment for novelty
        self.envs = []
        self.archive_envs = []

    def eligible_to_reproduce(self, E, theta):
        score = E(theta)
        print(f"Score: {score}, Threshold: {self.threshold_el}")
        return score >= self.threshold_el

    def mc_satisfied(self, child_list):
        res = []
        
        if not child_list:
            print("Empty child_list")
            return res

        for i in child_list:
            if not isinstance(i, tuple) or len(i) != 2:
                print(f"Item not valid in child_list: {i}")
                continue

        for E_child, theta_child in child_list:
            score = E_child(theta_child)
            if self.threshold_c_min <= score <= self.threshold_c_max:
                res.append((E_child, theta_child))

        print(f"Result in mc_satisfied: {res}")
        return res

    def rank_by_novelty(self, child_list):
        e = self.envs + self.archive_envs
        child_novelty = []

        for E_child, theta_child in child_list:
            dist = []
            for E, theta in e:
                dist.append(abs(E_child(np.zeros_like(theta)) - E(np.zeros_like(theta))))

            novelty_score = np.mean(dist)
            print(f"{theta_child} novelty score: {novelty_score}")
            child_novelty.append((E_child, theta_child, novelty_score))

        child_novelty_sorted = sorted(child_novelty, key=lambda x: x[2], reverse=True)
        print(f"child_novelty sorted: {child_novelty_sorted}")
        child_list_sorted = [(E_child, theta_child) for E_child, theta_child, _ in child_novelty_sorted]
        return child_list_sorted
    
    def remove_oldest(self, EA_list, num_removals):
        del EA_list[:num_removals]

    def env_reproduce(self, parent_list, max_children):
        child_list = []
        print(f"parent_list: {parent_list}")

        for E_parent, theta_parent in parent_list:
            for _ in range(max_children // len(parent_list)):
                obstacle_factor = np.random.normal(0, 0.5)
                print(f"obstacle_factor: {obstacle_factor}")
                score = E_parent(theta_parent) + obstacle_factor
                print(f"{theta_parent} score: {score} with obstacle_factor: {obstacle_factor}")

                theta_child = np.copy(theta_parent)
                child_tuple = (E_parent, theta_child)

                if not isinstance(child_tuple, tuple) or len(child_tuple) != 2:
                    print(f"Not valid child_tuple format: {child_tuple}")
                    continue

                self.envs.append((E_parent, theta_child))
                child_list.append(child_tuple)

        print(f"child_list: {child_list}")
        return child_list

    def mutate_envs(self, EA_list):
        parent_list = []

        M = len(EA_list)
        print("EA_List", EA_list)
        for m in range(M):
            E_m, theta_m = EA_list[m]
            if self.eligible_to_reproduce(E_m, theta_m):
                parent_list.append((E_m, theta_m))

        for i in parent_list:
            if not isinstance(i, tuple) or len(i) != 2:
                print(f"Item not valid in parent_list: {i}")
                continue

        child_list = self.env_reproduce(parent_list, self.max_children)
        print("child_list after env_reproduce", child_list)
        child_list = self.mc_satisfied(child_list)
        print("child_list after mc_satisfied", child_list)
        child_list = self.rank_by_novelty(child_list)
        print("child_list after rank_by_novelty", child_list)

        theta_list = [theta_m for _, theta_m in EA_list]
        admitted = 0
        for E_child, theta_child in child_list:
            theta_child = self.evaluate_candidates(theta_list, E_child, self.alpha, self.noise_std)
            if self.mc_satisfied([(E_child, theta_child)]):
                EA_list.append((E_child, theta_child))
                admitted += 1
                if admitted >= self.max_admitted:
                    break

        M = len(EA_list)
        if M > self.capacity:
            num_removals = M - self.capacity
            self.remove_oldest(EA_list, num_removals)

        return EA_list

    def es_step(self, theta_m_t, E_m, alpha, noise_std):
        theta_m_t = np.asarray(theta_m_t)

        if np.ndim(theta_m_t) == 1:
            epsilon = np.random.randn(self.n, len(theta_m_t))
        else:
            epsilon = np.random.randn(self.n, 1)

        E_i = np.array([E_m(theta_m_t + noise_std * epsilon_i) for epsilon_i in epsilon])

        res = 0
        for i in range(self.n):
            res += E_i[i] * epsilon[i]

        return alpha * (1 / self.n * noise_std) * res
    
    def evaluate_candidates(self, theta, E, alpha, noise_std):
        C = []
        
        M = len(theta)
        for m in range(M):
            theta_m = np.asarray(theta[m])
            C.append(theta[m])
            C.append(theta[m] + self.es_step(theta[m], E, alpha, noise_std))

        return np.argmax([E(theta) for theta in C])
    
    def update_environments(self, t):
        for i, (E, theta) in enumerate(self.envs):
            if E is None or not isinstance(E, CarEnvironment):
                E = CarEnvironment()
                self.envs[i] = (E, theta)

            obstacle_type = random.choice(["ramp", "bump", "hole"])

            width = random.uniform(6, 10)
            height = random.uniform(5, 20)

            if width <= 0 or height <= 0:
                print(f"Not valid width and/or height -> set default value.")
                width = 0.2
                height = 0.2

            modified_env_params = {
                "base_position": (random.uniform(10, 50), 1),
                "size": (width, height),
                "color": RED,
                "obstacle_type": obstacle_type
            }

            E.modify_env(modified_env_params)
            reward = E.evaluate_agent(self.agent)
            print(f"Reward: {reward}")

    def main_loop(self):
        EA_list = [(self.E_init, self.theta_init)]

        for t in range(self.T):
            if t >= 0 and t % self.N_mutate == 0:
                EA_list = self.mutate_envs(EA_list)

            self.update_environments(t)

            M = len(EA_list)
            for m in range(M):
                E_m, theta_m_t = EA_list[m]
                theta_m_t_1 = theta_m_t + self.es_step(theta_m_t, E_m, self.alpha, self.noise_std)

                if M > 1 and t % self.N_transfer == 0:
                    theta_b_a_m = [theta for j, (_, theta) in enumerate(EA_list) if j != m]
                    theta_top = self.evaluate_candidates(theta_b_a_m, E_m, self.alpha, self.noise_std)
                    print(f"theta_top: {theta_top}")
                    if E_m(theta_top) > E_m(theta_m_t_1):
                        theta_m_t_1 = theta_top

                EA_list[m] = (E_m, theta_m_t_1)

            pygame.display.update()