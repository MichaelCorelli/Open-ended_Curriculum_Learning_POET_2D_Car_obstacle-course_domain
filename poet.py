import numpy as np
import random
import pygame
from env import CarEnvironment
from utils import state_dict_to_vector, vector_to_state_dict
from tqdm import tqdm

BLACK = (0, 0, 0)

class POET:
    def __init__(self, car, ddqn_agent, E_init, theta_init, alpha, noise_std, T, N_mutate, N_transfer, env_input_dim, hidden_dim, action_dim):
        self.car = car
        self.ddqn_agent = ddqn_agent
        self.E_init = E_init
        self.theta_init = theta_init
        self.theta = theta_init
        self.alpha = alpha
        self.noise_std = noise_std
        self.T = T
        self.N_mutate = N_mutate
        self.N_transfer = N_transfer
        self.env_input_dim = env_input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.reference_state_dict = ddqn_agent.network.network.state_dict()

        self.n = 20
        self.max_children = 3
        self.max_admitted = 3
        self.capacity = 5
        self.threshold_el = 0.3
        self.threshold_c_min = 0.2
        self.threshold_c_max = 0.9
        self.envs = []
        self.archive_envs = []
        self.archive_envs_max_size = 5
        self.r_history = {}

    def update_threshold_c(self, r_history, window_size = 5, range = 0.1):

        if len(r_history) < window_size:
            r = r_history
        else:
            r = r_history[-window_size:]

        r_mean = np.mean(r)
        self.threshold_c_min = max(0, r_mean - range)
        self.threshold_c_max = min(1, r_mean + range)

        print(f"Thresholds: min = {self.threshold_c_min}, max = {self.threshold_c_max}")

    def update_threshold_el(self, r_history, window_size = 5, v = 0.03):

        if len(r_history) < window_size:
            r = r_history
        else:
            r = r_history[-window_size:]

        r_mean = np.mean(r)

        if r_mean < self.threshold_el:
            self.threshold_el = max(0, self.threshold_el - v)
        else:
            self.threshold_el = min(1, self.threshold_el + v)

        print(f"Eligibility threshold: {self.threshold_el}")
        
    def _evaluate_agent_with_vector(self, E, theta_vector):
        print(f"Vector shape: {theta_vector.shape}")
        total_params = sum(p.numel() for p in self.reference_state_dict.values())
        print(f"Expected total params: {total_params}")
        theta_sd = vector_to_state_dict(theta_vector, self.reference_state_dict)
        return E.evaluate_agent(self.ddqn_agent, theta_sd, verbose=False)

    def eligible_to_reproduce(self, E, theta):
        score = self._evaluate_agent_with_vector(E, theta)
        print(f"Score: {score}, Threshold: {self.threshold_el}")
        mean_reward = score['mean_reward']
        return mean_reward >= self.threshold_el

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
            score = self._evaluate_agent_with_vector(E_child, theta_child)
            mean_reward = score['mean_reward']
            if self.threshold_c_min <= mean_reward <= self.threshold_c_max:
                res.append((E_child, theta_child))

        print(f"Result in mc_satisfied: {res}")
        return res

    def rank_by_novelty(self, child_list):
        e = self.envs + self.archive_envs
        child_novelty = []

        for E_child, theta_child in child_list:
            dist = []
            for E, theta in e:
                score_diff = self._evaluate_agent_with_vector(E_child, theta_child) - self._evaluate_agent_with_vector(E, theta)
                dist.append(abs(score_diff))

            novelty_score = np.mean(dist) if dist else 0.0
            print(f"{theta_child} novelty score: {novelty_score}")
            child_novelty.append((E_child, theta_child, novelty_score))

        child_novelty_sorted = sorted(child_novelty, key=lambda x: x[2], reverse=True)
        print(f"child_novelty sorted: {child_novelty_sorted}")
        child_list_sorted = [(E_child, theta_child) for E_child, theta_child, _ in child_novelty_sorted]
        return child_list_sorted
    
    def remove_oldest(self, EA_list, num_removals):
        del EA_list[:num_removals]

    def archive(self, archive_envs_max_size):
        if len(self.archive_envs) > archive_envs_max_size:
            self.archive_envs = self.archive_envs[-archive_envs_max_size:]

    def env_reproduce(self, parent_list, max_children):
        child_list = []
        print(f"parent_list: {parent_list}")

        for E_parent, theta_parent in parent_list:
            for _ in range(max_children // len(parent_list)):
                
                E_child = E_parent.clone()  
                E_child.mutate_environment() 
                
                theta_parent_sd = vector_to_state_dict(theta_parent, self.reference_state_dict)
                score = E_child.evaluate_agent(self.ddqn_agent, theta_parent_sd)
                print(f"{theta_parent} score: {score} in new environment")

                theta_child = np.copy(theta_parent).flatten()
                child_tuple = (E_child, theta_child)

                if not isinstance(child_tuple, tuple) or len(child_tuple) != 2:
                    print(f"Not valid child_tuple format: {child_tuple}")
                    continue

                self.envs.append((E_child, theta_child))
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
            removed_envs = EA_list[:num_removals]
            self.archive_envs.extend(removed_envs)

            if len(self.archive_envs) > self.archive_envs_max_size:
                excess = len(self.archive_envs) - self.archive_envs_max_size
                self.archive_envs = self.archive_envs[excess:]

            self.remove_oldest(EA_list, num_removals)

        self.envs = EA_list.copy()

        for E, theta in EA_list:
            key = (E, tuple(theta))
            if key not in self.r_history:
                self.r_history[key] = []

            score = E.evaluate_agent(self.ddqn_agent, None)
            r_mean = score['r_mean']
            self.r_history[key].append(r_mean)

        self.update_threshold_c([r for rewards in self.r_history.values() for r in rewards])
        self.update_threshold_el([r for rewards in self.r_history.values() for r in rewards])

        return EA_list

    #es_step
    '''
    def es_step(self, theta_m_t, E_m, alpha, noise_std):
        theta_m_t = np.asarray(theta_m_t)

        if np.ndim(theta_m_t) == 1:
            epsilon = np.random.randn(self.n, len(theta_m_t))
        else:
            epsilon = np.random.randn(self.n, 1)

        E_i = np.array([E_m.evaluate_agent(self.ddqn_agent, theta_m_t + noise_std * epsilon_i) for epsilon_i in epsilon])

        res = np.sum(E_i[:, np.newaxis] * epsilon, axis = 0)

        return alpha * (1 / self.n * noise_std) * res
    '''
    
    #evaluate_candidiates with es_step
    '''
    def evaluate_candidates(self, theta_list, E, alpha, noise_std):
        C = []
        M = len(theta_list)
        for m in range(M):
            theta_m = np.asarray(theta_list[m])
            C.append(theta_m)
            C.append(theta_m + self.es_step(theta_m, E, alpha, noise_std))

        performances = [E.evaluate_agent(self.ddqn_agent, theta) for theta in C]
        i_best = np.argmax(performances)
        best_theta = C[i_best]
        return best_theta
    '''

    #evaluate_candidiates with ddqn_agent
    def evaluate_candidates(self, theta_list, E, alpha, noise_std):
        theta_m_t = np.mean(theta_list, axis=0)

        epsilon = np.random.randn(self.n, theta_m_t.shape[0])
        E_i = []

        for epsilon_i in epsilon:
            theta_variation = theta_m_t + noise_std * epsilon_i
            theta_variation_sd = vector_to_state_dict(theta_variation, self.reference_state_dict)
            self.ddqn_agent.network.network.load_state_dict(theta_variation_sd)
            performance = E.evaluate_agent(self.ddqn_agent, theta_variation_sd)
            E_i.append(performance)

        E_i = np.array(E_i)
        gradient_estimation = np.sum(E_i[:, np.newaxis] * epsilon, axis=0)
        theta_updated = theta_m_t + alpha * (1 / (self.n * noise_std)) * gradient_estimation

        C = theta_list + [theta_updated]
        performances = []
        for theta in C:
            theta_sd = vector_to_state_dict(theta, self.reference_state_dict)
            self.ddqn_agent.network.network.load_state_dict(theta_sd)
            performances.append(E.evaluate_agent(self.ddqn_agent, theta_sd))

        i_best = np.argmax(performances)
        best_theta = C[i_best]

        return best_theta
    
    def update_environments(self, t, d_min = 10, i_max = 100):

        threshold_incr_init = 0.8
        threshold_decr_init = 0.4
        rate_init = 0.1
        difficulty_max_init = 2.0

        for i, (E, theta) in enumerate(self.envs):
            if E is None:
                E = CarEnvironment()
                self.envs[i] = (E, theta)

            score = E.evaluate_agent(self.ddqn_agent, None)
            r_mean = score['r_mean']

            threshold_difficulty_incr = min(1, threshold_incr_init + 0.15 * (t / 1000))
            threshold_difficulty_decr = max(0, threshold_decr_init - 0.15 * (t / 1000))
            rate_difficulty = rate_init * (1 + 0.02 * t)
            difficulty_max = min(5.0, difficulty_max_init + 0.003 * t)

            difficulty_factor = 1
            if r_mean >= threshold_difficulty_incr:
                difficulty_factor += rate_difficulty * (r_mean - threshold_difficulty_incr)
            elif r_mean <= threshold_difficulty_decr:
                difficulty_factor -= rate_difficulty * (threshold_difficulty_decr - r_mean)

            difficulty_factor = min(difficulty_max, max(1, difficulty_factor))

            obstacles_n = int(difficulty_factor * 2)
            existing_p = E.obstacles if hasattr(E, "obstacles") else []

            for _ in range(obstacles_n):
                for _ in range(i_max):
                    p = (random.uniform(15, 60), random.uniform(2, 25))
                    
                    if all(np.linalg.norm(np.array(p) - np.array(pos)) >= d_min for pos in existing_p):
                        existing_p.append(p)
                        break

                else:
                    continue

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
                "color": BLACK,
                "obstacle_type": obstacle_type
            }

            E.modify_env(modified_env_params)

            E.obstacles = existing_p
            
            reward = E.evaluate_agent(self.ddqn_agent, None)
            print(f"Reward: {reward}")

            key = (E, tuple(theta))
            if key not in self.r_history:
                self.r_history[key] = []
            self.r_history[key].append(r_mean)

            self.update_threshold_c(self.r_history)
            self.update_threshold_el(self.r_history)

    def main_loop(self):
        EA_list = [(self.E_init, self.theta_init)]

        with tqdm(total=self.T, desc="POET Main Loop Progress") as pbar:
            for t in range(self.T):
                if t >= 0 and t % self.N_mutate == 0:
                    EA_list = self.mutate_envs(EA_list)

                self.update_environments(t)

                M = len(EA_list)
                for m in range(M):
                    E_m, theta_m_t = EA_list[m]

                    self.ddqn_agent.env = E_m
                    print(f"Training agent on env: {m}")
                    self.ddqn_agent.train(e_max=100, gamma=0.99, frequency_update=10, frequency_sync=100)

                    score = E_m.evaluate_agent(self.ddqn_agent, None)
                    print(f"Score: {score}")

                    r_mean = score['r_mean']

                    key = (E_m, tuple(theta_m_t))
                    if key not in self.r_history:
                        self.r_history[key] = []
                    self.r_history[key].append(r_mean)

                    self.update_threshold_c(self.r_history[key])
                    self.update_threshold_el(self.r_history[key])
                            
                    #theta_m_t_1 = theta_m_t + self.es_step(theta_m_t, E_m, self.alpha, self.noise_std)
                    updated_sd = self.ddqn_agent.network.network.state_dict()
                    updated_vec = state_dict_to_vector(updated_sd)
                    theta_m_t_1 = updated_vec

                    if M > 1 and t % self.N_transfer == 0:
                        theta_b_a_m = [theta for j, (_, theta) in enumerate(EA_list) if j != m]
                        theta_top = self.evaluate_candidates(theta_b_a_m, E_m, self.alpha, self.noise_std)
                        top_score = self._evaluate_agent_with_vector(E_m, theta_top)
                        current_score = self._evaluate_agent_with_vector(E_m, theta_m_t_1)
                        print(f"theta_top score: {top_score}, current score: {current_score}")
                        if top_score > current_score:
                            theta_m_t_1 = theta_top

                    EA_list[m] = (E_m, theta_m_t_1)
                pbar.update(1)
                pygame.display.update()