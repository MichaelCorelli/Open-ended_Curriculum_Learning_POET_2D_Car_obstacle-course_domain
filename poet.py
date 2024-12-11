import numpy as np
import random
import pygame
from env import CarEnvironment
from utils import state_dict_to_vector, vector_to_state_dict
from tqdm import tqdm


RED = (255, 0, 0)


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
        
        # Store a reference state_dict for conversions
        self.reference_state_dict = ddqn_agent.network.network.state_dict()

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
        #max size for archive_envs
        self.archive_envs_max_size = 5
        
    def _evaluate_agent_with_vector(self, E, theta_vector):
        # Convert vector to state_dict
        print(f"Vector shape: {theta_vector.shape}")
        total_params = sum(p.numel() for p in self.reference_state_dict.values())
        print(f"Expected total params: {total_params}")
        theta_sd = vector_to_state_dict(theta_vector, self.reference_state_dict)
        return E.evaluate_agent(self.ddqn_agent, theta_sd)

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
            mean_reward = score['mean_reward']  # Estrai il valore numerico
            if self.threshold_c_min <= mean_reward <= self.threshold_c_max:
                res.append((E_child, theta_child))

        print(f"Result in mc_satisfied: {res}")
        return res

    def rank_by_novelty(self, child_list):
        e = self.envs + self.archive_envs
        child_novelty = []
        # theta is no longer needed here separately since we evaluate each child's novelty
        # based on previously stored environments. They all have vectors now.

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
            removed = self.envs[:num_removals]
            self.envs = self.envs[num_removals:]
            self.archive_envs.extend(removed)

        self.archive(archive_envs_max_size = self.archive_envs_max_size)

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
    
    #evaluate_candidiates con es_step
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

    #evaluate_candidiates con ddqn_agent
    def evaluate_candidates(self, theta_list, E, alpha, noise_std):
        # theta_list is a list of vectors
        theta_m_t = np.mean(theta_list, axis=0)

        epsilon = np.random.randn(self.n, theta_m_t.shape[0])
        E_i = []

        # ES step approximation
        for epsilon_i in epsilon:
            theta_variation = theta_m_t + noise_std * epsilon_i
            # Convert to state_dict to load
            theta_variation_sd = vector_to_state_dict(theta_variation, self.reference_state_dict)
            self.ddqn_agent.network.network.load_state_dict(theta_variation_sd)
            performance = E.evaluate_agent(self.ddqn_agent, theta_variation_sd)
            E_i.append(performance)

        E_i = np.array(E_i)
        gradient_estimation = np.sum(E_i[:, np.newaxis] * epsilon, axis=0)
        theta_updated = theta_m_t + alpha * (1 / (self.n * noise_std)) * gradient_estimation

        # Now we have candidate set C: original theta_list plus theta_updated
        C = theta_list + [theta_updated]
        performances = []
        for theta in C:
            theta_sd = vector_to_state_dict(theta, self.reference_state_dict)
            self.ddqn_agent.network.network.load_state_dict(theta_sd)
            performances.append(E.evaluate_agent(self.ddqn_agent, theta_sd))

        i_best = np.argmax(performances)
        best_theta = C[i_best]

        return best_theta
    
    def update_environments(self, t):
        for i, (E, theta) in enumerate(self.envs):
            if E is None:
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
            # Evaluate agent in the updated environment with current ddqn weights (no theta provided means internal agent weights)
            reward = E.evaluate_agent(self.ddqn_agent, None)
            print(f"Reward: {reward}")

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
                    self.ddqn_agent.train(e_max=10, gamma=0.99, frequency_update=10, frequency_sync=100)
                            
                    #theta_m_t_1 = theta_m_t + self.es_step(theta_m_t, E_m, self.alpha, self.noise_std)
                    # After training, get the updated theta from agent (state_dict)
                    updated_sd = self.ddqn_agent.network.network.state_dict()
                    updated_vec = state_dict_to_vector(updated_sd)
                    theta_m_t_1 = updated_vec

                    if M > 1 and t % self.N_transfer == 0:
                        theta_b_a_m = [theta for j, (_, theta) in enumerate(EA_list) if j != m]
                        theta_top = self.evaluate_candidates(theta_b_a_m, E_m, self.alpha, self.noise_std)
                        # Compare performance of theta_top and theta_m_t_1
                        top_score = self._evaluate_agent_with_vector(E_m, theta_top)
                        current_score = self._evaluate_agent_with_vector(E_m, theta_m_t_1)
                        print(f"theta_top score: {top_score}, current score: {current_score}")
                        if top_score > current_score:
                            theta_m_t_1 = theta_top

                    EA_list[m] = (E_m, theta_m_t_1)
                pbar.update(1)
                pygame.display.update()