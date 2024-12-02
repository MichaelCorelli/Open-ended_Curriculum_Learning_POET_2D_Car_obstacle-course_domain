import numpy as np

class POET:
    def __init__(self, E_init, theta_init, alpha, noise_std, T, N_mutate, N_transfer):
        self.E_init = E_init
        self.theta_init = theta_init
        self.alpha = alpha
        self.noise_std = noise_std
        self.T = T
        self.N_mutate = N_mutate
        self.N_transfer = N_transfer

        #number of samples
        self.n = 20
        #max number of children per reproduction
        self.max_children = 3
        #max number of children admitted per reproduction
        self.max_admitted = 10
        #max number of active environments
        self.capacity = 10
        #eligibility threshold
        self.threshold_el = 0.6
        #minimal criterion threshold
        self.threshold_c_min = 0.3
        self.threshold_c_max = 0.7
        #environment for novelty
        self.envs = []
        self.archive_envs = []

    def eligible_to_reproduce(self, E_m, theta_m):
        return E_m(theta_m) >= self.threshold_el
    
    def env_reproduce(self, parent_list, max_children):
        child_list = []
        for E_parent, theta_parent in parent_list:
            for _ in range(max_children // len(parent_list)):

                def E_child(theta):
                    return E_parent(theta) + np.random.normal(0, 0.2)
                
                theta_child = np.copy(theta_parent)
                child_list.append((E_child, theta_child))
                
        return child_list
    
    def mc_satisfied(self, child_list):
        res = [(E_child, theta_child) for E_child, theta_child in child_list if self.threshold_c_min <= E_child(theta_child) <= self.threshold_c_max]
        return res
    
    def rank_by_novelty(self, child_list):

        def get_third_el(child_novelty):
            return child_novelty[2]
        
        e = self.envs + self.archive_envs
        child_novelty = []
        
        for E_child, theta_child in child_list:
            dist = []
            for E, _ in e:
                dist.append(abs(E_child(0) - E(0)))
            
            novelty_score = np.mean(dist)
            child_novelty.append((E_child, theta_child, novelty_score))
        
        child_novelty_sorted = sorted(child_novelty, key = get_third_el, reverse = True)
        child_list_sorted = [(E_child, theta_child) for E_child, theta_child, _ in child_novelty_sorted]

        return child_list_sorted
    
    def remove_oldest(self, EA_list, num_removals):
        del EA_list[:num_removals]

    def mutate_envs(self, EA_list):
        parent_list = []
        
        M = len(EA_list)
        m = 1
        for m in range(M+1):
            E_m, theta_m = EA_list[m]
            if self.eligible_to_reproduce(E_m, theta_m):
                parent_list.append((E_m, theta_m))
        
        child_list = self.env_reproduce(parent_list, self.max_children)
        child_list = self.mc_satisfied(child_list)
        child_list = self.rank_by_novelty(child_list)

        theta_list = [theta_m for _, theta_m in EA_list]
        admitted = 0
        for E_child, theta_child in child_list:
            theta_child = self.evaluate_candidates(theta_list, E_child, self.alpha, self.noise_std)
            if self.mc_satisfied([E_child, theta_child]):
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
        #N(0, I)
        epsilon = np.random.randn(self.n, len(theta_m_t))
        E_i = np.array([E_m(theta_m_t + noise_std * epsilon_i) for epsilon_i in epsilon])

        res = 0
        for i in range(self.n):
            res += E_i[i]*epsilon[i]
        
        return alpha*(1/self.n*noise_std)*res
    
    def evaluate_candidates(self, theta, E, alpha, noise_std):
        C = []

        m = 1
        M = len(E)
        for m in range(M+1):
            C.append(theta[m])
            C.append(theta[m] + self.es_step(theta[m], E, alpha, noise_std))

        return np.argmax([E(theta) for theta in C])

    def main_loop(self):
        EA_list = []
        EA_list.append((self.E_init, self.theta_init))

        for t in range(self.T):
            if t > 0 and t % self.N_mutate == 0:
                EA_list = self.mutate_envs(EA_list)

            m = 1
            M = len(EA_list)
            for m in range(M+1):
                E_m, theta_m_t = EA_list[m]
                theta_m_t_1 = theta_m_t + self.es_step(theta_m_t, E_m, self.alpha, self.noise_std)
            
            m = 1
            for m in range(M+1):
                if M > 1 and t % self.N_transfer == 0:
                    E_m, theta_m_t = EA_list[m]
                    theta_b_a_m = [theta for j, (_, theta) in enumerate(EA_list) if j != m]
                    theta_top = self.evaluate_candidates(theta_b_a_m, E_m, self.alpha, self.noise_std)
                    if E_m(theta_top) > E_m(theta_m_t_1):
                        theta_m_t_1 = theta_top
                EA_list[m] = (E_m, theta_m_t_1)