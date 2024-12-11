#main.py
import pygame
from model import DDQN, Buffer
from poet import POET, state_dict_to_vector
from env import CarEnvironment, Car


def main():
   
    env = CarEnvironment()
    car = Car(env.world, position=(10, 4))
    b = Buffer()
    
    
    agent = DDQN(
        env=env, 
        b=b, 
        lr=0.001, 
        epsilon_initial=0.5, 
        batch_size=64, 
        threshold_r=200
    )
    
    theta_init_sd = agent.network.network.state_dict()
    theta_init_vec = state_dict_to_vector(theta_init_sd)
    print(f"Initialized theta vector length: {len(theta_init_vec)}")
    E_init = env
    
    # Initialize POET
    poet = POET(
        car=car,
        ddqn_agent=agent,
        E_init=E_init,
        theta_init=theta_init_vec,
        alpha=0.1,
        noise_std=0.01,
        T=100,
        N_mutate=10,
        N_transfer=5,
        env_input_dim=env.observation_space.shape[0],
        hidden_dim=128,
        action_dim=2
    )
    
    #Agent training
    poet.main_loop()
    
    #Agent evaluation
    metrics = agent.evaluate(env, ep_n=10, render=False, verbose=True)
    print(metrics)
    
    display_envs = [poet.E_init] + [env for env, _ in poet.envs] + [env for env, _ in poet.archive_envs]
    if not display_envs:
        display_envs = [poet.E_init]
    
    current_idx = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            #Press RIGHT ARROW (→) to move to the next environment.  
            #Press LEFT ARROW (←) to go back to the previous environment.  
            #Press the window close button (X) to exit.  
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_idx = (current_idx + 1) % len(display_envs)
                elif event.key == pygame.K_LEFT:
                    current_idx = (current_idx - 1) % len(display_envs)

        pygame.display.flip()
        display_envs[current_idx].render() 

    #Close the environment when done
    env.close()
    pygame.quit()
    

if __name__ == "__main__":
    main()
