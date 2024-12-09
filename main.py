import pygame
import numpy as np
from model import CarAgent
from poet import POET
from env import CarEnvironment, Car

#Main function to test the environment
def main():
   
    env = CarEnvironment()
    car = Car(env.world, position=(10, 4))
    
    # Initialize agent
    agent = CarAgent(
        input_dim=env.observation_space.shape[0],
        hidden_dim=128,
        output_dim=2,
        lr=0.001,
        weight_decay=1e-4
    )
    
    #Initialize initial policy parameters theta
    theta_init = np.zeros(env.observation_space.shape[0])
    E_init = env
    
    # Initialize POET
    poet = POET(
        car=car,
        agent=agent,
        E_init=E_init,
        theta_init=theta_init,
        alpha=0.1,
        noise_std=0.01,
        T=100,
        N_mutate=10,
        N_transfer=5,
        env_input_dim=env.observation_space.shape[0],
        hidden_dim=128,
        action_dim=2
    )

    
    poet.main_loop()

    # Keep the window open until closed by the user
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Optionally render the environment or provide a delay
        pygame.display.flip()
        poet.E_init.render()  # Render the initial environment

    # Close the environment when done
    env.close()
    pygame.quit()
    

if __name__ == "__main__":
    main()
