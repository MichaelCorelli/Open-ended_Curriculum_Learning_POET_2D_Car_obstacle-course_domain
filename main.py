import pygame
import random
import numpy as np
from Box2D.b2 import world, polygonShape, circleShape, revoluteJointDef
from Box2D.b2 import RayCastCallback as b2RayCastCallback
import gymnasium as gym
from gymnasium import spaces
from model import CarAgent
from poet import POET


#Constants
PPM = 20.0  #pixels x meter
FPS = 60
STEP_T = 1.0 / FPS
SCREEN_W, SCREEN_H = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class CarEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CarEnvironment, self).__init__()

        #Initialize Pygame and Box2D world
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("POET 2D Car Simulation")
        self.clock = pygame.time.Clock()
        self.world = world(gravity=(0, -10), doSleep=True)
        self.bodies = []
        self.obstacles = []
        self.car = None
        self.camera = (0, 0)

        #Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(4,),
                                            dtype=np.float32)
        
        
        lidar_sensors = 10  # Number of LIDAR sensors
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 + lidar_sensors,),  #Car position (2), velocity (2), LIDAR data
            dtype=np.float32
        )

        self.done = False
        self.reset()

    def reset(self):
        #Reset the simulation environment
        self.world = world(gravity=(0, -10), doSleep=True)
        self.bodies = []
        self.obstacles = []
        self.car = Car(self.world, position=(10, 4))
        self.add_body(self.car.body, BLACK)
        self.add_body(self.car.wheel_front, BLACK)
        self.add_body(self.car.wheel_rear, BLACK)
        ground = self.world.CreateStaticBody(position=(50, 0),
                                             shapes=polygonShape(box=(50, 1)))
        self.add_body(ground, BLACK)
        self.camera = (0, 0)
        self.done = False

        #Initialize obstacles
        self.add_obstacle(base_position=(20, 1), size=(2, 2), color=RED, obstacle_type='ramp')
        self.add_obstacle(base_position=(40, 1), size=(3, 2), color=RED, obstacle_type='hole')
        self.add_obstacle(base_position=(60, 1), size=(4, 3), color=RED, obstacle_type='bump')

        return self._get_state()

    '''
    def step(self, action):
        #Apply action to the car
        self.car.apply_action(action)

        #Step the physics simulation
        self.world.Step(STEP_T, 6, 2)
        self.world.ClearForces()

        self._update_camera()

        #Check for termination conditions
        self.done = self._check_done()

        reward = self._calculate_reward()

        state = self._get_state()

        if self._should_add_obstacle():
            self._add_new_obstacle()

        return state, reward, self.done, {}
    '''

    def render(self, mode='human'):
        self.screen.fill(WHITE)
        for body, color in self.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, polygonShape):
                    vertices = [body.transform * v * PPM for v in shape.vertices]
                    vertices = [(v[0] - self.camera[0],
                                SCREEN_H - v[1] - self.camera[1]) for v in vertices]
                    pygame.draw.polygon(self.screen, color, vertices)
                elif isinstance(shape, circleShape):
                    position = body.transform * shape.pos * PPM
                    position = (position[0] - self.camera[0],
                                SCREEN_H - position[1] - self.camera[1])
                    pygame.draw.circle(self.screen, color,
                                    [int(x) for x in position],
                                    int(shape.radius * PPM))

        # Render holes as black rectangles
        for _, obstacle_type, base_position, width, _ in self.obstacles:
            if obstacle_type == "hole":
                x, y = base_position
                pygame.draw.rect(self.screen, BLACK,
                                pygame.Rect(x * PPM - self.camera[0],
                                            SCREEN_H - (y + 1) * PPM - self.camera[1],
                                            width * PPM, 20))  # Adjust height visually for ground

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()

    def add_body(self, body, color):
        self.bodies.append((body, color))

    def add_obstacle(self, base_position, size, color, obstacle_type):
        x, y = base_position
        width, height = size

        if obstacle_type == "ramp":
            # Create a ramp
            vertices = [(0, 0), (width, 0), (width, height)]
            ramp = self.world.CreateStaticBody(position=(x, y),
                                            shapes=polygonShape(vertices=vertices))
            self.add_body(ramp, color)
            self.obstacles.append((ramp, obstacle_type, base_position, width, height))

        elif obstacle_type == "hole":
            # Create a hole (gap with no physical body)
            self.obstacles.append((None, obstacle_type, base_position, width, height))

        elif obstacle_type == "bump":
            # Create a bump
            vertices = [(0, 0), (width, 0), (width / 2, height), (0, 0)]
            bump = self.world.CreateStaticBody(position=(x, y),
                                            shapes=polygonShape(vertices=vertices))
            self.add_body(bump, color)
            self.obstacles.append((bump, obstacle_type, base_position, width, height))

        else:
            raise ValueError(f"Unsupported obstacle type: {obstacle_type}")


    def evaluate_policy(self, theta):
        total_reward = 0
        state = self.reset()
        max_steps = 1000
        for _ in range(max_steps):
            action = self.policy(theta, state)
            state, reward, self.done, _ = self.step(action)
            total_reward += reward
            if self.done:
                break
        return total_reward

    def policy(self, theta, state):
        action = np.dot(theta, state)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def _get_state(self):
        car_pos = self.car.body.position
        car_vel = self.car.body.linearVelocity
        lidar_data = self.car.get_lidar_data()
        return np.concatenate(([car_pos[0], car_pos[1], car_vel[0], car_vel[1]], lidar_data))

    def _update_camera(self):
        car_x = self.car.body.position[0] * PPM
        car_y = self.car.body.position[1] * PPM
        x = car_x - SCREEN_W / 2
        y = car_y - SCREEN_H / 2
        x = max(0, x)
        y = max(0, y)
        self.camera = (x, y)

    def _check_done(self):
        car_x, car_y = self.car.body.position
        # Check if the car has fallen into a hole
        for _, obstacle_type, base_position, width, _ in self.obstacles:
            if obstacle_type == "hole":
                hole_x, hole_width = base_position[0], width
                if hole_x <= car_x <= hole_x + hole_width and car_y < 1:
                    return True  # Car fell into the hole
        return car_y < 0  # Existing condition for falling below the ground

    def _calculate_reward(self):
        reward = self.car.body.position[0]
        if self.done:
            reward -= 100
        return reward

    def _should_add_obstacle(self):
        if not self.obstacles:
            return False
        last_ramp_x = self.obstacles[-1][1][0] + self.obstacles[-1][2]
        car_x = self.car.body.position[0]
        return car_x > last_ramp_x - 20

    def _add_new_obstacle(self):
        last_ramp_x = self.obstacles[-1][2][0] + self.obstacles[-1][3]  # Last ramp position + width
        new_obstacle_x = last_ramp_x + random.uniform(5, 10)
        new_obstacle_y = 1

        obstacle_type = random.choice(["ramp", "hole", "bump"])  # Random selection
        if obstacle_type == "ramp":
            size = (random.uniform(6, 10), random.uniform(2, 5))
        elif obstacle_type == "hole":
            size = (random.uniform(5, 8), 0)
        elif obstacle_type == "bump":
            size = (random.uniform(2, 4), random.uniform(0.5, 1.5))

        self.add_obstacle(base_position=(new_obstacle_x, new_obstacle_y), 
                        size=size, 
                        color=RED, 
                        obstacle_type=obstacle_type)
        
    
    def evaluate_agent(self, agent, max_steps=1000):
        #Evaluates the agent's performance in the environment.
        #  - Resets the environment to start a new episode.
        #  - Runs a simulation loop where the agent selects actions based on its policy.
        #  - Applies actions, collects rewards, and checks for episode termination.
        #  - Returns the total reward as a performance metric.
        state = self.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(state)
            state, reward, done, _ = self.step(action)
            total_reward += reward
            if done:
                break
        return total_reward    


class Car:
    def __init__(self, world, position=(10, 4)):
        self.world = world
        self._create_car(position)

    def _create_car(self, position):
        car_width, car_height = 3.5, 1.4
        self.body = self.world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(box=(car_width / 2, car_height / 2),
                                       density=1.2, friction=0.5)
        self.wheel_front = self._create_wheel((position[0] + 1.3, position[1] - 1))
        self.wheel_rear = self._create_wheel((position[0] - 1.3, position[1] - 1))
        self.joint_front = self._create_wheel_joint(self.body, self.wheel_front, (1.4, -1))
        self.joint_rear = self._create_wheel_joint(self.body, self.wheel_rear, (-1.4, -1))
        self._create_wheel_joint(self.body, self.wheel_front, (1.4, -1))
        self._create_wheel_joint(self.body, self.wheel_rear, (-1.4, -1))

    def _create_wheel(self, position):
        radius = 0.5
        wheel = self.world.CreateDynamicBody(position=position)
        wheel.CreateCircleFixture(radius=radius, density=1.0, friction=0.9)
        return wheel

    def _create_wheel_joint(self, bodyA, bodyB, anchorA):
        joint_def = revoluteJointDef(bodyA=bodyA, bodyB=bodyB,
                                    localAnchorA=anchorA,
                                    enableMotor=True, maxMotorTorque=1000)
        joint = self.world.CreateJoint(joint_def)
        return joint
    
    #Scan LiDAR to get data from the environment
    def get_lidar_data(self, num_sensors=10, max_distance=20):
        angles = np.linspace(-np.pi / 4, np.pi / 4, num_sensors)
        lidar_data = []
        for angle in angles:
            ray_direction = (np.cos(angle), np.sin(angle))
            endpoint = (
                self.body.position[0] + max_distance * ray_direction[0],
                self.body.position[1] + max_distance * ray_direction[1],
            )
            callback = RayCastCallback()
            self.world.RayCast(callback, self.body.position, endpoint)
            if callback.hit:
                distance = callback.fraction * max_distance
            else:
                distance = max_distance
            lidar_data.append(distance)
        return np.array(lidar_data)

class RayCastCallback(b2RayCastCallback):
    def __init__(self):
        super().__init__()
        self.hit = False
        self.fraction = 1.0
    
    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True
        self.fraction = fraction
        return fraction
    

    def apply_action(self, action):
        motor_speed = action[0] * 50
        self.wheel_front.joint.motorSpeed = motor_speed
        self.wheel_rear.joint.motorSpeed = motor_speed


#Main function to test the environment
def main():
    env = CarEnvironment()
    theta = np.random.randn(env.observation_space.shape[0])  # Random policy parameters
    agent = CarAgent(input_dim=env.env_input_dim, hidden_dim=env.hidden_dim, output_dim=env.action_dim, lr=0.001, weight_decay=1e-4)

    env_input_dim = 4 + 10 #4 for position/velocity, 10 for LIDAR data
    hidden_dim = 128    #NN hidden layer size
    action_dim = 2      #Number of output actions

    
    poet = POET(
        E_init=env,
        theta_init=np.zeros(env_input_dim),
        alpha=0.01,
        noise_std=0.1,
        T=10,
        N_mutate=5,
        N_transfer=5,
        env_input_dim=env_input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )

    #Create an agent and test its performance
    agent = poet.create_new_agent()
    total_reward = env.evaluate_agent(agent)
    episodes = poet.main_loop()
    print(f"Total Reward: {total_reward}")
    
    state = env.reset()
    done = False
    while not done:
        action = env.policy(theta, state)
        state, reward, done, _ = env.step(action)
        env.render()

    env.close()


if __name__ == "__main__":
    #To-Do: verify this values
    main()
