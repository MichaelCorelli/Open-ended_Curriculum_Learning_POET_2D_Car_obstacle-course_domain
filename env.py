import pygame
import random
import numpy as np
from Box2D.b2 import world, polygonShape, circleShape, revoluteJointDef
from Box2D.b2 import rayCastCallback as b2RayCastCallback
import gymnasium as gym
from gymnasium import spaces
from utils import vector_to_state_dict


#Constants
PPM = 20.0  #pixels x meter
FPS = 60
STEP_T = 1.0 / FPS
SCREEN_W, SCREEN_H = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


class RayCastCallback(b2RayCastCallback):
    def __init__(self):
        super().__init__()
        self.hit = False
        self.fraction = 1.0
    
    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True
        self.fraction = fraction
        return fraction


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
        
        # Load textures
        # Background
        self.background_texture = pygame.image.load("textures/background.png").convert()
        self.background_texture = pygame.transform.scale(self.background_texture, (SCREEN_W, SCREEN_H))

        # Car body texture
        self.car_body_texture = pygame.image.load("textures/car_body.png").convert_alpha()
        self.car_body_texture = pygame.transform.scale(
            self.car_body_texture,
            (int(3.5 * PPM), int(1.4 * PPM))
        )

        # Ground/terrain texture
        self.ground_texture = pygame.image.load("textures/terrain.png").convert_alpha()
        self.ground_texture = pygame.transform.scale(self.ground_texture, (int(100 * PPM), int(2 * PPM)))

        # Wheel texture
        self.wheel_texture = pygame.image.load("textures/wheel.png").convert_alpha()
        self.wheel_texture = pygame.transform.scale(self.wheel_texture, (int(0.7 * PPM), int(0.7 * PPM)))
        
        
        lidar_sensors = 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 + lidar_sensors,),  
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
        self.modify_env({"base_position":(20, 1), "size":(2, 2), "color": RED, "obstacle_type":'ramp'})
        self.modify_env({"base_position":(40, 1), "size":(3, 2), "color":RED, "obstacle_type":'hole'})
        self.modify_env({"base_position":(60, 1), "size":(4, 3), "color":RED, "obstacle_type":'bump'})

        return self._get_state(), {}


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


    def render(self, mode='human'):
        #Draw background
        self.screen.blit(self.background_texture, (0,0))
        #Draw ground
        ground_x = 50 * PPM - self.camera[0] - self.ground_texture.get_width() / 2
        ground_y = SCREEN_H - (1 * PPM) - self.camera[1]
        self.screen.blit(self.ground_texture, (ground_x, ground_y))
        
        for body, color in self.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, polygonShape):
                    vertices = [body.transform * v * PPM for v in shape.vertices]
                    vertices = [(v[0] - self.camera[0],
                                SCREEN_H - v[1] - self.camera[1]) for v in vertices]
                    pygame.draw.polygon(self.screen, color, vertices)
                elif isinstance(shape, circleShape):
                    # Circle means a wheel
                    wheel_pos = body.position
                    wheel_angle = body.angle
                    rotated_wheel = pygame.transform.rotate(self.wheel_texture, -wheel_angle * 180 / np.pi)
                    wheel_x = wheel_pos[0]*PPM - self.camera[0] - rotated_wheel.get_width()/2
                    wheel_y = SCREEN_H - (wheel_pos[1]*PPM) - self.camera[1] - rotated_wheel.get_height()/2
                    self.screen.blit(rotated_wheel, (wheel_x, wheel_y))

        # Render holes as black rectangles
        for obs in self.obstacles:
            if obs['type'] == "hole":
                x, y = obs['params']['base_position']
                width = obs['params']['size'][0]
                pygame.draw.rect(self.screen, BLACK,
                                pygame.Rect(x * PPM - self.camera[0],
                                            SCREEN_H - (y + 1) * PPM - self.camera[1],
                                            width * PPM, 20))  # Adjust height visually for ground
        
        # Draw the car body using the custom texture
        car_pos = self.car.body.position
        car_angle = self.car.body.angle
        rotated_car_body = pygame.transform.rotate(self.car_body_texture, -car_angle * 180 / np.pi)
        car_x = car_pos[0]*PPM - self.camera[0] - rotated_car_body.get_width()/2
        car_y = SCREEN_H - (car_pos[1]*PPM) - self.camera[1] - rotated_car_body.get_height()/2
        self.screen.blit(rotated_car_body, (car_x, car_y))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()

    def add_body(self, body, color):
        self.bodies.append((body, color))

    def modify_env(self, params):
        x, y = params["base_position"]
        width, height = params["size"]
        obstacle_type = params["obstacle_type"]
        color = params["color"]

        if obstacle_type == "ramp":
            # Create a ramp
            vertices = [(0, 0), (width, 0), (width, height)]
            ramp = self.world.CreateStaticBody(
                position=(x, y),
                shapes=polygonShape(vertices=vertices)
            )
            self.add_body(ramp, color)
            self.obstacles.append({
                'body': ramp,
                'type': obstacle_type,
                'params': params
            })

        elif obstacle_type == "hole":
            # Create a hole (gap with no physical body)
            self.obstacles.append({
                'body': None,
                'type': obstacle_type,
                'params': params
            })

        elif obstacle_type == "bump":
            # Create a bump
            vertices = [(0, 0), (width, 0), (width / 2, height)]
            bump = self.world.CreateStaticBody(
                position=(x, y),
                shapes=polygonShape(vertices=vertices)
            )
            self.add_body(bump, color)
            self.obstacles.append({
                'body': bump,
                'type': obstacle_type,
                'params': params
            })

        else:
            raise ValueError("Unsupported obstacle type")
        
    def clone(self):
        # Create a new instance of CarEnvironment
        new_env = CarEnvironment()

        # Copy over the obstacle parameters and recreate the obstacles in new_env
        for obstacle in self.obstacles:
            params = obstacle['params']
            new_env.modify_env(params)

        # Optionally, copy other necessary state variables
        new_env.camera = self.camera

        # Return the cloned environment
        return new_env


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
        #Get LiDAR Data
        lidar_data = self.car.get_lidar_data(num_sensors=5, max_distance=20, angle_range=(-np.pi/6, np.pi/6))
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
        for obs in self.obstacles:
            if obs['type'] == "hole":
                base_pos = obs['params']['base_position']
                size = obs['params']['size']
                hole_x, hole_width = base_pos[0], size[0]

                if hole_x <= car_x <= hole_x + hole_width and car_y < 1:
                    return True  # Car fell into the hole

        return car_y < 0  # Existing condition for falling below the ground

    def _calculate_reward(self):
        reward = self.car.body.position[0]
        if self.done:
            reward -= 100
        return reward
    
    def mutate_environment(self):
        new_obstacle_x = random.uniform(10, 50)  
        new_obstacle_y = 1                      
        obstacle_type = random.choice(["ramp", "hole", "bump"])  
        width = random.uniform(2, 8)            
        height = random.uniform(1, 5)  
        
        params = {
            "base_position": (new_obstacle_x, new_obstacle_y),
            "size": (width, height),
            "color": RED,
            "obstacle_type": obstacle_type
        }
        
        self.modify_env(params)

    def _should_add_obstacle(self):
        if not self.obstacles:
            return False
        last_obs = self.obstacles[-1]
        base_pos = last_obs['params']['base_position']
        size = last_obs['params']['size']
        #Compute the end of the last obstacle
        last_pos_x = base_pos[0] + size[0]
        car_x = self.car.body.position[0]
        return car_x > last_pos_x - 20

    def _add_new_obstacle(self):
        last_obs = self.obstacles[-1]
        base_pos = last_obs['params']['base_position']
        size = last_obs['params']['size']
        last_pos_x = base_pos[0] + size[0]
        new_obstacle_x = last_pos_x + random.uniform(5, 10)
        new_obstacle_y = 1

        obstacle_type = random.choice(["ramp", "hole", "bump"])  # Random selection
        if obstacle_type == "ramp":
            size = (random.uniform(6, 10), random.uniform(2, 5))
        elif obstacle_type == "hole":
            size = (random.uniform(5, 8), 0)
        elif obstacle_type == "bump":
            size = (random.uniform(2, 4), random.uniform(0.5, 1.5))

        self.modify_env({
            "base_position":(new_obstacle_x, new_obstacle_y),
            "size": size,
            "color": RED,
            "obstacle_type": obstacle_type
        })
        
    
    def evaluate_agent(self, ddqn_agent, theta):
        if theta is not None:
            ddqn_agent.network.network.load_state_dict(theta)
        return ddqn_agent.evaluate(self)



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
    def get_lidar_data(self, num_sensors, max_distance, angle_range):
        start_angle, end_angle = angle_range
        angles = np.linspace(start_angle, end_angle, num_sensors)
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
    
    def apply_action(self, action):
        motor_speed = float(action[0] * 70)
        self.joint_front.motorSpeed = motor_speed
        self.joint_rear.motorSpeed = motor_speed

