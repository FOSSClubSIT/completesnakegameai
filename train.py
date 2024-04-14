import torch
import random
import numpy as np
from collections import deque
from queue import Queue
from game_ai import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plot
import multiprocessing as mp
import subprocess

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, shared_mean_score):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.shared_mean_score = shared_mean_score

    def bfs(self, start, game):
        visited = set()
        q = Queue()
        q.put((start, 0))
        visited.add(start)
        food_point = Point(game.food.x, game.food.y)  # Create a Point object for food coordinates
        while not q.empty():
            current, steps = q.get()
            if current == food_point:  # Check if current point is equal to food_point
                return steps
            for direction in [Direction.UP.value, Direction.DOWN.value, Direction.LEFT.value, Direction.RIGHT.value]:
                if direction == Direction.UP.value:
                    new_point = Point(current.x, current.y - 20)
                elif direction == Direction.DOWN.value:
                    new_point = Point(current.x, current.y + 20)
                elif direction == Direction.LEFT.value:
                    new_point = Point(current.x - 20, current.y)
                elif direction == Direction.RIGHT.value:
                    new_point = Point(current.x + 20, current.y)
                if not game.is_collision(new_point) and new_point not in visited:
                    q.put((new_point, steps + 1))
                    visited.add(new_point)
        return float('inf')

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done, steps_to_food, steps_to_food_new):
        self.memory.append((state, action, reward, next_state, done, steps_to_food, steps_to_food_new)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones, steps_to_food, steps_to_food_new = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, steps_to_food, steps_to_food_new)

        # Calculate mean score and store it
        mean_score = sum(self.shared_mean_score) / len(self.shared_mean_score)
        print("Mean score:", mean_score)
        self.trainer.train_step(states, actions, rewards, next_states, dones, steps_to_food, steps_to_food_new, mean_score)

    def train_short_memory(self, state, action, reward, next_state, done, steps_to_food, steps_to_food_new):
        self.trainer.train_step(state, action, reward, next_state, done, steps_to_food, steps_to_food_new)

    def get_action(self, state, model=None):
        if model is None:
            model=self.model
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def run_game():
    subprocess.Popen(['xterm', '-e', 'python', 'train.py'])

def train():
    shared_mean_score = mp.Manager().list()  # Shared list to store mean scores
    processes = []
    for i in range(5):
        mp.Process(target=run_game).start()

if __name__ == '__main__':
    train()
