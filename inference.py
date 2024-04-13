import pygame
import torch
from game_ai import SnakeGameAI, Direction, Point
from train import Agent
import os
import glob

# Function to find the latest model checkpoint file in a directory
def find_latest_model_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model.pth'))
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint

pygame.init()
game = SnakeGameAI()
checkpoint_dir = 'model'
latest_checkpoint = find_latest_model_checkpoint(checkpoint_dir)

if latest_checkpoint is None:
    print("No model checkpoint found in the directory:", checkpoint_dir)
    exit()

agent = Agent()
agent.model.load_state_dict(torch.load(latest_checkpoint))
# Set the model to evaluation mode (no gradient computation)
agent.model.eval()

running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    game_state = agent.get_state(game)
    action = agent.get_action(game_state)
    reward, game_over, score = game.play_step(action)

    if game_over:
        print("Game Over! Final Score:", score)
        running = False

    game.update_ui()
    pygame.display.flip()
    clock.tick(20)  


pygame.quit()
