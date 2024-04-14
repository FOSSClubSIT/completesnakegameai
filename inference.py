import pygame
import torch
from train import Agent  
from game_ai import SnakeGameAI, Direction


def load_agent(model_path):
    agent = Agent()
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval() 
    return agent

def main():
    pygame.init()
    game = SnakeGameAI()  

    model_path = 'model/model.pth' 
    agent = load_agent(model_path)
    agent.n_games=80

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game_state = agent.get_state(game)
        action = agent.get_action(game_state, agent.model)
        reward, game_over, score = game.play_step(action)

        if game_over:
            print("Game Over! Final Score:", score)
            running = False
        
        pygame.display.flip()
        clock.tick(20)  

    pygame.quit()

if __name__ == "__main__":
    main()
