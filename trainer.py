from game import reset_game, run
import pygame
from dqn_agent import DQNAgent
import sys
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def fake_keys_binary(direction):
    keys = defaultdict(int)
    if direction == 0:
        keys[pygame.K_LEFT] = 1
    elif direction == 1:
        keys[pygame.K_RIGHT] = 1
    return keys

epochs = 5000
state_size = 7  # square_x, square_y, vx, vy, paddle_x, game_over
action_size = 2  # left, right
window_size = 1000  

dqn_agent = DQNAgent(state_size, action_size, batch_size=512, max_memory_size=1000000, start_training=512*10)

epoch_scores = []
moving_avg_scores = []

os.makedirs("plots", exist_ok=True)

for epoch in range(epochs):
    state = reset_game()
    done = False
    total_reward = 0
    clock = pygame.time.Clock()
    while not done:
        save_state = state.copy()
        # save_state.pop('score')
        
        state_tensor = torch.tensor(list(save_state.values()), dtype=torch.float32).unsqueeze(0)
        action = dqn_agent.get_actions(state_tensor)
        keys = fake_keys_binary(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        next_state, reward, done, _ = run(state, keys, expect_maximum_score=500)
        save_next_state = next_state.copy()
        # save_next_state.pop('score')

        dqn_agent.update_memory(
            list(save_state.values()),
            action,
            reward,
            list(save_next_state.values()),
            done
        )
        # print(f"Epoch {epoch + 1}/{epochs}, Action: {action}, Reward: {reward}")
        total_reward += reward

        state = next_state
        pygame.display.update()
        epoch_scores.append(reward)
        if len(epoch_scores) >= window_size:
            moving_avg = np.mean(epoch_scores[-window_size:])
            moving_avg_scores.append(moving_avg)

        if epoch > 500:
            clock.tick(60)
    dqn_agent.update_model()

    if epoch%200 == 0 and epoch > 0:
        dqn_agent.update_target_network()

    print(f"Epoch {epoch + 1}/{epochs}, Total Reward: {total_reward}, Epsilon: {dqn_agent.epsilon:.2f}")

    if (epoch + 1) % 100 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(epoch_scores) + 1), epoch_scores, label="Current Score (per Epoch)", alpha=0.6)
        plt.plot(range(1, len(moving_avg_scores) + 1), moving_avg_scores, label=f"Moving Average (Window={window_size})", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"Score Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/score_plot.png")
        plt.close()
