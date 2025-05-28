import pygame
import random
import sys
from config import *
import numpy as np
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Falling Square Game")

# Font
font = pygame.font.SysFont(None, front_sys)
small_font = pygame.font.SysFont(None, small_font_size)

def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect(center=(x, y))
    surface.blit(text_obj, text_rect)

def reset_game():
    return {
        "square_x": random.randint(0, WIDTH - 20),
        "square_y": 0,
        "vx": random.choice([-3, -2, -1, 1, 2, 3]),
        "vy": 5,
        "paddle_x": WIDTH // 2 - 50,
        "game_over": False,
        "score": 0
    }

def run(state, keys, real_play=False, expect_maximum_score=100):
    screen.fill(WHITE)
    done = False

    if keys[pygame.K_LEFT] and state["paddle_x"] > 0:
        state["paddle_x"] -= paddle_speed
    if keys[pygame.K_RIGHT] and state["paddle_x"] < WIDTH - paddle_width:
        state["paddle_x"] += paddle_speed

    if not state["game_over"]:
        # Move square
        state["square_x"] += state["vx"]
        state["square_y"] += state["vy"]

        if state["square_y"] <= paddle_y:  
            ball_center = state["square_x"] + paddle_height
            paddle_center = state["paddle_x"] + paddle_width / 2
            horizontal_error = abs(ball_center - paddle_center)
            max_error = WIDTH/3
            #đưa ra giá trị -1 đến 0, càng gần paddle thì reward càng cao
            reward = -horizontal_error/max_error 
            if round(horizontal_error,2) == 0:
                reward = 1      
        # Wall collision
        if state["square_x"] <= 0 or state["square_x"] >= WIDTH - 20:
            state["vx"] = -state["vx"]

        # Paddle collision
        if (
            paddle_y <= state["square_y"] + 2*square_height <= paddle_y + paddle_height and
            state["paddle_x"] <= state["square_x"] + square_height <= state["paddle_x"] + paddle_width
        ):
            impact = (state["square_x"] + 10) - (state["paddle_x"] + paddle_width / 2)
            state["vx"] += impact / (paddle_width / 4)
            state["vy"] = -abs(state["vy"])  # Reflect upward
            state["square_y"] = paddle_y - 21
            state["score"] += 1
            if state["score"] % increase_speed_p == 0 and abs(state["vy"]) < 20:
                if state["vy"] < 0:
                    state["vy"] -= 1
                else:
                    state["vy"] += 1
            reward = 1

        # Ceiling
        if state["square_y"] <= 0:
            state["vy"] = abs(state["vy"])

        # Missed paddle
        if state["square_y"] > paddle_y:
            state["game_over"] = True
            reward = -1

    # Draw square and paddle
    pygame.draw.rect(screen, RED, (state["square_x"], state["square_y"], 20, 20))
    pygame.draw.rect(screen, BLACK, (state["paddle_x"], paddle_y, paddle_width, paddle_height))

    # Draw score
    score_text = small_font.render(f"Score: {state['score']}", True, BLACK)
    screen.blit(score_text, (10, 10))

    # Game Over screen
    if state["game_over"] and real_play:
        draw_text("Game Over", font, BLUE, screen, WIDTH // 2, HEIGHT // 2 - 50)
        draw_text("Press R to Restart", small_font, BLACK, screen, WIDTH // 2, HEIGHT // 2 + 10)
        draw_text("Press Q to Quit", small_font, BLACK, screen, WIDTH // 2, HEIGHT // 2 + 40)
        if keys[pygame.K_r]:
            return reset_game(), 0, 0, {}
        elif keys[pygame.K_q]:
            pygame.quit()
            sys.exit()
    elif state["game_over"]:
        done = True
    elif state['score'] == expect_maximum_score:
        done = True
    return state, reward, done, {}


def game_loop():
    state = reset_game()
    clock = pygame.time.Clock()
    while True:
        keys = pygame.key.get_pressed()
        # pygame.K_LEFT = True
        # pygame.K_RIGHT = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        state_ = run(state, keys, real_play=True)
        print(state_[1])
        state = state_[0]
        pygame.display.update()
        clock.tick(60)
# game_loop()
