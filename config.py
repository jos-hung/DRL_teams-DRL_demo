WIDTH = 400
HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
front_sys= 48
small_font_size = 32

paddle_y = HEIGHT - 50
paddle_width = 100
paddle_height = 10
square_height = 10
paddle_speed = 7
increase_speed_p = 10
#check GPU
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for training.")
else:
    device = torch.device("cpu")
    print("Using CPU for training.")
    
dqn_config = {
    # "state_size": 7,
    # "action_size": 2,
    "learning_rate": 0.0001,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "max_memory_size": 1000000,
    "start_training": 1024*10,
    "batch_size": 512
}