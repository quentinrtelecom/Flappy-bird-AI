import flappy_bird_gymnasium
import gymnasium

def make_env():
    return gymnasium.make("FlappyBird-v0", use_lidar=False)
