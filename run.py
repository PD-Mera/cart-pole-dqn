import gymnasium as gym
import onnxruntime
import random
import math
import numpy as np
from itertools import count
import imageio

class VirtualGymnasiumEnvInferOnnx:
    def __init__(self, weight_path, env_name = "CartPole-v1", render_mode = None) -> None:
        self.render_mode = render_mode
        self.env = gym.make(env_name, render_mode=render_mode)
        self.get_config()

        self.providers = ["CUDAExecutionProvider"]
        # Get the number of state observations
        state, info = self.env.reset()
        self.n_observations = len(state)

        self.policy_session = onnxruntime.InferenceSession(weight_path, providers = self.providers)

        self.steps_done = 0

    def get_config(self):
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

    def select_action(self, state):
        ort_inputs = {self.policy_session.get_inputs()[0].name: state}
        ort_outs = self.policy_session.run(None, ort_inputs)[0]
        indices = np.argmax(ort_outs, axis=1)
        result = indices.reshape(1, 1)
        return result


    def run(self, save = None):
        if save is not None:
            assert self.render_mode == "rgb_array"
        # Initialize the environment and get its state
        state, info = self.env.reset()
        state = np.array(state)
        state = np.expand_dims(state, axis = 0)
        frames = []
        for t in count():
            if save is not None:
                frames.append(self.env.render())
            action = self.select_action(state)
            # print(action)
            observation, reward, terminated, truncated, _ = self.env.step(int(action))
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = np.array(observation)
                next_state = np.expand_dims(observation, axis = 0)

            # Move to the next state
            state = next_state

            if done:
                break
        
        if save is not None:
            self.saveanimation(frames, save)

        
    def saveanimation(self, frames, address="./movie.gif"):
        """ 
            This method ,given the frames of images make the gif and save it in the folder
            
            params:
                frames:method takes in the array or np.array of images
                address:(optional)given the address/location saves the gif on that location
                        otherwise save it to default address './movie.gif'
            
            return :
                none
        """
        imageio.mimsave(address, frames)



if __name__ == "__main__":
    dqn_env = VirtualGymnasiumEnvInferOnnx("weights/cart-pole.onnx", env_name = "CartPole-v1", render_mode = "rgb_array")
    # If you want to save gif, render_mode set to "rgb_array"
    dqn_env.run(save = "visualize.gif")