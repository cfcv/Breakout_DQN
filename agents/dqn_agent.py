import keras
import random
import numpy as np
import agents.utils as utils

class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, transition):
    if len(self.memory) < self.capacity:
        self.memory.append(None)
    self.memory[self.position] = transition
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)


class Agent:
  def __init__(self, input_shape, n_actions, batch_size, memory_size, memory_initial_frames, gamma, env):
    #Parameters		
    self.input_shape = input_shape
    self.n_actions = n_actions
    self.batch_size = batch_size
    self.memory_size = memory_size
    self.memory_initial_frames = memory_initial_frames
    self.gamma = gamma
    self.env = env

    #Creating the replay memory
    self.replay_memory = ReplayMemory(self.memory_size)

      #Model that will be trained every step
    self.model = self.create_model()

      #model that will make the predictions
    self.target_model = self.create_model()
    self.target_model.set_weights(self.model.get_weights())



  def create_model(self):
    input_frames = keras.layers.Input(self.input_shape, name='frames')

    #Normalizing input from 0-255 to 0-1.
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(input_frames)

    #Convolutions
    conv_1 = keras.layers.Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu', kernel_initializer='he_normal')(normalized)

    conv_2 = keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(conv_1)

    conv_3 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conv_2)

    #Flattening
    conv_flattened = keras.layers.Flatten()(conv_2)

	  #Dense layers
    hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)

    output = keras.layers.Dense(self.n_actions)(hidden)


    model = keras.models.Model(inputs=input_frames, outputs=output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    return model
  
  def save(self):
    self.model.save('model.h5')
    self.target_model.save('target.h5')

  def predict(self, state):
    return self.model.predict(state)

  def update_target(self):
    self.target_model.set_weights(self.model.get_weights())

  def update_replay_memory(self, transition):
    #Transition = (initial_state, action, reward, next_state, is_terminal_state)
    self.replay_memory.push(transition)

  def initialize_replay_memory(self):
    count = 0
    while(count < self.memory_initial_frames):
        state = self.env.reset()
        state1, _, is_done1, _  = self.env.step(0)
        state2, _, is_done2, _  = self.env.step(0)
        state3, _, is_done3, _  = self.env.step(0)

        initial_state = utils.generate_input(state, state1, state2, state3)
        is_terminal_state = False

        while(not is_terminal_state):
            action = self.env.action_space.sample()

            state, reward, is_done, _  = self.env.step(action)
            state1, reward1, is_done1, _  = self.env.step(action)
            state2, reward2, is_done2, _  = self.env.step(action)
            state3, reward3, is_done3, _  = self.env.step(action)

            next_state = utils.generate_input(state, state1, state2, state3)
            total_reward = utils.transform_reward(reward + reward1 + reward2 + reward3)
            is_terminal_state = is_done or is_done1 or is_done2 or is_done3

            self.update_replay_memory([initial_state, action, total_reward, next_state, is_terminal_state])
            initial_state = next_state
            
            count += 1
            print('\r', end='')
            print(count, ' of ', self.memory_initial_frames, end='')
            if(count == self.memory_initial_frames):
                break
  
  def train(self):
    #Extract a batch from the replay memory 
    batch = self.replay_memory.sample(self.batch_size)
    
    #Extract the initial states and next states from the batch 
    initial_states = np.array([transition[0] for transition in batch])
    next_states = np.array([transition[3] for transition in batch])

    #Calculating the action_value(Q) values for current state and next states
    current_qs = self.model.predict(initial_states)
    futur_qs = self.target_model.predict(next_states)

    #Generating the train inputs and expected outputs
    batch_x = []
    batch_y = []

    for index, (initial_state, action, total_reward, next_state, is_terminal_state) in enumerate(batch):
      if(is_terminal_state):
        new_q = total_reward
      else:
        max_futur_q = np.max(futur_qs[index])
        new_q = total_reward + self.gamma * max_futur_q

      current_q = current_qs[index]
      current_q[action] = new_q

      batch_x.append(initial_state)
      batch_y.append(current_q)

    self.model.fit(np.array(batch_x), np.array(batch_y), batch_size=self.batch_size, epochs=1, shuffle=False, verbose=0)

