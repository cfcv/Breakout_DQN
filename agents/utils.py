import numpy as np

def toGrayScale(img):
    return np.mean(img, axis=2).astype(np.uint8) #storing as uint8 to consume less memory

#210x160 -> 105x80
def downSample(img):
    return img[::2, ::2]
  
#105x80 -> 82x72
def preProcess(img):
    gray = toGrayScale(downSample(img))
    return gray[16:98,4:76]

def transform_reward(reward):
    return np.sign(reward) #return -1 if reward is negativ or 1 if it is positive
    #return reward
    
def generate_input(state, state1, state2, state3):
    input_state = np.empty([82,72,4])
    input_state[:,:,0] = preProcess(state)
    input_state[:,:,1] = preProcess(state1)
    input_state[:,:,2] = preProcess(state2)
    input_state[:,:,3] = preProcess(state3)
    
    return input_state
