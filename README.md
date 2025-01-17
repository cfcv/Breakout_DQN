# Breakout_DQN

## Introduction
This project aims to implement the [Deepmind's paper](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning) in order to build an agent to play the Breakout Atari game using an OpenAI Gym environment. Knowing that there is not as much computational power and memory available for me now as Deepmind, some parameters as the size of the replay memory and the number of iterations are clearly smaller, degrading the performance. However, trying to compensate this fact, a more complex network(with more filters in the convolution layers) and fixed Q-targets were used.

Check out the resulting video by clicking in the image below:

[![Watch the video](http://i3.ytimg.com/vi/vH4TlVDMyms/hqdefault.jpg)](https://youtu.be/vH4TlVDMyms)

## Pre-processing
Before stocking the state transitions in the replay memory, they are pre-processed. At every time an action is taken, the environment outputs a color image of size 210x160x3, the reward and a boolean indicating if it is a terminal state or not. Each action is repeated 4 times so we can obtain four frames of the game, each frame is converted in grayscale, downsampled and cropped to remove useless information resulting in an image of size 82x72 and therefore, a state of size 82x72x4. 

![alt text](https://github.com/cfcv/Breakout_DQN/blob/master/images/processing.png)

## DQN architecture
For this task, a convolutional neural network was used. Aiming to optimize the amount of memory consumed, I choose to normalize the images on the fly, so then they can be stocked as uint8, consequently, the first layer of the neural network is a normalization layer. It has 3 convolutional layers and 2 dense layers. The convolutional layers have 32, 32 and 64 filters with a kernel size of 8x8, 4x4 and 3x3 with strides of 4x4, 2x2 and 1x1 respectively, each one with a ReLu activation and a He Normal initialization. After the convolutional layers, the features are flattened and a dense layer is added with 512 neurons and a ReLu activation. Finally, the last dense layer, representing the model output, has 4 neurons because we have 4 possible actions.

## Parameters
* Replay memory size: 100 000
* Memory initialization: 50 000
* Batch size: 32
* Update target network frequency: every 5 000 steps
* Number of Episodes: 10 600
* Number of Frames: 1 000 000
* initial Epsilon: 1.0
* Final Epsilon: 0.1(at the frame 600 000)
* Discount factor(gamma): 0.99

At the moment, the same model is training for 50 000 episodes for trying to get a better performance. 


## Conclusion
For this task, the reinforcement learning algorithm needs a really long time exploring the environment and training to give good results.

## Future improviments
* Prioritized Experience Replay 
* Dueling DQN 
* Double DQN
* Try other methods like Actor-Critic

