# **Behavioral Cloning for self-driving cars** 

In this project, we will see how convolutional neural networks can help predict steering angles from images. To showcase this, we will simulator to collect data of good driving behavior and then use the images to train our model. The simulator has been developed with Unity by Udacity (Google) and the code for is available [here](https://github.com/udacity/self-driving-car-sim).

You will find the code - using Keras - for this project is in the [IPython Notebook](https://github.com/itismouad/behavioral_cloning/blob/master/Behavioral%20Cloning.ipynb) and a [video](https://github.com/itismouad/behavioral_cloning/blob/master/video.mp4) displaying how our model can allow a car to successfully drive itself around a track without leaving the road drive.

[//]: # (Image References)

[simulator_pic]: ./data/simulator_pic.png "Simulator Picture Example"
[video_gif]: ./data/final_video.gif "Final video gif"
[loss]: ./data/loss.png "loss per epochs"

## Dataset Exploration

The data was gathered directly on the simulator that allows to collect your own training data by recording a video where one can drive the car manually around a track. Not only this allows to gather the images from the left/center/right cameras but also the logs regarding the steering angle, the speed, etc. You can see below a picture of the car in the simulator.

![alt text][simulator_pic]


The dataset consists of 6144 training observations and 1536 validation observations. Initially there is only one steering angle measurement per the center camera; hence, to increase the size of the dataset, one can also assign a steering angle to the images from the left and right cameras by applying a correction to the steering angle (20 degrees in our case). This helps significantly for dataset augmentation and then training.


## Data Pre-processing

I used two layers for the data-preprocessing. First, we can crop both the top and bottom portion of every image, this helps the model focus on the **right** information and by consequence improve the performance. I also made sure the data was normalized which is always a good practice.


## Model Design and Training

The Convolutional Neural Network that is used here is inspired on [NVIDIA's End to End Learning for Self-Driving Cars paper](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The model gave me full satisfaction after being trained for 4 epochs.

Specifically, the network is 9 layers deep with rectified linear units activation units :

| Layer         		| Description    	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Image                 	   			| 
| Cropping         		| 90x320x3 image                 	   			| 
| Normalization     	| Normalization		                            |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 43x158x24 	|
| RELU activation		|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 20x77x36   |
| RELU activation       |                                               |
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 8x37x48    |
| RELU activation       |                                               |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 6x35x64    |
| RELU activation       |                                               |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 34x33x64   |
| RELU activation       |                                               |
| Flatten               |                                               |
| Fully connected		| 8448 input, 100 output     					|
| Fully connected		| 100 input, 50 output     				     	|
| Fully connected		| 50 input, 10 output     				     	|
| Output         		| 10 input, 1 output     				     	|


After feeding the data to the network by using a generator function, the history losses evolved as following during the four epochs :

![alt text][loss]

I did not fight overfitting since my model gave me enough satisfaction. But I could have added a dropout layer. The model used an adam optimizer.

As a result, I was able to use my model in the simulator and drive the car in autonomous mode ! Here is a small capture below of how the car handles itself in a turn :

![alt text][simulator_gif]


## Final Thoughts

Even if the model is doing already well, we could done even better by collecting more data (driving other tracks, or even backwards). The setting in which our car drives itself is here very simple, everything get more difficulat wih obstacles and a moving environement.

Thanks for reading ! 🚘


