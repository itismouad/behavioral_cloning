# Behavioral Cloning for self-driving cars

[//]: # (Image References)

[video_gif]: ./data/final_video.gif "Final video gif"
[loss]: ./data/loss.png "loss per epochs"


## Overview


In this project, we use a deep convolutional neural networks to help predict steering angles from camera images. To showcase this, we use simulator to collect data of good driving behavior and then use the images to train a model. A video with the footage of the car driving itself in autonomous mode is available in the repository.

![alt text][video_gif] 

You will find the code - using Keras - for this project is in the [IPython Notebook](https://github.com/itismouad/behavioral_cloning/blob/master/Behavioral%20Cloning.ipynb). More details are available by reading the project [notes](https://github.com/itismouad/behavioral_cloning/blob/master/behavioral_cloning.md).

 

### The final model architecture :


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