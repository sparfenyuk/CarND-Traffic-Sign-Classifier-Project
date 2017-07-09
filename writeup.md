## **Build a Traffic Sign Recognition Project**

### The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/datavis.png "Visualization"
[image2]: ./writeup/samples.png "Samples"
[image3]: ./writeup/grayscale.png "Grayscaled images"
[image4]: ./writeup/5155701-German-traffic-sign-No-205-give-way-Stock-Photo.jpg "Traffic Sign 1"
[image5]: ./writeup/stock-photo-dutch-road-sign-road-narrows-on-both-sides-454308259.jpg "Traffic Sign 2"
[image6]: ./writeup/stock-photo-german-road-sign-stop-and-give-way-369157709.jpg "Traffic Sign 3"
[image7]: ./writeup/stock-photo-warning-triangle-slippery-road-traffic-sign-417252529.jpg "Traffic Sign 4"
[image8]: ./writeup/vaerloh-traffic-signs-double-curve-initially-right-cxdxff.jpg "Traffic Sign 5"
[image9]:./writeup/predictions.png "Predictions"

## Data Set Summary & Exploration


I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**.
* The size of the validation set is **4410**.
* The size of test set is **12630**.
* The shape of a traffic sign image is **32x32x3**.
* The number of unique classes/labels in the data set is **43**.

Here is an exploratory visualization of the data set. It is a horizontal bar chart showing how the data distributed by kind of signs. Probably the half of signs in training set has less than 500 samples when some other signs have roughly 2000 samples.

![Distribution of classes][image1]

Let's display some signs to see how they looks like. We can see that some images have low quality to understand what kind of sign is it. Some photos have perfect contrast when others haven't. Some images were taken probably at evening time when others were taken during day time. We can conclude that images differ in wide range of properties.

![Samples][image2]

## Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because having only one channel of data instead of three will speed up the training and it will take less time to finish. But in such a way color context is dropped and the result may be less accurate.

Then I normalized all images to have mean value equal to 0.

Here is an example of a traffic sign images before and after grayscaling.

![Grayscaled images][image3]

My final model consisted of the following layers:

| Layer             		|     Description	        				            	|
|:---------------------:|:---------------------------------------------:|
| Input             		| 32x32x1 grayscale image						            |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	  |
| RELU					        |												                        |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6  				        |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16	  |
| RELU					        |												                        |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x16   				        |
| Fully connected		    | input - 400, outputs 120     									|
| dropout       		    | keep 50%                    									|
| RELU          		    |                              									|
| Fully connected		    | input - 120, outputs 84     									|
| dropout       		    | keep 50%                    									|
| RELU          		    |                              									|
| Fully connected		    | input - 84, outputs 43      									|

The architecture is very similar to LeNet network with additional dropout layers.

To train the model, I used an AdamOptimizer with learning rate 0.001. AdamOptimizer is known to be the best overall choice. Also I've set batch size to be 128 and epochs to be 15. By the way, 10 will be enough, I believe.

My final model results were:
* validation set accuracy of 0.956
* test set accuracy of 0.930

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
>I've chosen LeNet architecture that is known to solve such tasks.

* What were some problems with the initial architecture?
>I couldn't get acceptable accuracy. So I had to modify model by adding dropout layers to overcome overfitting.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
> Only two modification was made - the network input shape changed (RGB channels reduced to grayscale) and dropout layers were added. Dropout helped me to increase validation accuracy higher than the acceptable value.

* Which parameters were tuned? How were they adjusted and why?
>I've tuned such parameters as learning rate - it was decreased from 0.05. Tried also change the batch size to fit to hardware possibilities of my laptop.

## Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

All stock photos are in a very good state because they need to be of great quality. Some of them are overlaid with watermarks so it may be confusing for network to make predictions. But I guess the accuracy will be very high.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Slippery Road			| Slippery Road      							|
| Yield					| Yield											|
| Narrows from both sides      		| Traffic signals sign   									|
| Keep right     			| Keep right 										|
| Stop	      		| Stop					 				|
| Dangerous curve to the right	      		| Dangerous curve to the right					 				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93%.

In the table above there are 6 images. But one is not a part of training set, so it cannot be recognized. But assume it's looking similar to sign 'Traffic signals' the model works fine.

The softmax probabilities for each images are displayed in following image:
![alt][image9]

The summary is given in next table:

| Probability         	|     Prediction	    |
|:---------------------:|:-------------------:|
| 0.38			| Slippery Road      							|
| 0.46			| Yield											      |
| 0.08      | Traffic signals sign  					|
| 0.51     	| Keep right 										  |
| 0.08	    | Stop					 				          |
| 0.14	    | Dangerous curve to the right		|			 				


Traffic signals sign was not in training set so the probability of wrongly detected sign is very low. The stop sign also has too low value of probability. It is possible because it slightly turned to the right.

Other predictions are relatively high.
