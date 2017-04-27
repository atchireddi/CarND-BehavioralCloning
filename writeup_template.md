# **Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Goal:**
The goal of this project is to drive car in simulator environment as autonomous mode.


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"



**Overview:**

Behavioral cloning is part of machine learing where we are trying to train a model with the data generated
from some behavior and see how well a model can mimic the behavior. Here data from our own driving 
behavior is used for model training. Trained model is used to drive car in autonomous mode. Results achieved
are quite impressive and its observed the final model behavior is close to user behavior.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




**Data Collection:**

Data collected from simulator by driving car in drive mode. The data spits out by simulator is, camera
frames and corresponding steering angle. we have camera data from 3 cameras mounted at center, right and
left sides. Simulator also generates throttle, brake and speed information, which we are not considered for
this project.

we have sample data provided as part of project resource and I drove around the track, couple of rounds 
in forward direction and another couple of rounds in opposite direction to collect additional data. since track 
has more left turns than right turns, now I have more of balanced data.


**Data Understanding:**

**Camera Images**
Simulator splits out 3 synchronous frames each from left, center and right camers.

**Steering Angle**
Steering angle produced is in radians and is normalised with in  -0.25 to 0.25 range. where negative values
indicates steer left and positive values indicates steer right.

I performed some data exploration to understand the driving behavior. Steering angle distribution shows most 
of the data points are concentrated with in -0.5 to 0.5, which represents a quite good driving behavior.
But this data doesn't teach a model how to steer back if it was going off the track. so, I gathered additional
data by explicitly drove car towards curb and pull back.

one way to spot for the recovery behavior is looking at the outliers of the steering angle distribution. For a 
typical machine learning problem, we don't like to have outliers in the data. But for automous vehicles 
precence of outliers helps a lot.

Below is steering distribution plot:


**Data Preparation**

**Data sources:**
I started inital trails with only center camera images. well, the trained model did ok for a small distance
and can't recovery from curb on the turning and goes off the track. For later trails, I used left and right 
camera images with their steer angle's shifted a bit. 


**Data Augmentation:**
since the images from center camera alone not enough for the model to mimic the behavior, though it does performed
well on the validation data. I decided to augment data, couple of ways to do it, either 

    -- 1. Flip center camera images
    -- 2. Use shifted center camera images
    -- 3. To adding left and right camera images. 

I choose to #1 and #3

Adding the entire set of left and right camera data would be a kill, so I decided to selectively pick left and right
camera images, which can help model learn to steer away from curb. I pick images where steer angle is 
    
    -- less than -0.15 from right camera
    -- greater than 0.15 from left camera.

Below is snapshot of fliped camera images.


**Model Training and Validation:**

**Model Architecture**

Below is the final model architecture, that produced impressive results.
Model has 3 convolution layers followed by 1 fully connected layer.
Output layer has one node whose values represents the predicted steering angle.
ELU activation function is employed.

```
model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    
    model.add(Cropping2D(cropping=((75,25),(0,0)), input_shape=(row,col,ch)))
    
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1024))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    keras.optimizers.adam(lr=0.0001)
    model.compile(optimizer="adam", loss="mse")
```

**Model Layers:**
My model has 3 convolution layers, followed by one fully connected layer:

    -- Conv1 Layer : 8x8 filter of depth 16 applied with (4, 4) subsample.
    -- Conv2 Layer : 5x5 filter of depth 32 applied with (2, 2) subsample. 
    -- Conv3 Layer : 5x5 filter of depth 64 applied with (2, 2) subsample.
    -- FC          : one fully connected layer of 1024 wide.
    -- Output Layer : Output layer with single node.


**Region of Interest:**
By intution, the independant features that would steering angle are lane lines/curb/road. so, it does make 
sense to constrain the focus to the image regions where likely these features appear. Its apparent, that
these features lie in the bottom half of the image. so, I expose only `bottom half` of image as features for 
the model.

**Normalization:**
I `normalized` all image pixels to `0-1 range`, to avoid feature biasing.

**Overfitting:**
A common issue with most of machine learning algorithms is overfitting. To avoid model overfitting, I use
`50% dropout` for all layers.

**Parameter Tuning:**
I used `adam` otpimiser with 0.0001 learning rate.

**Model Training:**


**Model Testing in Simulator:**

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```




####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
