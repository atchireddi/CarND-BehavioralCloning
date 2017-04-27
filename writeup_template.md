# **Behavioral Cloning** 


### **Goal:**

The goal of this project is to drive car in simulator environment as autonomous mode.


[//]: # (Image References)

[image1]: ./output_images/DataAnalysis.jpg "Steering Angle Data Distribution"
[image2]: ./output_images/leftCenterRightImage.jpg "Camera Images"
[image3]: ./output_images/flipImage.jpg "Center Camera Regular and Flip Images"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"



### **Overview:**

Behavioral cloning is part of machine learing where we are trying to train a model with the data generated
from some behavior and see how well a model can mimic the behavior. Here data from our own driving 
behavior is used for model training. Trained model is used to drive car in autonomous mode. Results achieved
are quite impressive and its observed the final model behavior is close to user behavior.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road



### **Data Collection:**

Data collected from simulator by driving car in drive mode. The data spits out by simulator is, camera
frames and corresponding steering angle. we have camera data from 3 cameras mounted at center, right and
left sides. Simulator also generates throttle, brake and speed information, which we are not considered for
this project.

we have sample data provided as part of project resource and I drove around the track, couple of rounds 
in `forward direction` and another couple of rounds in `opposite direction` to collect additional data. since track 
has more left turns than right turns, now I have more of balanced data.



### **Data Understanding:**

##### **Camera Images**

Simulator splits out 3 synchronous frames each from left, center and right camers.

##### **Steering Angle**

Steering angle produced is in radians and is normalised with in  -0.25 to 0.25 range. where negative values
indicates steer left and positive values indicates steer right.

I performed some data exploration to understand the driving behavior. Steering angle distribution shows most 
of the data points are concentrated with in -0.5 to 0.5, which represents a quite good driving behavior.
But this data doesn't teach a model how to steer back if it was going off the track. so, I gathered additional
data by explicitly drove car towards curb and pull back.

one way to spot for the recovery behavior, is, looking at the outliers of the steering angle distribution. For a 
typical machine learning problem, we don't like to have outliers in the data. But for automous vehicles 
precence of outliers helps a lot.

Code for data analysis can be found in `1_SampleDrvingData_Analysis.ipynb`

Below is steering distribution plot:

![alt text][image1]



### **Data Preparation**

##### **Data sources:**

I started inital trails with only center camera images. well, the trained model did ok for a small distance
and can't recovery from curb near left curvature and vehicle goes off the track. For later trails, I used l
eft and right camera images with their steer angle's shifted by offset. 


##### **Data Augmentation:**

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


Below is snapshot of camera images.

![alt text][image2]



Below is snapshot of center camera image and its flipped version.

![alt text][image2]


Code for data preparation can be found in `2p1_DataPreparation.ipynb`



### **Model Training and Validation:**

##### **Model Architecture**

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



##### **Model Layers:**

My model has 3 convolution layers, followed by one fully connected layer:

    -- Conv1 Layer : 8x8 filter of depth 16 applied with (4, 4) subsample.
    -- Conv2 Layer : 5x5 filter of depth 32 applied with (2, 2) subsample. 
    -- Conv3 Layer : 5x5 filter of depth 64 applied with (2, 2) subsample.
    -- FC          : one fully connected layer of 1024 wide.
    -- Output Layer : Output layer with single node.



##### **Region of Interest:**

By intution, the independant features that would steering angle are lane lines/curb/road. so, it does make 
sense to constrain the focus to the image regions where likely these features appear. Its apparent, that
these features lie in the bottom half of the image. so, I expose only `bottom half` of image as features for 
the model.



##### **Normalization:**
I `normalized` all image pixels to `0-1 range`, to avoid feature biasing.


##### **Overfitting:**
A common issue with most of machine learning algorithms is overfitting. To avoid model overfitting, I use
`50% dropout` for all layers.


##### **Parameter Tuning:**
I used `adam` otpimiser with 0.0001 learning rate.


##### **Model Training:**
 
The data I collected along with augmentation was around `76k images`. Date was `randomly shuffled` and put `20% 
of the data into a validation set`. Validation set was used to measure the model performance on unseen data.
I used `adam` optimizer with a `learning rate 0.0001`

code implementation for Model training is in `3_CommaModel_cropping.ipynb`

Trained model is saved as `h5` format. `model_sides0p15_ang0p25.h5`


### **Model Testing in Simulator:**

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

### **Discussion:**

My model could drive vehicle on track 2 for some distance, though it took higher throttle for uphill terrain. 
But it meserably failed to go past hairpin curvatures. Also track 2 wasn't flat terrain like track1. I feel, for an autonomous driving it needs models not just for predicting steering angle, also need similar models for predicting throttle, brake, speed.






