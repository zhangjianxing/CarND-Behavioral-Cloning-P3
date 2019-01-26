# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[steering_dist]: ./img_output/steering_dist.png "Model Visualization"
[steering_dist2]: ./img_output/steering_dist2.png "Model Visualization"
[center]: ./img_output/center.jpg "Model Visualization"
[center_shift]: ./img_output/center_shift.png "Model Visualization"
[center_flip]: ./img_output/center_flip.png "Model Visualization"
[nvidia_cnn]: ./img_output/nvidia_cnn.png "Model Visualization"
[training_curve]: ./img_output/training_curve.png "Model Visualization"
[steering_dist2]: ./img_output/steering_dist2.png "Model Visualization"
[steering_dist2]: ./img_output/steering_dist2.png "Model Visualization"
[steering_dist2]: ./img_output/steering_dist2.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model referenced from http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

#### 2. Attempts to reduce overfitting in the model

The model contains SpatialDropout2D and Dropout layers in order to reduce over-fitting.

The model use generator to feed training and validation data. 
Each iteration, the generator will randomly modify the input image and the angle (discuss lator) 
so that the feeding image can hardly be same. Thus we can prevent over-fitting.

The model was tested by running it through the simulator and ensuring 
that the vehicle could stay on the track.

#### 3. Model parameter tuning

after testing I find the default learning rate = 0.001 is good enough to train the model. 
So I do not need to tune it. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
And collected from multiple drives.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different models.

My first step is trains a single layer neural network, to try how to use neural network to do
behavioral cloning. The output model make the car drive out of the road very quick.

My second step was to use CNN model similar to LeNet. The CNN model performs much better, 
but still drive the car out of the road at second curve.

Finally, I apply [nvidia CNN](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
And That model always make my car hit the ledge or goes out of the road at one position. I keep tuning the model
by adding drop out layer. That helps a lot, but car still might hit the ledge.

Thus, I generate a new set of training data. Which helps the trained model have much better performance.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 21-52) consisted of a 
[nvidia CNN](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
And I have add dropout layers to prevent over-fitting

The model includes ELU layers to introduce nonlinearity 
```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 45, 160, 24)       1824      
_________________________________________________________________
spatial_dropout2d_1 (Spatial (None, 45, 160, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 23, 80, 36)        21636     
_________________________________________________________________
spatial_dropout2d_2 (Spatial (None, 23, 80, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 38, 48)        43248     
_________________________________________________________________
spatial_dropout2d_3 (Spatial (None, 10, 38, 48)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 18, 64)         27712     
_________________________________________________________________
spatial_dropout2d_4 (Spatial (None, 4, 18, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 8, 64)          36928     
_________________________________________________________________
spatial_dropout2d_5 (Spatial (None, 1, 8, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               51300     
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 188,219
Trainable params: 188,219
Non-trainable params: 0
_________________________________________________________________
```

Here is a visualization of the original architecture 
![alt-text][nvidia_cnn]

The model's training history:

![alt-text][training_curve]
In the log, epoch 40 has lowest validation rate. So I choose that one to be final model.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, 
I first recorded one laps on track one using center lane driving. 
Here is an example image of center lane driving:

##### Then I do some pre-process to feed the training data

* I assume the neighbour frame should yield close steering rate. Thus,
while I am reading `driving_log.csv`, I smooth out the steering angle by 
`samples[i][3] = angles[i-1]/5 + 3*angles[i]/5 + angles[i+1]/5`. 

* I found the training data has a lot images that has 0 steering rate:   
![alt_text][steering_dist]

So, I drop 90% of them and and 50% samples whose steering angle is between -0.1 and 0.1.
The resulting images distribution as shown:

![alt_text][steering_dist2]


##### in generator, I randomly modify images.
Thus, in each iteration, there can hardly contains duplicate input, which will prevent over-fitting.  

* I random choose center/left/right camera and tune the steering angle by 0.2 as mentioned in class.

```python
import numpy as np
import cv2
def _random_select_camera(line, adj_rate=0.2):
    img_idx = np.random.choice([0, 1, 2])
    image_dir = line[img_idx].strip()
    image = cv2.imread(image_dir)
    steering = line[3]
    if img_idx == 1:
        steering += adj_rate
    if img_idx == 2:
        steering -= adj_rate
    return image, steering
``` 

* I found the left/right camera is kind of 60 pixels shift from center camera. 
Thus, I randomly shift the chosen camera image by -60~60 pixels and 
correct the steering angle by -0.2~0.2. Thus, we can further augment the data.

Here is an example for shifting image:

![alt text][center]
![alt text][center_shift]

* The sample data set has bias turning left/right. So, I randomly flip the image and set 
`new _steering_rate = -1 * steering_rate` 

![alt text][center]
![alt text][center_flip]

Then I repeated this process on track two in order to get more data points.


After the collection process, I had X number of data points. 
