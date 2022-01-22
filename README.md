# Smart_Tech_SD4_CA2

### Project: 
This project was carried out by ***William Hadnett and Aaron Reihill*** as part of the final year ***Smart Technologies module***.
We were tasked with creating a smart driving car that could autonomously drive in a simulator, provided by [Udacity](https://github.com/udacity/self-driving-car-sim "Udacity").
When buidling the convolutional neural network modle that will carry out this function, we had to take into account preparing the data in which will allow our modle to output ***steering angle, throttle, reverse and speed***. 

### Goals 

- Collect data 
- Process data
- Construct model layers 
- Build model 
- Train model 
- Connect to simulator 
- Car drive around the track 
- Car goes fast around the track 

### Required Packages: 
#### drive.py
- pip install python-socketio==4.6.1
- pip install python-engineio==3.13.2
- pip install eventlet
- pip install tensorflow
- pip install Flask
- pip install numpy
- pip install opencv-python

#### bc.py
- pip install numpy
- pip install matplotlib
- pip install tensorflow
- pip install pandas
- pip install imgaug
- pip install scikit-learn
- pip install opencv-python



### Simulator Videos:
#### Track 1
https://user-images.githubusercontent.com/68790566/150642931-c995851b-6acc-456f-9e49-4a8a2f7a1d84.mp4
#### Track 2
https://user-images.githubusercontent.com/68790566/150643294-f084e704-cea0-4587-b80a-5016d5db3682.mp4

### Model Loss Graph
![Model_22](https://user-images.githubusercontent.com/68790566/150642323-705a520b-ee39-4674-b1c7-d943318222ee.png)

### Model Reliability
As seen in the above videos. The model works well on both tracks, but we found that this is not always the case. On Williams machine it works smoothly, then on Aarons machine it sometimes crashes on track 1. 
