# Deep Learning for sensor-based Human Activity Recognition.
A detailed analysis of my approach to HAR, using Deep Learning.
#
With the progress of Machine Intelligence in the past years, we are now able to use smart-watches, mobile applications empowered with Artificial Intelligence to predict what activity a Human being is doing based on raw accelerometer and gyroscope sensor signals. This problem is generally referred to as Sensor-based Human Activity Recognition (HAR). Its applications vary from healthcare to security (gait analysis for human identification, for instance).

Unfortunately, most of the classical approaches used for HAR heavily rely on heuristic hand-crafted feature extraction methods, which dramatically hinders their generalization performance. Moreover, there rarely exists efficient end products that perform on-device real-time activity recognition with high accuracy and low resource consumption. These are the realities that caught the attention of the computer scientist that I am, who believe there is possibility for better overall performance, taking advantage of existing improvements in Deep Learning. But before describing my solution, let’s stress out the limitations of existing approaches for the HAR problem in general.

## 1. Limitations of existing approaches

Decision tree, Support Vector Machine, Naive Bayes, Hidden Markov Models and K-Nearest Neighbors are the most used algorithms to tackle HAR problem. Models implementing these algorithms:

* **Require hand-crafted feature generation:** Ax, Ay, Az, Gx, Gy, Gz are the basic input parameters of HAR algorithms as time series. For a classic application, algorithms mentioned above are irreproachable. But classifying a time series (a window of accelerometer and gyroscope readings) requires another approach as the classification is instead done on a sample window. To achieve this goal using classical approaches, one has to generate statistical features related to time-domain or frequency-domain (median, min, max, std, L-SMA, DiffoY, SOR, etc.) for each training window (50 Hz sample rate by example). Sometimes, PCA is also applied after this for scale invariance but at the end the final model has to predict on an OVERVIEW of a sample window instead of on each record individually. Which feature to generate then? This is just the problem! There is no magic formula and this is where experience pays. It should not be the case though...
* **Do not generalize:** the accuracy of common algorithms tends to slow down when activities are performed by people not included in the training phase. This is a big problem for an application that has TO SCALE. Also, the algorithms are usually confused when we change the way we wear the sensor devices (device location on the body: hand, wrist, pocket, etc.; orientation of the device: axis change direction as well :relaxed: ).
* **Use Online APIs:** in existing approaches, trained models are deployed as web services or REST APIs that end products (mobile apps for instance) call periodically for real-time HAR. Even if there is nothing wrong with this approach, we believe on-device prediction deserves more attention. And this is exactly what these solutions cannot provide: the ability to deploy or embed the trained model in a mobile app.
* **Are not optimized for low resource consumption:** the very few existing solutions which rely on sensors data for on-device HAR using custom mobile apps lack resource optimization. Indeed, computing time and frequency domain statistics for each sample (at a given rate) before serving a model for prediction is one of the reasons why HAR is still a NOT FULLY SOLVED PROBLEM.

We believe we can take advantage of existing improvements in Artificial Intelligence to solve most of the problems mentioned above, with Deep Learning.

## 2. How we approach the problem

**TRAINING DATASET**

This heavily depends on our application. Generally, activities we are interested in are *Sitting, Standing, Walking, Running, Climbing Stairs Up, Climbing Starirs Down, etc.* but I have applied HAR once to predict different yoga steps (Bosch Hackathon 2017, Finalist). So, as I said before, it only depends on the application we want to apply HAR to. I have applied my approach on Classic HAR using the dataset collected by Allan et al (1,6 GB). This dataset contains readings from two sensors (accelerometer and gyroscope). Readings were recorded when users executed activities in no specific order, while carrying smartwatches and smartphones. The readings are from 9 users performing 6 activities (Sitting, Standing, Walking, Biking, ClimbStair-Up and ClimbStair-Down) using 6 types of mobile devices. 

Some common issues we face when applying HAR to a custom task are the imbalance of the dataset and the lack of enough training data. In the first case, I usually apply [SMOTE](https://www.jair.org/media/953/live-953-2037-jair.pdf) oversampling technique and naturally adapt it as a data augmentation solution for the second case.

**MODEL ARCHITECTURE**

We propose an architecture combining a CNN (Convolutional Neural Network) and a RNN (Recurrent Neural Network). Input sensor measurements are split into series of data intervals along time. The representation of each data interval is fed into a CNN to learn intra-interval local interactions within each sensing modality and intra-interval global interactions among different sensor inputs, hierarchically. The intra-interval representations along time are then fed into a RNN to learn the inter-interval relationships.

The CNN **automatically extracts local features** within each sensor modality and merges the local features of different sensor modalities into global features hierarchically. This beats the classical hand-crafted feature generation used in existing approaches.

The RNN **extracts temporal dependencies**. To understand how this could be useful, I tested existing approaches by performing an activity in various positions (changing my body configuration). My objective was to confuse the model; to obtain accelerometer and gyroscope values it was not used to, for that specific activity. The prediction accuracy expectedly slowed down and the explanation is quite simple: the model learned to predict the activity based on raw values instead of how they correlate or vary over time. But this is exactly how I think it should work because different performances of the same activity, independently of who performed it or the position used, produce approximately the same variation of Ax, Ay, Az, Gx, Gy, Gz over time. So why not take this into consideration in the training phase? This is exactly why our model features a RNN.

Because of the consistency of our architecture, our final model generalizes pretty well. The rest is up to fine-tuning.

**TRAINING THE MODEL**

Because building a deep learning model from scratch requires high performing computers and GPUs, it is in our advantage to build the model on a cloud platform. I used Google Cloud for this purpose. Also, I designed the algorithm using Google TensorFlow with Python 3.4. 

**ON-DEVICE PREDICTION**

It is nowadays crucial to be able to port a model to mobile platforms. This is because real-life applications of HAR are more and more integrated in mobile apps. Hopefully enough, Google provides libraries to use TensorFlow on Android. This makes possible the use in an android app of a model built in Python with TensorFlow. All we need to do is "Freeze" the model meta-graph and export it as a file (.pb format) for use in the app. Using Google inference library for Android, we can now feed the model with real-time sensors data.

More concretely:
*	Freeze the model and export it in a file (model.pb for instance)
*	Add Google inference library for Android in the Android app
*	Create an Activity in the Android App that feeds the model each 50 Hz (Sample rate) with sensors real-time accelerometer and gyroscope data.
*	Return the predicted activity or the activities with their prediction confidence (probability). 

We see that we do not need to compute extra features for prediction. the resource consumption of our technology is therefore limited to only the collection of sensors data periodically and the prediction by feeding our model graph.

## 3. Results
We were able to obtain 98% of training accuracy and 92% of testing accuracy using a customized cross-validation technique involving leaving one user out during training and testing on him.

## 4. Challenges (TO DO)

* Transfer Learning: research on how to apply existing models on custom activities without having to train from scratch.
* Incremental learning: this is a kind of dynamic learning where input data is continuously used to extend the existing model's knowledge without bearing the cost of retraining the entire model from scratch. It could be very helpful in HAR to be able to add, in production, training data of a new user and see it be reflected directly in the model knowledge.
* On-device learning: this is a modern challenge somehow beyond our capacities. Google is doing a great job here with [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite) and [TensorFlow Mobile](https://www.tensorflow.org/mobile/mobile_intro). 
