## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
<!-- [image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png -->
[video1]: ./readme/project_video.mp4

[image1]: ./readme/writeup_imgs/car.png
[image2]: ./readme/writeup_imgs/notcar.png
[image3]: ./readme/writeup_imgs/hog_visualization.png
[image4]: ./readme/writeup_imgs/test1_windows_64x64.png
[image5]: ./readme/writeup_imgs/test1_windows_96x96.png
[image6]: ./readme/writeup_imgs/test1_windows_128x128.png
[image7]: ./readme/writeup_imgs/debug_error.png
[image8]: ./readme/writeup_imgs/debug_solution.png
[image9]: ./readme/writeup_imgs/car_3d_plot.png
[image10]: ./readme/writeup_imgs/car_3d_plots.png
[image11]: ./readme/output_images/test1_output.png
[image12]: ./readme/output_images/test2_output.png
[image13]: ./readme/output_images/test3_output.png
[image14]: ./readme/output_images/test4_output.png
[image15]: ./readme/output_images/test5_output.png
[image16]: ./readme/output_images/test6_output.png
[image17]: ./readme/writeup_imgs/debug_13.png



---

### Histogram of Oriented Gradients (HOG)

#### 1. Feature Sets

The code for this step is contained in the beginning cells of the IPython notebook. Using the `glob` library, the file paths are dumped into variables. One for cars and another for images that are not cars.

In order combat false positives, I added set of my own set of 64 x 64 pixel images that the model had problems with. They can be found in the debug folder.

###### Car Image Set Sample (64px x 64px)
![alt text][image1]

###### Not Car Image Set Sample (64px x 64px)
![alt text][image2]

###### Troublesome False Positive
![alt text][image17]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I used the complete file set provided by the project. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

###### Sample HOG Visualization
![alt text][image3]


#### 2. HOG Parameters

I tried various combinations of parameters and get some results. After plotting out the image vector values, the LUV color space showed some interesting results. After switching to `LUV` from `RGB` and `YCrCb`, The model showed considerably less false positives in detection.

##### Sample Image
![alt text][image9]

##### Color Space 3D Plots
![alt text][image10]

The `LUV` color space yielded the most interesting results.

I assumed that increasing the modifying the spacial size and number of histogram bins would allow even better detection, but modifying these values in small increments did not show any major changes to the results.

```python
'color_space': 'LUV', #Colorspace
'orient': 9, # Orientation
'pix_per_cell': 16, # HOG pixels per cell
'cell_per_block': 2, # HOG cells per block
'hog_channel': "ALL", # Can be 0, 1, 2, or "ALL"
'transform_sqrt': False, # Square root transform
'spatial_size': (24, 24), # Spatial binning dimensions
'hist_bins': 24,    # Number of histogram bins
'spatial_feat': True, # Spatial features on or off
'hist_feat': True, # Histogram features on or off
'hog_feat': True # HOG features on or off
```

For the support vector classificator, the parameter `C` was set to `.2` in order to rule out false positives.

Using these parameters yielded a 98.96% test accuracy. With the result, I believed that it was safe to move on into the sliding window algorithm.

#### 3. Classifier Training

In the Lesson Helpers section, `extract_features()` did most of the heavy lifting for preparing the feature set. I then created a wrapper class that executed the function calls to train the model. The trainer is under the Training section of the notebook. It is modified version of the code presented in the lessons.

```python
class Trainer(object):


  def trainer(self,
              cars,
              notcars,
              color_space,
              spatial_size,
              hist_bins,
              orient,
              pix_per_cell,
              cell_per_block,
              hog_channel,
              spactial_feat,
              hist_feat,
              hog_feat):

  ...

      rand_state = np.random.randint(0, 100)
      X_train, X_test, y_train, y_test = train_test_split(
          scaled_X, y, test_size=0.2, random_state=72)

  ...

      # Use a linear SVC
      svc = LinearSVC()
      # Check the training time for the SVC
      t=time.time()
      svc.fit(X_train, y_train)

  ...

```

I used a linear SVM and yielded high performing results. The model was able to accurately detect all cars from the test image set. I kept the random state constant so that I would have a basis for improvement.  I split the feature set using 80% for training and 20% for testing.

### Sliding Window Search

#### 1. Scale and Overlapping

The sliding window algorithm was derived from the project lessons and can be found in the Lesson Helpers section. I wrote a wrapper for the sliding windows so that I could testing different window sizes and window container sizes. I kept the window container on the bottom half of the images to avoid noise. Then I used a variation of 32, 64, 96, and 128. I allowed for an overlap between .5 and .75 to allow for more data to be handled. For the smaller windows, I used a smaller overlap and a larger overlap for the larger windows.

The class `Slider` was created as a wrapper to minify all the parameter handling. It can be found under the Sliding Window Class section.

The scaling and overlapping of the windows was a resultant of the scientific method. Below shows my analysis of different sized windows with different overlapping parameters. The higher the overlapping percentage, the more windows would be needed to be searched.

###### Window Size 64px x 64px
![alt text][image4]

###### Window Size 96px x 96px
![alt text][image5]

###### Window Size 128px x 128px
![alt text][image6]

#### 2. Pipeline

I created a `Pipeline` class that allows for the adjustment of the hyper-parameters. It can be found in the Pipeline section.

Ultimately I searched on two scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

Use the `Pipeline` class I was able to create a debugging workhorse that helped me analyze frames that did not show very good predictive results. I would iterate through various frames while adjusting the hyper parameters in order to get sound results. The code can be found at the bottom of the IPython notebook.

###### Error
![alt text][image7]

###### Debugging Results
![alt text][image8]

---

### Video Implementation

#### 1. Video Output
Here's a [link to my video result](./readme/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The heatmap included all the boxes found for each size. From this standpoint all boxes have the same weight on deciding whether a car has been detected. I believe that by adding a variance to the boxes weight, a better model can be made.

In order to rule out false positives. The heat map object saves the previous heatmap into memory. It is then added with the current frame and adjusts itself depending on the heat. Heat maps have a max threshold of 10 while removing any negative values.

```python
def heater(self, img, box_list):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    current_heat = add_heat(heat, box_list)

    #cooling
    current_heat[current_heat < 1] -= self.dissipate

    self.heat += current_heat

    #set the lower limit to prevent false negatives
    self.heat[self.heat < 0] = 0

    #set the upper limit to prevent ghosting
    self.heat[self.heat > self.heatmax] = 10

    return self.heat
```

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps analysis:

###### Frame 1
![alt text][image11]

###### Frame 2
![alt text][image12]

###### Frame 3
![alt text][image13]

###### Frame 4
![alt text][image14]

###### Frame 5
![alt text][image15]

###### Frame 6
![alt text][image16]

---

### Discussion

#### 1. Problems

The pipeline took very long to complete. this was because I was using large overlaps with many different sized windows for the sliding window algorithm.

Vehicle detection became a lot easier for the model if more windows were used, however this also introduced a lot of false positives

The HOG parameters were largely based on the lessons and proved to be very accurate during training. With time as a constraint, it would be better to explore in-depth what changes could be made to create a better model.
