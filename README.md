# Lane_detection

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera calibration was performed using the calibrate_camera function. Chessboard images from the specified directory were used to find object points and image points. OpenCV's cv2.calibrateCamera() function was used to compute the camera matrix and distortion coefficients.

TODO: add an image

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![undistort](https://github.com/user-attachments/assets/9ba2adec-5c05-4cf9-ad9e-d77ea18f9c8e)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Color transform was performed using cv2.cvtColor() function with cv2.COLOR_BGR2GRAY parameter to turn the image into grayscale. Then the cv2.Sobel() function was used to isolate the edges in the grayscaled image. Meanwhile the
cv2.cvtColor() function with the cv2.COLOR_BGR2HLS parameter was used to convert the image into HLS color space. This was done in order to isolate bright lane colors (white and yellow). When combined with edges isolated by the Sobel gradient we get more robust way of detecting lane lines. This was done in threshold_image() function

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) //grayscale transform
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) //sobel edge detection
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) //hls transform
combined_binary = cv2.bitwise_or(sobel_binary, s_binary) //combining sobel edge detection and isolated hls bright colors 
```
 

![binary_th](https://github.com/user-attachments/assets/ded66850-8d86-40da-9df9-7d178eab733f)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective_transform() function was used to warp the image into a bird's-eye view using predefined source (src) and destination (dst) points. src defines a trapezoidal region in the original image, enclosing the visible lane area while dst defines a rectangular region in the transformed image where the trapezoidal region will be warped. The warp makes the lanes appear straight, enabling more accurate lane detection and polynomial fitting. It does this by using cv2.warpPerspective(img, M, (w, h)) on the transformation matrix M. Transformation matrix M was computed by using cv2.getPerspectiveTransform() function that maps points from one region to the other.

```
src = np.float32([[w * 0.45, h * 0.65],
                      [w * 0.55, h * 0.65],
                      [w * 0.1, h],
                      [w * 0.9, h]])  //defining source points 
dst = np.float32([[w * 0.2, 0],
                      [w * 0.8, 0],
                      [w * 0.2, h],
                      [w * 0.8, h]]) //defining destination points 
M = cv2.getPerspectiveTransform(src, dst) //computing the transformation matrix (src -> dst)
Minv = cv2.getPerspectiveTransform(dst, src) //computing inverse transformation matrix (dst -> src)
warped = cv2.warpPerspective(img, M, (w, h))
```

![perspective_warp](https://github.com/user-attachments/assets/b4e81884-07e0-44d7-a18c-2cbff7d5ca5d)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The detect_lane_lines() function identifies lane-line pixels using a sliding window approach, starting from histogram peaks that represent the base of the lanes. It then fits a second-degree polynomial to the identified pixels using NumPy's polyfit() function. The sliding window approach narrows down the search space for lane lines, improving computational efficiency and accuracy. Polynomial fitting provides a mathematical representation of lane boundaries, enabling curvature calculations and lane visualization.

```
histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0) #Create a histogram of the bottom half of the binary warped image to identify peaks

midpoint = np.int_(histogram.shape[0] / 2)
left_base = np.argmax(histogram[:midpoint])
right_base = np.argmax(histogram[midpoint:]) + midpoint #Find the base positions of the left and right lanes using the histogram peaks

nwindows = 9               # Number of sliding windows
margin = 100               # Width of the window (± from its center)
minpix = 50                # Minimum number of pixels to recenter window

window_height = np.int_(binary_warped.shape[0] / nwindows) # Divide the image into 9 horizontal slices (one per window)

nonzero = binary_warped.nonzero() # Get the indices of all non-zero pixels in the binary image

for window in range(nwindows) # Loop through each sliding window

good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0] # Identify pixels within the left and right windows

# Extract the x and y coordinates of the pixels for both lanes
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second-degree polynomial to each lane's pixels
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![output_image](https://github.com/user-attachments/assets/5ee45231-7191-4079-ac03-1351b2dc414b)


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

https://youtu.be/kgV9QEtHT6U

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

