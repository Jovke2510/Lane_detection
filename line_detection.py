import cv2
import numpy as np
import glob
import os

# Camera calibration
def calibrate_camera(calibration_images_path, chessboard_size=(9, 6)):
    """Calibrates the camera using passed chessboard images."""
    calibration_images = glob.glob(calibration_images_path)

    objpoints = []
    imgpoints = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    for calibration_image in calibration_images:
        img = cv2.imread(calibration_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist

def undistort_image(img, mtx, dist):
    """Undistorts the given image using the calibration matrix and distortion coefficients."""
    return cv2.undistort(img, mtx, dist, None, mtx)

def threshold_image(img):
    """Applies thresholding to isolate lane lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applies Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sobel_binary = cv2.inRange(scaled_sobel, 20, 100)

    # Threshold on color in HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = cv2.inRange(s_channel, 170, 255)

    # Combine thresholds
    combined_binary = cv2.bitwise_or(sobel_binary, s_binary)
    return combined_binary

def perspective_transform(img):
    """Applies a perspective transform to warp the image."""
    h, w = img.shape[:2]
    src = np.float32([[w * 0.45, h * 0.65],
                      [w * 0.55, h * 0.65],
                      [w * 0.1, h],
                      [w * 0.9, h]])
    dst = np.float32([[w * 0.2, 0],
                      [w * 0.8, 0],
                      [w * 0.2, h],
                      [w * 0.8, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, M, Minv

def detect_lane_lines(binary_warped):
    """Detects lane lines in a binary warped image."""
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    midpoint = np.int_(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50

    window_height = np.int_(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_current = left_base
    right_current = right_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_current = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current = np.int_(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def draw_lane(original_img, binary_warped, left_fit, right_fit, Minv):
    """Draws detected lanes on the original image."""
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result

def process_video_frame(frame, mtx, dist):
    """Processes a single video frame."""
    undistorted = undistort_image(frame, mtx, dist)
    binary = threshold_image(undistorted)
    binary_warped, M, Minv = perspective_transform(binary)
    left_fit, right_fit = detect_lane_lines(binary_warped)
    result = draw_lane(undistorted, binary_warped, left_fit, right_fit, Minv)
    return result

def process_video(video_input_path, video_output_path, mtx, dist):
    """Processes a video for lane detection."""
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            processed_frame = process_video_frame(frame, mtx, dist)
            print("processed frame")
            out.write(processed_frame)
        except Exception as e:
            print(f"Error processing frame: {e}")

    cap.release()
    out.release()

def proccess_image(image_path, output_image_path, mtx, dist):
    img = cv2.imread(image_path)
    undistorted = undistort_image(img, mtx, dist)
    binary = threshold_image(undistorted)
    binary_warped, M, Minv = perspective_transform(binary)
    left_fit, right_fit = detect_lane_lines(binary_warped)
    result = draw_lane(undistorted, binary_warped, left_fit, right_fit, Minv)
    cv2.imwrite(output_image_path, result)

    return result

# Paths and Initialization
calibration_images_path = "./Zadatak/camera_cal/*.jpg"
test_image_path = "./Zadatak/test_images/challange00101.jpg"
output_image_path = "./output_image.jpg"
test_video_path = "./Zadatak/test_videos/project_video03.mp4"
output_video_path = "./output_video.mp4"

# Calibration
ret, mtx, dist = calibrate_camera(calibration_images_path)

# Process Video
# process_video(video_input_path, video_output_path, mtx, dist)
# print(f"Video processing complete. Output saved to {video_output_path}.")

# Process Images
out_img = proccess_image(test_image_path, output_image_path, mtx, dist)
print(f"Image processing complete. Output saved to {output_image_path}.")
cv2.imshow("Lane Detection", out_img)
cv2.waitKey(5000)