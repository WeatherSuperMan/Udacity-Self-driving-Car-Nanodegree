### Import libaries
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.signal import find_peaks_cwt
import time
from moviepy.editor import VideoFileClip
from IPython.display import HTML


dist_pickle = pickle.load( open( "dist_pickle.p", "rb" ) )
camera_mtx = dist_pickle["mtx"]
camera_dist = dist_pickle["dist"]
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]




right_fit_count = 0
frame_count = 0


'''
def detect_lines(img_canny_masked):
   
    # Apply HoughLines to extract lines
    rho_res         = 2                 
    theta_res       = np.pi/180.      
    threshold       = 15               
    min_line_length = 40                
    max_line_gap    = 20               
    lines = cv2.HoughLinesP(img_canny_masked, rho_res, theta_res, threshold, np.array([]), 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

'''


# Display Function
def disp_img(original_image, augmented_image, aug_title = ""):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(original_image)
    ax2.imshow(augmented_image)
    ax1.set_title('Original Image', fontsize=30)
    ax2.set_title('Augmented Image: ' + aug_title, fontsize=30)
    plt.show()
    
'''
    
def color_threshold(img, color_thresh=(150, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = HLS[:,:,1]
    s_channel = HLS[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    combined_color_binary = np.zeros_like(s_binary)
    combined_color_binary[(sxbinary == 1) | (s_binary == 1)] = 1
    
    return combined_color_binary
    
'''

def color_threshold(img):
    
    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:,:,0]
    b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,2]
    #s_channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    
    b_thresh_min = 150
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
    
    l_thresh_min = 225  
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    
    '''
    s_thres_min = 180
    s_thres_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thres_min) & (s_channel <= s_thres_max)] = 1
    '''
    
    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1
    
    
    return combined_binary


#_______________________________________________________________________________
'''
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(150, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
     # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value before extracting the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary


def hls_select(img, thresh=(150, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    hls_binary = np.zeros_like(s_channel)
    hls_binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return hls_binary


def color_threshold(img, ksize = 3):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0, np.pi/2))
    hls_binary = hls_select(img, thresh=(180,255))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))|(hls_binary==1)] = 1
    
    return combined
    


#_______________________________________________________________________________
'''
def undistort_img(image):
    # Undistort test image
    undist_image = cv2.undistort(image, camera_mtx, camera_dist, None, camera_mtx)

    img_size = (undist_image.shape[1], undist_image.shape[0])
    
    # Lane line vertices
    # Upper and low are based on visual locations, not grid locations
    center_x = img_size[0]//2
    upper_y = img_size[1]//1.5  
    low_y = img_size[1]
    upper_left_x = center_x//1.33   #1.33
    upper_right_x = center_x//0.8  #0.8
    low_left_x = 0
    low_right_x = img_size[0]  # 2*center_x
    
    '''
    center_x = img_size[0]//2
    upper_y = img_size[1]//1.5
    low_y = img_size[1]
    upper_left_x = center_x//1.33
    upper_right_x = center_x//0.80
    low_left_x = 0
    low_right_x = 2*center_x
    
    '''

    
        
    # Calculate source points based on fractions of imade dimensions
    src_corners = np.float32([[low_left_x, low_y], 
                              [upper_left_x, upper_y], 
                              [upper_right_x, upper_y],
                              [low_right_x, low_y]])
   
    
    # Calculate destination points based on entire image's dimensions.
    
    dst_corners = np.float32([[0, img_size[1]],
                              [0, 0],
                              [img_size[0],0],
                              [img_size[0], img_size[1]]])
    
    
    return undist_image, src_corners, dst_corners



def perspective_transform(image):
    # Calculate perspective transform
    
    undist_image, src_corners, dst_corners = undistort_img(image)    
    
    img_size = (undist_image.shape[1], undist_image.shape[0])
    
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    warped = cv2.warpPerspective(undist_image, M, img_size)
    
    M_inv = cv2.getPerspectiveTransform(dst_corners, src_corners)

    # Draw points and lines to mark region for transform
    
    '''
    for i in range(4):
        
        cv2.circle(undist_image, (src_corners[i,0], src_corners[i,1]), 6, (255, 0, 0), 6)
    for i in range(4):
        
        cv2.line(undist_image, 
                 (src_corners[i-1,0], src_corners[i-1,1]), 
                 (src_corners[i,0], src_corners[i,1]),  
                 (0,255,0), 2)
    '''
        
    return warped, M_inv
        
        
        
### Color Thresholding on Bird-Eye View

def thresholded_img(image):
    # Image thresholding
    warped, M_inv = perspective_transform(image)
    blurred_warped = cv2.GaussianBlur(warped,(5,5),0)

    # Color Thresholding
    combined_color_binary = color_threshold(blurred_warped)
    
    return combined_color_binary, M_inv        
        

### Lane Finding
def lane_coordinates(image):
    
    # Define image height and window size
    img_dim_y = image.shape[0]
    img_dim_x = image.shape[1]
    img_slices = 10
    win_size = img_dim_y//img_slices
    
    # Histogram
    indexes = []
    #hist_val = []
    
    for i in range(img_slices):
        histogram = np.mean(image[(img_slices-i-1)*win_size:(img_slices-i)*win_size,:], axis=0)
        
        # Histogram search area
        left_half_min = 0
        left_half_max = img_dim_x//2.5
        right_half = img_dim_x-(img_dim_x//2.2)
        
        # Histogram peaks
        hist_peaksl = np.argmax(histogram[int(left_half_min):int(left_half_max)])
        hist_peaksr = np.argmax(histogram[int(right_half):])
        
        # Reduce noise
        if histogram[int(hist_peaksl+left_half_min)] > 0.01:
            hist_peaksl = int(hist_peaksl+left_half_min)
        else:
            hist_peaksl = 0
        
        if histogram[int(hist_peaksr + right_half)] > 0.01:
            hist_peaksr = int(hist_peaksr + right_half)
        else:
            hist_peaksr = 0
        
        # Append indices
        indexes.append([hist_peaksl, hist_peaksr])
    
    
    left_lane_idx = []
    right_lane_idx = []
    
    
    for i in range(img_slices):
        # Define window y positions
        win_y1 = (img_slices-i)*win_size
        win_y2 = (img_slices-i)*win_size - win_size        

        # Draw boxes and get indices of thresholded lane points
        # Left Lane
        if (indexes[i][0] != 0):
            # Define window x positions
            win_x1l = indexes[i][0] - (win_size//2)
            win_x2l = indexes[i][0] + (win_size//2)
            
            # Identify lane points where line was detected
            left_lane_idx_local = np.argwhere(image[win_y2:win_y1, win_x1l:win_x2l] > 0)
            # Append to list of lane indices and apply frame of reference transformation
            left_lane_idx.append(left_lane_idx_local + [win_y2,win_x1l])

        # Right lane
        if(indexes[i][1] != 0):
            # Define window x positions
            win_x1r = indexes[i][1] - (win_size//2)
            win_x2r = indexes[i][1] + (win_size//2)
            
            # Identify lane points where line was detected
            right_lane_idx_local = np.argwhere(image[win_y2:win_y1, win_x1r:win_x2r] > 0)
            
            # Append to list of lane indices and apply frame of reference transformation
            right_lane_idx.append(right_lane_idx_local + [win_y2,win_x1r])            
    
    # Concatenate all lane points to respective lane variables
    left_lane_idx = np.concatenate(left_lane_idx)
    right_lane_idx = np.concatenate(right_lane_idx)

    
    return left_lane_idx, right_lane_idx
    
    
### Draw Lane Points on new image
def draw_lane_points(img, left_lane_idx, right_lane_idx):
    
    new_img = np.zeros_like(img)
    for i in range(len(left_lane_idx)):
        for j in range(len(left_lane_idx[i])):
            cv2.circle(new_img, (left_lane_idx[i][1],left_lane_idx[i][0]), 1, (255, 255, 0), 1)

    for i in range(len(right_lane_idx)):
        cv2.circle(new_img, (right_lane_idx[i][1],right_lane_idx[i][0]), 1, (255, 0, 0), 1)
    
    return new_img
        
### Fit lane lines

def identify_lane(left_lane_idx, right_lane_idx, img_size):
    
    global prev_right_fit, right_fit_count
    
    # Obtain individual set of coordinates for each detected lane    
    left_lane_y = np.array([item[0] for item in left_lane_idx])
    left_lane_x = np.array([item[1] for item in left_lane_idx])
    right_lane_y = np.array([item[0] for item in right_lane_idx])
    right_lane_x = np.array([item[1] for item in right_lane_idx])

    # Fit a second order polynomial to lane lines
    left_fit = np.polyfit(left_lane_y , left_lane_x, 2)
    left_fit_x = left_fit[0]*left_lane_y **2 + left_fit[1]*left_lane_y  + left_fit[2]
    
    right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
    right_fit_x = right_fit[0]*right_lane_y**2 + right_fit[1]*right_lane_y + right_fit[2]    

    
    # Extrapolation of left lane
    top_left_y = top_right_y = 0
    bottom_left_y = bottom_right_y = img_size[1]
    
    top_left_x = left_fit[0]*top_left_y**2 + left_fit[1]*top_left_y  + left_fit[2]
    bottom_left_x = left_fit[0]*bottom_left_y**2 + left_fit[1]*bottom_left_y  + left_fit[2]
    
    left_fit_x = np.append(np.flipud(left_fit_x), top_left_x)
    left_lane_y = np.append(np.flipud(left_lane_y), top_left_y)
    
    left_fit_x = np.append(np.flipud(left_fit_x), bottom_left_x)
    left_lane_y = np.append(np.flipud(left_lane_y), bottom_left_y)
        
    
    # Use previous frames to keep track of right lane
    if len(right_lane_x)<1000 and right_fit_count==20:
        right_fit = prev_right_fit/20
        right_fit_count = 0
        
    elif right_fit_count == 0:
        prev_right_fit = right_fit
        
    elif 0 < right_fit_count < 20:
        right_fit_count += 1
        prev_right_fit += right_fit
    
    # Extrapolation of right lane
    top_right_x = right_fit[0]*top_right_y**2 + right_fit[1]*top_right_y + right_fit[2]
    bottom_right_x = right_fit[0]*bottom_right_y**2 + right_fit[1]*bottom_right_y + right_fit[2]
    
    right_fit_x = np.append(np.flipud(right_fit_x), top_right_x)
    right_lane_y = np.append(np.flipud(right_lane_y), top_right_y)

    right_lane_y = np.append(np.flipud(right_lane_y), bottom_right_y)
    right_fit_x = np.append(np.flipud(right_fit_x), bottom_right_x)
    
    return left_lane_y, right_lane_y, left_fit_x, right_fit_x, left_fit, right_fit
    

# Draw curved lines on image
def draw_curved_line(img, line_fit):
    p = np.poly1d(line_fit)
    x = list(range(0, img.shape[0]))
    y = list(map(int, p(x)))
    points = np.array([[y1,x1] for x1, y1 in zip(x, y)])
    points = points.reshape((-1,1,2))
    
    out_img = cv2.polylines(img, np.int32([points]), False, color=(255,255,255), thickness=10)
    
    
    return out_img


# Calculate lane curvature
def lane_curvature(lane_fit_x, lane_fit_y):
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension

    new_fit = np.polyfit(lane_fit_y*ym_per_pix, lane_fit_x*xm_per_pix, 2)
    
    y_eval = np.max(lane_fit_y)
    
    rad_curvature = ((1 + (2*new_fit[0]*y_eval*xm_per_pix + new_fit[1])**2)**1.5)/np.absolute(2*new_fit[0])
        
    return rad_curvature



# Calculate offset from lane center
def distance_from_lane(img, left_lane, right_lane):
    
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    
    img_center = (img.shape[1]//2, img.shape[0])
    
    car_pos = ((left_lane[-1] + right_lane[-1])//2 - img_center[0]) * xm_per_pix
    
    return car_pos

# Plot lane over original image

def draw_lane_line(image, left_lane_y, right_lane_y, left_fit_x, right_fit_x, M_inv):
    
    global frame_count
    frame_count += 1

    # Concatenate lane points
    combined_lane_left = np.array([np.flipud((np.transpose(np.vstack((left_fit_x,left_lane_y)))))])
    combined_lane_right = np.array([np.transpose(np.vstack((right_fit_x,right_lane_y)))])
    combined_lane_idx = np.hstack((combined_lane_left,combined_lane_right))
    
    # Draw lane lines and fill lane area
    img_draw = np.zeros_like(image)
    cv2.polylines(img_draw, np.int_([combined_lane_idx]), isClosed=False, color=(255,0,0), thickness = 40)
    cv2.fillPoly(img_draw, np.int_([combined_lane_idx]), (0,255,0))
    
    # Unwarp transformed image
    new_warp = cv2.warpPerspective(img_draw, M_inv, (image.shape[1], image.shape[0]))
    new_img = cv2.addWeighted(image, 1, new_warp, 0.5, 0)
    
    # Get Radius of Curvature
    left_lane_rad = lane_curvature(left_fit_x, left_lane_y)
    right_lane_rad = lane_curvature(right_fit_x, right_lane_y)
    
    average_lane_rad = (left_lane_rad+right_lane_rad)/2
    
    
    # Overlay Radius of Curvature (text)    
    
    
    cv2.putText(new_img, "Left Lane Radius: " + str("%5d" % left_lane_rad) + " metres", (100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0))
    
    
    '''
    cv2.putText(new_img, "Right Lane Radius: " + str("%5d" % right_lane_rad) + " metres", (100, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255))
    
    cv2.putText(new_img, "Lane Radius: " + str("%5d" % average_lane_rad) + " metres", (100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255))
    '''
    
    
    # Get car position
    vehicle_pos = distance_from_lane(new_img, left_fit_x, right_fit_x)
    
    # Overlay car position (text)
    cv2.putText(new_img, "Distance from Road Centre: " + str("%.3f" % vehicle_pos) + " metres", (100, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255))
    
    return new_img

### Video pipeline
def pipeline(image):
    
    # Threshold Image
    thresholded_image, M_inv = thresholded_img(image)
    img_size = [image.shape[1], image.shape[0]]
    
    # Obtain lane coordinates
    left_lane_idx, right_lane_idx = lane_coordinates(thresholded_image)
    
    # Identify Lane Lines
    left_lane_y, right_lane_y, left_fit_x, right_fit_x, left_fit, right_fit = identify_lane(left_lane_idx, 
                                                                                            right_lane_idx, img_size)
    
    # Draw lane lines
    final_img = draw_lane_line(image, left_lane_y, right_lane_y, left_fit_x, right_fit_x, M_inv)

    
    return final_img



def save():
    # Save the camera calibration result
    dist_pickle = {}
    dist_pickle["mtx"] = camera_mtx
    dist_pickle["dist"] = camera_dist
    dist_pickle["objpoints"] = objpoints
    dist_pickle["imgpoints"] = imgpoints
    pickle.dump( dist_pickle, open( "dist_pickle.p", "wb" ) )
    
    return 0
    
    
def load():
    
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load( open( "dist_pickle.p", "rb" ) )
    camera_mtx = dist_pickle["mtx"]
    camera_dist = dist_pickle["dist"]
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    
    return 0

'''
def make_video(video_path, file_out, ending):

    output = file_out
    length = ending
    clip1 = VideoFileClip(video_path).subclip(0,length)
    # NOTE: this function expects color images!!
    clip = clip1.fl_image(pipeline)
    clip.write_videofile(output, audio=False)

'''

def make_video(video_path, file_out):

    output = file_out
    clip1 = VideoFileClip(video_path)
    clip = clip1.fl_image(pipeline)
    clip.write_videofile(output, audio=False)

   




