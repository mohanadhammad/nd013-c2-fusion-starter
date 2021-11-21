# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?

### Lidar 3D Object Detection

#### Section 1 : Compute Lidar Point-Cloud from Range Image

This task is about extracting and visualizing the range images from the Waymo Open dataset. The intensity and range channels are extracted and converted to 8-bit integer value range. Then OpenCV library is used to stack the range and intensity images vertically and visualize it.

##### Ex.1 : Visualize range image channels (ID_S1_EX1)

###### Task Preparations:

```` python
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
show_only_frames = [0, 1]
exec_data = []
exec_detection = []
exec_visualization = ['show_range_image']
````

###### Steps:
1. Extract LiDAR data and range image for the roof-mounted LiDAR.
2. Extract the range and intensity channels from the range image.
3. Convert both channels from real float values to 8-bit range values.
4. Stack the range and intensity channels vertically and visualize it with OpenCV library.

###### Code Implementation:

```` python
# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first sequence
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    
    # step 2 : extract the range and the intensity channel from the range image
    ri_range_ch = ri[:, :, 0]
    ri_inten_ch = ri[:, :, 1]
    
    # step 3 : set values <0 to zero
    ri_range_ch[ri_range_ch < 0.0] = 0.0
    ri_inten_ch[ri_inten_ch < 0.0] = 0.0
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    ri_range_ch = ri_range_ch * 255 / (np.amax(ri_range_ch) - np.amin(ri_range_ch))
    
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    ri_inten_ch = np.amax(ri_inten_ch)/2 * ri_inten_ch * 255 / (np.amax(ri_inten_ch) - np.amin(ri_inten_ch))
    
    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    #img_range_intensity = [] # remove after implementing all steps
    ri_range_ch = ri_range_ch.astype(np.uint8)
    ri_inten_ch = ri_inten_ch.astype(np.uint8)

    deg45 = int(ri_range_ch.shape[1] / 8)
    center = int(ri_range_ch.shape[1] / 2)

    ri_range_ch = ri_range_ch[:, center-deg45 : center+deg45]
    ri_inten_ch = ri_inten_ch[:, center-deg45 : center+deg45]

    img_range_intensity = np.vstack((ri_range_ch, ri_inten_ch))
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity
````

###### Output:

![range_image_frame_0](doc/figures/S1_E1.png)

##### Ex.2 : Visualize lidar point-cloud (ID_S1_EX2)

#### Section 2 : Create Birds-Eye View from Lidar PCL
##### Ex.1 : Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)
##### Ex.2 : Compute intensity layer of the BEV map (ID_S2_EX2)
##### Ex.3 : Compute height layer of the BEV map (ID_S2_EX3)

#### Section 3 : Model-based Object Detection in BEV Image
##### Ex.1 : Add a second model from a GitHub repo (ID_S3_EX1)
##### Ex.2 : Extract 3D bounding boxes from model response (ID_S3_EX2)

#### Section 4 : Performance Evaluation for Object Detection
##### Ex.1 : Compute intersection-over-union between labels and detections (ID_S4_EX1)
##### Ex.2 : Compute false-negatives and false-positives (ID_S4_EX2)
##### Ex.3 : Compute precision and recall (ID_S4_EX3)

### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?


### 4. Can you think of ways to improve your tracking results in the future?

