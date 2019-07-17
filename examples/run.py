import pyfreenect2
import cv2
import scipy.misc
import signal
import numpy as np

# 'path to yolo config file'
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.cfg
CONFIG = './yolov3.cfg'

# 'path to text file containing class names'
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
CLASSES = './yolov3.txt'

# 'path to yolo pre-trained weights'
# wget https://pjreddie.com/media/files/yolov3.weights
WEIGHTS = './yolov3.weights'



import os

print(os.path.exists(CLASSES))
print(os.path.exists(CONFIG))
print(os.path.exists(WEIGHTS))




# read class names from text file
classes = None
with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.4

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))



# function to get the output layer names
# in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])+" "+str(round(confidence,3))
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y, x_plus_w-x, y_plus_h-y), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




def processImage(image):
    Width = image.shape[1]
    Height = image.shape[0]

    # read pre-trained model and config file
    net = cv2.dnn.readNet(WEIGHTS, CONFIG)

    # create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, class_ids[i], confidences[i], int(x), int(y), int(x + w), int(y + h))

    # display output image
    out_image_name = "object detection"# + str(index)
    cv2.imshow(out_image_name, image)
    # wait until any key is pressed
    # cv2.waitKey()
    # save output image to disk
    #cv2.imwrite("out/" + out_image_name + ".jpg", image)

"""
cap = cv2.VideoCapture("new-york-city-streets.jpg")

index = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    processImage(frame, index)
    index = index + 1

# release resources
cv2.destroyAllWindows()

"""
serialNumber = pyfreenect2.getDefaultDeviceSerialNumber()
kinect = pyfreenect2.Freenect2Device(serialNumber)

# Set up signal handler
shutdown = False
def sigint_handler(signum, frame):
	print( "Got SIGINT, shutting down...")
	global shutdown
	shutdown = True
signal.signal(signal.SIGINT, sigint_handler)

# Set up frame listener
frameListener = pyfreenect2.SyncMultiFrameListener(pyfreenect2.Frame.COLOR,
	pyfreenect2.Frame.IR,
	pyfreenect2.Frame.DEPTH)

print( frameListener)
kinect.setColorFrameListener(frameListener)
kinect.setIrAndDepthFrameListener(frameListener)
kinect.setDeepConfiguration(pyfreenect2.DeepConfig(MinDepth=.5,MaxDepth=10,EnableBilateralFilter=True,EnableEdgeAwareFilter=True))

# Start recording
kinect.start()

# Print useful info
print( "Kinect serial: %s" % kinect.serial_number)
print( "Kinect firmware: %s" % kinect.firmware_version)

# What's a registration?
print( kinect.ir_camera_params)

registration = pyfreenect2.Registration(kinect.ir_camera_params, kinect.color_camera_params)
#registration = pyfreenect2.Registration(kinect.color_camera_params, kinect.ir_camera_params)
#registration = pyfreenect2.Registration()

# Initialize OpenCV stuff
#cv2.namedWindow("RGB")
# cv2.namedWindow("IR")
cv2.namedWindow("Depth")

# Main loop

while not shutdown:
    frames = frameListener.waitForNewFrame()
    rgbFrame = frames.getFrame(pyfreenect2.Frame.COLOR)
    #irFrame = frames.getFrame(pyfreenect2.Frame.IR)
    depthFrame = frames.getFrame(pyfreenect2.Frame.DEPTH)
    rgb_frame = rgbFrame.getRGBData()
    bgr_frame = rgb_frame.copy()
    bgr_frame[:,:,0] = rgb_frame[:,:,2]
    bgr_frame[:,:,2] = rgb_frame[:,:,0]

    depth_frame = depthFrame.getDepthData()
    #depth_frame = frames.getFrame(pyfreenect2.Frame.DEPTH).getData()
    print(depth_frame)
    #bgr_frame_resize = scipy.misc.imresize(bgr_frame, size = .5)
    depth_frame_resize = scipy.misc.imresize(depth_frame, size = .5)

    bgr_frame_new=cv2.cvtColor(bgr_frame,cv2.COLOR_BGRA2BGR)



    processImage(bgr_frame_new)
    # TODO Display the frames w/ OpenCV
    #cv2.imshow("RGB", bgr_frame_resize)
    cv2.imshow("Depth", depth_frame)
    cv2.waitKey(20)
    frameListener.release(frames)

kinect.stop()
kinect.close()


"""
pf = pyfreenect2.PyFreeNect2()

cv2.startWindowThread()
cv2.namedWindow("RGB")
index=0
while True:
    extracted_frame = pf.get_new_frame(get_BGR = True)
    extracted_frame=cv2.cvtColor(extracted_frame, cv2.COLOR_BGRA2BGR)

    resized = scipy.misc.imresize(extracted_frame.BGR, size = 0.5)

    processImage(resized, index)
    index = index + 1
    cv2.imshow("RGB", resized)
    #cv2.waitKey(20)


"""