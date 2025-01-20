import cv2
import numpy as np
import matplotlib.pyplot as plt


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


width = 368
height = 368
inWidth = width
inHeight = height

# Load the pre-trained pose detection model (ensure the correct path to .pb file)
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

thres = 0.2  # Confidence threshold


def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    # Prepare the image as input to the network
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    # Run the forward pass
    out = net.forward()
    
    # The output layer has 19 parts (the body parts)
    out = out[:, :19, :, :]  # The 19 body parts

    assert(len(BODY_PARTS) == out.shape[1])  # Ensure the output matches the number of body parts
    
    points = []
    
    # Loop through each body part
    for i in range(len(BODY_PARTS)):
        # Get the heatmap for the body part
        heatMap = out[0, i, :, :]
        
        # Get the confidence and the location of the part
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        
        # Scale the point to the original image size
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        
        # Append the point if the confidence is above the threshold
        points.append((int(x), int(y)) if conf > thres else None)

    # Draw the pose connections (lines between body parts)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        # If both points are detected, draw a line between them
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
    # Return the output frame with pose detections
    return frame


# Load an input image
input_image = cv2.imread('stand.jpg')

# Detect poses on the input image
output_image = poseDetector(input_image)

# Save the output image with poses
cv2.imwrite("Output-image.png", output_image)

# Show the result
cv2.imshow("Pose Detection", output_image)

# Wait until a key is pressed
cv2.waitKey(0)

# Destroy all windows after key press
cv2.destroyAllWindows()
