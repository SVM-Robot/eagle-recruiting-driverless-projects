import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('video.mp4')

# Create feature detector (e.g., SIFT)
feature_detector = cv2.SIFT_create(nfeatures = 1000)
# Create a FLANN-based matcher
matcher = cv2.FlannBasedMatcher()

# Initialize previous frame and keypoints
prev_frame = None
prev_keypoints = None
prev_descriptors = None
row_list = []
scale_factor = 0.5
f = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute descriptors in the frame
    keypoints, descriptors = feature_detector.detectAndCompute(gray, None)

    if prev_frame is not None:
        # Match keypoints between current and previous frames
        matches = matcher.knnMatch(descriptors, prev_descriptors, k=2)
        
        # Apply ratio test to filter good matches
        good_matches = []
        dist1 = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                dist1.append(m.distance)
                good_matches.append(m)
        
        # Draw matches on the current frame
                
        #good_matches = sorted(good_matches, key=lambda x: x.distance)
        frame_with_matches = cv2.drawMatches(frame, keypoints, prev_frame, prev_keypoints, good_matches, None)
        cv2.imshow('Frame with Matches', cv2.resize(frame_with_matches, (1400, 600)))

        def avg_std(_good_matches):
            pixel_diffs = []
            angles = []
            for match in _good_matches:
                # Get keypoints indices
                idx1 = match.queryIdx
                idx2 = match.trainIdx
                # Get keypoints positions
                pt1 = keypoints[idx1].pt
                pt2 = prev_keypoints[idx2].pt
                # Calculate pixel difference
                diff = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                pixel_diffs.append(diff)
                # Calculate vector between keypoints
                vector = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
                # Calculate angle (in radians) between the vector and the x-axis
                angle = np.arctan2(vector[1], vector[0])
                angles.append(angle)    

            # Average pixel difference and std
            pixel_diffs = np.array(pixel_diffs)
            avg_pixel = np.mean(pixel_diffs)
            std_pixel = np.std(pixel_diffs)
            # Convert angles to numpy array
            angles = np.array(angles)
            # Convert angles to positive values (to handle negative angles)
            angles = (angles + 2 * np.pi) % (2 * np.pi)
            # Calculate average angle
            avg_angle = np.mean(angles)
            # Calculate standard deviation of angles
            std_angle = np.std(angles)

            def print_results1():
                print("Frame %s" %f)
                print("Average Pixel Difference:", avg_pixel)
                print("Standard Deviation of Pixel Differences:", std_pixel)
                print("Average Angle between Matched Keypoints (in radians):", avg_angle)
                print("Standard Deviation of Angles between Matched Keypoints (in radians):", std_angle)
            #print_results1()

            # Write results to csv file
            row_list.append([f, avg_pixel, std_pixel, avg_angle, std_angle])
            with open('results1.csv', 'w', newline='') as file:
                writer = csv.writer(file) 
                field = ["frame", "avg_pxl diff", "std_pxl", "avg_angle_diff","std_angle"]
                writer.writerow(field)               
                for row in row_list:
                    writer.writerow(row)                

        avg_std(good_matches)
        
    # Store current frame keypoints and descriptors for the next iteration
    prev_frame = frame.copy()
    prev_keypoints = keypoints
    prev_descriptors = descriptors    
    f += 1

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
