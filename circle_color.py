import cv2
import numpy as np

def is_quadrilateral(approx):
    return len(approx) == 4 and cv2.isContourConvex(approx)

def detect_circles_in_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use the HoughCircles method to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)
    
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # Draw a rectangle corresponding to the center of the circle
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            
            # Get the color of the detected circle (average color around the circle)
            circle_roi = frame[y-r:y+r, x-r:x+r]
            avg_color = np.mean(circle_roi, axis=(0, 1)).astype(int)
            
            # Classify color based on BGR ranges
            # if avg_color[0] > 150 and avg_color[1] < 100 and avg_color[2] < 100:  # Red
            #     color_name = "Red"
            # elif avg_color[0] < 100 and avg_color[1] > 150 and avg_color[2] < 100:  # Green
            #     color_name = "Green"
            # elif avg_color[0] < 100 and avg_color[1] < 100 and avg_color[2] > 150:  # Blue
            #     color_name = "Blue"
            # else:
            #     color_name = "Unknown"
            
            # Display the color name near the circle
            color_text = f"Color: {avg_color}"
            cv2.putText(frame, color_text, (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def main():
    # Hardcoded video file path
    video_path = '/home/jammy/soham task/AI Assignment video.mp4'

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 700x700 pixels
        scale=0.9
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the red color range
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create a mask for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Check if the contour is a quadrilateral
            if is_quadrilateral(approx):
                x, y, w, h = cv2.boundingRect(approx)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)

                # Extract the quadrilateral region
                quadrilateral = frame[y:y+h, x:x+w]

                # Detect circles and process on the quadrilateral frame
                processed_frame = detect_circles_in_frame(quadrilateral)

                # Display the processed frame with detected circles
                cv2.imshow('Processed Frame with Circles', processed_frame)

        # Display the original frame with contours
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
