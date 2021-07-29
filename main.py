from flask import Flask, render_template, Response
import cv2
import time
import numpy as np

"""
    Took idea for this from this article:: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
    
    Motion HeatMap: https://towardsdatascience.com/build-a-motion-heatmap-videousing-opencv-with-python-fd806e8a2340

"""

app = Flask(__name__)
camera = cv2.VideoCapture(0) # capture the video feed from the camera.

# Background subtraction instance ...
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()

def gen_frame():
    """
        This method returns frames captured by the camera as response in chunks.
        The function yields with the retrieved frame formatted as a response chunk with a
        content type of 'image/jpeg'.
    """
    t1 = time.time();

    # read the First Frame -> Initial frame
    success, initFrame = camera.read()

    if not success:
        # try reading from the camera feed again ...
        success, initFrame = camera.read()
        try:
            assert success == True
        except:
            # If the first frame could not be read from the camera feed -> then exit from the app
            print( 'The camera feed could not be retrieved. Please check your camera / device.' )
            exit(0)

    # the dimesnions for the captured frame from the camera feed
    height, width = initFrame.shape[:2]
    # initialized background for background subtraction...
    backgroundFrame = np.zeros( (height, width), np.uint8 )

    while(True):
        success, frame = camera.read()
        t2 = time.time()

        """
            success --> A boolean which states if Python can read the VideoCapture() object
            frame   --> A numpy array which gets the first image that the video captures
            .read() --> read frame correctly 
        """

        if not success:
            break # cannot render the image from the camera feed

        else:

            if (t2 - t1) > 10:
                # For every elapsed 10 seconds, re-initialize the rendered camera feed
                t1 = time.time()
                initFrame = frame
                height, width = initFrame.shape[:2]
                backgroundFrame = np.zeros( (height, width), np.uint8 ) # re-initialized background frame

            # Find the motion difference between two consecutive frames ...
            filtFrame = backgroundSubtractor.apply( frame ) # remove the background
            threshold = 2
            maxValue = 2
            retVal, threshFrame = cv2.threshold( filtFrame, threshold, maxValue, cv2.THRESH_BINARY) # applying threshold to remove nominal differences

            # add to the threshold frame to the background frame
            backgroundFrame = cv2.add( backgroundFrame, threshFrame )
            
            # applying colormap for generating the heatmap of the captured motion
            motionHeatmap = cv2.applyColorMap( backgroundFrame, cv2.COLORMAP_HOT )
            
            # final rendered frame should be a combination of the motion Heat-map frame and the captured frame from the camera feed
            motionFrame = cv2.addWeighted( frame, 1.25, motionHeatmap, 0.65, 0 )

            # read the incoming frames from the camera feed & serve them into the web browser
            ret, buffer = cv2.imencode('.jpg', motionFrame)
            renderFrame = buffer.tobytes()
            yield( bytes("--frame\r\nÇontent-Type: image/jpeg\r\n\r\n", "utf-8") + renderFrame + bytes("\r\n", "utf-8") )

            """
                cv2.imencode() ---> method to encode the image format into streaming data & assign
                                    it to memory cache. This is used mainly to compress image data format to facilitate network transmission
                
                yield          ---> lets the execution continue and keeps on generatingframe until alive
            """

@app.route('/')
def index():
    # This decorator method tells the app that whenever a user visits the URL with 
    # the specified 'route', executes the index() method 
    return render_template('index.html');


# define the app route for the Video feed to be rendered on the webpage
@app.route('/camera_capture')
def camera_capture():
    return Response( gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame' )


if __name__ == "__main__":
    app.run(debug = True)