from styx_msgs.msg import TrafficLight

import rospy
import cv2
import numpy as np
from keras import applications
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing import image as kimg

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        with CustomObjectScope({'relu6': applications.mobilenet.relu6,
            'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
            self.model = load_model('light_classification/models/model.MobileNet-3-classes.h5')
            self.model._make_predict_function()
        rospy.loginfo("TLClassifier: Finished to load model")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = cv2.resize(image, (224, 224))

        # Normalize
        x = kimg.img_to_array(image)
        x = np.expand_dims(x, axis=0).astype('float32') / 255

        return np.argmax(self.model.predict(x))
