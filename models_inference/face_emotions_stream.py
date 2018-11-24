import cv2
import numpy as np
from EmoPy.src.fermodel import FERModel


class FERStreamingModel(FERModel):

    def predict(self, image):
        """
        Predicts discrete emotion for given image.

        :param image: Image represented as a numpy array.
        :type image: np.ndarray
        :return: {emotion, score in percents}
        :rtype: {str, float}
        """
        gray_image = image
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, self.target_dimensions,
                                   interpolation=cv2.INTER_LINEAR)
        final_image = np.array([np.array([resized_image]).reshape(
            list(self.target_dimensions)+[self.channels])])
        prediction = self.model.predict(final_image)[0]
        self._print_prediction(prediction)

        normalized_prediction = [x / sum(prediction) for x in prediction]
        result = {}
        for emotion in self.emotion_map.keys():
            # conversion to percents
            result[emotion] = normalized_prediction[
                                  self.emotion_map[emotion]] * 100
        return result
