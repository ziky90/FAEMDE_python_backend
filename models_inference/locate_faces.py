
import cv2


CASCADE_FILE_PATH = 'haarcascade_frontalface_default.xml'


def locate_faces(input_image):
    """
    Locate faces in the original image.
    :param input_image: Array representing the input image.
    :type input_image: np.ndarray
    :return: (x, y, w, h)
    :rtype: (int, int, int, int)
    """
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    return faces
