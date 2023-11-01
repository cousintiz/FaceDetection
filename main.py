import cv2, sys
from deepface import DeepFace


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0) 


def setCam():
    """
    Set the camera to be used with the os index of the cam.
    """
    global cap
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("Error acessing Webcam")


def isPersonFace():
    """
    Determines whether is or not a person in the frame.
    Returns: confidence level of detection and the frame analized
    """
    try:
        _, _frame = cap.read()
        _result = DeepFace.extract_faces(_frame, enforce_detection = False)
        result = _result[0]["confidence"]
    except:
        print(f"Error:{result}")
    return result, _frame


def especifyFace(frame):
    """
    Recieves the rgb frame and converts to grayscale for analysis
    Args: frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def showCam(frame, conf):
    """
    Args: confidence level of detection and the frame analized
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, conf, (45, 45), font, 3,
                (0, 0, 255), 2, cv2.LINE_4, )
    cv2.imshow('Face Detection App:', frame)


def cleanUp():
    """
    clean up the environment, close camera, cv2 windows, and app.
    """
    cap.release()
    cv2.destroyAllWindows() 
    sys.exit()


if __name__ == "__main__":
    setCam()
    while True:
        try:
            conf, frame = isPersonFace()
            print(f"confidence percentage{conf}") 
            especifyFace(frame)
            showCam(frame, str(conf))
            cv2.waitKey(2)
        except KeyboardInterrupt:
            break

    cleanUp()






