import cv2
import os
import cv2.aruco as aruco
import numpy as np

cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

s_mask = cv2.imread('FaceSubstitution/images/wolverine.png', cv2.IMREAD_UNCHANGED)

#make mask of where the transparent bits are
trans_mask = s_mask[:,:,3] == 0

#replace areas of transparency with white and not transparent
s_mask[trans_mask] = [255, 255, 255, 0]

#video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture(1)

# Only IMAGES
# Mask
# https://stackoverflow.com/questions/60937583/is-it-possible-to-paste-an-image-on-top-of-another-in-opencv

cascPathface = os.path.dirname("")


#eye check
def put_mask(mask, frame, x, y, w, h):
    face_width = w
    face_height = h

    if y < 0:
        return frame

    hat_width = face_width + 1
    hat_height = int(1.0 * face_height) + 1

    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    mask = cv2.resize(mask, (hat_width,hat_height))
    #mask = cv2.resize(mask, (w, h))

    #cv2.line(frame,(x,y+h//2),(x+w,y+h//2),(255,0,255),3)

    # Por cada cara tenemos los ojos, la cosa sería mirar
    # a) Si hay dos ojos sacar la altura a partir de los 2 he incluso deformaciones para la profundidad
    # b) SI hay un solo ojo detectado, dibujas la mascara a esa altura.
    # print(eyes)

    # https://stackoverflow.com/questions/56002672/display-an-image-over-another-image-at-a-particular-co-ordinates-in-opencv
    #frame = cv2.add(src1=frame,src2=frame,mask=mask)



    # replace values at coordinates (100, 100) to (399, 399) of img3 with region of img2
    #frame[ y:y + h,x:x + w, :] = mask[:,:,:]

    white_img =  np.full((frame.shape[0], frame.shape[1], 4), 0, dtype=np.uint8)



    white_img[y : y + hat_height, x: x + hat_width,:] = mask[:,:,:]
    #white_img[y: y + h, x: x + w,:] = mask[:,:,:]


    mask = white_img
    alpha = mask[:, :, 3]
    alpha = cv2.merge([alpha, alpha, alpha])
    # extract bgr channels from foreground image
    front = mask[:, :, 0:3]
    # blend the two images using the alpha channel as controlling mask
    result = np.where(alpha == (0, 0, 0), frame, front)

    return result

# No eye check
def put_mask_eyes(mask, frame, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int(0.70 * face_height) + 1

    mask = cv2.resize(mask, (hat_width, hat_height))

    #cv2.line(frame,(x,y+h//2),(x+w,y+h//2),(255,0,255),3)

    # Por cada cara tenemos los ojos, la cosa sería mirar
    # a) Si hay dos ojos sacar la altura a partir de los 2 he incluso deformaciones para la profundidad
    # b) SI hay un solo ojo detectado, dibujas la mascara a esa altura.
    # print(eyes)

    # Manually writting
    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if mask[i][j][k] < 255:
                    frame[y + i - int(0.05 * face_height)][x + j][k] = mask[i][j][k]
    return frame

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)


    b_channel, g_channel, r_channel = cv2.split(frame)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)*100  # creating a dummy alpha channel image.

    #frame = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    for (x,y,w,h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        # Put the mask on the face



        faceROI = frame[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(faceROI)

        ##############

        frame = put_mask(s_mask, frame, x, y - 60, w, h)
        ########################




        #frame = put_mask(s_mask, frame, x, y, w, h)

        ##############
        #type_array = np.array(eyes)
        #if(type(eyes) == type(type_array)):
        #    if (eyes.shape[0] == 2):
        #        a = (eyes[0][0] + x + eyes[0][2]//2,eyes[0][1] + y + eyes[0][3]//2)
        #        b = (eyes[1][0] + x + eyes[1][2]//2,eyes[1][1] + y + eyes[1][3]//2)
                #cv2.line(frame, a,b, (255, 0, 255),3)

        #        frame = put_mask_eyes(s_mask, frame, x, y, w, h)
        #    else:
        #        frame = put_mask(s_mask, frame, x, y, w, h)
        #else:
        #    frame = put_mask(s_mask, frame, x, y, w, h)

        ########################


        #for (x2, y2, w2, h2) in eyes:

        #    eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)

            #radius = int(round((w2 + h2) * 0.25))
            #frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)



        # Display the resulting frame

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()