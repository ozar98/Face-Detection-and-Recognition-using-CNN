import cv2
import face_recognition
import numpy as np


video_capture=cv2.VideoCapture(0)

known_face_encodings=[]
known_face_names=[]

image_g=face_recognition.load_image_file(r'''D:\ozar files\ozar\ozar\UCA\Junior Year\image processing\project\photo.jpg''')
image_g_encoding=face_recognition.face_encodings(image_g)[0]

known_face_encodings=[image_g_encoding, img_g_encoding]
known_face_names=["Ozar"]



face_locations=[]
face_encodings=[]

face_names=[]
process_this_frame=True
i=0

while True:
    # Grab a single frame of video
    ret,frame=video_capture.read()
     # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame=cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame=small_frame[:,:,::-1]
    if process_this_frame:
         # Find all the faces and face encodings in the current frame of video
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        name_list=[]
        face_name=[]
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches=face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.5)
              # Or instead, use the known face with the smallest distance to the new face
            face_distances=face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index=np.argmin(face_distances)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]
                face_names.append(name)
    i=i+1
    if i==5:
        current_name=name
    if len(face_names)==0:
        i=0
    process_this_frame=not process_this_frame
    # Display the results
    for (top,right, bottom, left), name in zip(face_locations,face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top=top*2
        right*=2
        left*=2
        bottom*=2
        # Draw a box around the face
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)

        # Draw a label with a name below the face
        cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),cv2.FILLED)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6,bottom-6), font,1.0,(255,255,255),1)
    # Display the resulting image
    cv2.imshow('video',frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()









