import face_recognition
import cv2
import numpy as np
import os
import glob


def new_person(path, image):
    response = input("New Person Detected \nWould you like to add them to known people? y/n  ").lower()
    if response == 'y':
        name = input("What is there name?  ")
        cv2.imwrite(os.path.join(path, name + ".jpg"), image)
    print("Restarting Facial Recognition Software...")
    return 


def recognition():
    cap = cv2.VideoCapture(0)
   
    known_face_encodings = []
    known_face_names = []
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'known_people\\')

    list_of_files = [f for f in glob.iglob(path+'*.jpg')]
    names = list_of_files.copy()


    for i in range(len(list_of_files)):

        img_load = face_recognition.load_image_file(list_of_files[i])
        img_encoded = face_recognition.face_encodings(img_load)[0]
    
        known_face_encodings.append(img_encoded)

        names[i] = names[i].replace(path, "")
        names[i] = names[i].replace(".jpg", "")
        
        known_face_names.append(names[i])


    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        name =""

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
            
            
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.58)
                name = "?"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                best_match = np.argmin(face_distances)
                
                if matches[best_match]:
                    name = known_face_names[best_match]
                face_names.append(name)
        
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if name == "?":
                result, image = cap.read(frame)
                cv2.destroyAllWindows()
                cap.release()
                new_person(path, image)
                recognition()
                return
                
                
        cv2.imshow('vid',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

    cap.release()
    cv2.destroyAllWindows()
    return

recognition()
