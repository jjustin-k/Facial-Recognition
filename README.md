# Facial-Recognition
The motivation behind this project is that I noticed as I walked from class to class, I would not see people I know walk by me, this gave me the idea to look into how facial-recognition software works, and to see if I would ever be able to apply it in a way that would solve my problem. This specfic Facial recognition program uses pictures of people in a folder called known_people (folder must be in the same place as the script), if the person is not recognized, there is an option in the terminal to add this new person to the folder of known people. Like most facial-recognition algorithms, it is not 100% accurate, and will occasionly miss idenitfy similar looking people beacuse of the face_comparison tolerance.  This issue is improved for the most part if the indivdual being miss indetified manually adds there picture to known people.  For this program to work, you will have to download opencv, face_recognition, numpy, os, and glob.  In the future, I will be looking to expand on this project by adding voice transcription software to be able to tell the program what to do, instead of having to go to the terminal.
