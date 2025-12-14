
# Social-Media-App
This is a social media application built using Django framework with HTML,CSS,Javascript used for the front-end of the application.
It includes features such as posting, liking posts, suggesting friends, following, updating profiles, downloading photoes, etc.<br><br>
It comes along with a sensitive content detecting feature where-in the user is nudged with alerts whenever he/she tries to post anything sensitive like their personal information, political or religious opinions or about their health conditions or sexuality etc. This is aimed at safeguarding the user from possible problems to privacy and security that could arise due to posting such sensitive content on social media. <br><br>
The model for the sensitive content detection is built using a combination of bert and lstm layers. A feature to mask personal information before posting is also provided to the user. <br><br>
**Demo Video**


https://github.com/user-attachments/assets/543102d7-2451-4958-8071-cc22a35dfda6



**Steps to run the project**

**1. Creating a virtual Environment**

i. Create a folder named **"project-env"** or any other.

ii. Go to command prompt or vscode terminal and go inside the folder using **"cd project-env"**

iii. Type **"python -m venv venv"** (Here second venv is virtual environment name, you can keep any other name also eg. python -m venv myenv)

iv. Now a new folder named venv will be created inside project-env. 




**2. Activating a virtual Environment**

i. When you are in the project-env folder, type **venv\Scripts\activate** (or myenv\Scripts\activate based on your virtual environment name)



**3. Setting up the project**

i. Create a folder named **"social_book"** inside the project-env folder

ii. Now clone this repository using following command -- **"  git clone "https://github.com/snehasanganal/Final-Year-Project.git" social_book**  "

iii. Now you need to install multiple packages to run the project successfully. Ensure that when installing packages your virtual environment is activated.Run the following commands:

**"pip install numpy pandas torch spacy matplotlib seaborn scikit-learn tqdm transformers nltk requests"**

**"python -m spacy download en_core_web_sm"**
 
**"python -m nltk.downloader stopwords"**

iv. Now create a folder named ai_models inside "core" folder of the social_book folder . Now move the saved models inside this **"ai_models"** folder. The folder structure should be as below:

![image](https://github.com/user-attachments/assets/a050c0cf-919f-456d-afee-4509d76307bf)



**4. Running the project**

i. Enter the project folder by typing **"cd social_book"**

ii. Run the project using the command **python manage.py runserver**


When you want to exit use ctrl C and type deactivate to come out of virtual environment or else simply close command prompt/ terminal.

