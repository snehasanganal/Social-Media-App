<<<<<<< HEAD
# Social-Media-App
This is a social media application built using Django framework with HTML,CSS,Javascript used for the front-end of the application.
It includes features such as posting, liking posts, suggesting friends, following, updating profiles, downloading photoes, etc.<br><br>
**Demo Video**


https://github.com/snehasanganal/Social-Media-App/assets/116898126/32042118-25ed-4e1f-9834-3d9f99c73ecd

=======
<b> Steps to run the project <\b>
<b> 1. Creating a virtual Environment <\b>
i. Create a folder named <b>"project-env"<\b> or any other.
ii. Go to command prompt or vscode terminal and go inside the folder using <b>"cd project-env"<\b>
iii. Type <b>"python -m venv venv"<\b> (Here second venv is virtual environment name, you can keep any other name also eg. python -m venv myenv)
iv. Now a new folder named venv will be created inside project-env. 
<br><br>
<b> 2. Activating a virtual Environment <\b>
i. When you are in the project-env folder, type <b>venv\Scripts\activate<\b> (or myenv\Scripts\activate based on your virtual environment name)
<br><br>
<b> 3. Setting up the project <\b>
i. Create a folder named "social_book" inside the project-env folder
ii. Now clone this repository using following command -- <b>git clone "https://github.com/snehasanganal/Final-Year-Project.git" social_book <\b>
iii. Now you need to install multiple packages to run the project successfully. Ensure that when installing packages your virtual environment is activated.Run the following commands:
<b> "pip install numpy pandas torch spacy matplotlib seaborn scikit-learn tqdm transformers nltk requests"
 "python -m spacy download en_core_web_sm"
 "python -m nltk.downloader stopwords"<\b>
iv. Now create a folder named ai_models inside "core" folder of the social_book folder . Now move the saved models inside this <b>"ai_models"<\b> folder. The folder structure should be as below:
![image](https://github.com/user-attachments/assets/a050c0cf-919f-456d-afee-4509d76307bf)
<br><br>
<b> 3. Running the project <\b>
i. Enter the project folder by typing <b>"cd social_book"<\b>
ii. Run the project using the command <b> python manage.py runserver <\b>


When you want to exit use ctrl C and type deactivate to come out of virtual environment or else simply close command prompt/ terminal.
>>>>>>> a94a574 (Create README.md)
