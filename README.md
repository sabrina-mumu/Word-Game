## About:
- This project contains the api for word game
- Download the [software](https://sqlitebrowser.org/dl/) to check the database operations if needed


## Steps:
### Install necessary plugins
- Install virtual environment in the command prompt  `pip install virtualenv`
- Make a virtual environment in the directory  `python -m venv .venv`      (Here the environment name is .venv)
- Activate the environment  
	- For Windows `.venv\Scripts\activate`
	- For Unix `source .venv/bin/activate`
 - Download and install the necessary files  `pip install -r requirements.txt`


 ### Run the server
 - Run command `uvicorn word_game_api:app --reload`
 - Go to this link `localhost:8000/upload` [post method]
 - UI can be load from here:  `localhost:8000/` & `localhost:8000/docs`
 - On Postman go to the body and the Key parameter has to be `file`.