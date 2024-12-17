## Overview:
- This is a game where the AI throws a word to the user from the word pool, processes the user's response and gives a score based on the contextual similarity of the words
- The similarity is calculated using cosine similarity
- Once a proper response from the user is accepted, the given word (AI) is removed from the pool
- There are 3 levels of words based on the user score. Based on the score, the words will be merged
- Once the word pool is empty, it will reload all the words again and then there won't be any category for a given word based on the score
- When the pool is reloaded, the user won't be able to use the previous word pair a second time. For example, if the user has used the pair sky-star at stage 1, then he/she won't be able to use it again if the pool reloads
- The APIs are developed in such a manner that they can handle multiple users at the same time
- A database is also integrated to reduce the response time for already-used word pair


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
