from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, root_validator
from fastapi.middleware.cors import CORSMiddleware
from word_game_setup import WordGame, MultipleWordsError, InvalidWordError, SameInputError, AIWordError
from speech_to_text import Transcriber, CustomException
from db_setup import DuplicateUserError
from sqlalchemy.exc import SQLAlchemyError
import re

# Initialize FastAPI app
app = FastAPI(title="Word Game API")

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the word game instance
word_game = WordGame()


class MissingFieldError(Exception):
    """Custom exception raised when a required field is missing in the request."""

    def __init__(self, message="Request body is empty."):
        """Initialize with a default or custom error message.

        Args:
            message (str): Error message to describe the missing field.
        """
        self.message = message
        super().__init__(self.message)


class Initgame(BaseModel):
    """Model to validate the request body for initializing a game."""
    user_id: str


class WordPair(BaseModel):
    """Model to validate the request body for submitting a word pair in the game."""
    user_id: str
    ai_word: str
    human_word: str | None = None
    audio_url: str | None = None
    incoming_score: int

    @root_validator(pre=True)
    def check_empty(cls, values):
        """Validator to ensure the request body is not empty.

        Args:
            values (dict): Dictionary of values passed to the request.

        Raises:
            MissingFieldError: If all values are missing.

        Returns:
            dict: The validated values.
        """
        if not any(values.values()):
            raise MissingFieldError("Request body cannot be empty.")
        return values


class Endgame(BaseModel):
    """Model to validate the request body for ending a game."""
    user_id: str


@app.exception_handler(MissingFieldError)
async def missing_field_error_handler(request: Request, exc: MissingFieldError):
    """Exception handler for MissingFieldError.

    Args:
        request (Request): The incoming request object.
        exc (MissingFieldError): The exception instance.

    Returns:
        JSONResponse: JSON response indicating the missing input field error.
    """
    return JSONResponse(
        status_code=200,
        content={"status": 0, "error_type": "missing_input_field", "detail": exc.message},
    )


@app.post("/game_init")
async def start_game(init_game: Initgame):
    """Endpoint to initialize the word game.

    Args:
        init_game (Initgame): Contains the user ID for initializing the game.

    Returns:
        dict: A dictionary containing the status and the first AI word or an error message.
    """
    user_id = init_game.user_id
    print(f"Received request to start game for user: {user_id}")

    try:
        # Check if the user already has an active game initialized
        word_game.user_authentication(user_id)
        
        # Proceed with game initialization since user is not a duplicate
        word_game.initialize_game(user_id)
        
        # Get the first AI word to send to the user
        first_ai_word = word_game.throw_word_to_user(user_id)
        
        # Save the thrown word to the database
        word_game.save_thrown_word_to_db(user_id, first_ai_word)
        
        # If no word is available, inform the client
        if not first_ai_word:
            return {"status": 0, "message": "No words available in the pool."}

        # Return success response with the first AI word
        return {"status": 1, "first_ai_word": first_ai_word}
    
    except DuplicateUserError as e:
        # Handle duplicate user error by returning an appropriate message
        return {
            "status": 0,
            "error_type": "duplicate_user",
            "message": str(e)
        }
    
    except Exception as e:
        # Handle any unexpected exceptions
        return {
            "status": 0,
            "error_type": "initialization_error",
            "message": str(e)
        }


@app.post("/score_and_next_word")
async def play_word_game(word_pair: WordPair):
    """Endpoint to play the word game and get the next word.

    Args:
        word_pair (WordPair): Contains the AI word, human word, audio URL, and incoming score.

    Returns:
        dict: A dictionary containing the status, similarity score, updated score, and next AI word or error details.
    """
    user_id = word_pair.user_id
    ai_word = word_pair.ai_word
    human_word = word_pair.human_word
    audio_url = word_pair.audio_url
    incoming_score = word_pair.incoming_score

    if human_word:
        try:
            if " " in human_word:
                raise MultipleWordsError("The input contains multiple words.")
            
            if '"' in human_word:
                human_word = human_word.replace('"', "")

            # Check if the human word contains only special characters
            if not any(char.isalnum() for char in human_word):
                raise InvalidWordError("The input word contains only special characters.")

            # Check if the human word contains special characters along with alphabets
            if any(char.isalpha() for char in human_word):
                # Remove special characters from the word
                cleaned_word = re.sub(r'[^a-zA-Z]', '', human_word)
                
                # If cleaned word is not empty, assign it to human_word
                human_word = cleaned_word

        except MultipleWordsError as e:
            return {"status": 0, "error_type": "invalid_input", "detail": str(e)}
        
        except InvalidWordError as e:
            return {"status": 0, "error_type": "invalid_input", "detail": str(e)}

    elif audio_url:
        try:
            transcriber = Transcriber(audio_url)
            transcribed_text = transcriber.transcribe()

            if transcribed_text == "[Unintelligible]":
                raise CustomException("The audio was unclear. Please try again.", "speech_to_text_error")

            if not transcribed_text:
                raise CustomException("Failed to transcribe the audio.", "speech_to_text_error")
            
            if " " in transcribed_text:
                raise MultipleWordsError("The transcribed audio contains multiple words.")

            human_word = transcribed_text
        
        except CustomException as e:
            return {"status": 0, "error_type": e.error_type, "detail": str(e)}
        
        except Exception as e:
            return {"status": 0, "error_type": "speech_to_text_error", "detail": str(e)}

    if not human_word:
        return {
            "status": 0,
            "error_type": "invalid_input",
            "detail": "No valid word was provided or transcribed."
        }

    try:
        result = word_game.get_similarity_score_with_next_word(user_id, ai_word, human_word, score=incoming_score)

        if "error" in result:
            return {
                "status": 0, 
                "error_type": "invalid_input",
                "details": result["error"]
            }

        return {
            "status": 1,
            "data": {
                "similarity_score": result['similarity_score'],
                "updated_score": result['updated_score'],
                "next_ai_word": result['next_ai_word']
            }
        }

    except SQLAlchemyError as e:
        return {
            "status": 0,
            "error_type": "database_error",
            "detail": f"An error occurred while accessing the database: {str(e)}"
        }
    except SameInputError as e:
        return {
            "status": 0,
            "error_type": "duplicate_pair",
            "detail": "Please try a different word"
        }
    except AIWordError as e:
        return {
            "status": 0,
            "error_type": "ai_word_mismatch",
            "detail": "This word was not thrown by AI"
        }


@app.post("/end_game")
async def end_game(end_game: Endgame):
    """Endpoint to terminate the current game session.

    Args:
        end_game (Endgame): Contains the user ID of the game session to end.

    Returns:
        dict: A dictionary indicating successful termination of the game.
    """
    user_id = end_game.user_id
    word_game.clear_user_data(user_id) 
    word_game.clear_user_entries(user_id)
    return {"status": 1, "message": "Terminate the game."}
