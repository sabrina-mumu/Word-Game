import random
import torch
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from db_setup import GameResultDB


# Database setup
db = GameResultDB()  # Create an instance of the GameResultDB
db.create_table()
db_session = db.get_session()  # Get the session


class MultipleWordsError(Exception):
    """Exception raised when input contains multiple words instead of a single word."""

    def __init__(self, message="Input contains multiple words. Please provide only a single word."):
        """Initialize with a default or custom error message.

        Args:
            message (str): Error message describing the multiple words error.
        """
        self.message = message
        super().__init__(self.message)


class InvalidWordError(Exception):
    """Exception raised when the input word contains only special characters."""

    def __init__(self, message="The input word contains only special characters."):
        """Initialize with a default or custom error message.

        Args:
            message (str): Error message describing the invalid word error.
        """
        self.message = message
        super().__init__(self.message)


class SameInputError(Exception):
    """Exception raised when the same word is used again in the game."""

    def __init__(self, message="Please try a different word"):
        """Initialize with a default or custom error message.

        Args:
            message (str): Error message describing the duplicate input error.
        """
        self.message = message
        super().__init__(self.message)


class AIWordError(Exception):
    """Exception raised when the AI word does not match the last thrown word."""

    def __init__(self, message="This word was not thrown by AI"):
        """Initialize with a default or custom error message.

        Args:
            message (str): Error message describing the AI word mismatch.
        """
        self.message = message
        super().__init__(self.message)


class WordGame:
    """Main class for the word game, handling word selection, scoring, and database interactions."""

    def __init__(self, word_csv_path='words.csv'):
        """Initialize the WordGame instance with preloaded words and embeddings.

        Args:
            word_csv_path (str): Path to the CSV file containing words for different levels.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        # Load words from the CSV file, separated by levels
        self.level1_words, self.level2_words, self.level3_words = self.load_words_from_csv(word_csv_path)
        self.flag = 0
        self.user_data = {}
        self.updated_score = 0  # Track the total score

    def initialize_user_word_pool(self, user_id):
        """Initialize or reset the word pool and used words for the specified user.

        Args:
            user_id (str): The ID of the user to initialize.
        """
        self.user_data[user_id] = {
            'word_pool': self.level1_words[:],  # Start with level 1 words
            'level2_merged': False,
            'level3_merged': False,
            'current_level': 1,
            'last_thrown_word': None
        }

    def get_user_word_pool(self, user_id):
        """Get the current word pool for the specified user.

        Args:
            user_id (str): The ID of the user whose word pool to retrieve.

        Returns:
            list: The current word pool for the user.
        """
        if user_id not in self.user_data:
            self.initialize_user_word_pool(user_id)
        return self.user_data[user_id]['word_pool']

    def initialize_game(self, user_id):
        """Initialize the game for the user by saving the game status.

        Args:
            user_id (str): The ID of the user to initialize the game for.

        Raises:
            Exception: If there is an error in initializing the game in the database.
        """
        try:
            db.save_game_status(user_id)
            print(f"Game initialized for user {user_id}.")
        except Exception as e:
            print(f"Error initializing game for user {user_id}: {e}")
            raise Exception(f"Failed to initialize game for user {user_id}. Error: {e}")

    def user_authentication(self, user_id):
        """Authenticate the user by checking if they have an active game status.

        Args:
            user_id (str): The ID of the user to authenticate.

        Returns:
            bool: True if user does not exist, else raises a DuplicateUserError.
        """
        return db.authenticate_user(user_id)

    def load_words_from_csv(self, csv_path):
        """Load words from a CSV file for each level.

        Args:
            csv_path (str): Path to the CSV file containing the words.

        Returns:
            tuple: Three lists containing level1, level2, and level3 words respectively.
        """
        level1_words = []
        level2_words = []
        level3_words = []

        try:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                first_line = csvfile.readline().strip()
                delimiter = '\t' if '\t' in first_line else ',' if ',' in first_line else None

                csvfile.seek(0)
                reader = csv.DictReader(csvfile, delimiter=delimiter)

                for row in reader:
                    if row['level1']:
                        level1_words.append(row['level1'].strip())
                    if row['level2']:
                        level2_words.append(row['level2'].strip())
                    if row['level3']:
                        level3_words.append(row['level3'].strip())
            return level1_words, level2_words, level3_words
        except FileNotFoundError:
            print(f"Error: The file {csv_path} was not found.")
            return [], [], []

    def expand_word_pool(self, user_id, score):
        """Update the user's word pool based on their current score.

        Args:
            user_id (str): The ID of the user whose word pool to expand.
            score (int): The current score of the user.
        """
        user_info = self.user_data.get(user_id)

        # Add level 2 words if score is between 100 and 200 and not yet merged
        if 100 <= score < 200 and not user_info['level2_merged']:
            user_info['word_pool'] += self.level2_words
            user_info['level2_merged'] = True
            user_info['current_level'] = 2
        
        # Add level 3 words if score is 200 or more and not yet merged
        elif score >= 200 and not user_info['level3_merged']:
            user_info['word_pool'] += self.level3_words
            user_info['level3_merged'] = True
            user_info['current_level'] = 3

    def reload_word_pool(self, user_id):
        """Reload the user's word pool and reset used words when all levels are used.

        Args:
            user_id (str): The ID of the user whose word pool to reload.
        """
        user_info = self.user_data.get(user_id)

        if user_info['current_level'] == 1:
            user_info['word_pool'] = self.level1_words[:]
            user_info['current_level'] = 2
        elif user_info['current_level'] == 2:
            user_info['word_pool'] = self.level2_words[:]
            user_info['current_level'] = 3
        elif user_info['current_level'] == 3:
            user_info['word_pool'] = self.level3_words[:]
            user_info['current_level'] = 1

    def check_round(self, user_id):
        """Check the current round of the game for the user.

        Args:
            user_id (str): The ID of the user whose round to check.

        Returns:
            int: Flag indicating if it's the first round (0) or a subsequent round (1).
        """
        round = self.check_game_status(user_id)
        print("ROUND NO: ", round)
        if round > 1:
            self.flag = 1
        return self.flag

    def save_thrown_word_to_db(self, user_id, thrown_word):
        """Save the thrown word to the database.

        Args:
            user_id (str): The ID of the user for whom to save the word.
            thrown_word (str): The word thrown by the AI.

        Returns:
            bool: True if word is saved successfully, False otherwise.
        """
        return db.save_thrown_word(user_id, thrown_word)

    def get_word_embedding(self, word):
        """Get the embedding of a word using the SentenceTransformer model.

        Args:
            word (str): The word to be embedded.

        Returns:
            numpy.ndarray: The word's embedding vector.
        """
        return self.model.encode([word], device=self.device)

    def calculate_similarity(self, word1, word2):
        """Calculate the cosine similarity between two words.

        Args:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            float: The cosine similarity score.
        """
        vec1 = self.get_word_embedding(word1)
        vec2 = self.get_word_embedding(word2)
        return cosine_similarity(vec1, vec2)[0][0]

    def throw_word_to_user(self, user_id):
        """Provide a word to the user, excluding those already used.

        Args:
            user_id (str): The ID of the user.

        Returns:
            str or None: The word thrown to the user, or None if no words are available.
        """
        word_pool = self.get_user_word_pool(user_id)
        used_words = self.get_used_words_by_user_id(user_id)
        
        word_pool_set = set(word_pool)
        used_words_set = set(used_words)

        available_words = list(word_pool_set - used_words_set)

        if not available_words:
            self.increment_round_count(user_id)
            self.clear_used_words_by_user(user_id)
            self.reload_word_pool(user_id)
            word_pool = self.get_user_word_pool(user_id)
            used_words = set(self.get_used_words_by_user_id(user_id))
            available_words = [word for word in self.get_user_word_pool(user_id) if word not in used_words]
        
        if not available_words:
            return None

        thrown_word = random.choice(available_words)
        self.user_data[user_id]['last_thrown_word'] = thrown_word
        return thrown_word

    def save_game_result(self, ai_word, human_word, score):
        """Save the game result to the database using the GameResultDB class.

        Args:
            ai_word (str): The AI's word.
            human_word (str): The human player's word.
            score (int): The similarity score between the AI and human words.

        Returns:
            int: The primary key ID of the new game result entry in the database.
        """
        return db.save_game_result(ai_word, human_word, score)

    def save_used_word(self, user_id, ai_word):
        """Save a used word for a user to the database.

        Args:
            user_id (str): The ID of the user.
            ai_word (str): The word used in the game.

        Returns:
            bool: True if word is saved successfully, False otherwise.
        """
        return db.save_used_word(user_id, ai_word)
    
    def get_used_words_by_user_id(self, user_id):
        """Fetch the used words from the database for the given user ID.

        Args:
            user_id (str): The ID of the user.

        Returns:
            list: A list of words used by the user.
        """
        return db.get_used_words_by_user_id(user_id)

    def save_checkpoint(self, user_id, game_result_id):
        """Save the game checkpoint for the user in the database.

        Args:
            user_id (str): The ID of the user.
            game_result_id (int): The game result ID for the checkpoint.

        Returns:
            bool: True if checkpoint is saved successfully, False otherwise.
        """
        return db.save_checkpoint(user_id, game_result_id)

    def check_existing_score(self, ai_word, human_word):
        """Check if the AI and human word pair already exists in the database.

        Args:
            ai_word (str): The AI's word.
            human_word (str): The human player's word.

        Returns:
            tuple: The primary key ID and similarity score if the pair exists; (None, None) otherwise.
        """
        return db.check_existing_score(ai_word, human_word)

    def clear_used_words_by_user(self, user_id):
        """Clear all used words for the specified user ID.

        Args:
            user_id (str): The ID of the user.

        Returns:
            bool: True if used words are cleared successfully, False otherwise.
        """
        return db.clear_used_words_by_user_id(user_id)

    def increment_round_count(self, user_id):
        """Increment the round count for the specified user ID.

        Args:
            user_id (str): The ID of the user.

        Returns:
            bool: True if round count is incremented successfully, False otherwise.
        """
        return db.increment_round_count(user_id)

    def check_game_status(self, user_id):
        """Retrieve the current round count for the specified user ID.

        Args:
            user_id (str): The ID of the user.

        Returns:
            int: The current round count for the user.
        """
        return db.get_round_count(user_id)

    def check_checkpoint_entry(self, user_id, ai_word, human_word):
        """Check if a checkpoint entry exists for the specified user ID, AI word, and human word pair.

        Args:
            user_id (str): The ID of the user.
            ai_word (str): The AI's word.
            human_word (str): The human player's word.

        Returns:
            bool: True if the checkpoint entry exists, False otherwise.
        """
        return db.check_checkpoint_entry(user_id, ai_word, human_word)

    def clear_user_entries(self, user_id):
        """Clear all entries for the specified user ID from used words and checkpoints.

        Args:
            user_id (str): The ID of the user.

        Returns:
            bool: True if entries are cleared successfully, False otherwise.
        """
        return db.delete_user_entries(user_id)

    def get_similarity_score_with_next_word(self, user_id, ai_word, human_word, score, threshold=1.5):
        """Calculate similarity score and get the next AI word.

        Args:
            user_id (str): The ID of the user.
            ai_word (str): The word provided by the AI.
            human_word (str): The word provided by the human player.
            score (int): The current total score.
            threshold (float, optional): The threshold for adding similarity score to total. Defaults to 1.5.

        Returns:
            dict: Dictionary containing similarity score, next AI word, and updated score.

        Raises:
            AIWordError: If the AI word does not match the last thrown word for the user.
        """
        if ai_word != self.user_data[user_id].get('last_thrown_word'):
            raise AIWordError

        self.expand_word_pool(user_id, score)
        self.updated_score = score
        self.uid = user_id

        if ai_word.lower() == human_word.lower():
            return {
                "error": f"The word '{human_word}' is the same as the AI's word. Please try a different word.",
                "next_ai_word": ai_word
            }
        
        pid, sim_score = self.get_existing_or_calculate_score(user_id, ai_word, human_word)

        if sim_score >= threshold:
            self.updated_score += sim_score
        else:
            return {
                "similarity_score": sim_score,
                "next_ai_word": ai_word,
                "updated_score": self.updated_score
            }

        next_ai_word = self.throw_word_to_user(user_id)
        db.save_thrown_word(user_id, next_ai_word)

        return {
            "similarity_score": sim_score,
            "next_ai_word": next_ai_word,
            "updated_score": self.updated_score
        }

    def get_existing_or_calculate_score(self, user_id, ai_word, human_word):
        """Check if the pair exists in the database or calculate similarity if not.

        Args:
            user_id (str): The ID of the user.
            ai_word (str): The word provided by the AI.
            human_word (str): The word provided by the human player.

        Returns:
            tuple: Primary key ID and similarity score.
        """
        pid, existing_score = self.check_existing_score(ai_word, human_word)

        if existing_score is not None:
            round = self.check_round(user_id)
            if round == 0:
                self.save_checkpoint(user_id, pid)
                return pid, existing_score
            
            elif round == 1:
                entries = self.check_checkpoint_entry(user_id, ai_word, human_word)
                if entries:
                    raise SameInputError
                else:
                    self.save_checkpoint(user_id, pid)
                    return pid, existing_score
        
        elif existing_score is None:
            similarity = self.calculate_similarity(ai_word, human_word)
            sim_score = int(similarity * 10)
            pid = self.save_game_result(ai_word, human_word, sim_score)
            self.save_checkpoint(user_id, pid)
            return pid, sim_score
        return None, 0
    
    def clear_user_data(self, user_id):
        """Clear user data for the specified user ID from the in-memory dictionary.

        Args:
            user_id (str): The ID of the user to clear data for.
        """
        if user_id in self.user_data:
            del self.user_data[user_id]
            print(f"Cleared data for user '{user_id}' from memory.")
