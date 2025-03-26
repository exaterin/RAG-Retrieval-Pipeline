import os
import pandas as pd


class DatasetLoader:
    """
    Loads a text corpora file and questions.
    """

    def __init__(self, text_file: str, questions_file: str):
        self.text_file = text_file
        self.questions_file = questions_file

    def load_corpora(self) -> str:
        """
        Load the corpora.
        """
        if not os.path.isfile(self.text_file):
            raise FileNotFoundError(f"{self.text_file} not found.")
        
        with open(self.text_file, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_questions(self) -> pd.DataFrame:
        """
        Load questions and golden excerpts.
        """
        if not os.path.isfile(self.questions_file):
            raise FileNotFoundError(f"{self.questions_file} not found.")
        
        return pd.read_csv(self.questions_file)
