from dataset_loader import DatasetLoader

loader = DatasetLoader(text_file="corpora/wikitexts.md", 
                       questions_file="questions_df.csv")
corpus = loader.load_corpora()
questions_df = loader.load_questions()
