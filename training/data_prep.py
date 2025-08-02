import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_cleaner import DataSanitizer


class DataPreparer:
    def prepare_dataset(self, training_data: list):
        sanitizer = DataSanitizer()
        df = pd.DataFrame(
            training_data, 
            columns=["input", "response", "intent", "emotion"]
        )
        df['input'] = df['input'].apply(sanitizer.sanitize_text)
        df['response'] = df['response'].apply(sanitizer.sanitize_text)
        train, val = train_test_split(df, test_size=0.2)
        train.to_json("training/train.jsonl", orient="records", lines=True)
        val.to_json("training/val.jsonl", orient="records", lines=True)
