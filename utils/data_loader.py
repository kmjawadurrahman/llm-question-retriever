from typing import List, Dict, Tuple
import pandas as pd
from config.config import Config

class DataLoader:
    @staticmethod
    def load_questions_data() -> List[str]:
        """Load questions from CSV file"""
        return pd.read_csv(Config.QUESTIONS_DATA_PATH)['Question'].tolist()
    
    @staticmethod
    def load_jobs_data() -> pd.DataFrame:
        """Load job descriptions from CSV file"""
        return pd.read_csv(Config.JOBS_EVAL_PATH)
    
    @staticmethod
    def load_evaluation_data(job_num: int) -> pd.DataFrame:
        """Load evaluation data for a specific job"""
        eval_df = pd.read_csv(f'./data/evals/job{job_num}.csv')
        eval_df['Question ID'] = eval_df['Question ID'].apply(lambda x: int(x.replace('Q', '')))
        return eval_df

    @staticmethod
    def save_results(df: pd.DataFrame, filename: str):
        """Save results to CSV file"""
        df.to_csv(filename, index=False)