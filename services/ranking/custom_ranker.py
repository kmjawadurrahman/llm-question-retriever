from typing import List, Tuple
import pandas as pd
import numpy as np

class CustomRanker:
    @staticmethod
    def extract_top_two(df: pd.DataFrame) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Extract and sort first and second positions from each row"""
        first_positions = []
        second_positions = []
        
        for _, row in df.iterrows():
            try:
                ques_no = int(row['ques_no'])
                score = float(row['match_scores'])
                first_positions.append((ques_no, score))
            except (TypeError, ValueError) as e:
                continue
        
        # Sort by scores
        first_positions_sorted = sorted(first_positions, key=lambda x: x[1], reverse=True)
        
        # Get second positions if available
        if len(first_positions_sorted) > 1:
            second_positions = first_positions_sorted[1:]
            first_positions_sorted = [first_positions_sorted[0]]
        
        return first_positions_sorted, second_positions

    @staticmethod
    def extract_remaining(df: pd.DataFrame) -> List[Tuple[int, float]]:
        """Extract all items after position 2 and sort by scores"""
        remaining_items = []
        
        for idx, row in df.iterrows():
            if idx < 2:  # Skip first two positions
                continue
            try:
                ques_no = int(row['ques_no'])
                score = float(row['match_scores'])
                remaining_items.append((ques_no, score))
            except (TypeError, ValueError) as e:
                continue
        
        # Sort by scores
        remaining_sorted = sorted(remaining_items, key=lambda x: x[1], reverse=True)
        
        return remaining_sorted

    @staticmethod
    def deduplicate_list(question_list: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Remove duplicates while preserving order of first occurrence"""
        seen = set()
        deduplicated = []
        
        for ques_no, score in question_list:
            if ques_no not in seen:
                seen.add(ques_no)
                deduplicated.append((int(ques_no), float(score)))
        
        return deduplicated

    def process_ranked_questions(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """Process dataframe to get final ordered list of questions"""
        try:
            # Ensure numeric types
            df['ques_no'] = pd.to_numeric(df['ques_no'], errors='coerce')
            df['match_scores'] = pd.to_numeric(df['match_scores'], errors='coerce')
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            # Get sorted lists
            first_positions, second_positions = self.extract_top_two(df)
            remaining = self.extract_remaining(df)
            
            # Combine all lists
            combined = first_positions + second_positions + remaining
            
            # Deduplicate while preserving order
            final_list = self.deduplicate_list(combined)
            
            return final_list
            
        except Exception as e:
            print(f"Error in process_ranked_questions: {str(e)}")
            return []