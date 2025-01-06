import pandas as pd
from typing import Dict, List, Union, Tuple

class ResultProcessor:
    @staticmethod
    def process_question_results(questions_df: pd.DataFrame, 
                               templates: List[Dict]) -> pd.DataFrame:
        """Process questions and templates into DataFrame"""
        response_df = pd.DataFrame({
            'ques_no': range(1, len(questions_df) + 1),
            'question': questions_df,
            'ques_template': templates
        })
        return response_df
    
    @staticmethod
    def process_job_results(jobs_df: pd.DataFrame, 
                          job_templates: List[Dict],
                          ja_metadata: List[List[Dict]],
                          jd_metadata: List[List[Dict]],
                          hyde_questions: List[List[str]]) -> pd.DataFrame:
        """Process job analysis results into DataFrame"""
        jobs_df['job_info_template'] = job_templates
        jobs_df['job_ja_question_metadata'] = ja_metadata
        jobs_df['job_jd_question_metadata'] = jd_metadata
        jobs_df['job_questions_hyde_list'] = hyde_questions
        return jobs_df

    @staticmethod
    def combine_method_results(ja_results: pd.DataFrame,
                             jd_results: pd.DataFrame,
                             hyde_results: pd.DataFrame) -> pd.DataFrame:
        """Combine results from all methods"""
        all_results = pd.concat([
            ja_results.assign(method='JA'),
            jd_results.assign(method='JD'),
            hyde_results.assign(method='HyDE')
        ], ignore_index=True)
        return all_results

    @staticmethod
    def create_evaluation_dataframe(question_numbers: List[int],
                                  match_scores: List[float],
                                  evaluation_scores: List[float],
                                  method: str,
                                  top_n: int) -> pd.DataFrame:
        """Create evaluation DataFrame"""
        return pd.DataFrame({
            'method': [method],
            'top_n': [top_n],
            'question_order': [question_numbers],
            'match_scores': [match_scores],
            'evaluation_scores': [evaluation_scores]
        })

    @staticmethod
    def process_ranked_results(df: pd.DataFrame, 
                             job_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process both original and reranked results for a job"""
        # Extract scores and questions
        original_scores = df['match_scores'].tolist()
        original_questions = df['ques_no'].tolist()
        
        # Create original results DataFrame
        original_df = pd.DataFrame({
            'job_number': job_number,
            'match_scores': [original_scores],
            'question_order': [original_questions],
            'questions_list': [df['question'].tolist()]
        })
        
        # Create reranked results DataFrame
        reranked_df = pd.DataFrame({
            'job_number': job_number,
            'match_scores': [sorted(original_scores, reverse=True)],
            'question_order': [sorted(zip(original_questions, original_scores), 
                                   key=lambda x: x[1], reverse=True)],
            'questions_list': [df['question'].tolist()]
        })
        
        return original_df, reranked_df

    @staticmethod
    def save_results(results_dict: Dict[str, pd.DataFrame], 
                    base_filename: str):
        """Save results for each method"""
        for method, df in results_dict.items():
            df.to_csv(f'{base_filename}_{method}.csv', index=False)

    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'normalized_binary_score': df['binary_scores'].mean(),
            'normalized_score': df['evaluation_scores'].mean() / 4,
            'normalized_combined_score': df['combined_scores'].mean() / 6
        }

    @staticmethod
    def format_results_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """Format results for final analysis"""
        formatted_df = df.copy()
        # Convert string representations of lists back to actual lists
        for col in ['post_process_scores', 'reordered_question_numbers', 
                   'reordered_evaluation_scores', 'reordered_questions']:
            formatted_df[col] = formatted_df[col].apply(eval)
        return formatted_df

    @staticmethod
    def aggregate_results(ja_df: pd.DataFrame, 
                         jd_df: pd.DataFrame, 
                         hyde_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results from all methods"""
        all_results = []
        
        for df, method in [(ja_df, 'JA'), (jd_df, 'JD'), (hyde_df, 'HyDE')]:
            method_results = df.copy()
            method_results['method'] = method
            all_results.append(method_results)
        
        return pd.concat(all_results, ignore_index=True)
