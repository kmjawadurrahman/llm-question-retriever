from typing import Dict, List, Union
import pandas as pd
import json
import anthropic
from config.config import Config
from ..ranking.custom_ranker import CustomRanker

class Evaluator:
    def __init__(self):
        self.ranker = CustomRanker()
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def evaluate_results(self, final_list: List[tuple], eval_df: pd.DataFrame, 
                        top_n: int = 10, method: str = 'original') -> Dict:
        """
        Evaluate results against evaluation dataframe for top N items
        """
        # Get top N questions
        top_n_list = final_list[:top_n]
        
        # Get questions and scores
        questions = [q for q, _ in top_n_list]
        match_scores = [s for _, s in top_n_list]
        
        # Create dictionaries for quick lookup of different scores
        score_dict = dict(zip(eval_df['Question ID'], eval_df['Score']))
        binary_dict = dict(zip(eval_df['Question ID'], eval_df['Score Binary']))
        combined_dict = dict(zip(eval_df['Question ID'], eval_df['Score Combined']))
        
        # Get evaluation scores
        evaluation_scores = [score_dict.get(q, None) for q in questions]
        binary_scores = [binary_dict.get(q, None) for q in questions]
        combined_scores = [combined_dict.get(q, None) for q in questions]
        
        # Calculate normalized scores
        total_binary = sum(score for score in binary_scores if score is not None)
        total_score = sum(score for score in evaluation_scores if score is not None)
        total_combined = sum(score for score in combined_scores if score is not None)
        
        return {
            'method': method,
            'top_n': top_n,
            'question_order': questions,
            'match_scores': match_scores,
            'evaluation_scores': evaluation_scores,
            'binary_scores': binary_scores,
            'combined_scores': combined_scores,
            'normalized_binary_score': total_binary / top_n if top_n > 0 else 0,
            'normalized_score': total_score / (top_n * 4) if top_n > 0 else 0,
            'normalized_combined_score': total_combined / (top_n * 6) if top_n > 0 else 0
        }

    def _create_evaluation_prompt(self, job_description: str, 
                                question_numbers: List[int], 
                                questions: List[str]) -> str:
        """
        Create the prompt for question evaluation and missing aspects analysis
        """
        formatted_questions = "\n".join([f"Question {num}: {text}" 
                                       for num, text in zip(question_numbers, questions)])
        
        return f"""You are an expert technical interviewer tasked with evaluating interview questions against a job description.

Job Description:
{job_description}

Current Interview Questions Set:
{formatted_questions}

Your task has three parts:

1. Score each question's relevance using this rubric:
   - 4: Directly addresses critical job requirements, tests specific required skills, effectively evaluates core competencies
   - 3: Relates to job requirements and tests relevant skills, but could be more specific/targeted
   - 2: Has minimal connection to job requirements, too general/basic to effectively evaluate qualifications
   - 1: Completely unrelated to job requirements, or tests skills/knowledge not mentioned in description

2. Identify missing or under-represented areas in the question set.

3. Suggest additional interview questions that address the missing aspects.

Provide your analysis in JSON format with exactly three fields:
1. "scores": A list of integers (1-4) corresponding to each question in the order provided
2. "missing_aspects": A detailed paragraph about gaps in coverage, or "None" if adequate
3. "suggested_questions": A list of additional interview questions addressing the missing aspects

Rules:
- Scores MUST be integers from 1 to 4 based on the rubric
- Score each question in the exact order provided
- Be specific about missing aspects by referencing job requirements
- Suggested questions should directly address identified gaps
- Return ONLY valid JSON, nothing else

Example format:
{{
    "scores": [4, 3, 2, 4, 1, 3],
    "missing_aspects": "The question set lacks coverage of [specific aspects]...",
    "suggested_questions": [
        "First suggested question addressing gap...",
        "Second suggested question addressing gap..."
    ]
}}"""

    def analyze_question_coverage(self, job_description: str, 
                                question_numbers: List[int], 
                                questions_list: List[str], 
                                max_retries: int = 3) -> Dict:
        """
        Analyze question coverage and get missing aspects
        """
        prompt = self._create_evaluation_prompt(job_description, question_numbers, questions_list)
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=Config.CLAUDE_MODEL,
                    max_tokens=8192,
                    system="You are an expert technical interviewer who provides responses only in valid JSON format.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return json.loads(response.content[0].text)
            
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return {
                        "scores": [],
                        "missing_aspects": "Error in processing",
                        "suggested_questions": []
                    }

    def process_multiple_jobs(self, jobs_df: pd.DataFrame, question_orders: List[str],
                            questions_lists: List[List[str]], evaluation_scores: List[str],
                            max_retries: int = 5) -> pd.DataFrame:
        """Process multiple jobs with evaluation and reordering"""
        results_df = pd.DataFrame(columns=[
            'job_num',
            'post_process_scores',
            'reordered_question_numbers',
            'reordered_evaluation_scores',
            'reordered_questions',
            'normalized_post_score',
            'missing_aspects',
            'suggested_questions'
        ])
        
        for idx, (job_desc, question_order, questions_list, eval_scores) in enumerate(
            zip(jobs_df['Job Description'], question_orders, questions_lists, evaluation_scores), 1):
            
            # Convert strings to lists if needed
            question_numbers = eval(question_order) if isinstance(question_order, str) else question_order
            eval_scores_list = eval(eval_scores) if isinstance(eval_scores, str) else eval_scores
            
            # Process the results
            final_list = self.ranker.process_ranked_questions(pd.DataFrame({
                'ques_no': [question_numbers],
                'match_scores': [eval_scores_list]
            }))
            
            # Take top 20
            top_20 = final_list[:20]
            
            # Get post-processing analysis
            analysis = self.analyze_question_coverage(
                job_desc,
                [q for q, _ in top_20],
                questions_list[:20]
            )
            
            # Create new row
            new_row = pd.DataFrame({
                'job_num': [f'job{idx}'],
                'post_process_scores': [[s for _, s in top_20]],
                'reordered_question_numbers': [[q for q, _ in top_20]],
                'reordered_evaluation_scores': [eval_scores_list[:20]],
                'reordered_questions': [questions_list[:20]],
                'normalized_post_score': [sum(eval_scores_list[:20]) / (20 * 4)],
                'missing_aspects': [analysis['missing_aspects']],
                'suggested_questions': [analysis['suggested_questions']]
            })
            
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        
        return results_df