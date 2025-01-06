from typing import Dict, List
import json
import anthropic
import pandas as pd
from config.config import Config

from typing import Dict, List
import json
import anthropic
import pandas as pd
import time
import logging
from config.config import Config

class PostProcessor:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    
    def analyze_question_coverage(self, job_description: str, 
                                question_numbers: List[int], 
                                questions_list: List[str],
                                max_retries: int = 5) -> Dict:
        """
        Analyze question coverage and suggest missing aspects
        """
        prompt = self._create_evaluation_prompt(job_description, question_numbers, questions_list)
        
        retry_count = 0
        while retry_count < max_retries:
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
                
            except json.JSONDecodeError as e:
                retry_count += 1
                logging.warning(f"Retry {retry_count}/{max_retries} - Error parsing response: {e}")
                if retry_count < max_retries:
                    time.sleep(2)  # Wait before retrying
                    continue
                else:
                    logging.error(f"Failed to parse response after {max_retries} retries")
                    return {
                        "scores": [],
                        "missing_aspects": "Error in processing",
                        "suggested_questions": []
                    }
            except Exception as e:
                retry_count += 1
                logging.warning(f"Retry {retry_count}/{max_retries} - Unexpected error: {e}")
                if retry_count < max_retries:
                    time.sleep(2)
                    continue
                else:
                    logging.error(f"Failed after {max_retries} retries due to unexpected error")
                    return {
                        "scores": [],
                        "missing_aspects": "Error in processing",
                        "suggested_questions": []
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

    def process_multiple_jobs(self, jobs_df: pd.DataFrame, 
                            results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process multiple jobs to add missing aspects and suggested questions
        """
        for idx, row in results_df.iterrows():
            try:
                job_desc = jobs_df.iloc[idx]['Job Description']
                question_numbers = eval(row['reordered_question_numbers'])
                questions_list = eval(row['reordered_questions'])
                
                analysis = self.analyze_question_coverage(
                    job_desc,
                    question_numbers,
                    questions_list
                )
                
                results_df.at[idx, 'missing_aspects'] = analysis['missing_aspects']
                results_df.at[idx, 'suggested_questions'] = str(analysis['suggested_questions'])
                
            except Exception as e:
                logging.error(f"Error processing job {idx}: {str(e)}")
                results_df.at[idx, 'missing_aspects'] = "Error in processing"
                results_df.at[idx, 'suggested_questions'] = "[]"
        
        return results_df