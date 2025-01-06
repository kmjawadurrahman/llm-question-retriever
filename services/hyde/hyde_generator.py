from typing import List
import anthropic
from config.config import Config

class HydeGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        
    def generate_hyde_questions(self, job_description: str) -> List[str]:
        """Generate hypothetical questions using HyDE approach"""
        hyde_prompt = f"""
        You are an expert interviewer helping to generate relevant interview questions based on job descriptions.

        Given the following job description:
        {job_description}

        Generate 10 interview questions that would effectively assess candidates for this role. The questions should:
        1. Cover key technical skills and requirements from the job description
        2. Include relevant soft skills assessment
        3. Match the seniority level indicated
        4. Be specific and practical
        5. Range from technical knowledge to problem-solving scenarios
        6. Include behavioral and situational questions where appropriate

        FORMAT REQUIREMENT:
        Return ONLY a Python list from of your 10 questions. No other text, no introductions, no explanations, 
        no conclusions - just the list.
        """

        response = self.client.messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=8192,
            messages=[
                {"role": "user", "content": hyde_prompt}
            ]
        )
        
        return eval(response.content[0].text)