from typing import Dict, List
import json
import anthropic
from config.config import Config

class MetadataGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        
    def generate_ja_metadata(self, job_analysis: Dict) -> List[Dict]:
        """Generate metadata templates based on job analysis"""
        prompt = self._create_metadata_prompt_ja(job_analysis)
        return self._generate_templates(prompt)
    
    def generate_jd_metadata(self, job_description: str) -> List[Dict]:
        """Generate metadata templates based on job description"""
        prompt = self._create_metadata_prompt_jd(job_description)
        return self._generate_templates(prompt)
    
    def _create_metadata_prompt_ja(self, job_analysis: Dict) -> str:
        return f"""
        SYSTEM CONTEXT:
        You are assisting in generating comprehensive interview question metadata templates based on job requirements. 
        These templates will be used to match appropriate interview questions with job descriptions through vector embeddings.

        INPUT CONTEXT:
        Job Description Analysis: {job_analysis}

        TASK:
        Generate a list of 10 question metadata templates that comprehensively cover the skills, knowledge areas, 
        and competencies required for this role.

        OUTPUT STRUCTURE:
        Each template should follow this structure:
        {Config.get_question_template()}

        EXAMPLES:
        {Config.get_metadata_examples()}

        Generate a list of 10 question metadata templates. Output should be a valid JSON array of templates.
        """
    
    def _create_metadata_prompt_jd(self, job_description: str) -> str:
        return f"""
        SYSTEM CONTEXT:
        You are assisting in generating comprehensive interview question metadata templates based on job requirements. 
        These templates will be used to match appropriate interview questions with job descriptions through vector embeddings.

        INPUT CONTEXT:
        Job Description: {job_description}

        TASK:
        Generate a list of 10 question metadata templates that comprehensively cover the skills, knowledge areas, 
        and competencies required for this role.

        OUTPUT STRUCTURE:
        Each template should follow this structure:
        {Config.get_question_template()}

        EXAMPLES:
        {Config.get_metadata_examples()}

        Generate a list of 10 question metadata templates. Output should be a valid JSON array of templates.
        """
    
    def _generate_templates(self, prompt: str) -> List[Dict]:
        """Generate templates using AI service"""
        response = self.client.messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(response.content[0].text)