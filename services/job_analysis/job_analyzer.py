from typing import Dict, List
import json
from langchain.prompts import PromptTemplate
import anthropic
from config.config import Config

class JobAnalyzer:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.prompt = self._create_jd_prompt()
        
    def _create_jd_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["job_description"],
            partial_variables={
                "system_context": Config.get_jd_system_context(),
                "task": "Analyze the provided job description and extract structured information.",
                "json_template": Config.get_jd_template()
            },
            template=(
                "{system_context}\n\n{task}\n\n"
                "Template to follow:\n{json_template}\n\n"
                "Job Description to analyze: {job_description}\n\n"
                "Output the filled template in JSON format only."
            )
        )
    
    def analyze_job_description(self, job_description: str) -> Dict:
        """Analyze a job description and return structured template"""
        info_prompt = self.prompt.format(job_description=job_description)
        
        response = self.client.messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=8192,
            messages=[
                {"role": "user", "content": info_prompt}
            ]
        )
        
        return json.loads(response.content[0].text)