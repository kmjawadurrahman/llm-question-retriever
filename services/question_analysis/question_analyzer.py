from typing import Dict, List
import json
from langchain.prompts import PromptTemplate
import anthropic
from config.config import Config

class QuestionAnalyzer:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.prompt = self._create_question_prompt()
        
    def _create_question_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["question"],
            partial_variables={
                "system_context": Config.get_question_system_context(),
                "task": "Analyze the provided interview question and extract structured information.",
                "json_template": Config.get_question_template()
            },
            template=(
                "{system_context}\n\n{task}\n\n"
                "Template to follow:\n{json_template}\n\n"
                "Question to analyze: {question}\n\n"
                "Output the filled template in JSON format only."
            )
        )
    
    def analyze_question(self, question: str) -> Dict:
        """Analyze a single question and return structured template"""
        info_prompt = self.prompt.format(question=question)
        
        response = self.client.messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=8192,
            messages=[
                {"role": "user", "content": info_prompt}
            ]
        )
        
        return json.loads(response.content[0].text)
    
    def analyze_questions_batch(self, questions: List[str]) -> List[Dict]:
        """Analyze a batch of questions"""
        templates = []
        for question in questions:
            template = self.analyze_question(question)
            templates.append(template)
        return templates
