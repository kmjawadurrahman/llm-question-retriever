import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # AI Models
    CLAUDE_MODEL = "claude-3-5-haiku-latest"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Database
    QDRANT_PATH = "./qdrant_storage"
    QUESTIONS_COLLECTION = "questions_collection"
    QUESTIONS_RAW_COLLECTION = "questions_raw_collection"
    
    # File paths
    QUESTIONS_DATA_PATH = './data/interview_questions_data.csv'
    QUESTIONS_TEMPLATE_PATH = './data/questions_template_data.csv'
    QUESTIONS_EMBEDDINGS_PATH = './data/questions_embeddings.csv'
    JOBS_METADATA_PATH = './data/jobs_metadata_df.csv'
    JOBS_EVAL_PATH = './data/job_descriptions_eval.csv'
    
    # Templates
    @staticmethod
    def get_question_system_context():
        return """
        You are assisting in building a hierarchical system that maps job descriptions to relevant interview questions. 
        The system uses a structured approach to represent both interview questions and job descriptions in similar 
        vector spaces for effective matching. Your task is to analyze interview questions and extract structured 
        information that will help in this mapping.

        The extracted information will be used to:
        1. Create embeddings that align with job description requirements
        2. Enable effective filtering and/or matching of questions to job requirements
        """

    @staticmethod
    def get_question_template():
        return """
        {   
            "description": "",        # A generalized description focusing on core skills and competencies
            "technical_concepts": {
                "category": "",        # Probability, Statistics, ML, Coding, SQL, etc.
                "primary": [],        # Main technical concepts being tested
                "tools": [],         # Specific tools/technologies involved
                "methods": []        # Specific methods/algorithms/approaches
            },
            "skill_level": {
                "depth": "",         # basic/intermediate/advanced
                "knowledge_type": [], # theoretical/practical/implementation/design
                "experience_level": "", # junior/mid/senior/lead
            },
            "domain_context": {
                "industries": [],    # Relevant industry domains
            }
        }
        """

    @staticmethod
    def get_jd_system_context():
        return """
        You are assisting in building a hierarchical system that maps job descriptions to relevant interview questions. 
        The system analyzes job descriptions to extract structured information that will be used to match with 
        appropriate interview questions. This matching process relies on aligning the vector spaces of job requirements 
        with interview question characteristics.
        """

    @staticmethod
    def get_jd_template():
        return """
        {
            "technical_needs": {
                "required_skills": [],    # Must-have technical skills
                "preferred_skills": [],   # Nice-to-have technical skills
                "tools": [],             # Required tools/technologies
                "responsibilities": []    # Technical responsibilities
            },
            "experience_needs": {
                "years": "",             # Required years of experience
                "domain_experience": []   # Required domain experience
            },
            "domain_context": {
                "industry": "",          # Primary industry
            },
            "role_context": {
                "level": "",            # junior/mid/senior/lead
                "leadership_scope": "",  # Leadership responsibilities
            }
        }
        """

    @staticmethod
    def get_metadata_examples():
        return """
    {
        "description": "A probability analysis question testing understanding of sequential probability, combinatorics, and series outcomes in a sports/games context",
        "technical_concepts": {
            "category": "Probability",
            "primary": ["Sequential probability", "Binary outcomes", "Combinatorics", "Series probability"],
            "tools": [],
            "methods": ["Probability tree analysis", "Combination calculation"]
        },
        "skill_level": {
            "depth": "intermediate",
            "knowledge_type": ["theoretical", "analytical"],
            "experience_level": "mid"
        },
        "domain_context": {
            "industries": ["General", "Sports analytics", "Gaming"]
        }
    }

    {
        "description": "Tests understanding of maximum likelihood estimation (MLE) applied to uniform distribution parameters, combining probability theory and statistical inference concepts",
        "technical_concepts": {
            "category": "Statistics",
            "primary": ["Maximum Likelihood Estimation", "Uniform Distribution", "Parameter Estimation", "Sampling Theory"],
            "tools": [],
            "methods": ["MLE Method", "Parameter Estimation"]
        },
        "skill_level": {
            "depth": "intermediate",
            "knowledge_type": ["theoretical", "practical"],
            "experience_level": "mid"
        },
        "domain_context": {
            "industries": ["General"]
        }
    }

    {
        "description": "Tests understanding of probabilistic modeling for anomaly detection using Gaussian Mixture Models, including mathematical formulation, likelihood computation, and practical application to fraud detection",
        "technical_concepts": {
            "category": "Machine Learning",
            "primary": ["Gaussian Mixture Models", "Anomaly Detection", "Posterior Probability", "Maximum Likelihood Estimation", "Classification"],
            "tools": [],
            "methods": ["Likelihood Computation", "Posterior Probability Calculation", "Threshold-based Detection"]
        },
        "skill_level": {
            "depth": "advanced",
            "knowledge_type": ["theoretical", "practical", "implementation"],
            "experience_level": "senior"
        },
        "domain_context": {
            "industries": ["Financial Services", "Banking", "Payment Processing"]
        }
    }

    {
        "description": "Tests ability to write SQL queries involving grouping, counting, and identifying duplicates across multiple columns using self-joins or group by clauses",
        "technical_concepts": {
            "category": "SQL/Database",
            "primary": ["SQL querying", "Duplicate detection", "Aggregation functions", "Group By operations"],
            "tools": ["SQL"],
            "methods": ["COUNT aggregation", "GROUP BY filtering", "String comparison"]
        },
        "skill_level": {
            "depth": "intermediate",
            "knowledge_type": ["practical", "implementation"],
            "experience_level": "mid"
        },
        "domain_context": {
            "industries": ["Recruitment", "HR Tech", "Job Platforms"]
        }
    }

    {
        "description": "Algorithm problem testing string manipulation, pattern matching, and efficient computation of anagrams with sliding window technique. Tests understanding of hash maps, string operations, and optimization techniques.",
        "technical_concepts": {
            "category": "Coding",
            "primary": ["String Manipulation", "Anagrams", "Sliding Window", "Hash Maps"],
            "tools": ["Data Structures"],
            "methods": ["Sliding Window Technique", "Character Frequency Counting", "Array/String Iteration"]
        },
        "skill_level": {
            "depth": "intermediate",
            "knowledge_type": ["theoretical", "implementation"],
            "experience_level": "mid"
        },
        "domain_context": {
            "industries": ["General"]
        }
    }

    {
        "description": "Analysis of customer segmentation approaches for retail sales optimization, focusing on data-driven clustering techniques and business strategy application",
        "technical_concepts": {
            "category": "Machine Learning",
            "primary": ["Customer Segmentation", "Clustering Analysis", "Retail Analytics"],
            "tools": ["Statistical Analysis Software", "Machine Learning Libraries"],
            "methods": ["K-means Clustering", "Hierarchical Clustering", "RFM Analysis", "Demographic Segmentation"]
        },
        "skill_level": {
            "depth": "intermediate",
            "knowledge_type": ["practical", "implementation", "business strategy"],
            "experience_level": "mid"
        },
        "domain_context": {
            "industries": ["Retail", "Consumer Electronics", "Luxury Retail"]
        }
    }

    {
        "description": "Design a content recommendation system for a social media discovery feature, focusing on user engagement, personalization, and scalability aspects",
        "technical_concepts": {
            "category": "Product Sense",
            "primary": ["recommendation systems", "personalization algorithms", "content ranking", "user behavior analysis"],
            "tools": ["ranking algorithms", "user behavior tracking systems", "content classification systems"],
            "methods": ["matrix factorization", "deep learning", "similarity algorithms", "A/B testing"]
        },
        "skill_level": {
            "depth": "advanced",
            "knowledge_type": ["design", "theoretical", "practical"],
            "experience_level": "senior"
        },
        "domain_context": {
            "industries": ["social media", "content platforms", "digital entertainment"]
        }
    }
    """