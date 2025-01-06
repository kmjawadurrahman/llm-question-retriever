import streamlit as st
import pandas as pd
import logging
from typing import Dict, List, Tuple

from config.config import Config
from services.job_analysis.job_analyzer import JobAnalyzer
from services.job_analysis.metadata_generator import MetadataGenerator
from services.hyde.hyde_generator import HydeGenerator
from services.embedding.embedding_service import EmbeddingService
from services.storage.qdrant_service import QdrantService
from services.ranking.custom_ranker import CustomRanker
from services.evaluation.evaluator import Evaluator
from services.evaluation.post_processor import PostProcessor

# Configure page
st.set_page_config(
    page_title="Interview Question Recommender",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .log-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class StreamlitLogHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        
    def emit(self, record):
        log_entry = self.format(record)
        with self.placeholder.container():
            st.text(log_entry)

@st.cache_resource
def initialize_services():
    """Initialize all required services"""
    return {
        'job_analyzer': JobAnalyzer(),
        'metadata_generator': MetadataGenerator(),
        'hyde_generator': HydeGenerator(),
        'embedding_service': EmbeddingService(),
        'qdrant_service': QdrantService(),
        'custom_ranker': CustomRanker(),
        'evaluator': Evaluator(),
        'post_processor': PostProcessor()
    }

def setup_logging(placeholder):
    """Setup logging with Streamlit output"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Add Streamlit handler
    handler = StreamlitLogHandler(placeholder)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def process_job_description(job_description: str, method: str, services: Dict, 
                          log_placeholder, num_questions: int) -> Tuple[List[str], Dict]:
    """Process job description using selected method"""
    logging.info(f"Processing job description using {method} method...")

    try:
        if method == "Job Analysis":
            # JA method
            with st.spinner("Analyzing job description..."):
                job_info = services['job_analyzer'].analyze_job_description(job_description)
                st.success("✅ Job analysis completed")
                
                if st.session_state.show_details:
                    with st.expander("Job Analysis Results"):
                        st.json(job_info)
            
            with st.spinner("Generating metadata templates..."):
                metadata = services['metadata_generator'].generate_ja_metadata(job_info)
                st.success("✅ Metadata generation completed")
                
                if st.session_state.show_details:
                    with st.expander("Metadata Templates"):
                        st.json(metadata)
            
        elif method == "Job Description":
            # JD method
            with st.spinner("Generating metadata templates from job description..."):
                metadata = services['metadata_generator'].generate_jd_metadata(job_description)
                st.success("✅ Metadata generation completed")
                
                if st.session_state.show_details:
                    with st.expander("Metadata Templates"):
                        st.json(metadata)
            
        else:  # HyDE method
            with st.spinner("Generating hypothetical questions..."):
                metadata = services['hyde_generator'].generate_hyde_questions(job_description)
                st.success("✅ Question generation completed")
                
                if st.session_state.show_details:
                    with st.expander("Generated Questions"):
                        st.json(metadata)
        
        # Get embeddings and search
        with st.spinner("Processing embeddings and searching questions..."):
            collection_name = (Config.QUESTIONS_RAW_COLLECTION 
                            if method == "HyDE" 
                            else Config.QUESTIONS_COLLECTION)
            
            results = []
            for template in metadata:
                query_vector = services['embedding_service'].get_embedding(
                    str(template) if method == "HyDE" else template
                )
                
                search_results = services['qdrant_service'].search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=10
                )
                
                results.extend(search_results)
            
            st.success("✅ Search completed")
            
            if st.session_state.show_details:
                with st.expander("Search Results"):
                    st.write(f"Found {len(results)} matching questions")
        
        # Rank results
        with st.spinner("Ranking results..."):
            ranking_df = pd.DataFrame({
                'ques_no': [r.payload['ques_no'] for r in results],
                'match_scores': [float(r.score) for r in results],
                'question': [r.payload['question'] for r in results]
            })
            
            final_list = services['custom_ranker'].process_ranked_questions(ranking_df)
            
            # Get top questions using num_questions parameter
            top_questions = [
                ranking_df[ranking_df['ques_no'] == q]['question'].iloc[0]
                for q, _ in final_list[:num_questions]
            ]
            
            st.success("✅ Ranking completed")
            
            if st.session_state.show_details:
                with st.expander("Ranking Details"):
                    st.dataframe(ranking_df)
        
        # Get missing aspects analysis
        with st.spinner("Analyzing coverage and generating suggestions..."):
            analysis = services['post_processor'].analyze_question_coverage(
                job_description,
                [q for q, _ in final_list[:num_questions]],
                top_questions
            )
            st.success("✅ Analysis completed")
        
        return top_questions, analysis
        
    except Exception as e:
        st.error(f"Error processing job description: {str(e)}")
        logging.error(f"Error: {str(e)}")
        return [], {
            "missing_aspects": "Error in processing",
            "suggested_questions": []
        }

def main():
    # Title and introduction
    st.title("🎯 Interview Question Recommender")
    
    st.markdown("""
        This application helps you generate relevant interview questions based on job descriptions. 
        It uses advanced AI techniques to analyze job requirements and recommend appropriate questions 
        that assess the required skills and competencies.
    """)
    
    # Disclaimer
    st.warning("""
        **Disclaimer**: This tool uses AI models to generate and analyze content. The outputs, 
        including questions and suggestions, are generated by AI and should be reviewed by human 
        experts before use in actual interviews. The quality and appropriateness of results may vary.
    """)
    
    # Initialize services
    services = initialize_services()
    
    # Input section
    st.header("Input Parameters")
    
    # Job description input
    job_description = st.text_area(
        "Job Description",
        help="Paste the complete job description here",
        height=200,
        placeholder="Enter job description...",
        key="job_description"
    )
    
    # Method selection
    method = st.radio(
        "Select Method",
        ["Job Analysis", "Job Description", "HyDE"],
        help="""
        - Job Analysis: Analyzes job requirements first, then generates questions
        - Job Description: Directly generates questions from job description
        - HyDE: Uses hypothetical questions to find similar existing questions
        """
    )

    # Number of questions slider
    num_questions = st.slider(
        "Number of questions to retrieve",
        min_value=10,
        max_value=25,
        value=15,
        step=1,
        help="Select how many questions you want to retrieve"
    )
    
    # Detailed logging option
    if 'show_details' not in st.session_state:
        st.session_state.show_details = True
    
    st.session_state.show_details = st.checkbox(
        "Show detailed process logs",
        value=st.session_state.show_details,
        help="Display detailed logs and intermediate outputs during processing"
    )
    
    # Create placeholder for logs
    log_placeholder = st.empty()
    
    # Submit button
    if st.button("Generate Questions", type="primary"):
        if not job_description:
            st.error("Please enter a job description")
            return
        
        # Setup logging
        if st.session_state.show_details:
            setup_logging(log_placeholder)
        
        # Process job description with num_questions parameter
        questions, analysis = process_job_description(
            job_description, method, services, log_placeholder, num_questions
        )
        
        # Display results
        st.header("Results")
        
        # Display recommended questions
        st.subheader(f"Recommended Interview Questions (Top {num_questions})")
        for i, question in enumerate(questions, 1):
            st.markdown(f"{i}. {question}")
        
        # Display missing aspects and suggestions
        st.subheader("Coverage Analysis (AI Expert Feedback)")
        
        st.markdown("**Missing or Under-represented Areas:**")
        st.write(analysis['missing_aspects'])
        
        st.markdown("**Suggested Additional Questions:**")
        for i, question in enumerate(analysis['suggested_questions'], 1):
            st.markdown(f"{i}. {question}")

if __name__ == "__main__":
    main()