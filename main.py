import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Dict
from datetime import datetime
import traceback
import time
import json
import os
import anthropic
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config.config import Config
from services.question_analysis.question_analyzer import QuestionAnalyzer
from services.job_analysis.job_analyzer import JobAnalyzer
from services.job_analysis.metadata_generator import MetadataGenerator
from services.hyde.hyde_generator import HydeGenerator
from services.embedding.embedding_service import EmbeddingService
from services.storage.qdrant_service import QdrantService
from services.ranking.custom_ranker import CustomRanker
from services.evaluation.evaluator import Evaluator
from services.evaluation.post_processor import PostProcessor
from utils.data_loader import DataLoader
from utils.result_processor import ResultProcessor

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup logging with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/process_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def save_dataframe(df: pd.DataFrame, filename: str, directory: str = 'output'):
    """Save DataFrame with proper error handling and logging"""
    try:
        # Create full directory path
        if '/' in filename:
            # If filename contains subdirectories
            full_path = os.path.join(directory, os.path.dirname(filename))
        else:
            full_path = directory
            
        # Create all necessary directories
        os.makedirs(full_path, exist_ok=True)
        
        # Create full filepath
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        logging.info(f"Successfully saved DataFrame to {filepath}")
        
    except Exception as e:
        logging.error(f"Error saving DataFrame to {filename}: {str(e)}")
        raise

# def process_templates(templates: List, collection_name: str, 
#                      qdrant_service: QdrantService, 
#                      custom_ranker: CustomRanker,
#                      embedding_service: EmbeddingService,
#                      result_processor: ResultProcessor,
#                      job_number: int, 
#                      is_hyde: bool = False) -> tuple:
#     """Helper function to process templates and get search results"""
#     original_df = pd.DataFrame()
#     reranked_df = pd.DataFrame()
    
#     for template in templates:
#         try:
#             # Get embedding and search
#             query_vector = embedding_service.get_embedding(str(template) if is_hyde else template)
#             search_results = qdrant_service.search(
#                 collection_name=collection_name,
#                 query_vector=query_vector,
#                 limit=10
#             )
            
#             # Convert scores to float
#             scores = [float(r.score) for r in search_results]
#             question_nos = [r.payload['ques_no'] for r in search_results]
#             questions = [r.payload['question'] for r in search_results]
            
#             # Create original results DataFrame
#             original_result = pd.DataFrame({
#                 'template': [template],
#                 'match_scores': [scores],
#                 'ques_no': [question_nos],
#                 'question': [questions],
#                 'job_number': [job_number]
#             })
            
#             # Get reranked results using CustomRanker
#             final_list = custom_ranker.process_ranked_questions(pd.DataFrame({
#                 'ques_no': question_nos,
#                 'match_scores': scores
#             }))
            
#             # Create reranked results DataFrame
#             reranked_result = pd.DataFrame({
#                 'template': [template],
#                 'match_scores': [[float(s) for _, s in final_list]],
#                 'ques_no': [[q for q, _ in final_list]],
#                 'question': [questions],  # Keep original order for questions
#                 'job_number': [job_number]
#             })
            
#             original_df = pd.concat([original_df, original_result])
#             reranked_df = pd.concat([reranked_df, reranked_result])
            
#         except Exception as e:
#             logging.error(f"Error processing template for job {job_number}: {str(e)}")
#             continue
    
#     return original_df.reset_index(drop=True), reranked_df.reset_index(drop=True)

def process_templates(templates: List, collection_name: str, 
                     qdrant_service: QdrantService, 
                     custom_ranker: CustomRanker,
                     embedding_service: EmbeddingService,
                     result_processor: ResultProcessor,
                     job_number: int, 
                     is_hyde: bool = False) -> tuple:
    """Helper function to process templates and get search results"""
    original_df = pd.DataFrame()
    reranked_df = pd.DataFrame()
    
    for template_idx, template in enumerate(templates):
        try:
            # Get embedding and search
            query_vector = embedding_service.get_embedding(str(template) if is_hyde else template)
            search_results = qdrant_service.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=10
            )
            
            # Debug logging
            logging.info(f"Debug - Job {job_number}, Template {template_idx}:")
            logging.info(f"Search results type: {type(search_results)}")
            logging.info(f"First search result score type: {type(search_results[0].score)}")
            logging.info(f"First search result score: {search_results[0].score}")
            
            # Convert scores to float and extract data
            scores = [float(r.score) for r in search_results]
            question_nos = [int(r.payload['ques_no']) for r in search_results]
            questions = [r.payload['question'] for r in search_results]
            
            # Debug logging for processed data
            logging.info(f"Processed scores type: {type(scores)}")
            logging.info(f"First processed score type: {type(scores[0])}")
            logging.info(f"First processed score: {scores[0]}")
            
            # Create original results DataFrame
            original_result = pd.DataFrame({
                'template': [template],
                'match_scores': [scores],
                'ques_no': [question_nos],
                'question': [questions],
                'job_number': [job_number]
            })
            
            # Debug logging for DataFrame
            logging.info(f"Original DataFrame columns: {original_result.columns}")
            logging.info(f"match_scores type: {type(original_result['match_scores'].iloc[0])}")
            
            # Save intermediate DataFrame
            save_dataframe(original_result, 
                         f'debug/original_result_job{job_number}_template{template_idx}.csv')
            
            # Prepare data for ranking
            ranking_df = pd.DataFrame({
                'ques_no': question_nos,
                'match_scores': scores
            })
            
            # Debug logging for ranking DataFrame
            logging.info(f"Ranking DataFrame shape: {ranking_df.shape}")
            logging.info(f"Ranking DataFrame types - ques_no: {ranking_df['ques_no'].dtype}, match_scores: {ranking_df['match_scores'].dtype}")
            
            # Get reranked results
            final_list = custom_ranker.process_ranked_questions(ranking_df)
            
            # Create reranked results DataFrame
            reranked_result = pd.DataFrame({
                'template': [template],
                'match_scores': [[float(s) for _, s in final_list]],
                'ques_no': [[int(q) for q, _ in final_list]],
                'question': [questions],
                'job_number': [job_number]
            })
            
            original_df = pd.concat([original_df, original_result])
            reranked_df = pd.concat([reranked_df, reranked_result])
            
        except Exception as e:
            logging.error(f"Error processing template for job {job_number}: {str(e)}")
            logging.error(f"Error traceback: {traceback.format_exc()}")
            continue
    
    return original_df.reset_index(drop=True), reranked_df.reset_index(drop=True)


def main():
    # Setup logging
    setup_logging()
    logging.info("Starting interview question matching process")
    
    # Initialize services
    logging.info("Initializing services...")
    data_loader = DataLoader()
    question_analyzer = QuestionAnalyzer()
    job_analyzer = JobAnalyzer()
    metadata_generator = MetadataGenerator()
    hyde_generator = HydeGenerator()
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    custom_ranker = CustomRanker()
    evaluator = Evaluator()
    post_processor = PostProcessor()
    result_processor = ResultProcessor()

    # Step 1: Process Questions
    logging.info("Step 1: Processing questions...")
    questions_data = data_loader.load_questions_data()
    logging.info(f"Loaded {len(questions_data)} questions")
    
    # Generate question templates
    questions_df = pd.DataFrame()
    for idx, question_data in tqdm(enumerate(questions_data), desc="Processing questions"):
        try:
            template = question_analyzer.analyze_question(question_data)
            response_df = pd.DataFrame({
                'ques_no': [idx+1],
                'question': [question_data],
                'ques_template': [template]
            })
            questions_df = pd.concat([questions_df, response_df])
            
            # Save intermediate results every 100 questions
            if (idx + 1) % 100 == 0:
                save_dataframe(questions_df, f'questions_template_intermediate_{idx+1}.csv')
                
        except Exception as e:
            logging.error(f"Error processing question {idx+1}: {str(e)}")
            continue
    
    questions_df = questions_df.reset_index(drop=True)
    save_dataframe(questions_df, 'questions_template_final.csv')

    # Step 2: Generate Embeddings
    logging.info("Step 2: Generating embeddings...")
    try:
        for idx, row in tqdm(enumerate(questions_df.iterrows()), desc="Generating embeddings"):
            # Template embeddings
            template_embedding = embedding_service.get_embedding(str(row[1]['ques_template']))
            questions_df.loc[idx, 'embeddings'] = str(template_embedding)
            
            # Raw question embeddings
            question_embedding = embedding_service.get_embedding(row[1]['question'])
            questions_df.loc[idx, 'embeddings_ques_raw'] = str(question_embedding)
            
            # Save intermediate embeddings every 100 questions
            if (idx + 1) % 100 == 0:
                save_dataframe(questions_df, f'questions_embeddings_intermediate_{idx+1}.csv')
                logging.info(f"Saved embeddings for {idx+1} questions")
        
        # Convert embeddings strings to lists
        questions_df['embeddings'] = questions_df['embeddings'].apply(embedding_service.convert_embedding_str)
        questions_df['embeddings_ques_raw'] = questions_df['embeddings_ques_raw'].apply(embedding_service.convert_embedding_str)
        
        save_dataframe(questions_df, 'questions_embeddings_final.csv')
        logging.info("Completed embeddings generation")
        
    except Exception as e:
        logging.error(f"Error in embeddings generation: {str(e)}")
        raise

    # Step 3: Setup Vector Storage
    logging.info("Step 3: Setting up vector storage...")
    try:
        # Template collection
        vector_dim = len(questions_df['embeddings'].iloc[0])
        qdrant_service.create_collection(Config.QUESTIONS_COLLECTION, vector_dim)
        logging.info(f"Created template collection with dimension {vector_dim}")
        
        # Raw questions collection
        vector_dim_raw = len(questions_df['embeddings_ques_raw'].iloc[0])
        qdrant_service.create_collection(Config.QUESTIONS_RAW_COLLECTION, vector_dim_raw)
        logging.info(f"Created raw questions collection with dimension {vector_dim_raw}")

        # Upload points to both collections
        points = []
        points_raw = []
        for idx, row in questions_df.iterrows():
            points.append(qdrant_service.prepare_point(
                idx, row['embeddings'], row['ques_no'], row['question'], str(row['ques_template'])
            ))
            points_raw.append(qdrant_service.prepare_point(
                idx, row['embeddings_ques_raw'], row['ques_no'], row['question'], str(row['ques_template'])
            ))

        qdrant_service.upload_points(Config.QUESTIONS_COLLECTION, points)
        qdrant_service.upload_points(Config.QUESTIONS_RAW_COLLECTION, points_raw)
        logging.info("Completed vector storage setup")
        
    except Exception as e:
        logging.error(f"Error in vector storage setup: {str(e)}")
        raise

    # Step 4: Process Jobs
    logging.info("Step 4: Processing jobs...")
    jobs_df = data_loader.load_jobs_data()
    logging.info(f"Loaded {len(jobs_df)} jobs")
    
    # Generate job templates and metadata
    max_retries = 5  # Number of retries
    for idx, row in tqdm(jobs_df.iterrows(), desc="Processing jobs"):
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Job analysis
                job_info_template = job_analyzer.analyze_job_description(row['Job Description'])
                
                # Generate metadata templates (JA and JD)
                ja_metadata = metadata_generator.generate_ja_metadata(job_info_template)
                jd_metadata = metadata_generator.generate_jd_metadata(row['Job Description'])
                
                # Generate HyDE questions
                hyde_questions = hyde_generator.generate_hyde_questions(row['Job Description'])
                
                # Update DataFrame
                jobs_df.loc[idx, 'job_info_template'] = str(job_info_template)
                jobs_df.loc[idx, 'job_ja_question_metadata'] = str(ja_metadata)
                jobs_df.loc[idx, 'job_jd_question_metadata'] = str(jd_metadata)
                jobs_df.loc[idx, 'job_questions_hyde_list'] = str(hyde_questions)
                
                # Save intermediate results
                save_dataframe(jobs_df, f'jobs_processed_intermediate_{idx+1}.csv')
                logging.info(f"Processed job {idx+1}")
                
                success = True  # Mark as successful if we get here
                
            except json.JSONDecodeError as e:
                retry_count += 1
                logging.warning(f"Retry {retry_count}/{max_retries} for job {idx+1} due to JSON decode error: {str(e)}")
                if retry_count < max_retries:
                    time.sleep(2)  # Add delay between retries
                    continue
                else:
                    logging.error(f"Failed to process job {idx+1} after {max_retries} retries")
                    # Initialize empty/default values if all retries fail
                    jobs_df.loc[idx, 'job_info_template'] = '{}'
                    jobs_df.loc[idx, 'job_ja_question_metadata'] = '[]'
                    jobs_df.loc[idx, 'job_jd_question_metadata'] = '[]'
                    jobs_df.loc[idx, 'job_questions_hyde_list'] = '[]'
                    
            except Exception as e:
                retry_count += 1
                logging.warning(f"Retry {retry_count}/{max_retries} for job {idx+1} due to error: {str(e)}")
                if retry_count < max_retries:
                    time.sleep(2)  # Add delay between retries
                    continue
                else:
                    logging.error(f"Failed to process job {idx+1} after {max_retries} retries")
                    # Initialize empty/default values if all retries fail
                    jobs_df.loc[idx, 'job_info_template'] = '{}'
                    jobs_df.loc[idx, 'job_ja_question_metadata'] = '[]'
                    jobs_df.loc[idx, 'job_jd_question_metadata'] = '[]'
                    jobs_df.loc[idx, 'job_questions_hyde_list'] = '[]'
    
    save_dataframe(jobs_df, 'jobs_processed_final.csv')

    # Start from Step 5
    logging.info("Step 5: Processing all methods...")
    method_dfs = {
        'ja_original': pd.DataFrame(),
        'ja_reranked': pd.DataFrame(),
        'jd_original': pd.DataFrame(),
        'jd_reranked': pd.DataFrame(),
        'hyde_original': pd.DataFrame(),
        'hyde_reranked': pd.DataFrame()
    }

    for idx, row in tqdm(jobs_df.iterrows(), desc="Processing methods"):
        try:
            # Process each method
            for method, metadata_col, collection_name, is_hyde in [
                ('ja', 'job_ja_question_metadata', Config.QUESTIONS_COLLECTION, False),
                ('jd', 'job_jd_question_metadata', Config.QUESTIONS_COLLECTION, False),
                ('hyde', 'job_questions_hyde_list', Config.QUESTIONS_RAW_COLLECTION, True)
            ]:
                metadata = eval(row[metadata_col])
                original_df_temp, reranked_df_temp = process_templates(
                    templates=metadata,
                    collection_name=collection_name,
                    qdrant_service=qdrant_service,
                    custom_ranker=custom_ranker,
                    embedding_service=embedding_service,
                    result_processor=result_processor,
                    job_number=idx+1,
                    is_hyde=is_hyde
                )
                
                # Update method DataFrames
                method_dfs[f'{method}_original'] = pd.concat([method_dfs[f'{method}_original'], original_df_temp])
                method_dfs[f'{method}_reranked'] = pd.concat([method_dfs[f'{method}_reranked'], reranked_df_temp])
                
                # Save intermediate results
                save_dataframe(method_dfs[f'{method}_original'], f'{method}_original_intermediate_{idx+1}.csv')
                save_dataframe(method_dfs[f'{method}_reranked'], f'{method}_reranked_intermediate_{idx+1}.csv')
                
            logging.info(f"Processed all methods for job {idx+1}")
            
        except Exception as e:
            logging.error(f"Error processing methods for job {idx+1}: {str(e)}")
            continue

    # Reset indices and save final method results
    for name, df in method_dfs.items():
        df = df.reset_index(drop=True)
        save_dataframe(df, f'{name}_final.csv')
        method_dfs[name] = df

    # Step 6: Evaluate and Post-process Results
    logging.info("Step 6: Evaluating and post-processing results...")
    for job_num in range(1, 11):
        try:
            eval_df = data_loader.load_evaluation_data(job_num)
            job_desc = jobs_df.iloc[job_num-1]['Job Description']
            
            for method_name in ['ja', 'jd', 'hyde']:
                for variant in ['original', 'reranked']:
                    method_key = f'{method_name}_{variant}'
                    df = method_dfs[method_key]
                    job_df = df[df['job_number'] == job_num].reset_index(drop=True)
                    
                    if len(job_df) == 0:
                        logging.warning(f"No data found for {method_key} job {job_num}")
                        continue

                    # Convert string representations back to lists if needed
                    ques_nos = eval(job_df['ques_no'].iloc[0]) if isinstance(job_df['ques_no'].iloc[0], str) else job_df['ques_no'].iloc[0]
                    scores = eval(job_df['match_scores'].iloc[0]) if isinstance(job_df['match_scores'].iloc[0], str) else job_df['match_scores'].iloc[0]
                    questions = eval(job_df['question'].iloc[0]) if isinstance(job_df['question'].iloc[0], str) else job_df['question'].iloc[0]
                    
                    # Create list of tuples (question_number, score)
                    final_list = list(zip(ques_nos, scores))
                    
                    # Get evaluation results
                    eval_results = evaluator.evaluate_results(
                        final_list=final_list,
                        eval_df=eval_df,
                        method=method_key
                    )
                    
                    # Get post-processing analysis with retries
                    max_retries = 5
                    retry_count = 0
                    post_analysis = None
                    
                    while retry_count < max_retries:
                        try:
                            post_analysis = post_processor.analyze_question_coverage(
                                job_desc,
                                eval_results['question_order'],
                                questions
                            )
                            break  # Success, exit retry loop
                            
                        except Exception as e:
                            retry_count += 1
                            logging.warning(f"Retry {retry_count}/{max_retries} for post-processing job {job_num}, method {method_key}: {str(e)}")
                            if retry_count < max_retries:
                                time.sleep(2)  # Wait before retrying
                            else:
                                logging.error(f"Failed to get post-processing analysis after {max_retries} retries")
                                post_analysis = {
                                    'missing_aspects': "Error in processing",
                                    'suggested_questions': []
                                }
                    
                    # Combine and save results
                    combined_results = {**eval_results, **(post_analysis or {})}
                    save_dataframe(
                        pd.DataFrame([combined_results]),
                        f'results_{method_key}_job{job_num}.csv',
                        directory='evaluation_results'
                    )
            
            logging.info(f"Completed evaluation for job {job_num}")
            
        except Exception as e:
            logging.error(f"Error in evaluation for job {job_num}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            continue


# def main():
#     # Setup logging
#     setup_logging()
#     logging.info("Starting interview question matching process")
    
#     # Initialize services
#     logging.info("Initializing services...")
#     data_loader = DataLoader()
#     question_analyzer = QuestionAnalyzer()
#     job_analyzer = JobAnalyzer()
#     metadata_generator = MetadataGenerator()
#     hyde_generator = HydeGenerator()
#     embedding_service = EmbeddingService()
#     qdrant_service = QdrantService()
#     custom_ranker = CustomRanker()
#     evaluator = Evaluator()
#     post_processor = PostProcessor()
#     result_processor = ResultProcessor()

#     # Load previously processed data
#     questions_df = pd.read_csv('output/questions_embeddings_final.csv')
#     jobs_df = pd.read_csv('output/jobs_processed_final.csv')
    
#     # Convert embeddings back to lists
#     questions_df['embeddings'] = questions_df['embeddings'].apply(embedding_service.convert_embedding_str)
#     questions_df['embeddings_ques_raw'] = questions_df['embeddings_ques_raw'].apply(embedding_service.convert_embedding_str)

#     # Recreate vector storage from saved data
#     # Template collection
#     vector_dim = len(questions_df['embeddings'].iloc[0])
#     qdrant_service.create_collection(Config.QUESTIONS_COLLECTION, vector_dim)
    
#     # Raw questions collection
#     vector_dim_raw = len(questions_df['embeddings_ques_raw'].iloc[0])
#     qdrant_service.create_collection(Config.QUESTIONS_RAW_COLLECTION, vector_dim_raw)

#     # Upload points to both collections
#     points = []
#     points_raw = []
#     for idx, row in questions_df.iterrows():
#         points.append(qdrant_service.prepare_point(
#             idx, row['embeddings'], row['ques_no'], row['question'], str(row['ques_template'])
#         ))
#         points_raw.append(qdrant_service.prepare_point(
#             idx, row['embeddings_ques_raw'], row['ques_no'], row['question'], str(row['ques_template'])
#         ))

#     qdrant_service.upload_points(Config.QUESTIONS_COLLECTION, points)
#     qdrant_service.upload_points(Config.QUESTIONS_RAW_COLLECTION, points_raw)

#     # Start from Step 5
#     logging.info("Step 5: Processing all methods...")
#     method_dfs = {
#         'ja_original': pd.DataFrame(),
#         'ja_reranked': pd.DataFrame(),
#         'jd_original': pd.DataFrame(),
#         'jd_reranked': pd.DataFrame(),
#         'hyde_original': pd.DataFrame(),
#         'hyde_reranked': pd.DataFrame()
#     }

#     for idx, row in tqdm(jobs_df.iterrows(), desc="Processing methods"):
#         try:
#             # Process each method
#             for method, metadata_col, collection_name, is_hyde in [
#                 ('ja', 'job_ja_question_metadata', Config.QUESTIONS_COLLECTION, False),
#                 ('jd', 'job_jd_question_metadata', Config.QUESTIONS_COLLECTION, False),
#                 ('hyde', 'job_questions_hyde_list', Config.QUESTIONS_RAW_COLLECTION, True)
#             ]:
#                 metadata = eval(row[metadata_col])
#                 original_df_temp, reranked_df_temp = process_templates(
#                     templates=metadata,
#                     collection_name=collection_name,
#                     qdrant_service=qdrant_service,
#                     custom_ranker=custom_ranker,
#                     embedding_service=embedding_service,
#                     result_processor=result_processor,
#                     job_number=idx+1,
#                     is_hyde=is_hyde
#                 )
                
#                 # Update method DataFrames
#                 method_dfs[f'{method}_original'] = pd.concat([method_dfs[f'{method}_original'], original_df_temp])
#                 method_dfs[f'{method}_reranked'] = pd.concat([method_dfs[f'{method}_reranked'], reranked_df_temp])
                
#                 # Save intermediate results
#                 save_dataframe(method_dfs[f'{method}_original'], f'{method}_original_intermediate_{idx+1}.csv')
#                 save_dataframe(method_dfs[f'{method}_reranked'], f'{method}_reranked_intermediate_{idx+1}.csv')
                
#             logging.info(f"Processed all methods for job {idx+1}")
            
#         except Exception as e:
#             logging.error(f"Error processing methods for job {idx+1}: {str(e)}")
#             continue

#     # Reset indices and save final method results
#     for name, df in method_dfs.items():
#         df = df.reset_index(drop=True)
#         save_dataframe(df, f'{name}_final.csv')
#         method_dfs[name] = df

#     # Step 6: Evaluate and Post-process Results
#     logging.info("Step 6: Evaluating and post-processing results...")
#     for job_num in range(1, 11):
#         try:
#             eval_df = data_loader.load_evaluation_data(job_num)
#             job_desc = jobs_df.iloc[job_num-1]['Job Description']
            
#             for method_name in ['ja', 'jd', 'hyde']:
#                 for variant in ['original', 'reranked']:
#                     method_key = f'{method_name}_{variant}'
#                     df = method_dfs[method_key]
#                     job_df = df[df['job_number'] == job_num].reset_index(drop=True)
                    
#                     if len(job_df) == 0:
#                         logging.warning(f"No data found for {method_key} job {job_num}")
#                         continue

#                     # Convert string representations back to lists if needed
#                     ques_nos = eval(job_df['ques_no'].iloc[0]) if isinstance(job_df['ques_no'].iloc[0], str) else job_df['ques_no'].iloc[0]
#                     scores = eval(job_df['match_scores'].iloc[0]) if isinstance(job_df['match_scores'].iloc[0], str) else job_df['match_scores'].iloc[0]
#                     questions = eval(job_df['question'].iloc[0]) if isinstance(job_df['question'].iloc[0], str) else job_df['question'].iloc[0]
                    
#                     # Create list of tuples (question_number, score)
#                     final_list = list(zip(ques_nos, scores))
                    
#                     # Get evaluation results
#                     eval_results = evaluator.evaluate_results(
#                         final_list=final_list,
#                         eval_df=eval_df,
#                         method=method_key
#                     )
                    
#                     # Get post-processing analysis with retries
#                     max_retries = 5
#                     retry_count = 0
#                     post_analysis = None
                    
#                     while retry_count < max_retries:
#                         try:
#                             post_analysis = post_processor.analyze_question_coverage(
#                                 job_desc,
#                                 eval_results['question_order'],
#                                 questions
#                             )
#                             break  # Success, exit retry loop
                            
#                         except Exception as e:
#                             retry_count += 1
#                             logging.warning(f"Retry {retry_count}/{max_retries} for post-processing job {job_num}, method {method_key}: {str(e)}")
#                             if retry_count < max_retries:
#                                 time.sleep(2)  # Wait before retrying
#                             else:
#                                 logging.error(f"Failed to get post-processing analysis after {max_retries} retries")
#                                 post_analysis = {
#                                     'missing_aspects': "Error in processing",
#                                     'suggested_questions': []
#                                 }
                    
#                     # Combine and save results
#                     combined_results = {**eval_results, **(post_analysis or {})}
#                     save_dataframe(
#                         pd.DataFrame([combined_results]),
#                         f'results_{method_key}_job{job_num}.csv',
#                         directory='evaluation_results'
#                     )
            
#             logging.info(f"Completed evaluation for job {job_num}")
            
#         except Exception as e:
#             logging.error(f"Error in evaluation for job {job_num}: {str(e)}")
#             logging.error(f"Traceback: {traceback.format_exc()}")
#             continue


if __name__ == "__main__":
    main()