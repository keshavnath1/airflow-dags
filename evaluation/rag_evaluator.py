"""
RAG Evaluation Framework

Implements comprehensive evaluation metrics for RAG systems including:
- Faithfulness: Does the answer stick to the provided context?
- Correctness: Is the answer factually accurate?
- Contextual Relevancy: Was the retrieved context relevant?
- Answer Relevancy: Does the answer address the question?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
import ast
import subprocess
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class RAGEvaluator:
    """
    Comprehensive RAG evaluation framework that works without external LLM APIs.
    
    Uses rule-based and ML-based approaches for evaluation that can run entirely on CPU.
    """
    
    def __init__(self, use_advanced_metrics: bool = True):
        """
        Initialize RAG evaluator.
        
        Args:
            use_advanced_metrics: Whether to use advanced ML-based metrics
        """
        self.use_advanced_metrics = use_advanced_metrics
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self._setup_evaluation_patterns()
    
    def _setup_evaluation_patterns(self):
        """Setup regex patterns for rule-based evaluation"""
        
        # Patterns for detecting hallucination indicators
        self.hallucination_patterns = [
            r'\b(I think|I believe|probably|maybe|might be|could be)\b',
            r'\b(not sure|uncertain|unclear|don\'t know)\b',
            r'\b(typically|usually|generally|often)\b',  # Overgeneralization
        ]
        
        # Patterns for code-specific evaluation
        self.code_patterns = {
            'function_def': r'def\s+\w+\s*\(',
            'class_def': r'class\s+\w+\s*[\(:]',
            'import_stmt': r'(import|from)\s+\w+',
            'syntax_elements': r'[\{\}\[\]\(\)]',
            'comments': r'#.*$',
            'docstrings': r'""".*?"""',
        }
        
        # Keywords indicating different complexity levels
        self.complexity_keywords = {
            'simple': ['print', 'input', 'variable', 'basic', 'simple'],
            'moderate': ['function', 'loop', 'condition', 'list', 'dictionary'],
            'complex': ['class', 'inheritance', 'algorithm', 'optimization', 'async']
        }
    
    def evaluate_faithfulness(self, question: str, context: str, answer: str) -> Tuple[float, str]:
        """
        Evaluate faithfulness - does the answer stick to the provided context?
        
        Args:
            question: The input question
            context: The retrieved context
            answer: The generated answer
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        # Rule-based faithfulness evaluation
        score = 1.0
        issues = []
        
        # Check for hallucination indicators
        for pattern in self.hallucination_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                score -= 0.2
                issues.append(f"Contains uncertainty indicator: {pattern}")
        
        # Check if answer contains information not in context
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        context_words -= common_words
        answer_words -= common_words
        
        # Check for significant words in answer not in context
        unique_answer_words = answer_words - context_words
        if len(unique_answer_words) > len(answer_words) * 0.3:  # More than 30% unique words
            score -= 0.3
            issues.append("Answer contains significant information not in context")
        
        # Semantic similarity check using TF-IDF
        if self.use_advanced_metrics:
            try:
                docs = [context, answer]
                tfidf_matrix = self.vectorizer.fit_transform(docs)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                
                if similarity < 0.3:  # Low similarity threshold
                    score -= 0.2
                    issues.append(f"Low semantic similarity with context: {similarity:.2f}")
                    
            except Exception as e:
                # Fallback if TF-IDF fails
                pass
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        reasoning = "Faithful answer" if score > 0.7 else f"Issues found: {'; '.join(issues)}"
        
        return score, reasoning
    
    def evaluate_correctness(self, question: str, ground_truth: str, answer: str) -> Tuple[float, str]:
        """
        Evaluate correctness against ground truth.
        
        Args:
            question: The input question
            ground_truth: The reference answer
            answer: The generated answer
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        if not ground_truth or ground_truth.strip() == "":
            return 0.5, "No ground truth provided for comparison"
        
        # Normalize texts for comparison
        gt_normalized = self._normalize_text(ground_truth)
        answer_normalized = self._normalize_text(answer)
        
        # Exact match check
        if gt_normalized == answer_normalized:
            return 1.0, "Exact match with ground truth"
        
        # Semantic similarity using TF-IDF
        try:
            docs = [gt_normalized, answer_normalized]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Score based on similarity
            if similarity >= 0.8:
                score = 0.9
                reasoning = f"High similarity with ground truth: {similarity:.2f}"
            elif similarity >= 0.6:
                score = 0.7
                reasoning = f"Good similarity with ground truth: {similarity:.2f}"
            elif similarity >= 0.4:
                score = 0.5
                reasoning = f"Moderate similarity with ground truth: {similarity:.2f}"
            else:
                score = 0.2
                reasoning = f"Low similarity with ground truth: {similarity:.2f}"
                
        except Exception as e:
            # Fallback to simple word overlap
            gt_words = set(gt_normalized.split())
            answer_words = set(answer_normalized.split())
            
            if len(gt_words) == 0:
                return 0.0, "Empty ground truth"
            
            overlap = len(gt_words.intersection(answer_words))
            score = overlap / len(gt_words)
            reasoning = f"Word overlap score: {score:.2f}"
        
        return score, reasoning
    
    def evaluate_contextual_relevancy(self, question: str, context: str) -> Tuple[float, str]:
        """
        Evaluate if the retrieved context is relevant to the question.
        
        Args:
            question: The input question
            context: The retrieved context
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        if not context or context.strip() == "":
            return 0.0, "Empty context provided"
        
        # Normalize texts
        question_normalized = self._normalize_text(question)
        context_normalized = self._normalize_text(context)
        
        # Extract key terms from question
        question_words = set(question_normalized.split())
        context_words = set(context_normalized.split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why'}
        question_words -= common_words
        context_words -= common_words
        
        if len(question_words) == 0:
            return 0.5, "Question contains only common words"
        
        # Calculate word overlap
        overlap = len(question_words.intersection(context_words))
        word_overlap_score = overlap / len(question_words)
        
        # Semantic similarity using TF-IDF
        semantic_score = 0.5  # Default
        try:
            docs = [question_normalized, context_normalized]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            semantic_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception:
            pass
        
        # Combined score (weighted average)
        final_score = 0.4 * word_overlap_score + 0.6 * semantic_score
        
        if final_score >= 0.7:
            reasoning = f"High relevancy (word overlap: {word_overlap_score:.2f}, semantic: {semantic_score:.2f})"
        elif final_score >= 0.5:
            reasoning = f"Moderate relevancy (word overlap: {word_overlap_score:.2f}, semantic: {semantic_score:.2f})"
        else:
            reasoning = f"Low relevancy (word overlap: {word_overlap_score:.2f}, semantic: {semantic_score:.2f})"
        
        return final_score, reasoning
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> Tuple[float, str]:
        """
        Evaluate if the answer addresses the question.
        
        Args:
            question: The input question
            answer: The generated answer
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        if not answer or answer.strip() == "":
            return 0.0, "Empty answer provided"
        
        # Normalize texts
        question_normalized = self._normalize_text(question)
        answer_normalized = self._normalize_text(answer)
        
        # Check if answer directly addresses question type
        question_type_score = self._evaluate_question_type_match(question_normalized, answer_normalized)
        
        # Semantic similarity
        semantic_score = 0.5  # Default
        try:
            docs = [question_normalized, answer_normalized]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            semantic_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception:
            pass
        
        # Combined score
        final_score = 0.5 * question_type_score + 0.5 * semantic_score
        
        reasoning = f"Question type match: {question_type_score:.2f}, Semantic similarity: {semantic_score:.2f}"
        
        return final_score, reasoning
    
    def _evaluate_question_type_match(self, question: str, answer: str) -> float:
        """Evaluate if answer matches the question type (how, what, etc.)"""
        
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Question type patterns and expected answer patterns
        type_patterns = {
            'how': ['step', 'method', 'way', 'process', 'procedure', 'function', 'def'],
            'what': ['definition', 'meaning', 'is', 'are', 'class', 'variable'],
            'why': ['because', 'reason', 'due to', 'since', 'explanation'],
            'when': ['time', 'date', 'moment', 'during', 'while'],
            'where': ['location', 'place', 'in', 'at', 'directory', 'file'],
            'code': ['def', 'class', 'import', 'return', 'function', '```', 'python']
        }
        
        # Detect question type
        detected_type = None
        for q_type, patterns in type_patterns.items():
            if any(pattern in question_lower for pattern in [q_type]):
                detected_type = q_type
                break
        
        # Special case for code-related questions
        if any(word in question_lower for word in ['code', 'function', 'program', 'script', 'algorithm']):
            detected_type = 'code'
        
        if not detected_type:
            return 0.5  # Neutral score if type not detected
        
        # Check if answer contains expected patterns for the question type
        expected_patterns = type_patterns.get(detected_type, [])
        matches = sum(1 for pattern in expected_patterns if pattern in answer_lower)
        
        if len(expected_patterns) == 0:
            return 0.5
        
        return min(1.0, matches / len(expected_patterns))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (except for code)
        if not any(pattern in text for pattern in ['def ', 'class ', 'import ', 'return ']):
            text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def comprehensive_evaluate(self, question: str, context: str, answer: str, 
                             ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all metrics.
        
        Args:
            question: The input question
            context: The retrieved context
            answer: The generated answer
            ground_truth: Optional reference answer
            
        Returns:
            Dictionary with all evaluation results
        """
        
        results = {}
        
        # Core RAG metrics
        results['faithfulness'] = self.evaluate_faithfulness(question, context, answer)
        results['contextual_relevancy'] = self.evaluate_contextual_relevancy(question, context)
        results['answer_relevancy'] = self.evaluate_answer_relevancy(question, answer)
        
        # Correctness (if ground truth available)
        if ground_truth:
            results['correctness'] = self.evaluate_correctness(question, ground_truth, answer)
        else:
            results['correctness'] = (0.5, "No ground truth provided")
        
        # Calculate overall score
        scores = [result[0] for result in results.values()]
        results['overall_score'] = sum(scores) / len(scores)
        
        # Add metadata
        results['metadata'] = {
            'question_length': len(question.split()),
            'context_length': len(context.split()),
            'answer_length': len(answer.split()),
            'has_ground_truth': ground_truth is not None
        }
        
        return results
    
    def batch_evaluate(self, evaluation_data: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Evaluate multiple samples in batch.
        
        Args:
            evaluation_data: List of dictionaries with keys: question, context, answer, ground_truth (optional)
            
        Returns:
            DataFrame with evaluation results
        """
        
        results = []
        
        for i, sample in enumerate(evaluation_data):
            question = sample.get('question', '')
            context = sample.get('context', '')
            answer = sample.get('answer', '')
            ground_truth = sample.get('ground_truth')
            
            # Run comprehensive evaluation
            eval_result = self.comprehensive_evaluate(question, context, answer, ground_truth)
            
            # Flatten results for DataFrame
            row = {
                'sample_id': i,
                'question': question,
                'answer': answer,
                'faithfulness_score': eval_result['faithfulness'][0],
                'faithfulness_reasoning': eval_result['faithfulness'][1],
                'correctness_score': eval_result['correctness'][0],
                'correctness_reasoning': eval_result['correctness'][1],
                'contextual_relevancy_score': eval_result['contextual_relevancy'][0],
                'contextual_relevancy_reasoning': eval_result['contextual_relevancy'][1],
                'answer_relevancy_score': eval_result['answer_relevancy'][0],
                'answer_relevancy_reasoning': eval_result['answer_relevancy'][1],
                'overall_score': eval_result['overall_score'],
                'question_length': eval_result['metadata']['question_length'],
                'context_length': eval_result['metadata']['context_length'],
                'answer_length': eval_result['metadata']['answer_length']
            }
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def generate_evaluation_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results_df: DataFrame from batch_evaluate
            
        Returns:
            Dictionary with summary statistics and insights
        """
        
        report = {
            'summary_statistics': {
                'total_samples': len(results_df),
                'average_scores': {
                    'faithfulness': results_df['faithfulness_score'].mean(),
                    'correctness': results_df['correctness_score'].mean(),
                    'contextual_relevancy': results_df['contextual_relevancy_score'].mean(),
                    'answer_relevancy': results_df['answer_relevancy_score'].mean(),
                    'overall': results_df['overall_score'].mean()
                },
                'score_distributions': {
                    'faithfulness': results_df['faithfulness_score'].describe().to_dict(),
                    'correctness': results_df['correctness_score'].describe().to_dict(),
                    'contextual_relevancy': results_df['contextual_relevancy_score'].describe().to_dict(),
                    'answer_relevancy': results_df['answer_relevancy_score'].describe().to_dict()
                }
            },
            'performance_insights': {
                'high_performing_samples': len(results_df[results_df['overall_score'] >= 0.8]),
                'low_performing_samples': len(results_df[results_df['overall_score'] < 0.5]),
                'most_common_issues': self._analyze_common_issues(results_df)
            },
            'recommendations': self._generate_recommendations(results_df)
        }
        
        return report
    
    def _analyze_common_issues(self, results_df: pd.DataFrame) -> List[str]:
        """Analyze common issues from evaluation results"""
        
        issues = []
        
        # Check for low scores in specific metrics
        if results_df['faithfulness_score'].mean() < 0.6:
            issues.append("Low faithfulness scores - model may be hallucinating")
        
        if results_df['contextual_relevancy_score'].mean() < 0.6:
            issues.append("Low contextual relevancy - retrieval system needs improvement")
        
        if results_df['answer_relevancy_score'].mean() < 0.6:
            issues.append("Low answer relevancy - model not addressing questions properly")
        
        # Check for high variance
        if results_df['overall_score'].std() > 0.3:
            issues.append("High score variance - inconsistent model performance")
        
        return issues
    
    def _generate_recommendations(self, results_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        avg_scores = {
            'faithfulness': results_df['faithfulness_score'].mean(),
            'correctness': results_df['correctness_score'].mean(),
            'contextual_relevancy': results_df['contextual_relevancy_score'].mean(),
            'answer_relevancy': results_df['answer_relevancy_score'].mean()
        }
        
        # Find the lowest scoring metric
        lowest_metric = min(avg_scores, key=avg_scores.get)
        
        if lowest_metric == 'faithfulness':
            recommendations.append("Improve model training to reduce hallucination")
            recommendations.append("Add more context-grounding in training data")
        elif lowest_metric == 'contextual_relevancy':
            recommendations.append("Improve retrieval system with better embeddings")
            recommendations.append("Increase retrieval corpus quality and coverage")
        elif lowest_metric == 'answer_relevancy':
            recommendations.append("Fine-tune model on question-answer pairs")
            recommendations.append("Improve prompt engineering for better question understanding")
        elif lowest_metric == 'correctness':
            recommendations.append("Increase training data quality and accuracy")
            recommendations.append("Add fact-checking mechanisms to the pipeline")
        
        return recommendations

