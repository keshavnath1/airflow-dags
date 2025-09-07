"""
Evaluation Pipeline

Comprehensive evaluation pipeline that integrates RAG evaluation and code evaluation
to provide end-to-end assessment of RAG-based code generation systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .rag_evaluator import RAGEvaluator
from .code_evaluator import CodeExecutionEvaluator


class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline for RAG-based code generation systems.
    
    Combines RAG evaluation metrics with code-specific evaluation to provide
    a complete assessment of system performance.
    """
    
    def __init__(self, 
                 use_advanced_rag_metrics: bool = True,
                 code_timeout: int = 10,
                 max_memory_mb: int = 100):
        """
        Initialize evaluation pipeline.
        
        Args:
            use_advanced_rag_metrics: Whether to use advanced RAG metrics
            code_timeout: Timeout for code execution in seconds
            max_memory_mb: Maximum memory for code execution
        """
        
        self.rag_evaluator = RAGEvaluator(use_advanced_metrics=use_advanced_rag_metrics)
        self.code_evaluator = CodeExecutionEvaluator(timeout=code_timeout, max_memory_mb=max_memory_mb)
        
        self.evaluation_history = []
    
    def evaluate_single_sample(self, 
                              question: str,
                              context: str, 
                              generated_answer: str,
                              ground_truth: Optional[str] = None,
                              test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate a single RAG-generated code sample.
        
        Args:
            question: The input question/prompt
            context: Retrieved context used for generation
            generated_answer: Generated code/answer
            ground_truth: Optional reference answer
            test_cases: Optional test cases for code execution
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'input': {
                'question': question,
                'context': context,
                'generated_answer': generated_answer,
                'has_ground_truth': ground_truth is not None,
                'has_test_cases': test_cases is not None
            }
        }
        
        # RAG Evaluation
        print("🔍 Running RAG evaluation...")
        rag_results = self.rag_evaluator.comprehensive_evaluate(
            question, context, generated_answer, ground_truth
        )
        results['rag_evaluation'] = rag_results
        
        # Code Evaluation (if the answer contains code)
        if self._contains_code(generated_answer):
            print("💻 Running code evaluation...")
            
            # Extract code from the answer
            code = self._extract_code(generated_answer)
            
            if code:
                code_results = self.code_evaluator.comprehensive_code_evaluate(code, test_cases)
                results['code_evaluation'] = code_results
            else:
                results['code_evaluation'] = {
                    'error': 'Could not extract valid code from answer'
                }
        else:
            results['code_evaluation'] = {
                'skipped': 'Answer does not contain code'
            }
        
        # Calculate combined score
        results['combined_score'] = self._calculate_combined_score(results)
        
        # Add to history
        self.evaluation_history.append(results)
        
        return results
    
    def evaluate_batch(self, evaluation_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluate multiple samples in batch.
        
        Args:
            evaluation_data: List of evaluation samples with keys:
                - question: Input question
                - context: Retrieved context  
                - generated_answer: Generated answer
                - ground_truth: Optional reference answer
                - test_cases: Optional test cases
                
        Returns:
            DataFrame with evaluation results
        """
        
        print(f"📊 Starting batch evaluation of {len(evaluation_data)} samples...")
        
        batch_results = []
        
        for i, sample in enumerate(evaluation_data):
            print(f"Evaluating sample {i+1}/{len(evaluation_data)}")
            
            # Extract sample data
            question = sample.get('question', '')
            context = sample.get('context', '')
            generated_answer = sample.get('generated_answer', '')
            ground_truth = sample.get('ground_truth')
            test_cases = sample.get('test_cases')
            
            # Evaluate single sample
            try:
                result = self.evaluate_single_sample(
                    question, context, generated_answer, ground_truth, test_cases
                )
                
                # Flatten results for DataFrame
                flattened = self._flatten_evaluation_result(result, i)
                batch_results.append(flattened)
                
            except Exception as e:
                print(f"⚠️ Error evaluating sample {i}: {str(e)}")
                # Add error record
                error_record = {
                    'sample_id': i,
                    'error': str(e),
                    'question': question,
                    'generated_answer': generated_answer
                }
                batch_results.append(error_record)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(batch_results)
        
        print("✅ Batch evaluation completed!")
        return results_df
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code"""
        
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'return ',
            '```python', '```', 'print(', 'if __name__',
            'for ', 'while ', 'try:', 'except:'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)
    
    def _extract_code(self, text: str) -> str:
        """Extract code from text (handles markdown code blocks)"""
        
        # Try to extract from markdown code blocks first
        import re
        
        # Look for ```python or ``` code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        # Look for generic ``` blocks
        code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, check if the entire text looks like code
        if self._looks_like_code(text):
            return text.strip()
        
        return ""
    
    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to determine if text looks like code"""
        
        lines = text.strip().split('\n')
        code_line_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for code-like patterns
            if (line.startswith('def ') or 
                line.startswith('class ') or
                line.startswith('import ') or
                line.startswith('from ') or
                line.startswith('return ') or
                line.startswith('print(') or
                '=' in line and not line.startswith('#')):
                code_line_count += 1
        
        # If more than 30% of lines look like code, consider it code
        return code_line_count / len(lines) > 0.3 if lines else False
    
    def _calculate_combined_score(self, results: Dict[str, Any]) -> float:
        """Calculate combined score from RAG and code evaluation results"""
        
        rag_score = results.get('rag_evaluation', {}).get('overall_score', 0.0)
        
        code_eval = results.get('code_evaluation', {})
        
        if 'overall_score' in code_eval:
            code_score = code_eval['overall_score']
            # Weighted average: 60% RAG, 40% Code
            combined_score = 0.6 * rag_score + 0.4 * code_score
        else:
            # If no code evaluation, use only RAG score
            combined_score = rag_score
        
        return combined_score
    
    def _flatten_evaluation_result(self, result: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
        """Flatten nested evaluation result for DataFrame"""
        
        flattened = {
            'sample_id': sample_id,
            'timestamp': result['timestamp'],
            'question': result['input']['question'],
            'generated_answer': result['input']['generated_answer'],
            'has_ground_truth': result['input']['has_ground_truth'],
            'has_test_cases': result['input']['has_test_cases'],
            'combined_score': result['combined_score']
        }
        
        # RAG evaluation results
        rag_eval = result.get('rag_evaluation', {})
        if rag_eval:
            flattened.update({
                'rag_faithfulness_score': rag_eval.get('faithfulness', (0, ''))[0],
                'rag_faithfulness_reasoning': rag_eval.get('faithfulness', (0, ''))[1],
                'rag_correctness_score': rag_eval.get('correctness', (0, ''))[0],
                'rag_correctness_reasoning': rag_eval.get('correctness', (0, ''))[1],
                'rag_contextual_relevancy_score': rag_eval.get('contextual_relevancy', (0, ''))[0],
                'rag_contextual_relevancy_reasoning': rag_eval.get('contextual_relevancy', (0, ''))[1],
                'rag_answer_relevancy_score': rag_eval.get('answer_relevancy', (0, ''))[0],
                'rag_answer_relevancy_reasoning': rag_eval.get('answer_relevancy', (0, ''))[1],
                'rag_overall_score': rag_eval.get('overall_score', 0.0)
            })
        
        # Code evaluation results
        code_eval = result.get('code_evaluation', {})
        if 'overall_score' in code_eval:
            flattened.update({
                'code_syntax_score': code_eval.get('syntax', (0, ''))[0],
                'code_syntax_reasoning': code_eval.get('syntax', (0, ''))[1],
                'code_security_score': code_eval.get('security', (0, ''))[0],
                'code_security_reasoning': code_eval.get('security', (0, ''))[1],
                'code_execution_score': code_eval.get('execution', (0, ''))[0],
                'code_execution_reasoning': code_eval.get('execution', (0, ''))[1],
                'code_quality_score': code_eval.get('quality', (0, ''))[0],
                'code_quality_reasoning': code_eval.get('quality', (0, ''))[1],
                'code_overall_score': code_eval.get('overall_score', 0.0)
            })
        elif 'error' in code_eval:
            flattened['code_evaluation_error'] = code_eval['error']
        elif 'skipped' in code_eval:
            flattened['code_evaluation_status'] = code_eval['skipped']
        
        return flattened
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results_df: DataFrame from evaluate_batch
            
        Returns:
            Dictionary with comprehensive report
        """
        
        report = {
            'evaluation_summary': {
                'total_samples': len(results_df),
                'evaluation_timestamp': datetime.now().isoformat(),
                'samples_with_ground_truth': results_df['has_ground_truth'].sum(),
                'samples_with_test_cases': results_df['has_test_cases'].sum(),
                'samples_with_code': len(results_df[results_df.get('code_overall_score', pd.Series()).notna()])
            },
            'performance_metrics': {
                'combined_score': {
                    'mean': results_df['combined_score'].mean(),
                    'std': results_df['combined_score'].std(),
                    'min': results_df['combined_score'].min(),
                    'max': results_df['combined_score'].max(),
                    'median': results_df['combined_score'].median()
                }
            }
        }
        
        # RAG metrics (if available)
        rag_columns = [col for col in results_df.columns if col.startswith('rag_') and col.endswith('_score')]
        if rag_columns:
            report['rag_metrics'] = {}
            for col in rag_columns:
                metric_name = col.replace('rag_', '').replace('_score', '')
                report['rag_metrics'][metric_name] = {
                    'mean': results_df[col].mean(),
                    'std': results_df[col].std()
                }
        
        # Code metrics (if available)
        code_columns = [col for col in results_df.columns if col.startswith('code_') and col.endswith('_score')]
        if code_columns:
            report['code_metrics'] = {}
            for col in code_columns:
                metric_name = col.replace('code_', '').replace('_score', '')
                report['code_metrics'][metric_name] = {
                    'mean': results_df[col].mean(),
                    'std': results_df[col].std()
                }
        
        # Performance analysis
        report['performance_analysis'] = self._analyze_performance(results_df)
        
        # Recommendations
        report['recommendations'] = self._generate_comprehensive_recommendations(results_df)
        
        return report
    
    def _analyze_performance(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance patterns"""
        
        analysis = {}
        
        # Score distribution
        combined_scores = results_df['combined_score']
        analysis['score_distribution'] = {
            'excellent': len(combined_scores[combined_scores >= 0.8]) / len(combined_scores),
            'good': len(combined_scores[(combined_scores >= 0.6) & (combined_scores < 0.8)]) / len(combined_scores),
            'fair': len(combined_scores[(combined_scores >= 0.4) & (combined_scores < 0.6)]) / len(combined_scores),
            'poor': len(combined_scores[combined_scores < 0.4]) / len(combined_scores)
        }
        
        # Correlation analysis (if both RAG and code metrics available)
        if 'rag_overall_score' in results_df.columns and 'code_overall_score' in results_df.columns:
            correlation = results_df['rag_overall_score'].corr(results_df['code_overall_score'])
            analysis['rag_code_correlation'] = correlation
        
        # Common failure patterns
        analysis['failure_patterns'] = self._identify_failure_patterns(results_df)
        
        return analysis
    
    def _identify_failure_patterns(self, results_df: pd.DataFrame) -> List[str]:
        """Identify common failure patterns"""
        
        patterns = []
        
        # Low combined scores
        low_scores = results_df[results_df['combined_score'] < 0.5]
        if len(low_scores) > len(results_df) * 0.2:  # More than 20% low scores
            patterns.append(f"High failure rate: {len(low_scores)}/{len(results_df)} samples scored below 0.5")
        
        # RAG-specific issues
        if 'rag_faithfulness_score' in results_df.columns:
            low_faithfulness = results_df[results_df['rag_faithfulness_score'] < 0.6]
            if len(low_faithfulness) > len(results_df) * 0.3:
                patterns.append("High hallucination rate detected")
        
        # Code-specific issues
        if 'code_syntax_score' in results_df.columns:
            syntax_errors = results_df[results_df['code_syntax_score'] < 1.0]
            if len(syntax_errors) > 0:
                patterns.append(f"Syntax errors in {len(syntax_errors)} samples")
        
        return patterns
    
    def _generate_comprehensive_recommendations(self, results_df: pd.DataFrame) -> List[str]:
        """Generate comprehensive recommendations"""
        
        recommendations = []
        
        # Overall performance
        avg_combined_score = results_df['combined_score'].mean()
        if avg_combined_score < 0.6:
            recommendations.append("Overall system performance needs improvement")
        
        # RAG-specific recommendations
        if 'rag_overall_score' in results_df.columns:
            avg_rag_score = results_df['rag_overall_score'].mean()
            if avg_rag_score < 0.7:
                recommendations.append("Improve RAG pipeline (retrieval and generation)")
        
        # Code-specific recommendations
        if 'code_overall_score' in results_df.columns:
            avg_code_score = results_df['code_overall_score'].mean()
            if avg_code_score < 0.7:
                recommendations.append("Improve code generation quality and execution")
        
        # Data quality recommendations
        if results_df['has_ground_truth'].mean() < 0.5:
            recommendations.append("Increase ground truth coverage for better evaluation")
        
        if results_df['has_test_cases'].mean() < 0.3:
            recommendations.append("Add more test cases for comprehensive code evaluation")
        
        return recommendations
    
    def save_evaluation_results(self, results_df: pd.DataFrame, output_dir: str, 
                               experiment_name: str = "rag_evaluation") -> Dict[str, str]:
        """
        Save evaluation results to files.
        
        Args:
            results_df: Evaluation results DataFrame
            output_dir: Output directory
            experiment_name: Name for the experiment
            
        Returns:
            Dictionary with saved file paths
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{experiment_name}_{timestamp}"
        
        saved_files = {}
        
        # Save DataFrame as CSV
        csv_path = os.path.join(output_dir, f"{base_filename}_results.csv")
        results_df.to_csv(csv_path, index=False)
        saved_files['results_csv'] = csv_path
        
        # Generate and save comprehensive report
        report = self.generate_comprehensive_report(results_df)
        report_path = os.path.join(output_dir, f"{base_filename}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        saved_files['report_json'] = report_path
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"RAG Evaluation Summary - {experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Samples: {len(results_df)}\n")
            f.write(f"Average Combined Score: {results_df['combined_score'].mean():.3f}\n")
            f.write(f"Score Standard Deviation: {results_df['combined_score'].std():.3f}\n")
            f.write(f"Samples with Ground Truth: {results_df['has_ground_truth'].sum()}\n")
            f.write(f"Samples with Test Cases: {results_df['has_test_cases'].sum()}\n")
            
            if 'rag_overall_score' in results_df.columns:
                f.write(f"Average RAG Score: {results_df['rag_overall_score'].mean():.3f}\n")
            
            if 'code_overall_score' in results_df.columns:
                f.write(f"Average Code Score: {results_df['code_overall_score'].mean():.3f}\n")
        
        saved_files['summary_txt'] = summary_path
        
        print(f"📁 Evaluation results saved to {output_dir}")
        print(f"   - Results CSV: {csv_path}")
        print(f"   - Report JSON: {report_path}")
        print(f"   - Summary TXT: {summary_path}")
        
        return saved_files

