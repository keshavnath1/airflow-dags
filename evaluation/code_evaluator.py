"""
Code Execution Evaluator

Specialized evaluator for code generation tasks that includes:
- Syntax validation
- Execution testing
- Code quality metrics
- Security checks
"""

import ast
import subprocess
import tempfile
import os
import sys
import re
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class CodeExecutionEvaluator:
    """
    Evaluator specifically designed for code generation tasks.
    
    Provides safe code execution, syntax validation, and quality metrics.
    """
    
    def __init__(self, timeout: int = 10, max_memory_mb: int = 100):
        """
        Initialize code evaluator.
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self._setup_security_patterns()
    
    def _setup_security_patterns(self):
        """Setup patterns for detecting potentially unsafe code"""
        
        self.unsafe_patterns = [
            r'\bos\.system\b',
            r'\bsubprocess\.',
            r'\beval\b',
            r'\bexec\b',
            r'\b__import__\b',
            r'\bopen\s*\(',
            r'\bfile\s*\(',
            r'\binput\s*\(',
            r'\braw_input\s*\(',
            r'\bimport\s+os\b',
            r'\bfrom\s+os\b',
            r'\bimport\s+subprocess\b',
            r'\bfrom\s+subprocess\b'
        ]
        
        self.quality_patterns = {
            'has_docstring': r'""".*?"""',
            'has_comments': r'#.*$',
            'has_type_hints': r':\s*\w+\s*=',
            'follows_pep8_naming': r'def\s+[a-z_][a-z0-9_]*\s*\(',
            'has_error_handling': r'\btry\b|\bexcept\b|\braise\b'
        }
    
    def evaluate_syntax(self, code: str) -> Tuple[float, str]:
        """
        Evaluate code syntax validity.
        
        Args:
            code: Python code to evaluate
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        if not code or code.strip() == "":
            return 0.0, "Empty code provided"
        
        try:
            # Parse the code using AST
            ast.parse(code)
            return 1.0, "Code is syntactically valid"
            
        except SyntaxError as e:
            return 0.0, f"Syntax error: {str(e)}"
        except Exception as e:
            return 0.0, f"Parsing error: {str(e)}"
    
    def evaluate_security(self, code: str) -> Tuple[float, str]:
        """
        Evaluate code for potential security issues.
        
        Args:
            code: Python code to evaluate
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        if not code:
            return 1.0, "No code to evaluate"
        
        security_issues = []
        
        for pattern in self.unsafe_patterns:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                security_issues.append(f"Potentially unsafe pattern: {pattern}")
        
        if not security_issues:
            return 1.0, "No security issues detected"
        
        # Score decreases with number of issues
        score = max(0.0, 1.0 - (len(security_issues) * 0.2))
        reasoning = f"Security issues found: {'; '.join(security_issues)}"
        
        return score, reasoning
    
    def evaluate_execution(self, code: str, test_cases: Optional[List[Dict]] = None) -> Tuple[float, str]:
        """
        Evaluate code execution with optional test cases.
        
        Args:
            code: Python code to execute
            test_cases: Optional list of test cases with 'input' and 'expected_output'
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        # First check syntax
        syntax_score, syntax_reason = self.evaluate_syntax(code)
        if syntax_score == 0.0:
            return 0.0, f"Cannot execute due to syntax error: {syntax_reason}"
        
        # Check security
        security_score, security_reason = self.evaluate_security(code)
        if security_score < 0.5:
            return 0.0, f"Code execution blocked due to security concerns: {security_reason}"
        
        # Execute code
        execution_result = self._safe_execute(code)
        
        if not execution_result['success']:
            return 0.0, f"Execution failed: {execution_result['error']}"
        
        # If no test cases, just check if it runs
        if not test_cases:
            return 1.0, "Code executed successfully without errors"
        
        # Run test cases
        return self._run_test_cases(code, test_cases)
    
    def _safe_execute(self, code: str) -> Dict[str, Any]:
        """
        Safely execute code in a controlled environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary with execution results
        """
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout and capture output
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f'Code execution timed out after {self.timeout} seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Execution error: {str(e)}'
            }
    
    def _run_test_cases(self, code: str, test_cases: List[Dict]) -> Tuple[float, str]:
        """
        Run test cases against the code.
        
        Args:
            code: Python code to test
            test_cases: List of test cases
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        if not test_cases:
            return 1.0, "No test cases provided"
        
        passed_tests = 0
        failed_tests = []
        
        for i, test_case in enumerate(test_cases):
            test_input = test_case.get('input', '')
            expected_output = test_case.get('expected_output', '')
            
            # Create test code
            test_code = f"""
{code}

# Test case {i+1}
try:
    {test_input}
    print("TEST_OUTPUT:", result)
except Exception as e:
    print("TEST_ERROR:", str(e))
"""
            
            # Execute test
            execution_result = self._safe_execute(test_code)
            
            if execution_result['success']:
                output = execution_result['output']
                
                # Extract test output
                if "TEST_OUTPUT:" in output:
                    actual_output = output.split("TEST_OUTPUT:")[1].strip()
                    
                    # Compare with expected output
                    if self._compare_outputs(actual_output, expected_output):
                        passed_tests += 1
                    else:
                        failed_tests.append(f"Test {i+1}: Expected '{expected_output}', got '{actual_output}'")
                else:
                    failed_tests.append(f"Test {i+1}: No output produced")
            else:
                failed_tests.append(f"Test {i+1}: Execution error - {execution_result['error']}")
        
        score = passed_tests / len(test_cases)
        
        if score == 1.0:
            reasoning = f"All {len(test_cases)} test cases passed"
        else:
            reasoning = f"{passed_tests}/{len(test_cases)} tests passed. Failures: {'; '.join(failed_tests[:3])}"
        
        return score, reasoning
    
    def _compare_outputs(self, actual: str, expected: str) -> bool:
        """
        Compare actual and expected outputs with some tolerance.
        
        Args:
            actual: Actual output from code execution
            expected: Expected output
            
        Returns:
            True if outputs match within tolerance
        """
        
        # Normalize whitespace
        actual = actual.strip()
        expected = expected.strip()
        
        # Exact match
        if actual == expected:
            return True
        
        # Try numeric comparison for numbers
        try:
            actual_num = float(actual)
            expected_num = float(expected)
            return abs(actual_num - expected_num) < 1e-6
        except ValueError:
            pass
        
        # Case-insensitive comparison
        if actual.lower() == expected.lower():
            return True
        
        return False
    
    def evaluate_code_quality(self, code: str) -> Tuple[float, str]:
        """
        Evaluate code quality based on best practices.
        
        Args:
            code: Python code to evaluate
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        if not code or code.strip() == "":
            return 0.0, "Empty code provided"
        
        quality_scores = {}
        
        # Check for various quality indicators
        for quality_name, pattern in self.quality_patterns.items():
            if re.search(pattern, code, re.MULTILINE | re.DOTALL):
                quality_scores[quality_name] = 1.0
            else:
                quality_scores[quality_name] = 0.0
        
        # Additional quality checks
        lines = code.split('\n')
        
        # Check line length (PEP 8 recommends max 79 characters)
        long_lines = [i for i, line in enumerate(lines) if len(line) > 79]
        if long_lines:
            quality_scores['line_length'] = max(0.0, 1.0 - len(long_lines) / len(lines))
        else:
            quality_scores['line_length'] = 1.0
        
        # Check for meaningful variable names
        try:
            tree = ast.parse(code)
            variable_names = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    variable_names.append(node.id)
            
            # Check for single-letter variables (except common ones like i, j, x, y)
            single_letter_vars = [name for name in variable_names 
                                if len(name) == 1 and name not in ['i', 'j', 'x', 'y', 'n']]
            
            if single_letter_vars:
                quality_scores['meaningful_names'] = max(0.0, 1.0 - len(single_letter_vars) / len(variable_names))
            else:
                quality_scores['meaningful_names'] = 1.0
                
        except Exception:
            quality_scores['meaningful_names'] = 0.5  # Neutral score if can't parse
        
        # Calculate overall quality score
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        
        # Generate reasoning
        good_practices = [name for name, score in quality_scores.items() if score > 0.5]
        poor_practices = [name for name, score in quality_scores.items() if score <= 0.5]
        
        reasoning_parts = []
        if good_practices:
            reasoning_parts.append(f"Good practices: {', '.join(good_practices)}")
        if poor_practices:
            reasoning_parts.append(f"Areas for improvement: {', '.join(poor_practices)}")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Code quality assessment completed"
        
        return overall_score, reasoning
    
    def comprehensive_code_evaluate(self, code: str, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run comprehensive code evaluation.
        
        Args:
            code: Python code to evaluate
            test_cases: Optional test cases
            
        Returns:
            Dictionary with all evaluation results
        """
        
        results = {}
        
        # Core code metrics
        results['syntax'] = self.evaluate_syntax(code)
        results['security'] = self.evaluate_security(code)
        results['execution'] = self.evaluate_execution(code, test_cases)
        results['quality'] = self.evaluate_code_quality(code)
        
        # Calculate overall score
        scores = [result[0] for result in results.values()]
        results['overall_score'] = sum(scores) / len(scores)
        
        # Add metadata
        results['metadata'] = {
            'code_length': len(code),
            'line_count': len(code.split('\n')),
            'has_test_cases': test_cases is not None,
            'test_case_count': len(test_cases) if test_cases else 0
        }
        
        return results
    
    def batch_evaluate_code(self, code_samples: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluate multiple code samples in batch.
        
        Args:
            code_samples: List of dictionaries with 'code' and optional 'test_cases'
            
        Returns:
            DataFrame with evaluation results
        """
        
        results = []
        
        for i, sample in enumerate(code_samples):
            code = sample.get('code', '')
            test_cases = sample.get('test_cases')
            
            # Run comprehensive evaluation
            eval_result = self.comprehensive_code_evaluate(code, test_cases)
            
            # Flatten results for DataFrame
            row = {
                'sample_id': i,
                'code': code,
                'syntax_score': eval_result['syntax'][0],
                'syntax_reasoning': eval_result['syntax'][1],
                'security_score': eval_result['security'][0],
                'security_reasoning': eval_result['security'][1],
                'execution_score': eval_result['execution'][0],
                'execution_reasoning': eval_result['execution'][1],
                'quality_score': eval_result['quality'][0],
                'quality_reasoning': eval_result['quality'][1],
                'overall_score': eval_result['overall_score'],
                'code_length': eval_result['metadata']['code_length'],
                'line_count': eval_result['metadata']['line_count'],
                'has_test_cases': eval_result['metadata']['has_test_cases']
            }
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def generate_code_quality_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate code quality report from evaluation results.
        
        Args:
            results_df: DataFrame from batch_evaluate_code
            
        Returns:
            Dictionary with code quality insights
        """
        
        report = {
            'summary_statistics': {
                'total_samples': len(results_df),
                'average_scores': {
                    'syntax': results_df['syntax_score'].mean(),
                    'security': results_df['security_score'].mean(),
                    'execution': results_df['execution_score'].mean(),
                    'quality': results_df['quality_score'].mean(),
                    'overall': results_df['overall_score'].mean()
                },
                'pass_rates': {
                    'syntax_valid': len(results_df[results_df['syntax_score'] == 1.0]) / len(results_df),
                    'security_safe': len(results_df[results_df['security_score'] >= 0.8]) / len(results_df),
                    'execution_successful': len(results_df[results_df['execution_score'] >= 0.8]) / len(results_df),
                    'high_quality': len(results_df[results_df['quality_score'] >= 0.7]) / len(results_df)
                }
            },
            'code_characteristics': {
                'average_code_length': results_df['code_length'].mean(),
                'average_line_count': results_df['line_count'].mean(),
                'samples_with_tests': results_df['has_test_cases'].sum()
            },
            'common_issues': self._analyze_code_issues(results_df),
            'recommendations': self._generate_code_recommendations(results_df)
        }
        
        return report
    
    def _analyze_code_issues(self, results_df: pd.DataFrame) -> List[str]:
        """Analyze common code issues"""
        
        issues = []
        
        # Check for common problems
        syntax_failures = len(results_df[results_df['syntax_score'] < 1.0])
        if syntax_failures > len(results_df) * 0.1:  # More than 10% syntax errors
            issues.append(f"High syntax error rate: {syntax_failures}/{len(results_df)} samples")
        
        security_issues = len(results_df[results_df['security_score'] < 0.8])
        if security_issues > 0:
            issues.append(f"Security concerns in {security_issues} samples")
        
        execution_failures = len(results_df[results_df['execution_score'] < 0.5])
        if execution_failures > len(results_df) * 0.2:  # More than 20% execution failures
            issues.append(f"High execution failure rate: {execution_failures}/{len(results_df)} samples")
        
        quality_issues = len(results_df[results_df['quality_score'] < 0.5])
        if quality_issues > len(results_df) * 0.3:  # More than 30% low quality
            issues.append(f"Code quality concerns in {quality_issues} samples")
        
        return issues
    
    def _generate_code_recommendations(self, results_df: pd.DataFrame) -> List[str]:
        """Generate recommendations for code improvement"""
        
        recommendations = []
        
        avg_scores = {
            'syntax': results_df['syntax_score'].mean(),
            'security': results_df['security_score'].mean(),
            'execution': results_df['execution_score'].mean(),
            'quality': results_df['quality_score'].mean()
        }
        
        # Find areas needing improvement
        if avg_scores['syntax'] < 0.9:
            recommendations.append("Improve code generation to reduce syntax errors")
            recommendations.append("Add syntax validation to the training pipeline")
        
        if avg_scores['security'] < 0.8:
            recommendations.append("Filter training data to remove unsafe code patterns")
            recommendations.append("Add security-aware training objectives")
        
        if avg_scores['execution'] < 0.7:
            recommendations.append("Include more executable code examples in training")
            recommendations.append("Add execution testing to the training loop")
        
        if avg_scores['quality'] < 0.6:
            recommendations.append("Train on high-quality code repositories")
            recommendations.append("Add code style and best practices to training objectives")
        
        return recommendations

