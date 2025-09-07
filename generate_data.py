"""
Data Generation Script for RAG Pipeline

This script generates synthetic data for training and testing RAG models.
It creates realistic code generation datasets with questions, contexts, and answers.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_synthetic_data(num_samples=1000, output_path="data/synthetic_data.parquet"):
    """
    Generate synthetic data for RAG training and evaluation
    
    Args:
        num_samples (int): Number of samples to generate
        output_path (str): Path to save the generated data
    
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample questions for code generation
    question_templates = [
        "How do I create a function to {}?",
        "Write a Python function that {}",
        "Can you help me write code to {}?",
        "What's the best way to {} in Python?",
        "Create a function that {}",
        "How can I implement {} functionality?",
        "Write a class that {}",
        "Show me how to {} using Python",
        "Create a method to {}",
        "How do I write a function for {}?"
    ]
    
    # Sample tasks for code generation
    tasks = [
        "add two numbers",
        "multiply two numbers", 
        "calculate the factorial of a number",
        "check if a number is prime",
        "reverse a string",
        "find the maximum value in a list",
        "sort a list of numbers",
        "count vowels in a string",
        "calculate the fibonacci sequence",
        "check if a string is a palindrome",
        "find the sum of elements in a list",
        "remove duplicates from a list",
        "convert celsius to fahrenheit",
        "calculate the area of a circle",
        "find the greatest common divisor",
        "merge two sorted lists",
        "implement binary search",
        "create a simple calculator",
        "validate an email address",
        "generate random numbers"
    ]
    
    # Sample contexts (documentation snippets)
    contexts = [
        "Python functions are defined using the 'def' keyword followed by the function name and parameters in parentheses. The function body is indented and contains the code to be executed.",
        "In Python, you can use built-in functions like sum(), max(), min(), and len() to perform common operations on lists and other iterables.",
        "String manipulation in Python can be done using various methods like split(), join(), replace(), and slicing operations.",
        "Python's math module provides mathematical functions like sqrt(), pow(), factorial(), and trigonometric functions.",
        "List comprehensions in Python provide a concise way to create lists based on existing lists or other iterables.",
        "Python's random module can be used to generate random numbers, choose random elements, and shuffle sequences.",
        "Exception handling in Python uses try-except blocks to catch and handle errors gracefully.",
        "Python classes are defined using the 'class' keyword and can contain methods and attributes.",
        "Regular expressions in Python are handled by the 're' module for pattern matching and text processing.",
        "Python's itertools module provides functions for creating iterators and working with sequences efficiently."
    ]
    
    # Sample code answers
    code_templates = [
        """def {func_name}({params}):
    \"\"\"
    {description}
    
    Args:
        {param_docs}
    
    Returns:
        {return_doc}
    \"\"\"
    {implementation}
    return result""",
        
        """def {func_name}({params}):
    # {description}
    {implementation}
    return result""",
        
        """class {class_name}:
    def __init__(self{init_params}):
        {init_implementation}
    
    def {method_name}(self{method_params}):
        {method_implementation}
        return result"""
    ]
    
    # Generate synthetic data
    data = []
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_samples):
        # Select random components
        question_template = np.random.choice(question_templates)
        task = np.random.choice(tasks)
        context = np.random.choice(contexts)
        
        # Generate question
        question = question_template.format(task)
        
        # Generate a simple code answer based on the task
        if "add" in task:
            answer = """def add_numbers(a, b):
    \"\"\"Add two numbers and return the result\"\"\"
    return a + b"""
            ground_truth = "def add_numbers(a, b): return a + b"
            
        elif "multiply" in task:
            answer = """def multiply_numbers(a, b):
    \"\"\"Multiply two numbers and return the result\"\"\"
    return a * b"""
            ground_truth = "def multiply_numbers(a, b): return a * b"
            
        elif "factorial" in task:
            answer = """def factorial(n):
    \"\"\"Calculate factorial of a number\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
            ground_truth = "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)"
            
        elif "prime" in task:
            answer = """def is_prime(n):
    \"\"\"Check if a number is prime\"\"\"
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True"""
            ground_truth = "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))"
            
        elif "reverse" in task:
            answer = """def reverse_string(s):
    \"\"\"Reverse a string\"\"\"
    return s[::-1]"""
            ground_truth = "def reverse_string(s): return s[::-1]"
            
        else:
            # Generic template
            func_name = task.replace(" ", "_").replace("a ", "").replace("the ", "")
            answer = f"""def {func_name}():
    \"\"\"
    {task.capitalize()}
    \"\"\"
    # Implementation here
    pass"""
            ground_truth = f"def {func_name}(): pass"
        
        # Determine complexity based on answer length and structure
        complexity = "simple" if len(answer) < 100 else ("moderate" if len(answer) < 200 else "complex")
        
        # Create enhanced context with code examples
        code_context = f"Example: {answer.split('def')[1].split(':')[0] if 'def' in answer else 'function implementation'}"
        
        # Add to dataset
        data.append({
            'question': question,
            'context': context,
            'code_context': code_context,
            'answer': answer,
            'ground_truth': ground_truth,
            'complexity': complexity
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Generated {num_samples} synthetic samples")
    print(f"💾 Saved to {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df

def main():
    """Main function to generate synthetic data"""
    
    # Generate training data
    print("🔄 Generating synthetic training data...")
    train_data = generate_synthetic_data(
        num_samples=1000,
        output_path="data/synthetic_data.parquet"
    )
    
    # Generate smaller test dataset
    print("\n🔄 Generating synthetic test data...")
    test_data = generate_synthetic_data(
        num_samples=100,
        output_path="data/test_data.parquet"
    )
    
    print("\n🎉 Data generation completed successfully!")
    print(f"📁 Training data: data/synthetic_data.parquet ({len(train_data)} samples)")
    print(f"📁 Test data: data/test_data.parquet ({len(test_data)} samples)")

if __name__ == "__main__":
    main()