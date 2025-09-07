#!/usr/bin/env python3
"""
Create Proper RAG Dataset

This script creates a realistic RAG dataset with proper questions, contexts, and code answers
for training the RAG pipeline.
"""

import pandas as pd
import os
import numpy as np

def create_rag_dataset(num_samples=1000, output_path="data/rag_training_data.parquet"):
    """Create a proper RAG dataset for code generation"""
    
    print(f"🔄 Creating RAG dataset with {num_samples} samples...")
    
    # Realistic programming questions
    questions = [
        "How do I create a function to add two numbers in Python?",
        "Write a Python function that multiplies two numbers",
        "Can you help me write code to calculate the factorial of a number?",
        "What's the best way to reverse a string in Python?",
        "Create a function that checks if a number is prime",
        "How can I implement a function to find the maximum value in a list?",
        "Write a Python function to sort a list of numbers",
        "How do I create a function to count vowels in a string?",
        "Can you show me how to implement the Fibonacci sequence?",
        "Write a function to check if a string is a palindrome",
        "How do I create a function to calculate the sum of a list?",
        "Write a Python function to remove duplicates from a list",
        "How can I convert Celsius to Fahrenheit in Python?",
        "Create a function to calculate the area of a circle",
        "Write a function to find the greatest common divisor of two numbers",
        "How do I merge two sorted lists in Python?",
        "Create a function to implement binary search",
        "Write a simple calculator function in Python",
        "How can I validate an email address using Python?",
        "Create a function to generate random numbers within a range"
    ]
    
    # Corresponding contexts (documentation/examples)
    contexts = [
        "Python functions are defined using the 'def' keyword. Basic arithmetic operations use +, -, *, / operators. Functions should include docstrings for documentation.",
        "Multiplication in Python uses the * operator. Functions can take multiple parameters and return values using the 'return' statement.",
        "Factorial is the product of all positive integers up to n. It can be implemented recursively (n! = n * (n-1)!) or iteratively using loops.",
        "String reversal in Python can be done using slicing with [::-1] syntax, or by using loops, or the reversed() function with join().",
        "Prime numbers are only divisible by 1 and themselves. Check divisibility from 2 to sqrt(n). Numbers less than 2 are not prime.",
        "The max() function returns the largest item. For custom logic, iterate through the list comparing each element to track the maximum.",
        "Python provides built-in sorted() function. You can also implement sorting algorithms like bubble sort, quick sort, or use list.sort() method.",
        "Vowels are a, e, i, o, u (and sometimes y). Iterate through the string and count occurrences. Use lower() for case-insensitive counting.",
        "Fibonacci sequence: each number is sum of two preceding ones (0, 1, 1, 2, 3, 5, 8...). Can be implemented recursively or iteratively.",
        "A palindrome reads the same forwards and backwards. Compare string with its reverse, or use two pointers from start and end.",
        "Use Python's built-in sum() function for lists, or implement with a loop accumulating values. Handle empty lists appropriately.",
        "Remove duplicates using set(), dict.fromkeys(), or list comprehension with tracking. Preserve order if needed using collections.OrderedDict.",
        "Celsius to Fahrenheit formula: F = (C * 9/5) + 32. Fahrenheit to Celsius: C = (F - 32) * 5/9. Handle input validation.",
        "Circle area formula: A = π * r². Use math.pi for π value. Validate that radius is positive. Consider precision for floating-point results.",
        "GCD using Euclidean algorithm: gcd(a,b) = gcd(b, a%b) until b=0. Python's math.gcd() provides built-in implementation.",
        "Merge sorted lists by comparing elements from both lists. Use two pointers to track positions. Handle remaining elements after one list is exhausted.",
        "Binary search works on sorted arrays. Compare target with middle element, then search left or right half. Time complexity: O(log n).",
        "Calculator needs functions for basic operations (+, -, *, /). Handle division by zero. Consider using a menu system for user interaction.",
        "Email validation using regular expressions. Pattern: username@domain.extension. Use re module for pattern matching in Python.",
        "Use random module: random.randint(a,b) for integers, random.uniform(a,b) for floats. Set seed with random.seed() for reproducible results."
    ]
    
    # Corresponding code answers
    answers = [
        """def add_numbers(a, b):
    \"\"\"
    Add two numbers and return the result.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Sum of a and b
    \"\"\"
    return a + b""",
        
        """def multiply_numbers(a, b):
    \"\"\"
    Multiply two numbers and return the result.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Product of a and b
    \"\"\"
    return a * b""",
        
        """def factorial(n):
    \"\"\"
    Calculate the factorial of a number.
    
    Args:
        n: Non-negative integer
    
    Returns:
        Factorial of n
    \"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
        
        """def reverse_string(s):
    \"\"\"
    Reverse a string.
    
    Args:
        s: Input string
    
    Returns:
        Reversed string
    \"\"\"
    return s[::-1]""",
        
        """def is_prime(n):
    \"\"\"
    Check if a number is prime.
    
    Args:
        n: Integer to check
    
    Returns:
        True if n is prime, False otherwise
    \"\"\"
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True""",
        
        """def find_max(lst):
    \"\"\"
    Find the maximum value in a list.
    
    Args:
        lst: List of numbers
    
    Returns:
        Maximum value in the list
    \"\"\"
    if not lst:
        return None
    return max(lst)""",
        
        """def sort_numbers(lst):
    \"\"\"
    Sort a list of numbers.
    
    Args:
        lst: List of numbers
    
    Returns:
        Sorted list
    \"\"\"
    return sorted(lst)""",
        
        """def count_vowels(s):
    \"\"\"
    Count vowels in a string.
    
    Args:
        s: Input string
    
    Returns:
        Number of vowels
    \"\"\"
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)""",
        
        """def fibonacci(n):
    \"\"\"
    Generate the nth Fibonacci number.
    
    Args:
        n: Position in Fibonacci sequence
    
    Returns:
        nth Fibonacci number
    \"\"\"
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)""",
        
        """def is_palindrome(s):
    \"\"\"
    Check if a string is a palindrome.
    
    Args:
        s: Input string
    
    Returns:
        True if palindrome, False otherwise
    \"\"\"
    s = s.lower().replace(' ', '')
    return s == s[::-1]""",
        
        """def sum_list(lst):
    \"\"\"
    Calculate the sum of elements in a list.
    
    Args:
        lst: List of numbers
    
    Returns:
        Sum of all elements
    \"\"\"
    return sum(lst)""",
        
        """def remove_duplicates(lst):
    \"\"\"
    Remove duplicates from a list while preserving order.
    
    Args:
        lst: Input list
    
    Returns:
        List without duplicates
    \"\"\"
    return list(dict.fromkeys(lst))""",
        
        """def celsius_to_fahrenheit(celsius):
    \"\"\"
    Convert Celsius to Fahrenheit.
    
    Args:
        celsius: Temperature in Celsius
    
    Returns:
        Temperature in Fahrenheit
    \"\"\"
    return (celsius * 9/5) + 32""",
        
        """def circle_area(radius):
    \"\"\"
    Calculate the area of a circle.
    
    Args:
        radius: Radius of the circle
    
    Returns:
        Area of the circle
    \"\"\"
    import math
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * radius ** 2""",
        
        """def gcd(a, b):
    \"\"\"
    Find the greatest common divisor of two numbers.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Greatest common divisor
    \"\"\"
    while b:
        a, b = b, a % b
    return a""",
        
        """def merge_sorted_lists(list1, list2):
    \"\"\"
    Merge two sorted lists.
    
    Args:
        list1: First sorted list
        list2: Second sorted list
    
    Returns:
        Merged sorted list
    \"\"\"
    result = []
    i = j = 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result""",
        
        """def binary_search(arr, target):
    \"\"\"
    Perform binary search on a sorted array.
    
    Args:
        arr: Sorted array
        target: Value to search for
    
    Returns:
        Index of target if found, -1 otherwise
    \"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
        
        """def calculator(operation, a, b):
    \"\"\"
    Simple calculator function.
    
    Args:
        operation: Operation to perform (+, -, *, /)
        a: First number
        b: Second number
    
    Returns:
        Result of the operation
    \"\"\"
    if operation == '+':
        return a + b
    elif operation == '-':
        return a - b
    elif operation == '*':
        return a * b
    elif operation == '/':
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError("Invalid operation")""",
        
        """def validate_email(email):
    \"\"\"
    Validate an email address using regex.
    
    Args:
        email: Email address to validate
    
    Returns:
        True if valid, False otherwise
    \"\"\"
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))""",
        
        """def generate_random_numbers(start, end, count=1):
    \"\"\"
    Generate random numbers within a range.
    
    Args:
        start: Start of range (inclusive)
        end: End of range (inclusive)
        count: Number of random numbers to generate
    
    Returns:
        List of random numbers
    \"\"\"
    import random
    return [random.randint(start, end) for _ in range(count)]"""
    ]
    
    # Complexity levels
    complexities = [
        "simple", "simple", "moderate", "simple", "moderate",
        "simple", "simple", "simple", "moderate", "simple",
        "simple", "moderate", "simple", "moderate", "moderate",
        "complex", "complex", "moderate", "complex", "moderate"
    ]
    
    # Generate dataset
    data = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_samples):
        # Cycle through the questions to create variety
        idx = i % len(questions)
        
        data.append({
            'question': questions[idx],
            'context': contexts[idx],
            'answer': answers[idx],
            'ground_truth': answers[idx],
            'complexity': complexities[idx]
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Created RAG dataset: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🎯 Sample question: {df.iloc[0]['question']}")
    print(f"📝 Sample answer preview: {df.iloc[0]['answer'][:100]}...")
    
    return df

def main():
    """Main function to create RAG dataset"""
    
    print("🚀 Creating Proper RAG Training Dataset")
    print("=" * 50)
    
    # Create the dataset
    dataset = create_rag_dataset(
        num_samples=1000,
        output_path="data/rag_training_data.parquet"
    )
    
    print("\n🎉 RAG dataset created successfully!")
    print("\n💡 Now run the RAG training with:")
    print("python run_rag_training.py --engine dask --model-type rag --rag-type adaptive --evaluate --data-path data/rag_training_data.parquet")

if __name__ == "__main__":
    main()

