"""
Debug script to check data columns and fix column name issues
"""

import pandas as pd
import os

def debug_data_columns(data_path="data/synthetic_data.parquet"):
    """Debug and fix data column issues"""
    
    print("🔍 Debugging data columns...")
    print("-" * 50)
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    # Load and inspect data
    try:
        df = pd.read_parquet(data_path)
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Current columns: {list(df.columns)}")
        print(f"📝 Data types:\n{df.dtypes}")
        print("\n📖 First few rows:")
        print(df.head(2))
        
        # Expected columns for RAG training
        expected_columns = ['question', 'context', 'answer', 'ground_truth', 'complexity']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n⚠️ Missing expected columns: {missing_columns}")
            
            # Try to map existing columns to expected ones
            column_mapping = {}
            
            # Common column name variations
            for col in df.columns:
                col_lower = col.lower()
                if 'question' in col_lower or 'query' in col_lower:
                    column_mapping['question'] = col
                elif 'context' in col_lower or 'prompt' in col_lower:
                    column_mapping['context'] = col
                elif 'answer' in col_lower or 'response' in col_lower or 'output' in col_lower:
                    column_mapping['answer'] = col
                elif 'ground_truth' in col_lower or 'target' in col_lower or 'label' in col_lower:
                    column_mapping['ground_truth'] = col
                elif 'complexity' in col_lower or 'difficulty' in col_lower:
                    column_mapping['complexity'] = col
            
            print(f"\n🔄 Suggested column mapping: {column_mapping}")
            
            # Create fixed dataset
            fixed_df = df.copy()
            
            # Rename columns if mapping found
            if column_mapping:
                fixed_df = fixed_df.rename(columns={v: k for k, v in column_mapping.items()})
            
            # Add missing columns with defaults
            for col in expected_columns:
                if col not in fixed_df.columns:
                    if col == 'question':
                        fixed_df['question'] = "How do I write a Python function?"
                    elif col == 'context':
                        fixed_df['context'] = "Python functions are defined using the def keyword."
                    elif col == 'answer':
                        fixed_df['answer'] = "def example_function(): pass"
                    elif col == 'ground_truth':
                        fixed_df['ground_truth'] = fixed_df.get('answer', "def example_function(): pass")
                    elif col == 'complexity':
                        fixed_df['complexity'] = "simple"
            
            # Save fixed dataset
            fixed_path = data_path.replace('.parquet', '_fixed.parquet')
            fixed_df.to_parquet(fixed_path, index=False)
            
            print(f"\n✅ Fixed dataset saved to: {fixed_path}")
            print(f"📊 Fixed dataset shape: {fixed_df.shape}")
            print(f"📋 Fixed columns: {list(fixed_df.columns)}")
            
            return fixed_df
        else:
            print("\n✅ All expected columns are present!")
            return df
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_sample_dataset(output_path="data/sample_rag_data.parquet", num_samples=100):
    """Create a properly formatted sample dataset"""
    
    print(f"\n🔄 Creating sample dataset with {num_samples} samples...")
    
    # Sample data that matches expected format
    questions = [
        "How do I create a function to add two numbers?",
        "Write a Python function that multiplies two numbers",
        "Can you help me write code to calculate factorial?",
        "What's the best way to reverse a string in Python?",
        "Create a function that checks if a number is prime",
        "How can I implement list sorting functionality?",
        "Write a class that handles basic calculations",
        "Show me how to validate email using Python",
        "Create a method to find maximum in a list",
        "How do I write a function for fibonacci sequence?"
    ]
    
    contexts = [
        "Python functions are defined using the 'def' keyword followed by parameters.",
        "Mathematical operations in Python can be performed using arithmetic operators.",
        "Recursive functions call themselves with modified parameters until a base case.",
        "String slicing in Python uses bracket notation with start:end:step syntax.",
        "Prime number checking requires testing divisibility up to the square root.",
        "Python's built-in sorted() function provides efficient list sorting.",
        "Classes in Python encapsulate data and methods for object-oriented programming.",
        "Regular expressions provide pattern matching for string validation.",
        "The max() function returns the largest element from an iterable.",
        "Fibonacci sequences can be implemented recursively or iteratively."
    ]
    
    answers = [
        "def add_numbers(a, b):\n    return a + b",
        "def multiply_numbers(a, b):\n    return a * b",
        "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
        "def reverse_string(s):\n    return s[::-1]",
        "def is_prime(n):\n    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
        "def sort_list(lst):\n    return sorted(lst)",
        "class Calculator:\n    def add(self, a, b):\n        return a + b",
        "import re\ndef validate_email(email):\n    return bool(re.match(r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', email))",
        "def find_max(lst):\n    return max(lst) if lst else None",
        "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    ]
    
    complexities = ["simple", "simple", "moderate", "simple", "moderate", 
                   "simple", "moderate", "complex", "simple", "moderate"]
    
    # Generate dataset
    data = []
    for i in range(num_samples):
        idx = i % len(questions)
        data.append({
            'question': questions[idx],
            'context': contexts[idx],
            'answer': answers[idx],
            'ground_truth': answers[idx],
            'complexity': complexities[idx]
        })
    
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save dataset
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Sample dataset created: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df

def main():
    """Main debugging function"""
    
    print("🔧 RAG Data Column Debugger")
    print("=" * 50)
    
    # Check existing data
    data_path = "data/synthetic_data.parquet"
    fixed_data = debug_data_columns(data_path)
    
    if fixed_data is None:
        print("\n📝 Creating new sample dataset...")
        sample_data = create_sample_dataset()
        print(f"\n💡 Use this command to test with the sample data:")
        print(f"python run_rag_training.py --engine dask --model-type rag --rag-type adaptive --evaluate --data-path data/sample_rag_data.parquet")
    else:
        if 'fixed' in data_path:
            print(f"\n💡 Use this command to test with the fixed data:")
            print(f"python run_rag_training.py --engine dask --model-type rag --rag-type adaptive --evaluate --data-path {data_path.replace('.parquet', '_fixed.parquet')}")
        else:
            print(f"\n💡 Your data looks good! Try running again:")
            print(f"python run_rag_training.py --engine dask --model-type rag --rag-type adaptive --evaluate")

if __name__ == "__main__":
    main()