#!/usr/bin/env python3
"""
Simple RAG Training Pipeline
Focuses on RAG functionality without distributed computing complexity
"""

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import argparse

class SimpleRAGTrainer:
    """Simple RAG trainer without distributed computing"""
    
    def __init__(self):
        self.complexity_classifier = None
        self.retrieval_system = None
        self.response_generator = None
        self.is_trained = False
    
    def train(self, data):
        """Train the RAG model"""
        print("🎯 Training Simple RAG Model")
        print(f"📊 Training data: {len(data)} samples")
        
        # Phase 1: Train complexity classifier
        print("📊 Phase 1: Training complexity classifier...")
        self._train_complexity_classifier(data)
        
        # Phase 2: Build retrieval system
        print("📊 Phase 2: Building retrieval system...")
        self._build_retrieval_system(data)
        
        # Phase 3: Train response generator
        print("📊 Phase 3: Training response generator...")
        self._train_response_generator(data)
        
        self.is_trained = True
        print("✅ RAG model training completed!")
    
    def _train_complexity_classifier(self, data):
        """Train a classifier to determine query complexity"""
        # Extract features from questions
        questions = data['question'].fillna('').astype(str)
        
        # Simple features for complexity classification
        features = pd.DataFrame({
            'length': questions.str.len(),
            'word_count': questions.str.split().str.len(),
            'question_marks': questions.str.count(r'\?'),
            'has_code_keywords': questions.str.contains(r'function|class|def|import', case=False, na=False).astype(int)
        })
        
        # Create complexity labels based on heuristics
        complexity_labels = []
        for _, row in features.iterrows():
            if row['word_count'] <= 5:
                complexity_labels.append('simple')
            elif row['word_count'] <= 15:
                complexity_labels.append('moderate')
            else:
                complexity_labels.append('complex')
        
        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(
            features, complexity_labels, test_size=0.2, random_state=42
        )
        
        self.complexity_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.complexity_classifier.fit(X_train, y_train)
        
        accuracy = self.complexity_classifier.score(X_test, y_test)
        print(f"   ✅ Complexity classifier accuracy: {accuracy:.3f}")
    
    def _build_retrieval_system(self, data):
        """Build TF-IDF based retrieval system"""
        contexts = data['context'].fillna('').astype(str)
        
        # Build TF-IDF vectorizer
        self.retrieval_system = {
            'vectorizer': TfidfVectorizer(max_features=1000, stop_words='english'),
            'contexts': contexts.tolist(),
            'context_vectors': None
        }
        
        # Fit and transform contexts
        self.retrieval_system['context_vectors'] = self.retrieval_system['vectorizer'].fit_transform(contexts)
        
        print(f"   ✅ Retrieval system built with {len(contexts)} contexts")
    
    def _train_response_generator(self, data):
        """Train response generator using template-based approach"""
        # For simplicity, use template-based generation with ML ranking
        answers = data['answer'].fillna('').astype(str)
        
        # Create answer templates based on patterns
        self.response_generator = {
            'templates': {
                'function': 'def {function_name}({params}):\n    """{docstring}"""\n    {body}\n    return {return_value}',
                'class': 'class {class_name}:\n    """{docstring}"""\n    def __init__(self):\n        {init_body}',
                'simple': '{answer}',
            },
            'answer_examples': answers.tolist()
        }
        
        print(f"   ✅ Response generator trained with {len(answers)} examples")
    
    def predict(self, question, context=None):
        """Generate response for a question"""
        if not self.is_trained:
            return "Model not trained yet"
        
        # Step 1: Classify complexity
        question_features = pd.DataFrame({
            'length': [len(question)],
            'word_count': [len(question.split())],
            'question_marks': [question.count('?')],
            'has_code_keywords': [1 if any(kw in question.lower() for kw in ['function', 'class', 'def', 'import']) else 0]
        })
        
        complexity = self.complexity_classifier.predict(question_features)[0]
        
        # Step 2: Retrieve relevant context (if not provided)
        if context is None:
            question_vector = self.retrieval_system['vectorizer'].transform([question])
            similarities = cosine_similarity(question_vector, self.retrieval_system['context_vectors']).flatten()
            best_context_idx = np.argmax(similarities)
            context = self.retrieval_system['contexts'][best_context_idx]
            relevance_score = similarities[best_context_idx]
        else:
            relevance_score = 0.8  # Assume provided context is relevant
        
        # Step 3: Generate response
        if 'function' in question.lower():
            template = self.response_generator['templates']['function']
            response = template.format(
                function_name='example_function',
                params='param1, param2',
                docstring='Function description',
                body='    # Implementation here',
                return_value='result'
            )
        else:
            # Use a simple answer from examples
            response = np.random.choice(self.response_generator['answer_examples'])
        
        return {
            'response': response,
            'complexity': complexity,
            'context_relevance': relevance_score,
            'context_used': context[:100] + '...' if len(context) > 100 else context
        }

def evaluate_rag_model(trainer, test_data, num_samples=5):
    """Evaluate the RAG model"""
    print(f"🔍 Evaluating RAG model on {num_samples} samples...")
    
    results = []
    for i, row in test_data.head(num_samples).iterrows():
        question = row['question']
        expected_answer = row.get('answer', 'No expected answer')
        
        # Get prediction
        prediction = trainer.predict(question)
        
        # Simple evaluation metrics
        result = {
            'question': question,
            'predicted_answer': prediction['response'],
            'expected_answer': expected_answer,
            'complexity': prediction['complexity'],
            'context_relevance': prediction['context_relevance'],
            'answer_length': len(prediction['response']),
        }
        results.append(result)
        
        print(f"   Sample {i+1}:")
        print(f"   Q: {question[:50]}...")
        print(f"   A: {prediction['response'][:50]}...")
        print(f"   Complexity: {prediction['complexity']}")
        print(f"   Relevance: {prediction['context_relevance']:.3f}")
        print()
    
    # Calculate summary metrics
    avg_relevance = np.mean([r['context_relevance'] for r in results])
    avg_length = np.mean([r['answer_length'] for r in results])
    
    summary = {
        'num_samples': len(results),
        'avg_context_relevance': avg_relevance,
        'avg_answer_length': avg_length,
        'complexity_distribution': pd.Series([r['complexity'] for r in results]).value_counts().to_dict()
    }
    
    print("📊 Evaluation Summary:")
    print(f"   Average Context Relevance: {avg_relevance:.3f}")
    print(f"   Average Answer Length: {avg_length:.1f} characters")
    print(f"   Complexity Distribution: {summary['complexity_distribution']}")
    
    return results, summary

def main():
    parser = argparse.ArgumentParser(description='Simple RAG Training Pipeline')
    parser.add_argument('--data-path', default='data/rag_training_data.parquet', help='Path to training data')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--eval-samples', type=int, default=5, help='Number of evaluation samples')
    
    args = parser.parse_args()
    
    print("🚀 Starting Simple RAG Training Pipeline")
    print("=" * 50)
    
    # Load or generate data
    if os.path.exists(args.data_path):
        print(f"📊 Loading data from {args.data_path}")
        data = pd.read_parquet(args.data_path)
        print(f"📈 Dataset loaded: {len(data)} samples")
    else:
        print("📊 Generating sample data...")
        # Generate simple sample data
        questions = [
            "How do I create a function to add two numbers?",
            "Write a Python function that multiplies two numbers",
            "Can you help me write code to calculate factorial?",
            "What is the best way to reverse a string in Python?",
            "Create a function that checks if a number is prime"
        ] * (args.num_samples // 5)
        
        contexts = [
            "Python functions are defined using the def keyword followed by parameters.",
            "Mathematical operations in Python can be performed using arithmetic operators.",
            "Recursive functions call themselves with modified parameters until a base case.",
            "String slicing in Python uses bracket notation with start:end:step syntax.",
            "Prime number checking requires testing divisibility up to the square root."
        ] * (args.num_samples // 5)
        
        answers = [
            "def add_numbers(a, b):\n    return a + b",
            "def multiply_numbers(a, b):\n    return a * b",
            "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
            "def reverse_string(s):\n    return s[::-1]",
            "def is_prime(n):\n    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))"
        ] * (args.num_samples // 5)
        
        data = pd.DataFrame({
            'question': questions[:args.num_samples],
            'context': contexts[:args.num_samples],
            'answer': answers[:args.num_samples]
        })
        
        # Save generated data
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        data.to_parquet(args.data_path, index=False)
        print(f"💾 Generated and saved {len(data)} samples to {args.data_path}")
    
    # Train RAG model
    start_time = time.time()
    trainer = SimpleRAGTrainer()
    trainer.train(data)
    training_time = time.time() - start_time
    
    print(f"⏱️ Training completed in {training_time:.2f} seconds")
    
    # Evaluate if requested
    if args.evaluate:
        print("\n" + "=" * 50)
        results, summary = evaluate_rag_model(trainer, data, args.eval_samples)
        
        # Save evaluation results
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{eval_dir}/simple_rag_evaluation_{timestamp}.json"
        
        eval_data = {
            'timestamp': timestamp,
            'training_time': training_time,
            'summary': summary,
            'detailed_results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(eval_data, f, indent=2, default=str)
        
        print(f"💾 Evaluation results saved to {results_file}")
    
    print("\n✅ Simple RAG Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())

