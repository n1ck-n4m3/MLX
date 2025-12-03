#!/usr/bin/env python3
"""
Simple Model Testing Script
Test basic conversation with all 9 models
"""

from mlx_lm import load, generate
import time
import os

def test_model(model_path, model_name):
    """Test a single model with basic conversation"""
    print(f"\n{'='*60}")
    print(f"Testing Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print("Loading model...")
        start_time = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Test basic conversation
        test_prompt = "Hello! How are you today?"
        print(f"\nPrompt: {test_prompt}")
        
        # Generate response
        print("Generating response...")
        start_time = time.time()
        response = generate(
            model, 
            tokenizer, 
            prompt=test_prompt,
            max_tokens=100
        )
        generation_time = time.time() - start_time
        
        print(f"\nResponse ({generation_time:.2f}s):")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        return {
            'model_name': model_name,
            'load_time': load_time,
            'generation_time': generation_time,
            'response': response,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e)
        }

def main():
    """Test all 9 models"""
    print("🚀 Starting Model Testing - Basic Conversation")
    print("Testing prompt: 'Hello! How are you today?'")
    
    # Define all models
    models = [
        ("models/llama-3-8b-instruct", "Llama 3 8B Instruct"),
        ("models/llama-3.1-8b", "Llama 3.1 8B Base"),
        ("models/llama-3.1-8b-instruct", "Llama 3.1 8B Instruct"),
        ("models/mistral7b-v0_1", "Mistral 7B Base"),
        ("models/mistral7b-instruct-v0_3", "Mistral 7B Instruct"),
        ("models/ministral8b-instruct-2410", "Ministral 8B Instruct"),
        ("models/qwen2-7b-instruct", "Qwen2 7B Instruct"),
        ("models/qwen2.5-7b", "Qwen2.5 7B Base"),
        ("models/qwen2.5-7b-instruct", "Qwen2.5 7B Instruct")
    ]
    
    results = []
    total_start_time = time.time()
    
    for model_path, model_name in models:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"\n❌ Model not found: {model_path}")
            results.append({
                'model_name': model_name,
                'status': 'not_found',
                'path': model_path
            })
            continue
            
        # Test the model
        result = test_model(model_path, model_name)
        results.append(result)
        
        # Small delay between models
        time.sleep(1)
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Models tested: {len(models)}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    not_found = [r for r in results if r['status'] == 'not_found']
    
    print(f"✓ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⚠️  Not found: {len(not_found)}")
    
    if successful:
        print(f"\nAverage load time: {sum(r['load_time'] for r in successful)/len(successful):.2f}s")
        print(f"Average generation time: {sum(r['generation_time'] for r in successful)/len(successful):.2f}s")
    
    # Show failed models
    if failed:
        print(f"\nFailed models:")
        for result in failed:
            print(f"  - {result['model_name']}: {result['error']}")
    
    if not_found:
        print(f"\nModels not found:")
        for result in not_found:
            print(f"  - {result['model_name']}: {result['path']}")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
