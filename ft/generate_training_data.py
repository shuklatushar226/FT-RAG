"""
Generate training data from ingested repositories for fine-tuning
This script extracts code samples and creates instruction-following training pairs
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from app.settings import settings
from build_sft_data import SFTDataBuilder
import random

def get_qdrant_client():
    return QdrantClient(url=settings.QDRANT_URL)

def extract_training_data() -> List[Dict[str, Any]]:
    """Extract code examples from Qdrant to create training data"""
    client = get_qdrant_client()
    
    # Get all points from the collection
    points = client.scroll(
        collection_name=settings.COLLECTION,
        limit=1000,  # Adjust based on your data size
        with_payload=True
    )[0]
    
    training_examples = []
    
    for point in points:
        payload = point.payload
        content = payload.get('page_content', '')
        repo = payload.get('repo', '')
        path = payload.get('path', '')
        
        # Skip if content is too short
        if len(content.strip()) < 100:
            continue
            
        # Create different types of training examples
        examples = generate_code_examples(content, repo, path)
        training_examples.extend(examples)
    
    return training_examples

def generate_code_examples(content: str, repo: str, path: str) -> List[Dict[str, Any]]:
    """Generate various training examples from code content"""
    examples = []
    
    # Code explanation task
    if any(keyword in content for keyword in ['function', 'class', 'def', 'const', 'var']):
        examples.append({
            "instruction": f"Explain the following code from {repo}:",
            "input": content,
            "output": f"This code from {path} in the {repo} repository implements functionality that processes and handles data structures. The code uses standard patterns and best practices for maintainable software development."
        })
    
    # Code completion task
    if len(content) > 200:
        # Take first 70% as input, rest as expected output
        split_point = int(len(content) * 0.7)
        input_part = content[:split_point]
        output_part = content[split_point:]
        
        examples.append({
            "instruction": "Complete the following code:",
            "input": input_part,
            "output": output_part
        })
    
    # Documentation generation task
    if any(keyword in content for keyword in ['function', 'class', 'def']):
        examples.append({
            "instruction": f"Write documentation for this code from {repo}:",
            "input": content,
            "output": f"## Code Documentation\n\nThis code implements functionality in {path}. It follows established patterns and provides reliable functionality for the {repo} project."
        })
    
    return examples

def create_hyperswitch_training_data():
    """Create specific training data for Hyperswitch integration"""
    builder = SFTDataBuilder()
    
    # Add specific Hyperswitch examples
    builder.add_conversation(
        instruction="How do I integrate Hyperswitch payment processing in React Native?",
        output="""To integrate Hyperswitch in React Native:

1. Install the SDK:
```bash
npm install @juspay-tech/hyperswitch-client-core
```

2. Initialize the client:
```javascript
import { HyperSwitch } from '@juspay-tech/hyperswitch-client-core';

const client = new HyperSwitch({
  publishableKey: 'your_publishable_key',
  environment: 'sandbox' // or 'production'
});
```

3. Create a payment:
```javascript
const payment = await client.payments.create({
  amount: 1000,
  currency: 'USD',
  payment_method: {
    type: 'card',
    card: {
      number: '4111111111111111',
      exp_month: 12,
      exp_year: 2025,
      cvc: '123'
    }
  }
});
```"""
    )
    
    builder.add_conversation(
        instruction="What payment methods does Hyperswitch support?",
        output="""Hyperswitch supports multiple payment methods:

**Card Payments:**
- Credit/Debit cards (Visa, Mastercard, American Express)
- 3D Secure authentication
- Saved card payments

**Digital Wallets:**
- Apple Pay
- Google Pay
- PayPal
- Amazon Pay

**Bank Transfers:**
- ACH transfers
- SEPA Direct Debit
- Open Banking

**Buy Now Pay Later:**
- Klarna
- Afterpay
- Affirm

**Crypto:**
- Bitcoin
- Ethereum
- Stablecoins

All methods are unified under a single API interface."""
    )
    
    return builder

def main():
    """Main function to generate comprehensive training data"""
    print("🚀 Generating training data from ingested repositories...")
    
    # Extract training data from Qdrant
    try:
        training_examples = extract_training_data()
        print(f"📊 Extracted {len(training_examples)} examples from repositories")
    except Exception as e:
        print(f"⚠️  Warning: Could not extract from Qdrant: {e}")
        training_examples = []
    
    # Create SFT builder and add extracted examples
    builder = SFTDataBuilder()
    
    for example in training_examples:
        builder.add_conversation(
            instruction=example["instruction"],
            input_text=example["input"],
            output=example["output"]
        )
    
    # Add Hyperswitch-specific training data
    hyperswitch_builder = create_hyperswitch_training_data()
    builder.data.extend(hyperswitch_builder.data)
    
    # Add general coding examples
    builder.add_conversation(
        instruction="Write a TypeScript interface for a payment method",
        output="""```typescript
interface PaymentMethod {
  id: string;
  type: 'card' | 'wallet' | 'bank_transfer';
  card?: {
    last4: string;
    brand: string;
    exp_month: number;
    exp_year: number;
  };
  wallet?: {
    type: 'apple_pay' | 'google_pay' | 'paypal';
    email?: string;
  };
  bank_transfer?: {
    account_type: string;
    routing_number: string;
    account_number_last4: string;
  };
  created_at: string;
  is_default: boolean;
}
```"""
    )
    
    builder.add_conversation(
        instruction="Explain error handling in payment processing",
        output="""Error handling in payment processing should be comprehensive:

**1. Network Errors:**
```javascript
try {
  const payment = await client.payments.create(paymentData);
} catch (error) {
  if (error.type === 'network_error') {
    // Handle network issues
    showRetryOption();
  }
}
```

**2. Validation Errors:**
- Invalid card numbers
- Expired cards
- Insufficient funds
- Invalid CVV

**3. Authentication Errors:**
- 3D Secure failures
- Bank authentication issues

**4. Business Logic Errors:**
- Amount limits exceeded
- Currency not supported
- Payment method not allowed

Always provide clear user feedback and fallback options."""
    )
    
    # Validate and save
    builder.validate_data()
    
    # Save to the correct location for axolotl
    output_path = "sft.jsonl"
    builder.save_jsonl(output_path)
    
    print(f"✅ Training data saved to {output_path}")
    print(f"📈 Total training examples: {len(builder.data)}")
    
    # Also save a human-readable version
    builder.save_json("sft_readable.json")
    print("📄 Human-readable version saved to sft_readable.json")

if __name__ == "__main__":
    main()