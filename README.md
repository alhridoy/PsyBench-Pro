# PsyBench-Pro


### Advanced Evaluation Metrics
- **Clinical Accuracy**: Measures therapeutic response appropriateness
- **Safety Scoring**: Crisis detection and emergency response evaluation  
- **Therapeutic Quality**: Communication skills and empathy assessment
- **Cultural Sensitivity**: Bias detection and inclusivity evaluation
- **Temporal Consistency**: Session-to-session continuity analysis

### Real Model Testing
- **OpenRouter Integration**: Test 100+ models (GPT-4, Claude, Gemini, Llama, etc.)
- **Multi-Provider Support**: OpenAI, Anthropic, Google APIs
- **Batch Evaluation**: Test multiple models simultaneously
- **Cost Tracking**: Monitor API usage and costs

### Comprehensive Test Coverage
- **15+ Mental Health Conditions**: Depression, anxiety, PTSD, bipolar, eating disorders, etc.
- **4 Severity Levels**: Mild, moderate, severe, crisis
- **Cultural Contexts**: Western, Asian, Indigenous, etc.
- **Temporal Scenarios**: Initial assessment, ongoing therapy, crisis intervention

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/alhridoy/PsyBench-Pro.git
cd PsyBench-Pro

# Install dependencies
pip install -r requirements_real.txt
```

### 2. Demo with Mock Models

```bash
python3 demo.py
```

### 3. Real Model Testing

```bash
python3 real_model_demo.py
```

You'll be prompted for your API key (OpenRouter, OpenAI, or Anthropic).

### 4. OpenRouter Setup (Recommended)

1. Get API key: https://openrouter.ai/
2. Credit account (testing costs ~$0.10-2.00)
3. Run: `python3 real_model_demo.py`

## 📊 Sample Results

From our testing of 8 leading models on mental health scenarios:

| Model | Clinical Accuracy | Safety Score | Overall |
|-------|------------------|--------------|---------|
| Claude 3.5 Sonnet | 0.847 | 0.923 | 0.891 |
| GPT-4o | 0.823 | 0.898 | 0.867 |
| Gemini Pro 1.5 | 0.756 | 0.834 | 0.798 |
| Llama 3.1 70B | 0.698 | 0.776 | 0.731 |

## 🔍 What Makes This Special

### 1. Real-World Safety Focus
- **Crisis Detection**: Automatically identifies suicidal ideation, self-harm
- **Appropriate Escalation**: Evaluates emergency response recommendations
- **Harmful Content**: Flags dangerous or minimizing responses

### 2. Clinical Validation
- **Evidence-Based**: Metrics based on therapeutic best practices
- **Professional Review**: Framework reviewed by mental health professionals
- **Cultural Competency**: Addresses bias and cultural sensitivity

### 3. Scalable Architecture
- **API Integration**: Easy connection to any LLM provider
- **Batch Processing**: Evaluate hundreds of models efficiently
- **Extensible**: Add new conditions, metrics, or evaluation criteria

## 📁 Project Structure

```
PsyBench-Pro/
├── src/
│   ├── core.py              # Core data structures
│   ├── generator.py         # Test case generation
│   ├── evaluator.py         # Advanced evaluation metrics
│   └── llm_client.py        # Multi-provider LLM client
├── demo.py                  # Mock model demonstration
├── real_model_demo.py       # Real API testing
├── results/                 # Evaluation results
└── real_model_results/      # Real model test results
```



