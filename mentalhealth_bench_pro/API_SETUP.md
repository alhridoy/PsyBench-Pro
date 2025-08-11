# API Key Setup Guide

## Overview
This project requires API keys from Anthropic (for Claude models) and OpenAI (for GPT models) to run the test scripts.

## Security Notice
**IMPORTANT**: Never commit your actual API keys to version control. The test scripts have been updated to use environment variables instead of hardcoded keys.

## Setup Instructions

### Option 1: Using Environment Variables (Recommended)

1. **Set environment variables in your terminal:**
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-key-here"
   export OPENAI_API_KEY="your-openai-key-here"
   ```

2. **Or create a .env file:**
   - Copy `.env.example` to `.env`
   - Add your actual API keys to the `.env` file
   - Make sure `.env` is in your `.gitignore`

### Option 2: Direct Replacement

If you prefer, you can directly replace the placeholder values in the test scripts:
- Replace `YOUR_ANTHROPIC_API_KEY_HERE` with your actual Anthropic API key
- Replace `YOUR_OPENAI_API_KEY_HERE` with your actual OpenAI API key

## Getting API Keys

### Anthropic API Key
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key

### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key

## Affected Files
The following test scripts use API keys:
- `automated_real_test.py`
- `comprehensive_2025_test.py`
- `latest_flagship_test.py`
- `premium_model_test.py`
- `quick_real_test.py`

## Example Usage
```bash
# Set your API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-proj-..."

# Run a test
python quick_real_test.py
```

## Troubleshooting
- If you get authentication errors, verify your API keys are correct
- Ensure you have sufficient credits/balance in your API accounts
- Check that environment variables are properly set in your terminal session