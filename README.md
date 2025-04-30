# ChatGPT System Prompt Experimenter

A Streamlit app for experimenting with different system prompts for OpenAI's chat models and maintaining conversation context.

## Features

- Customize system prompts
- Save and load system prompt presets
- Maintain multiple conversations
- Switch between conversations
- Full conversation history

## Setup

1. Make sure you have Python installed
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the app

```
streamlit run mark-streamlit.py
```

## Usage

1. Enter your system prompt in the sidebar
2. Save presets for different system prompts you want to test
3. Create new conversations to test different prompts
4. Use the chat interface to interact with the model
5. Switch between conversations to compare responses 