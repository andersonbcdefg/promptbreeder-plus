# Promptbreeder

To get started, clone this repo. To make requests to OpenAI, you'll need to provide the following environment variables (put them in a .env file in the root directory):

```
OPENAI_API_KEY=<your_api_key_here>
MAX_TOKENS_PER_MINUTE=<maximum tokens per minute (according to your rate limits)>
MAX_REQUESTS_PER_MINUTE=<maximum requests per minute (according to your rate limits)>
```

Then run the following commands from the root directory to start a local server:
    
    ```bash
    pip install -e .
    uvicorn server:app --reload
    ```

A local server will begin running on localhost:8000. Go there in your browser to get started.