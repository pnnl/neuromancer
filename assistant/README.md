# NeuroMANCER GPT 

This folder contains a script that is intended to format the contents of the
NeuroMANCER repository in a way that is suitable for ingestion in RAG-based
"LLM-assistant" pipelines.


## Dependencies

The following pip packages are used by the script:
+ pypandoc
+ panflute
+ tqdm

``` sh
pip install pypandoc panflute tqdm
```

When run, the script generates the following files:

+ `knowledge/docs.txt` concatenated NeuroMANCER documentation.
+ `knowledge/examples.txt` concatenated NeuroMANCER examples, as python code.
+ `knowledge/src.txt` concatenated NeuroMANCER source code.

In addition, the following is included as a base system prompt:

+ `knowledge/prompt.txt` The system prompt for an AI assistant

## Usage

``` sh
python3 ingest.py 
```


