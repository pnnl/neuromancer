# NeuroMANCER GPT 

This folder contains a script that is intended to format the contents of the
NeuroMANCER repository in a way that is suitable for ingestion in RAG-based
"LLM-assistant" pipelines.

When run, this script will produce files that can be ingested/indexed
by RAG-based pipelines representing specialized contextual knowledge to
inform questions about NeuroMANCER.

We intend this script and its outputs to be used by the community to
leverage LLM-assistants where desired. We hope that by providing this script,
we lower the barrier to entry for using LLMs to assist the use, development,
and understanding of NeuroMANCER.

## Dependencies

The following pip packages are required by the script:
+ pypandoc
+ panflute
+ tqdm

They may be installed using:

``` sh
pip install pypandoc panflute tqdm
```

## Usage

When run, i.e., by executing


``` sh
python3 ingest.py 
```

the script generates the following files:

+ `knowledge/docs.txt` concatenated NeuroMANCER documentation.
+ `knowledge/examples.txt` concatenated NeuroMANCER examples, as python code.
+ `knowledge/src.txt` concatenated NeuroMANCER source code.

These files represent the contents of the NeuroMANCER repository, formatted
in a way that an LLM should be able to parse them more easily. An expected
use-case is for these files to be used in a RAG-based pipeline, which as of
writing have been developed to the point that using them should be as simple
as moving/uploading them the appropriate directory/service of a local. These
files should be useable with both open-source software for RAG that can be run
locally and with commercial "RAG-as-a-service" products.

Together with these files, we provide a base "system prompt" for such a pipeline:

+ `knowledge/prompt.txt` The system prompt for an AI assistant



