# LlamaIndex Graph Stores Integration: PostgreSQL

PostgreSQL is a powerful, open-source object-relational database system with over 35 years of active development. With the pgvector extension, PostgreSQL supports vector operations, making it suitable for AI applications and vector similarity searches.

In this project, we integrate PostgreSQL as the graph store to store the LlamaIndex graph data, and use PostgreSQL's SQL interface to query the graph data, so that people can use PostgreSQL to interact with LlamaIndex graph index.

- Property Graph Store: `PostgresPropertyGraphStore`
- Knowledge Graph Store: `PostgresGraphStore`

## Installation

```shell
pip install llama-index
pip install git+https://github.com/hmatsu47/llama-index-graph-stores-postgres.git
```

## Usage

### Property Graph Store

NOTE: `PostgresPropertyGraphStore` requires the pgvector extension to be installed in your PostgreSQL database.

Simple example to use `PostgresPropertyGraphStore`:

```python
from llama_index.core import PropertyGraphIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.bedrock import BedrockEmbedding, Models
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.graph_stores.postgres import PostgresPropertyGraphStore

documents = SimpleDirectoryReader("../../../examples/data/paul_graham/").load_data()

graph_store = PostgresPropertyGraphStore(
    db_connection_string="postgresql://user:password@host:5432/dbname",
)

llm = BedrockConverse(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-west-2",
    temperature=0.0,
)
embed_model = BedrockEmbedding(
    model_name=Models.TITAN_EMBEDDING_V2_0, region_name="us-west-2"
)

Settings.llm = llm
Settings.embed_model = embed_model

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=embed_model,
    kg_extractors=[
        SimpleLLMPathExtractor(llm=llm),
        ImplicitPathExtractor(),
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("What happened at Interleaf and Viaweb?")
print(response)
```

### Knowledge Graph Store

Simple example to use `PostgresGraphStore`:

```python
from llama_index.graph_stores.postgres import PostgresGraphStore
from llama_index.core import (
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
)

documents = SimpleDirectoryReader(
    "../../../examples/data/paul_graham/"
).load_data()

graph_store = PostgresGraphStore(
    db_connection_string="postgresql://user:password@host:5432/dbname"
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
index = KnowledgeGraphIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
)
query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)
print(response)
```
