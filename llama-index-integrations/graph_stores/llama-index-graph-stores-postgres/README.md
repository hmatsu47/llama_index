# LlamaIndex - PostgreSQL Graph Store

This package contains the PostgreSQL graph store integration with pgvector support for LlamaIndex.

## Installation

```bash
pip install llama-index-graph-stores-postgres
```

## Usage

```python
from llama_index.graph_stores.postgres import PostgresPropertyGraphStore

# Initialize the graph store
graph_store = PostgresPropertyGraphStore(
    db_connection_string="postgresql://user:password@localhost:5432/db_name",
    embedding_dim=1536,  # Dimension of your embeddings
)

# Use the graph store with LlamaIndex
from llama_index.core import StorageContext
from llama_index.core.indices import KnowledgeGraphIndex

storage_context = StorageContext.from_defaults(graph_store=graph_store)
index = KnowledgeGraphIndex([], storage_context=storage_context)
```

## Requirements

- PostgreSQL with pgvector extension installed
- Python 3.9+