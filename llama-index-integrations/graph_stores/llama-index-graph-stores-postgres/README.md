# LlamaIndex PostgreSQL Graph Store

This package contains the PostgreSQL graph store integration for [LlamaIndex](https://github.com/run-llama/llama_index).

## Installation

```bash
pip install llama-index-graph-stores-postgres
```

Or with `uv`:

```bash
uv pip install llama-index-graph-stores-postgres
```

## Features

- Basic graph store implementation with PostgreSQL
- Property graph store implementation with pgvector support for vector search
- SQLAlchemy integration with Psycopg 3 for PostgreSQL connectivity

## Requirements

- PostgreSQL database with [pgvector](https://github.com/pgvector/pgvector) extension installed
- Python 3.8+

## Usage

### Basic Graph Store

```python
from llama_index.graph_stores.postgres import PostgresGraphStore

# Initialize the graph store
graph_store = PostgresGraphStore(
    db_connection_string="postgresql+psycopg://user:password@localhost:5432/dbname",
    entity_table_name="entities",
    relation_table_name="relations",
)

# Add triplets
graph_store.upsert_triplet("Alice", "KNOWS", "Bob")
graph_store.upsert_triplet("Bob", "WORKS_WITH", "Charlie")

# Get triplets
triplets = graph_store.get("Alice")
print(triplets)  # [["KNOWS", "Bob"]]

# Get relationship map with depth
rel_map = graph_store.get_rel_map(subjs=["Alice"], depth=2)
print(rel_map)
```

### Property Graph Store with pgvector

```python
import numpy as np
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation, VectorStoreQuery
from llama_index.graph_stores.postgres import PostgresPropertyGraphStore

# Initialize the property graph store
graph_store = PostgresPropertyGraphStore(
    db_connection_string="postgresql+psycopg://user:password@localhost:5432/dbname",
    embedding_dim=1536,  # Dimension of your embeddings
    node_table_name="pg_nodes",
    relation_table_name="pg_relations",
)

# Create nodes
alice = EntityNode(
    name="Alice",
    label="Person",
    properties={"age": 30, "occupation": "Engineer"},
)

document_chunk = ChunkNode(
    id="chunk1",
    text="This is a test document chunk",
    label="Chunk",
    properties={"source": "test"},
    embedding=np.random.rand(1536),  # Add embedding for vector search
)

# Add nodes
graph_store.add_nodes([alice, document_chunk])

# Create and add relation
relation = Relation(
    source_id=alice.id,
    target_id=document_chunk.id,
    label="CREATED",
    properties={"date": "2023-01-01"},
)
graph_store.add_relation(relation)

# Get nodes by properties
engineers = graph_store.get(properties={"occupation": "Engineer"})

# Get triplets
triplets = graph_store.get_triplets(entity_names=["Alice"])

# Vector search
query_vector = np.random.rand(1536)
query = VectorStoreQuery(query_embedding=query_vector, similarity_top_k=5)
results = graph_store.vector_search(query)
```

## Setting up PostgreSQL with pgvector

1. Install PostgreSQL and pgvector extension:

```bash
# For Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-server-dev-all
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

2. Create a database and enable the pgvector extension:

```sql
CREATE DATABASE vectordb;
\c vectordb
CREATE EXTENSION vector;
```

## Development

### Testing

To run tests, set the PostgreSQL connection string as an environment variable:

```bash
export POSTGRES_CONNECTION_STRING="postgresql+psycopg://user:password@localhost:5432/dbname"
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.