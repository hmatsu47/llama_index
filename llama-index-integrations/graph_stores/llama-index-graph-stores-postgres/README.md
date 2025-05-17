# PostgreSQL Graph Store

This package contains the PostgreSQL graph store integration for LlamaIndex, supporting Graph RAG with pgvector.

## Installation

```bash
pip install llama-index-graph-stores-postgres
```

## Usage

Here's an example of how to use the PostgreSQL graph store:

```python
from llama_index.graph_stores.postgres import PostgresGraphStore

# Create a simple graph store
graph_store = PostgresGraphStore(
    db_connection_string="postgresql://postgres:postgres@localhost:5432/postgres",
    entity_table_name="entities",
    relation_table_name="relations",
)

# Add triplets
graph_store.upsert_triplet("LlamaIndex", "is a", "RAG framework")
graph_store.upsert_triplet("LlamaIndex", "supports", "Graph RAG")
graph_store.upsert_triplet("Graph RAG", "uses", "Knowledge Graphs")

# Get relations
relations = graph_store.get("LlamaIndex")
print(relations)  # [["is a", "RAG framework"], ["supports", "Graph RAG"]]

# Get depth-aware relations
rel_map = graph_store.get_rel_map(subjs=["LlamaIndex"], depth=2)
print(rel_map)
```

For property graph store with vector search capabilities:

```python
from llama_index.graph_stores.postgres import PostgresPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation

# Create a property graph store with vector search
graph_store = PostgresPropertyGraphStore(
    db_connection_string="postgresql://postgres:postgres@localhost:5432/postgres",
    embedding_dim=1536,  # OpenAI embedding dimension
    node_table_name="pg_nodes",
    relation_table_name="pg_relations",
)

# Create nodes with embeddings
node1 = EntityNode(name="LlamaIndex", label="Product", properties={"type": "framework"})
node2 = EntityNode(name="Graph RAG", label="Concept", properties={"field": "AI"})

# Add nodes and relations
graph_store.upsert_nodes([node1, node2])
graph_store.upsert_relations([
    Relation(label="supports", source_id=node1.id, target_id=node2.id)
])

# Query the graph
triplets = graph_store.get_triplets(entity_names=["LlamaIndex"])
print(triplets)
```

## Requirements

- PostgreSQL 17 or higher
- pgvector 0.8.0 or higher

## Development

To set up for development:

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest tests/`