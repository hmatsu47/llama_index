# PostgreSQL Graph Store

This package contains the PostgreSQL graph store integration for LlamaIndex.

## Installation

```bash
pip install llama-index-graph-stores-postgres
```

## Usage

### Property Graph Store

```python
from llama_index.graph_stores.postgres import PostgresPropertyGraphStore

# Initialize the graph store
graph_store = PostgresPropertyGraphStore(
    db_connection_string="postgresql+psycopg://user:password@localhost:5432/db_name",
    embedding_dim=1536,  # Dimension of your embeddings
    node_table_name="pg_nodes",  # Optional, defaults to "pg_nodes"
    relation_table_name="pg_relations",  # Optional, defaults to "pg_relations"
    drop_existing_table=False,  # Optional, defaults to False
    echo_queries=False,  # Optional, defaults to False
)

# Create nodes
from llama_index.core.graph_stores.types import EntityNode, Relation

# Create entity nodes
entity1 = EntityNode(name="entity1", label="person", properties={"age": 30})
entity2 = EntityNode(name="entity2", label="company", properties={"founded": 2020})

# Add nodes to the graph store
graph_store.upsert_nodes([entity1, entity2])

# Create a relation between nodes
relation = Relation(
    label="works_at",
    source_id=entity1.id,
    target_id=entity2.id,
    properties={"since": 2021}
)

# Add relation to the graph store
graph_store.upsert_relations([relation])

# Query the graph store
triplets = graph_store.get_triplets(entity_names=["entity1"])
nodes = graph_store.get(properties={"age": 30})

# Delete data
graph_store.delete(entity_names=["entity1"])
```

### Knowledge Graph Store

```python
from llama_index.graph_stores.postgres import PostgresGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import KnowledgeGraphIndex

# Initialize the graph store
graph_store = PostgresGraphStore(
    db_connection_string="postgresql+psycopg://user:password@localhost:5432/db_name",
    table_name="kg_table",  # Optional, defaults to "kg_triplets"
    drop_existing_table=False,  # Optional, defaults to False
)

# Create a storage context with the graph store
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Create a knowledge graph index
index = KnowledgeGraphIndex(
    [],  # Empty document list for now
    storage_context=storage_context,
)

# Add documents to the index
from llama_index.core import Document
documents = [Document(text="Alice works at Google. Bob works at Microsoft.")]
index.insert(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("Where does Alice work?")
print(response)
```