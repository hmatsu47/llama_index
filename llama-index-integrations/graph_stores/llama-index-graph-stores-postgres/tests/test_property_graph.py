"""Test PostgreSQL property graph store."""
import os
import pytest
import numpy as np
from typing import List, Generator

from llama_index.core.graph_stores.types import (
    LabelledNode,
    EntityNode,
    ChunkNode,
    Relation,
    VectorStoreQuery,
)
from llama_index.graph_stores.postgres import PostgresPropertyGraphStore

# Skip tests if environment variables are not set
pytestmark = pytest.mark.skipif(
    "POSTGRES_CONNECTION_STRING" not in os.environ,
    reason="PostgreSQL connection string not provided",
)


@pytest.fixture
def connection_string() -> str:
    """Get PostgreSQL connection string from environment variable."""
    return os.environ.get(
        "POSTGRES_CONNECTION_STRING",
        "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
    )


@pytest.fixture
def graph_store(connection_string: str) -> Generator[PostgresPropertyGraphStore, None, None]:
    """Create a PostgreSQL property graph store for testing."""
    store = PostgresPropertyGraphStore(
        db_connection_string=connection_string,
        embedding_dim=4,  # Small dimension for testing
        node_table_name="test_pg_nodes",
        relation_table_name="test_pg_relations",
        drop_existing_table=True,  # Always start with a clean slate for tests
    )
    yield store
    # Clean up after tests
    store.clear()


@pytest.fixture
def sample_nodes() -> List[LabelledNode]:
    """Create sample nodes for testing."""
    return [
        EntityNode(
            name="Alice",
            label="Person",
            properties={"age": 30, "occupation": "Engineer"},
        ),
        EntityNode(
            name="Bob",
            label="Person",
            properties={"age": 25, "occupation": "Designer"},
        ),
        ChunkNode(
            id="chunk1",
            text="This is a test chunk",
            label="Chunk",
            properties={"source": "test"},
            embedding=np.array([0.1, 0.2, 0.3, 0.4]),
        ),
    ]


@pytest.fixture
def sample_relations() -> List[Relation]:
    """Create sample relations for testing."""
    return [
        Relation(
            source_id="Person/Alice",
            target_id="Person/Bob",
            label="KNOWS",
            properties={"since": 2020},
        ),
        Relation(
            source_id="Person/Alice",
            target_id="Chunk/chunk1",
            label="CREATED",
            properties={"date": "2023-01-01"},
        ),
    ]


def test_add_and_get_nodes(graph_store: PostgresPropertyGraphStore, sample_nodes: List[LabelledNode]):
    """Test adding and retrieving nodes."""
    # Add nodes
    graph_store.add_nodes(sample_nodes)
    
    # Get all nodes
    nodes = graph_store.get()
    assert len(nodes) == 3
    
    # Get nodes by property
    engineers = graph_store.get(properties={"occupation": "Engineer"})
    assert len(engineers) == 1
    assert engineers[0].properties["occupation"] == "Engineer"
    
    # Get nodes by ID
    chunks = graph_store.get(ids=["Chunk/chunk1"])
    assert len(chunks) == 1
    assert chunks[0].id == "Chunk/chunk1"


def test_add_and_get_relations(
    graph_store: PostgresPropertyGraphStore, 
    sample_nodes: List[LabelledNode],
    sample_relations: List[Relation]
):
    """Test adding and retrieving relations."""
    # Add nodes first
    graph_store.add_nodes(sample_nodes)
    
    # Add relations
    graph_store.add_relations(sample_relations)
    
    # Get triplets by entity name
    triplets = graph_store.get_triplets(entity_names=["Alice"])
    assert len(triplets) == 2
    
    # Get triplets by relation name
    knows_triplets = graph_store.get_triplets(relation_names=["KNOWS"])
    assert len(knows_triplets) == 1
    assert knows_triplets[0][1].label == "KNOWS"
    
    # Get triplets by property
    date_triplets = graph_store.get_triplets(properties={"date": "2023-01-01"})
    assert len(date_triplets) == 1
    assert date_triplets[0][1].properties["date"] == "2023-01-01"


def test_vector_search(
    graph_store: PostgresPropertyGraphStore, 
    sample_nodes: List[LabelledNode]
):
    """Test vector search functionality."""
    # Add nodes with embeddings
    graph_store.add_nodes(sample_nodes)
    
    # Create a query vector
    query_vector = np.array([0.1, 0.2, 0.3, 0.4])
    query = VectorStoreQuery(query_embedding=query_vector, similarity_top_k=1)
    
    # Perform vector search
    results = graph_store.vector_search(query)
    
    # Check results
    assert len(results) == 1
    assert results[0].id == "Chunk/chunk1"
    assert results[0].score is not None
    assert results[0].score > 0.9  # Should be very similar to itself


def test_delete_node(
    graph_store: PostgresPropertyGraphStore, 
    sample_nodes: List[LabelledNode],
    sample_relations: List[Relation]
):
    """Test deleting a node and its relations."""
    # Add nodes and relations
    graph_store.add_nodes(sample_nodes)
    graph_store.add_relations(sample_relations)
    
    # Delete a node
    graph_store.delete_node("Person/Alice")
    
    # Check that the node is gone
    nodes = graph_store.get(ids=["Person/Alice"])
    assert len(nodes) == 0
    
    # Check that relations involving the node are gone
    triplets = graph_store.get_triplets(entity_names=["Alice"])
    assert len(triplets) == 0


def test_delete_relation(
    graph_store: PostgresPropertyGraphStore, 
    sample_nodes: List[LabelledNode],
    sample_relations: List[Relation]
):
    """Test deleting a relation."""
    # Add nodes and relations
    graph_store.add_nodes(sample_nodes)
    graph_store.add_relations(sample_relations)
    
    # Delete a relation
    relation = sample_relations[0]
    graph_store.delete_relation(relation)
    
    # Check that the relation is gone
    triplets = graph_store.get_triplets(relation_names=["KNOWS"])
    assert len(triplets) == 0
    
    # But other relations should still exist
    triplets = graph_store.get_triplets(relation_names=["CREATED"])
    assert len(triplets) == 1


def test_clear(
    graph_store: PostgresPropertyGraphStore, 
    sample_nodes: List[LabelledNode],
    sample_relations: List[Relation]
):
    """Test clearing all nodes and relations."""
    # Add nodes and relations
    graph_store.add_nodes(sample_nodes)
    graph_store.add_relations(sample_relations)
    
    # Clear the graph store
    graph_store.clear()
    
    # Check that everything is gone
    nodes = graph_store.get()
    assert len(nodes) == 0
    
    triplets = graph_store.get_triplets()
    assert len(triplets) == 0