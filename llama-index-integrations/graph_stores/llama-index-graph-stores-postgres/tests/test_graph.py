"""Test PostgreSQL graph store."""
import os
import pytest
from typing import Generator

from llama_index.graph_stores.postgres import PostgresGraphStore

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
def graph_store(connection_string: str) -> Generator[PostgresGraphStore, None, None]:
    """Create a PostgreSQL graph store for testing."""
    store = PostgresGraphStore(
        db_connection_string=connection_string,
        entity_table_name="test_entities",
        relation_table_name="test_relations",
    )
    yield store
    
    # Clean up after tests
    with store.get_client.connect() as conn:
        conn.execute(f"TRUNCATE TABLE {store._relation_table_name}")
        conn.execute(f"TRUNCATE TABLE {store._entity_table_name}")
        conn.commit()


def test_upsert_triplet(graph_store: PostgresGraphStore):
    """Test upserting a triplet."""
    # Add a triplet
    graph_store.upsert_triplet("Alice", "KNOWS", "Bob")
    
    # Check that it was added
    triplets = graph_store.get("Alice")
    assert len(triplets) == 1
    assert triplets[0][0] == "KNOWS"
    assert triplets[0][1] == "Bob"


def test_get_rel_map(graph_store: PostgresGraphStore):
    """Test getting a relationship map."""
    # Add some triplets
    graph_store.upsert_triplet("Alice", "KNOWS", "Bob")
    graph_store.upsert_triplet("Bob", "WORKS_WITH", "Charlie")
    graph_store.upsert_triplet("Charlie", "REPORTS_TO", "Dave")
    
    # Get relationship map with depth 1
    rel_map = graph_store.get_rel_map(subjs=["Alice"], depth=1)
    assert len(rel_map) == 1
    assert len(rel_map["Alice"]) == 1
    assert rel_map["Alice"][0][1] == "KNOWS"
    assert rel_map["Alice"][0][2] == "Bob"
    
    # Get relationship map with depth 2
    rel_map = graph_store.get_rel_map(subjs=["Alice"], depth=2)
    assert len(rel_map) == 1
    assert len(rel_map["Alice"]) == 2
    
    # The first relation should be Alice KNOWS Bob
    assert rel_map["Alice"][0][0] == "Alice"
    assert rel_map["Alice"][0][1] == "KNOWS"
    assert rel_map["Alice"][0][2] == "Bob"
    
    # The second relation should be Bob WORKS_WITH Charlie
    assert rel_map["Alice"][1][0] == "Bob"
    assert rel_map["Alice"][1][1] == "WORKS_WITH"
    assert rel_map["Alice"][1][2] == "Charlie"


def test_delete_triplet(graph_store: PostgresGraphStore):
    """Test deleting a triplet."""
    # Add a triplet
    graph_store.upsert_triplet("Alice", "KNOWS", "Bob")
    
    # Delete the triplet
    graph_store.delete("Alice", "KNOWS", "Bob")
    
    # Check that it was deleted
    triplets = graph_store.get("Alice")
    assert len(triplets) == 0


def test_query(graph_store: PostgresGraphStore):
    """Test querying the graph store."""
    # Add some triplets
    graph_store.upsert_triplet("Alice", "KNOWS", "Bob")
    graph_store.upsert_triplet("Bob", "WORKS_WITH", "Charlie")
    
    # Query the graph store
    result = graph_store.query(
        f"""
        SELECT e1.name AS subject, r.description AS relation, e2.name AS object
        FROM {graph_store._relation_table_name} r
        JOIN {graph_store._entity_table_name} e1 ON r.subject_id = e1.id
        JOIN {graph_store._entity_table_name} e2 ON r.object_id = e2.id
        WHERE e1.name = :name
        """,
        {"name": "Alice"}
    )
    
    # Check the result
    assert len(result) == 1
    assert result[0][0] == "Alice"
    assert result[0][1] == "KNOWS"
    assert result[0][2] == "Bob"