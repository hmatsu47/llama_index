from llama_index.core.graph_stores.types import GraphStore
from llama_index.graph_stores.postgres import PostgresGraphStore


def test_postgres_graph_store():
    names_of_bases = [b.__name__ for b in PostgresGraphStore.__bases__]
    assert GraphStore.__name__ in names_of_bases
