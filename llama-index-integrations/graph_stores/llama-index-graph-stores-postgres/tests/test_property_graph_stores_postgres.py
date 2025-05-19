import os
from unittest import TestCase, SkipTest

from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    VectorStoreQuery,
)

from llama_index.graph_stores.postgres import PostgresPropertyGraphStore


def get_store():
    return PostgresPropertyGraphStore(
        db_connection_string=os.environ.get("POSTGRES_TEST_CONNECTION_STRING"),
        drop_existing_table=True,
        relation_table_name="test_relations",
        node_table_name="test_nodes",
    )


class TestPostgresPropertyGraphStore(TestCase):
    @classmethod
    def setUp(self) -> None:
        try:
            get_store()
        except Exception:
            raise SkipTest("PostgreSQL database is not available")

        self.e1 = EntityNode(name="e1", properties={"p1": "v1"})
        self.e2 = EntityNode(name="e2")
        self.r = Relation(label="r", source_id=self.e1.id, target_id=self.e2.id)

    def test_add(self):
        g = get_store()

        g.upsert_nodes([self.e1, self.e2])
        g.upsert_relations([self.r])
        assert len(g.get_triplets(entity_names=["e1"])) == 1
        assert len(g.get_triplets(entity_names=["e3"])) == 0
        assert len(g.get_triplets(properties={"p1": "v1"})) == 1
        assert len(g.get_triplets(properties={"p1": "v2"})) == 0

    def test_delete_by_entity_names(self):
        g = get_store()

        g.upsert_nodes([self.e1, self.e2])
        g.upsert_relations([self.r])
        assert len(g.get_triplets(entity_names=["e1"])) == 1
        g.delete(entity_names=["e1"])
        assert len(g.get_triplets(entity_names=["e1"])) == 0

    def test_delete_by_entity_properties(self):
        g = get_store()

        g.upsert_nodes([self.e1, self.e2])
        g.upsert_relations([self.r])
        assert len(g.get_triplets(entity_names=["e1"])) == 1
        g.delete(properties={"p1": "not exist"})
        assert len(g.get_triplets(entity_names=["e1"])) == 1
        g.delete(properties={"p1": "v1"})
        assert len(g.get_triplets(entity_names=["e1"])) == 0

    def test_get(self):
        g = get_store()

        g.upsert_nodes([self.e1, self.e2])
        assert len(g.get(ids=[self.e1.id])) == 1
        assert len(g.get(ids=[self.e1.id, self.e2.id])) == 2
        assert len(g.get(properties={"p1": "v1"})) == 1

    def test_get_rel_map(self):
        g = get_store()

        g.upsert_nodes([self.e1, self.e2])
        g.upsert_relations([self.r])
        
        rel_map = g.get_rel_map([self.e1])
        assert len(rel_map) == 1
        
        rel_map = g.get_rel_map([self.e2])
        assert len(rel_map) == 0

    def test_vector_query(self):
        g = get_store()
        
        # Create nodes with embeddings
        e1 = EntityNode(name="e1", properties={"p1": "v1"})
        e1.embedding = [0.1] * 1536
        e2 = EntityNode(name="e2")
        e2.embedding = [0.2] * 1536
        
        g.upsert_nodes([e1, e2])
        
        # Create a vector query
        query = VectorStoreQuery(
            query_embedding=[0.1] * 1536,
            similarity_top_k=1
        )
        
        # Test vector query
        nodes, scores = g.vector_query(query)
        assert len(nodes) == 1
        assert len(scores) == 1