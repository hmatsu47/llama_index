import os
from unittest import TestCase, SkipTest

from llama_index.graph_stores.postgres import PostgresGraphStore


def get_store():
    return PostgresGraphStore(
        db_connection_string=os.environ.get("POSTGRES_TEST_CONNECTION_STRING"),
        entity_table_name="test_entities",
        relation_table_name="test_relations",
    )


class TestPostgresGraphStore(TestCase):
    @classmethod
    def setUp(self) -> None:
        try:
            get_store()
        except Exception:
            raise SkipTest("PostgreSQL database is not available")

    def test_add_get_triplet(self):
        g = get_store()
        g.upsert_triplet("subject", "relation", "object")
        triplets = g.get("subject")
        assert len(triplets) == 1
        assert triplets[0][0] == "relation"
        assert triplets[0][1] == "object"

    def test_delete_triplet(self):
        g = get_store()
        g.upsert_triplet("subject", "relation", "object")
        g.delete("subject", "relation", "object")
        triplets = g.get("subject")
        assert len(triplets) == 0

    def test_get_rel_map(self):
        g = get_store()
        g.upsert_triplet("subject1", "relation1", "object1")
        g.upsert_triplet("object1", "relation2", "object2")
        rel_map = g.get_rel_map(["subject1"], depth=2)
        assert len(rel_map) == 1
        assert len(rel_map["subject1"]) == 2
        assert rel_map["subject1"][0] == ["subject1", "relation1", "object1"]
        assert rel_map["subject1"][1] == ["object1", "relation2", "object2"]