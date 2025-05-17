"""PostgreSQL property graph store index."""
import json
from typing import Tuple, Optional, List, Dict, Any
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    TEXT,
    ForeignKey,
    sql,
    delete,
)
from sqlalchemy.orm import (
    Session,
    declarative_base,
    relationship,
    joinedload,
)

from pgvector.sqlalchemy import Vector
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    LabelledNode,
    EntityNode,
    ChunkNode,
    Relation,
    Triplet,
    VectorStoreQuery,
)
from llama_index.graph_stores.postgres.utils import (
    check_db_availability,
    remove_empty_values,
    get_or_create,
)


rel_depth_query = """
WITH RECURSIVE PATH AS
  (SELECT 1 AS depth,
          r.source_id,
          r.target_id,
          r.label,
          r.properties
   FROM {relation_table} r
   WHERE r.source_id IN :ids
   UNION ALL SELECT p.depth + 1,
                    r.source_id,
                    r.target_id,
                    r.label,
                    r.properties
   FROM PATH p
   JOIN {relation_table} r ON p.target_id = r.source_id
   WHERE p.depth < :depth )
SELECT e1.id AS e1_id,
       e1.name AS e1_name,
       e1.label AS e1_label,
       e1.properties AS e1_properties,
       p.label AS rel_label,
       p.properties AS rel_properties,
       e2.id AS e2_id,
       e2.name AS e2_name,
       e2.label AS e2_label,
       e2.properties AS e2_properties
FROM PATH p
JOIN {node_table} e1 ON p.source_id = e1.id
JOIN {node_table} e2 ON p.target_id = e2.id
ORDER BY p.depth
LIMIT :limit;
"""


class PostgresPropertyGraphStore(PropertyGraphStore):
    # PostgreSQL does not support graph cypher queries
    supports_structured_queries: bool = False
    supports_vector_queries: bool = True

    def __init__(
        self,
        db_connection_string: str,
        embedding_dim: int = 1536,
        node_table_name: str = "pg_nodes",
        relation_table_name: str = "pg_relations",
        drop_existing_table: bool = False,
        echo_queries: bool = False,
    ) -> None:
        self._engine = create_engine(
            db_connection_string, echo=echo_queries
        )
        check_db_availability(self._engine, check_vector=True)

        self._embedding_dim = embedding_dim
        self._node_table_name = node_table_name
        self._relation_table_name = relation_table_name
        self._drop_existing_table = drop_existing_table
        self._node_model, self._relation_model = self.init_schema()

    def init_schema(self) -> Tuple:
        """Initialize schema."""
        Base = declarative_base()

        class BaseMixin:
            created_at = Column(DateTime, nullable=False, server_default=sql.func.now())
            updated_at = Column(
                DateTime,
                nullable=False,
                server_default=sql.func.now(),
                onupdate=sql.func.now(),
            )

        class NodeModel(BaseMixin, Base):
            __tablename__ = self._node_table_name
            id = Column(String(512), primary_key=True)
            text = Column(TEXT, nullable=True)
            name = Column(String(512), nullable=True)
            label = Column(String(512), nullable=False, default="node")
            properties = Column(JSON, default={})
            embedding = Column(Vector(self._embedding_dim))

        class RelationModel(BaseMixin, Base):
            __tablename__ = self._relation_table_name
            id = Column(Integer, primary_key=True)
            label = Column(String(512), nullable=False)
            source_id = Column(String(512), ForeignKey(f"{self._node_table_name}.id"))
            target_id = Column(String(512), ForeignKey(f"{self._node_table_name}.id"))
            properties = Column(JSON, default={})

            source = relationship("NodeModel", foreign_keys=[source_id])
            target = relationship("NodeModel", foreign_keys=[target_id])

        if self._drop_existing_table:
            Base.metadata.drop_all(self._engine)
        Base.metadata.create_all(self._engine)
        return NodeModel, RelationModel
        
    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        with Session(self._engine) as session:
            query = session.query(self._node_model)
            if properties:
                for key, value in properties.items():
                    query = query.filter(self._node_model.properties[key] == value)
            if ids:
                query = query.filter(self._node_model.id.in_(ids))

            nodes = []
            for n in query.all():
                if n.text and n.name is None:
                    nodes.append(
                        ChunkNode(
                            id=n.id,
                            text=n.text,
                            label=n.label,
                            properties=remove_empty_values(n.properties),
                        )
                    )
                else:
                    nodes.append(
                        EntityNode(
                            name=n.name,
                            label=n.label,
                            properties=remove_empty_values(n.properties),
                        )
                    )
            return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets."""
        # if nothing is passed, return empty list
        if not ids and not properties and not entity_names and not relation_names:
            return []

        with Session(self._engine) as session:
            query = session.query(self._relation_model).options(
                joinedload(self._relation_model.source),
                joinedload(self._relation_model.target),
            )
            if ids:
                query = query.filter(
                    self._relation_model.source_id.in_(ids)
                    | self._relation_model.target_id.in_(ids)
                )
            if properties:
                for key, value in properties.items():
                    query = query.filter(
                        (self._relation_model.properties[key] == value)
                        | self._relation_model.source.has(
                            self._node_model.properties[key] == value
                        )
                        | self._relation_model.target.has(
                            self._node_model.properties[key] == value
                        )
                    )
            if entity_names:
                query = query.filter(
                    self._relation_model.source.has(
                        self._node_model.name.in_(entity_names)
                    )
                    | self._relation_model.target.has(
                        self._node_model.name.in_(entity_names)
                    )
                )
            if relation_names:
                query = query.filter(self._relation_model.label.in_(relation_names))

            triplets = []
            for r in query.all():
                source = EntityNode(
                    name=r.source.name,
                    label=r.source.label,
                    properties=remove_empty_values(r.source.properties),
                )
                target = EntityNode(
                    name=r.target.name,
                    label=r.target.label,
                    properties=remove_empty_values(r.target.properties),
                )
                relation = Relation(
                    label=r.label,
                    source_id=source.id,
                    target_id=target.id,
                    properties=remove_empty_values(r.properties),
                )
                triplets.append([source, relation, target])
            return triplets