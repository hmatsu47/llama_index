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
            
    def add_node(self, node: LabelledNode) -> None:
        """Add node."""
        with Session(self._engine) as session:
            if isinstance(node, ChunkNode):
                node_data = {
                    "id": node.id,
                    "text": node.text,
                    "label": node.label,
                    "properties": node.properties or {},
                    "embedding": node.embedding if hasattr(node, "embedding") else None,
                }
            else:
                node_data = {
                    "id": node.id,
                    "name": node.name,
                    "label": node.label,
                    "properties": node.properties or {},
                    "embedding": node.embedding if hasattr(node, "embedding") else None,
                }
            
            # Check if node exists
            existing_node = session.query(self._node_model).filter_by(id=node.id).first()
            if existing_node:
                # Update existing node
                for key, value in node_data.items():
                    if value is not None:
                        setattr(existing_node, key, value)
            else:
                # Create new node
                new_node = self._node_model(**node_data)
                session.add(new_node)
            
            session.commit()
    
    def add_nodes(self, nodes: List[LabelledNode]) -> None:
        """Add nodes."""
        for node in nodes:
            self.add_node(node)
    
    def add_relation(self, relation: Relation) -> None:
        """Add relation."""
        with Session(self._engine) as session:
            # Check if source and target nodes exist
            source_node = session.query(self._node_model).filter_by(id=relation.source_id).first()
            target_node = session.query(self._node_model).filter_by(id=relation.target_id).first()
            
            if not source_node or not target_node:
                raise ValueError(f"Source or target node not found for relation: {relation}")
            
            relation_data = {
                "label": relation.label,
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "properties": relation.properties or {},
            }
            
            # Check if relation exists
            existing_relation = session.query(self._relation_model).filter_by(
                source_id=relation.source_id,
                target_id=relation.target_id,
                label=relation.label
            ).first()
            
            if existing_relation:
                # Update existing relation
                for key, value in relation_data.items():
                    if value is not None:
                        setattr(existing_relation, key, value)
            else:
                # Create new relation
                new_relation = self._relation_model(**relation_data)
                session.add(new_relation)
            
            session.commit()
    
    def add_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        for relation in relations:
            self.add_relation(relation)
    
    def query_triplets(
        self,
        query: str,
        param_map: Optional[Dict[str, Any]] = None,
    ) -> List[Triplet]:
        """Query triplets."""
        with Session(self._engine) as session:
            result = session.execute(sql.text(query), param_map or {})
            triplets = []
            for row in result:
                # Convert row to triplet based on the query structure
                # This is a simplified implementation and may need to be adjusted
                # based on the actual query structure
                if len(row) >= 3:
                    source = EntityNode(
                        name=row[0],
                        label=row[1] if len(row) > 3 else "entity",
                    )
                    relation = Relation(
                        label=row[2],
                        source_id=source.id,
                        target_id=row[3] if len(row) > 3 else None,
                    )
                    target = EntityNode(
                        name=row[3] if len(row) > 3 else None,
                        label=row[4] if len(row) > 4 else "entity",
                    )
                    triplets.append([source, relation, target])
            return triplets
    
    def vector_search(
        self,
        query: VectorStoreQuery,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Vector search."""
        if query.query_embedding is None:
            raise ValueError("Query embedding is required for vector search")
        
        with Session(self._engine) as session:
            # Build the query
            db_query = session.query(
                self._node_model,
                self._node_model.embedding.cosine_distance(query.query_embedding).label("distance")
            )
            
            # Apply filters
            if ids:
                db_query = db_query.filter(self._node_model.id.in_(ids))
            
            # Apply similarity threshold if provided
            if query.similarity_top_k:
                db_query = db_query.order_by("distance").limit(query.similarity_top_k)
            
            # Execute query
            results = db_query.all()
            
            # Convert to nodes
            nodes = []
            for node, distance in results:
                if node.text and node.name is None:
                    nodes.append(
                        ChunkNode(
                            id=node.id,
                            text=node.text,
                            label=node.label,
                            properties=remove_empty_values(node.properties),
                            score=1.0 - distance,  # Convert distance to similarity score
                        )
                    )
                else:
                    nodes.append(
                        EntityNode(
                            name=node.name,
                            label=node.label,
                            properties=remove_empty_values(node.properties),
                            score=1.0 - distance,  # Convert distance to similarity score
                        )
                    )
            return nodes
    
    def delete_node(self, node_id: str) -> None:
        """Delete node."""
        with Session(self._engine) as session:
            # First delete all relations involving this node
            session.query(self._relation_model).filter(
                (self._relation_model.source_id == node_id) | 
                (self._relation_model.target_id == node_id)
            ).delete(synchronize_session=False)
            
            # Then delete the node
            session.query(self._node_model).filter_by(id=node_id).delete()
            session.commit()
    
    def delete_relation(self, relation: Relation) -> None:
        """Delete relation."""
        with Session(self._engine) as session:
            session.query(self._relation_model).filter_by(
                source_id=relation.source_id,
                target_id=relation.target_id,
                label=relation.label
            ).delete()
            session.commit()
    
    def clear(self) -> None:
        """Clear all nodes and relations."""
        with Session(self._engine) as session:
            session.query(self._relation_model).delete()
            session.query(self._node_model).delete()
            session.commit()