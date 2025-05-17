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