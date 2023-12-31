# This file is an example of how a SQLAlchemy schema can be used with the 
# ConFIRM package.

import os
from sqlalchemy import (create_engine, MetaData, Table, Column, String, 
                        Integer, Float)
from dotenv import load_dotenv

load_dotenv()

# Create Tables
metadata = MetaData()
table_1_name = "table_1_name"
table_1_desc = "Add some table 1 description here."
table_1_column = Table(
  table_1_name, metadata,
  Column("id", Integer, primary_key=True, comment="Identifier"),
  Column("column name", Float, comment="some description"),
)
table_2_name = "table_2_name"
table_2_desc = "Add some table 2 description here."
market_data = Table(
  table_2_name, metadata,
  Column("id", Integer, primary_key=True, comment="Identifier"),
  Column("column name", Float, comment="some description"),
)

# Initialise Database
engine = create_engine(os.getenv("PSQL_SYNC_DSN"),
                       isolation_level="AUTOCOMMIT",
                       pool_pre_ping=True)

metadata.create_all(bind=engine)
metadata.reflect(bind=engine)

# We can also add fields as stand-ins for datasources outside the postgres database
external_fields = [
  "field_name_1 - introduction to this field",
  "field_name_2 - introduction to this field",
]

# Extract all fields from the metadata object
all_fields = [f"{table}.{field} - {metadata.tables[table].columns[field].comment}" for table in metadata.tables.keys() for field in metadata.tables[table].columns.keys() if field != "id"]
all_fields += external_fields
