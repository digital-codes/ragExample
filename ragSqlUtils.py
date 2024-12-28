from sqlalchemy import create_engine, Column, Integer, String, Text, text, ForeignKey, LargeBinary, DateTime, MetaData, func
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from sqlalchemy import select, join

import numpy as np

import private_remote as pr

# Create the Declarative Base
Base = declarative_base()

# Define the database schema as before...
class Project(Base):
    """
    Represents a project entity in the database.

    Attributes:
        id (int): The primary key of the project.
        name (str): The name of the project. Cannot be null.
        description (str): A textual description of the project. Can be null.
    """
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)

class Item(Base):
    """
    Represents an item in the database.

    Attributes:
        id (int): The primary key of the item.
        name (str): The unique name of the item.
        code (int): The code associated with the item.
        project_id (int): The foreign key referencing the associated project.
        summary (str, optional): A brief summary of the item.
        fulltext (str, optional): The full text description of the item.
        tags (str, optional): Tags associated with the item.
        title (str, optional): The title of the item.
        created (datetime, optional): The creation date of the item. Defaults to the current date.
        modified (datetime, optional): The last modified date of the item.
        url (str, optional): The URL associated with the item.
        license (str, optional): The license information of the item.
        itemIndex (int): The index of the item.

    Relationships:
        project (Project): The project to which the item belongs.
    """
    __tablename__ = 'items'

    id = Column(Integer, primary_key=True)
    name = Column(String(256), unique=True, nullable=False)
    code = Column(Integer, nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    summary = Column(Text, nullable=True)
    fulltext = Column(Text, nullable=True)
    tags = Column(String(1024), nullable=True)
    title = Column(String(256), nullable=True)
    created = Column(DateTime, nullable=True, default=func.current_date())
    modified = Column(DateTime, nullable=True)
    url = Column(String(1024), nullable=True)
    license = Column(String(256), nullable=True)
    itemIndex = Column(Integer, nullable=False)

    project = relationship("Project", back_populates="items")

Project.items = relationship("Item", order_by=Item.id, back_populates="project")

class Chunk(Base):
    """
    Represents a chunk of text associated with an item.

    Attributes:
        id (int): Primary key of the chunk.
        chunkIdx (int): Index of the chunk.
        item_id (int): Foreign key referencing the associated item.
        text (str): The text content of the chunk.
        item (Item): Relationship to the Item model, back_populated by 'chunks'.
    """
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True)
    chunkIdx = Column(Integer, nullable=False)
    item_id = Column(Integer, ForeignKey('items.id'), nullable=False)
    text = Column(Text, nullable=True)

    item = relationship("Item", back_populates="chunks")

Item.chunks = relationship("Chunk", order_by=Chunk.id, back_populates="item")

class Vector(Base):
    """
    Represents a vector in the database.

    Attributes:
        id (int): The primary key of the vector.
        chunk_id (int): The foreign key referencing the associated chunk.
        value (bytes): The binary data representing the vector.
        chunk (Chunk): The relationship to the Chunk model, back_populated by "vectors".
    """
    __tablename__ = 'vectors'

    id = Column(Integer, primary_key=True)
    chunk_id = Column(Integer, ForeignKey('chunks.id'), nullable=False)
    value = Column(LargeBinary, nullable=False)  # Blob

    chunk = relationship("Chunk", back_populates="vectors")

Chunk.vectors = relationship("Vector", order_by=Vector.id, back_populates="chunk")

class TitleVector(Base):
    """
    TitleVector is a SQLAlchemy model representing the 'title_vectors' table.

    Attributes:
        id (int): Primary key of the table.
        item_id (int): Foreign key referencing the 'items' table.
        value (LargeBinary): Blob data representing the vector value.
        item (relationship): Relationship to the Item model, back_populated by 'title_vectors'.
    """
    __tablename__ = 'title_vectors'

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey('items.id'), nullable=False)
    value = Column(LargeBinary, nullable=False)  # Blob

    item = relationship("Item", back_populates="title_vectors")

Item.title_vectors = relationship("TitleVector", order_by=TitleVector.id, back_populates="item")

# Update the database connection
def setup_database():
    """
    Sets up the database connection and creates all tables defined in the metadata.

    This function uses SQLAlchemy to create an engine with the MySQL connection string
    provided in the `pr.mysql` dictionary. It then creates all tables defined in the
    `Base` metadata.

    Returns:
        sqlalchemy.engine.Engine: The SQLAlchemy engine connected to the MySQL database.
    """
    # Replace with your MySQL connection string
    engine = create_engine(f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}')
    Base.metadata.create_all(engine)
    return engine

# create a session
def create_session(engine):
    """
    Create a new SQLAlchemy session.

    Args:
        engine (Engine): The SQLAlchemy engine to bind the session to.

    Returns:
        Session: A new SQLAlchemy session.
    """
    Session = sessionmaker(bind=engine)
    return Session()


def delete_all():
    """
    Deletes all data from all tables in the MySQL database and drops all tables.

    This function connects to a MySQL database using SQLAlchemy, disables foreign key checks,
    deletes all data from all tables, drops all tables, and then re-enables foreign key checks.

    Note:
        - The database connection parameters are retrieved from the `pr.mysql` dictionary.
        - The function reflects the database schema to get the list of tables.
        - Tables are dropped in reverse order to respect foreign key constraints.

    Raises:
        Exception: If an error occurs during the execution of SQL commands, it will be caught and printed.

    """
    engine = create_engine(f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}')
    # Reflect the database schema
    meta = MetaData()
    meta.reflect(bind=engine)

    # Connect to the database
    with engine.connect() as conn:
        try:
            # Disable foreign key checks
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))

            # Delete all data from all tables
            # and drop all tables
            for table in reversed(meta.sorted_tables):  # Reverse order to respect FK constraints
                print(f"Deleting data from table: {table.name}")
                conn.execute(table.delete())
                conn.execute(text(f"DROP TABLE {table.name};"))

            # Re-enable foreign key checks
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        except Exception as e:
                    print(f"An error occurred: {e}")
        finally:
            conn.close()

    engine.dispose()        
        
def find_chunk(session: Session, chunk_idx: int, project_id: int):
    """
    Find a Chunk by its index and project ID.

    Args:
        session (Session): The SQLAlchemy session to use for the query.
        chunk_idx (int): The index of the chunk to find.
        project_id (int): The ID of the project to which the chunk belongs.

    Returns:
        Chunk: The found Chunk object, or None if no matching chunk is found.
    """
    stmt = (
        select(Chunk)
        .join(Item, Chunk.item_id == Item.id)  # Join Chunk -> Item
        .where(Chunk.chunkIdx == chunk_idx, Item.project_id == project_id)  # Conditions
    )

    # Execute the query
    result = session.execute(stmt).scalars().first()
    return result

def find_item(session: Session, chunk_idx: int, project_id: int):
    """
    Find an Item by a chunk index and project ID.

    Args:
        session (Session): The SQLAlchemy session to use for the query.
        chunk_idx (int): The index of the chunk to find.
        project_id (int): The ID of the project to which the chunk belongs.

    Returns:
        Item: The found Item object, or None if no matching Item is found.
    """
    stmt = (
        select(Item)
        .join(Chunk, Chunk.item_id == Item.id)  # Join Chunk -> Item
        .where(Chunk.chunkIdx == chunk_idx, Item.project_id == project_id)  # Conditions
    )

    # Execute the query
    result = session.execute(stmt).scalars().first()
    return result


def get_items(session: Session, project_id: int):
    """
    Get a list of all items for a given projectId, ordered by itemIdx (ascending).

    :param session: SQLAlchemy Session object
    :param project_id: ID of the project
    :return: List of Item objects
    """
    stmt = (
        select(Item)
        .where(Item.project_id == project_id)
        .order_by(Item.itemIndex.asc())  # Order by itemIndex in ascending order
    )

    # Execute the query
    result = session.execute(stmt).scalars().all()
    return result


def get_item(session: Session, name: str = None, code: int = None):
    """
    Get an item by name or code, where only one of the parameters is provided.

    :param session: SQLAlchemy Session object
    :param name: Name of the item (optional)
    :param code: Code of the item (optional)
    :return: Item object or None if not found
    :raises ValueError: If neither or both parameters are provided
    """
    if not (name or code):
        raise ValueError("Either name or code must be provided.")
    if name and code:
        raise ValueError("Only one of name or code must be provided, not both.")

    stmt = select(Item)
    if name:
        stmt = stmt.where(Item.name == name)
    elif code:
        stmt = stmt.where(Item.code == code)

    # Execute the query
    result = session.execute(stmt).scalars().first()
    return result


def get_table_layout(engine, table_name):
    """
    Retrieve the layout of a specific table in the database.

    :param engine: SQLAlchemy Engine
    :param table_name: Name of the table
    :return: Dictionary with column details
    """
    meta = MetaData()
    meta.reflect(bind=engine)
    table = meta.tables.get(table_name)
    
    if table == None:
        return f"Table '{table_name}' does not exist in the database."
    
    layout = []
    for column in table.columns:
        column_info = {
            "name": column.name,
            "type": str(column.type),
            "nullable": column.nullable,
            "default": column.default,
            "primary_key": column.primary_key,
            "unique": column.unique
        }
        layout.append(column_info)
    return layout



# Test function to populate and query dummy data
def test_database():
    delete_all()
    engine = setup_database()
    layout = get_table_layout(engine,"items")
    print(layout)
    session = create_session(engine)

    # Create dummy projects
    project1 = Project(name="Project Alpha", description="Description of Project Alpha")
    project2 = Project(name="Project Beta", description="Description of Project Beta")
    session.add_all([project1, project2])
    session.commit()

    # Create dummy items
    item1 = Item(name="Item One", code=101, project_id=project1.id, summary="Summary of item one", fulltext="Fulltext for item one", tags="tag1,tag2", title="Title One", itemIndex=1)
    item2 = Item(name="Item Two", code=102, project_id=project2.id, summary="Summary of item two", fulltext="Fulltext for item two", tags="tag3,tag4", title="Title Two", itemIndex=2)
    session.add_all([item1, item2])
    session.commit()

    # Create dummy chunks
    chunkIds = []
    chunk = Chunk(chunkIdx=1, item_id=item1.id, text="Chunk 1 text")
    session.add(chunk)
    session.flush()
    chunkIds.append(chunk.id)
    chunk = Chunk(chunkIdx=2, item_id=item2.id, text="Chunk 2 text")
    session.add(chunk)
    session.flush()
    chunkIds.append(chunk.id)
    session.commit()

    # Create dummy vectors
    vector = np.random.rand(384).astype('float32')
    binary_data = vector.tobytes()
    # Insert into the database
    vector1 = Vector(chunk_id=chunkIds[0], value=binary_data)
    vector = np.random.rand(384).astype('float32')
    binary_data = vector.tobytes()
    # Insert into the database
    vector2 = Vector(chunk_id=chunkIds[1], value=binary_data)
    session.add_all([vector1, vector2])
    session.commit()
    
    # Create dummy title_vectors
    vector = np.random.rand(384).astype('float32')
    binary_data = vector.tobytes()
    # Insert into the database
    vector1 = TitleVector(item_id=item1.id, value=binary_data)
    vector = np.random.rand(384).astype('float32')
    binary_data = vector.tobytes()
    # Insert into the database
    vector2 = TitleVector(item_id=item1.id, value=binary_data)
    session.add_all([vector1, vector2])
    session.commit()
    
    

    # Query and print data
    projects = session.query(Project).all()
    for project in projects:
        print(f"Project: {project.name}, Description: {project.description}")

    items = session.query(Item).all()
    for item in items:
        print(f"Item: {item.name}, Code: {item.code}, Tags: {item.tags}")

    chunks = session.query(Chunk).all()
    for chunk in chunks:
        print(f"Chunk: {chunk.text}, Index: {chunk.chunkIdx}")

    vectors = session.query(Vector).all()
    for vector in vectors:
        value = np.frombuffer(vector.value, dtype='float32')
        print(f"Vector: {value[:10]}...")  # Print the first 10 characters of the vector

    vectors = session.query(TitleVector).all()
    for vector in vectors:
        value = np.frombuffer(vector.value, dtype='float32')
        print(f"TitleVector: {value[:10]}...")  # Print the first 10 characters of the vector

    print(find_chunk(session, 1, 1).text)
    print(find_item(session, 1, 1).title)



if __name__ == "__main__":
    test_database()
