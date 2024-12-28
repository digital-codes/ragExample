from sqlalchemy import create_engine, Column, Integer, String, Text, text, ForeignKey, LargeBinary, DateTime, MetaData, func
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from sqlalchemy import select, join

import numpy as np

import private_remote as pr

# Create the Declarative Base
Base = declarative_base()

# Define the database schema as before...
class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)

class Item(Base):
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
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True)
    chunkIdx = Column(Integer, nullable=False)
    item_id = Column(Integer, ForeignKey('items.id'), nullable=False)
    text = Column(Text, nullable=True)

    item = relationship("Item", back_populates="chunks")

Item.chunks = relationship("Chunk", order_by=Chunk.id, back_populates="item")

class Vector(Base):
    __tablename__ = 'vectors'

    id = Column(Integer, primary_key=True)
    chunk_id = Column(Integer, ForeignKey('chunks.id'), nullable=False)
    value = Column(LargeBinary, nullable=False)  # Blob

    chunk = relationship("Chunk", back_populates="vectors")

Chunk.vectors = relationship("Vector", order_by=Vector.id, back_populates="chunk")

class TitleVector(Base):
    __tablename__ = 'title_vectors'

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey('items.id'), nullable=False)
    value = Column(LargeBinary, nullable=False)  # Blob

    item = relationship("Item", back_populates="title_vectors")

Item.title_vectors = relationship("TitleVector", order_by=TitleVector.id, back_populates="item")

# Update the database connection
def setup_database():
    # Replace with your MySQL connection string
    engine = create_engine(f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}')
    Base.metadata.create_all(engine)
    return engine

# create a session
def create_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def delete_all():
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
        
def find_chunk_by_idx_and_project(session: Session, chunk_idx: int, project_id: int):
    """
    Find a chunk by chunkIdx and projectId using a join.
    """
    stmt = (
        select(Chunk)
        .join(Item, Chunk.item_id == Item.id)  # Join Chunk -> Item
        .where(Chunk.chunkIdx == chunk_idx, Item.project_id == project_id)  # Conditions
    )

    # Execute the query
    result = session.execute(stmt).scalars().first()
    return result



# Test function to populate and query dummy data
def test_database():
    delete_all()
    engine = setup_database()
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

if __name__ == "__main__":
    test_database()
