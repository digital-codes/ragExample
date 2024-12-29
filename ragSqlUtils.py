from sqlalchemy import Column, Integer, String, Text
from sqlalchemy import ForeignKey, LargeBinary, DateTime, MetaData
from sqlalchemy import create_engine, text, func, select
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from contextlib import contextmanager

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
        name (str): The name of the project. Cannot be null. Unique
        description (str): A textual description of the project. Can be null.
        lang (str): The language of the project. Defaults to 'en'.
        vectorName (str): The name of the vector associated with the project. Cannot be null.
        vectorPath (str): The file path to the vector associated with the project. Cannot be null.
        indexName (str): The name of the index associated with the project. Can be null, then brute-force comparisons of vectors.
        indexPath (str): The file path to the index associated with the project. Can be null.
    """
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    name = Column(String(256), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    lang = Column(String(2), nullable=False, default="en")
    vectorName = Column(String(256), nullable=False)
    vectorPath = Column(String(1024), nullable=False)
    indexName = Column(String(256), nullable=True)
    indexPath = Column(String(1024), nullable=True)

class Item(Base):
    """
    Represents an item in the database.

    Attributes:
        id (int): The primary key of the item.
        name (str): The unique name of the item.
        code (int): The code associated with the item.
        projectId (int): The foreign key referencing the associated project.
        summary (str, optional): A brief summary of the item.
        fulltext (str, optional): The full text description of the item.
        tags (str, optional): Tags associated with the item.
        title (str): The title of the item.
        created (datetime, optional): The creation date of the item. Defaults to the current date.
        modified (datetime, optional): The last modified date of the item.
        url (str, optional): The URL associated with the item.
        license (str, optional): The license information of the item.
        itemIdx (int): The index of the item.

    Relationships:
        project (Project): The project to which the item belongs.
    """
    __tablename__ = 'items'

    id = Column(Integer, primary_key=True)
    name = Column(String(256), unique=True, nullable=False)
    code = Column(Integer, nullable=False)
    projectId = Column(Integer, ForeignKey('projects.id'), nullable=False)
    summary = Column(Text, nullable=True)
    fulltext = Column(Text, nullable=True)
    tags = Column(String(1024), nullable=True)
    title = Column(String(256), nullable=False)
    created = Column(DateTime, nullable=True, default=func.current_date())
    modified = Column(DateTime, nullable=True)
    url = Column(String(1024), nullable=True)
    dataurl = Column(String(1024), nullable=True)
    imgurl = Column(String(1024), nullable=True)
    license = Column(String(256), nullable=True)
    itemIdx = Column(Integer, nullable=False)

    project = relationship("Project", back_populates="items")

Project.items = relationship("Item", order_by=Item.id, back_populates="project")

class Chunk(Base):
    """
    Represents a chunk of text associated with an item.

    Attributes:
        id (int): Primary key of the chunk.
        chunkNum (int): Number of the chunk within item
        chunkIdx (int): Index of the chunk in project. Matches vector index.
        itemId (int): Foreign key referencing the associated item.
        text (str): The text content of the chunk.
        item (Item): Relationship to the Item model, back_populated by 'chunks'.
    """
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True)
    chunkIdx = Column(Integer, nullable=False)
    chunkNum = Column(Integer, nullable=False)
    itemId = Column(Integer, ForeignKey('items.id', ondelete="CASCADE"), nullable=False)
    text = Column(Text, nullable=True)
    preview = Column(Text, nullable=True)

    item = relationship("Item", back_populates="chunks")

Item.chunks = relationship("Chunk", order_by=Chunk.id, back_populates="item", cascade="all, delete-orphan")

class Vector(Base):
    """
    Represents a vector in the database.

    Attributes:
        id (int): The primary key of the vector.
        chunkId (int): The foreign key referencing the associated chunk.
        value (bytes): The binary data representing the vector.
        chunk (Chunk): The relationship to the Chunk model, back_populated by "vectors".
    """
    __tablename__ = 'vectors'

    id = Column(Integer, primary_key=True)
    chunkId = Column(Integer, ForeignKey('chunks.id', ondelete="CASCADE"), nullable=False)
    value = Column(LargeBinary, nullable=False)  # Blob

    chunk = relationship("Chunk", back_populates="vectors")

Chunk.vectors = relationship("Vector", order_by=Vector.id, back_populates="chunk", cascade="all, delete-orphan")

class TitleVector(Base):
    """
    TitleVector is a SQLAlchemy model representing the 'title_vectors' table.

    Attributes:
        id (int): Primary key of the table.
        itemId (int): Foreign key referencing the 'items' table.
        value (LargeBinary): Blob data representing the vector value.
        item (relationship): Relationship to the Item model, back_populated by 'title_vectors'.
    """
    __tablename__ = 'title_vectors'

    id = Column(Integer, primary_key=True)
    itemId = Column(Integer, ForeignKey('items.id', ondelete="CASCADE"), nullable=False)
    value = Column(LargeBinary, nullable=False)  # Blob

    item = relationship("Item", back_populates="title_vectors")

Item.title_vectors = relationship("TitleVector", order_by=TitleVector.id, back_populates="item")


# Database Utility Class
class DatabaseUtility:
    def __init__(self, connection_string):
        """
        Initialize the DatabaseUtility with the database connection string
        and create all tables.
        """
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False) 
        # expire_on_commit=False to prevent session from expiring after !!!
        self._initialize_tables()

    def _initialize_tables(self):
        """
        Create all tables in the database if they do not exist.
        """
        Base.metadata.create_all(self.engine)

    @contextmanager
    def get_session(self):
        """
        Context manager to provide a database session.
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    # create a session
    def create_session(self,engine):
        """
        Create a new SQLAlchemy session.

        Args:
            engine (Engine): The SQLAlchemy engine to bind the session to.

        Returns:
            Session: A new SQLAlchemy session.
        """
        Session = sessionmaker(bind=engine)
        return Session()

    def insert(self, obj):
        """
        Insert a single object into the database.
        """
        with self.get_session() as session:
            session.add(obj)
            session.flush()
            #session.refresh(obj)  # Forcefully load all attributes from the database
            return obj

    def search(self, model, filters=None, order_by=None):
        """
        Search for records in the database.
        """
        with self.get_session() as session:
            query = session.query(model)
            if filters:
                query = query.filter(*filters)
            if order_by:
                query = query.order_by(order_by)
            return query.all()

    def update(self, model, updated_obj):
        """
        Update an object in the database using the provided object.

        :param model: SQLAlchemy model class (e.g., Item, Chunk)
        :param updated_obj: SQLAlchemy model instance with updated values. Must have a valid ID.
        """
        with self.get_session() as session:
            # Query the existing object
            existing_obj = session.query(model).filter(model.id == updated_obj.id).first()
            if not existing_obj:
                raise ValueError(f"{model.__name__} with ID {updated_obj.id} not found.")

            # Update fields
            for key, value in updated_obj.__dict__.items():
                if not key.startswith("_"):  # Skip SQLAlchemy internals
                    setattr(existing_obj, key, value)

    def delete_id(self, model, obj_id):
        """
        Delete an object from the database, filtering by its ID.

        :param model: SQLAlchemy model class (e.g., Item, Chunk)
        :param obj_id: The ID of the object to delete
        """
        with self.get_session() as session:
            # Query the object to delete
            obj = session.query(model).filter(model.id == obj_id).first()
            if obj:
                session.delete(obj)
            else:
                raise ValueError(f"{model.__name__} with ID {obj_id} not found.")

            
    def find_chunk(self, chunkIdx: int, projectId: int):
        """
        Find a Chunk by its index and project ID.

        Args:
            session (Session): The SQLAlchemy session to use for the query.
            chunkIdx (int): The index of the chunk to find.
            projectId (int): The ID of the project to which the chunk belongs.

        Returns:
            Chunk: The found Chunk object, or None if no matching chunk is found.
        """
        stmt = (
            select(Chunk)
            .join(Item, Chunk.itemId == Item.id)  # Join Chunk -> Item
            .where(Chunk.chunkIdx == chunkIdx, Item.projectId == projectId)  # Conditions
        )

        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().first()
            return result


    def get_chunks(self, projectId: int):
        """
        Get a list of all chunks for a given projectId, ordered by itemIdx and then by chunkIdx.

        :param session: SQLAlchemy Session object
        :param projectId: ID of the project
        :return: List of Chunk objects
        """
        # stmt = (
        #     select(Chunk)
        #     .join(Item, Chunk.itemId == Item.id)  # Join Chunk -> Item
        #     .where(Item.projectId == projectId)  # Filter by projectId
        #     .order_by(Item.itemIdx.asc(), Chunk.chunkIdx.asc())  # Order by itemIdx and chunkIdx
        # )
        
        # stmt = (
        #     select(Chunk)
        #     .where(Item.projectId == projectId)  # Filter by projectId
        #     .order_by(Chunk.chunkIdx.asc())  # Order by itemIdx and chunkIdx
        # )
        stmt = (
            select(Chunk)
            .join(Item, Chunk.itemId == Item.id)  # Join Chunk with Item
            .where(Item.projectId == projectId)  # Filter by projectId
            .order_by(Chunk.chunkIdx.asc())  # Order by chunkIdx
        )
        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().all()
            return result


    def find_item(self, chunkIdx: int, projectId: int):
        """
        Find an Item by a chunk index and project ID.

        Args:
            session (Session): The SQLAlchemy session to use for the query.
            chunkIdx (int): The index of the chunk to find.
            projectId (int): The ID of the project to which the chunk belongs.

        Returns:
            Item: The found Item object, or None if no matching Item is found.
        """
        stmt = (
            select(Item)
            .join(Chunk, Chunk.itemId == Item.id)  # Join Chunk -> Item
            .where(Chunk.chunkIdx == chunkIdx, Item.projectId == projectId)  # Conditions
        )

        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().first()
            return result


    def get_items(self, projectId: int):
        """
        Get a list of all items for a given projectId, ordered by itemIdx (ascending).

        :param session: SQLAlchemy Session object
        :param projectId: ID of the project
        :return: List of Item objects
        """
        stmt = (
            select(Item)
            .where(Item.projectId == projectId)
            .order_by(Item.itemIdx.asc())  # Order by itemIdx in ascending order
        )

        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().all()
            return result


    def get_item(self, name: str = None, code: int = None):
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
        with self.get_session() as session:
            result = session.execute(stmt).scalars().first()
            return result


    def get_table_layout(self,table_name):
        """
        Retrieve the layout of a specific table in the database.

        :param engine: SQLAlchemy Engine
        :param table_name: Name of the table
        :return: Dictionary with column details
        """
        meta = MetaData()
        meta.reflect(bind=self.engine)
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
    
    @staticmethod
    def delete_all(connection_string):
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
        engine = create_engine(connection_string)
        # Reflect the database schema
        meta = MetaData()
        meta.reflect(bind=engine)

        # Connect to the database
        #with engine.connect() as conn:
        with engine.begin() as conn:
            try:
                # Disable foreign key checks
                conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))

                # Delete all data from all tables
                # and drop all tables
                for table in reversed(meta.sorted_tables):  # Reverse order to respect FK constraints
                    print(f"Deleting data from table: {table.name}")
                    conn.execute(text(f"DELETE FROM {table.name}"))
                    conn.execute(text(f"DROP TABLE {table.name};"))

                # Re-enable foreign key checks
                conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            except Exception as e:
                        print(f"An error occurred: {e}")
            finally:
                conn.close()

        engine.dispose()


if __name__ == "__main__":
    connection_string = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
    #db_util = DatabaseUtility(connection_string)
    # DatabaseUtility.delete_all(connection_string)
    db = DatabaseUtility(connection_string)    

    layout = db.get_table_layout("items")
    print(layout)

    # Create dummy projects
    project = Project(name="Project Alpha", description="Description of Project Alpha",vectorName="prj1.vec",vectorPath = "./data")
    project1 = db.insert(project)
    print(f"Project1 ID: {project1.id}")
    project = Project(name="Project Beta", description="Description of Project Beta",vectorName="prj2.vec",vectorPath = "./data")
    project2 = db.insert(project)
    print(f"Project2 ID: {project2.id}")

    # Create dummy items
    item = Item(name="Item One", code=101, projectId=project1.id, summary="Summary of item one", fulltext="Fulltext for item one", tags="tag1,tag2", title="Title One", itemIdx=1)
    item1 = db.insert(item)
    item = Item(name="Item Two", code=102, projectId=project2.id, summary="Summary of item two", fulltext="Fulltext for item two", tags="tag3,tag4", title="Title Two", itemIdx=1)
    item2 = db.insert(item)
    item = Item(name="Item Three", code=102, projectId=project1.id, summary="Summary of item three", fulltext="Fulltext for item three", tags="tag1,tag2", title="Title Three", itemIdx=2)
    item3 = db.insert(item)

    # Create dummy chunks
    chunkIds = []
    idx = 0
    chunk = Chunk(chunkIdx=idx, chunkNum = 1, itemId=item1.id, text="Chunk 1 text")
    chunk = db.insert(chunk)
    idx += 1
    chunkIds.append(chunk.id)
    chunk = Chunk(chunkIdx=idx, chunkNum = 1, itemId=item2.id, text="Chunk 2 text")
    chunk = db.insert(chunk)
    idx += 1
    chunkIds.append(chunk.id)
    chunk = Chunk(chunkIdx=idx, chunkNum = 2, itemId=item1.id, text="Chunk 3 text")
    chunk = db.insert(chunk)
    idx += 1
    chunkIds.append(chunk.id)
    chunk = Chunk(chunkIdx=idx, chunkNum = 1, itemId=item3.id, text="Chunk 4 text")
    chunk = db.insert(chunk)
    idx += 1
    chunkIds.append(chunk.id)

    # Create dummy vectors
    vector = np.random.rand(384).astype('float32')
    binary_data = vector.tobytes()
    # Insert into the database
    vector1 = Vector(chunkId=chunkIds[0], value=binary_data)
    vector = np.random.rand(384).astype('float32')
    binary_data = vector.tobytes()
    # Insert into the database
    vector2 = Vector(chunkId=chunkIds[1], value=binary_data)
    db.insert(vector1)
    db.insert(vector2)
    
    # Create dummy title_vectors
    vector = np.random.rand(384).astype('float32')
    binary_data = vector.tobytes()
    # Insert into the database
    vector1 = TitleVector(itemId=item1.id, value=binary_data)
    vector = np.random.rand(384).astype('float32')
    binary_data = vector.tobytes()
    # Insert into the database
    vector2 = TitleVector(itemId=item1.id, value=binary_data)
    db.insert(vector1)
    db.insert(vector2)
    
    

    # Query and print data
    projects = db.search(Project) #.all()
    for project in projects:
        print(f"Project: {project.name}, Description: {project.description}")

    items = db.search(Item)
    for item in items:
        print(f"Item: {item.name}, Code: {item.code}, Tags: {item.tags}")

    chunks = db.search(Chunk)
    for chunk in chunks:
        print(f"Chunk: {chunk.text}, Index: {chunk.chunkIdx}")

    vectors = db.search(Vector)
    for vector in vectors:
        value = np.frombuffer(vector.value, dtype='float32')
        print(f"Vector: {value[:10]}...")  # Print the first 10 characters of the vector

    vectors = db.search(TitleVector)
    for vector in vectors:
        value = np.frombuffer(vector.value, dtype='float32')
        print(f"TitleVector: {value[:10]}...")  # Print the first 10 characters of the vector

    c = db.find_chunk(1, 1)
    print("No chunk" if c == None else c.text)
    i = db.find_item(1, 1)
    print("No item" if i == None else i.title)
    c = db.find_chunk(1, 2)
    print("No chunk" if c == None else c.text)
    i = db.find_item(1, 2)
    print("No item" if i == None else i.title)

    results  = db.get_chunks(1)
    for chunk in results:
        print(f"Chunk, Item: {chunk.itemId}, Index: {chunk.chunkIdx}")
        #print(f"Chunk, Item: {item.id}, Index: {chunk.chunkIdx}")
