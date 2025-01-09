from sqlalchemy import Column, Integer, String, Text, Table
from sqlalchemy import ForeignKey, LargeBinary, DateTime, MetaData, CheckConstraint
from sqlalchemy import create_engine, text, func, select, event, distinct
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session, aliased
from sqlalchemy.sql.expression import over, case

from contextlib import contextmanager

from graphviz import Digraph



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
        langs (str): List of the languages of the project. Defaults to 'de,en'.
        embedModel (str): The name of the embedder model. Cannot be null.
        embedSize (int: The size of the embedder model. Cannot be null. 
        vectorName (str): The name of the vector associated with the project. Cannot be null.
        vectorPath (str): The file path to the vector associated with the project. Cannot be null.
        indexName (str): The name of the index associated with the project. Can be null, then brute-force comparisons of vectors.
        indexPath (str): The file path to the index associated with the project. Can be null.
    """
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True, autoincrement=False, default = 1)
    name = Column(String(256), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    langs = Column(String(256), nullable=False, default="de,en")
    embedModel = Column(String(256), nullable=False, default="all-minilm-l12-v2")
    embedSize = Column(Integer, nullable=False,default = 384)
    vectorName = Column(String(256), nullable=False)
    vectorPath = Column(String(1024), nullable=False)
    indexName = Column(String(256), nullable=True)
    indexPath = Column(String(1024), nullable=True)
    
    __table_args__ = (
        CheckConstraint('id = 1', name='check_id_equals_1'),  # CHECK constraint
    )


# Junction table for items and tags
item_tags = Table(
    'item_tags',
    Base.metadata,
    Column('itemId', Integer, ForeignKey('items.id', ondelete='CASCADE'), primary_key=True),
    Column('tagId', Integer, ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True)
)

class Tag(Base):
    """
    Represents an tag in the database.

    Attributes:
        id (int): The primary key of the item.
        name (str): The unique name of the item.

    Relationships:
        project (Project): The project to which the item belongs.
    """
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    name = Column(String(256), unique=True, nullable=False)

    # Relationship with items via the junction table
    items = relationship('Item', secondary=item_tags, back_populates='tags')


class Item(Base):
    """
    Represents an item in the database.

    Attributes:
        id (int): The primary key of the item.
        name (str): The unique name of the item.
        summary_<lang> (str, optional): A brief summary of the item.
        text_<lang> (str, optional): The full text description of the item.
        title_<lang> (str): The title of the item.
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
    created = Column(DateTime, nullable=True, default=func.current_date())
    modified = Column(DateTime, nullable=True)
    url = Column(String(1024), nullable=True)
    dataurl = Column(String(1024), nullable=True)
    imgurl = Column(String(1024), nullable=True)
    license = Column(String(256), nullable=True)
    itemIdx = Column(Integer, nullable=False)

    # Relationship with tags via the junction table
    tags = relationship('Tag', secondary=item_tags, back_populates='items')


class Chunk(Base):
    """
    Represents a chunk of text associated with an item.

    Attributes:
        id (int): Primary key of the chunk.
        chunkIdx (int): Index of the chunk in project. Matches vector index.
        itemId (int): Foreign key referencing the associated item.
        item (Item): Relationship to the Item model, back_populated by 'chunks'.
    """
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True)
    chunkIdx = Column(Integer, nullable=False)
    itemId = Column(Integer, ForeignKey('items.id', ondelete="CASCADE"), nullable=False)

    item = relationship("Item", back_populates="chunks")

Item.chunks = relationship("Chunk", order_by=Chunk.id, back_populates="item", cascade="all, delete-orphan")

class Snippet(Base):
    """
    Represents a piece of text associated with some other other element, either from item or chunk tables

    Attributes:
        id (int): Primary key of the chunk.
        refIdx (int): Index of referenced element. Matches vector index.
        itemId (int): Foreign key referencing the associated item. Nullabe with on delete cascade
        chunkId (int): Foreign key referencing the associated chunk. Nullable with on delete cascade
        lang (str): The language of the text.
        type (str): Type of the text. Can be 'title', 'summary', 'fact' or 'text'.
        content (str): The text content of the chunk. Must be in project.langs
        item (Item): Relationship to the Item model, back_populated by 'chunks'.
        chunk (Chunk): Relationship to the Item model, back_populated by 'chunks'.
    """
    __tablename__ = 'snippets'
    id = Column(Integer, primary_key=True)
    refIdx = Column(Integer, nullable=False)
    itemId = Column(Integer, ForeignKey('items.id', ondelete="CASCADE"), nullable=True)
    chunkId = Column(Integer, ForeignKey('chunks.id', ondelete="CASCADE"), nullable=True)
    lang = Column(String(10), nullable=False)
    type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)

    item = relationship("Item", back_populates="snippets")
    chunk = relationship("Chunk", back_populates="snippets")

    __table_args__ = (
        CheckConstraint("type IN ('content', 'title', 'summary','fact')", name='check_type_in_list'),
    )
    
Item.snippets = relationship("Snippet", order_by=Snippet.id, back_populates="item", cascade="all, delete-orphan")
Chunk.snippets = relationship("Snippet", order_by=Snippet.id, back_populates="chunk", cascade="all, delete-orphan")

@event.listens_for(Snippet, "before_insert")
def validate_language(mapper, connection, target):
    """
    Validate that the language of the text is in the project's allowed languages using FIND_IN_SET.
    """
    # Use FIND_IN_SET to check if the language is allowed
    query = text(f"""
        SELECT FIND_IN_SET('{target.lang}', langs) > 0
        FROM projects
        WHERE id = 1
    """
    )
    result = connection.execute(query).scalar()

    if not result:
        msg = f"Snippet: language {target.lang} not allowed"
        raise ValueError(msg)


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


    def updateTags(self, updated_item, tags):
        """
        Update an item in the database with tags names.

        :param model: SQLAlchemy model class (e.g., Item, Chunk)
        :param updated_item: SQLAlchemy Item model instance. Must have a valid ID.
        :param tags: List of tag names to associate with the item.
        """
        if not isinstance(updated_item, Item):
            raise ValueError("updated_item must be an instance of Item.")
        with self.get_session() as session:
            # Query the existing object
            existing_obj = session.query(Item).filter(Item.id == updated_item.id).first()
            if not existing_obj:
                raise ValueError(f"{Item.__name__} with ID {updated_item.id} not found.")

            # find tags from names
            tags = session.query(Tag).filter(Tag.name.in_(tags)).all()
            if not tags:
                raise ValueError(f"No tags found for names: {tags}")
            setattr(existing_obj, "tags", tags)


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

            
    def find_chunk(self, chunkIdx: int):
        """
        Find a Chunk by its index.

        Args:
            session (Session): The SQLAlchemy session to use for the query.
            chunkIdx (int): The index of the chunk to find.

        Returns:
            Chunk: The found Chunk object, or None if no matching chunk is found.
        """
        stmt = (
            select(Chunk)
            .join(Item, Chunk.itemId == Item.id)  # Join Chunk -> Item
            .where(Chunk.chunkIdx == chunkIdx)  # Conditions
        )

        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().first()
            return result


    def get_chunks(self):
        """
        Get a list of all chunks for a given projectId, ordered by itemIdx and then by chunkIdx.

        :param session: SQLAlchemy Session object
        :return: List of Chunk objects
        """
        stmt = (
            select(Chunk)
            .join(Item, Chunk.itemId == Item.id)  # Join Chunk with Item
            .order_by(Chunk.chunkIdx.asc())  # Order by chunkIdx
        )
        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().all()
            return result

    def get_items(self):
        """
        Get a list of all items, ordered by itemIdx (ascending).
        :return: List of Item objects
        """
        stmt = (
            select(Item)
            .order_by(Item.itemIdx.asc())  # Order by itemIdx in ascending order
        )

        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().all()
            return result

    def find_item(self, chunkIdx: int):
        """
        Find an Item by a chunk index.

        Args:
            session (Session): The SQLAlchemy session to use for the query.
            chunkIdx (int): The index of the chunk to find.

        Returns:
            Item: The found Item object, or None if no matching Item is found.
        """
        stmt = (
            select(Item)
            .join(Chunk, Chunk.itemId == Item.id)  # Join Chunk -> Item
            .where(Chunk.chunkIdx == chunkIdx)  # Conditions
        )

        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().first()
            return result


    def find_items(self, chunkList: list [int]):
        """
        Get a list of all items, referred to by chunk index list. deduplicate.
        :return: List of Item objects
        """
        # Define a CASE expression for custom ordering
        order_case = case(
            {key: i for i, key in enumerate(chunkList)},
            value=func.ifnull(Chunk.chunkIdx, -1)  # Default case if index_key not in list
        )        
        # Execute the query
        with self.get_session() as session:
            # Query to join Chunk and Item and apply ranking
            stmt = session.query(
                Chunk.chunkIdx.label('chunk_idx'),
                Item.id.label('item_id'),
                Item.name.label('item_name'),
                over(
                    func.row_number(),
                    partition_by=Item.id,  # Deduplicate by item
                    order_by=order_case   # Maintain input order
                ).label('rank')
            ).join(
                Item, Chunk.itemId == Item.id
            ).filter(
                Chunk.chunkIdx.in_(chunkList)  # Filter only by relevant chunks
            ).subquery()

            # Final query to select rows with rank = 1
            final_query = session.query(
                stmt.c.item_id,
                stmt.c.item_name,
                stmt.c.chunk_idx
                # Item
            ).filter(
                stmt.c.rank == 1
            )
            results = final_query.all()
            return results


    def get_item_by_name(self, name: str = None):
        """
        Get an item by name or code, where only one of the parameters is provided.
        :param name: Name of the item (optional)
        :return: Item object or None if not found
        :raises ValueError: If neither or both parameters are provided
        """
        if not name:
            raise ValueError("Name must be provided.")

        stmt = select(Item)
        stmt = stmt.where(Item.name == name)

        # Execute the query
        with self.get_session() as session:
            result = session.execute(stmt).scalars().first()
            return result

    def get_items_by_tags(self, tags = []):
        """
        Get all items matching on tags
        :param tags: List of tags
        :return: Item object or None if not found
        :raises ValueError: If neither or both parameters are provided
        """
        if len(tags) == 0:
            raise ValueError("Tag(s) must be provided.")

        tag_alias = aliased(Tag)
        stmt = (
            select(distinct(Item.id))
            .join(item_tags, Item.id == item_tags.c.itemId)  
            .join(tag_alias, tag_alias.id == item_tags.c.tagId) 
            .where(tag_alias.name.in_(tags))
        )
        with self.get_session() as session:
            result = session.execute(stmt).scalars().all()
            return result
            


    def get_item_tags(self, itemId):
        stmt = select(Item)
        stmt = stmt.where(Item.id == itemId)

        # Execute the query
        with self.get_session() as session:
            item = session.execute(stmt).scalars().first()
            if not item:
                return []
            return [tag.name for tag in item.tags]




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

    # Define function to generate Graphviz diagram
    # Generate UML diagram like so
    #   import ragSqlUtils as sq
    #   uml_diagram = sq.DatabaseUtility.create_uml()
    #   uml_diagram.render('uml_diagram', view=True)  # Save and open diagram
    @staticmethod
    def create_uml():
        dot = Digraph(comment='UML Diagram', format='png')
        
        classes = [Project, Item, Tag, Chunk, Snippet]
        # Create nodes for classes
        for cls in classes:
            table_name = cls.__tablename__
            columns = [f"{col.name}: {col.type}" for col in cls.__table__.columns]
            relationships = [rel.key for rel in cls.__mapper__.relationships]
            
            # Add table node
            label = f"{{ {table_name} | {'\\l'.join(columns)} | Relationships: {'\\l'.join(relationships)} }}"
            dot.node(table_name, label=label, shape='record')
        
        # Create relationships (edges)
        for cls in classes:
            table_name = cls.__tablename__
            for rel in cls.__mapper__.relationships:
                dot.edge(table_name, rel.target.name, label=f"1-to-{rel.direction.name.lower()}")
        
        return dot


if __name__ == "__main__":
    import private_remote as pr
    connection_string = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
    # DatabaseUtility.delete_all(connection_string)
    db = DatabaseUtility(connection_string)    

    layout = db.get_table_layout("items")
    print(layout)

    # Create dummy projects
    project = Project(name="Project Alpha", langs="de,en", description="Description of Project Alpha",vectorName="prj1.vec",vectorPath = "./data")
    project = db.insert(project)
    print(f"Project ID: {project.id}")
    try: 
        project = Project(name="Project Beta", description="Description of Project Beta",vectorName="prj2.vec",vectorPath = "./data")
        project2 = db.insert(project)
        print(f"Project2 ID: {project2.id}")
    except:
        print("Project already exists")

    # create some tags 
    tag = Tag(name="tag1")
    tag1 = db.insert(tag)
    tag = Tag(name="tag2")
    tag2 = db.insert(tag)
    tag = Tag(name="tag3")
    tag3 = db.insert(tag)

        
    # Create dummy items
    item = Item(name="Item One", itemIdx=0)
    item1 = db.insert(item)
    item = Item(name="Item Two",  itemIdx=1,tags = [tag1,tag2])
    item2 = db.insert(item)
    item = Item(name="Item Three",  itemIdx=2,tags = [tag3,tag2])
    item3 = db.insert(item)
    print(f"Item tags: {[t.name for t in item3.tags]}")
    items = db.search(Item)
    print(f"Items: {[(i.name) for i in items]}")

    # update item1 name
    item1.name = "Item One Updated"
    db.update(Item,item1)

    items = db.search(Item)
    print(f"Items: {[(i.name) for i in items]}")


    textIds = []

    txt = Snippet(content="Summary of item one", lang="de", itemId = item1.id, refIdx = item1.itemIdx, type="summary")
    txt1 = db.insert(txt)
    textIds.append(txt1.id)
    txt = Snippet(content="Title of item one", lang="de", itemId = item1.id, refIdx = item1.itemIdx, type="title")
    txt2 = db.insert(txt)
    textIds.append(txt2.id)
    txt = Snippet(content="Content of item one", lang="en", itemId = item1.id, refIdx = item1.itemIdx, type="content")
    txt3 = db.insert(txt)
    textIds.append(txt3.id)

    txt = Snippet(content="Summary of item two", lang="de", itemId = item2.id, refIdx = item2.itemIdx, type="summary")
    txt = db.insert(txt)
    textIds.append(txt.id)

    texts = db.search(Snippet)
    print(f"Texts: {[(i.content,i.itemId,i.chunkId,i.refIdx) for i in texts]}")

    # associate tags with items
    db.updateTags(item1,["tag1","tag3"])
    print(f"Updated item tags: {db.get_item_tags(item1.id)}")

    # find items by tags
    items = db.get_items_by_tags(["tag2"])
    print(f"Items by tags: {[i for i in items]}")


    # Create dummy chunks
    chunkIds = []
    for idx in range (0,4):
        chunk = Chunk(chunkIdx=idx, itemId=item1.id)
        chunk = db.insert(chunk)
        txt = Snippet(content=f"Content of chunk {idx + 1}", lang="de", chunkId = chunk.id, refIdx = chunk.chunkIdx, type="content")
        txt = db.insert(txt)
        textIds.append(txt.id)
        chunkIds.append(chunk.id)

    texts = db.search(Snippet)
    print(f"Texts: {[(i.content,i.itemId,i.chunkId,i.refIdx) for i in texts]}")


    # Query and print data
    projects = db.search(Project) #.all()
    for project in projects:
        print(f"Project: {project.name}, Description: {project.description}")

    items = db.search(Item)
    for item in items:
        print(f"Item: {item.name}")

    chunks = db.search(Chunk)
    for chunk in chunks:
        print(f"Chunk Ref: {chunk.itemId}, Index: {chunk.chunkIdx}")

    c = db.find_chunk(1)
    print("No chunk" if c == None else c.id)
    i = db.find_item(1)
    print("No item" if i == None else i.id)
    c = db.find_chunk(2)
    print("No chunk" if c == None else c.id)
    i = db.find_item(2)
    print("No item" if i == None else i.id)

    results  = db.get_chunks()
    for chunk in results:
        print(f"Chunk, Item: {chunk.itemId}, Index: {chunk.chunkIdx}")
        #print(f"Chunk, Item: {item.id}, Index: {chunk.chunkIdx}")

    # search titles from item list
    titles = db.search(Snippet,filters=[Snippet.itemId.in_([1,2]),Snippet.type == "title",Snippet.lang=="de"])
    print(f"Titles: {[t.content for t in titles]}")
    