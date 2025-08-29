# Importing necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
import chromadb
from chromadb.utils import embedding_functions

# Define the embedding function using SentenceTransformers
# This function will be used to generate embeddings (vector representations) for the data
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Creating an instance of ChromaClient to establish a connection with the Chroma database
client = chromadb.Client()

# Defining a name for the collection where data will be stored or accessed
# This collection is likely used to group related records, such as employee data
collection_name = "books_collection"

def main():
    try:
        # Creating a collection using the ChromaClient instance
        # The 'create_collection' method creates a new collection with the specified configuration
        collection = client.create_collection(
            # Specifying the name of the collection to be created
            name=collection_name,
            # Adding metadata to describe the collection
            metadata={"description": "A collection for storing books data"},
            # Configuring the collection with cosine distance and embedding function
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")

        # Defining a list of employee dictionaries
        # Each dictionary represents an individual employee with comprehensive information
        # List of book dictionaries with comprehensive details for advanced search
        books = [
            {
                "id": "book_1",
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "genre": "Classic",
                "year": 1925,
                "rating": 4.1,
                "pages": 180,
                "description": "A tragic tale of wealth, love, and the American Dream in the Jazz Age",
                "themes": "wealth, corruption, American Dream, social class",
                "setting": "New York, 1920s"
            },
            {
                "id": "book_2",
                "title": "To Kill a Mockingbird",
                "author": "Harper Lee",
                "genre": "Classic",
                "year": 1960,
                "rating": 4.3,
                "pages": 376,
                "description": "A powerful story of racial injustice and moral growth in the American South",
                "themes": "racism, justice, moral courage, childhood innocence",
                "setting": "Alabama, 1930s"
            },
            {
                "id": "book_3",
                "title": "1984",
                "author": "George Orwell",
                "genre": "Dystopian",
                "year": 1949,
                "rating": 4.4,
                "pages": 328,
                "description": "A chilling vision of totalitarian control and surveillance society",
                "themes": "totalitarianism, surveillance, freedom, truth",
                "setting": "Oceania, dystopian future"
            },
            {
                "id": "book_4",
                "title": "Harry Potter and the Philosopher's Stone",
                "author": "J.K. Rowling",
                "genre": "Fantasy",
                "year": 1997,
                "rating": 4.5,
                "pages": 223,
                "description": "A young wizard discovers his magical heritage and begins his education at Hogwarts",
                "themes": "friendship, courage, good vs evil, coming of age",
                "setting": "England, magical world"
            },
            {
                "id": "book_5",
                "title": "The Lord of the Rings",
                "author": "J.R.R. Tolkien",
                "genre": "Fantasy",
                "year": 1954,
                "rating": 4.5,
                "pages": 1216,
                "description": "An epic fantasy quest to destroy a powerful ring and save Middle-earth",
                "themes": "heroism, friendship, good vs evil, power corruption",
                "setting": "Middle-earth, fantasy realm"
            },
            {
                "id": "book_6",
                "title": "The Hitchhiker's Guide to the Galaxy",
                "author": "Douglas Adams",
                "genre": "Science Fiction",
                "year": 1979,
                "rating": 4.2,
                "pages": 224,
                "description": "A humorous space adventure following Arthur Dent across the galaxy",
                "themes": "absurdity, technology, existence, humor",
                "setting": "Space, various planets"
            },
            {
                "id": "book_7",
                "title": "Dune",
                "author": "Frank Herbert",
                "genre": "Science Fiction",
                "year": 1965,
                "rating": 4.3,
                "pages": 688,
                "description": "A complex tale of politics, religion, and ecology on a desert planet",
                "themes": "power, ecology, religion, politics",
                "setting": "Arrakis, distant future"
            },
            {
                "id": "book_8",
                "title": "The Hunger Games",
                "author": "Suzanne Collins",
                "genre": "Dystopian",
                "year": 2008,
                "rating": 4.2,
                "pages": 374,
                "description": "A teenage girl fights for survival in a brutal televised competition",
                "themes": "survival, oppression, sacrifice, rebellion",
                "setting": "Panem, dystopian future"
            },
        ]

        # Create comprehensive text documents for each employee
        # These documents will be used for similarity search based on skills, roles, and experience
        book_documents = []
        for book in books:
            document = f"{book['title']} written by {book['author']} in genre {book['genre']}. "
            document += f"Year published: {book['year']}. Rated {book['rating']}. "
            document += f"Themes: {book['themes']}."
            document += f"Setting: {book['setting']}. Description: {book['description']}."
            book_documents.append(document)
        
        # Adding data to the collection in the Chroma database
        # The 'add' method inserts or updates data into the specified collection
        collection.add(
            # Extracting employee IDs to be used as unique identifiers for each record
            ids=[book["id"] for book in books],
            # Using the comprehensive text documents we created
            documents=book_documents,
            # Adding comprehensive metadata for filtering and search
            metadatas=[{
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "year": book["year"],
                "rating": book["rating"],
                "pages": book["pages"],
                "description": book["description"],
                "themes": book["themes"],
                "setting": book["setting"]
            } for book in books]
        )

        # Retrieving all items from the specified collection
        # The 'get' method fetches all records stored in the collection
        all_items = collection.get()
        # Logging the retrieved items to the console for inspection or debugging
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        # Function to perform various types of searches within the collection
        def perform_advanced_search(collection):
            try:
                print("=== Similarity Search Examples ===")

                # Example 1: Search for Magical Fantasy Adventure books
                print("\n1. Searching for Magical Fantasy Adventure books")
                query_text = "magical fantasy adventure"
                results = collection.query(
                    query_texts=[query_text],
                    n_results=3
                )
                print(f"Query: '{query_text}'")
                for i, (doc_id, document, distance) in enumerate(zip(
                    results['ids'][0], results['documents'][0], results['distances'][0]
                )):
                    metadata = results['metadatas'][0][i]
                    print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
                    print(f"     Author: {metadata['author']}, Genre: {metadata['genre']}")
                    print(f"     Document: {document[:100]}...")
                
                print("\n=== Metadata Filtering Examples ===")

                #Filter by Genre
                print("\n2. Finding Fantasy Genre Books:")
                results = collection.get(
                    where={"genre": "Fantasy"}
                )
                print(f"Found {len(results['ids'])} Fantasy Books:")
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    print(f"  - {metadata['title']}: {metadata['author']} (Published in {metadata['year']})")

                #Filter by rating
                print("\n3. Finding Books with rating 4.0 or higher:")
                results = collection.get(
                    where={"rating": {"$gte": 4.0}}
                )

                print(f"Found {len(results['ids'])} rating 4.0 and higher Books:")
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    print(f"  - {metadata['title']}: {metadata['author']} (Published in {metadata['year']})")

                
                print("\n=== Combined Search: Similarity + Metadata Filtering ===")

                #Find highly-rated dystopian books
                print("\n4. Finding highly rated dystopian books:")
                query_text = "Dystopian books"
                results = collection.query(
                    query_texts=[query_text],
                    n_results=5,
                    where={"rating": {"$gte": 4.0}}
                )
                print(f"Query: '{query_text}' with filters (rating 4.0 or higher)")
                print(f"Found {len(results['ids'][0])} matching books:")
                for i, (doc_id, document, distance) in enumerate(zip(
                    results['ids'][0], results['documents'][0], results['distances'][0]
                )):
                    metadata = results['metadatas'][0][i]
                    print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
                    print(f"     Author: {metadata['author']}, Genre: {metadata['genre']}")
                    print(f"     Document: {document[:100]}...")
                
                # Check if the results are empty or undefined
                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    # Log a message if no similar documents are found for the query term
                    print(f'No documents found similar to "{query_text}"')
                    return

                print("\n5. Finding books that have a power theme published before year 2000")
                query_text = "Books with theme about power"
                results = collection.query(
                    query_texts=[query_text],
                    n_results=3,
                    where={"year": {"$lte": 2000}}
                )
                print(f"Query: '{query_text}' with filters (rating 4.0 or higher)")
                print(f"Found {len(results['ids'][0])} matching books:")
                for i, (doc_id, document, distance) in enumerate(zip(
                    results['ids'][0], results['documents'][0], results['distances'][0]
                )):
                    metadata = results['metadatas'][0][i]
                    print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
                    print(f"     Author: {metadata['author']}, Genre: {metadata['genre']}")
                    print(f"     Document: {document[:100]}...")
                
                # Check if the results are empty or undefined
                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    # Log a message if no similar documents are found for the query term
                    print(f'No documents found similar to "{query_text}"')
                    return
                
            except Exception as error:
                print(f"Error in advanced search: {error}")
        
        # Call the perform_advanced_search function with the collection and all_items as arguments
        perform_advanced_search(collection)

    except Exception as error:
        # Catching and handling any errors that occur within the 'try' block
        # Logs the error message to the console for debugging purposes
        print(f"Error: {error}")

if __name__ == "__main__":
    main()