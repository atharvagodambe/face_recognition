from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Replace with your actual connection string
DATABASE_URL = "mysql+pymysql://root:root@localhost/face_recognition"

try:
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    print("✅ Successfully connected to the MySQL database!")
    connection.close()
except SQLAlchemyError as e:
    print("❌ Failed to connect to the database.")
    print(str(e))
