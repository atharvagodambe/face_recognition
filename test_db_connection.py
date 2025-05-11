from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Replace with your actual connection string
# DATABASE_URL = "mysql+pymysql://root:root@localhost/face_recognition"
DATABASE_URL = "mysql+pymysql://sql12778143:lHsnT1RqH8@sql12.freesqldatabase.com:3306/sql12778143"

try:
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    print("✅ Successfully connected to the MySQL database!")
    connection.close()
except SQLAlchemyError as e:
    print("❌ Failed to connect to the database.")
    print(str(e))
