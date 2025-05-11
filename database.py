from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# DATABASE_URL = "mysql+pymysql://root:root@localhost/face_recognition"
DATABASE_URL = "mysql+pymysql://sql12778143:lHsnT1RqH8@sql12.freesqldatabase.com:3306/sql12778143"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
