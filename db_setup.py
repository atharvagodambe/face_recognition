from database import Base, engine
from models import KnownFace

Base.metadata.create_all(bind=engine)
