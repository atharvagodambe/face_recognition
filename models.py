from sqlalchemy import Column, Integer, String, LargeBinary
from database import Base

class KnownFace(Base):
    __tablename__ = "known_faces"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    encoding = Column(LargeBinary, nullable=False)
