from sqlalchemy import Column, Integer, Float, DateTime
from sqlalchemy.sql import func
from .database import Base

class SensorData(Base):
    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    heart_rate = Column(Float)
    temperature = Column(Float)
    gsr = Column(Float)
    lat = Column(Float)       
    lon = Column(Float)       
    #steps = Column(Integer)       
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class AppleHealthData(Base):
    __tablename__ = "apple_health"

    id = Column(Integer, primary_key=True, index=True)
    step_count = Column(Integer)
    active_energy = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
