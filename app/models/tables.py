from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.models.db import Base


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String, unique=True, index=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    objects: Mapped[list["DetectedObject"]] = relationship(
        "DetectedObject", back_populates="image", cascade="all, delete-orphan"
    )


class DetectedObject(Base):
    __tablename__ = "objects"
    __table_args__ = (
        UniqueConstraint("image_id", "ref_id", name="uq_image_ref"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), index=True)
    ref_id: Mapped[int] = mapped_column(Integer, index=True) 

    # Bounding box
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    w: Mapped[int] = mapped_column(Integer)
    h: Mapped[int] = mapped_column(Integer)

    # Circle info
    cx: Mapped[float] = mapped_column(Float)
    cy: Mapped[float] = mapped_column(Float)
    r: Mapped[float] = mapped_column(Float)

    image: Mapped[Image] = relationship("Image", back_populates="objects")


