from typing import Any

from pydantic import BaseModel, field_validator, model_validator
from qdrant_client import models as qmodels


class MetadataFilter(BaseModel):
    filename: str | None = None
    filenames: list[str] | None = None
    page: int | None = None
    section: str | None = None
    document_id: str | None = None

    @field_validator("page")
    @classmethod
    def validate_page(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("page must be greater than or equal to 1.")

        return value

    @model_validator(mode="after")
    def normalize(self) -> "MetadataFilter":
        # Chuẩn hóa filename đơn.
        if self.filename is not None:
            self.filename = self.filename.strip() or None

        # Chuẩn hóa danh sách filename.
        normalized_names = [
            name.strip()
            for name in (self.filenames or [])
            if isinstance(name, str) and name.strip()
        ]

        # Loại bỏ tên trùng nhưng vẫn giữ nguyên thứ tự.
        normalized_names = list(dict.fromkeys(normalized_names))

        # Nếu cả filename và filenames cùng được cung cấp,
        # đưa tất cả về filenames.
        if self.filename is not None:
            normalized_names.insert(0, self.filename)
            normalized_names = list(dict.fromkeys(normalized_names))

        if not normalized_names:
            self.filename = None
            self.filenames = None
        elif len(normalized_names) == 1:
            self.filename = normalized_names[0]
            self.filenames = None
        else:
            self.filename = None
            self.filenames = normalized_names

        if self.section is not None:
            self.section = self.section.strip() or None

        if self.document_id is not None:
            self.document_id = self.document_id.strip() or None

        return self


def coerce_filter(
    filters: MetadataFilter | dict[str, Any] | None,
) -> MetadataFilter | None:
    if filters is None:
        return None

    if isinstance(filters, MetadataFilter):
        return filters

    if isinstance(filters, dict):
        return MetadataFilter.model_validate(filters)

    raise TypeError(
        "filters must be MetadataFilter, dict, or None."
    )


def filters_to_dict(
    filters: MetadataFilter | dict[str, Any] | None,
) -> dict[str, Any] | None:
    normalized_filter = coerce_filter(filters)

    if normalized_filter is None:
        return None

    result = normalized_filter.model_dump(exclude_none=True)

    return result or None


def filters_to_qdrant(
    filters: MetadataFilter | dict[str, Any] | None,
) -> qmodels.Filter | None:
    flat_filter = filters_to_dict(filters)

    if not flat_filter:
        return None

    conditions: list[qmodels.FieldCondition] = []

    for field_name, value in flat_filter.items():
        if field_name == "filenames":
            conditions.append(
                qmodels.FieldCondition(
                    key="metadata.filename",
                    match=qmodels.MatchAny(any=value),
                )
            )
            continue

        conditions.append(
            qmodels.FieldCondition(
                key=f"metadata.{field_name}",
                match=qmodels.MatchValue(value=value),
            )
        )

    return qmodels.Filter(must=conditions)