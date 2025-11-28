"""
This module contains all dataset-related routes for the LightRAG-lly API.
"""

import asyncio
from functools import lru_cache
from uuid import uuid4

from lightrag.utils import logger, get_pinyin_sort_key
import aiofiles
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from io import BytesIO
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field, field_validator

from lightrag import LightRAG
from lightrag.base import DeletionResult, DocProcessingStatus, DocStatus
from lightrag.utils import generate_track_id
from lightrag.api.utils_api import get_combined_auth_dependency
from ..config import global_args
from lightrag.kg.dataset_metadata_impl import JsonDatasetMetadataStorage


router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)

"""Response model for  dataset metadata

Attributes:
    id (str): Unique identifier of the dataset
    name (str): Name of the dataset
    description (Optional[str]): Description of the dataset
    created_at (str): ISO format timestamp when the dataset was created
    updated_at (str): ISO format timestamp when the dataset was last updated
    docs_count (int): Number of documents in the dataset
    docs: List[str]: List of document IDs in the dataset
"""
# common class
class InsertResponse(BaseModel):
    """Response model for document insertion operations

    Attributes:
        status: Status of the operation (success, duplicated, partial_success, failure)
        message: Detailed message describing the operation result
        track_id: Tracking ID for monitoring processing status
    """
    id : Optional[str] = Field(description="Dataset ID")
    status: Literal["success", "duplicated", "partial_success", "failure"] = Field(
        description="Status of the operation"
    )
    message: str = Field(description="Message describing the operation result")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "File 'document.pdf' uploaded successfully. Processing will continue in background.",
                "track_id": "upload_20250729_170612_abc123",
            }
        }



# class


class LazyStorageProxy:
    def __init__(self, real_storage):
        self._storage = real_storage
        self._initialized = False

    async def _init(self):
        if not self._initialized:
            await self._storage.initialize()
            self._initialized = True

    def __getattr__(self, item):
        attr = getattr(self._storage, item)

        if callable(attr):
            async def wrapper(*args, **kwargs):
                await self._init()
                return await attr(*args, **kwargs)
            return wrapper

        return attr


class DatasetManager:
    """
    Dataset 管理器
    DatasetManager 的职责是管理 dataset 元数据与 doc 列表，
    """

    # ========== 加入 workspace 支持 ==========
    def __init__(
            self,
            workspace: str = "",
            storage: JsonDatasetMetadataStorage | None = None,  # 依赖注入
    ):
        """
        Args:
            workspace: 工作空间，用于数据隔离
            storage: dataset 元数据存储（JsonDatasetMetadataStorage）
        """

        self.workspace = workspace
        self.storage = LazyStorageProxy(storage)

        if self.storage is None:
            raise ValueError("DatasetManager 需要提供 JsonDatasetMetadataStorage 实例")



    # 简单缓存：dataset_id → metadata
        self._dataset_cache: Dict[str, dict[str, Any]] = {}


    # ========== 创建或更新数据集 ==========
    async def create_or_update_dataset(self, dataset_id: str, metadata: dict[str, Any]):
        """创建或更新 dataset 元数据"""
        await self.storage.create_or_update_dataset(dataset_id, metadata)
        self._dataset_cache[dataset_id] = metadata

    # ========== 根据dataset_id查询单个 dataset ==========
    async def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """获取 dataset 元数据"""
        if dataset_id in self._dataset_cache:
            return self._dataset_cache[dataset_id]

        data = await self.storage.get_dataset(dataset_id)
        if data:
            self._dataset_cache[dataset_id] = data
        return data


    # ========== 根据name精确查询单个 dataset ==========
    async def get_dataset_by_name(self, name: str) -> dict[str, Any] | None:
        """按名称获取 dataset 元数据"""
        data = await self.storage.get_dataset_by_exact_name(name)
        return data


    # ========== 根据name模糊查询单个 dataset ==========
    async def search_dataset_by_name(self, name: str) -> dict[str, Any] | None:
        """按名称获取 dataset 元数据"""
        data = await self.storage.search_datasets_by_name(name)
        return data

    # ========== 分页查询 dataset ==========
    async def list_datasets_paginated(
            self,
            page: int = 1,
            page_size: int = 50,
            sort_field: str = "updated_at",
            sort_direction: str = "desc",
    ):
        """分页查询 datasets"""
        return await self.storage.get_datasets_paginated(
            page=page,
            page_size=page_size,
            sort_field=sort_field,
            sort_direction=sort_direction,
        )

    # ========== 删除 dataset ==========
    async def delete_dataset(self, dataset_id: str):
        """删除 dataset"""
        await self.storage.delete_dataset(dataset_id)
        self._dataset_cache.pop(dataset_id, None)

    # ========== ：dataset 中添加 doc ==========
    async def add_doc_to_dataset(self, dataset_id: str, doc_id: str):
        """向 dataset 添加一个 doc"""
        await self.storage.add_doc_to_dataset(dataset_id, doc_id)

        # 更新缓存
        if dataset_id in self._dataset_cache:
            docs = self._dataset_cache[dataset_id].setdefault("docs", [])
            if doc_id not in docs:
                docs.append(doc_id)

    # ========== dataset 中移除 doc ==========
    async def remove_doc_from_dataset(self, dataset_id: str, doc_id: str):
        """从 dataset 移除一个 doc"""
        await self.storage.remove_doc_from_dataset(dataset_id, doc_id)

        if dataset_id in self._dataset_cache:
            docs = self._dataset_cache[dataset_id].setdefault("docs", [])
            if doc_id in docs:
                docs.remove(doc_id)


    # ==========统计 ==========
    async def get_dataset_counts(self):
        """获取各类状态数量统计，具体规则由 storage 决定"""
        return await self.storage.get_dataset_counts()

    # ========== 名称模糊搜索 ==========
    async def search_datasets_by_name(self, keyword: str):
        """按名称模糊搜索"""
        return await self.storage.search_datasets_by_name(keyword)


# create
# request and response models
class DatasetMetadataRequest(BaseModel):
    name: str
    description: Optional[str] = None

class DatasetMetadataResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    docs_count: int = 0
    docs: List[str] = Field(default_factory=list)




def create_dataset_routes(
    rag: LightRAG, dataset_manager:DatasetManager , api_key: Optional[str] = None
):
    # Create combined auth dependency for document routes
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/create",
        response_model=InsertResponse,
        dependencies=[Depends(combined_auth)]
    )
    async def create_dataset(
        metadata: DatasetMetadataRequest,
    ):
        try:
            name_lower = metadata.name.strip().lower()

            # —— 检查是否重名 ——
            existing = await dataset_manager.get_dataset_by_name(name_lower)
            if existing:
                return InsertResponse(
                    id = "",
                    status="duplicated",
                    message=f"Dataset '{name_lower}' already exists in the kv_store_dataset."
                )

            """创建一个新的 dataset"""
            dataset_id = "dataset-" +uuid4().hex
            now_iso = datetime.now(timezone.utc).isoformat()
            dataset_metadata = {
                "name": name_lower,
                "description": metadata.description or "",
                "created_at": now_iso,
                "updated_at": now_iso,
                "docs": [],
            }

            await dataset_manager.create_or_update_dataset(dataset_id, dataset_metadata)

            return InsertResponse(
                id = dataset_id,
                status="success",
                message=f"Dataset '{name_lower}' created successfully."
            )
        except Exception as e:
            logger.error(f"Error /documents/upload: {metadata.name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    return router




