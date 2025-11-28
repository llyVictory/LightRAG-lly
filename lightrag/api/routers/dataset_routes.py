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
from lightrag.kg.dataset_impl import JsonDatasetStorage


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


# request and response models


# create
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


class DatasetRequest(BaseModel):
    name: str
    description: Optional[str] = None


# paginated
class DatasetResponse(BaseModel):
    """Dataset 响应"""
    id: str
    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    docs_count: int = 0
    docs: List[str] = Field(default_factory=list)


class PaginationInfo(BaseModel):
    """分页信息"""
    page: int
    page_size: int
    total_count: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedDatasetsResponse(BaseModel):
    """分页返回 Dataset 响应"""

    datasets: List[DatasetResponse] = Field(description="当前页的 dataset 列表")
    pagination: PaginationInfo = Field(description="分页信息")


    class Config:
        json_schema_extra = {
            "example": {
                "datasets": [
                    {
                        "id": "dataset-4dcdd8e7507240eab4ab203766cf15c8",
                        "name": "my_first_dataset25",
                        "description": "用于测试的 dataset",
                        "created_at": "2025-11-28T02:29:15.454797+00:00",
                        "updated_at": "2025-11-28T02:29:15.454797+00:00",
                        "docs_count": 0,
                        "docs": []
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 50,
                    "total_count": 100,
                    "total_pages": 2,
                    "has_next": True,
                    "has_prev": False,
                },
            }
        }

class DatasetsRequest(BaseModel):
    """Request model for paginated dataset queries"""

    name_filter: Optional[str] = Field(
        default=None,
        description="按 dataset 名称模糊搜索，None 表示查询所有"
    )
    page: int = Field(default=1, ge=1, description="页码（1 起始）")
    page_size: int = Field(default=50, ge=10, le=200, description="每页返回 dataset 数量")
    sort_field: Literal["created_at", "updated_at", "id", "name"] = Field(
        default="updated_at",
        description="排序字段"
    )
    sort_direction: Literal["asc", "desc"] = Field(
        default="desc",
        description="排序方向"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name_filter": "my_first_dataset",
                "page": 1,
                "page_size": 50,
                "sort_field": "updated_at",
                "sort_direction": "desc",
            }
        }
#delete
class DeleteDatasetByIdResponse(BaseModel):
    """Response model for single dataset deletion operation."""

    status: Literal["deletion_started", "busy", "not_allowed"] = Field(
        description="Status of the deletion operation"
    )
    message: str = Field(description="Message describing the operation result")
    dataset_id: str = Field(description="The ID of the dataset to delete")

    class Config:
        schema_extra = {
            "example": {
                "status": "deletion_started",
                "message": "Dataset deletion has been initiated successfully.",
                "dataset_id": "dataset-4dcdd8e7507240eab4ab203766cf15c8"
            }
        }

class DeleteDatasetRequest(BaseModel):
    """Request model for deleting one or more datasets."""

    dataset_ids: List[str] = Field(
        ..., description="The IDs of the datasets to delete."
    )
    delete_files: bool = Field(
        default=False,
        description="Whether to delete all files associated with the datasets."
    )
    delete_llm_cache: bool = Field(
        default=False,
        description="Whether to delete cached LLM extraction results for the datasets."
    )

    class Config:
        schema_extra = {
            "example": {
                "dataset_ids": [
                    "dataset-4dcdd8e7507240eab4ab203766cf15c8",
                    "dataset-ade2f030d47d401dba7357e8e1dbe4e0"
                ],
                "delete_files": True,
                "delete_llm_cache": False
            }
        }




# common
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
    DatasetManager 的职责是管理 dataset 与 doc 列表，
    """

    # ========== 加入 workspace 支持 ==========
    def __init__(
            self,
            workspace: str = "",
            storage: JsonDatasetStorage | None = None,  # 依赖注入
    ):
        """
        Args:
            workspace: 工作空间，用于数据隔离
            storage: dataset 存储（JsonDatasetStorage）
        """

        self.workspace = workspace
        self.storage = LazyStorageProxy(storage)

        if self.storage is None:
            raise ValueError("DatasetManager 需要提供 JsonDatasetStorage 实例")



    # 简单缓存：dataset_id → metadata
        self._dataset_cache: Dict[str, dict[str, Any]] = {}


    # ========== 创建或更新数据集 ==========
    async def create_or_update_dataset(self, dataset_id: str, metadata: dict[str, Any]):
        """创建或更新 dataset """
        await self.storage.create_or_update_dataset(dataset_id, metadata)
        self._dataset_cache[dataset_id] = metadata

    # ========== 根据dataset_id查询单个 dataset ==========
    async def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """获取 dataset """
        if dataset_id in self._dataset_cache:
            return self._dataset_cache[dataset_id]

        data = await self.storage.get_dataset(dataset_id)
        if data:
            self._dataset_cache[dataset_id] = data
        return data


    # ========== 根据name精确查询单个 dataset ==========
    async def get_dataset_by_name(self, name: str) -> dict[str, Any] | None:
        """按名称获取 dataset """
        data = await self.storage.get_dataset_by_exact_name(name)
        return data


    # ========== 根据name模糊查询单个 dataset ==========
    async def search_dataset_by_name(self, name: str) -> dict[str, Any] | None:
        """按名称获取 dataset """
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



    # ========== 名称模糊搜索 ==========
    async def search_datasets_by_name(self, keyword: str):
        """按名称模糊搜索"""
        return await self.storage.search_datasets_by_name(keyword)






def create_dataset_routes(
    rag: LightRAG, dataset_manager:DatasetManager , api_key: Optional[str] = None
):
    # Create combined auth dependency for document routes
    combined_auth = get_combined_auth_dependency(api_key)

    # ====== 创建 dataset ======
    @router.post(
        "/create",
        response_model=InsertResponse,
        dependencies=[Depends(combined_auth)]
    )
    async def create_dataset(
        metadata: DatasetRequest,
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
            dataset = {
                "name": name_lower,
                "description": metadata.description or "",
                "created_at": now_iso,
                "updated_at": now_iso,
                "docs": [],
            }

            await dataset_manager.create_or_update_dataset(dataset_id, dataset)

            return InsertResponse(
                id = dataset_id,
                status="success",
                message=f"Dataset '{name_lower}' created successfully."
            )
        except Exception as e:
            logger.error(f"Error /documents/upload: {metadata.name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    # ====== 分页 dataset  ======
    @router.post(
        "/paginated",
        response_model=PaginatedDatasetsResponse,
        dependencies=[Depends(combined_auth)]
    )
    async def get_datasets_paginated(
        request: DatasetsRequest,
    )->PaginatedDatasetsResponse:
        """分页查询 dataset """
        try:
            datasets_data = await dataset_manager.list_datasets_paginated(
                page=request.page,
                page_size=request.page_size,
                sort_field=request.sort_field,
                sort_direction=request.sort_direction,
            )

            datasets_list, total_count = datasets_data  # 解包 tuple

            datasets_response = []
            for dataset_id, metadata in datasets_list:
                datasets_response.append(
                    DatasetResponse(
                        id=dataset_id,
                        name=metadata.get("name", ""),
                        description=metadata.get("description", ""),
                        created_at=metadata.get("created_at", ""),
                        updated_at=metadata.get("updated_at", ""),
                        docs_count=len(metadata.get("docs", [])),
                        docs=metadata.get("docs", []),
                    )
                )

            total_pages = (total_count + request.page_size - 1) // request.page_size
            pagination_info = PaginationInfo(
                page=request.page,
                page_size=request.page_size,
                total_count=total_count,
                total_pages=total_pages,
                has_next=request.page < total_pages,
                has_prev=request.page > 1,
            )



            return PaginatedDatasetsResponse(
                datasets=datasets_response,
                pagination=pagination_info,
            )
        except Exception as e:
            logger.error(f"Error /datasets/paginated: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))


    # ===== 更新 dataset  =====
    # @router.put(
    #     "/update/{dataset_id}",
    #     response_model=InsertResponse,
    #     dependencies=[Depends(combined_auth)]
    # )
    # async def update_dataset(
    #         dataset_id: str,
    #         metadata: DatasetRequest,
    # ):
    #     """
    #     更新 dataset ：
    #     - name（若与其他 dataset 重名则不允许）
    #     - description
    #     - updated_at 自动更新
    #     """
    #     try:
    #         # 1. 先检查 dataset 是否存在
    #         existing = await dataset_manager.get_dataset(dataset_id)
    #         if not existing:
    #             return InsertResponse(
    #                 id=dataset_id,
    #                 status="failure",
    #                 message=f"Dataset '{dataset_id}' not found."
    #             )
    #
    #         new_name = metadata.name.strip().lower()
    #
    #         # 2. 如果修改名称，则检查是否重名
    #         if new_name != existing.get("name"):
    #             existed = await dataset_manager.get_dataset_by_name(new_name)
    #             if existed:
    #                 return InsertResponse(
    #                     id=dataset_id,
    #                     status="duplicated",
    #                     message=f"Dataset '{new_name}' already exists, name update not allowed."
    #                 )
    #
    #         # 3. 执行更新
    #         now_iso = datetime.now(timezone.utc).isoformat()
    #
    #         updated_metadata = {
    #             "name": new_name,
    #             "description": metadata.description or "",
    #             "created_at": existing.get("created_at"),
    #             "updated_at": now_iso,
    #             "docs": existing.get("docs", []),
    #         }
    #
    #         await dataset_manager.create_or_update_dataset(dataset_id, updated_metadata)
    #
    #         return InsertResponse(
    #             id=dataset_id,
    #             status="success",
    #             message=f"Dataset '{dataset_id}' updated successfully."
    #         )
    #
    #     except Exception as e:
    #         logger.error(f"Error /datasets/update/{dataset_id}: {str(e)}")
    #         logger.error(traceback.format_exc())
    #         raise HTTPException(status_code=500, detail=str(e))





















    # ===== 删除 Dataset =====
    # @router.delete(
    #     "/{dataset_id}",
    #     response_model=DeleteDatasetByIdResponse,
    #     dependencies=[Depends(combined_auth)],
    #     summary="Delete a dataset and all its associated data by its ID.",
    # )
    # async def delete_dataset(
    #         dataset_id: str,
    #         delete_request: DeleteDatasetRequest,
    #         background_tasks: BackgroundTasks,
    # ) -> DeleteDatasetByIdResponse:
    #     """
    #     Delete a dataset and all its associated data including documents, files, and LLM caches.
    #
    #     Args:
    #         dataset_id: The ID of the dataset to delete.
    #         delete_request: Flags indicating whether to delete files and LLM caches.
    #         background_tasks: FastAPI BackgroundTasks for async deletion of heavy tasks.
    #
    #     Returns:
    #         DeleteDatasetByIdResponse: Status of the deletion operation.
    #     """
    #     from lightrag.kg.shared_storage import (
    #         get_namespace_data,
    #         get_namespace_lock,
    #     )
    #     try:
    #         # 检查 pipeline 是否忙
    #         pipeline_status = await get_namespace_data("pipeline_status", workspace=dataset_id)
    #         pipeline_status_lock = get_namespace_lock("pipeline_status", workspace=dataset_id)
    #
    #         async with pipeline_status_lock:
    #             if pipeline_status.get("busy", False):
    #                 return DeleteDatasetByIdResponse(
    #                     status="busy",
    #                     message=f"Cannot delete dataset {dataset_id} while pipeline is busy",
    #                     dataset_id=dataset_id,
    #                 )
    #             pipeline_status["busy"] = True
    #             pipeline_status["latest_message"] = f"Starting deletion of dataset {dataset_id}"
    #
    #         # 删除 Dataset 下所有文档和 LLM 缓存（如果指定）
    #         async def _delete_dataset():
    #             errors = []
    #             try:
    #                 docs = await dataset_manager.list_dataset_docs(dataset_id)
    #                 for doc in docs:
    #                     try:
    #                         await dataset_manager.delete_document(
    #                             doc_id=doc.id,
    #                             delete_file=delete_request.delete_files,
    #                             delete_llm_cache=delete_request.delete_llm_cache,
    #                         )
    #                     except Exception as e:
    #                         logger.error(f"Error deleting document {doc.id}: {e}")
    #                         errors.append(f"{doc.id}: {str(e)}")
    #
    #                 # 删除 Dataset 
    #                 await dataset_manager.delete_dataset(dataset_id)
    #
    #             except Exception as e:
    #                 logger.error(f"Error deleting dataset {dataset_id}: {e}")
    #                 errors.append(str(e))
    #             finally:
    #                 async with pipeline_status_lock:
    #                     pipeline_status["busy"] = False
    #                     pipeline_status["latest_message"] = f"Dataset deletion completed: {dataset_id}"
    #
    #             return errors
    #
    #         # 使用后台任务执行删除，避免阻塞 API
    #         background_tasks.add_task(_delete_dataset)
    #
    #         return DeleteDatasetByIdResponse(
    #             status="deletion_started",
    #             message=f"Dataset deletion started for {dataset_id}. It will run in the background.",
    #             doc_id=dataset_id,
    #         )
    #
    #     except Exception as e:
    #         logger.error(f"Failed to start deletion for dataset {dataset_id}: {str(e)}")
    #         raise HTTPException(status_code=500, detail=str(e))
    #
    #

















    return router




