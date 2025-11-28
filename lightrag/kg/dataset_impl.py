from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Union, final
import datetime
from lightrag.utils import (
    load_json,
    logger,
    write_json,
    get_pinyin_sort_key,
)
from lightrag.exceptions import StorageNotInitializedError
from lightrag.base import (DatasetStorage)

from .shared_storage import (
    get_namespace_data,
    get_namespace_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
)

@final
@dataclass
class JsonDatasetStorage(DatasetStorage):
    """JSON implementation of dataset metadata storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

    async def initialize(self):
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        async with get_data_init_lock():
            need_init = await try_initialize_namespace(
                self.namespace, workspace=self.workspace
            )
            self._data = await get_namespace_data(
                self.namespace, workspace=self.workspace
            )

            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    self._data.update(loaded_data)
                    logger.info(
                        f"[{self.workspace}] Process {os.getpid()} dataset metadata load {self.namespace} with {len(loaded_data)} records"
                    )

    # -----------------------
    # BaseKVStorage 抽象方法实现
    # -----------------------

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            return self._data.get(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            return [self._data[id] for id in ids if id in self._data]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._storage_lock:
            return {k for k in keys if k not in self._data}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        async with self._storage_lock:
            self._data.update(data)
            await set_all_update_flags(self.namespace, workspace=self.workspace)
        await self.index_done_callback()

    async def delete(self, ids: list[str]) -> None:
        async with self._storage_lock:
            for id in ids:
                self._data.pop(id, None)
            await set_all_update_flags(self.namespace, workspace=self.workspace)
        await self.index_done_callback()




    # =====================================================================
    #                           基础存取方法
    # =====================================================================

    async def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            data = self._data.get(dataset_id)
            return data.copy() if data else None

    async def get_dataset_by_exact_name(
            self, name: str
    ) -> dict[str, dict[str, Any]]:
        """按名称精确匹配"""
        async with self._storage_lock:
            result = {}
            for ds_id, meta in self._data.items():
                ds_name = meta.get("name", "")
                if ds_name.lower() == name.lower():  # 完全匹配
                    result[ds_id] = meta.copy()
                    return meta.copy()




    async def search_datasets_by_name(
            self, keyword: str
    ) -> dict[str, dict[str, Any]]:
        """按名称模糊搜索"""
        async with self._storage_lock:
            result = {}
            for ds_id, meta in self._data.items():
                name = meta.get("name", "")
                if keyword.lower() in name.lower():
                    result[ds_id] = meta.copy()
            return result

    async def get_datasets_paginated(
            self,
            page: int = 1,
            page_size: int = 50,
            sort_field: str = "updated_at",
            sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, dict[str, Any]]], int]:

        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "name", "id"]:
            sort_field = "updated_at"

        sort_desc = sort_direction.lower() == "desc"

        items = []

        async with self._storage_lock:
            for ds_id, meta in self._data.items():
                data = meta.copy()
                # 用于排序
                if sort_field == "name":
                    data["_sort_key"] = get_pinyin_sort_key(data.get("name", ""))
                elif sort_field == "id":
                    data["_sort_key"] = ds_id
                else:
                    data["_sort_key"] = data.get(sort_field, "")

                items.append((ds_id, data))

        # 排序
        items.sort(key=lambda x: x[1]["_sort_key"], reverse=sort_desc)

        # 清理排序字段
        for _, data in items:
            data.pop("_sort_key", None)

        total = len(items)

        start = (page - 1) * page_size
        end = start + page_size

        return items[start:end], total

    async def create_or_update_dataset(
            self, dataset_id: str, metadata: dict[str, Any]
    ) -> None:
        """新增或更新 dataset"""
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDatasetStorage")

        async with self._storage_lock:
            # 确保字段基本格式统一
            metadata.setdefault("docs", [])
            metadata.setdefault("created_at", "")
            metadata.setdefault("updated_at", "")
            metadata.setdefault("name", "")
            self._data[dataset_id] = metadata

            await set_all_update_flags(self.namespace, workspace=self.workspace)

        await self.index_done_callback()

    async def delete_dataset(self, dataset_id: str) -> None:
        async with self._storage_lock:
            existed = self._data.pop(dataset_id, None)
            if existed:
                await set_all_update_flags(self.namespace, workspace=self.workspace)

        await self.index_done_callback()

    # =====================================================================
    #                           文档关联操作
    # =====================================================================

    async def add_doc_to_dataset(self, dataset_id: str, doc_id: str) -> None:
        async with self._storage_lock:
            dataset = self._data.get(dataset_id)
            if not dataset:
                return

            docs = dataset.setdefault("docs", [])
            if doc_id not in docs:
                docs.append(doc_id)

            await set_all_update_flags(self.namespace, workspace=self.workspace)

        await self.index_done_callback()

    async def remove_doc_from_dataset(self, dataset_id: str, doc_id: str) -> None:
        async with self._storage_lock:
            dataset = self._data.get(dataset_id)
            if not dataset or "docs" not in dataset:
                return

            if doc_id in dataset["docs"]:
                dataset["docs"].remove(doc_id)
                await set_all_update_flags(self.namespace, workspace=self.workspace)

        await self.index_done_callback()


    # =====================================================================
    #                     生命周期 — 写盘、清空、基础方法
    # =====================================================================

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )
                logger.debug(
                    f"[{self.workspace}] Process {os.getpid()} dataset metadata writing {len(data_dict)} records to {self.namespace}"
                )

                needs_reload = write_json(data_dict, self._file_name)

                if needs_reload:
                    cleaned = load_json(self._file_name)
                    if cleaned is not None:
                        self._data.clear()
                        self._data.update(cleaned)

                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def is_empty(self) -> bool:
        async with self._storage_lock:
            return len(self._data) == 0

    async def drop(self) -> dict[str, str]:
        """清空所有 dataset metadata"""
        try:
            async with self._storage_lock:
                self._data.clear()
                await set_all_update_flags(self.namespace, workspace=self.workspace)

            await self.index_done_callback()

            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}

        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
