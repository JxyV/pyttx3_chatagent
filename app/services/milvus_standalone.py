"""
Milvus Standalone 数据库管理模块
专门针对Milvus Standalone版本优化
"""
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from app.config import get_rag_config


class MilvusStandaloneManager:
    """Milvus Standalone 数据库管理器"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.milvus_config = self.config.get_milvus_config()
        self.vector_dim = self.config.get_vector_dim()
        self.collection_name = self.config.get_collection_name()
        
    def connect(self) -> bool:
        """连接到Milvus Standalone"""
        try:
            # 构建连接参数
            connection_args = {
                "host": self.milvus_config["host"],
                "port": self.milvus_config["port"]
            }
            
            # 连接到Milvus Standalone
            connections.connect("default", **connection_args)
            logging.info(f"成功连接到Milvus Standalone: {self.milvus_config['host']}:{self.milvus_config['port']}")
            return True
            
        except Exception as e:
            logging.error(f"连接Milvus Standalone失败: {e}")
            return False
    
    def create_collection(self, collection_name: str, vector_dim: int = None) -> bool:
        """创建集合（Standalone版本）"""
        try:
            if not self.connect():
                return False
            
            vector_dim = vector_dim or self.vector_dim
            
            # 检查集合是否已存在
            if utility.has_collection(collection_name):
                logging.info(f"集合 '{collection_name}' 已存在")
                return True
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields, f"Collection for {collection_name}")
            
            # 创建集合
            collection = Collection(collection_name, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
            
            logging.info(f"成功创建集合: {collection_name}")
            return True
            
        except Exception as e:
            logging.error(f"创建集合失败: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        try:
            if not self.connect():
                return []
            
            collections = utility.list_collections()
            logging.info(f"找到 {len(collections)} 个集合: {collections}")
            return collections
            
        except Exception as e:
            logging.error(f"列出集合失败: {e}")
            return []
    
    def drop_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            if not self.connect():
                return False
            
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logging.info(f"成功删除集合: {collection_name}")
            else:
                logging.warning(f"集合 '{collection_name}' 不存在")
            
            return True
            
        except Exception as e:
            logging.error(f"删除集合失败: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """获取集合信息"""
        try:
            if not self.connect():
                return {}
            
            if not utility.has_collection(collection_name):
                logging.warning(f"集合 '{collection_name}' 不存在")
                return {}
            
            collection = Collection(collection_name)
            info = {
                "name": collection_name,
                "num_entities": collection.num_entities,
                "schema": collection.schema,
                "indexes": collection.indexes
            }
            
            return info
            
        except Exception as e:
            logging.error(f"获取集合信息失败: {e}")
            return {}
    
    def setup_museum_database(self) -> bool:
        """设置博物馆数据库（Standalone版本）"""
        try:
            # 直接创建集合，Standalone版本不需要数据库概念
            if not self.create_collection(self.collection_name):
                return False
            
            logging.info("博物馆数据库设置完成（Standalone版本）")
            return True
            
        except Exception as e:
            logging.error(f"设置博物馆数据库失败: {e}")
            return False


def main():
    """测试Milvus Standalone管理器"""
    logging.basicConfig(level=logging.INFO)
    
    manager = MilvusStandaloneManager()
    
    # 设置博物馆数据库
    if manager.setup_museum_database():
        print("✅ 博物馆数据库设置成功")
        
        # 列出集合
        collections = manager.list_collections()
        print(f"📋 当前集合: {collections}")
        
        # 获取集合信息
        for collection_name in collections:
            info = manager.get_collection_info(collection_name)
            if info:
                print(f"📊 集合 '{collection_name}' 信息:")
                print(f"   - 实体数量: {info.get('num_entities', 0)}")
    else:
        print("❌ 博物馆数据库设置失败")


if __name__ == "__main__":
    main()
