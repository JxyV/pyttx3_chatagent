"""
Milvus数据库管理模块
支持数据库命名、子文件夹存储等功能
"""
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from app.config import get_rag_config


class MilvusManager:
    """Milvus数据库管理器"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.milvus_config = self.config.get_milvus_config()
        self.vector_dim = self.config.get_vector_dim()
        self.collection_name = self.config.get_collection_name()
        
    def connect(self) -> bool:
        """连接到Milvus数据库"""
        try:
            # 构建连接参数
            connection_args = {
                "host": self.milvus_config["host"],
                "port": self.milvus_config["port"]
            }
            
            # 如果有用户名和密码，添加到连接参数
            if self.milvus_config["username"] and self.milvus_config["password"]:
                connection_args.update({
                    "user": self.milvus_config["username"],
                    "password": self.milvus_config["password"]
                })
            
            # 连接到Milvus
            connections.connect("default", **connection_args)
            logging.info(f"成功连接到Milvus: {self.milvus_config['host']}:{self.milvus_config['port']}")
            return True
            
        except Exception as e:
            logging.error(f"连接Milvus失败: {e}")
            return False
    
    def create_database(self, database_name: str) -> bool:
        """创建数据库"""
        try:
            if not self.connect():
                return False
            
            # 尝试创建数据库，如果已存在会抛出异常
            try:
                utility.create_database(database_name)
                logging.info(f"成功创建数据库: {database_name}")
            except Exception as db_error:
                # 如果数据库已存在，这是正常的
                if "already exists" in str(db_error).lower() or "exist" in str(db_error).lower():
                    logging.info(f"数据库 '{database_name}' 已存在")
                else:
                    raise db_error
            
            return True
            
        except Exception as e:
            logging.error(f"创建数据库失败: {e}")
            return False
    
    def use_database(self, database_name: str) -> bool:
        """切换到指定数据库"""
        try:
            connections.connect("default", 
                              host=self.milvus_config["host"],
                              port=self.milvus_config["port"],
                              database=database_name)
            logging.info(f"切换到数据库: {database_name}")
            return True
        except Exception as e:
            logging.error(f"切换数据库失败: {e}")
            return False
    
    def create_collection(self, collection_name: str, vector_dim: int = None) -> bool:
        """创建集合"""
        try:
            if not self.connect():
                return False
            
            # 对于Standalone版本，跳过数据库切换
            try:
                if self.milvus_config["database"]:
                    self.use_database(self.milvus_config["database"])
            except Exception as db_error:
                logging.warning(f"数据库切换跳过（Standalone版本限制）: {db_error}")
            
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
            
            # 对于Standalone版本，跳过数据库切换
            try:
                if self.milvus_config["database"]:
                    self.use_database(self.milvus_config["database"])
            except Exception as db_error:
                logging.warning(f"数据库切换跳过（Standalone版本限制）: {db_error}")
            
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
            
            if self.milvus_config["database"]:
                if not self.use_database(self.milvus_config["database"]):
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
            
            if self.milvus_config["database"]:
                if not self.use_database(self.milvus_config["database"]):
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
        """设置博物馆数据库"""
        try:
            # 对于Milvus Standalone，我们简化流程，直接创建集合
            # 数据库功能在Standalone版本中可能有限制
            
            # 尝试创建数据库（可选，Standalone可能不支持）
            try:
                self.create_database(self.milvus_config["database"])
            except Exception as db_error:
                logging.warning(f"数据库创建跳过（Standalone版本限制）: {db_error}")
            
            # 直接创建集合
            if not self.create_collection(self.collection_name):
                return False
            
            logging.info("博物馆数据库设置完成")
            return True
            
        except Exception as e:
            logging.error(f"设置博物馆数据库失败: {e}")
            return False
    
    def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """备份集合数据"""
        try:
            if not self.connect():
                return False
            
            if self.milvus_config["database"]:
                if not self.use_database(self.milvus_config["database"]):
                    return False
            
            # 确保备份目录存在
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 这里可以实现具体的备份逻辑
            # 由于Milvus的备份功能比较复杂，这里只是示例
            logging.info(f"备份集合 '{collection_name}' 到 '{backup_path}'")
            return True
            
        except Exception as e:
            logging.error(f"备份集合失败: {e}")
            return False


def main():
    """测试Milvus管理器"""
    logging.basicConfig(level=logging.INFO)
    
    manager = MilvusManager()
    
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
