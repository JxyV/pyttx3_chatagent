#!/usr/bin/env python3
"""
Milvus数据库管理CLI工具
支持数据库创建、集合管理、数据备份等功能
"""
import argparse
import logging
import sys
from pathlib import Path
from app.services.milvus import MilvusManager
from app.config import get_rag_config


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def cmd_setup(args):
    """设置博物馆数据库"""
    print("🔧 设置博物馆数据库...")
    manager = MilvusManager()
    
    if manager.setup_museum_database():
        print("✅ 博物馆数据库设置成功")
        
        # 显示数据库信息
        collections = manager.list_collections()
        print(f"📋 当前集合: {collections}")
        
        for collection_name in collections:
            info = manager.get_collection_info(collection_name)
            if info:
                print(f"📊 集合 '{collection_name}': {info.get('num_entities', 0)} 个实体")
    else:
        print("❌ 博物馆数据库设置失败")
        sys.exit(1)


def cmd_list(args):
    """列出所有集合"""
    print("📋 列出所有集合...")
    manager = MilvusManager()
    
    collections = manager.list_collections()
    if collections:
        print(f"找到 {len(collections)} 个集合:")
        for collection_name in collections:
            info = manager.get_collection_info(collection_name)
            if info:
                print(f"  - {collection_name}: {info.get('num_entities', 0)} 个实体")
    else:
        print("没有找到任何集合")


def cmd_info(args):
    """显示集合详细信息"""
    collection_name = args.collection or get_rag_config().get_collection_name()
    print(f"📊 显示集合 '{collection_name}' 的详细信息...")
    
    manager = MilvusManager()
    info = manager.get_collection_info(collection_name)
    
    if info:
        print(f"集合名称: {info['name']}")
        print(f"实体数量: {info['num_entities']}")
        print(f"字段信息:")
        for field in info['schema'].fields:
            print(f"  - {field.name}: {field.dtype} (主键: {field.is_primary})")
        print(f"索引信息:")
        for index in info['indexes']:
            print(f"  - 字段: {index.field_name}, 类型: {index.params}")
    else:
        print(f"❌ 集合 '{collection_name}' 不存在或无法访问")


def cmd_drop(args):
    """删除集合"""
    collection_name = args.collection or get_rag_config().get_collection_name()
    print(f"🗑️  删除集合 '{collection_name}'...")
    
    if not args.force:
        confirm = input(f"确定要删除集合 '{collection_name}' 吗? (y/N): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return
    
    manager = MilvusManager()
    if manager.drop_collection(collection_name):
        print(f"✅ 集合 '{collection_name}' 删除成功")
    else:
        print(f"❌ 删除集合 '{collection_name}' 失败")
        sys.exit(1)


def cmd_backup(args):
    """备份集合数据"""
    collection_name = args.collection or get_rag_config().get_collection_name()
    backup_path = args.path or f"./backups/{collection_name}"
    
    print(f"💾 备份集合 '{collection_name}' 到 '{backup_path}'...")
    
    manager = MilvusManager()
    if manager.backup_collection(collection_name, backup_path):
        print(f"✅ 备份完成: {backup_path}")
    else:
        print(f"❌ 备份失败")
        sys.exit(1)


def cmd_config(args):
    """显示当前配置"""
    print("⚙️  当前Milvus配置:")
    config = get_rag_config()
    milvus_config = config.get_milvus_config()
    
    print(f"  主机: {milvus_config['host']}")
    print(f"  端口: {milvus_config['port']}")
    print(f"  数据库: {milvus_config['database']}")
    print(f"  集合名称: {config.get_collection_name()}")
    print(f"  向量维度: {config.get_vector_dim()}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Milvus数据库管理工具")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # setup命令
    setup_parser = subparsers.add_parser("setup", help="设置博物馆数据库")
    setup_parser.set_defaults(func=cmd_setup)
    
    # list命令
    list_parser = subparsers.add_parser("list", help="列出所有集合")
    list_parser.set_defaults(func=cmd_list)
    
    # info命令
    info_parser = subparsers.add_parser("info", help="显示集合详细信息")
    info_parser.add_argument("--collection", help="集合名称")
    info_parser.set_defaults(func=cmd_info)
    
    # drop命令
    drop_parser = subparsers.add_parser("drop", help="删除集合")
    drop_parser.add_argument("--collection", help="集合名称")
    drop_parser.add_argument("--force", action="store_true", help="强制删除，不询问确认")
    drop_parser.set_defaults(func=cmd_drop)
    
    # backup命令
    backup_parser = subparsers.add_parser("backup", help="备份集合数据")
    backup_parser.add_argument("--collection", help="集合名称")
    backup_parser.add_argument("--path", help="备份路径")
    backup_parser.set_defaults(func=cmd_backup)
    
    # config命令
    config_parser = subparsers.add_parser("config", help="显示当前配置")
    config_parser.set_defaults(func=cmd_config)
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 执行命令
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
