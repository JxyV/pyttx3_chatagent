#!/usr/bin/env python3
"""
检查Ollama是否使用GPU
"""
import requests
import json
import os

def check_ollama_gpu():
    """检查Ollama GPU使用情况"""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        # 检查Ollama是否运行
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            print(f"❌ Ollama服务未运行或无法访问: {base_url}")
            return False
        
        print(f"✅ Ollama服务运行正常: {base_url}")
        
        # 检查模型信息
        model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        print(f"\n📦 检查模型: {model}")
        
        # 尝试拉取模型信息（如果模型不存在会自动拉取）
        show_response = requests.post(
            f"{base_url}/api/show",
            json={"name": model},
            timeout=30
        )
        
        if show_response.status_code == 200:
            model_info = show_response.json()
            print(f"✅ 模型信息获取成功")
            print(f"   模型大小: {model_info.get('modelfile', 'N/A')}")
        
        # 测试生成（检查GPU使用）
        print(f"\n🚀 测试生成（检查GPU使用）...")
        test_response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "你好",
                "stream": False,
                "options": {
                    "num_predict": 10,
                    "keep_alive": "5m"
                }
            },
            timeout=30
        )
        
        if test_response.status_code == 200:
            result = test_response.json()
            print(f"✅ 生成测试成功")
            print(f"   响应: {result.get('response', '')[:50]}...")
            print(f"\n💡 提示:")
            print(f"   - 如果生成速度很快（<1秒），说明可能在使用GPU")
            print(f"   - 如果生成速度很慢（>5秒），说明可能在使用CPU")
            print(f"   - 可以在任务管理器中查看GPU使用率")
        else:
            print(f"⚠️ 生成测试失败: {test_response.status_code}")
            print(f"   响应: {test_response.text}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到Ollama服务: {base_url}")
        print(f"   请确保Ollama正在运行: ollama serve")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Ollama GPU使用情况检查")
    print("=" * 60)
    check_ollama_gpu()
    print("=" * 60)


