#!/usr/bin/env python3
"""
修复音频格式问题 - 在WebSocket服务器中添加音频格式转换
"""
import wave
import io

def extract_pcm_from_wav(wav_data: bytes) -> bytes:
    """从WAV文件中提取PCM数据"""
    try:
        # 使用BytesIO包装WAV数据
        wav_buffer = io.BytesIO(wav_data)
        
        # 使用wave模块读取WAV文件
        with wave.open(wav_buffer, 'rb') as wav_file:
            # 获取音频参数
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            print(f"WAV文件信息: 采样率={sample_rate}Hz, 声道={channels}, 位深={sample_width*8}位")
            
            # 读取所有音频帧
            pcm_data = wav_file.readframes(wav_file.getnframes())
            
            print(f"提取的PCM数据大小: {len(pcm_data)} 字节")
            return pcm_data
            
    except Exception as e:
        print(f"提取PCM数据失败: {e}")
        return wav_data  # 如果失败，返回原始数据

def test_audio_conversion():
    """测试音频转换"""
    print("🧪 测试音频格式转换...")
    
    # 创建测试WAV数据
    import numpy as np
    
    # 生成测试音频
    sample_rate = 8000
    duration = 1
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = (np.sin(2 * np.pi * frequency * t) * 0.3 * 32767).astype(np.int16)
    
    # 创建WAV文件
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    wav_data = wav_buffer.getvalue()
    print(f"原始WAV数据大小: {len(wav_data)} 字节")
    
    # 提取PCM数据
    pcm_data = extract_pcm_from_wav(wav_data)
    print(f"提取的PCM数据大小: {len(pcm_data)} 字节")
    
    # 验证数据
    expected_pcm_size = sample_rate * duration * 2  # 8kHz * 1秒 * 2字节
    print(f"期望的PCM数据大小: {expected_pcm_size} 字节")
    
    if len(pcm_data) == expected_pcm_size:
        print("✅ 音频格式转换成功！")
        return True
    else:
        print("❌ 音频格式转换失败！")
        return False

if __name__ == "__main__":
    test_audio_conversion()

