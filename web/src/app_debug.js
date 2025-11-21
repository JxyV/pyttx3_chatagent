// 在浏览器控制台运行这些诊断函数

// 1. 检查WebSocket连接状态
function checkWebSocket() {
    if (!app.realtimeRecognition) {
        console.log('❌ WebSocket未初始化');
        return false;
    }
    
    const states = {
        0: 'CONNECTING',
        1: 'OPEN',
        2: 'CLOSING',
        3: 'CLOSED'
    };
    
    const state = app.realtimeRecognition.readyState;
    console.log(`WebSocket状态: ${states[state]} (${state})`);
    
    if (state === 1) {
        console.log('✅ WebSocket已连接');
        return true;
    } else {
        console.log('❌ WebSocket未连接');
        return false;
    }
}

// 2. 检查MediaRecorder状态
function checkMediaRecorder() {
    if (!app.mediaRecorder) {
        console.log('❌ MediaRecorder未初始化');
        return false;
    }
    
    console.log(`MediaRecorder状态: ${app.mediaRecorder.state}`);
    console.log(`麦克风激活状态: ${app.isMicrophoneActive}`);
    
    if (app.mediaRecorder.state === 'recording') {
        console.log('✅ 正在录音');
        return true;
    } else {
        console.log('❌ 未在录音');
        return false;
    }
}

// 3. 测试发送音频数据
async function testSendAudio() {
    console.log('开始测试音频发送...');
    
    // 检查WebSocket
    if (!checkWebSocket()) {
        console.log('请先点击麦克风按钮建立连接');
        return;
    }
    
    // 创建一个测试音频Blob
    const testData = new Uint8Array(1000);
    const blob = new Blob([testData], { type: 'audio/webm' });
    
    console.log('发送测试音频数据...');
    await app.sendAudioData(blob);
    console.log('测试音频数据已发送');
}

// 4. 监听WebSocket消息
function monitorWebSocket() {
    if (!app.realtimeRecognition) {
        console.log('❌ WebSocket未初始化');
        return;
    }
    
    const oldOnMessage = app.realtimeRecognition.onmessage;
    
    app.realtimeRecognition.onmessage = (event) => {
        console.log('📨 收到WebSocket消息:', event.data);
        const data = JSON.parse(event.data);
        console.log('消息类型:', data.type);
        console.log('消息内容:', data);
        
        // 调用原来的处理函数
        if (oldOnMessage) {
            oldOnMessage.call(app.realtimeRecognition, event);
        }
    };
    
    console.log('✅ WebSocket消息监控已启用');
}

// 5. 完整诊断
function fullDiagnostic() {
    console.log('='.repeat(60));
    console.log('前端语音识别完整诊断');
    console.log('='.repeat(60));
    
    console.log('\n1. WebSocket连接状态:');
    checkWebSocket();
    
    console.log('\n2. MediaRecorder状态:');
    checkMediaRecorder();
    
    console.log('\n3. 音频录制配置:');
    if (app.mediaRecorder) {
        console.log('   mimeType:', app.mediaRecorder.mimeType);
        console.log('   stream active:', app.mediaRecorder.stream.active);
        console.log('   audioBitsPerSecond:', app.mediaRecorder.audioBitsPerSecond);
    }
    
    console.log('\n4. 录音参数:');
    console.log('   format:', app.format);
    console.log('   channels:', app.channels);
    console.log('   rate:', app.rate);
    console.log('   chunk:', app.chunk);
    
    console.log('\n5. 音频发送参数:');
    console.log('   audioSendInterval:', app.audioSendInterval);
    console.log('   lastAudioSendTime:', app.lastAudioSendTime);
    
    console.log('\n6. 识别结果:');
    console.log('   recognizedText:', app.recognizedText);
    console.log('   isVoiceActive:', app.isVoiceActive);
    
    console.log('\n' + '='.repeat(60));
    console.log('诊断完成');
    console.log('='.repeat(60));
}

// 使用说明
console.log(`
前端语音识别诊断工具已加载

可用命令:
1. fullDiagnostic()     - 完整诊断
2. checkWebSocket()     - 检查WebSocket状态
3. checkMediaRecorder() - 检查录音器状态
4. testSendAudio()      - 测试发送音频
5. monitorWebSocket()   - 监控WebSocket消息

使用方法:
1. 打开浏览器控制台（F12）
2. 输入命令，例如: fullDiagnostic()
`);

