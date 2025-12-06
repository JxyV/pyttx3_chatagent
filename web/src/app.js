class MuseumChatApp {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.currentAudio = null;
        this.isVoiceMode = false;
        this.chatHistory = [];
        
        // 麦克风状态
        this.isMicrophoneActive = false;
        this.speechRecognition = null;
        
        // 实时语音识别相关
        this.realtimeRecognition = null;
        this.isVoiceActive = false;
        this.recognizedText = ''; // 当前显示的识别结果（用于实时显示）
        this.finalResults = []; // 累积所有的final识别结果
        this.recognitionTimeout = null;
        this.silenceTimeout = null; // 2秒静音检测定时器
        this.lastRecognitionTime = 0; // 最后一次识别到内容的时间
        this.lastAudioSendTime = 0;
        this.audioSendInterval = 100; // 最小发送间隔100ms，符合官方建议
        this.streamingSessionActive = false;
        this.waitingForFinalResult = false; // 标志：是否已停止录音，等待识别结果
        this.hasSentCurrentRecognition = false; // 标志：当前识别会话是否已发送，防止重复发送
        this.audioQueue = [];
        this.isPlayingQueue = false;
        this.isQueueClosing = false;
        this.hasStreamedTTS = false;
        
        // 流式语音合成相关
        this.ttsWebSocket = null;
        this.ttsBuffer = '';
        this.isStreamingTTS = false;
        this.currentTTSRequestId = null; // 当前TTS请求ID，用于取消旧的TTS
        this.streamingTTSBuffer = ''; // 流式TTS文本缓冲区
        this.streamingTTSRequestId = null; // 当前流式TTS的请求ID
        this.isStreamingTTSActive = false; // 是否正在流式TTS
        this.streamingTTSSentenceQueue = []; // 流式TTS句子队列
        this.isProcessingTTSSentence = false; // 是否正在处理TTS句子
        
        // 停止生成标志
        this.shouldIgnoreResponse = false;
        this.isGeneratingResponse = false; // 是否正在生成RAG响应
        this.currentResponseRequestId = null; // 当前响应请求ID，用于取消旧的响应
        
        this.init();
    }

    init() {
        this.initializeSocket();
        this.initializeElements();
        this.initializeEventListeners();
        this.initializeVoiceRecognition();
        this.updateConnectionStatus('connecting');
        
        // 不再自动启动语音识别，用户需要手动点击录音按钮启动
    }

    initializeSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('连接到服务器成功');
            this.isConnected = true;
            this.updateConnectionStatus('connected');
            this.hideLoading();
        });

        this.socket.on('disconnect', () => {
            console.log('与服务器断开连接');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
        });

        this.socket.on('response_start', (data) => {
            // 检查是否是当前请求的响应
            const requestId = data.requestId || null;
            if (requestId && requestId !== this.currentResponseRequestId) {
                console.log('⏭️ [忽略] 收到旧请求的响应，忽略');
                return;
            }
            
            // 重置忽略标志，开始新的响应
            this.shouldIgnoreResponse = false;
            this.isGeneratingResponse = true;
            
            // 如果是语音模式，准备流式TTS
            if (this.isVoiceMode) {
                this.streamingTTSRequestId = requestId;
                this.streamingTTSBuffer = '';
                this.isStreamingTTSActive = false; // 等待第一个chunk再启动
            }
            
            this.showTypingIndicator();
            this.addMessage('assistant', '', true); // 创建空消息用于流式更新
        });

        this.socket.on('response_chunk', (data) => {
            // 检查是否是当前请求的响应
            const requestId = data.requestId || null;
            if (requestId && requestId !== this.currentResponseRequestId) {
                console.log('⏭️ [忽略] 收到旧请求的响应块，忽略');
                return;
            }
            
            // 如果用户点击了停止，忽略后续的响应块
            if (this.shouldIgnoreResponse) {
                console.log('⏭️ [忽略] 忽略响应块（用户已停止）');
                return;
            }
            
            // 文字继续生成（即使TTS被取消）
            this.updateLastMessage(data.content, data.isFirst);
            
            // 流式TTS：如果是语音输入模式，在文字流式输出的同时进行TTS
            if ((this.isVoiceMode || data.autoTTS) && data.content) {
                this.handleStreamingTTS(data.content, requestId);
            }
        });

        this.socket.on('response_end', (data) => {
            // 检查是否是当前请求的响应
            const requestId = data.requestId || null;
            if (requestId && requestId !== this.currentResponseRequestId) {
                console.log('⏭️ [忽略] 收到旧请求的响应结束，忽略');
                return;
            }
            
            // 如果用户点击了停止，忽略响应结束事件，不播放音频
            if (this.shouldIgnoreResponse) {
                console.log('⏭️ [忽略] 忽略响应结束事件（用户已停止），不播放音频，不处理响应');
                this.hideTypingIndicator();
                this.isGeneratingResponse = false;
                // 清理可能残留的streaming消息
                const streamingMessage = this.elements.chatMessages.querySelector('.message[data-streaming="true"]');
                if (streamingMessage) {
                    const textDiv = streamingMessage.querySelector('.message-text');
                    const currentText = streamingMessage.dataset.rawText || '';
                    if (currentText) {
                        textDiv.innerHTML = this.formatMessageContent(currentText) + '<span style="color: #999; font-size: 0.9em;"> (已停止)</span>';
                    } else {
                        streamingMessage.remove();
                    }
                    delete streamingMessage.dataset.streaming;
                    delete streamingMessage.dataset.rawText;
                }
                // 确保不播放任何音频
                return;
            }
            
            this.isGeneratingResponse = false;
            
            this.hideTypingIndicator();
            this.finalizeLastMessage(data.fullResponse);
            
            // 完成流式TTS：处理剩余的缓冲区文本
            if (this.isStreamingTTSActive && this.streamingTTSRequestId === requestId) {
                this.finalizeStreamingTTS();
            }
            
            // 如果是语音输入（isVoiceMode为true）且后端允许自动TTS，但没有启动流式TTS，则播放完整回复
            if (!this.shouldIgnoreResponse && data.fullResponse && 
                (this.isVoiceMode || data.autoTTS) && !this.isStreamingTTSActive) {
                console.log('🎤 [自动播放] 语音输入，自动播放语音回复（非流式模式）...');
                // 生成新的TTS请求ID
                const ttsRequestId = Date.now();
                this.currentTTSRequestId = ttsRequestId;
                this.speakText(data.fullResponse, ttsRequestId);
            }

            // 本轮对话结束后重置语音模式状态
            this.isVoiceMode = false;
            this.isVoiceActive = false;
        });

        this.socket.on('error', (data) => {
            this.hideTypingIndicator();
            this.showError(data.message);
        });

        this.socket.on('user_typing', (data) => {
            // 处理其他用户打字状态
        });
    }

    initializeElements() {
        this.elements = {
            chatMessages: document.getElementById('chatMessages'),
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            micBtn: document.getElementById('micBtn'),
            stopBtn: document.getElementById('stopBtn'),
            clearChat: document.getElementById('clearChat'),
            typingIndicator: document.getElementById('typingIndicator'),
            voiceStatus: document.getElementById('voiceStatus'),
            connectionStatus: document.getElementById('connectionStatus'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            errorModal: document.getElementById('errorModal'),
            errorMessage: document.getElementById('errorMessage'),
            closeErrorModal: document.getElementById('closeErrorModal'),
            confirmError: document.getElementById('confirmError'),
            audioControls: document.getElementById('audioControls'),
            pauseAudio: document.getElementById('pauseAudio'),
            stopAudio: document.getElementById('stopAudio'),
            charCount: document.querySelector('.char-count'),
            voiceRecognitionArea: document.getElementById('voiceRecognitionArea'),
            voiceRecognitionText: document.getElementById('voiceRecognitionText')
        };
    }

    initializeEventListeners() {
        // 发送消息
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        this.elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // 输入框自动调整高度
        this.elements.messageInput.addEventListener('input', (e) => {
            this.autoResizeTextarea(e.target);
            this.updateCharCount();
        });

        // 麦克风按钮点击
        this.elements.micBtn.addEventListener('click', () => this.toggleMicrophone());
        
        // 停止按钮点击
        this.elements.stopBtn.addEventListener('click', () => this.stopGeneration());
        
        // 清空对话
        this.elements.clearChat.addEventListener('click', () => this.clearChat());

        // 建议问题点击
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                const question = e.target.getAttribute('data-question');
                this.elements.messageInput.value = question;
                this.sendMessage();
            });
        });

        // 错误模态框
        this.elements.closeErrorModal.addEventListener('click', () => this.hideError());
        this.elements.confirmError.addEventListener('click', () => this.hideError());

        // 音频控制
        this.elements.pauseAudio.addEventListener('click', () => this.pauseAudio());
        this.elements.stopAudio.addEventListener('click', () => this.stopAudio());

        // 点击模态框外部关闭
        this.elements.errorModal.addEventListener('click', (e) => {
            if (e.target === this.elements.errorModal) {
                this.hideError();
            }
        });
    }

    initializeVoiceRecognition() {
        // 初始化浏览器语音识别
        this.initializeBrowserSpeechRecognition();
        
        // 不在页面加载时就建立WebSocket连接，而是在用户点击麦克风时才建立
        // 这样可以避免在后端服务未启动时出现错误提示
    }

    initializeBrowserSpeechRecognition() {
        // 检查浏览器是否支持语音识别
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (SpeechRecognition) {
            this.speechRecognition = new SpeechRecognition();
            this.speechRecognition.continuous = false;
            this.speechRecognition.interimResults = true;
            this.speechRecognition.lang = 'zh-CN';
            
            this.speechRecognition.onstart = () => {
                console.log('语音识别开始');
                this.isMicrophoneActive = true;
                this.updateMicrophoneButton();
                this.elements.voiceStatus.style.display = 'flex';
            };
            
            this.speechRecognition.onresult = (event) => {
                let finalTranscript = '';
                let interimTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                    } else {
                        interimTranscript += transcript;
                    }
                }
                
                // 实时显示识别结果
                if (interimTranscript) {
                    this.elements.voiceRecognitionText.textContent = interimTranscript;
                    this.elements.voiceRecognitionArea.style.display = 'block';
                }
                
                // 最终结果，自动发送
                if (finalTranscript) {
                    this.elements.voiceRecognitionText.textContent = finalTranscript;
                    setTimeout(() => {
                        this.elements.voiceRecognitionArea.style.display = 'none';
                    }, 2000);
                    
                    // 将识别结果填入输入框
                    this.elements.messageInput.value = finalTranscript;
                    this.updateCharCount();
                    
                    // 自动发送消息
                    setTimeout(() => {
                        this.sendMessage();
                    }, 500);
                }
            };
            
            this.speechRecognition.onerror = (event) => {
                console.error('语音识别错误:', event.error);
                this.isMicrophoneActive = false;
                this.updateMicrophoneButton();
                this.elements.voiceStatus.style.display = 'none';
                this.elements.voiceRecognitionArea.style.display = 'none';
                
                let errorMessage = '语音识别失败';
                switch (event.error) {
                    case 'no-speech':
                        errorMessage = '未检测到语音，请重试';
                        break;
                    case 'audio-capture':
                        errorMessage = '无法访问麦克风';
                        break;
                    case 'not-allowed':
                        errorMessage = '麦克风权限被拒绝';
                        break;
                }
                this.showError(errorMessage);
            };
            
            this.speechRecognition.onend = () => {
                console.log('语音识别结束');
                this.isMicrophoneActive = false;
                this.updateMicrophoneButton();
                this.elements.voiceStatus.style.display = 'none';
                this.elements.voiceRecognitionArea.style.display = 'none';
            };
        } else {
            console.warn('浏览器不支持语音识别');
        }
    }

    async initializeRealtimeSpeechRecognition() {
        try {
            // 如果已经有连接且处于连接状态，直接返回
            if (this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.OPEN) {
                console.log('WebSocket已连接，无需重复建立');
                return true;
            }
            
            // 如果正在连接中，等待连接完成
            if (this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.CONNECTING) {
                console.log('WebSocket正在连接中...');
                return true;
            }
            
            // 建立WebSocket连接 - 使用本地实时语音识别
            console.log('正在建立WebSocket连接...');
            this.realtimeRecognition = new WebSocket('ws://localhost:8000/ws/realtime-speech');
            
            this.realtimeRecognition.onopen = () => {
                console.log('✅ 实时语音识别连接已建立 (本地FunASR模型)');
                // 仅建立连接，不自动启动流式识别，等用户真正开始录音时再发送start
                this.streamingSessionActive = false;
            };

            this.realtimeRecognition.onmessage = (event) => {
                console.log('📨 [DEBUG] 收到WebSocket消息:', event.data);
                const data = JSON.parse(event.data);
                console.log('📨 [DEBUG] 消息类型:', data.type);
                if (data.text) {
                    console.log('📨 [DEBUG] 识别文本:', data.text);
                }
                this.handleRealtimeSpeechResult(data);
            };

            this.realtimeRecognition.onerror = (error) => {
                console.error('❌ 实时语音识别连接错误:', error);
                // 不要立即弹窗，只记录错误
            };

            this.realtimeRecognition.onclose = () => {
                console.log('实时语音识别连接已关闭');
                // 如果当前仍在语音会话中，自动重新建立连接
                if (this.isVoiceActive) {
                    console.log('🔄 [DEBUG] 语音会话仍在进行，自动重新建立连接...');
                    // 延迟一点重连，避免立即重连失败
                    setTimeout(() => {
                        this.initializeRealtimeSpeechRecognition().catch(err => {
                            console.error('❌ [DEBUG] 自动重连失败:', err);
                        });
                    }, 500);
                } else {
                    this.realtimeRecognition = null;
                }
            };

            // 初始化音频录制
            await this.initializeAudioRecording();
            
            return true;
        } catch (error) {
            console.error('初始化实时语音识别失败:', error);
            // 不要立即弹窗，只记录错误
            return false;
        }
    }

    async initializeAudioRecording() {
        try {
            // 先清理旧的资源（如果存在）
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => track.stop());
            }
            
            // 获取麦克风权限 - 使用16kHz采样率匹配本地Paraformer
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,  // 16kHz采样率
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            // 保存stream引用，以便后续清理
            this.audioStream = stream;

            // 创建MediaRecorder（仅用于兼容性，实际使用AudioContext直接获取PCM）
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            this.audioChunks = [];

            // 使用AudioContext直接处理音频流，避免WebM解码问题
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.audioSource = this.audioContext.createMediaStreamSource(stream);
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            // 累积PCM数据
            this.pcmBuffer = [];
            this.lastSendTime = 0;
            
            // 检查AudioContext采样率，如果不是16kHz需要重采样
            const sourceSampleRate = this.audioContext.sampleRate;
            const targetSampleRate = 16000;
            const needsResample = sourceSampleRate !== targetSampleRate;
            
            console.log(`🎵 [DEBUG] AudioContext采样率: ${sourceSampleRate}Hz, 目标: ${targetSampleRate}Hz, 需要重采样: ${needsResample}`);
            
            // 辅助函数：将PCM数组转换为base64
            const pcmToBase64 = (pcm16Array) => {
                const pcmArray = new Uint8Array(pcm16Array.buffer);
                let binaryString = '';
                const chunkSize = 8192;
                for (let i = 0; i < pcmArray.length; i += chunkSize) {
                    const chunk = pcmArray.subarray(i, Math.min(i + chunkSize, pcmArray.length));
                    binaryString += String.fromCharCode.apply(null, chunk);
                }
                return btoa(binaryString);
            };
            
            this.processor.onaudioprocess = (event) => {
                // 只有在用户真正点击麦克风开始录音后才处理数据
                if (!this.isMicrophoneActive) {
                    return;
                }

                // 获取PCM数据（Float32Array，范围-1.0到1.0）
                const inputData = event.inputBuffer.getChannelData(0);
                
                let processedData = inputData;
                
                // 如果需要重采样
                if (needsResample) {
                    const targetLength = Math.floor(inputData.length * targetSampleRate / sourceSampleRate);
                    processedData = new Float32Array(targetLength);
                    
                    // 简单线性插值重采样
                    for (let i = 0; i < targetLength; i++) {
                        const srcIndex = (i * sourceSampleRate) / targetSampleRate;
                        const srcIndexFloor = Math.floor(srcIndex);
                        const srcIndexCeil = Math.min(srcIndexFloor + 1, inputData.length - 1);
                        const fraction = srcIndex - srcIndexFloor;
                        
                        processedData[i] = inputData[srcIndexFloor] * (1 - fraction) + inputData[srcIndexCeil] * fraction;
                    }
                }
                
                // 转换为16位PCM（Int16Array）
                const pcm16 = new Int16Array(processedData.length);
                for (let i = 0; i < processedData.length; i++) {
                    const sample = Math.max(-1, Math.min(1, processedData[i]));
                    pcm16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                }
                
                // 累积PCM数据
                this.pcmBuffer.push(pcm16);
                
                // 每100ms发送一次（约1600个样本，16kHz采样率，3200字节，与测试脚本一致）
                const now = Date.now();
                if (now - this.lastSendTime >= 100 && this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.OPEN && this.streamingSessionActive) {
                    // 合并累积的PCM数据
                    const totalLength = this.pcmBuffer.reduce((sum, arr) => sum + arr.length, 0);
                    if (totalLength > 0) {
                        const combinedPCM = new Int16Array(totalLength);
                        let offset = 0;
                        for (const arr of this.pcmBuffer) {
                            combinedPCM.set(arr, offset);
                            offset += arr.length;
                        }
                        this.pcmBuffer = []; // 清空缓冲区
                        
                        // 转换为base64发送
                        const base64Audio = pcmToBase64(combinedPCM);
                        
                        try {
                            this.realtimeRecognition.send(JSON.stringify({
                                type: 'audio',
                                audio: base64Audio
                            }));
                            
                            console.log('📤 [DEBUG] 发送PCM音频数据:', combinedPCM.length * 2, '字节 (', combinedPCM.length, '样本, 16kHz)');
                            this.lastSendTime = now;
                        } catch (error) {
                            console.error('❌ [DEBUG] 发送音频数据失败:', error);
                        }
                    }
                } else if (!this.streamingSessionActive) {
                    // 如果流式识别未启动，记录警告（但不要太频繁）
                    if (now - this.lastSendTime >= 1000) {
                        console.warn('⚠️ [DEBUG] 流式识别会话未激活，音频数据被丢弃');
                        this.lastSendTime = now;
                    }
                }
            };
            
            // 连接音频处理节点
            this.audioSource.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            // 保留MediaRecorder用于兼容性（但不再使用ondataavailable）
            this.mediaRecorder.ondataavailable = (event) => {
                // 只用于累积，不用于实时发送
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = async () => {
                console.log('🎙️ [DEBUG] 录音停止，开始处理音频...');
                console.log('📦 [DEBUG] 总共收集了', this.audioChunks.length, '个音频块');
                this.hideVoiceRecognitionArea();
                
                // 注意：不要在这里完全清理processor和audioSource
                // 因为下次录音时需要重新使用，但会在startMicrophone中检查并重新初始化
                // 这里只断开连接，保留引用以便检查
                if (this.processor) {
                    try {
                        this.processor.disconnect();
                    } catch (e) {
                        console.warn('⚠️ [DEBUG] 断开processor失败:', e);
                    }
                    // 不设为null，保留引用以便检查
                }
                if (this.audioSource) {
                    try {
                        this.audioSource.disconnect();
                    } catch (e) {
                        console.warn('⚠️ [DEBUG] 断开audioSource失败:', e);
                    }
                    // 不设为null，保留引用以便检查
                }
                
                // 发送剩余的PCM数据
                if (this.pcmBuffer.length > 0 && this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.OPEN) {
                    try {
                        // 合并累积的PCM数据
                        const totalLength = this.pcmBuffer.reduce((sum, arr) => sum + arr.length, 0);
                        const combinedPCM = new Int16Array(totalLength);
                        let offset = 0;
                        for (const arr of this.pcmBuffer) {
                            combinedPCM.set(arr, offset);
                            offset += arr.length;
                        }
                        this.pcmBuffer = [];
                        
                        // 转换为base64发送
                        const pcmArray = new Uint8Array(combinedPCM.buffer);
                        let binaryString = '';
                        const chunkSize = 8192;
                        for (let i = 0; i < pcmArray.length; i += chunkSize) {
                            const chunk = pcmArray.subarray(i, Math.min(i + chunkSize, pcmArray.length));
                            binaryString += String.fromCharCode.apply(null, chunk);
                        }
                        const base64Audio = btoa(binaryString);
                        
                        this.realtimeRecognition.send(JSON.stringify({
                            type: 'audio',
                            audio: base64Audio
                        }));
                        console.log('📤 [DEBUG] 发送最后一批PCM音频数据:', combinedPCM.length * 2, '字节');
                    } catch (error) {
                        console.error('❌ [DEBUG] 发送最后音频数据失败:', error);
                    }
                }
                
                // 发送结束信号
                if (this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.OPEN) {
                    this.realtimeRecognition.send(JSON.stringify({ type: 'end' }));
                    console.log('📤 [DEBUG] 已发送结束信号');
                    this.streamingSessionActive = false;
                }
                
                if (this.audioChunks.length > 0) {
                    // 合并所有音频块
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
                    console.log('✅ [DEBUG] 音频合并完成，总大小:', audioBlob.size, '字节');
                    
                    // 对于流式识别，音频数据已经在录音过程中发送
                    // 这里只需要清理
                    this.audioChunks = [];
                } else {
                    console.warn('⚠️ [DEBUG] 没有收集到音频数据');
                }
            };

            console.log('音频录制初始化成功');
        } catch (error) {
            console.error('音频录制初始化失败:', error);
            this.showError('无法访问麦克风，请检查权限设置');
        }
    }

    handleRealtimeSpeechResult(data) {
        // 无论RAG是否正在生成，都要处理识别结果
        console.log('🔍 [DEBUG] 处理语音识别结果，类型:', data.type, '文本:', data.text ? data.text.substring(0, 50) : '无', 
                   'RAG生成中:', this.isGeneratingResponse);
        
        switch (data.type) {
            case 'ready':
                console.log('✅ [DEBUG] 服务器就绪:', data.message);
                break;
            case 'partial':
            case 'interim':
                // 部分识别结果（中间结果），实时更新显示
                // 无论RAG是否正在生成，都要显示识别结果
                console.log('📝 [DEBUG] 中间识别结果:', data.text, 'RAG生成中:', this.isGeneratingResponse);
                if (data.text && data.text.trim()) {
                    // 检查当前识别会话是否已发送过，如果已发送则忽略后续识别结果
                    if (this.hasSentCurrentRecognition) {
                        console.log('⏭️ [忽略] 当前识别会话已发送，忽略新的interim结果');
                        return;
                    }
                    this.recognizedText = data.text;
                    // 更新黄色框框显示（确保显示区域可见）
                    this.updateVoiceRecognitionDisplay(this.recognizedText, false);
                    // 同时更新输入框，让用户看到实时识别结果
                    this.elements.messageInput.value = this.recognizedText;
                    this.updateCharCount();
                    this.resetRecognitionTimeout();
                    // 重置2秒静音检测定时器（即使在RAG生成期间也要重置）
                    this.lastRecognitionTime = Date.now();
                    this.resetSilenceTimeout();
                    console.log('✅ [识别显示] 已更新识别结果显示，文本:', this.recognizedText.substring(0, 50));
                }
                break;
            case 'final':
                // 最终识别结果
                // 无论RAG是否正在生成，都要处理并显示识别结果
                console.log('✅ [DEBUG] 最终识别结果:', data.text, 'RAG生成中:', this.isGeneratingResponse, '已发送:', this.hasSentCurrentRecognition);
                if (data.text && data.text.trim()) {
                    // 检查当前识别会话是否已发送过，如果已发送则忽略后续识别结果
                    if (this.hasSentCurrentRecognition) {
                        console.log('⏭️ [忽略] 当前识别会话已发送，忽略新的final结果');
                        return;
                    }
                    
                    const finalText = data.text.trim();
                    
                    // 检查这个final结果是否已经包含了之前累积的所有结果
                    // 后端在 stop_streaming 时会发送包含所有结果的完整合并文本
                    const previousText = this.finalResults.length > 0 
                        ? this.finalResults.join(' ') 
                        : '';
                    
                    // 如果新结果包含了之前的所有内容，说明是完整的合并结果，应该替换
                    // 或者新结果明显比之前的累积结果长很多，也说明是完整结果
                    const isCompleteResult = previousText.length > 0 && 
                                           (finalText.includes(previousText) || 
                                            finalText.length > previousText.length * 1.2);
                    
                    if (isCompleteResult) {
                        // 这是完整的合并结果，替换之前累积的结果，避免重复
                        console.log('📝 [DEBUG] 收到完整合并结果，替换累积结果（避免重复）');
                        this.finalResults = [finalText];
                        this.recognizedText = finalText;
                    } else {
                        // 这是单个句子，累积到数组中
                        this.finalResults.push(finalText);
                        // 更新当前显示的识别结果（合并所有结果）
                        this.recognizedText = this.finalResults.join(' ');
                    }
                    
                    // 更新显示（确保显示区域可见，即使在RAG生成期间）
                    this.updateVoiceRecognitionDisplay(this.recognizedText, true);
                    // 将最终结果填入输入框（用于显示）
                    this.elements.messageInput.value = this.recognizedText;
                    this.updateCharCount();
                    
                    // 重置2秒静音检测定时器（final结果后启动2秒定时器）
                    // 但如果当前会话已经发送过，就不要再启动定时器，避免重复发送
                    if (!this.hasSentCurrentRecognition) {
                        // 即使在RAG生成期间，也要重置定时器，以便用户可以打断
                        this.lastRecognitionTime = Date.now();
                        this.resetSilenceTimeout();
                        console.log('⏱️ [静音检测] final结果后启动2秒静音定时器，文本:', this.recognizedText.substring(0, 50), 'RAG生成中:', this.isGeneratingResponse);
                    } else {
                        console.log('⏭️ [忽略] 当前会话已发送，不再启动静音定时器，避免重复发送');
                    }
                    
                    console.log('💡 [提示] 识别结果已更新，当前完整文本长度:', this.recognizedText.length);
                    console.log('📊 [DEBUG] 累积的final结果数量:', this.finalResults.length);
                    console.log('✅ [识别显示] 已更新识别结果显示，文本:', this.recognizedText.substring(0, 50));
                }
                break;
            case 'end':
                console.log('🏁 [DEBUG] 识别会话结束');
                this.isVoiceActive = false;
                break;
            case 'status':
                console.log('📊 [DEBUG] 状态消息:', data.message);
                // 处理状态消息，比如"语音识别已结束"
                if (data.message && data.message.includes('语音识别已结束')) {
                    console.log('✅ [DEBUG] 语音识别会话正常结束');
                    this.isVoiceActive = false;
                    
                    // 如果已经停止录音，等待识别结果，则在识别结束后发送最终结果
                    if (this.waitingForFinalResult) {
                        // 直接使用 recognizedText，它已经在收到 final 消息时被更新为合并后的结果
                        // 这样可以避免重复合并，因为后端在 stop_streaming 时会发送包含所有结果的最终 final 消息
                        const finalText = this.recognizedText && this.recognizedText.trim() 
                            ? this.recognizedText.trim() 
                            : (this.finalResults.length > 0 ? this.finalResults.join(' ') : '');
                        
                        if (finalText && !this.hasSentCurrentRecognition) {
                            console.log('📤 [DEBUG] 识别已结束，发送最终完整识别结果:', finalText);
                            console.log('📊 [DEBUG] finalResults数量:', this.finalResults.length, 'recognizedText长度:', this.recognizedText.length);
                            
                            // 标记当前识别会话已发送，防止重复发送
                            this.hasSentCurrentRecognition = true;
                            
                            // 清除静音检测定时器，防止重复触发
                            if (this.silenceTimeout) {
                                clearTimeout(this.silenceTimeout);
                                this.silenceTimeout = null;
                            }
                            
                            this.sendRecognizedText();
                            
                            // 关键修复：发送后立即清空识别结果，防止下次意外重复发送
                            this.recognizedText = '';
                            this.finalResults = [];
                            this.elements.messageInput.value = '';
                            this.updateCharCount();
                        } else if (this.hasSentCurrentRecognition) {
                            console.log('⏭️ [忽略] 当前识别会话已发送，忽略status分支的重复发送');
                        } else {
                            console.log('⚠️ [提示] 识别已结束，但没有识别结果');
                            this.waitingForFinalResult = false;
                            this.hideVoiceRecognitionArea();
                        }
                    } else {
                        // 识别结束后，自动准备下次录音（不关闭连接，保持连接打开）
                        this.prepareForNextRecording();
                    }
                }
                break;
            case 'error':
                console.error('❌ [DEBUG] 语音识别错误:', data.error);
                this.isVoiceActive = false;
                this.isVoiceMode = false;
                this.showError('语音识别失败: ' + data.error);
                break;
            default:
                console.warn('⚠️ [DEBUG] 未知消息类型:', data.type);
        }
    }

    updateVoiceRecognitionDisplay(text, isFinal) {
        if (text.trim()) {
            this.elements.voiceRecognitionText.textContent = text;
            this.elements.voiceRecognitionArea.style.display = 'block';
            this.elements.voiceRecognitionArea.classList.add('listening');
            
            // 在常开模式下，不自动隐藏识别区域，保持显示以便用户看到识别内容
            // 只有在用户明确停止录音或发送消息后才隐藏
            // if (isFinal) {
            //     // 最终结果，3秒后隐藏
            //     setTimeout(() => {
            //         this.hideVoiceRecognitionArea();
            //     }, 3000);
            // }
        }
    }

    hideVoiceRecognitionArea() {
        // 在常开模式下，隐藏识别区域但不清空recognizedText，保持识别结果用于静音检测
        this.elements.voiceRecognitionArea.style.display = 'none';
        this.elements.voiceRecognitionArea.classList.remove('listening');
        // 不清空recognizedText，保持识别结果用于下次发送
        // this.recognizedText = '';
        // 不清空显示文本，保持显示以便用户看到识别内容（如果需要隐藏，可以清空）
        // this.elements.voiceRecognitionText.textContent = '';
    }

    toggleMicrophone() {
        if (this.isMicrophoneActive || this.streamingSessionActive) {
            // 如果正在录音或语音识别正在运行，则停止
            console.log('🛑 [切换] 停止录音和语音识别');
            this.stopMicrophone();
        } else {
            // 如果未在录音，则启动语音识别
            console.log('🎤 [切换] 启动录音和语音识别');
            // 如果WebSocket连接不存在或已关闭，先启动常开识别
            if (!this.realtimeRecognition || this.realtimeRecognition.readyState !== WebSocket.OPEN) {
                this.startContinuousRecognition();
            } else {
                // WebSocket连接存在，直接启动麦克风
                this.startMicrophone();
            }
        }
    }

    async startMicrophone() {
        try {
            console.log('🎤 [DEBUG] 开始启动麦克风...');
            
            // 立即停止所有正在播放的音频操作（仅在非常开模式下）
            // 在常开模式下，新识别输入时停止音频
            this.stopAudio();
            console.log('🛑 [DEBUG] 已停止所有音频播放');
            
            // 如果实时语音识别连接未建立，先建立连接
            if (!this.realtimeRecognition || this.realtimeRecognition.readyState !== WebSocket.OPEN) {
                console.log('🔌 [DEBUG] 建立语音识别连接...');
                await this.initializeRealtimeSpeechRecognition();
                
                // 等待连接建立（最多3秒）
                let waitCount = 0;
                while (this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.CONNECTING && waitCount < 30) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                    waitCount++;
                }
                
                // 检查连接状态
                if (!this.realtimeRecognition || this.realtimeRecognition.readyState !== WebSocket.OPEN) {
                    console.error('❌ [DEBUG] WebSocket连接失败');
                    this.showError('语音识别服务连接失败，请确认后端服务器已启动\n\n请运行: python websocket_server.py');
                    return;
                }
                console.log('✅ [DEBUG] WebSocket连接成功');
            }
            
            // 检查AudioContext和processor是否完整（可能在之前被清理了）
            const needsReinit = !this.mediaRecorder || 
                               !this.audioContext || 
                               !this.processor || 
                               !this.audioSource ||
                               this.audioContext.state === 'closed' ||
                               this.mediaRecorder.state === 'inactive' && (!this.processor || !this.audioSource);
            
            if (needsReinit) {
                console.log('⚠️ [DEBUG] 音频资源不完整，重新初始化...');
                console.log('    mediaRecorder:', !!this.mediaRecorder, this.mediaRecorder?.state);
                console.log('    audioContext:', !!this.audioContext, this.audioContext?.state);
                console.log('    processor:', !!this.processor);
                console.log('    audioSource:', !!this.audioSource);
                
                // 先清理旧的资源
                this.cleanupAudioResources();
                
                // 重新初始化
                await this.initializeAudioRecording();
            }
            
            // 再次检查MediaRecorder
            if (!this.mediaRecorder) {
                console.error('❌ [DEBUG] MediaRecorder初始化失败');
                this.showError('麦克风初始化失败，请检查浏览器权限');
                return;
            }
            
            // 检查processor是否正常连接
            if (!this.processor || !this.audioSource) {
                console.error('❌ [DEBUG] AudioContext或processor未正确初始化');
                this.showError('音频处理初始化失败，请刷新页面重试');
                return;
            }
            
            console.log('✅ [DEBUG] MediaRecorder已就绪');
            console.log('🎙️ [DEBUG] MediaRecorder状态:', this.mediaRecorder.state);
            console.log('🎙️ [DEBUG] MediaRecorder mimeType:', this.mediaRecorder.mimeType);
            console.log('🎵 [DEBUG] AudioContext状态:', this.audioContext.state);
            console.log('🎵 [DEBUG] processor已连接:', !!this.processor);

            // 标记语音会话激活，用于自动重连和语音回复
            this.isVoiceActive = true;

            // 确保AudioContext的节点重新连接
            if (this.audioSource && this.processor) {
                try {
                    this.audioSource.disconnect();
                } catch (e) {
                    // 可能已经断开
                }
                try {
                    this.processor.disconnect();
                } catch (e) {
                    // 可能已经断开
                }
                try {
                    this.audioSource.connect(this.processor);
                    this.processor.connect(this.audioContext.destination);
                    console.log('🔄 [DEBUG] AudioContext节点已重新连接');
                } catch (e) {
                    console.error('❌ [DEBUG] 重新连接AudioContext节点失败:', e);
                }
            }

            // 每次开始录音前，通知后端开启新的流式识别会话
            if (this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.OPEN) {
                try {
                    this.realtimeRecognition.send(JSON.stringify({ type: 'start' }));
                    console.log('🚀 [DEBUG] 已发送流式识别start指令');
                    this.streamingSessionActive = true;
                } catch (err) {
                    console.error('❌ [DEBUG] 发送start指令失败:', err);
                }
            } else {
                console.warn('⚠️ [DEBUG] WebSocket未连接，无法发送start指令');
            }

            // 清空之前的识别结果，开始新的录音会话（仅在首次启动时）
            // 在常开模式下，不清空识别结果，保持持续识别
            // 修正：每次启动麦克风时都清空之前的识别结果，防止残留
            if (!this.isMicrophoneActive) {
                console.log('🧹 [DEBUG] 启动麦克风，清空旧识别结果');
                this.recognizedText = '';
                this.finalResults = []; // 清空累积的final结果
                this.elements.messageInput.value = '';
                this.updateCharCount();
                // 重置识别会话发送标志，开始新的识别会话
                this.hasSentCurrentRecognition = false;
                console.log('🔓 [新会话] 重置识别会话发送标志，允许发送新的识别结果');
            }
            this.waitingForFinalResult = false; // 重置等待标志
            
            // 启动录音（AudioContext处理已在initializeAudioRecording中设置）
            // 如果已经在录音，不重复启动
            if (!this.isMicrophoneActive) {
                this.startRecording();
                this.elements.voiceRecognitionText.textContent = '请开始说话...';
                this.elements.voiceRecognitionArea.style.display = 'block';
                this.elements.voiceRecognitionArea.classList.add('listening');
            }
            
            console.log('✅ [DEBUG] 麦克风启动完成，使用AudioContext直接处理PCM');
        } catch (error) {
            console.error('❌ [DEBUG] 启动录音失败:', error);
            this.showError('录音启动失败：' + error.message);
        }
    }

    sendRecognizedText() {
        // 发送识别结果的通用方法（保留用于兼容）
        // 直接使用 recognizedText，它已经在收到 final 消息时被正确更新
        // 这样可以避免重复合并，因为后端在 stop_streaming 时会发送包含所有结果的最终 final 消息
        const finalText = (this.recognizedText && this.recognizedText.trim()) 
            ? this.recognizedText.trim() 
            : (this.finalResults.length > 0 ? this.finalResults.join(' ') : '');
        
        if (!finalText || !finalText.trim()) {
            console.warn('⚠️ [DEBUG] 没有识别结果可发送');
            this.waitingForFinalResult = false;
            return;
        }
        
        // 检查是否已经发送过
        if (this.hasSentCurrentRecognition) {
            console.log('⏭️ [忽略] 当前识别会话已发送，避免重复发送');
            return;
        }
        
        // 标记当前识别会话已发送
        this.hasSentCurrentRecognition = true;
        console.log('🔒 [防重复] 标记当前识别会话已发送');
        
        // 清除静音检测定时器，防止重复触发
        if (this.silenceTimeout) {
            clearTimeout(this.silenceTimeout);
            this.silenceTimeout = null;
            console.log('🧹 [清理] 清除静音检测定时器');
        }
        
        // 使用新的发送方法
        this.sendRecognizedTextToRAG(finalText);
        
        // 关键修复：发送后立即清空识别状态，防止重复发送
        this.recognizedText = '';
        this.finalResults = [];
        this.elements.messageInput.value = '';
        this.updateCharCount();
        
        // 重置等待标志
        this.waitingForFinalResult = false;
        
        // 隐藏欢迎消息
        this.hideWelcomeMessage();
    }

    stopMicrophone() {
        console.log('🛑 [DEBUG] 停止麦克风...');
        
        // 先停止MediaRecorder录音
        if (this.mediaRecorder && this.isMicrophoneActive) {
            this.stopRecording();
            // 结束信号会在 onstop 事件中的 sendAudioData 里发送
        } else {
            console.warn('⚠️ [DEBUG] 麦克风未在录音状态');
        }
        
        // 停止常开语音识别（这会关闭WebSocket连接和停止所有识别活动）
        this.stopContinuousRecognition();
        
        // 重置状态标志
        this.isMicrophoneActive = false;
        this.streamingSessionActive = false;
        this.isVoiceActive = false;
        this.waitingForFinalResult = false;
        
        // 更新按钮状态
        this.updateMicrophoneButton();
        
        console.log('💡 [提示] 停止录音，语音识别已完全关闭');
        console.log('📊 [DEBUG] 当前累积的final结果:', this.finalResults);
    }
    
    stopContinuousRecognition() {
        console.log('🛑 [停止] 停止持续语音识别...');
        
        // 停止流式识别会话
        if (this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.OPEN) {
            try {
                this.realtimeRecognition.send(JSON.stringify({ type: 'end' }));
                console.log('📤 [停止] 已发送结束信号');
            } catch (err) {
                console.error('❌ [停止] 发送结束信号失败:', err);
            }
        }
        
        // 关闭WebSocket连接
        if (this.realtimeRecognition) {
            try {
                // 移除所有事件监听器，避免内存泄漏
                this.realtimeRecognition.onopen = null;
                this.realtimeRecognition.onmessage = null;
                this.realtimeRecognition.onerror = null;
                this.realtimeRecognition.onclose = null;
                
                // 关闭连接
                if (this.realtimeRecognition.readyState === WebSocket.OPEN || 
                    this.realtimeRecognition.readyState === WebSocket.CONNECTING) {
                    this.realtimeRecognition.close();
                    console.log('🔌 [停止] 已关闭WebSocket连接');
                }
            } catch (err) {
                console.error('❌ [停止] 关闭WebSocket连接失败:', err);
            }
            this.realtimeRecognition = null;
        }
        
        // 清理音频资源
        this.cleanupAudioResources();
        
        // 隐藏语音识别区域
        this.hideVoiceRecognitionArea();
        
        // 重置状态标志
        this.streamingSessionActive = false;
        this.isVoiceActive = false;
        
        // 清空识别结果
        this.recognizedText = '';
        this.finalResults = [];
        if (this.elements.messageInput) {
            this.elements.messageInput.value = '';
            this.updateCharCount();
        }
        
        // 重置识别会话发送标志
        this.hasSentCurrentRecognition = false;
        
        console.log('✅ [停止] 持续语音识别已完全停止');
    }
    
    prepareForNextRecording() {
        console.log('🔄 [DEBUG] 准备下次录音...');
        
        // 确保WebSocket连接仍然打开
        if (!this.realtimeRecognition || this.realtimeRecognition.readyState !== WebSocket.OPEN) {
            console.log('⚠️ [DEBUG] WebSocket连接已关闭，重新建立连接...');
            // 自动重新建立连接
            this.initializeRealtimeSpeechRecognition().catch(err => {
                console.error('❌ [DEBUG] 重新建立连接失败:', err);
            });
        } else {
            console.log('✅ [DEBUG] WebSocket连接正常，准备就绪');
        }
        
        // 清空识别文本
        this.recognizedText = '';
        
        // 重置识别会话发送标志，准备新的识别会话
        this.hasSentCurrentRecognition = false;
        console.log('🔓 [新会话] 重置识别会话发送标志，准备新识别');
    }
    
    cleanupAudioResources() {
        console.log('🧹 [DEBUG] 清理音频资源...');
        
        // 断开processor
        if (this.processor) {
            try {
                this.processor.disconnect();
            } catch (e) {
                console.warn('⚠️ [DEBUG] 断开processor失败:', e);
            }
            this.processor = null;
        }
        
        // 断开audioSource
        if (this.audioSource) {
            try {
                this.audioSource.disconnect();
            } catch (e) {
                console.warn('⚠️ [DEBUG] 断开audioSource失败:', e);
            }
            this.audioSource = null;
        }
        
        // 关闭AudioContext（但不删除引用，以便检查）
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().catch(e => {
                console.warn('⚠️ [DEBUG] 关闭AudioContext失败:', e);
            });
        }
        this.audioContext = null;
        
        // 停止MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (e) {
                console.warn('⚠️ [DEBUG] 停止MediaRecorder失败:', e);
            }
        }
        this.mediaRecorder = null;
        
        // 停止音频流
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => {
                track.stop();
                console.log('🛑 [DEBUG] 停止音频轨道:', track.kind);
            });
            this.audioStream = null;
        }
        
        // 清空缓冲区
        this.pcmBuffer = [];
        this.audioChunks = [];
        this.lastSendTime = 0;
        
        console.log('✅ [DEBUG] 音频资源清理完成');
    }

    updateMicrophoneButton() {
        const micBtn = this.elements.micBtn;
        
        if (this.isMicrophoneActive || this.streamingSessionActive) {
            micBtn.classList.add('recording');
            micBtn.title = '结束语音对话';
            micBtn.textContent = '结束语音对话';
        } else {
            micBtn.classList.remove('recording');
            micBtn.title = '开始语音对话';
            micBtn.textContent = '开始语音对话';
        }
    }

    autoSendVoiceMessage(text) {
        if (text && text.trim() && this.isConnected) {
            console.log('📤 [DEBUG] 自动发送语音识别结果:', text);
            // 标记本轮对话为语音模式，以便自动语音回复
            this.isVoiceMode = true;
            
            // 添加用户消息到界面
            this.addMessage('user', text, false, true); // true表示这是语音消息
            
            // 发送到服务器
            this.socket.emit('send_message', {
                message: text,
                sessionId: this.socket.id
            });

            // 隐藏欢迎消息
            this.hideWelcomeMessage();
        }
    }

    resetRecognitionTimeout() {
        if (this.recognitionTimeout) {
            clearTimeout(this.recognitionTimeout);
        }
        this.recognitionTimeout = setTimeout(() => {
            this.hideVoiceRecognitionArea();
        }, 5000); // 5秒无语音输入后隐藏
    }
    
    // 重置2秒静音检测定时器
    resetSilenceTimeout() {
        // 清除旧的定时器
        if (this.silenceTimeout) {
            clearTimeout(this.silenceTimeout);
            this.silenceTimeout = null;
        }
        
        // 2秒静音后自动发送识别内容
        // 注意：即使在RAG生成期间，也要允许静音检测触发，以便用户可以打断
        this.silenceTimeout = setTimeout(() => {
            console.log('⏱️ [静音检测] 2秒定时器触发');
            this.onSilenceDetected();
        }, 2000);
        console.log('⏱️ [静音检测] 重置2秒静音定时器，将在2秒后检查');
    }
    
    // 2秒静音检测到，自动发送识别内容
    onSilenceDetected() {
        console.log('🔇 [静音检测] 2秒静音检测触发');
        console.log('📊 [静音检测] 当前recognizedText:', this.recognizedText);
        console.log('📊 [静音检测] 当前finalResults:', this.finalResults);
        console.log('📊 [静音检测] RAG生成中:', this.isGeneratingResponse);
        console.log('📊 [静音检测] 已发送:', this.hasSentCurrentRecognition);
        
        // 检查是否已经发送过
        if (this.hasSentCurrentRecognition) {
            console.log('⏭️ [忽略] 当前识别会话已发送，忽略静音检测触发');
            return;
        }
        
        const finalText = this.recognizedText && this.recognizedText.trim() 
            ? this.recognizedText.trim() 
            : (this.finalResults.length > 0 ? this.finalResults.join(' ') : '');
        
        if (!finalText) {
            console.log('⚠️ [静音检测] 2秒静音，但没有识别内容，不发送');
            return;
        }
        
        console.log('🔇 [静音检测] 2秒静音，自动发送识别内容:', finalText);
        
        // 标记当前识别会话已发送
        this.hasSentCurrentRecognition = true;
        console.log('🔒 [防重复] 标记当前识别会话已发送');
        
        // 清除定时器，避免重复触发
        if (this.silenceTimeout) {
            clearTimeout(this.silenceTimeout);
            this.silenceTimeout = null;
        }
        
        // 如果有正在进行的响应生成，取消其TTS播放（但文字继续生成）
        if (this.isGeneratingResponse) {
            console.log('🛑 [打断] 检测到新语音输入，取消当前响应的TTS播放');
            this.cancelCurrentTTS();
        }
        
        // 如果有正在播放的TTS，停止播放
        this.stopAudio();
        
        // 发送识别内容给RAG（即使RAG正在生成，也要发送新的请求）
        // 这会创建一个新的请求，旧的请求会被标记为已中断
        console.log('📤 [静音检测] 发送识别内容到RAG，即使RAG正在生成');
        this.sendRecognizedTextToRAG(finalText);
    }
    
    // 取消当前TTS任务
    cancelCurrentTTS() {
        // 设置标志，阻止后续TTS播放
        this.shouldIgnoreResponse = true;
        
        // 取消流式TTS
        if (this.isStreamingTTSActive) {
            this.cancelStreamingTTS();
        }
        
        // 停止音频播放
        this.stopAudio();
        
        // 关闭TTS WebSocket连接
        if (this.ttsWebSocket) {
            try {
                if (this.ttsWebSocket.readyState === WebSocket.OPEN || this.ttsWebSocket.readyState === WebSocket.CONNECTING) {
                    console.log('🛑 [取消TTS] 关闭TTS WebSocket连接');
                    this.ttsWebSocket.close();
                }
            } catch (e) {
                console.error('关闭TTS连接失败:', e);
            }
            this.ttsWebSocket = null;
        }
        
        // 生成新的请求ID，旧的TTS请求会被忽略
        this.currentTTSRequestId = Date.now();
    }
    
    // 发送识别内容给RAG（带打断机制）
    sendRecognizedTextToRAG(text) {
        if (!text || !text.trim()) {
            return;
        }
        
        console.log('📤 [发送RAG] 发送识别内容:', text);
        
        // 生成新的请求ID
        const requestId = Date.now();
        this.currentResponseRequestId = requestId;
        this.isGeneratingResponse = true;
        
        // 标记为语音模式，以便自动播放语音回复
        this.isVoiceMode = true;
        console.log('🎤 [语音模式] 标记为语音输入，将自动播放语音回复');
        
        // 如果有正在进行的响应，取消其TTS（但允许文字继续生成）
        if (this.isGeneratingResponse && this.currentResponseRequestId !== requestId) {
            console.log('🛑 [打断] 取消旧响应的TTS');
            this.cancelCurrentTTS();
        }
        
        // 添加用户消息到界面
        this.addMessage('user', text, false, true);
        
        // 发送到服务器（包含requestId）
        this.socket.emit('send_message', {
            message: text,
            sessionId: this.socket.id,
            requestId: requestId // 传递请求ID
        });
        
        // 彻底清除静音检测定时器，防止重复触发
        if (this.silenceTimeout) {
            clearTimeout(this.silenceTimeout);
            this.silenceTimeout = null;
            console.log('🧹 [清理] 发送后彻底清除静音检测定时器');
        }
        
        // 清空输入框
        this.elements.messageInput.value = '';
        this.updateCharCount();
        
        // 隐藏语音识别区域（但保持录音和识别继续）
        this.hideVoiceRecognitionArea();
        
        // 清空识别结果，准备下次识别（但保持录音继续）
        this.recognizedText = '';
        this.finalResults = [];
        
        // 延迟重置识别会话发送标志，防止后端残留消息导致状态混乱
        // 延长到3秒，确保所有残留的静音检测定时器都已过期，防止重复发送
        setTimeout(() => {
            this.hasSentCurrentRecognition = false;
            console.log('🔓 [新会话] 延迟重置识别会话发送标志，准备接收新识别');
        }, 3000);
    }
    
    // 启动持续语音识别（常开模式）
    async startContinuousRecognition() {
        try {
            console.log('🎤 [常开模式] 启动持续语音识别...');
            
            // 初始化实时语音识别连接
            await this.initializeRealtimeSpeechRecognition();
            
            // 等待连接建立
            let waitCount = 0;
            while (this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.CONNECTING && waitCount < 30) {
                await new Promise(resolve => setTimeout(resolve, 100));
                waitCount++;
            }
            
            if (!this.realtimeRecognition || this.realtimeRecognition.readyState !== WebSocket.OPEN) {
                console.error('❌ [常开模式] WebSocket连接失败');
                return;
            }
            
            // 启动流式识别
            if (this.realtimeRecognition.readyState === WebSocket.OPEN) {
                this.realtimeRecognition.send(JSON.stringify({ type: 'start' }));
                console.log('🚀 [常开模式] 已发送流式识别start指令');
                this.streamingSessionActive = true;
            }
            
            // 启动持续录音（不停止）
            await this.startMicrophone();
            
            console.log('✅ [常开模式] 持续语音识别已启动，将一直保持开启状态');
        } catch (error) {
            console.error('❌ [常开模式] 启动失败:', error);
        }
    }

    sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (!message || !this.isConnected) return;

        // 检查是否是语音识别结果（如果输入框内容与识别结果匹配或相似，认为是语音输入）
        const isVoiceInput = this.recognizedText && 
                            (message === this.recognizedText.trim() || 
                             message.includes(this.recognizedText.trim()) ||
                             this.recognizedText.trim().includes(message));
        
        if (isVoiceInput) {
            // 语音输入，标记为语音模式以便自动播放语音回复
            this.isVoiceMode = true;
            console.log('🎤 [DEBUG] 检测到语音输入，将自动播放语音回复');
        } else {
            // 普通文本输入，退出语音模式
            this.isVoiceMode = false;
        }

        // 添加用户消息到界面
        this.addMessage('user', message, false, isVoiceInput); // isVoiceInput 表示这是语音消息
        
        // 清空输入框和识别结果
        this.elements.messageInput.value = '';
        this.recognizedText = '';
        this.updateCharCount();
        this.autoResizeTextarea(this.elements.messageInput);

        // 发送到服务器
        this.socket.emit('send_message', {
            message: message,
            sessionId: this.socket.id
        });

        // 隐藏欢迎消息
        this.hideWelcomeMessage();
    }

    addMessage(role, content, isStreaming = false, isVoiceMessage = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (role === 'assistant') {
            const header = document.createElement('div');
            header.className = 'message-header';
            header.innerHTML = '<i class="fas fa-robot"></i> 博物馆AI助手';
            messageContent.appendChild(header);
        } else if (role === 'user' && isVoiceMessage) {
            const header = document.createElement('div');
            header.className = 'message-header';
            header.innerHTML = '<i class="fas fa-microphone"></i> 语音输入';
            messageContent.appendChild(header);
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-text';
        if (isStreaming) {
            contentDiv.innerHTML = '<span class="streaming-cursor">|</span>';
            messageDiv.dataset.streaming = 'true';
            messageDiv.dataset.rawText = '';
        } else {
            contentDiv.innerHTML = this.formatMessageContent(content);
        }
        messageContent.appendChild(contentDiv);

        // 添加操作按钮
        if (role === 'assistant' && content) {
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            
            // 创建朗读按钮
            const speakBtn = document.createElement('button');
            speakBtn.innerHTML = '<i class="fas fa-volume-up"></i> 朗读';
            speakBtn.dataset.text = content;
            speakBtn.addEventListener('click', () => {
                this.speakText(content);
            });
            
            // 创建复制按钮
            const copyBtn = document.createElement('button');
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> 复制';
            copyBtn.addEventListener('click', () => {
                this.copyToClipboard(content);
            });
            
            actions.appendChild(speakBtn);
            actions.appendChild(copyBtn);
            messageContent.appendChild(actions);
        }

        messageDiv.appendChild(messageContent);
        this.elements.chatMessages.appendChild(messageDiv);
        
        // 滚动到底部
        this.scrollToBottom();
        
        return messageDiv;
    }

    updateLastMessage(content, isFirst) {
        const lastMessage = this.elements.chatMessages.querySelector('.message[data-streaming="true"]');
        if (lastMessage) {
            const textDiv = lastMessage.querySelector('.message-text');
            const previousText = lastMessage.dataset.rawText || '';
            lastMessage.dataset.rawText = isFirst ? content : (previousText + content);

            if (isFirst) {
                textDiv.innerHTML = this.formatMessageContent(lastMessage.dataset.rawText) + '<span class="streaming-cursor">|</span>';
            } else {
                textDiv.innerHTML = this.formatMessageContent(lastMessage.dataset.rawText) + '<span class="streaming-cursor">|</span>';
            }
            this.scrollToBottom();
        }
    }

    finalizeLastMessage(fullResponse) {
        const lastMessage = this.elements.chatMessages.querySelector('.message[data-streaming="true"]');
        if (lastMessage) {
            const textDiv = lastMessage.querySelector('.message-text');
            textDiv.innerHTML = this.formatMessageContent(fullResponse);
            delete lastMessage.dataset.streaming;
            delete lastMessage.dataset.rawText;
            
            // 添加操作按钮
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            
            // 创建朗读按钮
            const speakBtn = document.createElement('button');
            speakBtn.innerHTML = '<i class="fas fa-volume-up"></i> 朗读';
            speakBtn.dataset.text = fullResponse;
            speakBtn.addEventListener('click', () => {
                this.speakText(fullResponse);
            });
            
            // 创建复制按钮
            const copyBtn = document.createElement('button');
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> 复制';
            copyBtn.addEventListener('click', () => {
                this.copyToClipboard(fullResponse);
            });
            
            actions.appendChild(speakBtn);
            actions.appendChild(copyBtn);
            lastMessage.querySelector('.message-content').appendChild(actions);
        }
    }

    toggleVoiceRecording() {
        if (this.isRecording) {
            this.stopVoiceRecording();
        } else {
            this.startVoiceRecording();
        }
    }

    startVoiceRecording() {
        if (!this.speechRecognition) {
            this.showError('您的浏览器不支持语音识别');
            return;
        }

        this.isRecording = true;
        this.elements.voiceRecordBtn.classList.add('recording');
        this.elements.voiceStatus.style.display = 'flex';
        this.speechRecognition.start();
    }

    stopVoiceRecording() {
        this.isRecording = false;
        this.elements.voiceRecordBtn.classList.remove('recording');
        this.elements.voiceStatus.style.display = 'none';
        if (this.speechRecognition) {
            this.speechRecognition.stop();
        }
    }

    startRecording() {
        console.log('🎙️ [DEBUG] startRecording被调用');
        console.log('🎙️ [DEBUG] mediaRecorder存在:', !!this.mediaRecorder);
        if (this.mediaRecorder) {
            console.log('🎙️ [DEBUG] mediaRecorder状态:', this.mediaRecorder.state);
        }
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'inactive') {
            this.audioChunks = [];
            this.pcmBuffer = []; // 重置PCM缓冲区
            this.lastSendTime = 0;
            // 启动MediaRecorder（用于兼容性，但主要使用AudioContext）
            this.mediaRecorder.start();
            this.isMicrophoneActive = true;
            this.updateMicrophoneButton();
            console.log('✅ [DEBUG] 录音已启动，使用AudioContext直接获取PCM数据');
        } else if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            console.warn('⚠️ [DEBUG] mediaRecorder状态不是inactive，当前状态:', this.mediaRecorder.state);
        } else {
            console.error('❌ [DEBUG] mediaRecorder不存在或未初始化');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.isMicrophoneActive = false;
            this.updateMicrophoneButton();
            console.log('停止录音');
        }
    }

    // 处理流式TTS：在RAG文字流式输出的同时进行TTS
    handleStreamingTTS(chunk, requestId) {
        // 如果用户已停止，不处理TTS
        if (this.shouldIgnoreResponse) {
            return;
        }
        
        // 如果是新的请求，重置流式TTS状态
        if (this.streamingTTSRequestId !== requestId) {
            // 取消旧的流式TTS
            if (this.isStreamingTTSActive) {
                this.cancelStreamingTTS();
            }
            // 初始化新的流式TTS
            this.streamingTTSRequestId = requestId;
            this.streamingTTSBuffer = '';
            this.isStreamingTTSActive = true;
            this.currentTTSRequestId = requestId;
            
            // 停止当前正在播放的音频
            this.stopAudio();
            
            // 初始化TTS WebSocket连接
            this.initStreamingTTS();
        }
        
        // 累积文本到缓冲区
        this.streamingTTSBuffer += chunk;
        
        // 检查是否有完整的句子（以句号、问号、感叹号结尾）
        const sentenceEndRegex = /[。！？\n]+/g;
        let lastIndex = 0;
        let match;
        const sentences = [];
        
        while ((match = sentenceEndRegex.exec(this.streamingTTSBuffer)) !== null) {
            const sentence = this.streamingTTSBuffer.substring(lastIndex, match.index + 1).trim();
            if (sentence) {
                sentences.push(sentence);
            }
            lastIndex = match.index + 1;
        }
        
        // 如果有完整的句子，发送到TTS
        if (sentences.length > 0) {
            sentences.forEach(sentence => {
                this.sendSentenceToTTS(sentence);
            });
            // 保留未完成的文本
            this.streamingTTSBuffer = this.streamingTTSBuffer.substring(lastIndex);
        }
    }
    
    // 初始化流式TTS WebSocket连接
    initStreamingTTS() {
        // 如果已有连接，先关闭
        if (this.ttsWebSocket) {
            try {
                this.ttsWebSocket.close();
            } catch (e) {
                console.error('关闭旧TTS连接失败:', e);
            }
            this.ttsWebSocket = null;
        }
        
        this.showAudioControls();
        this.audioQueue = [];
        this.isPlayingQueue = false;
        this.isQueueClosing = false;
        this.hasStreamedTTS = false;
        
        console.log('🎤 [流式TTS] 初始化流式TTS连接...');
    }
    
    // 发送句子到TTS进行合成（加入队列，按顺序处理）
    sendSentenceToTTS(sentence) {
        if (!sentence || !sentence.trim() || this.shouldIgnoreResponse) {
            return;
        }
        
        console.log('📤 [流式TTS] 句子加入队列:', sentence.substring(0, 50) + '...');
        
        // 将句子加入队列
        this.streamingTTSSentenceQueue.push(sentence.trim());
        
        // 如果当前没有在处理，开始处理队列
        if (!this.isProcessingTTSSentence) {
            this.processTTSSentenceQueue();
        }
    }
    
    // 处理TTS句子队列（按顺序发送和播放）
    async processTTSSentenceQueue() {
        if (this.shouldIgnoreResponse || this.streamingTTSSentenceQueue.length === 0) {
            this.isProcessingTTSSentence = false;
            return;
        }
        
        this.isProcessingTTSSentence = true;
        const sentence = this.streamingTTSSentenceQueue.shift();
        
        console.log('🎤 [流式TTS] 处理句子:', sentence.substring(0, 50) + '...');
        
        // 如果WebSocket未连接，建立连接
        if (!this.ttsWebSocket || this.ttsWebSocket.readyState !== WebSocket.OPEN) {
            await this.connectTTSWebSocket();
        }
        
        // 等待连接建立
        let retries = 0;
        while ((!this.ttsWebSocket || this.ttsWebSocket.readyState !== WebSocket.OPEN) && retries < 50) {
            await new Promise(resolve => setTimeout(resolve, 100));
            retries++;
        }
        
        if (!this.ttsWebSocket || this.ttsWebSocket.readyState !== WebSocket.OPEN) {
            console.error('❌ [流式TTS] WebSocket连接失败');
            this.isProcessingTTSSentence = false;
            // 继续处理队列
            if (this.streamingTTSSentenceQueue.length > 0) {
                this.processTTSSentenceQueue();
            }
            return;
        }
        
        // 创建一个Promise来等待这个句子的TTS完成
        const sentencePromise = new Promise((resolve) => {
            const originalOnMessage = this.ttsWebSocket.onmessage;
            let audioReceived = false;
            
            this.ttsWebSocket.onmessage = (event) => {
                if (this.shouldIgnoreResponse) {
                    resolve();
                    return;
                }
                
                const data = JSON.parse(event.data);
                
                if (data.type === 'audio') {
                    if (data.audio && !audioReceived) {
                        audioReceived = true;
                        console.log('🎵 [流式TTS] 收到句子音频，加入播放队列');
                        this.enqueueAudioChunk(data.audio);
                    }
                } else if (data.type === 'end') {
                    console.log('✅ [流式TTS] 句子TTS完成');
                    // 恢复原始onmessage处理器
                    this.ttsWebSocket.onmessage = originalOnMessage;
                    resolve();
                } else if (data.type === 'error') {
                    console.error('❌ [流式TTS] 句子TTS错误:', data.error);
                    this.ttsWebSocket.onmessage = originalOnMessage;
                    resolve();
                }
            };
            
            // 发送句子到TTS
            const message = {
                text: sentence,
                stream: false
            };
            this.ttsWebSocket.send(JSON.stringify(message));
        });
        
        // 等待当前句子完成后再处理下一个
        await sentencePromise;
        
        // 继续处理队列中的下一个句子
        if (this.streamingTTSSentenceQueue.length > 0 && !this.shouldIgnoreResponse) {
            this.processTTSSentenceQueue();
        } else {
            this.isProcessingTTSSentence = false;
        }
    }
    
    // 连接TTS WebSocket
    connectTTSWebSocket() {
        return new Promise((resolve, reject) => {
            if (this.ttsWebSocket && this.ttsWebSocket.readyState === WebSocket.OPEN) {
                resolve(); // 已连接
                return;
            }
            
            const ttsWebSocket = new WebSocket('ws://localhost:8000/ws/tts');
            this.ttsWebSocket = ttsWebSocket;
            
            ttsWebSocket.onopen = () => {
                console.log('✅ [流式TTS] WebSocket连接已建立');
                resolve();
            };
            
            // 默认的onmessage处理器（会被processTTSSentenceQueue中的临时处理器覆盖）
            ttsWebSocket.onmessage = (event) => {
                if (this.shouldIgnoreResponse) {
                    return;
                }
                
                const data = JSON.parse(event.data);
                
                if (data.type === 'audio') {
                    if (data.audio) {
                        this.enqueueAudioChunk(data.audio);
                    }
                } else if (data.type === 'end') {
                    console.log('🎵 [流式TTS] TTS完成');
                } else if (data.type === 'error') {
                    console.error('❌ [流式TTS] 错误:', data.error);
                }
            };
            
            ttsWebSocket.onerror = (error) => {
                console.error('❌ [流式TTS] WebSocket错误:', error);
                reject(error);
            };
            
            ttsWebSocket.onclose = (event) => {
                console.log('🔌 [流式TTS] WebSocket连接已关闭');
                if (this.ttsWebSocket === ttsWebSocket) {
                    this.ttsWebSocket = null;
                }
            };
        });
    }
    
    // 完成流式TTS：处理剩余的缓冲区文本
    finalizeStreamingTTS() {
        // 处理缓冲区中剩余的文本（即使没有句子结束符）
        if (this.streamingTTSBuffer && this.streamingTTSBuffer.trim()) {
            console.log('📤 [流式TTS] 发送剩余文本:', this.streamingTTSBuffer);
            this.sendSentenceToTTS(this.streamingTTSBuffer.trim());
            this.streamingTTSBuffer = '';
        }
        
        // 标记流式TTS完成（但继续处理队列中的句子）
        this.isStreamingTTSActive = false;
        this.streamingTTSRequestId = null;
        
        // 等待队列播放完成后关闭连接
        const checkAndClose = () => {
            if (!this.isProcessingTTSSentence && 
                this.streamingTTSSentenceQueue.length === 0 &&
                !this.isPlayingQueue && 
                this.audioQueue.length === 0) {
                if (this.ttsWebSocket) {
                    try {
                        this.ttsWebSocket.close();
                    } catch (e) {
                        console.error('关闭TTS连接失败:', e);
                    }
                    this.ttsWebSocket = null;
                }
                this.hideAudioControls();
            } else {
                // 如果还在处理，继续等待
                setTimeout(checkAndClose, 500);
            }
        };
        setTimeout(checkAndClose, 1000);
    }
    
    // 取消流式TTS
    cancelStreamingTTS() {
        console.log('🛑 [流式TTS] 取消流式TTS');
        this.streamingTTSBuffer = '';
        this.streamingTTSSentenceQueue = [];
        this.isStreamingTTSActive = false;
        this.isProcessingTTSSentence = false;
        this.streamingTTSRequestId = null;
        
        if (this.ttsWebSocket) {
            try {
                this.ttsWebSocket.close();
            } catch (e) {
                console.error('关闭TTS连接失败:', e);
            }
            this.ttsWebSocket = null;
        }
        
        this.stopAudio();
    }

    speakText(text, ttsRequestId = null) {
        // 如果提供了TTS请求ID，检查是否是最新的请求
        if (ttsRequestId && this.currentTTSRequestId && ttsRequestId !== this.currentTTSRequestId) {
            console.log('⏭️ [忽略TTS] 收到旧请求的TTS，忽略');
            return;
        }
        
        // 如果用户已停止生成，不播放语音
        if (this.shouldIgnoreResponse) {
            console.log('⏭️ [忽略] 用户已停止，不播放语音:', text.substring(0, 50) + '...');
            return;
        }
        
        // 停止当前正在播放的音频
        this.stopAudio();
        
        // 使用Qwen3-TTS实时语音合成
        this.speakWithQwenTTS(text);
    }

    async speakWithQwenTTS(text) {
        try {
            // 确保前一个TTS WebSocket连接完全关闭（停止旧的TTS）
            if (this.ttsWebSocket) {
                const oldSocket = this.ttsWebSocket;
                if (oldSocket.readyState === WebSocket.OPEN || oldSocket.readyState === WebSocket.CONNECTING) {
                    console.log('🔄 [TTS] 关闭前一个WebSocket连接（停止旧的TTS）...');
                    oldSocket.close();
                    // 等待连接关闭
                    await new Promise((resolve) => {
                        const checkClose = () => {
                            if (oldSocket.readyState === WebSocket.CLOSED) {
                                resolve();
                            } else {
                                setTimeout(checkClose, 50);
                            }
                        };
                        checkClose();
                    });
                }
                this.ttsWebSocket = null;
            }

            this.showAudioControls();
            this.audioQueue = [];
            this.isPlayingQueue = false;
            this.isQueueClosing = false;
            this.hasStreamedTTS = false;
            
            // 建立与Qwen3-TTS的WebSocket连接进行流式语音合成
            const ttsWebSocket = new WebSocket('ws://localhost:8000/ws/tts');
            this.ttsWebSocket = ttsWebSocket; // 保存引用以便后续管理
            
            ttsWebSocket.onopen = () => {
                console.log('✅ [TTS] WebSocket连接已建立');
                // 发送文本进行语音合成
                const message = {
                    text: text,
                    stream: true
                };
                console.log('📤 [TTS] 发送文本请求，长度:', text.length);
                ttsWebSocket.send(JSON.stringify(message));
            };

            ttsWebSocket.onmessage = (event) => {
                if (this.shouldIgnoreResponse) {
                    console.log('⏭️ [忽略] 用户已停止，丢弃TTS音频数据');
                    return;
                }
                const data = JSON.parse(event.data);
                console.log('🎵 [TTS] 收到消息:', data.type);
                
                if (data.type === 'audio_chunk') {
                    // 收到流式音频片段，加入播放队列
                    if (data.audio) {
                        this.hasStreamedTTS = true;
                        this.enqueueAudioChunk(data.audio);
                    }
                } else if (data.type === 'audio') {
                    // 仅在没有流式播放时才使用完整音频
                    if (this.hasStreamedTTS) {
                        console.log('🎵 [TTS] 已使用流式音频，忽略完整音频');
                        return;
                    }
                    console.log('🎵 [TTS] 收到完整音频数据，长度:', data.audio ? data.audio.length : 0);
                    this.playAudioChunk(data.audio);
                } else if (data.type === 'end') {
                    console.log('🎵 [TTS] 合成结束');
                    if (this.hasStreamedTTS) {
                        this.isQueueClosing = true;
                        if (ttsWebSocket.readyState === WebSocket.OPEN) {
                            ttsWebSocket.close();
                        }
                        if (!this.isPlayingQueue && (!this.audioQueue || this.audioQueue.length === 0)) {
                            this.isQueueClosing = false;
                            this.hasStreamedTTS = false;
                            this.hideAudioControls();
                        }
                    } else {
                        ttsWebSocket.close();
                        this.hideAudioControls();
                    }
                } else if (data.type === 'error') {
                    console.error('❌ [TTS] 错误:', data.error);
                    this.hideAudioControls();
                    ttsWebSocket.close();
                }
            };

            ttsWebSocket.onerror = (error) => {
                console.error('TTS WebSocket错误:', error);
                this.hideAudioControls();
            };

            ttsWebSocket.onclose = (event) => {
                console.log('🔌 [TTS] WebSocket连接已关闭', 'code:', event.code, 'reason:', event.reason);
                // 清理引用
                if (this.ttsWebSocket === ttsWebSocket) {
                    this.ttsWebSocket = null;
                }
            };

        } catch (error) {
            console.error('语音合成失败:', error);
            this.hideAudioControls();
            this.showError('语音合成失败');
        }
    }

    playAudioChunk(audioData) {
        if (this.shouldIgnoreResponse) {
            console.log('⏭️ [忽略] 已停止，跳过完整音频播放');
            return;
        }
        try {
            console.log('🎵 [播放] 开始播放完整音频，数据长度:', audioData ? audioData.length : 0);
            
            if (!audioData || audioData.length === 0) {
                console.error('❌ [播放] 音频数据为空');
                this.showError('音频数据为空，无法播放');
                return;
            }
            
            // 停止当前播放
            if (this.currentAudio) {
                try {
                    if (!this.currentAudio.paused) {
                this.currentAudio.pause();
                    }
                    this.currentAudio.currentTime = 0;
                    this.currentAudio = null;
                } catch (e) {
                    console.warn('⚠️ [播放] 停止旧音频失败:', e);
                }
            }
            
            // 创建音频对象
            const audioUrl = 'data:audio/wav;base64,' + audioData;
            console.log('🎵 [播放] 创建音频对象，URL长度:', audioUrl.length);
            
            const audio = new Audio(audioUrl);
            audio.volume = 1.0; // 设置为最大音量
            
            // 添加详细的事件监听
            audio.onloadstart = () => {
                console.log('🎵 [播放] 音频开始加载');
            };
            
            audio.onloadedmetadata = () => {
                console.log('🎵 [播放] 音频元数据加载完成，时长:', audio.duration, '秒');
            };
            
            audio.onloadeddata = () => {
                console.log('🎵 [播放] 音频数据加载完成');
            };
            
            audio.oncanplay = () => {
                console.log('🎵 [播放] 音频可以播放');
            };
            
            audio.oncanplaythrough = () => {
                console.log('🎵 [播放] 音频可以完整播放');
            };
            
            audio.onplay = () => {
                console.log('✅ [播放] 音频开始播放，当前时间:', audio.currentTime);
            };
            
            audio.onplaying = () => {
                console.log('🎵 [播放] 音频正在播放');
            };
            
            audio.onpause = () => {
                console.log('⏸️ [播放] 音频已暂停');
            };
            
            audio.onended = () => {
                console.log('🎵 [播放] 音频播放完成');
                this.currentAudio = null;
            };
            
            audio.onerror = (e) => {
                console.error('❌ [播放] 音频播放错误:', e);
                console.error('❌ [播放] 音频错误详情:', {
                    error: audio.error,
                    errorCode: audio.error ? audio.error.code : 'unknown',
                    errorMessage: audio.error ? audio.error.message : 'unknown',
                    readyState: audio.readyState,
                    networkState: audio.networkState
                });
                
                // 详细的错误信息
                let errorMsg = '音频播放失败';
                if (audio.error) {
                    switch (audio.error.code) {
                        case MediaError.MEDIA_ERR_ABORTED:
                            errorMsg = '音频播放被中止';
                            break;
                        case MediaError.MEDIA_ERR_NETWORK:
                            errorMsg = '网络错误，无法加载音频';
                            break;
                        case MediaError.MEDIA_ERR_DECODE:
                            errorMsg = '音频解码失败，请检查音频格式';
                            break;
                        case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
                            errorMsg = '不支持的音频格式';
                            break;
                        default:
                            errorMsg = `音频播放失败: ${audio.error.message || '未知错误'}`;
                    }
                }
                this.showError(errorMsg);
                this.currentAudio = null;
            };
            
            // 保存音频引用
            this.currentAudio = audio;
            
            // 尝试播放音频
            console.log('🎵 [播放] 尝试播放音频...');
            const playPromise = audio.play();
            
            if (playPromise !== undefined) {
                playPromise.then(() => {
                    console.log('✅ [播放] 音频播放成功');
                    console.log('🎵 [播放] 音频信息:', {
                        duration: audio.duration,
                        volume: audio.volume,
                        muted: audio.muted,
                        paused: audio.paused,
                        readyState: audio.readyState
                    });
                }).catch(error => {
                    console.error('❌ [播放] 音频播放失败:', error);
                    console.error('❌ [播放] 错误详情:', {
                        name: error.name,
                        message: error.message,
                        stack: error.stack
                    });
                    
                    // 检查是否是自动播放策略问题
                    if (error.name === 'NotAllowedError' || error.name === 'NotSupportedError') {
                        this.showError('浏览器阻止了自动播放，请点击播放按钮手动播放');
                        // 显示播放按钮
                        this.showAudioControls();
                    } else {
                        this.showError('音频播放失败: ' + error.message);
                    }
                    this.currentAudio = null;
                });
            } else {
                console.warn('⚠️ [播放] play()方法返回undefined，可能浏览器不支持');
            }
            
        } catch (error) {
            console.error('❌ [播放] 音频处理错误:', error);
            console.error('❌ [播放] 错误堆栈:', error.stack);
            this.showError('音频处理失败: ' + error.message);
            this.currentAudio = null;
        }
    }

    async playAudioStream() {
        if (!this.audioBufferQueue || this.audioBufferQueue.length === 0) {
            this.isPlayingStream = false;
            return;
        }
        
        this.isPlayingStream = true;
        
        while (this.audioBufferQueue.length > 0) {
            const audioBuffer = this.audioBufferQueue.shift();
            
            // 创建音频源
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            
            // 播放音频
            source.start();
            
            // 等待当前音频播放完成
            await new Promise((resolve) => {
                source.onended = resolve;
            });
        }
        
        this.isPlayingStream = false;
    }

    enqueueAudioChunk(audioData) {
        if (!audioData || this.shouldIgnoreResponse) {
            if (this.shouldIgnoreResponse) {
                console.log('⏭️ [忽略] 已停止，跳过新的音频片段');
            }
            return;
        }
        
        try {
            const audioUrl = 'data:audio/wav;base64,' + audioData;
            const audio = new Audio(audioUrl);
            audio.volume = 1.0;
            
            this.audioQueue.push(audio);
            console.log('🎵 [播放] 分段音频加入队列，当前长度:', this.audioQueue.length);
            
            if (!this.isPlayingQueue) {
                this.playAudioQueue();
            }
        } catch (error) {
            console.error('❌ [播放] 无法处理分段音频:', error);
        }
    }

    playAudioQueue() {
        if (this.shouldIgnoreResponse) {
            console.log('⏭️ [忽略] 已停止，清空播放队列');
            this.stopAudio();
            return;
        }
        if (!this.audioQueue || this.audioQueue.length === 0) {
            console.log('🎵 [播放] 队列为空，停止播放');
            this.isPlayingQueue = false;
            if (this.isQueueClosing) {
                console.log('🎵 [播放] 队列已播放完成，关闭音频控件');
                this.isQueueClosing = false;
                this.hasStreamedTTS = false;
                this.hideAudioControls();
            }
            return;
        }
        
        this.isPlayingQueue = true;
        const audio = this.audioQueue.shift();
        console.log('🎵 [播放] 播放队列中的音频，剩余:', this.audioQueue.length);
        
        // 播放当前音频
        audio.play().then(() => {
            console.log('✅ [播放] 音频播放成功');
        }).catch(error => {
            console.error('❌ [播放] 音频播放失败:', error);
            this.showError('音频播放失败，请检查音频格式或浏览器权限');
            this.isPlayingQueue = false;
            this.playAudioQueue();
        });
            
        // 当前音频播放完成后，播放下一个
        audio.onended = () => {
            console.log('🎵 [播放] 当前音频播放完成，播放下一个');
            this.isPlayingQueue = false;
            this.playAudioQueue();
        };
        
        // 保存当前音频引用（用于停止）
        this.currentAudio = audio;
    }

    speakTextChunk(text) {
        // 累积文本到缓冲区
        this.ttsBuffer += text;
        
        // 如果还没有开始流式TTS，则启动
        if (!this.isStreamingTTS) {
            this.startStreamingTTS();
        }
        
        // 发送文本块进行语音合成
        if (this.ttsWebSocket && this.ttsWebSocket.readyState === WebSocket.OPEN) {
            this.ttsWebSocket.send(JSON.stringify({
                type: 'chunk',
                text: text
            }));
        }
    }

    startStreamingTTS() {
        this.isStreamingTTS = true;
        this.ttsBuffer = '';
        
        // 建立与Qwen3-TTS的WebSocket连接
        this.ttsWebSocket = new WebSocket('ws://localhost:8000/ws/tts-stream');
        
        this.ttsWebSocket.onopen = () => {
            console.log('流式TTS WebSocket连接已建立');
            this.showAudioControls();
        };

        this.ttsWebSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'audio') {
                this.playAudioChunk(data.audio);
            } else if (data.type === 'error') {
                console.error('流式TTS错误:', data.error);
            }
        };

        this.ttsWebSocket.onerror = (error) => {
            console.error('流式TTS WebSocket错误:', error);
        };

        this.ttsWebSocket.onclose = () => {
            console.log('流式TTS WebSocket连接已关闭');
            this.isStreamingTTS = false;
        };
    }

    finalizeVoiceSynthesis() {
        if (this.ttsWebSocket && this.ttsWebSocket.readyState === WebSocket.OPEN) {
            // 发送结束信号
            this.ttsWebSocket.send(JSON.stringify({
                type: 'end'
            }));
        }
        
        // 延迟关闭，确保最后的音频播放完成
        setTimeout(() => {
            if (this.ttsWebSocket) {
                this.ttsWebSocket.close();
                this.ttsWebSocket = null;
            }
            this.isStreamingTTS = false;
            this.hideAudioControls();
        }, 2000);
    }

    pauseAudio() {
        if (this.currentAudio) {
            if (this.currentAudio.paused) {
                this.currentAudio.play();
                this.elements.pauseAudio.innerHTML = '<i class="fas fa-pause"></i>';
            } else {
                this.currentAudio.pause();
                this.elements.pauseAudio.innerHTML = '<i class="fas fa-play"></i>';
            }
        }
    }

    stopAudio() {
        // 停止当前播放的音频
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }
        
        // 停止Web Audio API播放
        if (this.audioSource) {
            try {
                this.audioSource.stop();
            } catch (e) {
                // 可能已经停止
            }
            this.audioSource = null;
        }
        
        // 清空音频缓冲区队列
        if (this.audioBufferQueue) {
            this.audioBufferQueue = [];
            this.isPlayingStream = false;
        }
        
        // 清空音频队列
        if (this.audioQueue) {
            this.audioQueue.forEach(audio => {
                audio.pause();
                audio.currentTime = 0;
            });
            this.audioQueue = [];
            this.isPlayingQueue = false;
        }
        
        // 关闭TTS WebSocket连接
        if (this.ttsWebSocket) {
            try {
                this.ttsWebSocket.close();
            } catch (e) {
                console.error('关闭TTS连接失败:', e);
            }
            this.ttsWebSocket = null;
        }
        
        this.hasStreamedTTS = false;
        this.isQueueClosing = false;
        this.hideAudioControls();
    }

    stopGeneration() {
        console.log('🛑 [停止] 用户点击停止按钮');
        
        // 立即设置标志，阻止后续的响应处理和语音播放
        this.shouldIgnoreResponse = true;
        
        // 立即停止所有音频播放和TTS连接
        this.stopAudio();
        
        // 关闭可能正在建立的TTS WebSocket连接
        if (this.ttsWebSocket) {
            try {
                if (this.ttsWebSocket.readyState === WebSocket.OPEN || this.ttsWebSocket.readyState === WebSocket.CONNECTING) {
                    console.log('🛑 [停止] 关闭TTS WebSocket连接');
                    this.ttsWebSocket.close();
                }
            } catch (e) {
                console.error('关闭TTS连接失败:', e);
            }
            this.ttsWebSocket = null;
        }
        
        // 检查是否正在生成文本
        const streamingMessage = this.elements.chatMessages.querySelector('.message[data-streaming="true"]');
        if (streamingMessage) {
            console.log('🛑 [停止] 检测到正在生成文本，停止生成');
            
            // 清理streaming消息
            const textDiv = streamingMessage.querySelector('.message-text');
            const currentText = streamingMessage.dataset.rawText || '';
            if (currentText) {
                // 如果有部分文本，保留并标记为已停止（不播放语音）
                textDiv.innerHTML = this.formatMessageContent(currentText) + '<span style="color: #999; font-size: 0.9em;"> (已停止)</span>';
            } else {
                // 如果没有文本，直接删除消息
                streamingMessage.remove();
            }
            delete streamingMessage.dataset.streaming;
            delete streamingMessage.dataset.rawText;
            
            // 隐藏打字指示器
            this.hideTypingIndicator();
        }
        
        // 等待下一次 response_start 再自动重置标志
        console.log('✅ [停止] 停止操作完成，已阻止文本生成和语音播放');
    }

    showAudioControls() {
        this.elements.audioControls.style.display = 'flex';
    }

    hideAudioControls() {
        this.elements.audioControls.style.display = 'none';
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showToast('已复制到剪贴板');
        }).catch(() => {
            this.showError('复制失败');
        });
    }

    formatMessageContent(text = '') {
        if (!text) return '';
        const escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
        return escaped.replace(/\n/g, '<br>');
    }

    clearChat() {
        if (confirm('确定要清空所有对话吗？')) {
            this.elements.chatMessages.innerHTML = '';
            this.chatHistory = [];
            this.showWelcomeMessage();
        }
    }

    showWelcomeMessage() {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <div class="welcome-content">
                <i class="fas fa-robot"></i>
                <h2>欢迎来到湖北博物馆智能问答系统</h2>
                <p>我是您的专属博物馆导游，可以为您介绍湖北博物馆的历史文化、展品信息、参观指南等。</p>
                <div class="suggested-questions">
                    <h3>您可以问我：</h3>
                    <div class="question-chips">
                        <button class="chip" data-question="湖北博物馆有哪些特色展品？">特色展品介绍</button>
                        <button class="chip" data-question="湖北博物馆的开放时间是什么？">开放时间</button>
                        <button class="chip" data-question="如何预约参观湖北博物馆？">参观预约</button>
                        <button class="chip" data-question="湖北博物馆的历史背景是什么？">历史背景</button>
                    </div>
                </div>
            </div>
        `;
        
        // 重新绑定建议问题的事件
        this.elements.chatMessages.appendChild(welcomeDiv);
        welcomeDiv.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                const question = e.target.getAttribute('data-question');
                this.elements.messageInput.value = question;
                this.sendMessage();
            });
        });
    }

    hideWelcomeMessage() {
        const welcomeMessage = this.elements.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
    }

    showTypingIndicator() {
        this.elements.typingIndicator.style.display = 'flex';
    }

    hideTypingIndicator() {
        this.elements.typingIndicator.style.display = 'none';
    }

    updateConnectionStatus(status) {
        const statusElement = this.elements.connectionStatus.querySelector('.status-indicator');
        statusElement.className = `status-indicator ${status}`;
        
        const statusText = {
            'connected': '已连接',
            'disconnected': '连接断开',
            'connecting': '连接中...'
        };
        
        statusElement.querySelector('span').textContent = statusText[status];
    }

    showLoading() {
        this.elements.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.elements.loadingOverlay.style.display = 'none';
    }

    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorModal.style.display = 'flex';
    }

    hideError() {
        this.elements.errorModal.style.display = 'none';
    }

    showToast(message) {
        // 简单的toast提示
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #4caf50;
            color: white;
            padding: 1rem 2rem;
            border-radius: 25px;
            z-index: 3000;
            animation: fadeInUp 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    updateCharCount() {
        const count = this.elements.messageInput.value.length;
        this.elements.charCount.textContent = `${count}/1000`;
        
        if (count > 800) {
            this.elements.charCount.style.color = '#f44336';
        } else if (count > 600) {
            this.elements.charCount.style.color = '#ff9800';
        } else {
            this.elements.charCount.style.color = '#666';
        }
    }

    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    async convertToWav(audioBlob) {
        return new Promise((resolve, reject) => {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const fileReader = new FileReader();
            
            fileReader.onload = async (e) => {
                try {
                    const arrayBuffer = e.target.result;
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    // 重采样到16kHz以匹配本地Paraformer
                    const resampledBuffer = await this.resampleAudioBuffer(audioContext, audioBuffer, 16000);
                    
                    // 转换为WAV格式
                    const wavBuffer = this.audioBufferToWav(resampledBuffer);
                    const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                    resolve(wavBlob);
                } catch (error) {
                    reject(error);
                }
            };
            
            fileReader.onerror = reject;
            fileReader.readAsArrayBuffer(audioBlob);
        });
    }

    audioBufferToPCM(audioBuffer) {
        /** 将AudioBuffer转换为PCM格式（16位，单声道） */
        const length = audioBuffer.length;
        const channelData = audioBuffer.getChannelData(0); // 获取第一个声道
        const pcmData = new Int16Array(length);
        
        for (let i = 0; i < length; i++) {
            // 将浮点数(-1.0 到 1.0)转换为16位整数
            const sample = Math.max(-1, Math.min(1, channelData[i]));
            pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        }
        
        return pcmData.buffer;
    }

    audioBufferToWav(buffer) {
        const length = buffer.length;
        const sampleRate = 16000; // 16kHz采样率，匹配本地Paraformer
        const arrayBuffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(arrayBuffer);
        
        // WAV文件头
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, length * 2, true);
        
        // 写入音频数据
        const channelData = buffer.getChannelData(0);
        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, channelData[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
        
        return arrayBuffer;
    }

    async resampleAudioBuffer(audioContext, audioBuffer, targetSampleRate) {
        const sourceSampleRate = audioBuffer.sampleRate;
        const targetLength = Math.floor(audioBuffer.length * targetSampleRate / sourceSampleRate);
        
        // 创建新的AudioBuffer
        const resampledBuffer = audioContext.createBuffer(
            audioBuffer.numberOfChannels,
            targetLength,
            targetSampleRate
        );
        
        // 简单的线性插值重采样
        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            const sourceData = audioBuffer.getChannelData(channel);
            const targetData = resampledBuffer.getChannelData(channel);
            
            for (let i = 0; i < targetLength; i++) {
                const sourceIndex = (i * sourceSampleRate) / targetSampleRate;
                const index = Math.floor(sourceIndex);
                const fraction = sourceIndex - index;
                
                if (index + 1 < sourceData.length) {
                    // 线性插值
                    targetData[i] = sourceData[index] * (1 - fraction) + sourceData[index + 1] * fraction;
                } else {
                    targetData[i] = sourceData[index] || 0;
                }
            }
        }
        
        return resampledBuffer;
    }

    async sendAudioData(audioBlob) {
        try {
            console.log('📦 [DEBUG] sendAudioData被调用，音频大小:', audioBlob.size, '字节');
            
            // 检查WebSocket连接状态
            if (!this.realtimeRecognition || this.realtimeRecognition.readyState !== WebSocket.OPEN) {
                console.error('❌ [DEBUG] WebSocket连接未就绪，无法发送音频数据');
                console.error('❌ [DEBUG] WebSocket状态:', this.realtimeRecognition ? this.realtimeRecognition.readyState : 'null');
                this.showError('语音识别连接已断开，请重新点击麦克风');
                return;
            }
            
            console.log('🔄 [DEBUG] 开始转换音频格式（webm -> WAV）...');
            // 将音频数据转换为WAV格式并发送
            const wavBlob = await this.convertToWav(audioBlob);
            console.log('✅ [DEBUG] 音频转换完成，WAV大小:', wavBlob.size, '字节');
            
            const reader = new FileReader();
            
            reader.onload = () => {
                const base64Audio = reader.result.split(',')[1];
                console.log('✅ [DEBUG] Base64编码完成，长度:', base64Audio.length, '字符');
                
                if (this.realtimeRecognition && this.realtimeRecognition.readyState === WebSocket.OPEN) {
                    try {
                        const message = JSON.stringify({
                            type: 'audio',
                            audio: base64Audio
                        });
                        console.log('📤 [DEBUG] 发送WebSocket消息，大小:', message.length, '字符');
                        this.realtimeRecognition.send(message);
                        console.log('✅ [DEBUG] 音频数据已成功发送');
                        
                        // 发送结束信号
                        console.log('📤 [DEBUG] 发送结束信号...');
                        this.realtimeRecognition.send(JSON.stringify({
                            type: 'end'
                        }));
                        console.log('✅ [DEBUG] 结束信号已发送');
                    } catch (error) {
                        console.error('❌ [DEBUG] 发送音频数据失败:', error);
                        this.showError('发送音频数据失败：' + error.message);
                    }
                } else {
                    console.error('❌ [DEBUG] WebSocket在发送时断开');
                    this.showError('语音识别连接已断开');
                }
            };
            
            reader.onerror = (error) => {
                console.error('❌ [DEBUG] 读取音频文件失败:', error);
                this.showError('读取音频文件失败：' + error.message);
            };
            
            reader.readAsDataURL(wavBlob);
        } catch (error) {
            console.error('❌ [DEBUG] 处理音频数据失败:', error);
            this.showError('处理音频数据失败：' + error.message);
        }
    }
}

// 初始化应用
const app = new MuseumChatApp();

// 页面加载完成后初始化语音
window.addEventListener('load', () => {
    // 等待语音API加载
    if ('speechSynthesis' in window) {
        speechSynthesis.getVoices();
    }
});

// 处理页面可见性变化
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // 页面隐藏时暂停语音
        if (speechSynthesis.speaking) {
            speechSynthesis.pause();
        }
    }
});
