"""脱敏处理钩子实现。"""

import re
import uuid
from collections import OrderedDict
from typing import Dict, Any, Union
from ..engine import BeforeRunHook, AfterRunHook, HookResult
from ..model_client import RunParams
from ..message import Message, ModelResponse, StreamingModelResponse


class DesensitizationHook(BeforeRunHook, AfterRunHook):
    """脱敏处理钩子，支持对敏感数据进行脱敏和反脱敏。
    
    支持的脱敏类型：
    - 手机号码
    - 身份证号
    - 银行卡号
    - 邮箱地址
    - 自定义正则表达式
    """
    
    def __init__(self, 
                 enable_phone: bool = True,
                 enable_id_card: bool = True, 
                 enable_bank_card: bool = True,
                 enable_email: bool = True,
                 custom_patterns: Dict[str, str] = None):
        """初始化脱敏钩子。
        
        Args:
            enable_phone: 是否启用手机号脱敏
            enable_id_card: 是否启用身份证号脱敏
            enable_bank_card: 是否启用银行卡号脱敏
            enable_email: 是否启用邮箱脱敏
            custom_patterns: 自定义脱敏模式，格式为 {名称: 正则表达式}
        """
        self.enable_phone = enable_phone
        self.enable_id_card = enable_id_card
        self.enable_bank_card = enable_bank_card
        self.enable_email = enable_email
        self.custom_patterns = custom_patterns or {}
        self._last_metadata = {}  # 保存最后的元数据用于trace
        self._streaming_buffer = ""  # 流式响应缓冲区
        self._max_placeholder_length = 100  # 最大占位符长度
        
        # 预定义的脱敏模式 - 使用OrderedDict确保处理顺序
        self.patterns = OrderedDict()
        # 身份证优先处理，避免被银行卡号误匹配
        if enable_id_card:
            # 修复身份证匹配，避免误匹配QQ号等短数字
            # 严格匹配18位身份证格式：前17位数字 + 最后一位数字或X/x
            # 使用负向前瞻和后瞻，兼容中文字符边界
            self.patterns['id_card'] = r'(?<!\d)\d{17}[\dXx](?!\d)'
        if enable_phone:
            self.patterns['phone'] = r'1[3-9]\d{9}'
        if enable_email:
            self.patterns['email'] = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        if enable_bank_card:
            self.patterns['bank_card'] = r'(?<!\d)\d{16,19}(?!\d)'
            
        # 添加自定义模式
        self.patterns.update(self.custom_patterns)
    
    def _desensitize_text(self, text: str) -> tuple[str, Dict[str, str]]:
        """对文本进行脱敏处理。
        
        Args:
            text: 原始文本
            
        Returns:
            tuple: (脱敏后文本, 映射关系)
        """
        mapping = {}
        result = text
        
        # 按顺序处理每种模式，避免重复匹配
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, result)
            for match in matches:
                # 检查该匹配是否已经被处理（被占位符替换）
                if match in result and not any(placeholder in match for placeholder in mapping.keys()):
                    # 生成占位符
                    placeholder = f"§{uuid.uuid4().hex[:8]}§"
                    mapping[placeholder] = match
                    result = result.replace(match, placeholder)
        
        return result, mapping
    
    def _recover_text(self, text: str, mapping: Dict[str, str]) -> str:
        """从脱敏文本恢复原始数据。
        
        Args:
            text: 脱敏后的文本
            mapping: 映射关系
            
        Returns:
            str: 恢复后的文本
        """
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result
    
    def _process_messages(self, messages: list[Message]) -> tuple[list[Message], Dict[str, str]]:
        """处理消息列表进行脱敏。
        
        Args:
            messages: 原始消息列表
            
        Returns:
            tuple: (脱敏后消息列表, 映射关系)
        """
        processed_messages = []
        combined_mapping = {}
        
        for msg in messages:
            new_msg = msg.model_copy()
            if msg.content:
                if isinstance(msg.content, str):
                    desensitized_content, mapping = self._desensitize_text(msg.content)
                    new_msg.content = desensitized_content
                    combined_mapping.update(mapping)
                elif isinstance(msg.content, list):
                    # 处理多模态内容
                    new_content = []
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_content = item.get('text', '')
                            desensitized_text, mapping = self._desensitize_text(text_content)
                            new_item = item.copy()
                            new_item['text'] = desensitized_text
                            new_content.append(new_item)
                            combined_mapping.update(mapping)
                        else:
                            new_content.append(item)
                    new_msg.content = new_content
            
            processed_messages.append(new_msg)
        
        return processed_messages, combined_mapping
    
    def process(self, params: RunParams) -> HookResult:
        """同步处理运行参数进行脱敏。"""
        processed_messages, mapping = self._process_messages(params.messages)
        
        # 创建新的RunParams对象
        new_params = params.model_copy()
        new_params.messages = processed_messages
        
        # 保存元数据用于trace
        metadata = {'desensitization_mapping': mapping}
        self._last_metadata = metadata
        
        return HookResult(
            data=new_params,
            metadata=metadata
        )
    
    async def aprocess(self, params: RunParams) -> HookResult:
        """异步处理运行参数进行脱敏。"""
        # 脱敏是CPU密集型操作，这里直接调用同步方法
        return self.process(params)
    
    def _process_streaming_chunk(self, chunk: str, mapping: Dict[str, str]) -> str:
        """实时处理流式响应块，立即恢复占位符而不等待。
        
        基于滑动缓冲区的实时算法：
        1. 立即替换完整的占位符
        2. 只保留最长占位符长度-1的安全缓冲
        3. 其余内容立即输出
        
        Args:
            chunk: 当前响应块
            mapping: 脱敏映射关系
            
        Returns:
            str: 可以立即输出的恢复内容
        """
        if not mapping:
            return chunk
        
        # 将新块添加到缓冲区
        self._streaming_buffer += chunk
        output = ""
        
        while True:
            # 1) 查找并替换完整的占位符
            found_placeholder = None
            for placeholder in mapping.keys():
                if placeholder in self._streaming_buffer:
                    found_placeholder = placeholder
                    break
            
            if found_placeholder:
                # 找到完整占位符，立即处理
                before, _, after = self._streaming_buffer.partition(found_placeholder)
                output += before  # 输出占位符前的内容
                output += mapping[found_placeholder]  # 输出恢复的原始内容
                self._streaming_buffer = after  # 保留占位符后的内容
                continue  # 继续查找其他占位符
            
            # 2) 没有完整占位符，实现滑动窗口策略
            max_placeholder_len = max(len(p) for p in mapping.keys()) if mapping else 0
            safety_buffer_size = max_placeholder_len - 1  # 滑动缓冲区大小
            
            if len(self._streaming_buffer) > safety_buffer_size:
                # 输出安全部分，保留安全缓冲区
                safe_output_length = len(self._streaming_buffer) - safety_buffer_size
                output += self._streaming_buffer[:safe_output_length]
                self._streaming_buffer = self._streaming_buffer[safe_output_length:]
            
            break  # 没有更多完整占位符可处理
        
        return output
    
    def _flush_streaming_buffer(self, mapping: Dict[str, str]) -> str:
        """刷新流式缓冲区的剩余内容。
        
        Args:
            mapping: 脱敏映射关系
            
        Returns:
            str: 剩余的恢复内容
        """
        if not self._streaming_buffer:
            return ""
        
        # 恢复剩余内容中的占位符
        result = self._streaming_buffer
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        
        self._streaming_buffer = ""
        return result

    def process_response(self, response: Union[ModelResponse, StreamingModelResponse], 
                         hook_metadata: Dict[str, Any]) -> HookResult:
        """同步处理响应进行反脱敏。"""
        mapping = hook_metadata.get('desensitization_mapping', {})
        if not mapping:
            return HookResult(data=response)
        
        new_response = response.model_copy()
        
        # 处理不同类型的响应
        if hasattr(new_response, 'content') and new_response.content:
            if isinstance(new_response.content, str):
                new_response.content = self._recover_text(new_response.content, mapping)
            elif isinstance(new_response.content, list):
                # 处理多模态响应内容
                recovered_content = []
                for item in new_response.content:
                    if isinstance(item, dict) and 'text' in item:
                        new_item = item.copy()
                        new_item['text'] = self._recover_text(item['text'], mapping)
                        recovered_content.append(new_item)
                    else:
                        recovered_content.append(item)
                new_response.content = recovered_content
        
        # 处理choices字段（如果存在）
        if hasattr(new_response, 'choices') and new_response.choices:
            for choice in new_response.choices:
                if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'content'):
                    if choice.message.content:
                        choice.message.content = self._recover_text(choice.message.content, mapping)
                elif hasattr(choice, 'delta') and choice.delta and hasattr(choice.delta, 'content'):
                    if choice.delta.content:
                        # 对于流式响应，使用缓冲处理
                        recovered_chunk = self._process_streaming_chunk(choice.delta.content, mapping)
                        choice.delta.content = recovered_chunk
        
        return HookResult(data=new_response)
    
    async def aprocess_response(self, response: Union[ModelResponse, StreamingModelResponse], 
                               hook_metadata: Dict[str, Any]) -> HookResult:
        """异步处理响应进行反脱敏。"""
        # 反脱敏是CPU密集型操作，这里直接调用同步方法
        return self.process_response(response, hook_metadata)
