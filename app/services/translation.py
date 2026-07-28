from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from app.core.config import TranslationNodeConfig, get_settings, normalize_output_voice_language
from app.core.logging import get_logger


logger = get_logger(__name__)


TARGET_LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "英文",
    "ja": "日语",
}


@dataclass(frozen=True, slots=True)
class TranslationResult:
    text: str
    source_language: str | None
    target_language: str
    node_name: str


def detect_text_language(text: str) -> str | None:
    content = (text or "").strip()
    if not content:
        return None
    has_hiragana = any("\u3040" <= ch <= "\u309f" for ch in content)
    has_katakana = any("\u30a0" <= ch <= "\u30ff" for ch in content)
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in content)
    has_latin = any(("a" <= ch.lower() <= "z") for ch in content if ch.isascii())
    if has_hiragana or has_katakana:
        return "ja"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return None


class TranslationService:
    def _get_node(self, node_id: str | None = None) -> TranslationNodeConfig | None:
        settings = get_settings()
        selected_id = node_id or settings.translation_node_id
        if not selected_id:
            return None
        for node in settings.translation_nodes:
            if node.node_id == selected_id:
                return node
        return None

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        cleaned = base_url.strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        if cleaned.endswith("/v1"):
            return f"{cleaned}/chat/completions"
        return f"{cleaned}/chat/completions"

    @staticmethod
    def _build_prompt(text: str, *, source_language: str | None, target_language: str) -> list[dict[str, str]]:
        target_label = TARGET_LANGUAGE_LABELS.get(target_language, "目标语言")
        source_label = TARGET_LANGUAGE_LABELS.get(source_language, source_language) if source_language else "未知"
        system_prompt = (
            "你是一个专门给 TTS 使用的高质量翻译引擎。"
            f"请把用户提供的文本翻译成自然、可直接朗读的{target_label}。"
            "你的目标不是逐字直译，而是在忠实原意、语气和信息的前提下，让译文听起来像母语者会自然说出来的话。"
            "严格遵守以下规则："
            "1. 只输出最终译文，不要解释，不要分析，不要加引号，不要加前缀。"
            "2. 保留原文的语气、情绪、敬语强弱、口语感和说话风格。"
            "3. 专有名词、品牌名、人名、地名、游戏名、产品名优先保留原文或采用最常见的标准译法。"
            "4. 数字、时间、百分比、单位、金额、链接、邮箱、代码标识尽量保持原样，除非目标语言必须转换表达。"
            "5. 保留换行、分句结构和大部分标点；但可以为了朗读自然度做轻微调整。"
            "6. 不要擅自补充原文没有的信息，不要总结，不要省略关键信息。"
            "7. 如果原文已经是目标语言，直接输出经过轻微润色后的原文；如果不需要润色，就原样输出。"
            "8. 如果原文混合多种语言，优先翻译主要内容到目标语言，同时保留必要的外文专有词。"
        )
        user_prompt = (
            f"源语言判断：{source_label}\n"
            f"目标语言：{target_label}\n"
            "请直接给出适合语音朗读的最终译文。\n"
            "原文如下：\n"
            f"{text}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_text_from_openai_payload(data: object) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = first.get("text")
        if isinstance(text, str):
            return text
        return ""

    def _translate_via_stream(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> str:
        chunks: list[str] = []
        saw_stream_chunk = False
        timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/event-stream" not in content_type:
                    data = response.json()
                    content = self._extract_text_from_openai_payload(data).strip()
                    if content:
                        return content
                    raise RuntimeError("翻译节点返回了空结果。")

                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    piece = self._extract_text_from_openai_payload(data)
                    if piece:
                        saw_stream_chunk = True
                        chunks.append(piece)
        translated = "".join(chunks).strip()
        if translated:
            return translated
        if saw_stream_chunk:
            return translated
        raise RuntimeError("翻译节点返回了空结果。")

    def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None = None,
        node_id: str | None = None,
    ) -> TranslationResult:
        normalized_target = normalize_output_voice_language(target_language)
        stripped = text.strip()
        if not stripped:
            return TranslationResult(
                text="",
                source_language=source_language,
                target_language=normalized_target,
                node_name="",
            )
        effective_source = source_language or detect_text_language(stripped)
        if effective_source == normalized_target:
            return TranslationResult(
                text=stripped,
                source_language=effective_source,
                target_language=normalized_target,
                node_name="",
            )
        node = self._get_node(node_id)
        if node is None:
            raise RuntimeError("未找到可用的翻译节点，请先在快速设置中配置 OpenAI / DeepSeek 节点。")

        url = self._normalize_base_url(node.base_url)
        headers = {
            "Content-Type": "application/json",
        }
        if node.api_key:
            headers["Authorization"] = f"Bearer {node.api_key}"
        payload = {
            "model": node.model,
            "messages": self._build_prompt(
                stripped,
                source_language=effective_source,
                target_language=normalized_target,
            ),
            "temperature": 0.1,
            "stream": True,
        }
        logger.info(
            "正在调用翻译节点: node={} provider={} target={} source_guess={}",
            node.name,
            node.provider,
            normalized_target,
            effective_source or "unknown",
        )
        translated = self._translate_via_stream(
            url=url,
            headers=headers,
            payload=payload,
        )
        return TranslationResult(
            text=translated,
            source_language=effective_source,
            target_language=normalized_target,
            node_name=node.name,
        )
