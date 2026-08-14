from __future__ import annotations

import asyncio
import json
import random
import threading
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Awaitable, TypeVar

import httpx

from app.core.config import TranslationNodeConfig, get_settings, normalize_output_voice_language
from app.core.logging import get_logger


logger = get_logger(__name__)
_T = TypeVar("_T")


TARGET_LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "英文",
    "ja": "日语",
}
DEEPL_LANGUAGE_CODES = {
    "zh": "ZH",
    "en": "EN",
    "ja": "JA",
}
GOOGLE_LANGUAGE_CODES = {
    "zh": "zh-CN",
    "en": "en",
    "ja": "ja",
}
BAIDU_LANGUAGE_CODES = {
    "zh": "zh",
    "en": "en",
    "ja": "jp",
}
BAIDU_ERROR_MESSAGES = {
    "52000": "请求成功。",
    "52001": "请求超时，请稍后重试。",
    "52002": "系统错误，请稍后重试。",
    "52003": "未授权用户，请检查 APP ID 是否正确。",
    "54000": "缺少必填参数，请检查请求内容。",
    "54001": "签名错误，请检查 APP ID、密钥和签名逻辑。",
    "54003": "访问频率受限，请稍后再试。",
    "54004": "账户余额不足，请检查百度翻译账户额度。",
    "54005": "长 query 请求频繁，请降低请求频率。",
    "58000": "客户端 IP 非法，请检查百度翻译开放平台的 IP 白名单。",
    "58001": "译文语言方向不支持，请检查源语言和目标语言。",
    "58002": "服务当前关闭，请前往百度翻译开放平台开启服务。",
    "90107": "认证失败或参数格式错误，请检查账号配置。",
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

    async def _translate_via_stream_async(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> str:
        chunks: list[str] = []
        saw_stream_chunk = False
        timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/event-stream" not in content_type:
                    raw_bytes = await response.aread()
                    data = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
                    content = self._extract_text_from_openai_payload(data).strip()
                    if content:
                        return content
                    raise RuntimeError("翻译节点返回了空结果。")

                async for raw_line in response.aiter_lines():
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

    @staticmethod
    def _resolve_deepl_language_code(language: str | None) -> str | None:
        if not language:
            return None
        return DEEPL_LANGUAGE_CODES.get(normalize_output_voice_language(language))

    @staticmethod
    def _resolve_google_language_code(language: str | None) -> str | None:
        if not language:
            return None
        return GOOGLE_LANGUAGE_CODES.get(normalize_output_voice_language(language))

    def _translate_via_deepl_sync(
        self,
        *,
        node: TranslationNodeConfig,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        try:
            import deepl
        except Exception as exc:
            raise RuntimeError("DeepL SDK 未安装，请先同步项目依赖。") from exc

        auth_key = node.api_key.strip()
        if not auth_key:
            raise RuntimeError("DeepL 节点缺少 API Key。")
        translator = deepl.Translator(
            auth_key,
            server_url=node.base_url.strip() or None,
        )
        target_code = self._resolve_deepl_language_code(target_language)
        if not target_code:
            raise RuntimeError(f"DeepL 暂不支持目标语言：{target_language}")
        translate_kwargs: dict[str, object] = {"target_lang": target_code}
        source_code = self._resolve_deepl_language_code(source_language)
        if source_code:
            translate_kwargs["source_lang"] = source_code
        model_type = node.model.strip()
        if model_type and model_type.casefold() not in {"default", "auto"}:
            translate_kwargs["model_type"] = model_type
        result = translator.translate_text(text, **translate_kwargs)
        translated = str(getattr(result, "text", result) or "").strip()
        if not translated:
            raise RuntimeError("DeepL 返回了空结果。")
        return translated

    async def _translate_via_deepl_async(
        self,
        *,
        node: TranslationNodeConfig,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        return await asyncio.to_thread(
            self._translate_via_deepl_sync,
            node=node,
            text=text,
            source_language=source_language,
            target_language=target_language,
        )

    @staticmethod
    def _load_google_credentials(credentials_value: str):
        try:
            from google.oauth2 import service_account
        except Exception as exc:
            raise RuntimeError("Google Cloud 认证依赖未安装完整，请先同步项目依赖。") from exc

        raw = credentials_value.strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return service_account.Credentials.from_service_account_file(str(candidate))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Google Cloud 凭证需要填写服务账号 JSON 路径或完整 JSON 内容。") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Google Cloud 凭证 JSON 格式无效。")
        return service_account.Credentials.from_service_account_info(data)

    def _translate_via_google_cloud_sync(
        self,
        *,
        node: TranslationNodeConfig,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        try:
            from google.cloud import translate_v3 as translate
        except Exception as exc:
            raise RuntimeError("google-cloud-translate SDK 未安装，请先同步项目依赖。") from exc

        project_id = node.base_url.strip()
        if not project_id:
            raise RuntimeError("Google Cloud Translate 节点缺少 Project ID。")
        location = node.model.strip() or "global"
        credentials = self._load_google_credentials(node.api_key)
        client = (
            translate.TranslationServiceClient(credentials=credentials)
            if credentials is not None
            else translate.TranslationServiceClient()
        )
        target_code = self._resolve_google_language_code(target_language)
        if not target_code:
            raise RuntimeError(f"Google Cloud Translate 暂不支持目标语言：{target_language}")
        request_args: dict[str, object] = {
            "parent": f"projects/{project_id}/locations/{location}",
            "contents": [text],
            "mime_type": "text/plain",
            "target_language_code": target_code,
        }
        source_code = self._resolve_google_language_code(source_language)
        if source_code:
            request_args["source_language_code"] = source_code
        response = client.translate_text(request=request_args)
        translations = getattr(response, "translations", None) or []
        translated = "".join(
            str(getattr(item, "translated_text", "") or "")
            for item in translations
        ).strip()
        if not translated:
            raise RuntimeError("Google Cloud Translate 返回了空结果。")
        return translated

    async def _translate_via_google_cloud_async(
        self,
        *,
        node: TranslationNodeConfig,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        return await asyncio.to_thread(
            self._translate_via_google_cloud_sync,
            node=node,
            text=text,
            source_language=source_language,
            target_language=target_language,
        )

    @staticmethod
    def _resolve_baidu_language_code(language: str | None) -> str | None:
        if not language:
            return None
        return BAIDU_LANGUAGE_CODES.get(normalize_output_voice_language(language))

    async def _translate_via_baidu_async(
        self,
        *,
        node: TranslationNodeConfig,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        app_id = node.base_url.strip()
        if not app_id:
            raise RuntimeError("百度翻译节点缺少 APP ID。")
        app_key = node.api_key.strip()
        if not app_key:
            raise RuntimeError("百度翻译节点缺少密钥。")
        endpoint = node.model.strip() or "https://api.fanyi.baidu.com/api/trans/vip/translate"
        target_code = self._resolve_baidu_language_code(target_language)
        if not target_code:
            raise RuntimeError(f"百度翻译暂不支持目标语言：{target_language}")
        source_code = self._resolve_baidu_language_code(source_language) or "auto"
        salt = random.randint(32768, 65536)
        sign = md5(f"{app_id}{text}{salt}{app_key}".encode("utf-8")).hexdigest()
        payload = {
            "appid": app_id,
            "q": text,
            "from": source_code,
            "to": target_code,
            "salt": salt,
            "sign": sign,
        }
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=20.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("百度翻译返回了无效响应。")
        error_code = str(data.get("error_code") or "").strip()
        if error_code:
            error_message = BAIDU_ERROR_MESSAGES.get(error_code) or str(data.get("error_msg") or "未知错误").strip()
            raise RuntimeError(f"百度翻译调用失败：{error_code} {error_message}".strip())
        trans_result = data.get("trans_result")
        if not isinstance(trans_result, list) or not trans_result:
            raise RuntimeError("百度翻译返回了空结果。")
        translated = "\n".join(
            str(item.get("dst") or "").strip()
            for item in trans_result
            if isinstance(item, dict)
        ).strip()
        if not translated:
            raise RuntimeError("百度翻译返回了空结果。")
        return translated

    async def _translate_via_openai_compatible_async(
        self,
        *,
        node: TranslationNodeConfig,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        url = self._normalize_base_url(node.base_url)
        headers = {
            "Content-Type": "application/json",
        }
        if node.api_key:
            headers["Authorization"] = f"Bearer {node.api_key}"
        payload = {
            "model": node.model,
            "messages": self._build_prompt(
                text,
                source_language=source_language,
                target_language=target_language,
            ),
            "temperature": 0.1,
            "stream": True,
        }
        return await self._translate_via_stream_async(
            url=url,
            headers=headers,
            payload=payload,
        )

    @staticmethod
    def _run_async_blocking(coroutine: Awaitable[_T]) -> _T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        result: dict[str, object] = {}
        error: dict[str, BaseException] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coroutine)
            except BaseException as exc:  # pragma: no cover - passthrough
                error["value"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if "value" in error:
            raise error["value"]
        return result["value"]  # type: ignore[return-value]

    async def translate_async(
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
            raise RuntimeError("未找到可用的翻译节点，请先配置百度翻译、DeepL、Google Cloud 或 OpenAI 兼容节点。")

        logger.info(
            "正在调用翻译节点: node={} provider={} target={} source_guess={}",
            node.name,
            node.provider,
            normalized_target,
            effective_source or "unknown",
        )
        match node.provider:
            case "baidu-translate":
                translated = await self._translate_via_baidu_async(
                    node=node,
                    text=stripped,
                    source_language=effective_source,
                    target_language=normalized_target,
                )
            case "deepl":
                translated = await self._translate_via_deepl_async(
                    node=node,
                    text=stripped,
                    source_language=effective_source,
                    target_language=normalized_target,
                )
            case "google-cloud-translate":
                translated = await self._translate_via_google_cloud_async(
                    node=node,
                    text=stripped,
                    source_language=effective_source,
                    target_language=normalized_target,
                )
            case _:
                translated = await self._translate_via_openai_compatible_async(
                    node=node,
                    text=stripped,
                    source_language=effective_source,
                    target_language=normalized_target,
                )
        return TranslationResult(
            text=translated,
            source_language=effective_source,
            target_language=normalized_target,
            node_name=node.name,
        )

    def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None = None,
        node_id: str | None = None,
    ) -> TranslationResult:
        return self._run_async_blocking(
            self.translate_async(
                text,
                target_language=target_language,
                source_language=source_language,
                node_id=node_id,
            )
        )
