from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import datasets


TNEWS_CHOICES = [
    "news_story",
    "news_culture",
    "news_entertainment",
    "news_sports",
    "news_finance",
    "news_house",
    "news_car",
    "news_edu",
    "news_tech",
    "news_military",
    "news_travel",
    "news_world",
    "news_stock",
    "news_agriculture",
    "news_game",
]

IFLYTEK_CHOICES = [
    "打车",
    "地图导航",
    "免费WIFI",
    "租车",
    "同城服务",
    "快递物流",
    "婚庆",
    "家政",
    "公共交通",
    "政务",
    "社区服务",
    "薅羊毛",
    "魔幻",
    "仙侠",
    "卡牌",
    "飞行空战",
    "射击游戏",
    "休闲益智",
    "动作类",
    "体育竞技",
    "棋牌中心",
    "经营养成",
    "策略",
    "MOBA",
    "辅助工具",
    "约会社交",
    "即时通讯",
    "工作社交",
    "论坛圈子",
    "婚恋社交",
    "情侣社交",
    "社交工具",
    "生活社交",
    "微博博客",
    "新闻",
    "漫画",
    "小说",
    "技术",
    "教辅",
    "问答交流",
    "搞笑",
    "杂志",
    "百科",
    "影视娱乐",
    "求职",
    "兼职",
    "视频",
    "短视频",
    "音乐",
    "直播",
    "电台",
    "K歌",
    "成人",
    "中小学",
    "职考",
    "公务员",
    "英语",
    "视频教育",
    "高等教育",
    "成人教育",
    "艺术",
    "语言(非英语)",
    "旅游资讯",
    "综合预定",
    "民航",
    "铁路",
    "酒店",
    "行程管理",
    "民宿短租",
    "出国",
    "工具",
    "亲子儿童",
    "母婴",
    "驾校",
    "违章",
    "汽车咨询",
    "汽车交易",
    "日常养车",
    "行车辅助",
    "租房",
    "买房",
    "装修家居",
    "电子产品",
    "问诊挂号",
    "养生保健",
    "医疗服务",
    "减肥瘦身",
    "美妆美业",
    "菜谱",
    "餐饮店",
    "体育咨讯",
    "运动健身",
    "支付",
    "保险",
    "股票",
    "借贷",
    "理财",
    "彩票",
    "记账",
    "银行",
    "美颜",
    "影像剪辑",
    "摄影修图",
    "相机",
    "绘画",
    "二手",
    "电商",
    "团购",
    "外卖",
    "电影票务",
    "社区超市",
    "购物咨询",
    "笔记",
    "办公",
    "日程管理",
    "女性",
    "经营",
    "收款",
    "其他",
]

_IDIOM_PATTERN = re.compile(r"#idiom\d+#")
_ANSWER_PREFIX = re.compile(r"^(?:答案|回答)\s*[:：]\s*")
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PUNCTUATION = set(
    string.punctuation
    + "，。！？；：、‘’“”《》〈〉【】〔〕（）()［］{}﹃﹄「」『』…·—～"
)


def afqmc_doc_to_text(doc: Mapping[str, Any]) -> str:
    return (
        "判断下面两个句子的含义是否相同。\n"
        f"句子一：{doc['sentence1']}\n"
        f"句子二：{doc['sentence2']}\n"
        "答案："
    )


def afqmc_doc_to_choice(_doc: Mapping[str, Any]) -> list[str]:
    return ["不同", "相同"]


def tnews_doc_to_text(doc: Mapping[str, Any]) -> str:
    return f"判断下面新闻标题的类别。\n新闻标题：{doc['sentence']}\n类别："


def tnews_doc_to_choice(_doc: Mapping[str, Any]) -> list[str]:
    return TNEWS_CHOICES


def iflytek_doc_to_text(doc: Mapping[str, Any]) -> str:
    return f"判断下面应用描述所属的类别。\n应用描述：{doc['sentence']}\n类别："


def iflytek_doc_to_choice(_doc: Mapping[str, Any]) -> list[str]:
    return IFLYTEK_CHOICES


def nli_doc_to_text(doc: Mapping[str, Any]) -> str:
    return (
        "判断句子二与句子一之间的关系。\n"
        f"句子一：{doc['sentence1']}\n"
        f"句子二：{doc['sentence2']}\n"
        "关系："
    )


def nli_doc_to_choice(_doc: Mapping[str, Any]) -> list[str]:
    # ClassLabel order: neutral, entailment, contradiction.
    return ["中立", "蕴含", "矛盾"]


def wsc_doc_to_text(doc: Mapping[str, Any]) -> str:
    target = doc["target"]
    return (
        f"句子：{doc['text']}\n"
        f"问题：其中“{target['span2_text']}”是否指代“{target['span1_text']}”？\n"
        "答案："
    )


def wsc_doc_to_choice(_doc: Mapping[str, Any]) -> list[str]:
    # The Parquet feature order is true, false.
    return ["是", "否"]


def csl_doc_to_text(doc: Mapping[str, Any]) -> str:
    keywords = "、".join(doc["keyword"])
    return (
        f"摘要：{doc['abst']}\n"
        f"关键词：{keywords}\n"
        "问题：这些关键词是否都真实属于该论文摘要？\n"
        "答案："
    )


def yes_no_doc_to_choice(_doc: Mapping[str, Any]) -> list[str]:
    return ["否", "是"]


def c3_doc_to_text(doc: Mapping[str, Any]) -> str:
    context = "\n".join(doc["context"])
    return f"阅读材料：\n{context}\n问题：{doc['question']}\n答案："


def c3_doc_to_choice(doc: Mapping[str, Any]) -> list[str]:
    return list(doc["choice"])


def c3_doc_to_target(doc: Mapping[str, Any]) -> int:
    choices = c3_doc_to_choice(doc)
    try:
        return choices.index(doc["answer"])
    except ValueError as error:
        raise ValueError(f"C3 answer is absent from choices: {doc['answer']!r}") from error


def _chid_answer_ids(answers: Any) -> list[int]:
    if isinstance(answers, Mapping):
        return [int(value) for value in answers.get("candidate_id", [])]

    if isinstance(answers, Sequence) and not isinstance(answers, (str, bytes)):
        return [int(answer["candidate_id"]) for answer in answers]

    raise TypeError(f"Unsupported CHID answers type: {type(answers).__name__}")


def process_chid(dataset: datasets.Dataset) -> datasets.Dataset:
    rows: list[dict[str, Any]] = []

    for doc in dataset:
        candidates = list(doc["candidates"])
        answer_ids = _chid_answer_ids(doc["answers"])
        answer_position = 0

        for content in doc["content"]:
            matches = list(_IDIOM_PATTERN.finditer(content))
            for match in matches:
                if answer_position >= len(answer_ids):
                    raise ValueError(f"CHID sample {doc['idx']} has fewer answers than blanks")

                label = answer_ids[answer_position]
                if not 0 <= label < len(candidates):
                    raise ValueError(f"CHID sample {doc['idx']} has invalid label {label}")

                masked = content[: match.start()] + "【待填成语】" + content[match.end() :]
                masked = _IDIOM_PATTERN.sub("【其他空缺】", masked)
                rows.append(
                    {
                        "idx": f"{doc['idx']}_{answer_position}",
                        "content": masked,
                        "candidates": candidates,
                        "label": label,
                    }
                )
                answer_position += 1

        if answer_position != len(answer_ids):
            raise ValueError(f"CHID sample {doc['idx']} has more answers than blanks")

    return datasets.Dataset.from_list(rows)


def chid_doc_to_text(doc: Mapping[str, Any]) -> str:
    return (
        f"句子：{doc['content']}\n"
        "请选择最恰当的成语填入【待填成语】处。\n"
        "答案："
    )


def cmrc_doc_to_text(doc: Mapping[str, Any]) -> str:
    return (
        "请根据文章回答问题，只输出文章中的答案原文，不要解释。\n"
        f"文章：{doc['context']}\n"
        f"问题：{doc['question']}\n"
        "答案："
    )


def drcd_doc_to_text(doc: Mapping[str, Any]) -> str:
    return (
        "請根據文章回答問題，只輸出文章中的答案原文，不要解釋。\n"
        f"文章：{doc['context']}\n"
        f"問題：{doc['question']}\n"
        "答案："
    )


def _answer_texts(doc: Mapping[str, Any]) -> list[str]:
    answers = doc["answers"]
    if isinstance(answers, Mapping):
        return [str(answer) for answer in answers.get("text", [])]
    return [str(answer["text"]) for answer in answers]


def qa_doc_to_target(doc: Mapping[str, Any]) -> str:
    answers = _answer_texts(doc)
    if not answers:
        raise ValueError(f"QA sample {doc.get('id', '<unknown>')} has no reference answer")
    return answers[0]


def _clean_prediction(prediction: str) -> str:
    prediction = prediction.strip().strip("`").strip()
    prediction = _ANSWER_PREFIX.sub("", prediction)
    return prediction.splitlines()[0].strip() if prediction else ""


def _normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        char for char in normalized if not char.isspace() and char not in _PUNCTUATION
    )


def _mixed_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    for char in normalized:
        if char.isspace() or char in _PUNCTUATION:
            flush_buffer()
        elif _CJK_PATTERN.fullmatch(char):
            flush_buffer()
            tokens.append(char)
        else:
            buffer.append(char)

    flush_buffer()
    return tokens


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _f1_score(prediction: str, reference: str) -> float:
    prediction_tokens = _mixed_tokens(prediction)
    reference_tokens = _mixed_tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    overlap = _lcs_length(prediction_tokens, reference_tokens)
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def process_results_qa(doc: Mapping[str, Any], results: list[str]) -> dict[str, float]:
    prediction = _clean_prediction(results[0])
    references = _answer_texts(doc)
    if not references:
        raise ValueError(f"QA sample {doc.get('id', '<unknown>')} has no reference answer")

    exact_match = max(
        float(_normalize_answer(prediction) == _normalize_answer(reference))
        for reference in references
    )
    f1 = max(_f1_score(prediction, reference) for reference in references)
    return {"exact_match": exact_match, "f1": f1}
