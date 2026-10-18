"""Choice extraction migrated from cubenlp/FuxiBench src/utils.py."""
import logging
import re

logger = logging.getLogger(__name__)

choices = ["A", "B", "C", "D"]
def extract_choice(response):
    '''
        Always return a choice, even cannot match by regex,
        to ensure fair comparison to other models.
    '''
    response = str(response)
    if response[0] in choices:
        return response[0]
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        (r'答案(是|为)选项 ?([ABCD])', 2),
        (r'故?选择?：? ?([ABCD])',1),
        (r'([ABCD]) ?选?项(是|为)?正确',1),
        (r'正确的?选项(是|为) ?([ABCD])',2),
        (r'答案(应该)?(是|为)([ABCD])',3),
        (r'选项 ?([ABCD]) ?(是|为)?正确',1),
        (r'选择答案 ?([ABCD])',1),
        (r'答案?：?([ABCD])',1),
        (r'([ABCD])(选?项)?是?符合题意',1),
        (r'答案选项：? ?([ABCD])', 1), # chatglm
        (r'答案(选项)?为(.*?)([ABCD])', 3), # chatgpt
        (r'content=(.*?)([ABCD])(.*?)', 2), # finetuned qwen

    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 2. Recursive match
    patterns = [
        (r'([ABCD])(.*?)当选', 1),
        (r'([ABCD])(.*?)正确', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer

    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCD])', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 4. Check the only mentioend choices
    pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    # return choices[random.randint(0,3)]
    return "Z"  # direct wrong


def multiple_choice_acc(pred, gold):
    """ 从 llm 输出的pred 字符串中提取 选项，与 gold 对比 """
    try:
        pred = extract_choice(pred)
        return pred.strip() == gold.strip()
    except Exception as e:
        logger.info(f"extract_choice failed: pred={pred}, gold={gold}")
        return False


