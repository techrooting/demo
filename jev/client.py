import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)
from langchain_typesafe import (
    Choice,
    Noul,
    NoulCriteria,
    Score,
    TypeSafeClassifier,
)
from rich import print as rich_print


jev_result = TypeSafeClassifier(
        model=os.getenv("BJEV_MODEL", "bjev"),
        base_url=os.getenv("BJEV_BASE_URL", "https://bjev.bytedance.net/api"),
    ).invoke(
        {
            # 待判断的业务上下文，同一份内容会被下方多个问题复用
            "state": "我的订阅今天又被扣费了两次，已经影响报销，请马上帮我处理。",
            # 一次调用中要评估的问题集合；department、frustration、is_urgent 是自定义问题名，也是读取结果时使用的键
            # instructions：向模型说明当前问题要判断什么
            # criteria：限定答案空间。Choice 使用键值对定义类别及其含义，Score 使用有序列表定义从轻到重的评分锚点
            "questions": {
                # 返回格式一：choice，从预设选项里选择一个，并返回概率分布和置信度。
                "department": Choice(
                    instructions="这条客服消息应该由哪个支持团队处理？",
                    criteria={
                        "billing": "支付、发票、退款、重复扣费、订阅相关问题",
                        "technical": "缺陷、故障、集成、API、产品错误相关问题",
                        "sales": "价格、升级、采购、新账号相关问题",
                    },
                ),
                # 返回格式二：score，按有序等级打分，可返回两个等级之间的位置。
                "frustration": Score(
                    instructions="客户的沮丧或不满程度有多高？",
                    criteria=["平静", "有些不满", "明显沮丧", "非常愤怒"],
                ),
                # 返回格式三：noul，是/否问题的概率，0 表示否，1 表示是。
                "is_urgent": Noul(
                    instructions="这条消息是否需要紧急处理？",
                    criteria=NoulCriteria(
                        true="客户描述了时间压力、重复失败或严重影响",
                        false="客户可以等待，或只是咨询一般信息",
                    ),
                ),
            },
        }
    )
    
rich_print(jev_result)


# ClassifierResponse(
#     model='bjev',
#     answers={
#         'department': ChoiceAnswer(
#             type='choice',
#             choice='billing',
#             probabilities={
#                 'billing': 0.9795,
#                 'technical': 0.0091,
#                 'sales': 0.0114
#             },
#             confidence=0.9693
#         ),
#         'frustration': ScoreAnswer(
#             type='score',
#             score=2.038,
#             legend={0: '平静', 1: '有些不满', 2: '明显沮丧', 3: '非常愤怒'},
#             probabilities={0: 0.0456, 1: 0.206, 2: 0.4131, 3: 0.3353},
#             confidence=0.3674
#         ),
#         'is_urgent': NoulAnswer(type='noul', noul=0.9284)
#     },
#     usage=Usage(input_tokens=139, output_tokens=256),
#     request_id=None
# )