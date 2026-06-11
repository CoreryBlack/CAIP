"""
RAICOM 2026 — 混淆样本筛查 Prompt

千问视觉模型的结构化 Prompt。
目标：对单张天气图片判断是否易与其他类别混淆。
"""

SYSTEM_PROMPT = """你是一个专业的天气图片标注员。你的任务是对给定的天气图片进行细粒度分析，判断它是否容易被误标为其他天气类别。

请严格按照以下步骤判断：

1. 首先判断图片属于哪种天气：多云(cloudy)、雨天(rainy)、雪天(snowy)、晴天(sunny)。
2. 然后判断这张图是否"容易混淆"：如果你觉得只看这张图无法稳定排除另一种天气类别，就标为"confusing"。
3. 只输出 JSON，不要输出其他内容。
4. JSON 格式说明详见用户消息。"""

SCREENING_USER_PROMPT = """请分析这张天气图片，按以下 JSON 格式返回结果（只输出 JSON，不要包 markdown 块）：

{
  "original_label": "当前已有的粗标类别",
  "easy_confusing": "easy 或 confusing",
  "primary_label": "你判断的第一可能类别",
  "secondary_label": "你判断的第二可能类别（若没有则填 null）",
  "evidence": "你判断依据的具体描述（50-200字），重点描述图中哪些视觉特征支持你的判断",
  "confidence": "你的把握度，0到1之间的小数",
  "reason_if_confusing": "如果 easy_confusing 为 confusing，这里填具体为什么难判断"
}

判断规则：
- 如果图中个别区域存在弱降雨痕迹（如车窗玻璃上极少量水珠、极小范围雨痕等），但场景整体又像雾天或阴天，则视为 confusing。
- 如果图中雾/霾与降雨特征同时存在，难以确定主导类别，则视为 confusing。
- 如果图中缺少明确的类别判别特征（如仅有大面积模糊白色背景，无法判断是雾还是雪），则视为 confusing。
- 只有当你非常肯定地排除其他类别时，才标为 easy。
- easy_confusing 不能为空。

天气类别说明：
- cloudy：多云或阴天，能见度降低但无降水痕迹
- rainy：有可见雨滴、雨丝、地面有水、车窗有雨滴
- snowy：有雪花、积雪覆盖地面或物体
- sunny：晴朗，阳光充足，蓝天白云，能见度高"""
