"""
读取并提取关键文献的核心信息
"""

from pdfminer.high_level import extract_text
import os

PAPERS = {
    "04_RandAugment": "raicom-weather/references/04_RandAugment_NeurIPS2020.pdf",
    "05_Mixup": "raicom-weather/references/05_Mixup_ICLR2018.pdf",
    "11_BalancedSoftmax": "raicom-weather/references/11_BalancedSoftmax_NeurIPS2020.pdf"
}

def extract_abstract(text: str) -> str:
    """简单的提取摘要方法"""
    # 尝试找到 Abstract 部分
    abstract_start = text.lower().find("abstract")
    if abstract_start != -1:
        # 从 "abstract" 开始，找到下一个换行符（通常是摘要开始的地方）
        text = text[abstract_start:abstract_start+2000]
        # 尝试截断到 Introduction
        intro_start = text.lower().find("1 introduction")
        if intro_start != -1:
            return text[:intro_start].replace("Abstract", "").strip()
        return text.replace("Abstract", "").strip()
    return "未找到摘要"

def safe_print(text):
    """安全打印，避免编码错误"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('utf-8', 'ignore').decode('utf-8'))

def main():
    for name, path in PAPERS.items():
        safe_print(f"正在读取 {name}...")
        try:
            text = extract_text(path)
            # 替换特殊字符避免编码问题
            text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl').replace('\u2217', '*').replace('\u00b2', '^2')
            abstract = extract_abstract(text)
            
            safe_print("-" * 50)
            safe_print(f"【{name}】")
            safe_print(f"摘要: \n{abstract[:500]}...")  # 只打印前500字符
            
            # 简单的关键词搜索
            keywords = ["hyperparameter", "result", "model", "method"]
            findings = []
            for kw in keywords:
                if kw in text.lower():
                    findings.append(kw)
            
            if findings:
                safe_print(f"提及关键词: {', '.join(findings)}")
            
        except Exception as e:
            safe_print(f"读取失败: {e}")

if __name__ == "__main__":
    main()
