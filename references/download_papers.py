"""
RAICOM 2026 — 智海算法调优赛
下载顶级文献 PDF 到 references/ 目录
"""

import urllib.request
import os
import sys
import time

PAPERS = [
    {
        "name": "01_EfficientNet_ICML2019",
        "url": "https://proceedings.mlr.press/v97/tan19a/tan19a.pdf",
        "citation": "Tan & Le. EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019",
    },
    {
        "name": "02_ConvNeXt_CVPR2022",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf",
        "citation": "Liu et al. A ConvNet for the 2020s. CVPR 2022",
    },
    {
        "name": "03_AutoAugment_CVPR2019",
        "url": "https://arxiv.org/pdf/1805.09501",
        "citation": "Cubuk et al. AutoAugment: Learning Augmentation Policies from Data. CVPR 2019",
    },
    {
        "name": "04_RandAugment_NeurIPS2020",
        "url": "https://arxiv.org/pdf/1909.13719",
        "citation": "Cubuk et al. RandAugment: Practical Automated Data Augmentation. NeurIPS 2020",
    },
    {
        "name": "05_Mixup_ICLR2018",
        "url": "https://openreview.net/pdf?id=r1Ddp1-Rb",
        "citation": "Zhang et al. mixup: Beyond Empirical Risk Minimization. ICLR 2018",
    },
    {
        "name": "06_CutMix_ICCV2019",
        "url": "https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf",
        "citation": "Yun et al. CutMix: Regularization Strategy to Train Strong Classifiers. ICCV 2019",
    },
    {
        "name": "07_RandomErasing_AAAI2020",
        "url": "https://arxiv.org/pdf/1708.04896",
        "citation": "Zhong et al. Random Erasing Data Augmentation. AAAI 2020",
    },
    {
        "name": "08_LabelSmoothing_NeurIPS2019",
        "url": "https://arxiv.org/pdf/1906.02629",
        "citation": "Müller et al. When Does Label Smoothing Help? NeurIPS 2019",
    },
    {
        "name": "09_FocalLoss_ICCV2017",
        "url": "https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf",
        "citation": "Lin et al. Focal Loss for Dense Object Detection. ICCV 2017",
    },
    {
        "name": "10_ClassBalancedLoss_CVPR2019",
        "url": "https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf",
        "citation": "Cui et al. Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019",
    },
    {
        "name": "11_BalancedSoftmax_NeurIPS2020",
        "url": "https://arxiv.org/pdf/2007.10740",
        "citation": "Ren et al. Balanced Meta-Softmax for Long-Tailed Visual Recognition. NeurIPS 2020",
    },
    {
        "name": "12_NoisyStudent_CVPR2020",
        "url": "https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf",
        "citation": "Xie et al. Self-Training With Noisy Student Improves ImageNet Classification. CVPR 2020",
    },
    {
        "name": "13_Quantization_CVPR2018",
        "url": "https://arxiv.org/pdf/1712.05877",
        "citation": "Jacob et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018",
    },
    {
        "name": "14_KnowledgeDistillation_2015",
        "url": "https://arxiv.org/pdf/1503.02531",
        "citation": "Hinton et al. Distilling the Knowledge in a Neural Network. 2015",
    },
]

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout.reconfigure(encoding='utf-8')  # 避免 GBK 编码问题


def download_pdf(url: str, save_path: str) -> bool:
    """下载 PDF，支持重试"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
                if len(data) < 1000:
                    print(f"       ⚠️  文件过小 ({len(data)} bytes)，可能不是有效 PDF")
                    return False
                with open(save_path, "wb") as f:
                    f.write(data)
                return True
        except Exception as e:
            print(f"       ⚠️  第 {attempt+1} 次失败: {e}")
            time.sleep(2)
    return False


def main():
    print(f"{'='*70}")
    print(f" RAICOM 2026 — 智海算法调优赛：顶级文献下载")
    print(f" 保存路径: {SAVE_DIR}")
    print(f"{'='*70}\n")

    success = 0
    failed = []

    for paper in PAPERS:
        name = paper["name"]
        url = paper["url"]
        citation = paper["citation"]
        save_path = os.path.join(SAVE_DIR, f"{name}.pdf")

        if os.path.exists(save_path):
            size = os.path.getsize(save_path)
            print(f"   [OK] {name}.pdf exists ({size/1024:.0f} KB)")
            success += 1
            continue

        print(f"   [DL] {name}")
        print(f"      来源: {citation}")
        print(f"      URL: {url}")

        ok = download_pdf(url, save_path)
        if ok:
            size = os.path.getsize(save_path)
            print(f"      [OK] download success ({size/1024:.0f} KB)")
            success += 1
        else:
            print(f"      [FAIL] download failed")
            failed.append(name)

        print()
        time.sleep(1)  # 避免请求过快

    # ── 总结 ──
    print(f"{'='*70}")
    print(f" Completed: {success}/{len(PAPERS)}")
    if failed:
        print(f" Failed: {', '.join(failed)}")
        print(f" Please manually download the above paper PDFs into {SAVE_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
