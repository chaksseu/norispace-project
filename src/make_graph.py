import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의 (순서는 각 메트릭별로 유지)
categories = ["Accuracy", "F1 Score", "Average Precision", "ROC AUC"]
classifier = [0.69203, 0.60465, 0.81728, 0.79022]
no_ocr = [0.7341, 0.7162, 0.7448, 0.7561]
ocr = [0.7658, 0.7672, 0.8203, 0.8096]
advanced_ocr = [0.7848, 0.7605, 0.8585, 0.8301]
advanced_ocr2 = [0.72101, 0.71161, 0.81479, 0.79343]

x = np.arange(len(categories))  # 각 메트릭의 x위치: 0, 1, 2, 3
width = 0.15  # 각 막대의 너비

# 저장 경로 설정
save_path = "0203_results/comparison_metrics_classifier.png"  # 파일 이름 및 경로 설정

# 스타일 지정
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 8))

# 막대 색상 (왼쪽부터 Classifier, No OCR, OCR, Advanced OCR, Advanced OCR2)
colors = ['#D4A017', '#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

# 각 메트릭별로 5개의 막대를 생성 (좌측부터 Classifier가 가장 왼쪽)
bars_classifier    = ax.bar(x - 2*width, classifier,   width, label='Classifier', color=colors[0], edgecolor='black')
bars_no_ocr        = ax.bar(x - width,   no_ocr,       width, label='No OCR',     color=colors[1], edgecolor='black')
bars_ocr           = ax.bar(x,           ocr,          width, label='OCR',        color=colors[2], edgecolor='black')
bars_advanced_ocr  = ax.bar(x + width,   advanced_ocr, width, label='Advanced OCR', color=colors[3], edgecolor='black')
bars_advanced_ocr2 = ax.bar(x + 2*width, advanced_ocr2,width, label='Advanced OCR2', color=colors[4], edgecolor='black')

# 축 및 제목 설정
ax.set_ylabel('Scores', fontsize=14)
ax.set_xlabel('Metrics', fontsize=14)
ax.set_title('Comparison of Metrics Across Different Methods', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0.6, 0.9)  # y축 범위 지정
ax.legend(fontsize=12, loc='upper left', frameon=True)

# 데이터 라벨 추가 함수
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            color='black',
            fontweight='bold'
        )

add_labels(bars_classifier)
add_labels(bars_no_ocr)
add_labels(bars_ocr)
add_labels(bars_advanced_ocr)
add_labels(bars_advanced_ocr2)

# 레이아웃 조정 및 저장
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 고해상도로 저장
plt.close(fig)  # 그래프 닫기
