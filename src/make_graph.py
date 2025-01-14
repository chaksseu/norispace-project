import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
categories = ["Accuracy", "F1 Score", "Average Precision", "ROC AUC"]
no_ocr = [0.7341, 0.7162, 0.7448, 0.7561]
ocr = [0.7658, 0.7672, 0.8203, 0.8096]
advanced_ocr = [0.7848, 0.7605, 0.8585, 0.8301]

x = np.arange(len(categories))  # x축 레이블 위치
width = 0.25  # 막대 너비

# 저장 경로 설정
save_path = "0114_results/comparison_metrics.png"  # 파일 이름 및 경로 설정

# 스타일 지정
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 8))

# 막대 그래프 생성
colors = ['#FF9999', '#66B2FF', '#99FF99']  # 컬러 팔레트
bars1 = ax.bar(x - width, no_ocr, width, label='No OCR', color=colors[0], edgecolor='black')
bars2 = ax.bar(x, ocr, width, label='OCR', color=colors[1], edgecolor='black')
bars3 = ax.bar(x + width, advanced_ocr, width, label='Advanced OCR', color=colors[2], edgecolor='black')

# 축 및 제목 설정
ax.set_ylabel('Scores', fontsize=14)
ax.set_xlabel('Metrics', fontsize=14)
ax.set_title('Comparison of Metrics Across Different Methods', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0.6, 0.9)  # y축 범위 지정
ax.legend(fontsize=12, loc='upper left', frameon=True)

# 데이터 라벨 추가
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# 레이아웃 조정 및 저장
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 고해상도로 저장
plt.close(fig)  # 그래프 닫기
