"""
eda_tools.py
============
EDA 시각화 유틸리티 (Designed by 의정)
  - HealthEatVisualizer : OS별 한글 폰트 자동 설정 + 분포 시각화
"""

import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


class HealthEatVisualizer:
    """EDA 및 시각화 분석 도구 (Designed by 의정)"""

    def __init__(self):
        system = platform.system()

        if system == 'Darwin':
            # ✅ Mac은 fname 대신 family로 지정 (경로 방식 미지원)
            self.fp = fm.FontProperties(family='AppleGothic')
            plt.rc('font', family='AppleGothic')

        elif system == 'Windows':
            font_path = 'C:/Windows/Fonts/malgun.ttf'
            if os.path.exists(font_path):
                self.fp = fm.FontProperties(fname=font_path)
                plt.rc('font', family=self.fp.get_name())
            else:
                print("⚠️  [Windows] malgun.ttf를 찾을 수 없습니다. 기본 폰트로 대체합니다.")
                self.fp = fm.FontProperties()

        else:
            # Linux (Colab 등) — NanumGothic 미설치 방어
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            if os.path.exists(font_path):
                self.fp = fm.FontProperties(fname=font_path)
                plt.rc('font', family=self.fp.get_name())
            else:
                print("⚠️  [Linux] NanumGothic 폰트 없음. 기본 폰트로 대체합니다.")
                print("   설치: !apt-get install -y fonts-nanum > /dev/null")
                self.fp = fm.FontProperties()

        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ [{system}] 폰트 설정 완료: {self.fp.get_name()}")

    def plot_distribution(self, df):
        """
        클래스별 알약 빈도 분포를 수평 막대 그래프로 시각화합니다.

        Args:
            df : 'class_name' 컬럼을 포함한 DataFrame (annotations_df)
        """
        class_counts = df['class_name'].value_counts()

        fig, ax = plt.subplots(figsize=(10, max(8, len(class_counts) * 0.25)))
        sns.barplot(x=class_counts.values, y=class_counts.index,
                    palette='coolwarm', ax=ax)

        ax.set_title('■ 클래스별 알약 빈도 분포', fontproperties=self.fp, fontsize=18)
        ax.set_xlabel('개수', fontproperties=self.fp, fontsize=12)
        ax.set_ylabel('클래스', fontproperties=self.fp, fontsize=12)

        for label in ax.get_yticklabels():
            label.set_fontproperties(self.fp)

        plt.tight_layout()
        plt.show()
