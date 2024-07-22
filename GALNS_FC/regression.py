import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 첫 번째 데이터셋
temperatures1 = np.array([-20, -10, 0, 10, 20, 30, 40])
percent_increases1 = np.array([238.4, 181.9, 93.1, 28.0, 0.0, 29.3, 45.2])

def model_func(T, a, b, c):
    return a * T**2 + b * T + c

params1, _ = curve_fit(model_func, temperatures1, percent_increases1)
a1, b1, c1 = params1
predicted_percent_increases1 = model_func(temperatures1, *params1)

# 두 번째 데이터셋
temperatures2 = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22.5, 25, 28, 30, 32, 33])
percentages2 = np.array([14, 17, 20, 23, 25, 27, 29, 30, 30.5, 31, 31.5, 32, 31.5, 30.5, 30, 28, 27])

params2, _ = curve_fit(model_func, temperatures2, percentages2)
a2, b2, c2 = params2
predicted_percent_increases2 = model_func(temperatures2, *params2)

# 세 번째 데이터셋
speeds = np.array([10/60, 20/60, 30/60, 40/60, 50/60, 60/60, 70/60, 80/60, 90/60, 100/60, 110/60, 120/60, 130/60, 140/60])
percent_increases3 = np.array([12, 10, 7, 2.8, 0.0, 6.9, 11.7, 15.2, 22.1, 29.0, 33.1, 41.4, 48.3, 54.5])

params3, _ = curve_fit(model_func, speeds, percent_increases3)
a3, b3, c3 = params3
predicted_percent_increases3 = model_func(speeds, *params3)

# 그래프 그리기
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# 첫 번째 그래프
axs[0].scatter(temperatures1, percent_increases1, color='red', label='Given Data')
axs[0].plot(temperatures1, predicted_percent_increases1, color='blue', label='Fitted Curve')
axs[0].set_xlabel('Temperature (°C)')
axs[0].set_ylabel('Percentage of Increase (%)')
axs[0].set_title('Impact of Temperature on Net Energy Consumption Increase')
axs[0].legend()
axs[0].grid(True)

# 첫 번째 그래프에 회귀식 추가
equation1 = f'$y = {a1:.2f}x^2 + {b1:.2f}x + {c1:.2f}$'
axs[0].text(0.05, 0.55, equation1, transform=axs[0].transAxes, fontsize=12,
            verticalalignment='top')

# 두 번째 그래프
axs[1].scatter(temperatures2, percentages2, color='red')
axs[1].plot(temperatures2, predicted_percent_increases2, color='blue')
axs[1].set_xlabel('Temperature (°C)')
axs[1].set_ylabel('Percentage (%)')
axs[1].set_title('Ratio of Regenerative Rotational Energy to Net Energy Consumption by Temperature')
axs[1].legend()
axs[1].grid(True)

# 두 번째 그래프에 회귀식 추가
equation2 = f'$y = {a2:.2f}x^2 + {b2:.2f}x + {c2:.2f}$'
axs[1].text(0.05, 0.75, equation2, transform=axs[1].transAxes, fontsize=12,
            verticalalignment='top')

# 세 번째 그래프
axs[2].scatter(speeds, percent_increases3, color='red')
axs[2].plot(speeds, predicted_percent_increases3, color='blue')
axs[2].set_xlabel('Speed (km/min)')
axs[2].set_ylabel('Percentage of Increase (%)')
axs[2].set_title('Impact of Speed on Energy Consumption Increase')
axs[2].legend()
axs[2].grid(True)

# 세 번째 그래프에 회귀식 추가
equation3 = f'$y = {a3:.2f}x^2 + {b3:.2f}x + {c3:.2f}$'
axs[2].text(0.05, 0.55, equation3, transform=axs[2].transAxes, fontsize=12,
            verticalalignment='top')

plt.subplots_adjust(hspace=0.5)  # 전체 서브플롯 간의 간격을 더 늘림

plt.show()
