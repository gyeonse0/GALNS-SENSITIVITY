import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 데이터
speeds = np.array([10/60, 20/60, 30/60, 40/60, 50/60, 60/60, 70/60, 80/60, 90/60, 100/60, 110/60, 120/60, 130/60, 140/60])
percent_increases = np.array([12, 10, 7, 2.8, 0.0, 6.9, 11.7, 15.2, 22.1, 29.0, 33.1, 41.4, 48.3, 54.5])

# 2차 다항식 모델 정의
def quadratic_model(V, a, b, c):
    return a * V**2 + b * V + c

# 회귀 분석
params, _ = curve_fit(quadratic_model, speeds, percent_increases)
a, b, c = params

# 예측된 데이터 계산
predicted_percent_increases = quadratic_model(speeds, *params)

# 결과 출력
print(f"계수 a: {a}")
print(f"계수 b: {b}")
print(f"계수 c: {c}")

# 데이터 및 회귀 결과 시각화
plt.figure(figsize=(12, 6))
plt.scatter(speeds, percent_increases, color='red', label='Given Data')
plt.plot(speeds, predicted_percent_increases, color='blue', label='Quadratic Fit')

plt.xlabel('Speed (km/min)')
plt.ylabel('Percent Increase (%)')
plt.title('Speed vs Percent Increase')
plt.legend()
plt.grid(True)
plt.show()
