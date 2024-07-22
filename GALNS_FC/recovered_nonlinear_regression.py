import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

temperatures = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22.5, 25, 28, 30, 32, 33])
percentage = np.array([14, 17, 20, 23, 25, 27, 29, 30, 30.5, 31, 31.5, 32, 31.5, 30.5, 30, 28, 27])

# 비선형 함수 정의 (파라볼릭 모델)
def model_func(T, a, b, c):
    return a * T**2 + b * T + c

# 회귀 분석
params, _ = curve_fit(model_func, temperatures, percentage)
a, b, c = params

# 예측된 데이터 계산
predicted_percent_increases = model_func(temperatures, *params)

# 결과 출력
print(f"계수 a: {a}")
print(f"계수 b: {b}")
print(f"계수 c: {c}")

# 데이터 및 회귀 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(temperatures, percentage, color='red', label='Given Data')
plt.plot(temperatures, predicted_percent_increases, color='blue', label='Fitted Curve')
plt.xlabel('Temperature (°C)')
plt.ylabel('Percentaqge (%)')
plt.title('Temperature vs Recoverd Percent')
plt.legend()
plt.grid(True)
plt.show()