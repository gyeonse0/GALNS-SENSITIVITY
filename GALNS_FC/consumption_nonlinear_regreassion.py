import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 주어진 데이터
temperatures = np.array([-20, -10, 0, 10, 20, 30, 40])
percent_increases = np.array([238.4, 181.9, 93.1, 28.0, 0.0, 29.3, 45.2])

# 비선형 함수 정의 (파라볼릭 모델)
def model_func(T, a, b, c):
    return a * T**2 + b * T + c

# 회귀 분석
params, _ = curve_fit(model_func, temperatures, percent_increases)
a, b, c = params

# 예측된 데이터 계산
predicted_percent_increases = model_func(temperatures, *params)

# 결과 출력
print(f"계수 a: {a}")
print(f"계수 b: {b}")
print(f"계수 c: {c}")

# 데이터 및 회귀 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(temperatures, percent_increases, color='red', label='Given Data')
plt.plot(temperatures, predicted_percent_increases, color='blue', label='Fitted Curve')
plt.xlabel('Temperature (°C)')
plt.ylabel('Percent Increase (%)')
plt.title('Temperature vs Percent Increase')
plt.legend()
plt.grid(True)
plt.show()
