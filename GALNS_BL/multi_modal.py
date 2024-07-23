import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import atexit
from FileReader import *
from RouteInitializer import *
from RouteGenerator import *

SEED = 1234
rnd_state = np.random.RandomState(None)

vrp_file_path = r'C:\Users\User\OneDrive\바탕 화면\realrealrealreal-main\realrealrealreal-main\GALNS_BL\data\multi_modal_data.vrp'
backup_file_path = r'C:\Users\User\OneDrive\바탕 화면\realrealrealreal-main\realrealrealreal-main\GALNS_BL\data\multi_modal_data_backup.vrp'

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
time_matrix = np.array(data["edge_km_t"])/data["speed_t"]

#max value 보다 1작은게 최대값으로 설정되고 있음
data["logistic_load"] = file_reader.randomize_vrp_data(data["logistic_load"], max_value=8, key_seed=78)
data["availability_landing_spot"] = file_reader.randomize_vrp_data(data["availability_landing_spot"], key_seed=12)
data["customer_drone_preference"] = file_reader.randomize_vrp_data(data["customer_drone_preference"], key_seed=34)
data["priority_delivery_time"] = file_reader.randomize_vrp_data(data["priority_delivery_time"], time=time_matrix, key_seed=56)

# price는 무게에 의존
logistic_load = np.array(list(data['logistic_load'].values()))
price = np.zeros_like(logistic_load)
price[logistic_load >= 5] = 6
price[(logistic_load <= 5) & (logistic_load > 3)] = 5
price[(logistic_load <= 3) & (logistic_load > 1)] = 4
price[logistic_load <= 1] = 3
data["price"] = {key: value for key, value in zip(data["price"].keys(), price)}

# price는 우선배송에게 추가요금 (충족못하면 100% 환불)
#priority_time = np.array(list(data["priority_delivery_time"].values()))
#price[priority_time != 0] += 3
#data["price"] = {key: value for key, value in zip(data["price"].keys(), price)}


revenue=0
for i in range(1, data["dimension"]):
    revenue = revenue + data["price"][i]


data_rf = {
    "Urban": {
        "speed": 17.6,
        "temperature": [-20, -10, 0, 10, 20, 30, 40],
        "increase": [238.4, 181.9, 93.1, 28, 0, 29.3, 45.2]
    },
    "WLTC Class 3": {
        "speed": 46,
        "temperature": [-20, -10, 0, 10, 20, 30, 40],
        "increase": [91.5, 65, 33.1, 10.6, 0, 7.9, 11.1]
    },
    "Highway": {
        "speed": 91.8,
        "temperature": [-20, -10, 0, 10, 20, 30, 40],
        "increase": [41, 29.9, 16, 5.6, 0, 1.1, 1.7]
    }
}

temperatures = np.array(data_rf["Urban"]["temperature"] + data_rf["WLTC Class 3"]["temperature"] + data_rf["Highway"]["temperature"])
speeds = np.array([data_rf["Urban"]["speed"]] * 7 + [data_rf["WLTC Class 3"]["speed"]] * 7 + [data_rf["Highway"]["speed"]] * 7)
increases = np.array(data_rf["Urban"]["increase"] + data_rf["WLTC Class 3"]["increase"] + data_rf["Highway"]["increase"])

X = np.column_stack((temperatures, speeds))
y = increases

rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X, y)

random_forest=rf_model.predict([[data["temperature"], data["speed_t"]]])[0]
base_efficiency=data["energy_kwh/km_t"]
data["energy_kwh/km_t"] = base_efficiency*(1+random_forest/100)
print(random_forest)

# 파일 백업
file_reader.create_backup(vrp_file_path, backup_file_path)

file_reader.update_vrp_file(vrp_file_path, data)

from SolutionPlotter import *
from MultiModalState import *
from Destroy import *
from Repair import *

destroyer = Destroy()
Rep = Repair()
plotter = SolutionPlotter(data)

initializer = RouteInitializer(data)
initial_truck = initializer.init_truck()
ga_count=0
k_opt_count=0

# 초기 설정
iteration_num=30

start_temperature = 100
end_temperature = 0.01
step = 0.1

temperature = start_temperature
current_num= 1
outcome_counts = {1: 0, 2: 0, 3: 0, 4: 0}
start_time = time.time()
min_objective_time = start_time

current_states = []  # 상태를 저장할 리스트
objectives = []  # 목적 함수 값을 저장할 리스트

current_states.append(initial_truck)
objective_value = MultiModalState(initial_truck).cost_objective()
objectives.append(objective_value)

destroy_operators = [destroyer.random_removal]

repair_operators = [Rep.greedy_truck_repair]
                    
destroy_counts = {destroyer.__name__: 0 for destroyer in destroy_operators}
repair_counts = {repairer.__name__: 0 for repairer in repair_operators}  

class RouletteWheel:
    def __init__(self):
        self.scores = [20, 5, 1, 0.5] 
        self.decay = 0.8
        self.operators = None
        self.weights = None

    def set_operators(self, operators):
        self.operators = operators
        self.weights = [1.0] * len(operators)

    def select_operators(self):
        total_weight = sum(self.weights)
        probabilities = [weight / total_weight for weight in self.weights]
        selected_index = np.random.choice(len(self.operators), p=probabilities)
        selected_operator = self.operators[selected_index]
        return selected_operator

    def update_weights(self, outcome_rank, selected_operator_idx):
        idx = selected_operator_idx
        score = self.scores[outcome_rank-1]
        self.weights[idx] = self.decay * self.weights[idx] + (1 - self.decay) * score                  

destroy_selector = RouletteWheel()
repair_selector = RouletteWheel()

destroy_selector.set_operators(destroy_operators)
repair_selector.set_operators(repair_operators)

while current_num <= iteration_num:
    if current_num==1:
        selected_destroy_operators = destroy_selector.select_operators()
        selected_repair_operators = repair_selector.select_operators()

        destroyed_state = selected_destroy_operators(initial_truck, rnd_state)
        repaired_state = selected_repair_operators(destroyed_state, rnd_state)

        current_states.append(repaired_state)
        objective_value = MultiModalState(repaired_state).cost_objective()
        objectives.append(objective_value)

        d_idx = destroy_operators.index(selected_destroy_operators)
        r_idx = repair_operators.index(selected_repair_operators)

        destroy_counts[destroy_operators[d_idx].__name__] += 1
        repair_counts[repair_operators[r_idx].__name__] += 1
        current_num+=1
        # 이때 새로 갱신했으면 이때 min_objective_time을 기억하도록 하기    
        
    elif current_num % 100 == 0 and current_num != 0:
        if current_num % 500 == 0:
            genetic_state = Rep.genetic_algorithm(current_states[-1],population_size=5, generations=100, mutation_rate=0.2)

            current_states.append(genetic_state)
            objective_value = MultiModalState(genetic_state).cost_objective()
            objectives.append(objective_value)
            if objective_value == min(objectives):
                min_objective_time = time.time()
            temperature = max(end_temperature, temperature - step)
            ga_count+=1
            current_num +=1

        else:
            k_opt_state = Rep.drone_k_opt(current_states[-1],rnd_state)
            current_states.append(k_opt_state)
            objective_value = MultiModalState(k_opt_state).cost_objective()
            objectives.append(objective_value)
            if objective_value == min(objectives):
                min_objective_time = time.time()
            temperature = max(end_temperature, temperature - step)
            k_opt_count+=1
            current_num +=1
        

    else:
        selected_destroy_operators = destroy_selector.select_operators()
        selected_repair_operators = repair_selector.select_operators()
    
        destroyed_state = selected_destroy_operators(current_states[-1].copy(), rnd_state)
        repaired_state = selected_repair_operators(destroyed_state, rnd_state)

        # 이전 objective 값과 비교하여 수락 여부 결정(accept)
        if np.exp((MultiModalState(current_states[-1]).cost_objective() - MultiModalState(repaired_state).cost_objective()) / temperature) >= rnd.random():
            current_states.append(repaired_state)
            objective_value = MultiModalState(repaired_state).cost_objective()
            objectives.append(objective_value)
            if objective_value == min(objectives):
                outcome = 1
                min_objective_time = time.time()

            elif objective_value <= MultiModalState(current_states[-1]).cost_objective():
                outcome = 2
            else: 
                outcome = 3

        else:
            # 이전 상태를 그대로 유지(reject)
            current_states.append(current_states[-1])
            objectives.append(MultiModalState(current_states[-1]).cost_objective())
            outcome = 4

        outcome_counts[outcome] += 1

        d_idx = destroy_operators.index(selected_destroy_operators)
        r_idx = repair_operators.index(selected_repair_operators)

        destroy_selector.update_weights(outcome, d_idx)
        repair_selector.update_weights(outcome, r_idx)

        # 온도 갱신
        temperature = max(end_temperature, temperature - step)

        destroy_counts[destroy_operators[d_idx].__name__] += 1
        repair_counts[repair_operators[r_idx].__name__] += 1
        current_num+=1


min_objective = min(objectives)
min_index = objectives.index(min_objective)
end_time = time.time()
execution_time = end_time - start_time
min_objective_time = min_objective_time - start_time

truck_soc, drone_soc = MultiModalState(current_states[min_index]).soc()[:2]
truck_time_arrival, drone_time_arrival = MultiModalState(current_states[min_index]).renew_time_arrival()
total_routes = MultiModalState(current_states[min_index]).routes
truck_current_kwh = data["battery_kwh_t"]
drone_current_kwh = data["battery_kwh_d"]


solution_printer = SolutionPrinter(
    current_states=current_states,
    initial_truck=initial_truck,
    min_index=min_index,
    objectives=objectives,
    outcome_counts=outcome_counts,
    destroy_counts=destroy_counts,
    repair_counts=repair_counts,
    ga_count=ga_count,
    k_opt_count=k_opt_count,
    execution_time=execution_time,
    min_objective_time=min_objective_time,
    revenue=revenue
)

saver = SolutionSaver(
    filename="GALNS_BL",
    data=data,
    current_states=current_states,
    min_index=min_index,
    initial_truck=initial_truck,
    ga_count=ga_count,
    k_opt_count=k_opt_count,
    execution_time=execution_time,
    min_objective_time=min_objective_time,
    revenue=revenue,
    outcome_counts=outcome_counts,
    destroy_counts=destroy_counts,
    repair_counts=repair_counts
)

#plotter.plot_current_solution(current_states[min_index])
#solution_printer.print_solution()
#solution_printer.plot_objectives()

# SOC 플롯 생성
#soc_plotter = SOCPlotter(data, truck_soc, drone_soc, truck_current_kwh, drone_current_kwh)
#soc_plotter.plot(total_routes)

# 경과 시간 플롯 생성
#elapsed_time_plotter = ElapsedTimePlotter(data, truck_time_arrival, drone_time_arrival)
#elapsed_time_plotter.plot(total_routes)

# 솔루션 텍스트 파일/사진 저장
saver.save_solution()
saver.save_current_solution(current_states[min_index], name="cex")
saver.save_convergence_graph(objectives, name="sex")

atexit.register(file_reader.restore_on_exit, backup_file_path, vrp_file_path)