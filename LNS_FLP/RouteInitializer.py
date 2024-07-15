import numpy as np
import random


from MultiModalState import *

class RouteInitializer:
    def __init__(self, data):
        self.data = data

    def neighbors_init_truck(self, customer):
        locations = np.argsort(self.data["edge_km_t"][customer])
        return locations[locations != 0]

    def validate_truck_routes(self, truck_routes):
        for route in truck_routes:
            consecutive_zeros = sum(1 for loc in route if loc == 0)
            if consecutive_zeros > 2:
                raise ValueError("Unable to satisfy demand with the given number of trucks!!")

    def nearest_neighbor_init_truck(self):
        truck_init_routes = [[] for _ in range(self.data["num_t"])]
        unvisited = set(range(1, self.data["dimension"]))

        while unvisited:
            for i in range(self.data["num_t"]):
                route = [0]
                route_demands = 0

                while unvisited:
                    current = route[-1]
                    neighbors = [nb for nb in self.neighbors_init_truck(current) if nb in unvisited]
                    nearest = neighbors[0]

                    if route_demands + self.data["demand"][nearest] > self.data["demand_t"]:
                        break

                    route.append(nearest)
                    unvisited.remove(nearest)
                    route_demands += self.data["demand"][nearest]

                route.append(0)
                truck_init_routes[i].extend(route[0:])

        self.validate_truck_routes(truck_init_routes)

        return {
            'num_t': len(truck_init_routes),
            'num_d': 0,
            'route': [{'vtype': 'truck', 'vid': f't{i + 1}', 'path': path} for i, path in enumerate(truck_init_routes)]
        }
    
    def init_truck(self):
        # nearest_neighbor_init_truck() 메서드 호출
        truck_init_routes = self.nearest_neighbor_init_truck()
        init_truck=[]
        # 각 경로를 튜플로 변환
        for route in truck_init_routes['route']:
            tuple_route = [(node, 0) for node in route['path']]
            init_truck.append(tuple_route)
        
        return MultiModalState(init_truck)
    
    def initialize_population(self, elements, population_size):
        # 초기 집단 생성: 랜덤으로 생성된 경로들
        population = []
        for _ in range(population_size):
            individual = self.random_route(elements)
            population.append(individual)
        return population
    
    def random_route(self, elements):
        # 랜덤한 경로를 생성하는 방법 (예: 초기 해에서 셔플)
        route = list(elements)  # 요소를 랜덤하게 변형
        random.shuffle(route)
        return route

    def evaluate_fitness_truck(self, individual, fly_node, catch_node):
        route = [fly_node] + individual + [catch_node]
        distance = 0
        for i in range(len(route) - 1):
            start_node = route[i]
            end_node = route[i + 1]
            edge_weight = data["edge_km_t"][start_node][end_node]
            distance += edge_weight
        fitness = 1000 / distance
        print(f"Route: {route}, Distance: {distance}, Fitness: {fitness}")
        return fitness
    
    def roulette_wheel_selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [fitness / total_fitness for fitness in fitness_scores]
        selected_individuals = random.choices(population, weights=selection_probs, k=len(population))
        return selected_individuals
    
    def crossover_and_mutate(self, selected_individuals, mutation_rate):
        # 교차 및 변이 연산을 통해 자손 생성
        offspring = []
        
        while len(offspring) < len(selected_individuals):
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = self.crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = self.mutate(child)
            offspring.append(child)
        
        return offspring

    def crossover(self, parent1, parent2):
        size = len(parent1)
        # 두 부모의 임의의 부분을 교환하여 자식 생성
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end+1] = parent1[start:end+1]
        
        current_pos = (end + 1) % size
        for i in range(size):
            if parent2[i] not in child:
                child[current_pos] = parent2[i]
                current_pos = (current_pos + 1) % size
    
        return child

    def mutate(self, individual):
        # 변이 연산: 랜덤으로 두 노드를 교환
        if len(individual) < 2:
            return individual
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    def run_ga_truck(self, elements, fly_node, catch_node, population_size, generations, mutation_rate):
        # 유전 알고리즘 실행
        population = self.initialize_population(elements, population_size)
        print(population)

        best_individual = max(population, key=lambda x: self.evaluate_fitness_truck(x, fly_node, catch_node))
        best_fitness = self.evaluate_fitness_truck(best_individual, fly_node, catch_node)

        for generation in range(generations):
            fitness_scores = [self.evaluate_fitness_truck(individual, fly_node, catch_node) for individual in population]
            selected_individuals = self.roulette_wheel_selection(population, fitness_scores)
            offspring = self.crossover_and_mutate(selected_individuals, mutation_rate)
            population = offspring
            
            current_best_individual = max(population, key=lambda x: self.evaluate_fitness_truck(x, fly_node, catch_node))
            current_best_fitness = self.evaluate_fitness_truck(current_best_individual, fly_node, catch_node)
            
            if current_best_fitness > best_fitness:
                best_individual = current_best_individual
                best_fitness = current_best_fitness
            
            print(f"Truck GA Generation {generation}: Best fitness = {self.evaluate_fitness_truck(best_individual, fly_node, catch_node)}")

        return best_individual
    
    def ga_idle(self, population_size, generations, mutation_rate):
        
        init_truck = []
        for i in range(1,50):
            init_truck.append(i)

        start_node = 0
        end_node = 0

        best_idle = self.run_ga_truck(init_truck, start_node, end_node, population_size, generations, mutation_rate)

        init_truck = [start_node] + best_idle + [end_node]

        tuple_route = []
        for node in init_truck:
            tuple_route.append((node, IDLE)) 

        return MultiModalState([tuple_route]) 

    def shuffle_middle_excluding_ends(self, route):
        """
        주어진 경로에서 맨 앞과 맨 뒤를 제외하고 중간 요소만 셔플합니다.
        """
        if len(route) <= 2:
            # 경로의 길이가 2 이하인 경우, 셔플할 요소가 없으므로 그대로 반환
            return route
        
        # 맨 앞과 맨 뒤를 제외한 중간 부분을 추출
        start = route[0]
        end = route[-1]
        middle = route[1:-1]
        
        # 중간 부분을 랜덤으로 섞음
        random.shuffle(middle)
        
        # 맨 앞, 셔플된 중간, 맨 뒤를 다시 결합
        shuffled_route = [start] + middle + [end]
        return shuffled_route

    def random_truck(self):
        truck_init_routes = self.nearest_neighbor_init_truck()
        init_truck = []
        
        # 각 경로를 튜플로 변환
        for route in truck_init_routes['route']:
            tuple_route = [(node, 0) for node in route['path']]
            init_truck.append(tuple_route)
        
        # 각 경로의 중간 부분만 랜덤으로 섞기
        for i in range(len(init_truck)):
            init_truck[i] = self.shuffle_middle_excluding_ends(init_truck[i])
        
        return MultiModalState(init_truck)


    def dividing_route(self, route_with_info, route_index):
        truck_route = [value for value in route_with_info if value[1] != 2]
        drone_route = [value for value in route_with_info if value[1] != 4]

        return [
            {'vtype': 'drone', 'vid': 'd' + str(route_index + 1), 'path': drone_route},
            {'vtype': 'truck', 'vid': 't' + str(route_index + 1), 'path': truck_route},
        ]
        
    
    def combine_paths(self, route_data):
        combined_paths = []
        initial_solution = self.nearest_neighbor_init_truck()
        nn_truck_paths = [route['path'] for route in initial_solution['route']]
        
        for i in range(len(route_data)):  # 모든 드론 및 트럭 경로에 대해 반복
            if i % 2 == 0:  # 짝수 인덱스인 경우 드론 경로
                nn_truck_path = nn_truck_paths[i // 2]
                drone_route = route_data[i]['path']
                truck_route = route_data[i + 1]['path']
                
                filled_path = []
                for node, value in drone_route[:-1]:
                    if node not in [point[0] for point in filled_path]:
                        filled_path.append((node, value))
                
                for node, value in truck_route[:-1]:
                    if node not in [point[0] for point in filled_path]:
                        filled_path.append((node, value))
            
                filled_path = sorted(filled_path, key=lambda x: nn_truck_path[:-1].index(x[0]))
                filled_path.append(drone_route[-1])
                combined_paths.append(filled_path)
        
        return combined_paths
