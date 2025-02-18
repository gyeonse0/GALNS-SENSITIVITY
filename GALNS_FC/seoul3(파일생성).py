import pandas as pd
import requests
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
import random
from haversine import haversine
import numpy as np
import os

a=89

# 서울 구 데이터 가져오기
API_key = '5357464a61716b7236306b6a746b45'
service = 'SearchFAQOfGUListService'
gu_url = f'http://openapi.seoul.go.kr:8088/{API_key}/json/{service}/1/25'
gu_list = requests.get(gu_url).json()
df_gu = pd.DataFrame(gu_list['SearchFAQOfGUListService']['row'])

# GeoJSON 데이터 가져오기
gu_json = []
vwolrd_key = '3C36C01F-FC3D-302C-BEB6-E697864F7DF3'
for gu in df_gu['CD_DESC']:
    url_vworld = f'https://api.vworld.kr/req/data?service=data&version=2.0&request=GetFeature&format=json&errorformat=json&size=10&page=1&data=LT_C_ADSIGG_INFO&attrfilter=sig_kor_nm:like:{gu}&columns=sig_cd,full_nm,sig_kor_nm,sig_eng_nm,ag_geom&geometry=true&attribute=true&key={vwolrd_key}&domain=https://localhost'
    result_dict = requests.get(url_vworld).json()
    gu_json.append(result_dict)

# GeoJSON 데이터 가공
features = []
for gu_data in gu_json:
    gu_name = gu_data['response']['result']['featureCollection']['features'][0]['properties']['sig_kor_nm']
    geometry = gu_data['response']['result']['featureCollection']['features'][0]['geometry']
    feature = {
        "type": "Feature",
        "id": gu_name,  # 구명을 id로 추가
        "geometry": geometry,
        "properties": {
            "name": gu_name
        }
    }
    features.append(feature)

geojson_data = {
    "type": "FeatureCollection",
    "features": features
}

# GeoJSON 파일 경로
file_path = 'C:/Users/User/OneDrive/바탕 화면/seoul_gu_boundaries.geojson'

# GeoJSON 파일로 저장
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(geojson_data, f, ensure_ascii=False, indent=4)

# 랜덤 포인트 생성 함수
def generate_random_points(polygon, num_points):
    points = []
    while len(points) < num_points:
        # 랜덤 위도와 경도 생성
        lon = random.uniform(polygon.bounds[0], polygon.bounds[2])
        lat = random.uniform(polygon.bounds[1], polygon.bounds[3])
        point = Point(lon, lat)
        
        # 폴리곤 내부에 있는지 확인
        if polygon.contains(point):
            points.append((lat, lon))
    return points

# 강남구의 랜덤 포인트 생성 및 출력
def generate_gangnam_points():
    # 강남구 포인트만 추출
    gangnam_points = []
    for feature in features:
        if feature['id'] == '강남구':  # 강남구의 id를 확인
            coords = feature['geometry']['coordinates']
            
            if feature['geometry']['type'] == 'Polygon':
                polygon = Polygon(coords)
                gangnam_points = generate_random_points(polygon, a)  # 49개의 포인트 생성
            
            elif feature['geometry']['type'] == 'MultiPolygon':
                for coord in coords:
                    polygon = Polygon(coord[0])
                    gangnam_points = generate_random_points(polygon, a)  # 49개의 포인트 생성
                    break  # 첫 번째 MultiPolygon만 사용
    
    latitudes, longitudes = zip(*gangnam_points)
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)
    
    # 중심 포인트를 0번으로 하고 나머지 포인트에 번호 붙이기
    node_coords = [(center_lat, center_lon)] + gangnam_points
    print(node_coords)
    return node_coords

    
def save_data_to_file(desktop_path):
    node_coords = generate_gangnam_points()
    
    lat_all = [lat for lat, _ in node_coords]
    lon_all = [lon for _, lon in node_coords]
    num_nodes = len(node_coords)
    
    # 디렉토리가 존재하는지 확인하고, 존재하지 않으면 생성
    desktop_dir = os.path.dirname(desktop_path)
    if not os.path.exists(desktop_dir):
        os.makedirs(desktop_dir)
    
    # 텍스트 파일에 추가할 데이터
    additional_data = f"""NAME: multi_modal_data
TYPE: MM

VEHICLES: 2
DIMENSION: {num_nodes}

NUM_T: 3
NUM_D: 1

MAXIMUM_SYSTEM_DURATION: 300
UNIT: MIN

SERVICETIME: 5
UNIT: MIN

TAKEOFF_LANDING_TIME: 2
UNIT: MIN

MAX_WAITING_TIME: 99999999
UNIT: MIN

INIT_SOC: 100
MAX_SOC: 100
MIN_SOC_T: 3
MIN_SOC_D: 3
UNIT: %

DEMAND_T: {(num_nodes/3)+1}
DEMAND_D: 0
UNIT: EA

MASS_T: 1940
MASS_D: 15
UNIT: KG

LIFT_TO_DRAG: 3
POWER_MOTOR_PROP: 0.5
UNIT: RATIO

POWER_ELEC: 0.1
UNIT: KW

CARGO_LIMIT_TRUCK: 99999999999999
UNIT: KG

CARGO_LIMIT_DRONE: 6
UNIT: KG

BATTERY_KWH_T: 120
BATTERY_KWH_D: 0.31
UNIT: KWH 

ENERGY_KWH/KM_T: 1.0
ENERGY_KWH/KM_D: 0.1
UNIT: KWH/KM

CHARGING_KW_D: 1.0
CHARGING_KW_T: 80
UNIT: KW 

LOSISTIC_KWH/KG_T: 0.001
LOSISTIC_KWH/KG_D: 0.001
UNIT: KWH/KG

SPEED_T: 0.5
SPEED_D: 1.0
UNIT: KM/MIN

ALTITUDE: 100
UNIT: M

CURVY_AIR: 1.1
UNIT: RATIO

CURVY_ROAD: 1.2
UNIT: RATIO

Temperature: 10.0
UNIT: degree

IDLE_T: 1.0
UNIT: KW

DEPOT_SECTION
0

EDGE_KM_D_TYPE: EUC_2D(haversine)
EDGE_KM_T_TYPE: MAN_2D(haversine)
EDGE_KM_D_FORMAT : FULL_MATRIX
EDGE_KM_T_FORMAT : FULL_MATRIX

"""
    # Add DEMAND_SECTION
    demand_section = "\nDEMAND_SECTION\n0 0\n" + "\n".join([f"{i} 1" for i in range(1, num_nodes)]) + "\n"
    
    # Add LOGISTIC_LOAD_SECTION
    logistic_load_section = "\nLOGISTIC_LOAD_SECTION\n" + "\n".join([f"{i}\t0" for i in range(num_nodes)]) + "\n"
    
    # Add AVAILABILITY_LANDING_SPOT_SECTION
    availability_landing_spot_section = "\nAVAILABILITY_LANDING_SPOT_SECTION\n" + "\n".join([f"{i}\t1" for i in range(num_nodes)]) + "\n"
    
    # Add CUSTOMER_DRONE_PREFERENCE_SECTION
    customer_drone_preference_section = "\nCUSTOMER_DRONE_PREFERENCE_SECTION\n" + "\n".join([f"{i}\t1" for i in range(num_nodes)]) + "\n"
    
    # Add PRICE_SECTION
    price_section = "\nPRICE_SECTION\n" + "\n".join([f"{i}\t3" for i in range(num_nodes)]) + "\n"
    
    # Add PRIORITY_DELIVERY_TIME_SECTION
    priority_delivery_time_section = "\nPRIORITY_DELIVERY_TIME_SECTION\n" + "\n".join([f"{i}\t0" for i in range(num_nodes)]) + "\n"
    
    with open(desktop_path, 'w', encoding='utf-8') as file:
        file.write(additional_data)
        
        file.write("NODE_COORD_SECTION\n")
        file.write(f"0\t{lat_all[0]:.8f}\t{lon_all[0]:.8f}\n")
        for i in range(1, num_nodes):
            file.write(f"{i}\t{lat_all[i]:.8f}\t{lon_all[i]:.8f}\n")
        
        file.write("\nEDGE_KM_D\n")
        
        # 유클리드 거리 행렬 계산 및 저장
        euclidean_distances_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    euclidean_distances_matrix[i, j] = haversine((lat_all[i], lon_all[i]), (lat_all[j], lon_all[j]))
        
        for i in range(num_nodes):
            file.write('\t'.join([f"{euclidean_distances_matrix[i, j]:.8f}" for j in range(num_nodes)]) + '\n')
        
        file.write("\nEDGE_KM_T\n")
        
        # 맨해튼 거리 행렬 계산 및 저장
        manhattan_distances_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    lat1, lon1 = lat_all[i], lon_all[i]
                    lat2, lon2 = lat_all[j], lon_all[j]
                    horizontal_distance = haversine((lat1, lon1), (lat1, lon2))
                    vertical_distance = haversine((lat1, lon2), (lat2, lon2))
                    manhattan_distance = horizontal_distance + vertical_distance
                    manhattan_distances_matrix[i, j] = manhattan_distance
        
        for i in range(num_nodes):
            file.write('\t'.join([f"{manhattan_distances_matrix[i, j]:.8f}" for j in range(num_nodes)]) + '\n')

        # EOF 추가
        file.write(demand_section)
        file.write(logistic_load_section)
        file.write(availability_landing_spot_section)
        file.write(customer_drone_preference_section)
        file.write(price_section)
        file.write(priority_delivery_time_section)
        file.write("\nEOF\n")

# 파일 경로 설정 및 데이터 저장
desktop_path = r"C:\Users\User\OneDrive\바탕 화면\data90.txt"
save_data_to_file(desktop_path)