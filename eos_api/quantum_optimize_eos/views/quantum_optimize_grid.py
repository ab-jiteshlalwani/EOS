from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from requests.exceptions import RequestException
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.throttling import UserRateThrottle
from pathlib import Path

import json
import ast
from dimod import Binary, Real
from dimod import ConstrainedQuadraticModel, ExactCQMSolver
from dwave.system import LeapHybridCQMSampler
from itertools import chain
import copy
import time
import warnings

warnings.filterwarnings('ignore')


@csrf_exempt
@throttle_classes([UserRateThrottle])
def quantum_optimize_grid(request):
    if request.method == 'POST':
        try:
            received_json_data = json.loads(request.body.decode("utf-8"))
            auth_header = request.META['HTTP_AUTHORIZATION']

            BASE_DIR = Path(__file__).resolve().parent.parent

            with open(str(BASE_DIR) + '/views/bearer.txt', 'r') as f:
                BEARER_TOKEN = f.read().strip()

            with open(str(BASE_DIR) + '/views/api_key.txt', 'r') as f:
                API_KEY = f.read().strip()

            if auth_header != BEARER_TOKEN:
                response = {"Results": "Missing or incorrect Bearer Token"}
            else:
                input_json = received_json_data['input_json']
                response = {"Results": optimize_grid_using_quantum(input_json, API_KEY)}

            return JsonResponse(response)
        except RequestException as request_error:
            response = {"error": str(request_error)}
            return JsonResponse(response)
        except KeyError as key_error:
            response = {"error": str(key_error) + ' is missing'}
            return JsonResponse(response)


def optimize_grid_using_quantum(input_json, API_KEY):
    output_json_array = []

    for input in input_json:
        fossil_fuels = input['fossil_fuels']
        capacity_fossil_fuels = input['capacity_fossil_fuels']
        operating_costs_fossil_fuels = input['operating_costs_fossil_fuels']

        solar = input['solar']
        capacity_solar = input['capacity_solar']
        operating_costs_solar = input['operating_costs_solar']

        wind = input['wind']
        capacity_wind = input['capacity_wind']
        operating_costs_wind = input['operating_costs_wind']

        battery = input['battery']
        capacity_battery = input['capacity_battery']
        operating_costs_battery = input['operating_costs_battery']
        charging_rates = input['charging_rates']
        discharging_rates = input['discharging_rates']
        efficiencies_charging = input['efficiencies_charging']
        efficiencies_discharging = input['efficiencies_discharging']

        demand = input['demand']
        r_capacities = input['r_capacities']

        # pre processing code
        len_fossil_fuels = len(fossil_fuels)
        len_solar = len(solar)
        len_wind = len(wind)
        len_battery = len(battery)
        n = len_fossil_fuels + len_solar + len_wind + len_battery
        capacities = capacity_fossil_fuels + capacity_solar + capacity_wind + capacity_battery
        operating_costs = operating_costs_fossil_fuels + operating_costs_solar + operating_costs_wind + operating_costs_battery
        name = fossil_fuels + solar + wind + battery
        solar_ = len_fossil_fuels
        wind_ = len_fossil_fuels + len_solar
        battery_ = len_fossil_fuels + len_solar + len_wind
        output_json = quantum(demand, r_capacities, n, capacities, operating_costs, name, solar_, wind_, battery_,
                              charging_rates, discharging_rates,
                              efficiencies_charging, efficiencies_discharging, API_KEY)
        output_json_array.append(output_json)
    return output_json_array


def quantum(demand, r_capacities, n, capacities, operating_costs, name, solar_, wind_, battery_,
            charging_rates, discharging_rates, efficiencies_charging, efficiencies_discharging, API_KEY):
    cqm = ConstrainedQuadraticModel()

    variables = []
    operating_costs_var = []
    for i in range(n):
        variables.append(Real(f'x{i}', lower_bound=0, upper_bound=capacities[i]))

    for i in range(len(variables)):
        operating_costs_var.append(operating_costs[i] * variables[i])

    battery_var = variables[0:battery_]

    for i in range(battery_, n):
        battery_var.append(-1 * variables[i])

    # Define the objective function
    cqm.set_objective(sum(operating_costs_var))

    cqm.add_constraint(sum(battery_var) == demand, label=f"Energy balance constraint at time t")

    for i in range(n):
        cqm.add_constraint(variables[i] <= capacities[i], label=f"Capacity constraint for power plant {i} at time t")

    cqm.add_constraint(sum(variables[solar_:battery_]) == r_capacities, label=f"Renewable Energy constraint at time t")

    for i in range(battery_, n):
        cqm.add_constraint(variables[i] <= efficiencies_charging[battery_ - i] * charging_rates[battery_ - i],
                           label=f"charging constraints {i}")

    for i in range(battery_, n):
        cqm.add_constraint(variables[i] <= efficiencies_discharging[battery_ - i] * discharging_rates[battery_ - i],
                           label=f"discharging constraints{i}")

    # Solve the optimization problem
    sampler = LeapHybridCQMSampler(token=API_KEY)
    sampleset = sampler.sample_cqm(cqm)
    # sampleset = ExactCQMSolver().sample_cqm(cqm)
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    result_sum = 0
    plant_name_q, energy_q = [], []
    for i in feasible_sampleset.first.sample.items():
        if float(i[1]) > 0:
            result_sum = result_sum + (operating_costs[int(i[0].replace("x", ""))] * float(i[1]))
            plant_name_q.append(name[int(i[0].replace("x", ""))])
            energy_q.append(float(i[1]))

    return {
        "Energy Source": plant_name_q,
        "Energy Amount": energy_q,
        "Total": result_sum
    }
