from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from requests.exceptions import RequestException
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.throttling import UserRateThrottle
from pathlib import Path

import json
import ast
import neal
import dimod
from dimod import Binary, Real
import numpy as np
from dimod import ConstrainedQuadraticModel, ExactCQMSolver
from dwave.system import LeapHybridCQMSampler
from itertools import chain
import copy
import copy
import time
from pulp import LpVariable, LpProblem, lpSum, LpMaximize, LpMinimize, LpStatus, listSolvers, getSolver
import warnings

warnings.filterwarnings('ignore')


@csrf_exempt
@throttle_classes([UserRateThrottle])
def quantum_optimize_eos_schedule(request):
    if request.method == 'POST':
        try:
            received_json_data = json.loads(request.body.decode("utf-8"))

            auth_header = request.META['HTTP_AUTHORIZATION']

            BASE_DIR = Path(__file__).resolve().parent.parent

            with open(str(BASE_DIR) + '/views/bearer.txt', 'r') as f:
                BEARER_TOKEN = f.read().strip()

            if auth_header is None or auth_header != BEARER_TOKEN:
                response = {"Results": "Missing or incorrect Bearer Token"}
            else:
                input_json = received_json_data['input_json']
                # print(request)
                response = optimize_eos_using_quantum(input_json)
                quantum_results_json = {"Results": response}
                return JsonResponse(quantum_results_json)
            return JsonResponse(response)
        except RequestException as request_error:
            response = {"error": str(request_error)}
            return JsonResponse(response)
        except KeyError as key_error:
            response = {"error": str(key_error) + ' key is missing'}
            return JsonResponse(response)


def optimize_eos_using_quantum(input_json):
    output_json_array = []

    for input in input_json:
        targets = [int(target.replace('T', '')) for target in ast.literal_eval(input['Targets'])]
        priorities = [priority for priority in ast.literal_eval(input['Priorities'])]
        sizes = [priority for priority in ast.literal_eval(input['Size of Targets'])]
        energies = [energy for energy in ast.literal_eval(input['Energies of Targets'])]
        ground_stations = [int(gs.replace('GS', '')) for gs in ast.literal_eval(input['Ground Stations'])]
        gs_limit = [-gs_limit for gs_limit in ast.literal_eval(input['Limit of Ground Stations'])]
        s1 = convert_gs_to_gs_limit([s1 for s1 in ast.literal_eval(input['Satellite 1'])],
                                    ground_stations, gs_limit)
        s2 = convert_gs_to_gs_limit([s1 for s1 in ast.literal_eval(input['Satellite 2'])],
                                    ground_stations, gs_limit)
        s3 = convert_gs_to_gs_limit([s1 for s1 in ast.literal_eval(input['Satellite 3'])],
                                    ground_stations, gs_limit)
        energy_limit = [energy_limit for energy_limit in ast.literal_eval(input['Energy Limit of Satellites'])]
        storage_limit = [storage_limit for storage_limit in ast.literal_eval(input['Storage Limit of Satellites'])]

        s1_, gs1_ = separate_gs_and_fix_non_gs_targets(s1)

        s2_, gs2_ = separate_gs_and_fix_non_gs_targets(s2)

        s3_, gs3_ = separate_gs_and_fix_non_gs_targets(s3)

        size_dict = dict(zip(targets, sizes))

        priorities_dict = dict(zip(targets, priorities))

        energies_dict = dict(zip(targets, energies))

        size1_list = [[size_dict[key] for key in sublist] for sublist in s1_]
        # print("size1_list",size1_list)

        size2_list = [[size_dict[key] for key in sublist] for sublist in s2_]
        # print("size2_list",size2_list)

        size3_list = [[size_dict[key] for key in sublist] for sublist in s3_]
        # print("size3_list",size3_list)

        priorities1_list = [[priorities_dict[key] for key in sublist] for sublist in s1_]
        # print("priorities1_list",priorities1_list)

        priorities2_list = [[priorities_dict[key] for key in sublist] for sublist in s2_]
        # print("priorities2_list",priorities2_list)

        priorities3_list = [[priorities_dict[key] for key in sublist] for sublist in s3_]
        # print("priorities3_list",priorities3_list)

        energies1_list = [[energies_dict[key] for key in sublist] for sublist in s1_]
        # print("energies1_list",energies1_list)

        energies2_list = [[energies_dict[key] for key in sublist] for sublist in s2_]
        # print("energies2_list",energies2_list)

        energies3_list = [[energies_dict[key] for key in sublist] for sublist in s3_]
        # print("energies3_list",energies3_list)

        intial_target = [s1_, s2_, s3_]
        # print("Targets = ",intial_target)

        target_dict = {}
        c = 0
        target_ = []
        for i in intial_target:
            j_ = []
            for j in i:
                k_ = []
                for k in j:
                    k_.append(c)
                    target_dict[c] = k
                    c = c + 1
                j_.append(k_)
            target_.append(j_)

        size_ = [size1_list, size2_list, size3_list]
        priority_ = [priorities1_list, priorities2_list, priorities3_list]
        energy_ = [energies1_list, energies2_list, energies3_list]

        energy_limit_ = energy_limit
        # energy limit for each satellite
        assert len(energy_) == len(energy_limit_)

        dl_ = [gs1_, gs2_, gs3_]
        dl_copy = copy.deepcopy(dl_)
        # downlink size limit in satellites

        assert len(dl_) == len(target_)
        dl_sizes = [len(i) for i in dl_]
        quadrent_sizes = [len(i) for i in target_]
        assert dl_sizes == quadrent_sizes

        gs_ = storage_limit
        # satellite storge in satellites

        assert len(gs_) == len(target_)

        name = 0

        q_targets = list(chain(*list(chain(*intial_target))))
        targets = list(chain(*list(chain(*target_))))
        sizes = list(chain(*list(chain(*size_))))
        priority = list(chain(*list(chain(*priority_))))
        energy = list(chain(*list(chain(*energy_))))

        assert len(targets) == len(sizes)
        assert len(priority) == len(targets)
        assert len(energy) == len(sizes)

        dl_ = [[-1 * j for j in i] for i in dl_]

        converted_dict = {}
        for key, value in target_dict.items():
            if value in converted_dict:
                converted_dict[value].append(key)
            else:
                converted_dict[value] = [key]

        keys = [k for k, v in converted_dict.items() if len(v) > 1]

        options_ = []
        for key in keys:
            options_.append(converted_dict[key])

        cqm = ConstrainedQuadraticModel()
        targets_included = []
        for j in range(len(priority)):
            targets_included.append(-priority[j] * Binary(f't{targets[j]}'))
        cqm.set_objective(sum(targets_included))
        target_result = [(k) * Binary(f't{j}') for j, k in zip(range(len(targets)), sizes)]

        energy_array = [(k) * Binary(f't{j}') for j, k in zip(range(len(targets)), energy)]

        for target, size, dl, gs, energy_limit in zip(target_, size_, dl_, gs_, energy_limit_):
            count = target[0][0]
            cons, dl_limits = constraints_func(dl, gs)
            cons.reverse()
            counts = []
            for i in zip(target, cons):
                cqm.add_constraint(sum(target_result[count:count + len(i[0])]) <= i[1],
                                   label=f'per q {name} {count}:{count + len(i[0])}<={i[1]}')
                name = name + 1
                counts.append(count)
                count = count + len(i[0])

            dl_limits.reverse()

            for i in zip(counts[:-1], dl_limits[:-1]):
                cqm.add_constraint(sum(target_result[i[0]:count]) <= i[1], label=f'per q {name} {i[0]}:{count}<={i[1]}')
                name = name + 1

            cqm.add_constraint(sum(energy_array[target[0][0]:count]) <= energy_limit,
                               label=f'{name} energy limit {target[0][0]}:{count} {energy_limit}')

        for i in range(len(options_)):
            cqm.add_constraint(sum([Binary(f't{j}') for j in options_[i]]) <= 1, label=f'common {i} {options_[i]}')

        qstart = time.time()

        sampler = LeapHybridCQMSampler(token="DEV-a837e60da7c343e9e89595965e4d264c5db6c402")
        sampleset = sampler.sample_cqm(cqm)
        # sampleset = ExactCQMSolver().sample_cqm(cqm)
        feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
        result = []
        for i in feasible_sampleset.first.sample.items():
            if i[1] == 1.0:
                result.append(int(i[0].replace("t", "")))
        result.sort()
        qend = time.time()
        q_time = (qend - qstart) * 10 ** 3

        quantum_out_json_inter = post(result, converted_dict, size_dict, priorities_dict, energies_dict, target_, gs1_,
                                      gs2_, gs3_, q_targets,
                                      sizes, energy,
                                      energy_limit_, q_time)

        cstart = time.time()

        bqm, invert = dimod.cqm_to_bqm(cqm)
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, seed=1234,
                                   beta_range=[0.1, 4.2],
                                   num_sweeps=20,
                                   beta_schedule_type='geometric')
        data = sampleset.first.sample

        sa = []
        for i in targets:
            if (data[f"t{i}"]) == 1:
                sa.append(i)

        cend = time.time()
        c_time = (cend - cstart) * 10 ** 3

        simulated_out_json_inter = post(sa, converted_dict, size_dict, priorities_dict, energies_dict, target_, gs1_,
                                        gs2_, gs3_, q_targets, sizes,
                                        energy, energy_limit_, c_time)

        output_json = {"Quantum Result": quantum_out_json_inter["Result"],
                       "Quantum Total Priority": quantum_out_json_inter["Total Priority"],
                       "Quantum Execution Time": quantum_out_json_inter["Execution Time"],
                       "Quantum Satisfied all Constraints?": quantum_out_json_inter["Satisfied all Constraints?"],
                       "Quantum Constraints Not Satisfied": quantum_out_json_inter["Constraints Not Satisfied"],
                       "Simulated Annealing Result": simulated_out_json_inter["Result"],
                       "Simulated Annealing Total Priority": simulated_out_json_inter["Total Priority"],
                       "Simulated Annealing Execution Time": simulated_out_json_inter["Execution Time"],
                       "Simulated Annealing Satisfied all Constraints?": simulated_out_json_inter[
                           "Satisfied all Constraints?"],
                       "Simulated Annealing Constraints Not Satisfied": simulated_out_json_inter[
                           "Constraints Not Satisfied"],
                       }
        output_json_array.append(output_json)
    return output_json_array


# create constraints values for sizes, based on different limits
def constraints_func(dl, gs):
    dl_limits, cons = [], []
    c = 0
    dl.reverse()
    for i in dl:
        c = c + i
        dl_limits.append(c)
        if c > gs:
            cons.append(gs)
        else:
            cons.append(c)
    return cons, dl_limits


# measure sum total of priority,size etc
def total(sizes, result):
    sum_size = []
    for i in result:
        sum_size.append(sizes[i])

    return sum(sum_size)


# data manupilation
def correct(lst):
    for i in range(int(len(lst) / 2)):
        lst = add_negative_pairs(lst)
    return lst


# takes series for targets and gs, returns target list and gs list well seprated with its limits
def separate_gs_and_fix_non_gs_targets(s):
    if s[-1] > 0:
        t = []
        for element in reversed(s):
            if element < 0:
                break
            t.append(element)

        s = t[::-1] + s

    s = correct(s)
    s_, temp, gs_ = [], [], []
    for i in s:
        if i < 0:
            gs_.append(i)
            s_.append(temp)
            temp = []
        else:
            temp.append(i)

    if not s_[0]:
        s_ = s_[1:]
        gs_ = gs_[1:]

    return s_, gs_


def is_equal_or_less(list1, list2):
    return all(x >= y for x, y in zip(list1, list2))


# post processing and mapping the results in quadrents and lower level checks on storage and energy
def post(res, converted_dict, size_dict, priorities_dict, energies_dict, target_, gs1_, gs2_, gs3_,
         q_targets, sizes, energy, energy_limit_, a_time):
    parsed_results = []
    total_size = 0
    priority_results = 0
    total_energy = 0
    conditions_satisfied = True
    unsatisfied_conditions = []
    for j in res:
        for i in converted_dict.items():
            if j in i[1]:
                parsed_results.append(i[0])
                # print(size_dict[i[0]])
                total_size = total_size + size_dict[i[0]]
                priority_results = priority_results + priorities_dict[i[0]]
                total_energy = total_energy + energies_dict[i[0]]

    i_, isize, iq, ienergy, sum_s, sum_e, sum_size, sum_energy = [], [], [], [], [], [], [], []
    for i in target_:
        j_, jsize, jq, je, um_s, um_e = [], [], [], [], [], []
        for j in i:
            k_, ksize, kq, ke = [], [], [], []
            for k in j:
                if k in res:
                    k_.append(k)
                    kq.append(q_targets[k])
                    ksize.append(sizes[k])
                    ke.append(energy[k])
            j_.append(k_)
            jq.append(kq)
            jsize.append(ksize)
            je.append(ke)
            um_s.append(sum(ksize))
            um_e.append(sum(ke))

        i_.append(j_)
        iq.append(jq)
        isize.append(jsize)
        ienergy.append(je)
        sum_s.append(um_s)
        sum_e.append(um_e)
        sum_size.append(sum(um_s))
        sum_energy.append(sum(um_e))

    gs_list = [gs1_, gs2_, gs3_]
    for i, j, k in zip(gs_list, sum_s, sum_size):
        if k > (-1 * sum(i)):
            print("Storage total is beyond", k, -1 * sum(i))
        if j[-1] > (-1 * i[-1]):
            conditions_satisfied = False
            if "Storage limit not satisfied" not in unsatisfied_conditions:
                unsatisfied_conditions.append("Storage limit not satisfied")

    if not is_equal_or_less(energy_limit_, sum_energy):
        conditions_satisfied = False
        if "Energy limit not satisfied" not in unsatisfied_conditions:
            unsatisfied_conditions.append("Energy limit not satisfied")

    out_json_inter = {"Result": res, "Total Priority": priority_results, "Execution Time": str(a_time) + " ms",
                      "Satisfied all Constraints?": conditions_satisfied,
                      "Constraints Not Satisfied": unsatisfied_conditions}
    return out_json_inter


def add_negative_pairs(lst):
    # base case: if the list is empty, return an empty list
    if not lst:
        return []

    # if the first element is positive, keep it and recurse on the rest of the list
    if lst[0] >= 0:
        return [lst[0]] + add_negative_pairs(lst[1:])

    # if the first two elements are negative, add them and recurse on the rest of the list
    if len(lst) > 1 and lst[1] < 0:
        return [lst[0] + lst[1]] + add_negative_pairs(lst[2:])

    # if the first element is negative and the second is positive, keep the negative element and recurse on the rest of the list
    return [lst[0]] + add_negative_pairs(lst[1:])


def convert_gs_to_gs_limit(satellite, ground_stations, gs_limit):
    satellite_coverage = []
    for sat in satellite:
        if isinstance(sat, int):
            satellite_coverage.append(sat)
        else:
            gs_index = ground_stations.index(int(sat.replace('GS', '')))
            satellite_coverage.append(gs_limit[gs_index])
    return satellite_coverage
