# import copy
# import json
#
# file_a = './gpt-4_15.json'
#
# with open(file_a, 'r') as fin:
#     data_a = json.load(fin)
#
# file_b = './gpt-4_15_60_outof_106.json'
#
# with open(file_b, 'r') as fin:
#     data_b = json.load(fin)
#
# data_c = {'results': copy.deepcopy(data_b['results'])}
# for item in data_a['results']:
#     data_c['results'].append(item)
#
# assert len(data_c['results']) == len(data_b['results']) + len(data_a['results'])
# with open('gpt-4_15.json', 'w') as fout:
#     json.dump(data_c, fout)
