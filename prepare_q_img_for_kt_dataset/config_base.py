import os

dir_curr = os.path.dirname(os.path.abspath(__file__))
dir_project = os.path.dirname(dir_curr)

map_dataset_name2datas_dir = {
    "XES3G5M": "data/XES3G5M",
    "DBE_KT22": "data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv",
    "MOOCRadar": "data/MOOCRadar",
    "NIPS_task34": "data/Eedi/data"
}

map_dataset_name2datas_dir = { dataset_name:os.path.join(dir_project, datas_dir) for dataset_name, datas_dir in map_dataset_name2datas_dir.items() }

map_dataset_name2course_id = { dataset_name:i for i, dataset_name in enumerate(map_dataset_name2datas_dir.keys()) }

file_configf = "my_configs/data_config.json"
file_configf = os.path.join(dir_project, file_configf)