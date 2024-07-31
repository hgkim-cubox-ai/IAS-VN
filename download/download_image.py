import os
from tqdm import tqdm

from google_api import get_service, list_up, download_file


def TNG_Employee(service, dataset_dir, id):
    file_dict = {}
    
    # CCCD, Employee_Fake_Paper
    folder_list = sorted(list_up(service, id, False), key=lambda x: x['name'])
    for folder in folder_list:
        folder_id = folder['id']
        folder_name = folder['name']
        
        # Json folders
        if folder_name.startswith('2024'):
            continue
        
        folder_dir = os.path.join(dataset_dir, folder_name)
        os.makedirs(folder_dir, exist_ok=True)
        
        # Person folders except fake_paper
        if folder_name == 'CCCD':
            person_list = sorted(list_up(service, folder_id, False), key=lambda x: x['name'])
            for person in person_list:
                person_name = person['name']
                person_id = person['id']
                
                person_dir = os.path.join(folder_dir, person_name)
                os.makedirs(person_dir, exist_ok=True)
                        
                label_list = list_up(service, person_id, False)
                for label in label_list:
                    label_name = label['name']  # Real/Fake
                    label_id = label['id']
                    
                    label_dir = os.path.join(person_dir, label_name)
                    os.makedirs(label_dir, exist_ok=True)
                    
                    file_list = list_up(service, label_id, True)
                    for file in file_list:
                        file_id = file['id']
                        file_name = file['name']
                        save_path = os.path.join(label_dir, file_name)
                        
                        file_dict[file_id] = [
                            f'{person_name}/{label_name}/{file_name}',
                            save_path
                        ]
        # Fake paper
        elif 'paper' in folder_name.lower():
            file_list = list_up(service, folder_id, True)
            for file in file_list:
                file_id = file['id']
                file_name = file['name']
                save_path = os.path.join(folder_dir, file_name)
                
                file_dict[file_id] = [
                    f'Employee_Fake_Paper/{file_name}',
                    save_path
                ]
    
    return file_dict


def TNGo_new(service, dataset_dir, id):
    file_dict = {}
    
    # Laptop/Monitor/Paper/Phone/Real
    label_list = sorted(list_up(service, id, False), key=lambda x: x['name'])
    for label in label_list:
        label_id = label['id']
        label_name = label['name']
        
        label_dir = os.path.join(dataset_dir, label_name)
        os.mkdir(label_dir)
        
        # Images
        file_list = list_up(service, label_id, True)
        for file in file_list:
            file_id = file['id']
            file_name = file['name']
            save_path = os.path.join(label_dir, file_name)
            
            file_dict[file_id] = [
                f'{label_name}/{file_name}',
                save_path
            ]
    
    return file_dict


def TNGo_new234(service, dataset_dir, id):
    file_dict = {}
    
    label_list = sorted(list_up(service, id, False), key=lambda x: x['name'])
    for label in label_list:
        label_id = label['id']
        label_name = label['name']
        
        # Json directory
        if label_name.startswith('2024'):
            continue
        
        label_dir = os.path.join(dataset_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)
        
        # Real data
        if label_name == 'Real':            
            worker_list = sorted(list_up(service, label_id, False), key=lambda x: x['name'])
            for worker in worker_list:
                worker_id = worker['id']
                worker_name = worker['name']
                
                worker_dir = os.path.join(label_dir, worker_name)
                os.makedirs(worker_dir, exist_ok=True)
                
                file_list = list_up(service, worker_id, True)
                for file in file_list:
                    file_id = file['id']
                    file_name = file['name']
                    save_path = os.path.join(worker_dir, file_name)
                    
                    file_dict[file_id] = [
                        f'{worker_name}/{file_name}',
                        save_path
                    ]        
        # Fake
        elif 'tngo' in label_name.lower():
            file_list = sorted(list_up(service, label_id, True), key=lambda x: x['name'])
            for file in file_list:
                file_id = file['id']
                file_name = file['name']
                save_path = os.path.join(label_dir, file_name)
                
                file_dict[file_id] = [
                    f'{label_name}/{file_name}',
                    save_path
                ]
        else:
            raise ValueError(f'Wrong label name: {label_name}')
    
    return file_dict


def Integration_Test(service, dataset_dir, id):
    file_dict = {}
    
    idcard_list = sorted(list_up(service, id, False), key=lambda x: x['name'])
    for idcard in idcard_list:
        idcard_id = idcard['id']
        idcard_name = idcard['name']
        
        idcard_dir = os.path.join(dataset_dir, idcard_name)
        os.makedirs(idcard_dir, exist_ok=True)
        
        file_list = sorted(list_up(service, idcard_id, True), key=lambda x: x['name'])
        for file in file_list:
            file_id = file['id']
            file_name = file['name']
            save_path = os.path.join(idcard_dir, file_name)
            
            file_dict[file_id] = [
                f'{idcard_name}/{file_name}',
                save_path
            ]

    return file_dict


def main():
    # Set service
    token_path = 'download/token.json'
    OAuth_key_path = 'download/client_secret.json'
    service = get_service(token_path, OAuth_key_path)
    
    # Root directory
    root_dir = 'C:/Users/heegyoon/Desktop/data/IAS/vn/raw'
    os.makedirs(root_dir, exist_ok=True)
    
    dataset_dict = {
        # 'TNG_Employee': {
        #     'fn': TNG_Employee, 'id': '1aXqx-bYC-Z3aPwtcrSH1g1maAVCKUvPP'
        # },
        # 'TNGo_new': {
        #     'fn': TNGo_new, 'id': '1j-PkGIExdAd7t8iI2MAYRz1R4ySQHDRI'
        # },
        # 'TNGo_new2': {
        #     'fn': TNGo_new234, 'id': '1-RFnI2gQsfz1ataWltjqozAnkKnOQmRL'
        # },
        # 'TNGo_new3': {
        #     'fn': TNGo_new234, 'id': '1989_NGBpdq2KmbexiEVEUlvJi4_oljij'
        # },
        # 'TNGo_new4': {
        #     'fn': TNGo_new234, 'id': '1uDtPgXSPkFWcKP-X2OVLtbTSasL7wGNJ'
        # },
        # 'Integration_Test': {
        #     'fn': Integration_Test, 'id': '1XKA66_zuQ_S_LqoraliMyjHmzGNxWEWQ'
        # }
        'Integration_test_1': {
            'fn': Integration_Test, 'id': '1XKA66_zuQ_S_LqoraliMyjHmzGNxWEWQ'
        },
        'Integration_test_2': {
            'fn': Integration_Test, 'id': '1Np0J2kCWWY0GaogEDS2k8Vxf9fDlyZdp'
        }
    }
    
    for dataset_name, info in dataset_dict.items():
        log = []
        dataset_dir = os.path.join(root_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        file_dict = info['fn'](service, dataset_dir, info['id'])
        for file_id, path in tqdm(file_dict.items(), desc=f'{dataset_name}'):
            # path[0]: path from google drive, path[1]: save_path
            ret = download_file(service, file_id, path[1])
            if not ret:
                log.append(f'{path[0]}, download fail.\n')
            
        log_path = os.path.join(root_dir, dataset_name, 'log_download.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.writelines(log)


if __name__ == "__main__":
    main()
    print('Done')
    
    # python download/download.py