import os
from tqdm import tqdm

from google_api import get_service, list_up, download_file


def TNG_Employee_data(service):
    root_path = 'C:/Users/heegyoon/Desktop/data/IAS/vn/raw/TNG_Employee'
    os.mkdir(root_path)
    log_path = os.path.join(root_path, 'log_download_TNG_Employee.txt')
    log = []
    
    CCCD_id = '1OS6DW5SVGS5BjAgEOHew6HcVr2rlu9wb'
    paper_id = '1m39ssjJq5nJxMRfIYBH-fWKIE7lI97Za'
    
    # Download in person folder except fake_paper
    person_list = sorted(list_up(service, CCCD_id, False), key=lambda x: x['name'])
    for person in tqdm(person_list):
        person_name = person['name']
        person_id = person['id']
        
        person_dir = os.path.join(root_path, person_name)
        os.mkdir(person_dir)
        
        spoof_labels = list_up(service, person_id, False)
        for label in spoof_labels:
            label_name = label['name']  # Real/Fake
            label_id = label['id']
            label_dir = os.path.join(person_dir, label_name)
            os.mkdir(label_dir)
            
            files = list_up(service, label_id, True)
            for file in files:
                file_id = file['id']
                file_name = file['name']
                save_path = os.path.join(label_dir, file_name)
                ret = download_file(service, file_id, save_path)
                if not ret:
                    log.append(f'{person}, {file_name}, download fail.\n')
    
    # Download fake_paper
    fake_dir = os.path.join(root_path, 'fake_paper')
    os.mkdir(fake_dir)
    files = list_up(service, paper_id, True)
    for file in files:
        file_id = file['id']
        file_name = file['name']
        save_path = os.path.join(fake_dir, file_name)
        ret = download_file(service, file_id, save_path)
        if not ret:
            log.append(f'fake_paper, {file_name}, download fail.\n')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines(log)


def TNGo_new_data(service):
    root_path = 'C:/Users/heegyoon/Desktop/data/IAS/vn/raw/TNGo_new'
    os.mkdir(root_path)
    log_path = os.path.join(root_path, 'log_download_TNGo_new.txt')
    log = []
        
    # ID of TNGo_new/CCCD
    root_id = '1j-PkGIExdAd7t8iI2MAYRz1R4ySQHDRI'
    
    # Laptop/Monitor/Paper/Phone/Real
    dir_list = sorted(list_up(service, root_id, False), key=lambda x: x['name'])
    for dir in dir_list:
        dir_id = dir['id']
        dir_name = dir['name']
        dir_path = os.path.join(root_path, dir_name)
        os.mkdir(dir_path)
        
        # Images
        file_list = list_up(service, dir_id, True)
        for file in tqdm(file_list, dir_name):
            file_id = file['id']
            file_name = file['name']
            save_path = os.path.join(dir_path, file_name)
            ret = download_file(service, file_id, save_path)
            if not ret:
                log.append(f'{dir_name}/{file_name}, download fail\n')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines(log)


def TNGo_new2_data(service):
    save_dir = 'C:/Users/heegyoon/Desktop/data/IAS/vn/raw/TNGo_new2'
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log_download_TNGo_new2.txt')
    log = []
    
    root_id = '1-RFnI2gQsfz1ataWltjqozAnkKnOQmRL'
    
    # Real data
    real_id = '1zcC_HiqTFgESFHYRgfO277Y31pbYwutq'
    real_dir = os.path.join(save_dir, 'Real')
    os.makedirs(real_dir, exist_ok=True)
    worker_list = sorted(list_up(service, real_id, False), key=lambda x: x['name'])
    for worker in worker_list:
        worker_id = worker['id']
        worker_name = worker['name']
        worker_dir = os.path.join(real_dir, worker_name)
        os.mkdir(worker_dir)
        
        file_list = list_up(service, worker_id, True)
        for file in tqdm(file_list, worker_name):
            file_id = file['id']
            file_name = file['name']
            save_path = os.path.join(worker_dir, file_name)
            ret = download_file(service, file_id, save_path)
            if not ret:
                log.append(f'{worker_name}/{file_name}, download fail.\n')
    
    # Others
    fake_list = sorted(list_up(service, root_id, False), key=lambda x: x['name'])
    for fake in fake_list:
        fake_id = fake['id']
        fake_name = fake['name']
        if not fake_name.startswith('Tngo'): # label, Real directory
            continue
        fake_dir = os.path.join(save_dir, fake_name)
        os.mkdir(fake_dir)
        
        file_list = list_up(service, fake_id, True)
        for file in tqdm(file_list, fake_name):
            file_id = file['id']
            file_name = file['name']
            save_path = os.path.join(fake_dir, file_name)
            ret = download_file(service, file_id, save_path)
            if not ret:
                log.append(f'{fake_name}/{file_name}, download fail.\n')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines(log)


def TNGo_new3_data(service):
    save_dir = 'C:/Users/heegyoon/Desktop/data/IAS/vn/raw/TNGo_new3'
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log_download_TNGo_new3.txt')
    log = []
    
    root_id = '1989_NGBpdq2KmbexiEVEUlvJi4_oljij'
    dir_list = sorted(list_up(service, root_id, False), key=lambda x: x['name'])
    for dir in dir_list:
        dir_id = dir['id']
        dir_name = dir['name']
        
        if dir_name.startswith('2024'):
            continue
        
        # Real data
        if dir_name == 'Real':
            real_dir = os.path.join(save_dir, 'Real')
            os.makedirs(real_dir, exist_ok=True)
            
            worker_list = sorted(list_up(service, dir_id, False), key=lambda x: x['name'])
            for worker in worker_list:
                worker_id = worker['id']
                worker_name = worker['name']
                worker_dir = os.path.join(real_dir, worker_name)
                os.mkdir(worker_dir)
                
                file_list = list_up(service, worker_id, True)
                for file in tqdm(file_list, worker_name):
                    file_id = file['id']
                    file_name = file['name']
                    save_path = os.path.join(worker_dir, file_name)
                    ret = download_file(service, file_id, save_path)
                    if not ret:
                        log.append(f'{worker_name}/{file_name}, download fail.\n')
        
        # Fake
        elif 'tngo' in dir_name:
            fake_dir = os.path.join(save_dir, dir_name)
            os.makedirs(fake_dir, exist_ok=True)
            file_list = sorted(list_up(service, dir_id, True), key=lambda x: x['name'])
            for file in tqdm(file_list, dir_name):
                file_id = file['id']
                file_name = file['name']
                save_path = os.path.join(fake_dir, file_name)
                ret = download_file(service, file_id, save_path)
                if not ret:
                    log.append(f'{dir_name}/{file_name}, download fail.\n')

    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines(log)


def TNG_Employee_dataset(service, dataset_dir):
    file_dict = {}
    
    # Person directory except fake_paper
    CCCD_id = '1OS6DW5SVGS5BjAgEOHew6HcVr2rlu9wb'
    person_list = sorted(list_up(service, CCCD_id, False), key=lambda x: x['name'])
    for person in person_list:
        person_name = person['name']
        person_id = person['id']
        
        person_dir = os.path.join(dataset_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
                
        label_list = list_up(service, person_id, False)
        for label in label_list:
            label_name = label['name']  # Real/Fake
            label_id = label['id']
            
            label_dir = os.path.join(person_dir, label_name)
            os.makedirs(label_dir)
            
            file_list = list_up(service, label_id, True)
            for file in file_list:
                file_id = file['id']
                file_name = file['name']
                save_path = os.path.join(dataset_dir, person_name, label_name, file_name)
                
                file_dict[file_id] = [
                    f'{person_name}/{label_name}/{file_name}',
                    save_path
                ]

    # Fake_paper
    paper_id = '1m39ssjJq5nJxMRfIYBH-fWKIE7lI97Za'
    paper_dir = os.path.join(dataset_dir, 'Employee_Fake_Paper')
    os.makedirs(paper_dir, exist_ok=True)
    
    file_list = list_up(service, paper_id, True)
    for file in file_list:
        file_id = file['id']
        file_name = file['name']
        save_path = os.path.join(paper_dir, file_name)
        
        file_dict[file_id] = [
            f'Employee_Fake_Paper/{file_name}',
            save_path
        ]
    
    return file_dict


def TNGo_new_dataset(service, dataset_dir):
    file_dict = {}
    dataset_id = '1j-PkGIExdAd7t8iI2MAYRz1R4ySQHDRI'
    
    # Laptop/Monitor/Paper/Phone/Real
    label_list = sorted(list_up(service, dataset_id, False), key=lambda x: x['name'])
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


def TNGo_new2_dataset(service, dataset_dir):
    file_dict = {}
    
    # Real data
    real_id = '1zcC_HiqTFgESFHYRgfO277Y31pbYwutq'
    real_dir = os.path.join(dataset_dir, 'Real')
    os.makedirs(real_dir, exist_ok=True)
    
    worker_list = sorted(list_up(service, real_id, False), key=lambda x: x['name'])
    for worker in worker_list:
        worker_id = worker['id']
        worker_name = worker['name']
        
        worker_dir = os.path.join(real_dir, worker_name)
        os.mkdir(worker_dir)
        
        file_list = list_up(service, worker_id, True)
        for file in file_list:
            file_id = file['id']
            file_name = file['name']
            save_path = os.path.join(worker_dir, file_name)
            
            file_dict[file_id] = [
                f'{worker_name}/{file_name}',
                save_path
            ]
    
    # Others
    root_id = '1-RFnI2gQsfz1ataWltjqozAnkKnOQmRL'
    fake_list = sorted(list_up(service, root_id, False), key=lambda x: x['name'])
    for fake in fake_list:
        fake_id = fake['id']
        fake_name = fake['name']
        if not fake_name.startswith('Tngo'): # label, Real directory
            continue
        
        fake_dir = os.path.join(dataset_dir, fake_name)
        os.mkdir(fake_dir)
        
        file_list = list_up(service, fake_id, True)
        for file in file_list:
            file_id = file['id']
            file_name = file['name']
            save_path = os.path.join(fake_dir, file_name)
            
            file_dict[file_id] = [
                f'{fake_name}/{file_name}',
                save_path
            ]
    
    return file_dict


def TNGo_new3_dataset(service, dataset_dir):
    file_dict = {}
    
    root_id = '1989_NGBpdq2KmbexiEVEUlvJi4_oljij'
    label_list = sorted(list_up(service, root_id, False), key=lambda x: x['name'])
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
                for file in file_list, worker_name:
                    file_id = file['id']
                    file_name = file['name']
                    save_path = os.path.join(worker_dir, file_name)
                    
                    file_dict[file_id] = [
                        f'{worker_name}/{file_name}',
                        save_path
                    ]        
        # Fake
        elif 'tngo' in label_name:
            file_list = sorted(list_up(service, label_id, True), key=lambda x: x['name'])
            for file in file_list, label_name:
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



def main():
    # Set service
    token_path = 'download/token.json'
    OAuth_key_path = 'download/client_secret.json'
    service = get_service(token_path, OAuth_key_path)
    
    # Root directory
    root_dir = 'C:/Users/heegyoon/Desktop/data/IAS/vn/raw2'
    
    dataset_dict = {
        # 'TNG_Employee': TNG_Employee_dataset,
        # 'TNGo_new': TNGo_new_dataset,
        # 'TNGo_new2': TNGo_new2_dataset,
        'TNGo_new3': TNGo_new3_dataset,
        # 'TNGo_new4': 
    }
    
    for dataset_name, func in dataset_dict.items():
        log = []
        dataset_dir = os.path.join(root_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        file_dict = func(service, dataset_dir)
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