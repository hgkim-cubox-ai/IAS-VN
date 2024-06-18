import os
from tqdm import tqdm

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


def authenticate(token_path, OAuth_key_path):
    # If modifying these scopes, delete the file token.json.
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                OAuth_key_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return creds


def list_up(service, folder_id, is_file):
    query = f"'{folder_id}' in parents and trashed=false and "
    if is_file:
        query += "mimeType!='application/vnd.google-apps.folder'"
    else:   # folder
        query += "mimeType='application/vnd.google-apps.folder'"
    page_token = None
    files = []
    
    while True:
        response = service.files().list(
            q=query,
            fields='nextPageToken, files(id, name)',
            pageSize=100,  # 각 페이지에 100개씩 가져옴
            pageToken=page_token
        ).execute()

        files.extend(response.get('files', []))

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    
    return files    # list[Dict[id, name]]


def download_file(service, file_id, save_to):
    request = service.files().get_media(fileId=file_id)
    with open(save_to, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
    return done


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
    

if __name__ == "__main__":
    # Set service
    token_path = 'download/token.json'
    OAuth_key_path = 'download/client_secret.json'
    creds = authenticate(token_path, OAuth_key_path)
    service = build('drive', 'v3', credentials=creds)
    
    data_dict = {
        'TNG_Employee': {},
        'TNGo_new': {},
        'TNGo_new2': {},
        'TNGo_new3': {},
        'TNGo_new4': {}
    }
    
    # TNGo_new_data(service)
    # TNG_Employee_data(service)
    # TNGo_new2_data(service)
    TNGo_new3_data(service)
    print('Done')