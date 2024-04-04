import os
from tqdm import tqdm

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def authenticate(token_path, OAuth_key_path):
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


def list_person_folders(service, root_id):
    query = f"'{root_id}' in parents and \
              mimeType='application/vnd.google-apps.folder' and \
              trashed=false"
    response = service.files().list(q=query, fields='files(id, name)').execute()
    folders = response.get('files')
    return folders  # list[Dict[id, name]]


def find_file_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents and \
              mimeType!='application/vnd.google-apps.folder' and \
              trashed=false"
    response = service.files().list(q=query, fields='files(id, name)').execute()
    files = response.get('files')
    return files    # list[Dict[id, name]]


def download_file(service, file_id, save_to):
    request = service.files().get_media(fileId=file_id)
    with open(save_to, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    return done


def main():
    debugging = False
    token_path = 'download/token.json'
    OAuth_key_path = 'download/client_secret.json'
    data_path = 'C:/Users/heegyoon/Desktop/data/IAS/vn/raw/20240404'
    person_exist = os.listdir(data_path)
    log_path = 'download/log_download_image.txt'
    log = []
    
    creds = authenticate(token_path, OAuth_key_path)
    service = build('drive', 'v3', credentials=creds)
    
    # ID of CCCD folder
    root_id = '118ZAWAnT1wIRoxqe66bcB230FbRVn9cE'
    person_list = list_person_folders(service, root_id)
    
    # Sample
    if debugging:
        person_list = person_list[2:6]
    
    for person in tqdm(person_list):
        person_name = person['name']
        if person_name in person_exist:
            log.append(f'{person_name}, already exists.\n')
            continue
        
        person_path = os.path.join(data_path, person_name)
        os.mkdir(person_path)
        
        files = find_file_in_folder(service, person['id'])
        for file in files:  # image file
            file_id = file['id']
            file_name = file['name']
            save_path = os.path.join(person_path, file_name)
            ret = download_file(service, file_id, save_path)
            if not ret:
                log.append(f'{person_name}, {file_name}, download fail.\n')
    
    mode = 'a' if os.path.exists(log_path) else 'w'
    with open(log_path, mode, encoding='utf-8') as f:
        f.writelines(log)


if __name__ == "__main__":
    main()
    print('Done')