import os
import time
from httplib2 import Http

import google_auth_httplib2
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import HttpError


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


def download_file(service, file_id, save_to, max_retries=5):
    request = service.files().get_media(fileId=file_id)
    with open(save_to, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        retries = 0
        while done is False:
            try:
                _, done = downloader.next_chunk()
            except HttpError as e:
                if retries < max_retries:
                    retries += 1
                    print(f"Retrying... ({retries}/{max_retries})")
                    time.sleep(2 ** retries)  # Exponential backoff
                else:
                    print(f"Failed to download file after {max_retries} attempts.")
                    raise e
    return done


def get_service(token_path, OAuth_key_path):
    creds = authenticate(token_path, OAuth_key_path)
    http = google_auth_httplib2.AuthorizedHttp(creds, Http(timeout=60))
    # http = Http(timeout=60)
    service = build('drive', 'v3', http=http)
    # service = build('drive', 'v3', credentials=creds)
    return service