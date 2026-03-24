import requests
import json

URL = "http://aitask.cbsmediahub.com/transcriptions.asp"

def get_transcription(job_id):
    
    try:
        response = requests.get(URL, params={"job_id": job_id})
        response.raise_for_status() 
        json_data = response.json()
        print(f"jobId: {json_data['job_id']}")
        return json_data
       
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP 에러 발생: {http_err}')
    except Exception as err:
        print(f'기타 에러 발생: {err}')

def get_transcriptions():
    
    try:
        response = requests.get(URL)
        response.raise_for_status() 
        json_data = response.json()
        jobIds = [ {idx: item['job_id']} for idx, item in enumerate(json_data['items'])]
        print(f"jobIds: {jobIds}")
        return json_data
       
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP 에러 발생: {http_err}')
    except Exception as err:
        print(f'기타 에러 발생: {err}')

def get_simple_segments(segments):
    try:
        return '\n'.join([f"[{seg['start']}] {seg['content']}" for seg in segments])
    except Exception as err:
        print(f'에러 발생: {err}')

def get_simple_transcriptions():
    response = get_transcriptions()
    result = []
    for i in response['items']:
        result.append(get_simple_segments(i['transcription']['transcription']['segments']))
    return result

def get_simple_transcription(job_id):
    response = get_transcription(job_id)
    result = get_simple_segments(response['transcription']['transcription']['segments'])
    return result
