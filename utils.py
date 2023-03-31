from dataclasses import dataclass
from dataclasses_json import dataclass_json
import logging
import os

@dataclass_json
@dataclass
class Config:
    save_folder: str
    file_path:str
    srt_filepath :str
    requests_filepath: str
    save_filepath: str
    api_key: str
    model: str = "medium.en"
    max_requests_per_minute: int = 3_000 * 0.5
    request_url: str = "https://api.openai.com/v1/chat/completions"
    max_tokens_per_minute: int = 250_000 * 0.5
    max_attempts: int = 5
    logging_level = logging.INFO

def create_config(folder_path:str, **kwargs) -> Config:
    return Config(save_folder=folder_path,
                  srt_filepath=folder_path + '\\tmp\\srt.csv',
                  requests_filepath=folder_path + '\\tmp\\request.jsonl',
                  save_filepath=folder_path + '\\tmp\\transcribe_result.csv',
                  api_key=os.getenv("OPENAI_API_KEY"),
                  **kwargs)
ydl_opts = {
    'format': 'm4a/bestaudio/best',
    'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
}
