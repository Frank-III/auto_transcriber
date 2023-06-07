# AutoTranscriber 
Auto-Transcriber is an experimental, openai-powered translation tool, generating .srt file in one call!

![screenshot1](https://gitee.com/jiangdawang/pic/raw/master/src/20230330235522.png)
 
## Main Features
- powered by Openai `GPT3.5` or `GPT4` model, excellent context and results. 
- Audio to Text using [Fast-Whisper](https://github.com/guillaumekln/faster-whisper)(a faster implementation of Openai Whisper model), up to 4x faster.
- Minimum ussage of `Pandas`, using `Polars` for data processing and generating GPT requests.
- Ascychronize `ChatCompletionCall`, minimum latency.
- Build with Streamlit, with support to modify transcription results.

## How to start with?
```sh
git clone //repo
pip install -r requirements.txt
streamlit run app.py
```

- Choose `From File` or `From Link` from sidebar
- Input `yt-link`(or video/audio file path), `saved folder path`
- Click `Generate`
