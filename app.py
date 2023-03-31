import streamlit as st
import polars as pl
from faster_whisper import WhisperModel
from faster_model_first import read_segments, parse_srt, gen_request_file, parse_transribed, align_ori_trans, get_each_batch
from async_better import process_api_requests_from_file, get_all_token_comsumptions
import asyncio  # for running API calls concurrently
from utils import Config, create_config
import logging  # for logging rate limit warnings and other messages
#import pathlib
#import subprocess
import os, glob

def check_tmpfile(save_folder: str) -> None:
    if not os.path.exists(save_folder + "\\tmp"):
        st.sidebar.info('tmp file directory does not exist', icon="ℹ️")
        os.mkdir(save_folder + "\\tmp")
        st.sidebar.info('tmp file directory created', icon="ℹ️")
    else:
        st.sidebar.success("tmp file directory existed", icon="✅")

def write_srt(srt_filepath:str, model: str, file_path:str) -> pl.DataFrame:
    if not os.path.isfile(srt_filepath):
        model = WhisperModel(model, device="cuda", compute_type="float16")
        segments, info = model.transcribe(file_path, beam_size=5, word_timestamps=True)
        srt_file = parse_srt(read_segments(segments))
        srt_file.write_csv(srt_filepath)
        st.sidebar.info('srt file created', icon="ℹ️")
    else:
        srt_file = pl.read_csv(srt_filepath)
        st.sidebar.success("srt file existed", icon="✅")
    return srt_file

def gen_request(requests_filepath:str, srt_filepath:str) -> None:
    if not os.path.isfile(requests_filepath):
        srt_file = pl.read_csv(srt_filepath)
        request_format = gen_request_file(srt_file)
        with open(requests_filepath, "w", encoding="Utf-8") as file:
            file.write(request_format.write_ndjson())
        st.sidebar.info('request file created', icon="ℹ️")
    else:
        st.sidebar.success("request file existed", icon="✅")

def call_openai(args: Config) -> None:
    if not os.path.isfile(args.save_filepath):
        # run script
        requests_token = get_all_token_comsumptions(args.requests_filepath)
        with st.form(key='do you want to continue?'):
            break_or_continue = st.form_submit_button(label='Submit')
        if not break_or_continue:
            # st.error("you choose not to use gpt")
            st.stop()
        asyncio.run(
            process_api_requests_from_file(
                requests_token=requests_token,
                save_filepath=args.save_filepath,
                request_url=args.request_url,
                api_key=args.api_key,
                max_requests_per_minute=float(args.max_requests_per_minute),
                max_tokens_per_minute=float(args.max_tokens_per_minute),
                # token_encoding_name=args.token_encoding_name,
                max_attempts=int(args.max_attempts),
                logging_level=int(args.logging_level),
            )
        )
        st.sidebar.info('transcribe file created', icon="ℹ️")
    else:
        st.sidebar.success("transcribe file existed", icon="✅")


def display_diff_pd(srt_file:pl.DataFrame, transcribed:pl.DataFrame):
    ori_dfs = {idx:val for (idx, val) in srt_file.lazy().with_columns(
        ((pl.col("is_end") - 1) // 10).alias("group")) \
        .collect().to_pandas().groupby("group")}

    each_batch = get_each_batch(transcribed, to_pandas=True)
    selected_batch = st.selectbox('select a batch:', each_batch.keys())
    tran_ = each_batch[selected_batch]#.to_pandas()
    or_ = ori_dfs[selected_batch]#.to_pandas()
    col4, col5 = st.columns([3, 3])
    or_df = col4.experimental_data_editor(or_, num_rows="dynamic")
    e_df = col5.experimental_data_editor(tran_, num_rows="dynamic")

def display_diff(srt_file:pl.DataFrame, transcribed:pl.DataFrame):
    ori_dfs = srt_file.lazy().with_columns(
        ((pl.col("is_end") - 1) // 10).alias("group")) \
        .collect().partition_by("group", maintain_order=True, as_dict=True)
    each_batch = get_each_batch(transcribed)
    selected_batch = st.selectbox('select a batch:', each_batch.keys())
    tran_ = each_batch[selected_batch].to_pandas()
    or_ = ori_dfs[selected_batch].to_pandas()
    col4, col5 = st.columns([3, 3])
    or_df = col4.experimental_data_editor(or_, num_rows="dynamic")
    e_df = col5.experimental_data_editor(tran_, num_rows="dynamic")

def write_transcribed(srt_file:pl.DataFrame, transcribed:pl.DataFrame,save_folder:str):
    combined = align_ori_trans(srt_file, transcribed)
    combined.write_csv(save_folder + "\\tmp\\combined.csv")
    with open(save_folder + "\\bilingal.srt", "w", encoding="Utf-8") as file:
        file.write("\n\n".join(["\n".join(row) for row in combined.iter_rows()]))
    with open(save_folder + "\\chinese.srt", "w", encoding="Utf-8") as file:
        file.write("\n\n".join(
            ["\n".join(row) for row in combined.select(pl.col("is_end", "timestamp", "transcribed")).iter_rows()]))
    with open(save_folder + "\\english.srt", "w", encoding="Utf-8") as file:
        file.write(
            "\n\n".join(["\n".join(row) for row in combined.select(pl.col("is_end", "timestamp", "text")).iter_rows()]))


def home():
    import streamlit as st
    from streamlit_ace import st_ace
    st.write("# Welcome to Auto-Transcriber! 👋")
    st.sidebar.success("Select From-File or From-Link to begin with")

    st.markdown(
        """

        Auto-Transcriber is an experimental, openai-powered translation tool, 
        generating .srt file in one call!
        

        ### How to start with?

        - Choose `From File` or `From Link` from sidebar
        - Input `yt-link`(or video/audio file path), `saved folder path`
        - Click `Generate`
        
        ### Main Features
    """
    )
    # content = st_ace()
    # content

def from_file():
    import streamlit as st
    if 'btn_clicked' not in st.session_state:
        st.session_state['btn_clicked'] = False

    def callback():
        # change state value
        st.session_state['btn_clicked'] = True

    col1, col2, col3 = st.columns([3, 3, 1])
    folder_path = col1.text_input("Place path to saved folder here 👇", ".")
    filenames = os.listdir(folder_path)
    video_file = col2.selectbox('Select a file', filenames)
    video_path = os.path.join(folder_path, video_file)

    if col3.button("Generate", on_click=callback) or st.session_state['btn_clicked']:
        if not (video_path and folder_path):
            st.warning('you should fill both `video_link` and `folder_path`', icon="⚠️")
        else:
            # args = Config(save_folder=folder_path,
            #               file_path=video_path,
            #               srt_filepath=folder_path + '\\tmp\\srt.csv',
            #               requests_filepath=folder_path + '\\tmp\\request.jsonl',
            #               save_filepath=folder_path + '\\tmp\\transcribe_result.csv',
            #               api_key= os.getenv("OPENAI_API_KEY"))
            st.sidebar.write("Logging:\n")
            args = create_config(folder_path, file_path=video_path)
            with st.expander("📝See your Config:"):
                st.write(args.to_dict())
            check_tmpfile(args.save_folder)
            srt_file = write_srt(args.srt_filepath, args.model, args.file_path)
            gen_request(args.requests_filepath, args.srt_filepath)
            call_openai(args)
            transcribed = parse_transribed(args.save_filepath)

            with st.expander("📝See details for each batch"):
                st.write(transcribed.filter(pl.col("num_sentences") != 10).select("group","num_sentences").to_struct(name="value_count").to_list())

            #add blocks to manipulate dataframe(as the implemenation here could no change the original one)
            display_diff_pd(srt_file, transcribed)

            form = st.form(key='continue?')
            name = form.selectbox('fill methods?', ["ffill", "bfill"])
            col6, col7 = form.columns([0.5, 0.5])
            create_final = col6.form_submit_button('Continue')
            not_create = col7.form_submit_button('Stop')
            if not create_final and not not_create:
                st.stop()
            if not_create:
                st.info("Sorry for not so perfect transcription", icon="💔")
                st.stop()


            write_transcribed(srt_file, transcribed, args.save_folder)
            st.success("Horay!\nDone in one call!", icon="🧋")


def from_link():
    import streamlit as st
    if 'btn_clicked' not in st.session_state:
        st.session_state['btn_clicked'] = False

    def callback():
        # change state value
        st.session_state['btn_clicked'] = True
    col1, col2, col3 = st.columns([3, 3, 1])
    video_link = col1.text_input("Place youtube link here 👇")
    folder_path = col2.text_input("Place path to saved folder here 👇")
    if col3.button("Generate", on_click=callback) or st.session_state['btn_clicked']:
        if not (video_link and folder_path):
            st.warning('you should fill both `video_link` and `folder_path`', icon="⚠️")
        else:
            try:
                from yt_dlp import YoutubeDL
                ydl_opts = {
                    'format': 'm4a/bestaudio/best',
                    'paths': {'home': folder_path},
                    'keepvideo': True,
                    'postprocessors': [{  # Extract audio using ffmpeg
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'postprocessor_args': [
                        '-ar', '16000',
                        '-ac', '1'
                    ]
                }
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download(video_link)
            except e as e:
                throw(e)

            st.sidebar.write("Logging:\n")
            video_name = glob.glob(folder_path+"\\*.wav")[0]
            args = create_config(folder_path, file_path=video_name)
            st.success('Video downloaded!', icon="✅")

            with st.expander("📝See your Config:"):
                st.write(args.to_dict())
            check_tmpfile(args.save_folder)
            srt_file = write_srt(args.srt_filepath, args.model, args.file_path)
            gen_request(args.requests_filepath, args.srt_filepath)
            call_openai(args)
            transcribed = parse_transribed(args.save_filepath)

            with st.expander("📝See details for each batch"):
                st.write(transcribed.filter(pl.col("num_sentences") != 10).select("group", "num_sentences").to_struct(
                    name="value_count").to_list())
            display_diff(srt_file, transcribed)
            form = st.form(key='continue?')
            name = form.selectbox('fill methods?', ["ffill", "bfill"])
            col6, col7 = form.columns([0.5, 0.5])
            create_final = col6.form_submit_button('Continue')
            not_create = col7.form_submit_button('Stop')
            if not create_final and not not_create:
                st.stop()
            if not_create:
                st.info("Sorry for not so perfect transcription", icon="💔")
                st.stop()


            write_transcribed(srt_file, transcribed, args.save_folder)
            st.success("Horay!\nDone in one call!", icon="🧋")

st.set_page_config(
    page_title="Transcribe App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded")

page_names_to_funcs = {
    "—": home,
    "From File": from_file,
    "From Link": from_link,
}

st.sidebar.markdown("# Main Menu")
demo_name = st.sidebar.selectbox("start with a video or YT link", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()