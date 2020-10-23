from youtube_transcript_api import YouTubeTranscriptApi
from itertools import accumulate
import pytube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import ffmpy

def get_transcript(vidid, start, stop):
    transcript = YouTubeTranscriptApi.get_transcripts(vidid, languages=['en'])
    filtered = filter(
        lambda x: 
            True if (x['start'] <= start and x['start'] + x['duration'] > start) or
                (x['start'] >= start and x['start'] <= stop)
            else False,
        transcript
    )

    return accumulate(filtered, lambda accum, new: accum.strip() + ' ' + new['text'], initial="")



def download_video(vidid, start, stop):
    path = "./data/yahoo/videos"
    url = f'https://youtube.com/watch?v={vidid}'
    print(url)
    youtube = pytube.YouTube(url)
    youtube.streams.filter(res="360p").first().download(path, vidid)
    ffmpeg_extract_subclip(path + "/" + vidid + ".mp4", start, stop, targetname=path + "/gif" + vidid + ".mp4")
    os.remove(path + "/" + vidid + ".mp4")
    os.rename(path + "/gif" + vidid + ".mp4", path + "/" + vidid + ".mp4")

try:
    print(get_transcript('psPWYsURvCo', 59.92, 62))
except Exception:
    print("failed to retrieve transcript")

download_video('psPWYsURvCo', 59.92, 62)