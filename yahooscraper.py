import os
import random
import pytube
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sys
from datetime import datetime
import xml.etree.ElementTree as ElementTree
from html import unescape

def get_caption_from_segment(xml_captions, start, stop):
    segments = []
    root = ElementTree.fromstring(xml_captions)
    for i, child in enumerate(list(root)):
        text = child.text or ""
        caption = unescape(text.replace("\n", " ").replace("  ", " "),)
        duration = float(child.attrib["dur"])
        capstart = float(child.attrib["start"])
        capend = capstart + duration

        if (start >= capstart and start <= capend) or (stop >= capstart and stop <= capend) or (start <= capstart and stop >= capend) or (start >= capstart and stop <= capend):
            segments.append(caption)

    return " ".join(segments).strip()


def download_video(vidid, start, stop, gif_title, index):
    url = f'https://youtube.com/watch?v={vidid}'
    youtube = pytube.YouTube(url)

    path = "./data/yahoo/videos/" + str(index)

    if not os.path.exists("./data/yahoo/videos/" + str(index)):
        os.mkdir("./data/yahoo/videos/" + str(index))

    # download correct
    youtube.streams.filter(res="360p").first().download(path, 'pos')
    ffmpeg_extract_subclip(path + "/" + 'pos' + ".mp4", start, stop, targetname=path + "/gif" + 'pos' + ".mp4")
    os.remove(path + "/" + 'pos' + ".mp4")
    os.rename(path + "/gif" + 'pos' + ".mp4", path + "/" + 'pos' + ".mp4")

    vid_length = stop - start
    negstart = random.randint(0, int(youtube.length - vid_length - 1))
    negstop = negstart + vid_length
    if youtube.length > vid_length + 3:
        # download negative example
        youtube.streams.filter(res="360p").first().download(path, 'neg')
        ffmpeg_extract_subclip(path + "/" + 'neg' + ".mp4", negstart, negstop, targetname=path + "/gif" + 'neg' + ".mp4")
        os.remove(path + "/" + 'neg' + ".mp4")
        os.rename(path + "/gif" + 'neg' + ".mp4", path + "/" + 'neg' + ".mp4")

    # caption retrieval
    pos_captions = ''
    neg_captions = ''
    if 'en' in youtube.captions:
        pos_captions = get_caption_from_segment(youtube.captions['en'].xml_captions, start, stop)
        if youtube.length > vid_length + 3:
            neg_captions = get_caption_from_segment(youtube.captions['en'].xml_captions, negstart, negstop)

    print('id', vidid)
    print('title', gif_title)
    print('description', youtube.description)
    print('pos_captions', pos_captions)
    print('neg_captions', neg_captions)

    # write metadata
    pd.DataFrame({
        'id': [vidid], 
        'title': [gif_title], 
        'description': [youtube.description],
        'poscaptions': [pos_captions],
        'neg_captions': [neg_captions]}
    ).to_csv(path + '/metadata.txt', index=False)

if __name__ == "__main__":
    random.seed(datetime.now())
    df = pd.read_csv('./data/yahoo/metadata/metadata.txt', sep=';\t')

    for index, row in df.iterrows():
        print(index / len(df))
        for i in range(3):
            try:
                download_video(
                    row['youtube_id'], 
                    float(row['gif_start_sec']), 
                    float(row['gif_end_sec']), 
                    row['gif_title'],
                    index)
                break
            except pytube.exceptions.RegexMatchError:
                print("Unexpected error:", sys.exc_info()[0])
            except pytube.exceptions.VideoPrivate:
                break