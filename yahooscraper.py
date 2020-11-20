import os
import random
import pytube
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sys
from datetime import datetime
import xml.etree.ElementTree as ElementTree
from html import unescape
import json
import argparse

basepath = '/users/zhoffman/Video2Gif/'

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

    global basepath
    path = basepath + "data/yahoo/videos/" + str(index)

    if not os.path.exists(basepath +  "data/yahoo/videos/" + str(index)):
        os.mkdir(basepath + "data/yahoo/videos/" + str(index))

    # download correct
    youtube.streams.filter(res="360p").first().download(path, 'pos')
    ffmpeg_extract_subclip(path + "/" + 'pos' + ".mp4", start, stop, targetname=path + "/gif" + 'pos' + ".mp4")
    os.remove(path + "/" + 'pos' + ".mp4")
    os.rename(path + "/gif" + 'pos' + ".mp4", path + "/" + 'pos' + ".mp4")

    vid_length = stop - start
    neg_captions = ''
    if youtube.length > vid_length + 3:
        negstart = random.randint(0, int(youtube.length - vid_length - 1))
        negstop = negstart + vid_length

        # download negative example
        youtube.streams.filter(res="360p").first().download(path, 'neg')
        ffmpeg_extract_subclip(path + "/" + 'neg' + ".mp4", negstart, negstop, targetname=path + "/gif" + 'neg' + ".mp4")
        os.remove(path + "/" + 'neg' + ".mp4")
        os.rename(path + "/gif" + 'neg' + ".mp4", path + "/" + 'neg' + ".mp4")

        if 'en' in youtube.captions:
             neg_captions = get_caption_from_segment(youtube.captions['en'].xml_captions, negstart, negstop)

    pos_captions = ''
    if 'en' in youtube.captions:
        pos_captions = get_caption_from_segment(youtube.captions['en'].xml_captions, start, stop)
           

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
    df = pd.read_csv(basepath + 'data/yahoo/metadata/metadata.txt', sep=';\t')

    parser = argparse.ArgumentParser(description='Download videos.')

    parser.add_argument("-s", "--start", type=int, default=0,
                    help="Index to start downloading")

    parser.add_argument("-e", "--end", type=int, default=len(df.index),
                help="Index to end downloading")

    args = parser.parse_args()
    start = args.start
    stop = args.end

    print("start", start, "end", stop)

    df = df[start:stop]

    total = len(df.index)
    for index, row in df.iterrows():
        print(index / total)
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
            except json.decoder.JSONDecodeError:
                print("Unexpected error:", sys.exc_info()[0])
            except pytube.exceptions.VideoPrivate:
                break
            except KeyboardInterrupt:
                raise
            except:
                print("Failed on video", index, sys.exc_info()[0])