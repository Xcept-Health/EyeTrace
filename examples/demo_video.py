import yt_dlp

def get_video_url(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']  # URL directe du flux

url = "https://www.youtube.com/watch?v=AAqATW60uFM"
direct_url = get_video_url(url)
print("URL directe :", direct_url)
# Exemple : https://rr2---sn-...googlevideo.com/videoplayback?...

# Maintenant, vous pouvez utiliser cette URL avec VideoReader
from eyetrace.io import VideoReader

with VideoReader(direct_url, resize=(640, 480), grayscale=True) as video:
    for frame in video:
        print(f"Frame {video.frame_count}")