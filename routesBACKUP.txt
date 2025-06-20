from flask import Blueprint, render_template, request
from app.sentiment_logic import analyse_video


main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        file = request.files.get('video_file')
        if file and file.filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            video_ids = [line.strip() for line in content.splitlines() if line.strip()]
            for vid in video_ids:
                resume, score = analyse_video(vid)
                results.append({'video_id': vid, 'resume': resume, 'score': score})
    return render_template('index.html', results=results)
