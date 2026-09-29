import base64
import io
import os

import torch
from flask import Flask, Request, render_template, send_from_directory, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from PIL import Image
from torchvision import transforms
from wtforms import FileField, FloatField, SubmitField

from utils.models import Decoder, VGGEncoder
from utils.utils import adaptive_instance_normalization


class InMemoryRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


app = Flask(__name__)
app.request_class = InMemoryRequest
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vgg_path = os.path.join(BASE_DIR, 'vgg_normalised.pth')
decoder_path = os.path.join(BASE_DIR, 'experiment', 'run2', 'decoder_10.pth')

encoder = VGGEncoder(vgg_path).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location=device))
encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def image_to_data_url(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().clone().squeeze(0).clamp(0, 1)
        image = transforms.ToPILImage()(image)
    else:
        image = image.convert('RGB')

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded_image}'



@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_image = None
    style_image = None
    error = None

    if request.method == 'POST' and form.validate_on_submit():
        content_file = form.content.data
        style_file = form.style.data

        if not content_file or not content_file.filename:
            error = 'Please upload content image'
        elif not style_file or not style_file.filename:
            error = 'Please upload style image'
        elif not allowed_file(content_file.filename) or not allowed_file(style_file.filename):
            error = 'Please upload PNG, JPG, or JPEG images'
        else:
            try:
                content_pil = Image.open(content_file.stream).convert('RGB')
                style_pil = Image.open(style_file.stream).convert('RGB')
                content_image = image_to_data_url(content_pil)
                style_image = image_to_data_url(style_pil)

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_pil, style_pil, encoder, decoder, alpha, device)
                result_image = image_to_data_url(stylized_image)
            except Exception as e:
                error = str(e)
    elif request.method == 'POST':
        error = 'Please upload both content and style images'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_image,
                           style_image=style_image, error=error)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    app.run(debug=True)