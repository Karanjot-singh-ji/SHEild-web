from flask import Flask, render_template, send_from_directory, jsonify
from keras.models import load_model
import cv2
import numpy as np
import datetime
from geopy.geocoders import Nominatim
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

app = Flask(__name__)

model = load_model("model/keras_model.h5", compile=False)
class_names = open("model/labels.txt", "r").readlines()

def send_email(image_path, video_path, current_time, location_details):
    from_email = "info.shield112@gmail.com"
    password = "foad ahkk sgrr zqwk"
    to_email = "vanshikasharma6553@gmail.com"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = "Alert Message"

    body = f"Alert!\nTime: {current_time}\nLocation: {location_details}"
    msg.attach(MIMEText(body, 'plain'))

    if image_path:
        with open(image_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(image_path)}")
            msg.attach(part)

    if video_path:
        with open(video_path, "rb") as video_attachment:
            video_part = MIMEBase('application', 'octet-stream')
            video_part.set_payload(video_attachment.read())
            encoders.encode_base64(video_part)
            video_part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(video_path)}")
            msg.attach(video_part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return "Email sent successfully"
    except smtplib.SMTPAuthenticationError:
        return "Failed to authenticate. Check your email and password."
    except Exception as e:
        return f"An error occurred while sending email: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alert', methods=['GET'])
def alert():
    video_path = "static/uploads/vi.mp4"  # Ensure this path is correct
    camera = cv2.VideoCapture(video_path)
    message = ""

    # Initialize variables to capture alert segment
    alert_frame_number = None
    fps = camera.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frame_preprocessed = np.asarray(frame_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        frame_preprocessed = (frame_preprocessed / 127.5) - 1

        prediction = model.predict(frame_preprocessed)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        if index == 1:
            alert_frame_number = int(camera.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            break

    if alert_frame_number is not None:
        # Save snapshot
        image_path = 'static/snapshot.jpg'
        camera.set(cv2.CAP_PROP_POS_FRAMES, alert_frame_number)
        ret, frame = camera.read()
        if ret:
            cv2.imwrite(image_path, frame)

        # Crop 10-second segment from alert frame
        video_output_path = 'static/alert_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

        start_frame = alert_frame_number
        camera.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        end_frame = start_frame + int(fps * 10)  # 10 seconds of video
        while camera.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = camera.read()
            if ret:
                out.write(frame)
            else:
                break

        out.release()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get location
        geolocator = Nominatim(user_agent="SHEild/1.0 (karanjot032004@gmail.com)")
        try:
            location = geolocator.geocode("12A State Highway, CGC Jhanjeri, Mohali, Punjab, India")
            if location is not None:
                location_details = f"{location.address}, Lat: {location.latitude}, Lon: {location.longitude}"
            else:
                location_details = "12A State Highway, CGC Jhanjeri, Mohali, Punjab, India"
        except Exception:
            location_details = "12A State Highway, CGC Jhanjeri, Mohali, Punjab, India"

        # Send email with image and cropped video
        result = send_email(image_path, video_output_path, current_time, location_details)
        message = result

    camera.release()

    # Clean up
    try:
        os.remove(image_path)
        os.remove(video_output_path)
    except Exception as e:
        print(f"Error cleaning up files: {e}")

    return render_template('alert.html', message=message)

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory('static/uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
