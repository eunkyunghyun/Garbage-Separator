from flask import Flask, render_template, request
import cv2
import smtplib
from email.mime.text import MIMEText
import pandas as pd
from utils.config import parse_args
from utils.data_loader import get_data_loader
from models.nk_model import nkModel

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('base.html')


@app.route('/introduction', methods=["GET", "POST"])
def introduction():
    if request.method == "POST":
        text = request.form.get('text')
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            msg = MIMEText(text, _charset='euc-kr')
            msg['Subject'] = "{}-Feedback".format(email)
            msg['From'] = email
            msg['To'] = "eunandy09@naver.com"
            server = smtplib.SMTP_SSL("smtp.naver.com", 465)
            server.login(email, password)
            server.sendmail(email, "eunandy09@naver.com", msg.as_string())
            server.quit()
            error = None
            return render_template('introduction.html', error=error)
        except (OSError, TypeError, UnicodeEncodeError):
            error = "The wrong account has been inputted."
            return render_template('introduction.html', error=error)
        except smtplib.SMTPAuthenticationError:
            error = "The account has never been activated."
            return render_template('introduction.html', error=error)
    return render_template('introduction.html')


@app.route('/separation', methods=["GET", "POST"])
def separation():
    if request.method == "POST":
        try:
            cam = cv2.VideoCapture(0)
            while True:
                ret, frame = cam.read()
                if ret:
                    color = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
                    cv2.imshow('Camera', color)
                    q = cv2.waitKey(1) & 0xFF
                    if q == 27:
                        break
                    elif q == 13:
                        image = cv2.resize(color, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                        cv2.imwrite("image/garbage.jpg", image)
                        f = open("./labels/test_label.csv", 'w')
                        f.close()
                        f = open("./labels/test_result_label.csv", 'w')
                        f.close()
                        f = open("./labels/test_label.csv", 'a')
                        f.write("file,label\n")
                        f.write("image/garbage.jpg")
                        f.close()
                        break
                else:
                    raise cv2.error
            cam.release()
            cv2.destroyAllWindows()
            image = cv2.imread("./image/garbage.jpg")
            cv2.imshow('Garbage', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            config = parse_args()
            separate(args=config)
            f = open("./labels/test_result_label.csv", 'r')
            directory = f.readlines()
            if int(directory[1].strip('\n')) == 0:
                result = "Food Waste"
            else:
                result = "General Waste"
            return render_template('separation.html', result=result)
        except cv2.error:
            error = "An error has occurred, executing corresponded camera system."
            return render_template('separation.html', error=error)
    return render_template('separation.html')


def separate(args):
    train_loader, test_loader = get_data_loader(args)
    model = nkModel(args, train_loader, test_loader)
    if args.is_train:
        model.train()
    else:
        temp_list = model.test()
        my_df = pd.DataFrame(temp_list)
        my_df.to_csv('./labels/test_result_label.csv', index=False, header=False)


if __name__ == '__main__':
    app.run()