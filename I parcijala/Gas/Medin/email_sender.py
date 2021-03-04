import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email import encoders
import os.path
def salji_mail(file_location):
    email = 'medinwisstaa@gmail.com'
    
    send_to_email = 'karamuratovicdina@gmail.com'
    subject = 'Presjek stanja'
    message = 'Ovo je poruka od poslovođe. U prilogu maila excel tabela i grafik današnje proizvodnje. Sve je automatski generisano iz Pythona. Ugodan ostatak dana'
    email = 'medinwisstaa@gmail.com'
    
    
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = send_to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))
    filename = os.path.basename(file_location)
    attachment = open(file_location, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(part)
    img = 'C:\\Users\\HP\Desktop\\Pokusaji\\zadnja.png'
    img_data = open(img, "rb").read()
    image = MIMEImage(img_data, name = 'C:\\Users\\HP\Desktop\\Pokusaji\\zadnja.png')
    msg.attach(image)
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login("medinwisstaa@gmail.com", "nezaboravi123")
    text = msg.as_string()
    server.sendmail(email, send_to_email, text)
    server.quit()
