import smtplib
for i in range(10):
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login("medinwisstaa@gmail.com", "nezaboravi123")
    server.sendmail("medinwisstaa@gmail.com", "harunhajdarevic@outlook.com", "Gasss")
    server.quit()