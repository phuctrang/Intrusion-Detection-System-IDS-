import telegram

def send_telegram(photo_path="alert.png"):
    try:
        my_token = "5588456298:AAHIWsaDN8_J766l64ygfXicFiOtutpcNm0"
        bot = telegram.Bot(token=my_token)
        bot.sendPhoto(chat_id="5427289697", photo=open(photo_path, "rb"), caption="Cẩn thận, có đột nhập !!! ")
    except Exception as ex:
        print("Can not send message telegram ", ex)

    print("Send sucess") #pip install python-telegram-bot