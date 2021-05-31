import telegram
import os


TOKEN = 'enter_your_token'
bot = telegram.Bot(token=TOKEN)


async def send_detect_foto(filename):
    chat_id = bot.get_updates()[-1].message.chat_id
    if os.path.exists(filename):
        bot.send_photo(chat_id=chat_id, photo=open(filename, 'rb'))
