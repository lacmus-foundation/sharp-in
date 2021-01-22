from fastapi import FastAPI
import telebot
from api.v0.routes.api import router as api_router
from core.config import get_config


def get_application() -> FastAPI:
    project_name = get_config().project_name
    debug = get_config().debug
    version = get_config().version
    prefix = get_config().api_prefix

    application = FastAPI(title=project_name, debug=debug, version=version)
    application.include_router(api_router, prefix=prefix)
    return application

# TG bot setup

WEBHOOK_HOST        = get_config().WEBHOOK_HOST
WEBHOOK_PORT        = 443
WEBHOOK_URL_BASE    = "https://%s:%s" % (WEBHOOK_HOST, WEBHOOK_PORT)
WEBHOOK_URL_PATH    = "/%s" % ("api/v0/tg")

bot = telebot.TeleBot(get_config().token)
bot.remove_webhook()
bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)

app = get_application()