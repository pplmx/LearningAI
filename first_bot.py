#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2/27/2018 14:10

from chatterbot import ChatBot

chat_bot = ChatBot(
    'Ada',
    # trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
    database='db.sqlite3'
)

# Train
# chat_bot.train('chatterbot.corpus.english')
# chat_bot.train('chatterbot.corpus.chinese')
# chat_bot.train('chatterbot.corpus.tchinese')

if __name__ == "__main__":
    # Get a response to an input statement
    print('Chatting with Ada.(Print \'exit\' to exit.)')
    while True:
        print('Master: ')
        try:
            statement = input()
            if statement == 'exit':
                break
            resp = chat_bot.get_response(statement)
            print('%s: %s' % (chat_bot.name, resp))
        except(KeyboardInterrupt, EOFError, SystemExit):
            break
