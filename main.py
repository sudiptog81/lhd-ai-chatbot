import os
import discord

from dotenv import load_dotenv

from support import *


class Client(discord.Client):
    counters = dict()

    async def on_ready(self):
        print('Logged in as', self.user.name, self.user.id)
        guilds = await client.fetch_guilds(limit=150).flatten()
        guilds = [g.id for g in guilds]
        for g in guilds:
            self.counters[g] = {
                'pos': 1,
                'neg': 1
            }

    async def on_message(self, message):
        if message.author.id == self.user.id \
                or len(message.content.strip()) == 0:
            return
        else:
            text = message.content.strip()
            prediction = model.predict([extract_features(text, freqs)[0]])[0]
            if prediction == 1.:
                self.counters[message.guild.id]['pos'] += 1
                if self.counters[message.guild.id]['pos'] >= 5:
                    await message.channel.send('Yay')
                    self.counters[message.guild.id]['pos'] = 1
                    self.counters[message.guild.id]['neg'] = 1
            else:
                self.counters[message.guild.id]['neg'] += 1
                if self.counters[message.guild.id]['neg'] >= 5:
                    await message.channel.send('I think that the last few messages sent on this channel express a negative sentiment.')
                    self.counters[message.guild.id]['neg'] = 1


if __name__ == '__main__':
    load_dotenv()
    client = Client()
    client.run(os.environ['DISCORD_TOKEN'])
