from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline
from config import API_TOKEN

pipe = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  
    device_map="auto"
)
session_data = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! I'm your chatbot. Ask me anything!")

async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global session_data
    session_data.clear()  
    await update.message.reply_text("The bot has been restarted! Start fresh by asking me anything.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text  
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of an engineer.",
        },
        {"role": "user", "content": user_input},
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    bot_response = outputs[0]["generated_text"]

    
    await update.message.reply_text(bot_response)

def main():
    app = ApplicationBuilder().token(API_TOKEN).build()

    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("restart", restart))  

    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    
    app.run_polling()

if __name__ == "__main__":
    main()
