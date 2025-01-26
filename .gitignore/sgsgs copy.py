from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline
from config import API_TOKEN
# Initialize your Llama pipeline
pipe = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  
    device_map="auto"
)

# A global variable to store the bot's state if necessary (optional)
session_data = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text("Hello! I'm your chatbot. Ask me anything!")

async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /restart command."""
    global session_data
    session_data.clear()  # Clear any session data if applicable
    await update.message.reply_text("The bot has been restarted! Start fresh by asking me anything.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages."""
    user_input = update.message.text  # Get the user's input from the message
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of an engineer.",
        },
        {"role": "user", "content": user_input},
    ]
    # Format the input for Llama
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate a response
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    bot_response = outputs[0]["generated_text"]

    # Send the response back to the user
    await update.message.reply_text(bot_response)

def main():
    app = ApplicationBuilder().token(API_TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("restart", restart))  # Add the restart command

    # Add message handler for all other messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    app.run_polling()

if __name__ == "__main__":
    main()
