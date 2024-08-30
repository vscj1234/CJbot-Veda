from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_session import Session
from mysql.connector import pooling
from mysql.connector import Error as MySQLError
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import re
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime, timedelta
import time
import threading

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=3)  # Set session lifetime to 3 minutes
Session(app)

# Initialize SendGrid API key
SENDGRID_API_KEY = ""
os.environ["SENDGRID_API_KEY"] = SENDGRID_API_KEY

# MySQL Database Configuration for Conversations and Appointments with Pooling
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Mysql@27#', 
    'database': 'cloudjune',
    'pool_name': 'cloudjune_pool',
    'pool_size': 5,
    'pool_reset_session': False,
    'autocommit': True,
    'get_warnings': True,
    'raise_on_warnings': True,
    'connection_timeout': 30,
}

# Initialize the connection pool
connection_pool = pooling.MySQLConnectionPool(**db_config)

os.environ["OPENAI_API_KEY"] = "sk-cloudjune-H0RLPEiyk8tHQri0KPArT3BlbkFJM61NxbpwCOhnOJVflWQP"

# Load the persisted vector store
embeddings = OpenAIEmbeddings()
docsearch = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

llm = ChatOpenAI(model="gpt-4o")

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

def is_connection_valid(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        return True
    except MySQLError:
        return False

def get_db_connection(max_retries=3, retry_delay=1):
    for attempt in range(max_retries):
        try:
            conn = connection_pool.get_connection()
            return conn
        except MySQLError as e:
            print(f"Error getting database connection (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

def refresh_connection_pool():
    global connection_pool
    while True:
        time.sleep(300)  # Check every 5 minutes
        try:
            new_pool = pooling.MySQLConnectionPool(**db_config)
            old_pool = connection_pool
            connection_pool = new_pool
            del old_pool
            print("Connection pool refreshed successfully")
        except Exception as e:
            print(f"Error refreshing connection pool: {e}")

# Start the refresh thread
refresh_thread = threading.Thread(target=refresh_connection_pool, daemon=True)
refresh_thread.start()

# Define SendGrid email sending function
def send_email(user_message, bot_response):
    message = Mail(
        from_email='support@cloudjune.com',
        to_emails='vedarutvija.joopally@cloudjune.com',  # Update with recipient email
        subject='User Enquiry',
        html_content=f'<p>User Message: {user_message}</p><p>Bot Response: {bot_response}</p>'
    )

    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))

@app.before_request
def before_request():
    session.permanent = True  # Make session permanent so it uses PERMANENT_SESSION_LIFETIME

    # Check if 'last_activity' exists in the session
    if 'last_activity' in session:
        last_activity = session['last_activity']
        now = datetime.now()

        # If more than 3 minutes have passed since the last activity, clear the session
        if now - last_activity > timedelta(minutes=3):
            session.clear()
            return redirect(url_for('home'))

    # Update the last activity time
    session['last_activity'] = datetime.now()

@app.route('/')
def home():
    return render_template('base.html')

# Function to check available appointment slots
def get_available_slots():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT appointment_time FROM appointments")
        booked_slots = [row[0] for row in cursor.fetchall()]

        available_slots = []
        now = datetime.now()
        for i in range(7):  # Next 7 days
            date = now.date() + timedelta(days=i)
            for hour in range(9, 18):  # 9 AM to 5 PM
                slot = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
                if slot not in booked_slots:
                    available_slots.append(slot.strftime("%Y-%m-%d %H:00"))

        return available_slots[:5]  # Return the first 5 available slots
    finally:
        cursor.close()
        conn.close()

# Function to book an appointment
def book_appointment(user_id, appointment_time, email):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO appointments (user_id, appointment_time, email) VALUES (%s, %s, %s)", (user_id, appointment_time, email))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data['message']

    # Initialize chat history and booking stage if not present in session
    session.setdefault('chat_history', [])
    session.setdefault('booking_stage', 'initial')

    # Prepare the conversation history
    chat_history = session['chat_history']

    # Add system message if it's the first message in the conversation
    if not chat_history:
        system_message = ("You are a pro conversational chatbot created for a company named CloudJune. You should act as a chat assistant on behalf of CloudJune and should always refuse to answer questions that are not related to CloudJune. "
                          "You are pro at understanding context and deliver meaningful conversational responses. "
                          "You have the ability to schedule appointments for users. When a user expresses interest in booking an appointment, guide them through the process using the available appointment slots. "
                          "After every response you should ask the user to share their name and email so that the team can reach out to them. "
                          "Do it only until they provide their name and email. Act accordingly:")
        chat_history.append(("system", system_message))

    # Handle appointment booking process
    if session['booking_stage'] == 'initial' and any(keyword in query.lower() for keyword in ["book", "appointment", "schedule", "meet"]):
        session['booking_stage'] = 'started'
        response = "Certainly! I'd be happy to help you book an appointment with CloudJune. First, could you please provide your email address?"
    elif session['booking_stage'] == 'started':
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails_found = re.findall(email_pattern, query)
        if emails_found:
            session['user_email'] = emails_found[0]
            available_slots = get_available_slots()
            response = "Thank you for providing your email. Here are the available slots:\n\n"
            for i, slot in enumerate(available_slots, 1):
                response += f"{i}. {slot}\n"
            response += "\nPlease choose a slot by entering its number."
            session['booking_stage'] = 'slot_selection'
        else:
            response = "I'm sorry, I couldn't find a valid email address in your message. Could you please provide your email address?"
    elif session['booking_stage'] == 'slot_selection':
        try:
            slot_index = int(query) - 1
            available_slots = get_available_slots()
            chosen_slot = available_slots[slot_index]
            book_appointment(session.sid, chosen_slot, session['user_email'])
            response = f"Great! Your appointment with CloudJune has been booked for {chosen_slot}. We'll send a confirmation to {session['user_email']}. Is there anything else I can help you with?"
            session['booking_stage'] = 'initial'
            session.pop('user_email', None)
        except (ValueError, IndexError):
            response = "I'm sorry, that's not a valid selection. Please choose a number from the list of available slots."
    else:
        # Generate response using the existing chain
        result = chain({"question": query, "chat_history": chat_history})
        response = result['answer']

        # Check if the response indicates inability to book appointments
        if "don't have the ability to schedule appointments" in response:
            response = "I apologize for the confusion. I can actually help you book an appointment with CloudJune. Would you like me to show you the available slots?"
            session['booking_stage'] = 'initial'

    # Update chat history
    chat_history.append(("user", query))
    chat_history.append(("bot", response))

    # Trim chat history if it gets too long (keep last 10 exchanges)
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    # Save updated chat history to session
    session['chat_history'] = chat_history

    # Check if the user message contains an email address
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_found = re.findall(email_pattern, query)

    # If an email address is found, send an email using SendGrid
    if emails_found:
        send_email(query, response)

    # Insert user query into conversations table
    # conn = get_db_connection()
    try:
        conn = get_db_connection()
        if not is_connection_valid(conn):
            conn.close()
            conn = get_db_connection()  # Try to get a new connection
        
        cursor = conn.cursor()
        user_id = session.sid
        cursor.execute('INSERT INTO conversations (user_id, user_query, bot_response) VALUES (%s, %s, %s)', (user_id, query, response))
        conn.commit()
    except MySQLError as e:
        print(f"Database error: {e}")
        # Handle the error appropriately
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)
