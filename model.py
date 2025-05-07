from fastapi import FastAPI, Form, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import requests

# Initialize FastAPI app
app = FastAPI()

# Database setup (SQLite)
DATABASE_URL = "sqlite:///./reminders.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define Reminder table
class Reminder(Base):
    __tablename__ = "reminders"
    id = Column(Integer, primary_key=True, index=True)
    plant_name = Column(String)
    phone_number = Column(String)
    reminder_time = Column(DateTime)

# Create the table
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False)

# Infobip API URL and credentials
API_URL = "https://api.infobip.com/sms/2/text/advanced"
API_KEY = "a3450df078285d4b0c5e023cb1cfca7b-fe452fd5-6001-4811-9354-a3db7bf9ffee"  # Replace with your Infobip API key
headers = {
    "Authorization": f"App {API_KEY}",
    "Content-Type": "application/json"
}

# Function to send SMS via Infobip API
def send_sms(phone_number: str, message: str):
    payload = {
        "messages": [
            {
                "to": phone_number,
                "from": "YourSenderID",  # Replace with your Infobip Sender ID or phone number
                "text": message
            }
        ]
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    return response.json()

# Jinja2 Templates setup
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit/")
async def submit_form(
    background_tasks: BackgroundTasks,
    plant_name: str = Form(...),
    phone_number: str = Form(...),
    reminder_time: str = Form(...)
):
    db = SessionLocal()
    dt = datetime.fromisoformat(reminder_time)
    
    # Store reminder in the database
    reminder = Reminder(
        plant_name=plant_name,
        phone_number=phone_number,
        reminder_time=dt
    )
    db.add(reminder)
    db.commit()
    db.close()
    
    # Schedule SMS reminder
    message = f"Reminder: Time to water your {plant_name}!"
    background_tasks.add_task(send_sms, phone_number, message)
    
    return {"message": "Reminder set successfully!"}
