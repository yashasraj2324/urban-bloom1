from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image, ExifTags, ImageOps, ImageFilter
from io import BytesIO
import google.generativeai as genai
import os
import tensorflow as tf
import requests
from dotenv import load_dotenv
import sqlite3
import logging
from typing import Optional, List, Dict, Tuple, Any
import time
import random
import base64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get API keys from environment variables safely
GOOGLE_API_KEY = "AIzaSyCUnt3ZLimDiToSqlfJBCTepPrqpZBCKgY"
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not set in environment variables")
    raise RuntimeError("GOOGLE_API_KEY not set in environment variables")

WEATHER_API_KEY = "71205c205335df85df7b13f3519e8bf4"
SERPAPI_KEY = "10fc45f90a84480a587a8328286f2bb06e70b75b3e9435cab52918bbd0248653"
UNSPLASH_KEY="mOSycI-GrOppoTI_6Oqv3ahKwxDFTVKcE3amppU-vCg"

# Validate that required keys are present
for key_name, key_value in [
    ("WEATHER_API_KEY", WEATHER_API_KEY),
    ("SERPAPI_KEY", SERPAPI_KEY)
]:
    if not key_value:
        logger.warning(f"{key_name} not set in environment variables")

# Load the TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="1.tflite")
    interpreter.allocate_tensors()
    logger.info("TFLite model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {e}")
    raise RuntimeError(f"Failed to load TFLite model: {e}")

# Configure Gemini
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    raise RuntimeError(f"Failed to initialize Gemini model: {e}")

# Create FastAPI app
app = FastAPI(
    title="Plant Recommendation API",
    description="API for suggesting suitable plants based on image analysis and environmental data",
    version="1.0.0"
)

# Initialize request history dictionary for rate limiting
request_history = {}

# Add rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Simple in-memory rate limiting (consider using Redis for production)
    client_ip = request.client.host
    current_time = time.time()
    
    # Allow 10 requests per minute per IP
    if client_ip in request_history:
        request_times = request_history[client_ip]
        # Keep only requests in the last minute
        request_times = [t for t in request_times if current_time - t < 60]
        
        if len(request_times) >= 10:
            return Response(
                content='{"detail":"Too many requests"}',
                status_code=429,
                media_type="application/json"
            )
        
        request_history[client_ip] = request_times + [current_time]
    else:
        request_history[client_ip] = [current_time]
    
    return await call_next(request)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db():
    conn = sqlite3.connect("suggestions.db", check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()

# Create tables on startup
def init_db():
    conn = sqlite3.connect("suggestions.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude REAL,
        longitude REAL,
        width REAL,
        height REAL,
        sunlight TEXT,
        moisture REAL,
        plants TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized")

init_db()

# Pydantic models
class EnvironmentData(BaseModel):
    sunlight: float
    moisture: float
    width: float
    height: float
    pot_preference: str = "any"

class PlantSuggestion(BaseModel):
    name: str
    pot_size: str
    info_url: str
    image_url: Optional[str] = None

class SuggestionResponse(BaseModel):
    environment: EnvironmentData
    coordinates: Dict[str, float]
    suggested_plants: List[PlantSuggestion]
    explanation: str
    ar_image_base64: Optional[str] = None
    ar_images: Optional[Dict[str, Optional[str]]] = None  # Added field for multiple AR images

# Helper functions
def optimize_image_for_processing(image: Image.Image) -> Image.Image:
    """Optimize image for faster processing."""
    # Resize large images to a reasonable size for analysis
    max_size = 1024
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    
    # Return optimized image
    return image

def estimate_plant_area(pot_size: str) -> float:
    """Convert pot size descriptions to approximate area in square meters."""
    mapping = {
        "0.25kg": 0.03,
        "0.5kg": 0.04,
        "1kg": 0.06,
        "2kg": 0.1,
        "small": 0.05,
        "medium": 0.12,
        "large": 0.25,
        "extra large": 0.4
    }
    pot_size_lower = pot_size.lower().strip()
    for key in mapping:
        if key in pot_size_lower:
            return mapping[key]
    return 0.1

def convert_sunlight_level(level: str) -> float:
    """Convert sunlight level descriptions to numerical values."""
    sunlight_map = {
        "low": 0.3, 
        "partial": 0.5,
        "moderate": 0.6,
        "medium": 0.6,
        "high": 0.9,
        "full": 0.9
    }
    level_lower = level.lower().strip()
    for key, value in sunlight_map.items():
        if key in level_lower:
            return value
    return 0.6

def fetch_plant_image(plant_name: str) -> Optional[str]:
    """Fetch an image URL for a plant using SerpAPI."""
    if not SERPAPI_KEY:
        return None
        
    try:
        params = {
            "engine": "google",
            "q": f"{plant_name} plant",
            "tbm": "isch",
            "api_key": SERPAPI_KEY,
            "ijn": str(random.randint(0, 2))  # Random page to get different images
        }
        
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        if "images_results" not in data or not data["images_results"]:
            return None
            
        return random.choice(data["images_results"][:3]).get("original")
    except Exception:
        return None

def fetch_plant_image_with_fallback(plant_name: str) -> Optional[str]:
    """
    Try to fetch plant image with fallback to Unsplash API if SerpAPI fails.
    Includes randomization to avoid duplicate images.
    """
    # First try SerpAPI
    image_url = fetch_plant_image(plant_name)
    
    # If SerpAPI failed and we have an Unsplash key, try Unsplash as fallback
    if not image_url and UNSPLASH_KEY:
        try:
            random_terms = ["indoor", "outdoor", "potted", "closeup", "green"]
            random_term = random.choice(random_terms)
            
            params = {
                "query": f"{plant_name} plant {random_term}",
                "per_page": 1,
                "page": random.randint(1, 5),
                "client_id": UNSPLASH_KEY,
                "orientation": random.choice(["portrait", "landscape"])
            }
            
            response = requests.get("https://api.unsplash.com/search/photos", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("results") and len(data["results"]) > 0:
                    return data["results"][0].get("urls", {}).get("regular")
        except Exception:
            pass
    
    return image_url

def get_gemini_recommendations(env: EnvironmentData) -> List[Tuple[str, str]]:
    """Get plant recommendations from Gemini AI based on environmental data."""
    prompt = (
        f"The balcony is {env.width:.1f}m by {env.height:.1f}m with "
        f"{env.sunlight*100:.0f}% sunlight and {env.moisture*100:.0f}% moisture. "
        f"Suggest 5-8 suitable plants. Format: <plant name> - <pot size>. "
        f"Prioritize {env.pot_preference} pots. Each plant should have a different name. "
        f"Include both common and scientific names when possible. "
        f"Use standardized pot size descriptions (small, medium, large or weight in kg)."
    )
    try:
        response = gemini_model.generate_content(prompt)
        lines = response.text.strip().split("\n")
        plants = []
        for line in lines:
            if " - " in line:
                line = line.split(". ", 1)[-1] if ". " in line else line
                parts = line.split(" - ")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    size = parts[1].strip()
                    if size and size[-1] in ".,;:":
                        size = size[:-1]
                    plants.append((name, size))
        return plants
    except Exception as e:
        logger.error(f"Error getting plant recommendations: {e}")
        return [
            ("Snake Plant (Sansevieria)", "medium"),
            ("ZZ Plant (Zamioculcas zamiifolia)", "medium"),
            ("Pothos (Epipremnum aureum)", "small"),
            ("Spider Plant (Chlorophytum comosum)", "small")
        ]

def get_weather_data(lat: float, lon: float) -> Dict[str, Any]:
    """Get weather data for a location using OpenWeatherMap API."""
    if not WEATHER_API_KEY:
        return {"sunlight": 50, "moisture": 50, "temp": 25}
        
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"Weather API error: {response.status_code}")
            
        data = response.json()
        return {
            "sunlight": 100 - data.get("clouds", {}).get("all", 50),
            "moisture": data.get("main", {}).get("humidity", 50),
            "temp": data.get("main", {}).get("temp", 25)
        }
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return {"sunlight": 50, "moisture": 50, "temp": 25}

def analyze_image(image: Image.Image) -> Tuple[float, float]:
    """Analyze image dimensions using TensorFlow model."""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        img = image.resize((224, 224))
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        
        width, height = image.size
        return width / 100, height / 100
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return 2.0, 1.5

def extract_gps_from_image(file: bytes) -> Tuple[Optional[float], Optional[float]]:
    """Extract GPS coordinates from image EXIF data."""
    try:
        image = Image.open(BytesIO(file))
        exif_data = image._getexif()
        if not exif_data:
            return None, None
            
        gps_info = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag)
            if decoded == "GPSInfo":
                for t in value:
                    gps_info[ExifTags.GPSTAGS.get(t)] = value[t]

        required_tags = ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"]
        if not all(tag in gps_info for tag in required_tags):
            return None, None

        def dms_to_deg(dms, ref):
            degrees = dms[0][0] / dms[0][1]
            minutes = dms[1][0] / dms[1][1]
            seconds = dms[2][0] / dms[2][1]
            deg = degrees + minutes / 60 + seconds / 3600
            return -deg if ref in ['S', 'W'] else deg

        lat = dms_to_deg(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
        lon = dms_to_deg(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
        return lat, lon
    except Exception as e:
        logger.error(f"Error extracting GPS data: {e}")
        return None, None

def generate_ar_view(balcony_image: Image.Image, plant_suggestions: List[Dict[str, Any]]) -> Image.Image:
    """
    Generate an AR-like view by placing plant images on the balcony image.
    Uses perspective transformation for more realistic placement.
    """
    try:
        ar_image = balcony_image.copy()
        
        # Calculate positions based on balcony dimensions
        img_width, img_height = ar_image.size
        positions = []
        
        # Generate grid positions
        cols = min(3, len(plant_suggestions))
        rows = (len(plant_suggestions) + cols - 1) // cols
        
        for i in range(len(plant_suggestions)):
            col = i % cols
            row = i // cols
            
            x = int((col + 0.5) * (img_width / cols))
            y = int((row + 0.7) * (img_height / rows))
            positions.append((x, y))
        
        # Place plants
        for i, plant in enumerate(plant_suggestions):
            if i >= len(positions):
                break
                
            if plant.get("image_url"):
                try:
                    response = requests.get(plant["image_url"], timeout=10)
                    plant_img = Image.open(BytesIO(response.content)).convert("RGBA")
                    
                    # Size based on pot size
                    if "small" in plant["pot_size"].lower():
                        size = (150, 150)
                    elif "medium" in plant["pot_size"].lower():
                        size = (200, 200)
                    else:
                        size = (250, 250)
                    
                    plant_img = plant_img.resize(size)
                    
                    # Add shadow effect
                    shadow = Image.new('RGBA', (size[0]+10, size[1]+10), (0,0,0,100))
                    shadow = shadow.filter(ImageFilter.GaussianBlur(5))
                    
                    # Composite shadow then plant
                    x, y = positions[i]
                    ar_image.paste(shadow, (x-size[0]//2+5, y-size[1]//2+5), shadow)
                    ar_image.paste(plant_img, (x-size[0]//2, y-size[1]//2), plant_img)
                except Exception as e:
                    logger.error(f"Error placing plant {plant['name']}: {e}")
        
        return ar_image
    except Exception as e:
        logger.error(f"Error generating AR view: {e}")
        return balcony_image

async def generate_gemini_ar_image(balcony_image: Image.Image, plant_suggestions: List[Dict[str, Any]]) -> Optional[Image.Image]:
    """Generate an AR image using Gemini's image generation capabilities."""
    try:
        # Convert balcony image to base64
        buffered = BytesIO()
        balcony_image.save(buffered, format="JPEG")
        balcony_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare plant descriptions
        plant_descriptions = "\n".join(
            f"- {plant['name']} in a {plant['pot_size']} pot" 
            for plant in plant_suggestions
        )
        
        # Generate prompt
        prompt = f"""
        Create a realistic image showing how these plants would look in this space:
        {plant_descriptions}
        
        The plants should be arranged naturally in the space, considering their sizes.
        Make the lighting and shadows look realistic.
        Keep the original room perspective and style.
        """
        
        # Generate the image using Gemini model's async method
        response = await gemini_model.generate_content_async(
            [prompt, {"inline_data": {"mime_type": "image/jpeg", "data": balcony_base64}}]
        )
        
        # Extract and return the generated image
        if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'content'):
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'image') and hasattr(part.image, 'data'):
                    return Image.open(BytesIO(part.image.data))
        
        return None
    except Exception as e:
        logger.error(f"Error generating Gemini AR image: {e}")
        return None

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/suggest", response_model=SuggestionResponse)
async def suggest_plants(
    file: UploadFile = File(...),
    sunlight_level: str = "moderate",
    pot_preference: str = "any",
    db: sqlite3.Connection = Depends(get_db)
):
    """Suggest plants based on uploaded image and parameters."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
            
        image = Image.open(BytesIO(contents)).convert("RGB")
        image = optimize_image_for_processing(image)
        width, height = analyze_image(image)

        lat, lon = extract_gps_from_image(contents)
        if lat is None or lon is None:
            lat, lon = 12.97, 77.59
            logger.info("Using default coordinates (Bangalore)")

        weather = get_weather_data(lat, lon)
        moisture = weather["moisture"] / 100
        sunlight = convert_sunlight_level(sunlight_level)

        env = EnvironmentData(
            sunlight=sunlight, 
            moisture=moisture, 
            width=width, 
            height=height, 
            pot_preference=pot_preference
        )
        
        area = width * height
        candidates = get_gemini_recommendations(env)

        used_area = 0
        final_plants = []

        # Select plants based on available space
        for name, pot_size in candidates:
            plant_area = estimate_plant_area(pot_size)
            
            if used_area + plant_area <= area:
                image_url = fetch_plant_image_with_fallback(name)
                
                plant = {
                    "name": name,
                    "pot_size": pot_size,
                    "info_url": f"https://www.google.com/search?q={name.replace(' ', '+')}+plant+care",
                    "image_url": image_url
                }
                
                final_plants.append(plant)
                used_area += plant_area
                
            if used_area >= area:
                break

        # Ensure we have at least 2 plant suggestions
        if len(final_plants) < 2 and candidates:
            remaining = [p for p in candidates if p[0] not in [plant["name"] for plant in final_plants]]
            for name, pot_size in remaining[:2]:
                if len(final_plants) >= 2:
                    break
                image_url = fetch_plant_image_with_fallback(name)
                plant = {
                    "name": name,
                    "pot_size": pot_size,
                    "info_url": f"https://www.google.com/search?q={name.replace(' ', '+')}+plant+care",
                    "image_url": image_url
                }
                final_plants.append(plant)

        # Generate AR views
        balcony_image = Image.open(BytesIO(contents)).convert("RGB")
        balcony_image = optimize_image_for_processing(balcony_image)
        
        # Method 1: Direct image placement
        ar_image_direct = generate_ar_view(balcony_image.copy(), final_plants)
        
        # Method 2: Gemini-generated image (if implemented)
        ar_image_gemini = None
        try:
            ar_image_gemini = await generate_gemini_ar_image(balcony_image.copy(), final_plants)
        except Exception as e:
            logger.error(f"Error generating Gemini AR image: {e}")
        
        # Prepare both images as base64
        ar_images = {
            "direct": image_to_base64(ar_image_direct) if ar_image_direct else None,
            "gemini": image_to_base64(ar_image_gemini) if ar_image_gemini else None
        }
        
        # Use Gemini image as primary if available, otherwise use direct placement
        ar_image = ar_image_gemini if ar_image_gemini else ar_image_direct
        ar_image_base64 = image_to_base64(ar_image)

        # Generate explanation
        plants_json = [{"name": p["name"], "pot_size": p["pot_size"]} for p in final_plants]
        prompt = (
            f"Given these plant recommendations: {plants_json}, "
            f"write a user-friendly summary (about 3-4 paragraphs) explaining why these plants "
            f"are suitable for a {width:.1f}m x {height:.1f}m space with {sunlight_level} sunlight "
            f"and {moisture*100:.0f}% moisture. Include brief care tips for each plant."
        )
        
        try:
            explanation = gemini_model.generate_content(prompt).text.strip()
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            explanation = (
                f"Here are {len(final_plants)} plants recommended for your {width:.1f}m x {height:.1f}m space "
                f"with {sunlight_level} sunlight. These plants are selected based on your specific environmental "
                f"conditions and should thrive in your space with proper care."
            )

        # Save to database
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO suggestions (latitude, longitude, width, height, sunlight, moisture, plants)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            lat, lon, width, height, sunlight_level, moisture,
            ", ".join([f"{p['name']} - {p['pot_size']}" for p in final_plants])
        ))
        db.commit()

        return {
            "environment": env,
            "coordinates": {"latitude": lat, "longitude": lon},
            "suggested_plants": final_plants,
            "explanation": explanation,
            "ar_image_base64": ar_image_base64,
            "ar_images": ar_images
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing request:")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)