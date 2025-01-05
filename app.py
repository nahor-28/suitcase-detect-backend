from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import google.generativeai as genai
import base64
from flask_cors import CORS
import logging
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PORT=os.getenv("PORT")
ORIGIN_URL_PROD=os.getenv("ORIGIN_PROD")

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

CORS(app, resources={r"/api/*": {"origins": [ORIGIN_URL_PROD]}})

# Initialize the GenerativeAI client
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/api/estimate-size', methods=['POST'])
@limiter.limit("10 per minute")
def estimate_size():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        if not data or 'image' not in data:
            logger.error("No image data found in request")
            return jsonify({'error': 'No image provided'}), 400

        # Extract the base64 image data
        image_data = data['image']
        
        # Remove the data URL prefix if present
        if ',' in image_data:
            logger.debug("Removing data URL prefix")
            image_data = image_data.split(',')[1]

        try:
            # Validate base64 data
            base64.b64decode(image_data)
        except Exception as e:
            logger.error("Invalid base64 data: %s", str(e))
            return jsonify({'error': 'Invalid image data'}), 400

        # The new prompt
        prompt_text = """
        You are a highly intelligent and perceptive AI specializing in analyzing images from airport settings. Your primary task is to identify and estimate the dimensions of luggage, including suitcases, carry-ons, backpacks, and other travel bags. You should focus exclusively on objects clearly intended for carrying personal belongings during travel.

When presented with an image, meticulously analyze it, focusing only on luggage items. Ignore other objects such as people, furniture, or airport infrastructure unless they provide crucial context for determining the scale of the luggage. Consider perspective, shadows, and any visual cues that might indicate the size of the luggage. If no luggage is present, indicate this clearly.

For each piece of luggage detected, provide an estimation of its dimensions in centimeters, a brief descriptive text, an estimation of its volumetric capacity in cubic centimeters. **Crucially, when estimating the weight, assume the luggage is packed with typical clothing items (e.g., shirts, pants, shoes). Do not estimate the empty weight of the luggage itself. Provide an estimated total weight, including the contents.** Also estimate how much space the luggage will take up in the overhead cabin of a standard commercial airplane. Additionally calculate the sum of the length, width, and height in centimeters.

Output the results in the following JSON format:

```json
[
  {
    "object_type": "Type of luggage (e.g., suitcase, carry-on, backpack)",
    "description": "A brief description of the luggage (e.g., 'Large, hard-shell suitcase, dark blue', 'Small, rolling carry-on with a front pocket', 'Red backpack with visible straps')",
    "length_cm": "Estimated length in centimeters",
    "width_cm": "Estimated width in centimeters",
    "height_cm": "Estimated height in centimeters",
    "sum_dimensions_cm": "Sum of length, width, and height in centimeters",
    "volumetric_capacity_cubic_cm": "Estimated volumetric capacity in cubic centimeters",
    "estimated_weight_kg": "Estimated total weight (including contents) in kilograms",
    "confidence": "Confidence level in the estimations (e.g., 'High', 'Medium', 'Low')",
    "bounding_box": {
        "x_min": "x coordinate of top-left corner",
        "y_min": "y coordinate of top-left corner",
        "x_max": "x coordinate of bottom-right corner",
        "y_max": "y coordinate of bottom-right corner"
    }
  },
    {
      "object_type":"No Luggage",
      "description": "No luggage detected in the image.",
      "length_cm": null,
      "width_cm": null,
      "height_cm": null,
      "sum_dimensions_cm":null,
      "volumetric_capacity_cubic_cm": null,
      "estimated_weight_kg": null,
      "confidence": null,
      "bounding_box": null
  }
  // ... more luggage objects if present
]
"""

        # Generate content
        logger.debug("Sending request to Gemini")
        response = model.generate_content([
            prompt_text,
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        
        # Parse the response text as JSON
        try:
            # Extract JSON from the response text
            response_text = response.text
            
            # Clean up the response text to extract just the JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            # Parse the JSON
            parsed_data = json.loads(json_str)
            
            # Ensure the response is a list
            if not isinstance(parsed_data, list):
                parsed_data = [parsed_data]
            
            logger.debug(f"Successfully parsed response: {json.dumps(parsed_data, indent=2)}")
            return jsonify(parsed_data), 200
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_text}")
            return jsonify([{
                "object_type": "Error",
                "description": "Failed to process luggage information",
                "length_inches": None,
                "width_inches": None,
                "height_inches": None,
                "confidence": None,
                "bounding_box": None
            }]), 500

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=PORT)