# app.py

import os
from flask import Flask, request, Response, g, render_template, jsonify
import marko
import google.generativeai as genai

genai.configure(api_key=os.getenv("API_KEY"))

app = Flask(__name__)
app.debug = True

config = {
    'temperature': 0,
    'top_k': 20,
    'top_p': 0.9,
    'max_output_tokens': 500
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=config,
                              safety_settings=safety_settings)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template("chat.html")

# Inside the /chat route in app.py
@app.route('/chat', methods=['POST'])
def chat():
    user_image = request.files.get('user_image')
    user_text = request.form.get('user_text', '')

    if not user_image and not user_text.strip():
        return jsonify({"error": "No input provided"})

    prompt_parts = [
        "You are Batman. The dark knight. The user is reaching out to you.\n\n",
    ]

    if user_text.strip():
        prompt_parts.extend([
            "User's text:\n\n",
            user_text,
            "\n\n",
        ])

    if user_image:
        image_data = user_image.read()
        image_parts = [
            {
                "mime_type": user_image.content_type,
                "data": image_data
            },
        ]
        prompt_parts.extend([
            "User's image:\n\n",
            image_parts[0],
            "\n\n",
        ])

    prompt_parts.append("Your response:\n")

    if user_text.strip() and user_image:
        # If both image and text are provided, use gemini-pro-vision model
        image_model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=config,
                              safety_settings=safety_settings)
        response = image_model.generate_content(prompt_parts)
    elif user_text.strip():
        # If only text is provided, switch to gemini-pro model
        text_model = genai.GenerativeModel(model_name="gemini-pro", generation_config=config)
        response = text_model.generate_content(prompt_parts)
    elif user_image:
        # If only image is provided, use gemini-pro-vision model
        image_model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=config,
                              safety_settings=safety_settings)
        response = image_model.generate_content(prompt_parts)

    return jsonify({
        "response": marko.convert(response.text)
    })



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
